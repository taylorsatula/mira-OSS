"""
Generic OpenAI-compatible client with Anthropic format translation.

This client accepts Anthropic-format messages and tools, translates them to
OpenAI format, makes HTTP requests to any OpenAI-compatible endpoint, and
translates responses back to Anthropic format.

Use cases:
- Emergency failover during Anthropic outages
- On-demand calls to alternative providers (Groq, OpenRouter, local models)
- Cost optimization for simple operations (sentiment, classification, etc.)
- Local inference for development/testing

⚠️ IMPORTANT FOR CODE GENERATORS (CLAUDE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT instantiate this class directly when writing application code!

WRONG:
    from utils.generic_openai_client import GenericOpenAIClient
    client = GenericOpenAIClient(endpoint="...", api_key="...", model="...")
    response = client.messages.create(...)

CORRECT:
    from clients.llm_provider import LLMProvider
    llm = LLMProvider()
    response = llm.generate_response(
        messages=[...],
        endpoint_url="https://openrouter.ai/api/v1/chat/completions",
        model_override="anthropic/claude-3-5-sonnet",
        api_key_override=api_key
    )

WHY: Using LLMProvider.generate_response() provides:
- Consistent interface for all LLM calls (Anthropic + third-party)
- Easy migration between providers (just omit overrides for Anthropic)
- Automatic failover and error handling
- Unified logging and monitoring
- Future flexibility for routing changes

This class is ONLY used internally by LLMProvider. Application code should
never import or instantiate GenericOpenAIClient directly.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
import logging
import requests
from types import SimpleNamespace
from typing import Dict, List, Any, NoReturn, Optional, Union

# System prompt: plain string or list of cache_control blocks (Anthropic format)
SystemPrompt = Union[str, List[Dict[str, str]]]

logger = logging.getLogger(__name__)


class ToolNotLoadedError(Exception):
    """
    Raised when model attempts to use a tool not in the request.

    This exception is raised when a generic provider (e.g., Groq) rejects a tool call
    because the tool wasn't included in the tools parameter. The LLMProvider catches
    this and returns a synthetic response that triggers invokeother_tool to load the
    requested tool.
    """
    def __init__(self, tool_name: str, original_message: str):
        self.tool_name = tool_name
        self.original_message = original_message
        super().__init__(f"Tool '{tool_name}' not loaded")


class GenericProviderErrorParser:
    """Extract a readable message from generic provider error bodies."""

    @staticmethod
    def extract_message(error_body: Optional[dict], fallback_text: str) -> str:
        if not error_body:
            return fallback_text
        error_info = error_body.get("error", {})
        return error_info.get("message") or fallback_text


class GenericOpenAIResponse:
    """
    Mimics anthropic.types.Message structure for compatibility.

    This allows GenericOpenAIClient to be used as a drop-in replacement
    for anthropic.Anthropic client in LLMProvider codepaths.

    ⚠️ INTERNAL CLASS - DO NOT USE DIRECTLY IN APPLICATION CODE
    This is only used by LLMProvider when routing to third-party providers.

    🚧 REPLACEMENT-LAYER INTEGRATION NOTE (2026-05-20):
    This class does NOT expose a `.model` attribute, while `anthropic.types.Message`
    does. `tools/implementations/phoneafriend_tool.py` relies on this asymmetry to
    detect silent claude-high failover swaps (failover response has `.model`,
    successful generic-provider response does not). When integrating the wholesale
    LLM-provider rewrite, preserve a model-identity signal on the response object
    (e.g. add `self.model = model` here, or attach the resolved endpoint/model on
    the returned wrapper) so callers can verify the requested provider was actually
    served. Otherwise the phoneafriend mismatch check silently breaks.
    """

    def __init__(self, content: List, stop_reason: str, usage: Dict,
                 reasoning_details: Optional[List] = None):
        """
        Initialize response with Anthropic-compatible structure.

        Args:
            content: List of content blocks (text, tool_use)
            stop_reason: Anthropic-style stop reason (end_turn, tool_use, max_tokens)
            usage: Token usage dictionary
            reasoning_details: Raw reasoning_details from OpenRouter for round-tripping
        """
        self.content = content
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(**usage)
        self.reasoning_details = reasoning_details


class GenericOpenAIClient:
    """
    Generic client for OpenAI-compatible API endpoints with Anthropic translation.

    Provides a messages.create() interface that accepts Anthropic-format inputs
    and returns Anthropic-compatible response objects, enabling seamless integration
    with existing LLMProvider codepaths.

    ⚠️⚠️⚠️ CRITICAL - DO NOT USE THIS CLASS DIRECTLY ⚠️⚠️⚠️
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    This class is INTERNAL to LLMProvider. Application code should NEVER import
    or instantiate GenericOpenAIClient directly.

    WRONG EXAMPLE (DO NOT DO THIS):
        from utils.generic_openai_client import GenericOpenAIClient
        client = GenericOpenAIClient(
            endpoint="https://api.groq.com/openai/v1/chat/completions",
            api_key="gsk_...",
            model="llama-3.1-70b-versatile"
        )
        response = client.messages.create(messages=[...])

    CORRECT EXAMPLE (ALWAYS DO THIS):
        from clients.llm_provider import LLMProvider
        llm = LLMProvider()
        response = llm.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            endpoint_url="https://api.groq.com/openai/v1/chat/completions",
            model_override="llama-3.1-70b-versatile",
            api_key_override=groq_api_key
        )

    The ONLY code that should instantiate this class is LLMProvider._generate_non_streaming()
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_tokens: int = 4096,
        temperature: float = 1.0
    ):
        """
        Initialize generic OpenAI-compatible client.

        Args:
            endpoint: Full URL to chat completions endpoint
            model: Model identifier to use
            api_key: API key for authentication (optional for local providers like Ollama)
            timeout: Request timeout in seconds
            max_tokens: Default max tokens (can be overridden per request)
            temperature: Default temperature (can be overridden per request)
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.default_max_tokens = max_tokens
        self.default_temperature = temperature

        # Create messages namespace to mimic anthropic.Anthropic interface
        self.messages = SimpleNamespace(
            create=self._create_message,
            create_streaming=self._create_message_streaming
        )

        logger.info(f"GenericOpenAIClient initialized: {endpoint} / {model}")

    def _create_message(
        self,
        messages: List[Dict],
        system: Optional[SystemPrompt] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking_effort: Optional[str] = None,
        thinking_budget: int = 0,
        **kwargs
    ) -> GenericOpenAIResponse:
        """
        Create a message using OpenAI-compatible endpoint (non-streaming).

        Mimics anthropic.Anthropic.messages.create() interface for drop-in compatibility.

        Args:
            messages: Anthropic-format messages (user/assistant with content blocks)
            system: System prompt (string or list of blocks with cache_control)
            tools: Anthropic-format tool definitions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            thinking_effort: Effort level string for reasoning (e.g. "low", "medium", "high")
            thinking_budget: Token budget for thinking (adjusts max_tokens)
            **kwargs: Additional parameters (ignored for compatibility)

        Returns:
            GenericOpenAIResponse with Anthropic-compatible structure

        Raises:
            TimeoutError: If request times out
            PermissionError: If authentication fails
            RuntimeError: On other API errors
        """
        # Use defaults if not specified
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        # Adjust max_tokens for thinking if enabled
        if thinking_effort:
            max_tokens = max_tokens + thinking_budget

        # Prepare messages - Anthropic API uses different format than OpenAI
        is_anthropic = "api.anthropic.com" in self.endpoint

        if is_anthropic:
            # Anthropic: system as top-level param, messages without system role
            anthropic_messages = self._convert_messages(messages)
            logger.debug(f"Converted {len(messages)} messages for Anthropic API")

            payload = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
            }
            # Anthropic uses top-level system parameter
            if system:
                if isinstance(system, str):
                    payload["system"] = system
                elif isinstance(system, list):
                    # Convert list of blocks to string for simplicity
                    payload["system"] = " ".join(
                        block.get("text", "") for block in system if block.get("type") == "text"
                    )
        else:
            # OpenAI-compatible: system as message with role "system"
            openai_messages = []
            if system:
                openai_messages.append(self._convert_system_prompt(system))
            openai_messages.extend(self._convert_messages(messages))
            logger.debug(f"Converted {len(messages)} Anthropic messages to {len(openai_messages)} OpenAI messages")

            payload = {
                "model": self.model,
                "messages": openai_messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

        # Add tools if provided
        if tools:
            openai_tools = self._convert_tools(tools)
            payload["tools"] = openai_tools
            logger.debug(f"Converted {len(tools)} Anthropic tools to {len(openai_tools)} OpenAI tools")

        # Add reasoning for providers that support it (OpenRouter/Gemini format)
        if thinking_effort:
            payload["reasoning"] = {"effort": thinking_effort}
            logger.debug(f"Reasoning enabled for generic provider (effort={thinking_effort})")

        # Traffic tap: log request before sending
        from utils.llm_tap import is_active as _tap_active, log_request as _tap_request
        if _tap_active():
            _tap_request(provider="generic", endpoint=self.endpoint, model=self.model, body=payload)

        # Make HTTP request
        try:
            logger.debug(f"Generic OpenAI client request to {self.endpoint} with model {self.model}")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                # Anthropic uses x-api-key header, not Bearer token
                if "api.anthropic.com" in self.endpoint:
                    headers["x-api-key"] = self.api_key
                    headers["anthropic-version"] = "2023-06-01"
                else:
                    headers["Authorization"] = f"Bearer {self.api_key}"
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return self._wrap_response(response.json())

        except requests.Timeout:
            logger.error("Generic OpenAI client request timed out")
            raise TimeoutError("Generic OpenAI client request timed out")
        except requests.HTTPError as e:
            # Log full error for debugging and extract error details
            error_body = None
            try:
                error_body = e.response.json()
                logger.error(f"Generic OpenAI client HTTP error: {e.response.status_code} - {error_body}")
            except:
                logger.error(f"Generic OpenAI client HTTP error: {e.response.status_code} - {e.response.text}")
            self._handle_http_error(e, error_body)
        except Exception as e:
            logger.error(f"Generic OpenAI client error: {e}")
            raise RuntimeError(f"Generic OpenAI client error: {e}")

    def _create_message_streaming(
        self,
        messages: List[Dict],
        system: Optional[SystemPrompt] = None,
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        thinking_effort: Optional[str] = None,
        thinking_budget: int = 0,
        **kwargs
    ):
        """
        Stream a message using OpenAI-compatible endpoint with SSE.

        Yields raw SSE chunks as dicts. Each chunk has structure:
        {"choices": [{"delta": {"content": "...", "tool_calls": [...]}, "finish_reason": None}]}

        Final chunk has finish_reason set ("stop" or "tool_calls").

        Args:
            messages: Anthropic-format messages
            system: System prompt
            tools: Anthropic-format tool definitions
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            thinking_effort: Effort level string for reasoning
            thinking_budget: Token budget for thinking

        Yields:
            Dict chunks from SSE stream
        """
        from utils import http_client

        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        if thinking_effort:
            max_tokens = max_tokens + thinking_budget

        # Build messages list
        openai_messages = []
        if system:
            openai_messages.append(self._convert_system_prompt(system))
        openai_messages.extend(self._convert_messages(messages))

        # Build request payload with stream=True
        payload = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        if tools:
            openai_tools = self._convert_tools(tools)
            payload["tools"] = openai_tools
            logger.debug(f"Streaming with {len(tools)} tools")

        if thinking_effort:
            payload["reasoning"] = {"effort": thinking_effort}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Anthropic uses x-api-key header, not Bearer token
            if "api.anthropic.com" in self.endpoint:
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        # Traffic tap: log streaming request before sending
        from utils.llm_tap import is_active as _tap_active, log_request as _tap_request
        if _tap_active():
            _tap_request(provider="generic", endpoint=self.endpoint, model=self.model, body=payload)

        logger.debug(f"Starting streaming request to {self.endpoint}")

        with http_client.stream("POST", self.endpoint, json=payload, headers=headers, timeout=self.timeout) as response:
            if response.status_code >= 400:
                error_text = response.read().decode('utf-8', errors='replace')
                logger.error(f"Streaming request failed: {response.status_code} - {error_text[:500]}")
                error_body = None
                try:
                    error_body = json.loads(error_text)
                except json.JSONDecodeError:
                    pass
                self._raise_provider_http_error(
                    status=response.status_code,
                    error_body=error_body,
                    fallback_text=error_text,
                    mode="streaming",
                )

            for line in response.iter_lines():
                line = line.strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    logger.debug("SSE stream complete")
                    break
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])  # Strip "data: " prefix
                        yield chunk
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse SSE chunk: {line[:100]}")
                        continue

    def _convert_system_prompt(self, system: SystemPrompt) -> Dict[str, str]:
        """
        Convert Anthropic system prompt to OpenAI system message.

        Args:
            system: String or list of blocks with cache_control

        Returns:
            OpenAI system message dict
        """
        if isinstance(system, list):
            # Extract text from structured blocks (strip cache_control)
            text_parts = [b.get("text", "") for b in system if b.get("type") == "text"]
            text = "".join(text_parts)
            return {"role": "system", "content": text}
        elif isinstance(system, str):
            return {"role": "system", "content": system}
        else:
            return {"role": "system", "content": str(system)}

    def _convert_messages(self, anthropic_messages: List[Dict]) -> List[Dict]:
        """
        Convert Anthropic messages to OpenAI format.

        Handles:
        - Text content blocks → string content
        - Tool use blocks → tool_calls array
        - Tool result blocks → tool role messages
        - Thinking blocks → stripped (not supported in OpenAI)

        Args:
            anthropic_messages: Messages in Anthropic format

        Returns:
            Messages in OpenAI format
        """
        openai_messages = []
        logger.debug(f"Converting {len(anthropic_messages)} Anthropic messages")

        for msg in anthropic_messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                # Handle user messages with text or tool results
                if isinstance(content, list):
                    # Check for tool_result blocks
                    tool_results = [b for b in content if b.get("type") == "tool_result"]
                    if tool_results:
                        # Add tool result messages
                        for tr in tool_results:
                            # Handle content - could be string, dict, or other types
                            result_content = tr.get("content", "")
                            if isinstance(result_content, (dict, list)):
                                result_content = json.dumps(result_content)
                            else:
                                result_content = str(result_content)

                            openai_messages.append({
                                "role": "tool",
                                "tool_call_id": tr["tool_use_id"],
                                "content": result_content
                            })
                    else:
                        # Extract text from content blocks
                        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
                        if text:
                            openai_messages.append({"role": "user", "content": text})
                else:
                    # Simple string content
                    openai_messages.append({"role": "user", "content": content})

            elif role == "assistant":
                # Handle assistant messages with text and tool_use blocks
                text_parts = []
                tool_calls = []

                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"])
                                }
                            })
                        elif block.get("type") == "thinking":
                            # Skip thinking blocks (not supported in generic providers)
                            logger.debug("Skipping thinking block in generic OpenAI client")
                elif isinstance(content, str):
                    # Simple string content
                    text_parts.append(content)

                msg_obj = {"role": "assistant"}
                if text_parts:
                    msg_obj["content"] = "".join(text_parts)
                if tool_calls:
                    msg_obj["tool_calls"] = tool_calls
                # Preserve reasoning_details for round-trip (OpenRouter/Gemini requirement)
                if msg.get("reasoning_details"):
                    msg_obj["reasoning_details"] = msg["reasoning_details"]

                openai_messages.append(msg_obj)

        logger.debug(f"Converted to {len(openai_messages)} OpenAI messages")
        return openai_messages

    def _convert_tools(self, anthropic_tools: List[Dict]) -> List[Dict]:
        """
        Convert Anthropic tool schemas to OpenAI format.

        Args:
            anthropic_tools: Tool definitions in Anthropic format

        Returns:
            Tool definitions in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            for tool in anthropic_tools
        ]

    def _wrap_response(self, response: Dict[str, Any]) -> GenericOpenAIResponse:
        """
        Convert API response to Anthropic-compatible structure.

        Handles both OpenAI format (choices array) and Anthropic format (content array).

        Args:
            response: Full API response (OpenAI or Anthropic format)

        Returns:
            GenericOpenAIResponse with Anthropic-compatible structure

        Raises:
            ValueError: If response structure is invalid
        """
        # Check if this is native Anthropic response format
        if "content" in response and "type" in response and response.get("type") == "message":
            # Native Anthropic format - already in correct structure
            content_blocks = []
            for block in response.get("content", []):
                if block.get("type") == "text":
                    content_blocks.append(SimpleNamespace(
                        type="text",
                        text=block.get("text", "")
                    ))
                elif block.get("type") == "tool_use":
                    content_blocks.append(SimpleNamespace(
                        type="tool_use",
                        id=block.get("id"),
                        name=block.get("name"),
                        input=block.get("input", {})
                    ))

            usage = response.get("usage", {})
            return GenericOpenAIResponse(
                content=content_blocks,
                stop_reason=response.get("stop_reason", "end_turn"),
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0)
                }
            )

        # OpenAI format - validate and convert
        openai_response = response
        if "choices" not in openai_response or not openai_response["choices"]:
            logger.error(f"Invalid OpenAI response: missing or empty choices - {openai_response}")
            raise ValueError("Invalid OpenAI response: missing or empty choices")

        if "usage" not in openai_response:
            logger.error(f"Invalid OpenAI response: missing usage data - {openai_response}")
            raise ValueError("Invalid OpenAI response: missing usage data")

        choice = openai_response["choices"][0]
        message = choice.get("message", {})

        if not message:
            logger.error(f"Invalid OpenAI response: empty message - {openai_response}")
            raise ValueError("Invalid OpenAI response: empty message")

        content_blocks = []

        # Handle reasoning (OpenRouter format - may be string or in reasoning_details)
        # Add before text content to match Anthropic's thinking-first ordering
        reasoning_text = None
        reasoning_details = openai_response.get("reasoning_details") or message.get("reasoning_details")
        if reasoning_details and isinstance(reasoning_details, list):
            # Gemini format: list of dicts with type/text fields
            text_parts = []
            for item in reasoning_details:
                if isinstance(item, dict) and item.get("type") == "reasoning.text" and item.get("text"):
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            if text_parts:
                reasoning_text = "\n".join(text_parts)
        elif message.get("reasoning") and isinstance(message["reasoning"], str):
            # Simple string reasoning field
            reasoning_text = message["reasoning"]

        if reasoning_text:
            content_blocks.append(SimpleNamespace(
                type="thinking",
                thinking=reasoning_text,
                signature=None  # Stub - not validated downstream
            ))
            logger.debug(f"Parsed reasoning into thinking block ({len(reasoning_text)} chars)")

        # Add text content
        if message.get("content"):
            content_blocks.append(SimpleNamespace(
                type="text",
                text=message["content"]
            ))

        # Add tool calls (preserve IDs unchanged)
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                # OpenRouter omits 'arguments' entirely for no-parameter tools when proxying
                # Claude, instead of returning '{}' per OpenAI spec. Handle defensively.
                arguments_str = tc["function"].get("arguments", "{}")
                try:
                    arguments = json.loads(arguments_str) if arguments_str else {}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments: {arguments_str}")
                    arguments = {}
                content_blocks.append(SimpleNamespace(
                    type="tool_use",
                    id=tc["id"],
                    name=tc["function"]["name"],
                    input=arguments
                ))

        # Map finish reason to Anthropic stop_reason
        stop_reason_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens"
        }
        stop_reason = stop_reason_map.get(choice["finish_reason"], "end_turn")

        # Build usage object
        usage = {
            "input_tokens": openai_response["usage"]["prompt_tokens"],
            "output_tokens": openai_response["usage"]["completion_tokens"],
            "cache_creation_input_tokens": 0,  # N/A for OpenAI
            "cache_read_input_tokens": 0
        }

        logger.debug(f"Generic OpenAI response: {len(content_blocks)} blocks, {stop_reason}")
        return GenericOpenAIResponse(content_blocks, stop_reason, usage, reasoning_details=reasoning_details)

    def _handle_http_error(self, error: requests.HTTPError, error_body: Optional[dict] = None) -> NoReturn:
        """
        Map HTTP errors to exceptions that LLMProvider expects.

        Args:
            error: requests.HTTPError from failed request
            error_body: Optional parsed JSON error response

        Raises:
            PermissionError: For authentication failures
            ValueError: For context length exceeded errors
            RuntimeError: For rate limits and server errors
        """
        status = error.response.status_code
        fallback_text = error.response.text if error.response is not None else str(error)
        self._raise_provider_http_error(
            status=status,
            error_body=error_body,
            fallback_text=fallback_text,
            mode="non-streaming",
        )

    def _raise_provider_http_error(
        self,
        *,
        status: int,
        error_body: Optional[dict],
        fallback_text: str,
        mode: str,
    ) -> NoReturn:
        """Map provider HTTP status failures to typed exceptions."""
        error_message = GenericProviderErrorParser.extract_message(error_body, fallback_text)

        # Check for context length exceeded error (400 with specific error code)
        if status == 400 and error_body:
            error_info = error_body.get("error", {})
            error_code = error_info.get("code", "")

            if "context_length" in str(error_code) or "reduce the length" in error_message.lower():
                logger.error("Generic OpenAI client context length exceeded")
                # Import locally to avoid circular dependency
                from clients.llm_provider import ContextOverflowError
                raise ContextOverflowError(0, 200000, 'generic')

            # Handle tool validation failures - model tried to use unavailable tool
            # Groq returns: {"error": {"code": "tool_use_failed", "message": "attempted to call tool 'X' which was not in request.tools"}}
            if error_code == "tool_use_failed":
                import re
                match = re.search(r"attempted to call tool '(\w+)'", error_message)
                tool_name = match.group(1) if match else "unknown_tool"
                logger.info(f"Tool '{tool_name}' not loaded - raising ToolNotLoadedError for auto-load handling")
                raise ToolNotLoadedError(tool_name, error_message)

        if status == 401 or status == 403:
            logger.error(f"Generic OpenAI client authentication failed: {status}")
            raise PermissionError("Generic OpenAI client authentication failed")
        elif status == 429:
            logger.error("Generic OpenAI client rate limit exceeded")
            from clients.llm_provider import ProviderHTTPStatusError
            raise ProviderHTTPStatusError(self.endpoint, status, mode, error_message)
        elif status >= 500:
            logger.error(f"Generic OpenAI client server error: {status}")
            from clients.llm_provider import ProviderHTTPStatusError
            raise ProviderHTTPStatusError(self.endpoint, status, mode, error_message)
        else:
            logger.error(f"Generic OpenAI client API error: {status}")
            raise RuntimeError(f"Generic OpenAI client API error: {status} - {error_message}")
