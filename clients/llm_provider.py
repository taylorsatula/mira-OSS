"""
LLM Provider with Anthropic SDK and streaming events.

Provides a generator-based streaming API using native Anthropic SDK
with prompt caching, tool use, and type-safe message handling.

⚠️ IMPORTANT FOR CODE GENERATORS (CLAUDE):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS use LLMProvider.generate_response() for ALL LLM calls in application code.

This is the UNIVERSAL interface for:
- Anthropic API calls (default behavior, no overrides needed)
- Third-party providers (Groq, OpenRouter, local models) via overrides
- Emergency failover (automatic)

CORRECT - Third-party provider call:
    llm = LLMProvider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": "Hello"}],
        endpoint_url="https://openrouter.ai/api/v1/chat/completions",
        model_override="anthropic/claude-3-5-sonnet",
        api_key_override=api_key
    )

CORRECT - Anthropic API call:
    llm = LLMProvider()
    response = llm.generate_response(
        messages=[{"role": "user", "content": "Hello"}]
    )

WRONG - NEVER do this:
    from utils.generic_openai_client import GenericOpenAIClient
    client = GenericOpenAIClient(...)
    response = client.messages.create(...)

Benefits of using generate_response():
- Consistent interface across all providers
- Easy migration between providers (just change/omit overrides)
- Automatic error handling and failover
- Unified logging and monitoring
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import anthropic
import concurrent.futures
import contextvars
import json
import logging
import os
import requests
import threading
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Generator, Tuple

from config import config
from cns.core.stream_events import (
    StreamEvent, TextEvent, ThinkingEvent, ToolDetectedEvent, ToolExecutingEvent,
    ToolCompletedEvent, ToolErrorEvent, CompleteEvent, ErrorEvent,
    CircuitBreakerEvent, RetryEvent
)


class CircuitBreaker:
    """
    Simple circuit breaker for tool execution chains.
    Stops chains on:
    - Any tool error
    - Repeated identical results
    - Maximum iteration limit
    """

    def __init__(self, max_iterations: int = 10):
        self.max_iterations = max_iterations
        self.tool_results: List[Tuple[str, Optional[str], Optional[Exception]]] = []

    def record_execution(self, tool_name: str, result: Any, error: Optional[Exception] = None):
        """Record each tool execution"""
        result_hash = self._hash_result(result) if not error else None
        self.tool_results.append((tool_name, result_hash, error))

    def should_continue(self) -> Tuple[bool, str]:
        """Check if we should continue the tool chain"""
        if not self.tool_results:
            return True, "First tool"

        # Check for any errors
        if self.tool_results[-1][2] is not None:
            return False, f"Tool error: {self.tool_results[-1][2]}"

        # Check for repeated results (last 2 executions)
        if len(self.tool_results) >= 2:
            last_result = self.tool_results[-1][1]
            second_last_result = self.tool_results[-2][1]

            if last_result == second_last_result and last_result is not None:
                return False, "Repeated identical results"

        # Check max iterations
        if len(self.tool_results) >= self.max_iterations:
            return False, f"Reached maximum iterations ({self.max_iterations})"

        return True, "Continue"

    def _hash_result(self, result: Any) -> str:
        """Create a hash of the result for comparison"""
        # Convert result to string and hash it
        return hashlib.md5(str(result).encode()).hexdigest()


class GenericProviderClient:
    """
    Lightweight HTTP client for OpenAI-compatible API endpoints.

    Use this for third-party providers like OpenRouter, or any API
    that follows the OpenAI chat completions format.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_endpoint: str = "https://api.openai.com/v1/chat/completions",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout: int = 60
    ):
        """
        Initialize generic provider client.

        Args:
            api_key: API key for authentication
            model: Model identifier
            api_endpoint: Full URL for chat completions endpoint
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.model = model
        self.endpoint = api_endpoint
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make HTTP call to provider and return response.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Full API response dict

        Raises:
            requests.HTTPError: If request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def extract_text_content(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from API response.

        Args:
            response: Full API response dict

        Returns:
            Extracted text content or empty string
        """
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        return ""


class LLMProvider:
    """
    Generator-based LLM provider with clean streaming architecture.

    Yields StreamEvents during processing and returns the final response
    when generation is complete.
    """

    # Class-level failover state (shared across all instances)
    _failover_active = False
    _failover_lock = threading.Lock()
    _recovery_timer: Optional[threading.Timer] = None
    _logger = logging.getLogger("llm_provider")

    def __init__(self,
                 model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 timeout: Optional[int] = None,
                 api_key: Optional[str] = None,
                 tool_repo = None,
                 enable_prompt_caching: bool = True):
        """Initialize LLM provider with Anthropic SDK."""
        self.logger = logging.getLogger("llm_provider")

        # Apply configuration with overrides
        self.model = model if model is not None else config.api.model  # Reasoning model
        self.execution_model = config.api.execution_model  # Fast model for simple tools
        self.execution_endpoint = config.api.execution_endpoint  # OpenRouter endpoint for execution model
        self.simple_tools = set(config.api.simple_tools)  # Convert to set for fast lookup
        self.max_tokens = max_tokens if max_tokens is not None else config.api.max_tokens
        self.temperature = temperature if temperature is not None else config.api.temperature
        self.timeout = timeout if timeout is not None else config.api.timeout
        self.api_key = api_key if api_key is not None else config.api_key
        self.enable_prompt_caching = enable_prompt_caching

        # Get execution model API key from Vault
        try:
            from clients.vault_client import get_api_key
            self.execution_api_key = get_api_key(config.api.execution_api_key_name)
            if not self.execution_api_key:
                self.logger.warning(f"Execution model API key '{config.api.execution_api_key_name}' not found in Vault - execution model disabled")
        except Exception as e:
            self.logger.error(f"Failed to get execution model API key: {e}")
            self.execution_api_key = None

        # Extended thinking configuration
        self.extended_thinking = config.api_server.extended_thinking
        self.extended_thinking_budget = config.api_server.extended_thinking_budget

        # Initialize Anthropic SDK client
        self.anthropic_client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.timeout
        )

        # Initialize emergency fallback if enabled
        self.emergency_fallback_enabled = config.api.emergency_fallback_enabled
        self.emergency_fallback_api_key = None

        if self.emergency_fallback_enabled:
            try:
                from clients.vault_client import get_api_key
                self.emergency_fallback_api_key = get_api_key(config.api.emergency_fallback_api_key_name)
                if not self.emergency_fallback_api_key:
                    self.logger.warning(f"Emergency fallback API key not found - fallback disabled")
                    self.emergency_fallback_enabled = False
                else:
                    self.logger.info("Emergency fallback enabled")
            except Exception as e:
                self.logger.error(f"Failed to get emergency fallback API key: {e}")
                self.emergency_fallback_enabled = False

        # Optional tool repository for tool execution
        self.tool_repo = tool_repo
        self.max_tool_iterations = config.system.max_tool_iterations

        # Check for firehose mode
        self.firehose_enabled = getattr(config.system, '_firehose_enabled', False)

        self.logger.info(f"LLM Provider initialized with Anthropic SDK")
        self.logger.info(f"Reasoning model: {self.model}")
        self.logger.info(f"Execution model: {self.execution_model}")
        self.logger.info(f"Simple tools for execution model: {self.simple_tools}")
        if self.tool_repo:
            self.logger.info("Tool execution enabled")
        else:
            self.logger.info("Tool execution disabled (no tool_repo provided)")
        if self.enable_prompt_caching:
            self.logger.info("Prompt caching enabled")
        if self.emergency_fallback_enabled:
            self.logger.info("Emergency fallback enabled")
        if self.firehose_enabled:
            self.logger.info("Firehose mode enabled - logging to firehose_output.json")

    def _select_model(self, last_response: Optional[anthropic.types.Message]) -> str:
        """
        Select optimal model based on last response tool usage.

        Uses execution model (Haiku) for simple tool operations, reasoning model (Sonnet)
        for everything else.

        Args:
            last_response: Previous API response containing tool calls

        Returns:
            Model identifier to use for next API call
        """
        if last_response and last_response.stop_reason == "tool_use":
            tool_names = {b.name for b in last_response.content if b.type == "tool_use"}
            if tool_names and tool_names.issubset(self.simple_tools):
                self.logger.debug(f"Using execution model ({self.execution_model}) for simple tools: {tool_names}")
                return self.execution_model

        # Default to reasoning model for all other cases
        return self.model

    def _emit_events_from_response(self, response: anthropic.types.Message) -> Generator[StreamEvent, None, None]:
        """
        Emit events from a completed response.

        Args:
            response: Anthropic Message object with content blocks

        Yields:
            StreamEvent objects for text, tool_use, and completion
        """
        for block in response.content:
            if block.type == "text":
                yield TextEvent(content=block.text)
            elif block.type == "tool_use":
                yield ToolDetectedEvent(tool_name=block.name, tool_id=block.id)
        yield CompleteEvent(response=response)

    @classmethod
    def _activate_failover(cls):
        """
        Activate emergency failover for all users.

        Cancels any existing recovery timer and schedules a new one.
        """
        with cls._failover_lock:
            if cls._failover_active:
                # Cancel existing timer if any
                if cls._recovery_timer:
                    cls._recovery_timer.cancel()

            cls._failover_active = True
            cls._logger.warning("🚨 EMERGENCY FAILOVER ACTIVATED - All traffic routing to fallback provider")

            # Schedule recovery test
            recovery_minutes = config.api.emergency_fallback_recovery_minutes
            cls._recovery_timer = threading.Timer(
                recovery_minutes * 60,
                cls._test_recovery
            )
            cls._recovery_timer.daemon = True
            cls._recovery_timer.start()

            cls._logger.info(f"Recovery test scheduled in {recovery_minutes} minutes")

    @classmethod
    def _test_recovery(cls):
        """Test if Anthropic is back online by clearing failover flag."""
        with cls._failover_lock:
            cls._failover_active = False
            cls._logger.info("🔄 Testing Anthropic recovery - next request will try primary provider")

    @classmethod
    def _is_failover_active(cls) -> bool:
        """Check if failover is currently active."""
        return cls._failover_active

    def _prepare_messages(self, messages: List[Dict]) -> tuple[Union[str, List[Dict]], List[Dict]]:
        """
        Extract system prompt and prepare messages for Anthropic API.

        Anthropic requires system prompts as a separate parameter, not in messages array.

        Args:
            messages: OpenAI-format messages with possible system message

        Returns:
            Tuple of (system_content, anthropic_messages) where system_content can be
            str (simple prompt) or List[Dict] (structured blocks with cache_control)
        """
        system_content = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # Preserve type - can be str or List[Dict]
                system_content = msg["content"]
            else:
                # User and assistant messages keep same format
                anthropic_messages.append(msg)

        return system_content, anthropic_messages

    def _write_firehose(self, system_prompt: str, messages: List[Dict], tools: Optional[List[Dict]]) -> None:
        """Write request data to firehose_output.json for debugging."""
        if not self.firehose_enabled:
            return

        firehose_data = {
            "timestamp": time.time(),
            "model": self.model,
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": tools,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        try:
            with open("firehose_output.json", "w") as f:
                json.dump(firehose_data, f, indent=2)
            self.logger.debug("Wrote request data to firehose_output.json")
        except Exception as e:
            self.logger.error(f"Failed to write firehose output: {e}")

    def _prepare_tools_for_caching(self, tools: List[Dict]) -> List[Dict]:
        """
        Prepare tools for prompt caching by marking the last tool.

        Args:
            tools: Anthropic-format tool definitions

        Returns:
            Tools with cache control added to last tool if caching enabled
        """
        if not self.enable_prompt_caching or not tools:
            return tools

        # Mark last tool for caching (caches all tools)
        tools_copy = tools.copy()
        tools_copy[-1] = tools_copy[-1].copy()
        tools_copy[-1]["cache_control"] = {"type": "ephemeral"}
        return tools_copy

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        callback: Optional[Any] = None,
        endpoint_url: Optional[str] = None,
        model_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        thinking_enabled: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        **kwargs
    ) -> anthropic.types.Message:
        """
        Generate a response from the LLM using Anthropic SDK or generic provider.

        ⚠️ UNIVERSAL LLM INTERFACE - Use this for ALL LLM calls in application code!
        This method handles both Anthropic API calls and third-party providers through
        a unified interface. NEVER instantiate GenericOpenAIClient directly.

        Usage patterns:
        1. Anthropic API (default):
           response = llm.generate_response(messages=[...])

        2. Third-party provider (Groq, OpenRouter, etc):
           response = llm.generate_response(
               messages=[...],
               endpoint_url="https://openrouter.ai/api/v1/chat/completions",
               model_override="anthropic/claude-3-5-sonnet",
               api_key_override=api_key
           )

        - When stream=False (default): returns Anthropic Message object.
        - When stream=True: streams events to `callback` if provided, and
          returns the final Message object.
        - When endpoint_url is provided: routes to GenericOpenAIClient for that endpoint.

        Args:
            messages: Anthropic-format messages
            tools: Optional tool definitions
            stream: Enable streaming mode
            callback: Callback for streaming events
            endpoint_url: Optional custom OpenAI-compatible endpoint URL (for third-party providers)
            model_override: Optional model identifier override
            api_key_override: Optional API key override (REQUIRED when using endpoint_url)
            system_override: Optional system prompt override
            thinking_enabled: Override instance-level extended thinking setting
            thinking_budget: Override instance-level thinking budget tokens
            **kwargs: Additional parameters
        """
        # Pass thinking parameters through kwargs to internal methods
        if thinking_enabled is not None:
            kwargs['thinking_enabled'] = thinking_enabled
        if thinking_budget is not None:
            kwargs['thinking_budget'] = thinking_budget

        # Non-streaming path for simple consumers
        if not stream:
            return self._generate_non_streaming(
                messages, tools,
                endpoint_url=endpoint_url,
                model_override=model_override,
                api_key_override=api_key_override,
                system_override=system_override,
                **kwargs
            )

        # Streaming path: iterate events and forward to callback
        final_response: Optional[anthropic.types.Message] = None
        for event in self.stream_events(messages, tools, **kwargs):
            # Capture completion
            if isinstance(event, CompleteEvent):
                final_response = event.response

            if not callback:
                continue

            # Forward minimal event types expected by current consumers
            # Note: Callback failures are logged but don't stop the stream (user code shouldn't break LLM streaming)
            if isinstance(event, TextEvent):
                try:
                    callback({"type": "text", "content": event.content})
                except Exception as e:
                    self.logger.error(f"Callback failed on TextEvent: {e}", exc_info=True)
            elif isinstance(event, ThinkingEvent):
                try:
                    callback({"type": "thinking", "content": event.content})
                except Exception as e:
                    self.logger.error(f"Callback failed on ThinkingEvent: {e}", exc_info=True)
            elif isinstance(event, ToolDetectedEvent):
                try:
                    callback({"type": "tool_event", "event": "detected", "tool": event.tool_name})
                except Exception as e:
                    self.logger.error(f"Callback failed on ToolDetectedEvent: {e}", exc_info=True)
            elif isinstance(event, ToolExecutingEvent):
                try:
                    callback({"type": "tool_event", "event": "executing", "tool": event.tool_name})
                except Exception as e:
                    self.logger.error(f"Callback failed on ToolExecutingEvent: {e}", exc_info=True)
            elif isinstance(event, ToolCompletedEvent):
                try:
                    callback({"type": "tool_event", "event": "completed", "tool": event.tool_name})
                except Exception as e:
                    self.logger.error(f"Callback failed on ToolCompletedEvent: {e}", exc_info=True)
            elif isinstance(event, ToolErrorEvent):
                try:
                    callback({"type": "tool_event", "event": "failed", "tool": event.tool_name})
                except Exception as e:
                    self.logger.error(f"Callback failed on ToolErrorEvent: {e}", exc_info=True)

        if final_response is None:
            raise RuntimeError("No completion event received from stream")
        return final_response

    def stream_events(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[StreamEvent, None, None]:
        """Stream LLM events as a generator for real-time UIs."""
        try:
            # Validate messages before processing
            self._validate_messages(messages)

            # Route to appropriate handler
            if tools and self.tool_repo:
                yield from self._execute_with_tools(messages, tools, **kwargs)
            else:
                yield from self._stream_response(messages, tools, **kwargs)

        except Exception as e:
            self.logger.error(f"LLM API request failed: {e}")
            yield ErrorEvent(error=str(e), technical_details=repr(e))
            raise

    def _generate_non_streaming(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model_override: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        **kwargs
    ) -> anthropic.types.Message:
        """Non-streaming generation with Anthropic SDK or generic provider - returns Message object."""
        try:
            # Check failover FIRST - route to emergency fallback if active
            if self._is_failover_active() and self.emergency_fallback_api_key:
                self.logger.warning("Using emergency fallback (failover active)")
                endpoint_url = config.api.emergency_fallback_endpoint
                model_override = config.api.emergency_fallback_model
                api_key_override = self.emergency_fallback_api_key
                # Fall through to generic provider routing below

            # Route to generic provider if endpoint_url is provided
            if endpoint_url:
                # Require complete OpenAI request parameters
                if not model_override or not api_key_override:
                    raise ValueError(
                        "When using endpoint_url, both model_override and api_key_override must be provided. "
                        "Generic provider calls require explicit, complete parameters."
                    )

                self.logger.info(f"Routing to generic OpenAI-compatible endpoint: {endpoint_url} / {model_override}")

                # Create generic client instance (internal utility)
                from utils.generic_openai_client import GenericOpenAIClient

                generic_client = GenericOpenAIClient(
                    endpoint=endpoint_url,
                    api_key=api_key_override,
                    model=model_override,
                    timeout=self.timeout,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )

                # Prepare system prompt
                system_prompt = None
                if system_override:
                    system_prompt = system_override
                else:
                    system_prompt, messages = self._prepare_messages(messages)

                # Strip cache_control from tools (not supported in generic providers)
                generic_tools = None
                if tools:
                    generic_tools = [{k: v for k, v in tool.items() if k != "cache_control"} for tool in tools]

                # Call generic client
                return generic_client.messages.create(
                    messages=messages,
                    system=system_prompt,
                    tools=generic_tools,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )

            # If tools are enabled, reuse streaming pipeline and consume to completion
            if tools and self.tool_repo:
                final: Optional[anthropic.types.Message] = None
                for event in self.stream_events(messages, tools, **kwargs):
                    if isinstance(event, CompleteEvent):
                        final = event.response
                if final is None:
                    raise RuntimeError("No completion event received")
                return final

            # No tools: plain non-streaming API call
            # Validate messages before API call
            self._validate_messages(messages)

            # Select model (use override if provided, otherwise default to reasoning model)
            selected_model = model_override if model_override else self.model

            # Adjust max_tokens for model constraints (Haiku: 8192, Sonnet: 10000+)
            max_tokens = self.max_tokens
            if "haiku" in selected_model.lower() and max_tokens > 8192:
                max_tokens = 8192

            system_prompt, anthropic_messages = self._prepare_messages(messages)
            anthropic_tools = self._prepare_tools_for_caching(tools) if tools else None

            # Build system parameter based on content type
            if isinstance(system_prompt, list):
                # Already structured with cache_control
                system_param = system_prompt
            elif isinstance(system_prompt, str) and system_prompt:
                # Simple string - apply caching if enabled
                if self.enable_prompt_caching:
                    system_param = [{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }]
                else:
                    system_param = system_prompt
            else:
                system_param = None

            # Determine if thinking should be enabled (respect explicit override, otherwise use instance config)
            use_thinking = kwargs.get('thinking_enabled', self.extended_thinking and selected_model != self.execution_model)
            thinking_budget = kwargs.get('thinking_budget', self.extended_thinking_budget)

            # Add thinking budget to max_tokens when thinking is enabled
            if use_thinking:
                max_tokens = max_tokens + thinking_budget

            # Filter thinking blocks when thinking is disabled
            messages_to_send = anthropic_messages if use_thinking else [
                {**msg, "content": [b for b in msg["content"] if b.get("type") != "thinking"]}
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
                else msg
                for msg in anthropic_messages
            ]

            # Build API call parameters
            api_params = {
                "model": selected_model,
                "max_tokens": max_tokens,
                "messages": messages_to_send,
                "temperature": self.temperature
            }

            # Only include system if provided (Anthropic API rejects system: None)
            if system_param is not None:
                api_params["system"] = system_param

            # Only include tools if provided (Anthropic API rejects tools: None)
            if anthropic_tools:
                api_params["tools"] = anthropic_tools

            # Add extended thinking if enabled
            if use_thinking:
                api_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }

            message = self.anthropic_client.messages.create(**api_params)

            # Log cache usage if available
            if hasattr(message, 'usage') and message.usage:
                usage = message.usage
                self.logger.debug(
                    f"Model: {selected_model} - Token usage - Input: {usage.input_tokens}, Output: {usage.output_tokens}, "
                    f"Cache created: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
                    f"Cache read: {getattr(usage, 'cache_read_input_tokens', 0)}"
                )

            return message

        except anthropic.APITimeoutError:
            self.logger.error("Request timed out")
            raise TimeoutError("Request timed out")
        except anthropic.APIStatusError as e:
            # Yield error events for stream compatibility (empty generator)
            for _ in self._handle_anthropic_error(e):
                pass
        except anthropic.APIError as e:
            # ACTIVATE FAILOVER on Anthropic errors
            if self.emergency_fallback_enabled and self.emergency_fallback_api_key:
                self.logger.error(f"Anthropic error: {e} - activating emergency failover")
                self._activate_failover()
                return self._generate_non_streaming(
                    messages, tools,
                    endpoint_url=config.api.emergency_fallback_endpoint,
                    model_override=config.api.emergency_fallback_model,
                    api_key_override=self.emergency_fallback_api_key,
                    **kwargs
                )
            else:
                self.logger.error(f"API error: {e}")
                raise RuntimeError(f"API error: {e}")

    def _stream_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        model_override: Optional[str] = None,
        **kwargs
    ) -> Generator[StreamEvent, None, None]:
        """Execute streaming request with Anthropic SDK."""
        # Check failover FIRST - use non-streaming fallback if active
        if self._is_failover_active() and self.emergency_fallback_api_key:
            self.logger.warning("Using emergency fallback (failover active, non-streaming)")
            response = self._generate_non_streaming(
                messages, tools,
                endpoint_url=config.api.emergency_fallback_endpoint,
                model_override=config.api.emergency_fallback_model,
                api_key_override=self.emergency_fallback_api_key,
                **kwargs
            )
            yield from self._emit_events_from_response(response)
            return

        # Select model (use override if provided, otherwise default to reasoning model)
        selected_model = model_override if model_override else self.model

        # Route execution model to OpenRouter (non-streaming)
        if selected_model == self.execution_model and self.execution_api_key:
            self.logger.debug(f"Routing execution model to OpenRouter: {self.execution_model}")
            response = self._generate_non_streaming(
                messages, tools,
                endpoint_url=self.execution_endpoint,
                model_override=self.execution_model,
                api_key_override=self.execution_api_key,
                **kwargs
            )
            yield from self._emit_events_from_response(response)
            return

        # Adjust max_tokens for model constraints (Haiku: 8192, Sonnet: 10000+)
        max_tokens = self.max_tokens
        if "haiku" in selected_model.lower() and max_tokens > 8192:
            max_tokens = 8192

        # Prepare messages and tools
        system_prompt, anthropic_messages = self._prepare_messages(messages)
        anthropic_tools = self._prepare_tools_for_caching(tools) if tools else None

        # Write to firehose if enabled
        self._write_firehose(system_prompt, anthropic_messages, anthropic_tools)

        # Build system parameter based on content type
        if isinstance(system_prompt, list):
            # Already structured with cache_control
            system_param = system_prompt
        elif isinstance(system_prompt, str) and system_prompt:
            # Simple string - apply caching if enabled
            if self.enable_prompt_caching:
                system_param = [{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }]
            else:
                system_param = system_prompt
        else:
            system_param = None

        # Track tool uses for detection events
        tool_uses_seen = set()

        try:
            # Determine if thinking should be enabled (respect explicit override, otherwise use instance config)
            use_thinking = kwargs.get('thinking_enabled', self.extended_thinking and selected_model != self.execution_model)
            thinking_budget = kwargs.get('thinking_budget', self.extended_thinking_budget)

            # Add thinking budget to max_tokens when thinking is enabled
            if use_thinking:
                max_tokens = max_tokens + thinking_budget

            # Filter thinking blocks when thinking is disabled
            messages_to_send = anthropic_messages if use_thinking else [
                {**msg, "content": [b for b in msg["content"] if b.get("type") != "thinking"]}
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
                else msg
                for msg in anthropic_messages
            ]

            # Build API call parameters
            stream_params = {
                "model": selected_model,
                "max_tokens": max_tokens,
                "messages": messages_to_send,
                "temperature": self.temperature
            }

            # Only include system if provided (Anthropic API rejects system: None)
            if system_param is not None:
                stream_params["system"] = system_param

            # Only include tools if provided (Anthropic API rejects tools: None)
            if anthropic_tools:
                stream_params["tools"] = anthropic_tools

            # Add extended thinking if enabled
            if use_thinking:
                stream_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                }

            with self.anthropic_client.messages.stream(**stream_params) as stream:
                # Stream events
                for event in stream:
                    if event.type == "text":
                        yield TextEvent(content=event.text)

                    elif event.type == "content_block_delta":
                        # Handle thinking deltas from extended thinking
                        if hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                            if event.delta.type == "thinking_delta":
                                yield ThinkingEvent(content=event.delta.thinking)

                    elif event.type == "content_block_start":
                        # Emit tool detected events
                        if event.content_block.type == "tool_use":
                            tool_id = event.content_block.id
                            if tool_id not in tool_uses_seen:
                                tool_uses_seen.add(tool_id)
                                yield ToolDetectedEvent(
                                    tool_name=event.content_block.name,
                                    tool_id=tool_id
                                )

                # Get final message (Anthropic Message object)
                final_message = stream.get_final_message()

                # Log cache usage if available
                if hasattr(final_message, 'usage') and final_message.usage:
                    usage = final_message.usage
                    self.logger.debug(
                        f"Model: {selected_model} - Token usage - Input: {usage.input_tokens}, Output: {usage.output_tokens}, "
                        f"Cache created: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
                        f"Cache read: {getattr(usage, 'cache_read_input_tokens', 0)}"
                    )

                yield CompleteEvent(response=final_message)

        except anthropic.APITimeoutError:
            error_msg = f"Request timed out"
            self.logger.error(error_msg)
            yield ErrorEvent(error=error_msg)
            raise TimeoutError(error_msg)
        except anthropic.APIStatusError as e:
            yield from self._handle_anthropic_error(e)
        except anthropic.APIError as e:
            # ACTIVATE FAILOVER on Anthropic errors
            if self.emergency_fallback_enabled and self.emergency_fallback_api_key:
                self.logger.error(f"Anthropic error: {e} - activating emergency failover")
                self._activate_failover()
                response = self._generate_non_streaming(
                    messages, tools,
                    endpoint_url=config.api.emergency_fallback_endpoint,
                    model_override=config.api.emergency_fallback_model,
                    api_key_override=self.emergency_fallback_api_key,
                    **kwargs
                )
                yield from self._emit_events_from_response(response)
            else:
                error_msg = f"API error: {str(e)}"
                self.logger.error(error_msg)
                yield ErrorEvent(error=error_msg)
                raise RuntimeError(error_msg)

    def _execute_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Generator[StreamEvent, None, None]:
        """Execute with tool loop using Anthropic format with dynamic model selection."""
        circuit_breaker = CircuitBreaker(self.max_tool_iterations)
        current_messages = messages.copy()
        last_response = None  # Track previous response for model selection

        while True:
            # Select model based on previous response
            selected_model = self._select_model(last_response)

            # Stream LLM response with selected model
            response = None
            for event in self._stream_response(current_messages, tools, model_override=selected_model, **kwargs):
                if isinstance(event, CompleteEvent):
                    response = event.response
                else:
                    yield event

            if not response:
                yield ErrorEvent(error="No response received from LLM")
                return

            # Extract tool calls from Anthropic Message
            tool_calls = self.extract_tool_calls(response)

            # If no tool calls, we're done
            if not tool_calls:
                yield CompleteEvent(response=response)
                return

            # Add assistant message with tool calls
            assistant_message = self._build_assistant_message(response)
            current_messages.append(assistant_message)

            # Update last_response for next iteration's model selection
            last_response = response

            # Execute all tools in parallel
            # Emit executing events immediately
            for tool_call in tool_calls:
                yield ToolExecutingEvent(
                    tool_name=tool_call["tool_name"],
                    tool_id=tool_call["id"],
                    arguments=tool_call["input"]
                )

            # Execute tools concurrently
            tool_results = []
            # Capture context value to propagate to worker threads
            from utils.user_context import _user_context
            user_context_value = _user_context.get()

            def invoke_with_context(tool_name: str, tool_input: Dict[str, Any]):
                """Wrapper that sets context in worker thread without re-entry issues"""
                _user_context.set(user_context_value)
                return self.tool_repo.invoke_tool(tool_name, tool_input)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit all tool executions with manual context propagation
                future_to_tool = {
                    executor.submit(invoke_with_context, tc["tool_name"], tc["input"]): tc
                    for tc in tool_calls
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_call = future_to_tool[future]
                    tool_name = tool_call["tool_name"]
                    tool_id = tool_call["id"]

                    error = None
                    try:
                        result = future.result()
                        tool_result_content = str(result)

                        yield ToolCompletedEvent(
                            tool_name=tool_name,
                            tool_id=tool_id,
                            result=tool_result_content
                        )

                    except Exception as e:
                        self.logger.error(f"Tool execution failed for {tool_name}: {e}")
                        tool_result_content = f"Error: {e}"
                        error = e
                        result = None

                        yield ToolErrorEvent(
                            tool_name=tool_name,
                            tool_id=tool_id,
                            error=str(e)
                        )

                    # Record in circuit breaker
                    circuit_breaker.record_execution(tool_name, result, error)

                    # Store result for batching
                    tool_results.append({
                        "tool_use_id": tool_id,
                        "content": tool_result_content
                    })

            # Check circuit breaker after all tools complete
            should_continue, reason = circuit_breaker.should_continue()
            if not should_continue:
                self.logger.info(f"Circuit breaker triggered: {reason}")
                yield CircuitBreakerEvent(reason=reason)

                # Add all tool results in ONE user message
                current_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": r["content"]}
                        for r in tool_results
                    ]
                })

                # Get final response after circuit breaker
                for event in self._stream_response(current_messages, tools, model_override=self.model, **kwargs):
                    if isinstance(event, CompleteEvent):
                        yield event
                        return
                    else:
                        yield event

            # Add all tool results in ONE user message
            current_messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": r["content"]}
                    for r in tool_results
                ]
            })

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate messages before sending to API."""
        # Check for empty messages list
        if not messages:
            self.logger.error("Empty messages list detected")
            raise ValueError("Cannot send empty messages list to LLM API")

        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')

            # Allow assistant messages with tool calls but no content
            if role == 'assistant' and not content and msg.get('tool_calls'):
                continue

            # Block truly empty messages for all roles
            if not content or not str(content).strip():
                self.logger.error(f"Empty {role} message detected in continuum")
                raise ValueError(f"Cannot send empty {role} message to LLM API")

    def _build_assistant_message(self, message: anthropic.types.Message) -> Dict[str, Any]:
        """
        Build assistant message with Anthropic content blocks.

        Converts Anthropic Message to message dict suitable for continuum history.
        Preserves thinking blocks when extended thinking is enabled.

        Args:
            message: Anthropic Message object

        Returns:
            Message dict with role and content blocks
        """
        content_blocks = []

        for block in message.content:
            if block.type == "thinking":
                content_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature
                })
            elif block.type == "text":
                content_blocks.append({
                    "type": "text",
                    "text": block.text
                })
            elif block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })

        return {
            "role": "assistant",
            "content": content_blocks
        }

    def _handle_anthropic_error(self, error: anthropic.APIStatusError) -> Generator[StreamEvent, None, None]:
        """Handle Anthropic SDK errors and yield appropriate events."""
        status_code = error.status_code
        message = str(error)

        if status_code == 401:
            self.logger.error("Authentication failed")
            yield ErrorEvent(error="Authentication failed. Check your API key.")
            raise PermissionError("Invalid API key")
        elif status_code == 429:
            self.logger.error("Rate limit exceeded")
            yield ErrorEvent(error="Rate limit exceeded. Please try again later.")
            raise RuntimeError("Rate limit exceeded")
        elif status_code >= 500:
            self.logger.error(f"Server error: {message}")
            yield ErrorEvent(error=f"Server error: {message}")
            raise RuntimeError(f"Server error: {message}")
        else:
            self.logger.error(f"API error ({status_code}): {message}")
            yield ErrorEvent(error=f"API error ({status_code}): {message}")
            raise ValueError(f"API error ({status_code}): {message}")

    def extract_text_content(self, message: anthropic.types.Message) -> str:
        """Extract text content from Anthropic Message."""
        text_blocks = [block.text for block in message.content if block.type == "text"]
        return "".join(text_blocks)

    def extract_tool_calls(self, message: anthropic.types.Message) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Anthropic Message in standardized format.

        Returns list of dicts with keys: id, tool_name, input
        """
        return [
            {
                "id": block.id,
                "tool_name": block.name,
                "input": block.input  # Already parsed dict
            }
            for block in message.content
            if block.type == "tool_use"
        ]
