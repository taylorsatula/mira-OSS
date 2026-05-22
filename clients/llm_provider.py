"""
LLM Provider with Anthropic SDK and streaming events.

Provides a generator-based streaming API using native Anthropic SDK
with prompt caching, tool use, and type-safe message handling.

# TODO: Add generic provider failover (same pattern as Anthropic failover)
# When generic provider (Groq) returns 401/403 PermissionError:
# 1. Add _generic_failover_active class flag (like _failover_active)
# 2. In _generate_non_streaming, check flag before calling generic client
# 3. Catch PermissionError, activate flag, retry with emergency_fallback_endpoint
# 4. Add recovery timer to test if Groq is back
# This enables automatic Ollama fallback when Groq credentials are invalid/placeholder

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
import hashlib
import httpx
import json
import logging
import os
import random
import re
import signal
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Any, Literal, Optional, Union, Generator, Tuple, NamedTuple, TypedDict, TypeVar

from config import config

from cns.core.stream_events import (
    StreamEvent, TextEvent, ThinkingEvent, ToolDetectedEvent, ToolExecutingEvent,
    ToolCompletedEvent, ToolErrorEvent, CompleteEvent, ErrorEvent,
    CircuitBreakerEvent, FileArtifactEvent, GenerationCancelled, ProviderSwitchEvent,
)
from utils.user_context import check_cancelled, get_cancel_event
from tools.repo import ANTHROPIC_BETA_FLAGS

# Maximum file size for code execution artifacts (Anthropic server-enforced limit: 32MB)
MAX_FILE_ARTIFACT_SIZE = 32 * 1024 * 1024

# LLM traffic tap: toggle with `kill -USR1 <pid>`
from utils.llm_tap import toggle as _toggle_traffic_tap
signal.signal(signal.SIGUSR1, _toggle_traffic_tap)


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename from external source for safe filesystem storage."""
    safe = Path(filename).name                          # Strip directory components
    safe = safe.replace('\x00', '').replace('\n', '').replace('\r', '')  # Control chars
    safe = safe.lstrip('.')                             # No hidden files
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', safe)       # Allowlist chars
    if len(safe) > max_length:                          # Cap length
        stem, sep, ext = safe.rpartition('.')
        if sep and len(ext) + 1 < max_length:           # Has real extension that fits
            safe = stem[:max_length - len(ext) - 1] + '.' + ext
        else:                                            # No extension or ext too long
            safe = safe[:max_length]
    return safe or 'file'                               # Fallback


class ContextOverflowError(Exception):
    """
    Raised when request exceeds model context window.

    Used for both proactive (pre-flight estimation) and reactive (API error) detection.
    Enables structured retry logic with remediation strategies.
    """
    def __init__(self, estimated_tokens: int, context_window: int, provider: str):
        self.estimated_tokens = estimated_tokens
        self.context_window = context_window
        self.provider = provider
        super().__init__(f"Context overflow: ~{estimated_tokens} tokens vs {context_window} limit ({provider})")


class ProviderStallError(Exception):
    """Raised when an LLM provider accepts a request but produces no output."""

    def __init__(self, endpoint: str, timeout_seconds: int, mode: str):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.mode = mode
        super().__init__(
            f"Provider {endpoint} stalled for {timeout_seconds}s without producing output ({mode})"
        )


class ProviderHTTPStatusError(Exception):
    """Raised when a provider returns a retryable HTTP status."""

    def __init__(self, endpoint: str, status_code: int, mode: str, message: str):
        self.endpoint = endpoint
        self.status_code = status_code
        self.mode = mode
        self.provider_message = message
        super().__init__(
            f"Provider {endpoint} returned HTTP {status_code} during {mode}: {message}"
        )


T = TypeVar("T")


# Retry configuration for transient API errors (overloaded_error)
OVERLOAD_MAX_RETRIES = 3
OVERLOAD_BASE_DELAY = 1.0  # seconds
OVERLOAD_MAX_DELAY = 8.0   # seconds


class ToolCall(TypedDict):
    """Standardized tool call extracted from Anthropic Message."""
    id: str
    tool_name: str
    input: Dict[str, Any]


class ToolExecution(NamedTuple):
    """Record of a single tool execution for circuit breaker tracking."""
    tool_name: str
    result_hash: Optional[str]
    error: Optional[Exception]


class ToolExecutionResult(NamedTuple):
    """Result of executing a single tool."""
    tool_call: ToolCall
    result_content: Union[str, List[Dict[str, Any]]]
    raw_result: Any
    error: Optional[Exception]


def _displayable_result(
    result_content: Union[str, List[Dict[str, Any]]], raw_result: Any
) -> Union[str, List[Dict[str, Any]]]:
    """Pass through tool result content for persistence and downstream use.

    Content blocks (images + text) flow through as-is so they persist to
    conversation history and remain visible to the LLM on subsequent turns.
    String results pass through unchanged.
    """
    return result_content


# Maps named effort levels to token budgets for non-adaptive models
EFFORT_BUDGET_MAP: Dict[str, int] = {
    "low": 1024,
    "medium": 2048,
    "high": 8192,
    "max": 31999,  # Anthropic max budget_tokens ceiling
}


def _uses_adaptive_thinking(model: str) -> bool:
    """Claude 4.6+ uses adaptive thinking; older models require token budgets.

    4.6+ rejects budget_tokens with a 400 error — must use type: "adaptive"
    with output_config.effort instead. Matches the -4-N version suffix where N >= 6.
    """
    m = re.search(r'-4-(\d+)', model)
    return m is not None and int(m.group(1)) >= 6


@dataclass(frozen=True)
class ThinkingConfig:
    """Caller's intent for thinking/effort on a single LLM request.

    Two mutually exclusive modes:
    - effort: Named level ("low"/"medium"/"high"/"max") — translated per-backend
    - budget_tokens: Exact token count — always uses type: "enabled"
    - Both None: no thinking
    """
    effort: Optional[Literal["low", "medium", "high", "max"]] = None
    budget_tokens: Optional[int] = None

    def __post_init__(self):
        if self.effort is not None and self.budget_tokens is not None:
            raise ValueError("Mutually exclusive: set effort or budget_tokens, not both")

    @property
    def active(self) -> bool:
        return self.effort is not None or self.budget_tokens is not None

    def resolved_budget(self) -> int:
        """Effective budget in tokens. Only valid when active."""
        if self.budget_tokens is not None:
            return self.budget_tokens
        return EFFORT_BUDGET_MAP[self.effort]


def build_batch_params(
    purpose: str,
    system_prompt: str,
    messages: List[Dict[str, Any]],
    *,
    cache_ttl: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build Anthropic Batch API params dict from internal_llm config.

    All LLM-tuning params (model, max_tokens, effort) are resolved from
    InternalLLMConfig — callers just pass the purpose key.

    Two states:
    - effort set: send thinking/effort params (model-era-aware translation)
    - effort NULL: vanilla API call — no thinking, no effort. Correct for
      models that don't support effort (Haiku 4.5, Llama, Gemini Flash).

    Args:
        purpose: Internal LLM config key (e.g., 'extraction', 'relationship')
        system_prompt: System prompt text
        messages: Anthropic-format messages list
        cache_ttl: Optional cache TTL (e.g., "1h" for 1-hour at 2x cost).
            Omit for default 5-minute ephemeral caching.

    Returns:
        Complete params dict ready for {"custom_id": ..., "params": params}
    """
    from utils.user_context import get_internal_llm
    llm_cfg = get_internal_llm(purpose)

    cache_control: Dict[str, str] = {"type": "ephemeral"}
    if cache_ttl:
        cache_control["ttl"] = cache_ttl

    params: Dict[str, Any] = {
        "model": llm_cfg.model,
        "max_tokens": llm_cfg.max_tokens,
        "system": [{"type": "text", "text": system_prompt, "cache_control": cache_control}],
        "messages": messages,
    }

    if llm_cfg.effort:
        if _uses_adaptive_thinking(llm_cfg.model):
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            params["output_config"] = {"effort": llm_cfg.effort}
        else:
            budget = EFFORT_BUDGET_MAP[llm_cfg.effort]
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}
    # No else — vanilla API call for models without effort support

    return params


def _is_overloaded_error(error: Exception) -> bool:
    """Check if an exception is an Anthropic overloaded error (transient, should retry)."""
    error_str = str(error).lower()
    return "overloaded" in error_str or "overloaded_error" in error_str


class CircuitBreaker:
    """
    Simple circuit breaker for tool execution chains.
    Stops chains on:
    - Any tool error
    - Repeated identical results (loop detection)
    """

    def __init__(self):
        self.tool_results: List[ToolExecution] = []

    def record_execution(self, tool_name: str, result: Any, error: Optional[Exception] = None):
        """Record each tool execution"""
        result_hash = self._hash_result(result) if not error else None
        self.tool_results.append(ToolExecution(tool_name, result_hash, error))

    def should_continue(self) -> Tuple[bool, str]:
        """Check if we should continue the tool chain"""
        if not self.tool_results:
            return True, "First tool"

        last = self.tool_results[-1]

        # Check for errors - allow ONE retry per tool before triggering
        if last.error is not None:
            # Count previous errors for this SAME tool
            prior_errors = sum(1 for ex in self.tool_results[:-1]
                              if ex.tool_name == last.tool_name and ex.error is not None)
            if prior_errors > 0:
                # Second failure for same tool - give up
                return False, f"Tool '{last.tool_name}' failed after correction attempt: {last.error}"
            # First failure - allow retry (model will see schema in error message)

        # Check for repeated results (last 2 executions of SAME tool) - loop detection
        if len(self.tool_results) >= 2:
            current = self.tool_results[-1]
            previous = self.tool_results[-2]

            if (current.tool_name == previous.tool_name
                    and current.result_hash == previous.result_hash
                    and current.result_hash is not None):
                return False, "Repeated identical results"

        return True, "Continue"

    def _hash_result(self, result: Any) -> str:
        """Create a hash of the result for comparison"""
        # Convert result to string and hash it
        return hashlib.md5(str(result).encode()).hexdigest()


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
                 model: str = config.api.model,
                 max_tokens: int = config.api.max_tokens,
                 temperature: float = config.api.temperature,
                 timeout: int = config.api.timeout,
                 api_key: Optional[str] = None,
                 tool_repo: Optional["ToolRepository"] = None,
                 enable_prompt_caching: bool = True):
        """Initialize LLM provider with Anthropic SDK."""
        self.logger = logging.getLogger("llm_provider")

        # Apply configuration with overrides
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else config.api_key
        self.enable_prompt_caching = enable_prompt_caching

        # Initialize Anthropic SDK client
        self.anthropic_client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
        )

        # Attach request logging hooks (req_id → content mapping)
        from utils.logging_config import instrument_anthropic_client
        instrument_anthropic_client(self.anthropic_client)

        # Initialize emergency fallback if enabled
        self.emergency_fallback_enabled = config.api.emergency_fallback_enabled
        self.emergency_fallback_api_key = None

        if self.emergency_fallback_enabled:
            # API key is optional for local providers like Ollama
            if config.api.emergency_fallback_api_key_name:
                try:
                    from clients.vault_client import get_api_key
                    self.emergency_fallback_api_key = get_api_key(config.api.emergency_fallback_api_key_name)
                    if not self.emergency_fallback_api_key:
                        self.logger.warning(f"Emergency fallback API key '{config.api.emergency_fallback_api_key_name}' not found in Vault")
                except Exception as e:
                    self.logger.warning(f"Failed to get emergency fallback API key: {e}")

            # Log fallback configuration
            if self.emergency_fallback_api_key:
                self.logger.info(f"Emergency fallback enabled with API key")
            else:
                self.logger.info(f"Emergency fallback enabled (no API key - local provider: {config.api.emergency_fallback_endpoint})")

        # Optional tool repository for tool execution
        self.tool_repo = tool_repo

        # Firehose mode checked live at call time via os.environ.get('MIRA_FIREHOSE')

        # Consolidated initialization log
        features = []
        if self.tool_repo: features.append("tools")
        if self.enable_prompt_caching: features.append("cache")
        if self.emergency_fallback_enabled: features.append("fallback")

        self.logger.info(
            f"LLMProvider: model={self.model}, features={','.join(features) if features else 'none'}"
        )

    def download_file_artifact(self, file_id: str, filename: str, mime_type: str, size_bytes: int) -> Path:
        """Download file from Anthropic Files API to user's artifacts directory.

        Files are stored with opaque names ({uuid4_hex}.bin) to prevent filename-based
        attacks. Original filename is preserved in a .meta sidecar for Content-Disposition
        at download time.

        Args:
            file_id: Anthropic file ID
            filename: Original filename from Anthropic metadata
            mime_type: MIME type from Anthropic metadata
            size_bytes: File size in bytes

        Returns:
            Path to the downloaded .bin file

        Raises:
            ValueError: If file exceeds MAX_FILE_ARTIFACT_SIZE
        """
        from utils.user_context import get_current_user_id
        from tools.repo import FILES_API_BETA_FLAG
        from uuid import uuid4

        if size_bytes > MAX_FILE_ARTIFACT_SIZE:
            raise ValueError(f"File too large: {size_bytes} bytes (max {MAX_FILE_ARTIFACT_SIZE})")

        user_id = str(get_current_user_id())
        artifacts_dir = Path("data/users") / user_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = sanitize_filename(filename)

        response = self.anthropic_client.beta.files.download(
            file_id, betas=[FILES_API_BETA_FLAG]
        )

        # Store in file_id subdirectory with opaque name
        file_dir = artifacts_dir / file_id
        file_dir.mkdir(exist_ok=True)
        random_stem = uuid4().hex

        content_path = file_dir / f"{random_stem}.bin"
        content_path.resolve().relative_to(file_dir.resolve())  # Path traversal guard
        content_path.write_bytes(response.read())

        # Metadata sidecar for original filename at download time
        meta_path = file_dir / f"{random_stem}.meta"
        meta_path.write_text(json.dumps({"filename": safe_filename, "mime_type": mime_type}))

        self.logger.info(f"Downloaded file artifact {file_id} ({safe_filename}, {size_bytes} bytes)")
        return content_path

    def _emit_server_tool_completed_events(self, content_blocks: list) -> Generator[StreamEvent, None, None]:
        """Emit ToolCompletedEvent for server-side code_execution results.

        Code execution is executed by Anthropic's sandbox and its result appears
        as a BetaCodeExecutionToolResultBlock envelope at the top level of the
        response content. The orchestrator needs ToolCompletedEvent to fire
        downstream completion logging and UI stream events.
        """
        from anthropic.types.beta import BetaCodeExecutionResultBlock

        server_tool_map: dict[str, str] = {}
        for block in content_blocks:
            if block.type == "server_tool_use":
                server_tool_map[block.id] = block.name

        for block in content_blocks:
            if block.type != "code_execution_tool_result":
                continue
            tool_id = getattr(block, 'tool_use_id', '')
            tool_name = server_tool_map.get(tool_id, "code_execution")
            if isinstance(block.content, BetaCodeExecutionResultBlock):
                inner = block.content
                file_count = sum(1 for ob in inner.content if getattr(ob, 'file_id', None))
                result_summary = (
                    f"[code_execution: rc={inner.return_code}, "
                    f"stdout={len(inner.stdout or '')}B, stderr={len(inner.stderr or '')}B, "
                    f"files={file_count}]"
                )
            else:
                err = getattr(block.content, 'error_code', 'unknown')
                result_summary = f"[code_execution error: {err}]"
            yield ToolCompletedEvent(
                tool_name=tool_name,
                tool_id=tool_id,
                result=result_summary,
            )

    def _extract_file_artifacts(self, content_blocks: list) -> Generator[StreamEvent, None, None]:
        """Extract file_ids from code_execution_tool_result envelopes and
        download them to the user's artifacts directory.

        The SDK wraps code execution results in a two-level envelope:
        BetaCodeExecutionToolResultBlock (outer, type='code_execution_tool_result')
        → Union[BetaCodeExecutionToolResultError, BetaCodeExecutionResultBlock] (inner)
        → List[BetaCodeExecutionOutputBlock] on the success variant.
        We unwrap by isinstance against the inner success type — any future
        union expansion surfaces as a type error rather than silent misclassification.
        """
        from anthropic.types.beta import BetaCodeExecutionResultBlock
        from tools.repo import FILES_API_BETA_FLAG

        for block in content_blocks:
            if block.type != "code_execution_tool_result":
                continue
            if not isinstance(block.content, BetaCodeExecutionResultBlock):
                continue  # error variant — nothing to extract
            for ob in block.content.content:
                fid = getattr(ob, 'file_id', None)
                if not fid:
                    continue
                try:
                    meta = self.anthropic_client.beta.files.retrieve_metadata(
                        fid, betas=[FILES_API_BETA_FLAG]
                    )
                    self.download_file_artifact(fid, meta.filename, meta.mime_type, meta.size_bytes)
                    yield FileArtifactEvent(
                        file_id=fid,
                        filename=meta.filename,
                        mime_type=meta.mime_type,
                        size_bytes=meta.size_bytes,
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to process file artifact {fid}: {e}")

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
            cls._logger.toast("🔄 Anthropic recovery initiated - testing primary provider")

    @classmethod
    def _is_failover_active(cls) -> bool:
        """Check if failover is currently active."""
        return cls._failover_active

    def _run_with_response_timeout(
        self,
        operation: Callable[[], T],
        *,
        endpoint: Optional[str],
        mode: str,
    ) -> T:
        """Run a provider operation with time-to-output stall detection."""
        timeout_seconds = config.api.provider_response_timeout
        result: list[T] = []
        errors: list[BaseException] = []

        def invoke() -> None:
            try:
                result.append(operation())
            except BaseException as e:
                errors.append(e)

        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout=timeout_seconds)

        if worker.is_alive():
            raise ProviderStallError(endpoint or "anthropic", timeout_seconds, mode)
        if errors:
            raise errors[0]
        return result[0]

    def _prepare_messages(self, messages: List[Dict]) -> tuple[Optional[Union[str, List[Dict]]], List[Dict]]:
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
                if system_content is not None:
                    raise ValueError("Multiple system messages found — only one is supported")
                # Preserve type - can be str or List[Dict]
                system_content = msg["content"]
            else:
                # User and assistant messages keep same format
                anthropic_messages.append(msg)

        return system_content, anthropic_messages

    def _strip_container_uploads_from_messages(
        self,
        messages: List[Dict]
    ) -> List[Dict]:
        """
        Convert container_upload blocks to text warnings for generic providers.

        Generic providers (Groq, OpenRouter) don't support Files API or
        container_upload blocks. This strips them out and replaces with
        warning text.

        Args:
            messages: Messages with potential container_upload blocks

        Returns:
            Messages with container_upload blocks replaced by text warnings
        """
        stripped = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                new_blocks = []
                for block in msg["content"]:
                    if block.get("type") == "container_upload":
                        file_id = block.get("source", {}).get("file_id", "unknown")
                        new_blocks.append({
                            "type": "text",
                            "text": f"[File upload not supported by this provider: {file_id}]"
                        })
                    else:
                        new_blocks.append(block)
                stripped.append({**msg, "content": new_blocks})
            else:
                stripped.append(msg)
        return stripped

    def _bill_cancelled_stream(
        self,
        stream: object,
        streamed_output_chars: int,
        model: str,
        endpoint_url: Optional[str],
        allow_negative: bool,
    ) -> None:
        """Record billing for a cancelled stream using snapshot input tokens and estimated output tokens."""
        try:
            from billing import get_billing_backend, UsageRecord
            from billing.pricing import resolve_pricing_key
            from utils.user_context import get_current_user_id, has_user_context

            if not has_user_context():
                return

            snapshot = getattr(stream, 'current_message_snapshot', None)
            if not snapshot or not getattr(snapshot, 'usage', None):
                return

            pricing_key = resolve_pricing_key(model, endpoint_url)
            if not pricing_key:
                return

            estimated_output = max(streamed_output_chars // 4, 1)
            billing = get_billing_backend()
            billing.record_usage(
                str(get_current_user_id()),
                UsageRecord(
                    pricing_key=pricing_key,
                    model=model,
                    input_tokens=snapshot.usage.input_tokens,
                    output_tokens=estimated_output,
                    cache_read_tokens=getattr(snapshot.usage, 'cache_read_input_tokens', 0) or 0,
                    cache_write_tokens=getattr(snapshot.usage, 'cache_creation_input_tokens', 0) or 0,
                ),
                allow_negative=allow_negative,
            )
            self.logger.info(
                f"Cancelled generation billed: in={snapshot.usage.input_tokens}, "
                f"out~={estimated_output} (from {streamed_output_chars} chars)"
            )
        except ImportError:
            pass  # OSS mode
        except Exception as e:
            self.logger.error(f"Billing on cancel failed: {e}", exc_info=True)

        # Cost accumulator hook (independent of billing module — active when a
        # request handler has opted into per-request cost feedback).
        try:
            from utils import cost_accumulator
            snapshot = getattr(stream, 'current_message_snapshot', None)
            if cost_accumulator.is_active() and snapshot and getattr(snapshot, 'usage', None):
                cost_accumulator.record(
                    internal_llm_name=None,  # stream cancellation only occurs on user-tier chat
                    model=model,
                    input_tokens=snapshot.usage.input_tokens,
                    output_tokens=max(streamed_output_chars // 4, 1),
                    cache_read_tokens=getattr(snapshot.usage, 'cache_read_input_tokens', 0) or 0,
                    cache_write_tokens=getattr(snapshot.usage, 'cache_creation_input_tokens', 0) or 0,
                )
        except Exception as e:
            self.logger.debug(f"cost_accumulator record failed (non-fatal): {e}")

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
        tools_copy = [t.copy() for t in tools]
        tools_copy[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        return tools_copy

    def _anthropic_thinking_params(self, thinking: ThinkingConfig, model: str) -> Tuple[Dict, int]:
        """Translate ThinkingConfig to Anthropic API params.

        Returns:
            (api_params_dict, max_tokens_adjustment) — for 4.6 models the adjustment
            is 0 (adaptive thinking doesn't need budget headroom); for older models
            the adjustment equals the budget so max_tokens accommodates thinking.
        """
        if not thinking.active:
            return {}, 0

        if _uses_adaptive_thinking(model):
            # Claude 4.6+: adaptive thinking with native effort level.
            # 4.7+ rejects budget_tokens entirely (400 error) and omits
            # thinking content by default — "summarized" opts in.
            effort = thinking.effort or "high"  # Default when only budget_tokens set
            return {
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": effort}
            }, 0
        else:
            # Older Anthropic models: explicit token budget
            budget = thinking.resolved_budget()
            return {
                "thinking": {"type": "enabled", "budget_tokens": budget}
            }, budget

    def _generic_thinking_params(self, thinking: ThinkingConfig) -> Tuple[Optional[str], int]:
        """Translate ThinkingConfig to generic OpenAI-compatible provider params.

        Returns:
            (effort_string_or_None, max_tokens_adjustment) — effort string flows
            to the reasoning payload. max_tokens adjustment is only non-zero when
            the caller specified an explicit budget_tokens (provider may deduct
            thinking from max_tokens). When using effort mode, the provider manages
            thinking tokens via the reasoning payload — no inflation needed.
        """
        if not thinking.active:
            return None, 0
        effort = thinking.effort or "high"  # Default when only budget_tokens set
        budget = thinking.budget_tokens or 0
        return effort, budget

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        callback: Optional[Callable[[dict[str, str]], None]] = None,
        internal_llm: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model_override: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        effort: Optional[Literal["low", "medium", "high", "max"]] = None,
        container_id: Optional[str] = None,
        thinking_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        allow_negative: bool = False,
    ) -> anthropic.types.Message:
        """
        Generate a response from the LLM using Anthropic SDK or generic provider.

        ⚠️ UNIVERSAL LLM INTERFACE - Use this for ALL LLM calls in application code!
        This method handles both Anthropic API calls and third-party providers through
        a unified interface. NEVER instantiate GenericOpenAIClient directly.

        Usage patterns:
        1. Anthropic API (default):
           response = llm.generate_response(messages=[...])

        2. Internal LLM config (resolves endpoint/model/key from database schema):
           response = llm.generate_response(messages=[...], internal_llm="summary")

        3. Third-party provider (Groq, OpenRouter, etc) — one-off overrides:
           response = llm.generate_response(
               messages=[...],
               endpoint_url="https://openrouter.ai/api/v1/chat/completions",
               model_override="anthropic/claude-3-5-sonnet",
               api_key_override=api_key
           )

        4. With thinking (token budget):
           response = llm.generate_response(messages=[...], thinking_tokens=4096)

        5. With thinking (named effort level):
           response = llm.generate_response(messages=[...], effort="high")

        - When stream=False (default): returns Anthropic Message object.
        - When stream=True: streams events to `callback` if provided, and
          returns the final Message object.
        - When endpoint_url is provided: routes to GenericOpenAIClient for that endpoint.

        Args:
            messages: Anthropic-format messages
            tools: Optional tool definitions
            stream: Enable streaming mode
            callback: Callback for streaming events
            internal_llm: Config key resolving endpoint/model/key from internal_llm schema.
                Mutually exclusive with endpoint_url/model_override/api_key_override.
            endpoint_url: Optional custom OpenAI-compatible endpoint URL (for third-party providers)
            model_override: Optional model identifier override
            api_key_override: Optional API key override (REQUIRED when using endpoint_url)
            system_override: Optional system prompt override
            effort: Named thinking effort level — translated per-backend
            thinking_tokens: Exact thinking token budget — always uses type: "enabled"
            container_id: Optional container ID to reuse (for multi-turn file access)
            temperature: Per-call temperature override (defaults to self.temperature)
            max_tokens: Per-call max_tokens override (defaults to self.max_tokens)
            allow_negative: Allow billing balance to go negative (for system tasks)
        """
        # Resolve internal_llm config into endpoint/model/key + LLM-tuning params
        _llm_cfg = None
        if internal_llm is not None:
            if endpoint_url is not None or model_override is not None or api_key_override is not None:
                raise ValueError(
                    "internal_llm= is mutually exclusive with endpoint_url/model_override/api_key_override"
                )
            from utils.user_context import get_internal_llm
            from clients.vault_client import get_api_key
            _llm_cfg = get_internal_llm(internal_llm)
            endpoint_url = _llm_cfg.endpoint_url
            model_override = _llm_cfg.model
            if _llm_cfg.api_key_name:
                api_key_override = get_api_key(_llm_cfg.api_key_name)
            # Auto-resolve max_tokens (caller can still override)
            # temperature: not stored in DB — instance default (1.0) or caller override
            if max_tokens is None:
                max_tokens = _llm_cfg.max_tokens

        # Resolve thinking config — priority: thinking_tokens > effort > DB config > inactive
        if thinking_tokens is not None:
            thinking = ThinkingConfig(budget_tokens=thinking_tokens)
        elif effort is not None:
            thinking = ThinkingConfig(effort=effort)
        elif _llm_cfg and _llm_cfg.effort:
            thinking = ThinkingConfig(effort=_llm_cfg.effort)
        else:
            thinking = ThinkingConfig()

        # Non-streaming path for simple consumers
        if not stream:
            return self._generate_non_streaming(
                messages, tools,
                endpoint_url=endpoint_url,
                model_override=model_override,
                api_key_override=api_key_override,
                system_override=system_override,
                thinking=thinking,
                temperature=temperature,
                max_tokens=max_tokens,
                container_id=container_id,
                allow_negative=allow_negative,
                internal_llm_name=_llm_cfg.name if _llm_cfg else None,
            )

        # Streaming path
        final_response: Optional[anthropic.types.Message] = None
        for event in self.stream_events(
            messages, tools,
            thinking=thinking,
            temperature=temperature,
            max_tokens=max_tokens,
            container_id=container_id,
            endpoint_url=endpoint_url,
            model_override=model_override,
            api_key_override=api_key_override,
            system_override=system_override,
            allow_negative=allow_negative,
        ):
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
        *,
        effort: Optional[Literal["low", "medium", "high", "max"]] = None,
        thinking_tokens: Optional[int] = None,
        thinking: Optional[ThinkingConfig] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        container_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model_override: Optional[str] = None,
        model_preference: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        allow_negative: bool = False,
    ) -> Generator[StreamEvent, None, None]:
        """Stream LLM events as a generator for real-time UIs.

        Self-sufficient entry point — resolves thinking config from
        effort/thinking_tokens and accepts all LLM-tuning params as
        explicit keyword arguments. Direct callers (e.g. orchestrator)
        get the same resolution that generate_response() provides.
        """
        # Resolve thinking config — priority: pre-built > thinking_tokens > effort > inactive
        if thinking is None:
            if thinking_tokens is not None:
                thinking = ThinkingConfig(budget_tokens=thinking_tokens)
            elif effort is not None:
                thinking = ThinkingConfig(effort=effort)
            else:
                thinking = ThinkingConfig()

        try:
            # Validate messages before processing
            self._validate_messages(messages)
            if endpoint_url:
                # Agentic loop for generic providers with real-time streaming
                current_messages = list(messages)  # Copy to avoid mutating original
                circuit_breaker = CircuitBreaker()

                # Import streaming dependencies
                from utils.generic_openai_client import GenericOpenAIClient, GenericOpenAIResponse
                from types import SimpleNamespace

                while True:
                    check_cancelled()

                    # Prepare for streaming
                    system_prompt, prepared_messages = self._prepare_messages(current_messages)

                    # Strip unsupported features from tools
                    generic_tools = None
                    if tools:
                        filtered_tools = [t for t in tools if t.get("type") != "code_execution_20250825"]
                        generic_tools = [{k: v for k, v in t.items() if k != "cache_control"} for t in filtered_tools]

                    # Create generic client for streaming (resolve per-call overrides)
                    effective_temperature = temperature if temperature is not None else self.temperature
                    effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
                    generic_client = GenericOpenAIClient(
                        endpoint=endpoint_url,
                        model=model_override,
                        api_key=api_key_override,
                        timeout=self.timeout,
                        max_tokens=effective_max_tokens,
                        temperature=effective_temperature
                    )

                    thinking_effort, thinking_budget = self._generic_thinking_params(thinking)

                    # Stream response with real-time event emission
                    accumulated_text = ""
                    accumulated_tool_calls = {}  # {index: {"id": ..., "name": ..., "arguments": ""}}
                    accumulated_reasoning_details = []  # Required for OpenRouter reasoning model tool calling
                    finish_reason = None
                    stream_usage = None  # Captured from final chunk (stream_options.include_usage)

                    chunk_iterator = iter(generic_client.messages.create_streaming(
                        messages=prepared_messages,
                        system=system_prompt,
                        tools=generic_tools,
                        max_tokens=effective_max_tokens,
                        temperature=effective_temperature,
                        thinking_effort=thinking_effort,
                        thinking_budget=thinking_budget
                    ))
                    while True:
                        try:
                            chunk = self._run_with_response_timeout(
                                lambda: next(chunk_iterator),
                                endpoint=endpoint_url,
                                mode="streaming",
                            )
                        except StopIteration:
                            break
                        # Capture usage from final chunk (OpenRouter includes automatically;
                        # other providers need stream_options.include_usage)
                        if chunk.get("usage"):
                            u = chunk["usage"]
                            prompt_details = u.get("prompt_tokens_details") or {}
                            stream_usage = {
                                "input_tokens": u.get("prompt_tokens", 0),
                                "output_tokens": u.get("completion_tokens", 0),
                                "cache_creation_input_tokens": prompt_details.get("cache_write_tokens", 0) or 0,
                                "cache_read_input_tokens": prompt_details.get("cached_tokens", 0) or 0
                            }

                        if not chunk.get("choices"):
                            continue

                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason

                        # Stream text content in real-time
                        if delta.get("content"):
                            yield TextEvent(content=delta["content"])
                            accumulated_text += delta["content"]

                        # Stream reasoning/thinking content in real-time
                        if delta.get("reasoning"):
                            yield ThinkingEvent(content=delta["reasoning"])

                        # Accumulate reasoning_details for round-trip (required by OpenRouter reasoning models)
                        # These must be passed back unmodified when returning tool results
                        if delta.get("reasoning_details"):
                            accumulated_reasoning_details.extend(delta["reasoning_details"])

                        # Handle tool calls - accumulate arguments across chunks
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                idx = tc["index"]
                                if idx not in accumulated_tool_calls:
                                    # New tool call - initialize and emit ToolDetectedEvent
                                    accumulated_tool_calls[idx] = {
                                        "id": tc.get("id", ""),
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": ""
                                    }
                                    if accumulated_tool_calls[idx]["name"]:
                                        yield ToolDetectedEvent(
                                            tool_name=accumulated_tool_calls[idx]["name"],
                                            tool_id=accumulated_tool_calls[idx]["id"]
                                        )
                                else:
                                    # Update existing tool call ID/name if provided
                                    if tc.get("id"):
                                        accumulated_tool_calls[idx]["id"] = tc["id"]
                                    if tc.get("function", {}).get("name"):
                                        accumulated_tool_calls[idx]["name"] = tc["function"]["name"]

                                # Accumulate arguments
                                args_delta = tc.get("function", {}).get("arguments", "")
                                accumulated_tool_calls[idx]["arguments"] += args_delta

                    # Build GenericOpenAIResponse from accumulated data
                    content_blocks = []

                    # Add text block if present
                    if accumulated_text:
                        content_blocks.append(SimpleNamespace(type="text", text=accumulated_text))

                    # Add tool_use blocks - parse accumulated JSON arguments
                    for idx in sorted(accumulated_tool_calls.keys()):
                        tc = accumulated_tool_calls[idx]
                        try:
                            arguments = json.loads(tc["arguments"]) if tc["arguments"] else {}
                        except json.JSONDecodeError:
                            self.logger.warning(f"Failed to parse tool arguments: {tc['arguments'][:100]}", exc_info=True)
                            arguments = {}
                        content_blocks.append(SimpleNamespace(
                            type="tool_use",
                            id=tc["id"],
                            name=tc["name"],
                            input=arguments
                        ))

                    # Map finish reason to Anthropic stop_reason (case-insensitive to handle
                    # Gemini uppercase values like "STOP", "MAX_TOKENS" that OpenRouter may pass through)
                    stop_reason_map = {
                        "stop": "end_turn",
                        "tool_calls": "tool_use",
                        "length": "max_tokens",
                        "max_tokens": "max_tokens",
                        # Gemini-specific: content policy / copyright filter
                        "recitation": "end_turn",
                        "safety": "end_turn",
                        # OpenRouter-specific: mid-stream provider error
                        "error": "end_turn",
                    }
                    normalized_finish = finish_reason.lower() if isinstance(finish_reason, str) else None
                    stop_reason = stop_reason_map.get(normalized_finish, "end_turn")

                    # Always log raw finish_reason for generic providers — essential for
                    # diagnosing Gemini truncation (thinking budget cannibalization,
                    # premature stop bugs, OpenRouter stream drops)
                    self.logger.debug(
                        f"Generic stream complete: model={model_override} "
                        f"finish_reason={finish_reason!r} stop_reason={stop_reason} "
                        f"text_chars={len(accumulated_text)} tool_calls={len(accumulated_tool_calls)}"
                    )

                    if normalized_finish is None:
                        self.logger.warning(
                            f"Stream from {model_override} ended without finish_reason — "
                            f"possible premature termination ({len(accumulated_text)} chars accumulated)"
                        )
                    elif normalized_finish in ("recitation", "safety", "error"):
                        self.logger.warning(
                            f"Non-standard finish_reason='{finish_reason}' from {model_override} — "
                            f"response may be truncated or filtered ({len(accumulated_text)} chars accumulated)"
                        )
                    elif normalized_finish not in stop_reason_map:
                        self.logger.warning(
                            f"Unmapped finish_reason='{finish_reason}' from {model_override}, "
                            f"defaulted to end_turn ({len(accumulated_text)} chars accumulated)"
                        )

                    response = GenericOpenAIResponse(
                        content=content_blocks,
                        stop_reason=stop_reason,
                        usage=stream_usage or {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                        reasoning_details=accumulated_reasoning_details or None
                    )

                    # Traffic tap: log assembled generic streaming response
                    from utils.llm_tap import is_active as _tap_active, log_response as _tap_response
                    if _tap_active():
                        _tap_response(
                            provider="generic", model=model_override or self.model,
                            response_data=response, endpoint=endpoint_url,
                        )

                    if not stream_usage:
                        self.logger.warning(
                            f"No usage data in stream from {endpoint_url} / {model_override}. "
                            f"Billing will be skipped for this request."
                        )

                    # Billing hook for generic provider streaming (mirrors Anthropic path)
                    try:
                        from billing import get_billing_backend, UsageRecord
                        from billing.exceptions import BillingConfigurationError
                        from billing.pricing import resolve_pricing_key
                        from utils.user_context import get_current_user_id, has_user_context

                        if has_user_context() and response.usage and response.usage.input_tokens:
                            pricing_key = resolve_pricing_key(model_override, endpoint_url)
                            if not pricing_key:
                                raise BillingConfigurationError(
                                    f"No pricing config found for model '{model_override}' "
                                    f"(endpoint: {endpoint_url}). "
                                    f"Add entry to usage_pricing table."
                                )
                            billing = get_billing_backend()
                            billing_usage = UsageRecord(
                                pricing_key=pricing_key,
                                model=model_override,
                                input_tokens=response.usage.input_tokens,
                                output_tokens=response.usage.output_tokens,
                                cache_read_tokens=getattr(response.usage, 'cache_read_input_tokens', 0) or 0,
                                cache_write_tokens=getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                            )
                            billing.record_usage(
                                str(get_current_user_id()),
                                billing_usage,
                                allow_negative=allow_negative,
                            )
                    except ImportError:
                        pass  # OSS mode - billing module not present
                    except Exception as e:
                        try:
                            from billing.exceptions import InsufficientBalanceError, BillingConfigurationError as BCE
                            if isinstance(e, (InsufficientBalanceError, BCE)):
                                raise
                        except ImportError:
                            pass  # OSS mode
                        self.logger.error(f"Billing record_usage failed (generic streaming): {e}", exc_info=True)

                    # Cost accumulator hook (stream_events never carries internal_llm context —
                    # it's only called from the user-facing chat path).
                    try:
                        from utils import cost_accumulator
                        if cost_accumulator.is_active() and response.usage:
                            cost_accumulator.record(
                                internal_llm_name=None,
                                model=model_override,
                                input_tokens=response.usage.input_tokens,
                                output_tokens=response.usage.output_tokens,
                                cache_read_tokens=getattr(response.usage, 'cache_read_input_tokens', 0) or 0,
                                cache_write_tokens=getattr(response.usage, 'cache_creation_input_tokens', 0) or 0,
                            )
                    except Exception as e:
                        self.logger.debug(f"cost_accumulator record failed (non-fatal): {e}")

                    # Check for tool_use blocks
                    tool_blocks = [b for b in response.content if b.type == "tool_use"]

                    if not tool_blocks:
                        # No tools = done, emit final response
                        yield CompleteEvent(response=response)
                        return

                    # Execute tools in parallel (matches Anthropic path pattern)
                    from utils.user_context import _user_context
                    user_context_value = _user_context.get()

                    def invoke_with_context(tool_name: str, tool_input: dict):
                        """Wrapper that propagates user context to worker thread."""
                        _user_context.set(user_context_value)
                        return self.tool_repo.invoke_tool(tool_name, tool_input)

                    # Emit executing events for all tools immediately
                    for block in tool_blocks:
                        yield ToolExecutingEvent(
                            tool_name=block.name,
                            tool_id=block.id,
                            arguments=block.input
                        )

                    # Execute tools concurrently
                    tool_results = []
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_to_block = {
                            executor.submit(invoke_with_context, block.name, block.input): block
                            for block in tool_blocks
                        }

                        for future in concurrent.futures.as_completed(future_to_block):
                            block = future_to_block[future]
                            error = None
                            try:
                                result = future.result()
                                if isinstance(result, dict) and "_content_blocks" in result:
                                    result_str = result.pop("_content_blocks")
                                else:
                                    result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                                yield ToolCompletedEvent(
                                    tool_name=block.name,
                                    tool_id=block.id,
                                    result=_displayable_result(result_str, result),
                                )
                            except Exception as e:
                                self.logger.error(f"Tool execution failed for {block.name}: {e}", exc_info=True)
                                # Include schema for parameter validation errors to help model correct itself
                                schema_hint = ""
                                error_str = str(e).lower()
                                is_param_error = isinstance(e, ValueError) or any(
                                    kw in error_str for kw in ["unknown operation", "invalid", "required", "missing", "parameter"]
                                )
                                if is_param_error and self.tool_repo:
                                    schema = self.tool_repo.get_tool_definition(block.name)
                                    if schema:
                                        props = schema.get("input_schema", {}).get("properties", {})
                                        schema_hint = f"\n\nCORRECT PARAMETERS:\n{json.dumps(props, indent=2)}"
                                result_str = f"Error: {e}{schema_hint}"
                                result = None
                                error = e
                                yield ToolErrorEvent(
                                    tool_name=block.name,
                                    tool_id=block.id,
                                    error=str(e)
                                )

                            circuit_breaker.record_execution(block.name, result if not error else None, error)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                                **({"is_error": True} if error else {})
                            })

                    # Build assistant message with tool_use blocks (Anthropic format)
                    assistant_content = []
                    for block in response.content:
                        if block.type == "text":
                            assistant_content.append({"type": "text", "text": block.text})
                        elif block.type == "tool_use":
                            assistant_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })

                    assistant_msg = {"role": "assistant", "content": assistant_content}
                    # Preserve reasoning_details for round-trip (required by OpenRouter reasoning models)
                    if hasattr(response, 'reasoning_details') and response.reasoning_details:
                        assistant_msg["reasoning_details"] = response.reasoning_details
                    current_messages.append(assistant_msg)
                    current_messages.append({"role": "user", "content": tool_results})

                    # Check circuit breaker before continuing loop
                    should_continue, reason = circuit_breaker.should_continue()
                    if not should_continue:
                        self.logger.warning(f"Circuit breaker triggered: {reason}")
                        yield CircuitBreakerEvent(reason=reason)

                        # Add instruction to respond without more tools
                        current_messages[-1]["content"].append({
                            "type": "text",
                            "text": f"[Automated system message: Tool call issue detected - {reason}. No more tool calls available. Provide your response to the user based on information gathered so far.]"
                        })

                        # Force final response without tools
                        response = self._generate_non_streaming(
                            current_messages, None,
                            thinking=thinking,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            container_id=container_id,
                            endpoint_url=endpoint_url,
                            model_override=model_override,
                            api_key_override=api_key_override,
                            system_override=system_override,
                            allow_negative=allow_negative,
                        )
                        yield CompleteEvent(response=response)
                        return
                    # Loop continues - call API again with tool results

            # Route to appropriate handler (Anthropic streaming)
            if tools and self.tool_repo:
                yield from self._execute_with_tools(
                    messages, tools,
                    thinking=thinking,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    container_id=container_id,
                    endpoint_url=endpoint_url,
                    model_preference=model_preference,
                    api_key_override=api_key_override,
                    system_override=system_override,
                    allow_negative=allow_negative,
                )
            else:
                yield from self._stream_response(
                    messages, tools,
                    model_override=model_override,
                    thinking=thinking,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    container_id=container_id,
                    endpoint_url=endpoint_url,
                    api_key_override=api_key_override,
                    system_override=system_override,
                    allow_negative=allow_negative,
                )

        except GenerationCancelled:
            raise
        except (ProviderStallError, ProviderHTTPStatusError) as e:
            from utils.user_context import get_internal_llm
            from clients.vault_client import get_api_key
            self.logger.error(f"Provider bailed: {e} — switching to claude-high")
            _llm_cfg = get_internal_llm("claude-high")
            self.logger.info(f"Retrying with claude-high: {_llm_cfg.model} via {_llm_cfg.endpoint_url}")
            yield ProviderSwitchEvent(
                original_endpoint=e.endpoint,
                backup_model=_llm_cfg.model,
                reason=str(e),
            )
            # Retry with claude-high — resolve API key from vault if needed
            backup_api_key = _llm_cfg.api_key_name and get_api_key(_llm_cfg.api_key_name) or None
            yield from self._stream_response(
                messages, tools,
                model_override=_llm_cfg.model,
                thinking=ThinkingConfig(),
                temperature=temperature,
                max_tokens=_llm_cfg.max_tokens,
                container_id=container_id,
                endpoint_url=_llm_cfg.endpoint_url,
                api_key_override=backup_api_key,
                system_override=system_override,
                allow_negative=allow_negative,
            )
            return
        except Exception as e:
            self.logger.error(f"LLM API request failed: {e}", exc_info=True)
            yield ErrorEvent(error=str(e))
            raise

    def _generate_non_streaming(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        model_override: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        thinking: ThinkingConfig = ThinkingConfig(),
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        container_id: Optional[str] = None,
        allow_negative: bool = False,
        allow_provider_stall_fallback: bool = True,
        internal_llm_name: Optional[str] = None,
    ) -> anthropic.types.Message:
        """Non-streaming generation with Anthropic SDK or generic provider - returns Message object."""
        try:
            # Check failover FIRST - route to emergency fallback if active
            if self._is_failover_active():
                self.logger.debug("Using emergency fallback (failover active)")
                endpoint_url = config.api.emergency_fallback_endpoint
                model_override = config.api.emergency_fallback_model
                api_key_override = self.emergency_fallback_api_key
                # Disable thinking for fallback providers (not supported)
                thinking = ThinkingConfig()
                # Fall through to generic provider routing below

            # Route to generic provider if endpoint_url is provided (but not for Anthropic)
            # Anthropic endpoints should use native SDK to preserve features (batch API, prompt caching, etc.)
            is_anthropic_endpoint = endpoint_url and "api.anthropic.com" in endpoint_url
            if endpoint_url and not is_anthropic_endpoint:
                # Require model_override; api_key_override is optional for local providers like Ollama
                if not model_override:
                    raise ValueError(
                        "When using endpoint_url, model_override must be provided. "
                        "Generic provider calls require an explicit model identifier."
                    )

                self.logger.info(f"Routing to generic OpenAI-compatible endpoint: {endpoint_url} / {model_override}")

                # Create generic client instance (internal utility)
                from utils.generic_openai_client import (
                    GenericOpenAIClient, GenericOpenAIResponse, ToolNotLoadedError
                )

                # Per-call overrides (fall back to instance defaults)
                effective_temperature = temperature if temperature is not None else self.temperature
                effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

                generic_client = GenericOpenAIClient(
                    endpoint=endpoint_url,
                    model=model_override,
                    api_key=api_key_override,  # Optional for local providers like Ollama
                    timeout=self.timeout,
                    max_tokens=effective_max_tokens,
                    temperature=effective_temperature
                )

                # Prepare system prompt
                system_prompt = None
                if system_override:
                    system_prompt = system_override
                else:
                    system_prompt, messages = self._prepare_messages(messages)

                # Strip unsupported features from tools for generic providers
                # - cache_control: not supported
                # - code_execution: Anthropic-specific server-side tool
                generic_tools = None
                if tools:
                    # Filter out code_execution (Anthropic server-side tool)
                    filtered_tools = [tool for tool in tools if tool.get("type") != "code_execution_20250825"]
                    # Strip cache_control from remaining tools
                    generic_tools = [{k: v for k, v in tool.items() if k != "cache_control"} for tool in filtered_tools]

                # Strip container_upload blocks from messages (Files API not supported)
                messages = self._strip_container_uploads_from_messages(messages)

                # Forward thinking params to generic client
                thinking_effort, thinking_budget = self._generic_thinking_params(thinking)

                # Call generic client - handle tool validation errors with auto-load
                try:
                    _generic_resp = self._run_with_response_timeout(
                        lambda: generic_client.messages.create(
                            messages=messages,
                            system=system_prompt,
                            tools=generic_tools,
                            max_tokens=effective_max_tokens,
                            temperature=effective_temperature,
                            thinking_effort=thinking_effort,
                            thinking_budget=thinking_budget
                        ),
                        endpoint=endpoint_url,
                        mode="non-streaming",
                    )
                    # Traffic tap: log generic non-streaming response
                    from utils.llm_tap import is_active as _tap_active, log_response as _tap_response
                    if _tap_active():
                        _tap_response(
                            provider="generic", model=model_override or self.model,
                            response_data=_generic_resp, endpoint=endpoint_url,
                        )
                    return _generic_resp
                except ToolNotLoadedError as e:
                    # Model tried to use a tool that isn't in the request
                    # Return synthetic response with invokeother_tool call
                    # The agentic loop will execute this, load the tool, and continue
                    import uuid
                    from types import SimpleNamespace

                    self.logger.info(
                        f"Tool '{e.tool_name}' not loaded - returning synthetic invokeother_tool call"
                    )

                    return GenericOpenAIResponse(
                        content=[
                            SimpleNamespace(
                                type="tool_use",
                                id=f"toolu_{uuid.uuid4().hex[:24]}",
                                name="invokeother_tool",
                                input={"mode": "load", "query": e.tool_name}
                            )
                        ],
                        stop_reason="tool_use",
                        usage={
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0
                        }
                    )

            # If tools are enabled, reuse streaming pipeline and consume to completion
            if tools and self.tool_repo:
                final: Optional[anthropic.types.Message] = None
                for event in self.stream_events(
                    messages, tools,
                    thinking=thinking,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    container_id=container_id,
                    endpoint_url=endpoint_url,
                    model_override=model_override,
                    api_key_override=api_key_override,
                    system_override=system_override,
                    allow_negative=allow_negative,
                ):
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

            # Per-call overrides (fall back to instance defaults)
            max_tokens = max_tokens if max_tokens is not None else self.max_tokens
            temperature = temperature if temperature is not None else self.temperature

            # Adjust max_tokens for model constraints (Haiku: 8192, Sonnet: 10000+)
            if "haiku" in selected_model.lower() and max_tokens > 8192:
                max_tokens = 8192

            # Use system_override if provided, otherwise extract from messages
            if system_override:
                system_prompt = system_override
                # Still need to prepare messages (convert to Anthropic format)
                _, anthropic_messages = self._prepare_messages(messages)
            else:
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
                        "cache_control": {"type": "ephemeral", "ttl": "1h"}
                    }]
                else:
                    system_param = system_prompt
            else:
                system_param = None

            # Translate thinking config to Anthropic API params
            thinking_params, thinking_budget_adj = self._anthropic_thinking_params(thinking, selected_model)
            max_tokens += thinking_budget_adj

            # Filter thinking blocks:
            # - When thinking disabled: strip all thinking blocks
            # - When thinking enabled: strip only generic provider thinking (signature=None)
            #   to prevent Anthropic rejecting blocks with invalid signatures
            def keep_block(block: dict) -> bool:
                if block.get("type") != "thinking":
                    return True  # Keep non-thinking blocks
                if not thinking.active:
                    return False  # Strip all thinking when disabled
                # Keep only thinking blocks with valid signatures (from Anthropic)
                return block.get("signature") is not None

            messages_to_send = [
                {**msg, "content": [b for b in msg["content"] if keep_block(b)]}
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
                else msg
                for msg in anthropic_messages
            ]

            # Build API call parameters
            api_params = {
                "model": selected_model,
                "max_tokens": max_tokens,
                "messages": messages_to_send,
            }

            # Thinking is incompatible with temperature (API rejects both together).
            # When thinking is active, omit temperature entirely.
            if thinking.active:
                api_params.update(thinking_params)
            else:
                api_params["temperature"] = temperature

            # Only include system if provided (Anthropic API rejects system: None)
            if system_param is not None:
                api_params["system"] = system_param

            # Only include tools if provided (Anthropic API rejects tools: None)
            if anthropic_tools:
                api_params["tools"] = anthropic_tools
                # Per-tool parallel_safe is handled at execution time in _execute_with_tools

            # Add container ID for multi-turn file persistence
            # ONLY when code_execution tool is in the tools list (Anthropic API requirement)
            has_code_execution = anthropic_tools and any(
                tool.get("type") == "code_execution_20250825"
                for tool in anthropic_tools
            )
            if container_id and has_code_execution:
                api_params["container"] = container_id
                self.logger.debug(f"Reusing container: {container_id}")
            elif container_id:
                self.logger.debug("Skipping container (code_execution not in tools)")

            # Retry loop for transient overloaded errors
            for attempt in range(OVERLOAD_MAX_RETRIES):
                try:
                    # Use beta API for code execution and Files API
                    message = self._run_with_response_timeout(
                        lambda: self.anthropic_client.beta.messages.create(
                            **api_params,
                            betas=ANTHROPIC_BETA_FLAGS
                        ),
                        endpoint=endpoint_url or "anthropic",
                        mode="non-streaming",
                    )
                    # Traffic tap: log non-streaming response
                    from utils.llm_tap import is_active as _tap_active, log_response as _tap_response
                    if _tap_active():
                        _tap_response(provider="anthropic", model=message.model, response_data=message)
                    break  # Success - exit retry loop
                except anthropic.APIStatusError as e:
                    if _is_overloaded_error(e) and attempt < OVERLOAD_MAX_RETRIES - 1:
                        delay = min(OVERLOAD_BASE_DELAY * (2 ** attempt), OVERLOAD_MAX_DELAY)
                        delay = delay * (0.5 + random.random())  # Add jitter
                        self.logger.warning(f"API overloaded, retry {attempt + 1}/{OVERLOAD_MAX_RETRIES} after {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    raise  # Not overloaded or exhausted - re-raise for outer handler

            # Capture container ID from response for reuse
            if hasattr(message, 'container') and message.container:
                response_container_id = message.container.id
                self.logger.debug(f"Container ID captured: {response_container_id}")
                # Store in message metadata for orchestrator to retrieve
                if not hasattr(message, '_container_id'):
                    message._container_id = response_container_id
            elif container_id:
                # Container was reused, preserve the ID
                if not hasattr(message, '_container_id'):
                    message._container_id = container_id

            # Log cache usage if available
            if hasattr(message, 'usage') and message.usage:
                usage = message.usage
                self.logger.debug(
                    f"Model: {selected_model} - Token usage - Input: {usage.input_tokens}, Output: {usage.output_tokens}, "
                    f"Cache created: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
                    f"Cache read: {getattr(usage, 'cache_read_input_tokens', 0)}"
                )

            # Billing hook (silently skipped if billing module absent - OSS mode)
            try:
                from billing import get_billing_backend, UsageRecord
                from billing.exceptions import BillingConfigurationError
                from billing.pricing import resolve_pricing_key
                from utils.user_context import get_current_user_id, has_user_context

                if has_user_context() and hasattr(message, 'usage') and message.usage:
                    pricing_key = resolve_pricing_key(selected_model, endpoint_url)
                    if not pricing_key:
                        raise BillingConfigurationError(
                            f"No pricing config found for model '{selected_model}' "
                            f"(endpoint: {endpoint_url}). "
                            f"Add entry to usage_pricing table."
                        )
                    billing = get_billing_backend()
                    billing_usage = UsageRecord(
                        pricing_key=pricing_key,
                        model=selected_model,
                        input_tokens=message.usage.input_tokens,
                        output_tokens=message.usage.output_tokens,
                        cache_read_tokens=getattr(message.usage, 'cache_read_input_tokens', 0) or 0,
                        cache_write_tokens=getattr(message.usage, 'cache_creation_input_tokens', 0) or 0
                    )
                    billing.record_usage(
                        str(get_current_user_id()),
                        billing_usage,
                        allow_negative=allow_negative,
                    )
            except ImportError:
                pass  # OSS mode - billing module not present
            except Exception as e:
                # InsufficientBalanceError and BillingConfigurationError MUST propagate
                try:
                    from billing.exceptions import InsufficientBalanceError, BillingConfigurationError as BCE
                    if isinstance(e, (InsufficientBalanceError, BCE)):
                        raise
                except ImportError:
                    pass  # OSS mode - no billing exceptions module
                self.logger.error(f"Billing record_usage failed: {e}", exc_info=True)

            # Cost accumulator hook (independent of billing module).
            try:
                from utils import cost_accumulator
                if cost_accumulator.is_active() and hasattr(message, 'usage') and message.usage:
                    cost_accumulator.record(
                        internal_llm_name=internal_llm_name,
                        model=selected_model,
                        input_tokens=message.usage.input_tokens,
                        output_tokens=message.usage.output_tokens,
                        cache_read_tokens=getattr(message.usage, 'cache_read_input_tokens', 0) or 0,
                        cache_write_tokens=getattr(message.usage, 'cache_creation_input_tokens', 0) or 0,
                    )
            except Exception as e:
                self.logger.debug(f"cost_accumulator record failed (non-fatal): {e}")

            return message

        except (ProviderStallError, ProviderHTTPStatusError) as e:
            if not allow_provider_stall_fallback:
                self.logger.error(f"Backup provider failed: {e}")
                raise
            from utils.user_context import get_internal_llm
            from clients.vault_client import get_api_key
            self.logger.error(f"Provider bailed: {e} — switching to claude-high")
            _llm_cfg = get_internal_llm("claude-high")
            self.logger.info(f"Retrying with claude-high: {_llm_cfg.model} via {_llm_cfg.endpoint_url}")
            backup_api_key = _llm_cfg.api_key_name and get_api_key(_llm_cfg.api_key_name) or None
            return self._generate_non_streaming(
                messages, tools,
                model_override=_llm_cfg.model,
                endpoint_url=_llm_cfg.endpoint_url,
                api_key_override=backup_api_key,
                thinking=ThinkingConfig(),
                temperature=temperature,
                max_tokens=_llm_cfg.max_tokens,
                container_id=container_id,
                system_override=system_override,
                allow_negative=allow_negative,
                allow_provider_stall_fallback=False,
            )
        except anthropic.APITimeoutError:
            self.logger.error("Request timed out")
            raise TimeoutError("Request timed out")
        except anthropic.APIStatusError as e:
            # Yield error events for stream compatibility (empty generator)
            for _ in self._handle_anthropic_error(e):
                pass
        except anthropic.APIError as e:
            # ACTIVATE FAILOVER on Anthropic errors
            if self.emergency_fallback_enabled:
                self.logger.error(f"Anthropic error: {e} - activating emergency failover", exc_info=True)
                self._activate_failover()
                # Disable thinking for fallback providers (not supported)
                return self._generate_non_streaming(
                    messages, tools,
                    endpoint_url=config.api.emergency_fallback_endpoint,
                    model_override=config.api.emergency_fallback_model,
                    api_key_override=self.emergency_fallback_api_key,
                    thinking=ThinkingConfig(),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    container_id=container_id,
                    system_override=system_override,
                    allow_negative=allow_negative,
                )
            else:
                self.logger.error(f"API error: {e}", exc_info=True)
                raise RuntimeError(f"API error: {e}")

    def _stream_response(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        *,
        model_override: Optional[str] = None,
        thinking: ThinkingConfig = ThinkingConfig(),
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        container_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        allow_negative: bool = False,
    ) -> Generator[StreamEvent, None, None]:
        """Execute streaming request with Anthropic SDK."""
        # Check failover FIRST - use non-streaming fallback if active
        if self._is_failover_active():
            self.logger.debug("Using emergency fallback (failover active, non-streaming)")
            # Disable thinking for fallback providers (not supported)
            response = self._generate_non_streaming(
                messages, tools,
                endpoint_url=config.api.emergency_fallback_endpoint,
                model_override=config.api.emergency_fallback_model,
                api_key_override=self.emergency_fallback_api_key,
                thinking=ThinkingConfig(),
                temperature=temperature,
                max_tokens=max_tokens,
                container_id=container_id,
                system_override=system_override,
                allow_negative=allow_negative,
            )
            yield from self._emit_events_from_response(response)
            return

        # Select model (use override if provided, otherwise default)
        selected_model = model_override if model_override else self.model

        # Per-call overrides (fall back to instance defaults)
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        temperature = temperature if temperature is not None else self.temperature

        # Adjust max_tokens for model constraints (Haiku: 8192, Sonnet: 10000+)
        if "haiku" in selected_model.lower() and max_tokens > 8192:
            max_tokens = 8192

        # Prepare messages and tools
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
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                }]
            else:
                system_param = system_prompt
        else:
            system_param = None

        # Track tool uses for detection events
        tool_uses_seen = set()

        try:
            # Translate thinking config to Anthropic API params
            thinking_params, thinking_budget_adj = self._anthropic_thinking_params(thinking, selected_model)
            max_tokens += thinking_budget_adj

            # Filter thinking blocks:
            # - When thinking disabled: strip all thinking blocks
            # - When thinking enabled: strip only generic provider thinking (signature=None)
            #   to prevent Anthropic rejecting blocks with invalid signatures
            def keep_block(block: dict) -> bool:
                if block.get("type") != "thinking":
                    return True  # Keep non-thinking blocks
                if not thinking.active:
                    return False  # Strip all thinking when disabled
                # Keep only thinking blocks with valid signatures (from Anthropic)
                return block.get("signature") is not None

            messages_to_send = [
                {**msg, "content": [b for b in msg["content"] if keep_block(b)]}
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list)
                else msg
                for msg in anthropic_messages
            ]

            # Build API call parameters
            stream_params = {
                "model": selected_model,
                "max_tokens": max_tokens,
                "messages": messages_to_send,
            }

            # Thinking is incompatible with temperature (API rejects both together).
            # When thinking is active, omit temperature entirely.
            if thinking.active:
                stream_params.update(thinking_params)
            else:
                stream_params["temperature"] = temperature

            # Only include system if provided (Anthropic API rejects system: None)
            if system_param is not None:
                stream_params["system"] = system_param

            # Only include tools if provided (Anthropic API rejects tools: None)
            if anthropic_tools:
                stream_params["tools"] = anthropic_tools
                # Per-tool parallel_safe is handled at execution time in _execute_with_tools

            # Add container ID for multi-turn file persistence
            # ONLY when code_execution tool is in the tools list (Anthropic API requirement)
            has_code_execution = anthropic_tools and any(
                tool.get("type") == "code_execution_20250825"
                for tool in anthropic_tools
            )
            if container_id and has_code_execution:
                stream_params["container"] = container_id
                self.logger.debug(f"Reusing container: {container_id}")
            elif container_id:
                self.logger.debug("Skipping container (code_execution not in tools)")

            # Retry loop for transient overloaded errors
            for attempt in range(OVERLOAD_MAX_RETRIES):
                try:
                    # Use beta API for code execution and Files API
                    with self.anthropic_client.beta.messages.stream(
                        **stream_params,
                        betas=ANTHROPIC_BETA_FLAGS
                    ) as stream:
                        # Stream events — track output chars for cancel billing
                        streamed_output_chars = 0

                        stream_iterator = iter(stream)
                        while True:
                            try:
                                event = self._run_with_response_timeout(
                                    lambda: next(stream_iterator),
                                    endpoint=endpoint_url or "anthropic",
                                    mode="streaming",
                                )
                            except StopIteration:
                                break

                            # Check cancellation between chunks
                            cancel_evt = get_cancel_event()
                            if cancel_evt is not None and cancel_evt.is_set():
                                self._bill_cancelled_stream(
                                    stream, streamed_output_chars,
                                    selected_model, endpoint_url, allow_negative,
                                )
                                raise GenerationCancelled()

                            if event.type == "text":
                                streamed_output_chars += len(event.text)
                                yield TextEvent(content=event.text)

                            elif event.type == "content_block_delta":
                                # Handle thinking deltas from extended thinking
                                if hasattr(event, 'delta') and hasattr(event.delta, 'type'):
                                    if event.delta.type == "thinking_delta":
                                        streamed_output_chars += len(event.delta.thinking)
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
                                elif event.content_block.type == "server_tool_use":
                                    yield ToolDetectedEvent(
                                        tool_name=event.content_block.name,
                                        tool_id=event.content_block.id
                                    )
                                    yield ToolExecutingEvent(
                                        tool_name=event.content_block.name,
                                        tool_id=event.content_block.id,
                                        arguments={}
                                    )

                        # Get final message (Anthropic Message object)
                        final_message = stream.get_final_message()

                        # Traffic tap: log assembled streaming response
                        from utils.llm_tap import is_active as _tap_active, log_response as _tap_response
                        if _tap_active():
                            _tap_response(provider="anthropic", model=final_message.model, response_data=final_message)

                        # Capture container ID from response for reuse
                        if hasattr(final_message, 'container') and final_message.container:
                            response_container_id = final_message.container.id
                            self.logger.debug(f"📦 Container ID captured: {response_container_id}")
                            # Store in message metadata for orchestrator to retrieve
                            if not hasattr(final_message, '_container_id'):
                                final_message._container_id = response_container_id
                        elif container_id:
                            # Container was reused, preserve the ID
                            self.logger.debug(f"📦 Container reused (no new ID in response): {container_id}")
                            if not hasattr(final_message, '_container_id'):
                                final_message._container_id = container_id

                        # Log cache usage if available
                        if hasattr(final_message, 'usage') and final_message.usage:
                            usage = final_message.usage
                            self.logger.debug(
                                f"Model: {selected_model} - Token usage - Input: {usage.input_tokens}, Output: {usage.output_tokens}, "
                                f"Cache created: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
                                f"Cache read: {getattr(usage, 'cache_read_input_tokens', 0)}"
                            )

                        # Billing hook (silently skipped if billing module absent - OSS mode)
                        try:
                            from billing import get_billing_backend, UsageRecord
                            from billing.exceptions import BillingConfigurationError
                            from billing.pricing import resolve_pricing_key
                            from utils.user_context import get_current_user_id, has_user_context

                            if has_user_context() and hasattr(final_message, 'usage') and final_message.usage:
                                pricing_key = resolve_pricing_key(selected_model, endpoint_url)
                                if not pricing_key:
                                    raise BillingConfigurationError(
                                        f"No pricing config found for model '{selected_model}' "
                                        f"(endpoint: {endpoint_url}). "
                                        f"Add entry to usage_pricing table."
                                    )
                                billing = get_billing_backend()
                                billing_usage = UsageRecord(
                                    pricing_key=pricing_key,
                                    model=selected_model,
                                    input_tokens=final_message.usage.input_tokens,
                                    output_tokens=final_message.usage.output_tokens,
                                    cache_read_tokens=getattr(final_message.usage, 'cache_read_input_tokens', 0) or 0,
                                    cache_write_tokens=getattr(final_message.usage, 'cache_creation_input_tokens', 0) or 0
                                )
                                billing.record_usage(
                                    str(get_current_user_id()),
                                    billing_usage,
                                    allow_negative=allow_negative,
                                )
                        except ImportError:
                            pass  # OSS mode - billing module not present
                        except Exception as e:
                            # InsufficientBalanceError and BillingConfigurationError MUST propagate
                            try:
                                from billing.exceptions import InsufficientBalanceError, BillingConfigurationError as BCE
                                if isinstance(e, (InsufficientBalanceError, BCE)):
                                    raise
                            except ImportError:
                                pass  # OSS mode - no billing exceptions module
                            self.logger.error(f"Billing record_usage failed: {e}", exc_info=True)

                        # Cost accumulator hook (stream_events is always user-tier chat).
                        try:
                            from utils import cost_accumulator
                            if cost_accumulator.is_active() and hasattr(final_message, 'usage') and final_message.usage:
                                cost_accumulator.record(
                                    internal_llm_name=None,
                                    model=selected_model,
                                    input_tokens=final_message.usage.input_tokens,
                                    output_tokens=final_message.usage.output_tokens,
                                    cache_read_tokens=getattr(final_message.usage, 'cache_read_input_tokens', 0) or 0,
                                    cache_write_tokens=getattr(final_message.usage, 'cache_creation_input_tokens', 0) or 0,
                                )
                        except Exception as e:
                            self.logger.debug(f"cost_accumulator record failed (non-fatal): {e}")

                        # Emit ToolCompletedEvent for server-side tools so the
                        # orchestrator can persist tool history across turns.
                        # Without this, code_execution results are lost from
                        # conversation history and the model loses context.
                        yield from self._emit_server_tool_completed_events(final_message.content)

                        # Extract file artifacts from code execution result blocks
                        yield from self._extract_file_artifacts(final_message.content)

                        yield CompleteEvent(response=final_message)
                        break  # Success - exit retry loop

                except anthropic.APIStatusError as e:
                    if _is_overloaded_error(e) and attempt < OVERLOAD_MAX_RETRIES - 1:
                        # Exponential backoff with jitter for overloaded errors
                        delay = min(OVERLOAD_BASE_DELAY * (2 ** attempt), OVERLOAD_MAX_DELAY)
                        delay = delay * (0.5 + random.random())  # Add jitter
                        self.logger.warning(f"API overloaded, retry {attempt + 1}/{OVERLOAD_MAX_RETRIES} after {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    # Not overloaded or exhausted retries - re-raise for outer handler
                    raise

        except anthropic.APITimeoutError:
            error_msg = f"Request timed out"
            self.logger.error(error_msg)
            yield ErrorEvent(error=error_msg)
            raise TimeoutError(error_msg)
        except anthropic.APIStatusError as e:
            yield from self._handle_anthropic_error(e)
        except anthropic.APIError as e:
            # ACTIVATE FAILOVER on Anthropic errors
            if self.emergency_fallback_enabled:
                self.logger.error(f"Anthropic error: {e} - activating emergency failover", exc_info=True)
                self._activate_failover()
                # Disable thinking for fallback providers (not supported)
                response = self._generate_non_streaming(
                    messages, tools,
                    endpoint_url=config.api.emergency_fallback_endpoint,
                    model_override=config.api.emergency_fallback_model,
                    api_key_override=self.emergency_fallback_api_key,
                    thinking=ThinkingConfig(),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    container_id=container_id,
                    system_override=system_override,
                    allow_negative=allow_negative,
                )
                yield from self._emit_events_from_response(response)
            else:
                error_msg = f"API error: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                yield ErrorEvent(error=error_msg)
                raise RuntimeError(error_msg)

    def _execute_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        *,
        thinking: ThinkingConfig = ThinkingConfig(),
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        container_id: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        model_preference: Optional[str] = None,
        api_key_override: Optional[str] = None,
        system_override: Optional[str] = None,
        allow_negative: bool = False,
    ) -> Generator[StreamEvent, None, None]:
        """Execute with tool loop using Anthropic format.

        KNOWN BUG: System prompt staleness during agentic loop.
        current_messages is copied once and never re-rendered between rounds.
        Tools that mutate trinket-backing state (domaindoc expand/collapse,
        reminder create/complete) leave the model seeing a stale system prompt
        that contradicts its own tool results. Most visible when enable →
        expand produces zero visible content because the trinket wasn't
        rendering that domaindoc at compose time. The model falls back to
        search as a read-through workaround, burning tokens and tool calls.
        Fix requires bridging this loop back to the orchestrator's working memory.
        """
        circuit_breaker = CircuitBreaker()
        current_messages = messages.copy()

        # Check for user model preference
        selected_model = model_preference if model_preference else self.model

        while True:
            check_cancelled()

            # Stream LLM response with selected model
            response = None
            for event in self._stream_response(
                current_messages, tools,
                model_override=selected_model,
                thinking=thinking,
                temperature=temperature,
                max_tokens=max_tokens,
                container_id=container_id,
                endpoint_url=endpoint_url,
                api_key_override=api_key_override,
                system_override=system_override,
                allow_negative=allow_negative,
            ):
                if isinstance(event, CompleteEvent):
                    response = event.response
                else:
                    yield event

            if not response:
                yield ErrorEvent(error="No response received from LLM")
                return

            # Extract tool calls from Anthropic Message
            tool_calls = self.extract_tool_calls(response)
            self.logger.debug(f"Extracted {len(tool_calls)} tool calls: {[tc['tool_name'] for tc in tool_calls]}")

            # If no tool calls, we're done
            if not tool_calls:
                yield CompleteEvent(response=response)
                return

            # Add assistant message with tool calls
            assistant_message = self._build_assistant_message(response)
            current_messages.append(assistant_message)

            # Separate tools into sequential (parallel_safe=False) and parallel groups
            # Filter out server-side tools first (code_execution is executed by Anthropic)
            local_tool_calls = [tc for tc in tool_calls if tc["tool_name"] != "code_execution"]

            sequential_tools = []
            parallel_tools = []
            for tc in local_tool_calls:
                tool_class = self.tool_repo.tool_classes.get(tc["tool_name"])
                if tool_class and not tool_class.is_call_parallel_safe(tc["input"]):
                    sequential_tools.append(tc)
                else:
                    parallel_tools.append(tc)

            if sequential_tools:
                self.logger.debug(f"Sequential tools: {[t['tool_name'] for t in sequential_tools]}")
            if parallel_tools:
                self.logger.debug(f"Parallel tools: {[t['tool_name'] for t in parallel_tools]}")

            check_cancelled()

            # Emit executing events for all tools
            for tool_call in local_tool_calls:
                yield ToolExecutingEvent(
                    tool_name=tool_call["tool_name"],
                    tool_id=tool_call["id"],
                    arguments=tool_call["input"]
                )

            tool_results = []
            # Capture context value to propagate to worker threads
            from utils.user_context import _user_context
            user_context_value = _user_context.get()

            def invoke_with_context(tool_name: str, tool_input: Dict[str, Any]):
                """Wrapper that sets context in worker thread without re-entry issues"""
                _user_context.set(user_context_value)
                return self.tool_repo.invoke_tool(tool_name, tool_input)

            def execute_tool(tc: ToolCall) -> ToolExecutionResult:
                """Execute a single tool and return structured result."""
                tool_name = tc["tool_name"]
                error = None
                result = None
                try:
                    result = invoke_with_context(tool_name, tc["input"])
                    if isinstance(result, dict) and "_content_blocks" in result:
                        result_content = result.pop("_content_blocks")
                    else:
                        result_content = json.dumps(result) if isinstance(result, dict) else str(result)
                except Exception as e:
                    self.logger.error(f"Tool execution failed for {tool_name}: {e}")
                    schema_hint = ""
                    error_str = str(e).lower()
                    is_param_error = isinstance(e, ValueError) or any(
                        kw in error_str for kw in ["unknown operation", "invalid", "required", "missing", "parameter"]
                    )
                    if is_param_error and self.tool_repo:
                        schema = self.tool_repo.get_tool_definition(tool_name)
                        if schema:
                            props = schema.get("input_schema", {}).get("properties", {})
                            schema_hint = f"\n\nCORRECT PARAMETERS:\n{json.dumps(props, indent=2)}"
                    result_content = f"Error: {e}{schema_hint}"
                    error = e
                return ToolExecutionResult(
                    tool_call=tc,
                    result_content=result_content,
                    raw_result=result,
                    error=error,
                )

            # Execute sequential tools first (in request order)
            for tc in sequential_tools:
                exec_result = execute_tool(tc)
                tool_name = exec_result.tool_call["tool_name"]
                tool_id = exec_result.tool_call["id"]

                if exec_result.error:
                    yield ToolErrorEvent(tool_name=tool_name, tool_id=tool_id, error=str(exec_result.error))
                else:
                    yield ToolCompletedEvent(
                        tool_name=tool_name, tool_id=tool_id,
                        result=_displayable_result(exec_result.result_content, exec_result.raw_result),
                    )

                circuit_breaker.record_execution(tool_name, exec_result.raw_result, exec_result.error)
                tool_results.append({
                    "tool_use_id": tool_id,
                    "content": exec_result.result_content,
                    **({"is_error": True} if exec_result.error else {})
                })

            # Execute parallel tools concurrently
            if parallel_tools:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_tool = {
                        executor.submit(execute_tool, tc): tc
                        for tc in parallel_tools
                    }

                    for future in concurrent.futures.as_completed(future_to_tool):
                        exec_result = future.result()
                        tool_name = exec_result.tool_call["tool_name"]
                        tool_id = exec_result.tool_call["id"]

                        if exec_result.error:
                            yield ToolErrorEvent(tool_name=tool_name, tool_id=tool_id, error=str(exec_result.error))
                        else:
                            yield ToolCompletedEvent(
                                tool_name=tool_name, tool_id=tool_id,
                                result=_displayable_result(exec_result.result_content, exec_result.raw_result),
                            )

                        circuit_breaker.record_execution(tool_name, exec_result.raw_result, exec_result.error)
                        tool_results.append({
                            "tool_use_id": tool_id,
                            "content": exec_result.result_content,
                            **({"is_error": True} if exec_result.error else {})
                        })

            # Check circuit breaker after all tools complete
            should_continue, reason = circuit_breaker.should_continue()
            if not should_continue:
                self.logger.warning(f"Circuit breaker triggered: {reason}")
                yield CircuitBreakerEvent(reason=reason)

                # Add tool results + instruction to respond without more tools
                current_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": r["content"]}
                        for r in tool_results
                    ] + [
                        {"type": "text", "text": f"[Automated system message: Tool call issue detected - {reason}. No more tool calls available. Provide your response to the user based on information gathered so far.]"}
                    ]
                })

                # Get final response - pass tools=None to force text-only response
                for event in self._stream_response(
                    current_messages, None,
                    model_override=selected_model,
                    thinking=thinking,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    container_id=container_id,
                    endpoint_url=endpoint_url,
                    api_key_override=api_key_override,
                    system_override=system_override,
                    allow_negative=allow_negative,
                ):
                    if isinstance(event, CompleteEvent):
                        yield event
                        return
                    else:
                        yield event

            # Add tool results in ONE user message (only if we have local tool results)
            # Server-side tools (code_execution) are handled by Anthropic and don't need results from us
            if tool_results:
                current_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": r["tool_use_id"], "content": r["content"]}
                        for r in tool_results
                    ]
                })
            else:
                # All tools were server-side (e.g., only code_execution) - continue to next iteration
                # Anthropic will handle the execution and may call more tools
                self.logger.debug("No local tool results - all tools executed server-side")

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate messages before sending to API."""
        # Check for empty messages list
        if not messages:
            self.logger.error("Empty messages list detected")
            raise ValueError("Cannot send empty messages list to LLM API")

        for idx, msg in enumerate(messages):
            content = msg.get('content', '')
            role = msg.get('role', 'unknown')

            # Allow assistant messages with tool calls but no content
            if role == 'assistant' and not content and msg.get('tool_calls'):
                continue

            # Block truly empty messages for all roles
            if not content or not str(content).strip():
                self.logger.error(f"Empty {role} message detected in continuum")
                raise ValueError(f"Cannot send empty {role} message to LLM API")

            # Validate container_upload blocks have required file_id
            if isinstance(content, list):
                for block_idx, block in enumerate(content):
                    if isinstance(block, dict) and block.get('type') == 'container_upload':
                        # file_id is directly on the block, not nested in source
                        file_id = block.get('file_id')
                        if not file_id:
                            self.logger.error(
                                f"Malformed container_upload block in message {idx}, block {block_idx}: "
                                f"missing file_id. Block: {block}"
                            )
                            raise ValueError(
                                f"container_upload block in message {idx} is missing required file_id field"
                            )

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
        # Max stdout/stderr size to prevent context inflation during agentic loop.
        # Code execution can produce megabytes of output; only the tail matters
        # for the model to reason about what happened.
        MAX_TOOL_OUTPUT_CHARS = 50_000

        def _truncate_output(text: str) -> str:
            if not text or len(text) <= MAX_TOOL_OUTPUT_CHARS:
                return text
            truncated = len(text) - MAX_TOOL_OUTPUT_CHARS
            return f"[...truncated {truncated} chars...]\n{text[-MAX_TOOL_OUTPUT_CHARS:]}"

        content_blocks = []

        for block in message.content:
            if block.type == "thinking":
                content_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature
                })
            elif block.type == "redacted_thinking":
                # Opaque blocks required by the API for thinking continuity.
                # Dropping these causes 400 errors on the next agentic loop iteration.
                content_blocks.append({
                    "type": "redacted_thinking",
                    "data": block.data
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
            elif block.type == "server_tool_use":
                content_blocks.append({
                    "type": "server_tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
            elif block.type == "code_execution_tool_result":
                content_blocks.append(block.model_dump())
            else:
                # Unrecognized block type — pass through as-is to avoid
                # silently dropping blocks the API requires.
                self.logger.warning(
                    f"Unrecognized content block type '{block.type}' in assistant message — "
                    f"passing through raw. Update _build_assistant_message to handle this type."
                )
                # Convert SDK object to dict via model_dump if available (Pydantic v2),
                # otherwise fall back to __dict__
                if hasattr(block, 'model_dump'):
                    content_blocks.append(block.model_dump())
                elif hasattr(block, 'to_dict'):
                    content_blocks.append(block.to_dict())
                else:
                    content_blocks.append({"type": block.type, **{
                        k: v for k, v in vars(block).items() if not k.startswith('_')
                    }})

        return {
            "role": "assistant",
            "content": content_blocks
        }

    def _handle_anthropic_error(self, error: anthropic.APIStatusError) -> Generator[StreamEvent, None, None]:
        """Handle Anthropic SDK errors and yield appropriate events."""
        status_code = error.status_code
        message = str(error)

        # Check for container expiration (rare - 30 day TTL)
        if "container" in message.lower() and ("expired" in message.lower() or "not found" in message.lower()):
            self.logger.warning(f"Container expired or not found: {message}")
            self.logger.warning("A new container will be created on next request with file upload")
            # Fall through to general error handling - container_id will be replaced on next file upload

        # Check for context length exceeded (400 with specific patterns)
        if status_code == 400:
            error_lower = message.lower()
            if "prompt is too long" in error_lower or "context" in error_lower or "too many tokens" in error_lower:
                self.logger.error(f"Anthropic context length exceeded: {message}")
                yield ErrorEvent(error="Request too large for model context window")
                raise ContextOverflowError(0, config.api.context_window_tokens, 'anthropic')

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

    def extract_thinking_content(self, message: anthropic.types.Message) -> str:
        """Extract concatenated thinking content from Anthropic Message.

        Returns empty string if no thinking blocks present (extended thinking
        was disabled or model produced no reasoning for this turn).
        """
        thinking_blocks = [block.thinking for block in message.content if block.type == "thinking"]
        return "\n\n".join(thinking_blocks)

    def extract_tool_calls(self, message: anthropic.types.Message) -> list[ToolCall]:
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
