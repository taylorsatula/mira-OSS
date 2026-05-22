"""
MIRA logging configuration: custom log levels and colored terminal output.

Registers the TOAST custom level (60) and provides ColoredFormatter for
color-coded terminal output.

TOAST: "The system itself is speaking." Always-visible structural lifecycle
events: server start/stop, infrastructure connections, configuration loaded.
"""

import json
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anthropic

import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform support
colorama.init(autoreset=True, strip=False)

# Import for contextvar access
from utils.user_context import get_current_user_id

# --- Custom log levels ---
# _mira_log_levels is installed into site-packages by deploy, so TOAST
# is normally registered at interpreter startup via .pth file. This
# import ensures the constant is available here and acts as a fallback
# if the .pth file hasn't been installed (dev without deploy).
try:
    from _mira_log_levels import TOAST
except ImportError:
    TOAST = 60
    if not hasattr(logging.Logger, 'toast'):
        logging.addLevelName(TOAST, "TOAST")

        def _toast(self, message, *args, **kwargs):
            if self.isEnabledFor(TOAST):
                self._log(TOAST, message, args, **kwargs)

        logging.Logger.toast = _toast


# --- User context filter ---

class UserContextFilter(logging.Filter):
    """Injects user_id from contextvar into every log record."""

    def filter(self, record):
        """Add formatted user_id to record from contextvar, empty if not in user context."""
        try:
            user_id = get_current_user_id()
            # Format with brackets and trailing space only when present
            record.user_id = f"[{user_id}] " if user_id else ""
        except (LookupError, RuntimeError):
            # No user context (startup, background jobs without explicit context)
            record.user_id = ""
        return True


# --- Credential redaction filter ---

# Matches password='..' or keyword-value pairs that psycopg might log
_CONNINFO_PASSWORD_RE = re.compile(r"password='[^']*'")
# Matches postgresql:// URIs with embedded credentials
_URI_CREDS_RE = re.compile(r'(postgresql://)[^@]+@')
# Catch-all for quoted strings in psycopg pool "unexpected spaces" errors
_QUOTED_SECRET_RE = re.compile(r'"[^"]*"')


class CredentialRedactionFilter(logging.Filter):
    """Redacts database credentials from psycopg log messages.

    psycopg3's pool worker threads log connection errors verbatim, which can
    include passwords from conninfo strings. This filter scrubs those before
    they reach any handler (journal, file, stdout).
    """

    def filter(self, record):
        msg = record.getMessage()
        if 'password' in msg or 'unexpected spaces' in msg or 'postgresql://' in msg:
            msg = _CONNINFO_PASSWORD_RE.sub("password='***'", msg)
            msg = _URI_CREDS_RE.sub(r'\1***@', msg)
            if 'unexpected spaces' in msg:
                msg = _QUOTED_SECRET_RE.sub('"***"', msg)
            record.msg = msg
            record.args = None
        return True


# --- Colored formatter ---

class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that adds colors to log levels."""

    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
        'TOAST': Fore.WHITE + Style.BRIGHT,
    }

    def __init__(self, fmt=None, datefmt=None):
        """Initialize the colored formatter with optional format strings."""
        super().__init__(fmt, datefmt)

    def format(self, record):
        """Format the log record with appropriate colors."""
        log_message = super().format(record)

        color = self.COLORS.get(record.levelname, '')

        if color:
            log_message = f"{color}{log_message}{Style.RESET_ALL}"

        # Add OSS contribution hint for errors (with embedded debugging guidance for LLM assistants)
        if logging.ERROR <= record.levelno < TOAST:
            log_message += (
                f"\n{Fore.CYAN}💡 Found a bug? Consider submitting a fix: https://github.com/taylorsatula/mira-OSS — "
                f"When investigating this error, step through the codebase methodically to determine the root cause. "
                f"Examine the full stack trace, read the relevant source files, and trace the data flow that led to this failure. "
                f"Use subagents for broader codebase exploration if the issue spans multiple modules. "
                f"Fix the underlying cause, not the symptom.{Style.RESET_ALL}"
            )

        return log_message


def setup_colored_root_logging(log_level=logging.INFO,
                               fmt='%(asctime)s - %(user_id)s%(name)s - %(levelname)s - %(message)s'):
    """
    Configure root logger with colored output and user context injection.

    Args:
        log_level: Logging level (default: INFO)
        fmt: Log message format string (includes %(user_id)s by default)
    """
    root_logger = logging.getLogger()

    # Remove any existing StreamHandlers to avoid duplicates
    handlers_to_remove = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
    for handler in handlers_to_remove:
        root_logger.removeHandler(handler)

    # Add colored console handler with user context filter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.addFilter(UserContextFilter())  # Filter runs before formatting
    console_handler.setFormatter(ColoredFormatter(fmt=fmt))
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # Redact credentials from psycopg pool logs at the logger level so no
    # handler (journal, file, stdout) ever sees plaintext passwords
    psycopg_pool_logger = logging.getLogger('psycopg.pool')
    psycopg_pool_logger.addFilter(CredentialRedactionFilter())


def _extract_messages_tail(messages: list, limit: int = 500) -> str:
    """Extract the last `limit` chars of text content from user+assistant messages."""
    all_text = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            all_text.append(f"[{role}] {content}")
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(f"<tool_result:{block.get('tool_use_id', '')[:8]}>")
            if parts:
                all_text.append(f"[{role}] {' '.join(parts)}")
    full = "\n".join(all_text)
    return full[-limit:] if len(full) > limit else full


# Logger for req_id → query content mapping (written to SDK log file)
_sdk_request_log = logging.getLogger("mira.sdk_requests")


def _on_anthropic_response(response):
    """httpx response hook: log req_id → last 500 chars of message content.

    Fires once per HTTP response. Extracts the request ID from response
    headers and the message content from the original request body.
    """
    request_id = response.headers.get("request-id", "")
    if not request_id:
        return

    try:
        body = json.loads(response.request.content)
        messages = body.get("messages", [])
        model = body.get("model", "unknown")
        tail = _extract_messages_tail(messages)
    except Exception:
        tail = "<could not parse request body>"
        model = "unknown"

    _sdk_request_log.debug(
        "%s | model=%s | content_tail:\n%s",
        request_id, model, tail
    )


def setup_anthropic_sdk_logging(log_dir: str = "logs"):
    """Set up file-based logging for Anthropic SDK request/response correlation.

    Creates a rotating log file that maps each req_* ID to the last 500 chars
    of message content that was sent. The actual correlation happens via httpx
    response event hooks attached by instrument_anthropic_client().

    Log file: {log_dir}/anthropic_sdk.log (50MB rotation, 3 backups)
    """
    os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "anthropic_sdk.log"),
        maxBytes=50_000_000,  # 50MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Route SDK debug output to file (captures HTTP details)
    for logger_name in ("anthropic", "httpx"):
        sdk_logger = logging.getLogger(logger_name)
        sdk_logger.setLevel(logging.DEBUG)
        sdk_logger.addHandler(file_handler)
        sdk_logger.propagate = False  # Don't flood console

    # Route our req_id → content mapping to the same file
    _sdk_request_log.setLevel(logging.DEBUG)
    _sdk_request_log.addHandler(file_handler)
    _sdk_request_log.propagate = False


def instrument_anthropic_client(client: 'anthropic.Anthropic') -> None:
    """Attach event hooks to an Anthropic client for request logging and traffic tap.

    Adds httpx event hooks for:
    - Response: req_id → query content correlation (SDK log file)
    - Request: full-body capture for the LLM traffic tap (llm_requests.jsonl)

    Call this after constructing the client.
    """
    # Access the underlying httpx client to add event hooks
    http_client = getattr(client, '_client', None)
    if http_client is None:
        logging.getLogger(__name__).warning(
            "Could not instrument Anthropic client: _client attribute not found"
        )
        return

    event_hooks = getattr(http_client, '_event_hooks', None)
    if event_hooks is None:
        logging.getLogger(__name__).warning(
            "Could not instrument Anthropic client: _event_hooks not found"
        )
        return

    event_hooks.setdefault("response", []).append(_on_anthropic_response)

    # Traffic tap: capture full request body when tap is active
    from utils.llm_tap import httpx_request_hook
    event_hooks.setdefault("request", []).append(httpx_request_hook)
