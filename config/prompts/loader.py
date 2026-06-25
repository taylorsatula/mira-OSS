"""
Centralized prompt loader for config/prompts/.

All production services should use load_prompt() instead of rolling their own
open()/read_text() calls. This ensures consistent:
  - UTF-8 encoding
  - Existence checks with descriptive errors
  - Whitespace stripping (.strip())
  - Path resolution (always relative to this directory, never cwd-dependent)
  - Path traversal protection (cannot escape config/prompts/)
  - Empty-file detection (required prompts must not be empty after strip)
  - File type enforcement (.txt only)
  - Debug logging for load events

Usage:
    from config.prompts.loader import load_prompt

    system_prompt = load_prompt("subcortical_system.txt")
    user_template = load_prompt("subcortical_user.txt")

    # Subdirectories:
    agent_prompt = load_prompt("agents/forage_system.txt")

    # Optional prompts (missing = empty string, no error):
    instructions = load_prompt("thinking_block_instructions.txt", required=False)
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.resolve()
_ALLOWED_SUFFIX = ".txt"


def load_prompt(filename: str, *, required: bool = True) -> str:
    """Load a prompt file from config/prompts/.

    Args:
        filename: Relative path within config/prompts/ (e.g. "subcortical_system.txt",
                  "agents/forage_system.txt"). Must end in .txt.
        required: If True (default), raises FileNotFoundError when the file is missing
                  or empty after stripping. If False, returns an empty string when the
                  file is missing — use for optional addendum prompts that may not exist
                  for all deployments.

    Returns:
        The file contents as a string, stripped of leading/trailing whitespace.

    Raises:
        ValueError: If filename attempts path traversal or has a disallowed extension.
        FileNotFoundError: When required=True and the file does not exist or is empty.
    """
    # --- filetype enforcement ---
    if not filename.endswith(_ALLOWED_SUFFIX):
        raise ValueError(
            f"Prompt filename must end in {_ALLOWED_SUFFIX!r}, "
            f"got {filename!r}"
        )

    # --- path traversal protection ---
    resolved = (_PROMPTS_DIR / filename).resolve()
    if not resolved.is_relative_to(_PROMPTS_DIR):
        raise ValueError(
            f"Prompt filename {filename!r} resolves outside config/prompts/ "
            f"(resolved to {resolved})"
        )

    # --- existence check ---
    if not resolved.exists():
        if required:
            raise FileNotFoundError(
                f"Required prompt file not found: {resolved}. "
                f"Prompts are system configuration, not optional features."
            )
        return ""

    # --- read + strip ---
    content = resolved.read_text(encoding="utf-8").strip()

    # --- empty-file detection ---
    if not content:
        if required:
            raise FileNotFoundError(
                f"Required prompt file is empty: {resolved}. "
                f"Prompt files must contain non-whitespace content."
            )
        return ""

    logger.debug("Loaded prompt: %s (%d chars)", filename, len(content))
    return content
