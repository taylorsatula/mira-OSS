"""Shared types for dialect-owned thinking/effort translation."""

from __future__ import annotations

import re
from dataclasses import dataclass


def uses_adaptive_thinking(model: str) -> bool:
    """Claude 4.6+ uses adaptive thinking instead of explicit token budgets."""
    match = re.search(r"-4-(\d+)(?:-|$)", model)
    if match is None:
        return False
    version = match.group(1)
    return len(version) <= 2 and int(version) >= 6


@dataclass(frozen=True)
class TranslationNote:
    """Describes a lossy or model-clamped thinking translation a dialect performed.

    Dialects emit a TranslationNote whenever a caller's ThinkingConfig is not
    honored verbatim — e.g., budget_tokens mapped to effort heuristically, or
    effort clamped to a per-model maximum. Notes are logged at WARNING level so
    operators can see the seam between caller intent and provider behavior.
    """

    field: str           # "effort" or "budget_tokens"
    requested: object    # the value the caller provided
    applied: object      # the value the dialect will use
    reason: str          # human-readable explanation of why translation happened
