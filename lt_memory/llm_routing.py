"""LLM dialect routing helpers for LT memory batch decisions."""

from clients.llm.resolver import ModelResolver


def uses_anthropic_batch_dialect(internal_llm: str) -> bool:
    """Return whether an internal LLM purpose is routed to Anthropic batch transport."""
    return ModelResolver().resolve(internal_llm=internal_llm).dialect_name == "anthropic"
