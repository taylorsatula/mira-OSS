"""
Per-request LLM cost accumulator for user-facing "what did this cost" feedback.

A request handler (typically `cns/api/chat.py`) calls `start()` when the user
has asked to see cost. Every LLM call site in `clients/llm_provider.py` then
calls `record()` with the call's token usage and context (`internal_llm` name
if applicable). At the end of the request the handler calls `drain()` which
returns a structured summary.

Pricing is looked up from the `usage_pricing` table. Rows with NULL prices
fall back to `FALLBACK_PRICES` below — keeps the feature useful in OSS where
the closed `billing` module isn't present to auto-populate from OpenRouter.

The accumulator is scoped to a contextvar, so call-site code can be ignorant
of whether cost tracking is active — `record()` is a no-op when it isn't.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict

from utils.database_session_manager import get_shared_session_manager


# Public pricing as of 2026-04. Used only when `usage_pricing` has NULL for a
# key the user exercised. Values are USD per million tokens.
# Updating this table is a low-risk change; Anthropic and Groq publish their
# prices openly. A hosted install with a real billing backend will have
# populated `usage_pricing` rows and never hit this fallback.
FALLBACK_PRICES: Dict[str, Dict[str, float]] = {
    # Anthropic Claude
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00, "cache_read": 1.50,  "cache_write": 18.75},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00, "cache_read": 0.30,  "cache_write":  3.75},
    "claude-haiku-4-5":          {"input":  1.00, "output":  5.00, "cache_read": 0.10,  "cache_write":  1.25},
    "claude-haiku-4-5-20251001": {"input":  1.00, "output":  5.00, "cache_read": 0.10,  "cache_write":  1.25},
    # Groq (subcortical)
    "qwen/qwen3-32b":            {"input":  0.29, "output":  0.39, "cache_read": 0.0,   "cache_write":  0.0},
}


class _UsageRecord(TypedDict):
    pricing_key: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True)
class CostSummary:
    """Structured cost summary for a single request."""
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    calls: int
    unpriced_calls: int
    fallback_prices_used: bool

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "fallback_prices_used": self.fallback_prices_used,
        }


_records: ContextVar[Optional[List[_UsageRecord]]] = ContextVar(
    "cost_accumulator_records", default=None
)


def start() -> None:
    """Begin accumulating usage records for the current request."""
    _records.set([])


def is_active() -> bool:
    """True if the accumulator has been started for the current request."""
    return _records.get() is not None


def record(
    *,
    internal_llm_name: Optional[str],
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> None:
    """Append a usage record. No-op if no accumulator is active.

    `internal_llm_name` is the `internal_llm` purpose key (e.g. `"summary"`)
    when the call was made with `internal_llm=...`, or None when the call
    used the user's account tier (regular user-facing chat).
    """
    records = _records.get()
    if records is None:
        return
    pricing_key = _resolve_pricing_key(internal_llm_name)
    if pricing_key is None:
        # Called outside user context (rare — startup, admin jobs). Skip silently;
        # cost tracking is only meaningful inside a user request anyway.
        return
    records.append(_UsageRecord(
        pricing_key=pricing_key,
        model=model,
        input_tokens=int(input_tokens or 0),
        output_tokens=int(output_tokens or 0),
        cache_read_tokens=int(cache_read_tokens or 0),
        cache_write_tokens=int(cache_write_tokens or 0),
    ))


def drain() -> Optional[CostSummary]:
    """Consume the accumulator and return a priced summary.

    Returns None if no accumulator was active. The accumulator is always
    cleared, even on partial pricing data.
    """
    records = _records.get()
    _records.set(None)
    if not records:
        return None

    prices = _fetch_prices({r["pricing_key"] for r in records})

    total_in = sum(r["input_tokens"] for r in records)
    total_out = sum(r["output_tokens"] for r in records)
    total_cr = sum(r["cache_read_tokens"] for r in records)
    total_cw = sum(r["cache_write_tokens"] for r in records)

    total_cost = 0.0
    unpriced = 0
    used_fallback = False

    for r in records:
        db_price = prices.get(r["pricing_key"])
        price, is_fallback = _price_for_record(db_price, r["model"])
        if price is None:
            unpriced += 1
            continue
        if is_fallback:
            used_fallback = True
        total_cost += (
            (r["input_tokens"]       / 1_000_000) * price["input"] +
            (r["output_tokens"]      / 1_000_000) * price["output"] +
            (r["cache_read_tokens"]  / 1_000_000) * price["cache_read"] +
            (r["cache_write_tokens"] / 1_000_000) * price["cache_write"]
        )

    return CostSummary(
        input_tokens=total_in,
        output_tokens=total_out,
        cache_read_tokens=total_cr,
        cache_write_tokens=total_cw,
        cost_usd=total_cost,
        calls=len(records),
        unpriced_calls=unpriced,
        fallback_prices_used=used_fallback,
    )


def _resolve_pricing_key(internal_llm_name: Optional[str]) -> Optional[str]:
    """Compute the `usage_pricing.name` for the call being recorded.

    Internal LLM purposes key by "{name}:{tier}" (e.g. "summary:cof");
    user-facing chat keys by the user's `account_tiers.name` (e.g. "primary").
    Returns None when there is no active user context.
    """
    from utils.user_context import has_user_context, get_current_user_id, _resolve_user_internal_tier

    if not has_user_context():
        return None

    if internal_llm_name is not None:
        return f"{internal_llm_name}:{_resolve_user_internal_tier()}"

    # User-tier chat — read llm_tier from the users row.
    from clients.postgres_client import PostgresClient
    db = PostgresClient("mira_service", admin=True)
    row = db.execute_single(
        "SELECT llm_tier FROM users WHERE id = %(id)s",
        {"id": str(get_current_user_id())},
    )
    return row["llm_tier"] if row else None


def _fetch_prices(keys: set[str]) -> Dict[str, Optional[Dict[str, Optional[float]]]]:
    """Return {pricing_key: price_dict_or_None} for the given keys.

    Each price_dict has 'input'/'output'/'cache_read'/'cache_write' floats, any
    of which may be None if that column is NULL. Keys with no row at all map
    to None.
    """
    if not keys:
        return {}
    with get_shared_session_manager().get_admin_session() as session:
        rows = session.execute_query(
            """SELECT name,
                      input_price_per_mtok,
                      output_price_per_mtok,
                      cache_read_price_per_mtok,
                      cache_write_price_per_mtok
                 FROM usage_pricing
                WHERE name = ANY(%(keys)s)""",
            {"keys": list(keys)},
        )
    result: Dict[str, Optional[Dict[str, Optional[float]]]] = {k: None for k in keys}
    for row in rows:
        result[row["name"]] = {
            "input":       float(row["input_price_per_mtok"])       if row["input_price_per_mtok"]       is not None else None,
            "output":      float(row["output_price_per_mtok"])      if row["output_price_per_mtok"]      is not None else None,
            "cache_read":  float(row["cache_read_price_per_mtok"])  if row["cache_read_price_per_mtok"]  is not None else None,
            "cache_write": float(row["cache_write_price_per_mtok"]) if row["cache_write_price_per_mtok"] is not None else None,
        }
    return result


def _price_for_record(
    db_price: Optional[Dict[str, Optional[float]]],
    model: str,
) -> tuple[Optional[Dict[str, float]], bool]:
    """Choose the price to use for a single record.

    Preference order:
      1. usage_pricing row with all four columns populated.
      2. FALLBACK_PRICES by model name.
      3. None — record is unpriced and will be counted but not costed.

    Returns (price_dict, is_fallback). `price_dict` always has all four
    fields as concrete floats when not None.
    """
    if db_price is not None and all(
        db_price.get(k) is not None for k in ("input", "output", "cache_read", "cache_write")
    ):
        return (
            {
                "input":       db_price["input"],       # type: ignore[typeddict-item]
                "output":      db_price["output"],      # type: ignore[typeddict-item]
                "cache_read":  db_price["cache_read"],  # type: ignore[typeddict-item]
                "cache_write": db_price["cache_write"], # type: ignore[typeddict-item]
            },
            False,
        )
    fallback = FALLBACK_PRICES.get(model)
    if fallback is not None:
        return (dict(fallback), True)
    return (None, False)
