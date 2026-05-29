"""Explicit usage-accounting lifecycle policy for LLM calls."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

from clients.llm.types import Result, Usage

logger = logging.getLogger(__name__)


class UsageAccountingError(RuntimeError):
    """Usage accounting could not be completed for a billable LLM result."""


@dataclass(frozen=True)
class UsageAccountingPolicy:
    """Controls whether lifecycle completion must account for provider usage."""

    required: bool
    allow_negative: bool = False

    @classmethod
    def for_current_build(cls, *, allow_negative: bool = False) -> UsageAccountingPolicy:
        """Require billing when the hosted billing package is present."""
        return cls(
            required=importlib.util.find_spec("billing") is not None,
            allow_negative=allow_negative,
        )


class UsageAccountingService:
    """Finalizes token usage after the provider lifecycle produces a result."""

    def finalize(self, result: Result, policy: UsageAccountingPolicy) -> None:
        usage = result.usage
        if usage is None:
            if policy.required:
                logger.warning(
                    "Provider %s did not return usage data; billing skipped for model=%s endpoint=%s",
                    result.provider_metadata.dialect_name,
                    result.provider_metadata.model,
                    result.provider_metadata.endpoint_url,
                )
            return

        if policy.required:
            self._record_billing(result, usage, allow_negative=policy.allow_negative)

        self._record_cost_accumulator(result, usage)

    def _record_billing(self, result: Result, usage: Usage, *, allow_negative: bool) -> None:
        from billing import get_billing_backend
        from billing.exceptions import BillingConfigurationError
        from billing.models import UsageRecord
        from billing.pricing import resolve_pricing_key
        from utils.user_context import get_current_user_id, has_user_context

        if not has_user_context():
            raise UsageAccountingError("User context required for billable LLM usage accounting")

        metadata = result.provider_metadata
        if not metadata.model:
            raise UsageAccountingError("Provider metadata missing model required for billing")

        pricing_key = resolve_pricing_key(metadata.model, metadata.endpoint_url)
        if not pricing_key:
            raise BillingConfigurationError(
                f"No pricing configured for model={metadata.model}, endpoint={metadata.endpoint_url}"
            )

        get_billing_backend().record_usage(
            get_current_user_id(),
            UsageRecord(
                pricing_key=pricing_key,
                model=metadata.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens,
                cache_write_tokens=usage.cache_creation_input_tokens,
            ),
            allow_negative=allow_negative,
        )

    def _record_cost_accumulator(self, result: Result, usage: Usage) -> None:
        try:
            from utils import cost_accumulator

            if cost_accumulator.is_active():
                cost_accumulator.record(
                    internal_llm_name=result.provider_metadata.internal_llm_name,
                    model=result.provider_metadata.model or "unknown",
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_input_tokens,
                    cache_write_tokens=usage.cache_creation_input_tokens,
                )
        except Exception as error:
            logger.debug("cost_accumulator record failed (non-fatal): %s", error)
