-- Security hardening schema updates for billing idempotency.

ALTER TABLE stripe_webhook_events
    ADD COLUMN IF NOT EXISTS processing_lock_token UUID,
    ADD COLUMN IF NOT EXISTS processing_started_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS processing_finished_at TIMESTAMP WITH TIME ZONE;

CREATE INDEX IF NOT EXISTS idx_stripe_webhook_processing_lock
    ON stripe_webhook_events(processing_lock_token)
    WHERE processing_lock_token IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_positive_stripe_deposit_once
    ON billing_transactions(stripe_payment_intent_id)
    WHERE transaction_type = 'deposit'
      AND amount_usd > 0
      AND stripe_payment_intent_id IS NOT NULL;
