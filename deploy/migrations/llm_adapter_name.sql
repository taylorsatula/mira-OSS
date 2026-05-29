-- Migration: Replace conversation_llm.provider with adapter_name.
--
-- Run: psql -U postgres -h localhost -d mira_service -f deploy/migrations/llm_adapter_name.sql

BEGIN;

ALTER TABLE conversation_llm
    RENAME COLUMN provider TO adapter_name;

ALTER TABLE conversation_llm
    DROP CONSTRAINT IF EXISTS conversation_llm_provider_check;

UPDATE conversation_llm
SET adapter_name = 'openai_compat'
WHERE adapter_name = 'generic';

ALTER TABLE conversation_llm
    ALTER COLUMN adapter_name SET DEFAULT 'anthropic';

ALTER TABLE conversation_llm
    ADD CONSTRAINT conversation_llm_adapter_name_check
    CHECK (adapter_name IN ('anthropic', 'openai_compat'));

UPDATE conversation_llm
SET api_key_name = 'anthropic_key'
WHERE adapter_name = 'anthropic'
  AND api_key_name IS NULL;

COMMIT;
