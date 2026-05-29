-- Rollback for llm_dialect_split.sql.
--
-- Reverses the dialect_name column rename and restores the pre-migration
-- two-value CHECK constraint, collapsing openai/openrouter/groq back to the
-- historical 'openai_compat' bucket.
--
-- Run: psql -U postgres -h localhost -d mira_service \
--          -f deploy/migrations/llm_dialect_split.rollback.sql

BEGIN;

-- ----------------------------------------------------------------------
-- internal_llm
-- ----------------------------------------------------------------------

ALTER TABLE internal_llm DROP CONSTRAINT internal_llm_dialect_name_check;
ALTER TABLE internal_llm RENAME COLUMN dialect_name TO adapter_name;
UPDATE internal_llm SET adapter_name = 'openai_compat'
  WHERE adapter_name IN ('openai', 'openrouter', 'groq');
ALTER TABLE internal_llm
  ADD CONSTRAINT internal_llm_adapter_name_check
  CHECK (adapter_name IN ('anthropic', 'openai_compat'));

-- ----------------------------------------------------------------------
-- conversation_llm
-- ----------------------------------------------------------------------

ALTER TABLE conversation_llm DROP CONSTRAINT conversation_llm_dialect_name_check;
ALTER TABLE conversation_llm RENAME COLUMN dialect_name TO adapter_name;
UPDATE conversation_llm SET adapter_name = 'openai_compat'
  WHERE adapter_name IN ('openai', 'openrouter', 'groq');
ALTER TABLE conversation_llm
  ADD CONSTRAINT conversation_llm_adapter_name_check
  CHECK (adapter_name IN ('anthropic', 'openai_compat'));

COMMIT;
