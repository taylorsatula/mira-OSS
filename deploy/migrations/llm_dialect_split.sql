-- Migration: Rename adapter_name → dialect_name across the LLM tables and
-- expand the value set from {anthropic, openai_compat} to {anthropic, openai,
-- openrouter, groq}. Reclassifies existing rows by endpoint_url.
--
-- Run: psql -U postgres -h localhost -d mira_service \
--          -f deploy/migrations/llm_dialect_split.sql

BEGIN;

-- ----------------------------------------------------------------------
-- internal_llm
-- ----------------------------------------------------------------------

-- Drop the old CHECK constraint before reclassifying rows.
ALTER TABLE internal_llm DROP CONSTRAINT internal_llm_adapter_name_check;

-- Reclassify rows by endpoint URL (still under the old column name).
UPDATE internal_llm SET adapter_name = 'openrouter'
  WHERE endpoint_url LIKE 'https://openrouter.ai/%';
UPDATE internal_llm SET adapter_name = 'groq'
  WHERE endpoint_url LIKE 'https://api.groq.com/%';
UPDATE internal_llm SET adapter_name = 'openai'
  WHERE endpoint_url LIKE 'https://aiplatform.googleapis.com/%';
-- (anthropic rows remain 'anthropic')

-- Rename the column.
ALTER TABLE internal_llm RENAME COLUMN adapter_name TO dialect_name;

-- Add the new four-value CHECK constraint under the new name.
ALTER TABLE internal_llm
  ADD CONSTRAINT internal_llm_dialect_name_check
  CHECK (dialect_name IN ('anthropic', 'openai', 'openrouter', 'groq'));

-- ----------------------------------------------------------------------
-- conversation_llm
-- ----------------------------------------------------------------------

ALTER TABLE conversation_llm DROP CONSTRAINT conversation_llm_adapter_name_check;

UPDATE conversation_llm SET adapter_name = 'openrouter'
  WHERE endpoint_url LIKE 'https://openrouter.ai/%';
UPDATE conversation_llm SET adapter_name = 'groq'
  WHERE endpoint_url LIKE 'https://api.groq.com/%';
UPDATE conversation_llm SET adapter_name = 'openai'
  WHERE endpoint_url LIKE 'https://aiplatform.googleapis.com/%';
-- (anthropic rows remain 'anthropic')

ALTER TABLE conversation_llm RENAME COLUMN adapter_name TO dialect_name;

ALTER TABLE conversation_llm
  ADD CONSTRAINT conversation_llm_dialect_name_check
  CHECK (dialect_name IN ('anthropic', 'openai', 'openrouter', 'groq'));

COMMIT;
