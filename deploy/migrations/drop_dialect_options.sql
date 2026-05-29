-- Migration: Drop unused dialect_options JSONB columns.
--
-- dialect_options was loaded into ConversationLLMConfig, InternalLLMConfig,
-- and ModelSelection but never consumed by any dialect, provider, or request
-- builder. Full pipeline load → validate → throw away.
--
-- Run: psql -U postgres -h localhost -d mira_service \
--          -f deploy/migrations/drop_dialect_options.sql

BEGIN;

ALTER TABLE internal_llm DROP COLUMN dialect_options;
ALTER TABLE conversation_llm DROP COLUMN dialect_options;

COMMIT;
