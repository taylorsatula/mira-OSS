-- Migration: Cache TTL, adapter_name cleanup, and thinking_blocks removal.
--
-- Changes:
-- 1. Drop DEFAULT 'anthropic' from adapter_name on both tables.
--    All seed rows already specify adapter_name explicitly; the default was
--    a silent misclassification risk for new inserts.
-- 2. Remove metadata.thinking_blocks from stored messages.
--    Superseded by thinking_signatures (read by Anthropic adapter for
--    round-trip). thinking_blocks was only ever written, never read.
--
-- Run: psql -U postgres -h localhost -d mira_service -f deploy/migrations/cache_ttl_and_adapter_cleanup.sql

BEGIN;

-- 1a: Drop DEFAULT from conversation_llm.adapter_name
ALTER TABLE conversation_llm
    ALTER COLUMN adapter_name DROP DEFAULT;

-- 1b: Drop DEFAULT from internal_llm.adapter_name
ALTER TABLE internal_llm
    ALTER COLUMN adapter_name DROP DEFAULT;

-- 2: Remove thinking_blocks from message metadata
-- thinking_signatures is the active field; thinking_blocks was a dead
-- bridge that was never read by any code path.
UPDATE messages
SET metadata = metadata - 'thinking_blocks'
WHERE metadata ? 'thinking_blocks';

COMMIT;
