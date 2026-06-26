-- Migration: Ensure messages.is_error exists for existing installs.
--
-- The neutral message format promotes tool error state from metadata JSONB to a
-- first-class messages.is_error column. Fresh schemas already include this
-- column, but upgraded installs can reach newer message persistence code before
-- that schema repair has been applied.
--
-- Run: psql -U postgres -h localhost -d mira_service \
--          -f deploy/migrations/add_messages_is_error_column.sql

BEGIN;

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS is_error BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE messages
SET is_error = COALESCE((metadata->>'is_error')::boolean, FALSE)
WHERE metadata ? 'is_error';

COMMENT ON COLUMN messages.is_error IS 'Whether this tool message reports an error.';

COMMIT;
