-- Neutral message format migration.
--
-- Materializes tool_call_id and is_error as first-class columns (promoted
-- from metadata JSONB), and rewrites stored content blocks from the old
-- Anthropic wire format to the provider-neutral internal format.
--
-- After this migration:
--   - Message.to_db_tuple() no longer needs to write tool_call_id/is_error
--     into the metadata dict
--   - Message.from_dict() / _parse_message_rows() no longer need to
--     read them from metadata
--   - _convert_legacy_blocks() no longer needs to run at DB read time
--   - The Valkey cache should be cleared (FLUSHALL) since cached content
--     blocks will be in the old format

-- =========================================================================
-- Step 1: Add first-class columns
-- =========================================================================

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS tool_call_id TEXT,
    ADD COLUMN IF NOT EXISTS is_error BOOLEAN NOT NULL DEFAULT FALSE;

-- =========================================================================
-- Step 2: Migrate tool_call_id / is_error from metadata JSONB
-- =========================================================================

UPDATE messages
SET tool_call_id = metadata->>'tool_call_id',
    is_error = COALESCE((metadata->>'is_error')::boolean, FALSE)
WHERE metadata ? 'tool_call_id'
   OR metadata ? 'is_error';

-- Clean the promoted keys out of metadata so they aren't duplicated
UPDATE messages
SET metadata = metadata - 'tool_call_id' - 'is_error'
WHERE metadata ? 'tool_call_id'
   OR metadata ? 'is_error';

-- =========================================================================
-- Step 3: Rewrite content blocks to neutral format
-- =========================================================================
-- The content column stores either a plain string or a JSON array of
-- content blocks. We only need to rewrite rows where content is a JSON
-- array containing old-format block types.

-- 3a: tool_use → tool_call
UPDATE messages
SET content = (
    SELECT jsonb_agg(
        CASE WHEN elem->>'type' = 'tool_use'
            THEN elem || '{"type":"tool_call"}' - 'type' || jsonb_build_object('type', 'tool_call')
            ELSE elem
        END
    )
    FROM jsonb_array_elements(content::jsonb) elem
)
WHERE content LIKE '%"type":"tool_use"%'
  AND content LIKE '[%';

-- 3b: image blocks — flatten nested source wrapper
-- Old: {"type":"image","source":{"type":"base64","media_type":"...","data":"..."}}
-- New: {"type":"image","media_type":"...","data":"..."}
UPDATE messages
SET content = (
    SELECT jsonb_agg(
        CASE WHEN elem->>'type' = 'image' AND elem ? 'source'
            THEN jsonb_build_object(
                'type', 'image',
                'media_type', elem->'source'->>'media_type',
                'data', elem->'source'->>'data'
            )
            ELSE elem
        END
    )
    FROM jsonb_array_elements(content::jsonb) elem
)
WHERE content LIKE '%"type":"image"%'  -- only rows that might contain image blocks
  AND content LIKE '%source%'         -- only rows with old nested source format
  AND content LIKE '[%';

-- 3c: document blocks — flatten nested source wrapper (same as image)
UPDATE messages
SET content = (
    SELECT jsonb_agg(
        CASE WHEN elem->>'type' = 'document' AND elem ? 'source'
            THEN jsonb_build_object(
                'type', 'document',
                'media_type', elem->'source'->>'media_type',
                'data', elem->'source'->>'data'
            )
            ELSE elem
        END
    )
    FROM jsonb_array_elements(content::jsonb) elem
)
WHERE content LIKE '%"type":"document"%'
  AND content LIKE '%source%'
  AND content LIKE '[%';

-- 3d: container_upload → file_ref
-- Old: {"type":"container_upload","container_id":"...","file_id":"..."}
-- New: {"type":"file_ref","file_id":"..."}
UPDATE messages
SET content = (
    SELECT jsonb_agg(
        CASE WHEN elem->>'type' = 'container_upload'
            THEN jsonb_build_object(
                'type', 'file_ref',
                'file_id', elem->>'file_id'
            )
            ELSE elem
        END
    )
    FROM jsonb_array_elements(content::jsonb) elem
)
WHERE content LIKE '%"type":"container_upload"%'
  AND content LIKE '[%';

-- 3e: Remove thinking / redacted_thinking blocks from content
-- Their data is preserved in metadata.thinking_blocks for providers that
-- need it.  In the neutral format, reasoning is stored as ReasoningBlock
-- (type="reasoning") and signatures in metadata.thinking_signatures.
-- Old stored thinking blocks have no neutral equivalent — they are
-- provider-specific wire blocks that belong in metadata, not content.
UPDATE messages
SET content = (
    SELECT jsonb_agg(elem)
    FROM jsonb_array_elements(content::jsonb) elem
    WHERE elem->>'type' NOT IN ('thinking', 'redacted_thinking')
)
WHERE (content LIKE '%"type":"thinking"%'
    OR content LIKE '%"type":"redacted_thinking"%')
  AND content LIKE '[%';

-- 3f: Strip cache_control keys from all content blocks
-- Adapters now own caching at serialization time.
UPDATE messages
SET content = (
    SELECT jsonb_agg(elem - 'cache_control')
    FROM jsonb_array_elements(content::jsonb) elem
)
WHERE content LIKE '%cache_control%'
  AND content LIKE '[%';

-- 3g: Convert tool_result blocks in user-role content to separate
--     role="tool" messages.  This is the trickiest case: a single old
--     user message may contain both tool_result blocks and regular
--     content, which must be split into separate rows.

-- For each user message containing tool_result blocks, we:
--   1. Emit one role="tool" message per tool_result block
--   2. Keep the original user message with only the non-tool_result blocks
--      (or delete it if it has no remaining content)

-- First, insert the new role="tool" messages
INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at, tool_call_id, is_error, segment_embedding)
SELECT
    gen_random_uuid(),
    m.continuum_id,
    m.user_id,
    'tool',
    COALESCE(tr.elem->>'content', ''),
    m.metadata - 'tool_call_id' - 'is_error',
    m.created_at,
    COALESCE(tr.elem->>'tool_use_id', tr.elem->>'call_id', ''),
    COALESCE((tr.elem->>'is_error')::boolean, FALSE),
    NULL
FROM messages m
CROSS JOIN LATERAL jsonb_array_elements(m.content::jsonb) AS tr(elem)
WHERE m.role = 'user'
  AND m.content LIKE '%"type":"tool_result"%'
  AND m.content LIKE '[%'
  AND tr.elem->>'type' = 'tool_result';

-- Then, rewrite the original user messages to remove tool_result blocks
UPDATE messages
SET content = (
    SELECT jsonb_agg(elem)
    FROM jsonb_array_elements(content::jsonb) elem
    WHERE elem->>'type' != 'tool_result'
)
WHERE role = 'user'
  AND content LIKE '%"type":"tool_result"%'
  AND content LIKE '[%';

-- Delete user messages that became empty after removing tool_result blocks
-- (i.e., they contained only tool_result blocks and nothing else)
DELETE FROM messages
WHERE role = 'user'
  AND content = '[]'::jsonb::text;

-- =========================================================================
-- Step 4: Finalize
-- =========================================================================

-- Set NOT NULL on tool_call_id for tool messages (data has been migrated)
-- We use a partial index approach instead of NOT NULL since tool_call_id
-- is only meaningful for role='tool' messages.
CREATE INDEX IF NOT EXISTS idx_messages_tool_call_id
    ON messages(tool_call_id)
    WHERE tool_call_id IS NOT NULL;

COMMENT ON COLUMN messages.tool_call_id IS 'Tool call identifier for role=tool messages. Promoted from metadata JSONB in neutral format migration.';
COMMENT ON COLUMN messages.is_error IS 'Whether this tool message reports an error. Promoted from metadata JSONB in neutral format migration.';
