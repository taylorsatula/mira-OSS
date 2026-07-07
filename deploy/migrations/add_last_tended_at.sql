-- Migration: Add last_tended_at to memories table
-- Purpose: Track when the MemoryCuratorAgent last tended each memory (linked,
--   merged, archived, or salvaged). The floor trigger uses this to sample
--   "memories not tended in N days" without re-treading recently curated ones.
--   Backfills existing rows to created_at so legacy memories become floor-
--   eligible after the staleness cutoff instead of being stranded at NULL.
--   Memories created after this migration start NULL ("integration not yet run")
--   and become floor-eligible once older than the cutoff without being tended.
-- Usage: psql -U postgres -h localhost -d mira_service -f deploy/migrations/add_last_tended_at.sql

-- Add column (nullable: NULL means "never tended by curator").
-- No DEFAULT: new memories must start NULL so the floor query's NULL+age
-- branch can catch memories integration never tended.
ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_tended_at TIMESTAMPTZ;

-- Backfill: treat pre-curation memories as tended-at-creation. This is a
-- one-time update; rows created after migration keep last_tended_at = NULL
-- until the agent tends them. Using created_at (not NOW()) means legacy
-- memories age into floor eligibility at their natural age, not all at once.
UPDATE memories
SET last_tended_at = created_at
WHERE last_tended_at IS NULL;

-- Partial index supporting the floor query's filter on
-- (importance_score, last_tended_at) over non-archived rows.
CREATE INDEX IF NOT EXISTS idx_memories_floor_candidates
    ON memories (importance_score, last_tended_at)
    WHERE is_archived = FALSE;

COMMENT ON COLUMN memories.last_tended_at IS 'Last timestamp the MemoryCuratorAgent tended this memory (linked/merged/archived/salvaged). NULL = never tended (integration not yet run). Backfilled to created_at for pre-curation memories.';
