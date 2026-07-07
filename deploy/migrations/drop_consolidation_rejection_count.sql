-- Migration: Drop consolidation_rejection_count from memories table
-- Purpose: remove the now-unused consolidation rejection counter. The old
--   programmatic consolidation pipeline tracked per-memory rejection counts
--   (after 3 rejections a memory was excluded from consolidation candidates).
--   Curation is now handled by the MemoryCuratorAgent, which does not read or
--   write this column. No code reads the column; the writer method was removed.
-- Usage: psql -U postgres -h localhost -d mira_service -f deploy/migrations/drop_consolidation_rejection_count.sql

ALTER TABLE memories DROP COLUMN IF EXISTS consolidation_rejection_count;
