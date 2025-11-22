-- Migration: Remove RLS-ineffective indexes
-- Date: 2025-11-09
-- Reason: Indexes without user_id as leading column provide no benefit with RLS
--
-- Run with: psql -U taylut -h localhost -d mira_service -f deploy/migrations/001_remove_rls_ineffective_indexes.sql

-- =====================================================================
-- DROP MESSAGE INDEXES
-- =====================================================================

-- Drop vector index on segment embeddings
-- This index scans all users' segment embeddings, then RLS filters - no benefit
DROP INDEX IF EXISTS idx_messages_segment_embedding;

-- Drop partial index for active segments
-- No user_id in index means RLS filters after scan - no benefit
DROP INDEX IF EXISTS idx_messages_active_segments;

-- Drop GIN index on segment metadata
-- Indexes across all users' metadata, RLS filters after - no benefit
DROP INDEX IF EXISTS idx_messages_segment_metadata;

-- =====================================================================
-- DROP MEMORIES INDEXES
-- =====================================================================

-- Drop GIN index on search_vector for full-text search
-- Indexes all users' text vectors, RLS filters after - no benefit
DROP INDEX IF EXISTS idx_memories_search_vector;

-- =====================================================================
-- VERIFICATION
-- =====================================================================

-- Verify indexes are dropped
SELECT
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('messages', 'memories')
ORDER BY tablename, indexname;
