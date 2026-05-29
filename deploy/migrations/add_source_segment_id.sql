-- Migration: Add source_segment_id to memories table
-- Purpose: Enable context exploration by linking memories to their source segment
-- Usage: psql -U postgres -h localhost -d mira_service -f deploy/migrations/add_source_segment_id.sql

-- Add column (nullable to allow existing memories to remain valid)
ALTER TABLE memories ADD COLUMN IF NOT EXISTS source_segment_id UUID;

-- Add column comment
COMMENT ON COLUMN memories.source_segment_id IS 'Segment this memory was extracted from (enables context exploration via continuum_tool search_within_segment)';
