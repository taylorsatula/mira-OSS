-- Migration: Prune extraction_ref links from memories JSONB arrays
-- Purpose: The extraction_ref link type was written by deterministic extraction-
--   time storage code that has been removed (curation moved to the
--   MemoryCuratorAgent, which produces only typed relationship links).
--   extraction_ref is no longer in VALID_RELATIONSHIP_TYPES / RelationshipType,
--   so any surviving JSONB entries are stale and must be pruned so the data
--   matches the new schema. Links are stored as JSONB arrays on the memories
--   row (inbound_links / outbound_links), not a separate table.
-- Effect: removes every {link_type: "extraction_ref"} entry from both arrays.
-- Usage: psql -U postgres -h localhost -d mira_service -f deploy/migrations/prune_extraction_ref_links.sql

-- Prune outbound_links: drop entries whose link_type is extraction_ref.
UPDATE memories
SET outbound_links = (
    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
    FROM jsonb_array_elements(COALESCE(outbound_links, '[]'::jsonb)) AS elem
    WHERE elem->>'link_type' IS DISTINCT FROM 'extraction_ref'
)
WHERE outbound_links @> '[{"link_type":"extraction_ref"}]'::jsonb
   OR outbound_links IS NULL;

-- Prune inbound_links: drop entries whose link_type is extraction_ref.
UPDATE memories
SET inbound_links = (
    SELECT COALESCE(jsonb_agg(elem), '[]'::jsonb)
    FROM jsonb_array_elements(COALESCE(inbound_links, '[]'::jsonb)) AS elem
    WHERE elem->>'link_type' IS DISTINCT FROM 'extraction_ref'
)
WHERE inbound_links @> '[{"link_type":"extraction_ref"}]'::jsonb
   OR inbound_links IS NULL;
