-- Migration: Prune dead internal_llm + usage_pricing rows for removed curation configs
-- Purpose: The relationship, consolidation, and entity_gc internal_llm configs drove
--   the deleted batch-judgment handlers (relationship classification, consolidation
--   verdict, entity GC). Curation is now handled by the MemoryCuratorAgent, which
--   reuses the system 'summary' internal_llm. These three configs have no callers
--   and no rows in the fresh-deploy schema seed; any rows surviving in already-
--   deployed databases are orphans that never resolve (no internal_llm lookup hits
--   them, so their usage_pricing rows are never billed). Prune both tables so the
--   data matches the new schema.
-- Effect: deletes internal_llm rows (both cof/free tiers) and usage_pricing rows
--   (both :cof/:free qualified names) for relationship, consolidation, entity_gc.
-- Usage: psql -U postgres -h localhost -d mira_service -f deploy/migrations/prune_dead_curation_llm_rows.sql

DELETE FROM usage_pricing
WHERE name IN (
    'relationship:cof', 'relationship:free',
    'consolidation:cof', 'consolidation:free',
    'entity_gc:cof', 'entity_gc:free'
);

DELETE FROM internal_llm
WHERE name IN ('relationship', 'consolidation', 'entity_gc');
