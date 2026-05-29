-- Migration: Tier-qualify internal_llm pricing keys in usage_pricing
--
-- With the free|cof tier split in internal_llm, each config name maps to two
-- different models with different costs. Split each internal_llm usage_pricing
-- entry into {name}:cof and {name}:free so billing resolves the correct price
-- for the model actually used.
--
-- Run: psql -U postgres -h localhost -d mira_service -f deploy/migrations/tier_qualified_pricing_keys.sql

BEGIN;

-- Remove unqualified internal_llm entries
DELETE FROM usage_pricing WHERE name IN (
    'analysis', 'summary', 'injection_defense', 'tidyup',
    'extraction', 'relationship', 'consolidation',
    'entity_gc', 'domaindoc_summary'
);

-- Insert tier-qualified entries (NULL prices auto-resolve from OpenRouter at startup)
INSERT INTO usage_pricing (name) VALUES
    ('analysis:cof'), ('analysis:free'),
    ('consolidation:cof'), ('consolidation:free'),
    ('domaindoc_summary:cof'), ('domaindoc_summary:free'),
    ('entity_gc:cof'), ('entity_gc:free'),
    ('extraction:cof'), ('extraction:free'),
    ('injection_defense:cof'), ('injection_defense:free'),
    ('relationship:cof'), ('relationship:free'),
    ('summary:cof'), ('summary:free'),
    ('tidyup:cof'), ('tidyup:free')
ON CONFLICT (name) DO NOTHING;

COMMIT;
