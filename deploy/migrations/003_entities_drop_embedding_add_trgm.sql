-- Migration: Remove embedding column from entities, add pg_trgm index
-- Entity matching now uses PostgreSQL trigram fuzzy matching instead of vector similarity.
-- Run: psql -U postgres -h localhost -d mira_service -f deploy/migrations/003_entities_drop_embedding_add_trgm.sql

BEGIN;

-- Drop the embedding column (spaCy 300d vectors no longer used for entity matching)
ALTER TABLE entities DROP COLUMN IF EXISTS embedding;

-- Change unique constraint from (user_id, name, entity_type) to (user_id, name)
-- Entity type is now stored metadata only, not part of identity
ALTER TABLE entities DROP CONSTRAINT IF EXISTS entities_user_name_type_unique;
ALTER TABLE entities ADD CONSTRAINT entities_user_name_unique UNIQUE (user_id, name);

-- Add GIN index for pg_trgm fuzzy matching on entity names
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON entities USING gin (name gin_trgm_ops);

COMMIT;
