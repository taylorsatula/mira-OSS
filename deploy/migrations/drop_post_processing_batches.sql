-- Migration: Drop the post_processing_batches table
-- Purpose: remove the now-unused post-processing batch tracking table.
--   Links are stored as JSONB arrays on the memories row; no separate
--   relationship-tracking table is needed. Curation is handled by the
--   MemoryCuratorAgent.
-- Usage: psql -U postgres -h localhost -d mira_service -f deploy/migrations/drop_post_processing_batches.sql

DROP TABLE IF EXISTS post_processing_batches CASCADE;
