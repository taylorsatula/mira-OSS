-- Migration: Add overarching_knowledge column to users table
-- Created: 2025-11-02
-- Run with: psql -U mira_admin -h localhost -d mira_service -f deploy/migrations/add_overarching_knowledge.sql

BEGIN;

-- Add overarching_knowledge column to users table
ALTER TABLE users
ADD COLUMN IF NOT EXISTS overarching_knowledge TEXT;

-- Add overarching_knowledge column to users_trash table for consistency
ALTER TABLE users_trash
ADD COLUMN IF NOT EXISTS overarching_knowledge TEXT;

COMMIT;

-- Verify changes
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'users'
AND column_name = 'overarching_knowledge';
