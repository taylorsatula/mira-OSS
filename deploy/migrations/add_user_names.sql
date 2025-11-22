-- Migration: Add first_name and last_name columns to users table
-- Created: 2025-11-02
-- Run with: psql -U mira_admin -h localhost -d mira_service -f deploy/migrations/add_user_names.sql

BEGIN;

-- Add name columns to users table
ALTER TABLE users
ADD COLUMN IF NOT EXISTS first_name VARCHAR(100),
ADD COLUMN IF NOT EXISTS last_name VARCHAR(100);

-- Add name columns to users_trash table for consistency
ALTER TABLE users_trash
ADD COLUMN IF NOT EXISTS first_name VARCHAR(100),
ADD COLUMN IF NOT EXISTS last_name VARCHAR(100);

COMMIT;

-- Verify changes
SELECT column_name, data_type, character_maximum_length
FROM information_schema.columns
WHERE table_name = 'users'
AND column_name IN ('first_name', 'last_name')
ORDER BY ordinal_position;
