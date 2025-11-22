-- MIRA Service Database Schema
-- Unified schema combining all application and memory tables
-- Updated: 2025-10-19
--
-- Run this to create a fresh mira_service database:
-- psql -U mira_admin -h localhost -f deploy/mira_service_schema.sql
--
-- =====================================================================
-- INDEX STRATEGY WITH ROW LEVEL SECURITY (RLS)
-- =====================================================================
--
-- RLS policies function as additional WHERE clauses during query planning.
-- PostgreSQL can effectively use indexes with RLS when policies use
-- LEAKPROOF functions (like current_setting() and type casts).
--
-- INDEXING REQUIREMENTS FOR RLS TABLES:
--   1. ALWAYS index columns used in RLS policies (e.g., user_id)
--   2. Create specialized indexes for query patterns (vector, full-text)
--   3. PostgreSQL combines multiple indexes for optimal query plans
--
-- PERFORMANCE STRATEGY:
--   - Primary: Proper indexes on filtered columns (user_id, timestamps)
--   - Secondary: Specialized indexes (IVFFlat for vectors, GIN for full-text)
--   - Tertiary: Application caching to reduce database load
--
-- Vector indexes (IVFFlat, HNSW) CANNOT be composite with scalar columns,
-- so we use separate indexes that PostgreSQL combines during execution:
--   - B-tree index on user_id (for RLS filtering)
--   - IVFFlat index on embedding (for similarity search)
--   - Query planner uses both: vector index finds candidates, RLS filters them

-- =====================================================================
-- CREATE ROLES
-- =====================================================================

-- Database owner role (schema management, migrations, backups)
-- NOT a superuser - can only manage mira_service database
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_admin') THEN
        CREATE ROLE mira_admin WITH
            LOGIN
            CREATEDB
            NOCREATEROLE
            NOREPLICATION
            NOSUPERUSER
            PASSWORD 'new_secure_password_2024';
    END IF;
END
$$;

-- Application runtime role (data operations only)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN
        CREATE ROLE mira_dbuser WITH
            LOGIN
            NOCREATEDB
            NOCREATEROLE
            NOREPLICATION
            NOSUPERUSER
            PASSWORD 'new_secure_password_2024';
    END IF;
END
$$;

-- =====================================================================
-- CREATE DATABASE
-- =====================================================================

CREATE DATABASE mira_service OWNER mira_admin;

\c mira_service

-- =====================================================================
-- EXTENSIONS
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- =====================================================================
-- SCHEMA PERMISSIONS
-- =====================================================================

GRANT USAGE ON SCHEMA public TO mira_dbuser;
GRANT ALL ON SCHEMA public TO mira_admin;

-- =====================================================================
-- USERS & AUTH TABLES
-- =====================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE,
    webauthn_credentials JSONB DEFAULT '{}',
    memory_manipulation_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    daily_manipulation_last_run TIMESTAMP WITH TIME ZONE,
    timezone VARCHAR(100) NOT NULL DEFAULT 'America/Chicago',
    overarching_knowledge TEXT,

    -- Activity-based time tracking (vacation-proof scoring)
    cumulative_activity_days INT DEFAULT 0,
    last_activity_date DATE
);

COMMENT ON COLUMN users.cumulative_activity_days IS 'Total number of days user has sent at least one message (activity-based time metric)';
COMMENT ON COLUMN users.last_activity_date IS 'Last date user sent a message (prevents double-counting same day)';

-- NOTE: Currently unused - reserved for future soft delete implementation
CREATE TABLE IF NOT EXISTS users_trash (
    id UUID PRIMARY KEY,
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    webauthn_credentials JSONB,
    memory_manipulation_enabled BOOLEAN,
    daily_manipulation_last_run TIMESTAMP WITH TIME ZONE,
    timezone VARCHAR(100),
    overarching_knowledge TEXT,
    cumulative_activity_days INT,
    last_activity_date DATE,
    deleted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE users_trash IS 'Soft delete storage for deleted users (currently unused - users are hard-deleted via CASCADE)';

CREATE TABLE IF NOT EXISTS magic_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    email VARCHAR(255) NOT NULL,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE magic_links IS 'Passwordless authentication tokens for magic link login flow';

-- =====================================================================
-- ACTIVITY TRACKING (for vacation-proof scoring)
-- =====================================================================

CREATE TABLE IF NOT EXISTS user_activity_days (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    activity_date DATE NOT NULL,
    first_message_at TIMESTAMP WITH TIME ZONE NOT NULL,
    message_count INT DEFAULT 1,
    PRIMARY KEY (user_id, activity_date)
);

COMMENT ON TABLE user_activity_days IS 'Granular per-day activity tracking for users (one row per active day)';
COMMENT ON COLUMN user_activity_days.first_message_at IS 'Timestamp of first message on this day';
COMMENT ON COLUMN user_activity_days.message_count IS 'Number of messages sent by user on this day';

-- =====================================================================
-- DOMAIN KNOWLEDGE (Letta agent memory blocks)
-- =====================================================================

CREATE TABLE IF NOT EXISTS domain_knowledge_blocks (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    domain_label VARCHAR(100) NOT NULL,
    domain_name VARCHAR(255) NOT NULL,
    block_description TEXT NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE domain_knowledge_blocks IS 'Domain-specific knowledge blocks for Letta agent memory system';
COMMENT ON COLUMN domain_knowledge_blocks.domain_label IS 'Short identifier for the domain (e.g., "customer_db")';
COMMENT ON COLUMN domain_knowledge_blocks.domain_name IS 'Human-readable domain name';
COMMENT ON COLUMN domain_knowledge_blocks.block_description IS 'Description of what knowledge this block provides';
COMMENT ON COLUMN domain_knowledge_blocks.agent_id IS 'Letta agent ID this block is associated with';

CREATE TABLE IF NOT EXISTS domain_knowledge_block_content (
    id SERIAL PRIMARY KEY,
    block_id INTEGER NOT NULL UNIQUE REFERENCES domain_knowledge_blocks(id) ON DELETE CASCADE,
    block_value TEXT NOT NULL,
    letta_block_id VARCHAR(255),
    synced_at TIMESTAMP NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE domain_knowledge_block_content IS 'Actual content/value of domain knowledge blocks';
COMMENT ON COLUMN domain_knowledge_block_content.block_value IS 'The knowledge content text';
COMMENT ON COLUMN domain_knowledge_block_content.letta_block_id IS 'External Letta block ID for sync tracking';

-- =====================================================================
-- CONTINUUM & MESSAGES (conversation architecture)
-- =====================================================================

CREATE TABLE IF NOT EXISTS continuums (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE continuums IS 'Continuous timeline of user interactions (one per user, replaces discrete conversations)';
COMMENT ON COLUMN continuums.metadata IS 'Flexible storage for last_touchstone, last_touchstone_embedding, etc.';

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    continuum_id UUID NOT NULL REFERENCES continuums(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON COLUMN messages.content IS 'Message content - text for simple messages, JSON for multimodal content blocks';
COMMENT ON COLUMN messages.metadata IS 'Message metadata: has_tool_calls, tool_calls, tool_call_id, is_summary, summary_type, etc.';

-- Set LZ4 compression for large text columns
ALTER TABLE messages ALTER COLUMN content SET COMPRESSION lz4;

-- Add segment embedding column for segment sentinels
ALTER TABLE messages ADD COLUMN IF NOT EXISTS segment_embedding vector(384);

COMMENT ON COLUMN messages.segment_embedding IS 'AllMiniLM embedding (384-dim) for segment boundary sentinels (used for segment search)';

-- =====================================================================
-- MESSAGE INDEXES
-- =====================================================================

-- User ID index for RLS policy filtering
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);

-- Continuum ID index for conversation retrieval
CREATE INDEX IF NOT EXISTS idx_messages_continuum_id ON messages(continuum_id);

-- Created timestamp index for temporal queries
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

COMMENT ON TABLE messages IS 'All conversation messages; segments implemented as sentinel messages with is_segment_boundary=true in metadata';
COMMENT ON COLUMN messages.segment_embedding IS 'Vector embedding for segment sentinels, enables semantic segment search. See docs/SystemsOverview/segment_system_overview.md for architecture details.';

-- =====================================================================
-- MEMORIES TABLE (core long-term memory storage)
-- =====================================================================

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding vector(384),  -- AllMiniLM embeddings for fast memory search
    search_vector tsvector,  -- Full-text search vector for BM25-style retrieval
    importance_score NUMERIC(5,3) NOT NULL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE,
    happens_at TIMESTAMP WITH TIME ZONE,

    -- Link tracking arrays for efficient hub scoring
    inbound_links JSONB DEFAULT '[]'::jsonb,  -- Array of {source_id, link_type, confidence, reasoning, created_at}
    outbound_links JSONB DEFAULT '[]'::jsonb, -- Array of {target_id, link_type, confidence, reasoning, created_at}
    entity_links JSONB DEFAULT '[]'::jsonb,   -- Array of {uuid, type, name}

    -- Metadata
    confidence NUMERIC(3,2) DEFAULT 0.9 CHECK (confidence >= 0 AND confidence <= 1),
    is_archived BOOLEAN DEFAULT FALSE,
    archived_at TIMESTAMP WITH TIME ZONE,

    -- Refinement tracking to prevent repeated refinement
    is_refined BOOLEAN DEFAULT FALSE,
    last_refined_at TIMESTAMP WITH TIME ZONE,
    refinement_rejection_count INTEGER DEFAULT 0,

    -- Activity day snapshots for vacation-proof scoring
    activity_days_at_creation INT,
    activity_days_at_last_access INT
);

COMMENT ON TABLE memories IS 'Long-term memory storage with embeddings, links, and activity-based decay';
COMMENT ON COLUMN memories.text IS 'Memory content text';
COMMENT ON COLUMN memories.embedding IS 'AllMiniLM 384-dimensional embedding for semantic similarity search';
COMMENT ON COLUMN memories.search_vector IS 'Full-text search vector for BM25-style retrieval';
COMMENT ON COLUMN memories.importance_score IS 'Memory importance (0.0-1.0) used for retrieval ranking';
COMMENT ON COLUMN memories.happens_at IS 'When the memory event occurred (for temporal context)';
COMMENT ON COLUMN memories.inbound_links IS 'JSONB array of memories that link TO this memory';
COMMENT ON COLUMN memories.outbound_links IS 'JSONB array of memories this memory links TO';
COMMENT ON COLUMN memories.entity_links IS 'JSONB array of entity references this memory mentions';
COMMENT ON COLUMN memories.refinement_rejection_count IS 'Number of times memory was marked do_nothing during refinement. After 3 rejections, excluded from future refinement.';
COMMENT ON COLUMN memories.activity_days_at_creation IS 'User cumulative_activity_days when memory was created (snapshot for decay calculation)';
COMMENT ON COLUMN memories.activity_days_at_last_access IS 'User cumulative_activity_days when memory was last accessed (snapshot for recency calculation)';

-- Set LZ4 compression for large text columns
ALTER TABLE memories ALTER COLUMN text SET COMPRESSION lz4;

-- =====================================================================
-- MEMORY INDEXES
-- =====================================================================

-- User ID index for RLS policy filtering (CRITICAL for performance)
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);

-- Full-text search index for keyword-based retrieval
CREATE INDEX IF NOT EXISTS idx_memories_search_vector ON memories USING gin (search_vector);

-- Vector similarity index for semantic search (IVFFlat algorithm)
-- lists=100 is optimal for ~1000-10000 rows (adjust if dataset grows significantly)
-- This index enables O(log n) similarity search instead of O(n) full table scans
CREATE INDEX IF NOT EXISTS idx_memories_embedding_ivfflat
    ON memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON INDEX idx_memories_user_id IS 'B-tree index for RLS policy filtering - essential for multi-user performance';
COMMENT ON INDEX idx_memories_search_vector IS 'GIN index for full-text search operations';
COMMENT ON INDEX idx_memories_embedding_ivfflat IS 'IVFFlat index for fast cosine similarity search - prevents O(n) sequential scans during deduplication and retrieval';

-- Trigger function to maintain search vectors
CREATE OR REPLACE FUNCTION update_memories_search_vector() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', NEW.text);
    NEW.search_text := NEW.text;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

-- Create trigger to maintain search vectors on insert/update
CREATE TRIGGER memories_search_vector_update
BEFORE INSERT OR UPDATE OF text
ON memories
FOR EACH ROW
EXECUTE FUNCTION update_memories_search_vector();

-- =====================================================================
-- ENTITIES TABLE (knowledge graph nodes)
-- =====================================================================

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- PERSON, ORG, GPE, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, NORP, FAC
    embedding VECTOR(300),       -- spaCy en_core_web_lg word vector (300-dimensional)
    link_count INTEGER DEFAULT 0,
    last_linked_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    is_archived BOOLEAN DEFAULT FALSE,
    archived_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT entities_user_name_type_unique UNIQUE (user_id, name, entity_type)
);

COMMENT ON TABLE entities IS 'Persistent knowledge anchors (people, organizations, products, etc.) that memories link to';
COMMENT ON COLUMN entities.name IS 'Canonical normalized entity name';
COMMENT ON COLUMN entities.entity_type IS 'spaCy NER entity type (PERSON, ORG, GPE, PRODUCT, etc.)';
COMMENT ON COLUMN entities.embedding IS 'spaCy word vector for semantic similarity (300d from en_core_web_lg)';
COMMENT ON COLUMN entities.link_count IS 'Number of memories linking to this entity';
COMMENT ON COLUMN entities.last_linked_at IS 'Timestamp of most recent memory link (for dormancy detection)';

-- =====================================================================
-- EXTRACTION BATCHES (async memory extraction tracking)
-- =====================================================================

CREATE TABLE IF NOT EXISTS extraction_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id TEXT NOT NULL,  -- Anthropic batch API ID
    custom_id TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    request_payload JSONB NOT NULL,
    chunk_metadata JSONB,
    memory_context JSONB,
    status TEXT NOT NULL CHECK (status IN ('submitted', 'processing', 'completed', 'failed', 'expired', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    result_url TEXT,
    result_payload JSONB,
    extracted_memories JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    tokens_used INTEGER
);

COMMENT ON TABLE extraction_batches IS 'Batch extraction job tracking for async memory extraction via Anthropic batch API';
COMMENT ON COLUMN extraction_batches.batch_id IS 'Anthropic batch API batch ID';
COMMENT ON COLUMN extraction_batches.custom_id IS 'Custom ID for batch request tracking';
COMMENT ON COLUMN extraction_batches.chunk_index IS 'Index of conversation chunk being processed';
COMMENT ON COLUMN extraction_batches.status IS 'Batch processing status';
COMMENT ON COLUMN extraction_batches.extracted_memories IS 'JSON array of extracted memories from batch response';

-- =====================================================================
-- POST-PROCESSING BATCHES (relationship classification & consolidation)
-- =====================================================================

CREATE TABLE IF NOT EXISTS post_processing_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id TEXT NOT NULL,  -- Anthropic batch API ID
    batch_type TEXT NOT NULL CHECK (batch_type IN ('relationship_classification', 'consolidation', 'consolidation_review')),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    request_payload JSONB NOT NULL,
    input_data JSONB NOT NULL,
    items_submitted INTEGER NOT NULL,
    items_completed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('submitted', 'processing', 'completed', 'failed', 'expired', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    submitted_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    result_payload JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    tokens_used INTEGER,
    links_created INTEGER DEFAULT 0,
    conflicts_flagged INTEGER DEFAULT 0,
    memories_consolidated INTEGER DEFAULT 0
);

COMMENT ON TABLE post_processing_batches IS 'Post-processing batch tracking for relationship classification and memory consolidation';
COMMENT ON COLUMN post_processing_batches.batch_type IS 'Type of post-processing: relationship_classification or consolidation';
COMMENT ON COLUMN post_processing_batches.input_data IS 'Input data for batch processing (memory pairs, clusters, etc.)';
COMMENT ON COLUMN post_processing_batches.items_submitted IS 'Number of items in batch';
COMMENT ON COLUMN post_processing_batches.links_created IS 'Number of memory links created from batch results';
COMMENT ON COLUMN post_processing_batches.conflicts_flagged IS 'Number of conflicting memories detected';
COMMENT ON COLUMN post_processing_batches.memories_consolidated IS 'Number of memories consolidated from batch';

-- =====================================================================
-- TRIGGERS
-- =====================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_memories_updated_at ON memories;
CREATE TRIGGER update_memories_updated_at
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_entities_updated_at ON entities;
CREATE TRIGGER update_entities_updated_at
BEFORE UPDATE ON entities
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================================
-- PERMISSIONS
-- =====================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON
            users, users_trash, magic_links,
            user_activity_days, domain_knowledge_blocks, domain_knowledge_block_content,
            continuums, messages,
            memories, entities, extraction_batches, post_processing_batches
        TO mira_dbuser;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mira_dbuser;
    END IF;
END
$$;

-- Grant default privileges on future objects created by mira_admin
ALTER DEFAULT PRIVILEGES FOR ROLE mira_admin IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO mira_dbuser;

ALTER DEFAULT PRIVILEGES FOR ROLE mira_admin IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO mira_dbuser;

-- Grant mira_admin access to all existing tables for admin operations
-- (mira_admin has BYPASSRLS to perform cross-user queries)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO mira_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mira_admin;

-- =====================================================================
-- ROW LEVEL SECURITY (user isolation)
-- =====================================================================

-- Note: Authentication tables (users, magic_links) do NOT have RLS
-- These are accessed during authentication flow before user context is established
-- Application code handles access control via token validation
--
-- Note: Sessions are stored in Valkey (not PostgreSQL) - see auth/session.py
-- Note: User credentials stored via UserDataManager (SQLite) - see auth/user_credentials.py

ALTER TABLE user_activity_days ENABLE ROW LEVEL SECURITY;
CREATE POLICY user_activity_days_user_policy ON user_activity_days
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE domain_knowledge_blocks ENABLE ROW LEVEL SECURITY;
CREATE POLICY domain_knowledge_blocks_user_policy ON domain_knowledge_blocks
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE domain_knowledge_block_content ENABLE ROW LEVEL SECURITY;
CREATE POLICY domain_knowledge_block_content_user_policy ON domain_knowledge_block_content
    FOR ALL TO PUBLIC
    USING (block_id IN (SELECT id FROM domain_knowledge_blocks WHERE user_id = current_setting('app.current_user_id')::uuid));

ALTER TABLE continuums ENABLE ROW LEVEL SECURITY;
CREATE POLICY continuums_user_policy ON continuums
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
CREATE POLICY messages_user_policy ON messages
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
CREATE POLICY memories_user_policy ON memories
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
CREATE POLICY entities_user_policy ON entities
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

-- Note: extraction_batches and post_processing_batches do NOT have RLS
-- These are system tracking tables accessed by admin polling jobs
-- They contain no user data, only batch job metadata
