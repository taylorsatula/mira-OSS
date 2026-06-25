-- MIRA Service Database Schema
-- Unified schema combining all application and memory tables
-- Updated: 2026-03-24
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
--   - BYPASSRLS required for admin operations that need to bypass Row Level Security
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
            BYPASSRLS
            PASSWORD 'changethisifdeployingpwd';
    END IF;
END
$$;

-- Ensure BYPASSRLS is set even if role already existed
-- (IF NOT EXISTS above won't update existing roles)
ALTER ROLE mira_admin BYPASSRLS;

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
            PASSWORD 'changethisifdeployingpwd';
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
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Trigram fuzzy matching for typo-tolerant search

-- =====================================================================
-- SCHEMA PERMISSIONS
-- =====================================================================

GRANT USAGE ON SCHEMA public TO mira_dbuser;
GRANT ALL ON SCHEMA public TO mira_admin;

-- =====================================================================
-- USERS & AUTH TABLES
-- =====================================================================

-- Conversation LLM table (user-selectable models for conversations)
-- Must be created before users table due to FK reference
CREATE TABLE IF NOT EXISTS conversation_llm (
    name VARCHAR(20) PRIMARY KEY,
    model VARCHAR(100) NOT NULL,
    thinking_budget INT NOT NULL DEFAULT 0,
    description TEXT,
    display_order INT NOT NULL DEFAULT 0,
    dialect_name VARCHAR(20) NOT NULL CHECK (dialect_name IN ('anthropic', 'openai', 'openrouter', 'groq')),
    endpoint_url TEXT DEFAULT NULL,
    api_key_name VARCHAR(50) DEFAULT NULL,
    hidden BOOLEAN NOT NULL DEFAULT FALSE
);

INSERT INTO conversation_llm (name, model, thinking_budget, description, display_order, dialect_name, endpoint_url, api_key_name, hidden) VALUES
    ('primary', 'claude-sonnet-4-6', 0, 'Primary', 1, 'anthropic', NULL, NULL, FALSE)
ON CONFLICT (name) DO NOTHING;

-- Internal LLM configurations for system operations (not user-facing)
-- Contrasts with conversation_llm which handles user-facing model selection
-- Every config has two rows: 'free' (no card on file) and 'cof' (card on file)
-- For cheap models, both rows point to the same model
CREATE TABLE IF NOT EXISTS internal_llm (
    name VARCHAR(50) NOT NULL,
    tier VARCHAR(10) NOT NULL CHECK (tier IN ('free', 'cof')),
    model VARCHAR(200) NOT NULL,
    endpoint_url TEXT NOT NULL,
    api_key_name VARCHAR(50),
    description TEXT,
    max_tokens INT NOT NULL,
    effort VARCHAR(10) CHECK (effort IN ('low', 'medium', 'high', 'xhigh', 'max')),
    dialect_name VARCHAR(20) NOT NULL CHECK (dialect_name IN ('anthropic', 'openai', 'openrouter', 'groq')),
    PRIMARY KEY (name, tier)
);

INSERT INTO internal_llm (name, tier, model, endpoint_url, api_key_name, description, max_tokens, effort, dialect_name) VALUES
    -- COF configs: Anthropic models via direct API
    ('summary', 'cof', 'google/gemma-4-31b-it', 'https://openrouter.ai/api/v1/chat/completions', 'provider_key', 'Segment summary generation', 10000, NULL, 'openrouter'),
    ('assessment', 'cof', 'claude-opus-4-6', 'https://api.anthropic.com/v1/messages', 'anthropic_key', 'Assessment extraction', 10000, NULL, 'anthropic'),
    ('synthesis', 'cof', 'claude-opus-4-6', 'https://api.anthropic.com/v1/messages', 'anthropic_key', 'User model synthesis', 10000, NULL, 'anthropic'),
    ('extraction', 'cof', 'claude-sonnet-4-6', 'https://api.anthropic.com/v1/messages', 'anthropic_batch_key', 'Memory extraction', 16000, 'high', 'anthropic'),
    ('consolidation', 'cof', 'claude-sonnet-4-6', 'https://api.anthropic.com/v1/messages', 'anthropic_batch_key', 'Memory consolidation', 4096, NULL, 'anthropic'),
    ('tidyup', 'cof', 'claude-sonnet-4-6', 'https://api.anthropic.com/v1/messages', 'anthropic_batch_key', 'Context tidyup', 10000, NULL, 'anthropic'),
    ('relationship', 'cof', 'claude-haiku-4-5-20251001', 'https://api.anthropic.com/v1/messages', 'anthropic_batch_key', 'Relationship classification', 500, NULL, 'anthropic'),
    ('entity_gc', 'cof', 'claude-haiku-4-5', 'https://api.anthropic.com/v1/messages', 'anthropic_batch_key', 'Entity garbage collection', 2048, 'high', 'anthropic'),
    ('critic', 'cof', 'claude-sonnet-4-6', 'https://api.anthropic.com/v1/messages', 'anthropic_key', 'User model critic', 10000, NULL, 'anthropic'),
    -- Subcortical: same model for both tiers via Groq
    ('analysis', 'cof', 'qwen/qwen3.6-27b', 'https://api.groq.com/openai/v1/chat/completions', 'subcortical_key', 'Subcortical analysis', 3072, NULL, 'groq'),
    -- Forage: COF gets Kimi K2 via OpenRouter, free gets OSS 120B via Groq
    ('forage', 'cof', 'moonshotai/kimi-k2-thinking', 'https://openrouter.ai/api/v1/chat/completions', 'provider_key', 'Forage agent tool-calling loop', 4096, NULL, 'openrouter'),
    -- Overwatch: passive agent iteration observer, same cheap model as subcortical
    ('overwatch', 'cof', 'qwen/qwen3.6-27b', 'https://api.groq.com/openai/v1/chat/completions', 'subcortical_key', 'Passive agent iteration observer', 100, NULL, 'groq'),
    -- Phone-a-friend: high-capability outside voices for synchronous subagent consultation
    ('phoneafriend_claude', 'cof', 'claude-opus-4-7', 'https://api.anthropic.com/v1/messages', 'anthropic_key', 'Phone-a-friend level-headed thought partner', 10000, 'high', 'anthropic'),
    ('phoneafriend_gemini', 'cof', 'google/gemini-3.1-pro-preview', 'https://openrouter.ai/api/v1/chat/completions', 'provider_key', 'Phone-a-friend outside voice with broad world knowledge', 10000, NULL, 'openrouter'),
    -- Repulsion feedback rewriter: register-aware rewrite pass for captured AI-tells
    ('rewriter', 'cof', 'openai/gpt-5.5', 'https://openrouter.ai/api/v1/chat/completions', 'provider_key', 'Repulsion feedback rewrite generation', 10000, 'high', 'openrouter')
ON CONFLICT (name, tier) DO NOTHING;

GRANT SELECT ON internal_llm TO mira_dbuser;

-- Usage pricing (keyed by tier name or internal config name, not model string)
-- NULL prices auto-resolve from OpenRouter on startup. NOT NULL = manual override.
CREATE TABLE IF NOT EXISTS usage_pricing (
    name VARCHAR(50) PRIMARY KEY,
    input_price_per_mtok DECIMAL(10,6),
    output_price_per_mtok DECIMAL(10,6),
    cache_read_price_per_mtok DECIMAL(10,6),
    cache_write_price_per_mtok DECIMAL(10,6),
    effective_date DATE NOT NULL DEFAULT CURRENT_DATE
);

INSERT INTO usage_pricing (name) VALUES
    -- Conversation LLM
    ('primary'),
    -- Internal LLM configs (tier-qualified: different models per free/cof)
    ('analysis:cof'), ('analysis:free'),
    ('consolidation:cof'), ('consolidation:free'),
    ('entity_gc:cof'), ('entity_gc:free'),
    ('extraction:cof'), ('extraction:free'),
    ('forage:cof'), ('forage:free'),
    ('phoneafriend_claude:cof'), ('phoneafriend_claude:free'),
    ('phoneafriend_gemini:cof'), ('phoneafriend_gemini:free'),
    ('relationship:cof'), ('relationship:free'),
    ('rewriter:cof'), ('rewriter:free'),
    ('summary:cof'), ('summary:free'),
    ('tidyup:cof'), ('tidyup:free')
ON CONFLICT (name) DO NOTHING;

UPDATE usage_pricing SET input_price_per_mtok = 5.000000, output_price_per_mtok = 25.000000 WHERE name = '__default__';

GRANT SELECT ON usage_pricing TO mira_dbuser;

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
    temperature_unit VARCHAR(20) NOT NULL DEFAULT 'fahrenheit' CHECK (temperature_unit IN ('fahrenheit', 'celsius')),

    -- Activity-based time tracking (vacation-proof scoring)
    cumulative_activity_days INT DEFAULT 0,
    last_activity_date DATE,

    -- Conversation LLM preference
    conversation_llm VARCHAR(20) DEFAULT 'primary' REFERENCES conversation_llm(name),

    -- Balance in USD (OSS: seeded high since user brings own API key)
    balance_usd DECIMAL(12,6) NOT NULL DEFAULT 0.00,

    -- Stripe billing
    stripe_customer_id VARCHAR(255),
    stripe_payment_method_id VARCHAR(255),
    auto_recharge_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    auto_recharge_amount_usd DECIMAL(10,6) NOT NULL DEFAULT 10.00,
    auto_recharge_acknowledged_at TIMESTAMP WITH TIME ZONE,
    last_drip_applied_at DATE,

    -- User portrait (synthesized from segment summaries every 14 activity days)
    portrait TEXT,
    portrait_generated_at TIMESTAMP WITH TIME ZONE
);

-- Grant SELECT on conversation_llm to application user
GRANT SELECT ON conversation_llm TO mira_dbuser;

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

CREATE TABLE IF NOT EXISTS api_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA256 hex = 64 chars
    name VARCHAR(100) NOT NULL DEFAULT 'API Token',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,  -- NULL = never expires
    last_used_at TIMESTAMP WITH TIME ZONE,
    revoked_at TIMESTAMP WITH TIME ZONE  -- soft delete for audit trail
);

-- Index for fast token validation (most common operation)
CREATE INDEX IF NOT EXISTS idx_api_tokens_hash ON api_tokens(token_hash) WHERE revoked_at IS NULL;

-- Index for listing user's tokens
CREATE INDEX IF NOT EXISTS idx_api_tokens_user ON api_tokens(user_id) WHERE revoked_at IS NULL;

-- Unique constraint on token name per user (only for non-revoked tokens)
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_tokens_user_name_unique ON api_tokens(user_id, name) WHERE revoked_at IS NULL;

COMMENT ON TABLE api_tokens IS 'Persistent API tokens for programmatic access (hashed, shown once at creation)';

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
-- DOMAINDOC SHARING (cross-user domain document collaboration)
-- =====================================================================

CREATE TABLE IF NOT EXISTS domaindoc_shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    domaindoc_label TEXT NOT NULL,
    collaborator_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected', 'revoked')),
    invited_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    accepted_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(owner_user_id, domaindoc_label, collaborator_user_id)
);

CREATE INDEX idx_domaindoc_shares_collaborator ON domaindoc_shares(collaborator_user_id, status);
CREATE INDEX idx_domaindoc_shares_owner ON domaindoc_shares(owner_user_id, status);
CREATE INDEX idx_domaindoc_shares_label ON domaindoc_shares(domaindoc_label);

COMMENT ON TABLE domaindoc_shares IS 'Cross-user domaindoc sharing with consent flow (pending→accepted)';
COMMENT ON COLUMN domaindoc_shares.status IS 'pending: awaiting acceptance, accepted: collaborator has access, rejected: collaborator declined, revoked: owner revoked access';

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
COMMENT ON COLUMN continuums.metadata IS 'Flexible storage for continuum-level configuration and state';

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    continuum_id UUID NOT NULL REFERENCES continuums(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    tool_call_id TEXT,
    is_error BOOLEAN NOT NULL DEFAULT FALSE
);

COMMENT ON COLUMN messages.content IS 'Message content - text for simple messages, JSON for multimodal content blocks';
COMMENT ON COLUMN messages.metadata IS 'Message metadata: has_tool_calls, tool_calls, is_summary, summary_type, etc.';
COMMENT ON COLUMN messages.tool_call_id IS 'Tool call identifier for role=tool messages.';
COMMENT ON COLUMN messages.is_error IS 'Whether this tool message reports an error.';

-- Set LZ4 compression for large text columns
ALTER TABLE messages ALTER COLUMN content SET COMPRESSION lz4;

-- Add segment embedding column for segment sentinels
ALTER TABLE messages ADD COLUMN IF NOT EXISTS segment_embedding vector(768);

COMMENT ON COLUMN messages.segment_embedding IS 'mdbr-leaf-ir-asym embedding (768-dim) for segment boundary sentinels (used for segment search)';

-- =====================================================================
-- MESSAGE INDEXES
-- =====================================================================

-- User ID index for RLS policy filtering
CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);

-- Tool call ID index for joining tool results to tool calls
CREATE INDEX IF NOT EXISTS idx_messages_tool_call_id
    ON messages(tool_call_id)
    WHERE tool_call_id IS NOT NULL;

-- Continuum ID index for conversation retrieval
CREATE INDEX IF NOT EXISTS idx_messages_continuum_id ON messages(continuum_id);

-- Created timestamp index for temporal queries
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- Unique partial index: at most one active segment sentinel per continuum
-- Prevents TOCTOU race in _ensure_active_segment from creating duplicate sentinels
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_active_segment_unique ON messages (continuum_id)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND metadata->>'status' = 'active';

-- Composite index for active segment queries (continuum + temporal ordering)
CREATE INDEX IF NOT EXISTS idx_messages_active_segments ON messages (continuum_id, created_at)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND metadata->>'status' = 'active';

-- GIN index on metadata for segment boundary queries
CREATE INDEX IF NOT EXISTS idx_messages_segment_metadata ON messages USING gin (metadata)
    WHERE metadata->>'is_segment_boundary' = 'true';

-- HNSW vector index for segment embedding similarity search
-- Partial index: only segment boundaries with embeddings
CREATE INDEX IF NOT EXISTS idx_messages_segment_embedding ON messages
    USING hnsw (segment_embedding vector_cosine_ops)
    WHERE metadata->>'is_segment_boundary' = 'true'
      AND segment_embedding IS NOT NULL;

COMMENT ON TABLE messages IS 'All conversation messages; segments implemented as sentinel messages with is_segment_boundary=true in metadata';
COMMENT ON COLUMN messages.segment_embedding IS 'mdbr-leaf-ir-asym 768d embedding for segment sentinels. See docs/SystemsOverview/segment_system_overview.md for architecture details.';

-- =====================================================================
-- MEMORIES TABLE (core long-term memory storage)
-- =====================================================================

CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding vector(768),  -- mdbr-leaf-ir-asym embeddings for memory search
    search_vector tsvector,  -- Full-text search vector for BM25-style retrieval
    importance_score NUMERIC(5,3) NOT NULL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER NOT NULL DEFAULT 0,
    mention_count INTEGER NOT NULL DEFAULT 0,  -- Explicit LLM references (strongest importance signal)
    last_accessed TIMESTAMP WITH TIME ZONE,
    happens_at TIMESTAMP WITH TIME ZONE,

    -- Link tracking arrays for efficient hub scoring
    inbound_links JSONB DEFAULT '[]'::jsonb,  -- Array of {source_id, link_type, reasoning, created_at}
    outbound_links JSONB DEFAULT '[]'::jsonb, -- Array of {target_id, link_type, reasoning, created_at}
    entity_links JSONB DEFAULT '[]'::jsonb,   -- Array of {uuid, type, name}

    -- Metadata
    is_archived BOOLEAN DEFAULT FALSE,
    archived_at TIMESTAMP WITH TIME ZONE,

    consolidation_rejection_count INTEGER DEFAULT 0,

    -- Activity day snapshots for vacation-proof scoring
    activity_days_at_creation INT,
    activity_days_at_last_access INT,

    -- Annotations for contextual notes
    annotations JSONB DEFAULT '[]'::jsonb,

    -- Source segment for context exploration
    source_segment_id UUID  -- Segment this memory was extracted from (enables context exploration via continuum_tool)
);

COMMENT ON TABLE memories IS 'Long-term memory storage with embeddings, links, and activity-based decay';
COMMENT ON COLUMN memories.text IS 'Memory content text';
COMMENT ON COLUMN memories.embedding IS 'mdbr-leaf-ir-asym 768-dimensional embedding for semantic similarity search';
COMMENT ON COLUMN memories.search_vector IS 'Full-text search vector for BM25-style retrieval';
COMMENT ON COLUMN memories.importance_score IS 'Memory importance (0.0-1.0) used for retrieval ranking';
COMMENT ON COLUMN memories.happens_at IS 'When the memory event occurred (for temporal context)';
COMMENT ON COLUMN memories.inbound_links IS 'JSONB array of memories that link TO this memory';
COMMENT ON COLUMN memories.outbound_links IS 'JSONB array of memories this memory links TO';
COMMENT ON COLUMN memories.entity_links IS 'JSONB array of entity references this memory mentions';
COMMENT ON COLUMN memories.activity_days_at_creation IS 'User cumulative_activity_days when memory was created (snapshot for decay calculation)';
COMMENT ON COLUMN memories.activity_days_at_last_access IS 'User cumulative_activity_days when memory was last accessed (snapshot for recency calculation)';
COMMENT ON COLUMN memories.annotations IS 'Contextual notes: [{text, created_at, source}]';
COMMENT ON COLUMN memories.source_segment_id IS 'Segment this memory was extracted from (enables context exploration via continuum_tool search_within_segment)';

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

-- Partial index on source_segment_id for segment-to-memory tracing
CREATE INDEX IF NOT EXISTS idx_memories_source_segment_id
    ON memories(source_segment_id)
    WHERE source_segment_id IS NOT NULL;

COMMENT ON INDEX idx_memories_user_id IS 'B-tree index for RLS policy filtering - essential for multi-user performance';
COMMENT ON INDEX idx_memories_search_vector IS 'GIN index for full-text search operations';
COMMENT ON INDEX idx_memories_embedding_ivfflat IS 'IVFFlat index for fast cosine similarity search - prevents O(n) sequential scans during deduplication and retrieval';
COMMENT ON INDEX idx_memories_source_segment_id IS 'Partial B-tree index for tracing memories back to source segments';

-- Trigger function to maintain search vectors
CREATE OR REPLACE FUNCTION update_memories_search_vector() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', NEW.text);
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
-- GLOBAL MEMORIES TABLE (centralized, no RLS, no decay)
-- =====================================================================
-- Global memories are manually curated facts accessible to all users.
-- They surface through ProactiveService retrieval but cannot be queried,
-- linked to, or annotated via the memory_tool (which only sees personal memories).
-- Cross-table linking silently fails by design - global memory IDs return
-- "memory not found" when users attempt to link to them.

CREATE TABLE IF NOT EXISTS global_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    embedding vector(768),  -- mdbr-leaf-ir-asym embeddings for memory search
    search_vector tsvector,  -- Full-text search vector for BM25-style retrieval
    importance_score NUMERIC(5,3) NOT NULL DEFAULT 1.0 CHECK (importance_score >= 0 AND importance_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE,
    happens_at TIMESTAMP WITH TIME ZONE,  -- Optional temporal context
    entity_links JSONB DEFAULT '[]'::jsonb,  -- Array of {uuid, type, name}
    -- Link arrays for global ↔ global relationships (manually curated)
    inbound_links JSONB DEFAULT '[]'::jsonb,
    outbound_links JSONB DEFAULT '[]'::jsonb,
    is_archived BOOLEAN DEFAULT FALSE,
    archived_at TIMESTAMP WITH TIME ZONE
);

COMMENT ON TABLE global_memories IS 'Centralized memories accessible to all users - no RLS, no decay. Manually curated via psql.';
COMMENT ON COLUMN global_memories.importance_score IS 'Fixed at 1.0 for global memories (no decay applied)';
COMMENT ON COLUMN global_memories.entity_links IS 'Entity references for potential future hub discovery integration';
COMMENT ON COLUMN global_memories.inbound_links IS 'JSONB array of other global memories that link TO this memory';
COMMENT ON COLUMN global_memories.outbound_links IS 'JSONB array of other global memories this memory links TO';

-- Set LZ4 compression for large text columns
ALTER TABLE global_memories ALTER COLUMN text SET COMPRESSION lz4;

-- =====================================================================
-- GLOBAL MEMORY INDEXES
-- =====================================================================

-- Full-text search index for keyword-based retrieval
CREATE INDEX IF NOT EXISTS idx_global_memories_search_vector ON global_memories USING gin (search_vector);

-- Vector similarity index for semantic search (IVFFlat algorithm)
CREATE INDEX IF NOT EXISTS idx_global_memories_embedding_ivfflat
    ON global_memories USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON INDEX idx_global_memories_search_vector IS 'GIN index for full-text search operations';
COMMENT ON INDEX idx_global_memories_embedding_ivfflat IS 'IVFFlat index for fast cosine similarity search';

-- Trigger to maintain search vectors (reuses existing function)
CREATE TRIGGER global_memories_search_vector_update
BEFORE INSERT OR UPDATE OF text
ON global_memories
FOR EACH ROW
EXECUTE FUNCTION update_memories_search_vector();

-- =====================================================================
-- GLOBAL MEMORY PERMISSIONS (NO RLS)
-- =====================================================================
-- Note: RLS is NOT enabled on this table - all users can read global memories.
-- Write access is intended for manual curation via psql only.

GRANT SELECT, INSERT, UPDATE, DELETE ON global_memories TO mira_dbuser;
GRANT ALL ON global_memories TO mira_admin;

-- =====================================================================
-- ENTITIES TABLE (knowledge graph nodes)
-- =====================================================================

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- PERSON, ORG, GPE, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, NORP, FAC
    embedding vector(300),  -- spaCy word vector for semantic similarity (300d from en_core_web_lg)
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
    status TEXT NOT NULL CHECK (status IN ('submitted', 'processing', 'result_processing', 'completed', 'failed', 'expired', 'cancelled')),
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
    batch_type TEXT NOT NULL CHECK (batch_type IN ('relationship_classification', 'consolidation')),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    request_payload JSONB NOT NULL,
    input_data JSONB NOT NULL,
    items_submitted INTEGER NOT NULL,
    items_completed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    status TEXT NOT NULL CHECK (status IN ('submitted', 'processing', 'result_processing', 'completed', 'failed', 'expired', 'cancelled')),
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
-- FEEDBACK SIGNALS TABLE (DIY reinforcement loop)
-- =====================================================================

CREATE TABLE IF NOT EXISTS feedback_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    segment_id UUID NOT NULL,
    continuum_id UUID NOT NULL,
    signal_type TEXT NOT NULL CHECK (signal_type IN ('alignment', 'misalignment', 'contextual_pass')),
    section_id TEXT NOT NULL,
    strength TEXT NOT NULL CHECK (strength IN ('strong', 'moderate', 'mild')),
    evidence TEXT NOT NULL,
    extracted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    synthesized BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE feedback_signals IS 'Assessment signals for the user model pipeline';
COMMENT ON COLUMN feedback_signals.signal_type IS 'Type: alignment, misalignment, contextual_pass';
COMMENT ON COLUMN feedback_signals.section_id IS 'System prompt section ID this signal references';
COMMENT ON COLUMN feedback_signals.strength IS 'Signal strength: strong, moderate, mild';
COMMENT ON COLUMN feedback_signals.synthesized IS 'True after user model synthesis has processed this signal';

CREATE INDEX IF NOT EXISTS idx_feedback_signals_user_id ON feedback_signals(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_signals_user_type ON feedback_signals(user_id, signal_type);
CREATE INDEX IF NOT EXISTS idx_feedback_signals_unsynthesized ON feedback_signals(user_id) WHERE NOT synthesized;
CREATE INDEX IF NOT EXISTS idx_feedback_signals_section_id ON feedback_signals(user_id, section_id) WHERE NOT synthesized;

-- =====================================================================
-- FEEDBACK SYNTHESIS TRACKING TABLE
-- =====================================================================

CREATE TABLE IF NOT EXISTS feedback_synthesis_tracking (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    activity_days_at_last_synthesis INTEGER NOT NULL DEFAULT 0,
    last_synthesis_at TIMESTAMP WITH TIME ZONE,
    last_synthesis_output TEXT,
    needs_checkin BOOLEAN NOT NULL DEFAULT FALSE,
    checkin_response TEXT
);

COMMENT ON TABLE feedback_synthesis_tracking IS 'Tracks synthesis state for the user model pipeline (modular arithmetic on cumulative_activity_days)';
COMMENT ON COLUMN feedback_synthesis_tracking.activity_days_at_last_synthesis IS 'Snapshot of users.cumulative_activity_days when synthesis last ran (modular arithmetic base)';
COMMENT ON COLUMN feedback_synthesis_tracking.last_synthesis_output IS 'User model XML from previous synthesis for evolutionary refinement';
COMMENT ON COLUMN feedback_synthesis_tracking.needs_checkin IS 'True when user model contains check-in topics for behavioral debrief';

-- =====================================================================
-- BILLING & STRIPE TABLES
-- =====================================================================

CREATE TABLE IF NOT EXISTS billing_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    transaction_type VARCHAR(20) NOT NULL,
    amount_usd DECIMAL(12,6) NOT NULL,
    balance_after DECIMAL(12,6) NOT NULL,
    model_name VARCHAR(100),
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    stripe_payment_intent_id VARCHAR(255),
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE billing_transactions IS 'Audit log of all billing events: usage charges, deposits, drips, refunds';

CREATE INDEX IF NOT EXISTS idx_billing_txn_user_created ON billing_transactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_billing_txn_type ON billing_transactions(transaction_type)
    WHERE transaction_type IN ('recharge_failed', 'deposit');
CREATE UNIQUE INDEX IF NOT EXISTS idx_billing_positive_stripe_deposit_once
    ON billing_transactions(stripe_payment_intent_id)
    WHERE transaction_type = 'deposit'
      AND amount_usd > 0
      AND stripe_payment_intent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS stripe_webhook_events (
    event_id VARCHAR(255) PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_successfully BOOLEAN NOT NULL DEFAULT FALSE,
    processing_lock_token UUID,
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_finished_at TIMESTAMP WITH TIME ZONE,
    payload JSONB NOT NULL
);

COMMENT ON TABLE stripe_webhook_events IS 'Idempotency tracking for Stripe webhooks - prevents double-processing';

CREATE INDEX IF NOT EXISTS idx_stripe_webhook_processed ON stripe_webhook_events(processed_at);
CREATE INDEX IF NOT EXISTS idx_stripe_webhook_processing_lock
    ON stripe_webhook_events(processing_lock_token)
    WHERE processing_lock_token IS NOT NULL;

-- =====================================================================
-- PERMISSIONS
-- =====================================================================

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mira_dbuser') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON
            users, users_trash, magic_links,
            user_activity_days, domain_knowledge_blocks, domain_knowledge_block_content,
            domaindoc_shares,
            continuums, messages,
            memories, entities, extraction_batches, post_processing_batches,
            feedback_signals, feedback_synthesis_tracking,
            billing_transactions
        TO mira_dbuser;
        GRANT SELECT, INSERT, UPDATE ON stripe_webhook_events TO mira_dbuser;
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

ALTER TABLE api_tokens ENABLE ROW LEVEL SECURITY;
CREATE POLICY api_tokens_user_policy ON api_tokens
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

ALTER TABLE feedback_signals ENABLE ROW LEVEL SECURITY;
CREATE POLICY feedback_signals_user_policy ON feedback_signals
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE feedback_synthesis_tracking ENABLE ROW LEVEL SECURITY;
CREATE POLICY feedback_synthesis_tracking_user_policy ON feedback_synthesis_tracking
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id')::uuid);

ALTER TABLE billing_transactions ENABLE ROW LEVEL SECURITY;
CREATE POLICY billing_transactions_user_policy ON billing_transactions
    FOR ALL TO PUBLIC
    USING (user_id = current_setting('app.current_user_id', true)::uuid);

-- Domaindoc sharing: owner manages their own shares, collaborator views/accepts/rejects theirs
ALTER TABLE domaindoc_shares ENABLE ROW LEVEL SECURITY;
CREATE POLICY domaindoc_shares_owner_all ON domaindoc_shares
    FOR ALL TO PUBLIC
    USING (owner_user_id = current_setting('app.current_user_id')::uuid)
    WITH CHECK (owner_user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY domaindoc_shares_collaborator_select ON domaindoc_shares
    FOR SELECT TO PUBLIC
    USING (collaborator_user_id = current_setting('app.current_user_id')::uuid);
CREATE POLICY domaindoc_shares_collaborator_update ON domaindoc_shares
    FOR UPDATE TO PUBLIC
    USING (collaborator_user_id = current_setting('app.current_user_id')::uuid)
    WITH CHECK (collaborator_user_id = current_setting('app.current_user_id')::uuid);

-- Note: extraction_batches and post_processing_batches do NOT have RLS
-- These are system tracking tables accessed by admin polling jobs
-- They contain no user data, only batch job metadata
