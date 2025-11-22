# MIRA OSS Release Notes

**Version:** 1.0.0 (Initial Open Source Release)
**Date:** 2025

---

## What is MIRA?

MIRA (Memory-Integrated Reasoning Assistant) is a conversational AI system that maintains persistent memory across interactions. It uses sophisticated context window curation to provide relevant historical context while managing token budgets efficiently.

**Core Philosophy:** Context windows aren't constraints to work around—they're design surfaces. MIRA treats every token as valuable real estate, using complexity-based loading and information density metrics to ensure the most relevant context appears in every conversation.

---

## Key Features

### Complexity-Based Context Curation

MIRA doesn't just load recent conversations—it prioritizes **information density** over message count.

**How it works:**
- Conversations are segmented and scored 1-3 based on cognitive complexity
- A 5-message quantum physics discussion scores higher than a 40-message chat about lunch
- Context budget fills with high-value segments first
- Configuration: `session_summary_complexity_limit` (default: 8)

**Implementation:** `cns/core/segment_cache_loader.py:110-187`

### Long-Term Memory System

Automated memory extraction, deduplication, and surfacing:

- **Batch Processing:** Uses Anthropic Batch API for 50% cost savings on extraction
- **Dual Deduplication:** Fuzzy text matching (rapidfuzz) + vector similarity search
- **Relationship Classification:** Memories linked with types: consolidates, conflicts, supports, supersedes, related
- **Entity Linking:** SpaCy NER extracts entities, creates memory→entity relationships
- **Momentum Decay:** `access_count * 0.95^(activity_days_since_last_access)` - vacation-proof engagement

**Implementation:** `lt_memory/` directory with modular processing pipeline

### Event-Driven Architecture

Three coordinated systems communicate via event bus:

1. **CNS (Central Nervous System):** Conversation management via immutable Continuum aggregate
2. **Working Memory:** Trinket-based system prompt composition with cache optimization
3. **LT_Memory:** Batch memory extraction, linking, and refinement

**Key Events:**
- `TurnCompletedEvent` - Triggers memory extraction, trinket updates
- `SegmentTimeoutEvent` - Initiates segment collapse after 3 hours inactivity
- `SegmentCollapsedEvent` - Summary generated, triggers downstream processing

### Tool System with On-Demand Loading

Tools load when needed, not at startup:

- **Essential tools** load at startup: `webaccess_tool`, `reminder_tool`, `invokeother_tool`
- **Other tools** load on-demand via `invokeother_tool`
- **Result:** Faster startup, smaller initial context

**Included Tools:**
- `contacts_tool` - Contact management with UUID linking
- `reminder_tool` - Time-aware reminders with timezone support
- `punchclock_tool` - Work session tracking
- `weather_tool` - Weather lookups
- `kasa_tool` - TP-Link smart home control
- `getcontext_tool` - Semantic search across conversation history
- `webaccess_tool` - Web browsing via Playwright

**Creating tools:** See `tools/HOW_TO_BUILD_A_TOOL.md` (598 lines, pattern-indexed)

### Working Memory Trinkets

Trinkets inject dynamic content into system prompts based on events:

- `user_info_trinket` - User's overarching knowledge/preferences
- `tool_loader_trinket` - Available and loaded tools display
- `reminder_manager` - Active reminders with relative timestamps
- `proactive_memory_trinket` - Automatically surfaced memories with confidence scores
- `domain_knowledge_trinket` - Context-specific knowledge blocks
- `time_manager` - Current datetime in user's timezone
- `punchclock_trinket` - Work session status

**Cache optimization:** Cached trinkets placed sequentially for Claude's prefix caching efficiency.

**Creating trinkets:** See `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md` (606 lines)

### Thinking Budget Control

Per-conversation control over Claude's extended thinking:

```python
# Via API actions endpoint
POST /v0/api/actions
{
  "domain": "continuum",
  "action": "set_thinking_budget_preference",
  "params": {"budget": 8000}  # or 0 to disable, null for system default
}
```

Preference cached in Valkey, reset after segment collapse.

### Relative Time Formatting

All timestamps displayed as relative time for reduced cognitive load:

- "26 days ago" instead of "2024-10-14"
- Granularity: just now → minutes → hours → days → weeks → months → years

**Implementation:** `utils/timezone_utils.py:format_relative_time()`

---

## Architecture Overview

### Technology Stack

- **Backend:** FastAPI with async support
- **Database:** PostgreSQL 17 with pgvector extension
- **Cache:** Valkey (Redis-compatible)
- **Secrets:** HashiCorp Vault
- **LLM:** Anthropic Claude (primary), Groq (fast execution), OpenAI (fallback)
- **Embeddings:** AllMiniLM-L6-v2 (384-dim for real-time), OpenAI (1536-dim for storage)
- **NER:** SpaCy with en_core_web_lg model
- **Reranking:** BGE-reranker-base

### User Isolation

PostgreSQL Row Level Security (RLS) with contextvars provides automatic user isolation:

- All user-scoped queries enforce `user_id` filtering at database level
- Context flows automatically from authentication through to RLS enforcement
- Tools automatically get user-scoped data access via `self.db` property

### Request → Response Flow

1. User input → HTTP endpoint
2. Get/create continuum from pool (Valkey → DB if miss)
3. Add user message to continuum cache
4. Generate analysis (touchstone) for weighted context
5. Generate 384-dim embedding (shared across turn)
6. Surface relevant memories via vector similarity
7. Compose system prompt via event bus + trinkets
8. Get enabled tool definitions
9. Call LLM with streaming
10. Parse tags from response
11. Add assistant message
12. Batch persist via Unit of Work
13. Publish TurnCompletedEvent
14. Return response

### Memory Lifecycle

1. Segment collapse after 3 hours inactivity
2. Summary generation with complexity score
3. Batch submission to Anthropic Batch API
4. Memory extraction from user messages
5. Fuzzy + vector deduplication
6. Embedding generation
7. Relationship classification
8. Entity extraction (SpaCy NER)
9. Entity linking
10. Consolidation review (weekly)
11. Refinement pass (verbose trimming)
12. Decay scoring based on activity

---

## API Endpoints

### Chat

```
POST /v0/api/chat
Authorization: Bearer <api_key>

{
  "message": "string",
  "image": "base64_string",      // optional
  "image_type": "image/jpeg"     // required if image present
}

Response:
{
  "success": true,
  "data": {
    "continuum_id": "uuid",
    "response": "assistant reply",
    "metadata": {
      "tools_used": [],
      "referenced_memories": [],
      "surfaced_memories": [],
      "processing_time_ms": 1234
    }
  }
}
```

### Actions

```
POST /v0/api/actions
Authorization: Bearer <api_key>

{
  "domain": "continuum",
  "action": "set_thinking_budget_preference",
  "params": {"budget": 8000}
}
```

### Health

```
GET /v0/api/health

Response:
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "valkey": "connected",
    "vault": "connected"
  }
}
```

---

## Installation

### Automated Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mira-OSS.git
cd mira-OSS

# Run deployment script
./deploy.sh
```

The deployment script handles:
- Platform detection (Linux/macOS)
- System dependencies (PostgreSQL, Valkey, Vault)
- Python environment and dependencies
- AI model downloads (AllMiniLM, BGE reranker, spaCy)
- Vault initialization and credential storage
- Database schema creation
- Service configuration

### System Requirements

**Linux (Ubuntu/Debian):**
- PostgreSQL 17 with pgvector
- Valkey
- HashiCorp Vault 1.18.3
- Python 3.12+
- 10GB+ disk space

**macOS:**
- Homebrew
- Same software stack
- 10GB+ disk space

### Port Requirements

- **1993** - MIRA API server
- **8200** - HashiCorp Vault
- **6379** - Valkey cache
- **5432** - PostgreSQL

### API Keys Required

**Required:**
- Anthropic API key
- Groq API key

**Optional:**
- OpenAI API key (emergency fallback)
- Google Maps API key (maps_tool)
- TP-Link Kasa account (kasa_tool)
- Kagi API key (web search)

---

## Configuration

### Main Configuration

Configuration uses Pydantic models in `config/config.py`:

```python
# LLM Configuration
execution_model = "openai/gpt-oss-20b"  # Via Groq
execution_endpoint = "https://api.groq.com/openai/v1/chat/completions"
extended_thinking = False  # Enable for complex reasoning
extended_thinking_budget = 1024  # Minimum tokens when enabled

# Memory Configuration
session_summary_complexity_limit = 8  # Complexity budget for context loading
segment_timeout = 180  # Minutes before segment collapse

# Tool Configuration
essential_tools = ["webaccess_tool", "reminder_tool", "invokeother_tool"]
```

### Prompts

LLM prompts are in `config/prompts/`:

- `memory_extraction_system.txt` - Memory extraction guidelines (216 lines)
- `segment_summary_system.txt` - Segment summarization with complexity scoring (126 lines)
- `memory_relationship_classification.txt` - Relationship type classification

### System Prompt

The AI persona and behavior is defined in `config/system_prompt.txt`. Sections include:
- Identity and background
- Foundational understanding
- Communication style
- Response formatting

---

## CLI Usage

### Interactive Mode

```bash
# Auto-starts server if not running
python talkto_mira.py
```

### Headless Mode

```bash
# Requires running server
python talkto_mira.py --headless "What's the weather today?"
```

### Shell Alias

After installation, use the `mira` alias:

```bash
source ~/.bashrc
mira
```

---

## Development

### Running Tests

```bash
# All tests
pytest

# Specific test
pytest tests/test_file.py::test_function
```

### Code Quality

```bash
# Linting
flake8

# Type checking
mypy .

# Formatting
black .
```

### Database Access

```bash
psql -U postgres -h localhost -d mira_service
```

### Developer Guides

- `tools/HOW_TO_BUILD_A_TOOL.md` - Pattern-indexed tool development guide
- `working_memory/trinkets/HOW_TO_BUILD_A_TRINKET.md` - Trinket development guide
- `CLAUDE.md` - AI assistant instructions with design principles
- `docs/mira-architecture-overview.md` - Comprehensive architecture documentation

---

## Database Schema

### Core Tables

**users** - User accounts with preferences
- `id` (UUID, PK)
- `email` (VARCHAR)
- `first_name`, `last_name` (VARCHAR)
- `timezone` (VARCHAR)
- `overarching_knowledge` (TEXT) - User-level context
- `cumulative_activity_days` (INT) - Engagement metric

**continuums** - Conversation containers
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `metadata` (JSONB) - Touchstones, embeddings

**messages** - Conversation messages
- `id` (UUID, PK)
- `continuum_id` (UUID, FK)
- `role` (VARCHAR) - user, assistant, system
- `content` (TEXT)
- `segment_embedding` (vector(384)) - For segment boundaries

**memories** - Extracted long-term memories
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `text` (TEXT)
- `embedding` (vector(1536))
- `importance_score` (FLOAT)
- `confidence` (FLOAT)
- `created_at`, `last_accessed_at` (TIMESTAMP)

**memory_relationships** - Memory links
- `source_memory_id`, `target_memory_id` (UUID, FK)
- `relationship_type` (VARCHAR) - consolidates, conflicts, supports, supersedes, related

**entities** - Extracted named entities
- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `name` (VARCHAR)
- `entity_type` (VARCHAR) - PERSON, ORG, GPE, etc.

**user_activity_days** - Daily activity tracking
- `user_id` (UUID, FK)
- `activity_date` (DATE)
- `message_count` (INT)

---

## Performance Characteristics

### Typical Timings

- **Message processing:** 2-5 seconds
- **Memory surfacing:** 100-300ms
- **Vector search:** 50-200ms (ivfflat index)
- **Segment collapse:** 500ms-2s (includes LLM summary)

### Optimization Strategies

1. **Complexity-based loading** - Rich segments prioritized over trivial ones
2. **Prefix caching** - Static content ordered for Claude's cache
3. **Singleton initialization** - Models loaded at startup
4. **Background extraction** - Non-blocking boot extraction
5. **Batch API** - 50% cost savings on memory extraction

### Memory Requirements

- AllMiniLM-L6-v2: ~80MB
- BGE-reranker-base: ~1.1GB
- spaCy en_core_web_lg: ~800MB
- **Total models:** ~2GB RAM

---

## Security

### Credential Management

All sensitive values stored in HashiCorp Vault:

- API keys at `secret/mira/api_keys`
- Database URLs at `secret/mira/database`
- User-specific credentials via `UserCredentialService`

### Authentication

Single-user mode with Bearer token authentication:

```bash
curl -H "Authorization: Bearer mira_<token>" http://localhost:1993/v0/api/chat
```

API key generated on first startup, stored in Vault.

### User Isolation

PostgreSQL Row Level Security (RLS) enforces user isolation at database level. All queries automatically filtered by `user_id`.

### Production Hardening

1. **Change default passwords:**
   ```sql
   ALTER USER mira_admin WITH PASSWORD 'strong-password';
   ```

2. **Secure Vault init keys:**
   ```bash
   chmod 600 /opt/vault/init-keys.txt
   # Move to secure storage
   ```

3. **Configure firewall:**
   ```bash
   ufw allow 1993/tcp   # MIRA API
   ufw deny 8200/tcp    # Vault (internal only)
   ufw deny 6379/tcp    # Valkey (internal only)
   ufw deny 5432/tcp    # PostgreSQL (internal only)
   ```

---

## Troubleshooting

### Common Issues

**Server won't start:**
- Check port availability: `lsof -i :1993`
- Verify Vault is running: `vault status`
- Check PostgreSQL: `pg_isready -h localhost`
- Check Valkey: `valkey-cli ping`

**Authentication errors:**
- Verify API key in Vault: `vault kv get secret/mira/api_keys`
- Check Authorization header format: `Bearer mira_<token>`

**Memory extraction not working:**
- Check scheduled tasks: logs show batch polling
- Verify Anthropic API key in Vault
- Check batch status: `lt_memory/batching.py` logs

**Tool not loading:**
- Verify tool enabled in config
- Check `invokeother_tool` logs
- Ensure tool file exists in `tools/implementations/`

### Logs

Application logs to stdout. For systemd:
```bash
journalctl -u mira -f
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow code style guidelines in `CLAUDE.md`
4. Write tests for new functionality
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use Black formatter (88 char lines)
- Type hints required for all function signatures
- Google-style docstrings

---

## License

See LICENSE.md for details.

---

## Support

- **GitHub Issues:** https://github.com/yourusername/mira-OSS/issues
- **Documentation:** https://github.com/yourusername/mira-OSS/docs
- **Hosted Version:** https://miraos.org

---

## Acknowledgments

Built with:
- [Anthropic Claude](https://anthropic.com) - Primary LLM
- [FastAPI](https://fastapi.tiangolo.com) - Web framework
- [PostgreSQL](https://postgresql.org) + [pgvector](https://github.com/pgvector/pgvector) - Database
- [HashiCorp Vault](https://vaultproject.io) - Secrets management
- [Sentence Transformers](https://sbert.net) - Embeddings
- [SpaCy](https://spacy.io) - NLP/NER

---

**End of Release Notes**
