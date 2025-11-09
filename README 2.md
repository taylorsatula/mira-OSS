# MIRA

**Memory-Integrated Reasoning Assistant**

A conversational AI that remembers.

---

> **Note**: This is the first commit. I'm sure there are going to be some weird broken things I'll fix over the next week or two (I'll add a proper deploy script). For now if you'd like to use a real-live hosted instance of MIRA, you can access it at [miraos.org](https://miraos.org). Otherwise please enjoy looking through the codebase.

---

## What MIRA Is

MIRA is a FastAPI server that runs Claude with persistent memory. Conversations build on each other. The system extracts facts from what you talk about, stores them in PostgreSQL, and surfaces relevant memories when they matter.

You get a conversational partner that knows your history, your preferences, your projects. It learns through use.

### ---

A note from the developer: I strive to build software I would personally use. I loathe garbo half-implemented buggy slopcode. **All functionality described below works properly and in concert with the other components. The large majority of this README.md document was programmatically generated from the actual code in the repo.**

I toiled for months on this project and actualized the final product I wanted to release to the world and use in my own life.

I hope you find it interesting and perhaps contribute back to the MIRA repository. I pledge that MIRA will have an OSS release branch for as long as I control the repository. I believe in OSS and the OSS foundational technologies used to build MIRA.

## ---

## What Makes MIRA Different

**Memory that earns its keep**: MIRA doesn't just dump everything into a vector database. Memories are extracted as discrete facts, deduplicated through consolidation, refined into concise statements, and linked through typed relationships (supports, conflicts, supersedes). Memories track their own access patterns. Get used enough, they stick around. Sit idle, they decay and delete.

**Pre-computed semantic intent**: Before your main LLM call, MIRA generates an "evolved semantic touchstone" - a semantic summary of what you're trying to accomplish. This touchstone gets embedded once and used everywhere: memory retrieval, context building, tool selection. Your conversation context gets weighted with this intent signal. This prevents the "one turn behind" problem where memory searches are based on stale context.

**Progressive disclosure everywhere**: Tools load on demand through the `invokeother_tool` pattern. You see tool hints (name + brief description), call invokeother to load what you need, then use it. Idle tools auto-unload after 5 turns. This cuts context usage by 80-90%. Search results work the same way - summaries first, then message details on demand.

**Event-driven architecture**: Everything coordinates through domain events. Message arrives, event fires. Memory extracted, event fires. Tool used, event fires. Components subscribe to what they care about. The orchestrator, working memory, tool system, and memory extraction all stay synchronized through this event bus.

**Domain knowledge blocks**: MIRA integrates with Letta to maintain topic-specific context containers. Enable your "work" block when at work, your "home" block for personal stuff. Sleeptime agents process conversation batches and update blocks automatically. These inject into your system prompt when enabled.

---

## Memory Systems

MIRA has two separate memory systems that work together.

### Working Memory

Working memory builds your system prompt dynamically every turn. It coordinates specialized components called "trinkets" - each one handles a specific type of context:

- **TimeManager**: Current date, time, and timezone
- **ReminderManager**: Active reminders and deadlines
- **ProactiveMemoryTrinket**: Long-term memories relevant to this conversation
- **ToolGuidanceTrinket**: Hints for tools you haven't loaded yet
- **ToolLoaderTrinket**: Currently loaded tools and their usage
- **ManifestTrinket**: Current conversation state and metadata
- **DomainKnowledgeTrinket**: Enabled domain knowledge blocks from Letta
- **PunchclockTrinket**: Active work sessions

The process: API request triggers `ComposeSystemPromptEvent` → working memory publishes `UpdateTrinketEvent` to each trinket → trinkets publish `TrinketContentEvent` with their content → working memory assembles it all into the final system prompt.

Trinkets are event-driven. They subscribe to relevant events (`TurnCompletedEvent`, `ToolUsedEvent`, etc.) and update their state automatically. Adding new context is just adding a new trinket.

### Long-Term Memory

Long-term memory runs completely in the background. After conversations, it extracts facts, consolidates duplicates, refines verbose statements, links relationships, and tracks what gets used.

**The extraction flow**:

1. Conversation completes → pointer summary coalescence creates extraction event
2. System hydrates messages from the conversation chunk
3. Batches messages and sends to Claude via Batch API
4. LLM extracts discrete facts with metadata (confidence, category, timestamps)
5. Consolidation engine finds duplicates and superseded memories
6. Refinement engine distills verbose memories into concise statements
7. Entity extraction (spaCy) finds people, places, organizations
8. Memories get embedded (384-dim via AllMiniLM)
9. Storage in PostgreSQL with pgvector indexing

**The surfacing flow**:

1. User asks a question → touchstone generated → embedded
2. Proactive memory service searches: vector similarity + graph traversal (linked memories) + access score weighting
3. Top memories ranked by combined score
4. Memories injected into ProactiveMemoryTrinket
5. Access patterns tracked → frequently used memories get higher scores
6. Unused memories decay over time → eventually deleted

Memories form typed relationships:
- **supports**: Memory A provides evidence for memory B
- **conflicts**: Memory A contradicts memory B
- **supersedes**: Memory A replaces outdated memory B
- **related**: Memory A connects thematically to memory B

The graph structure lets MIRA traverse relationships. If memory X is relevant, its supporting memories become relevant too.

### Domain Knowledge Blocks

Domain knowledge is Letta-powered. You create blocks for different contexts (work, home, projects, trips). Enable/disable blocks as needed. Sleeptime agents process conversation batches and update blocks in the background.

When a block is enabled, its content injects into your system prompt through DomainKnowledgeTrinket. This gives Claude persistent, updatable context containers beyond discrete memories.

---

## CNS (Central Nervous System)

CNS is MIRA's conversation orchestrator. It follows Domain-Driven Design with immutable value objects and event sourcing patterns.

### How a Conversation Works

When you send a message:

1. **API receives message** → distributed lock acquired per user (prevents race conditions)
2. **Continuum loaded** → conversation aggregate retrieved from database or created new
3. **Message added to cache** → not yet persisted, lives in memory
4. **Touchstone generated** → fast model summarizes semantic intent before main LLM call
5. **Embedding computed** → single 384-dim vector from weighted context (user message + surrounding messages + touchstone)
6. **Memory surfaced** → proactive memory searches using the embedding
7. **Trinkets updated** → ComposeSystemPromptEvent published, all trinkets contribute content
8. **System prompt assembled** → working memory collects all trinket contributions
9. **LLM called** → Claude processes with composed prompt + message cache + tool schemas
10. **Response parsed** → tag parser looks for special instructions (`<mira:topic_changed/>`, `<mira:need_tool/>`)
11. **TurnCompletedEvent published** → carries the continuum object with full conversation state
12. **Database writes** → messages persisted, caches updated, background tasks queued

### The Continuum Aggregate

A `Continuum` is an immutable value object representing a complete conversation:

- **id**: UUID for the conversation
- **user_id**: Owner (enforced by Row-Level Security)
- **_message_cache**: Recent messages in memory (hot cache)
- **_cumulative_tokens**: Running token count for cache management
- **metadata**: Stores touchstone, embedding, topic, segment info

All state-changing methods return new Continuum objects + domain events. No mutation. This makes the system testable and prevents concurrent modification bugs.

### Domain Events

Everything state-changing emits an event:

- `MessageReceivedEvent` → user message added
- `ToolCallsRequestedEvent` → LLM wants to execute tools
- `TurnCompletedEvent` → conversation turn finished
- `TopicChangedEvent` → conversation subject shifted
- `WorkflowDetectedEvent` → multi-step workflow identified
- `ComposeSystemPromptEvent` → working memory needs updating
- `UpdateTrinketEvent` → specific trinket needs refresh
- `TrinketContentEvent` → trinket publishing new content

Events are immutable dataclass objects. The event bus publishes them synchronously. Subscribers handle what they care about.

### Tag Parser

The tag parser extracts special XML-style tags from LLM responses:

- `<mira:memory_ref="uuid" />` → references specific memories
- `<mira:touchstone>content</mira:touchstone>` → semantic intent summary
- `<mira:my_emotion>emoji</mira:my_emotion>` → emotion tracking (for continuity)
- `<mira:display_title>title</mira:display_title>` → segment display titles
- `<error_analysis error_id="...">content</error_analysis>` → error analysis

These tags let Claude communicate metadata and structure beyond just generating text.

### Segments and Collapse

MIRA uses segments to organize conversation history. Each continuum (one per user) contains messages, with special **sentinel messages** marking segment boundaries.

When a segment needs collapsing (triggered by `SegmentTimeoutEvent`):

1. System finds the segment boundary sentinel message
2. Loads messages between sentinels
3. Generates segment summary with embedding via LLM
4. Updates sentinel message metadata with summary and embedding
5. Publishes downstream events

This creates searchable conversation segments. Later searches can find relevant segments by summary similarity, then drill into message details.

---

## Tool System

MIRA has 9 tools built on a dynamic loading system. Tools you haven't loaded show up as hints. Call `invokeother_tool` to load what you need. Idle tools auto-unload after 5 turns.

### Available Tools

**Core tools** (loaded by default):
- **reminder_tool**: SQLite-based reminders with contact linking, natural language dates, and timezone handling
- **webaccess_tool**: Web scraping and HTTP requests with retry logic and domain filtering

**Integration tools** (load on demand):
- **continuumsearch_tool**: Hybrid vector + BM25 search across conversation history and memories with progressive disclosure
- **contacts_tool**: Contact management with UUID-based linking for cross-tool references
- **punchclock_tool**: Work session tracking with pause/resume support and trinket integration
- **weather_tool**: OpenMeteo forecasts + WBGT heat stress calculations with caching
- **kasa_tool**: TP-Link smart home control with fuzzy device name matching and async operations
- **getcontext_tool**: Async agentic search that runs in background

**Meta tool**:
- **invokeother_tool**: Dynamic tool loader with three modes: load, unload, fallback

### How Dynamic Loading Works

The system uses the `invokeother_tool` pattern to manage context:

1. **Tool hints visible**: Working memory shows all available tools with brief descriptions
2. **Load on demand**: Call `invokeother_tool(mode="load", query="weather_tool")`
3. **Tool becomes available**: Next turn, weather_tool's full definition loads into context
4. **Use the tool**: Call weather_tool directly with your query
5. **Auto-cleanup**: After 5 turns of non-use, tool auto-unloads

**Fallback mode**: When stuck, call `invokeother_tool(mode="fallback")` to load all tools for one turn. They auto-unload aggressively after.

Context savings: 80-90% reduction by loading only what's needed.

### User Data Isolation

Every tool inherits from `Tool` base class with automatic user scoping:

- **self.db**: User-scoped database connection (SQLite or PostgreSQL)
- **self.user_data_path**: User-specific file directory
- **File operations**: `make_dir()`, `get_file_path()`, `open_file()` all scoped automatically

Tool code never filters by `user_id` manually. The architecture handles it.

Storage locations:
- SQLite tool data: `data/users/{user_id}/tools/{tool_name}/`
- Cache files: Tool-specific subdirectories under user path
- Encrypted credentials: Per-user SQLite databases

### Tool State Management

ToolLoaderTrinket tracks everything:

```python
available_tools: Dict[str, str]     # Tools NOT loaded (show as hints)
loaded_tools: Dict[str, LoadedToolInfo]  # Currently loaded tools
essential_tools: List[str]          # Can't auto-unload
current_turn: int                   # For idle detection
```

On each `TurnCompletedEvent`:
1. Check idle time for each loaded tool
2. Unload tools where `idle_turns > 5`
3. Update working memory with new hints
4. Publish trinket content event

Tools can publish trinket updates when their state changes (punchclock_tool does this for active work sessions).

### Tool Execution Flow

When Claude calls a tool:

1. Name validated in `ToolRepository.enabled_tools`
2. Fresh instance created with dependency injection (no caching between users)
3. User context automatically applied
4. `tool.run(**params)` executes
5. Database queries scoped to current user
6. Results returned to Claude
7. ToolLoaderTrinket notified if state changed

Tools are lazy-instantiated. No tool instances exist until called. This prevents user data leakage.

---

## Security & User Isolation

MIRA runs in single-user mode. One user per instance, isolated at the application layer.

### Single-User Enforcement

On startup, MIRA:
1. Checks for existing users in the database
2. Creates a default user if none exist
3. Exits with error if multiple users found
4. Sets global user context via `contextvars`

The user context is thread-safe and works for both CLI and async web scenarios. Every database query, tool call, and file operation automatically scopes to this user.

### User Context Architecture

```python
# From utils/user_context.py
set_current_user_id(user_id)  # Sets thread-local context
get_current_user_id()          # Gets context, raises if not set
```

Components use `get_current_user_id()` internally. Tool base class provides `self.user_id` property that calls this. Database clients receive `user_id` parameter and filter queries accordingly.

This is application-layer isolation, implemented via explicit filtering in queries. Stores user ID in thread-local context variables for transparent scoping.

### Credential Management

Two-tier architecture:

**System-level credentials** → HashiCorp Vault
- API keys for Anthropic, OpenAI, Google Maps, etc.
- Database URLs and connection strings
- Retrieved at runtime via `vault_client.get_api_key()`
- Startup fails if required credentials missing (fail-fast)

**Per-user credentials** → Encrypted SQLite
- Tool-specific credentials (email passwords, API tokens)
- Stored in user-specific encrypted database
- Accessed via `UserCredentialService`
- Located at `data/users/{user_id}/credentials.db`

Vault is initialized during deployment. Credentials saved to `.vault_keys` (gitignored). The deploy script handles Vault setup automatically.

---

## Performance Optimizations

### Connection Pooling

PostgreSQL uses `psycopg2.pool.ThreadedConnectionPool`:
- Min 2 connections, max 50 per database
- Thread-safe pool management with `RLock`
- Pools shared across all instances
- 30-second connection timeout

Valkey client uses singleton pattern - one instance reused across the application.

### Three-Tier Caching

**1. Embedding Cache** (Valkey, 15-minute TTL):
- Stores embeddings as fp16 to save memory
- Separate caches for AllMiniLM (384-dim) and OpenAI (1024-dim)
- Prevents dimension conflicts

**2. Message Cache** (Valkey, event-driven invalidation):
- Stores recent messages for hot continuums
- Invalidated on segment timeout events
- No fixed TTL, clears based on activity

**3. Database Connection Pool Cache**:
- Per-user DB client caching in repository
- Avoids recreating connections

### Hybrid Embeddings

Dual-model strategy for speed vs quality:
- **AllMiniLM (384-dim)**: Local, ~1ms latency, used for real-time operations
- **OpenAI text-embedding-3-small (1024-dim)**: Remote, higher quality, used for advanced features
- BGE reranker available for search refinement (configurable pool size)

Single embedding generated per turn, shared across all services (memory surfacing, context building, search).

### Model Routing

Simple tools can route to faster execution model:
- Reasoning tasks → Claude Sonnet 4.5
- Simple tools (weather, reminders, punchclock) → Groq gpt-oss-20b
- Configurable via `config.api.simple_tools` list

### Prompt Caching

Anthropic's prompt caching reduces costs ~90% on cached content. MIRA implements this intelligently:

- **System prompt base**: Always cached
- **Static trinket sections**: Cached (time-insensitive context)
- **Dynamic sections**: Not cached (reminders, recent context)
- **Tool definitions**: Last tool marked for caching (caches all tools)

The system prompt composer separates cached from non-cached sections. Cache metrics logged after each API call.

### Resource Pre-Initialization

Expensive resources load at startup:
- Hybrid embeddings provider (loads AllMiniLM + OpenAI embeddings)
- Continuum repository (creates DB connection pool)
- LT memory factory (initializes all memory services)

This avoids first-request latency. Everything's ready before the first message.

---

## Technology Stack

**Web Framework:**
- FastAPI (async REST API)
- Hypercorn (HTTP/2 ASGI server)
- Pydantic (data validation and config)

**Database:**
- PostgreSQL 14+ with pgvector extension
- psycopg2 with ThreadedConnectionPool
- SQLite for per-user tool data

**Caching:**
- Valkey (Redis-compatible)
- 15-minute TTL for embeddings
- Event-driven invalidation for messages

**AI/ML:**
- Anthropic Claude Sonnet 4.5 (reasoning)
- Anthropic Batch API (memory extraction)
- Groq gpt-oss-20b (fast execution for simple tools)
- OpenAI (emergency fallback + embeddings)
- Ollama (offline LLM support)

**Embeddings:**
- sentence-transformers (AllMiniLM-L6-v2, 384-dim, local)
- OpenAI text-embedding-3-small (1024-dim, cloud)
- BGE-M3 + BGE reranker (cross-encoder reranking)
- ONNX Runtime (optimized inference)
- PyTorch + Transformers (ML foundation)

**NLP:**
- spaCy 3.7+ (entity extraction, en_core_web_lg model)
- rapidfuzz (fuzzy string matching for deduplication)

**Security:**
- HashiCorp Vault (secrets management)
- UserCredentialService (encrypted per-user credentials)
- WebAuthn (passkey authentication - legacy from multi-user mode)
- PyJWT (token handling)

**Background Processing:**
- APScheduler (scheduled tasks for memory extraction, consolidation, refinement)
- asyncio + threading for concurrent operations

**Integrations:**
- Letta SDK (domain knowledge blocks via sleeptime agents)
- Playwright (headless browser for JavaScript-rendered pages)
- httpx (HTTP/2 client for API calls)
- TP-Link Kasa (smart home control)
- Google Maps API (geocoding)
- Kagi Search API (privacy-focused search)
- OpenMeteo API (weather forecasts)

---

## Getting Started

### Prerequisites

**Required:**
- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- Valkey or Redis
- HashiCorp Vault

**LLM Provider (choose one):**
- Anthropic API key (recommended for Claude Sonnet 4.5)
- OpenRouter API key (alternative cloud provider)
- Ollama (fully offline operation)

**Optional:**
- OpenAI API key (for deep embeddings; falls back to local AllMiniLM)
- Letta API credentials (for domain knowledge blocks; system disables if unavailable)
- Kagi API key (for enhanced web search)

### Installation

Run the deployment script:

```bash
./deploy.sh
```

The script handles:
1. Platform detection (macOS/Linux)
2. Dependency checking and optional auto-installation
3. Python venv creation
4. pip install of requirements
5. Model downloads (spaCy ~800MB, AllMiniLM ~80MB, BGE reranker ~1.1GB, optional Playwright ~300MB)
6. HashiCorp Vault initialization
7. Database deployment (creates mira_service database)
8. Service verification (PostgreSQL, Valkey, Vault running and accessible)

After deployment:
```bash
source .vault_keys  # Load Vault credentials
python main.py      # Start MIRA
```

First startup creates the default user automatically.

### Configuration

Configuration uses Pydantic BaseModel schemas in `config/config.py`:

- **ApiConfig**: Model selection, token limits, simple tools list
- **ApiServerConfig**: Host, port, CORS settings
- **EmbeddingsConfig**: Embedding model selection, cache settings
- **LTMemoryConfig**: Extraction intervals, consolidation thresholds
- **ToolConfigs**: Per-tool configuration (idle thresholds, cache durations)

Access via global config instance:
```python
from config.config_manager import config

config.api.model  # "claude-sonnet-4-5-20250929"
config.tools.invokeother_tool.idle_threshold  # 5
```

System prompt loaded from `config/system_prompt.txt` at startup. Missing file causes startup failure (fail-fast).

---

## Development

### Running Tests

```bash
pytest                                    # All tests
pytest tests/test_file.py                # Specific file
pytest tests/test_file.py::test_function # Specific test
pytest --cov=. --cov-report=html         # With coverage
```

### Code Quality

```bash
black .        # Format (88 char line length)
mypy .         # Type checking
flake8         # Linting
```

### Development Workflow

1. Create feature branch
2. Write tests (follow patterns in `tests/fixtures/`)
3. Implement feature
4. Run tests and quality checks
5. Commit with semantic prefix (feat:, fix:, refactor:, etc.)
6. Submit pull request

See `CLAUDE.md` for comprehensive engineering principles, code style guidelines, and architectural patterns.

---

## Contributing

Read `CLAUDE.md` for:
- Technical integrity and evidence-based decisions
- Security and reliability standards (fail-fast infrastructure, credential management)
- Core engineering practices (minimize dependencies, fix upstream, simple solutions first)
- Architecture and design patterns (Domain-Driven Design, event sourcing, immutable value objects)
- Performance optimization guidelines

**Code Style:**
- PEP 8 with Black formatting (88 char lines)
- Type hints required for all function signatures
- Google-style docstrings for public methods
- Group imports: stdlib, third-party, local (sorted alphabetically)
- Document all raised exceptions

**Adding New Tools:**
- Inherit from `BaseTool`
- Use `self.db` for user-scoped database access
- Use `self.user_data_path` for file storage
- Register Pydantic config via `registry.register()`
- Add comprehensive tests

See `tools/HOW_TO_BUILD_A_TOOL.md` for detailed guide.

---

## Documentation

**System Overviews** (in `docs/SystemsOverview/`):
- CNS System: Event-driven conversation orchestration
- Event Bus System: Publish/subscribe coordination
- LT_Memory System: Long-term memory lifecycle
- Working Memory System: Trinket-based prompt composition
- Tool System: Tool framework and patterns
- Domain Knowledge System: Letta-powered context blocks
- Segment System: Conversation segmentation

**Architecture Decision Records** (in `docs/ADRs/`):
- Model Routing: Fast vs reasoning models
- InvokeOther Tool Pattern: Dynamic tool loading
- OSS Single User Mode: Single-user deployment
- Bucket-Based Conversations: Organization strategy

**Design Patterns** (in `docs/SystemsOverview/patterns/`):
- Heal on Read Pattern: Self-correcting data structures
- Drag and Drop Tools: Frontend tool integration
- Event-Aware Trinkets: Event-driven context providers

---

## License & Credits

### A Note on Claude Sonnet 4.5

Claude Sonnet 4.5 is closed source but has a gestalt quality that enables incredibly accurate mimicry. Claude models have always been good but Sonnet 4.5 has a unique sense-of-self I've never encountered in all my time working with large language models. I don't know what Anthropic has done but it speaks to you with such realism and its creative tool use combos are remarkable. MIRA will work fine with gpt-oss-120b or even pretty well with hermes3-8b but they'll never have that OEM feel that Sonnet MIRA has. I felt it was important to include other provider support but I have to say it. Dear reader, if you're feeling really zesty you should shrink the context window to a 20,000 token buffer and use claude-opus-4-20250514 for a bit (remember its $15/75 per million tokens). The sense that you're talking to something that talks back to you (we know not what that means vs. human experience) is uncanny.

Check out the system_prompt.txt for MIRA in the config/ folder. Its content and brevity stand out from many system prompts in other LLM-based projects.

Also, I'd like to give an earnest shoutout to the Claude Code team. God honest truth I'm really good at reviewing Python and articulating my design decisions but I don't know a lick about writing Python even after all this time. Thank you to Boris from the Claude Code team for being a wellknown developer and STILL replying to messages on Threads like a good developer should. I'm sure he surrounds himself with a high-quality team of folks too and thank you to them as well. I built this entire application from the CC terminal (except the CSS because LLMs by their nature struggle with inheritance properly. thats not their fault.). I got access to CC on the second day that it was available and dove in. As of writing this readme that was 8 months ago. I built MIRA and wrote tests that pass as a one-person development team. Call me the Yung David Hahn but I just slaved away at my vision for a result and with enough tenacity I actualized it. Thanks Claude Code team. You're a real one for this.

### Dependencies

MIRA builds on excellent open-source projects:
- **FastAPI**: Modern async web framework
- **PostgreSQL**: Robust relational database with pgvector
- **Valkey**: High-performance Redis-compatible cache
- **HashiCorp Vault**: Secrets management
- **Letta**: Stateful AI agents for domain knowledge
- **Playwright**: Headless browser automation
- **spaCy**: Industrial-strength NLP

See `requirements.txt` for complete list.

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/taylorsatula/mira/issues)
- **Discussions**: [GitHub Discussions](https://github.com/taylorsatula/mira/discussions)
- **Documentation**: [docs/](docs/)

---

**Built with ❤️ by Taylor Satula**
