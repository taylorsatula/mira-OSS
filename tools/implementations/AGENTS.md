# tools/implementations/ — Concrete Tool subclasses

## Rules

Every tool follows this exact four-part structure:
1. Define `XxxToolConfig(BaseModel)` with `enabled: bool = Field(default=True, ...)`
2. `registry.register("xxx_tool", XxxToolConfig)` at module level — discovery depends on this
3. Subclass `Tool` (from `tools.repo`) with `name`, `tool_schema`, `description`, `simple_description` class attributes
4. Implement `run(**params)` routing on `params.pop("operation")`

Per-user credentials via `UserCredentialService().get_credential(type, service_name)`. Raise on missing — never fall back to a default or env var.

User-scoped file storage: `data/users/{user_id}/artifacts/{file_id}/{uuid4_hex}.bin` + `.meta` sidecar (`filename`, `mime_type`). Served by `/files/{file_id}` (download) or `/images/{file_id}` (inline). Persistent across segments — no segment-collapse cleanup.

Tools returning images use two special result-dict keys: `_content_blocks` (provider-neutral rich content blocks passed through the LLM lifecycle) and `_image_artifact` (`{file_id, alt_text}` — orchestrator emits `![alt](/v0/api/images/{file_id})` to stream).

Every `tool_schema` parameter description is an LLM caller contract. Use exact token names, state co-dependencies inline, name the failure mode. No hedging, no internal jargon.

## Files

- `contacts_tool.py` — Contact CRUD, search, and group management; owns `contacts` SQLite schema
- `continuum_tool.py` — Conversation and segment operations (search, navigation) against the CNS continuum
- `domaindoc_tool.py` — Domain knowledge document management; always-active essential tool; `tool_schema` is a `@property` that builds a live catalog from SQLite at schema-read time; `request_create`/`request_delete` return UI guidance only (noops). Supports shared domaindocs via `utils.domaindoc_shares` — collaborators can edit sections, version history records actor info
- `email_tool.py` — Email send/read via user-configured provider credential; `reasoning` param required on all mutating ops (send, reply, delete, move, mark, draft), logged to `email_action_audit` SQLite table with `encrypted__` fields for Stage 2 autonomy analysis
- `homeassistant_tool.py` — Home Assistant smart home control via REST API; local SQLite entity registry for fuzzy name resolution; toggle states and query device data
- `forage_tool.py` — Speculative background context gathering; fire-and-forget trigger that dispatches `ForageAgent` via `agents/implementations/forage_agent.py`; two params (`query` + `context`) to launch, `dismiss_task_id` to clear results; publishes to `ForageTrinket`
- `whilethecatsaway_tool.py` — Curiosity-driven background research; fire-and-forget dispatch of `WhileTheCatsAwayAgent` in batch mode; two params (`topic` + `context`); findings stored as LT_Memory, summary published to `WhileTheCatsAwayTrinket`
- `imagegen_tool.py` — Image generation/refinement via Google Gemini with Chat-based lineage; `generate` opens a new session, `refine` replays full history, `publish` emits `_image_artifact`; per-user API key via `UserCredentialService('google_genai')`; images uploaded to Anthropic Files API uncompressed
- `inbox_tool.py` — Local file drop-off dropbox. User drops files into a configured filesystem path (default `/tmp/mira-dropbox`); tool exposes `list` / `read` (paginated text extraction for TXT/CSV/JSON/DOCX/XLSX) / `archive` (moves to `dropbox/archive/` with a `.meta.json` sidecar). Does not write to domain docs or memory itself — model composes with `domaindoc_tool` or `memory_tool` between `read` and `archive`. PDFs and images are archive-only. Disabled by default (`InboxToolConfig.enabled=False`); not in `ESSENTIAL_TOOLS` — opt in via config to load
- `invokeother_tool.py` — Meta-tool for on-demand loading of non-essential tools at runtime
- `maps_tool.py` — Google Maps geocoding, places, and distance; lazy client init
- `memory_tool.py` — LT_Memory search, create, link, annotate, touch, archive, and merge; `create_memory` queues to Valkey for deferred processing at segment collapse (no spaCy at init time). `archive` soft-deletes a memory; `merge_memories` consolidates 2+ memories via `ConsolidationHandler.execute_consolidation` (sources archived, links transferred, new memory stamped `last_tended_at`). `link_memories` carries an optional 3-word `bond` (the extraction bond from candidate hints), persisted on the link as `extraction_bond` and surfaced when a memory is referenced via `proactive_memory_trinket`. `CURATOR_MEMORY_SCHEMA` module constant excludes `create_memory` for the `MemoryCuratorAgent` (anti-silt).
- `pager_tool.py` — Lattice federation messaging
- `phoneafriend_tool.py` — Synchronous outside-model consultation; tool choices `claude`/`gemini` map to dedicated `internal_llm` keys `phoneafriend_claude`/`phoneafriend_gemini`; disables provider stall fallback so the requested outside voice cannot silently swap to `claude-high`; stores user-bound, segment-scoped subagent transcripts in Valkey under `phoneafriend:{user_id}:{segment_id}:{thread_id}` with TTL cleanup
- `punchclock_tool.py` — Time tracking
- `reminder_tool.py` — Reminder scheduling and management
- `weather_tool.py` — Weather data retrieval
- `web_tool.py` — Web search (Kagi primary, DuckDuckGo fallback), page content extraction (trafilatura; HTTP first, Playwright escalation for JS-heavy pages), and HTTP requests
- `sidebar_tool.py` — Sidebar agent lifecycle: scratchpad (write/read/clear notes) and `complete_task` (writes to `sidebar_activity` SQLite). Disabled (`enabled: False`) — sidebar agents only, not main conversation
- `sidebaragents_tool.py` — Main conversation tool for managing sidebar activity: `list_activity`, `get_details`, `dismiss`, `resolve`. Reads from same `sidebar_activity`/`scratchpad` tables that `sidebar_tool` writes. Takes `WorkingMemory` for event_bus access
- `schemas/contacts_tool.sql` — DDL for the contacts SQLite schema owned by `contacts_tool.py`
