# agents/ — Autonomous sidebar agents

## Rules

Agents extend `SidebarAgent` (in `base.py`). The base class owns the LLM-in-a-loop mechanics: LLM init, tool schema assembly (always includes `sidebar_tool`), input sanitization (when `sanitize_untrusted_input=True`), message loop with heartbeat, tool execution, `complete_task` detection, trace capture, and completion publishing via `on_completion()`.

Implementations define: `agent_id`, `internal_llm_key`, `available_tools`, `get_agent_prompt(work_item)`, `build_initial_message()`. Override `_get_completion_trinket()` and `_build_completion_context()` to publish to a different trinket (see `ForageAgent`). Default agent prompts live in `config/prompts/agents/` as `.txt` files, loaded via `load_agent_prompt()` from `base.py`. Per-rule prompts override the default when `work_item.context['agent_prompt']` is set — the trigger populates this from the matched rule's `prompt` column. If multiple rules with different prompts match the same item, the trigger writes a `conflict` record to `sidebar_activity` and skips the item.

### Sentry Gate (Opt-In)

Set `sentry_llm_key` to an `internal_llm` name to activate a cheap one-shot LLM call before the main loop. Override `build_sentry_message(work_item)` to provide the evaluation prompt. The sentry response must contain `<decision>proceed|skip</decision>` and `<reason>...</reason>` XML tags (override `parse_sentry_response()` for custom formats). If the sentry says skip, the agent exits with `status='dismissed'` — no main loop tokens burned. Fails open on errors (LLM failure, parse failure → proceed). Use for periodic/speculative agents where most evaluations result in "nothing to do."

The base class injects `thread_id` into ALL `sidebar_tool` calls (scratchpad and completion) — thread identity is a system concern, never passed by the LLM. For `complete_task`, it also injects `interface_name`, `agent_id`, and `run_count`. Loop terminates when the LLM calls `complete_task`, which writes to `sidebar_activity` SQLite via UPSERT and exits.

Agents are spawned in background threads with `contextvars.copy_context()`. Two spawn paths:
1. **SidebarDispatcher** — polls registered `SidebarTrigger` instances on an APScheduler interval, spawns agents for new `WorkItem`s. Owns dedup via `sidebar_activity` SQLite + in-flight tracking. Triggers return all discovered items; the dispatcher decides what to act on.
2. **Direct invocation** — a tool (e.g. `ForageTool`) creates a `WorkItem` and calls `agent.run()` in a background thread.

### Batch Mode (Opt-In)

Set `use_batch = True` on a `SidebarAgent` subclass to route LLM calls through the Anthropic Batch API (50% cost reduction). Each `batch_generate_response()` call submits a single-request batch, polls until completion, and returns the same neutral `Result` type as the sync path.

**Class attributes**: `use_batch: bool`, `batch_timeout_seconds: int` (default 3600). **Timeout coupling**: subclasses must also set `timeout_seconds >= batch_timeout_seconds * max_iterations`, since the per-iteration timeout gate still applies.

**Concurrency**: Batch agents are tracked in `_active_batch_agents` (separate from `_active_agents`) with their own cap (`max_concurrent_batch_agents` in `SidebarDispatcherConfig`). Batch agents don't consume sync agent slots.

**Restart behavior**: If the app restarts mid-poll, the in-flight batch work is orphaned. The trigger rediscovers the item on the next poll cycle and a new run starts fresh. Acceptable for non-urgent background work.

**Transport module**: `agents/batch.py` — lazy Anthropic client using `anthropic_batch_key` from Vault, builds an Anthropic batch request from the pre-resolved `InternalLLMConfig`, and normalizes the batch message to `Result`.

### Overwatch (Opt-In)

Set `overwatch_llm_key` to an `internal_llm` name to activate a passive iteration observer. After each non-terminal iteration, the base class spawns a daemon thread that calls a cheap model (e.g. Qwen3-32B via Groq) with a compact summary of the iteration's tool calls and results. The agent loop is unaware of the observer — zero latency impact.

**Class attributes**: `overwatch_llm_key: str | None`, `overwatch_max_tokens: int` (default 80). **Hooks**: override `get_overwatch_context(work_item)` for task-specific context (default: interface name), override `on_overwatch_update(event_bus, work_item, iteration, summary)` to publish the one-sentence summary to the appropriate trinket.

The overwatch thread uses `contextvars.copy_context()` so user context propagates correctly. Late arrivals (overwatch finishes after agent completes) are handled at the trinket layer — terminal states are never downgraded to `in_progress`.

**ForageAgent** is the first consumer: sets `overwatch_llm_key = "overwatch"`, publishes stacked per-iteration summaries to `ForageTrinket` so the primary LLM sees the full research arc.

### Dedup & Retry

Dedup is the dispatcher's responsibility, not the trigger's. The dispatcher checks `sidebar_activity` for prior runs and uses `_dispatch_decision()` to determine: first dispatch, retry (if `max_retries > 0` and `run_count` allows), or skip.

For retries, the dispatcher sets `work_item.context['prior_run']` with the previous activity record. The base class calls `build_recovery_context(prior_run)` and prepends the result to the initial message. Override `build_recovery_context()` (default returns `None`) to inject failure-aware context.

Old `sidebar_activity` records and their scratchpad notes are cleaned up after 30 days (`_CLEANUP_RETENTION_DAYS`), rate-limited to once per 24h per user.

## Files

- `batch.py` — Batch transport for sidebar agents. `batch_generate_response()` submits a single LLM call to the Anthropic Batch API, polls for completion, returns `clients.llm.types.Result`. Lazy thread-safe Anthropic client using `anthropic_batch_key` from Vault; accepts pre-resolved `InternalLLMConfig` directly (no re-resolution). Deep-copies last tool schema before adding `cache_control`.
- `base.py` — `SidebarAgent` ABC. Shared loop mechanics, sentry gate (opt-in cheap pre-filter via `sentry_llm_key`), overwatch (opt-in passive iteration observer via `overwatch_llm_key`), `sanitize_untrusted_input` flag for pre-loop injection defense, `ACTIVITY_TABLE_DDL` (with `run_count` column), `ensure_activity_schema()` (creates table + runs migrations), trace TypedDicts (`ToolCallTrace`, `IterationTrace`, `AgentTrace`). The system prompt is rebuilt every iteration; override `_iteration_status(iteration)` to append a per-turn addendum (e.g. progress bar). Exports `load_agent_prompt()` for loading prompts from `config/prompts/agents/`.
- `sidebar.py` — `WorkItem` model, `SidebarTrigger` protocol (`on_dispatched` for side effects, not dedup), `SidebarDispatcher` (APScheduler poll loop with SQLite-backed dedup, in-flight tracking, retry support, cleanup).
- `implementations/forage_agent.py` — `ForageAgent`: background research with 20-iteration cap. Uses `_get_completion_trinket()` / `_build_completion_context()` overrides to publish to `ForageTrinket`. Uses overwatch (`overwatch_llm_key = "overwatch"`) for per-iteration progress summaries. Overrides `_iteration_status()` to inject a Unicode block-character progress bar into the system prompt each turn. Tools: `continuum_tool`, `memory_tool`, `web_tool`.
- `implementations/whilethecatsaway_agent.py` — `WhileTheCatsAwayAgent`: curiosity-driven background research in batch mode (50% cost). High iteration cap (25). Uses `_get_completion_trinket()` / `_build_completion_context()` overrides to publish to `WhileTheCatsAwayTrinket`. Tools: `web_tool`, `memory_tool`, `continuum_tool`.
- `implementations/memory_curator_agent.py` — `MemoryCuratorAgent`: agentic memory-graph curation in two modes (integration + floor) selected per work-item via `work_item.context['mode']`. Integration: spawned at memory storage (forage-style background thread) to link/merge/stand-alone new memories. Floor: dispatcher-driven via `MemoryFloorTrigger`, triages a random sample of low-value unseen memories (archive/salvage). Sole link typer — `tool_schema_overrides` excludes `create_memory` from `memory_tool` (anti-silt). Stamps `last_tended_at` on successful runs. Publishes to `MemoryCuratorTrinket`. Sync, no sentry, no overwatch. Context TypedDicts (`CandidateRef`, `NewMemory`, `IntegrationContext`, `FloorMemory`, `FloorContext`) define the trigger→agent contract.
- `triggers/memory_floor_trigger.py` — `MemoryFloorTrigger`: dispatcher-driven floor-mode curation trigger. Pure deterministic SQL discovery (no LLM) — random sample of low-importance memories not tended in `floor_unseen_days` via `LTMemoryDB.get_floor_candidates()`. Use-day cadence gate (inline `activity_days % floor_use_days`). Stable `item_id` per activity-day cycle for dispatcher dedup. `agent_class` lazy-resolves `MemoryCuratorAgent` to avoid import cycle.
