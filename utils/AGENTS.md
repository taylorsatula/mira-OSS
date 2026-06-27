# utils/ — Cross-cutting infrastructure and shared utilities

## Rules

- `user_context.py` is the single source of truth for user identity, tier resolution, internal LLM config, and user preferences — do not duplicate any of these in other modules.
- `get_internal_llm(name)` requires `load_internal_llm_configs()` called at startup; calling it before that raises `RuntimeError`. Every internal LLM call site must pass a `name` that exists in the `internal_llm` DB table.
- `get_user_preferences()` has a hardcoded SELECT and explicit field mapping. Adding a field to `UserPreferences` is not enough — update both the SELECT and the `UserPreferences(...)` construction or the new field silently returns its default on cache miss.
- `database_session_manager.py` is a module-level singleton (`get_shared_session_manager()`); never instantiate `LTMemorySessionManager` directly. Its `execute_*` methods are monkey-patched by `perf.py` at startup — do not add timing or logging to the methods themselves.
- All perf instrumentation stays in `perf.py`. If you add a new class with `execute_*` methods, register it in `install_db_instrumentation()`.
- Use-day-gated jobs: all scheduled work that should fire on user engagement intervals calls `get_users_due_for_job(interval)` from `scheduled_tasks.py`, not a custom tracking table. Calendar jobs use `IntervalTrigger` directly without this filter.

## Files

- `user_context.py` — Owns user identity contextvar (`set_current_user_id` / `get_current_user_id`), segment context, `UserPreferences` model + `get_user_preferences()`, `ConversationLLMConfig` / `get_conversation_llms()` with `adapter_name`, and `InternalLLMConfig` / `get_internal_llm()`.
- `database_session_manager.py` — Singleton PostgreSQL connection pool (`get_shared_session_manager()`); `LTMemorySession` enforces RLS, `AdminSession` bypasses it.
- `perf.py` — Performance instrumentation; monkey-patches `execute_*` methods at startup via `install_db_instrumentation()`. Gated by `mira.perf` logger level.
- `logging_config.py` — TOAST custom log level (60), `UserContextFilter` (injects user_id into all log records), `ColoredFormatter`, `setup_anthropic_sdk_logging()`. `instrument_anthropic_client()` attaches httpx hooks for SDK log correlation and traffic tap.
- `llm_tap.py` — Debugging firehose for sniffing all LLM request/response traffic. Toggle: `kill -USR1 <pid>`. Output: `llm_requests.jsonl`, `llm_responses.jsonl` (project root, append mode). Request capture via httpx hooks on Anthropic clients; provider adapters log normalized requests and `Result` responses.
- `scheduled_tasks.py` — `get_users_due_for_job(interval)` use-day platform function; `initialize_all_scheduled_tasks()` entry point; `register_sidebar_dispatcher_job()` called separately after tool_repo init.
- `lt_memory_jobs.py` — APScheduler job registration for all LT_Memory periodic work (extraction retry, batch polling, consolidation, score recalc, GC, cleanup).
- `sidebar_jobs.py` — APScheduler registration for the sidebar dispatcher poll loop. Creates `SidebarDispatcher` and registers it on a configurable interval.
- `scheduler_service.py` — APScheduler wrapper; `register_job()` is the only job registration path.
- `scheduled_task_monitor.py` — Tracks scheduled job execution history, durations, and failures.
- `thread_monitor.py` — `@monitored_operation` decorator and `MonitoredThreadPoolExecutor`; thresholds: 30s slow, 300s stuck.
- `timezone_utils.py` — UTC datetime utilities; `utc_now()` and `format_utc_iso()`.
- `profile_validation.py` — Shared profile display-name normalization and validation for signup, demo conversion, and profile updates.
- `url_safety.py` — Public HTTP URL and domain-filter validation for outbound tool requests; blocks private, metadata, and non-global IP targets including DNS-resolved addresses.
- `tool_config_store.py` — Per-user tool config storage helpers; splits Pydantic fields marked `json_schema_extra={"secret": True}` into separate encrypted credentials and redacts API responses.
- `distributed_lock.py` — Valkey-backed `DistributedLock` for cross-process coordination via atomic SET NX.
- `user_credentials.py` — `UserCredentialService` bridging tool credential storage to `UserDataManager`'s encrypted SQLite.
- `user_activity.py` — `increment_user_activity_day()` (use-day clock); first-message-of-day hook point.
- `userdata_manager.py` — Per-user encrypted SQLite (`UserDataManager`); one persistent connection per user, cached at module level. Owns `trigger_rules` table schema (sidebar trigger filter rules — trigger_id, scope, field, pattern, prompt, enabled).
- `domaindoc_shares.py` — Cross-user domaindoc sharing resolution. Single source of truth for resolving domaindocs across user boundaries: `resolve_domaindoc()` (label → db + doc, own or shared), `get_accepted_shares()` (all accepted shares for a user), `invalidate_domaindoc_cache()` (Valkey trinket cache). All PostgresClient usage includes `user_id` for RLS enforcement.
- `tag_parser.py` — Extracts structured tags from LLM responses; owns `format_memory_id()` / `parse_memory_id()` and the `mem_XXXXXXXX` short-ID format.
- `text_sanitizer.py` — Text cleaning and sanitization utilities.
- `document_processing.py` — Document parsing and text extraction.
- `image_compression.py` — Image resizing/compression for uploads.
- `artifact_store.py` — User-scoped `.bin` + `.meta` artifact persistence for generated/provider files. Validates artifact IDs, caps file size, and sanitizes display filenames before writing under `data/users/{user_id}/artifacts/`.
- `prompt_injection_defense.py` — Input validation against prompt injection.
- `http_client.py` — Shared HTTP client utilities.
- `mcp_client.py` — Model Context Protocol client integration.
- `playwright_service.py` — Lazy-start Playwright service for JS-heavy pages: Chromium launches on first `fetch_rendered_html()`, auto-shuts down after `IDLE_TIMEOUT_SECONDS` (10 min) of inactivity. Used as escalation path by web_tool when HTTP+trafilatura produces insufficient content. Browser subrequests are validated through `utils.url_safety.validate_public_http_url()` before route continuation. Simplified rendering: `goto()` with `networkidle`/`domcontentloaded` fallback. No accordion expansion or progressive scroll.
- `synthetic_toolexample_generator.py` — Generates synthetic tool usage examples for training data.

## Wiring

`user_context.py` → `database_session_manager.py`: session manager reads `get_current_user_id()` to enforce RLS on every `LTMemorySession` query.

`user_activity.py` → `scheduled_tasks.py`: `increment_user_activity_day()` advances `cumulative_activity_days`; `get_users_due_for_job(interval)` reads it via `MOD(cumulative_activity_days, interval) = 0` to gate use-day jobs.

`perf.py` → `database_session_manager.py`: `install_db_instrumentation()` monkey-patches `LTMemorySessionManager.execute_*` at startup. Order matters — call `install_db_instrumentation()` after the session manager singleton is initialized.
