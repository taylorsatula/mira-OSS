# config/ — Application configuration and system prompt template

## Rules

- `config.py` contains only **operational and infrastructure** settings — feature flags, infrastructure coordinates, scheduling cadences, deployment settings. Algorithm tuning constants live inline in their consumer modules as `UPPER_SNAKE_CASE` module-level constants.
- All settings are Pydantic `BaseModel` schemas with hardcoded defaults. No required fields — the system must boot with zero external input.
- Secrets (API keys, DB URLs) are never stored here. They are lazy Vault lookups via properties on `AppConfig` (e.g., `config.api_key` calls `get_api_key(self.api.api_key_name)`). No env-var fallbacks.
- The `config` singleton is module-level state created at import time: `config = initialize_config()` in `config_manager.py`. Import it as `from config import config`. Never instantiate `AppConfig` directly in application code.
- `__init__.py` imports `tools.registry` before `config_manager` — this import order is load-bearing for circular dependency avoidance. Do not reorder.
- `config.<tool_name>_tool` triggers `AppConfig.__getattr__` → `get_tool_config()` → `registry.get_or_create()`. This is the only correct way to access tool configs.
- `ScheduledJobsConfig` fields ending in `_use_days` are modular activity-day intervals, not calendar intervals. `floor_use_days=7` means "run when `MOD(cumulative_activity_days, 7) = 0`", not "run every 7 calendar days".
- `system_prompt.txt` is loaded once at startup by `AppConfig._load_system_prompt()`. Template variables `{first_name}`, `{user_context}`, and `{relative time since account creation}` are substituted by `working_memory/core.py`, not by config.

## Config Models (6 total)

- `ApiConfig` — LLM API: feature flags (`analysis_enabled`, `show_openai_compat_thinking`, `emergency_fallback_enabled`), infrastructure coordinates (vault key names, endpoints), generation defaults (model, max_tokens, temperature), the absolute live-compaction threshold (`compaction_trigger_tokens`), and timeouts.
- `ApiServerConfig` — Server deployment: host/port/workers, CORS, uvicorn log level, extended thinking toggle.
- `SystemConfig` — System-level: `log_level`, `timezone`, `peanutgallery_enabled`.
- `ScheduledJobsConfig` — Background job cadences: extraction retry, batch polling, consolidation, score recalc, entity GC, batch cleanup, portrait synthesis. All operational knobs.
- `SidebarDispatcherConfig` — Sidebar agent: `enabled`, poll interval, max concurrent agents.

## Where Algorithm Constants Live

Algorithm tuning constants were moved from config.py to their consumer modules:
- `lt_memory/proactive.py` — surfacing thresholds, link weights, debut boost, context window caps
- `lt_memory/hybrid_search.py` — intent weights, RRF k, search defaults
- `lt_memory/linking.py` — link discovery thresholds, TF-IDF settings
- `lt_memory/processing/batch_coordinator.py` — batch processing limits
- `lt_memory/processing/memory_processor.py` — dedup thresholds
- `cns/services/orchestrator.py` — context overflow, topic drift, tool result limits
- `cns/services/peanutgallery_service.py` — trigger interval, seed count, TTL
- `cns/services/peanutgallery_model.py` — prerunner tokens, message window
- `cns/core/segment_cache_loader.py` — session cache tier settings
- `cns/services/manifest_query_service.py` — manifest depth, cache TTL
- `cns/services/segment_timeout_service.py` — segment timeout minutes

## Files

- `config.py` — Pydantic schema definitions for the 6 config models. No logic, no side effects.
- `config_manager.py` — `AppConfig` (root aggregate), `initialize_config()`, and the `config` singleton. Owns Vault property lookups and dynamic tool config via `__getattr__`.
- `__init__.py` — Re-exports `config` and `AppConfig`; enforces registry-before-config import order.
- `system_prompt.txt` — Mira's core identity prompt. Section order is semantics: foundational identity appears before behavioral directives because earlier tokens condition interpretation of later ones.
- `announcement.py` — Module-level cache for `announcement.json`. `load_announcement()` called once at startup lifespan; `get_cached_announcement()` used by `cns/api/data.py`.
- `announcement.json` — Active announcement state. Set `id` + `message` to show banner; set both to `null` to suppress. Requires app restart to take effect.
- `announcement.sample.json` — Reference format for `announcement.json`.
- `vault.hcl` — Local dev Vault server config (file storage, `127.0.0.1:8200`, no TLS). Consumed by the Vault binary, not Python.
- `prompts/` — LLM prompt templates for all subsystems. See `prompts/AGENTS.md`.
