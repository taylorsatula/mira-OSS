# cns/ — Conversation orchestration: Continuum aggregate, LLM execution, memory surfacing, segment lifecycle

## Rules

- No database access in `core/` or `services/` — all persistence goes through `infrastructure/` repositories.
- Singleton services follow the `_instance = None` + `get_*()` / `initialize_*()` pattern. `integration/factory.py` calls initializers; all other callers use getters. Never instantiate services directly outside the factory.
- All inter-component communication goes through `EventBus.publish(event)`. New events subclass one of the four base categories in `core/events.py` and use `.create()` — never construct event dicts ad hoc.
- LLM calls use `internal_llm='purpose'` on `generate_response()` — never hardcode model names. Purpose keys: `'summary'`, `'assessment'`, `'synthesis'`, `'critic'`, `'portrait'`, `'analysis'`.
- All LLM output parses `<mira:tag>` XML via regex with flexible whitespace. Use `utils.tag_parser.TagParser` for standard tags; follow the regex patterns in `subcortical.py` and `assessment_extractor.py` for custom tags.
- Critical path services (`orchestrator`, `summary_generator`, `segment_collapse_handler`) propagate failures. Non-critical services (`peanutgallery_service`, feedback loop) swallow exceptions and log.
- Thread spawns copy user context: `contextvars.copy_context().run(fn)`. See `peanutgallery_service.py` for the canonical pattern.

## Files

- `core/` — Immutable domain model: Continuum aggregate, Message value objects, domain events, stream events, segment cache reconstruction. No I/O.
- `services/` — Stateless service layer: orchestrator, subcortical processing, summary generation, segment collapse, user model pipeline, peanut gallery, manifest query.
- `infrastructure/` — Persistence and caching: PostgreSQL continuum repository, Valkey message cache, feedback storage.
- `api/` — FastAPI endpoints: chat (HTTP + WebSocket), actions, data queries, health, tool config, federation.
- `integration/` — Event bus (pub/sub) and factory (dependency wiring for the entire CNS graph).

## Wiring

Segment lifecycle has three states: `active` (timeout ticking), `paused` (timeout suspended), `collapsed` (processed). Users can pause sessions indefinitely; segments auto-resume to `active` when the user sends their next message (via `increment_segment_turn()`). Segment collapse chain (triggered by `SegmentTimeoutEvent` on active-only segments): `segment_timeout_service` publishes → `segment_collapse_handler` handles → `summary_generator` produces summary → embedding → sentinel collapse in `infrastructure/continuum_repository` → `lt_memory` extraction → portrait synthesis gate (every 10 activity days via `ScheduledJobsConfig.portrait_synthesis_use_days`).

Memory surfacing pipeline (inside `orchestrator.process_message()`): `subcortical.generate()` (query expansion + retention + entities + complexity) → pinned cap (`max_pinned_memories=15`) → fresh budget (`max(min_fresh, max_surfaced - pinned)`) → `memory_relevance_service` embedding search → merge → `UpdateTrinketEvent` to `ProactiveMemoryTrinket`. Total bounded by `max_surfaced_memories=20`.

`actions.py` calls `segment_collapse_handler.collapse_segment(event, force_immediate=True)` — the `force_immediate` flag skips batch scheduling so memories are available before the user's next turn.
