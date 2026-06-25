# cns/infrastructure/ — CNS persistence and caching layer

## Rules

All DB access here uses either `PostgresClient` (user-scoped, RLS-enforced) or `get_shared_session_manager()` (shared pool). Never create a new `PostgresClient` per call — `ContinuumRepository` caches per-user clients in `_db_cache`. Cross-user admin queries use `AdminSession` to bypass RLS.

`ContinuumRepository` and `ContinuumPool` are singletons accessed via `get_continuum_repository()` and `get_continuum_pool()`. Never instantiate them directly outside `integration/factory.py`.

Valkey cache is event-driven — no TTL. A cache miss in `ContinuumPool.get_or_create()` means new session, not stale data. Do not add TTL-based expiry.

`UnitOfWork` is the only path for persisting messages during a turn. Call `uow.add_messages()` during processing, `uow.commit()` once at the end. Bypass is not acceptable — direct `save_messages_batch()` calls skip Valkey sync.

`FeedbackTracker` synthesis timing uses `users.cumulative_activity_days % threshold == 0` with a dedup guard against `activity_days_at_last_synthesis`. Never add a separate counter — it's stateless by design.

## Files

- `continuum_repository.py` — Owns PostgreSQL persistence for continuums, messages, and segments. Exports `HistoryResult`, `FailedSegment`, `ActiveSegmentRow` TypedDicts. Segment lifecycle: `find_active_segment()` matches both `active` and `paused` status; `pause_segment()` / `unpause_segment()` toggle pause state; `increment_segment_turn()` auto-resumes paused segments on user message. `load_messages_for_live_compaction_range()` is the DB-backed source for active-context compaction and excludes segment boundaries, system notifications, and compaction synopsis messages. `load_tool_result_by_id()` returns raw persisted tool-result text only when user, active session, and retrieval ID all match. Collapsed segments are final — no reactivation path.
- `continuum_pool.py` — Owns the Valkey-backed session cache and `UnitOfWork`. `UnitOfWork.add_messages()` enforces a per-message character limit (config: `context.message_max_chars`, default 150k) as a safety net against oversized content. `get_or_create()` is the entry point for all continuum session resolution.
- `valkey_message_cache.py` — Owns `Message` ↔ Valkey JSON serialization and compare-and-set tool-result compaction patches. Key format: `continuum:{user_id}:messages`. No constructor DI — calls `get_valkey_client()` directly.
- `feedback_repository.py` — Owns CRUD for `feedback_signals` table. Exports `FeedbackSignalRow`. Uses `get_shared_session_manager()`, not `PostgresClient`.
- `feedback_tracker.py` — Owns synthesis lifecycle state in `feedback_synthesis_tracking` table. Exports `LoraContent`, `TrackingStatus`. `get_and_clear_checkin_response()` atomically nulls the stored response via `UPDATE ... RETURNING`.

## Wiring

`UnitOfWork.commit()` calls `repository.save_messages_batch()` then `valkey_cache.set_continuum()` in that order — DB write always precedes cache write. A crash between the two leaves DB authoritative; next cache miss reconstructs via `SegmentCacheLoader` (defined in `cns/core/segment_cache_loader.py`).

Post-commit callbacks run only after database, cache, and metadata persistence succeed. Tool-result compaction uses this hook so the exact original is durable before its hot-cache copy is shortened.

`FeedbackRepository` takes `AssessmentSignal` from `cns/services/assessment_extractor.py`. `FeedbackTracker.get_lora_content()` returns `LoraContent` consumed by `LoraTrinket` in working memory. These two files are the boundary between the assessment pipeline and the user model pipeline.
