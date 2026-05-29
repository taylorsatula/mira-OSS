# cns/core/ — Immutable domain model: no DB, no HTTP, no external dependencies

## Rules

No database access, no HTTP calls, no infrastructure imports in this directory — except `segment_cache_loader.py`, which is the one permitted infrastructure boundary crosser (it takes a `ContinuumRepository` as an injected dependency, not a direct import of infrastructure singletons). `message_formatter.py` has the existing deferred service import for collapsed segment display formatting only.

`Message` and `ContinuumState` are `frozen=True` dataclasses. Mutations produce new objects via `with_metadata()` or `from_dict()`. The `Continuum` aggregate itself is mutable (its `_message_cache` list is appended to), but its embedded `_state` is frozen.

Domain events: subclass one of the four base categories (`MessageEvent`, `ToolEvent`, `WorkingMemoryEvent`, `ContinuumCheckpointEvent`), use `frozen=True, kw_only=True`, add a `.create()` classmethod that generates `event_id`/`occurred_at` and sources `user_id` from `get_current_user_id()`. Don't create new categories — adapt existing ones.

Stream events: subclass `StreamEvent`, set a unique `type` string as a non-init field default. These are mutable (no `frozen=True`) for streaming performance.

`ThinkingBlock` dicts must be passed to the API unmodified — the `signature` field is cryptographic and the API rejects tampered blocks.

`preprocess_content_blocks()` in `message.py` is the single shared path for stripping media and truncating tool results across extraction, summarization, and peanut gallery. Don't duplicate that logic; call this function. Consumers that need custom block loops (e.g., `assessment_extractor`) import `_MEDIA_BLOCK_TYPES` directly from `cns.core.message`. `TOOL_RESULT_TRUNCATION_LIMIT` has moved to `cns.services.live_context_compaction_service`.

## Files

- `continuum.py` — Continuum aggregate root. Owns message cache mutation (`apply_cache()`, `add_*_message()`, `add_tool_history()`). `get_messages_for_api()` is a wrapper over `message_formatter.format_messages_for_api(self.messages)`.
- `message.py` — `Message` frozen dataclass (value object), `MessageMetadata` TypedDict (canonical list of all known metadata keys), `ContentBlock` union, `ThinkingBlock` TypedDict, and `preprocess_content_blocks()` shared preprocessing function.
- `message_formatter.py` — Shared provider-neutral formatter for `Message` sequences. Handles tool result messages, collapsed segment synthetic `continuum_tool` framing, timestamp injection, assistant string normalization, and adapter metadata fields.
- `state.py` — `ContinuumState` frozen dataclass and `ContinuumStateDict` TypedDict for serialization round-trips. Intentionally minimal — real state lives in the message cache.
- `events.py` — Domain event hierarchy. Four abstract base categories plus concrete event subclasses. `TurnCompletedEvent.continuum` is typed via `TYPE_CHECKING` guard to break the circular import with `continuum.py`.
- `segment_cache_loader.py` — Reconstructs message history on Valkey cache miss. Two-tier segment selection: Tier 1 loads extended summaries by accumulated `complexity_score` (up to 4.5), Tier 2 loads up to 4 additional segments using only their 2-sentence precis for broader lookback. Segments are marked with ephemeral `display_mode` ('extended'/'precis') for display routing. Loads behavioral primer dialogue from `config/prompts/behavioral_primer.txt` at init and injects between summaries and continuity turns when collapsed segments exist.

## Wiring

`message_formatter.format_messages_for_api()` imports `format_segment_for_display` from `cns.services.segment_helpers` at call time (deferred import to avoid circular dependency). Any change to collapsed segment display format must be coordinated between those two files.

`SegmentCacheLoader` is the only file in this directory that imports from `cns.infrastructure`. It receives `ContinuumRepository` via constructor injection — never create it with a direct infrastructure import.
