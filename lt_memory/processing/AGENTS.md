# lt_memory/processing/ — Extraction pipeline

## Rules

- No direct `anthropic_client.beta.messages.batches.create()` calls outside `BatchCoordinator.submit_batch()` — it is the sole submission point for all batch types.
- All LLM params for batch paths come from `build_batch_params('purpose', ...)`. Never construct batch param dicts inline.
- All LLM params for immediate paths come from `internal_llm='purpose'` in `generate_response()`. Never pass model, endpoint, or API key explicitly.
- Batch-vs-immediate routing uses `lt_memory.llm_routing.uses_anthropic_batch_dialect('purpose')`. Non-Anthropic dialect selections trigger immediate mode.
- `store_and_tend_extraction()` in `execution_strategy.py` is the single source of truth for memory storage — called by both `ImmediateExecutionStrategy._process_and_store_memories()` and `ExtractionBatchResultHandler.process_result()`. It stores memories with embeddings, persists LLM-extracted entities, builds typeless candidate hints, and notifies the integration curator via `LTMemoryFactory.on_memories_stored`. Never duplicate this logic.
- No typed links are written by extraction/storage code. Extraction-time `related_memory_ids` + bonds flow to the `MemoryCuratorAgent` as `CandidateRef` hints (discovery_signal `"extraction"`), not as edges. Relationship typing is the agent's job.
- `MemoryProcessor` has no side effects — pure data transformation. All DB writes happen in callers.
- `ExtractionEngine` has no LLM calls — pure payload construction. LLM calls happen in strategies.

## Files

- `orchestrator.py` — Owns the segment extraction lifecycle: load messages from `ContinuumRepository`, build `ProcessingChunk`, select strategy, mark `memories_extracted=true`. Two entry points: `submit_segment_extraction()` (per-segment) and `extract_unprocessed_segments()` (6-hour safety-net sweep).
- `execution_strategy.py` — Owns the `ExecutionStrategy` ABC, `BatchExecutionStrategy`, `ImmediateExecutionStrategy`, the `create_execution_strategy()` factory, and the module-level `store_and_tend_extraction()` / `_persist_llm_entities()` / `_build_candidate_hints()` helpers. `execute_extraction()` returns `str` (batch ID or synthetic `bypass_<uuid>`) or raises `ValueError` — never `None`.
- `extraction_engine.py` — Owns `ExtractionPayload` construction: prompt loading, UUID shortening/mapping via `format_memory_id()`, memory context retrieval from `ProcessingChunk.memory_context_snapshot`, and message formatting via `preprocess_content_blocks()`. File-local types: `ExtractionMessage`, `ExtractionPayload`.
- `memory_processor.py` — Owns LLM response parsing: JSON repair fallback, short→full UUID remapping, field validation/sanitization, and fuzzy+vector duplicate detection. File-local types: `DuplicateCheckResult`, `RawMemoryDict`.
- `batch_coordinator.py` — Owns the Anthropic Batch API lifecycle: submission, polling, expiry, retry, and result dispatch via `BatchResultProcessor` ABC. `poll_extraction_batches()` is a convenience wrapper over generic `poll_batches()`.
- `consolidation_handler.py` — Owns memory merge execution: link bundle transfer (inbound, outbound, entity), outbound-link rewriting on source memories, and archival of old memories. Pure business logic — no routing decisions, no LLM calls. Called by `memory_tool.merge_memories` (agent-invoked).

## Wiring

**Strategy selection at init vs. per-call:**
`LTMemoryFactory` calls `create_execution_strategy()` once at startup, producing either `BatchExecutionStrategy` or `ImmediateExecutionStrategy` as `ExtractionOrchestrator.execution_strategy`. A separate `ImmediateExecutionStrategy` is always created as `ExtractionOrchestrator.immediate_strategy`. Per-call, `submit_segment_extraction()` overrides to `immediate_strategy` when `force_immediate=True` (manual segment collapse) or when the extraction endpoint is non-Anthropic.

**`store_and_tend_extraction()` call sites:**
- `ImmediateExecutionStrategy._process_and_store_memories()` — called inline after `generate_response()`, via the shared helper.
- `ExtractionBatchResultHandler.process_result()` in `lt_memory/batch_result_handlers.py` — called from `BatchCoordinator.poll_batches()` after Anthropic returns results, via the shared helper.

Both pass `segment_id` so the integration curator work-item can record the source segment. The helper's `on_memories_stored` callback (late-registered by `SegmentCollapseHandler`) spawns the `MemoryCuratorAgent` in integration mode; `lt_memory` never imports from `agents/`.

**`chunk.segment_id` pipeline:**
`BatchExecutionStrategy` stores `str(chunk.segment_id)` in `ExtractionBatch.chunk_metadata["segment_id"]`. `ExtractionBatchResultHandler.process_result()` reads it back and sets `memory.source_segment_id` before storage. Required for segment-scoped memory cleanup on session resume.
