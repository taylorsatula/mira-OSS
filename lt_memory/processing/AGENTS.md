# lt_memory/processing/ — Extraction and post-processing pipeline

## Rules

- No direct `anthropic_client.beta.messages.batches.create()` calls outside `BatchCoordinator.submit_batch()` — it is the sole submission point for all batch types.
- All LLM params for batch paths come from `build_batch_params('purpose', ...)`. Never construct batch param dicts inline.
- All LLM params for immediate paths come from `internal_llm='purpose'` in `generate_response()`. Never pass model, endpoint, or API key explicitly.
- Batch-vs-immediate routing uses `lt_memory.llm_routing.uses_anthropic_batch_dialect('purpose')`. Non-Anthropic dialect selections trigger immediate mode.
- `_process_and_store_memories()` on `ExecutionStrategy` is the single path for response parsing → storage → entity persistence → extraction-time link creation. Both `batch_result_handlers.py` (in `lt_memory/`) and `ImmediateExecutionStrategy` call it. Never duplicate this logic.
- `MemoryProcessor` has no side effects — pure data transformation. All DB writes happen in callers.
- `ExtractionEngine` has no LLM calls — pure payload construction. LLM calls happen in strategies.

## Files

- `orchestrator.py` — Owns the segment extraction lifecycle: load messages from `ContinuumRepository`, build `ProcessingChunk`, select strategy, mark `memories_extracted=true`. Two entry points: `submit_segment_extraction()` (per-segment) and `extract_unprocessed_segments()` (6-hour safety-net sweep).
- `execution_strategy.py` — Owns the `ExecutionStrategy` ABC, `BatchExecutionStrategy`, `ImmediateExecutionStrategy`, and `create_execution_strategy()` factory. `execute_extraction()` returns `str` (batch ID or synthetic `bypass_<uuid>`) or raises `ValueError` — never `None`.
- `extraction_engine.py` — Owns `ExtractionPayload` construction: prompt loading, UUID shortening/mapping via `format_memory_id()`, memory context retrieval from `ProcessingChunk.memory_context_snapshot`, and message formatting via `preprocess_content_blocks()`. File-local types: `ExtractionMessage`, `ExtractionPayload`.
- `memory_processor.py` — Owns LLM response parsing: JSON repair fallback, short→full UUID remapping, field validation/sanitization, fuzzy+vector duplicate detection, and linking pair construction. File-local types: `DuplicateCheckResult`, `RawMemoryDict`.
- `batch_coordinator.py` — Owns the Anthropic Batch API lifecycle: submission, polling, expiry, retry, and result dispatch via `BatchResultProcessor` ABC. `poll_extraction_batches()` and `poll_post_processing_batches()` are convenience wrappers over generic `poll_batches()`.
- `post_processing_orchestrator.py` — Owns consolidation batch submission. Routes to `BatchCoordinator.submit_batch()` or `ConsolidationHandler` immediate-mode based on resolved dialect support. Uses `internal_llm='consolidation'`.
- `consolidation_handler.py` — Owns memory merge execution: link bundle transfer (inbound, outbound, entity), outbound-link rewriting on source memories, and archival of old memories. Pure business logic — no routing decisions.

## Wiring

**Strategy selection at init vs. per-call:**
`LTMemoryFactory` calls `create_execution_strategy()` once at startup, producing either `BatchExecutionStrategy` or `ImmediateExecutionStrategy` as `ExtractionOrchestrator.execution_strategy`. A separate `ImmediateExecutionStrategy` is always created as `ExtractionOrchestrator.immediate_strategy`. Per-call, `submit_segment_extraction()` overrides to `immediate_strategy` when `force_immediate=True` (manual segment collapse) or when the extraction endpoint is non-Anthropic.

**`_process_and_store_memories()` call sites:**
- `ImmediateExecutionStrategy.execute_extraction()` — calls inline after `generate_response()`
- `ExtractionBatchResultHandler.process_result()` in `lt_memory/batch_result_handlers.py` — called from `BatchCoordinator.poll_batches()` after Anthropic returns results

**`chunk.segment_id` pipeline:**
`BatchExecutionStrategy` stores `str(chunk.segment_id)` in `ExtractionBatch.chunk_metadata["segment_id"]`. `ExtractionBatchResultHandler.process_result()` reads it back and sets `memory.source_segment_id` before storage. Required for segment-scoped memory cleanup on session resume.
