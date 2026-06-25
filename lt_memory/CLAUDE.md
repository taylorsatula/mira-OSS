# lt_memory/ — Long-term memory: extraction, linking, consolidation, and retrieval

## Rules

- `LTMemoryFactory` is the sole constructor for all services here — never instantiate `LTMemoryDB`, `VectorOps`, `LinkingService`, etc. directly outside tests. Access the singleton via `get_lt_memory_factory()`. The factory no longer receives a config object — all algorithm tuning constants are module-level `UPPER_SNAKE_CASE` values in each service file.
- `BatchCoordinator.submit_batch()` is the only path for submitting batches to Anthropic. No direct `anthropic_client.beta.messages.batches.create()` calls elsewhere in this directory.
- Callers store `input_data` in their own batch records (`create_extraction_batch()` or `create_post_processing_batch()`) before calling `submit_batch()` — `submit_batch()` does not accept `input_data`.
- `BatchKind` (`"extraction"` | `"post_processing"`) discriminates all generic batch DB methods. Use it; don't duplicate per-kind SQL.
- All shared types cross-used across files live in `models.py`. File-local types (e.g., `EntityNode`, `EntityGCDecision` in `entity_gc.py`) stay in their file. `ProcessingChunk.messages` stays `List[Any]` — Pydantic can't resolve TYPE_CHECKING-only imports at runtime.
- `factory.immediate_strategy` bypasses batch submission and is reserved for manual collapse only (segment collapse must complete before the user's next conversation).

## Files

- `models.py` — All Pydantic models and shared TypedDicts. Source of truth for `Memory`, `ExtractedMemory`, `MemoryLink`, `Entity`, `ExtractionBatch`, `PostProcessingBatch`, `ProcessingChunk`, `ConsolidationCluster`, `PendingManualMemory`, and the 18 cross-file TypedDicts. Literal aliases: `RelationshipType`, `BatchStatus`, `BatchKind`.
- `db_access.py` — All SQL for memories, entities, batches, and scoring. Loads scoring formula from `scoring_formula.sql` at import time. Uses `LTMemorySessionManager` for RLS-scoped connections. `get_memories_by_segment_id(segment_id)` traces memories back to their source segment (uses partial index `idx_memories_source_segment_id`).
- `scoring_formula.sql` — SQL expression for importance score recalculation, loaded by `db_access.py` at import time. Edit here, not inline. See below for formula details.
- `factory.py` — Creates and wires all service instances in dependency order with reverse-order cleanup. Singleton via `get_lt_memory_factory()`.
- `vector_ops.py` — Embedding generation and vector similarity search using mdbr-leaf-ir-asym (768d). Owns `HybridSearcher` composition.
- `hybrid_search.py` — BM25 + vector hybrid search with reciprocal rank fusion. `HybridSearcher.hybrid_search()` returns `List[Memory]`.
- `linking.py` — Three-axis candidate discovery (vector similarity, entity co-occurrence with similarity floor, TF-IDF term overlap) and bidirectional link creation. TF-IDF state lazily initialized on `LinkingService`, rebuilt when memory count changes.
- `proactive.py` — Proactive memory surfacing: merges similarity pool and hub-derived pool, reranks with link traversal. Returns `List[MemoryDict]`.
- `llm_routing.py` — Resolver-backed helper for LT memory batch-vs-immediate decisions. `uses_anthropic_batch_adapter(purpose)` is the only routing check for Anthropic Batch API eligibility.
- `hub_discovery.py` — Entity-driven retrieval: pg_trgm fuzzy entity match → linked memories → expansion-similarity ranking. DB errors propagate from `_match_entities()`.
- `refinement.py` — Consolidation cluster identification via connected-components and payload building for batch consolidation.
- `entity_extraction.py` — spaCy NER (en_core_web_lg, parser/lemmatizer disabled) with fuzzy normalization. Returns `List[NamedEntity]`.
- `entity_gc.py` — Entity deduplication: pg_trgm self-join → BFS grouping → Anthropic Batch API review → merge/delete/keep execution. File-local types: `EntityNode`, `EntityGCDecision`, `ReviewPrompt`.
- `memory_formatter.py` — XML formatting of memories and annotations for prompt inclusion.
- `batch_result_handlers.py` — `BatchResultProcessor` implementations for extraction, relationship classification, consolidation, and entity GC. `PostProcessingBatchDispatcher` routes by batch type.
- `__init__.py` — Public re-exports for all shared models and types.
- `processing/` — Extraction and post-processing pipeline (see `processing/CLAUDE.md`).

## Scoring Formula (`scoring_formula.sql`)

The importance score is a single SQL expression producing a value in [0, 1] via sigmoid transform. It uses an "earn your keep" philosophy — new memories start at ~0.5 and must accumulate behavioral signals to maintain relevance.

**Formula pipeline** (computed in this order):

1. **Expiration hard zero** — If `expires_at` is >5 days in the past → `0.0` immediately
2. **Value score** — `LN(1 + access_rate / 0.02) * 0.8` where access_rate = `(access_count × 0.95^activity_days_since_last_access) / MAX(7, activity_days_since_creation)`. The momentum decay (`0.95^days`) means old accesses fade; the `MAX(7, age)` prevents new memories from spiking on day 1
3. **Hub score** — Linear `0.04/link` up to 10 inbound links, then diminishing returns: `0.4 + (links-10)×0.02 / (1 + (links-10)×0.05)`
4. **Entity hub score** — Weighted sum of entity links where weight = `entity.link_count × type_weight` (PERSON=1.0, EVENT=0.9, ORG=0.8, PRODUCT=0.7, etc.). Linear `×0.005` up to 50 weighted links, then logarithmic `0.25 + LN(sum/50)×0.075`
5. **Mention score** — `0.08/mention` up to 5 mentions, then `0.4 + LN(1 + (mentions-5))×0.1`. Explicit LLM references are the strongest behavioral signal
6. **Newness boost** — `MAX(0, 2.0 - age_in_activity_days × 0.133)` gives new memories a 15-day grace period to accumulate signals before decay kicks in
7. **Raw score** = value + hub + entity_hub + mention + newness
8. **Recency boost** — `1 / (1 + activity_days_since_last_access × 0.015)`, half-life ~67 activity days. Gentle transition to cold storage
9. **Temporal multiplier** (calendar-based) — Upcoming events get boosts (≤1 day: 2.0, ≤7 days: 1.5, ≤14 days: 1.2). Past events decay from 0.8→0.4 over 45 days, floor at 0.4
10. **Expiration trailoff** (calendar-based) — Linear decay from 1.0→0.0 over 5 days after `expires_at`
11. **Sigmoid transform** — `1 / (1 + EXP(-(raw × recency × temporal × trailoff - 2.0)))`, center=2.0 maps average memories to ~0.5

**Activity days vs calendar days:** Decay calculations (momentum, recency, newness) use **activity days** (user engagement days) so vacations don't degrade scores. Temporal events (`happens_at`, `expires_at`) use **calendar days** since real-world deadlines don't pause.

**SQL interface:** The formula expects aliases `m` (memories) and `u` (users) with `m.user_id = u.id` join.
