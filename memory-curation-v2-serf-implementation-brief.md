# Serf Implementation Brief: Memory Graph Curation v2

## Run Envelope

You are an autonomous, non-interactive implementation agent running inside a fresh branch/worktree of the MIRA repository from **before** Memory Graph Curation v2 was implemented.

Your objective is to implement **Memory Graph Curation v2** from this document. This is a harness evaluation: the goal is not to match a known patch textually, but to produce the same behavior and architecture from a decision record + actionable plan.

### Operating rules

- Do not ask the user questions. Infer from local code, document assumptions, and continue.
- Read local code before editing. Files and line numbers may differ from this brief; symbol names and behavior are authoritative.
- Prefer idiomatic integration with existing project patterns over blind patching.
- Preserve existing behavior outside the stated scope.
- If a local symbol/file named here is absent or moved, search for the equivalent and adapt.
- Do not implement sentry, batch execution for the curator, or wording revision in this iteration.
- The `extraction_ref` relationship type is removed entirely from `VALID_RELATIONSHIP_TYPES` / `RelationshipType`. Extraction bonds flow to the agent as candidate hints **and** are persisted on the link when the agent links (see the bond-persistence contract).
- Final response must include changed files, tests/commands run, pass/fail evidence, and any known gaps.

---

## ADR Summary

### Context / Forces

MIRA's long-term memory graph (`lt_memory/`) was designed to self-organize through an importance score, typed links, and consolidation. In production it accumulates silt instead:

- The importance score is a **sensor without an actor**. Low scores lower retrieval ranking but do not archive/delete memories.
- Deterministic discovery can identify that memories touch (vector/entity/TF-IDF overlap), but cannot type the relationship.
- Programmatic batch-LLM judgment paths duplicate editorial work across relationship typing and consolidation verdicts. Entity garbage collection is the one programmatic path worth keeping — but as a lean background LLM-merge service, not the old batch-judgment handler.
- The 3-word extraction bond produced at memory creation is useful and must be preserved: as agent input (candidate hints) **and** persisted on the link the agent writes, so it surfaces when either memory is referenced downstream.
- Curation must be bounded. The agent must work from pre-computed context and random low-score floor samples, not sweep the whole graph.
- The agent must not manufacture silt: no `create_memory` in its tool schema.
- New-memory integration and floor triage have different rubrics but belong to one conceptual curator.

### Decision

Create one agentic `MemoryCuratorAgent`, with two modes selected by `work_item.context["mode"]`:

- `integration` — tend newly-created memories: link / merge / stand-alone.
- `floor` — triage low-score, not-recently-tended memories: salvage by link/annotation or archive.

Programmatic code does deterministic work only:

- extraction, embedding, scoring, retrieval
- candidate discovery as typeless `CandidateRef` hints
- merge mechanics through `ConsolidationHandler.execute_consolidation()`
- floor sampling via SQL

Agentic code makes editorial decisions:

- typed relationship links
- merge decisions
- archive/salvage decisions
- stand-alone decisions

### Explicit deferrals / non-goals

- No sentry gate in v2.
- No batch execution for the curator in v2.
- No revise-wording operation in v2.
- Do not fix the existing `corroborates` spelling.
- Entity merge is NOT an agent capability — it runs as a background scheduled service (see the Entity Merge contract). The curator has no entity tool.

---

## Required Contracts and Exact Shapes

### CandidateRef

Define near `MemoryCuratorAgent` or in a small shared module if local conventions favor that.

```python
class CandidateRef(TypedDict):
    """A pre-computed candidate relationship for the curator to judge."""
    memory_id: str            # short ID, e.g. "mem_XXXXXXXX"
    bond: str                 # 3-word extraction bond, "" if none
    discovery_signal: str     # "extraction" | "vector" | "entity" | "tfidf"
    similarity: float | None  # vector similarity only when discovery_signal == "vector"
```

### Work-item contexts

Use full UUID strings inside work-item context where later programmatic code needs full IDs. Render short `mem_XXXXXXXX` IDs only in the initial message shown to the LLM.

```python
class NewMemory(TypedDict):
    memory_id: str  # full UUID string
    text: str

class IntegrationContext(TypedDict):
    mode: Literal["integration"]
    segment_id: str
    new_memories: list[NewMemory]
    candidate_hints: dict[str, list[CandidateRef]]  # full UUID string -> hints

class FloorMemory(TypedDict):
    memory_id: str  # full UUID string
    text: str
    importance_score: float

class FloorContext(TypedDict):
    mode: Literal["floor"]
    memories: list[FloorMemory]
```

### MemoryCuratorAgent class contract

- File: `agents/implementations/memory_curator_agent.py`
- Subclass existing `SidebarAgent`.
- Class attrs:

```python
agent_id = "memory_curator"
# Reuse the system summary model — no dedicated internal_llm row is needed.
internal_llm_key = "summary"
available_tools = ["memory_tool"]  # sidebar_tool is auto-included by base if that is local pattern
tool_schema_overrides = {"memory_tool": CURATOR_MEMORY_SCHEMA}
inherit_base_prompt = False
max_iterations = 8
timeout_seconds = 480
```

- Do **not** set `use_batch`.
- Do **not** set `sentry_llm_key`.
- Do **not** set `overwatch_llm_key`.
- `get_agent_prompt(work_item)` selects one of two prompt files by `context["mode"]`:
  - `config/prompts/agents/memory_curator_integration.txt`
  - `config/prompts/agents/memory_curator_floor.txt`
- `build_initial_message(work_item)` renders:
  - integration: each new memory, its short ID, full text, candidate hints with bond/signal/similarity
  - floor: sampled memories, short IDs, full text, scores
- `on_completion(event_bus, work_item, status, summary)` stamps `last_tended_at` **only on success**:
  - integration: all `new_memories[*].memory_id`
  - floor: all `memories[*].memory_id`
  - failed/timeout runs stamp nothing so floor can resample later
- Publish to `MemoryCuratorTrinket` via the existing base completion event mechanism.

### Prompt rubrics

Create two self-contained prompt files because `inherit_base_prompt = False`. Include loop/`complete_task` instructions; do not rely on base prompt text.

#### Integration prompt must say

- You are in **INTEGRATION MODE**.
- A segment collapsed and new memories were extracted.
- For each new memory decide exactly one of:
  - **MERGE** into redundant existing memory/memories via `memory_tool.merge_memories`
  - **LINK** to existing memories via `memory_tool.link_memories` with exact relationship type
  - **STAND ALONE** if no confident relation exists
- Read candidate hints first; search/confirm/expand neighborhood with `memory_tool.search`.
- Use the extraction bond as context, not as a pre-decided relationship. When you LINK and the candidate hint carried a bond, pass it as the `bond` argument to `link_memories` — it is preserved on the link and surfaces when either memory is referenced elsewhere.
- Do not create memories.
- Do not revise wording.
- Do not manufacture links.
- Finish with `sidebar_tool.complete_task` and summarize actions per memory.

#### Floor prompt must say

- You are in **FLOOR MODE**.
- You were handed low-score memories not tended recently.
- This mode is harsher triage.
- For each memory decide:
  - **ARCHIVE** if low-value/redundant/stale/no longer useful
  - **SALVAGE** by link/annotate/touch if there is recoverable value
  - **LEAVE** only if unsure and no action is justified
- Do not create memories.
- Do not revise wording.
- Do not manufacture links.
- Finish with `sidebar_tool.complete_task` and summarize archive/salvage/leave actions.

### Memory tool contract

File: `tools/implementations/memory_tool.py`

Add base operations:

```text
search, create_memory, link_memories, annotate_memory, touch, archive, merge_memories
```

Add `CURATOR_MEMORY_SCHEMA` module constant: deepcopy/full copy of the base tool schema with `operation.enum` restricted to:

```text
search, link_memories, annotate_memory, touch, archive, merge_memories
```

`CURATOR_MEMORY_SCHEMA` must exclude `create_memory`.

Operation semantics:

- `archive(memory_id)` — resolve short ID, call `LTMemoryDB.archive_memory`, return confirmation.
- `link_memories(source_memory_id, target_memory_id, link_type, reasoning, bond="")` — create a typed bidirectional link. The optional `bond` (the 3-word extraction bond from the candidate hint) is persisted on the link as `extraction_bond` so it survives and surfaces on reference (see the Database contract).
- `merge_memories(memory_ids, consolidated_text, merge_note=None)` — require >=2 distinct non-archived memories, delegate to `ConsolidationHandler.execute_consolidation(old_memory_ids, consolidated_text, user_id, merge_note)`, stamp `last_tended_at` on the new memory, return new short ID.
- No `confidence` field anywhere on this tool — link confidence was a half-hearted feature and is removed (the `MemoryLink` model has no `confidence` field; do not add one). Search results do not report a `confidence`/`high_confidence` status.

### Database contract

Add to `Memory` model:

```python
last_tended_at: Optional[datetime] = None
```

Add migration and fresh schema update:

- `ALTER TABLE memories ADD COLUMN last_tended_at TIMESTAMPTZ;`
- Backfill existing rows:

```sql
UPDATE memories
SET last_tended_at = created_at
WHERE last_tended_at IS NULL;
```

- Add partial index for floor sampling:

```sql
CREATE INDEX IF NOT EXISTS idx_memories_floor_candidates
ON memories (importance_score, last_tended_at)
WHERE is_archived = FALSE;
```

Add DB helpers:

```python
def update_last_tended(self, memory_ids: list[UUID], user_id: Optional[str] = None) -> None: ...

def get_floor_candidates(
    self,
    floor_threshold: float,
    unseen_days: int,
    sample_size: int,
    user_id: Optional[str] = None,
) -> list[UUID]: ...
```

The 3-word extraction bond is stored on the link, not just carried as a hint:

- `MemoryLink` (the pydantic model) carries `extraction_bond: str = ""`.
- `MemoryLinkEntry` (the JSONB `TypedDict`) carries `extraction_bond: NotRequired[str]`.
- `LTMemoryDB.create_links` writes `extraction_bond` onto both the outbound and inbound JSONB entries when the `MemoryLink.extraction_bond` is non-empty.
- `LinkingService.traverse_related` carries `bond` through each `TraversalResult`, so the proactive-memory trinket can render it (see Bond surfacing).

Floor query invariant:

```sql
SELECT id
FROM memories
WHERE is_archived = FALSE
  AND importance_score < :floor_threshold
  AND (
        last_tended_at < now() - interval ':unseen_days days'
     OR (last_tended_at IS NULL AND created_at < now() - interval ':unseen_days days')
  )
ORDER BY random()
LIMIT :sample_size;
```

Use the project's existing RLS-scoped session manager pattern.

### Config / LLM contract

Add `MemoryCuratorConfig`, wired into app config:

```python
class MemoryCuratorConfig(BaseModel):
    enabled: bool = True
    floor_threshold: float = 0.1
    floor_unseen_days: int = 14
    floor_sample_size: int = 8
    floor_use_days: int = 7
```

Add the entity-merge use-day cadence to `ScheduledJobsConfig` (not `MemoryCuratorConfig`):

```python
entity_merge_use_days: int = Field(default=7, ge=1, description="Use-day cadence for background entity dedup/merge")
```

The curator reuses the existing system `summary` `internal_llm` row — **no dedicated `memory_curator` internal_llm row and no `memory_curator` usage_pricing rows.** Do not create them. The three old batch-judgment `internal_llm` rows (`relationship`, `consolidation`, `entity_gc`) are removed from the fresh schema and pruned from already-deployed databases by migration; their `usage_pricing` rows are pruned too.

### Candidate discovery contract

File: `lt_memory/linking.py`

Add:

```python
def find_candidate_hints(self, memory_id: UUID) -> list[dict]: ...
```

Behavior:

- Run existing vector similarity axis.
- Run existing entity co-occurrence axis.
- Run existing TF-IDF axis.
- Deduplicate candidates by UUID.
- Exclude `memory_id` itself.
- Return CandidateRef-shaped dicts:
  - vector: `discovery_signal="vector"`, include similarity if available
  - entity: `discovery_signal="entity"`, `similarity=None`
  - tfidf: `discovery_signal="tfidf"`, `similarity=None`
  - `bond=""` for all these; extraction bonds are merged by the storage helper
- Do not write links.

`create_bidirectional_link` / `create_bidirectional_links` are **deleted** — they had zero callers once extraction-time link writing was removed. The agent writes links through `memory_tool.link_memories` → `LTMemoryDB.create_links` directly. Do not re-add them.

### Storage / spawn contract

File: `lt_memory/processing/execution_strategy.py`

Add shared helpers:

```python
def _persist_llm_entities(user_id, memories, memory_ids, db) -> int: ...

def _build_candidate_hints(memories, memory_ids, linking) -> dict[str, list[dict]]: ...

def store_and_tend_extraction(*, user_id, segment_id, memories, vector_ops, db, linking) -> list[UUID]: ...
```

`_build_candidate_hints` merges:

1. `linking.find_candidate_hints(new_memory_id)`
2. extraction LLM `related_memory_ids` + `bond` from each extracted memory as:

```python
{
  "memory_id": format_memory_id(related_uuid),
  "bond": bond,
  "discovery_signal": "extraction",
  "similarity": None,
}
```

`store_and_tend_extraction` does:

1. store memories with embeddings
2. persist LLM entities
3. build candidate hints
4. call `get_lt_memory_factory().on_memories_stored(...)` if registered
5. never fail storage if the callback fails; log warning and continue

Modify both storage paths to use it:

- immediate execution path
- extraction batch result handler path

Remove all `extraction_ref` link writes from both paths.

### Factory callback contract

File: `lt_memory/factory.py`

Add:

```python
self.on_memories_stored = None
```

This is a late-registered callback. `lt_memory` must not import `agents/`.

File: `cns/services/segment_collapse_handler.py`

- Accept/store `tool_repo` in constructor if not already available.
- In `__init__`, register:

```python
self.lt_memory_factory.on_memories_stored = self._on_memories_stored
```

Implement:

```python
def _on_memories_stored(self, *, user_id, segment_id, memory_ids, memories, candidate_hints) -> None: ...
def _tend_manual_memories(self, user_id, segment_id, stored_manual) -> None: ...
def _spawn_integration_curator(self, user_id, segment_id, new_memories, candidate_hints) -> None: ...
```

Spawn pattern:

- construct `WorkItem(item_id=f"integrate_{segment_id}_{user_id}", interface_name="memory_curator_integration", context=IntegrationContext)`
- construct `MemoryCuratorAgent(tool_repo=self.tool_repo)`
- use `contextvars.copy_context()` and `threading.Thread(..., daemon=True)` to run `agent.run(work_item, event_bus)`
- spawn failure must not fail memory storage/collapse

Manual memories:

- collect stored manual memory IDs/texts after the pending-memory loop
- build discovery-only hints with `linking.find_candidate_hints`
- call `_spawn_integration_curator`
- preserve user-specified score / happens_at / expires_at / supersedes behavior

### Floor trigger contract

File: `agents/triggers/memory_floor_trigger.py`

Implement a trigger compatible with the local `SidebarTrigger` protocol:

- `trigger_id = "memory_floor"`
- `interface_name = "memory_curator_floor"`
- `agent_class` returns/lazy-resolves `MemoryCuratorAgent`
- `check_for_new_items(user_id)`:
  1. load config
  2. if disabled, return `[]`
  3. get user's cumulative activity days
  4. if `activity_days <= 0` or `activity_days % floor_use_days != 0`, return `[]`
  5. call `db.get_floor_candidates(floor_threshold, floor_unseen_days, floor_sample_size, user_id)`
  6. fetch memory texts/scores
  7. return one `WorkItem` with `FloorContext`
- stable `item_id = f"floor_{user_id}_{activity_days}"` for dispatcher dedup
- trigger performs no LLM calls

Register it in sidebar job registration (`utils/sidebar_jobs.py`) and export from `agents/triggers/__init__.py` if that package exists.

### Trinket contract

Add `working_memory/trinkets/memory_curator_trinket.py`, following the closest existing trinket pattern (e.g. forage/activity trinket).

Purpose:

- show curator mode
- status
- segment id if integration
- summary
- actions if derivable from completion context
- success persists until next curation/replacement
- failed/timeout auto-expire after a few turns, matching local trinket conventions

Register it with working memory/trinket setup.

### Deletion contract

Delete or remove all non-test references to:

- `RelationshipBatchResultHandler`
- `ConsolidationBatchResultHandler`
- `EntityGCBatchResultHandler`
- `PostProcessingBatchDispatcher`
- `PostProcessingOrchestrator`
- `EntityGCService` (the old batch-LLM handler — replaced by the lean `entity_merge` service, not re-added)
- `RefinementService` entirely (including `identify_consolidation_clusters` — it had zero callers) and the whole `lt_memory/refinement.py` file
- `LinkingService.build_classification_payload`
- `LinkingService._build_relationship_prompt`
- `LinkingService._format_temporal_fields`
- `LinkingService.find_similar_candidates`
- `LinkingService.create_bidirectional_link` and `create_bidirectional_links` (zero callers once extraction-time link writing was removed)
- `ImmediateExecutionStrategy._trigger_relationship_classification`
- the `post_processing_batches` table, the `PostProcessingBatch` model, the `BatchKind` alias, and the `kind` / `batch_type` parameters threaded through the generic batch DB / coordinator methods (the batch surface is extraction-only now)
- post-processing batch polling job
- consolidation scheduled job
- entity-GC scheduled job (replaced by the entity-merge job — see the Entity Merge contract)
- `consolidation_use_days` config field
- `entity_gc_use_days` config field (the `entity_merge_use_days` field replaces it)
- all `extraction_ref` link writes, and remove `"extraction_ref"` from `VALID_RELATIONSHIP_TYPES` / `RelationshipType`; prune surviving JSONB rows by migration
- `Memory.consolidation_rejection_count` and `LTMemoryDB.increment_consolidation_rejection_count` (old consolidation tracking; zero callers)
- `ConsolidationHandler.validate_consolidation_cluster` (old pre-validation; zero callers)
- `BatchCoordinator.poll_batches` / `BatchResultProcessor.finalize_batch` post-chunk hook (base no-op + sole override no-op + the call site + the write-only `any_succeeded` flag that gated it)
- `lt_memory/memory_formatter.py` entirely (`format_memories_for_consolidation` and its helpers `format_annotations_xml` / `format_links_summary_xml` — an orphan module; nothing imported it)
- the extraction-side `consolidates_memory_ids` handling on `ExtractedMemory` (the field stays on the model — `ConsolidationHandler` writes it — but the extraction prompts never emit it, so the `RawMemoryDict` typing, construction, short→UUID remap, validation, and the `_is_duplicate_memory` guard are dead and removed)
- the extraction-side `relationship_type` on `ExtractedMemory` and its parse/validate scaffolding (the prompt says the downstream pipeline determines the relationship type; not stored on `Memory`; the `MemoryLink.link_type` field and `VALID_RELATIONSHIP_TYPES` set stay)
- the dead entity DB methods `delete_entity`, `archive_entity`, `get_active_entities`, `get_entities_by_ids` (only the old `entity_gc.py` called them). The live entity methods `get_or_create_entity`, `link_memory_to_entity`, `get_memories_for_entity` stay; `find_similar_entity_pairs`, `get_entity`, `merge_entities` are restored for the entity-merge service (see Entity Merge contract).
- the `memory_curator` `internal_llm` row and its `usage_pricing` rows if present (the curator reuses `summary`); and the now-dead `relationship` / `consolidation` / `entity_gc` `internal_llm` + `usage_pricing` rows, pruned by migration.
- all `confidence`-related fields on `memory_tool` / `MemoryLink`: `manual_link_confidence` config, the phantom `confidence=` kwarg, the search `confidence` / `high_confidence`/`medium_confidence`/`low_confidence` status, and the unused `confidence_cluster_threshold` config. (Unrelated confidence systems — `continuum_tool`, prompt-injection defense, user-model observation confidence, CRM — are out of scope and untouched.)

Keep:

- `ExtractionBatchResultHandler`
- `BatchCoordinator` extraction polling (extraction-only — no `kind` / `batch_type` dispatch param)
- `ConsolidationHandler.execute_consolidation`
- `MemoryLink.extraction_bond` / `MemoryLinkEntry.extraction_bond` and the `create_links` write logic (restored — see the Database contract)
- the `find_candidate_hints` discovery accessor and its vector/entity/tfidf internals + `traverse_related` + `cleanup`
- the restored entity-merge DB helpers (`find_similar_entity_pairs`, `get_entity`, `merge_entities`, `EntityPairRow`)

### Entity Merge contract

Entity dedup/merge runs as a **background scheduled service**, not an agent capability and not the old batch-LLM handler. Over time, entities like `"Round Lake"` / `"Round Lake, IL"` / `"round lake"` accumulate as separate rows across sessions (`get_or_create_entity` only fuzzy-dedups at creation, never after), fragmenting entity-driven retrieval. A lean service fixes this.

File: `lt_memory/entity_merge.py` (free functions + a single entry point; factory-singleton deps like the domaindoc summary service — no factory wiring).

Flow for `run_entity_merge_for_user(user_id) -> dict[str, int]`:

1. `db.find_similar_entity_pairs(threshold, user_id=...)` — pg_trgm self-join over non-archived entities (threshold `0.6`, calibrated from the old service's production data).
2. `_build_merge_groups(pairs)` — BFS connected-components over the pairs → groups of `{id, name, type, link_count}`.
3. Cap to `GROUPS_PER_CALL` (25) groups per call — surplus groups are re-found next run; merging shrinks the candidate pool each pass.
4. `_build_groups_prompt(groups)` — render groups as JSON-ish blocks with 8-char short entity IDs; keep a `short_to_full` map.
5. Single LLM call via the `analysis` `internal_llm` (the system's cheap model), `allow_negative=True`. System prompt `entity_merge_system.txt`, user template `entity_merge_user.txt`. Ask which entities are the same and which is canonical. JSON output `{"merges": [{"canonical": "<8char>", "merge": ["<8char>", ...]}]}`.
6. `_parse_merge_response` — validate every short ID exists in the input (reject hallucinated IDs); skip entries with unknown canonical or empty merge list. Robust JSON extraction (markdown fences or first `{...}` block).
7. Execute each merge via `db.merge_entities(source_id, target_id, user_id=...)` (rewrites memory `entity_links` to point at the canonical, deduplicating memories already linked to both, bumps `link_count`, archives the source). Best-effort — log and skip individual failures.

Files:

- `lt_memory/entity_merge.py`
- `config/prompts/entity_merge_system.txt`
- `config/prompts/entity_merge_user.txt`
- `lt_memory/db_access.py` — `find_similar_entity_pairs`, `get_entity`, `merge_entities` (restored); `lt_memory/models.py` — `EntityPairRow` (restored)
- config: `entity_merge_use_days` on `ScheduledJobsConfig`
- job registration: `lt_memory_entity_merge` job (use-day-gated, `IntervalTrigger(days=1)` + `get_users_due_for_job`) in `utils/lt_memory_jobs.py`
- POST: add `lt_memory_entity_merge` to the `power_on_self_test` required-jobs set

Acceptance:

- the service uses **one LLM call** presenting candidate groups and parsing JSON — no batch API, no agent tools, no regex heuristics for the merge decision.
- the LLM only decides; execution is deterministic and RLS-scoped.
- runs infrequently (use-day cadence).
- `entities.is_archived` column stays — it is read live by `hub_discovery`.

### AGENTS.md / orientation maps

Update nested orientation docs required by the project:

- `agents/AGENTS.md`
- `lt_memory/AGENTS.md`
- `tools/implementations/AGENTS.md`

They must describe the final state, not the old programmatic-judgment flow.

---

## Implementation Plan

### Step 1 — Add `last_tended_at` and floor DB helpers

Files:

- `lt_memory/models.py`
- `lt_memory/db_access.py`
- fresh schema SQL if the repo has one
- new migration under the repo's migration directory

Acceptance:

- `Memory` has `last_tended_at`.
- migration adds column, backfills existing rows to `created_at`, adds index/comment if local schema uses comments.
- fresh schema provisions column/index without migration.
- `update_last_tended` updates RLS-scoped rows.
- `get_floor_candidates` returns only non-archived, below-threshold, old-tended or old-never-tended rows, capped by sample size.

### Step 2 — Add memory tool ops and curator schema

Files:

- `tools/implementations/memory_tool.py`

Acceptance:

- base schema includes `archive` and `merge_memories`; keeps `create_memory`.
- `CURATOR_MEMORY_SCHEMA` excludes `create_memory` and includes the six curator ops.
- archive op soft-deletes memory.
- merge op delegates to `ConsolidationHandler.execute_consolidation` and stamps new memory tended.
- module imports cleanly.

### Step 3 — Implement `MemoryCuratorAgent` and prompts

Files:

- `agents/implementations/memory_curator_agent.py`
- `config/prompts/agents/memory_curator_integration.txt`
- `config/prompts/agents/memory_curator_floor.txt`

Acceptance:

- class attrs match contract.
- prompt selected by mode.
- initial messages render hints/scores.
- success stamps tended UUIDs; failure/timeout stamps nothing.
- tool schema override excludes `create_memory`.
- prompts contain required boundaries and `complete_task` instruction.

### Step 4 — Add config (no dedicated internal LLM row)

Files:

- config model files
- fresh schema
- migration SQL

Acceptance:

- `MemoryCuratorConfig` loadable from app config.
- `entity_merge_use_days` is on `ScheduledJobsConfig`.
- the curator resolves the existing system `summary` `internal_llm` row — **no `memory_curator` row is created.**
- the dead `relationship` / `consolidation` / `entity_gc` `internal_llm` + `usage_pricing` rows are removed from the fresh schema and pruned by migration.
- migration is idempotent.

### Step 5 — Implement floor trigger and register it

Files:

- `agents/triggers/memory_floor_trigger.py`
- `agents/triggers/__init__.py`
- `utils/sidebar_jobs.py` or equivalent registration site

Acceptance:

- disabled/not-due/no-candidates return `[]`.
- due+candidates returns one WorkItem with `mode="floor"`.
- stable item_id dedups same activity day.
- trigger performs no LLM calls.

### Step 6 — Consolidate extraction storage and spawn integration curator

Files:

- `lt_memory/processing/execution_strategy.py`
- `lt_memory/batch_result_handlers.py`
- `lt_memory/factory.py`
- `cns/services/segment_collapse_handler.py`
- any construction site for `SegmentCollapseHandler`

Acceptance:

- immediate and batch-result storage paths call `store_and_tend_extraction`.
- no `extraction_ref` links written anywhere.
- candidate_hints include deterministic discovery plus extraction bonds.
- `on_memories_stored` callback seam exists.
- integration curator spawn is best-effort and non-blocking.
- manual pending memories are also tended.

### Step 7 — Delete programmatic judgment paths and scheduled jobs

Files:

- `lt_memory/batch_result_handlers.py`
- `lt_memory/processing/post_processing_orchestrator.py`
- `lt_memory/entity_gc.py`
- `lt_memory/refinement.py`
- `lt_memory/factory.py`
- `lt_memory/processing/__init__.py`
- `lt_memory/memory_formatter.py`
- scheduled jobs/config files

Acceptance:

- grep for deleted symbols returns clean in non-test code.
- extraction batch path still imports/works.
- `refinement.py` and `RefinementService` are gone entirely (including `identify_consolidation_clusters`).
- the `post_processing_batches` table, `PostProcessingBatch` model, `BatchKind`, and the `kind` / `batch_type` dispatch params are gone; `BatchCoordinator` is extraction-only.
- `memory_formatter.py` is gone; `finalize_batch` post-chunk hook and the `any_succeeded` flag are gone.
- `consolidation_rejection_count` / `increment_consolidation_rejection_count` and `validate_consolidation_cluster` are gone.
- the extraction-side `consolidates_memory_ids` and `relationship_type` scaffolding are gone (the `MemoryLink.link_type` field and `VALID_RELATIONSHIP_TYPES` set stay).
- the dead entity DB methods (`delete_entity`, `archive_entity`, `get_active_entities`, `get_entities_by_ids`) are gone; the live ones stay.
- app modules compile/import.

### Step 8 — Build the entity-merge background service

Files:

- `lt_memory/entity_merge.py` (new)
- `config/prompts/entity_merge_system.txt` (new)
- `config/prompts/entity_merge_user.txt` (new)
- `lt_memory/db_access.py` — restore `find_similar_entity_pairs`, `get_entity`, `merge_entities`
- `lt_memory/models.py` — restore `EntityPairRow`
- `config/config.py` — add `entity_merge_use_days` to `ScheduledJobsConfig`
- `utils/lt_memory_jobs.py` — register the `lt_memory_entity_merge` job (use-day-gated)
- `utils/power_on_self_test.py` — add `lt_memory_entity_merge` to the required-jobs set

Acceptance:

- the service makes **one LLM call** (the `analysis` model) presenting candidate groups and parsing JSON — no batch API, no agent tools.
- the LLM only decides merges; execution is deterministic (`merge_entities`) and RLS-scoped.
- the job is use-day-gated (`entity_merge_use_days`) and registered in the POST required set.
- `entities.is_archived` stays (read by `hub_discovery`).
- modules compile/import.

### Step 9 — Demote linking to typeless candidate hints

Files:

- `lt_memory/linking.py`
- `lt_memory/processing/execution_strategy.py`

Acceptance:

- `find_candidate_hints` is the discovery accessor.
- `find_similar_candidates`, classification payload/prompt builders, and immediate relationship classification are gone.
- `create_bidirectional_link(s)` are **deleted** (zero programmatic callers once extraction-time link writing was removed); the agent writes links via `memory_tool.link_memories` → `LTMemoryDB.create_links`.
- the 3-word bond is persisted on the link (`MemoryLink.extraction_bond`) and surfaced on reference (`traverse_related` carries `bond`; the proactive-memory trinket renders it).
- grep invariant: no programmatic typed-link writes remain; typed links originate through `memory_tool.link_memories` / agent-driven path.

### Step 10 — Observability and docs

Files:

- `working_memory/trinkets/memory_curator_trinket.py`
- trinket registration site
- `agents/implementations/memory_curator_agent.py`
- nested `AGENTS.md` files

Acceptance:

- curator completion publishes to `MemoryCuratorTrinket`.
- all code compiles/imports.
- orientation docs describe final state and do not reference deleted runtime paths.

---

## Required Verification Commands

Adapt commands to local environment. At minimum run compile/import checks and grep invariants.

```bash
# Compile changed Python files broadly
python -m py_compile \
  agents/implementations/memory_curator_agent.py \
  agents/triggers/memory_floor_trigger.py \
  tools/implementations/memory_tool.py \
  lt_memory/models.py \
  lt_memory/db_access.py \
  lt_memory/linking.py \
  lt_memory/entity_merge.py \
  lt_memory/processing/execution_strategy.py \
  lt_memory/batch_result_handlers.py \
  lt_memory/factory.py \
  utils/lt_memory_jobs.py

# Deleted judgment paths must be absent
grep -rn "RelationshipBatchResultHandler\|ConsolidationBatchResultHandler\|EntityGCBatchResultHandler\|PostProcessingBatchDispatcher\|PostProcessingOrchestrator\|EntityGCService\|build_consolidation_payload\|RefinementService\|identify_consolidation_clusters" --include='*.py' . | grep -v '/.git/' || true

# Deleted link classifiers + extraction-time link writers must be absent
grep -rn "build_classification_payload\|_build_relationship_prompt\|find_similar_candidates\|_trigger_relationship_classification\|create_bidirectional_link" --include='*.py' . | grep -v '/.git/' || true

# extraction_ref gone from the relationship type set; no link writes of it
grep -rn 'extraction_ref' --include='*.py' lt_memory/ tools/ || true

# memory_tool has no confidence (the half-hearted feature is removed)
grep -rn 'confidence\|manual_link_confidence\|high_confidence' tools/implementations/memory_tool.py || true

# Dead consolidation tracking / dead hooks / orphan formatter must be absent
grep -rn 'consolidation_rejection_count\|validate_consolidation_cluster\|finalize_batch\|memory_formatter' --include='*.py' . | grep -v '/.git/' || true

# Entity-merge service is wired: config field + job + POST required set
python - <<'PY'
from config import config
assert config.scheduled_jobs.entity_merge_use_days >= 1
import lt_memory.entity_merge as m
assert hasattr(m, 'run_entity_merge_for_user')
print('entity_merge wired')
PY
grep -rn 'lt_memory_entity_merge' utils/lt_memory_jobs.py utils/power_on_self_test.py

# Curator schema excludes create_memory
python - <<'PY'
from tools.implementations.memory_tool import MemoryTool, CURATOR_MEMORY_SCHEMA
base = set(MemoryTool.tool_schema['input_schema']['properties']['operation']['enum'])
cur = set(CURATOR_MEMORY_SCHEMA['input_schema']['properties']['operation']['enum'])
assert 'create_memory' in base
assert 'create_memory' not in cur
assert {'search','link_memories','annotate_memory','touch','archive','merge_memories'} <= cur
assert 'bond' in CURATOR_MEMORY_SCHEMA['input_schema']['properties'], 'link_memories must accept a bond param'
print('schema ok')
PY
```

If a local Postgres instance is available, additionally verify:

- migrations apply cleanly (the set: `add_last_tended_at.sql`, `drop_post_processing_batches.sql`, `prune_extraction_ref_links.sql`, `drop_consolidation_rejection_count.sql`, `prune_dead_curation_llm_rows.sql`)
- fresh schema creates `last_tended_at`; the `post_processing_batches` table is absent; the `consolidation_rejection_count` column is absent; the dead `relationship` / `consolidation` / `entity_gc` `internal_llm` + `usage_pricing` rows are absent.
- floor query returns only eligible memories across these cases:
  - low + old + tended → eligible
  - low + old + never-tended → eligible
  - low + recent-tended → excluded
  - low + new + never-tended → excluded
  - high + old → excluded
  - archived + old → excluded

---

## Gotchas / Failure Modes to Avoid

- Do not spawn integration only at the collapse hook; batch-dialect extraction stores later. Spawn at storage points.
- Do not let `lt_memory` import from `agents/`; use the factory callback seam.
- `extraction_ref` is removed from the relationship type set; do not write it. The 3-word bond is preserved as agent input **and** persisted on the link the agent writes (`link_memories` `bond` arg → `MemoryLink.extraction_bond`) and surfaced on reference — do not treat bonds as hints-only.
- Do not stamp `last_tended_at` on failed/timeout curator runs.
- Do not store only short IDs in work-item context; programmatic completion needs full UUIDs.
- `create_bidirectional_link(s)` are **deleted** — do not re-add them. The agent writes links through `memory_tool.link_memories` → `LTMemoryDB.create_links`.
- Do not implement sentry or batch mode for curator.
- Do not add `create_memory` to `CURATOR_MEMORY_SCHEMA`.
- Do not create a dedicated `memory_curator` `internal_llm` row or its `usage_pricing` rows — the curator reuses the system `summary` model.
- Do not add any `confidence` field to `memory_tool` / `MemoryLink` — link/search confidence was a half-hearted feature and is removed.
- Entity merge is a background scheduled service, not an agent capability — do not give the curator an entity-merge tool.
- Do not delete extraction batch handling; only post-processing judgment paths go away. The `post_processing_batches` table, `PostProcessingBatch` model, `BatchKind`, and the `kind` / `batch_type` dispatch params are all removed — `BatchCoordinator` is extraction-only.
- Do not forget fresh schema updates in addition to migrations.
- Do not forget nested `AGENTS.md` updates.

---

## Final Report Required

When done, report:

1. Summary of behavior implemented.
2. Files changed/created/deleted.
3. Tests and commands run with outputs/summaries.
4. Grep invariants and whether they pass.
5. DB verification performed or why skipped.
6. Known gaps or assumptions.
7. Any intentional deviations from this brief and why.
