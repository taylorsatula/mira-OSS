# Memory Gardener — Implementation Roughout

> **NOT FINISHED.** This document is a working design and partial implementation
> sketch. It is not production-ready. Gaps are called out explicitly throughout.

---

## Background

The scoring formula correctly decays unused memories to a floor of ~0.119 (sigmoid of -2.0),
but the system has no lifecycle policy beyond that point. Currently:

- ~906 memories are permanently pinned at the floor with no path to rehabilitation or removal
- The existing consolidation job (`refinement.py`, 7-day cadence) is deadheaded
- 64.5% of all memories have zero accesses — the extraction pipeline produces far more
  memories than the retrieval pipeline ever uses

The memory gardener replaces the scheduled consolidation job with a per-user agent that
traverses the memory graph and makes incremental lifecycle decisions daily. Its advantage
over the formula is that it can read content, reason about semantic uniqueness, and make
judgment calls the math cannot: whether a silt memory's content is already captured
elsewhere, whether a fossil event leaves behind a durable fact worth keeping, whether a
stranded node should be wired into the graph or discarded.

See `NEARFUTURE_FEATURES.md` for the original one-line concept.

---

## Tending Zones

The gardener works across four zones per daily run, bounded by a per-user memory budget
to keep each run tractable:

| Zone       | Criteria                                                                   | Gardener action                              |
|------------|----------------------------------------------------------------------------|----------------------------------------------|
| **Silt**   | `importance_score <= 0.12`, age ≥ 15 activity days, not archived           | Check coverage, merge or archive             |
| **Fossil** | `happens_at < NOW() - 45 days`, not archived                               | Extract durable fact if any, archive shell   |
| **Strand** | 0 entity links, 0 inbound links, 0 accesses, age ≥ 3 activity days        | Attempt entity wiring, else archive          |
| **Thicket**| 3+ memories sharing a high-link-count entity (future zone — not in v1)    | Consolidate redundant cluster                |

Thicket detection requires a clustering query that is non-trivial to scope correctly.
Leave it out of v1 and let the existing consolidation pipeline handle it until the
gardener is stable.

---

## Files to Create

```
tools/implementations/memory_garden_tool.py
agents/implementations/memory_gardener_agent.py
config/prompts/agents/memory_gardener_system.txt
```

## Files to Modify

```
config/config.py                  — add gardener_use_days to ScheduledJobsConfig
utils/scheduled_tasks.py          — register gardener trigger + dispatch
agents/CLAUDE.md                  — document new agent
tools/implementations/CLAUDE.md   — document new tool
```

---

## `memory_garden_tool.py`

```python
"""
Memory garden tool — administrative write operations for the MemoryGardenerAgent.

Bypasses the Valkey/segment-collapse path used by memory_tool.create_memory.
The gardener runs outside conversation context (no active segment), so merged
memories must be persisted directly via LTMemoryDB + embeddings provider.

Operations:
    garden_status       — counts per zone, used by trigger to skip no-work days
    read_zone           — pull a batch of memories from a named zone
    archive_memory      — soft-delete with audit annotation
    update_text         — rewrite memory text (enrichment, fossil extraction)
    create_merged       — persist a merged memory and archive the absorbed originals
"""
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from lt_memory.db_access import LTMemoryDB
from lt_memory.models import ExtractedMemory
from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now, format_utc_iso
from utils.user_context import get_current_user_id, get_user_cumulative_activity_days
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider

logger = logging.getLogger(__name__)

# Silt zone threshold — matches sigmoid(-2.0) = 0.119 floor, with a small buffer
SILT_SCORE_CEILING = 0.12
SILT_MIN_AGE_ACTIVITY_DAYS = 15
FOSSIL_PAST_DAYS = 45
STRAND_MIN_AGE_ACTIVITY_DAYS = 3


class MemoryGardenToolConfig(BaseModel):
    max_zone_batch: int = Field(
        default=20,
        description="Max memories returned per read_zone call"
    )


registry.register("memory_garden_tool", MemoryGardenToolConfig)


class MemoryGardenTool(Tool):
    name = "memory_garden_tool"

    simple_description = (
        "Administrative memory lifecycle operations for the gardener agent: "
        "zone reads, archival, text updates, and merge-creation."
    )

    anthropic_schema = {
        "name": "memory_garden_tool",
        "description": (
            "Administrative memory lifecycle operations. "
            "Not for use outside the memory_gardener agent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "garden_status",
                        "read_zone",
                        "archive_memory",
                        "update_text",
                        "create_merged",
                    ],
                    "description": "Operation to perform"
                },
                "zone": {
                    "type": "string",
                    "enum": ["silt", "fossil", "strand"],
                    "description": "Zone to read. Required for read_zone."
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max memories to return. Default 10. Used by read_zone."
                },
                "memory_id": {
                    "type": "string",
                    "description": "Full UUID of the memory to act on. Required for archive_memory and update_text."
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Required for archive_memory, update_text, and create_merged. "
                        "Stored as an audit annotation. Min 10 characters."
                    )
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text for the memory. Required for update_text."
                },
                "merged_text": {
                    "type": "string",
                    "description": (
                        "Text for the new merged memory. Required for create_merged. "
                        "Should synthesize the content of all absorbed memories."
                    )
                },
                "absorbs_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Full UUIDs of memories to archive after creating the merge. "
                        "Required for create_merged. Min 1 item."
                    )
                },
            },
            "required": ["operation"]
        }
    }

    def __init__(self):
        super().__init__()
        config_cls = registry.get("memory_garden_tool") or MemoryGardenToolConfig
        self._config = config_cls()
        self._memory_db = LTMemoryDB(get_shared_session_manager())
        self._embeddings_provider = get_hybrid_embeddings_provider()

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        if operation == "garden_status":
            return self._garden_status()
        elif operation == "read_zone":
            return self._read_zone(**kwargs)
        elif operation == "archive_memory":
            return self._archive_memory(**kwargs)
        elif operation == "update_text":
            return self._update_text(**kwargs)
        elif operation == "create_merged":
            return self._create_merged(**kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _garden_status(self) -> Dict[str, Any]:
        """Return zone counts. Used by the trigger to decide whether to dispatch."""
        user_id = get_current_user_id()
        activity_days = get_user_cumulative_activity_days()

        with self._memory_db.session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT
                    SUM(CASE
                        WHEN m.importance_score <= %(silt_ceiling)s
                         AND NOT m.is_archived
                         AND (%(activity_days)s - COALESCE(m.activity_days_at_creation, 0)) >= %(silt_min_age)s
                        THEN 1 ELSE 0
                    END) AS silt_count,
                    SUM(CASE
                        WHEN m.happens_at IS NOT NULL
                         AND m.happens_at < NOW() - INTERVAL '45 days'
                         AND NOT m.is_archived
                        THEN 1 ELSE 0
                    END) AS fossil_count,
                    SUM(CASE
                        WHEN jsonb_array_length(COALESCE(m.entity_links, '[]'::jsonb)) = 0
                         AND jsonb_array_length(COALESCE(m.inbound_links, '[]'::jsonb)) = 0
                         AND m.access_count = 0
                         AND NOT m.is_archived
                         AND (%(activity_days)s - COALESCE(m.activity_days_at_creation, 0)) >= %(strand_min_age)s
                        THEN 1 ELSE 0
                    END) AS strand_count
                FROM memories m
            """, {
                'silt_ceiling': SILT_SCORE_CEILING,
                'silt_min_age': SILT_MIN_AGE_ACTIVITY_DAYS,
                'strand_min_age': STRAND_MIN_AGE_ACTIVITY_DAYS,
                'activity_days': activity_days,
            })

        return {
            'silt_count': result['silt_count'] or 0,
            'fossil_count': result['fossil_count'] or 0,
            'strand_count': result['strand_count'] or 0,
            'has_work': any([
                (result['silt_count'] or 0) > 0,
                (result['fossil_count'] or 0) > 0,
                (result['strand_count'] or 0) > 0,
            ])
        }

    def _read_zone(self, zone: str, limit: int = 10, **kwargs) -> Dict[str, Any]:
        """Pull a batch of memories from a named zone."""
        user_id = get_current_user_id()
        activity_days = get_user_cumulative_activity_days()
        limit = min(int(limit), self._config.max_zone_batch)

        zone_filters = {
            'silt': """
                m.importance_score <= %(silt_ceiling)s
                AND NOT m.is_archived
                AND (%(activity_days)s - COALESCE(m.activity_days_at_creation, 0)) >= %(silt_min_age)s
                ORDER BY m.importance_score ASC, m.created_at ASC
            """,
            'fossil': """
                m.happens_at IS NOT NULL
                AND m.happens_at < NOW() - INTERVAL '45 days'
                AND NOT m.is_archived
                ORDER BY m.happens_at ASC
            """,
            'strand': """
                jsonb_array_length(COALESCE(m.entity_links, '[]'::jsonb)) = 0
                AND jsonb_array_length(COALESCE(m.inbound_links, '[]'::jsonb)) = 0
                AND m.access_count = 0
                AND NOT m.is_archived
                AND (%(activity_days)s - COALESCE(m.activity_days_at_creation, 0)) >= %(strand_min_age)s
                ORDER BY m.created_at ASC
            """,
        }

        if zone not in zone_filters:
            raise ValueError(f"Unknown zone: {zone}. Must be one of: {list(zone_filters)}")

        query = f"""
            SELECT
                m.id, m.text, m.importance_score, m.access_count,
                m.mention_count, m.happens_at, m.expires_at,
                m.entity_links, m.inbound_links, m.outbound_links,
                m.annotations, m.source_segment_id,
                m.activity_days_at_creation,
                m.activity_days_at_last_access,
                (%(activity_days)s - COALESCE(m.activity_days_at_creation, 0)) AS age_activity_days
            FROM memories m
            WHERE {zone_filters[zone]}
            LIMIT %(limit)s
        """

        with self._memory_db.session_manager.get_session(user_id) as session:
            rows = session.execute_query(query, {
                'activity_days': activity_days,
                'silt_ceiling': SILT_SCORE_CEILING,
                'silt_min_age': SILT_MIN_AGE_ACTIVITY_DAYS,
                'strand_min_age': STRAND_MIN_AGE_ACTIVITY_DAYS,
                'limit': limit,
            })

        return {
            'zone': zone,
            'count': len(rows),
            'memories': [
                {
                    'id': str(r['id']),
                    'text': r['text'],
                    'importance_score': float(r['importance_score']),
                    'access_count': r['access_count'],
                    'mention_count': r['mention_count'],
                    'happens_at': format_utc_iso(r['happens_at']) if r['happens_at'] else None,
                    'expires_at': format_utc_iso(r['expires_at']) if r['expires_at'] else None,
                    'entity_links': r['entity_links'] or [],
                    'inbound_links_count': len(r['inbound_links'] or []),
                    'outbound_links_count': len(r['outbound_links'] or []),
                    'annotations': r['annotations'] or [],
                    'source_segment_id': str(r['source_segment_id']) if r['source_segment_id'] else None,
                    'age_activity_days': r['age_activity_days'],
                }
                for r in rows
            ]
        }

    def _archive_memory(self, memory_id: str, reason: str, **kwargs) -> Dict[str, Any]:
        """Soft-delete a memory with an audit annotation."""
        if not reason or len(reason.strip()) < 10:
            raise ValueError("reason must be at least 10 characters")

        uid = UUID(memory_id)
        user_id = get_current_user_id()

        # Append audit annotation before archiving so reason is in the record
        memory = self._memory_db.get_memory(uid, user_id=user_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        existing = memory.annotations or []
        audit = {
            "text": f"[gardener] Archived: {reason.strip()}",
            "created_at": format_utc_iso(utc_now()),
            "source": "gardener",
        }
        import json
        self._memory_db.update_memory(uid, {"annotations": json.dumps(existing + [audit])})
        self._memory_db.archive_memory(uid, user_id=user_id)

        return {"status": "archived", "memory_id": memory_id}

    def _update_text(self, memory_id: str, new_text: str, reason: str, **kwargs) -> Dict[str, Any]:
        """Rewrite memory text. Used for enrichment and fossil fact extraction."""
        if not new_text or len(new_text.strip()) < 10:
            raise ValueError("new_text must be at least 10 characters")
        if not reason or len(reason.strip()) < 10:
            raise ValueError("reason must be at least 10 characters")

        uid = UUID(memory_id)
        user_id = get_current_user_id()

        # Regenerate embedding for new text
        new_embedding = self._embeddings_provider.encode_realtime(new_text.strip())

        import json
        memory = self._memory_db.get_memory(uid, user_id=user_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        existing = memory.annotations or []
        audit = {
            "text": f"[gardener] Text updated: {reason.strip()}",
            "created_at": format_utc_iso(utc_now()),
            "source": "gardener",
        }

        self._memory_db.update_memory(uid, {
            "text": new_text.strip(),
            "embedding": new_embedding.tolist(),
            "annotations": json.dumps(existing + [audit]),
        })

        # Force tsvector refresh — update_memory triggers update_memories_search_vector
        # only on text column change, which this is, so no extra step needed.

        return {"status": "updated", "memory_id": memory_id}

    def _create_merged(
        self,
        merged_text: str,
        absorbs_ids: List[str],
        reason: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Persist a merged memory and archive the absorbed originals.

        Uses store_memories (direct DB insert, no Valkey) so this works
        outside conversation context. The merged memory starts at the
        newness boost score (0.5) and must earn its keep like any new memory.

        NOTE: Entity extraction and linking do NOT run here — those pipeline
        steps require the extraction batch job. The merged memory will be
        unlinked until the next consolidation/linking pass. This is acceptable
        for background maintenance work.

        TODO: Consider triggering a lightweight entity extraction pass inline
        so the merged memory enters the graph connected from the start.
        """
        if not merged_text or len(merged_text.strip()) < 10:
            raise ValueError("merged_text must be at least 10 characters")
        if not absorbs_ids:
            raise ValueError("absorbs_ids must contain at least one ID")
        if not reason or len(reason.strip()) < 10:
            raise ValueError("reason must be at least 10 characters")

        # Generate embedding for merged text
        embedding = self._embeddings_provider.encode_realtime(merged_text.strip())

        new_memory = ExtractedMemory(
            text=merged_text.strip(),
            importance_score=0.5,  # Enter at newness boost baseline
        )

        created_ids = self._memory_db.store_memories(
            memories=[new_memory],
            embeddings=[embedding.tolist()],
        )
        new_id = str(created_ids[0])

        # Archive all absorbed memories
        archived = []
        failed = []
        for abs_id in absorbs_ids:
            try:
                self._archive_memory(
                    abs_id,
                    reason=f"Absorbed into merged memory {new_id}: {reason}"
                )
                archived.append(abs_id)
            except Exception as e:
                logger.warning(f"create_merged: failed to archive {abs_id}: {e}")
                failed.append(abs_id)

        return {
            "status": "merged",
            "new_memory_id": new_id,
            "absorbed_count": len(archived),
            "archived_ids": archived,
            "failed_to_archive": failed,
        }
```

---

## `memory_gardener_agent.py`

```python
"""
MemoryGardenerAgent — Daily per-user memory graph curator.

Replaces the deadheaded scheduled consolidation job. Runs in batch mode
(background, no latency requirement). Dispatched by MemoryGardenerTrigger
via SidebarDispatcher on each user's use-day schedule.

Does NOT use the sentry gate — the trigger already pre-filters via
garden_status counts, so no tokens are burned on no-work days.
"""
import logging
from typing import Any, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


class MemoryGardenerAgent(SidebarAgent):
    agent_id = "memory_gardener"
    internal_llm_key = "memory_gardener"      # TODO: add to internal_llm table
    available_tools = ["memory_garden_tool", "memory_tool"]
    inherit_base_prompt = False

    max_iterations = 10
    use_batch = True
    batch_timeout_seconds = 3600
    timeout_seconds = 14400  # 4h ceiling: 3600 * 10 iterations worst-case

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        return load_agent_prompt("memory_gardener_system.txt")

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        counts = work_item.context.get('zone_counts', {})
        return (
            f"Today's garden status:\n"
            f"  Silt (floor memories, never accessed): {counts.get('silt_count', 0)}\n"
            f"  Fossils (past-due event memories): {counts.get('fossil_count', 0)}\n"
            f"  Stranded (unlinked, never accessed): {counts.get('strand_count', 0)}\n\n"
            "Tend to what's here. Prioritize silt first — it's most likely to contain "
            "something worth absorbing before archiving. Work through each zone "
            "systematically. Use memory_tool search to check whether silt content "
            "is already captured in a stronger memory before deciding to archive or merge."
        )
```

---

## `MemoryGardenerTrigger`

Attach to `agents/implementations/memory_gardener_agent.py` or a standalone
`agents/implementations/memory_gardener_trigger.py`.

```python
class MemoryGardenerTrigger:
    """
    Fires daily per-user when garden_status reports work to do.

    WorkItem.item_id format: "memory_gardener:{user_id}:{activity_day}"
    This means one gardener run per user per activity day — dedup via
    sidebar_activity prevents re-dispatch within the same day.
    """
    trigger_id = "memory_gardener"
    interface_name = "memory_gardener"
    agent_class = MemoryGardenerAgent

    def check_for_new_items(self, user_id: str) -> list['WorkItem']:
        from agents.sidebar import WorkItem
        from utils.user_context import set_current_user_id, clear_user_context, get_user_cumulative_activity_days
        from tools.implementations.memory_garden_tool import MemoryGardenTool

        try:
            set_current_user_id(user_id)
            tool = MemoryGardenTool()
            status = tool.run(operation="garden_status")

            if not status.get('has_work'):
                return []

            activity_day = get_user_cumulative_activity_days()
            return [WorkItem(
                item_id=f"memory_gardener:{user_id}:{activity_day}",
                interface_name=self.interface_name,
                context={'zone_counts': status},
            )]
        except Exception as e:
            logger.warning(f"MemoryGardenerTrigger: status check failed for {user_id}: {e}")
            return []
        finally:
            clear_user_context()

    def on_dispatched(self, user_id: str, item_id: str) -> None:
        pass  # No side effects needed
```

---

## System Prompt Sketch (`memory_gardener_system.txt`)

> **NOT WRITTEN.** This is the most consequential piece and needs careful authoring.
> The gardener makes irreversible decisions (archival). The prompt must make the
> agent conservative: when uncertain, annotate and skip rather than archive.

Key directives to include:

- **Default to caution.** Archive only when confident the content is covered elsewhere
  or genuinely valueless. Annotate uncertain cases with a `[gardener] reviewed, retained`
  note and move on.
- **Silt zone protocol**: For each silt memory, call `memory_tool search` with the memory
  text as the query. If a memory scoring > 0.5 clearly covers the same ground, archive the
  silt memory noting which memory supersedes it. If nothing surfaced, check whether the
  silt memory states a unique fact. Unique facts → attempt rehabilitation via `update_text`
  to improve clarity, then retain. Generic/vague/ephemeral content → archive.
- **Fossil zone protocol**: Read the event memory. Extract any durable fact it implies
  (e.g. "planned to visit X" → "has visited or attempted X"). If a durable fact exists
  and isn't already captured, call `update_text` to rewrite the memory as a past-tense
  fact (remove the `happens_at` framing). Then archive only the now-replaced event version
  if a new memory was created. If no durable fact, archive directly.
- **Strand zone protocol**: Read the text. Use `memory_tool search` to find entity neighbors.
  If a clear entity match exists, call `memory_tool link_memories` to wire the strand in.
  A connected strand is no longer stranded. If no connections found and content is
  thin/ephemeral, archive.
- **Merge conservatively.** Only merge when two memories make the same claim with enough
  redundancy that a combined version is genuinely better than either alone. The merged text
  must be richer than either source — not a lossy summary.
- **Budget your iterations.** Process ~5–8 memories per iteration. Use `sidebar_tool`
  scratchpad to track what you've done. Call `complete_task` with a summary when all
  zones are tended or iteration budget approaches.

---

## Config Changes

### `config/config.py` — add to `ScheduledJobsConfig`

```python
gardener_use_days: int = Field(
    default=1,
    description="Use-days between memory gardener runs"
)
```

### `utils/scheduled_tasks.py` — registration

```python
# In initialize_all_scheduled_tasks(), after sidebar dispatcher registration:

from agents.implementations.memory_gardener_agent import MemoryGardenerTrigger

gardener_trigger = MemoryGardenerTrigger()
sidebar_dispatcher.register_trigger(gardener_trigger)
logger.info("Registered memory_gardener trigger")
```

The trigger fires on the existing SidebarDispatcher poll loop (daily calendar tick).
No new APScheduler job needed — the dispatcher already polls all registered triggers.

### `internal_llm` table — add gardener LLM entry

```sql
-- TODO: choose model. Sonnet — needs real semantic reasoning, not Haiku.
-- Haiku will make bad archival decisions. The gardener is low-frequency
-- (once/user/day) and batch-mode (50% cost), so Sonnet is affordable.
INSERT INTO internal_llm (name, model, max_tokens, endpoint_url, api_key_name)
VALUES ('memory_gardener', 'claude-sonnet-4-6', 8192, NULL, 'anthropic_api_key');
```

---

## What's Left

- [ ] Write `memory_gardener_system.txt` prompt (most important, not started)
- [ ] Add `gardener` LLM config to `internal_llm` table and migrations
- [ ] Decide whether `create_merged` should trigger inline entity extraction — currently
      merged memories enter the graph unlinked and depend on a future pipeline pass
- [ ] Thicket zone: clustering query to find redundant same-topic memory groups
- [ ] Add `gardener_use_days` to `ScheduledJobsConfig` and verify it threads through correctly
- [ ] Tests for `memory_garden_tool` operations, especially `create_merged` archival
      rollback behavior when some `absorbs_ids` fail
- [ ] Decide whether the gardener should emit a `UpdateTrinketEvent` to a new
      `MemoryGardenerTrinket` or just write silently to `sidebar_activity`. Silent
      is probably right for a maintenance agent — no user-facing progress needed
- [ ] Audit `_archive_memory` → `_read_zone` race: if the gardener reads a silt batch,
      then separately tries to archive an ID from that batch, and the memory was
      archived by another path in between, the current `archive_memory` call will
      silently no-op (UPDATE WHERE id=X matches 0 rows). Should it raise or return
      a `was_already_archived` flag?
- [ ] The `update_text` / fossil extraction path leaves the old `happens_at` on the
      memory. After rewriting a fossil to a past-tense fact, the `happens_at` should
      be cleared so the temporal multiplier stops applying. Add `happens_at = NULL`
      to the update in `_update_text` when the gardener calls it on a fossil.
- [ ] Register `memory_garden_tool` in `ESSENTIAL_TOOLS` or as a gardener-only
      restricted tool — it should never be accessible from the primary conversation
      LLM, only from the gardener agent. Check how `sidebar_tool` handles this
      (`enabled: False` in config, sidebar agents only).
