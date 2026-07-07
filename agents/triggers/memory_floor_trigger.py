"""
MemoryFloorTrigger — dispatcher-driven floor-mode curation trigger.

Surfaces a random sample of low-value memories not tended recently as a single
floor-mode work-item for the MemoryCuratorAgent. Pure deterministic discovery:
no LLM, no graph scanning — just SQL heuristics over importance_score +
last_tended_at staleness. The agent makes all judgment decisions (archive /
salvage) on the surfaced sample.

Cadence: use-day gated. The floor fires only on activity days where
MOD(cumulative_activity_days, floor_use_days) == 0, mirroring the stateless
get_users_due_for_job() modular condition but checked inline per user (the
dispatcher already iterates users, so a per-user modular check is cheaper and
semantically correct here than re-querying all users each poll).

Dedup: item_id is stable per activity-day cycle (floor_{user_id}_{activity_days}),
so the dispatcher's sidebar_activity table skips re-dispatch after the first
successful floor run that day. A failed run is not retried (max_retries=0); the
next due activity day re-samples.
"""
import logging
from typing import TYPE_CHECKING

from agents.sidebar import WorkItem

if TYPE_CHECKING:
    from agents.base import SidebarAgent

logger = logging.getLogger(__name__)


class MemoryFloorTrigger:
    """Surfaces floor-mode curation work-items for the MemoryCuratorAgent."""

    trigger_id = "memory_floor"
    interface_name = "memory_curator_floor"

    # agent_class is resolved lazily at class-access time to avoid an import
    # cycle at module load (MemoryCuratorAgent -> tools.implementations.memory_tool
    # -> ...). The dispatcher reads trigger.agent_class per dispatch.

    def __init__(self):
        from utils.database_session_manager import get_shared_session_manager
        from lt_memory.db_access import LTMemoryDB
        self._db = LTMemoryDB(get_shared_session_manager())

    @property
    def agent_class(self) -> type:
        from agents.implementations.memory_curator_agent import MemoryCuratorAgent
        return MemoryCuratorAgent

    def check_for_new_items(self, user_id: str) -> list[WorkItem]:
        """Return at most one floor work-item for this user, or []."""
        from config.config_manager import config
        from utils.user_context import get_user_cumulative_activity_days

        cfg = config.memory_curator
        if not cfg.enabled:
            return []

        # Use-day cadence gate (inline modular check for THIS user; the
        # dispatcher already iterates users, so this is cheaper than calling
        # get_users_due_for_job() per poll and semantically correct).
        activity_days = get_user_cumulative_activity_days()
        if activity_days <= 0 or activity_days % cfg.floor_use_days != 0:
            return []

        # Deterministic sample — no LLM, no graph scan.
        uuids = self._db.get_floor_candidates(
            floor_threshold=cfg.floor_threshold,
            unseen_days=cfg.floor_unseen_days,
            sample_size=cfg.floor_sample_size,
            user_id=user_id,
        )
        if not uuids:
            return []

        # Fetch texts + scores so the agent can judge without a search round-trip
        # just to read the memories it was handed (bounded, pre-computed context).
        memories = self._db.get_memories_by_ids(uuids, user_id=user_id)
        if not memories:
            return []

        floor_memories = [
            {
                "memory_id": str(m.id),
                "text": m.text,
                "importance_score": m.importance_score,
            }
            for m in memories
        ]

        # Stable per-cycle key: same activity day → same item_id → dispatcher
        # dedup prevents re-dispatch after the first successful run.
        item_id = f"floor_{user_id}_{activity_days}"

        logger.info(
            "MemoryFloorTrigger: surfacing %d floor memories for user %s "
            "(activity day %d)",
            len(floor_memories), user_id, activity_days,
        )

        return [WorkItem(
            item_id=item_id,
            interface_name=self.interface_name,
            context={
                "mode": "floor",
                "memories": floor_memories,
            },
        )]

    def on_dispatched(self, user_id: str, item_id: str) -> None:
        """No side effects (floor curation leaves no IMAP-style flags)."""
        pass
