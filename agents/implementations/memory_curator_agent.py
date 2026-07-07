"""
MemoryCuratorAgent -- Agentic memory-graph curation on the SidebarAgent base.

Replaces the programmatic batch-LLM judgment paths (relationship typing,
consolidation verdict, entity GC verdict) with a single agent that makes all
memory-graph judgment decisions: linking, merging, and (floor mode) archiving.

One class, two modes, selected per work-item via work_item.context['mode']:
  - 'integration': a segment just collapsed; tend each new memory (link / merge
    / stand-alone) before the user's next conversation. Spawned directly at the
    collapse hook (forage-style).
  - 'floor': a random sample of low-value memories not tended in a while;
    triage each (archive / salvage). Dispatcher-driven on a use-day cadence.

The agent is the SOLE link typer: deterministic discovery emits typeless
candidate hints; this agent writes every typed relationship. It can NEVER
create memories (create_memory is excluded from its tool schema) -- preventing
manufactured silt.

See agents/HOW_TO_BUILD_AN_AGENT.md and the v2 memory-graph curation plan.
"""
import logging
from typing import Any, TypedDict, TYPE_CHECKING

from agents.base import SidebarAgent, load_agent_prompt
from tools.implementations.memory_tool import CURATOR_MEMORY_SCHEMA

if TYPE_CHECKING:
    from agents.sidebar import WorkItem
    from tools.repo import ToolRepository

logger = logging.getLogger(__name__)


# ============================================================================
# Work-item context contract
# ============================================================================
# These TypedDicts are the contract between the triggers (MemoryFloorTrigger,
# segment-collapse spawn) and this agent. The triggers populate
# work_item.context to one of these shapes; build_initial_message renders it
# for the LLM and on_completion reads the memory IDs back out to stamp
# last_tended_at.
#
# Full UUID strings are carried (not short IDs) because last_tended_at updates
# need full UUIDs and short IDs are an irreversible prefix. build_initial_message
# formats them to mem_XXXXXXXX for display and tool calls.

class CandidateRef(TypedDict):
    """A pre-computed candidate relationship for the agent to judge.

    Deterministic discovery (LinkingService.find_candidate_hints +
    extraction-time related_memory_ids) produces these. The discovery code can
    see that two memories touch; it cannot classify HOW they relate -- that is
    the agent's job.
    """
    memory_id: str               # SHORT id (mem_XXXXXXXX) -- for the LLM's tool calls
    bond: str                    # 3-word extraction bond, "" if none
    discovery_signal: str         # "extraction" | "vector" | "entity" | "tfidf"
    similarity: float | None     # vector similarity if discovery_signal == "vector"


class NewMemory(TypedDict):
    """A freshly extracted memory for the integration agent to tend."""
    memory_id: str               # FULL UUID string -- for on_completion's last_tended stamp
    text: str


class IntegrationContext(TypedDict):
    """work_item.context shape for mode='integration'."""
    mode: str                    # "integration"
    segment_id: str
    new_memories: list[NewMemory]
    candidate_hints: dict[str, list[CandidateRef]]  # full-UUID-string(new memory) -> candidates


class FloorMemory(TypedDict):
    """A sampled low-value memory for the floor agent to triage."""
    memory_id: str               # FULL UUID string -- for on_completion's last_tended stamp
    text: str
    importance_score: float


class FloorContext(TypedDict):
    """work_item.context shape for mode='floor'."""
    mode: str                    # "floor"
    memories: list[FloorMemory]


# ============================================================================
# Agent
# ============================================================================

class MemoryCuratorAgent(SidebarAgent):
    """Agentic curator of the memory graph (integration + floor modes)."""

    agent_id = "memory_curator"
    # Reuse the system summary model (internal_llm 'summary') for curation
    # judgment — no dedicated internal_llm row is needed.
    internal_llm_key = "summary"
    available_tools = ["memory_tool"]  # sidebar_tool auto-appended by the base

    # The curator operates on existing memories only -- never creates new ones.
    # This override replaces the memory_tool schema handed to the LLM with a
    # surface that excludes create_memory (prevents manufactured silt).
    tool_schema_overrides = {"memory_tool": CURATOR_MEMORY_SCHEMA}

    # Self-contained rubric prompts (include their own loop/complete_task framing).
    inherit_base_prompt = False

    # Sync (no use_batch): integration must finish before the user's next
    # conversation; per-mode batch behavior isn't cleanly expressible in one
    # class. The rubric drives early complete_task; this ceiling is generous
    # enough for ~8 memories (multiple tool calls per turn).
    max_iterations = 8
    timeout_seconds = 480

    # No sentry gate in v2 -- the floor trigger bounds volume via deterministic
    # SQL heuristics; the agent judges the surfaced items. No overwatch -- the
    # trinket completion event is the observability surface.
    sanitize_untrusted_input = False

    def __init__(self, tool_repo: 'ToolRepository'):
        super().__init__(tool_repo)

    # ------------------------------------------------------------------
    # Prompt + initial message (mode-branched)
    # ------------------------------------------------------------------

    def get_agent_prompt(self, work_item: 'WorkItem') -> str:
        mode = work_item.context.get('mode')
        if mode == 'integration':
            return load_agent_prompt("memory_curator_integration.txt")
        if mode == 'floor':
            return load_agent_prompt("memory_curator_floor.txt")
        # Unknown mode is a trigger bug -- fail loud, not silent.
        raise ValueError(
            f"MemoryCuratorAgent: work_item.context['mode'] must be "
            f"'integration' or 'floor', got {mode!r}"
        )

    def build_initial_message(self, work_item: 'WorkItem') -> str:
        mode = work_item.context.get('mode')
        if mode == 'integration':
            return self._build_integration_message(work_item)
        if mode == 'floor':
            return self._build_floor_message(work_item)
        raise ValueError(
            f"MemoryCuratorAgent: work_item.context['mode'] must be "
            f"'integration' or 'floor', got {mode!r}"
        )

    def _build_integration_message(self, work_item: 'WorkItem') -> str:
        ctx: IntegrationContext = work_item.context  # type: ignore[assignment]
        segment_id = ctx.get('segment_id', 'unknown')
        new_memories: list[NewMemory] = ctx.get('new_memories', [])
        hints: dict[str, list[CandidateRef]] = ctx.get('candidate_hints', {})

        lines: list[str] = [
            f"You are tending {len(new_memories)} new "
            f"{'memories' if len(new_memories) != 1 else 'memory'} from segment "
            f"{segment_id} (integration mode).",
            "",
            "For each, candidate hints are pre-computed suggestions of existing "
            "memories that might relate. Evaluate each hint, optionally use "
            "memory_tool search to confirm or expand the neighborhood, then "
            "decide: MERGE, LINK (with the exact link_type), or STAND ALONE. "
            "Decide every memory, then call sidebar_tool complete_task.",
            "",
        ]

        for mem in new_memories:
            full_id = mem['memory_id']
            short_id = _short(full_id)
            lines.append(f"## {short_id}")
            lines.append(mem['text'].strip())
            lines.append("")

            mem_hints = hints.get(full_id, [])
            if not mem_hints:
                lines.append("Candidate hints: (none — search if you suspect a duplicate)")
            else:
                lines.append("Candidate hints:")
                for h in mem_hints:
                    parts = [f"- {h['memory_id']} [{h['discovery_signal']}]"]
                    if h.get('similarity') is not None:
                        parts.append(f"sim {h['similarity']:.2f}")
                    if h.get('bond'):
                        parts.append(f"bond: \"{h['bond']}\"")
                    lines.append(" — ".join(parts))
            lines.append("")

        return "\n".join(lines).strip()

    def _build_floor_message(self, work_item: 'WorkItem') -> str:
        ctx: FloorContext = work_item.context  # type: ignore[assignment]
        memories: list[FloorMemory] = ctx.get('memories', [])

        lines: list[str] = [
            f"You are tending {len(memories)} low-value "
            f"{'memories' if len(memories) != 1 else 'memory'} sampled from the "
            "store (floor mode).",
            "",
            "For each, use memory_tool search to verify against the store, then "
            "decide: ARCHIVE (redundant / stale / trivial / wrong) or SALVAGE "
            "(link to a genuinely related memory, or annotate with needed context). "
            "This is the harsher mode — these already score low, so archive when "
            "the evidence supports it. Decide every memory, then call sidebar_tool "
            "complete_task.",
            "",
        ]

        for mem in memories:
            short_id = _short(mem['memory_id'])
            score = mem.get('importance_score', 0.0)
            lines.append(f"## {short_id} (score {score:.2f})")
            lines.append(mem['text'].strip())
            lines.append("")

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Completion — stamp last_tended_at, publish observability
    # ------------------------------------------------------------------

    def on_completion(
        self,
        event_bus: 'EventBus',
        work_item: 'WorkItem',
        status: str,
        summary: str,
    ) -> None:
        """Stamp tended memories and publish a completion event.

        Only stamps last_tended_at on a successful run. A failed/timeout run
        leaves memories un-tended so the floor trigger re-samples them later
        (new memories stay NULL until tended and age into floor eligibility
        via the NULL+age branch).
        """
        tended_uuids = self._tended_uuids_for_status(work_item, status)
        if tended_uuids:
            self._stamp_tended(tended_uuids)

        # Publish the completion event to the MemoryCuratorTrinket.
        super().on_completion(event_bus, work_item, status, summary)

    def _tended_uuids_for_status(
        self, work_item: 'WorkItem', status: str
    ) -> list[Any]:
        """Return the full UUIDs to stamp, or [] if the run shouldn't stamp."""
        if status != 'success':
            return []

        mode = work_item.context.get('mode')
        if mode == 'integration':
            raw = [m['memory_id'] for m in work_item.context.get('new_memories', [])]
        elif mode == 'floor':
            raw = [m['memory_id'] for m in work_item.context.get('memories', [])]
        else:
            return []

        uuids: list[Any] = []
        for mid in raw:
            try:
                from uuid import UUID
                uuids.append(UUID(mid))
            except (ValueError, TypeError):
                logger.warning(
                    "MemoryCuratorAgent: skipping unparseable memory_id %r in "
                    "last_tended stamp", mid,
                )
        return uuids

    def _stamp_tended(self, uuids: list[Any]) -> None:
        """Stamp last_tended_at on the tended memories (RLS-scoped, ambient user)."""
        try:
            from utils.database_session_manager import get_shared_session_manager
            from lt_memory.db_access import LTMemoryDB
            db = LTMemoryDB(get_shared_session_manager())
            db.update_last_tended(uuids)
        except Exception:
            # Observability only — never let the stamp failure mask a
            # successful curation run. Un-stamped memories will re-surface via
            # the floor trigger's NULL+age branch.
            logger.exception("MemoryCuratorAgent: last_tended stamp failed")

    def _get_completion_trinket(self) -> str:
        return 'MemoryCuratorTrinket'

    def _build_completion_context(
        self,
        status: str,
        summary: str,
        work_item: 'WorkItem',
    ) -> dict[str, Any]:
        ctx: dict[str, Any] = {
            'task_id': work_item.item_id,
            'status': status,
            'mode': work_item.context.get('mode'),
            'summary': summary,
        }
        if work_item.context.get('mode') == 'integration':
            ctx['segment_id'] = work_item.context.get('segment_id', '')
        return ctx


# ============================================================================
# Helpers
# ============================================================================

def _short(full_uuid: str) -> str:
    """Format a full UUID string to the mem_XXXXXXXX display form."""
    from utils.tag_parser import format_memory_id
    return format_memory_id(full_uuid)
