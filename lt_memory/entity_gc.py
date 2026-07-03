"""
Entity garbage collection service.

Finds entity pairs with similar names (pg_trgm self-join), groups them
via BFS connected-components, submits to Anthropic Batch API for review
(XML structured output), and executes merge/delete/keep decisions on completion.

Two-phase flow:
  1. submit_entity_gc_batch(): pairs → groups → batch requests → Anthropic Batch API
  2. EntityGCBatchResultHandler: poll → parse XML → execute merge/delete/keep

Synchronous fallback (run_entity_gc_for_user) when batch transport is unavailable.
"""
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from datetime import timedelta
from typing import Any, List, Dict, NamedTuple, Optional, TypedDict
from uuid import UUID

from lt_memory.db_access import LTMemoryDB
from lt_memory.llm_routing import uses_anthropic_batch_dialect
from lt_memory.models import PostProcessingBatch, EntityPairRow, GCStats
from lt_memory.processing.batch_coordinator import BatchCoordinator, BATCH_EXPIRY_HOURS
from clients.llm_provider import LLMProvider, build_batch_params
from utils.user_context import get_current_user_id
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

# pg_trgm similarity threshold for entity name matching
# Calibrated from production data (887 entities): 0.6 yields ~408 pairs
ENTITY_GC_SIMILARITY_THRESHOLD = 0.6

VALID_ACTIONS = {'canonical', 'merge', 'delete', 'keep'}

# Max groups per LLM call. 10 groups x ~3 entities avg = ~30 entities per batch.
# Keeps prompt size manageable and avoids models dropping groups from output.
GROUPS_PER_BATCH = 10


# ============================================================================
# File-local types (used only within entity_gc.py)
# ============================================================================

class EntityNode(TypedDict):
    """Node in the BFS connected-components graph."""
    id: UUID
    name: str
    type: str
    link_count: int


class EntityGCDecision(TypedDict):
    """Single entity decision from LLM GC response."""
    id: UUID
    action: str


class ReviewPrompt(NamedTuple):
    """Result of _build_review_prompt: prompt text and short-to-full UUID mapping."""
    user_prompt: str
    short_to_full: Dict[str, UUID]


def _format_entity_id(uuid_str: str) -> str:
    """8-char short ID for LLM prompts (same pattern as format_memory_id)."""
    return uuid_str.replace('-', '')[:8]


class EntityGCService:
    """
    Service for entity garbage collection.

    Normal flow (batch API):
        submit_entity_gc_batch() → Anthropic Batch API → polling picks up result →
        EntityGCBatchResultHandler calls parse_gc_response() + execute_gc_decisions()

    Failover flow (synchronous):
        run_entity_gc_for_user() → direct LLM calls → immediate execution
    """

    def __init__(
        self,
        db: LTMemoryDB,
        llm_provider: LLMProvider,
        batch_coordinator: BatchCoordinator,
    ):
        self.db = db
        self.llm_provider = llm_provider
        self.batch_coordinator = batch_coordinator
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load GC prompts from external files."""
        from config.prompts.loader import load_prompt
        self.gc_system_prompt = load_prompt("entity_gc_system.txt")
        self.gc_user_template = load_prompt("entity_gc_user.txt")

    # ============================================================================
    # Pair Discovery & Grouping
    # ============================================================================

    def _build_merge_groups(
        self,
        pairs: List[EntityPairRow]
    ) -> List[List[EntityNode]]:
        """
        BFS connected-components over entity pairs.

        Returns list of groups, each group is a list of
        {id: UUID, name: str, type: str, link_count: int}.
        """
        graph: Dict[str, set] = defaultdict(set)
        entity_info: Dict[str, Dict[str, Any]] = {}

        for pair in pairs:
            id_a = str(pair['id_a'])
            id_b = str(pair['id_b'])
            graph[id_a].add(id_b)
            graph[id_b].add(id_a)
            entity_info[id_a] = {
                'id': pair['id_a'],
                'name': pair['name_a'],
                'type': pair['type_a'],
                'link_count': pair['links_a'],
            }
            entity_info[id_b] = {
                'id': pair['id_b'],
                'name': pair['name_b'],
                'type': pair['type_b'],
                'link_count': pair['links_b'],
            }

        visited: set = set()
        groups: List[List[Dict[str, Any]]] = []

        for node in graph:
            if node in visited:
                continue
            component: List[str] = []
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                queue.extend(graph[current] - visited)

            groups.append([entity_info[eid] for eid in component])

        return groups

    # ============================================================================
    # Prompt Building
    # ============================================================================

    def _build_review_prompt(
        self,
        groups: List[List[EntityNode]]
    ) -> ReviewPrompt:
        """
        Build XML prompt with all groups for LLM review.

        Returns:
            (user_prompt, short_to_full) where short_to_full maps
            8-char short IDs to full UUIDs.
        """
        short_to_full: Dict[str, UUID] = {}
        group_xml_parts: List[str] = []

        for i, group in enumerate(groups, 1):
            entity_lines: List[str] = []
            for entity in group:
                full_id = entity['id']
                short_id = _format_entity_id(str(full_id))
                short_to_full[short_id] = UUID(str(full_id))
                entity_lines.append(
                    f'  <entity id="{short_id}" name="{entity["name"]}" '
                    f'type="{entity["type"]}" links="{entity["link_count"]}"/>'
                )
            group_xml_parts.append(
                f'<group id="{i}">\n' +
                '\n'.join(entity_lines) +
                '\n</group>'
            )

        groups_xml = '\n'.join(group_xml_parts)
        user_prompt = self.gc_user_template.format(groups=groups_xml)
        return ReviewPrompt(user_prompt, short_to_full)

    # ============================================================================
    # Response Parsing
    # ============================================================================

    def parse_gc_response(
        self,
        response_text: str,
        short_to_full: Dict[str, UUID]
    ) -> List[List[EntityGCDecision]]:
        """
        Parse XML response into per-group action lists.

        Validates:
        - Every entity ID exists in input (rejects hallucinated IDs)
        - Every action is in {canonical, merge, delete, keep}
        - Groups with merge must have exactly one canonical (skip group if violated)
        - Missing entities default to keep

        Returns:
            List of groups, each group is a list of {id: UUID, action: str}.
            Groups that fail validation are excluded entirely.
        """
        gc_start = response_text.find('<gc_decisions>')
        if gc_start == -1:
            logger.error("No <gc_decisions> tag in GC response")
            return []
        xml_text = response_text[gc_start:]

        gc_end = xml_text.find('</gc_decisions>')
        if gc_end != -1:
            xml_text = xml_text[:gc_end + len('</gc_decisions>')]

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            logger.exception("Failed to parse GC XML response")
            return []

        parsed_groups: List[List[Dict[str, Any]]] = []

        for group_elem in root.findall('group'):
            group_actions: List[Dict[str, Any]] = []
            group_id = group_elem.get('id', '?')

            for entity_elem in group_elem.findall('entity'):
                short_id = entity_elem.get('id', '')
                action = entity_elem.get('action', '').lower()

                if short_id not in short_to_full:
                    logger.warning(f"Hallucinated entity ID '{short_id}' in group {group_id}")
                    continue
                if action not in VALID_ACTIONS:
                    logger.warning(
                        f"Unknown action '{action}' for entity {short_id} — defaulting to keep"
                    )
                    action = 'keep'

                group_actions.append({
                    'id': short_to_full[short_id],
                    'action': action,
                })

            # Split into subgroups by canonical.
            # Walk actions in order — each canonical starts a new subgroup.
            # Merges after a canonical belong to it. Keep/delete are standalone.
            # Orphaned merges before the first canonical default to keep.
            canonicals = [a for a in group_actions if a['action'] == 'canonical']

            if len(canonicals) == 0:
                # No canonicals — entire group is keep/delete only, still valid
                parsed_groups.append(group_actions)
            elif len(canonicals) == 1:
                # Single canonical — validate it has at least one merge
                merges = [a for a in group_actions if a['action'] == 'merge']
                if not merges:
                    logger.warning(
                        f"Group {group_id}: canonical without any merges — treating as keep"
                    )
                    for a in group_actions:
                        if a['action'] == 'canonical':
                            a['action'] = 'keep'
                parsed_groups.append(group_actions)
            else:
                # Multiple canonicals — split into subgroups.
                # Each canonical + subsequent merges form a subgroup.
                # Keep/delete entities go into a standalone group.
                subgroups: List[List[Dict[str, Any]]] = []
                current_subgroup: List[Dict[str, Any]] = []
                standalone: List[Dict[str, Any]] = []

                for a in group_actions:
                    if a['action'] == 'canonical':
                        # Flush previous subgroup if it has a canonical
                        if current_subgroup:
                            subgroups.append(current_subgroup)
                        current_subgroup = [a]
                    elif a['action'] == 'merge':
                        if current_subgroup:
                            current_subgroup.append(a)
                        else:
                            # Orphaned merge before first canonical
                            a['action'] = 'keep'
                            standalone.append(a)
                    else:
                        standalone.append(a)

                # Flush last subgroup
                if current_subgroup:
                    subgroups.append(current_subgroup)

                # Validate each subgroup has at least one merge
                for sg in subgroups:
                    sg_merges = [a for a in sg if a['action'] == 'merge']
                    if sg_merges:
                        parsed_groups.append(sg)
                    else:
                        # Canonical without merges — demote to keep
                        for a in sg:
                            if a['action'] == 'canonical':
                                a['action'] = 'keep'
                        standalone.extend(sg)

                if standalone:
                    parsed_groups.append(standalone)

        return parsed_groups

    # ============================================================================
    # Decision Execution
    # ============================================================================

    def execute_gc_decisions(
        self,
        parsed_groups: List[List[EntityGCDecision]],
        user_id: str,
    ) -> GCStats:
        """
        Execute parsed GC decisions (merge/delete/keep).

        Used by both the batch result handler and synchronous fallback.

        Returns:
            Statistics: {"merged": N, "deleted": N, "kept": N, "errors": N}
        """
        stats = {"merged": 0, "deleted": 0, "kept": 0, "errors": 0}

        for group_actions in parsed_groups:
            canonical_id = None
            for a in group_actions:
                if a['action'] == 'canonical':
                    canonical_id = a['id']
                    break

            for entity_action in group_actions:
                entity_id = entity_action['id']
                action = entity_action['action']

                try:
                    if action == 'merge':
                        self.db.merge_entities(
                            source_id=entity_id,
                            target_id=canonical_id,
                            user_id=user_id
                        )
                        logger.info(f"Merged entity {entity_id} → {canonical_id}")
                        stats["merged"] += 1

                    elif action == 'delete':
                        self.db.archive_entity(entity_id, user_id=user_id)
                        logger.info(f"Archived entity {entity_id}")
                        stats["deleted"] += 1

                    elif action in ('keep', 'canonical'):
                        stats["kept"] += 1

                except Exception as e:
                    logger.error(
                        f"Error executing {action} for entity {entity_id}: {e}",
                        exc_info=True
                    )
                    stats["errors"] += 1

        return stats

    # ============================================================================
    # Batch API Submission
    # ============================================================================

    def submit_entity_gc_batch(self) -> Optional[str]:
        """
        Find similar entity pairs, group them, and submit to Anthropic Batch API.

        In batch mode: builds requests and submits to Anthropic Batch API.
        In immediate mode: executes synchronously via run_entity_gc_for_user().

        Returns:
            Batch ID if submitted, None if no pairs found or immediate mode executed.
        """
        user_id = get_current_user_id()

        # 1. Find similar entity pairs
        pairs = self.db.find_similar_entity_pairs(
            similarity_threshold=ENTITY_GC_SIMILARITY_THRESHOLD,
            user_id=user_id
        )

        if not pairs:
            logger.info(f"No similar entity pairs for user {user_id}")
            return None

        # 2. Group via BFS connected-components
        groups = self._build_merge_groups(pairs)
        logger.info(
            f"Found {len(pairs)} similar pairs → {len(groups)} groups "
            f"for user {user_id}"
        )

        if not uses_anthropic_batch_dialect("entity_gc"):
            logger.warning(
                f"Bypassing entity GC batch for user {user_id} — "
                f"executing {len(groups)} groups synchronously"
            )
            self._execute_synchronous(groups, user_id)
            return None

        # 4. Build batch requests (one request per GROUPS_PER_BATCH chunk)
        requests = []
        input_data = {}

        for batch_idx, batch_start in enumerate(range(0, len(groups), GROUPS_PER_BATCH)):
            batch_groups = groups[batch_start:batch_start + GROUPS_PER_BATCH]
            user_prompt, short_to_full = self._build_review_prompt(batch_groups)

            custom_id = f"{user_id}_entitygc_{batch_idx}"

            params = build_batch_params(
                'entity_gc',
                system_prompt=self.gc_system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            requests.append({
                "custom_id": custom_id,
                "params": params
            })

            # Store short_to_full as serializable strings
            input_data[str(batch_idx)] = {
                "short_to_full": {
                    short_id: str(full_uuid)
                    for short_id, full_uuid in short_to_full.items()
                }
            }

        # 5. Submit to Anthropic Batch API
        batch_id = self.batch_coordinator.submit_batch(
            requests=requests,
            batch_type="entity_gc",
            user_id=user_id,
        )

        # 6. Store batch record for polling
        expires_at = utc_now() + timedelta(hours=BATCH_EXPIRY_HOURS)
        batch_record = PostProcessingBatch(
            batch_id=batch_id,
            batch_type="entity_gc",
            user_id=user_id,
            request_payload={"requests": requests},
            input_data=input_data,
            items_submitted=len(requests),
            status="submitted",
            created_at=utc_now(),
            submitted_at=utc_now(),
            expires_at=expires_at
        )
        self.db.create_post_processing_batch(batch_record, user_id=user_id)

        logger.info(
            f"Submitted entity GC batch {batch_id}: "
            f"{len(requests)} requests ({len(groups)} groups) for user {user_id}"
        )
        return batch_id

    # ============================================================================
    # Synchronous Fallback (Failover Mode)
    # ============================================================================

    def _execute_synchronous(
        self,
        groups: List[List[EntityNode]],
        user_id: str,
    ) -> None:
        """Execute entity GC synchronously when Anthropic failover is active."""
        stats = {"merged": 0, "deleted": 0, "kept": 0, "errors": 0}

        for batch_start in range(0, len(groups), GROUPS_PER_BATCH):
            batch = groups[batch_start:batch_start + GROUPS_PER_BATCH]
            user_prompt, short_to_full = self._build_review_prompt(batch)

            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=self.gc_system_prompt,
                internal_llm='entity_gc',
                allow_negative=True,
            )
            response_text = self.llm_provider.extract_text_content(response)

            parsed_groups = self.parse_gc_response(response_text, short_to_full)
            batch_stats = self.execute_gc_decisions(parsed_groups, user_id)

            for key in stats:
                stats[key] += batch_stats[key]

        logger.info(
            f"Entity GC (synchronous) complete for user {user_id}: "
            f"{stats['merged']} merged, {stats['deleted']} deleted, "
            f"{stats['kept']} kept, {stats['errors']} errors"
        )

    def run_entity_gc_for_user(self) -> GCStats:
        """
        Run full entity GC cycle synchronously for current user.

        This is the synchronous fallback path, used when:
        - Anthropic failover is active
        - Called from tuning harness or manual invocation

        For normal operation, use submit_entity_gc_batch() instead.

        Returns:
            Statistics: {"merged": N, "deleted": N, "kept": N, "errors": N}
        """
        user_id = get_current_user_id()

        # 1. Find similar entity pairs
        pairs = self.db.find_similar_entity_pairs(
            similarity_threshold=ENTITY_GC_SIMILARITY_THRESHOLD,
            user_id=user_id
        )

        if not pairs:
            logger.info(f"No similar entity pairs for user {user_id}")
            return {"merged": 0, "deleted": 0, "kept": 0, "errors": 0}

        # 2. Group via BFS connected-components
        groups = self._build_merge_groups(pairs)
        logger.info(
            f"Found {len(pairs)} similar pairs → {len(groups)} groups "
            f"for user {user_id}"
        )

        # 3. Process synchronously
        stats = {"merged": 0, "deleted": 0, "kept": 0, "errors": 0}

        for batch_start in range(0, len(groups), GROUPS_PER_BATCH):
            batch = groups[batch_start:batch_start + GROUPS_PER_BATCH]
            batch_num = batch_start // GROUPS_PER_BATCH + 1
            total_batches = (len(groups) + GROUPS_PER_BATCH - 1) // GROUPS_PER_BATCH
            logger.info(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} groups)"
            )

            user_prompt, short_to_full = self._build_review_prompt(batch)

            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=self.gc_system_prompt,
                internal_llm='entity_gc',
                allow_negative=True,
            )
            response_text = self.llm_provider.extract_text_content(response)

            parsed_groups = self.parse_gc_response(response_text, short_to_full)
            batch_stats = self.execute_gc_decisions(parsed_groups, user_id)

            for key in stats:
                stats[key] += batch_stats[key]

        logger.info(
            f"Entity GC complete for user {user_id}: "
            f"{stats['merged']} merged, {stats['deleted']} deleted, "
            f"{stats['kept']} kept, {stats['errors']} errors"
        )
        return stats

    def cleanup(self) -> None:
        """No-op: Dependencies managed by factory lifecycle."""
        logger.debug("EntityGCService cleanup completed (no-op)")
