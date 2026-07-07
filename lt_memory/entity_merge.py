"""
Entity merge service — background LLM-driven deduplication of entities.

Over time, entities like "Round Lake" / "Round Lake, IL" / "round lake" accumulate
as separate rows across sessions (get_or_create_entity only fuzzy-dedups at
creation, never after). This service finds similar entity pairs via pg_trgm,
groups them by connected components, asks a single LLM call to judge which are
the same entity, and merges the duplicates (rewriting memory entity_links to
point at the canonical, bumping its link_count, archiving the losers).

Runs infrequently as a use-day-gated scheduled job. No agent tools — a plain LLM
call presenting candidate groups and parsing JSON output. The LLM only decides
merges; execution is deterministic (merge_entities) and per-user (RLS-scoped).
"""
import json
import logging
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)

# pg_trgm similarity threshold for candidate pair discovery. Calibrated on the
# old entity_gc service's production data (887 entities → ~408 pairs at 0.6).
ENTITY_MERGE_SIMILARITY_THRESHOLD = 0.6

# Max groups presented to the LLM per call (bounds prompt size). Surplus groups
# are re-found on the next run — merging shrinks the candidate pool each pass.
GROUPS_PER_CALL = 25


def _format_entity_id(uuid_str: str) -> str:
    """8-char short ID for LLM prompts (same pattern as format_memory_id)."""
    return uuid_str.replace('-', '')[:8]


def _build_merge_groups(pairs: List[Dict]) -> List[List[Dict]]:
    """BFS connected-components over entity pairs → list of groups.

    Each group is a list of {id, name, type, link_count} dicts for entities the
    trigram join connected. The LLM judges whether each group's members are the
    same entity.
    """
    graph: Dict[str, set] = defaultdict(set)
    entity_info: Dict[str, Dict] = {}

    for pair in pairs:
        id_a, id_b = str(pair['id_a']), str(pair['id_b'])
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
    groups: List[List[Dict]] = []
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


def _build_groups_prompt(groups: List[List[Dict]]) -> Tuple[str, Dict[str, UUID]]:
    """Render groups for the LLM user prompt. Returns (prompt_text, short_to_full)."""
    short_to_full: Dict[str, UUID] = {}
    group_blocks: List[str] = []

    for i, group in enumerate(groups, 1):
        entries: List[str] = []
        for entity in group:
            full_id = str(entity['id'])
            short_id = _format_entity_id(full_id)
            short_to_full[short_id] = UUID(full_id)
            entries.append(
                f'  {{"id": "{short_id}", "name": "{entity["name"]}", '
                f'"type": "{entity["type"]}", "links": {entity["link_count"]}}}'
            )
        group_blocks.append(f'Group {i}:\n' + '\n'.join(entries))

    from config.prompts.loader import load_prompt
    user_template = load_prompt("entity_merge_user.txt")
    return user_template.format(groups='\n\n'.join(group_blocks)), short_to_full


def _extract_json(text: str) -> Dict:
    """Robustly extract the first JSON object from an LLM response."""
    # Strip markdown code fences if present
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    # Otherwise find the first {...} block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")


def _parse_merge_response(
    response_text: str,
    short_to_full: Dict[str, UUID],
) -> List[Tuple[UUID, List[UUID]]]:
    """Parse the LLM JSON response into [(target_uuid, [source_uuid, ...])].

    Validates every short ID exists in the input (rejects hallucinated IDs) and
    skips any merge entry with an unknown canonical or empty merge list.
    """
    data = _extract_json(response_text)
    merges = data.get('merges', [])
    if not isinstance(merges, list):
        return []

    decisions: List[Tuple[UUID, List[UUID]]] = []
    for entry in merges:
        canonical_short = entry.get('canonical')
        merge_shorts = entry.get('merge', [])
        if not canonical_short or not isinstance(merge_shorts, list):
            continue
        canonical_uuid = short_to_full.get(canonical_short)
        if canonical_uuid is None:
            logger.warning("Entity merge: unknown canonical id %r, skipping", canonical_short)
            continue
        source_uuids: List[UUID] = []
        for short_id in merge_shorts:
            uuid = short_to_full.get(short_id)
            if uuid is None:
                logger.warning("Entity merge: unknown merge id %r, skipping it", short_id)
                continue
            source_uuids.append(uuid)
        if source_uuids:
            decisions.append((canonical_uuid, source_uuids))
    return decisions


def _assess_groups(
    llm,
    user_prompt: str,
    short_to_full: Dict[str, UUID],
) -> List[Tuple[UUID, List[UUID]]]:
    """Single LLM call presenting candidate groups → parsed merge decisions."""
    from config.prompts.loader import load_prompt
    system_prompt = load_prompt("entity_merge_system.txt")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = llm.generate_response(
        messages=messages,
        internal_llm='analysis',
        allow_negative=True,  # Background maintenance task
    )
    raw = llm.extract_text_content(response).strip()
    try:
        return _parse_merge_response(raw, short_to_full)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Entity merge: failed to parse LLM response: %r", raw[:300])
        return []


def run_entity_merge_for_user(user_id: str) -> Dict[str, int]:
    """Entry point: find candidate pairs → group → LLM assess → merge, for one user.

    Returns a stats dict. Best-effort: individual merge failures are logged and
    skipped; the caller (scheduled job) iterates users with set/clear user context.
    """
    from lt_memory.factory import get_lt_memory_factory
    from clients.llm_provider import LLMProvider

    factory = get_lt_memory_factory()
    db = factory.db

    pairs = db.find_similar_entity_pairs(
        ENTITY_MERGE_SIMILARITY_THRESHOLD, user_id=user_id
    )
    if not pairs:
        logger.info("Entity merge for user %s: no candidate pairs", user_id)
        return {"candidate_pairs": 0, "merged": 0}

    groups = _build_merge_groups(pairs)[:GROUPS_PER_CALL]
    user_prompt, short_to_full = _build_groups_prompt(groups)

    decisions = _assess_groups(LLMProvider(), user_prompt, short_to_full)
    if not decisions:
        logger.info(
            "Entity merge for user %s: LLM approved no merges (of %d groups)",
            user_id, len(groups),
        )
        return {"candidate_groups": len(groups), "merged": 0}

    merged = 0
    for target_id, source_ids in decisions:
        for source_id in source_ids:
            try:
                db.merge_entities(source_id, target_id, user_id=user_id)
                merged += 1
            except Exception:
                logger.exception(
                    "Entity merge failed: %s -> %s (user %s)", source_id, target_id, user_id
                )

    logger.info(
        "Entity merge for user %s: merged %d entities (of %d candidate groups)",
        user_id, merged, len(groups),
    )
    return {"candidate_groups": len(groups), "merged": merged}
