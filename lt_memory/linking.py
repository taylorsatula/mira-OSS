"""
Relationship discovery and link management for LT_Memory system.

Handles finding semantically related memories, classifying relationship types,
and creating bidirectional links in the memory graph. Supports both synchronous
link creation and batch classification payload building.

Discovery uses three axes:
1. Vector similarity — embedding cosine distance
2. Entity co-occurrence — shared entities filtered by embedding similarity floor
3. TF-IDF term overlap — catches orphan memories the embedding model smooths over
"""
import logging
from typing import List, Optional
from uuid import UUID

from lt_memory.models import Memory, TraversalResult
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from utils.tag_parser import format_memory_id
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)

# Link discovery thresholds
SIMILARITY_THRESHOLD_FOR_LINKING = 0.75
MAX_CANDIDATES_PER_MEMORY = 20
MAX_LINK_TRAVERSAL_DEPTH = 3
CLASSIFICATION_MAX_TOKENS = 500
ENTITY_SIMILARITY_FLOOR = 0.55     # pg_trgm floor for entity co-occurrence candidates
TFIDF_SIMILARITY_THRESHOLD = 0.20  # TF-IDF cosine floor for term-based discovery
TFIDF_MAX_CANDIDATES = 10


class _TfidfState:
    """TF-IDF matrix state scoped to a single user."""

    def __init__(self, vectorizer, matrix, memory_ids: List[UUID], memory_count: int):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.memory_ids = memory_ids
        self.memory_count = memory_count


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5 or 1e-8
    norm_b = sum(b * b for b in vec_b) ** 0.5 or 1e-8
    return dot / (norm_a * norm_b)


class LinkingService:
    """
    Service for discovering and managing memory relationships.

    Provides:
    - Similarity-based candidate discovery
    - Relationship classification (sync or batch payload building)
    - Bidirectional link creation and management
    - Link traversal for graph navigation
    """

    def __init__(
        self,
        vector_ops: VectorOps,
        db: LTMemoryDB
    ):
        self.vector_ops = vector_ops
        self.db = db

        self._tfidf_states: dict[str, _TfidfState] = {}

    def find_candidate_hints(
        self,
        memory_id: UUID
    ) -> List[dict]:
        """
        Typeless candidate discovery for the MemoryCuratorAgent.

        Runs three discovery axes (vector similarity, entity co-occurrence,
        TF-IDF term overlap) and returns CandidateRef-shaped dicts (short id + discovery_signal +
        similarity) with NO typed relationship classification. The agent
        decides how — or whether — each candidate relates.

        This is the programmatic half of the curation delineation:
        deterministic code finds *that* memories touch; the agent classifies
        *how*. No links are written here.

        Args:
            memory_id: Source memory UUID

        Returns:
            List of candidate dicts: {memory_id, bond, discovery_signal,
            similarity}. bond is always "" here (extraction bonds are merged
            in by the caller, not discovered by this service).
        """
        vector_candidates = self.vector_ops.find_similar_to_memory(
            memory_id=memory_id,
            limit=MAX_CANDIDATES_PER_MEMORY,
            similarity_threshold=SIMILARITY_THRESHOLD_FOR_LINKING,
            min_importance=0.001
        )
        entity_candidates = self._find_entity_candidates(memory_id)
        tfidf_candidates = self._find_tfidf_candidates(memory_id)

        seen_ids = set()
        hints: List[dict] = []

        for mem in vector_candidates:
            if mem.id not in seen_ids and mem.id != memory_id:
                seen_ids.add(mem.id)
                hints.append({
                    "memory_id": format_memory_id(str(mem.id)),
                    "bond": "",
                    "discovery_signal": "vector",
                    "similarity": getattr(mem, "similarity_score", None),
                })
        for mem in entity_candidates:
            if mem.id not in seen_ids and mem.id != memory_id:
                seen_ids.add(mem.id)
                hints.append({
                    "memory_id": format_memory_id(str(mem.id)),
                    "bond": "",
                    "discovery_signal": "entity",
                    "similarity": None,
                })
        for mem in tfidf_candidates:
            if mem.id not in seen_ids and mem.id != memory_id:
                seen_ids.add(mem.id)
                hints.append({
                    "memory_id": format_memory_id(str(mem.id)),
                    "bond": "",
                    "discovery_signal": "tfidf",
                    "similarity": None,
                })

        logger.debug(
            f"find_candidate_hints: {len(hints)} candidates for memory "
            f"{memory_id} (vector={len(vector_candidates)}, "
            f"entity={len(entity_candidates)}, tfidf={len(tfidf_candidates)})"
        )
        return hints

    def _find_entity_candidates(
        self,
        memory_id: UUID
    ) -> List[Memory]:
        """
        Find memories that share entities with the source memory.

        For each entity linked to the source memory, retrieves other memories
        also linked to that entity, filtered by embedding similarity floor to
        suppress O(N²) noise from common entities like "MIRA".

        Args:
            memory_id: Source memory UUID

        Returns:
            List of candidate Memory objects (excludes source, cold storage,
            and candidates below entity_similarity_floor)
        """
        source_memory = self.db.get_memory(memory_id)
        if not source_memory or not source_memory.entity_links:
            return []

        source_embedding = source_memory.embedding
        if not source_embedding:
            return []

        seen_ids = {memory_id}  # Exclude source
        candidates = []
        floor = ENTITY_SIMILARITY_FLOOR

        for entity_link in source_memory.entity_links:
            entity_id = entity_link.get("uuid")
            if not entity_id:
                continue

            try:
                entity_uuid = UUID(entity_id)
            except ValueError:
                logger.warning(f"Malformed entity UUID in entity_links: {entity_id}")
                continue

            co_occurring = self.db.get_memories_for_entity(entity_uuid)

            for mem in co_occurring:
                if mem.id in seen_ids:
                    continue
                if mem.importance_score is not None and mem.importance_score <= 0.0:
                    continue  # Skip cold storage
                # Similarity floor: suppress noise from common entities
                if floor > 0 and mem.embedding:
                    if _cosine_similarity(source_embedding, mem.embedding) < floor:
                        continue
                seen_ids.add(mem.id)
                candidates.append(mem)

        return candidates

    def _ensure_tfidf(self) -> _TfidfState:
        """Rebuild TF-IDF matrix for the current user if stale or uninitialized."""
        user_id = get_current_user_id()
        state = self._tfidf_states.get(user_id)

        memories = self.db.get_all_memories()
        active = [
            m for m in memories
            if m.importance_score and m.importance_score > 0
            and m.embedding is not None
            and not m.is_archived
        ]

        if state is not None and len(active) == state.memory_count:
            return state  # still fresh

        from sklearn.feature_extraction.text import TfidfVectorizer

        memory_ids = [m.id for m in active]
        texts = [m.text for m in active]

        vectorizer = TfidfVectorizer(
            max_features=10000, stop_words='english', min_df=2, max_df=0.8
        )
        matrix = vectorizer.fit_transform(texts)

        state = _TfidfState(
            vectorizer=vectorizer,
            matrix=matrix,
            memory_ids=memory_ids,
            memory_count=len(active),
        )
        self._tfidf_states[user_id] = state
        logger.info(f"Rebuilt TF-IDF matrix: {len(active)} memories, {len(vectorizer.vocabulary_)} terms")
        return state

    def _find_tfidf_candidates(
        self,
        memory_id: UUID
    ) -> List[Memory]:
        """
        Find candidate memories via TF-IDF term overlap.

        Rescues orphan memories (no entities, distant embeddings) that share
        rare terms the embedding model smooths over — e.g., wine preference
        and specific bottle, water filtration research and system plan.

        Args:
            memory_id: Source memory UUID

        Returns:
            List of candidate Memory objects above tfidf_similarity_threshold,
            capped at tfidf_max_candidates
        """
        source_memory = self.db.get_memory(memory_id)
        if not source_memory:
            return []

        state = self._ensure_tfidf()

        if state.vectorizer is None or state.matrix is None:
            return []

        # Transform source text against fitted vocabulary
        source_vector = state.vectorizer.transform([source_memory.text])

        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(source_vector, state.matrix).flatten()

        # Collect candidates above threshold, excluding source
        threshold = TFIDF_SIMILARITY_THRESHOLD
        max_candidates = TFIDF_MAX_CANDIDATES
        scored = []

        for idx, sim in enumerate(similarities):
            mid = state.memory_ids[idx]
            if mid == memory_id:
                continue
            if sim >= threshold:
                scored.append((sim, mid))

        # Sort by similarity descending, cap at max
        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [mid for _, mid in scored[:max_candidates]]

        if not top_ids:
            return []

        # Batch-fetch Memory objects (RLS-filtered by user context)
        return self.db.get_memories_by_ids(top_ids)

    def traverse_related(
        self,
        memory_id: UUID,
        depth: Optional[int] = None
    ) -> List[TraversalResult]:
        """
        Traverse memory graph from starting memory with link metadata.

        Follows outbound links up to specified depth, collecting related memories
        with their link information (type, reasoning) and hierarchical
        position preserved for display.

        Args:
            memory_id: Starting memory UUID
            depth: Maximum traversal depth (uses config default if None)

        Returns:
            List of dicts with Memory object and link metadata:
            [
                {
                    "memory": Memory,
                    "link_type": str,
                    "reasoning": str,
                    "depth": int,
                    "linked_from_id": UUID
                },
                ...
            ]
        """
        if depth is None:
            depth = MAX_LINK_TRAVERSAL_DEPTH

        if depth < 1:
            return []

        visited_ids = {memory_id}
        current_level = [(memory_id, None, 0)]  # (uuid, link_metadata, depth)
        all_related = []

        for current_depth in range(1, depth + 1):
            if not current_level:
                break

            # Get UUIDs for this level
            level_uuids = [item[0] for item in current_level]
            current_memories = self.db.get_memories_by_ids(level_uuids)

            # Heal-on-read: detect and remove dead links
            found_memory_ids = {m.id for m in current_memories}
            dead_links = [uuid for uuid in level_uuids if uuid not in found_memory_ids]

            if dead_links:
                removed_count = self.db.remove_dead_links(dead_links)
                if removed_count > 0:
                    logger.info(
                        f"Heal-on-read removed {removed_count} dead link references "
                        f"for {len(dead_links)} UUIDs during traversal"
                    )

            # Build memory lookup
            memory_lookup = {m.id: m for m in current_memories}

            # Process current level and extract next level
            next_level = []
            for uuid, link_meta, depth_level in current_level:
                memory = memory_lookup.get(uuid)
                if not memory:
                    continue

                # Add to results (skip starting memory)
                if uuid != memory_id:
                    all_related.append({
                        "memory": memory,
                        "link_type": link_meta.get("type") if link_meta else None,
                        "reasoning": link_meta.get("reasoning") if link_meta else None,
                        "bond": link_meta.get("bond") if link_meta else None,
                        "depth": depth_level,
                        "linked_from_id": link_meta.get("source_id") if link_meta else None
                    })

                # Extract outbound links for next level
                for link in memory.outbound_links:
                    target_uuid = UUID(link["uuid"])

                    if target_uuid not in visited_ids:
                        visited_ids.add(target_uuid)
                        next_level.append((
                            target_uuid,
                            {
                                "type": link.get("type"),
                                "reasoning": link.get("reasoning"),
                                "bond": link.get("extraction_bond"),
                                "source_id": uuid
                            },
                            current_depth
                        ))

            current_level = next_level

        return all_related

    def cleanup(self) -> None:
        """
        Clean up resources.

        No-op: Dependencies managed by factory lifecycle.
        Nulling references breaks in-flight scheduler jobs.
        """
        logger.debug("LinkingService cleanup completed (no-op)")
