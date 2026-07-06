"""
Proactive memory surfacing for CNS integration.

Provides intelligent memory search using pre-computed subcortical embeddings,
hub-based entity discovery, and automatic inclusion of linked memories for
context enrichment.

The surfacing flow combines two retrieval paths:
1. Similarity Pool: Traditional hybrid search (BM25 + vector similarity)
2. Hub-Derived Pool: Entity-driven discovery (entity → memories → hubs → memories)

These pools are merged to provide comprehensive memory coverage.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import List, Optional, TYPE_CHECKING
import numpy as np

from lt_memory.db_access import LTMemoryDB
from lt_memory.linking import LinkingService
from lt_memory.models import Memory, MemoryDict, TraversalResult
from lt_memory.vector_ops import VectorOps

if TYPE_CHECKING:
    from lt_memory.hub_discovery import HubDiscoveryService

logger = logging.getLogger(__name__)

# Similarity and search
PROACTIVE_SIMILARITY_THRESHOLD = 0.42
PROACTIVE_MAX_LINK_TRAVERSAL_DEPTH = 3
PROACTIVE_MAX_MEMORIES = 10
MIN_IMPORTANCE_SCORE = 0.1
SEARCH_MAX_WORKERS = 2
SEARCH_OVERSAMPLE_FACTOR = 2

# Context window caps (cross-module: also imported by orchestrator, subcortical)
# MAX_SURFACED_MEMORIES is the authoritative hard cap on primary memories shown
# to the model/frontend each turn. The orchestrator enforces it after the
# pinned+fresh merge, so the ProactiveMemoryTrinket cache (and thus next-turn's
# subcortical retention input) is also capped to this count.
#
# Tuned for attention, not tokens: 15-20 memories dilute attention so each
# becomes noise; a small curated subset means each one earns integration into
# the response. The ranking pipeline (RRF + effective_score + link rerank)
# already does the curation — this cap just bounds how many survive to the
# window. 8 keeps the subset small while leaving room for pinned continuity plus
# fresh discovery. MAX_PINNED is intentionally below MAX_SURFACED so the pinned
# cap can bind and preserve MIN_FRESH slots.
MAX_SURFACED_MEMORIES = 8    # Hard cap on primary memories shown per turn
MAX_PINNED_MEMORIES = 6      # Max retained from the previous cache; also pressure warning trigger
MIN_FRESH_MEMORIES = 2       # Min fresh fetched; sliding budget = max(this, MAX_SURFACED - pinned)
MAX_LINKED_PER_PRIMARY = 2

# Assistant message embedding for memory surfacing
NUM_ASSISTANT_MESSAGES = 2
ASSISTANT_SIMILARITY_THRESHOLD = 0.75
MAX_ASSISTANT_MEMORIES = 3

# Debut boost: temporary ranking boost for new memories before they build hub connections
DEBUT_FULL_BOOST_DAYS = 7    # Full boost for days 0-6 (activity days, not calendar)
DEBUT_END_DAYS = 10           # Boost trails off completely by this day
DEBUT_BOOST_AMOUNT = 0.15
HUB_CONNECTION_THRESHOLD = 2  # Entity link count at which debut boost no longer applies

# Supersedes penalty for soft demotion of superseded memories
SUPERSEDES_PENALTY_MULTIPLIER = 0.3  # 0.3 = 70% reduction

# Link type weights for reranking (higher = more important)
LINK_WEIGHT_CONFLICTS = 1.0
LINK_WEIGHT_CORROBORATES = 0.9
LINK_WEIGHT_SUPERSEDES = 0.9
LINK_WEIGHT_REFINES = 0.8
LINK_WEIGHT_PRECEDES = 0.7
LINK_WEIGHT_CONTEXTUALIZES = 0.7
LINK_WEIGHT_EXEMPLIFIES = 0.75
LINK_WEIGHT_SHARES_ENTITY = 0.4
LINK_WEIGHT_DEFAULT = 0.5
# Importance inheritance: (linked * weight) + (primary * (1 - weight))
LINK_IMPORTANCE_INHERITANCE_WEIGHT = 0.7


class ProactiveService:
    """
    Service for proactive memory surfacing in conversations.

    Finds relevant memories using pre-computed subcortical embeddings,
    hub-based entity discovery, and automatically includes linked memories
    for richer context.
    """

    def __init__(
        self,
        vector_ops: VectorOps,
        linking_service: LinkingService,
        db: LTMemoryDB,
        hub_discovery: 'HubDiscoveryService'
    ):
        self.vector_ops = vector_ops
        self.linking = linking_service
        self.db = db
        self.hub_discovery = hub_discovery

    def search_with_embedding(
        self,
        embedding: np.ndarray,
        query_expansion: str,
        limit: Optional[int] = None,
        extracted_entities: Optional[List[str]] = None
    ) -> List[MemoryDict]:
        """
        Search for relevant memories using expansion embedding and hub discovery.

        Combines two retrieval paths:
        1. Similarity Pool: Hybrid search (BM25 + vector similarity)
        2. Hub-Derived Pool: Entity-driven discovery via hub navigation

        Args:
            embedding: Pre-computed 768d expansion embedding
            query_expansion: Expanded query text (for BM25 and reranking)
            limit: Maximum number of memories to return
            extracted_entities: Entity names from subcortical layer for hub discovery

        Returns:
            List of relevant memory dictionaries with metadata

        Raises:
            Exception: If search operations fail
        """
        if limit is None:
            limit = PROACTIVE_MAX_MEMORIES

        # Run both retrieval paths in parallel
        # Each thread gets its own DB connection and sets app.current_user_id via RLS
        # Each thread needs its OWN context copy - a single context object cannot be
        # entered by multiple threads simultaneously (causes "already entered" error)

        def _fetch_similarity_pool():
            """Fetch similarity-based memories via hybrid search."""
            return self.vector_ops.hybrid_search(
                query_text=query_expansion,
                query_embedding=embedding,
                search_intent="general",
                limit=limit * SEARCH_OVERSAMPLE_FACTOR,
                similarity_threshold=PROACTIVE_SIMILARITY_THRESHOLD,
                min_importance=MIN_IMPORTANCE_SCORE
            )

        def _fetch_hub_pool():
            """Fetch hub-derived memories via entity navigation."""
            if not extracted_entities:
                return []
            return self.hub_discovery.discover_hub_memories(
                extracted_entities=extracted_entities,
                expansion_embedding=embedding
            )

        # Execute both searches in parallel - each thread gets its own context copy
        with ThreadPoolExecutor(max_workers=SEARCH_MAX_WORKERS) as executor:
            ctx_similarity = copy_context()
            ctx_hub = copy_context()
            similarity_future = executor.submit(ctx_similarity.run, _fetch_similarity_pool)
            hub_future = executor.submit(ctx_hub.run, _fetch_hub_pool)

            # Wait for both to complete - exceptions will propagate on .result()
            similarity_pool = similarity_future.result()
            hub_pool = hub_future.result()

        # Filter similarity pool by minimum importance score
        similarity_pool = [
            memory for memory in similarity_pool
            if memory.importance_score >= MIN_IMPORTANCE_SCORE
        ]

        # Merge pools (deduplicate, similarity pool takes precedence for scores)
        merged_results = self._merge_memory_pools(similarity_pool, hub_pool)

        if not merged_results:
            logger.debug("No relevant memories found")
            return []

        # Include linked memories for context enrichment
        # (Hub chain already provides some traversal, but this adds explicit link metadata)
        expanded_results = self._include_linked_memories(merged_results[:limit])

        # Rerank and filter linked memories by type and importance
        reranked_results = self._rerank_with_links(expanded_results)
        final_results = reranked_results[:limit]

        logger.info(
            f"Found {len(final_results)} relevant memories "
            f"(similarity: {len(similarity_pool)}, hub: {len(hub_pool)})"
        )

        # Track access for retrieved memories to update importance scores
        self._track_memory_access(final_results)

        return [self._memory_to_dict(m) for m in final_results]

    def _merge_memory_pools(
        self,
        similarity_pool: List[Memory],
        hub_pool: List[Memory]
    ) -> List[Memory]:
        """
        Merge similarity and hub-derived pools with debut boost for new memories.

        New memories (< 10 activity days old) with few entity links receive a
        ranking boost to help them surface before building hub connections.
        Boost is full for days 0-6, then trails off linearly days 7-10.

        Args:
            similarity_pool: Memories from hybrid search
            hub_pool: Memories from hub discovery

        Returns:
            Merged, deduplicated list of Memory objects sorted by effective score
        """
        # Debut boost from config
        debut_full_boost_days = DEBUT_FULL_BOOST_DAYS
        debut_end_days = DEBUT_END_DAYS
        debut_boost = DEBUT_BOOST_AMOUNT
        hub_connection_threshold = HUB_CONNECTION_THRESHOLD

        # Get user's current activity days for vacation-proof age calculation
        from utils.user_context import get_user_cumulative_activity_days
        current_activity_days = get_user_cumulative_activity_days()

        merged = list(similarity_pool)
        seen_ids = {m.id for m in similarity_pool}

        for memory in hub_pool:
            if memory.id not in seen_ids:
                merged.append(memory)
                seen_ids.add(memory.id)

        # Calculate effective scores with debut boost and supersedes penalty
        def effective_score(memory) -> float:
            score = memory.importance_score

            # Global memories skip debut boost (they're curated, not new user memories)
            is_global = getattr(memory, 'source', 'personal') == 'global'

            # Debut boost logic (apply to score, not early return)
            if not is_global and memory.activity_days_at_creation is not None:
                age_in_days = current_activity_days - memory.activity_days_at_creation
                entity_link_count = len(memory.entity_links) if memory.entity_links else 0

                if entity_link_count < hub_connection_threshold:
                    if age_in_days < debut_full_boost_days:
                        score += debut_boost
                    elif age_in_days <= debut_end_days:
                        remaining = debut_end_days - age_in_days
                        trailoff_window = debut_end_days - debut_full_boost_days
                        score += debut_boost * (remaining / trailoff_window)

            # Supersedes penalty (soft demotion for superseded memories)
            if memory.inbound_links:
                has_supersedes = any(
                    link.get('type') == 'supersedes'
                    for link in memory.inbound_links
                )
                if has_supersedes:
                    score *= SUPERSEDES_PENALTY_MULTIPLIER
                    logger.debug(f"Supersedes penalty for memory {memory.id}")

            return score

        # Sort by effective score
        merged.sort(key=effective_score, reverse=True)
        return merged

    def _track_memory_access(self, memories: List[Memory]) -> None:
        """
        Track access for retrieved memories to update importance scores.

        Updates access_count, last_accessed, and recalculates importance
        scores for each retrieved memory. Only tracks primary memories,
        not linked memories (which are secondary context).

        Args:
            memories: List of Memory objects that were retrieved
        """
        for memory in memories:
            try:
                self.db.update_access_stats(memory.id)
            except Exception as e:
                # Log but don't fail the search - access tracking is enhancement
                logger.warning(
                    f"Failed to update access stats for memory {memory.id}: {e}"
                )

    def _include_linked_memories(
        self,
        primary_memories: List[Memory]
    ) -> List[Memory]:
        """
        Include memories linked to primary search results.

        Traverses memory graph and attaches related memories as children
        of primary memories, preserving link metadata.

        Args:
            primary_memories: List of primary memory search results

        Returns:
            List of primary memories with linked_memories populated
        """
        if not primary_memories:
            return []

        for primary_memory in primary_memories:
            linked_with_metadata = self.linking.traverse_related(
                memory_id=primary_memory.id,
                depth=PROACTIVE_MAX_LINK_TRAVERSAL_DEPTH
            )

            primary_memory.linked_memories = []

            for linked_data in linked_with_metadata:
                linked_memory = linked_data["memory"]
                linked_memory.link_metadata = {
                    "link_type": linked_data["link_type"],
                    "reasoning": linked_data["reasoning"],
                    "depth": linked_data["depth"],
                    "linked_from_id": linked_data["linked_from_id"]
                }
                primary_memory.linked_memories.append(linked_memory)

            logger.debug(
                f"Attached {len(primary_memory.linked_memories)} linked memories "
                f"to primary memory {primary_memory.id}"
            )

        return primary_memories

    def _rerank_with_links(self, primary_memories: List[Memory]) -> List[Memory]:
        """
        Rerank and filter memories considering link types and importance.

        Ranking formula: type_weight × inherited_importance
        """
        # Link type weights from config
        link_type_weights = {
            "conflicts": LINK_WEIGHT_CONFLICTS,
            "corroborates": LINK_WEIGHT_CORROBORATES,
            "supersedes": LINK_WEIGHT_SUPERSEDES,
            "refines": LINK_WEIGHT_REFINES,
            "precedes": LINK_WEIGHT_PRECEDES,
            "contextualizes": LINK_WEIGHT_CONTEXTUALIZES,
            "exemplifies": LINK_WEIGHT_EXEMPLIFIES,
            "shares_entity": LINK_WEIGHT_SHARES_ENTITY,
        }
        default_weight = LINK_WEIGHT_DEFAULT
        inheritance_weight = LINK_IMPORTANCE_INHERITANCE_WEIGHT

        primary_ids = {m.id for m in primary_memories}

        for primary_memory in primary_memories:
            if not hasattr(primary_memory, 'linked_memories'):
                continue

            linked_memories = primary_memory.linked_memories
            if not linked_memories:
                continue

            scored_linked = []

            for linked in linked_memories:
                # Deduplication
                if linked.id in primary_ids:
                    continue

                link_meta = getattr(linked, 'link_metadata', {})
                link_type = link_meta.get('link_type', 'unknown')

                # Type-based weighting
                type_weight = default_weight
                for known_type, weight in link_type_weights.items():
                    if link_type == known_type or link_type.startswith(f"{known_type}:"):
                        type_weight = weight
                        break

                # Importance inheritance: (linked * weight) + (primary * (1 - weight))
                linked_importance = getattr(linked, 'importance_score', 0.5)
                primary_importance = getattr(primary_memory, 'importance_score', 0.5)
                inherited_importance = (linked_importance * inheritance_weight) + (primary_importance * (1 - inheritance_weight))

                final_score = type_weight * inherited_importance
                scored_linked.append((linked, final_score))

            scored_linked.sort(key=lambda x: x[1], reverse=True)
            max_linked = MAX_LINKED_PER_PRIMARY
            primary_memory.linked_memories = [linked for linked, score in scored_linked[:max_linked]]

        return primary_memories

    def _memory_to_dict(self, memory: Memory) -> MemoryDict:
        """Convert Memory model to dictionary with hierarchical structure."""
        result = {
            "id": str(memory.id),
            "text": memory.text,
            "importance_score": memory.importance_score,
            "similarity_score": memory.similarity_score,  # Sigmoid-normalized RRF score (0-1)
            "vector_similarity": getattr(memory, '_vector_similarity', None),  # Raw cosine similarity
            "_raw_rrf_score": getattr(memory, '_raw_rrf_score', None),  # Raw RRF before sigmoid
            "source": getattr(memory, 'source', 'personal'),  # 'personal' or 'global'
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "access_count": memory.access_count,
            "happens_at": memory.happens_at.isoformat() if memory.happens_at else None,
            "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
            "inbound_links": memory.inbound_links if hasattr(memory, 'inbound_links') else [],
            "outbound_links": memory.outbound_links if hasattr(memory, 'outbound_links') else [],
            # Entity links for self-improving loop - shows which hubs this memory connects to
            "entity_links": memory.entity_links if hasattr(memory, 'entity_links') else [],
            # Annotations for contextual notes
            "annotations": memory.annotations if hasattr(memory, 'annotations') else [],
        }

        if hasattr(memory, 'link_metadata'):
            result['link_metadata'] = memory.link_metadata

        if hasattr(memory, 'linked_memories') and memory.linked_memories:
            result['linked_memories'] = [
                self._memory_to_dict(linked)
                for linked in memory.linked_memories
            ]
        else:
            result['linked_memories'] = []

        return result
