"""
Memory refinement service for LT_Memory system.

Handles consolidation clustering: identifying and merging redundant similar memories
via connected-components graph analysis and LLM-driven merge decisions.
"""
import logging
from collections import defaultdict, deque
from typing import List, Optional, Set
from uuid import UUID

from lt_memory.models import (
    Memory,
    ConsolidationCluster,
    ConsolidationPayload,
)
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB

logger = logging.getLogger(__name__)

# Consolidation tuning
CONSOLIDATION_SIMILARITY_THRESHOLD = 0.85
MIN_CLUSTER_SIZE = 2
MAX_CONSOLIDATION_REJECTION_COUNT = 3


class RefinementService:
    """
    Service for refining memories through consolidation.

    Provides:
    - Consolidation cluster discovery via connected-components
    - Prompt building for batch consolidation
    """

    def __init__(
        self,
        vector_ops: VectorOps,
        db: LTMemoryDB
    ):
        self.vector_ops = vector_ops
        self.db = db
        self._load_prompts()

    def _load_prompts(self) -> None:
        """
        Load consolidation prompt.

        Raises:
            FileNotFoundError: If required prompt file not found (prompts are required configuration)
        """
        from config.prompts.loader import load_prompt
        self.consolidation_system_prompt = load_prompt("memory_consolidation_system.txt")
        logger.info("Loaded consolidation prompt")

    def identify_consolidation_clusters(
        self,
        min_cluster_size: Optional[int] = None
    ) -> List[ConsolidationCluster]:
        """
        Identify clusters of similar memories via full vector sweep + connected components.

        Builds a similarity graph across all eligible memories (not yet at max rejection
        count), then extracts connected components via BFS. Each component becomes a
        consolidation cluster sent to the LLM for merge decisions.

        Args:
            min_cluster_size: Minimum memories in cluster (default from config)

        Returns:
            List of ConsolidationCluster objects
        """
        if min_cluster_size is None:
            min_cluster_size = MIN_CLUSTER_SIZE

        all_memories = self.db.get_all_memories(include_archived=False)
        eligible = [
            m for m in all_memories
            if m.consolidation_rejection_count < MAX_CONSOLIDATION_REJECTION_COUNT
        ]
        eligible_ids = {m.id for m in eligible}

        # Build similarity graph using pre-loaded embeddings (avoids N+1 get_memory calls)
        graph: dict[UUID, set[UUID]] = defaultdict(set)
        for memory in eligible:
            if not memory.embedding:
                continue
            neighbors = self.vector_ops.find_similar_by_embedding(
                query_embedding=memory.embedding,
                limit=21,  # +1 to account for self-match
                similarity_threshold=CONSOLIDATION_SIMILARITY_THRESHOLD,
                min_importance=0.001
            )
            for neighbor in neighbors:
                if neighbor.id != memory.id and neighbor.id in eligible_ids:
                    graph[memory.id].add(neighbor.id)
                    graph[neighbor.id].add(memory.id)

        # Extract connected components via BFS
        visited: Set[UUID] = set()
        components: List[Set[UUID]] = []
        for node in graph:
            if node in visited:
                continue
            component: Set[UUID] = set()
            queue = deque([node])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                queue.extend(graph[current] - visited)
            if len(component) >= min_cluster_size:
                components.append(component)

        # Build ConsolidationCluster objects
        memory_lookup = {m.id: m for m in eligible}
        clusters = []
        for i, component in enumerate(components):
            if len(component) > 20:
                logger.warning(
                    f"Large consolidation cluster: {len(component)} memories in component_{i}. "
                    f"May indicate similarity threshold is too low or a broad topic cluster."
                )
            mem_ids = list(component)
            mem_texts = [memory_lookup[mid].text for mid in mem_ids]
            clusters.append(ConsolidationCluster(
                cluster_id=f"component_{i}",
                memory_ids=mem_ids,
                memory_texts=mem_texts,
            ))

        logger.info(
            f"Found {len(clusters)} consolidation clusters from "
            f"{len(eligible)} eligible memories ({len(all_memories)} total)"
        )
        return clusters

    def build_consolidation_payload(
        self,
        cluster: ConsolidationCluster,
        full_memories: Optional[List[Memory]] = None
    ) -> ConsolidationPayload:
        """
        Build consolidation analysis payload for batch API.

        Args:
            cluster: ConsolidationCluster to analyze
            full_memories: Optional pre-loaded Memory objects (if None, loads from DB)

        Returns:
            ConsolidationPayload with prompt and parameters

        Raises:
            ValueError: If memories cannot be loaded (fail-fast, don't proceed with incomplete data)
        """
        from lt_memory.memory_formatter import format_memories_for_consolidation

        # Load full memories if not provided (for annotation/link context)
        if full_memories is None:
            full_memories = self.db.get_memories_by_ids(cluster.memory_ids)

        # Fail-fast: don't proceed with consolidation if memories can't be loaded
        if not full_memories:
            raise ValueError(
                f"Cannot build consolidation payload: memories not found for cluster {cluster.cluster_id}. "
                f"Memory IDs: {cluster.memory_ids}"
            )

        # Format memories with annotations and links
        memories_text = format_memories_for_consolidation(
            full_memories,
            include_annotations=True,
            include_links=True
        )

        user_prompt = f"""Analyze these memories and identify which ones are saying the same thing in different words:

{memories_text}

Respond with JSON:
{{
    "merge_groups": [
        {{
            "memory_ids": ["mem_XXXXXXXX", "mem_YYYYYYYY"],
            "merged_text": "The consolidated text preserving all details",
            "reason": "Brief explanation"
        }}
    ],
    "independent_ids": ["mem_ZZZZZZZZ"],
    "summary": "Brief overall explanation"
}}"""

        return {
            "cluster_id": cluster.cluster_id,
            "memory_ids": [str(mid) for mid in cluster.memory_ids],
            "system_prompt": self.consolidation_system_prompt,  # From file
            "user_prompt": user_prompt  # Data with context
        }

    def cleanup(self) -> None:
        """
        Clean up resources.

        No-op: Dependencies managed by factory lifecycle.
        Nulling references breaks in-flight scheduler jobs.
        """
        logger.debug("RefinementService cleanup completed (no-op)")
