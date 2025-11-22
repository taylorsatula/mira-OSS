"""
Consolidation handler - memory consolidation with link transfer.

Handles consolidating multiple memories into one while preserving all
relationship information through link bundle transfer, source memory
outbound link rewriting, and proper archival of consolidated memories.
"""
import logging
import statistics
from typing import List, Dict, Any
from uuid import UUID

from lt_memory.models import ExtractedMemory
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB

logger = logging.getLogger(__name__)


class ConsolidationHandler:
    """
    Handle memory consolidation with link bundle transfer.

    Single Responsibility: Consolidate multiple memories into one,
    preserving all relationship information.

    Pure business logic - no I/O decisions or orchestration.
    """

    def __init__(self, vector_ops: VectorOps, db: LTMemoryDB):
        """
        Initialize consolidation handler.

        Args:
            vector_ops: Vector operations for memory storage
            db: Database access for memory queries and updates
        """
        self.vector_ops = vector_ops
        self.db = db

    def execute_consolidation(
        self,
        old_memory_ids: List[UUID],
        consolidated_text: str,
        confidence: float,
        user_id: str
    ) -> UUID:
        """
        Consolidate multiple memories into one with link preservation.

        Creates new consolidated memory with median importance from source
        memories, transfers all link bundles (inbound, outbound, entity),
        rewrites source memory references to point to new memory, and
        archives old memories.

        Args:
            old_memory_ids: Memory UUIDs to consolidate
            consolidated_text: Text of consolidated memory
            confidence: Confidence score for consolidation
            user_id: User ID

        Returns:
            UUID of newly created consolidated memory

        Raises:
            ValueError: If old memories cannot be loaded
            RuntimeError: If consolidated memory storage fails
        """
        # Step 1: Load old memories
        old_memories = self.db.get_memories_by_ids(old_memory_ids, user_id=user_id)

        # Validate ALL requested memories were found (not just some)
        if len(old_memories) != len(old_memory_ids):
            found_ids = {m.id for m in old_memories}
            missing_ids = set(old_memory_ids) - found_ids
            raise ValueError(
                f"Failed to load {len(missing_ids)} of {len(old_memory_ids)} memories for consolidation. "
                f"Missing IDs: {missing_ids}. Cannot consolidate with incomplete memory set."
            )

        # Step 2: Calculate median importance from old memories
        importance_scores = [m.importance_score for m in old_memories if m.importance_score]
        median_importance = statistics.median(importance_scores) if importance_scores else 0.5

        logger.debug(
            f"Consolidating {len(old_memories)} memories with median importance {median_importance:.3f}"
        )

        # Step 3: Collect all unique links from old memories
        all_inbound_links = []
        all_outbound_links = []
        all_entity_links = []

        for memory in old_memories:
            all_inbound_links.extend(memory.inbound_links)
            all_outbound_links.extend(memory.outbound_links)
            all_entity_links.extend(memory.entity_links)

        # Step 4: Deduplicate links (exclude self-references, keep highest confidence)
        old_memory_id_strs = {str(mid) for mid in old_memory_ids}

        # Deduplicate inbound links by UUID, keeping link with highest confidence
        unique_inbound = {}
        for link in all_inbound_links:
            uuid = link['uuid']
            if uuid in old_memory_id_strs:
                continue  # Skip self-references to old memories
            if uuid not in unique_inbound:
                unique_inbound[uuid] = link
            else:
                # Keep link with higher confidence
                if link.get('confidence', 0) > unique_inbound[uuid].get('confidence', 0):
                    unique_inbound[uuid] = link

        # Deduplicate outbound links by UUID, keeping link with highest confidence
        unique_outbound = {}
        for link in all_outbound_links:
            uuid = link['uuid']
            if uuid in old_memory_id_strs:
                continue
            if uuid not in unique_outbound:
                unique_outbound[uuid] = link
            else:
                # Keep link with higher confidence
                if link.get('confidence', 0) > unique_outbound[uuid].get('confidence', 0):
                    unique_outbound[uuid] = link

        # Deduplicate entity links by UUID (entity links don't have confidence scores)
        unique_entities = {
            link['uuid']: link for link in all_entity_links
        }

        logger.debug(
            f"Collected link bundles: {len(unique_inbound)} inbound, "
            f"{len(unique_outbound)} outbound, {len(unique_entities)} entities"
        )

        # Step 5: Create consolidated memory with median importance
        consolidated_memory = ExtractedMemory(
            text=consolidated_text,
            importance_score=median_importance,
            confidence=confidence,
            consolidates_memory_ids=[str(mid) for mid in old_memory_ids]
        )

        # Step 6: Store consolidated memory with embeddings
        new_ids = self.vector_ops.store_memories_with_embeddings(
            [consolidated_memory]
        )

        if not new_ids:
            raise RuntimeError("Failed to store consolidated memory")

        new_memory_id = new_ids[0]
        new_memory_id_str = str(new_memory_id)

        # Step 7: Transfer link bundles to new consolidated memory
        if unique_inbound or unique_outbound or unique_entities:
            self.db.update_memory(new_memory_id, {
                'inbound_links': list(unique_inbound.values()),
                'outbound_links': list(unique_outbound.values()),
                'entity_links': list(unique_entities.values())
            }, user_id=user_id)

            logger.info(
                f"Transferred links to consolidated memory {new_memory_id}: "
                f"{len(unique_inbound)} inbound, {len(unique_outbound)} outbound, "
                f"{len(unique_entities)} entities"
            )

        # Step 8: Update all memories that were linking TO old memories
        # Rewrite their outbound_links to point to new_memory_id
        source_memory_ids = {
            UUID(link['uuid']) for link in all_inbound_links
            if link['uuid'] not in old_memory_id_strs
        }

        for source_memory_id in source_memory_ids:
            # Get the source memory
            source_memory = self.db.get_memory(source_memory_id, user_id=user_id)
            if not source_memory:
                continue

            # Rewrite outbound_links: replace old_ids with new_id
            updated_outbound = []
            for outbound_link in source_memory.outbound_links:
                if outbound_link['uuid'] in old_memory_id_strs:
                    # Update link to point to new consolidated memory
                    updated_link = outbound_link.copy()
                    updated_link['uuid'] = new_memory_id_str
                    updated_outbound.append(updated_link)
                else:
                    updated_outbound.append(outbound_link)

            # Update the source memory with rewritten links
            self.db.update_memory(source_memory_id, {
                'outbound_links': updated_outbound
            }, user_id=user_id)

        if source_memory_ids:
            logger.debug(f"Rewrote outbound_links for {len(source_memory_ids)} source memories")

        # Step 9: Archive the old memories
        for old_id in old_memory_ids:
            self.db.archive_memory(old_id, user_id=user_id)

        logger.info(
            f"Consolidated {len(old_memory_ids)} memories into {new_memory_id} "
            f"(median importance: {median_importance:.3f}): {consolidated_text[:80]}..."
        )

        return new_memory_id

    def validate_consolidation_cluster(
        self,
        memory_ids: List[UUID],
        user_id: str
    ) -> bool:
        """
        Validate that a consolidation cluster is valid.

        Checks:
        - At least 2 memories (can't consolidate 1)
        - All memories exist and belong to user
        - Memories are not already archived

        Args:
            memory_ids: List of memory UUIDs to consolidate
            user_id: User ID

        Returns:
            True if cluster is valid, False otherwise
        """
        if len(memory_ids) < 2:
            logger.warning(f"Consolidation cluster too small: {len(memory_ids)} memories")
            return False

        memories = self.db.get_memories_by_ids(memory_ids, user_id=user_id)

        if len(memories) != len(memory_ids):
            logger.warning(
                f"Some memories not found: expected {len(memory_ids)}, got {len(memories)}"
            )
            return False

        # Check for archived memories
        archived = [m for m in memories if getattr(m, 'archived', False)]
        if archived:
            logger.warning(f"Cluster contains {len(archived)} archived memories")
            return False

        return True
