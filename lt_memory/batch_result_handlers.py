"""
Batch result handlers for processing completed Anthropic Batch API results.

Contains complete implementations of result processing logic that was
previously embedded in BatchingService.
"""
import json
import logging
from typing import List, Tuple
from uuid import UUID

import anthropic
from json_repair import repair_json

from lt_memory.db_access import LTMemoryDB
from lt_memory.processing.batch_coordinator import BatchResultProcessor
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps
from lt_memory.linking import LinkingService
from lt_memory.models import ExtractionBatch, PostProcessingBatch
from utils.user_context import set_current_user_id, clear_user_context

logger = logging.getLogger(__name__)


class ExtractionBatchResultHandler(BatchResultProcessor):
    """
    Handle extraction batch results: parse, store, persist entities, trigger relationships.

    Complete implementation extracted from BatchingService._process_extraction_result().
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        linking_service: LinkingService
    ):
        self.anthropic_client = anthropic_client
        self.memory_processor = memory_processor
        self.vector_ops = vector_ops
        self.db = db
        self.linking = linking_service

    def process_result(self, batch_id: str, batch: ExtractionBatch) -> bool:
        """
        Process extraction batch result.

        Args:
            batch_id: Anthropic batch ID
            batch: Extraction batch record

        Returns:
            True if processing succeeded
        """
        # Retrieve result from Anthropic
        for result in self.anthropic_client.beta.messages.batches.results(batch_id):
            if result.custom_id != batch.custom_id:
                continue

            if result.result.type == "succeeded":
                # Extract text from response
                text_blocks = [
                    block.text for block in result.result.message.content
                    if block.type == "text" and block.text
                ]

                # Skip if no text content
                if not text_blocks:
                    logger.debug(f"Batch {batch_id} result had no text content - skipping")
                    self.db.delete_extraction_batch(batch.id, user_id=batch.user_id)
                    return True

                response_text = "\n".join(text_blocks)

                # Process extraction result
                set_current_user_id(batch.user_id)
                try:
                    short_to_uuid = batch.chunk_metadata.get("short_to_uuid", {}) if batch.chunk_metadata else {}

                    # Parse with memory processor
                    result = self.memory_processor.process_extraction_response(
                        response_text=response_text,
                        short_to_uuid=short_to_uuid,
                        memory_context=batch.memory_context or {}
                    )

                    memories = result.memories
                    linking_pairs = result.linking_pairs

                    # Store memories with embeddings
                    memory_ids = []
                    if memories:
                        memory_ids = self.vector_ops.store_memories_with_embeddings(memories)

                        # Persist entities and create memory→entity links
                        self._persist_entities_for_memories(batch.user_id, memory_ids)

                        # Trigger relationship classification with linking hints
                        self._trigger_relationship_classification(batch.user_id, memory_ids, linking_pairs)

                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: {len(memory_ids)} stored")

                        # Update last extraction timestamp
                        self.db.update_extraction_timestamp(batch.user_id)
                    else:
                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: no memories extracted")

                    # Delete batch record - processing complete
                    self.db.delete_extraction_batch(batch.id, user_id=batch.user_id)
                    return True
                finally:
                    clear_user_context()

            elif result.result.type == "errored":
                self.db.update_extraction_batch_status(
                    batch.id,
                    "failed",
                    error_message=str(result.result.error),
                    user_id=batch.user_id
                )
                return False

        return False

    def _persist_entities_for_memories(self, user_id: str, memory_ids: List[UUID]) -> None:
        """Extract and persist entities for newly created memories."""
        if not memory_ids:
            return

        try:
            from lt_memory.entity_extraction import EntityExtractor

            entity_extractor = EntityExtractor()
            memories = self.db.get_memories_by_ids(memory_ids, user_id=user_id)
            if not memories:
                return

            total_links_created = 0

            for memory in memories:
                entities_with_types = entity_extractor.extract_entities_with_types(memory.text)
                if not entities_with_types:
                    continue

                # Deduplicate entities within this memory
                unique_entities = {}
                for entity_name, entity_type in entities_with_types:
                    key = (entity_name, entity_type)
                    if key not in unique_entities:
                        unique_entities[key] = (entity_name, entity_type)

                # Persist each unique entity and link to this memory
                for entity_name, entity_type in unique_entities.values():
                    entity_embedding = entity_extractor.nlp(entity_name).vector.tolist()

                    entity = self.db.get_or_create_entity(
                        name=entity_name,
                        entity_type=entity_type,
                        embedding=entity_embedding,
                        user_id=user_id
                    )

                    if entity:
                        self.db.link_memory_to_entity(
                            memory_id=memory.id,
                            entity_id=entity.id,
                            entity_name=entity_name,
                            entity_type=entity_type,
                            user_id=user_id
                        )
                        total_links_created += 1

            logger.info(f"Persisted entities for {len(memories)} memories: {total_links_created} links created")
            entity_extractor.cleanup()

        except Exception as e:
            logger.warning(f"Background entity persistence failed for user {user_id} (non-critical): {e}", exc_info=True)

    def _trigger_relationship_classification(
        self,
        user_id: str,
        memory_ids: List[UUID],
        linking_hints: List[Tuple[int, int]] = None
    ):
        """Trigger relationship classification for new memories."""
        if not memory_ids:
            return

        all_pairs = []

        # Process extraction hints first
        if linking_hints:
            new_memories = {m.id: m for m in self.db.get_memories_by_ids(memory_ids, user_id=user_id)}

            for src_idx, tgt_idx in linking_hints:
                if src_idx < len(memory_ids) and tgt_idx < len(memory_ids):
                    src_id = memory_ids[src_idx]
                    tgt_id = memory_ids[tgt_idx]

                    if src_id in new_memories and tgt_id in new_memories:
                        all_pairs.append({
                            "new_memory_id": str(src_id),
                            "similar_memory_id": str(tgt_id),
                            "new_memory": new_memories[src_id],
                            "similar_memory": new_memories[tgt_id],
                            "from_extraction_hint": True
                        })

            logger.info(f"Added {len(all_pairs)} pairs from extraction hints for user {user_id}")

        # Find similar existing memories
        for mem_id in memory_ids:
            candidates = self.linking.find_similar_candidates(mem_id)
            for candidate in candidates:
                all_pairs.append({
                    "new_memory_id": str(mem_id),
                    "similar_memory_id": str(candidate.id),
                    "new_memory": self.db.get_memory(mem_id, user_id=user_id),
                    "similar_memory": candidate,
                    "from_extraction_hint": False
                })

        if all_pairs:
            logger.info(f"Triggered relationship classification for {len(all_pairs)} memory pairs")


class RelationshipBatchResultHandler(BatchResultProcessor):
    """
    Handle relationship classification results: parse and create bidirectional links.

    Complete implementation extracted from BatchingService._process_relationship_result().
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        linking_service: LinkingService,
        db: LTMemoryDB
    ):
        self.anthropic_client = anthropic_client
        self.linking = linking_service
        self.db = db

    def process_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """Process relationship classification result."""
        try:
            # Retrieve results from Anthropic
            classifications = {}
            for result in self.anthropic_client.beta.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    response_text = "\n".join([
                        block.text for block in result.result.message.content
                        if block.type == "text"
                    ])

                    if not response_text.strip():
                        logger.warning(f"Empty response in {result.custom_id}")
                        continue

                    # Parse JSON with repair fallback
                    try:
                        classifications[result.custom_id] = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed for {result.custom_id}: {e}, attempting repair")
                        try:
                            repaired = repair_json(response_text)
                            classifications[result.custom_id] = json.loads(repaired)
                            logger.debug(f"Successfully repaired JSON for {result.custom_id}")
                        except (json.JSONDecodeError, Exception) as repair_error:
                            logger.error(f"JSON repair failed for {result.custom_id}: {repair_error}")
                            logger.debug(f"Response text (first 500 chars): {response_text[:500]}")
                            continue

            if not classifications:
                self.db.update_relationship_batch_status(
                    batch.id,
                    "failed",
                    error_message="No valid results",
                    user_id=batch.user_id
                )
                return False

            # Create links
            set_current_user_id(batch.user_id)
            links_created = 0

            for custom_id, classification in classifications.items():
                rel_type = classification.get("relationship_type")
                if rel_type == "null":
                    continue

                # Find memory IDs from input_data
                pair_data = next(
                    (data for new_id, data in batch.input_data.items() if new_id[:8] in custom_id),
                    None
                )

                if not pair_data:
                    continue

                # Create bidirectional link
                if self.linking.create_bidirectional_link(
                    source_id=UUID(pair_data["new_memory_id"]),
                    target_id=UUID(pair_data["similar_memory_id"]),
                    link_type=rel_type,
                    confidence=classification.get("confidence", 0.9),
                    reasoning=classification.get("reasoning", "")
                ):
                    links_created += 1

            clear_user_context()

            # Delete batch record
            self.db.delete_relationship_batch(batch.id, user_id=batch.user_id)
            logger.info(f"Relationship batch {batch_id}: {links_created} links created")

            return True

        except Exception as e:
            logger.error(f"Error processing relationship result: {e}", exc_info=True)
            self.db.update_relationship_batch_status(
                batch.id,
                "failed",
                error_message=str(e),
                user_id=batch.user_id
            )
            return False
