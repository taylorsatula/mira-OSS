"""
Batch result handlers for processing completed Anthropic Batch API results.

ExtractionBatchResultHandler implements BatchResultProcessor and is invoked by
the batch polling job (BatchCoordinator.poll_batches) when an extraction batch
completes. It parses, stores, persists entities, and notifies the
MemoryCuratorAgent — all relationship/consolidation judgment moved to the
agentic curator in v2.
"""
import logging
from uuid import UUID

import anthropic

from lt_memory.db_access import LTMemoryDB
from lt_memory.processing.batch_coordinator import BatchResultProcessor
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps
from lt_memory.linking import LinkingService
from lt_memory.models import ExtractionBatch, ExtractedMemory
from lt_memory.processing.execution_strategy import store_and_tend_extraction
from clients.llm_provider import LLMProvider
from utils.user_context import set_current_user_id, clear_user_context

logger = logging.getLogger(__name__)


class ExtractionBatchResultHandler(BatchResultProcessor):
    """
    Handle extraction batch results: parse, store, persist entities, trigger relationships.
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        linking_service: LinkingService,
        llm_provider: LLMProvider,
        batch_coordinator: 'BatchCoordinator'
    ):
        self.anthropic_client = anthropic_client
        self.memory_processor = memory_processor
        self.vector_ops = vector_ops
        self.db = db
        self.linking = linking_service
        self.llm_provider = llm_provider
        self.batch_coordinator = batch_coordinator

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
                    self.db.delete_batch(batch.id, user_id=batch.user_id)
                    return True

                response_text = "\n".join(text_blocks)

                # Process extraction result
                set_current_user_id(batch.user_id)
                try:
                    short_to_uuid = batch.chunk_metadata.get("short_to_uuid", {}) if batch.chunk_metadata else {}
                    segment_id = batch.chunk_metadata.get("segment_id") if batch.chunk_metadata else None

                    # Parse with memory processor
                    result = self.memory_processor.process_extraction_response(
                        response_text=response_text,
                        short_to_uuid=short_to_uuid,
                        memory_context=batch.memory_context or {}
                    )

                    memories = result.memories

                    # Set source_segment_id on extracted memories for resume cleanup
                    if segment_id:
                        for memory in memories:
                            memory.source_segment_id = UUID(segment_id)

                    # Store memories, persist entities, and notify the
                    # integration curator (shared with the immediate path).
                    # Related-memory bonds flow to the curator as typeless
                    # candidate hints, not typed links.
                    memory_ids = []
                    if memories:
                        memory_ids = store_and_tend_extraction(
                            user_id=batch.user_id,
                            segment_id=str(segment_id) if segment_id else None,
                            memories=memories,
                            vector_ops=self.vector_ops,
                            db=self.db,
                            linking=self.linking,
                        )

                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: {len(memory_ids)} stored")
                    else:
                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: no memories extracted")

                    # Apply importance boosts to existing memories
                    memory_context = batch.memory_context or {}
                    pinned_short_ids = memory_context.get("pinned_short_ids", [])
                    if pinned_short_ids:
                        self.db.apply_pin_boost(pinned_short_ids, user_id=batch.user_id)

                    # Delete batch record - processing complete
                    self.db.delete_batch(batch.id, user_id=batch.user_id)
                    return True
                finally:
                    clear_user_context()

            elif result.result.type == "errored":
                self.db.update_batch_status(
                    batch.id,
                    "failed",
                    error_message=str(result.result.error),
                    user_id=batch.user_id
                )
                return False

        return False
