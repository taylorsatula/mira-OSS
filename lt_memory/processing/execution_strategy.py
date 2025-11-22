"""
Execution strategy - execute extraction via batch or immediate.

Eliminates duplication between batch API and immediate (fallback) execution paths.

Key insight: Batch vs Immediate differ ONLY in transport (how to call LLM).
All business logic (parsing, validation, storage, entity persistence, relationship
triggering) is IDENTICAL.

Strategy pattern: Share business logic, vary only the LLM call mechanism.
"""
import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID, uuid4

import anthropic

from lt_memory.models import ProcessingChunk, ExtractionBatch
from lt_memory.processing.extraction_engine import ExtractionEngine, ExtractionPayload
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from clients.llm_provider import LLMProvider
from config.config import BatchingConfig, ExtractionConfig
from utils.timezone_utils import utc_now
from utils.user_context import set_current_user_id, clear_user_context

logger = logging.getLogger(__name__)


class ExecutionStrategy(ABC):
    """
    Abstract base for extraction execution strategies.

    Concrete implementations:
    - BatchExecutionStrategy: Submit to Anthropic Batch API
    - ImmediateExecutionStrategy: Execute immediately via OpenAI fallback
    """

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB
    ):
        """
        Initialize execution strategy.

        Args:
            extraction_engine: Builds extraction payloads
            memory_processor: Parses and validates LLM responses
            vector_ops: Vector operations for memory storage
            db: Database access
        """
        self.extraction_engine = extraction_engine
        self.memory_processor = memory_processor
        self.vector_ops = vector_ops
        self.db = db

    @abstractmethod
    def execute_extraction(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> Optional[str]:
        """
        Execute extraction for chunks.

        Args:
            user_id: User ID
            chunks: Processing chunks to extract from

        Returns:
            Batch ID (for batch strategy) or synthetic ID (for immediate strategy)
        """
        pass

    def _process_and_store_memories(
        self,
        user_id: str,
        response_text: str,
        payload: ExtractionPayload
    ) -> Tuple[List[UUID], List[Tuple[int, int]]]:
        """
        Shared business logic: process LLM response and store memories.

        This is the IDENTICAL logic that was duplicated in both batch and immediate paths.

        Args:
            user_id: User ID
            response_text: LLM response text
            payload: Extraction payload (for UUID mapping and context)

        Returns:
            Tuple of (memory_ids, linking_pairs)
        """
        # Parse and validate using MemoryProcessor
        result = self.memory_processor.process_extraction_response(
            response_text=response_text,
            short_to_uuid=payload.short_to_uuid,
            memory_context=payload.memory_context
        )

        memories = result.memories
        linking_pairs = result.linking_pairs

        # Store memories with embeddings
        memory_ids = []
        if memories:
            memory_ids = self.vector_ops.store_memories_with_embeddings(memories)
            logger.info(f"Stored {len(memory_ids)} memories for user {user_id}")

        return memory_ids, linking_pairs


class BatchExecutionStrategy(ExecutionStrategy):
    """
    Execute extraction via Anthropic Batch API.

    Submits requests to Batch API and stores tracking records.
    Results processed asynchronously by BatchCoordinator polling.
    """

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        anthropic_client: anthropic.Anthropic,
        batching_config: BatchingConfig,
        extraction_config: ExtractionConfig
    ):
        super().__init__(extraction_engine, memory_processor, vector_ops, db)
        self.anthropic_client = anthropic_client
        self.batching_config = batching_config
        self.extraction_config = extraction_config

    def execute_extraction(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> Optional[str]:
        """
        Submit extraction batch to Anthropic.

        Args:
            user_id: User ID
            chunks: Processing chunks

        Returns:
            Batch ID if successful, None if no chunks to process

        Raises:
            Exception: If Anthropic API call fails
        """
        requests = []
        chunk_request_mapping = []  # Track which chunks got requests

        for chunk in chunks:
            # Build extraction payload
            payload = self.extraction_engine.build_extraction_payload(
                chunk,
                for_batch=True
            )

            if not payload.messages:
                continue

            request = {
                "custom_id": f"{user_id}_{chunk.chunk_index}",
                "params": {
                    "model": self.extraction_config.extraction_model,
                    "max_tokens": self.extraction_config.max_extraction_tokens,
                    "temperature": self.extraction_config.extraction_temperature,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "system": [{
                        "type": "text",
                        "text": payload.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    "messages": payload.messages
                }
            }

            requests.append(request)
            chunk_request_mapping.append((chunk, len(requests) - 1, payload))

        if not requests:
            return None

        # Submit to Anthropic
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)
        expires_at = utc_now() + timedelta(hours=self.batching_config.batch_expiry_hours)

        # Store batch records with UUID mapping
        for chunk, request_idx, payload in chunk_request_mapping:
            batch_record = ExtractionBatch(
                batch_id=batch.id,
                custom_id=f"{user_id}_{chunk.chunk_index}",
                user_id=user_id,
                chunk_index=chunk.chunk_index,
                request_payload=requests[request_idx],
                chunk_metadata={
                    "message_count": len(chunk.messages),
                    "short_to_uuid": payload.short_to_uuid
                },
                memory_context=payload.memory_context,
                status="submitted",
                created_at=utc_now(),
                submitted_at=utc_now(),
                expires_at=expires_at
            )
            self.db.create_extraction_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted batch {batch.id} for user {user_id}: {len(requests)} chunks")
        return batch.id


class ImmediateExecutionStrategy(ExecutionStrategy):
    """
    Execute extraction immediately via OpenAI fallback.

    Used when Anthropic Batch API is unavailable (failover mode).
    Executes synchronously and stores results immediately.
    """

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        llm_provider: LLMProvider
    ):
        super().__init__(extraction_engine, memory_processor, vector_ops, db)
        self.llm_provider = llm_provider

    def execute_extraction(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> Optional[str]:
        """
        Execute extraction immediately using OpenAI fallback.

        Args:
            user_id: User ID
            chunks: Processing chunks

        Returns:
            Synthetic batch ID for tracking

        Raises:
            Exception: If LLM call or result processing fails
        """
        set_current_user_id(user_id)
        total_memories_stored = 0
        all_linking_pairs = []

        try:
            for chunk in chunks:
                # Build extraction payload
                payload = self.extraction_engine.build_extraction_payload(
                    chunk,
                    for_batch=False  # Use immediate format (system + user prompt)
                )

                if not payload.user_prompt:
                    continue

                # Call LLM directly (routes to OpenAI fallback)
                response = self.llm_provider.generate_response(
                    messages=[{"role": "user", "content": payload.user_prompt}],
                    system_override=payload.system_prompt,
                    thinking_enabled=True,
                    thinking_budget=1024
                )

                # Extract text from response
                response_text = self.llm_provider.extract_text_content(response)

                # Process and store memories (shared business logic)
                memory_ids, linking_pairs = self._process_and_store_memories(
                    user_id,
                    response_text,
                    payload
                )

                total_memories_stored += len(memory_ids)
                all_linking_pairs.extend(linking_pairs)

                logger.info(
                    f"Immediate extraction chunk {chunk.chunk_index}: "
                    f"{len(memory_ids)} memories stored"
                )

            # Update timestamp to prevent reprocessing
            if total_memories_stored > 0:
                self.db.update_extraction_timestamp(user_id)
                logger.info(
                    f"Immediate extraction complete for user {user_id}: "
                    f"{total_memories_stored} total memories"
                )

            return f"bypass_{uuid4()}"

        finally:
            clear_user_context()


def create_execution_strategy(
    extraction_engine: ExtractionEngine,
    memory_processor: MemoryProcessor,
    vector_ops: VectorOps,
    db: LTMemoryDB,
    llm_provider: LLMProvider,
    anthropic_client: Optional[anthropic.Anthropic],
    batching_config: BatchingConfig,
    extraction_config: ExtractionConfig
) -> ExecutionStrategy:
    """
    Factory function to create appropriate execution strategy.

    Automatically selects batch or immediate based on failover status.

    Args:
        extraction_engine: Extraction engine instance
        memory_processor: Memory processor instance
        vector_ops: Vector operations instance
        db: Database instance
        llm_provider: LLM provider instance
        anthropic_client: Anthropic client (None if unavailable)
        batching_config: Batching configuration
        extraction_config: Extraction configuration

    Returns:
        Appropriate ExecutionStrategy (Batch or Immediate)
    """
    # Check if failover mode active
    if llm_provider._is_failover_active() or anthropic_client is None:
        logger.warning("Creating ImmediateExecutionStrategy (failover mode active)")
        return ImmediateExecutionStrategy(
            extraction_engine,
            memory_processor,
            vector_ops,
            db,
            llm_provider
        )
    else:
        return BatchExecutionStrategy(
            extraction_engine,
            memory_processor,
            vector_ops,
            db,
            anthropic_client,
            batching_config,
            extraction_config
        )
