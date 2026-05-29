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

from lt_memory.models import ProcessingChunk, ExtractedMemory, ExtractionBatch, MemoryLink, LinkingPair
from lt_memory.processing.extraction_engine import ExtractionEngine, ExtractionPayload
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from lt_memory.linking import LinkingService
from lt_memory.llm_routing import uses_anthropic_batch_adapter
from clients.llm_provider import LLMProvider, build_batch_params
from lt_memory.processing.batch_coordinator import BATCH_EXPIRY_HOURS
from utils.timezone_utils import utc_now

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
    ) -> str:
        """
        Execute extraction for chunks.

        Args:
            user_id: User ID
            chunks: Processing chunks to extract from

        Returns:
            Batch ID (for batch strategy) or synthetic ID (for immediate strategy)

        Raises:
            ValueError: If no valid payloads could be built from the chunks
        """
        pass

    def _process_and_store_memories(
        self,
        user_id: str,
        response_text: str,
        payload: ExtractionPayload
    ) -> Tuple[List[UUID], List[LinkingPair]]:
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

            # Persist LLM-extracted entities via fuzzy matching
            self._persist_llm_entities(user_id, memories, memory_ids)

            # Persist extraction-time links to existing memories (related_memory_ids)
            extraction_links = []
            for idx, memory in enumerate(memories):
                if idx < len(memory_ids) and memory.related_memory_ids:
                    new_id = memory_ids[idx]
                    for ref in memory.related_memory_ids:
                        related_id = UUID(ref["id"])
                        bond = ref.get("bond", "")
                        extraction_links.append(MemoryLink(
                            source_id=new_id,
                            target_id=related_id,
                            link_type="extraction_ref",
                            reasoning=bond if bond else "Referenced during conversation",
                            extraction_bond=bond,
                            created_at=utc_now()
                        ))

            if extraction_links:
                self.db.create_links(extraction_links)
                logger.info(f"Created {len(extraction_links)} extraction_ref links for user {user_id}")

        return memory_ids, linking_pairs

    def _persist_llm_entities(
        self,
        user_id: str,
        memories: List[ExtractedMemory],
        memory_ids: List[UUID]
    ) -> int:
        """
        Persist entities extracted by LLM during memory extraction.

        Uses pg_trgm fuzzy matching via get_or_create_entity to resolve
        name variations to existing entities before creating new ones.

        Args:
            user_id: User ID
            memories: ExtractedMemory objects (with entities field populated)
            memory_ids: Parallel list of stored memory UUIDs

        Returns:
            Number of entity links created
        """
        if len(memories) != len(memory_ids):
            logger.error(
                f"Memory/ID length mismatch: {len(memories)} memories vs {len(memory_ids)} IDs"
            )
            return 0

        total_links = 0

        try:
            for memory, memory_id in zip(memories, memory_ids):
                if not memory.entities:
                    continue

                # Deduplicate entities by name within this memory
                seen_names = set()
                for entity_dict in memory.entities:
                    entity_name = entity_dict['name']
                    entity_type = entity_dict.get('type', 'UNKNOWN')

                    if entity_name in seen_names:
                        continue
                    seen_names.add(entity_name)

                    entity = self.db.get_or_create_entity(
                        name=entity_name,
                        entity_type=entity_type,
                        user_id=user_id
                    )

                    self.db.link_memory_to_entity(
                        memory_id=memory_id,
                        entity_id=entity.id,
                        entity_name=entity_name,
                        entity_type=entity.entity_type,
                        user_id=user_id
                    )
                    total_links += 1

            if total_links:
                logger.info(f"Persisted LLM entities: {total_links} links for {len(memories)} memories")

        except Exception as e:
            logger.warning(f"Entity persistence failed for user {user_id} (non-critical): {e}", exc_info=True)

        return total_links


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
        batch_coordinator: 'BatchCoordinator',
    ):
        super().__init__(extraction_engine, memory_processor, vector_ops, db)
        self.batch_coordinator = batch_coordinator

    def execute_extraction(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> str:
        """
        Submit extraction batch to Anthropic.

        Args:
            user_id: User ID
            chunks: Processing chunks

        Returns:
            Batch ID from Anthropic

        Raises:
            ValueError: If no valid payloads could be built from the chunks
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

            params = build_batch_params(
                'extraction',
                system_prompt=payload.system_prompt,
                messages=payload.messages,
            )

            request = {
                "custom_id": f"{user_id}_{chunk.chunk_index}",
                "params": params
            }

            requests.append(request)
            chunk_request_mapping.append((chunk, len(requests) - 1, payload))

        if not requests:
            raise ValueError(
                f"No valid extraction payloads built from {len(chunks)} chunks for user {user_id}"
            )

        # Submit via BatchCoordinator (single submission path)
        batch_id = self.batch_coordinator.submit_batch(
            requests=requests,
            batch_type="extraction",
            user_id=user_id,
        )
        expires_at = utc_now() + timedelta(hours=BATCH_EXPIRY_HOURS)

        # Store batch records with UUID mapping
        for chunk, request_idx, payload in chunk_request_mapping:
            batch_record = ExtractionBatch(
                batch_id=batch_id,
                custom_id=f"{user_id}_{chunk.chunk_index}",
                user_id=user_id,
                chunk_index=chunk.chunk_index,
                request_payload=requests[request_idx],
                chunk_metadata={
                    "message_count": len(chunk.messages),
                    "short_to_uuid": payload.short_to_uuid,
                    "segment_id": str(chunk.segment_id) if chunk.segment_id else None
                },
                memory_context=payload.memory_context,
                status="submitted",
                created_at=utc_now(),
                submitted_at=utc_now(),
                expires_at=expires_at
            )
            self.db.create_extraction_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted batch {batch_id} for user {user_id}: {len(requests)} chunks")
        return batch_id


class ImmediateExecutionStrategy(ExecutionStrategy):
    """
    Execute extraction immediately via the live LLM provider.

    Used when Anthropic Batch API is unavailable for the resolved adapter.
    Executes synchronously and stores results immediately, including
    entity persistence and relationship classification.
    """

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        llm_provider: LLMProvider,
        linking_service: LinkingService
    ):
        super().__init__(extraction_engine, memory_processor, vector_ops, db)
        self.llm_provider = llm_provider
        self.linking = linking_service

    def execute_extraction(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> str:
        """
        Execute extraction immediately using OpenAI fallback.

        Caller is responsible for user context lifecycle — this method
        does not set or clear user context.

        Args:
            user_id: User ID
            chunks: Processing chunks

        Returns:
            Synthetic batch ID for tracking

        Raises:
            Exception: If LLM call or result processing fails
        """
        total_memories_stored = 0
        all_memory_ids: List[UUID] = []
        all_linking_pairs: List[LinkingPair] = []

        for chunk in chunks:
            # Build extraction payload
            payload = self.extraction_engine.build_extraction_payload(
                chunk,
                for_batch=False  # Use immediate format (system + user prompt)
            )

            if not payload.user_prompt:
                continue

            # Call LLM directly using extraction internal LLM config
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": payload.user_prompt}],
                system_prompt=payload.system_prompt,
                internal_llm='extraction',
                allow_negative=True,  # System task — segment already paid for
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
            all_memory_ids.extend(memory_ids)
            all_linking_pairs.extend(linking_pairs)

            logger.info(
                f"Immediate extraction chunk {chunk.chunk_index}: "
                f"{len(memory_ids)} memories stored"
            )

        # Post-storage processing: relationships
        if all_memory_ids:
            # Trigger relationship classification
            self._trigger_relationship_classification(user_id, all_memory_ids, all_linking_pairs)

        if total_memories_stored > 0:
            logger.info(
                f"Immediate extraction complete for user {user_id}: "
                f"{total_memories_stored} total memories"
            )

        return f"bypass_{uuid4()}"

    def _trigger_relationship_classification(
        self,
        user_id: str,
        memory_ids: List[UUID],
        linking_hints: List[LinkingPair]
    ) -> None:
        """
        Execute relationship classification immediately for new memories.

        ImmediateExecutionStrategy executes classifications synchronously via
        the LLM provider.
        """
        if not memory_ids:
            return

        all_pairs = []

        # Process extraction hints first
        if linking_hints:
            new_memories = {m.id: m for m in self.db.get_memories_by_ids(memory_ids, user_id=user_id)}

            for pair in linking_hints:
                src_idx = pair["source_idx"]
                tgt_idx = pair["target_idx"]
                bond = pair.get("bond", "")

                if src_idx < len(memory_ids) and tgt_idx < len(memory_ids):
                    src_id = memory_ids[src_idx]
                    tgt_id = memory_ids[tgt_idx]

                    if src_id in new_memories and tgt_id in new_memories:
                        all_pairs.append({
                            "new_memory_id": src_id,
                            "similar_memory_id": tgt_id,
                            "new_memory": new_memories[src_id],
                            "similar_memory": new_memories[tgt_id],
                            "from_extraction_hint": True,
                            "bond": bond
                        })

            logger.info(f"Added {len(all_pairs)} pairs from extraction hints for user {user_id}")

        # Find similar existing memories
        for mem_id in memory_ids:
            candidates = self.linking.find_similar_candidates(mem_id)
            for candidate in candidates:
                all_pairs.append({
                    "new_memory_id": mem_id,
                    "similar_memory_id": candidate.id,
                    "new_memory": self.db.get_memory(mem_id, user_id=user_id),
                    "similar_memory": candidate,
                    "from_extraction_hint": False
                })

        if not all_pairs:
            return

        # Execute classifications immediately.
        # Circuit breaker: if the model returns garbage N times in a row,
        # stop wasting LLM calls — the endpoint is broken.
        MAX_CONSECUTIVE_FAILURES = 5

        links_created = 0
        consecutive_failures = 0
        for pair in all_pairs:
            try:
                # Build classification payload
                payload = self.linking.build_classification_payload(
                    pair["new_memory"],
                    pair["similar_memory"],
                    bond=pair.get("bond", "")
                )

                # Call LLM directly using relationship internal LLM config
                response = self.llm_provider.generate_response(
                    messages=[{"role": "user", "content": payload["user_prompt"]}],
                    system_prompt=payload["system_prompt"],
                    internal_llm='relationship',
                    allow_negative=True,  # System task — segment already paid for
                )

                # Extract text from response
                response_text = self.llm_provider.extract_text_content(response)

                # Parse classification result
                import json
                classification = json.loads(response_text)
                rel_type = classification.get("relationship_type")

                consecutive_failures = 0  # Reset on successful parse

                if rel_type and rel_type != "null":
                    # Create bidirectional link
                    if self.linking.create_bidirectional_link(
                        source_id=pair["new_memory_id"],
                        target_id=pair["similar_memory_id"],
                        link_type=rel_type,
                        reasoning=classification.get("reasoning", ""),
                        extraction_bond=pair.get("bond", "")
                    ):
                        links_created += 1

            except json.JSONDecodeError:
                consecutive_failures += 1
                logger.error(
                    f"Invalid JSON in relationship classification response "
                    f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): "
                    f"{response_text[:200]}"
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"Circuit breaker tripped: {MAX_CONSECUTIVE_FAILURES} consecutive "
                        f"JSON failures — aborting relationship classification for user {user_id}. "
                        f"The relationship LLM endpoint is likely returning non-JSON responses."
                    )
                    break
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Relationship classification failed for pair: {e}")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        f"Circuit breaker tripped: {MAX_CONSECUTIVE_FAILURES} consecutive "
                        f"failures — aborting relationship classification for user {user_id}."
                    )
                    break

        logger.info(
            f"Immediate: relationship classification complete for {len(all_pairs)} pairs, "
            f"{links_created} links created"
        )


def create_execution_strategy(
    extraction_engine: ExtractionEngine,
    memory_processor: MemoryProcessor,
    vector_ops: VectorOps,
    db: LTMemoryDB,
    llm_provider: LLMProvider,
    batch_coordinator: 'BatchCoordinator | None',
    linking_service: Optional[LinkingService] = None
) -> ExecutionStrategy:
    """
    Factory function to create appropriate execution strategy.

    Automatically selects batch or immediate based on the resolved adapter.

    Args:
        extraction_engine: Extraction engine instance
        memory_processor: Memory processor instance
        vector_ops: Vector operations instance
        db: Database instance
        llm_provider: LLM provider instance
        batch_coordinator: BatchCoordinator (None if unavailable)
        linking_service: Linking service (required for immediate mode)

    Returns:
        Appropriate ExecutionStrategy (Batch or Immediate)
    """
    if batch_coordinator is None or not uses_anthropic_batch_adapter("extraction"):
        if linking_service is None:
            raise ValueError(
                "ImmediateExecutionStrategy requires linking_service "
                "for relationship classification"
            )
        logger.warning("Creating ImmediateExecutionStrategy (batch adapter unavailable)")
        return ImmediateExecutionStrategy(
            extraction_engine,
            memory_processor,
            vector_ops,
            db,
            llm_provider,
            linking_service
        )
    else:
        return BatchExecutionStrategy(
            extraction_engine,
            memory_processor,
            vector_ops,
            db,
            batch_coordinator,
        )
