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

from lt_memory.models import ProcessingChunk, ExtractedMemory, ExtractionBatch, MemoryLink
from lt_memory.processing.extraction_engine import ExtractionEngine, ExtractionPayload
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from lt_memory.linking import LinkingService
from lt_memory.llm_routing import uses_anthropic_batch_dialect
from clients.llm_provider import LLMProvider, build_batch_params
from lt_memory.processing.batch_coordinator import BATCH_EXPIRY_HOURS
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


# ============================================================================
# Consolidated extraction storage (shared by ImmediateExecutionStrategy and
# ExtractionBatchResultHandler). Stores memories with embeddings, persists
# entities, builds typeless candidate hints, and notifies the integration
# curator to spawn via the factory's on_memories_stored callback.
# ============================================================================

def _persist_llm_entities(
    user_id: str,
    memories: List[ExtractedMemory],
    memory_ids: List[UUID],
    db: LTMemoryDB,
) -> int:
    """Persist LLM-extracted entities via pg_trgm fuzzy matching.

    Free function so both execution paths share one implementation. Uses
    get_or_create_entity to resolve name variations before creating new ones.
    Non-critical: failures log a warning and return partial counts.
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
            seen_names = set()
            for entity_dict in memory.entities:
                entity_name = entity_dict['name']
                entity_type = entity_dict.get('type', 'UNKNOWN')
                if entity_name in seen_names:
                    continue
                seen_names.add(entity_name)
                entity = db.get_or_create_entity(
                    name=entity_name, entity_type=entity_type, user_id=user_id
                )
                db.link_memory_to_entity(
                    memory_id=memory_id, entity_id=entity.id,
                    entity_name=entity_name, entity_type=entity.entity_type,
                    user_id=user_id,
                )
                total_links += 1
        if total_links:
            logger.info(f"Persisted LLM entities: {total_links} links for {len(memories)} memories")
    except Exception as e:
        logger.warning(f"Entity persistence failed for user {user_id} (non-critical): {e}", exc_info=True)
    return total_links


def _build_candidate_hints(
    memories: List[ExtractedMemory],
    memory_ids: List[UUID],
    linking: LinkingService,
) -> Dict[str, List[dict]]:
    """Assemble typeless candidate hints for the integration curator.

    Merges deterministic discovery (LinkingService.find_candidate_hints:
    vector/entity/tfidf axes) with extraction-time related_memory_ids + bonds.
    Keyed by the new memory's full UUID string; each value is a list of
    CandidateRef dicts (short memory_id + bond + discovery_signal +
    similarity). No links written — discovery finds that memories touch;
    the agent classifies how they relate.
    """
    from utils.tag_parser import format_memory_id
    hints: Dict[str, List[dict]] = {}
    if len(memories) != len(memory_ids):
        logger.error(
            f"Candidate-hints length mismatch: {len(memories)} vs {len(memory_ids)}"
        )
        return hints
    for memory, mem_id in zip(memories, memory_ids):
        full_id = str(mem_id)
        refs: List[dict] = []
        try:
            refs.extend(linking.find_candidate_hints(mem_id))
        except Exception:
            logger.warning("find_candidate_hints failed for %s", mem_id, exc_info=True)
        if memory.related_memory_ids:
            for ref in memory.related_memory_ids:
                try:
                    related_id = ref["id"]
                    refs.append({
                        "memory_id": format_memory_id(related_id),
                        "bond": ref.get("bond", ""),
                        "discovery_signal": "extraction",
                        "similarity": None,
                    })
                except (KeyError, TypeError):
                    continue
        if refs:
            hints[full_id] = refs
    return hints


def store_and_tend_extraction(
    *,
    user_id: str,
    segment_id: Optional[str],
    memories: List[ExtractedMemory],
    vector_ops: VectorOps,
    db: LTMemoryDB,
    linking: LinkingService,
) -> List[UUID]:
    """Single source of truth for extraction storage.

    Called by both ImmediateExecutionStrategy and ExtractionBatchResultHandler.
    Stores memories with embeddings, persists LLM-extracted entities, builds
    candidate hints, and notifies the integration curator to spawn (via the
    factory's on_memories_stored callback). No links are written here —
    relationship typing is the MemoryCuratorAgent's job.
    """
    memory_ids = vector_ops.store_memories_with_embeddings(memories)
    logger.info(f"Stored {len(memory_ids)} memories for user {user_id}")

    _persist_llm_entities(user_id, memories, memory_ids, db)

    candidate_hints = _build_candidate_hints(memories, memory_ids, linking)

    # Notify the integration curator (CNS-layer callback, late-registered on
    # the factory). None until the SegmentCollapseHandler registers it.
    # Best-effort: never let a spawn failure break the storage path.
    try:
        from lt_memory.factory import get_lt_memory_factory
        callback = get_lt_memory_factory().on_memories_stored
        if callback:
            callback(
                user_id=user_id,
                segment_id=segment_id,
                memory_ids=memory_ids,
                memories=memories,
                candidate_hints=candidate_hints,
            )
    except Exception:
        logger.warning("Integration curator notify failed", exc_info=True)

    return memory_ids


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
        payload: ExtractionPayload,
        segment_id: Optional[str] = None,
    ) -> List[UUID]:
        """
        Shared business logic: parse, store, and tend extracted memories.

        Delegates storage + entity persistence + candidate-hint assembly +
        curator spawn to store_and_tend_extraction (shared with the batch
        result handler). Related-memory bonds flow to the MemoryCuratorAgent
        as typeless candidate hints, not typed links.

        Args:
            user_id: User ID
            response_text: LLM response text
            payload: Extraction payload (for UUID mapping and context)
            segment_id: Segment UUID string (for the integration curator work-item)

        Returns:
            List of stored memory UUIDs
        """
        # Parse and validate using MemoryProcessor
        result = self.memory_processor.process_extraction_response(
            response_text=response_text,
            short_to_uuid=payload.short_to_uuid,
            memory_context=payload.memory_context
        )

        memories = result.memories

        memory_ids: List[UUID] = []
        if memories:
            memory_ids = store_and_tend_extraction(
                user_id=user_id,
                segment_id=segment_id,
                memories=memories,
                vector_ops=self.vector_ops,
                db=self.db,
                linking=self.linking,
            )

        return memory_ids

    def _persist_llm_entities(
        self,
        user_id: str,
        memories: List[ExtractedMemory],
        memory_ids: List[UUID]
    ) -> int:
        """Delegates to the module-level free function (consolidation)."""
        return _persist_llm_entities(user_id, memories, memory_ids, self.db)


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

    Used when Anthropic Batch API is unavailable for the resolved dialect.
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
            memory_ids = self._process_and_store_memories(
                user_id,
                response_text,
                payload,
                segment_id=str(chunk.segment_id) if chunk.segment_id else None,
            )

            total_memories_stored += len(memory_ids)

            logger.info(
                f"Immediate extraction chunk {chunk.chunk_index}: "
                f"{len(memory_ids)} memories stored"
            )

        if total_memories_stored > 0:
            logger.info(
                f"Immediate extraction complete for user {user_id}: "
                f"{total_memories_stored} total memories"
            )

        return f"bypass_{uuid4()}"


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

    Automatically selects batch or immediate based on the resolved dialect.

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
    if batch_coordinator is None or not uses_anthropic_batch_dialect("extraction"):
        if linking_service is None:
            raise ValueError(
                "ImmediateExecutionStrategy requires linking_service "
                "for relationship classification"
            )
        logger.warning("Creating ImmediateExecutionStrategy (batch dialect unavailable)")
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
