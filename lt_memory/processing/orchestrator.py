"""
Extraction orchestrator - high-level extraction workflows.

Consolidates high-level extraction workflows from batching.py:
- Boot extraction sweep (all users with memory enabled)
- Segment extraction (after segment collapse)
- Failed extraction retry (safety net)

These are THIN coordinators that delegate to specialized components.
"""
import json
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID

from cns.core.message import Message
from lt_memory.models import ProcessingChunk
from lt_memory.processing.extraction_engine import ExtractionEngine
from lt_memory.processing.execution_strategy import ExecutionStrategy
from lt_memory.db_access import LTMemoryDB
from config.config import BatchingConfig
from utils.user_context import set_current_user_id, get_current_user_id, clear_user_context

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """
    High-level extraction workflow coordination.

    Single Responsibility: Orchestrate extraction workflows (boot, segment, retry)

    Thin coordinator - delegates to:
    - ExtractionEngine: Build payloads
    - ExecutionStrategy: Execute extraction
    - ContinuumRepository: Load messages
    - AuthDB: Get users, update timestamps
    - LTMemoryDB: Safety valve checks, memory context loading
    """

    def __init__(
        self,
        config: BatchingConfig,
        extraction_engine: ExtractionEngine,
        execution_strategy: ExecutionStrategy,
        continuum_repo,
        db: LTMemoryDB
    ):
        """
        Initialize extraction orchestrator.

        Args:
            config: Batching configuration
            extraction_engine: Extraction engine for payload building
            execution_strategy: Execution strategy (batch or immediate)
            continuum_repo: Continuum repository for message loading
            db: LT Memory database for safety checks and context loading
        """
        self.config = config
        self.extraction_engine = extraction_engine
        self.execution_strategy = execution_strategy
        self.continuum_repo = continuum_repo
        self.db = db

    def run_boot_extraction(self) -> Dict[str, Any]:
        """
        Run on-boot extraction sweep for all users with memory enabled.

        Loads conversation messages since last extraction and submits for processing.
        Implements safety valve: skips users with pending batches.

        Returns:
            Boot extraction statistics
        """
        logger.info("Starting on-boot extraction sweep")

        users = self.db.get_users_with_memory_enabled()
        results = {
            "users_checked": len(users),
            "batches_submitted": 0,
            "users_skipped": 0,
            "errors": []
        }

        for user in users:
            uid = str(user["id"])
            try:
                # Safety valve: skip if pending batches (but warn - could indicate stuck batches)
                pending_batches = self.db.get_pending_extraction_batches_for_user(uid)
                if pending_batches:
                    results["users_skipped"] += 1
                    logger.warning(
                        f"Skipping user {uid}: {len(pending_batches)} pending extraction batches exist. "
                        "If batches consistently pending across multiple boot cycles, investigate "
                        "batch polling job or Anthropic API issues."
                    )
                    continue

                set_current_user_id(uid)
                chunks = self._load_conversation_chunks(uid)

                if chunks:
                    batch_id = self.execution_strategy.execute_extraction(uid, chunks)
                    if batch_id:
                        results["batches_submitted"] += 1
            except Exception as e:
                logger.error(f"Error in boot extraction for {uid}: {e}", exc_info=True)
                results["errors"].append(str(e))
            finally:
                # Only clear if we set it for this user
                try:
                    if get_current_user_id() == uid:
                        clear_user_context()
                except Exception:
                    pass  # Context wasn't set, nothing to clear

        logger.info(f"Boot extraction complete: {results['batches_submitted']} batches submitted")
        return results

    def submit_segment_extraction(
        self,
        user_id: str,
        messages: List[Message],
        segment_id: str
    ) -> bool:
        """
        Submit segment messages for memory extraction with chunking.

        Called when a segment collapses. Chunks the segment's messages using
        segment_chunk_size and submits to execution strategy (batch or immediate).

        Args:
            user_id: User ID
            messages: Segment messages (already filtered, no summaries/boundaries)
            segment_id: Segment UUID for tracking

        Returns:
            True if submission succeeded

        Raises:
            Exception: If extraction submission fails
        """
        # Build chunks from segment messages using segment_chunk_size
        chunks = []
        chunk_size = self.config.segment_chunk_size

        for i in range(0, len(messages), chunk_size):
            chunk_messages = messages[i:i + chunk_size]
            chunk = ProcessingChunk.from_conversation_messages(
                chunk_messages,
                chunk_index=i // chunk_size
            )

            # Build memory context for this chunk (only referenced memories)
            memory_context = self._build_memory_context(chunk_messages, user_id)
            chunk.memory_context_snapshot = memory_context

            chunks.append(chunk)

        logger.info(
            f"Built {len(chunks)} chunks from segment {segment_id} ({len(messages)} messages)"
        )

        # Submit via execution strategy
        set_current_user_id(user_id)
        try:
            batch_id = self.execution_strategy.execute_extraction(user_id, chunks)

            if batch_id:
                logger.info(
                    f"Submitted segment {segment_id} for extraction "
                    f"(batch: {batch_id}, {len(chunks)} chunks)"
                )
                return True
            else:
                logger.warning(f"Failed to submit segment {segment_id} for extraction")
                return False
        finally:
            # Only clear if we set it for this user
            try:
                if get_current_user_id() == user_id:
                    clear_user_context()
            except Exception:
                pass  # Context wasn't set, nothing to clear

    def retry_failed_extractions(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Safety net for failed segment extractions.

        Primary extraction happens during segment collapse. This job catches
        collapsed segments where extraction failed and retries them.

        Args:
            user_id: Optional specific user. If None, processes all users.

        Returns:
            Extraction statistics
        """
        logger.info(
            f"Starting extraction failure retry sweep"
            f"{f' for user {user_id}' if user_id else ''}"
        )

        users = (
            [{"id": user_id}] if user_id
            else self.db.get_users_with_memory_enabled()
        )
        results = {"segments_retried": 0, "users_processed": 0, "errors": []}

        for user in users:
            uid = str(user["id"])
            try:
                # Find collapsed segments that failed extraction
                failed_segments = self.continuum_repo.find_failed_extraction_segments(uid)

                if not failed_segments:
                    continue

                logger.info(f"Found {len(failed_segments)} failed segments for user {uid}")
                set_current_user_id(uid)

                for segment in failed_segments:
                    segment_id = segment['segment_id']

                    # Load messages for this segment
                    messages = self._load_segment_messages(segment['message_id'], uid)

                    if not messages:
                        logger.warning(f"No messages found for segment {segment_id}")
                        continue

                    # Retry extraction using existing method
                    if self.submit_segment_extraction(uid, messages, segment_id):
                        results["segments_retried"] += 1
                        logger.info(f"Retried extraction for segment {segment_id}")

                results["users_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing user {uid}: {e}", exc_info=True)
                results["errors"].append(str(e))
            finally:
                # Only clear if we set it for this user
                try:
                    if get_current_user_id() == uid:
                        clear_user_context()
                except Exception:
                    pass  # Context wasn't set, nothing to clear

        logger.info(
            f"Extraction failure retry sweep complete: "
            f"{results['segments_retried']} segments retried"
        )
        return results

    # ============================================================================
    # Helper Methods (Continuum Loading)
    # ============================================================================

    def _load_conversation_chunks(self, user_id: str) -> List[ProcessingChunk]:
        """
        Load continuum messages and build chunks with memory context.

        Args:
            user_id: User ID

        Returns:
            List of ProcessingChunk objects

        Raises:
            Exception: If database query or continuum operations fail
        """
        # Get continuum (must exist from signup)
        continuum = self.continuum_repo.get_continuum(user_id)
        if not continuum:
            raise RuntimeError(
                f"Continuum not found for user {user_id}. "
                f"Continuum should be created during signup."
            )

        # Get last extraction timestamp
        last_run = self.db.get_extraction_timestamp(user_id)

        # Load messages since last run
        db_client = self.continuum_repo._get_client(user_id)
        query = """
        SELECT * FROM messages
        WHERE continuum_id = %s
        """ + (
            "AND created_at > %s" if last_run else ""
        ) + """
        AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' = 'false')
        ORDER BY created_at
        """

        params = (str(continuum.id), last_run) if last_run else (str(continuum.id),)
        message_rows = db_client.execute_query(query, params)

        # Convert to Message objects
        messages = []
        for row in message_rows:
            metadata = self._parse_metadata(row.get('metadata', {}))
            messages.append(Message(
                id=row['id'],
                content=row['content'],
                role=row['role'],
                created_at=row['created_at'],
                metadata=metadata
            ))

        # Check minimum messages
        conversational = [m for m in messages if m.role in ('user', 'assistant')]
        if len(conversational) < self.config.min_messages_for_boot_extraction:
            logger.info(
                f"Skipping user {user_id}: only {len(conversational)} conversational messages "
                f"(minimum {self.config.min_messages_for_boot_extraction} required)"
            )
            return []

        # Build chunks with memory context
        chunks = []
        chunk_size = self.config.max_chunk_size

        for i in range(0, len(messages), chunk_size):
            chunk_messages = messages[i:i + chunk_size]
            chunk = ProcessingChunk.from_conversation_messages(
                chunk_messages,
                chunk_index=i // chunk_size
            )

            # Build memory context for this chunk (only referenced)
            memory_context = self._build_memory_context(chunk_messages, user_id)
            chunk.memory_context_snapshot = memory_context

            chunks.append(chunk)

        logger.info(f"Built {len(chunks)} chunks for user {user_id}")
        return chunks

    def _load_segment_messages(self, segment_message_id: str, user_id: str) -> List[Message]:
        """
        Load messages for a specific segment.

        Loads all messages from segment boundary (exclusive) to next boundary or end.

        Args:
            segment_message_id: UUID of segment boundary message
            user_id: User ID

        Returns:
            List of Message objects in segment

        Raises:
            Exception: If database query fails
        """
        db = self.continuum_repo._get_client(user_id)

        # Get the segment boundary message first to find continuum_id and position
        segment_query = """
            SELECT continuum_id, created_at
            FROM messages
            WHERE id = %s
        """
        segment_row = db.execute_query(segment_query, (segment_message_id,))
        if not segment_row:
            logger.warning(f"Segment boundary message {segment_message_id} not found")
            return []

        continuum_id = segment_row[0]['continuum_id']
        segment_time = segment_row[0]['created_at']

        # Load all messages after this segment boundary
        messages_query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at > %s
                AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' = 'false')
                AND (metadata->>'is_segment_boundary' IS NULL OR metadata->>'is_segment_boundary' = 'false')
            ORDER BY created_at
        """

        message_rows = db.execute_query(messages_query, (str(continuum_id), segment_time))

        # Convert to Message objects and stop at next segment boundary
        messages = []
        for row in message_rows:
            metadata = self._parse_metadata(row.get('metadata', {}))

            # Stop if we hit another segment boundary
            if metadata.get('is_segment_boundary'):
                break

            messages.append(Message(
                id=row['id'],
                content=row['content'],
                role=row['role'],
                created_at=row['created_at'],
                metadata=metadata
            ))

        logger.debug(f"Loaded {len(messages)} messages for segment {segment_message_id}")
        return messages

    def _build_memory_context(
        self,
        messages: List[Message],
        user_id: str
    ) -> Dict[str, Any]:
        """
        Build memory context from referenced memories.

        Loads memory texts from database for all referenced memory IDs.
        Only includes memories user explicitly referenced (not surfaced).

        Args:
            messages: Chunk messages
            user_id: User ID

        Returns:
            Memory context dict with referenced memory texts (dict format)
        """
        referenced_ids = set()

        for msg in messages:
            metadata = getattr(msg, "metadata", {}) or {}

            # Only extract referenced memories (not surfaced)
            if isinstance(metadata.get("referenced_memories"), list):
                for ref in metadata["referenced_memories"]:
                    if isinstance(ref, str):
                        referenced_ids.add(ref)

        # Load memory texts from database
        memory_texts = {}
        if referenced_ids:
            memory_uuids = [UUID(ref_id) for ref_id in referenced_ids]
            memories = self.db.get_memories_by_ids(memory_uuids, user_id=user_id)
            for mem in memories:
                memory_texts[str(mem.id)] = mem.text

            logger.debug(f"Loaded {len(memories)} referenced memories for context")

        return {
            "memory_ids": list(referenced_ids),
            "referenced_memory_ids": sorted(referenced_ids),
            "memory_texts": memory_texts  # Dict format: {uuid: text}
        }

    @staticmethod
    def _parse_metadata(raw_metadata) -> Dict[str, Any]:
        """Parse message metadata from various formats."""
        if isinstance(raw_metadata, str):
            try:
                return json.loads(raw_metadata) if raw_metadata else {}
            except json.JSONDecodeError:
                return {}
        return dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
