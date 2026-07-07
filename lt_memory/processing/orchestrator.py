"""
Extraction orchestrator - high-level extraction workflows.

Two entry points:
- submit_segment_extraction(user_id, boundary_message_id): Self-contained segment
  extraction. Loads messages, submits batch, marks boundary. Called by collapse handler
  and extract_unprocessed_segments.
- extract_unprocessed_segments(): Safety-net sweep for collapsed segments where
  extraction failed or was never attempted. Runs on a 6-hour schedule.

All complexity lives in submit_segment_extraction. Callers are trivial.
"""
import json
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
from uuid import UUID

from cns.core.message import Message
from lt_memory.models import ProcessingChunk, MemoryContextSnapshot
from lt_memory.processing.extraction_engine import ExtractionEngine
from lt_memory.processing.execution_strategy import ExecutionStrategy, ImmediateExecutionStrategy
from lt_memory.db_access import LTMemoryDB
from lt_memory.llm_routing import uses_anthropic_batch_dialect
from utils.user_context import set_current_user_id, get_current_user_id, clear_user_context

if TYPE_CHECKING:
    from cns.infrastructure.continuum_repository import ContinuumRepository

logger = logging.getLogger(__name__)


class ExtractionOrchestrator:
    """
    High-level extraction workflow coordination.

    Single entry point: submit_segment_extraction owns the full lifecycle
    (load messages, build chunk, submit, mark boundary).

    Delegates to:
    - ExtractionEngine: Build payloads
    - ExecutionStrategy: Execute extraction (batch or immediate)
    - ContinuumRepository: Load messages, find segments
    - LTMemoryDB: Safety valve checks, memory context loading
    """

    def __init__(
        self,
        extraction_engine: ExtractionEngine,
        execution_strategy: ExecutionStrategy,
        continuum_repo: 'ContinuumRepository',
        db: LTMemoryDB,
        immediate_strategy: ImmediateExecutionStrategy = None
    ):
        self.extraction_engine = extraction_engine
        self.execution_strategy = execution_strategy
        self.continuum_repo = continuum_repo
        self.db = db
        self.immediate_strategy = immediate_strategy

    def submit_segment_extraction(
        self,
        user_id: str,
        boundary_message_id: str,
        force_immediate: bool = False
    ) -> bool:
        """
        Self-contained segment extraction: load, submit, mark.

        Owns the full lifecycle for extracting memories from a single segment:
        1. Query boundary message to get segment_id and position
        2. Load messages between this boundary and the next
        3. Build single ProcessingChunk (no chunking - segments are natural units)
        4. Submit to execution_strategy (or immediate_strategy if force_immediate)
        5. Mark memories_extracted=true on the boundary message

        Args:
            user_id: User ID
            boundary_message_id: UUID string of the segment boundary sentinel message
            force_immediate: If True, bypass batch and execute extraction inline
                via ImmediateExecutionStrategy. Used for manual segment collapse
                so memories are ready before the user's next conversation.

        Returns:
            True if extraction was submitted successfully

        Raises:
            RuntimeError: If boundary message not found or has no messages
        """
        db_client = self.continuum_repo._get_client(user_id)

        # Step 1: Query boundary row for segment_id, continuum_id, position
        boundary_row = db_client.execute_query("""
            SELECT id, continuum_id, created_at, metadata
            FROM messages
            WHERE id = %s
        """, (boundary_message_id,))

        if not boundary_row:
            raise RuntimeError(
                f"Boundary message {boundary_message_id} not found for user {user_id}"
            )

        row = boundary_row[0]
        continuum_id = row['continuum_id']
        boundary_time = row['created_at']
        metadata = self._parse_metadata(row.get('metadata', {}))
        segment_id = metadata.get('segment_id', boundary_message_id)
        segment_uuid = UUID(segment_id) if isinstance(segment_id, str) else segment_id

        # Step 2: Load messages after boundary, stop at next boundary
        message_rows = db_client.execute_query("""
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at > %s
                AND (metadata->>'system_notification' IS NULL
                     OR metadata->>'system_notification' = 'false')
            ORDER BY created_at
        """, (str(continuum_id), boundary_time))

        messages = []
        for msg_row in message_rows:
            msg_metadata = self._parse_metadata(msg_row.get('metadata', {}))

            # Stop at next segment boundary
            if msg_metadata.get('is_segment_boundary'):
                break

            messages.append(Message(
                id=msg_row['id'],
                content=msg_row['content'],
                role=msg_row['role'],
                created_at=msg_row['created_at'],
                metadata=msg_metadata
            ))

        if not messages:
            logger.warning(
                f"No messages found for segment {segment_id} "
                f"(boundary: {boundary_message_id})"
            )
            return False

        # Step 3: Build single ProcessingChunk (full segment, no chunking)
        chunk = ProcessingChunk.from_conversation_messages(
            messages,
            chunk_index=0,
            segment_id=segment_uuid
        )
        chunk.memory_context_snapshot = self._build_memory_context(messages, user_id)

        # Step 4: Submit via execution strategy (immediate when forced or batch dialect unavailable)
        strategy = self.execution_strategy
        if force_immediate and self.immediate_strategy is not None:
            strategy = self.immediate_strategy
            logger.info(
                f"Using immediate extraction for segment {segment_id} "
                f"(manual collapse — skipping batch)"
            )
        elif self.immediate_strategy is not None and not uses_anthropic_batch_dialect("extraction"):
            strategy = self.immediate_strategy
            logger.info(
                f"Using immediate extraction for segment {segment_id} "
                f"(batch dialect unavailable — skipping batch)"
            )
        batch_id = strategy.execute_extraction(user_id, [chunk])

        # Step 5: Mark boundary as extracted
        db_client.execute_query("""
            UPDATE messages
            SET metadata = jsonb_set(metadata, '{memories_extracted}', 'true')
            WHERE id = %s
        """, (boundary_message_id,))

        logger.info(
            f"Submitted segment {segment_id} for extraction "
            f"(batch: {batch_id}, {len(messages)} messages)"
        )
        return True

    def extract_unprocessed_segments(self, user_id: str = None) -> Dict[str, Any]:
        """
        Safety-net sweep for collapsed segments where extraction failed.

        Finds all collapsed segments with memories_extracted != true and
        submits each via submit_segment_extraction. Per-segment error isolation
        ensures one bad segment doesn't block the rest.

        Args:
            user_id: Optional specific user. If None, processes all users.

        Returns:
            Extraction statistics
        """
        logger.info(
            f"Starting unprocessed segment extraction sweep"
            f"{f' for user {user_id}' if user_id else ''}"
        )

        users = (
            [{"id": user_id}] if user_id
            else self.db.get_users_with_memory_enabled()
        )
        results = {"segments_submitted": 0, "users_processed": 0, "errors": []}

        for user in users:
            uid = str(user["id"])
            try:
                # Safety valve: skip users with pending batches
                pending_batches = self.db.get_pending_batches_for_user(uid)
                if pending_batches:
                    logger.info(
                        f"Skipping user {uid}: {len(pending_batches)} pending extraction batches"
                    )
                    continue

                # Find collapsed segments needing extraction
                failed_segments = self.continuum_repo.find_failed_extraction_segments(uid)
                if not failed_segments:
                    continue

                logger.info(f"Found {len(failed_segments)} unprocessed segments for user {uid}")
                set_current_user_id(uid)

                for segment in failed_segments:
                    # Increment attempt counter before expensive work (persists via jsonb_set)
                    attempts = segment.get('extraction_attempts', 0)
                    db_client = self.continuum_repo._get_client(uid)
                    db_client.execute_returning("""
                        UPDATE messages
                        SET metadata = jsonb_set(metadata, '{extraction_attempts}', to_jsonb(%s))
                        WHERE id = %s
                            AND metadata->>'is_segment_boundary' = 'true'
                        RETURNING id
                    """, (attempts + 1, segment['message_id']))

                    try:
                        if self.submit_segment_extraction(uid, segment['message_id']):
                            results["segments_submitted"] += 1
                    except Exception as e:
                        logger.error(
                            f"Error extracting segment {segment.get('segment_id', '?')} "
                            f"for user {uid} (attempt {attempts + 1}): {e}",
                            exc_info=True
                        )
                        results["errors"].append(str(e))

                results["users_processed"] += 1

            except Exception as e:
                logger.error(f"Error processing user {uid}: {e}", exc_info=True)
                results["errors"].append(str(e))
            finally:
                try:
                    if get_current_user_id() == uid:
                        clear_user_context()
                except Exception:
                    pass

        logger.info(
            f"Unprocessed segment sweep complete: "
            f"{results['segments_submitted']} segments submitted"
        )
        return results

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_memory_context(
        self,
        messages: list[Message],
        user_id: str
    ) -> MemoryContextSnapshot:
        """
        Build memory context from referenced and pinned memories.

        Loads memory texts from database for all referenced memory IDs.
        Also collects pinned memory IDs (8-char) for importance boosting.

        Args:
            messages: Chunk messages
            user_id: User ID

        Returns:
            Memory context dict with:
            - memory_ids: Full UUIDs of referenced memories
            - referenced_memory_ids: Sorted full UUIDs (for extraction context)
            - memory_texts: Dict of {uuid: text}
            - pinned_short_ids: Deduplicated 8-char IDs for importance boost
        """
        referenced_ids = set()
        pinned_short_ids = set()

        for msg in messages:
            metadata = getattr(msg, "metadata", {}) or {}

            # Extract referenced memories (explicit LLM references)
            if isinstance(metadata.get("referenced_memories"), list):
                for ref in metadata["referenced_memories"]:
                    if isinstance(ref, str):
                        referenced_ids.add(ref)

            # Extract pinned memory IDs (8-char short IDs from retention)
            if isinstance(metadata.get("pinned_memory_ids"), list):
                for pin_id in metadata["pinned_memory_ids"]:
                    if isinstance(pin_id, str) and pin_id:
                        pinned_short_ids.add(pin_id.lower())

        # Load memory texts from database
        memory_texts = {}
        if referenced_ids:
            memory_uuids = [UUID(ref_id) for ref_id in referenced_ids]
            memories = self.db.get_memories_by_ids(memory_uuids, user_id=user_id)
            for mem in memories:
                memory_texts[str(mem.id)] = mem.text

            logger.debug(f"Loaded {len(memories)} referenced memories for context")

        if pinned_short_ids:
            logger.debug(f"Collected {len(pinned_short_ids)} unique pinned memory IDs")

        return {
            "memory_ids": list(referenced_ids),
            "referenced_memory_ids": sorted(referenced_ids),
            "memory_texts": memory_texts,
            "pinned_short_ids": sorted(pinned_short_ids),
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
