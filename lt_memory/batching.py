"""
Batch orchestration for memory extraction and relationship classification.

Thin coordinator that manages Anthropic Batch API workflows and delegates
processing to specialized modules (extraction, linking, vector_ops, db_access).
"""
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from uuid import UUID, uuid4

import anthropic
from cns.core.message import Message

from lt_memory.models import (
    ProcessingChunk,
    ExtractionBatch,
    PostProcessingBatch
)
from config.config import BatchingConfig
from lt_memory.db_access import LTMemoryDB
from lt_memory.extraction import ExtractionService
from lt_memory.linking import LinkingService
from lt_memory.vector_ops import VectorOps
from utils.timezone_utils import utc_now
from auth import set_current_user_id
from utils.user_context import clear_user_context
from auth.database import AuthDatabase

logger = logging.getLogger(__name__)


class BatchingService:
    """
    Orchestrates memory extraction, batch processing, and relationship classification.

    Thin coordinator responsible for:
    - Loading continuum messages and building chunks
    - Submitting/polling Anthropic Batch API requests
    - Delegating processing to specialized modules
    """

    def __init__(
        self,
        config: BatchingConfig,
        db: LTMemoryDB,
        extraction_service: ExtractionService,
        linking_service: LinkingService,
        vector_ops: VectorOps,
        anthropic_client: anthropic.Anthropic,
        conversation_repo,
        llm_provider
    ):
        self.config = config
        self.db = db
        self.extraction = extraction_service
        self.linking = linking_service
        self.vector_ops = vector_ops
        self.anthropic_client = anthropic_client
        self.conversation_repo = conversation_repo
        self.llm_provider = llm_provider
        self.auth_db = AuthDatabase()

        # Load prompts
        self.extraction_prompt = self._load_prompt("memory_extraction_system.txt")
        self.relationship_prompt = self._load_prompt("memory_relationship_classification.txt")

        logger.info("BatchingService initialized")

    def _load_prompt(self, filename: str) -> str:
        """
        Load system prompt from file.

        Args:
            filename: Prompt filename in config/prompts/ directory

        Returns:
            Prompt text

        Raises:
            FileNotFoundError: If prompt file not found (prompts are required configuration)
        """
        path = Path(f"config/prompts/{filename}")
        if not path.exists():
            raise FileNotFoundError(
                f"Required prompt file not found: {path}. "
                f"Prompts are system configuration, not optional features."
            )
        return path.read_text()

    # ============================================================================
    # Entry Points - Extraction Orchestration
    # ============================================================================

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
        logger.info(f"Starting extraction failure retry sweep{f' for user {user_id}' if user_id else ''}")

        users = [{"id": user_id}] if user_id else self.auth_db.get_users_with_memory_enabled()
        results = {"segments_retried": 0, "users_processed": 0, "errors": []}

        for user in users:
            uid = str(user["id"])
            try:
                # Find collapsed segments that failed extraction
                failed_segments = self._get_failed_extraction_segments(uid)

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
                clear_user_context()

        logger.info(f"Extraction failure retry sweep complete: {results['segments_retried']} segments retried")
        return results

    def run_boot_extraction(self) -> Dict[str, Any]:
        """
        Run on-boot extraction sweep.

        Returns:
            Boot extraction statistics
        """
        logger.info("Starting on-boot extraction sweep")

        users = self.auth_db.get_users_with_memory_enabled()
        results = {"users_checked": len(users), "batches_submitted": 0, "users_skipped": 0, "errors": []}

        for user in users:
            uid = str(user["id"])
            try:
                # Safety valve: skip if pending batches
                if self.db.get_pending_extraction_batches_for_user(uid):
                    results["users_skipped"] += 1
                    continue

                set_current_user_id(uid)
                chunks = self._load_conversation_chunks(uid)

                if chunks:
                    batch_id = self._submit_extraction_batch(uid, chunks)
                    if batch_id:
                        results["batches_submitted"] += 1
            except Exception as e:
                logger.error(f"Error in boot extraction for {uid}: {e}", exc_info=True)
                results["errors"].append(str(e))
            finally:
                clear_user_context()

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
        segment_chunk_size and submits to Batch API (or executes immediately
        if in fallback mode). No minimum message requirement for segments.

        Args:
            user_id: User ID
            messages: Segment messages (already filtered, no summaries/boundaries)
            segment_id: Segment UUID for tracking

        Returns:
            True if submission succeeded

        Raises:
            Exception: If batch submission fails
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

        logger.info(f"Built {len(chunks)} chunks from segment {segment_id} ({len(messages)} messages)")

        # Submit via existing batch infrastructure
        set_current_user_id(user_id)
        try:
            batch_id = self._submit_extraction_batch(user_id, chunks)

            if batch_id:
                logger.info(f"Submitted segment {segment_id} for extraction (batch: {batch_id}, {len(chunks)} chunks)")
                return True
            else:
                logger.warning(f"Failed to submit segment {segment_id} for extraction")
                return False
        finally:
            clear_user_context()

    # ============================================================================
    # Batch Polling - APScheduler Jobs
    # ============================================================================

    def poll_extraction_batches(self) -> Dict[str, int]:
        """
        Poll pending extraction batches (1-minute interval).

        Uses database-driven approach: queries for users with pending batches,
        then processes each user's batches with proper security scoping.

        Returns:
            Polling statistics
        """
        stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        # Get users with pending batches (auth DB query)
        users_with_pending = self.db.get_users_with_pending_extraction_batches()

        if not users_with_pending:
            return stats

        logger.info(f"Polling extraction batches for {len(users_with_pending)} users")

        for user_id in users_with_pending:
            # Get batches for THIS user with proper scoping
            pending = self.db.get_pending_extraction_batches_for_user(user_id)
            stats["checked"] += len(pending)

            if not pending:
                continue

            # Group by batch_id to minimize API calls
            batch_groups = {}
            for batch in pending:
                if batch.batch_id not in batch_groups:
                    batch_groups[batch.batch_id] = []
                batch_groups[batch.batch_id].append(batch)

            for batch_id, batches in batch_groups.items():
                # Check expiration
                if batches[0].expires_at and utc_now() > batches[0].expires_at:
                    for b in batches:
                        self.db.update_extraction_batch_status(
                            b.id, "expired", user_id=user_id
                        )
                    stats["expired"] += len(batches)
                    continue

                # Query Anthropic
                batch_info = self.anthropic_client.beta.messages.batches.retrieve(batch_id)

                if batch_info.processing_status == "ended":
                    for b in batches:
                        try:
                            if self._process_extraction_result(batch_id, b):
                                stats["completed"] += 1
                            else:
                                stats["failed"] += 1
                        except ValueError as e:
                            logger.error(f"Failed to process extraction result for batch {b.id}: {e}")
                            # Update batch status to prevent infinite retry loop
                            self.db.update_extraction_batch_status(
                                b.id, "failed", error_message=str(e), user_id=user_id
                            )
                            stats["failed"] += 1
                        except Exception as e:
                            logger.error(f"Unexpected error processing extraction result for batch {b.id}: {e}", exc_info=True)
                            # Update batch status to prevent infinite retry loop
                            self.db.update_extraction_batch_status(
                                b.id, "failed", error_message=str(e), user_id=user_id
                            )
                            stats["failed"] += 1
                elif batch_info.processing_status == "in_progress":
                    for b in batches:
                        if b.status != "processing":
                            self.db.update_extraction_batch_status(
                                b.id, "processing", user_id=user_id
                            )
                elif batch_info.processing_status in ("canceling", "canceled"):
                    for b in batches:
                        self.db.update_extraction_batch_status(
                            b.id, "cancelled", error_message="Cancelled", user_id=user_id
                        )
                    stats["failed"] += len(batches)

        logger.info(f"Extraction polling: {stats['completed']} completed, {stats['failed']} failed")
        return stats

    def poll_linking_batches(self) -> Dict[str, int]:
        """
        Poll pending relationship batches (1-minute interval).

        Uses database-driven approach: queries for users with pending batches,
        then processes each user's batches with proper security scoping.

        Returns:
            Polling statistics
        """
        stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        # Get users with pending batches (auth DB query)
        users_with_pending = self.db.get_users_with_pending_relationship_batches()

        if not users_with_pending:
            return stats

        logger.info(f"Polling relationship batches for {len(users_with_pending)} users")

        for user_id in users_with_pending:
            try:
                # Get batches for THIS user with proper scoping
                pending = self.db.get_pending_relationship_batches_for_user(user_id)
                stats["checked"] += len(pending)

                if not pending:
                    continue

                # Group by batch_id
                batch_groups = {}
                for batch in pending:
                    if batch.batch_id not in batch_groups:
                        batch_groups[batch.batch_id] = []
                    batch_groups[batch.batch_id].append(batch)

                for batch_id, batches in batch_groups.items():
                    try:
                        # Check expiration
                        if batches[0].expires_at and utc_now() > batches[0].expires_at:
                            for b in batches:
                                self.db.update_relationship_batch_status(
                                    b.id, "expired", user_id=user_id
                                )
                            stats["expired"] += len(batches)
                            continue

                        # Query Anthropic
                        batch_info = self.anthropic_client.beta.messages.batches.retrieve(batch_id)

                        if batch_info.processing_status == "ended":
                            for b in batches:
                                # Route to correct processor based on batch type
                                if b.batch_type == "relationship_classification":
                                    success = self._process_relationship_result(batch_id, b)
                                elif b.batch_type == "consolidation":
                                    success = self._process_consolidation_result(batch_id, b)
                                elif b.batch_type == "consolidation_review":
                                    success = self._process_consolidation_review_result(batch_id, b)
                                else:
                                    logger.error(f"Unknown batch type: {b.batch_type}")
                                    success = False

                                if success:
                                    stats["completed"] += 1
                                else:
                                    stats["failed"] += 1
                        elif batch_info.processing_status == "in_progress":
                            for b in batches:
                                if b.status != "processing":
                                    self.db.update_relationship_batch_status(
                                        b.id, "processing", user_id=user_id
                                    )
                        elif batch_info.processing_status in ("canceling", "canceled"):
                            for b in batches:
                                self.db.update_relationship_batch_status(
                                    b.id, "cancelled", error_message="Cancelled", user_id=user_id
                                )
                            stats["failed"] += len(batches)
                    except Exception as e:
                        logger.error(f"Error polling batch {batch_id}: {e}", exc_info=True)
                        stats["failed"] += len(batches)

            except Exception as e:
                logger.error(f"Error polling batches for user {user_id}: {e}", exc_info=True)

        logger.info(f"Relationship polling: {stats['completed']} completed, {stats['failed']} failed")
        return stats

    # ============================================================================
    # Continuum Loading
    # ============================================================================

    def _get_failed_extraction_segments(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Query for collapsed segments where memory extraction failed.

        Args:
            user_id: User ID

        Returns:
            List of segment dicts with segment_id and message_id

        Raises:
            Exception: If query fails
        """
        segments = self.conversation_repo.find_failed_extraction_segments(user_id)
        logger.debug(f"Found {len(segments)} failed extraction segments for user {user_id}")
        return segments

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
        db = self.conversation_repo._get_client(user_id)

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
        continuum = self.conversation_repo.get_continuum(user_id)
        if not continuum:
            raise RuntimeError(f"Continuum not found for user {user_id}. Continuum should be created during signup.")

        # Get last extraction timestamp
        user_details = self.auth_db.get_user_by_id(user_id)
        last_run = user_details.get('daily_manipulation_last_run')

        # Load messages since last run
        db_client = self.conversation_repo._get_client(user_id)
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

            # Build memory context for this chunk (only referenced memories)
            memory_context = self._build_memory_context(chunk_messages, user_id)
            chunk.memory_context_snapshot = memory_context

            chunks.append(chunk)

        logger.info(f"Built {len(chunks)} chunks for user {user_id}")
        return chunks

    def _parse_metadata(self, raw_metadata) -> Dict[str, Any]:
        """Parse message metadata from various formats."""
        if isinstance(raw_metadata, str):
            try:
                return json.loads(raw_metadata) if raw_metadata else {}
            except json.JSONDecodeError:
                return {}
        return dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

    def _build_memory_context(self, messages: List[Message], user_id: str) -> Dict[str, Any]:
        """
        Build memory context from referenced memories only.

        Only includes memories the user explicitly referenced (not surfaced ones,
        which could be unrelated and cause context explosion).

        Args:
            messages: Chunk messages
            user_id: User ID

        Returns:
            Memory context dict with referenced memory texts
        """
        referenced_ids = set()

        for msg in messages:
            metadata = getattr(msg, "metadata", {}) or {}

            # Only extract referenced memories (not surfaced)
            if isinstance(metadata.get("referenced_memories"), list):
                for ref in metadata["referenced_memories"]:
                    if isinstance(ref, str):
                        referenced_ids.add(ref)

        # Load memory texts (this automatically tracks access at DB layer)
        memory_texts = {}

        if referenced_ids:
            # Convert string UUIDs to UUID objects
            memory_uuids = [UUID(ref_id) for ref_id in referenced_ids]
            memories = self.db.get_memories_by_ids(memory_uuids, user_id=user_id)
            for mem in memories:
                memory_texts[str(mem.id)] = mem.text

            logger.debug(f"Loaded {len(memories)} referenced memories for context")

        return {
            "memory_ids": list(referenced_ids),
            "referenced_memory_ids": sorted(referenced_ids),
            "memory_texts": memory_texts
        }

    # ============================================================================
    # Batch Submission
    # ============================================================================

    def _submit_extraction_batch(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> Optional[str]:
        """
        Submit extraction batch to Anthropic or execute immediately if in fallback mode.

        Can submit multiple chunks as a single batch. Next extraction will pick up
        messages that arrived after last_consolidation_run timestamp.

        Args:
            user_id: User ID
            chunks: Processing chunks

        Returns:
            Batch ID if successful, None if no chunks to process

        Raises:
            Exception: If Anthropic API call fails
        """
        # Check if we should bypass batching (emergency fallback mode)
        if self.llm_provider._is_failover_active():
            logger.warning(f"🔄 Bypassing Anthropic Batch API for user {user_id} - executing {len(chunks)} chunks immediately")
            return self._execute_extraction_immediately(user_id, chunks)

        requests = []
        chunk_request_mapping = []  # Track which chunks got requests

        for chunk in chunks:
            # Get UUID mapping from extraction service
            payload = self.extraction.build_extraction_payload(chunk)
            short_to_uuid = payload.get("short_to_uuid", {})

            messages = self._build_batch_messages(chunk)
            if not messages:
                continue

            request = {
                "custom_id": f"{user_id}_{chunk.chunk_index}",
                "params": {
                    "model": self.extraction.config.extraction_model,
                    "max_tokens": self.extraction.config.max_extraction_tokens,
                    "temperature": self.extraction.config.extraction_temperature,
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "system": [{
                        "type": "text",
                        "text": self.extraction_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    "messages": messages
                }
            }

            requests.append(request)
            chunk_request_mapping.append((chunk, len(requests) - 1, payload))

        if not requests:
            return None

        # Submit to Anthropic
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)
        expires_at = utc_now() + timedelta(hours=self.config.batch_expiry_hours)

        # Store batch records with UUID mapping (using correct chunk-to-request mapping)
        for chunk, request_idx, payload in chunk_request_mapping:
            batch_record = ExtractionBatch(
                batch_id=batch.id,
                custom_id=f"{user_id}_{chunk.chunk_index}",
                user_id=user_id,
                chunk_index=chunk.chunk_index,
                request_payload=requests[request_idx],
                chunk_metadata={
                    "message_count": len(chunk.messages),
                    "short_to_uuid": payload.get("short_to_uuid", {})
                },
                memory_context=chunk.memory_context_snapshot,
                status="submitted",
                created_at=utc_now(),
                submitted_at=utc_now(),
                expires_at=expires_at
            )
            self.db.create_extraction_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted batch {batch.id} for user {user_id}: {len(requests)} chunks")
        return batch.id

    def _execute_extraction_immediately(
        self,
        user_id: str,
        chunks: List[ProcessingChunk]
    ) -> Optional[str]:
        """
        Execute extraction immediately using OpenAI fallback (bypass Anthropic Batch API).

        Called when emergency fallback mode is active.

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

        try:
            for chunk in chunks:
                # Build extraction payload
                payload = self.extraction.build_extraction_payload(chunk)
                short_to_uuid = payload.get("short_to_uuid", {})
                messages = self._build_batch_messages(chunk)

                if not messages:
                    continue

                # Call LLM directly (routes to OpenAI fallback)
                response = self.llm_provider.generate_response(
                    messages=messages,
                    system_override=self.extraction_prompt
                )

                # Extract text from response
                response_text = self.llm_provider.extract_text_content(response)

                # Process extraction result
                result = self.extraction.process_batch_result(
                    response_text=response_text,
                    memory_context=chunk.memory_context_snapshot or {},
                    short_to_uuid=short_to_uuid
                )

                memories = result.memories
                linking_pairs = result.linking_pairs

                # Store memories with embeddings
                if memories:
                    memory_ids = self.vector_ops.store_memories_with_embeddings(
                        memories
                    )

                    # Persist entities and create links
                    self._persist_entities_for_memories(user_id, memory_ids)

                    # Trigger relationship classification with linking hints
                    self._trigger_relationship_classification(user_id, memory_ids, linking_pairs)

                    total_memories_stored += len(memory_ids)
                    logger.info(f"Immediate extraction chunk {chunk.chunk_index}: {len(memory_ids)} memories stored")

            # Update timestamp to prevent reprocessing
            if total_memories_stored > 0:
                self.auth_db.update_user_manipulation_timestamp(user_id, 'daily')
                logger.info(f"Immediate extraction complete for user {user_id}: {total_memories_stored} total memories")

            return f"bypass_{uuid4()}"

        finally:
            clear_user_context()

    def _build_batch_messages(self, chunk: ProcessingChunk) -> List[Dict[str, Any]]:
        """Build Anthropic messages from chunk."""
        messages = []

        # Add memory context if available
        if chunk.memory_context_snapshot and chunk.memory_context_snapshot.get("memory_texts"):
            context_lines = ["EXISTING MEMORIES (referenced in this continuum):"]
            for mem_id, text in chunk.memory_context_snapshot["memory_texts"].items():
                context_lines.append(f"- [{mem_id}] {text}")
            context_lines.append("\nNow extract memories from the following continuum:")
            messages.append({"role": "user", "content": "\n".join(context_lines)})

        # Add continuum messages
        for msg in chunk.messages:
            metadata = getattr(msg, "metadata", {}) or {}

            if metadata.get("system_notification"):
                continue
            if msg.role == "tool":
                continue

            if msg.role == "user":
                content = self._extract_content(msg)
                if content:
                    messages.append({"role": "user", "content": content})
            elif msg.role == "assistant":
                content = self._extract_content(msg)
                if content or not metadata.get("has_tool_calls"):
                    messages.append({"role": "assistant", "content": content or ""})

        # Add extraction instructions as final user message
        extraction_instruction = "Extract NEW memories from the above continuum following all extraction principles. Return as JSON array."

        if messages:
            if messages[-1]["role"] == "assistant":
                messages.append({"role": "user", "content": extraction_instruction})
            else:
                # Last message is user - append instruction to it
                messages[-1]["content"] += f"\n\n{extraction_instruction}"

        return messages

    def _extract_content(self, message: Message) -> str:
        """Extract text content from message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            parts = [
                item.get("text", "")
                for item in message.content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return " ".join(parts)
        return str(message.content)

    # ============================================================================
    # Result Processing
    # ============================================================================

    def _process_extraction_result(self, batch_id: str, batch: ExtractionBatch) -> bool:
        """
        Process extraction batch result.

        Delegates to extraction service for parsing, then vector_ops for storage.

        Raises:
            Exception: If result processing or storage fails
        """
        # Retrieve result from Anthropic
        for result in self.anthropic_client.beta.messages.batches.results(batch_id):
            if result.custom_id != batch.custom_id:
                continue

            if result.result.type == "succeeded":
                # Extract text from response
                response_text = "\n".join([
                    block.text for block in result.result.message.content
                    if block.type == "text"
                ])

                # Delegate to extraction service for parsing with UUID remapping
                set_current_user_id(batch.user_id)
                try:
                    short_to_uuid = batch.chunk_metadata.get("short_to_uuid", {}) if batch.chunk_metadata else {}

                    result = self.extraction.process_batch_result(
                        response_text=response_text,
                        memory_context=batch.memory_context or {},
                        short_to_uuid=short_to_uuid
                    )

                    memories = result.memories
                    linking_pairs = result.linking_pairs

                    # Delegate to vector_ops for storage with embeddings
                    memory_ids = []
                    if memories:
                        memory_ids = self.vector_ops.store_memories_with_embeddings(
                            memories
                        )

                        # Persist entities and create memory→entity links
                        self._persist_entities_for_memories(batch.user_id, memory_ids)

                        # Trigger relationship classification with linking hints
                        self._trigger_relationship_classification(batch.user_id, memory_ids, linking_pairs)

                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: {len(memory_ids)} stored")

                        # Update last extraction timestamp to prevent reprocessing
                        self.auth_db.update_user_manipulation_timestamp(batch.user_id, 'daily')
                        logger.debug(f"Updated daily_manipulation_last_run timestamp for user {batch.user_id}")
                    else:
                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: no memories extracted")

                    # Delete batch record - processing complete
                    self.db.delete_extraction_batch(batch.id, user_id=batch.user_id)
                    logger.debug(f"Deleted extraction batch {batch.id} after storing {len(memory_ids)} memories")

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

    def _persist_entities_for_memories(
        self,
        user_id: str,
        memory_ids: List[UUID]
    ) -> None:
        """
        Extract and persist entities for newly created memories.

        Extracts entities from ALL memories (not just hubs) and creates persistent
        entity records with memory→entity links. No clustering at ingestion -
        entities are stored as-is. Monthly GC handles merging with LLM review.

        Background operation: failures don't prevent memory storage but are logged.

        Args:
            user_id: User ID
            memory_ids: Newly stored memory UUIDs
        """
        if not memory_ids:
            return

        try:
            from lt_memory.entity_extraction import EntityExtractor

            entity_extractor = EntityExtractor()

            # Get all memories from batch
            memories = self.db.get_memories_by_ids(memory_ids, user_id=user_id)
            if not memories:
                return

            # Extract entities with types from each memory independently
            total_entities_created = 0
            total_links_created = 0

            for memory in memories:
                # Extract entities from this memory's text
                entities_with_types = entity_extractor.extract_entities_with_types(memory.text)

                if not entities_with_types:
                    continue

                # Deduplicate entities within this memory (same name + type)
                unique_entities = {}  # (name, type) → first occurrence
                for entity_name, entity_type in entities_with_types:
                    key = (entity_name, entity_type)
                    if key not in unique_entities:
                        unique_entities[key] = (entity_name, entity_type)

                # Persist each unique entity and link to this memory
                for entity_name, entity_type in unique_entities.values():
                    # Get entity embedding from spaCy
                    entity_embedding = entity_extractor.nlp(entity_name).vector.tolist()

                    # Get or create entity (unique on user_id + name + type)
                    entity = self.db.get_or_create_entity(
                        name=entity_name,
                        entity_type=entity_type,
                        embedding=entity_embedding,
                        user_id=user_id
                    )

                    if entity:
                        # Link this memory to the entity
                        self.db.link_memory_to_entity(
                            memory_id=memory.id,
                            entity_id=entity.id,
                            entity_name=entity_name,
                            entity_type=entity_type,
                            user_id=user_id
                        )
                        total_links_created += 1

            logger.info(
                f"Persisted entities for {len(memories)} memories: "
                f"{total_links_created} memory→entity links created"
            )

            entity_extractor.cleanup()

        except Exception as e:
            logger.warning(
                f"Background entity persistence failed for user {user_id} (non-critical): {e}",
                exc_info=True
            )

    def _trigger_relationship_classification(self, user_id: str, memory_ids: List[UUID], linking_hints: List[Tuple[int, int]] = None):
        """
        Trigger relationship classification for new memories or execute immediately if in fallback mode.

        Delegates to linking service to find candidates and build payloads.

        Args:
            user_id: User ID
            memory_ids: List of newly created memory UUIDs
            linking_hints: List of (source_idx, target_idx) pairs from extraction hints

        Raises:
            Exception: If relationship classification submission fails
        """
        if not memory_ids:
            return

        all_pairs = []

        # Process extraction hints first (highest priority)
        if linking_hints:
            # Fetch all new memories once for efficiency
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

        # Then find similar existing memories (existing logic)
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

        if not all_pairs:
            return

        # Check if we should bypass batching (emergency fallback mode)
        if self.llm_provider._is_failover_active():
            logger.warning(f"🔄 Bypassing relationship batch for user {user_id} - executing {len(all_pairs)} classifications immediately")
            self._execute_relationship_classification_immediately(user_id, all_pairs)
            return

        # Build and submit relationship batch
        requests = []
        input_data = {}

        for idx, pair in enumerate(all_pairs):
            # Delegate to linking service to build payload
            payload = self.linking.build_classification_payload(
                pair["new_memory"],
                pair["similar_memory"]
            )

            custom_id = f"{user_id}_rel_{pair['new_memory_id'][:8]}_{idx}"

            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": self.config.relationship_model,
                    "max_tokens": self.config.relationship_max_tokens,
                    "temperature": self.config.relationship_temperature,
                    "system": payload["system_prompt"],
                    "messages": [{"role": "user", "content": payload["user_prompt"]}]
                }
            })

            input_data[pair["new_memory_id"]] = {
                "new_memory_id": pair["new_memory_id"],
                "similar_memory_id": pair["similar_memory_id"]
            }

        # Submit to Anthropic
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)
        expires_at = utc_now() + timedelta(hours=self.config.batch_expiry_hours)

        # Store batch record
        batch_record = PostProcessingBatch(
            batch_id=batch.id,
            batch_type="relationship_classification",
            user_id=user_id,
            request_payload={"requests": requests},  # Wrap list in dict
            input_data=input_data,
            items_submitted=len(requests),
            status="submitted",
            created_at=utc_now(),
            submitted_at=utc_now(),
            expires_at=expires_at
        )
        self.db.create_relationship_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted relationship batch {batch.id}: {len(requests)} classifications")

    def _execute_relationship_classification_immediately(
        self,
        user_id: str,
        all_pairs: List[Dict[str, Any]]
    ) -> None:
        """
        Execute relationship classification immediately using OpenAI fallback.

        Called when emergency fallback mode is active.

        Args:
            user_id: User ID
            all_pairs: List of memory pair dicts with new_memory and similar_memory
        """
        try:
            set_current_user_id(user_id)
            links_created = 0

            for pair in all_pairs:
                # Build classification payload
                payload = self.linking.build_classification_payload(
                    pair["new_memory"],
                    pair["similar_memory"]
                )

                # Call LLM directly (routes to OpenAI fallback)
                response = self.llm_provider.generate_response(
                    messages=[{"role": "user", "content": payload["user_prompt"]}],
                    system_override=payload["system_prompt"]
                )

                # Extract text from response
                response_text = self.llm_provider.extract_text_content(response)

                # Parse classification result
                try:
                    classification = json.loads(response_text)
                    rel_type = classification.get("relationship_type")

                    if rel_type and rel_type != "null":
                        # Create bidirectional link (user context set at line 1137)
                        if self.linking.create_bidirectional_link(
                            source_id=UUID(pair["new_memory_id"]),
                            target_id=UUID(pair["similar_memory_id"]),
                            link_type=rel_type,
                            confidence=classification.get("confidence", 0.9),
                            reasoning=classification.get("reasoning", "")
                        ):
                            links_created += 1
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in relationship classification response")
                    continue

            logger.info(f"Immediate relationship classification complete for user {user_id}: {links_created} links created")
            clear_user_context()

        except Exception as e:
            logger.error(f"Error in immediate relationship classification for {user_id}: {e}", exc_info=True)
            clear_user_context()

    def submit_consolidation_batch(self, user_id: str) -> Optional[str]:
        """
        Identify consolidation clusters and submit batch for processing or execute immediately if in fallback mode.

        Args:
            user_id: User ID

        Returns:
            Batch ID if successful, None if no clusters found

        Raises:
            Exception: If Anthropic API call fails
        """
        # Delegate to refinement service to identify clusters
        clusters = self.refinement.identify_consolidation_clusters()

        if not clusters:
            logger.info(f"No consolidation clusters found for user {user_id}")
            return None

        # Check if we should bypass batching (emergency fallback mode)
        if self.llm_provider._is_failover_active():
            logger.warning(f"🔄 Bypassing consolidation batch for user {user_id} - executing {len(clusters)} clusters immediately")
            self._execute_consolidation_immediately(user_id, clusters)
            return f"bypass_{uuid4()}"

        # Build batch requests
        requests = []
        input_data = {}

        for cluster in clusters:
            # Delegate to refinement service to build payload
            payload = self.refinement.build_consolidation_payload(cluster)

            custom_id = f"{user_id}_consol_{cluster.cluster_id}"

            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": "claude-sonnet-4-5-20250929",  # Reasoning model for consequential consolidation decisions
                    "max_tokens": 1000,
                    "temperature": 1.0,  # Must be 1.0 when thinking is enabled
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                    "system": payload["system_prompt"],
                    "messages": [{"role": "user", "content": payload["user_prompt"]}]
                }
            })

            input_data[cluster.cluster_id] = {
                "cluster_id": cluster.cluster_id,
                "memory_ids": payload["memory_ids"]
            }

        # Submit to Anthropic
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)
        expires_at = utc_now() + timedelta(hours=self.config.batch_expiry_hours)

        # Store batch record
        batch_record = PostProcessingBatch(
            batch_id=batch.id,
            batch_type="consolidation",
            user_id=user_id,
            request_payload={"requests": requests},  # Wrap list in dict
            input_data=input_data,
            items_submitted=len(requests),
            status="submitted",
            created_at=utc_now(),
            submitted_at=utc_now(),
            expires_at=expires_at
        )
        self.db.create_relationship_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted consolidation batch {batch.id}: {len(requests)} clusters")
        return batch.id

    def _execute_consolidation_immediately(
        self,
        user_id: str,
        clusters: List[Any]
    ) -> None:
        """
        Execute consolidation immediately using OpenAI fallback (bypass Anthropic Batch API).

        Performs single-pass consolidation without the two-phase review process used in batching.

        Args:
            user_id: User ID
            clusters: List of consolidation clusters
        """
        import statistics

        try:
            set_current_user_id(user_id)
            memories_consolidated = 0

            for cluster in clusters:
                # Build consolidation payload
                payload = self.refinement.build_consolidation_payload(cluster)

                # Call LLM directly (routes to OpenAI fallback)
                response = self.llm_provider.generate_response(
                    messages=[{"role": "user", "content": payload["user_prompt"]}],
                    system_override=payload["system_prompt"]
                )

                # Extract text from response
                response_text = self.llm_provider.extract_text_content(response)

                # Parse consolidation decision
                try:
                    decision = json.loads(response_text)
                    should_consolidate = decision.get("should_consolidate", False)

                    if not should_consolidate:
                        logger.debug(f"Skipping cluster {cluster.cluster_id}: {decision.get('reason')}")
                        continue

                    consolidated_text = decision.get("consolidated_text", "").strip()
                    if not consolidated_text:
                        logger.warning(f"Consolidation for {cluster.cluster_id} has no text, skipping")
                        continue

                    # Get memory IDs for this cluster
                    memory_ids = [UUID(mid) for mid in payload["memory_ids"]]
                    old_memories = self.db.get_memories_by_ids(memory_ids, user_id=user_id)

                    if not old_memories:
                        logger.warning(f"Could not load old memories for consolidation: {memory_ids}")
                        continue

                    # Calculate median importance
                    importance_scores = [m.importance_score for m in old_memories if m.importance_score]
                    median_importance = statistics.median(importance_scores) if importance_scores else 0.5

                    # Collect link bundles
                    all_inbound_links = []
                    all_outbound_links = []
                    all_entity_links = []

                    for memory in old_memories:
                        all_inbound_links.extend(memory.inbound_links)
                        all_outbound_links.extend(memory.outbound_links)
                        all_entity_links.extend(memory.entity_links)

                    # Deduplicate links (exclude self-references)
                    old_memory_id_strs = {str(mid) for mid in memory_ids}
                    unique_inbound = {
                        link['uuid']: link for link in all_inbound_links
                        if link['uuid'] not in old_memory_id_strs
                    }
                    unique_outbound = {
                        link['uuid']: link for link in all_outbound_links
                        if link['uuid'] not in old_memory_id_strs
                    }
                    unique_entities = {
                        link['uuid']: link for link in all_entity_links
                    }

                    # Create consolidated memory
                    from lt_memory.models import ExtractedMemory
                    consolidated_memory = ExtractedMemory(
                        text=consolidated_text,
                        importance_score=median_importance,
                        confidence=decision.get("confidence", 0.9),
                        consolidates_memory_ids=[str(mid) for mid in memory_ids]
                    )

                    # Store consolidated memory
                    new_ids = self.vector_ops.store_memories_with_embeddings(
                        [consolidated_memory]
                    )

                    if new_ids:
                        new_memory_id = new_ids[0]
                        new_memory_id_str = str(new_memory_id)

                        # Transfer link bundles
                        if unique_inbound or unique_outbound or unique_entities:
                            self.db.update_memory(new_memory_id, {
                                'inbound_links': list(unique_inbound.values()),
                                'outbound_links': list(unique_outbound.values()),
                                'entity_links': list(unique_entities.values())
                            }, user_id=user_id)

                        # Update memories linking TO old memories
                        source_memory_ids = {UUID(link['uuid']) for link in all_inbound_links if link['uuid'] not in old_memory_id_strs}

                        for source_memory_id in source_memory_ids:
                            source_memory = self.db.get_memory(source_memory_id, user_id=user_id)
                            if not source_memory:
                                continue

                            # Rewrite outbound_links: replace old_ids with new_id
                            updated_outbound = []
                            for outbound_link in source_memory.outbound_links:
                                if outbound_link['uuid'] in old_memory_id_strs:
                                    updated_link = outbound_link.copy()
                                    updated_link['uuid'] = new_memory_id_str
                                    updated_outbound.append(updated_link)
                                else:
                                    updated_outbound.append(outbound_link)

                            self.db.update_memory(source_memory_id, {
                                'outbound_links': updated_outbound
                            }, user_id=user_id)

                        # Archive old memories
                        for old_id in memory_ids:
                            self.db.archive_memory(old_id, user_id=user_id)

                        memories_consolidated += len(memory_ids)
                        logger.info(f"Consolidated {len(memory_ids)} memories into {new_memory_id}: {consolidated_text[:80]}...")

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON in consolidation response")
                    continue

            logger.info(f"Immediate consolidation complete for user {user_id}: {memories_consolidated} memories consolidated")
            clear_user_context()

        except Exception as e:
            logger.error(f"Error in immediate consolidation for {user_id}: {e}", exc_info=True)
            clear_user_context()

    def _process_relationship_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """
        Process relationship classification result.

        Delegates to linking service to create bidirectional links.
        """
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

                    # Try parsing as-is first
                    try:
                        classifications[result.custom_id] = json.loads(response_text)
                        continue
                    except json.JSONDecodeError as e:
                        # Model may have added explanation after JSON - extract just the JSON object
                        if "{" in response_text and "}" in response_text:
                            try:
                                start = response_text.index("{")
                                end = response_text.rindex("}") + 1
                                json_str = response_text[start:end]
                                classifications[result.custom_id] = json.loads(json_str)
                                logger.debug(f"Extracted JSON from response with extra text in {result.custom_id}")
                                continue
                            except (json.JSONDecodeError, ValueError):
                                pass

                        logger.error(f"Invalid JSON in {result.custom_id}: {e}")
                        logger.debug(f"Response text: {response_text[:500]}")

            if not classifications:
                self.db.update_relationship_batch_status(
                    batch.id,
                    "failed",
                    error_message="No valid results",
                    user_id=batch.user_id
                )
                return False

            # Delegate to linking service to create links
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

                # Delegate to linking service to create bidirectional link (user context set at line 1457)
                if self.linking.create_bidirectional_link(
                    source_id=UUID(pair_data["new_memory_id"]),
                    target_id=UUID(pair_data["similar_memory_id"]),
                    link_type=rel_type,
                    confidence=classification.get("confidence", 0.9),
                    reasoning=classification.get("reasoning", "")
                ):
                    links_created += 1

            clear_user_context()

            # Delete batch record - processing complete, links created
            self.db.delete_relationship_batch(batch.id, user_id=batch.user_id)
            logger.info(f"Relationship batch {batch_id}: {links_created} links created, batch record deleted")

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

    def _process_consolidation_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """
        Process consolidation batch result.

        Consolidates clusters by archiving old memories and creating new consolidated ones.
        """
        try:
            # Retrieve results from Anthropic
            consolidation_decisions = {}
            for result in self.anthropic_client.beta.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    response_text = "\n".join([
                        block.text for block in result.result.message.content
                        if block.type == "text"
                    ])
                    try:
                        consolidation_decisions[result.custom_id] = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in {result.custom_id}")

            if not consolidation_decisions:
                self.db.update_relationship_batch_status(
                    batch.id, "failed", error_message="No valid results", user_id=batch.user_id
                )
                return False

            # Process consolidation decisions - submit for final review
            set_current_user_id(batch.user_id)

            # Collect approved consolidations for final review
            review_candidates = []
            for custom_id, decision in consolidation_decisions.items():
                should_consolidate = decision.get("should_consolidate", False)
                if not should_consolidate:
                    logger.debug(f"Skipping cluster {custom_id}: {decision.get('reason')}")
                    continue

                consolidated_text = decision.get("consolidated_text", "").strip()
                if not consolidated_text:
                    logger.warning(f"Consolidation for {custom_id} has no text, skipping")
                    continue

                cluster_data = next(
                    (data for cluster_id, data in batch.input_data.items() if cluster_id in custom_id),
                    None
                )

                if cluster_data:
                    review_candidates.append({
                        "custom_id": custom_id,
                        "cluster_data": cluster_data,
                        "decision": decision
                    })

            if review_candidates:
                # Submit final review batch
                review_batch_id = self._submit_consolidation_review_batch(
                    batch.user_id,
                    review_candidates
                )

                if review_batch_id:
                    logger.info(
                        f"Submitted {len(review_candidates)} consolidations for final review "
                        f"(batch: {review_batch_id})"
                    )

            clear_user_context()

            # Delete batch record - processing complete, review batch submitted
            self.db.delete_relationship_batch(batch.id, user_id=batch.user_id)
            logger.info(
                f"Consolidation batch {batch_id}: {len(review_candidates)} sent for final review, batch record deleted"
            )

            return True

        except Exception as e:
            logger.error(f"Error processing consolidation result: {e}", exc_info=True)
            self.db.update_relationship_batch_status(
                batch.id,
                "failed",
                error_message=str(e),
                user_id=batch.user_id
            )
            return False

    def _submit_consolidation_review_batch(
        self,
        user_id: str,
        review_candidates: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Submit final consolidation review batch to Haiku.

        Uses minimal prompts with first-pass reasoning for final approval.

        Args:
            user_id: User ID
            review_candidates: List of dicts with cluster_data and first decision

        Returns:
            Batch ID if successful

        Raises:
            Exception: If Anthropic API call or database operations fail
        """
        requests = []
        input_data = {}

        system_prompt = "You are a final review system for memory consolidation. Verify that consolidations preserve information and improve clarity."

        for candidate in review_candidates:
            cluster_data = candidate["cluster_data"]
            first_decision = candidate["decision"]
            custom_id = candidate["custom_id"]

            # Build minimal review prompt
            memory_ids = cluster_data["memory_ids"]
            memories = self.db.get_memories_by_ids(
                [UUID(mid) for mid in memory_ids],
                user_id=user_id
            )

            original_texts = "\n".join([
                f"{i+1}. {m.text}"
                for i, m in enumerate(memories)
            ])

            user_prompt = f"""Original memories:
{original_texts}

Proposed consolidation: "{first_decision['consolidated_text']}"
Reasoning: {first_decision.get('reason', 'None provided')}

Approve consolidation?

Respond with JSON:
{{"approved": true/false, "reason": "Brief explanation"}}"""

            review_custom_id = f"{user_id}_review_{custom_id}"

            requests.append({
                "custom_id": review_custom_id,
                "params": {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 200,
                    "temperature": 0.0,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]
                }
            })

            input_data[review_custom_id] = {
                "original_custom_id": custom_id,
                "memory_ids": memory_ids,
                "consolidated_text": first_decision["consolidated_text"],
                "confidence": first_decision.get("confidence", 0.9)
            }

        # Submit to Anthropic (no caching)
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)
        expires_at = utc_now() + timedelta(hours=self.config.batch_expiry_hours)

        # Store batch record
        batch_record = PostProcessingBatch(
            batch_id=batch.id,
            batch_type="consolidation_review",
            user_id=user_id,
            request_payload={"requests": requests},  # Wrap list in dict
            input_data=input_data,
            items_submitted=len(requests),
            status="submitted",
            created_at=utc_now(),
            submitted_at=utc_now(),
            expires_at=expires_at
        )
        self.db.create_relationship_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted consolidation review batch {batch.id}: {len(requests)} reviews")
        return batch.id

    def _process_consolidation_review_result(
        self,
        batch_id: str,
        batch: PostProcessingBatch
    ) -> bool:
        """
        Process final consolidation review results and execute approved consolidations.
        """
        try:
            # Retrieve review results from Anthropic
            reviews = {}
            for result in self.anthropic_client.beta.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    response_text = "\n".join([
                        block.text for block in result.result.message.content
                        if block.type == "text"
                    ])
                    try:
                        reviews[result.custom_id] = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in review {result.custom_id}")

            if not reviews:
                self.db.update_relationship_batch_status(
                    batch.id, "failed", error_message="No valid review results", user_id=batch.user_id
                )
                return False

            # Execute approved consolidations
            set_current_user_id(batch.user_id)
            memories_consolidated = 0

            for custom_id, review in reviews.items():
                approved = review.get("approved", False)
                if not approved:
                    logger.info(f"Consolidation rejected: {review.get('reason')}")
                    continue

                # Get consolidation data from input_data
                consolidation_data = batch.input_data.get(custom_id)
                if not consolidation_data:
                    logger.warning(f"No consolidation data for {custom_id}")
                    continue

                memory_ids = [UUID(mid) for mid in consolidation_data["memory_ids"]]
                consolidated_text = consolidation_data["consolidated_text"]

                # Collect link bundles and metadata from old memories
                old_memories = self.db.get_memories_by_ids(memory_ids, user_id=batch.user_id)
                if not old_memories:
                    logger.warning(f"Could not load old memories for consolidation: {memory_ids}")
                    continue

                # Calculate median importance score from old memories
                import statistics
                importance_scores = [m.importance_score for m in old_memories if m.importance_score]
                median_importance = statistics.median(importance_scores) if importance_scores else 0.5

                # Collect all unique links from old memories (build thicker bundles)
                all_inbound_links = []
                all_outbound_links = []
                all_entity_links = []

                for memory in old_memories:
                    all_inbound_links.extend(memory.inbound_links)
                    all_outbound_links.extend(memory.outbound_links)
                    all_entity_links.extend(memory.entity_links)

                # Deduplicate links (exclude self-references to old memories)
                old_memory_id_strs = {str(mid) for mid in memory_ids}
                unique_inbound = {
                    link['uuid']: link for link in all_inbound_links
                    if link['uuid'] not in old_memory_id_strs
                }
                unique_outbound = {
                    link['uuid']: link for link in all_outbound_links
                    if link['uuid'] not in old_memory_id_strs
                }
                unique_entities = {
                    link['uuid']: link for link in all_entity_links
                }

                # Create consolidated memory with median importance
                from lt_memory.models import ExtractedMemory
                consolidated_memory = ExtractedMemory(
                    text=consolidated_text,
                    importance_score=median_importance,
                    confidence=consolidation_data.get("confidence", 0.9),
                    consolidates_memory_ids=[str(mid) for mid in memory_ids]
                )

                # Store consolidated memory
                new_ids = self.vector_ops.store_memories_with_embeddings(
                    [consolidated_memory]
                )

                if new_ids:
                    new_memory_id = new_ids[0]
                    new_memory_id_str = str(new_memory_id)

                    # Transfer link bundles to new consolidated memory
                    if unique_inbound or unique_outbound or unique_entities:
                        self.db.update_memory(new_memory_id, {
                            'inbound_links': list(unique_inbound.values()),
                            'outbound_links': list(unique_outbound.values()),
                            'entity_links': list(unique_entities.values())
                        }, user_id=batch.user_id)

                        logger.info(
                            f"Transferred links to consolidated memory {new_memory_id}: "
                            f"{len(unique_inbound)} inbound, {len(unique_outbound)} outbound, "
                            f"{len(unique_entities)} entities"
                        )

                    # Update all memories that were linking TO old memories
                    # Rewrite their outbound_links to point to new_memory_id
                    source_memory_ids = {UUID(link['uuid']) for link in all_inbound_links if link['uuid'] not in old_memory_id_strs}

                    for source_memory_id in source_memory_ids:
                        # Get the source memory
                        source_memory = self.db.get_memory(source_memory_id, user_id=batch.user_id)
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
                        }, user_id=batch.user_id)

                    # Archive the old memories
                    for old_id in memory_ids:
                        self.db.archive_memory(old_id, user_id=batch.user_id)

                    memories_consolidated += len(memory_ids)
                    logger.info(
                        f"Consolidated {len(memory_ids)} memories into {new_memory_id} "
                        f"(median importance: {median_importance:.3f}): {consolidated_text[:80]}..."
                    )

            clear_user_context()

            # Delete batch record - processing complete, memories consolidated
            self.db.delete_relationship_batch(batch.id, user_id=batch.user_id)
            logger.info(
                f"Consolidation review batch {batch_id}: {memories_consolidated} memories consolidated "
                f"({len(reviews) - memories_consolidated} rejected), batch record deleted"
            )

            return True

        except Exception as e:
            logger.error(f"Error processing consolidation review result: {e}", exc_info=True)
            self.db.update_relationship_batch_status(
                batch.id,
                "failed",
                error_message=str(e),
                user_id=batch.user_id
            )
            return False

    def cleanup(self):
        """
        Clean up batching service resources.

        Note: BatchingService primarily coordinates other services and holds
        references to external clients (anthropic_client, conversation_repo).
        These are managed elsewhere and will be garbage collected when the
        factory is destroyed. We don't null references because scheduler jobs
        may still be running during shutdown sequence.
        """
        # No-op: service dependencies are managed by factory lifecycle
        # Nulling references here would break in-flight scheduler jobs
        logger.debug("BatchingService cleanup completed (no-op)")
