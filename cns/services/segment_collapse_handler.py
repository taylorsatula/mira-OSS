"""
Segment collapse handler for processing timeout events.

Handles SessionTimeoutEvent by:
1. Finding the segment boundary sentinel
2. Loading segment messages
3. Generating summary with embedding (Continuity Engine)
4. Updating sentinel metadata
5. Triggering downstream processing (memory extraction, domain updates)
6. Extracting feedback signals (DIY reinforcement loop)
7. Running pattern synthesis if use-day threshold reached (every 7 use-days)
8. Portrait synthesis if use-day threshold reached (every 10 use-days)
"""
from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from lt_memory.factory import LTMemoryFactory
    from lt_memory.models import Memory
    from lt_memory.db_access import LTMemoryDB
    from cns.infrastructure.continuum_pool import ContinuumPool

from cns.core.events import SegmentTimeoutEvent, SegmentCollapsedEvent, ManifestUpdatedEvent
from cns.core.message import Message
from cns.services.segment_helpers import collapse_segment_sentinel
from cns.services.summary_generator import SummaryGenerator, SummaryResult, SummaryType
from cns.infrastructure.continuum_repository import ContinuumRepository
from clients.hybrid_embeddings_provider import HybridEmbeddingsProvider
from clients.valkey_client import get_valkey_client
from cns.integration.event_bus import EventBus
from utils.timezone_utils import utc_now, parse_time_string
from utils.user_context import set_current_user_id, get_current_user_id, clear_user_context

logger = logging.getLogger(__name__)

# Maximum collapse attempts before tombstoning a segment.
# Prevents infinite retry loops when a persistent failure (billing, DB schema,
# missing config) causes every attempt to fail after the expensive LLM call.
MAX_COLLAPSE_ATTEMPTS = 3


class SegmentCollapseHandler:
    """
    Handles segment collapse when timeout is reached.

    Subscribes to SessionTimeoutEvent and orchestrates the collapse pipeline:
    summary generation, embedding, sentinel update, downstream processing,
    and the DIY reinforcement loop (feedback extraction + pattern synthesis).
    """

    def __init__(
        self,
        continuum_repo: ContinuumRepository,
        summary_generator: SummaryGenerator,
        embeddings_provider: HybridEmbeddingsProvider,
        event_bus: EventBus,
        continuum_pool: ContinuumPool,
        lt_memory_factory: LTMemoryFactory,
    ):
        """
        Initialize collapse handler.

        Args:
            continuum_repo: Repository for loading/saving messages
            summary_generator: Service for generating segment summaries
            embeddings_provider: Provider for generating embeddings
            event_bus: Event bus for publishing events
            continuum_pool: Continuum pool for cache invalidation
            lt_memory_factory: LT_Memory factory for extraction
        """
        self.continuum_repo = continuum_repo
        self.summary_generator = summary_generator
        self.embeddings_provider = embeddings_provider
        self.event_bus = event_bus
        self.continuum_pool = continuum_pool
        self.lt_memory_factory = lt_memory_factory

        # Initialize feedback loop components (lazy - may not be available yet)
        self._assessment_extractor = None
        self._feedback_repo = None
        self._feedback_tracker = None
        self._synthesizer = None
        self._feedback_loop_initialized = False

        # Subscribe to timeout events
        self.event_bus.subscribe('SegmentTimeoutEvent', self.handle_timeout)
        logger.info("SegmentCollapseHandler subscribed to SegmentTimeoutEvent")

    def _init_feedback_loop(self) -> bool:
        """
        Lazy initialization of feedback loop components.

        Returns True if initialization successful, False otherwise.
        Retries on each call until successful — transient failures (missing
        prompt files, import issues) don't permanently kill the pipeline.
        """
        if self._feedback_loop_initialized:
            return True

        try:
            from cns.services.assessment_extractor import AssessmentExtractor
            from cns.infrastructure.feedback_repository import FeedbackRepository
            from cns.infrastructure.feedback_tracker import FeedbackTracker
            from cns.services.user_model_synthesizer import UserModelSynthesizer

            self._assessment_extractor = AssessmentExtractor()
            self._feedback_repo = FeedbackRepository()
            self._feedback_tracker = FeedbackTracker()
            self._synthesizer = UserModelSynthesizer(self._feedback_repo)

            self._feedback_loop_initialized = True
            logger.info("Feedback loop components initialized successfully")
            return True

        except Exception as e:
            logger.error(
                "USER MODEL PIPELINE INIT FAILED: %s — "
                "No assessment signals or synthesis will run until this is fixed. "
                "Will retry on next segment collapse.",
                e, exc_info=True
            )
            return False

    def handle_timeout(self, event: SegmentTimeoutEvent) -> None:
        """
        Event subscriber wrapper for segment timeout.

        Catches all exceptions so the event bus never sees failures.
        Segment remains active on failure and retries on the next timeout check.

        Args:
            event: SegmentTimeoutEvent with segment details
        """
        try:
            self.collapse_segment(event)
        except Exception as e:
            logger.error(
                f"COLLAPSE FAILURE: Segment {event.segment_id} collapse failed. "
                f"Segment will remain active and retry on next timeout check. "
                f"Operators should investigate if this persists. Error: {e}",
                exc_info=True
            )

    def collapse_segment(
        self,
        event: SegmentTimeoutEvent,
        force_immediate: bool = False
    ) -> Message:
        """
        Collapse a segment: generate summary, update sentinel, trigger downstream.

        Called by handle_timeout (event subscriber, swallows exceptions) and by the
        actions API (needs exceptions to propagate and the collapsed sentinel returned).

        Args:
            event: SegmentTimeoutEvent with segment details
            force_immediate: If True, run memory extraction inline instead of via
                batch. Used for manual collapse so memories are ready before the
                user's next conversation.

        Returns:
            Collapsed sentinel Message

        Raises:
            RuntimeError: On any collapse failure (sentinel not found, no messages,
                summary generation failed, etc.)
        """
        # Clear stale per-user cached fields (e.g. cumulative_activity_days)
        # before setting new identity — required when the timeout service
        # processes multiple users sequentially on the same thread.
        clear_user_context()
        set_current_user_id(event.user_id)

        logger.info(
            f"Processing timeout for segment {event.segment_id}, "
            f"continuum {event.continuum_id}, "
            f"inactive_duration={event.inactive_duration_minutes}min, "
            f"local_hour={event.local_hour}"
        )

        # Find the segment boundary sentinel
        sentinel = self._find_segment_sentinel(
            event.continuum_id,
            event.segment_id
        )

        if not sentinel:
            raise RuntimeError(
                f"Segment sentinel {event.segment_id} not found. "
                f"Data consistency violation - timeout event published for non-existent segment."
            )

        # Circuit breaker: stop retrying after MAX_COLLAPSE_ATTEMPTS
        attempts = sentinel.metadata.get('collapse_attempts', 0)
        if attempts >= MAX_COLLAPSE_ATTEMPTS:
            logger.critical(
                "Segment %s has failed %d collapse attempts — tombstoning to stop retry loop. "
                "Check previous COLLAPSE FAILURE logs for root cause.",
                event.segment_id, MAX_COLLAPSE_ATTEMPTS
            )
            # Force-collapse with tombstone so segment exits timeout queue
            tombstone = collapse_segment_sentinel(
                sentinel,
                summary="[Segment collapse failed after maximum retry attempts]",
                precis="[Collapse Failed]",
                display_title="[Collapse Failed]",
                embedding=[0.0] * 768,
                inactive_duration_minutes=event.inactive_duration_minutes,
                processing_failed=True,
                tools_used=sentinel.metadata.get('tools_used', []),
                segment_end_time=utc_now(),
                complexity_score=0.0
            )
            user_id = get_current_user_id()
            self.continuum_repo.save_message(tombstone, event.continuum_id, user_id)
            self.continuum_pool.invalidate()
            return tombstone

        # Increment attempt counter before expensive LLM call (persists to DB via jsonb_set)
        db = self.continuum_repo._get_client(get_current_user_id())
        db.execute_returning("""
            UPDATE messages
            SET metadata = jsonb_set(metadata, '{collapse_attempts}', to_jsonb(%s))
            WHERE id = %s
                AND metadata->>'is_segment_boundary' = 'true'
            RETURNING id
        """, (attempts + 1, str(sentinel.id)))

        # Load messages in segment (between this sentinel and next, or end of continuum)
        messages = self._load_segment_messages(
            event.continuum_id,
            sentinel
        )

        if not messages:
            raise RuntimeError(
                f"Segment {event.segment_id} has no committed messages yet. "
                f"This usually means the conversation is still being processed "
                f"(messages are committed after the full turn completes). "
                f"The segment will collapse automatically once the conversation finishes."
            )

        # Generate summary and embedding (raises on failure)
        result, embedding = self._generate_summary(
            messages,
            sentinel,
            event.continuum_id
        )

        # Extract tools used from actual messages (not sentinel metadata)
        tools_used = self._extract_tools_from_messages(messages)

        # Set segment_end_time from last message (guaranteed to exist at this point)
        segment_end_time = messages[-1].created_at

        # Collapse sentinel (returns new Message with collapsed state)
        collapsed_sentinel = collapse_segment_sentinel(
            sentinel,
            summary=result.synopsis,
            precis=result.precis,
            display_title=result.display_title,
            embedding=embedding,
            inactive_duration_minutes=event.inactive_duration_minutes,
            processing_failed=False,  # Always False - failures raise instead of degrading
            tools_used=tools_used,
            segment_end_time=segment_end_time,
            complexity_score=result.complexity
        )

        # Save collapsed sentinel to database
        # Note: segment_embedding will be extracted from sentinel.metadata during save
        user_id = get_current_user_id()
        self.continuum_repo.save_message(
            collapsed_sentinel,
            event.continuum_id,
            user_id
        )

        # Invalidate Valkey cache to force reload with collapsed sentinel
        self.continuum_pool.invalidate()

        # Publish collapsed event
        self.event_bus.publish(SegmentCollapsedEvent.create(
            continuum_id=event.continuum_id,
            segment_id=event.segment_id,
            summary=result.synopsis,
            tools_used=tools_used
        ))

        # Trigger downstream processing
        self._trigger_downstream_processing(
            event.continuum_id,
            event.segment_id,
            collapsed_sentinel,
            messages,
            result.synopsis,
            force_immediate=force_immediate
        )

        # Cleanup Files API uploads for this segment
        self._cleanup_segment_files(event.segment_id)

        # DIY Reinforcement Loop: Extract feedback and run synthesis if due
        self._process_feedback_loop(
            messages=messages,
            segment_id=UUID(event.segment_id),
            continuum_id=UUID(event.continuum_id),
        )

        # Portrait synthesis if use-day threshold reached
        self._process_portrait_synthesis()

        # Publish manifest updated event (for cache invalidation)
        self.event_bus.publish(ManifestUpdatedEvent.create(
            continuum_id=event.continuum_id,
            segment_count=self._count_user_segments()
        ))

        logger.info(f"Successfully collapsed segment {event.segment_id}")
        return collapsed_sentinel

    def _find_segment_sentinel(
        self,
        continuum_id: str,
        segment_id: str
    ) -> Optional[Message]:
        """
        Find segment boundary sentinel by segment_id.

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            segment_id: Segment UUID from sentinel metadata

        Returns:
            Sentinel message or None if not found
        """
        user_id = get_current_user_id()
        return self.continuum_repo.find_segment_by_id(continuum_id, segment_id, user_id)

    def _load_segment_messages(
        self,
        continuum_id: str,
        sentinel: Message
    ) -> List[Message]:
        """
        Load messages belonging to this segment.

        Messages are from this sentinel to next sentinel (exclusive) or end of continuum,
        excluding session boundaries and summaries.

        NOTE: This method uses direct database access (encapsulation violation) to implement
        defensive "stop at next boundary" logic. While system constraints ensure only one
        active segment exists at a time (making this check theoretically unnecessary), the
        boundary check provides protection against:
        - Future race conditions in segment creation
        - Data inconsistencies from manual database operations
        - Changes to segment lifecycle management

        This defensive programming is accepted technical debt - the encapsulation violation
        is acknowledged but deemed acceptable for this single-use case with defensive value.

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            sentinel: Segment boundary sentinel

        Returns:
            List of messages in segment
        """
        user_id = get_current_user_id()
        db = self.continuum_repo._get_client(user_id)

        # Load messages after sentinel timestamp, excluding boundaries/system notifications
        query = """
            SELECT * FROM messages
            WHERE continuum_id = %s
                AND created_at > %s
                AND COALESCE(metadata->>'is_segment_boundary', 'false') != 'true'
                AND COALESCE(metadata->>'system_notification', 'false') != 'true'
            ORDER BY created_at ASC
        """

        rows = db.execute_query(query, (continuum_id, sentinel.created_at))

        # Stop at next segment boundary by filtering results
        segment_rows = []
        for row in rows:
            # Check for next segment boundary in remaining results
            metadata = row.get('metadata', {})
            if isinstance(metadata, str):
                import json
                metadata = json.loads(metadata) if metadata else {}

            if metadata.get('is_segment_boundary'):
                break

            segment_rows.append(row)

        return self.continuum_repo._parse_message_rows(segment_rows)

    def _generate_summary(
        self,
        messages: List[Message],
        sentinel: Message,
        continuum_id: str
    ) -> tuple[SummaryResult, list[float]]:
        """
        Generate segment summary and embedding.

        Fetches recent collapsed segments for narrative continuity - the summarizer
        can then use connective phrases like "Building on Tuesday's auth work..."

        Args:
            messages: Messages in segment
            sentinel: Segment boundary sentinel (for tools_used)
            continuum_id: Continuum ID for fetching previous summaries

        Returns:
            Tuple of (SummaryResult, embedding)

        Raises:
            RuntimeError: If summary generation or embedding generation fails
        """
        tools_used = sentinel.metadata.get('tools_used', [])
        user_id = get_current_user_id()

        # Fetch recent collapsed segments for narrative continuity (sliding window)
        previous_summaries = self.continuum_repo.find_collapsed_segments(
            continuum_id=continuum_id,
            user_id=user_id,
            limit=5  # Last 5 segments for context
        )

        try:
            # Generate summary using SummaryGenerator with previous summaries for continuity
            result = self.summary_generator.generate_summary(
                messages=messages,
                summary_type=SummaryType.SEGMENT,
                tools_used=tools_used,
                previous_summaries=previous_summaries
            )

            # Generate embedding for segment search (required for semantic segment search)
            embedding = self.embeddings_provider.encode_deep(result.synopsis)

            # Convert ndarray to list for JSON serialization (storage boundary)
            embedding_list = embedding.tolist()

            return result, embedding_list

        except Exception as e:
            # Re-raise to fail the entire collapse operation
            # Segment will remain active and retry on next timeout check
            logger.exception(
                "Segment summary generation failed; segment %s will remain active and retry",
                sentinel.metadata.get('segment_id')
            )
            raise RuntimeError(f"Segment collapse failed: summary generation error") from e

    def _trigger_downstream_processing(
        self,
        continuum_id: str,
        segment_id: str,
        sentinel: Message,
        messages: List[Message],
        summary: str,
        force_immediate: bool = False
    ) -> None:
        """
        Trigger downstream processing after segment collapse.

        Submits segment to:
        1. Memory extraction (batch or immediate) - skipped for tombstoned segments
        2. Domain knowledge updates (if enabled)

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            segment_id: Segment UUID
            sentinel: Collapsed segment sentinel
            messages: Messages in segment
            summary: Generated summary text (checked for tombstone)
            force_immediate: If True, run extraction inline (skips batch)

        Raises:
            RuntimeError: If memory extraction submission fails
        """
        user_id = get_current_user_id()

        # Skip memory extraction for tombstoned segments (LLM refused to summarize)
        if summary == "[Segment content not summarized]":
            logger.warning(f"Skipping memory extraction for tombstoned segment {segment_id}")
            return

        # Skip memory extraction for demo users (ephemeral sessions)
        from utils.user_context import get_user_preferences
        prefs = get_user_preferences()
        if prefs.conversation_llm == 'demo':
            logger.info(f"Skipping memory extraction for demo user segment {segment_id}")
            return

        # Memory extraction (immediate for manual collapse, batch otherwise)
        if messages:
            # submit_segment_extraction is self-contained: loads messages, submits, marks boundary
            batch_submitted = self.lt_memory_factory.extraction_orchestrator.submit_segment_extraction(
                user_id=user_id,
                boundary_message_id=str(sentinel.id),
                force_immediate=force_immediate
            )

            if not batch_submitted:
                raise RuntimeError(f"Failed to submit segment {segment_id} for memory extraction - submission failed")

            logger.info(f"{'Immediate' if force_immediate else 'Batch'} extraction submitted for segment {segment_id}")

        # Process pending manual memories (from memory_tool.create_memory)
        self._process_pending_manual_memories(user_id, segment_id)

        # Domain knowledge updates (if user has blocks enabled)
        # NOTE (2025-11-07): Considered implementing segment collapse flush to domain knowledge service.
        # Trade-off: Letta requires consistent batches of 10 messages for iterative learning,
        # but segment collapse would flush partial batches (< 10 messages). Need to resolve
        # whether to prioritize: (a) Letta batch consistency vs (b) data completeness on collapse.
        # Current behavior: Messages buffer until batch size (10) or explicit disable/delete.
        # Deferred for future architectural decision.

    def _process_pending_manual_memories(self, user_id: str, segment_id: str) -> None:
        """
        Process pending manual memories queued by memory_tool.create_memory().

        Fetches pending memories from Valkey and processes them synchronously:
        - Generates embeddings
        - Extracts entities and links to knowledge graph
        - Creates supersedes links
        - Stores to database

        This runs at segment collapse to defer heavy operations from tool invocation.

        Args:
            user_id: User UUID
            segment_id: Segment UUID being collapsed
        """
        from lt_memory.models import PendingManualMemory, ExtractedMemory, MemoryLink
        from lt_memory.db_access import LTMemoryDB
        from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
        from utils.database_session_manager import get_shared_session_manager
        from utils.tag_parser import parse_memory_id

        valkey = get_valkey_client()

        # Check both segment-specific queue and presegment queue
        queue_keys = [
            f"pending_memories:{user_id}:{segment_id}",
            f"pending_memories:{user_id}:presegment"
        ]

        all_pending = []
        for queue_key in queue_keys:
            pending_json_list = valkey.lrange(queue_key, 0, -1)
            for json_str in pending_json_list:
                try:
                    pending = PendingManualMemory.from_json(json_str)
                    all_pending.append(pending)
                except Exception:
                    logger.warning("Failed to parse pending memory", exc_info=True)

            # Delete queue after fetching
            if pending_json_list:
                valkey.delete(queue_key)

        if not all_pending:
            logger.debug(f"No pending manual memories for segment {segment_id}")
            return

        logger.info(f"Processing {len(all_pending)} pending manual memories for segment {segment_id}")

        embeddings_provider = get_hybrid_embeddings_provider()
        session_manager = get_shared_session_manager()
        db = LTMemoryDB(session_manager)

        for mem in all_pending:
            try:
                # Generate embedding (768d deep encoder)
                embedding = embeddings_provider.encode_deep([mem.text])[0].tolist()

                # Parse temporal fields
                parsed_happens_at = None
                parsed_expires_at = None
                if mem.happens_at:
                    try:
                        parsed_happens_at = parse_time_string(mem.happens_at)
                    except Exception:
                        logger.warning(f"Invalid happens_at for pending memory {mem.pending_id}")
                if mem.expires_at:
                    try:
                        parsed_expires_at = parse_time_string(mem.expires_at)
                    except Exception:
                        logger.warning(f"Invalid expires_at for pending memory {mem.pending_id}")

                # Create ExtractedMemory
                extracted = ExtractedMemory(
                    text=mem.text,
                    importance_score=mem.importance_score,
                    happens_at=parsed_happens_at,
                    expires_at=parsed_expires_at
                )

                # Store memory
                created_ids = db.store_memories([extracted], embeddings=[embedding])
                memory_id = created_ids[0]

                # Manual memories skip entity extraction (no LLM extraction context).
                # Entities get linked when batch extraction processes the segment.

                # Create supersedes links if provided
                for short_id in mem.supersedes_memory_ids:
                    target = self._find_memory_by_short_id(short_id, db)
                    if target:
                        link = MemoryLink(
                            source_id=memory_id,
                            target_id=target.id,
                            link_type="supersedes",
                            reasoning="Manual supersedes link",
                            created_at=utc_now()
                        )
                        db.create_links([link])
                        logger.debug(f"Created supersedes link: {memory_id} -> {target.id}")

                logger.info(
                    f"Processed manual memory {memory_id} "
                    f"(pending_id: {mem.pending_id})"
                )

            except Exception:
                # Log but don't fail segment collapse on individual memory failures
                logger.exception("Failed to process pending memory %s", mem.pending_id)

    def _find_memory_by_short_id(self, short_id: str, db: LTMemoryDB) -> Memory | None:
        """
        Find a memory by short ID for supersedes linking.

        Args:
            short_id: Either "mem_XXXXXXXX" or raw "XXXXXXXX"
            db: LTMemoryDB instance

        Returns:
            Memory model or None if not found
        """
        from utils.tag_parser import parse_memory_id
        from lt_memory.models import Memory

        clean_id = parse_memory_id(short_id)
        if not clean_id or len(clean_id) < 8:
            return None

        user_id = get_current_user_id()
        with db.session_manager.get_session(user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE REPLACE(id::text, '-', '') LIKE %(pattern)s
              AND is_archived = FALSE
            LIMIT 1
            """
            result = session.execute_single(query, {'pattern': f"{clean_id.lower()}%"})

            if result:
                return Memory(**result)
            return None

    def _extract_tools_from_messages(self, messages: List[Message]) -> List[str]:
        """
        Extract unique tools used by parsing message content for tool_call blocks.

        Args:
            messages: Messages in segment

        Returns:
            Sorted list of unique tool names
        """
        tools_used = set()

        for msg in messages:
            # Skip non-assistant messages (tools only in assistant responses)
            if msg.role != "assistant":
                continue

            # Check if content is structured (list of blocks)
            if isinstance(msg.content, list):
                for block in msg.content:
                    # Extract tool name from tool_call blocks
                    if isinstance(block, dict) and block.get('type') == 'tool_call':
                        tool_name = block.get('name')
                        if tool_name:
                            tools_used.add(tool_name)

        return sorted(list(tools_used))

    def _cleanup_segment_files(self, segment_id: str) -> None:
        """
        Cleanup Files API uploads for collapsed segment.

        Called after segment collapse to delete uploaded files from Anthropic storage.
        Files are tracked per-segment and deleted when segment is archived.

        Args:
            segment_id: Segment UUID to cleanup files for
        """
        try:
            from cns.services.orchestrator import get_orchestrator

            orchestrator = get_orchestrator()
            files_manager = orchestrator.llm_provider.create_files_manager()
            files_manager.cleanup_segment_files(segment_id)

            logger.debug(f"Cleaned up Files API uploads for segment {segment_id}")
        except Exception:
            # Log but don't fail segment collapse on cleanup errors
            logger.warning(
                "Failed to cleanup Files API uploads for segment %s",
                segment_id,
                exc_info=True
            )

    def _count_user_segments(self) -> int:
        """
        Count total segments for user (for ManifestUpdatedEvent).

        Requires: Active user context (set via set_current_user_id at handler entry)

        Returns:
            Number of segments for user

        Raises:
            RuntimeError: If database query fails
        """
        from utils.database_session_manager import get_shared_session_manager
        session_manager = get_shared_session_manager()

        user_id = get_current_user_id()
        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT COUNT(*) as count
                FROM messages
                WHERE user_id = %s
                    AND metadata->>'is_segment_boundary' = 'true'
            """, (user_id,))

            return result['count'] if result else 0

    def _process_feedback_loop(
        self,
        messages: List[Message],
        segment_id: UUID,
        continuum_id: UUID,
    ) -> None:
        """
        Process the user model pipeline: assess behavior and synthesize if due.

        Evaluates conversation against the system prompt's behavioral contract,
        producing section-anchored signals. When synthesis threshold is reached,
        evolves the user model with critic validation.

        Args:
            messages: Messages in the collapsed segment
            segment_id: Segment UUID
            continuum_id: Continuum UUID
        """
        if not self._init_feedback_loop():
            return

        user_id = get_current_user_id()

        try:
            # Step 1: Get current user model (needed by assessor for calibration)
            current_model_xml = self._feedback_tracker.get_last_synthesis_output(user_id)

            # Step 2: Extract assessment signals from this segment
            signals = self._assessment_extractor.extract_signals(
                messages=messages,
                segment_id=segment_id,
                continuum_id=continuum_id,
                user_model_xml=current_model_xml
            )

            # Step 3: Persist signals
            if signals:
                self._feedback_repo.save_signals(signals)
                logger.info("Extracted %d assessment signals from segment %s", len(signals), segment_id)

            # Step 4: Synthesize user model if threshold reached (every 7 use-days)
            if self._feedback_tracker.should_synthesize(user_id):
                logger.info("Running user model synthesis for user %s", user_id)

                result = self._synthesizer.synthesize(
                    user_id=user_id,
                    current_model_xml=current_model_xml
                )

                # Store result
                self._feedback_tracker.mark_synthesized(
                    user_id,
                    result.raw_xml,
                    needs_checkin=len(result.checkin_topics) > 0
                )

                # Mark signals as synthesized
                unsynthesized = self._feedback_repo.get_unsynthesized_signals(user_id)
                signal_ids = [s['id'] for s in unsynthesized]
                if signal_ids:
                    self._feedback_repo.mark_signals_synthesized(user_id, signal_ids)

                # Invalidate LoRA trinket cache so next turn picks up new model
                self._invalidate_lora_trinket_cache(user_id)

                logger.info(
                    "User model synthesis complete: %d observations, %d checkin topics",
                    len(result.observations), len(result.checkin_topics)
                )

        except Exception as e:
            # Don't re-raise — segment is already collapsed and saved at this point.
            # But log at error so operators notice immediately.
            logger.error(
                "USER MODEL PIPELINE FAILED for segment %s: %s — "
                "Assessment signals and/or synthesis were lost for this segment.",
                segment_id, e, exc_info=True
            )

    def _process_portrait_synthesis(self) -> None:
        """
        Synthesize user portrait if use-day threshold reached.

        Runs after the feedback loop in the collapse chain. Portrait synthesis
        is optional content — failures are logged but do not fail the collapse.
        """
        from config import config
        from cns.services.portrait_service import should_synthesize_portrait, synthesize_and_store

        user_id = get_current_user_id()
        threshold = config.scheduled_jobs.portrait_synthesis_use_days

        try:
            if not should_synthesize_portrait(user_id, threshold):
                return

            logger.info("Running portrait synthesis for user %s", user_id)
            synthesize_and_store(user_id)

        except Exception as e:
            logger.error(
                "PORTRAIT SYNTHESIS FAILED for user %s: %s — "
                "Portrait will be stale until next qualifying collapse.",
                user_id, e, exc_info=True
            )

    def _invalidate_lora_trinket_cache(self, user_id: str) -> None:
        """Invalidate LoRA trinket cache after synthesis."""
        try:
            from working_memory.trinkets.base import TRINKET_KEY_PREFIX
            valkey = get_valkey_client()
            valkey.hdel_with_retry(f"{TRINKET_KEY_PREFIX}:{user_id}", "behavioral_directives")
        except Exception as e:
            logger.warning("Failed to invalidate LoRA trinket cache: %s", e, exc_info=True)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_collapse_handler_instance: Optional[SegmentCollapseHandler] = None


def get_segment_collapse_handler() -> SegmentCollapseHandler:
    """
    Get singleton SegmentCollapseHandler instance.

    Returns:
        The initialized SegmentCollapseHandler

    Raises:
        RuntimeError: If handler not initialized (factory not run)
    """
    global _collapse_handler_instance
    if _collapse_handler_instance is None:
        raise RuntimeError(
            "SegmentCollapseHandler not initialized. "
            "Ensure CNSIntegrationFactory has been initialized."
        )
    return _collapse_handler_instance


def initialize_segment_collapse_handler(handler: SegmentCollapseHandler) -> None:
    """
    Initialize the singleton handler instance (called by factory).

    Args:
        handler: The handler instance to store
    """
    global _collapse_handler_instance
    _collapse_handler_instance = handler
    logger.info("SegmentCollapseHandler singleton initialized")
