"""
Segment collapse handler for processing timeout events.

Handles SessionTimeoutEvent by:
1. Finding the segment boundary sentinel
2. Loading segment messages
3. Generating summary with embedding
4. Updating sentinel metadata
5. Triggering downstream processing (memory extraction, domain updates)
"""
import logging
from typing import List, Optional

from cns.core.events import SegmentTimeoutEvent, SegmentCollapsedEvent, ManifestUpdatedEvent
from cns.core.message import Message
from cns.services.segment_helpers import (
    collapse_segment_sentinel,
    mark_segment_processed,
    get_segment_id,
    is_segment_boundary
)
from cns.services.summary_generator import SummaryGenerator, SummaryType
from cns.infrastructure.continuum_repository import ContinuumRepository
from clients.hybrid_embeddings_provider import HybridEmbeddingsProvider
from cns.integration.event_bus import EventBus
from utils.timezone_utils import utc_now
from utils.user_context import set_current_user_id, get_current_user_id

logger = logging.getLogger(__name__)


class SegmentCollapseHandler:
    """
    Handles segment collapse when timeout is reached.

    Subscribes to SessionTimeoutEvent and orchestrates the collapse pipeline:
    summary generation, embedding, sentinel update, and downstream processing.
    """

    def __init__(
        self,
        continuum_repo: ContinuumRepository,
        summary_generator: SummaryGenerator,
        embeddings_provider: HybridEmbeddingsProvider,
        event_bus: EventBus,
        continuum_pool,
        lt_memory_factory=None
    ):
        """
        Initialize collapse handler.

        Args:
            continuum_repo: Repository for loading/saving messages
            summary_generator: Service for generating segment summaries
            embeddings_provider: Provider for generating embeddings
            event_bus: Event bus for publishing events
            continuum_pool: Continuum pool for cache invalidation
            lt_memory_factory: LT_Memory factory for extraction (optional)
        """
        self.continuum_repo = continuum_repo
        self.summary_generator = summary_generator
        self.embeddings_provider = embeddings_provider
        self.event_bus = event_bus
        self.continuum_pool = continuum_pool
        self.lt_memory_factory = lt_memory_factory

        # Subscribe to timeout events
        self.event_bus.subscribe('SegmentTimeoutEvent', self.handle_timeout)
        logger.info("SegmentCollapseHandler subscribed to SegmentTimeoutEvent")

    def handle_timeout(self, event: SegmentTimeoutEvent) -> None:
        """
        Handle segment timeout by collapsing the segment.

        Collapse failures are caught and logged - segment remains active and will retry
        on next timeout check. However, persistent failures are escalated to alert operators.

        Args:
            event: SegmentTimeoutEvent with segment details
        """
        # Set user context once at entry point
        set_current_user_id(event.user_id)

        try:
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
                # Sentinel missing is a data consistency error - alert operators
                logger.error(
                    f"COLLAPSE FAILURE: Segment sentinel {event.segment_id} not found. "
                    f"Data consistency violation - timeout event published for non-existent segment. "
                    f"Segment will remain in timeout queue and retry."
                )
                return

            # Load messages in segment (between this sentinel and next, or end of continuum)
            messages = self._load_segment_messages(
                event.continuum_id,
                sentinel
            )

            if not messages:
                raise RuntimeError(
                    f"Segment {event.segment_id} has no messages - this violates system invariants. "
                    f"Messages are always persisted in pairs (user + assistant). "
                    f"Possible data corruption or segment boundary logic error."
                )

            # Generate summary, display title, complexity, and embedding (raises on failure)
            summary, display_title, complexity, embedding = self._generate_summary_with_fallback(
                messages,
                sentinel
            )

            # Extract tools used from actual messages (not sentinel metadata)
            tools_used = self._extract_tools_from_messages(messages)

            # Set segment_end_time from last message (guaranteed to exist at this point)
            segment_end_time = messages[-1].created_at

            # Collapse sentinel (returns new Message with collapsed state)
            collapsed_sentinel = collapse_segment_sentinel(
                sentinel,
                summary=summary,
                display_title=display_title,
                embedding=embedding,
                inactive_duration_minutes=event.inactive_duration_minutes,
                processing_failed=False,  # Always False - failures raise instead of degrading
                tools_used=tools_used,
                segment_end_time=segment_end_time,
                complexity_score=complexity
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
                summary=summary,
                tools_used=tools_used
            ))

            # Trigger downstream processing
            self._trigger_downstream_processing(
                event.continuum_id,
                event.segment_id,
                collapsed_sentinel,
                messages
            )

            # Publish manifest updated event (for cache invalidation)
            self.event_bus.publish(ManifestUpdatedEvent.create(
                continuum_id=event.continuum_id,
                segment_count=self._count_user_segments()
            ))

            logger.info(f"Successfully collapsed segment {event.segment_id}")

            # Reset thinking budget preference to system default after segment collapse
            continuum = self.continuum_pool.get_or_create()
            continuum.set_thinking_budget_preference(None)
            logger.debug(f"Reset thinking budget preference to system default after segment collapse")

        except Exception as e:
            # Collapse failure - segment remains active and will retry on next timeout check
            # Log at ERROR level with full stack trace to alert operators
            logger.error(
                f"COLLAPSE FAILURE: Segment {event.segment_id} collapse failed. "
                f"Segment will remain active and retry on next timeout check. "
                f"Operators should investigate if this persists. Error: {e}",
                exc_info=True
            )

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
                AND (metadata->>'is_segment_boundary' IS NULL OR metadata->>'is_segment_boundary' != 'true')
                AND (metadata->>'system_notification' IS NULL OR metadata->>'system_notification' != 'true')
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

    def _generate_summary_with_fallback(
        self,
        messages: List[Message],
        sentinel: Message
    ) -> tuple[str, str, int, List[float]]:
        """
        Generate segment summary, complexity score, and embedding.

        Args:
            messages: Messages in segment
            sentinel: Segment boundary sentinel (for tools_used)

        Returns:
            Tuple of (summary_text, display_title, complexity_score, embedding)

        Raises:
            RuntimeError: If summary generation or embedding generation fails
        """
        tools_used = sentinel.metadata.get('tools_used', [])

        try:
            # Generate summary using SummaryGenerator (returns 3-tuple)
            summary_text, display_title, complexity = self.summary_generator.generate_summary(
                messages=messages,
                summary_type=SummaryType.SEGMENT,
                tools_used=tools_used
            )

            # Generate embedding for segment search (required for semantic segment search)
            embedding = self.embeddings_provider.encode_realtime(summary_text)

            # Convert ndarray to list for JSON serialization (storage boundary)
            embedding_list = embedding.tolist()

            return summary_text, display_title, complexity, embedding_list

        except Exception as e:
            # Re-raise to fail the entire collapse operation
            # Segment will remain active and retry on next timeout check
            logger.error(f"Segment summary generation failed - segment {sentinel.metadata.get('segment_id')} will remain active and retry")
            raise RuntimeError(f"Segment collapse failed: summary generation error") from e

    def _trigger_downstream_processing(
        self,
        continuum_id: str,
        segment_id: str,
        sentinel: Message,
        messages: List[Message]
    ) -> None:
        """
        Trigger downstream processing after segment collapse.

        Submits segment to:
        1. Memory extraction (via Batch API)
        2. Domain knowledge updates (if enabled)

        Requires: Active user context (set via set_current_user_id at handler entry)

        Args:
            continuum_id: Continuum UUID
            segment_id: Segment UUID
            sentinel: Collapsed segment sentinel
            messages: Messages in segment

        Raises:
            RuntimeError: If memory extraction submission fails
        """
        user_id = get_current_user_id()

        # Memory extraction via Batch API
        if self.lt_memory_factory and messages:
            # Submit to Batch API (returns immediately, polling happens separately)
            batch_submitted = self.lt_memory_factory.extraction_orchestrator.submit_segment_extraction(
                user_id=user_id,
                messages=messages,
                segment_id=segment_id
            )

            if not batch_submitted:
                raise RuntimeError(f"Failed to submit segment {segment_id} for memory extraction - batch submission failed")

            # Mark as submitted (actual completion tracked by polling job)
            mark_segment_processed(sentinel, memories_extracted=True)

            # Re-persist sentinel with updated metadata
            self.continuum_repo.save_message(sentinel, continuum_id, user_id)

            logger.info(f"Submitted segment {segment_id} for memory extraction")

        # Domain knowledge updates (if user has blocks enabled)
        # NOTE (2025-11-07): Considered implementing segment collapse flush to domain knowledge service.
        # Trade-off: Letta requires consistent batches of 10 messages for iterative learning,
        # but segment collapse would flush partial batches (< 10 messages). Need to resolve
        # whether to prioritize: (a) Letta batch consistency vs (b) data completeness on collapse.
        # Current behavior: Messages buffer until batch size (10) or explicit disable/delete.
        # Deferred for future architectural decision.

    def _extract_tools_from_messages(self, messages: List[Message]) -> List[str]:
        """
        Extract unique tools used by parsing message content for tool_use blocks.

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
                    # Extract tool name from tool_use blocks
                    if isinstance(block, dict) and block.get('type') == 'tool_use':
                        tool_name = block.get('name')
                        if tool_name:
                            tools_used.add(tool_name)

        return sorted(list(tools_used))

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

    def cleanup(self) -> None:
        """Unsubscribe from events during cleanup."""
        try:
            self.event_bus.unsubscribe('SegmentTimeoutEvent', self.handle_timeout)
            logger.info("SegmentCollapseHandler unsubscribed from events")
        except Exception as e:
            logger.warning(f"Error unsubscribing from events: {e}")
