"""
Segment helper utilities for managing segment boundary sentinels.

Segments are represented as sentinel messages in the messages table with
metadata.is_segment_boundary = True, following the same pattern as session boundaries.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from cns.core.message import Message
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


def create_segment_boundary_sentinel(
    first_message_time: datetime,
    continuum_id: str
) -> Message:
    """
    Create segment boundary sentinel message.

    Sentinel is created in 'active' status when second message arrives.
    Summary and embedding are added during collapse.

    Args:
        first_message_time: Timestamp of first message in segment
        continuum_id: Continuum UUID (for reference)

    Returns:
        Message object with segment boundary metadata
    """
    metadata = {
        'is_segment_boundary': True,
        'status': 'active',
        'segment_id': str(uuid4()),  # Unique segment identifier
        'segment_start_time': first_message_time.isoformat(),
        'segment_end_time': first_message_time.isoformat(),  # Will update as messages arrive
        'tools_used': [],
        'memories_extracted': False,
        'domain_blocks_updated': False
    }

    # Content is placeholder until collapse generates summary
    sentinel = Message(
        content="[Segment in progress]",
        role="assistant",
        metadata=metadata
    )

    logger.info(
        f"Created segment boundary sentinel {metadata['segment_id']} "
        f"for continuum {continuum_id}"
    )

    return sentinel


def add_tools_to_segment(sentinel: Message, tools_used: List[str]) -> None:
    """
    Add tools to segment's tools_used list (deduplicated).

    Args:
        sentinel: Segment boundary sentinel message
        tools_used: List of tool names used in this turn
    """
    current_tools = set(sentinel.metadata.get('tools_used', []))
    current_tools.update(tools_used)
    sentinel.metadata['tools_used'] = sorted(list(current_tools))


def collapse_segment_sentinel(
    sentinel: Message,
    summary: str,
    display_title: str,
    embedding: Optional[List[float]],
    inactive_duration_minutes: int,
    processing_failed: bool = False,
    tools_used: Optional[List[str]] = None,
    segment_end_time: Optional[datetime] = None,
    complexity_score: int = 2
) -> Message:
    """
    Collapse segment sentinel with summary and embedding.

    Returns new Message with collapsed state (Message is immutable).

    Args:
        sentinel: Segment boundary sentinel message
        summary: Generated telegraphic summary
        display_title: Short telegraphic title for manifest display
        embedding: AllMiniLM 384-dim embedding of summary
        inactive_duration_minutes: Minutes of inactivity that triggered collapse
        processing_failed: True if summary generation failed and fallback was used
        tools_used: Tools used in segment (extracted from messages)
        segment_end_time: Timestamp of last message in segment
        complexity_score: Cognitive complexity score (1=simple, 2=moderate, 3=complex)

    Returns:
        New Message with status='collapsed' and summary in content
    """
    # Build collapsed metadata
    collapsed_metadata = {
        **sentinel.metadata,
        'status': 'collapsed',
        'collapsed_at': utc_now().isoformat(),
        'inactive_duration_minutes': inactive_duration_minutes,
        'summary_generated_at': utc_now().isoformat(),
        'processing_failed': processing_failed,
        'display_title': display_title,
        'complexity_score': complexity_score
    }

    # Set tools_used from actual messages
    if tools_used is not None:
        collapsed_metadata['tools_used'] = tools_used

    # Set segment_end_time from last message
    if segment_end_time is not None:
        collapsed_metadata['segment_end_time'] = segment_end_time.isoformat()

    # Store embedding in metadata for repository to extract and persist
    if embedding is not None:
        collapsed_metadata['segment_embedding_value'] = embedding
        collapsed_metadata['has_segment_embedding'] = True

    logger.info(
        f"Collapsed segment {sentinel.metadata['segment_id']}: "
        f"inactive_duration={inactive_duration_minutes}min, "
        f"tools_used={tools_used}, "
        f"summary='{summary[:50]}...'"
    )

    # Return new Message with clean summary - display formatting happens at display time
    return Message(
        id=sentinel.id,
        content=summary,
        role=sentinel.role,
        created_at=sentinel.created_at,
        metadata=collapsed_metadata
    )


def mark_segment_processed(
    sentinel: Message,
    memories_extracted: bool = False,
    domain_blocks_updated: bool = False,
    memory_count: Optional[int] = None
) -> None:
    """
    Mark segment as processed for memory extraction or domain updates.

    Idempotent processing flags prevent duplicate work.

    Args:
        sentinel: Segment boundary sentinel message
        memories_extracted: Set memories_extracted flag
        domain_blocks_updated: Set domain_blocks_updated flag
        memory_count: Number of memories extracted
    """
    if memories_extracted:
        sentinel.metadata['memories_extracted'] = True
        sentinel.metadata['memory_extraction_at'] = utc_now().isoformat()
        if memory_count is not None:
            sentinel.metadata['memory_count'] = memory_count

    if domain_blocks_updated:
        sentinel.metadata['domain_blocks_updated'] = True
        sentinel.metadata['domain_update_at'] = utc_now().isoformat()


def get_segment_id(sentinel: Message) -> str:
    """Extract segment ID from sentinel metadata."""
    return sentinel.metadata.get('segment_id', '')


def is_segment_boundary(message: Message) -> bool:
    """Check if message is a segment boundary sentinel."""
    return message.metadata.get('is_segment_boundary', False)


def is_active_segment(sentinel: Message) -> bool:
    """Check if segment is still active (not collapsed)."""
    return sentinel.metadata.get('status') == 'active'


def get_segment_time_range(sentinel: Message) -> tuple[datetime, datetime]:
    """
    Get segment time range from sentinel metadata.

    Returns:
        Tuple of (start_time, end_time) as datetime objects
    """
    start_str = sentinel.metadata.get('segment_start_time')
    end_str = sentinel.metadata.get('segment_end_time')

    start_time = datetime.fromisoformat(start_str) if start_str else utc_now()
    end_time = datetime.fromisoformat(end_str) if end_str else utc_now()

    return start_time, end_time


def format_segment_for_display(sentinel: Message) -> str:
    """
    Format collapsed segment content for display in working memory.

    Adds display title, timespan, and formatting to the stored summary.
    This separation keeps storage clean while allowing flexible display formatting.

    Args:
        sentinel: Collapsed segment sentinel message

    Returns:
        Formatted content string with title, timespan, and summary

    Raises:
        KeyError: If required metadata is missing
        ValueError: If timestamp format is invalid
    """
    from utils.timezone_utils import format_relative_time

    display_title = sentinel.metadata['display_title']
    summary = sentinel.content
    start_time_iso = sentinel.metadata['segment_start_time']

    # Convert ISO timestamp to datetime object
    start_time = datetime.fromisoformat(start_time_iso)

    # Format as relative time (grouped timeframe using segment start)
    relative_time = format_relative_time(start_time)

    return f"This is an extended summary of: {display_title}\nTimespan: {relative_time}\n\n{summary}"


def create_collapse_marker() -> Message:
    """
    Create collapse marker for segment display.

    This marker appears between collapsed segment summaries and recent messages,
    indicating older content has been compressed.

    Returns:
        Message with collapse_marker notification type
    """
    return Message(
        content="[...older messages and summaries available through search. use continuumsearch to find specific information from past conversations...]",
        role="assistant",
        metadata={'system_notification': True, 'notification_type': 'collapse_marker'}
    )


def create_session_boundary_marker(segment_summaries: List[Message]) -> Message:
    """
    Create session boundary marker showing time gap between sessions.

    The boundary shows when the previous session ended (based on collapsed segments)
    and when the current session is beginning.

    Args:
        segment_summaries: Collapsed segment summaries from previous sessions

    Returns:
        Message with session_break notification type
    """
    from utils.timezone_utils import convert_from_utc, format_datetime
    from utils.user_context import get_user_timezone

    current_time = utc_now()

    # Determine when the last session ended
    if segment_summaries:
        # Use the most recent collapsed segment's end time
        last_segment = segment_summaries[-1]
        end_time_str = last_segment.metadata.get('segment_end_time')

        if end_time_str:
            last_session_end = datetime.fromisoformat(end_time_str)
        else:
            # Fallback to segment creation time if no end_time
            last_session_end = last_segment.created_at
    else:
        # No previous segments - this is the first conversation
        # Show a generic message without specific end time
        try:
            user_tz = get_user_timezone()
        except Exception:
            user_tz = 'UTC'

        current_time_user_tz = convert_from_utc(current_time, user_tz)
        current_time_str = format_datetime(current_time_user_tz, "time_only", include_timezone=True)

        return Message(
            content=f"NOTIFICATION: THIS CHAT SESSION BEGAN AT {current_time_str}",
            role="assistant",
            metadata={'system_notification': True, 'notification_type': 'session_break'}
        )

    # Convert times to user timezone
    try:
        user_tz = get_user_timezone()
    except Exception:
        user_tz = 'UTC'

    last_time_user_tz = convert_from_utc(last_session_end, user_tz)
    current_time_user_tz = convert_from_utc(current_time, user_tz)

    # Format times consistently
    last_time_str = format_datetime(last_time_user_tz, "time_only", include_timezone=True)
    current_time_str = format_datetime(current_time_user_tz, "time_only", include_timezone=True)

    # Create boundary message
    boundary_content = (
        f"NOTIFICATION: LAST CHAT SESSION ENDED AT {last_time_str} | "
        f"THIS CHAT SESSION BEGAN AT {current_time_str}"
    )

    return Message(
        content=boundary_content,
        role="assistant",
        metadata={'system_notification': True, 'notification_type': 'session_break'}
    )
