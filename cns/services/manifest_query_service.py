"""
Manifest query service for generating conversation segment manifests.

Provides ASCII tree-formatted segment manifests for system prompt display,
with Valkey caching and event-driven invalidation.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from clients.valkey_client import get_valkey_client
from cns.core.events import ManifestUpdatedEvent
from cns.integration.event_bus import EventBus
from cns.infrastructure.continuum_repository import get_continuum_repository
from utils.timezone_utils import utc_now, convert_from_utc
from utils.user_context import get_user_timezone
from config import config

logger = logging.getLogger(__name__)

# Module-level singleton instance
_manifest_service_instance = None


def get_manifest_query_service(event_bus: Optional[EventBus] = None) -> 'ManifestQueryService':
    """
    Get or create singleton ManifestQueryService instance.

    Args:
        event_bus: Event bus for cache invalidation (only needed on first call)

    Returns:
        Singleton ManifestQueryService instance
    """
    global _manifest_service_instance
    if _manifest_service_instance is None:
        logger.info("Creating singleton ManifestQueryService instance")
        _manifest_service_instance = ManifestQueryService(event_bus)
    return _manifest_service_instance


class ManifestQueryService:
    """
    Generates conversation manifests from segment boundaries.

    Queries segment sentinels from messages table, groups by relative time,
    and formats as ASCII tree for system prompt display.
    """

    def __init__(self, event_bus: Optional[EventBus] = None, continuum_repository=None):
        """
        Initialize manifest query service.

        Args:
            event_bus: Event bus for subscribing to ManifestUpdatedEvent
            continuum_repository: Continuum repository (uses singleton if not provided)
        """
        self.valkey = get_valkey_client()
        self.cache_ttl = config.system.manifest_cache_ttl
        self.continuum_repository = continuum_repository or get_continuum_repository()

        # Subscribe to manifest update events for cache invalidation
        if event_bus:
            event_bus.subscribe('ManifestUpdatedEvent', self._handle_manifest_updated)
            logger.info("ManifestQueryService subscribed to ManifestUpdatedEvent")

    def _handle_manifest_updated(self, event: ManifestUpdatedEvent) -> None:
        """
        Handle manifest update event by invalidating cache.

        Args:
            event: ManifestUpdatedEvent with user_id
        """
        cache_key = f"manifest:{event.user_id}"
        try:
            self.valkey.delete(cache_key)
            logger.debug(f"Invalidated manifest cache for user {event.user_id}")
        except Exception as e:
            logger.warning(f"Failed to invalidate manifest cache: {e}")

    def get_manifest_for_prompt(self, user_id: str, limit: Optional[int] = None) -> str:
        """
        Generate ASCII tree manifest from segments for system prompt.

        Format:
        CONVERSATION MANIFEST
        ├─ Today
        │  ├─ [2:15pm - Active] Manifest architecture discussion
        │  └─ [9:00am - 10:15am] Morning standup notes
        ├─ Yesterday
        │  └─ [8:00am - 8:27am] Arduino servo debugging with webaccess research
        └─ Jan 18
           └─ [3:00pm - 4:12pm] Nacho recipe research and planning

        Args:
            user_id: User ID for query
            limit: Maximum number of segments to include (uses config.system.manifest_depth if None)

        Returns:
            Formatted manifest string, or empty string if no segments
        """
        # Use config default if not specified
        if limit is None:
            limit = config.system.manifest_depth

        # Try cache first
        cache_key = f"manifest:{user_id}"
        try:
            cached = self.valkey.get(cache_key)
            if cached:
                logger.debug(f"Manifest cache hit for user {user_id}")
                return cached.decode('utf-8') if isinstance(cached, bytes) else cached
        except Exception as e:
            logger.debug(f"Manifest cache miss: {e}")

        # Query segments from database
        segments = self._query_segments(user_id, limit)

        if not segments:
            return ""

        # Generate manifest
        manifest = self._format_manifest(segments)

        # Cache the result
        try:
            self.valkey.setex(cache_key, self.cache_ttl, manifest)
            logger.debug(f"Cached manifest for user {user_id} (TTL={self.cache_ttl}s)")
        except Exception as e:
            logger.warning(f"Failed to cache manifest: {e}")

        return manifest

    def _query_segments(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """
        Query segment boundary sentinels from messages table.

        Args:
            user_id: User ID for RLS
            limit: Maximum segments to return

        Returns:
            List of segment dictionaries with metadata

        Raises:
            RuntimeError: If database query fails
        """
        segment_messages = self.continuum_repository.find_all_segments(user_id, limit)

        segments = []
        for msg in segment_messages:
            metadata = msg.metadata

            # Use display_title from metadata for manifest tree, fallback to content for in-progress segments
            display_title = metadata.get('display_title')
            if not display_title:
                # Fallback for active segments or segments without display_title
                display_title = msg.content if msg.content else '[Segment in progress]'
                # Truncate if using full content as fallback
                if len(display_title) > 50:
                    display_title = display_title[:47] + "..."

            segments.append({
                'id': str(msg.id),
                'display_title': display_title,
                'synopsis': msg.content,  # Full synopsis available if needed
                'status': metadata.get('status', 'unknown'),
                'start_time': metadata.get('segment_start_time'),
                'end_time': metadata.get('segment_end_time'),
                'created_at': msg.created_at.isoformat() if msg.created_at else None
            })

        # Reverse to chronological order (oldest first)
        segments.reverse()

        logger.debug(f"Queried {len(segments)} segments for user {user_id}")
        return segments

    def _format_manifest(self, segments: List[Dict[str, Any]]) -> str:
        """
        Format segments as ASCII tree grouped by relative time.

        Requires: Active user context (set via set_current_user_id during authentication)

        Args:
            segments: List of segment dictionaries

        Returns:
            Formatted manifest string
        """
        if not segments:
            return ""

        # Get user's timezone for local time display
        try:
            user_tz = get_user_timezone()
        except Exception:
            user_tz = 'UTC'

        # Group segments by date
        grouped = self._group_segments_by_date(segments, user_tz)

        # Build ASCII tree
        lines = ["=== CONVERSATION MANIFEST ==="]

        date_groups = list(grouped.items())
        for date_idx, (date_label, date_segments) in enumerate(date_groups):
            is_last_date = (date_idx == len(date_groups) - 1)
            date_branch = "└─" if is_last_date else "├─"

            lines.append(f"{date_branch} {date_label}")

            # Format segments within date group
            for seg_idx, segment in enumerate(date_segments):
                is_last_seg = (seg_idx == len(date_segments) - 1)

                # Determine indentation
                date_indent = "   " if is_last_date else "│  "
                seg_branch = "└─" if is_last_seg else "├─"

                # Format time range
                time_range = self._format_time_range(
                    segment['start_time'],
                    segment['end_time'],
                    segment['status'],
                    user_tz
                )

                # Format segment line using telegraphic display title
                display_title = segment['display_title']
                lines.append(f"{date_indent}{seg_branch} {time_range} {display_title}")

        return "\n".join(lines)

    def _group_segments_by_date(
        self,
        segments: List[Dict[str, Any]],
        timezone: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group segments by relative date (Today, Yesterday, or date string).

        Args:
            segments: List of segment dictionaries
            timezone: User's timezone for date calculation

        Returns:
            Ordered dict mapping date labels to segment lists
        """
        from collections import OrderedDict

        now = utc_now()
        now_local = convert_from_utc(now, timezone)
        today = now_local.date()
        yesterday = today - timedelta(days=1)

        grouped = OrderedDict()

        for segment in reversed(segments):  # Process newest first for date grouping
            # Parse segment start time
            try:
                if segment['start_time']:
                    start_time_utc = datetime.fromisoformat(segment['start_time'])
                    start_time_local = convert_from_utc(start_time_utc, timezone)
                    segment_date = start_time_local.date()
                else:
                    segment_date = today
            except Exception:
                segment_date = today

            # Determine date label
            if segment_date == today:
                date_label = "TODAY"
            elif segment_date == yesterday:
                date_label = "YESTERDAY"
            else:
                # Format as "JAN 18" or similar
                date_label = segment_date.strftime("%b %d").upper()

            # Add to grouped dict
            if date_label not in grouped:
                grouped[date_label] = []
            grouped[date_label].insert(0, segment)  # Insert at beginning for chronological order

        return grouped

    def _format_time_range(
        self,
        start_time_str: Optional[str],
        end_time_str: Optional[str],
        status: str,
        timezone: str
    ) -> str:
        """
        Format time range for segment display.

        Args:
            start_time_str: ISO format start time
            end_time_str: ISO format end time
            status: Segment status (active/collapsed)
            timezone: User's timezone

        Returns:
            Formatted time range like "[2:15pm - Active]" or "[9:00am - 10:15am]"
        """
        try:
            if not start_time_str:
                return "[Unknown]"

            start_time_utc = datetime.fromisoformat(start_time_str)
            start_time_local = convert_from_utc(start_time_utc, timezone)

            if status == 'active':
                # Active segment - show start time and "ACTIVE"
                start_str = start_time_local.strftime("%-I:%M%p").upper()
                return f"[{start_str} - ACTIVE]"
            else:
                # Collapsed segment - show time range
                if end_time_str:
                    end_time_utc = datetime.fromisoformat(end_time_str)
                    end_time_local = convert_from_utc(end_time_utc, timezone)

                    start_str = start_time_local.strftime("%-I:%M%p").upper()
                    end_str = end_time_local.strftime("%-I:%M%p").upper()
                    return f"[{start_str} - {end_str}]"
                else:
                    start_str = start_time_local.strftime("%-I:%M%p").upper()
                    return f"[{start_str}]"

        except Exception as e:
            logger.warning(f"Failed to format time range: {e}")
            return "[Unknown]"
