"""
Tests for segment_helpers.py - Segment sentinel management utilities.

Focus: Real contract guarantees, not implementation details.
"""
import pytest
from datetime import datetime

from cns.core.message import Message
from cns.services.segment_helpers import (
    create_segment_boundary_sentinel,
    add_tools_to_segment,
    collapse_segment_sentinel,
    mark_segment_processed,
    get_segment_id,
    is_segment_boundary,
    is_active_segment,
    get_segment_time_range
)
from utils.timezone_utils import utc_now


class TestSegmentLifecycle:
    """Tests enforce segment state transitions: active → collapsed."""

    def test_creates_segment_in_active_state(self):
        """CONTRACT: New sentinels start in 'active' state."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        assert is_active_segment(sentinel)
        assert is_segment_boundary(sentinel)

    def test_collapse_transitions_to_collapsed_state(self):
        """CONTRACT: collapse_segment_sentinel() returns new Message with collapsed status."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        assert is_active_segment(sentinel)

        collapsed = collapse_segment_sentinel(sentinel, "Summary text", "Test segment", None, 60)

        # Original unchanged (immutable)
        assert is_active_segment(sentinel)
        # New message is collapsed
        assert not is_active_segment(collapsed)
        assert collapsed.metadata['status'] == "collapsed"

    def test_collapsed_sentinel_has_summary_not_placeholder(self):
        """CONTRACT: Collapsed message has summary in content, not placeholder."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        original_content = sentinel.content

        summary = "Arduino debugging session"
        display_title = "Arduino debugging"
        collapsed = collapse_segment_sentinel(sentinel, summary, display_title, None, 60)

        # New message has formatted summary with title
        assert display_title in collapsed.content
        assert summary in collapsed.content
        assert collapsed.content != original_content
        # Original unchanged
        assert sentinel.content == original_content

    def test_segment_ids_are_unique(self):
        """CONTRACT: Each segment gets unique identifier."""
        s1 = create_segment_boundary_sentinel(utc_now(), "cid")
        s2 = create_segment_boundary_sentinel(utc_now(), "cid")

        assert get_segment_id(s1) != get_segment_id(s2)
        assert get_segment_id(s1) != ""
        assert get_segment_id(s2) != ""


class TestToolDeduplication:
    """Tests enforce tool deduplication guarantee."""

    def test_deduplicates_tools_across_calls(self):
        """CONTRACT: No duplicate tools in tools_used list."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        # Add same tool multiple times
        add_tools_to_segment(sentinel, ["webaccess_tool"])
        add_tools_to_segment(sentinel, ["webaccess_tool", "calendar_tool"])
        add_tools_to_segment(sentinel, ["webaccess_tool"])

        tools = sentinel.metadata['tools_used']
        # No duplicates
        assert len(tools) == len(set(tools))
        # Both tools present exactly once
        assert tools.count("webaccess_tool") == 1
        assert tools.count("calendar_tool") == 1

    def test_deduplicates_tools_within_single_call(self):
        """CONTRACT: Duplicates removed even within single add_tools call."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        add_tools_to_segment(sentinel, ["tool_a", "tool_a", "tool_b", "tool_a"])

        tools = sentinel.metadata['tools_used']
        assert len(tools) == len(set(tools))
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_tools_are_sorted(self):
        """CONTRACT: tools_used list is deterministically sorted."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        # Add in random order
        add_tools_to_segment(sentinel, ["zebra_tool", "aardvark_tool", "middle_tool"])

        tools = sentinel.metadata['tools_used']
        # Alphabetically sorted
        assert tools == sorted(tools)


class TestEmbeddingHandling:
    """Tests enforce optional embedding contract."""

    def test_collapse_with_embedding_stores_embedding(self):
        """CONTRACT: Embedding is stored when provided."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        embedding = [0.1] * 384

        collapsed = collapse_segment_sentinel(sentinel, "summary", "Test title", embedding, 60)

        assert collapsed.metadata['segment_embedding_value'] == embedding
        assert collapsed.metadata['has_segment_embedding'] is True

    def test_collapse_without_embedding_works(self):
        """CONTRACT: Embedding is optional - collapse works with None."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        collapsed = collapse_segment_sentinel(sentinel, "summary", "Test title", None, 60)

        # No embedding metadata when None provided
        assert 'segment_embedding_value' not in collapsed.metadata
        assert 'has_segment_embedding' not in collapsed.metadata
        # But collapse still succeeded
        assert collapsed.metadata['status'] == "collapsed"


class TestProcessingFlags:
    """Tests enforce independent processing flag guarantees."""

    def test_can_mark_memories_extracted_independently(self):
        """CONTRACT: memories_extracted can be set without domain_blocks_updated."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        mark_segment_processed(sentinel, memories_extracted=True)

        assert sentinel.metadata['memories_extracted'] is True
        assert sentinel.metadata['domain_blocks_updated'] is False

    def test_can_mark_domain_updated_independently(self):
        """CONTRACT: domain_blocks_updated can be set without memories_extracted."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        mark_segment_processed(sentinel, domain_blocks_updated=True)

        assert sentinel.metadata['domain_blocks_updated'] is True
        assert sentinel.metadata['memories_extracted'] is False

    def test_memory_count_is_optional(self):
        """CONTRACT: memory_count can be omitted even when memories_extracted=True."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        mark_segment_processed(sentinel, memories_extracted=True, memory_count=None)

        assert sentinel.metadata['memories_extracted'] is True
        assert 'memory_count' not in sentinel.metadata

    def test_memory_count_zero_is_valid(self):
        """CONTRACT: memory_count=0 is distinct from memory_count=None."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        mark_segment_processed(sentinel, memories_extracted=True, memory_count=0)

        # 0 is a valid value and should be stored
        assert sentinel.metadata['memory_count'] == 0


class TestMissingMetadata:
    """Tests enforce graceful degradation when metadata is missing/corrupted."""

    def test_get_segment_id_handles_missing_metadata(self):
        """CONTRACT: get_segment_id() returns empty string when metadata missing."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        del sentinel.metadata['segment_id']

        result = get_segment_id(sentinel)

        assert result == ""

    def test_is_segment_boundary_handles_missing_flag(self):
        """CONTRACT: is_segment_boundary() returns False for non-sentinel messages."""
        regular_message = Message(content="test", role="user")

        assert is_segment_boundary(regular_message) is False

    def test_is_active_segment_handles_missing_status(self):
        """CONTRACT: is_active_segment() returns False when status missing."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        del sentinel.metadata['status']

        assert is_active_segment(sentinel) is False

    def test_is_active_segment_handles_unexpected_status(self):
        """CONTRACT: is_active_segment() returns False for non-active statuses."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        # Collapsed
        sentinel.metadata['status'] = "collapsed"
        assert is_active_segment(sentinel) is False

        # Unexpected value
        sentinel.metadata['status'] = "archived"
        assert is_active_segment(sentinel) is False

    def test_get_segment_time_range_handles_missing_timestamps(self):
        """CONTRACT: get_segment_time_range() doesn't crash on missing timestamps."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        del sentinel.metadata['segment_start_time']
        del sentinel.metadata['segment_end_time']

        # Should return valid datetimes (fallback to now)
        start, end = get_segment_time_range(sentinel)

        assert isinstance(start, datetime)
        assert isinstance(end, datetime)


    def test_add_tools_handles_missing_tools_used_key(self):
        """CONTRACT: add_tools_to_segment() works even if tools_used missing."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        del sentinel.metadata['tools_used']

        add_tools_to_segment(sentinel, ["webaccess_tool"])

        assert sentinel.metadata['tools_used'] == ["webaccess_tool"]


class TestRealWorkflow:
    """Integration tests using sentinels in realistic segment workflow."""

    def test_complete_segment_lifecycle(self):
        """Verify sentinel can go through complete lifecycle: create → update → collapse → process."""
        # Create active segment
        sentinel = create_segment_boundary_sentinel(utc_now(), "continuum-123")
        assert is_active_segment(sentinel)

        # Add tools as conversation progresses
        add_tools_to_segment(sentinel, ["webaccess_tool"])
        add_tools_to_segment(sentinel, ["reminder_tool"])

        # Segment times out and collapses (returns new Message)
        embedding = [0.5] * 384
        summary = "Web research and reminders"
        display_title = "Web research"
        collapsed = collapse_segment_sentinel(sentinel, summary, display_title, embedding, 60)
        assert not is_active_segment(collapsed)
        assert display_title in collapsed.content
        assert summary in collapsed.content

        # Downstream processing marks segment (mutates metadata)
        mark_segment_processed(collapsed, memories_extracted=True, memory_count=3)
        mark_segment_processed(collapsed, domain_blocks_updated=True)

        # Verify final state
        assert collapsed.metadata['status'] == "collapsed"
        assert collapsed.metadata['tools_used'] == ["reminder_tool", "webaccess_tool"]  # Sorted
        assert collapsed.metadata['segment_embedding_value'] == embedding
        assert collapsed.metadata['memories_extracted'] is True
        assert collapsed.metadata['memory_count'] == 3
        assert collapsed.metadata['domain_blocks_updated'] is True

    def test_multiple_segments_have_unique_ids(self):
        """Verify multiple segments for same user/continuum have unique IDs."""
        # Create 5 segments for same continuum
        segments = [
            create_segment_boundary_sentinel(utc_now(), "continuum-1")
            for _ in range(5)
        ]

        # All segment IDs are unique
        segment_ids = [get_segment_id(s) for s in segments]
        assert len(segment_ids) == len(set(segment_ids))


class TestComplexityScoring:
    """Tests enforce complexity score storage and handling guarantees."""

    def test_collapse_stores_complexity_score(self):
        """CONTRACT: collapse_segment_sentinel() stores complexity_score in metadata."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        collapsed = collapse_segment_sentinel(
            sentinel,
            summary="Test summary",
            display_title="Test",
            embedding=None,
            inactive_duration_minutes=60,
            complexity_score=3
        )

        assert collapsed.metadata['complexity_score'] == 3

    def test_collapse_defaults_complexity_to_two(self):
        """CONTRACT: complexity_score defaults to 2 (moderate) when not provided."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        # Omit complexity_score parameter
        collapsed = collapse_segment_sentinel(
            sentinel,
            summary="Test summary",
            display_title="Test",
            embedding=None,
            inactive_duration_minutes=60
        )

        assert collapsed.metadata['complexity_score'] == 2

    def test_all_complexity_values_are_valid(self):
        """CONTRACT: Complexity scores 1, 2, 3 are all valid and stored correctly."""
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")

        for complexity in [1, 2, 3]:
            collapsed = collapse_segment_sentinel(
                sentinel,
                summary=f"Complexity {complexity} summary",
                display_title=f"Complexity {complexity}",
                embedding=None,
                inactive_duration_minutes=60,
                complexity_score=complexity
            )

            assert collapsed.metadata['complexity_score'] == complexity

    def test_complexity_persists_through_lifecycle(self):
        """CONTRACT: Complexity score persists through complete segment lifecycle."""
        # Create and collapse with complexity
        sentinel = create_segment_boundary_sentinel(utc_now(), "cid")
        embedding = [0.3] * 384

        collapsed = collapse_segment_sentinel(
            sentinel,
            summary="Complex analysis summary",
            display_title="Complex analysis",
            embedding=embedding,
            inactive_duration_minutes=90,
            complexity_score=3,
            tools_used=["tool_a", "tool_b", "tool_c", "tool_d"]
        )

        # Mark as processed
        mark_segment_processed(collapsed, memories_extracted=True, memory_count=5)

        # Complexity still present
        assert collapsed.metadata['complexity_score'] == 3
        assert collapsed.metadata['status'] == "collapsed"
        assert collapsed.metadata['memories_extracted'] is True
