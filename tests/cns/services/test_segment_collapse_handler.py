"""
Tests for segment_collapse_handler.py - Real segment collapse orchestration.

Focus: Tests use real database, real events, real repository to verify the
complete collapse pipeline from timeout → summary → persistence → downstream.
Only mocks LLM calls (external/expensive) - everything else is real infrastructure.
"""
import pytest
from unittest.mock import Mock
from uuid import uuid4

from cns.services.segment_collapse_handler import SegmentCollapseHandler
from cns.core.events import SegmentTimeoutEvent, SegmentCollapsedEvent, ManifestUpdatedEvent
from cns.core.message import Message
from cns.services.segment_helpers import (
    create_segment_boundary_sentinel,
    add_tools_to_segment,
    get_segment_id,
    is_active_segment
)
from cns.infrastructure.continuum_repository import get_continuum_repository
from cns.integration.event_bus import EventBus
from utils.timezone_utils import utc_now
from utils.user_context import set_current_user_id
from tests.fixtures.core import TEST_USER_EMAIL, ensure_test_user_exists


@pytest.fixture
def mock_summary_generator():
    """Provide mock summary generator (LLM call is expensive)."""
    generator = Mock()
    # Default: return successful summary
    generator.generate_summary.return_value = Message(
        content="Test summary",
        role="assistant"
    )
    return generator


@pytest.fixture
def mock_embeddings_provider():
    """Provide mock embeddings provider (external API call)."""
    provider = Mock()
    # Default: return 384-dim embedding
    provider.encode_384.return_value = [0.5] * 384
    return provider


class TestRealCollapseOrchestration:
    """Tests verify real collapse flow using actual database and repository."""

    def test_complete_collapse_flow_persists_to_database(
        self,
        authenticated_user,
        conversation_repository,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: Complete flow from timeout → database persistence."""
        user_id = authenticated_user['user_id']
        set_current_user_id(user_id)
        continuum_repo = conversation_repository
        event_bus = EventBus()

        # Create continuum
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        # Create segment sentinel and messages in database
        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        add_tools_to_segment(sentinel, ["webaccess_tool"])
        segment_id = get_segment_id(sentinel)

        msg1 = Message(content="user message", role="user")
        msg2 = Message(content="assistant response", role="assistant")

        # Persist to database
        continuum_repo.save_messages_batch([sentinel, msg1, msg2], continuum_id, user_id)

        # Create handler with real dependencies
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Fire timeout event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        handler.handle_timeout(event)

        # Verify collapsed sentinel persisted to database
        messages = continuum_repo.load_messages(continuum_id, user_id)

        # Find collapsed sentinel
        collapsed_sentinel = next(
            (m for m in messages if get_segment_id(m) == segment_id),
            None
        )

        assert collapsed_sentinel is not None
        assert collapsed_sentinel.metadata['status'] == 'collapsed'
        assert collapsed_sentinel.content == "Test summary"
        assert not is_active_segment(collapsed_sentinel)

    def test_collapsed_sentinel_has_embedding_in_database(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: Embedding persisted to segment_embedding column."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        segment_id = get_segment_id(sentinel)

        continuum_repo.save_message(sentinel, continuum_id, user_id)

        # Create handler
        embedding_vector = [0.7] * 384
        mock_embeddings_provider.encode_384.return_value = embedding_vector

        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        handler.handle_timeout(event)

        # Query database directly for segment_embedding
        from utils.database_session_manager import get_shared_session_manager
        session_manager = get_shared_session_manager()

        with session_manager.get_session(user_id) as session:
            result = session.execute_single("""
                SELECT segment_embedding
                FROM messages
                WHERE id = %s
            """, (sentinel.id,))

            assert result is not None
            assert result['segment_embedding'] is not None  # segment_embedding should be populated

    def test_events_published_to_real_event_bus(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: SegmentCollapsedEvent and ManifestUpdatedEvent published."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        sentinel.metadata['message_count'] = 3
        segment_id = get_segment_id(sentinel)

        msg1 = Message(content="test", role="user")
        continuum_repo.save_messages_batch([sentinel, msg1], continuum_id, user_id)

        # Subscribe to events
        collapsed_events = []
        manifest_events = []

        def capture_collapsed(event):
            collapsed_events.append(event)

        def capture_manifest(event):
            manifest_events.append(event)

        event_bus.subscribe('SegmentCollapsedEvent', capture_collapsed)
        event_bus.subscribe('ManifestUpdatedEvent', capture_manifest)

        # Create handler
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        handler.handle_timeout(event)

        # Verify events published
        assert len(collapsed_events) == 1
        assert len(manifest_events) == 1

        collapsed_event = collapsed_events[0]
        assert collapsed_event.continuum_id == continuum_id
        assert collapsed_event.user_id == user_id
        assert collapsed_event.segment_id == segment_id
        assert collapsed_event.summary == "Test summary"
        assert collapsed_event.message_count == 3

    def test_summary_generator_called_with_correct_messages(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: Summary generator receives correct message list."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        segment_id = get_segment_id(sentinel)

        msg1 = Message(content="first", role="user")
        msg2 = Message(content="second", role="assistant")
        session_boundary = Message(
            content="[Session]",
            role="assistant",
            metadata={'system_notification': True, 'notification_type': 'session_break'}
        )
        msg3 = Message(content="third", role="user")

        # Next segment boundary to stop collection
        next_sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id, user_id)

        continuum_repo.save_messages_batch(
            [sentinel, msg1, msg2, session_boundary, msg3, next_sentinel],
            continuum_id,
            user_id
        )

        # Create handler
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        handler.handle_timeout(event)

        # Verify summary generator called with correct messages
        mock_summary_generator.generate_summary.assert_called_once()
        call_kwargs = mock_summary_generator.generate_summary.call_args.kwargs
        messages_arg = call_kwargs['messages']

        # Should have msg1, msg2, msg3 (session_boundary skipped)
        assert len(messages_arg) == 3
        assert messages_arg[0].content == "first"
        assert messages_arg[1].content == "second"
        assert messages_arg[2].content == "third"

    def test_handles_missing_sentinel_gracefully(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: Returns early without errors when sentinel not found."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum but no sentinel
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        # Subscribe to events
        events_published = []
        event_bus.subscribe('SegmentCollapsedEvent', events_published.append)

        # Create handler
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Fire event for nonexistent segment
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id="nonexistent-segment-id",
            inactive_duration_minutes=60,
            local_hour=14
        )

        # Should not raise
        handler.handle_timeout(event)

        # No events should be published
        assert len(events_published) == 0

        # Summary generator should not be called
        mock_summary_generator.generate_summary.assert_not_called()

    def test_continues_with_fallback_on_summary_generation_failure(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_embeddings_provider
    ):
        """CONTRACT: Uses fallback summary when LLM generation fails."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        segment_id = get_segment_id(sentinel)

        msg1 = Message(content="test", role="user")
        continuum_repo.save_messages_batch([sentinel, msg1], continuum_id, user_id)

        # Summary generator raises exception
        failing_generator = Mock()
        failing_generator.generate_summary.side_effect = Exception("LLM error")

        # Create handler
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=failing_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Subscribe to events
        collapsed_events = []
        event_bus.subscribe('SegmentCollapsedEvent', collapsed_events.append)

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        # Should not raise
        handler.handle_timeout(event)

        # Event should still be published with fallback summary
        assert len(collapsed_events) == 1
        assert "Conversation segment" in collapsed_events[0].summary

    def test_continues_with_none_embedding_on_encoding_failure(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator
    ):
        """CONTRACT: Processing continues when embedding generation fails."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        segment_id = get_segment_id(sentinel)

        continuum_repo.save_message(sentinel, continuum_id, user_id)

        # Embeddings provider raises exception
        failing_embeddings = Mock()
        failing_embeddings.encode_384.side_effect = Exception("Encoding error")

        # Create handler
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=failing_embeddings,
            event_bus=event_bus,
            lt_memory_factory=None
        )

        # Subscribe to events
        collapsed_events = []
        event_bus.subscribe('SegmentCollapsedEvent', collapsed_events.append)

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        # Should not raise
        handler.handle_timeout(event)

        # Event should still be published
        assert len(collapsed_events) == 1

        # Verify sentinel saved without embedding
        messages = continuum_repo.load_messages(continuum_id, user_id)
        collapsed_sentinel = next(m for m in messages if get_segment_id(m) == segment_id)
        assert collapsed_sentinel.metadata['status'] == 'collapsed'
        assert 'segment_embedding_value' not in collapsed_sentinel.metadata


class TestDownstreamOrchestration:
    """Tests verify downstream processing submission."""

    def test_submits_to_lt_memory_when_factory_provided(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: Calls batching.submit_segment_extraction when factory provided."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        segment_id = get_segment_id(sentinel)

        msg1 = Message(content="test", role="user")
        continuum_repo.save_messages_batch([sentinel, msg1], continuum_id, user_id)

        # Create mock lt_memory_factory
        lt_memory_factory = Mock()
        batching_service = Mock()
        batching_service.submit_segment_extraction.return_value = True
        lt_memory_factory.batching = batching_service

        # Create handler with factory
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=lt_memory_factory
        )

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        handler.handle_timeout(event)

        # Verify submission called
        batching_service.submit_segment_extraction.assert_called_once()
        call_kwargs = batching_service.submit_segment_extraction.call_args.kwargs
        assert call_kwargs['user_id'] == user_id
        assert call_kwargs['segment_id'] == segment_id
        assert len(call_kwargs['messages']) == 1

    def test_skips_downstream_when_no_messages(
        self,
        test_user,
        continuum_repo,
        event_bus,
        mock_summary_generator,
        mock_embeddings_provider
    ):
        """CONTRACT: Skips memory extraction when messages list is empty."""
        user_id = test_user['id']
        set_current_user_id(user_id)

        # Create continuum and segment with no messages
        continuum = continuum_repo.create_continuum(user_id)
        continuum_id = str(continuum.id)

        sentinel = create_segment_boundary_sentinel(utc_now(), continuum_id)
        segment_id = get_segment_id(sentinel)

        continuum_repo.save_message(sentinel, continuum_id, user_id)

        # Create mock lt_memory_factory
        lt_memory_factory = Mock()
        batching_service = Mock()
        lt_memory_factory.batching = batching_service

        # Create handler
        handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=mock_summary_generator,
            embeddings_provider=mock_embeddings_provider,
            event_bus=event_bus,
            lt_memory_factory=lt_memory_factory
        )

        # Fire event
        event = SegmentTimeoutEvent.create(
            continuum_id=continuum_id,
            user_id=user_id,
            segment_id=segment_id,
            inactive_duration_minutes=60,
            local_hour=14
        )

        handler.handle_timeout(event)

        # Verify submission NOT called (no messages)
        batching_service.submit_segment_extraction.assert_not_called()
