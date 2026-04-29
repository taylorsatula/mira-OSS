"""
Peanut Gallery metacognitive observer service.

Orchestrates the Peanut Gallery observation system:
- Triggers every N turns (fire-and-forget async)
- Curates seed memories from recent conversation
- Runs two-stage LLM evaluation (prerunner + observer)
- Applies results: compaction modifies cache, guidance injects into HUD
"""
import json
import logging
import contextvars
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from uuid import uuid4

from cns.core.message import Message
from cns.core.events import TurnCompletedEvent, UpdateTrinketEvent
from cns.integration.event_bus import EventBus
from cns.infrastructure.valkey_message_cache import ValkeyMessageCache
from cns.services.peanutgallery_model import PeanutGalleryModel, PeanutGalleryResult
# Peanut Gallery tuning
PG_TRIGGER_INTERVAL = 10     # Run observer every N turns
PG_SEED_MEMORY_COUNT = 10    # Seed memories for prerunner relevance selection
PG_GUIDANCE_TTL_TURNS = 5    # Turns before guidance expires from HUD
from lt_memory.models import MemoryDict
from lt_memory.proactive import ProactiveService
from utils.timezone_utils import format_utc_iso, utc_now
from utils.user_context import get_current_user_id

logger = logging.getLogger(__name__)

from utils.instance import user_data_base

USER_DATA_BASE = user_data_base()


class PeanutGalleryService:
    """
    Orchestrates the Peanut Gallery metacognitive observer.

    Every N turns, asynchronously:
    1. Gets recent messages from Valkey cache
    2. Curates seed memories via ProactiveService
    3. Runs PeanutGalleryModel evaluation
    4. Applies results (compaction, guidance injection)

    Fire-and-forget pattern ensures observation doesn't block conversation.
    """

    def __init__(
        self,
        model: PeanutGalleryModel,
        valkey_cache: ValkeyMessageCache,
        event_bus: EventBus,
        proactive_service: ProactiveService
    ):
        self.model = model
        self.valkey_cache = valkey_cache
        self.event_bus = event_bus
        self.proactive = proactive_service

        # Thread pool for async execution (single worker - one observation at a time)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="peanutgallery"
        )

        # Subscribe to turn completed events
        event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)

        logger.info(
            f"PeanutGalleryService initialized: "
            f"trigger_interval={PG_TRIGGER_INTERVAL}"
        )

    def _handle_turn_completed(self, event: TurnCompletedEvent) -> None:
        """
        Handle turn completion - check if it's time for observation.

        Runs every N turns based on segment_turn_number.
        Fire-and-forget async execution to avoid blocking the conversation.
        """
        if event.segment_turn_number % PG_TRIGGER_INTERVAL != 0:
            return

        logger.debug(
            f"Triggering Peanut Gallery observation at segment turn {event.segment_turn_number}"
        )

        # Copy context for the background thread
        ctx = contextvars.copy_context()

        # Fire-and-forget: submit to thread pool with context
        self._executor.submit(ctx.run, self._run_observation)

    def _run_observation(self) -> None:
        """
        Execute Peanut Gallery observation and apply results.

        Errors are logged but don't affect the main conversation flow.
        """
        try:
            # Get current messages from cache
            messages = self.valkey_cache.get_continuum()
            if not messages:
                logger.debug("No messages in cache for observation")
                return

            # Curate seed memories
            seed_memories = self._curate_seed_memories(messages)

            # Run model evaluation
            result = self.model.evaluate(messages, seed_memories)

            # Apply result based on action type
            self._apply_result(result, messages)

        except Exception as e:
            # Non-critical: log and continue
            logger.warning(f"Peanut Gallery observation failed (non-critical): {e}")

    def _curate_seed_memories(self, messages: List[Message]) -> list[MemoryDict]:
        """
        Curate seed memories from recent conversation for observer context.

        Uses the last user message as a query to find relevant memories.
        Returns list of memory dicts with id, text, importance_score.
        """
        # Find last user message for query
        last_user_content = ""
        for msg in reversed(messages):
            if msg.role == "user" and msg.content:
                last_user_content = msg.content
                if isinstance(last_user_content, list):
                    # Extract text from multimodal content
                    last_user_content = " ".join(
                        item.get('text', '')
                        for item in last_user_content
                        if isinstance(item, dict) and item.get('type') == 'text'
                    )
                break

        if not last_user_content:
            return []

        # Use ProactiveService to search for relevant memories
        from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
        embeddings = get_hybrid_embeddings_provider()

        query_embedding = embeddings.encode_realtime(last_user_content[:500])

        return self.proactive.search_with_embedding(
            embedding=query_embedding,
            query_expansion=last_user_content[:500],
            limit=PG_SEED_MEMORY_COUNT
        )

    def _apply_result(
        self,
        result: PeanutGalleryResult,
        messages: List[Message]
    ) -> None:
        """
        Apply Peanut Gallery result based on action type.

        Args:
            result: Evaluation result from model
            messages: Current message list from cache
        """
        if result.action_type == "noop":
            logger.debug("Peanut Gallery: noop (no action needed)")
            return

        # DISABLED: Compaction breaks the Anthropic cache prefix because it rewrites
        # conversation history. The Anthropic API caches requests based on a prefix
        # derived from the message sequence. When compaction replaces a range of
        # messages with a synopsis, subsequent requests no longer match the cached
        # prefix, causing cache misses and potentially confusing the model with
        # the reflowed history.
        # FUTURE: Refactor to work like Windows Defragment - reorganize blocks across
        # the whole context window rather than collapsing ranges inline. This would
        # preserve cache prefix integrity by keeping message boundaries stable while
        # still making room in the context window. That is a larger project for later.
        # if result.action_type == "compaction":
        #     self._apply_compaction(result, messages)

        elif result.action_type in ("concern", "coaching"):
            self._inject_guidance(result.action_type, result.guidance)

    def _apply_compaction(
        self,
        result: PeanutGalleryResult,
        messages: List[Message]
    ) -> None:
        """
        Apply compaction result - collapse message range into synopsis.

        Args:
            result: Result with range_start, range_end, synopsis
            messages: Current message list from cache
        """
        if not all([result.range_start, result.range_end, result.synopsis]):
            logger.debug("Incomplete compaction result, skipping")
            return

        # Build ID -> index mapping (8-char prefix)
        id_to_idx = {str(m.id)[:8]: i for i, m in enumerate(messages)}

        start_idx = id_to_idx.get(result.range_start)
        end_idx = id_to_idx.get(result.range_end)

        if start_idx is None or end_idx is None:
            logger.warning(
                f"Compaction IDs not found: {result.range_start}, {result.range_end}"
            )
            return

        # Ensure start < end
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Validate range makes sense (at least 2 messages)
        if end_idx - start_idx < 2:
            logger.debug("Compaction range too small, skipping")
            return

        # Get timestamps for continuum_tool drill-down access
        start_time = format_utc_iso(messages[start_idx].created_at)
        end_time = format_utc_iso(messages[end_idx].created_at)

        # Create synopsis message with metadata for drill-down
        synopsis_content = (
            f'<mira:compacted_segment start_time="{start_time}" end_time="{end_time}">'
            f'{result.synopsis}'
            f'</mira:compacted_segment>'
        )

        synopsis_message = Message(
            id=str(uuid4()),
            content=synopsis_content,
            role="assistant",
            created_at=utc_now(),
            metadata={
                "is_compaction_synopsis": True,
                "compacted_count": end_idx - start_idx + 1,
                "original_start_time": start_time,
                "original_end_time": end_time
            }
        )

        # Replace range in cache
        collapsed_messages = messages[start_idx:end_idx + 1]
        new_messages = (
            messages[:start_idx] +
            [synopsis_message] +
            messages[end_idx + 1:]
        )

        # Log the compaction to JSONL file before modifying cache
        self._log_compaction(collapsed_messages, result.synopsis, start_time, end_time)

        self.valkey_cache.set_continuum(new_messages)

        logger.info(
            f"Peanut Gallery compaction: collapsed {end_idx - start_idx + 1} messages "
            f"({result.range_start}..{result.range_end})"
        )

    def _inject_guidance(
        self,
        guidance_type: str,
        guidance_text: str
    ) -> None:
        """
        Inject guidance into HUD via PeanutGalleryTrinket.

        Args:
            guidance_type: "concern" or "coaching"
            guidance_text: The guidance message
        """
        if not guidance_text:
            logger.debug(f"Empty {guidance_type} guidance, skipping")
            return

        # Publish update to PeanutGalleryTrinket
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id="peanutgallery",  # Placeholder, trinket uses context
            target_trinket="PeanutGalleryTrinket",
            context={
                "action": "add_guidance",
                "type": guidance_type,
                "text": guidance_text,
                "ttl": PG_GUIDANCE_TTL_TURNS
            }
        ))

        logger.info(f"Peanut Gallery {guidance_type}: {guidance_text[:80]}...")

    def _get_log_dir(self) -> Path:
        """Get or create the peanutgallery log directory for current user."""
        user_id = get_current_user_id()
        log_dir = USER_DATA_BASE / user_id / "peanutgallery"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def _log_compaction(
        self,
        collapsed_messages: List[Message],
        synopsis: str,
        start_time: str,
        end_time: str
    ) -> None:
        """
        Log compaction operation to JSONL file in user's data directory.

        Creates a searchable audit trail of all compaction operations.
        """
        try:
            log_dir = self._get_log_dir()

            # Generate unique filename with timestamp
            timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid4())[:8]
            log_file = log_dir / f"compaction_{timestamp}_{unique_id}.jsonl"

            # Build transcript of collapsed messages
            transcript = []
            for msg in collapsed_messages:
                transcript.append({
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": format_utc_iso(msg.created_at) if msg.created_at else None
                })

            # Create log record
            record = {
                "timestamp": format_utc_iso(utc_now()),
                "operation": "peanutgallery_compaction",
                "collapsed_count": len(collapsed_messages),
                "time_range": {
                    "start": start_time,
                    "end": end_time
                },
                "synopsis": synopsis,
                "collapsed_transcript": transcript
            }

            # Write as single JSONL line
            with open(log_file, 'w') as f:
                f.write(json.dumps(record) + '\n')

            logger.debug(f"Compaction logged to {log_file}")

        except Exception as e:
            # Logging failure is non-critical - don't block the compaction
            logger.warning(f"Failed to log compaction operation: {e}")
