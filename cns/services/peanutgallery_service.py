"""
Peanut Gallery metacognitive observer service.

Orchestrates the Peanut Gallery observation system:
- Triggers every N turns (fire-and-forget async)
- Uses the TurnCompletedEvent continuum snapshot as its evidence source
- Runs observer evaluation on current-session evidence
- Applies guidance results into HUD
"""
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor

from cns.core.events import TurnCompletedEvent, UpdateTrinketEvent
from cns.core.message import Message
from cns.integration.event_bus import EventBus
from cns.services.peanutgallery_model import PeanutGalleryModel, PeanutGalleryResult

logger = logging.getLogger(__name__)

# Peanut Gallery tuning
PG_TRIGGER_INTERVAL = 5      # Run observer every N turns (broad periodic correction)
PG_GUIDANCE_TTL_TURNS = 2    # Short-lived directive: repair now or let it disappear


class PeanutGalleryService:
    """
    Orchestrates the Peanut Gallery metacognitive observer.

    Every N turns, asynchronously:
    1. Captures the turn-completed continuum snapshot
    2. Runs PeanutGalleryModel evaluation
    3. Applies guidance results into HUD

    Fire-and-forget pattern ensures observation doesn't block conversation.
    """

    def __init__(
        self,
        model: PeanutGalleryModel,
        event_bus: EventBus,
    ):
        self.model = model
        self.event_bus = event_bus

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="peanutgallery"
        )

        event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)

        logger.info(
            "PeanutGalleryService initialized: trigger_interval=%d",
            PG_TRIGGER_INTERVAL,
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
            "Triggering Peanut Gallery observation at segment turn %d",
            event.segment_turn_number,
        )

        messages_snapshot = tuple(event.continuum.messages)
        ctx = contextvars.copy_context()
        self._executor.submit(ctx.run, self._run_observation, messages_snapshot)

    def _run_observation(self, messages_snapshot: tuple[Message, ...]) -> None:
        """
        Execute Peanut Gallery observation and apply results.

        Errors are logged but don't affect the main conversation flow.
        """
        try:
            if not messages_snapshot:
                logger.debug("No messages in snapshot for observation")
                return

            result = self.model.evaluate(list(messages_snapshot))
            self._apply_result(result)

        except Exception as e:
            logger.warning("Peanut Gallery observation failed (non-critical): %s", e)

    def _apply_result(self, result: PeanutGalleryResult) -> None:
        """Apply Peanut Gallery result to working memory."""
        if result.action_type == "noop":
            logger.debug("Peanut Gallery: noop (no action needed)")
            return

        if result.action_type in ("concern", "coaching"):
            self._inject_guidance(result.action_type, result.guidance, result.critical)
            return

        logger.warning("Unsupported Peanut Gallery action type: %s", result.action_type)

    def _inject_guidance(
        self,
        guidance_type: str,
        guidance_text: str,
        critical: bool = False
    ) -> None:
        """
        Inject guidance into HUD via PeanutGalleryTrinket.

        Args:
            guidance_type: "concern" or "coaching"
            guidance_text: The guidance message
            critical: Whether this guidance warrants 4th-wall-breaking
        """
        if not guidance_text:
            logger.debug("Empty %s guidance, skipping", guidance_type)
            return

        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id="peanutgallery",
            target_trinket="PeanutGalleryTrinket",
            context={
                "action": "add_guidance",
                "type": guidance_type,
                "text": guidance_text,
                "ttl": PG_GUIDANCE_TTL_TURNS,
                "critical": critical,
            }
        ))

        logger.info("Peanut Gallery %s (critical=%s): %s...", guidance_type, critical, guidance_text[:80])
