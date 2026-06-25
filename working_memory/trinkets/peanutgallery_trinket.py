"""
Peanut Gallery metacognitive observer trinket.

Displays metacognitive directives from the Peanut Gallery observer system in
the notification center (HUD). Supports concern alerts and coaching directives
with turn-based TTL expiry.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Any, TypedDict
from uuid import uuid4

from working_memory.trinkets.base import StatefulTrinket
from utils.user_context import get_current_user_id


class ActiveGuidance(TypedDict):
    """Active guidance entry returned by get_active_guidance()."""
    id: str
    type: Literal["concern", "coaching", "initiative"]
    text: str
    turns_remaining: int
    critical: bool

logger = logging.getLogger(__name__)


@dataclass
class GuidanceEntry:
    """A single guidance message with TTL tracking."""
    id: str
    guidance_type: Literal["concern", "coaching", "initiative"]
    text: str
    expires_at_turn: int
    critical: bool = False


class PeanutGalleryTrinket(StatefulTrinket):
    """
    Displays metacognitive directives from the Peanut Gallery observer.

    Guidance automatically expires after a configurable number of turns (TTL)
    and is cleared entirely when the segment collapses (via WorkingMemory flush).
    """

    variable_name = "peanutgallery_guidance"

    def __init__(self, event_bus, working_memory, default_ttl: int = 2):
        """
        Initialize with state tracking.

        Args:
            event_bus: CNS event bus for publishing content
            working_memory: Working memory instance for registration
            default_ttl: Default turns until guidance expires
        """
        super().__init__(event_bus, working_memory)

        self._user_guidance: dict[str, Dict[str, GuidanceEntry]] = {}
        self._default_ttl = default_ttl

        logger.info(f"PeanutGalleryTrinket initialized with {default_ttl}-turn default TTL")

    def _expire_items(self) -> bool:
        """Remove guidance entries past their TTL."""
        user_id = get_current_user_id()
        guidance = self._user_guidance.get(user_id, {})
        expired = [
            gid for gid, entry in guidance.items()
            if self.current_turn > entry.expires_at_turn
        ]

        for gid in expired:
            del guidance[gid]

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired guidance entries")

        return bool(expired)

    def _clear_all_state(self) -> None:
        """Clear guidance entries for the current user on segment collapse."""
        guidance = self._user_guidance.pop(get_current_user_id(), None)
        if guidance:
            logger.info(f"Clearing {len(guidance)} guidance entries on segment collapse")

    def get_active_guidance(self) -> list[ActiveGuidance]:
        """Get all currently active (non-expired) guidance messages."""
        user_id = get_current_user_id()
        guidance = self._user_guidance.get(user_id, {})
        return [
            {
                "id": entry.id,
                "type": entry.guidance_type,
                "text": entry.text,
                "turns_remaining": max(0, entry.expires_at_turn - self.current_turn),
                "critical": entry.critical,
            }
            for entry in guidance.values()
            if self.current_turn <= entry.expires_at_turn
        ]

    def add_guidance(
        self,
        guidance_type: Literal["concern", "coaching", "initiative"],
        text: str,
        ttl: int | None = None,
        critical: bool = False
    ) -> str:
        """
        Add a new guidance message.

        Args:
            guidance_type: "concern" or "coaching"
            text: The guidance text
            ttl: Turns until expiry (uses default if None)
            critical: Whether this warrants 4th-wall-breaking with the user

        Returns:
            Unique guidance ID
        """
        guidance_id = str(uuid4())[:8]
        ttl = ttl if ttl is not None else self._default_ttl

        user_id = get_current_user_id()
        guidance = self._user_guidance.setdefault(user_id, {})
        guidance[guidance_id] = GuidanceEntry(
            id=guidance_id,
            guidance_type=guidance_type,
            text=text,
            expires_at_turn=self.current_turn + ttl,
            critical=critical
        )

        logger.info(f"Added {guidance_type} guidance (id={guidance_id}, ttl={ttl}, critical={critical})")
        return guidance_id

    def handle_update_request(self, event) -> None:
        """Process incoming guidance from PeanutGalleryService."""
        context = event.context
        if context.get('action') == 'add_guidance':
            guidance_type = context.get('type')
            text = context.get('text')
            if guidance_type and text:
                self.add_guidance(
                    guidance_type, text,
                    ttl=context.get('ttl'),
                    critical=context.get('critical', False)
                )

        super().handle_update_request(event)

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate HUD content showing all active guidance.

        If any active guidance is marked critical, includes the 4th-wall-breaking
        directive authorizing the model to explicitly acknowledge slips to the user.
        Standard (non-critical) guidance is delivered quietly in the background.
        """
        active = self.get_active_guidance()
        if not active:
            return ""

        standard_guidance = [g for g in active if not g["critical"] and g["type"] != "initiative"]
        initiative_guidance = [g for g in active if g["type"] == "initiative"]
        critical_guidance = [guidance for guidance in active if guidance["critical"]]
        parts = ['<mira:peanutgallery>']

        if standard_guidance:
            parts.append('  <section mode="standard">')
            parts.append('    <instruction>Metacognitive guidance: silently repair the flagged issue in your next reply. Do not reference this notification to the user.</instruction>')
            for guidance in standard_guidance:
                parts.append(
                    f'    <guidance type="{guidance["type"]}" expires_in="{guidance["turns_remaining"]}_turns">'
                    f'{guidance["text"]}'
                    f'</guidance>'
                )
            parts.append('  </section>')

        if initiative_guidance:
            parts.append('  <section mode="initiative">')
            parts.append('    <instruction>Conversational stewardship directive: make the one bounded improvement described below. Do not expand into a full guide, ask extra questions, or turn this into a lecture.</instruction>')
            for guidance in initiative_guidance:
                parts.append(
                    f'    <guidance type="{guidance["type"]}" expires_in="{guidance["turns_remaining"]}_turns">'
                    f'{guidance["text"]}'
                    f'</guidance>'
                )
            parts.append('  </section>')

        if critical_guidance:
            parts.append('  <section mode="critical">')
            parts.append('    ⟦PEANUT_GALLERY_DIRECTIVE⟧')
            parts.append('    <instruction>Immediate metacognitive directive. If the flagged issue is still unresolved, repair it in your next reply. Self-aware repair behavior is allowed: explicitly acknowledge the slip to the user when it helps restore accuracy and trust. Example: "Oops, I\'m sorry, I promised to make a tool call for you a moment ago and I got a notification in my context window that I didn\'t make it. Would you like me to do it now?" Example: "Earlier I told you something inaccurate. I got an alert in my context window just now that let me know. I checked it and it was in fact wrong. Here\'s the correction."</instruction>')
            for guidance in critical_guidance:
                parts.append(
                    f'    <guidance type="{guidance["type"]}" expires_in="{guidance["turns_remaining"]}_turns" critical="true">'
                    f'{guidance["text"]}'
                    f'</guidance>'
                )
            parts.append('  </section>')

        parts.append('</mira:peanutgallery>')
        return "\n".join(parts)
