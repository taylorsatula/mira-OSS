"""
Domain Knowledge Trinket - Injects enabled domain-specific knowledge blocks into system prompt.

Fetches enabled domain blocks from Letta and formats them for system prompt injection.
"""
import logging
from typing import Dict, Any

from working_memory.trinkets.base import EventAwareTrinket
from cns.services.domain_knowledge_service import get_domain_knowledge_service

logger = logging.getLogger(__name__)


class DomainKnowledgeTrinket(EventAwareTrinket):
    """
    Trinket that injects enabled domain knowledge blocks into the system prompt.

    Retrieves enabled domain blocks (managed by Letta sleeptime agents) and formats
    them as XML-tagged sections for injection into MIRA's context.
    """

    def _get_variable_name(self) -> str:
        """Return variable name for system prompt composition."""
        return "domain_knowledge"

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate domain knowledge content from enabled blocks.

        Domain knowledge is an optional feature. If not configured, returns empty.
        If configured but infrastructure fails, exceptions propagate.

        Args:
            context: Context dict with user_id

        Returns:
            Formatted domain blocks content or empty string

        Raises:
            DatabaseError: If database queries fail
            ValkeyError: If cache operations fail
        """
        user_id = context.get('user_id')
        if not user_id:
            logger.warning("No user_id in context for DomainKnowledgeTrinket")
            return ""

        # Domain knowledge is optional - if not configured, return empty
        service = get_domain_knowledge_service(self.event_bus)
        if not service:
            logger.debug("Domain knowledge service not configured (optional feature)")
            return ""

        # Service is configured - infrastructure failures should propagate
        enabled_domains = service.get_enabled_domains()

        if not enabled_domains:
            logger.debug(f"No enabled domain blocks for user {user_id}")
            return ""  # Legitimately empty - user hasn't enabled any domains

        # Fetch content for each enabled domain
        domain_contents = []
        for domain in enabled_domains:
            domain_label = domain['domain_label']
            content = service.get_block_content(
                domain_label=domain_label,
                prompt_formatted=True  # Get XML-wrapped format
            )
            if content:
                domain_contents.append(content)

        if not domain_contents:
            return ""  # Domains enabled but no content available

        # Build formatted section
        header = "<domain_knowledge>\nContext-specific knowledge blocks currently active:\n\n"
        footer = "\n</domain_knowledge>"

        return header + "\n\n".join(domain_contents) + footer
