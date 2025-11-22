"""
Summary generation service for CNS.

Provides out-of-band summary generation with predefined prompts,
proper message storage, and continuum integration.
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

from cns.core.message import Message
from cns.core.continuum import Continuum
from cns.infrastructure.continuum_repository import ContinuumRepository
from clients.llm_provider import LLMProvider
from utils.timezone_utils import utc_now
from utils.tag_parser import TagParser
from config.config_manager import config

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    SEGMENT = "segment_summary"  # Time-based segment summary for manifest


@dataclass
class SummaryPrompt:
    """Predefined prompt configuration for summary generation."""
    system_prompt: str
    user_template: str
    max_tokens: int = 400
    temperature: float = 0.1
    include_metadata: bool = True

    def format(self, **kwargs) -> str:
        """Format the user template with provided values."""
        return self.user_template.format(**kwargs)


class SummaryGenerator:
    """
    Service for generating summaries outside normal continuum flow.

    Handles LLM-based summary generation, message persistence, and
    continuum cache integration.
    """

    def __init__(self,
                 repository: ContinuumRepository,
                 llm_provider: Optional[LLMProvider] = None):
        """
        Initialize summary generator.

        Args:
            repository: Repository for message persistence
            llm_provider: Optional LLM provider (creates default if not provided)
        """
        self.repository = repository
        self.llm_provider = llm_provider or self._create_default_llm()
        self.tag_parser = TagParser()
        self._load_prompts()
        
    def _create_default_llm(self) -> LLMProvider:
        """Create default LLM provider for summary generation (uses Haiku for cost/speed)."""
        return LLMProvider(
            model="claude-haiku-4-5",
            temperature=1.0,
            max_tokens=1000,  # Sufficient for rich summaries without extended thinking
            enable_prompt_caching=False  # No caching needed for one-shot summaries
        )

    def _load_prompts(self):
        """Load prompts from files."""
        prompts_dir = Path("config/prompts")

        # Load segment summary prompts
        segment_system_path = prompts_dir / "segment_summary_system.txt"
        segment_user_path = prompts_dir / "segment_summary_user.txt"

        if not segment_system_path.exists() or not segment_user_path.exists():
            raise FileNotFoundError(f"Segment summary prompts not found in {prompts_dir}")

        with open(segment_system_path, 'r') as f:
            segment_system_prompt = f.read().strip()
        with open(segment_user_path, 'r') as f:
            segment_user_template = f.read().strip()

        # Store prompts in a dictionary
        self.PROMPTS = {
            SummaryType.SEGMENT: SummaryPrompt(
                system_prompt=segment_system_prompt,
                user_template=segment_user_template,
                max_tokens=600,  # Rich synopsis + telegraphic title
                temperature=1.0  # Higher temp for natural, varied summaries
            )
        }
    
    def generate_summary(self,
                             messages: Optional[List[Message]],
                             summary_type: SummaryType,
                             tools_used: Optional[List[str]] = None,
                             content_override: Optional[str] = None) -> Tuple[str, str, int]:
        """
        Generate a summary (text only, does not persist).

        Args:
            messages: Messages to summarize (optional if content_override provided)
            summary_type: Type of summary to generate (SEGMENT)
            tools_used: List of tool names used in segment
            content_override: Pre-formatted content to summarize

        Returns:
            Tuple of (synopsis_text, display_title, complexity_score)
            where complexity_score is 1 (simple), 2 (moderate), or 3 (complex)

        Raises:
            ValueError: If summary generation fails
        """
        try:
            # Get prompt configuration
            if summary_type in self.PROMPTS:
                prompt_config = self.PROMPTS[summary_type]
            else:
                raise ValueError(f"No prompt defined for {summary_type}")
            
            # Prepare prompt based on summary type
            if content_override:
                # Use pre-formatted content (e.g., for coalescence)
                prompt_text = prompt_config.format(conversation_text=content_override)
            else:
                prompt_text = self._prepare_prompt(messages, summary_type, prompt_config, tools_used)
            
            # Format system prompt with segment timestamp
            segment_time = self._get_segment_time(messages)
            formatted_system_prompt = prompt_config.system_prompt.format(
                current_time=segment_time
            )
            
            # Generate summary via LLM with system/user messages
            llm_messages = [
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": prompt_text}
            ]

            response = self.llm_provider.generate_response(
                messages=llm_messages,
                temperature=prompt_config.temperature,
                max_tokens=prompt_config.max_tokens,
                thinking_enabled=False  # Disable extended thinking for summaries
            )

            raw_summary_output = self.llm_provider.extract_text_content(response)

            # Extract synopsis, display title, and complexity from output
            synopsis, display_title, complexity = self._extract_summary_components(raw_summary_output)

            logger.info(
                f"Generated {summary_type.value} (complexity={complexity}), "
                f"summarizing {len(messages) if messages else 'pre-formatted'} messages"
            )

            return synopsis.strip(), display_title, complexity
        except Exception as e:
            logger.error(f"Failed to generate {summary_type.value}: {str(e)}")
            raise ValueError(f"Summary generation failed: {str(e)}") from e
    
    def _prepare_prompt(self,
                       messages: List[Message],
                       summary_type: SummaryType,
                       prompt_config: SummaryPrompt,
                       tools_used: Optional[List[str]] = None) -> str:
        """Prepare the prompt text based on summary type."""
        # Format messages for summarization
        conversation_text = self._format_messages_for_llm(messages)

        # For SEGMENT summaries, include tools_used information
        if summary_type == SummaryType.SEGMENT:
            tools_text = "None" if not tools_used else ", ".join(tools_used)
            return prompt_config.format(
                conversation_text=conversation_text,
                tools_used=tools_text
            )

        # For other summary types, just use conversation text
        return prompt_config.format(conversation_text=conversation_text)
    
    def _format_messages_for_llm(self, messages: List[Message]) -> str:
        """Format messages into readable continuum text."""
        formatted_lines = []

        for msg in messages:
            # Skip system notifications
            if msg.metadata.get('system_notification', False):
                continue

            role_label = msg.role.upper()
            content = msg.content
            formatted_lines.append(f"{role_label}: {content}")

        return "\n\n".join(formatted_lines)

    def _extract_summary_components(self, summary_output: str) -> Tuple[str, str, int]:
        """
        Extract synopsis, display title, and complexity from LLM output using tag_parser.

        Expected format:
        [Synopsis text here]

        <mira:display_title>[Title here]</mira:display_title>
        <mira:complexity>[1-3]</mira:complexity>

        Args:
            summary_output: Raw LLM output containing synopsis, display title, and complexity

        Returns:
            Tuple of (synopsis, display_title, complexity_score)

        Raises:
            ValueError: If display_title tag is missing (instruction-following failure)
        """
        # Parse tags using tag_parser
        parsed = self.tag_parser.parse_response(summary_output)

        display_title = parsed.get('display_title')
        synopsis = parsed.get('clean_text', summary_output).strip()
        complexity = parsed.get('complexity')

        # Missing display_title indicates LLM instruction-following failure
        if not display_title:
            logger.error(
                f"LLM failed to generate <mira:display_title> tag. "
                f"Output (first 200 chars): {summary_output[:200]}"
            )
            raise ValueError(
                "Summary generation failed: LLM did not include required <mira:display_title> tag. "
                "This indicates instruction-following failure."
            )

        # Default complexity to 2 (moderate) if missing or invalid
        if complexity is None or complexity not in [1, 2, 3]:
            logger.warning(
                f"LLM did not provide valid complexity score (got {complexity}), "
                f"defaulting to 2 (moderate)"
            )
            complexity = 2

        return synopsis, display_title, complexity
    
    def _get_segment_time(self, messages: Optional[List[Message]]) -> str:
        """
        Get formatted timestamp from segment messages.
        
        Args:
            messages: Messages in the segment
            
        Returns:
            Formatted timestamp string
        """
        if messages and len(messages) > 0:
            # Use first message timestamp as segment time
            first_msg_time = messages[0].created_at
            return first_msg_time.strftime("%B %d, %Y")
        else:
            # Fallback to current time if no messages
            return utc_now().strftime("%B %d, %Y")
