"""
Summary generation service for CNS.

Provides out-of-band summary generation with predefined prompts,
proper message storage, and continuum integration.
"""
import logging
from typing import List, Optional
from enum import Enum
from dataclasses import dataclass

from cns.core.message import Message, preprocess_content_blocks
from cns.infrastructure.continuum_repository import ContinuumRepository
from clients.llm_provider import LLMProvider, ContextOverflowError
from utils.timezone_utils import utc_now
from utils.tag_parser import TagParser

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries that can be generated."""
    SEGMENT = "segment_summary"  # Time-based segment summary for manifest


@dataclass
class SummaryPrompt:
    """Predefined prompt configuration for summary generation."""
    system_prompt: str
    user_template: str
    include_metadata: bool = True

    def format(self, **kwargs) -> str:
        """Format the user template with provided values."""
        return self.user_template.format(**kwargs)


@dataclass(frozen=True)
class SummaryResult:
    """Result of summary generation: synopsis, precis, display title, and complexity score."""
    synopsis: str
    precis: str
    display_title: str
    complexity: float


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

        # Use provided LLM or create default (routing via internal_llm= per-call)
        self.llm_provider = llm_provider or LLMProvider()
        self.tag_parser = TagParser()
        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files."""
        from config.prompts.loader import load_prompt

        # Load segment summary prompts
        segment_system_prompt = load_prompt("segment_summary_system.txt")
        segment_user_template = load_prompt("segment_summary_user.txt")

        # Store prompts in a dictionary
        self.PROMPTS = {
            SummaryType.SEGMENT: SummaryPrompt(
                system_prompt=segment_system_prompt,
                user_template=segment_user_template,
            )
        }

        # Load synthesis prompts (for merging chunk summaries of oversized segments)
        # Optional at init - will fail at runtime if chunking needed but prompts missing
        self._synthesis_system_prompt = load_prompt("synthesis_summary_system.txt", required=False) or None
        self._synthesis_user_template = load_prompt("synthesis_summary_user.txt", required=False) or None
    
    def generate_summary(self,
                             messages: Optional[List[Message]],
                             summary_type: SummaryType,
                             tools_used: Optional[List[str]] = None,
                             content_override: Optional[str] = None,
                             previous_summaries: Optional[List[Message]] = None) -> SummaryResult:
        """
        Generate a summary (text only, does not persist).

        Args:
            messages: Messages to summarize (optional if content_override provided)
            summary_type: Type of summary to generate (SEGMENT)
            tools_used: List of tool names used in segment
            content_override: Pre-formatted content to summarize
            previous_summaries: Recent collapsed segment sentinels for narrative continuity.
                               The LLM sees these to enable connective phrases like
                               "Building on the auth work from Tuesday..."

        Returns:
            SummaryResult with synopsis, display_title, and complexity score
            (0.5=trivial, 1=simple, 2=moderate, 3=complex)

        Raises:
            ValueError: If summary generation fails
        """
        # Get prompt configuration
        if summary_type in self.PROMPTS:
            prompt_config = self.PROMPTS[summary_type]
        else:
            raise ValueError(f"No prompt defined for {summary_type}")

        # Prepare prompt based on summary type
        if content_override:
            # Use pre-formatted content (e.g., chunked synthesis)
            prompt_text = prompt_config.format(conversation_text=content_override)
        else:
            prompt_text = self._prepare_prompt(
                messages, summary_type, prompt_config, tools_used, previous_summaries
            )

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

        try:
            response = self.llm_provider.generate_response(
                messages=llm_messages,
                internal_llm='summary',
                allow_negative=True  # System task — segment already paid for
            )

            raw_summary_output = self.llm_provider.extract_text_content(response)

            # Extract synopsis, display title, and complexity from output
            result = self._extract_summary_components(raw_summary_output)

            logger.info(
                f"Generated {summary_type.value} (complexity={result.complexity}), "
                f"summarizing {len(messages) if messages else 'pre-formatted'} messages"
            )

            return result

        except ContextOverflowError:
            # Segment too large for single pass - use hierarchical summarization
            if messages and summary_type == SummaryType.SEGMENT:
                logger.info("Segment exceeded context limit, falling back to chunked summarization")
                try:
                    return self._generate_chunked_summary(
                        messages, tools_used, previous_summaries
                    )
                except Exception as e:
                    logger.error(f"Chunked summarization also failed: {e}")
                    return SummaryResult(synopsis="[Segment content not summarized]", precis="", display_title="Large segment archived", complexity=1.0)
            else:
                # Non-segment summaries can't be chunked
                raise

        except Exception as e:
            logger.error(f"Failed to generate {summary_type.value}: {str(e)}")
            raise ValueError(f"Summary generation failed: {str(e)}") from e
    
    def _prepare_prompt(self,
                       messages: List[Message],
                       summary_type: SummaryType,
                       prompt_config: SummaryPrompt,
                       tools_used: Optional[List[str]] = None,
                       previous_summaries: Optional[List[Message]] = None) -> str:
        """
        Prepare the prompt text based on summary type.

        For segment summaries, includes previous summaries for narrative continuity.
        The LLM can then use connective phrases like "Building on Tuesday's work..."
        """
        # Format messages for summarization
        conversation_text = self._format_messages_for_llm(messages)

        # Format previous summaries for continuity (if provided)
        previous_context = self._format_previous_summaries(previous_summaries)

        # For SEGMENT summaries, include tools_used and previous_summaries
        if summary_type == SummaryType.SEGMENT:
            tools_text = "None" if not tools_used else ", ".join(tools_used)
            return prompt_config.format(
                conversation_text=conversation_text,
                tools_used=tools_text,
                previous_summaries=previous_context
            )

        # For other summary types, just use conversation text
        return prompt_config.format(conversation_text=conversation_text)

    def _format_previous_summaries(self, previous_summaries: Optional[List[Message]]) -> str:
        """
        Format previous segment summaries for narrative continuity.

        Each summary includes the display title and timestamp so the LLM can
        reference previous work naturally.
        """
        if not previous_summaries:
            return "No previous segments in this continuum yet."

        formatted_parts = []
        for segment in previous_summaries:
            title = segment.metadata.get('display_title', 'Untitled segment')
            end_time = segment.metadata.get('segment_end_time', '')
            summary_text = segment.content if isinstance(segment.content, str) else str(segment.content)

            # Format as: **Title** (timestamp): summary
            formatted_parts.append(f"**{title}** ({end_time}):\n{summary_text}")

        return "\n\n".join(formatted_parts)
    
    def _format_messages_for_llm(self, messages: List[Message]) -> str:
        """Format messages into readable continuum text, stripping binary/media content."""
        formatted_lines = []

        for msg in messages:
            # Skip system notifications
            if msg.metadata.get('system_notification', False):
                continue

            role_label = msg.role.upper()
            content = msg.content

            # Preprocess structured content (strips media, truncates tool results)
            if isinstance(content, list):
                preprocessed = preprocess_content_blocks(content)
                content = ' '.join(preprocessed.text_parts)
                if preprocessed.image_count > 0:
                    content = f"[{preprocessed.image_count} image(s) shared] {content}".strip()
                if not content:
                    content = f"[{preprocessed.image_count} image(s) shared, no text]"

            formatted_lines.append(f"{role_label}: {content}")

        return "\n\n".join(formatted_lines)

    def _extract_summary_components(self, summary_output: str) -> SummaryResult:
        """
        Extract synopsis, display title, and complexity from LLM output using tag_parser.

        Expected format:
        [Synopsis text here]

        <mira:display_title>[Title here]</mira:display_title>
        <mira:complexity>[0.5-3]</mira:complexity>

        Args:
            summary_output: Raw LLM output containing synopsis, display title, and complexity

        Returns:
            SummaryResult with parsed synopsis, display_title, and complexity_score

        Raises:
            ValueError: If display_title tag is missing (instruction-following failure)
        """
        # Parse tags using tag_parser
        parsed = self.tag_parser.parse_response(summary_output)

        display_title = parsed.get('display_title')
        synopsis = parsed.get('clean_text', summary_output).strip()
        complexity = parsed.get('complexity')
        precis = parsed.get('precis') or ''

        # Missing display_title indicates LLM refused or failed to follow instructions
        # Autocollapse with tombstone instead of retrying forever
        if not display_title:
            logger.warning(
                f"LLM did not generate <mira:display_title> tag - autocollapsing with tombstone. "
                f"Output (first 200 chars): {summary_output[:200]}"
            )
            return SummaryResult(synopsis="[Segment content not summarized]", precis="", display_title="Archived segment", complexity=1.0)

        # Default complexity to 2 (moderate) if missing or invalid
        if complexity is None or complexity not in [0.5, 1, 2, 3]:
            logger.warning(
                f"LLM did not provide valid complexity score (got {complexity}), "
                f"defaulting to 2 (moderate)"
            )
            complexity = 2

        if not precis:
            logger.warning("LLM did not generate <mira:precis> tag — precis will be empty")

        return SummaryResult(synopsis=synopsis, precis=precis, display_title=display_title, complexity=complexity)
    
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

    # --- Hierarchical summarization for oversized segments ---

    # Target ~50k tokens per chunk, estimate 4 chars/token
    _CHUNK_TARGET_CHARS = 200000

    def _chunk_messages(self, messages: List[Message]) -> List[List[Message]]:
        """Split messages into chunks of ~50k tokens each."""
        chunks: List[List[Message]] = []
        current_chunk: List[Message] = []
        current_chars = 0

        for msg in messages:
            msg_chars = len(str(msg.content))
            if current_chars + msg_chars > self._CHUNK_TARGET_CHARS and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [msg]
                current_chars = msg_chars
            else:
                current_chunk.append(msg)
                current_chars += msg_chars

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _summarize_chunk(
        self,
        messages: List[Message],
        tools_used: Optional[List[str]] = None,
        previous_summaries: Optional[List[Message]] = None
    ) -> SummaryResult:
        """Generate summary for a single chunk using standard segment prompts."""
        prompt_config = self.PROMPTS[SummaryType.SEGMENT]
        prompt_text = self._prepare_prompt(
            messages, SummaryType.SEGMENT, prompt_config, tools_used, previous_summaries
        )

        segment_time = self._get_segment_time(messages)
        formatted_system_prompt = prompt_config.system_prompt.format(current_time=segment_time)

        response = self.llm_provider.generate_response(
            messages=[
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            internal_llm='summary',
            allow_negative=True  # System task — segment already paid for
        )

        raw_output = self.llm_provider.extract_text_content(response)
        return self._extract_summary_components(raw_output)

    def _synthesize_chunk_summaries(self, chunk_summaries: List[str], tools_used: Optional[List[str]]) -> SummaryResult:
        """Combine chunk summaries into final synopsis + display_title + complexity."""
        if not self._synthesis_system_prompt or not self._synthesis_user_template:
            raise RuntimeError("Synthesis prompts not found - cannot merge chunk summaries")

        combined_text = "\n\n---\n\n".join(
            f"**Part {i+1}:**\n{summary}"
            for i, summary in enumerate(chunk_summaries)
        )
        tools_text = "None" if not tools_used else ", ".join(tools_used)

        user_prompt = self._synthesis_user_template.format(
            chunk_summaries=combined_text,
            tools_used=tools_text
        )

        response = self.llm_provider.generate_response(
            messages=[
                {"role": "system", "content": self._synthesis_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            internal_llm='summary',
            allow_negative=True  # System task — segment already paid for
        )

        raw_output = self.llm_provider.extract_text_content(response)
        return self._extract_summary_components(raw_output)

    def _generate_chunked_summary(
        self,
        messages: List[Message],
        tools_used: Optional[List[str]],
        previous_summaries: Optional[List[Message]] = None
    ) -> SummaryResult:
        """
        Hierarchical summarization for oversized segments.

        Previous summaries are only passed to the first chunk for narrative
        continuity - subsequent chunks don't need that context.
        """
        chunks = self._chunk_messages(messages)
        logger.info(f"Split into {len(chunks)} chunks for hierarchical summarization")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Summarizing chunk {i+1}/{len(chunks)}")
            # Only first chunk gets previous summaries for continuity
            chunk_context = previous_summaries if i == 0 else None
            chunk_result = self._summarize_chunk(chunk, tools_used, chunk_context)
            # Use just the synopsis text for synthesis (ignore per-chunk title/complexity)
            chunk_summaries.append(chunk_result.synopsis)

        return self._synthesize_chunk_summaries(chunk_summaries, tools_used)
