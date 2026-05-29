"""
Assessment extraction service for the user model pipeline.

Evaluates conversation segments against Mira's behavioral contract (system prompt),
producing section-anchored signals (alignment, misalignment, contextual_pass) with
specific evidence. Replaces the old blind feedback extraction approach.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional
from uuid import UUID

from cns.core.message import Message, _MEDIA_BLOCK_TYPES
from cns.services.system_prompt_parser import (
    anonymize_prompt,
    format_section_list,
    get_assessable_sections,
)
from clients.llm_provider import LLMProvider
from config import config
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

# Thinking content threshold: include thinking traces when they exceed this
# proportion of total conversation content
THINKING_INCLUSION_THRESHOLD = 0.7


@dataclass
class AssessmentSignal:
    """A single assessment signal extracted from a conversation segment."""
    signal_type: Literal['alignment', 'misalignment', 'contextual_pass']
    section_id: str       # e.g., 'authenticity', 'collaboration'
    strength: Literal['strong', 'moderate', 'mild']
    evidence: str
    segment_id: UUID
    continuum_id: UUID
    extracted_at: datetime


class AssessmentExtractor:
    """
    Extracts assessment signals from conversation segments by evaluating
    behavior against the system prompt's behavioral contract.

    Each signal is anchored to a specific system prompt section and includes
    concrete evidence from the conversation.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider or LLMProvider()
        self._load_prompts()

        # Pre-compute system prompt sections and anonymized prompt
        raw_prompt = config.system_prompt
        self._sections = get_assessable_sections(raw_prompt)
        self._anonymized_prompt = anonymize_prompt(raw_prompt)
        self._section_list = format_section_list(self._sections)

        logger.info("AssessmentExtractor initialized with %d assessable sections", len(self._sections))

    def _load_prompts(self) -> None:
        """Load assessment extraction prompts from files."""
        prompts_dir = Path("config/prompts")

        system_path = prompts_dir / "assessment_extraction_system.txt"
        user_path = prompts_dir / "assessment_extraction_user.txt"
        thinking_path = prompts_dir / "thinking_block_instructions.txt"

        if not system_path.exists() or not user_path.exists():
            raise FileNotFoundError(f"Assessment extraction prompts not found in {prompts_dir}")

        self._system_template = system_path.read_text().strip()
        self._user_template = user_path.read_text().strip()
        self._thinking_instructions = thinking_path.read_text().strip() if thinking_path.exists() else ""

    def extract_signals(
        self,
        messages: List[Message],
        segment_id: UUID,
        continuum_id: UUID,
        user_model_xml: Optional[str] = None
    ) -> List[AssessmentSignal]:
        """
        Extract assessment signals from a conversation segment.

        Args:
            messages: Messages in the segment
            segment_id: Segment UUID
            continuum_id: Continuum UUID
            user_model_xml: Current user model XML for calibration context

        Returns:
            List of AssessmentSignal objects (may be empty)
        """
        if not messages:
            logger.debug("No messages to assess")
            return []

        # Format messages as XML
        conversation_text = self._format_messages(messages)

        # Compile system prompt with conditional thinking instructions
        system_prompt = self._compile_system_prompt(messages)

        # Format user model context
        user_model_text = self._format_user_model(user_model_xml)

        # Build user prompt
        user_prompt = self._user_template.format(
            anonymized_system_prompt=self._anonymized_prompt,
            user_model_observations=user_model_text,
            section_id_list=self._section_list,
            conversation_text=conversation_text
        )

        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            internal_llm='assessment',
            allow_negative=True  # System task — segment already paid for
        )

        raw_output = self.llm_provider.extract_text_content(response)
        signals = self._parse_assessment_xml(raw_output, segment_id, continuum_id)

        logger.info("Extracted %d assessment signals from segment %s", len(signals), segment_id)
        return signals

    def _compile_system_prompt(self, messages: List[Message]) -> str:
        """
        Build the system prompt, conditionally including thinking block instructions
        when thinking content exceeds the inclusion threshold.
        """
        thinking_pct = self._calculate_thinking_percentage(messages)

        if thinking_pct >= THINKING_INCLUSION_THRESHOLD and self._thinking_instructions:
            return self._system_template.replace(
                "{{THINKING_BLOCK_INSTRUCTIONS}}",
                self._thinking_instructions
            )

        return self._system_template.replace("{{THINKING_BLOCK_INSTRUCTIONS}}", "")

    def _calculate_thinking_percentage(self, messages: List[Message]) -> float:
        """Calculate what fraction of assistant content is thinking traces."""
        total_len = 0
        thinking_len = 0

        for msg in messages:
            if msg.role != "assistant":
                continue

            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'reasoning':
                            thinking_len += len(block.get('text', ''))
                            total_len += len(block.get('text', ''))
                        elif block.get('type') == 'text':
                            total_len += len(block.get('text', ''))
            elif isinstance(content, str):
                total_len += len(content)

        return thinking_len / total_len if total_len > 0 else 0.0

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages as XML tags for the assessor."""
        parts = []
        include_thinking = self._calculate_thinking_percentage(messages) >= THINKING_INCLUSION_THRESHOLD

        for msg in messages:
            if msg.metadata.get('system_notification', False):
                continue

            content = msg.content

            if isinstance(content, list):
                # Structured content (multimodal)
                text_parts = []
                thinking_parts = []
                media_count = 0
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type', '')
                        if block_type == 'text':
                            text_parts.append(block.get('text', ''))
                        elif block_type == 'reasoning' and include_thinking:
                            thinking_parts.append(block.get('text', ''))
                        elif block_type == 'tool_call':
                            text_parts.append(f"[Used tool: {block.get('name', 'unknown')}]")
                        elif block_type in _MEDIA_BLOCK_TYPES:
                            media_count += 1

                if media_count > 0:
                    text_parts.insert(0, f"[{media_count} image(s) shared]")

                if msg.role == "assistant":
                    if thinking_parts:
                        parts.append(f"<think>{''.join(thinking_parts)}</think>")
                    if text_parts:
                        parts.append(f"<assistant>{' '.join(text_parts)}</assistant>")
                else:
                    if text_parts:
                        parts.append(f"<user>{' '.join(text_parts)}</user>")

            elif isinstance(content, str):
                tag = "assistant" if msg.role == "assistant" else "user"
                parts.append(f"<{tag}>{content}</{tag}>")

        return "\n\n".join(parts)

    def _format_user_model(self, user_model_xml: Optional[str]) -> str:
        """
        Format user model observations as readable context for the assessor.

        Parses the XML and presents observations as natural text. If no model
        exists yet (early cycles), returns a placeholder.
        """
        if not user_model_xml:
            return "No user model observations yet."

        observations = []
        pattern = r'<mira:observation\s+section="([^"]+)"\s+confidence="([^"]+)">(.*?)</mira:observation>'

        for match in re.finditer(pattern, user_model_xml, re.DOTALL):
            section_id = match.group(1)
            confidence = match.group(2)
            # Strip the changelog from the observation text
            obs_text = re.sub(r'<changelog>.*?</changelog>', '', match.group(3), flags=re.DOTALL).strip()
            observations.append(f"[{section_id}, {confidence}] {obs_text}")

        if not observations:
            return "No user model observations yet."

        return "\n".join(observations)

    def _parse_assessment_xml(
        self,
        xml_output: str,
        segment_id: UUID,
        continuum_id: UUID
    ) -> List[AssessmentSignal]:
        """
        Parse assessment signals from XML output.

        Expected format:
        <mira:assessment>
            <mira:section id="...">
                <mira:signal type="..." strength="...">
                    <evidence>...</evidence>
                </mira:signal>
            </mira:section>
        </mira:assessment>
        """
        signals = []
        extracted_at = utc_now()

        valid_types = {'alignment', 'misalignment', 'contextual_pass'}
        valid_strengths = {'strong', 'moderate', 'mild'}

        # Match section blocks
        section_pattern = r'<mira:section\s+id="([^"]+)"\s*>(.*?)</mira:section>'

        for section_match in re.finditer(section_pattern, xml_output, re.DOTALL):
            section_id = section_match.group(1).strip()
            section_body = section_match.group(2)

            # Match signals within this section
            signal_pattern = r'<mira:signal\s+type="([^"]+)"\s+strength="([^"]+)"\s*>\s*<evidence>(.*?)</evidence>\s*</mira:signal>'

            for sig_match in re.finditer(signal_pattern, section_body, re.DOTALL):
                signal_type = sig_match.group(1).strip()
                strength = sig_match.group(2).strip()
                evidence = sig_match.group(3).strip()

                if signal_type not in valid_types:
                    logger.warning("Unknown signal type: %s", signal_type)
                    continue

                if strength not in valid_strengths:
                    logger.warning("Unknown strength: %s", strength)
                    continue

                if not evidence:
                    continue

                signals.append(AssessmentSignal(
                    signal_type=signal_type,
                    section_id=section_id,
                    strength=strength,
                    evidence=evidence,
                    segment_id=segment_id,
                    continuum_id=continuum_id,
                    extracted_at=extracted_at
                ))

        return signals
