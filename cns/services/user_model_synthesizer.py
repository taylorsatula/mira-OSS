"""
User model synthesizer for the user model pipeline.

Synthesizes assessment signals into a descriptive user model: observations about the
user anchored to system prompt sections. Includes a critic validation loop that
catches observation laundering, personality labels, and internal contradictions.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Literal, Optional

from cns.infrastructure.feedback_repository import FeedbackRepository, FeedbackSignalRow
from cns.infrastructure.feedback_tracker import FeedbackTracker
from cns.services.system_prompt_parser import format_section_list, get_assessable_sections
from clients.llm_provider import LLMProvider
from config import config

logger = logging.getLogger(__name__)

CRITIC_MAX_ATTEMPTS = 3


@dataclass
class UserObservation:
    """A parsed observation from the user model."""
    section_id: str
    observation: str
    confidence: Literal['high', 'moderate', 'low']
    changelog: str


@dataclass
class CheckinTopic:
    """A check-in topic for behavioral debrief."""
    section_id: str
    reason: str


@dataclass
class SynthesisResult:
    """Complete result from user model synthesis."""
    observations: List[UserObservation]
    checkin_topics: List[CheckinTopic]
    raw_xml: str


@dataclass
class CriticResult:
    """Result of critic validation on a candidate user model."""
    passed: bool
    feedback: str


class UserModelSynthesizer:
    """
    Synthesizes assessment signals into a user model with critic validation.

    Pipeline:
    1. Fetch unsynthesized signals
    2. Call synthesis LLM to produce candidate user model
    3. Run critic validation (Haiku) to check for quality issues
    4. If critic fails: rerun synthesis with feedback (up to 3 attempts)
    5. Return final user model
    """

    def __init__(
        self,
        feedback_repo: FeedbackRepository,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.feedback_repo = feedback_repo
        self.llm_provider = llm_provider or LLMProvider()
        self._load_prompts()

        # Pre-compute section list for critic context
        raw_prompt = config.system_prompt
        sections = get_assessable_sections(raw_prompt)
        self._section_list = format_section_list(sections)

        logger.info("UserModelSynthesizer initialized")

    def _load_prompts(self) -> None:
        """Load synthesis and critic prompts."""
        from config.prompts.loader import load_prompt
        self._synthesis_system_prompt = load_prompt("user_model_synthesis_system.txt")
        self._synthesis_user_template = load_prompt("user_model_synthesis_user.txt")
        self._critic_system_prompt = load_prompt("user_model_critic_system.txt")
        self._critic_user_template = load_prompt("user_model_critic_user.txt")

    def synthesize(self, user_id: str, current_model_xml: Optional[str] = None) -> SynthesisResult:
        """
        Synthesize a user model from accumulated assessment signals.

        Args:
            user_id: User whose signals to synthesize
            current_model_xml: Current user model XML (for evolutionary continuity)

        Returns:
            SynthesisResult with observations, checkin topics, and raw XML

        Raises:
            Exception: On synthesis failure (caller handles)
        """
        signals = self.feedback_repo.get_unsynthesized_signals(user_id)

        if not signals and not current_model_xml:
            logger.info("No signals or existing model for user %s", user_id)
            return SynthesisResult(observations=[], checkin_topics=[], raw_xml="")

        # Fetch and consume any pending check-in feedback (atomically cleared)
        tracker = FeedbackTracker()
        checkin_feedback = tracker.get_and_clear_checkin_response(user_id)

        # Format signals grouped by section
        signals_text = self._format_signals_by_section(signals) if signals else "No new signals."
        current_model_text = current_model_xml if current_model_xml else "No existing user model (first synthesis)."

        # Initial synthesis
        candidate_xml = self._run_synthesis(signals_text, current_model_text, checkin_feedback)

        # Critic validation loop
        for attempt in range(CRITIC_MAX_ATTEMPTS):
            critic = self._validate_with_critic(candidate_xml)

            if critic.passed:
                logger.info("User model passed critic validation (attempt %d)", attempt + 1)
                break

            logger.warning("Critic rejected user model (attempt %d): %s", attempt + 1, critic.feedback[:200])

            if attempt < CRITIC_MAX_ATTEMPTS - 1:
                candidate_xml = self._rerun_synthesis_with_feedback(
                    critic.feedback, signals_text, current_model_text, checkin_feedback
                )
        else:
            # Circuit breaker: fall back to previous model rather than injecting
            # a candidate that failed critic validation (known-suspect content).
            # The user model stays stale for one more cycle but remains correct.
            logger.warning(
                "Critic validation exhausted %d attempts, keeping previous model", CRITIC_MAX_ATTEMPTS
            )
            if current_model_xml:
                candidate_xml = current_model_xml
            # If no previous model exists, we have no choice but to use the candidate

        result = self._parse_user_model_xml(candidate_xml)
        logger.info(
            "Synthesized user model: %d observations, %d checkin topics",
            len(result.observations), len(result.checkin_topics)
        )
        return result

    def _run_synthesis(
        self,
        signals_text: str,
        current_model_text: str,
        checkin_feedback: str | None = None
    ) -> str:
        """Run the synthesis LLM call and return raw XML output."""
        user_prompt = self._synthesis_user_template.format(
            current_user_model=current_model_text,
            assessment_signals=signals_text
        )

        if checkin_feedback:
            user_prompt += f"\n\n## User Check-in Feedback\n{checkin_feedback}"

        llm_messages = [
            {"role": "system", "content": self._synthesis_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            internal_llm='synthesis',
            allow_negative=True  # System task — segment already paid for
        )

        return self.llm_provider.extract_text_content(response)

    def _validate_with_critic(self, candidate_xml: str) -> CriticResult:
        """
        Run critic validation on a candidate user model.

        Returns:
            CriticResult with passed=True and empty feedback on success,
            or passed=False with actionable revision instructions.
        """
        user_prompt = self._critic_user_template.format(
            section_id_list=self._section_list,
            candidate_user_model=candidate_xml
        )

        llm_messages = [
            {"role": "system", "content": self._critic_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            internal_llm='critic',
            allow_negative=True  # System task — segment already paid for
        )

        raw_output = self.llm_provider.extract_text_content(response)

        # Parse critic result
        status_match = re.search(r'<mira:critic_review\s+status="(\w+)"', raw_output)
        if not status_match:
            logger.warning("Could not parse critic output, treating as pass")
            return CriticResult(passed=True, feedback="")

        status = status_match.group(1)

        if status == "pass":
            return CriticResult(passed=True, feedback="")

        # Extract issue descriptions as feedback
        issues = []
        issue_pattern = r'<mira:issue\s+type="([^"]+)"\s+section="([^"]+)">(.*?)</mira:issue>'
        for match in re.finditer(issue_pattern, raw_output, re.DOTALL):
            issue_type = match.group(1)
            section = match.group(2)
            detail = match.group(3).strip()
            issues.append(f"[{issue_type} in {section}] {detail}")

        return CriticResult(passed=False, feedback="\n".join(issues))

    def _rerun_synthesis_with_feedback(
        self,
        feedback: str,
        signals_text: str,
        current_model_text: str,
        checkin_feedback: str | None = None
    ) -> str:
        """Rerun synthesis with critic feedback appended to the prompt."""
        user_prompt = self._synthesis_user_template.format(
            current_user_model=current_model_text,
            assessment_signals=signals_text
        )

        if checkin_feedback:
            user_prompt += f"\n\n## User Check-in Feedback\n{checkin_feedback}"

        user_prompt += f"\n\n## Quality Critic Feedback\nThe quality critic flagged these issues in the previous attempt. Revise the user model to address them:\n\n{feedback}"

        llm_messages = [
            {"role": "system", "content": self._synthesis_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm_provider.generate_response(
            messages=llm_messages,
            internal_llm='synthesis',
            allow_negative=True  # System task — segment already paid for
        )

        return self.llm_provider.extract_text_content(response)

    def _format_signals_by_section(self, signals: list[FeedbackSignalRow]) -> str:
        """Group and format signals by section_id for the synthesis prompt."""
        by_section: dict[str, list[FeedbackSignalRow]] = {}

        for signal in signals:
            section = signal['section_id']
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(signal)

        parts = []
        for section_id, section_signals in sorted(by_section.items()):
            parts.append(f"### {section_id}")
            for s in section_signals:
                signal_type = s['signal_type']
                strength = s['strength']
                evidence = s['evidence']
                parts.append(f"- **{signal_type}** ({strength}): {evidence}")
            parts.append("")

        return "\n".join(parts) if parts else "No new signals."

    def _parse_user_model_xml(self, xml_output: str) -> SynthesisResult:
        """
        Parse a user model XML document into structured data.

        Expected format:
        <mira:user_model>
            <mira:observation section="..." confidence="...">
                Observation text.
                <changelog>What changed.</changelog>
            </mira:observation>
            <mira:checkin>
                <mira:topic section="..." reason="..."/>
            </mira:checkin>
        </mira:user_model>
        """
        observations = []
        checkin_topics = []

        # Parse observations
        obs_pattern = r'<mira:observation\s+section="([^"]+)"\s+confidence="([^"]+)">(.*?)</mira:observation>'

        for match in re.finditer(obs_pattern, xml_output, re.DOTALL):
            section_id = match.group(1).strip()
            confidence = match.group(2).strip()
            body = match.group(3)

            # Extract changelog
            changelog_match = re.search(r'<changelog>(.*?)</changelog>', body, re.DOTALL)
            changelog = changelog_match.group(1).strip() if changelog_match else ""

            # Observation text is everything except the changelog
            obs_text = re.sub(r'<changelog>.*?</changelog>', '', body, flags=re.DOTALL).strip()

            if confidence not in ('high', 'moderate', 'low'):
                logger.warning("Unknown confidence level: %s", confidence)
                confidence = 'moderate'

            observations.append(UserObservation(
                section_id=section_id,
                observation=obs_text,
                confidence=confidence,
                changelog=changelog
            ))

        # Parse check-in topics
        topic_pattern = r'<mira:topic\s+section="([^"]+)"\s+reason="([^"]+)"\s*(?:/>|>\s*</mira:topic>)'

        for match in re.finditer(topic_pattern, xml_output, re.DOTALL):
            checkin_topics.append(CheckinTopic(
                section_id=match.group(1).strip(),
                reason=match.group(2).strip()
            ))

        return SynthesisResult(
            observations=observations,
            checkin_topics=checkin_topics,
            raw_xml=xml_output
        )
