"""
Peanut Gallery metacognitive observer model.

Single-stage LLM pipeline for conversation observation.
The observer reviews recent conversation plus a compact execution trace and
emits metacognitive guidance when MIRA needs evidence-backed correction.

Actions: noop, concern, coaching
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from cns.core.message import Message, preprocess_content_blocks
from clients.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class PeanutGalleryResult:
    """Result from the Peanut Gallery observer evaluation."""
    action_type: Literal["noop", "concern", "coaching"]
    guidance: Optional[str] = None
    critical: bool = False


class PeanutGalleryModel:
    """
    Single-stage LLM model for metacognitive conversation observation.

    The observer reviews recent conversation and execution evidence, then
    emits either noop or a high-salience corrective directive.

    The observer can output:
    - noop: No action needed
    - concern: Direct corrective guidance for an active failure
    - coaching: Strong steering guidance before a bigger failure lands
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        analysis_interval_turns: int
    ):
        self.llm_provider = llm_provider
        self._observer_window_turns = analysis_interval_turns * 2

        prompts_dir = Path("config/prompts")
        self._system_prompt = self._load_prompt(prompts_dir / "peanutgallery_system.txt")
        self._user_template = self._load_prompt(prompts_dir / "peanutgallery_user.txt")

        logger.info(
            "PeanutGalleryModel initialized (observer_window_turns=%d)",
            self._observer_window_turns,
        )

    def _load_prompt(self, path: Path) -> str:
        """Load prompt template from file."""
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text()

    def evaluate(self, messages: List[Message]) -> PeanutGalleryResult:
        """
        Evaluate recent conversation for metacognitive intervention.

        Args:
            messages: Recent messages from conversation cache

        Returns:
            PeanutGalleryResult with action_type and guidance when warranted
        """
        observation_messages = self._get_observation_messages(messages)
        if not observation_messages:
            return PeanutGalleryResult(action_type="noop")

        execution_trace = self._build_execution_trace(observation_messages)
        return self._run_observer(observation_messages, execution_trace)

    def _get_observation_messages(self, messages: List[Message]) -> list[Message]:
        """Scope raw messages to the same recent turn window the observer reviews."""
        conversational_messages = self._extract_conversational_messages(messages)
        recent_conversational_messages = conversational_messages[-(self._observer_window_turns * 2):]

        if not recent_conversational_messages:
            return []

        first_recent_message_id = recent_conversational_messages[0].id
        for idx, message in enumerate(messages):
            if message.id == first_recent_message_id:
                return messages[idx:]

        logger.warning("Failed to align Peanut Gallery observation window; using filtered messages only")
        return recent_conversational_messages

    def _run_observer(
        self,
        messages: List[Message],
        execution_trace: str
    ) -> PeanutGalleryResult:
        """
        Run observer to evaluate conversation and decide action.

        Args:
            messages: Scoped recent conversation messages
            execution_trace: Compact action/evidence summary from recent turns

        Returns:
            PeanutGalleryResult with action and details
        """
        conversational_messages = self._extract_conversational_messages(messages)

        if len(conversational_messages) < 3:
            logger.debug("Not enough conversational messages for observer evaluation")
            return PeanutGalleryResult(action_type="noop")

        formatted_messages = self._format_with_ids(conversational_messages)

        user_prompt = self._user_template.replace(
            "{formatted_messages}", formatted_messages
        ).replace(
            "{execution_trace}", execution_trace
        )

        response = self.llm_provider.generate_response(
            messages=[{"role": "user", "content": user_prompt}],
            stream=False,
            internal_llm='tidyup',
            system_override=self._system_prompt
        )

        response_text = self.llm_provider.extract_text_content(response).strip()

        if not response_text:
            logger.debug("Observer returned empty response")
            return PeanutGalleryResult(action_type="noop")

        return self._parse_observer_response(response_text)

    def _parse_observer_response(self, response_text: str) -> PeanutGalleryResult:
        """
        Parse observer response into PeanutGalleryResult.

        Expected formats:
        - <mira:noop/>
        - <mira:peanutgallery type="concern">...</mira:peanutgallery>
        - <mira:peanutgallery type="coaching">...</mira:peanutgallery>
        """
        if re.search(r'<mira:noop\s*/>', response_text, re.IGNORECASE):
            logger.debug("Observer: noop")
            return PeanutGalleryResult(action_type="noop")

        pg_match = re.search(
            r'<mira:peanutgallery\b([^>]*)>(.*?)</mira:peanutgallery>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )

        if not pg_match:
            logger.debug("Malformed observer response, treating as noop")
            return PeanutGalleryResult(action_type="noop")

        attributes = pg_match.group(1)
        content = pg_match.group(2)

        action_type_match = re.search(r'\btype="(\w+)"', attributes, re.IGNORECASE)
        if not action_type_match:
            logger.debug("Observer response missing action type, treating as noop")
            return PeanutGalleryResult(action_type="noop")

        severity_match = re.search(r'\bseverity="(\w+)"', attributes, re.IGNORECASE)

        action_type = action_type_match.group(1).lower()
        severity = severity_match.group(1) if severity_match else None

        if action_type in ("concern", "coaching") and content is not None:
            return self._parse_guidance(action_type, content, severity)

        logger.warning("Unknown action type: %s", action_type)
        return PeanutGalleryResult(action_type="noop")

    def _parse_guidance(
        self,
        guidance_type: Literal["concern", "coaching"],
        content: str,
        severity: Optional[str] = None
    ) -> PeanutGalleryResult:
        """Parse concern or coaching guidance from content."""
        guidance_match = re.search(
            r'<guidance>(.*?)</guidance>',
            content,
            re.DOTALL
        )

        if not guidance_match:
            logger.debug("Malformed %s response", guidance_type)
            return PeanutGalleryResult(action_type="noop")

        return PeanutGalleryResult(
            action_type=guidance_type,
            guidance=guidance_match.group(1).strip(),
            critical=severity is not None and severity.lower() == "critical"
        )

    def _extract_conversational_messages(self, messages: List[Message]) -> list[Message]:
        """Extract human-visible user/assistant messages for observer review."""
        return [message for message in messages if self._is_conversational_message(message)]

    def _is_conversational_message(self, message: Message) -> bool:
        """Return True when a message contributes to the visible conversation flow."""
        metadata = message.metadata or {}
        if message.role not in ('user', 'assistant'):
            return False

        if metadata.get('is_segment_boundary') or metadata.get('system_notification'):
            return False

        if isinstance(message.content, str):
            return bool(message.content.strip())

        has_text_content = any(
            (isinstance(block, str) and block.strip()) or
            (
                isinstance(block, dict) and
                block.get("type") == "text" and
                bool(str(block.get("text", "")).strip())
            )
            for block in message.content
        )
        preprocessed = preprocess_content_blocks(message.content)
        has_media_content = preprocessed.image_count > 0
        return has_text_content or has_media_content

    def _build_execution_trace(self, messages: List[Message]) -> str:
        """Build a compact evidence ledger for dropped-task and hallucination checks."""
        conversational_messages = self._extract_conversational_messages(messages)
        assistant_commitments: list[str] = []
        tool_activity: list[str] = []
        latest_user_request = "none"
        latest_assistant_response = "none"

        for msg in reversed(conversational_messages):
            rendered = self._render_message_content(msg)
            if latest_user_request == "none" and msg.role == "user" and rendered:
                latest_user_request = rendered
            if latest_assistant_response == "none" and msg.role == "assistant" and rendered:
                latest_assistant_response = rendered
            if latest_user_request != "none" and latest_assistant_response != "none":
                break

        for msg in messages:
            metadata = msg.metadata or {}
            if metadata.get('is_segment_boundary') or metadata.get('system_notification'):
                continue

            if msg.role == "assistant":
                if self._is_conversational_message(msg):
                    assistant_commitments.extend(
                        self._extract_commitments(self._render_message_content(msg))
                    )
                tool_activity.extend(self._extract_tool_uses(msg))
            elif msg.role == "tool":
                tool_call_id = msg.metadata.get("tool_call_id", "unknown")
                result_text = self._render_message_content(msg)
                if len(result_text) > 180:
                    result_text = result_text[:180] + "..."
                if not result_text:
                    tool_activity.append(
                        f"tool result received for call {tool_call_id} (empty content)"
                    )
                else:
                    tool_activity.append(
                        f"tool result received for call {tool_call_id}: {result_text}"
                    )

        if not assistant_commitments:
            assistant_commitments.append("none")
        if not tool_activity:
            tool_activity.append("none observed")

        def _escape(value: str) -> str:
            return (
                value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        trace_parts = [
            "<execution_trace>",
            f"<latest_user_request>{_escape(latest_user_request)}</latest_user_request>",
            f"<latest_assistant_response>{_escape(latest_assistant_response)}</latest_assistant_response>",
            "<assistant_commitments>",
        ]
        trace_parts.extend(
            f"- {_escape(commitment)}" for commitment in assistant_commitments[:5]
        )
        trace_parts.append("</assistant_commitments>")
        trace_parts.append("<tool_activity>")
        trace_parts.extend(
            f"- {_escape(activity)}" for activity in tool_activity[:8]
        )
        trace_parts.append("</tool_activity>")
        trace_parts.append("</execution_trace>")
        return "\n".join(trace_parts)

    def _extract_commitments(self, rendered_message: str) -> list[str]:
        """Extract concrete assistant commitments from a rendered message."""
        if not rendered_message:
            return []

        commitment_patterns = [
            r"\bI'll\s+([^.!?]+)",
            r"\bI will\s+([^.!?]+)",
            r"\bLet me\s+([^.!?]+)",
            r"\bI(?:'m| am)\s+going to\s+([^.!?]+)",
        ]

        commitments: list[str] = []
        for pattern in commitment_patterns:
            for match in re.finditer(pattern, rendered_message, re.IGNORECASE):
                commitment = match.group(0).strip()
                if commitment not in commitments:
                    commitments.append(commitment)
        return commitments

    def _extract_tool_uses(self, message: Message) -> list[str]:
        """Extract tool-use summaries from assistant content blocks and metadata."""
        tool_activity: list[str] = []
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input")
                    if tool_input:
                        input_text = json.dumps(tool_input, sort_keys=True)
                        if len(input_text) > 180:
                            input_text = input_text[:180] + "..."
                        tool_activity.append(f"assistant invoked tool {tool_name} with input {input_text}")
                    else:
                        tool_activity.append(f"assistant invoked tool {tool_name}")

        tools_used = message.metadata.get("tools_used", [])
        for tool_name in tools_used:
            summary = f"assistant used tool {tool_name}"
            if summary not in tool_activity:
                tool_activity.append(summary)

        return tool_activity

    def _render_message_content(self, message: Message) -> str:
        """Render message content into compact plain text for observer context."""
        content = message.content
        if isinstance(content, list):
            preprocessed = preprocess_content_blocks(content)
            rendered = " ".join(preprocessed.text_parts)
            if preprocessed.image_count > 0:
                rendered = f"[{preprocessed.image_count} image(s) shared] {rendered}".strip()
        else:
            rendered = content

        if len(rendered) > 500:
            return rendered[:500] + "... [truncated]"
        return rendered

    def _format_with_ids(self, messages: List[Message]) -> str:
        """
        Format messages with 8-char IDs for observer evaluation.

        Format: [ID:xxxxxxxx] role: content
        """
        lines = []
        for msg in messages:
            msg_id = str(msg.id)[:8]
            content = self._render_message_content(msg)
            lines.append(f"[ID:{msg_id}] {msg.role}: {content}")

        return "\n".join(lines)
