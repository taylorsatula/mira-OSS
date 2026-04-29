"""
Subcortical layer service for retrieval-optimized query expansion.

Transforms fragmentary user queries into detailed, specific queries
optimized for finding relevant memories via embedding similarity.

Key principle: The query expansion REPLACES the original query for retrieval,
rather than augmenting it. Research showed this approach outperforms
query augmentation for personal memory search.

Also handles memory retention decisions - evaluating which previously
surfaced memories should remain in context based on conversation trajectory.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Set, Optional, TypedDict, TYPE_CHECKING

import json_repair

from cns.core.continuum import Continuum

if TYPE_CHECKING:
    from clients.llm_provider import LLMProvider
    from cns.core.message import Message
from utils.tag_parser import format_memory_id, TagParser
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

# Matches all <mira:*> internal tags — both paired (<mira:my_emotion>…</mira:my_emotion>)
# and self-closing (<mira:memory_ref="…" />). Stripped from conversation turns before
# subcortical processing to avoid rare-token attention sinks.
_MIRA_TAG_PATTERN = re.compile(r'<mira:[^>]*>.*?</mira:[^>]*>|<mira:[^/]*/\s*>', re.DOTALL)


class SurfacedMemory(TypedDict):
    """Memory previously surfaced in conversation, evaluated for retention."""
    id: str  # Full UUID string
    text: str
    importance_score: float  # 0.0-1.0


@dataclass
class SubcorticalResult:
    """Result of subcortical layer processing and memory retention analysis."""
    query_expansion: str
    pinned_memory_ids: Set[str] = field(default_factory=set)  # 8-char hex IDs
    entities: List[str] = field(default_factory=list)
    complexity: Literal["straightforward", "complex"] = "complex"  # Default to complex for safety

    def get_effort_level(self) -> str:
        """
        Return the named effort level based on complexity assessment.

        Maps subcortical complexity vocabulary to Anthropic effort levels:
        - "straightforward" → "medium" (simple questions, status updates, casual chat)
        - "complex" → "high" (multi-step reasoning, debugging, decision-making)
        """
        if self.complexity == "straightforward":
            return "medium"
        return "high"

# Number of user/assistant pairs to include as context (3 pairs = 6 messages).
# Reduced from 6 based on MI analysis: conv_turns token mass was 9.5x current_message,
# driving 2.34x context bleed ratio. At 3 pairs the ratio drops to ~1.1x.
CONTEXT_PAIRS = 3


class SubcorticalLayer:
    """
    Subcortical processing layer for retrieval-optimized query expansion.

    Uses a fast model (Groq) to expand fragmentary queries into detailed
    specifics that match stored memory vocabulary for better embedding similarity.
    """

    def __init__(self, analysis_enabled: bool, llm_provider: 'LLMProvider'):
        """
        Initialize subcortical layer.

        Args:
            analysis_enabled: Whether subcortical processing is enabled
            llm_provider: LLM provider for subcortical processing calls

        Raises:
            FileNotFoundError: If prompt files not found
            ValueError: If API key not found in Vault
            RuntimeError: If subcortical processing is disabled
        """
        self.llm_provider = llm_provider

        if not analysis_enabled:
            raise RuntimeError(
                "SubcorticalLayer requires analysis_enabled=True"
            )

        # Load prompt templates
        system_prompt_path = Path("config/prompts/subcortical_system.txt")
        user_prompt_path = Path("config/prompts/subcortical_user.txt")

        if not system_prompt_path.exists():
            raise FileNotFoundError(
                f"Subcortical system prompt not found at {system_prompt_path}"
            )

        if not user_prompt_path.exists():
            raise FileNotFoundError(
                f"Subcortical user prompt not found at {user_prompt_path}"
            )

        with open(system_prompt_path, 'r') as f:
            self.system_prompt = f.read()

        with open(user_prompt_path, 'r') as f:
            self.user_prompt_template = f.read()

        logger.info("SubcorticalLayer initialized")

    def generate(
        self,
        continuum: Continuum,
        current_user_message: str,
        previous_memories: list[SurfacedMemory] | None = None
    ) -> SubcorticalResult:
        """
        Generate retrieval-optimized query expansion, evaluate retention, and extract entities.

        Expands fragmentary queries into detailed specifics:
        - Resolves "that", "it", "the one" to concrete references
        - Expands implicit context to explicit names, places, dates
        - Outputs vocabulary that matches stored memories

        Also evaluates which previously surfaced memories should remain in context,
        and extracts named entities for hub-based memory discovery.

        Args:
            continuum: Current continuum with message history
            current_user_message: User message to expand
            previous_memories: Memories from previous turn to evaluate for retention

        Returns:
            SubcorticalResult with query_expansion, pinned_memory_ids, and entities.

        Raises:
            RuntimeError: On empty response or parse failure.
        """
        # Under memory pressure, narrow the conversation window so the model
        # scopes retention decisions to the most recent exchange — makes it
        # easier to identify and drop memories irrelevant to the immediate context
        from lt_memory.proactive import MAX_PINNED_MEMORIES
        max_pinned = MAX_PINNED_MEMORIES
        under_pressure = (
            previous_memories is not None
            and len(previous_memories) >= max_pinned
        )
        pairs = CONTEXT_PAIRS - 1 if under_pressure else CONTEXT_PAIRS

        conversation_turns = self._format_recent_turns(
            continuum,
            current_user_message,
            max_pairs=pairs
        )

        # Piggyback: extract memory IDs mentioned in conversation (regex on already-built string)
        conversation_pinned_ids = set(
            m.lower() for m in TagParser.MEMORY_ID_PATTERN.findall(conversation_turns)
        )

        # Format previous memories for prompt (unfiltered - LLM needs full context)
        # When under pressure, _format_previous_memories injects a pruning alert
        memories_block = self._format_previous_memories(previous_memories)

        # Collapse newlines — template is single-line by design, so injected
        # content must not reintroduce them.
        def _collapse(text: str) -> str:
            return " ".join(text.split())

        user_message = self.user_prompt_template.replace(
            "{conversation_turns}",
            _collapse(conversation_turns)
        ).replace(
            "{user_message}",
            _collapse(current_user_message)
        ).replace(
            "{previous_memories}",
            _collapse(memories_block)
        )

        logger.debug(f"Generating query expansion for: {current_user_message[:100]}...")
        if previous_memories:
            logger.debug(f"Evaluating retention for {len(previous_memories)} memories")

        response = self.llm_provider.generate_response(
            messages=[{"role": "user", "content": user_message}],
            stream=False,
            internal_llm='analysis',
            system_override=self.system_prompt,
        )

        response_text = self.llm_provider.extract_text_content(response).strip()

        if not response_text:
            raise RuntimeError("Subcortical returned empty response")

        # Attempt to repair malformed response structure before parsing
        try:
            # json_repair can fix common structural issues even in XML-like responses
            repaired_text = json_repair.repair_json(response_text, return_objects=False, skip_json_loads=True)
            if isinstance(repaired_text, str) and repaired_text.strip():
                response_text = repaired_text
                logger.debug("Applied structural repair to response")
        except Exception as repair_error:
            logger.debug(f"Repair attempt skipped: {repair_error}, using original response")

        # Parse query expansion, pinned IDs, and entities from response
        result = self._parse_response(response_text, previous_memories)

        # Guarantee retention for conversation-pinned memories (union with LLM's picks)
        result.pinned_memory_ids.update(conversation_pinned_ids)

        if conversation_pinned_ids:
            logger.info(f"Conversation pinning: {len(conversation_pinned_ids)} memories auto-retained")

        logger.info(f"Generated query expansion: {result.query_expansion[:150]}...")
        if previous_memories:
            logger.info(
                f"Retention: {len(result.pinned_memory_ids)}/{len(previous_memories)} memories pinned"
            )
        if result.entities:
            logger.info(f"Extracted {len(result.entities)} entities for hub discovery")

        # Persist subcortical output for prompt improvement
        self._save_output(
            user_message=current_user_message,
            query_expansion=result.query_expansion,
            pinned_ids=result.pinned_memory_ids,
            entities=result.entities,
            complexity=result.complexity,
            previous_memory_count=len(previous_memories) if previous_memories else 0
        )

        return result

    def _save_output(
        self,
        user_message: str,
        query_expansion: str,
        pinned_ids: Set[str],
        entities: List[str],
        complexity: str,
        previous_memory_count: int
    ) -> None:
        """
        Persist subcortical output to user's data directory for prompt improvement.

        Saves to data/users/{user_id}/subcortical_outputs.jsonl with full context
        needed to evaluate and improve subcortical expansion prompts.
        """
        try:
            from utils.user_context import get_current_user_id
            user_id = get_current_user_id()
            if not user_id:
                logger.debug("No user context, skipping subcortical output persistence")
                return

            from utils.instance import user_data_base
            output_dir = user_data_base() / str(user_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "subcortical_outputs.jsonl"

            record = {
                "timestamp": utc_now().isoformat(),
                "user_message": user_message,
                "query_expansion": query_expansion,
                "pinned_ids": list(pinned_ids),
                "entities": entities,
                "complexity": complexity,
                "previous_memory_count": previous_memory_count
            }

            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            logger.debug(f"Saved subcortical output to {output_file}")

        except Exception as e:
            # Don't let persistence failures break the subcortical pipeline
            logger.warning(f"Failed to save subcortical output: {e}")

    @staticmethod
    def _importance_to_dots(importance_score: float) -> str:
        """
        Convert importance score (0.0-1.0) to 5-dot visual indicator.

        Scale:
            ●●●●● = 0.8-1.0 (high importance)
            ●●●●○ = 0.6-0.8
            ●●●○○ = 0.4-0.6
            ●●○○○ = 0.2-0.4
            ●○○○○ = 0.0-0.2 (low importance)
        """
        score = max(0.0, min(1.0, importance_score))
        filled = int(score * 5) + (1 if score > 0 else 0)  # At least 1 dot if score > 0
        filled = min(5, max(1, filled)) if score > 0 else 1
        return "●" * filled + "○" * (5 - filled)

    def _format_previous_memories(
        self,
        memories: list[SurfacedMemory] | None
    ) -> str:
        """
        Format previous memories for the prompt with 8-char IDs and importance indicators.

        Matches system prompt input format: mem_a1B2c3D4 [●●●○○] - Passage text

        Uses a two-tier graduated pressure system derived from max_pinned_memories:
        - Warning (at max_pinned - 4): early signal with runway to self-regulate
        - Critical (at max_pinned): aggressive pruning instruction, force-truncation imminent

        Args:
            memories: Surfaced memories to format for retention evaluation

        Returns:
            Formatted passage lines or empty string if no memories
        """
        if not memories:
            return ""

        lines = []

        # Graduated pressure alerts derived from max_pinned_memories
        from lt_memory.proactive import MAX_PINNED_MEMORIES
        import math
        max_pinned = MAX_PINNED_MEMORIES
        warning_threshold = max_pinned - 4
        count = len(memories)

        if count >= max_pinned:
            # Critical: force-truncation imminent
            prune_count = math.ceil(count / 2)
            lines.append(
                f'<mira:system_alert>\U0001f6a8 TOO MANY PINNED MEMORIES ({count}/{max_pinned}). '
                f'The system WILL force-drop lowest-importance memories after this evaluation. '
                f'Aggressively prune \u2014 keep only memories that provide necessary context for '
                f'the active conversation topic. Drop tangential, redundant, or background '
                f'memories that aren\u2019t actively informing the discussion. You MUST remove at '
                f'least {prune_count} to avoid forced truncation. \U0001f6a8</mira:system_alert>'
            )
        elif count >= warning_threshold:
            # Warning: budget tightening
            lines.append(
                f'<mira:system_alert>\u26a0\ufe0f {count} memories are currently pinned. The system will '
                f'force-drop lowest-importance memories if pinned count exceeds {max_pinned}. '
                f'Be selective \u2014 retain memories that support the active topic, but let go of '
                f'memories from topics the conversation has moved past. Pinning more memories '
                f'reduces the budget for discovering new relevant ones.</mira:system_alert>'
            )

        # Truncate passage text to reduce token mass competing with current_message.
        # MI analysis (round 2) showed 15 words preserves enough context for relevance
        # judgment while freeing ~60% of passage token budget. Per-token attention for
        # current_message gained +18.1% with this truncation.
        max_passage_words = 15

        for memory in memories:
            text = memory.get('text', '')
            memory_id = memory.get('id', '')
            importance = memory.get('importance_score', 0.5)
            formatted_id = format_memory_id(memory_id)
            dots = SubcorticalLayer._importance_to_dots(importance)

            # Truncate to first N words — ID + topic is sufficient for relevance filtering
            words = text.split()
            if len(words) > max_passage_words:
                text = " ".join(words[:max_passage_words]) + "..."

            if text and formatted_id:
                lines.append(f"{formatted_id} [{dots}] - {text}")
            elif text:
                lines.append(f"[{dots}] - {text}")

        return "\n".join(lines)

    def _parse_response(
        self,
        response_text: str,
        previous_memories: list[SurfacedMemory] | None
    ) -> SubcorticalResult:
        """
        Parse query expansion, pinned memory IDs, entities, and complexity from LLM response.

        Args:
            response_text: Raw LLM response
            previous_memories: Surfaced memories (used to determine if passage parsing needed)

        Returns:
            SubcorticalResult with parsed query_expansion, pinned_memory_ids, entities, complexity

        Raises:
            RuntimeError: If <query_expansion> tag missing or empty
        """
        # Extract query expansion from <query_expansion> tags
        expansion_match = re.search(
            r'<query_expansion>(.*?)</query_expansion>',
            response_text,
            re.DOTALL
        )
        if expansion_match:
            query_expansion = expansion_match.group(1).strip()
        else:
            raise RuntimeError("Failed to extract query expansion from response - no <query_expansion> tag found")

        if not query_expansion:
            raise RuntimeError("Failed to extract query expansion from response - empty <query_expansion>")

        # Extract entities from <entities> block
        entities = SubcorticalLayer._parse_entities(response_text)

        # Extract complexity from <complexity> tag (default to "complex" if missing/invalid)
        complexity = SubcorticalLayer._parse_complexity(response_text)

        # Extract pinned memory IDs from <relevant_passages> tags
        pinned_ids: Set[str] = set()

        if previous_memories:
            # Extract 8-char IDs from <passage id="mem_xxx"> elements
            # Format: <passage id="mem_a1B2c3D4">passage text</passage>
            # UUIDs only contain hex chars (0-9, a-f)
            id_matches = re.findall(
                r'<passage\s+id="mem_([a-fA-F0-9]{8})"',
                response_text,
                re.IGNORECASE
            )
            pinned_ids = {match.lower() for match in id_matches}
            logger.debug(f"Parsed {len(pinned_ids)} pinned IDs from response")

        return SubcorticalResult(
            query_expansion=query_expansion,
            pinned_memory_ids=pinned_ids,
            entities=entities,
            complexity=complexity
        )

    @staticmethod
    def _parse_entities(response_text: str) -> List[str]:
        """
        Parse extracted entities from <entities> block.

        Expected format:
        <entities>
        <ne>Annika</ne>
        <ne>Mom</ne>
        </entities>

        Or for no entities:
        <entities>None</entities>

        Args:
            response_text: Raw LLM response

        Returns:
            List of entity names
        """
        entities: List[str] = []

        entities_match = re.search(
            r'<entities>(.*?)</entities>',
            response_text,
            re.DOTALL
        )

        if not entities_match:
            logger.debug("No <entities> block found in response")
            return entities

        entities_block = entities_match.group(1).strip()

        # Handle "None" case (no entities)
        if entities_block.lower() == "none":
            return entities

        # Parse <ne> tags
        entity_matches = re.findall(
            r'<ne[^>]*>(.*?)</ne>',
            entities_block,
            re.DOTALL
        )

        for name in entity_matches:
            name = name.strip()
            if name and name.lower() != "none":
                entities.append(name)

        logger.debug(f"Parsed {len(entities)} entities from response")
        return entities

    @staticmethod
    def _parse_complexity(response_text: str) -> Literal["straightforward", "complex"]:
        """
        Parse complexity assessment from <complexity> tag.

        Args:
            response_text: Raw LLM response

        Returns:
            "straightforward" or "complex" (defaults to "complex" if missing/invalid)
        """
        complexity_match = re.search(
            r'<complexity>(.*?)</complexity>',
            response_text,
            re.DOTALL | re.IGNORECASE
        )

        if not complexity_match:
            logger.debug("No <complexity> tag found, defaulting to complex")
            return "complex"

        value = complexity_match.group(1).strip().lower()

        if value == "straightforward":
            return "straightforward"
        elif value == "complex":
            return "complex"
        else:
            logger.warning(f"Invalid complexity value '{value}', defaulting to complex")
            return "complex"

    def _format_recent_turns(
        self,
        continuum: Continuum,
        current_user_message: str,
        max_pairs: int = CONTEXT_PAIRS
    ) -> str:
        """
        Format recent conversation turns for context.

        Skips collapsed segment summaries to only include actual conversation pairs.
        Uses XML <turn> elements with timestamps for structured representation.
        Does NOT include the current user message (template handles that separately).

        Args:
            continuum: Continuum with message cache
            current_user_message: Current user message (unused, kept for signature compat)
            max_pairs: Maximum conversation pairs to include (default: CONTEXT_PAIRS).
                       Reduced under memory pressure to narrow scope for retention decisions.

        Returns:
            Formatted string with last N pairs as timestamped <turn> elements
        """
        lines = []
        pairs_found = 0
        i = len(continuum.messages) - 1

        # Walk backwards to extract user/assistant pairs
        while i >= 0 and pairs_found < max_pairs:
            # Find assistant message (skip segment summaries)
            while i >= 0:
                msg = continuum.messages[i]
                if msg.role == "assistant" and not self._is_segment_summary(msg):
                    break
                i -= 1
            if i < 0:
                break
            assistant_msg = continuum.messages[i]
            i -= 1

            # Find preceding user message
            while i >= 0 and continuum.messages[i].role != "user":
                i -= 1
            if i < 0:
                break
            user_msg = continuum.messages[i]
            i -= 1

            # Format timestamps as HH:MM
            user_time = user_msg.created_at.strftime("%H:%M")
            assistant_time = assistant_msg.created_at.strftime("%H:%M")

            # Prepend pair (we're walking backwards), truncating long messages.
            # Strip all <mira:*> internal tags (emotion emojis, memory refs, etc.)
            # — rare-token attention sinks that waste budget without contributing
            # to entity/expansion/passage tasks.
            user_content = self._extract_text_content(user_msg.content)[:2000]
            assistant_content = _MIRA_TAG_PATTERN.sub('', str(assistant_msg.content))[:2000]
            lines.insert(0, f"<turn speaker=\"assistant\" time=\"{assistant_time}\">{assistant_content}</turn>")
            lines.insert(0, f"<turn speaker=\"user\" time=\"{user_time}\">{user_content}</turn>")
            pairs_found += 1

        return "\n".join(lines)

    def _is_segment_summary(self, message: 'Message') -> bool:
        """Check if message is a collapsed segment summary."""
        metadata = getattr(message, 'metadata', {}) or {}
        return (
            metadata.get('is_segment_boundary', False) and
            metadata.get('status') == 'collapsed'
        )

    def _extract_text_content(self, content: str | list[dict[str, object]]) -> str:
        """Extract text from potentially multimodal content."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = [
                item['text'] for item in content
                if isinstance(item, dict) and item.get('type') == 'text'
            ]
            return ' '.join(text_parts) if text_parts else '[non-text content]'

        return str(content)
