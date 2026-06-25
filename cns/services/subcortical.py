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
import contextvars
import json
import logging
import re
import threading
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

    def __init__(
        self,
        analysis_enabled: bool,
        llm_provider: 'LLMProvider',
        prefill_warmup_enabled: bool = False,
    ):
        """
        Initialize subcortical layer.

        Args:
            analysis_enabled: Whether subcortical processing is enabled
            llm_provider: LLM provider for subcortical processing calls
            prefill_warmup_enabled: When True, warm_cache() fires a background
                max_tokens=1 request after each turn so vLLM's prefix cache is
                populated before the next user message arrives. Wastes billed
                tokens on cloud providers; only enable for local vLLM.

        Raises:
            FileNotFoundError: If prompt files not found
            ValueError: If API key not found in Vault
            RuntimeError: If subcortical processing is disabled
        """
        self.llm_provider = llm_provider
        self.prefill_warmup_enabled = prefill_warmup_enabled

        if not analysis_enabled:
            raise RuntimeError(
                "SubcorticalLayer requires analysis_enabled=True"
            )

        # Load prompt templates
        from config.prompts.loader import load_prompt
        self.system_prompt = load_prompt("subcortical_system.txt")
        self.user_prompt_template = load_prompt("subcortical_user.txt")

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
        user_message, conversation_turns = self._build_user_prompt(
            continuum, current_user_message, previous_memories
        )

        # Piggyback: extract memory IDs mentioned in conversation (regex on already-built string)
        conversation_pinned_ids = set(
            m.lower() for m in TagParser.MEMORY_ID_PATTERN.findall(conversation_turns)
        )

        logger.debug(f"Generating query expansion for: {current_user_message[:100]}...")
        if previous_memories:
            logger.debug(f"Evaluating retention for {len(previous_memories)} memories")

        response = self.llm_provider.generate_response(
            messages=[{"role": "user", "content": user_message}],
            internal_llm='analysis',
            system_prompt=self.system_prompt,
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

        # Persist full prompt/response for fine-tuning curation
        self._save_output(
            prompt=user_message,
            raw_response=response_text,
        )

        return result

    def warm_cache(
        self,
        continuum: Continuum,
        previous_memories: list[SurfacedMemory] | None = None,
    ) -> None:
        """
        Speculatively warm the subcortical KV cache for the next turn.

        Fires a fire-and-forget max_tokens=1 request with the prefix that's
        already known at end-of-turn (system prompt + conversation_turns +
        previous_memories) so vLLM's prefix cache populates those blocks while
        the user is reading and typing. The next real subcortical call only
        prefills the new user-message tokens, removing prefill from the
        user-perceived critical path.

        No-op when prefill_warmup_enabled is False. Failures are logged at
        warning and swallowed — warmup is an optimization, not required
        infrastructure, so it must not break the response path.

        Only worthwhile against a vLLM (or other prefix-cache-aware) backend.
        On cloud providers without prefix caching, the call burns a billed
        token per turn for no benefit.
        """
        if not self.prefill_warmup_enabled:
            return
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(self._do_warmup, continuum, previous_memories),
            daemon=True,
        ).start()

    def _do_warmup(
        self,
        continuum: Continuum,
        previous_memories: list[SurfacedMemory] | None,
    ) -> None:
        try:
            user_message, _ = self._build_user_prompt(
                continuum, "", previous_memories
            )
            self.llm_provider.generate_response(
                messages=[{"role": "user", "content": user_message}],
                internal_llm='analysis',
                system_prompt=self.system_prompt,
                max_tokens=1,
            )
            logger.debug("Subcortical KV cache warmed")
        except Exception as e:
            logger.warning("Subcortical prefill warmup failed: %s", e)

    def _build_user_prompt(
        self,
        continuum: Continuum,
        current_user_message: str,
        previous_memories: list[SurfacedMemory] | None,
    ) -> tuple[str, str]:
        """
        Build the subcortical user prompt; returns (prompt, conversation_turns).

        Single source of truth shared by generate() and warm_cache(). Producing
        byte-exact-same tokens here for warmup and real calls is what makes the
        vLLM prefix cache hit — any divergence breaks the optimization silently.
        """
        # Under memory pressure, narrow the conversation window so the model
        # scopes retention decisions to the most recent exchange — makes it
        # easier to identify and drop memories irrelevant to the immediate context
        from lt_memory.proactive import MAX_PINNED_MEMORIES
        under_pressure = (
            previous_memories is not None
            and len(previous_memories) >= MAX_PINNED_MEMORIES
        )
        pairs = CONTEXT_PAIRS - 1 if under_pressure else CONTEXT_PAIRS

        conversation_turns = self._format_recent_turns(
            continuum,
            current_user_message,
            max_pairs=pairs,
        )

        # Format previous memories for prompt (unfiltered - LLM needs full context)
        # When under pressure, _format_previous_memories injects a pruning alert
        memories_block = self._format_previous_memories(previous_memories)

        # Collapse newlines — template is single-line by design, so injected
        # content must not reintroduce them.
        def _collapse(text: str) -> str:
            return " ".join(text.split())

        user_prompt = self.user_prompt_template.replace(
            "{conversation_turns}",
            _collapse(conversation_turns)
        ).replace(
            "{user_message}",
            _collapse(current_user_message)
        ).replace(
            "{previous_memories}",
            _collapse(memories_block)
        )
        return user_prompt, conversation_turns

    def _save_output(
        self,
        prompt: str,
        raw_response: str,
    ) -> None:
        """
        Persist raw prompt/response pairs for fine-tuning curation.

        Saves to data/users/{user_id}/subcortical_debuglogs/{YYYY-MM-DD}.jsonl
        with '------------------' delimiters between invocations.
        """
        try:
            from utils.user_context import get_current_user_id
            user_id = get_current_user_id()
            if not user_id:
                logger.debug("No user context, skipping subcortical output persistence")
                return

            output_dir = Path("data/users") / str(user_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            debug_dir = output_dir / "subcortical_debuglogs"
            debug_dir.mkdir(parents=True, exist_ok=True)
            date = utc_now().strftime("%Y-%m-%d")
            debug_file = debug_dir / f"{date}.jsonl"
            file_has_content = debug_file.exists() and debug_file.stat().st_size > 0

            debug_record = {
                "timestamp": utc_now().isoformat(),
                "prompt": prompt,
                "response": raw_response,
            }

            with open(debug_file, "a") as f:
                if file_has_content:
                    f.write("\n------------------\n")
                f.write(json.dumps(debug_record))

            logger.debug(f"Saved subcortical debug log to {debug_file}")

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
