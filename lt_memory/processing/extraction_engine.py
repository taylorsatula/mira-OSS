"""
Extraction engine - build extraction payloads and prompts.

Consolidates all prompt building, context loading, and message formatting:
- Prompt loading from config files
- UUID shortening and bidirectional mapping
- Memory context retrieval and caching
- Message formatting (conversation → XML-tagged turns)
- Extraction prompt building with context
- Anthropic message batch building

This module handles WHAT to extract and HOW to ask the LLM.
MemoryProcessor handles parsing the LLM's response.
"""
import logging
from typing import List, Dict, Any, Tuple, TypedDict
from uuid import UUID

from cns.core.message import preprocess_content_blocks
from lt_memory.models import ProcessingChunk, MemoryContext
from lt_memory.db_access import LTMemoryDB
from utils.tag_parser import format_memory_id

logger = logging.getLogger(__name__)


class ExtractionMessage(TypedDict):
    """Single message in Anthropic batch request format."""
    role: str
    content: str


class ExtractionPayload:
    """
    Complete extraction payload for batch or immediate execution.

    Contains everything needed to make an extraction request:
    - Prompts (system + user)
    - Messages (Anthropic format)
    - UUID mappings (for response parsing)
    - Memory context (for deduplication)
    """

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        messages: List[ExtractionMessage],
        short_to_uuid: Dict[str, str],
        memory_context: MemoryContext,
        chunk_index: int
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.messages = messages
        self.short_to_uuid = short_to_uuid
        self.memory_context = memory_context
        self.chunk_index = chunk_index


class ExtractionEngine:
    """
    Build extraction payloads for LLM extraction.

    Single Responsibility: Prepare all inputs needed for memory extraction

    Handles prompt management, context building, message formatting.
    Does NOT make LLM calls or process responses - that's for ExecutionStrategy and MemoryProcessor.
    """

    def __init__(self, db: LTMemoryDB):
        self.db = db
        self._load_prompts()

    def _load_prompts(self) -> None:
        """
        Load extraction prompts from configuration files.

        Raises:
            FileNotFoundError: If prompt files not found (fail-fast)
        """
        from config.prompts.loader import load_prompt
        self.extraction_system_prompt = load_prompt("memory_extraction_system.txt")
        self.extraction_user_template = load_prompt("memory_extraction_user.txt")
        logger.info("Loaded memory extraction prompts")

    def build_extraction_payload(
        self,
        chunk: ProcessingChunk,
        for_batch: bool = True
    ) -> ExtractionPayload:
        """
        Build complete extraction payload for chunk.

        Creates everything needed for LLM extraction:
        - Memory context (existing memories referenced in chunk)
        - UUID mappings (full ↔ short for compact prompts)
        - Formatted messages (conversation → XML-tagged turns with attribution)
        - Extraction prompt with context

        Args:
            chunk: ProcessingChunk containing continuum messages
            for_batch: Kept for caller compatibility (both paths produce same prompt)

        Returns:
            ExtractionPayload with all components
        """
        if not chunk.messages:
            raise ValueError(
                f"Cannot build extraction payload for empty chunk {chunk.chunk_index}. "
                f"Chunks should be filtered before payload building."
            )

        # Get memory context for this chunk
        memory_context = self._get_memory_context_for_chunk(chunk)
        short_to_uuid = memory_context.get("short_to_uuid", {})

        # Format messages with memory attribution injected into assistant turns
        formatted_messages = self._format_chunk_for_extraction(
            chunk,
            memory_texts=memory_context.get("memory_texts", {})
        )
        extraction_prompt = self._build_extraction_prompt(formatted_messages)

        if for_batch:
            return ExtractionPayload(
                system_prompt=self.extraction_system_prompt,
                user_prompt="",
                messages=[{"role": "user", "content": extraction_prompt}],
                short_to_uuid=short_to_uuid,
                memory_context=memory_context,
                chunk_index=chunk.chunk_index
            )
        else:
            return ExtractionPayload(
                system_prompt=self.extraction_system_prompt,
                user_prompt=extraction_prompt,
                messages=[],
                short_to_uuid=short_to_uuid,
                memory_context=memory_context,
                chunk_index=chunk.chunk_index
            )

    def _build_identifier_maps(
        self,
        memory_ids: List[str]
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Build bidirectional UUID mapping for short identifiers.

        Creates maps between full UUIDs and 8-character shortened versions
        for efficient prompt representation.

        Args:
            memory_ids: List of full UUID strings

        Returns:
            Tuple of (full_to_short, short_to_full) dictionaries

        Raises:
            RuntimeError: If UUID collision detected (indicates broken UUID generator)
        """
        full_to_short: Dict[str, str] = {}
        short_to_full: Dict[str, str] = {}

        for full_id in memory_ids:
            short_id = format_memory_id(full_id)
            if not short_id:
                continue

            # Detect collisions - this should NEVER happen with proper UUID v4
            existing = short_to_full.get(short_id)
            if existing and existing != full_id:
                error_msg = (
                    f"CRITICAL: UUID collision detected for shortened ID '{short_id}': "
                    f"{existing} vs {full_id}. This indicates a broken UUID generator. "
                    f"Collision probability for UUID v4 is ~2^-122 - this should be impossible."
                )
                logger.critical(error_msg)
                raise RuntimeError(error_msg)

            short_to_full[short_id] = full_id
            full_to_short[full_id] = short_id

        return full_to_short, short_to_full

    def _get_memory_context_for_chunk(
        self,
        chunk: ProcessingChunk
    ) -> MemoryContext:
        """
        Retrieve memory context for extraction chunk.

        Memory texts are cached during chunk creation (before batch submission).
        This avoids redundant DB queries and ensures consistency - texts reflect
        memory state at chunk-creation time, not extraction time.

        Falls back to DB if cache missing (old chunks, tests, migrations).

        Args:
            chunk: Processing chunk with memory_context_snapshot

        Returns:
            Dictionary containing:
            - memory_ids: List of memory UUIDs
            - memory_texts: Dict {uuid: text} (always dict format)
            - uuid_to_short: Full UUID to short ID mapping
            - short_to_uuid: Short ID to full UUID mapping
        """
        # Validate chunk has memory context (should always be present for proper chunks)
        if chunk.memory_context_snapshot is None:
            logger.warning(
                f"Chunk {chunk.chunk_index} missing memory_context_snapshot. "
                "Expected for test chunks, unexpected for production chunks created by orchestrator."
            )
            memory_context = {}
        else:
            memory_context = chunk.memory_context_snapshot

        # Extract memory IDs from snapshot
        # Only include explicitly referenced memories as context
        # Surfaced memories that weren't explicitly mentioned are not relevant
        memory_ids = memory_context.get("memory_ids")
        if memory_ids is None:
            # Use only referenced memories, not surfaced
            memory_ids = memory_context.get("referenced_memory_ids", [])

        if memory_ids is None:
            # Neither field present - context is malformed or empty
            memory_ids = []

        # Retrieve memory texts (always dict format)
        memory_texts: Dict[str, str] = {}
        if memory_ids:
            # Try cached texts first
            cached_texts = memory_context.get("memory_texts")

            if isinstance(cached_texts, dict):
                # Use cached dict directly
                memory_texts = cached_texts
            else:
                # Fall back to database if no cached texts
                # Cache miss should only happen for old chunks or migrations
                logger.warning(
                    f"Chunk {chunk.chunk_index} missing memory_texts cache, falling back to DB. "
                    "Expected for old chunks/tests, unexpected for newly created chunks."
                )

                memory_uuids = [UUID(mid) for mid in memory_ids]
                memories = self.db.get_memories_by_ids(memory_uuids)

                # Verify all requested memories were found
                if len(memories) != len(memory_uuids):
                    found_ids = {m.id for m in memories}
                    missing_ids = set(memory_uuids) - found_ids
                    raise ValueError(
                        f"Failed to load {len(missing_ids)} of {len(memory_uuids)} memories from DB "
                        f"for chunk {chunk.chunk_index}. Missing IDs: {missing_ids}. "
                        "Indicates deleted/corrupted memory references in chunk context."
                    )

                # Build dict: {uuid: text}
                memory_texts = {str(m.id): m.text for m in memories}

                logger.debug(
                    f"Retrieved {len(memory_texts)} memory texts for "
                    f"chunk {chunk.chunk_index} from database"
                )

        # Build bidirectional identifier maps
        uuid_to_short, short_to_uuid = self._build_identifier_maps(memory_ids)

        return {
            "memory_ids": memory_ids,
            "memory_texts": memory_texts,
            "snapshot_timestamp": memory_context.get("timestamp", ""),
            "uuid_to_short": uuid_to_short,
            "short_to_uuid": short_to_uuid
        }

    def _format_chunk_for_extraction(
        self,
        chunk: ProcessingChunk,
        memory_texts: dict[str, str] | None = None
    ) -> str:
        """
        Format chunk messages for LLM extraction.

        Converts structured message format to XML-tagged turns for consistency
        with the XML-structured extraction system prompt. Handles text blocks,
        skips images, includes tool results. Filters system notifications and
        tool-role messages. Injects <meta> attribution on assistant messages
        that were informed by existing memories.

        Args:
            chunk: ProcessingChunk containing message list
            memory_texts: Mapping of memory UUID -> text for attribution lookup

        Returns:
            Formatted continuum text with XML turn tags
        """
        if memory_texts is None:
            memory_texts = {}
        formatted_lines = []

        for msg in chunk.messages:
            role = getattr(msg, "role", "unknown")
            metadata = getattr(msg, "metadata", {}) or {}

            # Filter system notifications and tool messages
            if metadata.get("system_notification"):
                continue
            if role == "tool":
                continue

            content = getattr(msg, "content", "")

            # Preprocess structured content (strips media, truncates tool results)
            preprocessed = preprocess_content_blocks(content)
            content = " ".join(preprocessed.text_parts)
            if preprocessed.image_count > 0:
                content = f"[{preprocessed.image_count} image(s) shared] {content}".strip()

            # Format with XML tags matching system prompt style
            if role == "user":
                formatted_lines.append(f"<user>{content}</user>")
            elif role == "assistant":
                # Build <meta> attribution block if this message used existing memories
                meta_block = self._build_attribution_meta(metadata, memory_texts)
                if meta_block:
                    formatted_lines.append(f"<assistant>\n{meta_block}\n{content}</assistant>")
                else:
                    formatted_lines.append(f"<assistant>{content}</assistant>")
            else:
                # Discard unsupported roles (developer, system, etc.)
                logger.debug(f"Skipping message with unsupported role: {role}")
                continue

        return "\n".join(formatted_lines)

    def _build_attribution_meta(
        self,
        metadata: Dict[str, Any],
        memory_texts: Dict[str, str]
    ) -> str:
        """
        Build <meta> attribution block for an assistant message.

        If the message's metadata contains referenced_memories UUIDs,
        looks up each in memory_texts and builds a <meta> block so the
        extraction LLM knows which statements were memory-informed.

        Args:
            metadata: Message metadata dict
            memory_texts: Mapping of memory UUID -> text

        Returns:
            Meta block string, or empty string if no attributions
        """
        if not memory_texts:
            return ""

        referenced = metadata.get("referenced_memories")
        if not isinstance(referenced, list) or not referenced:
            return ""

        lines = ["<meta>"]
        for ref_id in referenced:
            if not isinstance(ref_id, str):
                continue
            text = memory_texts.get(ref_id)
            if text:
                short_id = format_memory_id(ref_id)
                lines.append(f'<references_memory id="{short_id}">{text}</references_memory>')

        if len(lines) == 1:
            # No valid attributions found
            return ""

        lines.append("</meta>")
        return "\n".join(lines)

    def _build_extraction_prompt(self, formatted_messages: str) -> str:
        """Build extraction user prompt from formatted conversation text."""
        return self.extraction_user_template.format(
            formatted_messages=formatted_messages,
        )

