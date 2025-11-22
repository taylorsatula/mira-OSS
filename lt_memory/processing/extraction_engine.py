"""
Extraction engine - build extraction payloads and prompts.

Consolidates all prompt building, context loading, and message formatting:
- Prompt loading from config files
- UUID shortening and bidirectional mapping
- Memory context retrieval and caching
- Message formatting (conversation → Human:/Assistant: format)
- Extraction prompt building with context
- Anthropic message batch building

This module handles WHAT to extract and HOW to ask the LLM.
MemoryProcessor handles parsing the LLM's response.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID

from cns.core.message import Message
from lt_memory.models import ProcessingChunk
from lt_memory.db_access import LTMemoryDB
from config.config import ExtractionConfig

logger = logging.getLogger(__name__)


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
        messages: List[Dict[str, Any]],
        short_to_uuid: Dict[str, str],
        memory_context: Dict[str, Any],
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

    def __init__(self, config: ExtractionConfig, db: LTMemoryDB):
        """
        Initialize extraction engine.

        Args:
            config: Extraction configuration parameters
            db: Database access for memory queries
        """
        self.config = config
        self.db = db
        self._load_prompts()

    def _load_prompts(self):
        """
        Load extraction prompts from configuration files.

        Raises:
            FileNotFoundError: If prompt files not found (fail-fast)
        """
        prompts_dir = Path("config/prompts")

        system_path = prompts_dir / "memory_extraction_system.txt"
        user_path = prompts_dir / "memory_extraction_user.txt"

        if not system_path.exists() or not user_path.exists():
            raise FileNotFoundError(
                f"Memory extraction prompts not found in {prompts_dir}"
            )

        with open(system_path, 'r', encoding='utf-8') as f:
            self.extraction_system_prompt = f.read().strip()

        with open(user_path, 'r', encoding='utf-8') as f:
            self.extraction_user_template = f.read().strip()

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
        - Formatted messages (conversation → Human:/Assistant: format)
        - Extraction prompt with context

        Args:
            chunk: ProcessingChunk containing continuum messages
            for_batch: If True, format for Anthropic Batch API; if False, for immediate execution

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

        # Build messages (different formats for batch vs immediate)
        if for_batch:
            messages = self._build_batch_messages(chunk, memory_context)
            # For batch API, return messages directly
            return ExtractionPayload(
                system_prompt=self.extraction_system_prompt,
                user_prompt="",  # Not used in batch format
                messages=messages,
                short_to_uuid=short_to_uuid,
                memory_context=memory_context,
                chunk_index=chunk.chunk_index
            )
        else:
            # For immediate execution, use old format (system + user prompt)
            formatted_messages = self._format_chunk_for_extraction(chunk)
            extraction_prompt = self._build_extraction_prompt(
                formatted_messages,
                memory_context
            )

            return ExtractionPayload(
                system_prompt=self.extraction_system_prompt,
                user_prompt=extraction_prompt,
                messages=[],  # Not used in immediate format
                short_to_uuid=short_to_uuid,
                memory_context=memory_context,
                chunk_index=chunk.chunk_index
            )

    @staticmethod
    def _shorten_memory_id(memory_id: str) -> str:
        """
        Shorten UUID to first 8 hex characters for compact prompt representation.

        Args:
            memory_id: Full UUID string

        Returns:
            First 8 hex characters (no dashes)
        """
        if not memory_id:
            return ""
        return memory_id.replace('-', '')[:8]

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
            short_id = self._shorten_memory_id(full_id)
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
    ) -> Dict[str, Any]:
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
                    found_ids = {str(m.id) for m in memories}
                    missing_ids = set(memory_ids) - found_ids
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

    def _format_chunk_for_extraction(self, chunk: ProcessingChunk) -> str:
        """
        Format chunk messages for LLM extraction.

        Converts structured message format to Human:/Assistant: dialog format.
        Handles text blocks, skips images, includes tool results.

        Args:
            chunk: ProcessingChunk containing message list

        Returns:
            Formatted continuum text
        """
        formatted_lines = []

        for msg in chunk.messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")

            # Handle structured content (list of content blocks)
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")

                        if item_type == "text":
                            text_parts.append(item.get("text", ""))

                        elif item_type == "image_url":
                            # Skip images (too expensive for memory extraction)
                            continue

                        elif item_type == "tool_result":
                            # Include tool results for context
                            tool_name = item.get("tool_name", "tool")
                            result = item.get("content", "")
                            text_parts.append(f"[{tool_name} result: {result}]")

                    elif isinstance(item, str):
                        text_parts.append(item)

                content = " ".join(text_parts)

            # Format with role prefixes
            # Only process user/assistant roles (system not in continuum, tool/developer unsupported)
            if role == "user":
                formatted_lines.append(f"Human: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")
            else:
                # Discard unsupported roles (tool, developer, system, etc.)
                logger.debug(f"Skipping message with unsupported role: {role}")
                continue

        return "\n".join(formatted_lines)

    def _build_extraction_prompt(
        self,
        formatted_messages: str,
        memory_context: Dict[str, Any]
    ) -> str:
        """
        Build extraction prompt with memory context.

        Formats template with continuum and already-known information.
        Uses shortened UUIDs for compact representation.

        Args:
            formatted_messages: Formatted continuum text
            memory_context: Memory context with IDs and texts

        Returns:
            Complete extraction prompt
        """
        # Build memory context section
        memory_context_text = ""
        memory_texts = memory_context.get("memory_texts")

        if memory_texts:
            memory_lines = ["## Already Known Information"]
            uuid_to_short = memory_context.get("uuid_to_short", {})

            # memory_texts is always dict: {uuid: text}
            for memory_id, memory_text in memory_texts.items():
                short_id = uuid_to_short.get(memory_id, memory_id[:8])
                memory_lines.append(f"- [{short_id}] {memory_text}")

            memory_context_text = "\n".join(memory_lines)

        # Fill template
        extraction_prompt = self.extraction_user_template.format(
            formatted_messages=formatted_messages,
            known_memories_section=memory_context_text
        )

        return extraction_prompt

    def _build_batch_messages(
        self,
        chunk: ProcessingChunk,
        memory_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Build Anthropic messages from ProcessingChunk for Batch API.

        Args:
            chunk: ProcessingChunk with messages
            memory_context: Memory context dict

        Returns:
            List of message dicts in Anthropic format
        """
        messages = []

        # Add memory context if available
        if memory_context and memory_context.get("memory_texts"):
            context_lines = ["EXISTING MEMORIES (referenced in this continuum):"]
            memory_texts = memory_context["memory_texts"]

            # memory_texts is always dict: {uuid: text}
            for mem_id, text in memory_texts.items():
                context_lines.append(f"- [{mem_id}] {text}")

            context_lines.append("\nNow extract memories from the following continuum:")
            messages.append({"role": "user", "content": "\n".join(context_lines)})

        # Add continuum messages
        for msg in chunk.messages:
            metadata = getattr(msg, "metadata", {}) or {}

            if metadata.get("system_notification"):
                continue
            if msg.role == "tool":
                continue

            if msg.role == "user":
                content = self._extract_content(msg)
                if content:
                    messages.append({"role": "user", "content": content})
            elif msg.role == "assistant":
                content = self._extract_content(msg)
                if content or not metadata.get("has_tool_calls"):
                    messages.append({"role": "assistant", "content": content or ""})

        # Always add extraction instruction as standalone user message with clear barrier
        # This ensures proper turn-taking and clear instruction separation
        extraction_instruction = """============================
============================

Extract NEW memories from the above continuum following all extraction principles.

Claude, respond with only valid JSON array. Start with [ and end with ]"""

        if messages:
            # Always append as new message, even if last is user
            messages.append({"role": "user", "content": extraction_instruction})

        return messages

    @staticmethod
    def _extract_content(message: Message) -> str:
        """
        Extract text content from Message object.

        Args:
            message: Message with potentially structured content

        Returns:
            Extracted text content
        """
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            parts = [
                item.get("text", "")
                for item in message.content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return " ".join(parts)
        return str(message.content)
