"""
Memory extraction service for LT_Memory system.

Extracts discrete, meaningful memories from continuum chunks using LLM analysis.
Handles memory context awareness, UUID shortening for efficient prompts, and
semantic deduplication to prevent re-extraction of known information.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from rapidfuzz import fuzz

from lt_memory.models import ExtractedMemory, ProcessingChunk, Memory, ExtractionResult
from config.config import ExtractionConfig
from lt_memory.vector_ops import VectorOps
from lt_memory.db_access import LTMemoryDB
from clients.llm_provider import LLMProvider
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class ExtractionService:
    """
    Extracts memories from continuum chunks using LLM-driven analysis.

    Coordinates:
    - Prompt management and formatting
    - Memory context retrieval (existing memories surfaced during continuum)
    - LLM extraction with retry logic
    - Semantic deduplication against existing memories
    - Conversion to structured ExtractedMemory objects
    """

    def __init__(
        self,
        config: ExtractionConfig,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        llm_provider: LLMProvider
    ):
        """
        Initialize extraction service.

        Args:
            config: Extraction configuration parameters
            vector_ops: Vector operations for similarity search
            db: Database access for memory queries
            llm_provider: LLM provider for extraction calls
        """
        self.config = config
        self.vector_ops = vector_ops
        self.db = db
        self.llm_provider = llm_provider
        self._load_prompts()

    def _load_prompts(self):
        """Load extraction prompts from configuration files."""
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

        Gets actual memory texts for UUIDs captured during continuum
        (surfaced memories and referenced memories). Falls back to database
        query if texts not cached in snapshot.

        Args:
            chunk: Processing chunk with memory_context_snapshot

        Returns:
            Dictionary containing:
            - memory_ids: List of memory UUIDs
            - memory_texts: List of memory texts
            - uuid_to_short: Full UUID to short ID mapping
            - short_to_uuid: Short ID to full UUID mapping
        """
        memory_context = chunk.memory_context_snapshot or {}

        # Extract memory IDs from snapshot
        # Only include explicitly referenced memories as context
        # Surfaced memories that weren't explicitly mentioned are not relevant
        memory_ids = memory_context.get("memory_ids")
        if memory_ids is None:
            # Use only referenced memories, not surfaced
            referenced_ids = memory_context.get("referenced_memory_ids", [])
            memory_ids = referenced_ids

        memory_ids = memory_ids or []

        # Retrieve memory texts
        memory_texts: List[str] = []
        if memory_ids:
            # Try cached texts first
            cached_texts = memory_context.get("memory_texts")

            if isinstance(cached_texts, dict):
                # Dict format: {uuid: text}
                ordered_texts = []
                ordered_ids = []
                for memory_id in memory_ids:
                    text = cached_texts.get(memory_id)
                    if text:
                        ordered_ids.append(memory_id)
                        ordered_texts.append(text)
                memory_ids = ordered_ids
                memory_texts = ordered_texts

            elif isinstance(cached_texts, list) and len(cached_texts) == len(memory_ids):
                # List format (parallel arrays)
                memory_texts = cached_texts

            # Fall back to database if no cached texts
            if not memory_texts:
                memory_uuids = [UUID(mid) for mid in memory_ids]
                memories = self.db.get_memories_by_ids(memory_uuids)

                # Build ID->text mapping
                text_by_id = {str(m.id): m.text for m in memories}

                # Preserve original order, filtering missing memories
                filtered_ids = []
                filtered_texts = []
                for memory_id in memory_ids:
                    text = text_by_id.get(memory_id)
                    if text:
                        filtered_ids.append(memory_id)
                        filtered_texts.append(text)

                memory_ids = filtered_ids
                memory_texts = filtered_texts

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
            if role == "user":
                formatted_lines.append(f"Human: {content}")
            elif role == "assistant":
                formatted_lines.append(f"Assistant: {content}")
            elif role == "system":
                formatted_lines.append(f"System: {content}")
            else:
                formatted_lines.append(f"{role}: {content}")

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
        if memory_context.get("memory_texts"):
            memory_lines = ["## Already Known Information"]
            uuid_to_short = memory_context.get("uuid_to_short", {})

            for memory_id, memory_text in zip(
                memory_context["memory_ids"],
                memory_context["memory_texts"]
            ):
                short_id = uuid_to_short.get(memory_id, memory_id[:8])
                memory_lines.append(f"- [{short_id}] {memory_text}")

            memory_context_text = "\n".join(memory_lines)

        # Fill template
        extraction_prompt = self.extraction_user_template.format(
            formatted_messages=formatted_messages,
            known_memories_section=memory_context_text
        )

        return extraction_prompt

    def _validate_memory_list_structure(self, parsed: Any) -> bool:
        """
        Validate that parsed result is a list of memory dicts.

        Ensures the parsed JSON matches the expected schema: a list where each
        element is a dictionary containing at minimum a "text" field.

        Args:
            parsed: Result from json.loads()

        Returns:
            True if valid structure, False otherwise
        """
        if not isinstance(parsed, list):
            logger.warning(f"Parse result is not a list: {type(parsed).__name__}")
            return False

        for idx, item in enumerate(parsed):
            if not isinstance(item, dict):
                logger.warning(
                    f"Memory list item {idx} is not a dict: {type(item).__name__} = {item}"
                )
                return False

            if "text" not in item:
                logger.warning(f"Memory list item {idx} missing required 'text' field")
                return False

        return True

    def _parse_extraction_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON extraction response from LLM.

        Handles both list format and single object format.
        Strips markdown wrappers and explanatory text if present.
        Attempts JSON repair if initial parsing fails.

        Args:
            response_text: LLM response text (JSON format)

        Returns:
            List of memory dictionaries

        Raises:
            ValueError: If response is not valid JSON and repair fails
        """
        original_text = response_text
        response_text = response_text.strip()

        # Handle empty responses (LLM returned nothing)
        if not response_text:
            logger.warning("Empty response from LLM - no memories extracted")
            return []

        # Try parsing as-is first (handles compliant responses)
        try:
            parsed = json.loads(response_text)

            # Handle different response formats
            if isinstance(parsed, list):
                if not self._validate_memory_list_structure(parsed):
                    raise ValueError("Parsed list contains invalid memory structures")
                return parsed
            elif isinstance(parsed, dict):
                # Single memory object or {"memories": [...]} wrapper
                if "memories" in parsed:
                    memories = parsed["memories"]
                    if isinstance(memories, list):
                        if not self._validate_memory_list_structure(memories):
                            raise ValueError("Parsed 'memories' field contains invalid structures")
                        return memories
                    else:
                        return [memories] if memories else []
                else:
                    # Single memory object
                    return [parsed]
            else:
                raise ValueError(
                    f"Invalid extraction response format: expected list or dict, got {type(parsed).__name__}"
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Malformed JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

            # Try json_repair if available
            try:
                from json_repair import repair_json
            except ImportError:
                logger.warning("json_repair module not installed - cannot attempt repair")
                logger.info("Install with: pip install json-repair")
                raise ValueError(f"Invalid JSON response and json_repair not available: {e}")

            try:
                repaired = repair_json(response_text)
                parsed = json.loads(repaired)

                logger.info("Successfully repaired malformed JSON")

                # Handle repaired response formats
                if isinstance(parsed, list):
                    if not self._validate_memory_list_structure(parsed):
                        logger.error(f"Repaired JSON has invalid structure: {parsed}")
                        raise ValueError("Repaired JSON does not match memory list schema")
                    return parsed
                elif isinstance(parsed, dict):
                    if "memories" in parsed:
                        memories = parsed["memories"]
                        if isinstance(memories, list):
                            if not self._validate_memory_list_structure(memories):
                                logger.error(f"Repaired JSON 'memories' field invalid: {memories}")
                                raise ValueError("Repaired JSON memories field does not match schema")
                            return memories
                        else:
                            return [memories] if memories else []
                    else:
                        return [parsed]
                else:
                    raise ValueError(
                        f"Invalid extraction response format after repair: expected list or dict, got {type(parsed).__name__}"
                    )

            except Exception as repair_error:
                logger.error(f"JSON repair failed: {repair_error}")
                raise ValueError(f"Invalid JSON response: {e}")

    def _remap_short_ids_to_full(
        self,
        memories_data: List[Dict[str, Any]],
        short_to_full: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Remap shortened UUID identifiers back to full UUIDs.

        LLM uses shortened IDs in response; this converts them back.

        Args:
            memories_data: List of memory dicts with shortened IDs
            short_to_full: Mapping from short ID to full UUID

        Returns:
            Updated memory dicts with full UUIDs
        """
        for memory_dict in memories_data:
            # Remap related_memory_ids
            if "related_memory_ids" in memory_dict:
                related_ids = memory_dict["related_memory_ids"]
                if isinstance(related_ids, list):
                    full_ids = [
                        short_to_full.get(short_id, short_id)
                        for short_id in related_ids
                    ]
                    memory_dict["related_memory_ids"] = full_ids

            # Remap consolidates_memory_ids
            if "consolidates_memory_ids" in memory_dict:
                consolidate_ids = memory_dict["consolidates_memory_ids"]
                if isinstance(consolidate_ids, list):
                    full_ids = [
                        short_to_full.get(short_id, short_id)
                        for short_id in consolidate_ids
                    ]
                    memory_dict["consolidates_memory_ids"] = full_ids

        return memories_data

    def _validate_extracted_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """
        Validate and sanitize extracted memory structure.

        Rejects invalid memories, applies intelligent fallbacks for recoverable issues.

        Args:
            memory_dict: Memory dictionary from LLM (modified in-place)

        Returns:
            True if valid (possibly after fixes), False if unrecoverable
        """
        # REJECT: Not a dictionary (defensive type guard)
        if not isinstance(memory_dict, dict):
            logger.warning(f"Rejecting memory: expected dict, got {type(memory_dict).__name__}")
            return False

        # REJECT: Missing or invalid text (unrecoverable)
        if not memory_dict.get("text"):
            logger.warning("Rejecting memory: no text field")
            return False

        if not isinstance(memory_dict["text"], str):
            logger.warning(f"Rejecting memory: text is {type(memory_dict['text'])}, not string")
            return False

        # REJECT: Text too short (unrecoverable)
        text = memory_dict["text"].strip()
        if len(text) < 10:
            logger.warning(f"Rejecting memory: text too short ({len(text)} chars): {text}")
            return False

        # FIX: Normalize relationship type
        relationship_type = memory_dict.get("relationship_type")
        if relationship_type:
            valid_types = {"conflicts", "supports", "supersedes", "related", "null"}
            if relationship_type not in valid_types:
                logger.warning(f"Fixing invalid relationship_type '{relationship_type}' -> null")
                memory_dict["relationship_type"] = None

        # FIX: Ensure UUID lists are valid
        for uuid_field in ["related_memory_ids", "consolidates_memory_ids"]:
            if uuid_field in memory_dict:
                uuid_list = memory_dict[uuid_field]
                if not isinstance(uuid_list, list):
                    logger.warning(f"Fixing {uuid_field}: converting {type(uuid_list)} to empty list")
                    memory_dict[uuid_field] = []

        # FIX: Validate linking hints
        if "linking_hints" in memory_dict:
            hints = memory_dict["linking_hints"]
            if not isinstance(hints, list):
                logger.warning(f"Fixing linking_hints: converting {type(hints)} to empty list")
                memory_dict["linking_hints"] = []
            else:
                # Filter out invalid hints (keep only non-negative integers)
                valid_hints = []
                for hint in hints:
                    if isinstance(hint, int) and hint >= 0:
                        valid_hints.append(hint)
                    else:
                        logger.warning(f"Removing invalid linking hint: {hint} (type: {type(hint)})")
                memory_dict["linking_hints"] = valid_hints

        # FIX: Validate numeric fields with fallbacks
        if "confidence" in memory_dict:
            confidence = memory_dict["confidence"]
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                logger.warning(f"Fixing invalid confidence {confidence} -> 0.9")
                memory_dict["confidence"] = 0.9

        if "importance_score" in memory_dict:
            importance = memory_dict["importance_score"]
            if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
                logger.warning(f"Fixing invalid importance_score {importance} -> None (will use default)")
                memory_dict.pop("importance_score", None)

        return True

    def _is_duplicate_memory(
        self,
        memory_dict: Dict[str, Any],
        memory_context: Dict[str, Any]
    ) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Check if extracted memory is duplicate of existing memory.

        Uses three-stage checking:
        1. Fuzzy text matching against memory context (fast, catches variations)
        2. Vector similarity search (slower but semantic)
        3. Garbage collection handles deep similarity checks later

        Args:
            memory_dict: Extracted memory dictionary
            memory_context: Memory context with existing IDs and texts

        Returns:
            Tuple of (is_duplicate, similarity_score, duplicate_id)
        """
        # Consolidation memories intentionally mirror existing memories
        if memory_dict.get("consolidates_memory_ids"):
            return False, None, None

        memory_text = memory_dict.get("text", "").strip()
        if not memory_text:
            return False, None, None

        # Stage 1: Fuzzy text matching (wider net than exact, cheaper than vector)
        if memory_context:
            context_ids = memory_context.get("memory_ids", [])
            context_texts = memory_context.get("memory_texts", [])

            for idx, existing_text in enumerate(context_texts):
                if not existing_text:
                    continue

                # Use rapidfuzz for fuzzy matching
                similarity = fuzz.ratio(existing_text.strip(), memory_text) / 100.0
                if similarity >= 0.95:  # High fuzzy threshold for near-duplicates
                    duplicate_id = context_ids[idx] if idx < len(context_ids) else None
                    logger.debug(
                        f"Fuzzy match found: similarity {similarity:.3f} "
                        f"with existing memory {duplicate_id}"
                    )
                    return True, similarity, duplicate_id

        # Stage 2: Vector similarity search
        similar_memories = self.vector_ops.find_similar_memories(
            query=memory_text,
            limit=5,
            similarity_threshold=self.config.dedup_similarity_threshold,
            min_importance=0.001  # Filter cold storage (0.0) memories
        )

        if not similar_memories:
            return False, None, None

        # Extract best match score and ID
        best_score = None
        best_memory_id = None

        for memory in similar_memories:
            score = memory.similarity_score
            memory_id = memory.id

            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_memory_id = str(memory_id)

        # Fallback if no scores extracted
        if best_score is None:
            best_score = self.config.dedup_similarity_threshold
            if similar_memories:
                best_memory_id = str(similar_memories[0].id)

        return True, best_score, best_memory_id

    def build_extraction_payload(
        self,
        chunk: ProcessingChunk
    ) -> Dict[str, Any]:
        """
        Build extraction request payload for batch API.

        Creates the prompt and parameters needed for LLM extraction
        without making the actual call. Use this for cost-efficient
        batch processing via Anthropic Message Batches API.

        Args:
            chunk: ProcessingChunk containing continuum messages

        Returns:
            Dictionary with system_prompt, user_prompt, and metadata
        """
        if not chunk.messages:
            return {
                "chunk_index": chunk.chunk_index,
                "system_prompt": self.extraction_system_prompt,
                "user_prompt": "",
                "memory_context": {},
                "short_to_uuid": {}
            }

        # Get memory context for this chunk
        memory_context = self._get_memory_context_for_chunk(chunk)
        short_to_full = memory_context.get("short_to_uuid", {})

        # Format chunk messages
        formatted_messages = self._format_chunk_for_extraction(chunk)

        # Build extraction prompt with memory context
        extraction_prompt = self._build_extraction_prompt(
            formatted_messages,
            memory_context
        )

        return {
            "chunk_index": chunk.chunk_index,
            "system_prompt": self.extraction_system_prompt,
            "user_prompt": extraction_prompt,
            "memory_context": memory_context,
            "short_to_uuid": short_to_full
        }

    def process_batch_result(
        self,
        response_text: str,
        memory_context: Dict[str, Any],
        short_to_uuid: Dict[str, str]
    ) -> ExtractionResult:
        """
        Process batch extraction result.

        Parses LLM response and converts to ExtractedMemory objects with linking hints.
        Used by batch orchestrator after retrieving batch results.

        Args:
            response_text: LLM response text from batch result
            memory_context: Memory context used during extraction
            short_to_uuid: UUID mapping from extraction payload

        Returns:
            ExtractionResult containing memories and linking pairs

        Raises:
            ValueError: If response parsing fails
            RuntimeError: If extraction processing fails
        """
        # Parse JSON response
        memories_data = self._parse_extraction_response(response_text)

        # Remap shortened UUIDs to full UUIDs
        memories_data = self._remap_short_ids_to_full(memories_data, short_to_uuid)

        # Process and validate each extracted memory
        # Track index mapping: original LLM response index → filtered list index
        extracted_memories = []
        original_to_filtered_idx = {}

        for original_idx, memory_dict in enumerate(memories_data):
            # Validate structure
            if not self._validate_extracted_memory(memory_dict):
                continue

            # Check for duplicates
            is_duplicate, similarity, duplicate_id = self._is_duplicate_memory(
                memory_dict,
                memory_context
            )

            if is_duplicate:
                logger.info(
                    f"Skipping duplicate memory: similarity {similarity:.3f} "
                    f"with existing memory {duplicate_id}"
                )
                continue

            # Create ExtractedMemory object
            # Temporal fields are validated by Pydantic model
            extracted_memory = ExtractedMemory(
                text=memory_dict["text"],
                importance_score=self.config.default_importance_score,
                expires_at=memory_dict.get("expires_at"),
                happens_at=memory_dict.get("happens_at"),
                confidence=memory_dict.get("confidence", 0.9),
                relationship_type=memory_dict.get("relationship_type"),
                related_memory_ids=memory_dict.get("related_memory_ids", []),
                consolidates_memory_ids=memory_dict.get("consolidates_memory_ids", []),
                linking_hints=memory_dict.get("linking_hints", [])
            )

            # Track mapping from original index to filtered index
            filtered_idx = len(extracted_memories)
            original_to_filtered_idx[original_idx] = filtered_idx
            extracted_memories.append(extracted_memory)

        # Build linking pairs from hints with index remapping
        linking_pairs = []
        for filtered_idx, memory in enumerate(extracted_memories):
            # Deduplicate hints and remap to filtered indices
            for original_hint_idx in set(memory.linking_hints):
                # Check if hinted memory survived filtering
                if original_hint_idx not in original_to_filtered_idx:
                    logger.debug(
                        f"Ignoring linking hint to filtered memory: "
                        f"memory[{filtered_idx}] -> original[{original_hint_idx}]"
                    )
                    continue

                target_filtered_idx = original_to_filtered_idx[original_hint_idx]

                # Prevent self-references
                if target_filtered_idx == filtered_idx:
                    logger.warning(
                        f"Ignoring self-referencing linking hint: memory[{filtered_idx}]"
                    )
                    continue

                linking_pairs.append((filtered_idx, target_filtered_idx))

        logger.info(f"Processed batch result: {len(extracted_memories)} memories, {len(linking_pairs)} linking hints")

        # Return ExtractionResult with memories and linking pairs
        return ExtractionResult(
            memories=extracted_memories,
            linking_pairs=linking_pairs
        )

    def extract_from_chunk(
        self,
        chunk: ProcessingChunk
    ) -> ExtractionResult:
        """
        Synchronously extract memories from a continuum chunk.

        Makes immediate LLM call for extraction. Use sparingly -
        prefer batch extraction for cost efficiency (50% savings).

        Orchestrates the complete pipeline:
        1. Load memory context
        2. Format chunk messages
        3. Build extraction prompt
        4. Call LLM with retry logic
        5. Parse and validate response
        6. Deduplicate against existing memories
        7. Return ExtractedMemory objects

        Args:
            chunk: ProcessingChunk containing continuum messages

        Returns:
            ExtractionResult containing memories and linking pairs

        Raises:
            RuntimeError: If extraction fails after all retry attempts
        """
        if not chunk.messages:
            logger.info("Empty chunk, no extraction needed")
            return ExtractionResult(memories=[], linking_pairs=[])

        # Get memory context for this chunk
        memory_context = self._get_memory_context_for_chunk(chunk)
        short_to_full = memory_context.get("short_to_uuid", {})

        # Format chunk messages
        formatted_messages = self._format_chunk_for_extraction(chunk)

        # Build extraction prompt with memory context
        extraction_prompt = self._build_extraction_prompt(
            formatted_messages,
            memory_context
        )

        # Extraction with retry logic
        for attempt in range(self.config.retry_attempts + 1):
            try:
                response = self.llm_provider.generate_response(
                    messages=[
                        {"role": "system", "content": self.extraction_system_prompt},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    model_override=self.config.extraction_model,
                    temperature=self.config.extraction_temperature,
                    max_tokens=self.config.max_extraction_tokens,
                    top_p=0.3,
                    frequency_penalty=0.5,
                    response_format={"type": "json_object"}
                )

                # Extract text content from response
                response_text = self.llm_provider.extract_text_content(response)

                # Parse JSON response
                memories_data = self._parse_extraction_response(response_text)

                # Remap shortened UUIDs to full UUIDs
                memories_data = self._remap_short_ids_to_full(memories_data, short_to_full)

                # Process and validate each extracted memory
                # Track index mapping: original LLM response index → filtered list index
                extracted_memories = []
                original_to_filtered_idx = {}

                for original_idx, memory_dict in enumerate(memories_data):
                    # Validate structure
                    if not self._validate_extracted_memory(memory_dict):
                        continue

                    # Check for duplicates
                    is_duplicate, similarity, duplicate_id = self._is_duplicate_memory(
                        memory_dict,
                        memory_context
                    )

                    if is_duplicate:
                        logger.info(
                            f"Skipping duplicate memory: similarity {similarity:.3f} "
                            f"with existing memory {duplicate_id}"
                        )
                        continue

                    # Create ExtractedMemory object
                    # Temporal fields are validated by Pydantic model
                    extracted_memory = ExtractedMemory(
                        text=memory_dict["text"],
                        importance_score=self.config.default_importance_score,
                        expires_at=memory_dict.get("expires_at"),
                        happens_at=memory_dict.get("happens_at"),
                        confidence=memory_dict.get("confidence", 0.9),
                        relationship_type=memory_dict.get("relationship_type"),
                        related_memory_ids=memory_dict.get("related_memory_ids", []),
                        consolidates_memory_ids=memory_dict.get("consolidates_memory_ids", []),
                        linking_hints=memory_dict.get("linking_hints", [])
                    )

                    # Track mapping from original index to filtered index
                    filtered_idx = len(extracted_memories)
                    original_to_filtered_idx[original_idx] = filtered_idx
                    extracted_memories.append(extracted_memory)

                # Build linking pairs from hints with index remapping
                linking_pairs = []
                for filtered_idx, memory in enumerate(extracted_memories):
                    # Deduplicate hints and remap to filtered indices
                    for original_hint_idx in set(memory.linking_hints):
                        # Check if hinted memory survived filtering
                        if original_hint_idx not in original_to_filtered_idx:
                            logger.debug(
                                f"Ignoring linking hint to filtered memory: "
                                f"memory[{filtered_idx}] -> original[{original_hint_idx}]"
                            )
                            continue

                        target_filtered_idx = original_to_filtered_idx[original_hint_idx]

                        # Prevent self-references
                        if target_filtered_idx == filtered_idx:
                            logger.warning(
                                f"Ignoring self-referencing linking hint: memory[{filtered_idx}]"
                            )
                            continue

                        linking_pairs.append((filtered_idx, target_filtered_idx))

                logger.info(
                    f"Extracted {len(extracted_memories)} memories from "
                    f"chunk {chunk.chunk_index}, {len(linking_pairs)} linking hints"
                )

                # Return ExtractionResult with memories and linking pairs
                return ExtractionResult(
                    memories=extracted_memories,
                    linking_pairs=linking_pairs
                )

            except Exception as e:
                logger.warning(f"Extraction attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts:
                    continue
                else:
                    # All attempts failed
                    raise RuntimeError(
                        f"Extraction failed after {self.config.retry_attempts + 1} "
                        f"attempts: {str(e)}"
                    )

    def cleanup(self):
        """
        Clean up resources.

        No-op: Dependencies managed by factory lifecycle.
        Nulling references breaks in-flight scheduler jobs.
        """
        logger.debug("ExtractionService cleanup completed (no-op)")
