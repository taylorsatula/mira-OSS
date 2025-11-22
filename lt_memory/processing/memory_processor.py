"""
Memory processor - parse and validate LLM extraction responses.

Consolidates all response processing logic into a single, focused module:
- JSON parsing with repair fallback
- Structure validation
- UUID remapping (short → full)
- Memory field validation and sanitization
- Duplicate detection (fuzzy + vector)
- Index remapping for filtered memories
- Linking pair construction

This module is pure data processing with no side effects.
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from rapidfuzz import fuzz

from lt_memory.models import ExtractedMemory, ExtractionResult
from lt_memory.vector_ops import VectorOps
from config.config import ExtractionConfig

logger = logging.getLogger(__name__)


class MemoryProcessor:
    """
    Process LLM extraction responses into validated ExtractedMemory objects.

    Single Responsibility: Transform raw LLM text → validated memories + linking pairs

    No side effects - pure data processing that can be tested independently.
    """

    def __init__(self, config: ExtractionConfig, vector_ops: VectorOps):
        """
        Initialize memory processor.

        Args:
            config: Extraction configuration
            vector_ops: Vector operations for similarity search (deduplication)
        """
        self.config = config
        self.vector_ops = vector_ops

    def process_extraction_response(
        self,
        response_text: str,
        short_to_uuid: Dict[str, str],
        memory_context: Dict[str, Any]
    ) -> ExtractionResult:
        """
        Process batch extraction result from LLM response.

        Complete pipeline: Parse → Remap IDs → Validate → Deduplicate → Build linking pairs

        Args:
            response_text: LLM response text (JSON format)
            short_to_uuid: Mapping from shortened IDs to full UUIDs
            memory_context: Memory context used during extraction (for deduplication)

        Returns:
            ExtractionResult containing validated memories and linking pairs

        Raises:
            ValueError: If response parsing fails catastrophically
        """
        # Step 1: Parse JSON response
        memories_data = self._parse_extraction_response(response_text)

        # Step 2: Remap shortened UUIDs to full UUIDs
        memories_data = self._remap_short_ids_to_full(memories_data, short_to_uuid)

        # Step 3: Validate and deduplicate memories
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

        # Step 4: Build linking pairs from hints with index remapping
        linking_pairs = self._build_linking_pairs(
            extracted_memories,
            original_to_filtered_idx
        )

        logger.info(
            f"Processed extraction response: {len(extracted_memories)} memories, "
            f"{len(linking_pairs)} linking hints"
        )

        return ExtractionResult(
            memories=extracted_memories,
            linking_pairs=linking_pairs
        )

    def _parse_extraction_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON extraction response from LLM.

        Handles multiple formats with repair fallback:
        - List of memory dicts (standard)
        - Single memory dict → wrapped in list
        - {"memories": [...]} wrapper → extracted
        - Empty/whitespace responses → empty list

        Args:
            response_text: LLM response text (JSON format)

        Returns:
            List of memory dictionaries

        Raises:
            ValueError: If response is not valid JSON and repair fails
        """
        response_text = response_text.strip()

        # Handle empty responses (LLM returned nothing)
        if not response_text:
            logger.warning(
                "LLM returned empty response - should return valid JSON like [] or {\"memories\": []}. "
                "May indicate API issue or prompt problem."
            )
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
            # Check if it's actually an empty response that looks like whitespace/special chars
            if not response_text or response_text.isspace():
                logger.debug("Response contains only whitespace - no memories extracted")
                return []

            # Log the actual error with the first part of the response for debugging
            logger.warning(f"JSON parsing failed: {e}")
            logger.debug(f"Response text (first 200 chars): {response_text[:200]!r}")

            # Try json_repair (required dependency)
            try:
                from json_repair import repair_json
            except ImportError:
                raise ValueError(f"Invalid JSON response and json_repair unavailable: {e}")

            try:
                repaired = repair_json(response_text)

                # Check if repair actually changed something
                if repaired == response_text:
                    logger.error(
                        f"json_repair could not repair response (returned unchanged). "
                        f"Response is not valid JSON. First 200 chars: {response_text[:200]!r}"
                    )
                    raise ValueError(
                        "LLM response is not valid JSON and json_repair could not fix it. "
                        "Indicates LLM output format issue."
                    )

                parsed = json.loads(repaired)
                logger.debug("Successfully repaired malformed JSON")

                # Handle repaired response formats
                if isinstance(parsed, list):
                    if not self._validate_memory_list_structure(parsed):
                        logger.debug(f"Repaired JSON has invalid structure: {parsed}")
                        raise ValueError("Repaired JSON does not match memory list schema")
                    return parsed
                elif isinstance(parsed, dict):
                    if "memories" in parsed:
                        memories = parsed["memories"]
                        if isinstance(memories, list):
                            if not self._validate_memory_list_structure(memories):
                                logger.debug(f"Repaired JSON 'memories' field invalid: {memories}")
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

            except json.JSONDecodeError as e:
                # Even after repair, still not valid JSON - this is an error
                logger.error(
                    f"Repaired response still not valid JSON: {e}. "
                    f"Repaired text (first 200 chars): {repaired[:200]!r}"
                )
                raise ValueError(
                    f"LLM response invalid even after json_repair attempt: {e}"
                ) from e
            except Exception as repair_error:
                # Unexpected error during repair - propagate it
                logger.error(f"Unexpected error during JSON repair: {repair_error}")
                raise ValueError(
                    f"JSON repair failed with unexpected error: {repair_error}"
                ) from repair_error

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
        Modifies memory_dict in-place with fixes.

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

            # Handle dict format: {uuid: text}
            if isinstance(context_texts, dict):
                for memory_id, existing_text in context_texts.items():
                    if not existing_text:
                        continue

                    # Use rapidfuzz for fuzzy matching
                    similarity = fuzz.ratio(existing_text.strip(), memory_text) / 100.0
                    if similarity >= 0.95:  # High fuzzy threshold for near-duplicates
                        logger.debug(
                            f"Fuzzy match found: similarity {similarity:.3f} "
                            f"with existing memory {memory_id}"
                        )
                        return True, similarity, memory_id
            # Handle list format (parallel arrays)
            elif isinstance(context_texts, list):
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

    def _build_linking_pairs(
        self,
        extracted_memories: List[ExtractedMemory],
        original_to_filtered_idx: Dict[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Build linking pairs from extraction hints with index remapping.

        Handles memories that were filtered out during validation/deduplication.

        Args:
            extracted_memories: List of validated ExtractedMemory objects
            original_to_filtered_idx: Mapping from original LLM index to filtered index

        Returns:
            List of (source_idx, target_idx) tuples
        """
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

        return linking_pairs
