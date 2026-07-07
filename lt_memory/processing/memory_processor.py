"""
Memory processor - parse and validate LLM extraction responses.

Consolidates all response processing logic into a single, focused module:
- JSON parsing with repair fallback
- Structure validation
- UUID remapping (short → full)
- Memory field validation and sanitization
- Duplicate detection (fuzzy + vector)
- Index remapping for filtered memories

This module is pure data processing with no side effects.
"""
import json
import logging
from typing import List, Dict, Any, NamedTuple, NotRequired, Optional, Tuple, TypedDict
from uuid import UUID

from rapidfuzz import fuzz

from lt_memory.models import ExtractedMemory, ExtractionResult, MemoryContext
from lt_memory.vector_ops import VectorOps

logger = logging.getLogger(__name__)

# Extraction deduplication tuning
DEDUP_SIMILARITY_THRESHOLD = 0.92  # Cosine similarity for duplicate detection
DEFAULT_IMPORTANCE_SCORE = 0.5     # Default importance for newly extracted memories


class DuplicateCheckResult(NamedTuple):
    """Result of duplicate memory check."""
    is_duplicate: bool
    similarity: float | None
    duplicate_id: str | None


class RawMemoryDict(TypedDict, total=False):
    """Raw memory dict from LLM extraction response before validation."""
    text: str
    importance_score: float
    expires_at: str | None
    happens_at: str | None
    related_memory_ids: list[dict[str, str]]
    entities: list[dict[str, str]]


class MemoryProcessor:
    """
    Process LLM extraction responses into validated ExtractedMemory objects.

    Single Responsibility: Transform raw LLM text → validated memories

    No side effects - pure data processing that can be tested independently.
    """

    def __init__(self, vector_ops: VectorOps):
        self.vector_ops = vector_ops

    def process_extraction_response(
        self,
        response_text: str,
        short_to_uuid: Dict[str, str],
        memory_context: MemoryContext
    ) -> ExtractionResult:
        """
        Process batch extraction result from LLM response.

        Complete pipeline: Parse → Remap IDs → Validate → Deduplicate

        Args:
            response_text: LLM response text (JSON format)
            short_to_uuid: Mapping from shortened IDs to full UUIDs
            memory_context: Memory context used during extraction (for deduplication)

        Returns:
            ExtractionResult containing validated memories

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
                importance_score=DEFAULT_IMPORTANCE_SCORE,
                expires_at=memory_dict.get("expires_at"),
                happens_at=memory_dict.get("happens_at"),
                related_memory_ids=memory_dict.get("related_memory_ids", []),
                entities=memory_dict.get("entities", [])
            )

            # Track mapping from original index to filtered index
            filtered_idx = len(extracted_memories)
            original_to_filtered_idx[original_idx] = filtered_idx
            extracted_memories.append(extracted_memory)

        logger.info(
            f"Processed extraction response: {len(extracted_memories)} memories"
        )

        return ExtractionResult(
            memories=extracted_memories
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

    def _validate_memory_list_structure(self, parsed: object) -> bool:
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
        memories_data: List[RawMemoryDict],
        short_to_full: Dict[str, str]
    ) -> List[RawMemoryDict]:
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
            # Remap related_memory_ids: remap id field inside dicts, drop unresolved
            if "related_memory_ids" in memory_dict:
                related_ids = memory_dict["related_memory_ids"]
                if isinstance(related_ids, list):
                    valid_refs = []
                    for ref in related_ids:
                        if isinstance(ref, dict) and "id" in ref:
                            ref["id"] = short_to_full.get(ref["id"], ref["id"])
                            try:
                                UUID(ref["id"])
                                valid_refs.append(ref)
                            except ValueError:
                                logger.warning(f"Dropping unresolved related_memory_id after remap: {ref['id']}")
                        else:
                            valid_refs.append(ref)
                    memory_dict["related_memory_ids"] = valid_refs

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

        # REJECT: Validate related_memory_ids as list of ExtractionRef dicts
        if "related_memory_ids" in memory_dict:
            refs = memory_dict["related_memory_ids"]
            if not isinstance(refs, list):
                logger.error(f"Rejecting memory: related_memory_ids is {type(refs)}, not list")
                return False
            for ref in refs:
                if not isinstance(ref, dict) or "id" not in ref or "bond" not in ref:
                    logger.error(
                        f"Rejecting memory: malformed related_memory_ids entry {ref!r}. "
                        f"Expected {{'id': str, 'bond': str}}"
                    )
                    return False

        # FIX: Validate numeric fields with fallbacks
        if "importance_score" in memory_dict:
            importance = memory_dict["importance_score"]
            if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
                logger.warning(f"Fixing invalid importance_score {importance} -> None (will use default)")
                memory_dict.pop("importance_score", None)

        # Validate and filter entities list
        if "entities" in memory_dict:
            entities = memory_dict["entities"]
            if not isinstance(entities, list):
                logger.warning(f"Fixing entities: converting {type(entities)} to empty list")
                memory_dict["entities"] = []
            else:
                # Filter to valid entity dicts
                valid_entities = []
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    if 'name' not in entity or 'type' not in entity:
                        continue
                    if not isinstance(entity['name'], str) or not isinstance(entity['type'], str):
                        continue
                    # Normalize entity name (strip whitespace)
                    entity['name'] = entity['name'].strip()
                    if len(entity['name']) < 2:
                        continue
                    valid_entities.append(entity)
                memory_dict["entities"] = valid_entities

        return True

    def _is_duplicate_memory(
        self,
        memory_dict: Dict[str, Any],
        memory_context: MemoryContext
    ) -> DuplicateCheckResult:
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
        memory_text = memory_dict.get("text", "").strip()
        if not memory_text:
            return DuplicateCheckResult(False, None, None)

        # Stage 1: Fuzzy text matching (wider net than exact, cheaper than vector)
        if memory_context:
            context_ids = memory_context.get("memory_ids", [])
            context_texts = memory_context.get("memory_texts", [])

            # Dict format: {uuid: text} — produced by extraction_engine.py
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
                        return DuplicateCheckResult(True, similarity, memory_id)

        # Stage 2: Vector similarity search
        similar_memories = self.vector_ops.find_similar_for_dedup(
            query_text=memory_text,
            limit=5,
            similarity_threshold=DEDUP_SIMILARITY_THRESHOLD,
            min_importance=0.001  # Filter cold storage (0.0) memories
        )

        if not similar_memories:
            return DuplicateCheckResult(False, None, None)

        # Extract best match score and ID
        best_score = None
        best_memory_id = None

        for memory in similar_memories:
            score = memory.similarity_score
            if score is not None and (best_score is None or score > best_score):
                best_score = score
                best_memory_id = memory.id

        # Fallback if no scores extracted
        if best_score is None:
            best_score = DEDUP_SIMILARITY_THRESHOLD
            if similar_memories:
                best_memory_id = similar_memories[0].id

        return DuplicateCheckResult(True, best_score, best_memory_id)
