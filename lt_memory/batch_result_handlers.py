"""
Batch result handlers for processing completed Anthropic Batch API results.

Each handler implements BatchResultProcessor and is routed via
PostProcessingBatchDispatcher from the batch polling job.
"""
import json
import logging
from datetime import timedelta
from typing import List, Dict, Any, Optional
from uuid import UUID

import anthropic
from json_repair import repair_json

from lt_memory.db_access import LTMemoryDB
from lt_memory.processing.batch_coordinator import BatchResultProcessor
from lt_memory.processing.memory_processor import MemoryProcessor
from lt_memory.vector_ops import VectorOps
from lt_memory.linking import LinkingService
from lt_memory.llm_routing import uses_anthropic_batch_dialect
from lt_memory.models import ExtractionBatch, PostProcessingBatch, ExtractedMemory, MemoryLink, LinkingPair, ClassificationPair
from lt_memory.processing.batch_coordinator import BATCH_EXPIRY_HOURS
from clients.llm_provider import LLMProvider, build_batch_params
from utils.user_context import set_current_user_id, clear_user_context
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class ExtractionBatchResultHandler(BatchResultProcessor):
    """
    Handle extraction batch results: parse, store, persist entities, trigger relationships.
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        memory_processor: MemoryProcessor,
        vector_ops: VectorOps,
        db: LTMemoryDB,
        linking_service: LinkingService,
        llm_provider: LLMProvider,
        batch_coordinator: 'BatchCoordinator'
    ):
        self.anthropic_client = anthropic_client
        self.memory_processor = memory_processor
        self.vector_ops = vector_ops
        self.db = db
        self.linking = linking_service
        self.llm_provider = llm_provider
        self.batch_coordinator = batch_coordinator

        # Accumulates memory_ids and linking_pairs across chunks within a batch,
        # so relationship classification is triggered once per Anthropic batch
        # rather than once per chunk.
        self._pending_relationships: Dict[str, Dict[str, Any]] = {}

    def process_result(self, batch_id: str, batch: ExtractionBatch) -> bool:
        """
        Process extraction batch result.

        Args:
            batch_id: Anthropic batch ID
            batch: Extraction batch record

        Returns:
            True if processing succeeded
        """
        # Retrieve result from Anthropic
        for result in self.anthropic_client.beta.messages.batches.results(batch_id):
            if result.custom_id != batch.custom_id:
                continue

            if result.result.type == "succeeded":
                # Extract text from response
                text_blocks = [
                    block.text for block in result.result.message.content
                    if block.type == "text" and block.text
                ]

                # Skip if no text content
                if not text_blocks:
                    logger.debug(f"Batch {batch_id} result had no text content - skipping")
                    self.db.delete_batch("extraction", batch.id, user_id=batch.user_id)
                    return True

                response_text = "\n".join(text_blocks)

                # Process extraction result
                set_current_user_id(batch.user_id)
                try:
                    short_to_uuid = batch.chunk_metadata.get("short_to_uuid", {}) if batch.chunk_metadata else {}
                    segment_id = batch.chunk_metadata.get("segment_id") if batch.chunk_metadata else None

                    # Parse with memory processor
                    result = self.memory_processor.process_extraction_response(
                        response_text=response_text,
                        short_to_uuid=short_to_uuid,
                        memory_context=batch.memory_context or {}
                    )

                    memories = result.memories
                    linking_pairs = result.linking_pairs

                    # Set source_segment_id on extracted memories for resume cleanup
                    if segment_id:
                        for memory in memories:
                            memory.source_segment_id = UUID(segment_id)

                    # Store memories with embeddings
                    memory_ids = []
                    if memories:
                        memory_ids = self.vector_ops.store_memories_with_embeddings(memories)

                        # Persist extraction-time links to existing memories (related_memory_ids)
                        extraction_links = []
                        for idx, memory in enumerate(memories):
                            if idx < len(memory_ids) and memory.related_memory_ids:
                                new_id = memory_ids[idx]
                                for ref in memory.related_memory_ids:
                                    related_id = UUID(ref["id"])
                                    bond = ref.get("bond", "")
                                    extraction_links.append(MemoryLink(
                                        source_id=new_id,
                                        target_id=related_id,
                                        link_type="extraction_ref",
                                        reasoning=bond if bond else "Referenced during conversation",
                                        extraction_bond=bond,
                                        created_at=utc_now()
                                    ))

                        if extraction_links:
                            self.db.create_links(extraction_links)
                            logger.info(f"Batch {batch_id}: created {len(extraction_links)} extraction_ref links")

                        # Persist LLM-extracted entities via fuzzy matching
                        self._persist_llm_entities(batch.user_id, memories, memory_ids)

                        # Accumulate for finalize_batch() — relationship classification
                        # triggers once after all chunks are processed, not per-chunk
                        if batch_id not in self._pending_relationships:
                            self._pending_relationships[batch_id] = {
                                "user_id": batch.user_id,
                                "memory_ids": [],
                                "linking_pairs": []
                            }
                        acc = self._pending_relationships[batch_id]
                        offset = len(acc["memory_ids"])
                        acc["memory_ids"].extend(memory_ids)
                        if linking_pairs:
                            for pair in linking_pairs:
                                acc["linking_pairs"].append({
                                    "source_idx": pair["source_idx"] + offset,
                                    "target_idx": pair["target_idx"] + offset,
                                    "bond": pair.get("bond", "")
                                })

                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: {len(memory_ids)} stored")
                    else:
                        logger.info(f"Batch {batch_id} chunk {batch.custom_id}: no memories extracted")

                    # Apply importance boosts to existing memories
                    memory_context = batch.memory_context or {}
                    pinned_short_ids = memory_context.get("pinned_short_ids", [])
                    if pinned_short_ids:
                        self.db.apply_pin_boost(pinned_short_ids, user_id=batch.user_id)

                    # Delete batch record - processing complete
                    self.db.delete_batch("extraction", batch.id, user_id=batch.user_id)
                    return True
                finally:
                    clear_user_context()

            elif result.result.type == "errored":
                self.db.update_batch_status("extraction",
                    batch.id,
                    "failed",
                    error_message=str(result.result.error),
                    user_id=batch.user_id
                )
                return False

        return False

    def finalize_batch(self, batch_id: str, user_id: str) -> None:
        """
        Trigger relationship classification once for all chunks in this batch.

        Called by poll_batches() after all chunk records for a batch_id have
        been processed. Consumes the accumulated memory_ids and linking_pairs.
        """
        acc = self._pending_relationships.pop(batch_id, None)
        if not acc or not acc["memory_ids"]:
            return

        set_current_user_id(user_id)
        try:
            self._trigger_relationship_classification(
                user_id,
                acc["memory_ids"],
                acc["linking_pairs"] or None
            )
        finally:
            clear_user_context()

    def _persist_llm_entities(
        self,
        user_id: str,
        memories: List[ExtractedMemory],
        memory_ids: List[UUID]
    ) -> None:
        """Persist LLM-extracted entities via pg_trgm fuzzy matching."""
        if not memory_ids or len(memories) != len(memory_ids):
            return

        try:
            total_links = 0

            for memory, memory_id in zip(memories, memory_ids):
                if not memory.entities:
                    continue

                seen_names = set()
                for entity_dict in memory.entities:
                    entity_name = entity_dict['name']
                    entity_type = entity_dict.get('type', 'UNKNOWN')

                    if entity_name in seen_names:
                        continue
                    seen_names.add(entity_name)

                    entity = self.db.get_or_create_entity(
                        name=entity_name,
                        entity_type=entity_type,
                        user_id=user_id
                    )

                    self.db.link_memory_to_entity(
                        memory_id=memory_id,
                        entity_id=entity.id,
                        entity_name=entity_name,
                        entity_type=entity.entity_type,
                        user_id=user_id
                    )
                    total_links += 1

            if total_links:
                logger.info(f"Persisted LLM entities: {total_links} links for {len(memories)} memories")

        except Exception:
            logger.warning(
                "Entity persistence failed for user %s (non-critical)",
                user_id,
                exc_info=True
            )

    def _trigger_relationship_classification(
        self,
        user_id: str,
        memory_ids: List[UUID],
        linking_hints: List[LinkingPair] | None = None
    ) -> None:
        """
        Trigger relationship classification for new memories.

        Builds memory pairs from extraction hints and similarity search,
        then submits batch to Anthropic for classification.
        """
        if not memory_ids:
            return

        all_pairs = []
        seen_pairs: set[frozenset] = set()

        # Process extraction hints first (LinkingPair dicts)
        if linking_hints:
            new_memories = {m.id: m for m in self.db.get_memories_by_ids(memory_ids, user_id=user_id)}

            for pair in linking_hints:
                src_idx = pair["source_idx"]
                tgt_idx = pair["target_idx"]
                bond = pair.get("bond", "")

                if src_idx < len(memory_ids) and tgt_idx < len(memory_ids):
                    src_id = memory_ids[src_idx]
                    tgt_id = memory_ids[tgt_idx]

                    if src_id in new_memories and tgt_id in new_memories:
                        pair_key = frozenset({src_id, tgt_id})
                        seen_pairs.add(pair_key)
                        all_pairs.append({
                            "new_memory_id": src_id,
                            "similar_memory_id": tgt_id,
                            "new_memory": new_memories[src_id],
                            "similar_memory": new_memories[tgt_id],
                            "from_extraction_hint": True,
                            "bond": bond
                        })

            logger.info(f"Added {len(all_pairs)} pairs from extraction hints for user {user_id}")

        # Find similar existing memories (deduplicate bidirectional pairs)
        for mem_id in memory_ids:
            candidates = self.linking.find_similar_candidates(mem_id)
            for candidate in candidates:
                pair_key = frozenset({mem_id, candidate.id})
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                all_pairs.append({
                    "new_memory_id": mem_id,
                    "similar_memory_id": candidate.id,
                    "new_memory": self.db.get_memory(mem_id, user_id=user_id),
                    "similar_memory": candidate,
                    "from_extraction_hint": False
                })

        if not all_pairs:
            return

        if not uses_anthropic_batch_dialect("relationship"):
            logger.warning(
                f"Bypassing relationship batch for user {user_id} - "
                f"executing {len(all_pairs)} classifications immediately"
            )
            self._execute_relationship_classification_immediately(user_id, all_pairs)
            return

        # Build and submit relationship batch
        requests = []
        input_data = {}

        for idx, pair in enumerate(all_pairs):
            # Delegate to linking service to build payload
            bond = pair.get("bond", "")
            payload = self.linking.build_classification_payload(
                pair["new_memory"],
                pair["similar_memory"],
                bond=bond
            )

            new_id_str = str(pair["new_memory_id"])
            similar_id_str = str(pair["similar_memory_id"])
            custom_id = f"{user_id}_rel_{new_id_str[:8]}_{idx}"

            params = build_batch_params(
                'relationship',
                system_prompt=payload["system_prompt"],
                messages=[{"role": "user", "content": payload["user_prompt"]}],
            )
            requests.append({"custom_id": custom_id, "params": params})

            input_data[new_id_str] = {
                "new_memory_id": new_id_str,
                "similar_memory_id": similar_id_str,
                "bond": bond
            }

        # Submit via BatchCoordinator (single submission path)
        batch_id = self.batch_coordinator.submit_batch(
            requests=requests,
            batch_type="relationship_classification",
            user_id=user_id,
        )
        expires_at = utc_now() + timedelta(hours=BATCH_EXPIRY_HOURS)

        # Store batch record
        batch_record = PostProcessingBatch(
            batch_id=batch_id,
            batch_type="relationship_classification",
            user_id=user_id,
            request_payload={"requests": requests},
            input_data=input_data,
            items_submitted=len(requests),
            status="submitted",
            created_at=utc_now(),
            submitted_at=utc_now(),
            expires_at=expires_at
        )
        self.db.create_post_processing_batch(batch_record, user_id=user_id)

        logger.info(f"Submitted relationship batch {batch_id}: {len(requests)} classifications")

    def _execute_relationship_classification_immediately(
        self,
        user_id: str,
        all_pairs: List[ClassificationPair]
    ) -> None:
        """
        Execute relationship classification immediately using OpenAI fallback.

        Called when emergency fallback mode is active.

        Args:
            user_id: User ID
            all_pairs: List of memory pair dicts with new_memory and similar_memory
        """
        try:
            set_current_user_id(user_id)
            links_created = 0

            total = len(all_pairs)
            skipped = 0

            for idx, pair in enumerate(all_pairs):
                # Build classification payload
                payload = self.linking.build_classification_payload(
                    pair["new_memory"],
                    pair["similar_memory"],
                    bond=pair.get("bond", "")
                )

                # Call LLM directly using relationship internal LLM config
                try:
                    response = self.llm_provider.generate_response(
                        messages=[{"role": "user", "content": payload["user_prompt"]}],
                        system_prompt=payload["system_prompt"],
                        internal_llm='relationship',
                        allow_negative=True,  # System task — segment already paid for
                    )
                except Exception:
                    logger.exception("[%s/%s] LLM call failed", idx + 1, total)
                    skipped += 1
                    continue

                # Extract text from response
                response_text = self.llm_provider.extract_text_content(response)

                # Strip think tags, markdown fences, and whitespace
                import re
                cleaned = response_text.strip()
                cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
                cleaned = cleaned.strip()

                # Parse classification result
                try:
                    classification = json.loads(cleaned)
                    rel_type = classification.get("relationship_type")

                    if rel_type and rel_type != "null":
                        # Create bidirectional link
                        if self.linking.create_bidirectional_link(
                            source_id=pair["new_memory_id"],
                            target_id=pair["similar_memory_id"],
                            link_type=rel_type,
                            reasoning=classification.get("reasoning", ""),
                            extraction_bond=pair.get("bond", "")
                        ):
                            links_created += 1
                except json.JSONDecodeError:
                    logger.error(f"[{idx + 1}/{total}] Invalid JSON: {cleaned[:120]}")
                    skipped += 1
                    continue

                if (idx + 1) % 25 == 0 or (idx + 1) == total:
                    logger.info(
                        f"[{idx + 1}/{total}] Progress: {links_created} links created, "
                        f"{skipped} skipped"
                    )

            logger.info(
                f"Immediate relationship classification complete for user {user_id}: "
                f"{links_created} links created"
            )
            clear_user_context()

        except Exception as e:
            logger.error(
                f"Error in immediate relationship classification for {user_id}: {e}",
                exc_info=True
            )
            clear_user_context()


class RelationshipBatchResultHandler(BatchResultProcessor):
    """
    Handle relationship classification results: parse and create bidirectional links.
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        linking_service: LinkingService,
        db: LTMemoryDB
    ):
        self.anthropic_client = anthropic_client
        self.linking = linking_service
        self.db = db

    def process_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """Process relationship classification result."""
        try:
            # Retrieve results from Anthropic
            classifications = {}
            for result in self.anthropic_client.beta.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    response_text = "\n".join([
                        block.text for block in result.result.message.content
                        if block.type == "text"
                    ])

                    if not response_text.strip():
                        logger.warning(f"Empty response in {result.custom_id}")
                        continue

                    # Parse JSON with repair fallback
                    try:
                        classifications[result.custom_id] = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.warning(
                            "JSON parsing failed for %s, attempting repair",
                            result.custom_id,
                            exc_info=True
                        )
                        try:
                            repaired = repair_json(response_text)
                            classifications[result.custom_id] = json.loads(repaired)
                            logger.debug(f"Successfully repaired JSON for {result.custom_id}")
                        except json.JSONDecodeError:
                            logger.exception("JSON repair failed for %s", result.custom_id)
                            logger.debug(
                                "Response text length for %s: %s",
                                result.custom_id,
                                len(response_text)
                            )
                            continue

            if not classifications:
                self.db.update_batch_status("post_processing",
                    batch.id,
                    "failed",
                    error_message="No valid results",
                    user_id=batch.user_id
                )
                return False

            # Create links
            set_current_user_id(batch.user_id)
            links_created = 0

            for custom_id, classification in classifications.items():
                rel_type = classification.get("relationship_type")
                if rel_type == "null":
                    continue

                # Find memory IDs from input_data
                pair_data = next(
                    (data for new_id, data in batch.input_data.items() if new_id[:8] in custom_id),
                    None
                )

                if not pair_data:
                    continue

                # Create bidirectional link
                if self.linking.create_bidirectional_link(
                    source_id=UUID(pair_data["new_memory_id"]),
                    target_id=UUID(pair_data["similar_memory_id"]),
                    link_type=rel_type,
                    reasoning=classification.get("reasoning", ""),
                    extraction_bond=pair_data.get("bond", "")
                ):
                    links_created += 1

            clear_user_context()

            # Delete batch record
            self.db.delete_batch("post_processing", batch.id, user_id=batch.user_id)
            logger.info(f"Relationship batch {batch_id}: {links_created} links created")

            return True

        except Exception as e:
            logger.exception("Error processing relationship result")
            self.db.update_batch_status("post_processing",
                batch.id,
                "failed",
                error_message=str(e),
                user_id=batch.user_id
            )
            return False


class ConsolidationBatchResultHandler(BatchResultProcessor):
    """
    Handle consolidation batch results: parse multi-group LLM decisions,
    execute merges via ConsolidationHandler, track rejections for independents.
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        db: LTMemoryDB,
        consolidation_handler: 'ConsolidationHandler',
    ):
        self.anthropic_client = anthropic_client
        self.db = db
        self.consolidation_handler = consolidation_handler

    def process_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """
        Process consolidation batch result.

        Parses multi-group JSON from each cluster result, executes merges
        via ConsolidationHandler, and increments rejection counts for independents.
        """
        from utils.tag_parser import format_memory_id

        try:
            # Retrieve results from Anthropic
            cluster_decisions = {}
            for result in self.anthropic_client.beta.messages.batches.results(batch_id):
                if result.result.type == "succeeded":
                    response_text = "\n".join([
                        block.text for block in result.result.message.content
                        if block.type == "text"
                    ])
                    try:
                        cluster_decisions[result.custom_id] = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Try json_repair fallback
                        try:
                            cluster_decisions[result.custom_id] = json.loads(
                                repair_json(response_text)
                            )
                        except Exception:
                            logger.error(f"Invalid JSON in {result.custom_id}, repair also failed")

            if not cluster_decisions:
                self.db.update_batch_status("post_processing",
                    batch.id, "failed", error_message="No valid results", user_id=batch.user_id
                )
                return False

            set_current_user_id(batch.user_id)
            try:
                memories_consolidated = 0
                memories_rejected = 0

                for custom_id, decision in cluster_decisions.items():
                    # Extract cluster_id from custom_id format: {user_id}_consol_{cluster_id}
                    cluster_id = custom_id.split("_consol_", 1)[-1] if "_consol_" in custom_id else None
                    cluster_data = batch.input_data.get(cluster_id) if cluster_id else None
                    if not cluster_data:
                        logger.warning(f"No cluster data for {custom_id}")
                        continue

                    # Build short-to-full UUID mapping
                    short_to_full = {}
                    for full_id in cluster_data["memory_ids"]:
                        short_id = format_memory_id(full_id)
                        short_to_full[short_id] = UUID(full_id)

                    # Resolve independent IDs first — independence wins over merge
                    independent_uuids = set()
                    for short_id in decision.get("independent_ids", []):
                        full_uuid = short_to_full.get(short_id)
                        if full_uuid:
                            independent_uuids.add(full_uuid)
                            self.db.increment_consolidation_rejection_count(
                                full_uuid, user_id=str(batch.user_id)
                            )
                            memories_rejected += 1

                    # Process merge groups, excluding any memory marked independent
                    for group in decision.get("merge_groups", []):
                        merged_text = group.get("merged_text", "").strip()
                        if not merged_text:
                            logger.warning(f"Merge group in {custom_id} has no text, skipping")
                            continue

                        group_uuids = []
                        for short_id in group.get("memory_ids", []):
                            full_uuid = short_to_full.get(short_id)
                            if full_uuid and full_uuid not in independent_uuids:
                                group_uuids.append(full_uuid)

                        if len(group_uuids) < 2:
                            continue

                        try:
                            new_memory_id = self.consolidation_handler.execute_consolidation(
                                old_memory_ids=group_uuids,
                                consolidated_text=merged_text,
                                user_id=str(batch.user_id),
                                merge_note=group.get("reason"),
                            )
                            memories_consolidated += len(group_uuids)
                            logger.info(
                                f"Batch consolidation: {len(group_uuids)} memories -> {new_memory_id}"
                            )
                        except (ValueError, RuntimeError):
                            logger.exception("Consolidation failed for group in %s", custom_id)

                # Delete batch record
                self.db.delete_batch("post_processing", batch.id, user_id=batch.user_id)
                logger.info(
                    f"Consolidation batch {batch_id}: "
                    f"{memories_consolidated} memories consolidated, {memories_rejected} rejected"
                )
                return True
            finally:
                clear_user_context()

        except Exception as e:
            logger.exception("Error processing consolidation result")
            self.db.update_batch_status("post_processing",
                batch.id,
                "failed",
                error_message=str(e),
                user_id=batch.user_id
            )
            return False


class EntityGCBatchResultHandler(BatchResultProcessor):
    """
    Handle entity GC batch results: parse XML decisions, execute merge/delete/keep.

    Delegates parsing and execution to EntityGCService (shared with synchronous path).
    """

    def __init__(
        self,
        anthropic_client: anthropic.Anthropic,
        db: LTMemoryDB,
        entity_gc_service: 'EntityGCService',
    ):
        self.anthropic_client = anthropic_client
        self.db = db
        self.entity_gc = entity_gc_service

    def process_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """
        Process entity GC batch result.

        Iterates all results in the Anthropic batch, resolves short IDs from
        input_data, parses XML via EntityGCService, and executes decisions.
        """
        try:
            total_stats = {"merged": 0, "deleted": 0, "kept": 0, "errors": 0}

            for result in self.anthropic_client.beta.messages.batches.results(batch_id):
                if result.result.type != "succeeded":
                    logger.warning(
                        f"Entity GC result {result.custom_id} failed: "
                        f"{result.result.type}"
                    )
                    continue

                # Extract text from response
                text_blocks = [
                    block.text for block in result.result.message.content
                    if block.type == "text" and block.text
                ]
                if not text_blocks:
                    logger.warning(f"Entity GC result {result.custom_id} had no text content")
                    continue

                response_text = "\n".join(text_blocks)

                # Extract batch_idx from custom_id: {user_id}_entitygc_{batch_idx}
                batch_idx = (
                    result.custom_id.split("_entitygc_", 1)[-1]
                    if "_entitygc_" in result.custom_id
                    else None
                )
                batch_data = batch.input_data.get(batch_idx) if batch_idx else None
                if not batch_data:
                    logger.warning(f"No input_data for custom_id {result.custom_id}")
                    continue

                # Rebuild short_to_full UUID mapping from stored strings
                short_to_full = {
                    short_id: UUID(full_str)
                    for short_id, full_str in batch_data["short_to_full"].items()
                }

                # Parse XML and execute decisions
                parsed_groups = self.entity_gc.parse_gc_response(
                    response_text, short_to_full
                )

                set_current_user_id(batch.user_id)
                try:
                    batch_stats = self.entity_gc.execute_gc_decisions(
                        parsed_groups, str(batch.user_id)
                    )
                finally:
                    clear_user_context()

                for key in total_stats:
                    total_stats[key] += batch_stats[key]

            # Delete batch record
            self.db.delete_batch("post_processing", batch.id, user_id=batch.user_id)
            logger.info(
                f"Entity GC batch {batch_id}: "
                f"{total_stats['merged']} merged, {total_stats['deleted']} deleted, "
                f"{total_stats['kept']} kept, {total_stats['errors']} errors"
            )
            return True

        except Exception as e:
            logger.exception("Error processing entity GC result")
            self.db.update_batch_status("post_processing",
                batch.id,
                "failed",
                error_message=str(e),
                user_id=batch.user_id
            )
            return False


class PostProcessingBatchDispatcher(BatchResultProcessor):
    """
    Routes post-processing batches to appropriate handlers based on batch_type.

    Implements the BatchResultProcessor interface, allowing it to be used with
    the existing poll_batches() infrastructure in BatchCoordinator.

    Usage:
        dispatcher = PostProcessingBatchDispatcher()
        dispatcher.register('relationship_classification', relationship_handler)
        dispatcher.register('consolidation', consolidation_handler)

        # Pass to batch coordinator
        batch_coordinator.poll_post_processing_batches(result_processor=dispatcher)
    """

    def __init__(self):
        self._handlers: Dict[str, BatchResultProcessor] = {}

    def register(self, batch_type: str, handler: BatchResultProcessor) -> None:
        """
        Register a handler for a specific batch_type.

        Args:
            batch_type: The batch type this handler processes
            handler: Handler implementing BatchResultProcessor
        """
        self._handlers[batch_type] = handler
        logger.debug(f"Registered handler for batch_type: {batch_type}")

    def process_result(self, batch_id: str, batch: PostProcessingBatch) -> bool:
        """
        Route batch to appropriate handler based on batch_type.

        Args:
            batch_id: Anthropic batch ID
            batch: PostProcessingBatch record from database

        Returns:
            True if processing succeeded, False otherwise
        """
        handler = self._handlers.get(batch.batch_type)

        if not handler:
            logger.error(
                f"No handler registered for batch_type: {batch.batch_type}. "
                f"Available types: {list(self._handlers.keys())}"
            )
            return False

        logger.info(f"Processing {batch.batch_type} batch {batch_id}")
        return handler.process_result(batch_id, batch)
