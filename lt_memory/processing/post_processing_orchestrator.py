"""
Post-processing orchestrator - consolidation batch submission.

Thin coordinator that delegates to:
- RefinementService: Identify consolidation clusters, build payloads
- BatchCoordinator: Submit to Anthropic Batch API
- ConsolidationHandler: Execute immediate-mode consolidation with link transfer
- LTMemoryDB: Batch record storage, memory queries
"""
import json
import logging
from datetime import timedelta
from typing import Optional
from uuid import UUID, uuid4

from json_repair import repair_json

from clients.llm_provider import LLMProvider, build_batch_params
from lt_memory.db_access import LTMemoryDB
from lt_memory.llm_routing import uses_anthropic_batch_dialect
from lt_memory.models import PostProcessingBatch, ConsolidationCluster
from lt_memory.processing.batch_coordinator import BatchCoordinator, BATCH_EXPIRY_HOURS
from lt_memory.processing.consolidation_handler import ConsolidationHandler
from lt_memory.refinement import RefinementService
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class PostProcessingOrchestrator:
    """
    Orchestrate consolidation batch submission.

    Single Responsibility: Submit consolidation work via batch API
    or execute immediately when the selected dialect cannot use Anthropic batch.

    Parallel to ExtractionOrchestrator (which handles extraction submission).
    """

    def __init__(
        self,
        refinement: RefinementService,
        batch_coordinator: BatchCoordinator,
        consolidation_handler: ConsolidationHandler,
        db: LTMemoryDB,
        llm_provider: LLMProvider
    ):
        self.refinement = refinement
        self.batch_coordinator = batch_coordinator
        self.consolidation_handler = consolidation_handler
        self.db = db
        self.llm_provider = llm_provider

    def submit_consolidation_batch(self, user_id: str) -> Optional[str]:
        """
        Identify consolidation clusters and submit for processing.

        In batch mode: builds requests and submits to Anthropic Batch API.
        In immediate mode: executes consolidation via LLM + ConsolidationHandler.

        Args:
            user_id: User ID

        Returns:
            Batch ID if submitted, synthetic ID if immediate, None if no clusters found

        Raises:
            Exception: If Anthropic API call or LLM call fails
        """
        clusters = self.refinement.identify_consolidation_clusters()

        if not clusters:
            logger.info(f"No consolidation clusters found for user {user_id}")
            return None

        if not uses_anthropic_batch_dialect("consolidation"):
            logger.warning(
                f"Bypassing consolidation batch for user {user_id} - "
                f"executing {len(clusters)} clusters immediately"
            )
            self._execute_consolidation_immediately(user_id, clusters)
            return f"bypass_{uuid4()}"

        # Batch mode: submit to Anthropic
        requests = []
        input_data = {}

        for cluster in clusters:
            payload = self.refinement.build_consolidation_payload(cluster)

            custom_id = f"{user_id}_consol_{cluster.cluster_id}"

            params = build_batch_params(
                'consolidation',
                system_prompt=payload["system_prompt"],
                messages=[{"role": "user", "content": payload["user_prompt"]}],
            )
            requests.append({"custom_id": custom_id, "params": params})

            input_data[cluster.cluster_id] = {
                "cluster_id": cluster.cluster_id,
                "memory_ids": payload["memory_ids"]
            }

        # Submit to Anthropic
        batch_id = self.batch_coordinator.submit_batch(
            requests=requests,
            batch_type="consolidation",
            user_id=user_id,
        )

        # Store batch record for polling
        expires_at = utc_now() + timedelta(hours=BATCH_EXPIRY_HOURS)
        batch_record = PostProcessingBatch(
            batch_id=batch_id,
            batch_type="consolidation",
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

        logger.info(f"Submitted consolidation batch {batch_id}: {len(requests)} clusters")
        return batch_id

    # ============================================================================
    # Immediate Mode (Failover)
    # ============================================================================

    def _execute_consolidation_immediately(
        self,
        user_id: str,
        clusters: list[ConsolidationCluster]
    ) -> None:
        """
        Execute consolidation immediately when Anthropic failover is active.

        Parses multi-group JSON format from LLM, delegates merge execution to
        ConsolidationHandler, and tracks rejections for independent memories.

        Args:
            user_id: User ID
            clusters: List of ConsolidationCluster objects
        """
        from utils.tag_parser import format_memory_id

        memories_consolidated = 0
        memories_rejected = 0

        for cluster in clusters:
            payload = self.refinement.build_consolidation_payload(cluster)

            # Build short-to-full UUID mapping for resolving LLM output
            short_to_full = {}
            for full_id in payload["memory_ids"]:
                short_id = format_memory_id(full_id)
                short_to_full[short_id] = UUID(full_id)

            # Call LLM directly
            response = self.llm_provider.generate_response(
                messages=[{"role": "user", "content": payload["user_prompt"]}],
                system_prompt=payload["system_prompt"],
                internal_llm='consolidation',
                allow_negative=True
            )

            response_text = self.llm_provider.extract_text_content(response)

            try:
                decision = json.loads(response_text)
            except json.JSONDecodeError:
                try:
                    decision = json.loads(repair_json(response_text))
                except Exception:
                    logger.error(f"Invalid JSON in consolidation response for cluster {cluster.cluster_id}")
                    continue

            # Resolve independent IDs first — independence wins over merge
            independent_uuids = set()
            for short_id in decision.get("independent_ids", []):
                full_uuid = short_to_full.get(short_id)
                if full_uuid:
                    independent_uuids.add(full_uuid)
                    self.db.increment_consolidation_rejection_count(full_uuid, user_id=user_id)
                    memories_rejected += 1

            # Process merge groups, excluding any memory marked independent
            for group in decision.get("merge_groups", []):
                merged_text = group.get("merged_text", "").strip()
                if not merged_text:
                    logger.warning(f"Merge group in {cluster.cluster_id} has no text, skipping")
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
                        user_id=user_id,
                        merge_note=group.get("reason"),
                    )
                    memories_consolidated += len(group_uuids)
                    logger.info(
                        f"Immediate consolidation: {len(group_uuids)} memories -> {new_memory_id}"
                    )
                except (ValueError, RuntimeError) as e:
                    logger.error(f"Consolidation failed for cluster {cluster.cluster_id}: {e}")

        logger.info(
            f"Immediate consolidation complete for user {user_id}: "
            f"{memories_consolidated} memories consolidated, {memories_rejected} rejected"
        )

    def cleanup(self) -> None:
        """No-op: dependencies managed by factory lifecycle."""
        logger.debug("PostProcessingOrchestrator cleanup completed (no-op)")
