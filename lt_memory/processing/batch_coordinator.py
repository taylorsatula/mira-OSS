"""
Batch coordinator - generic Anthropic Batch API orchestration.

Eliminates duplication between extraction and relationship batch polling
(poll_extraction_batches and poll_linking_batches were 90% identical).

Provides generic polling infrastructure with pluggable result processors.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import anthropic

from lt_memory.db_access import LTMemoryDB
from lt_memory.models import ExtractionBatch, PostProcessingBatch
from config.config import BatchingConfig
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)


class BatchResultProcessor(ABC):
    """
    Abstract base for batch result processors.

    Concrete implementations handle specific batch types:
    - ExtractionResultProcessor: Process memory extraction results
    - RelationshipResultProcessor: Process relationship classification results
    - ConsolidationResultProcessor: Process consolidation results
    """

    @abstractmethod
    def process_result(self, batch_id: str, batch: Any) -> bool:
        """
        Process completed batch result.

        Args:
            batch_id: Anthropic batch ID
            batch: Batch record from database

        Returns:
            True if processing succeeded, False otherwise
        """
        pass


class BatchCoordinator:
    """
    Generic Anthropic Batch API coordinator.

    Single Responsibility: Orchestrate batch submission, polling, and result processing

    Handles the generic batch lifecycle:
    1. Submit batches to Anthropic
    2. Poll for completion
    3. Handle expiry, retries, failures
    4. Delegate result processing to specialized processors

    Eliminates 90% duplication between extraction and relationship polling.
    """

    def __init__(
        self,
        config: BatchingConfig,
        db: LTMemoryDB,
        anthropic_client: anthropic.Anthropic
    ):
        """
        Initialize batch coordinator.

        Args:
            config: Batching configuration
            db: Database access for batch tracking
            anthropic_client: Anthropic client for API calls
        """
        self.config = config
        self.db = db
        self.anthropic_client = anthropic_client

    def submit_batch(
        self,
        requests: List[Dict[str, Any]],
        batch_type: str,
        user_id: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit batch to Anthropic API.

        Generic submission that works for any batch type
        (extraction, relationship, consolidation, etc.).

        Args:
            requests: List of Anthropic batch request dicts
            batch_type: Type of batch ("extraction", "relationship_classification", etc.)
            user_id: User ID
            input_data: Optional input data to store with batch

        Returns:
            Batch ID from Anthropic

        Raises:
            Exception: If Anthropic API call fails
        """
        if not requests:
            raise ValueError("Cannot submit empty batch")

        # Submit to Anthropic
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)
        # Always 24 hours (Anthropic API limit)
        expires_at = utc_now() + timedelta(hours=24)

        logger.info(
            f"Submitted {batch_type} batch {batch.id} for user {user_id}: "
            f"{len(requests)} requests"
        )

        return batch.id

    def poll_batches(
        self,
        batch_type: str,
        get_pending_batches_fn: Callable[[], List[Any]],
        result_processor: BatchResultProcessor,
        update_status_fn: Callable[..., None],
        increment_retry_fn: Callable[[Any, str], None],
        delete_batch_fn: Callable[[Any, str], None]
    ) -> Dict[str, int]:
        """
        Generic batch polling loop.

        Polls Anthropic for batch completion and delegates result processing.

        This method eliminates the duplication between poll_extraction_batches
        and poll_linking_batches (which were 90% identical).

        Args:
            batch_type: Type of batch being polled
            get_pending_batches_fn: Function to get pending batches from DB
            result_processor: Processor for completed batch results
            update_status_fn: Function to update batch status.
                Called as: update_status_fn(batch_id, status, error_message=..., user_id=...)
            increment_retry_fn: Function to increment retry count (batch_id, user_id)
            delete_batch_fn: Function to delete batch after success (batch_id, user_id)

        Returns:
            Stats dict (checked, completed, failed, expired)
        """
        stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        # Get pending batches
        pending_batches = get_pending_batches_fn()

        if not pending_batches:
            return stats

        logger.info(f"Polling {batch_type} batches: {len(pending_batches)} pending")

        # Filter out batches older than max age (Anthropic results expire after 24h)
        batch_age_cutoff = utc_now() - timedelta(hours=self.config.batch_max_age_hours)
        fresh_batches = []
        for batch in pending_batches:
            if batch.created_at < batch_age_cutoff:
                # Mark as expired - results no longer available
                update_status_fn(
                    batch.id,
                    "expired",
                    error_message=f"Batch older than {self.config.batch_max_age_hours}h - Anthropic results expired",
                    user_id=batch.user_id
                )
                stats["expired"] += 1
                logger.warning(f"Expired old batch {batch.batch_id} (age: {utc_now() - batch.created_at})")
            else:
                fresh_batches.append(batch)

        if not fresh_batches:
            logger.info(f"All {len(pending_batches)} pending batches were expired")
            return stats

        # Group by batch_id to minimize API calls
        batch_groups = {}
        for batch in fresh_batches:
            if batch.batch_id not in batch_groups:
                batch_groups[batch.batch_id] = []
            batch_groups[batch.batch_id].append(batch)

        for batch_id, batches in batch_groups.items():
            stats["checked"] += len(batches)

            # Check expiration
            if batches[0].expires_at and utc_now() > batches[0].expires_at:
                for b in batches:
                    update_status_fn(b.id, "expired", error_message=None, user_id=b.user_id)
                stats["expired"] += len(batches)
                continue

            # Query Anthropic (only retry transient API errors)
            try:
                batch_info = self.anthropic_client.beta.messages.batches.retrieve(batch_id)
            except (anthropic.APIError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                # Transient API errors - retry with backoff
                logger.warning(f"Transient API error retrieving batch {batch_id}: {e}")
                for b in batches:
                    # Increment retry counter
                    increment_retry_fn(b.id, b.user_id)

                    # Fail permanently after max retries
                    if b.retry_count + 1 >= self.config.max_retry_count:
                        update_status_fn(
                            b.id,
                            "failed",
                            error_message=f"Failed after {self.config.max_retry_count} retries: {str(e)}",
                            user_id=b.user_id
                        )
                        stats["failed"] += 1
                continue
            except Exception as e:
                # Programming errors (AttributeError, TypeError, etc.) should propagate
                logger.error(
                    f"Unexpected error retrieving batch {batch_id}: {e}. "
                    "This indicates a programming error, not a transient API issue.",
                    exc_info=True
                )
                raise

            # Handle batch status
            if batch_info.processing_status == "ended":
                for b in batches:
                    try:
                        # Process with timeout to prevent infinite hangs
                        # ThreadPoolExecutor provides thread-safe timeout (works in APScheduler background threads)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(result_processor.process_result, batch_id, b)
                            try:
                                if future.result(timeout=self.config.batch_processing_timeout_seconds):
                                    stats["completed"] += 1
                                    delete_batch_fn(b.id, b.user_id)
                                else:
                                    raise RuntimeError("Processing returned False")
                            except FuturesTimeoutError:
                                raise TimeoutError(
                                    f"Batch processing exceeded {self.config.batch_processing_timeout_seconds}s timeout"
                                )

                    except TimeoutError as e:
                        logger.error(f"Batch {b.id} processing timeout: {e}", exc_info=True)
                        update_status_fn(
                            b.id,
                            "failed",
                            error_message=str(e),
                            user_id=b.user_id
                        )
                        stats["failed"] += 1
                    except Exception as e:
                        logger.error(f"Error processing batch {b.id}: {e}", exc_info=True)

                        # Increment retry counter
                        increment_retry_fn(b.id, b.user_id)

                        # Fail permanently after max retries
                        if b.retry_count + 1 >= self.config.max_retry_count:
                            update_status_fn(
                                b.id,
                                "failed",
                                error_message=f"Failed after {self.config.max_retry_count} retries: {str(e)}",
                                user_id=b.user_id
                            )
                        stats["failed"] += 1

            elif batch_info.processing_status == "in_progress":
                for b in batches:
                    if b.status != "processing":
                        update_status_fn(b.id, "processing", error_message=None, user_id=b.user_id)

            elif batch_info.processing_status in ("canceling", "canceled"):
                for b in batches:
                    update_status_fn(b.id, "cancelled", error_message="Cancelled", user_id=b.user_id)
                stats["failed"] += len(batches)

        logger.info(
            f"{batch_type} polling: {stats['completed']} completed, "
            f"{stats['failed']} failed, {stats['expired']} expired"
        )
        return stats

    def poll_extraction_batches(
        self,
        result_processor: BatchResultProcessor
    ) -> Dict[str, int]:
        """
        Poll extraction batches (convenience wrapper).

        Args:
            result_processor: Processor for extraction results

        Returns:
            Polling statistics
        """
        # Get users with pending extraction batches
        users_with_pending = self.db.get_users_with_pending_extraction_batches()

        all_stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        for user_id in users_with_pending:
            stats = self.poll_batches(
                batch_type="extraction",
                get_pending_batches_fn=lambda: self.db.get_pending_extraction_batches_for_user(user_id),
                result_processor=result_processor,
                update_status_fn=self.db.update_extraction_batch_status,
                increment_retry_fn=self.db.increment_extraction_batch_retry,
                delete_batch_fn=self.db.delete_extraction_batch
            )

            # Aggregate stats
            for key in all_stats:
                all_stats[key] += stats[key]

        return all_stats

    def poll_relationship_batches(
        self,
        result_processor: BatchResultProcessor
    ) -> Dict[str, int]:
        """
        Poll relationship batches (convenience wrapper).

        Args:
            result_processor: Processor for relationship results

        Returns:
            Polling statistics
        """
        # Get users with pending relationship batches
        users_with_pending = self.db.get_users_with_pending_relationship_batches()

        all_stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        for user_id in users_with_pending:
            stats = self.poll_batches(
                batch_type="relationship",
                get_pending_batches_fn=lambda: self.db.get_pending_relationship_batches_for_user(user_id),
                result_processor=result_processor,
                update_status_fn=self.db.update_relationship_batch_status,
                increment_retry_fn=self.db.increment_relationship_batch_retry,
                delete_batch_fn=self.db.delete_relationship_batch
            )

            # Aggregate stats
            for key in all_stats:
                all_stats[key] += stats[key]

        return all_stats
