"""
Batch coordinator - Anthropic Batch API orchestration for extraction.

Provides generic polling infrastructure with pluggable result processors.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import anthropic

from lt_memory.db_access import LTMemoryDB
from lt_memory.models import ExtractionBatch
from utils.timezone_utils import utc_now

logger = logging.getLogger(__name__)

# Batch processing limits
BATCH_EXPIRY_HOURS = 24              # Hours before Anthropic batch expires
BATCH_MAX_AGE_HOURS = 48             # Max age for batches to poll (Anthropic results expire after 24h)
MAX_RETRY_COUNT = 3                  # Max retry attempts before permanent failure
MAX_BATCHES_PER_POLL = 3             # Max batch IDs to process per poll cycle
BATCH_PROCESSING_TIMEOUT_SECONDS = 300  # Max seconds per single batch result (5 minutes)


class BatchResultProcessor(ABC):
    """
    Abstract base for batch result processors.

    Concrete implementations handle specific batch types:
    - ExtractionResultProcessor: Process memory extraction results
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
    """

    def __init__(
        self,
        db: LTMemoryDB,
        anthropic_client: anthropic.Anthropic
    ):
        self.db = db
        self.anthropic_client = anthropic_client

    def submit_batch(
        self,
        requests: List[Dict[str, Any]],
        user_id: str,
    ) -> str:
        """
        Submit extraction batch to Anthropic API.

        Args:
            requests: List of Anthropic batch request dicts
            user_id: User ID

        Returns:
            Batch ID from Anthropic

        Raises:
            Exception: If Anthropic API call fails
        """
        if not requests:
            raise ValueError("Cannot submit empty batch")

        # Submit to Anthropic
        batch = self.anthropic_client.beta.messages.batches.create(requests=requests)

        logger.info(
            f"Submitted extraction batch {batch.id} for user {user_id}: "
            f"{len(requests)} requests"
        )

        return batch.id

    def poll_batches(
        self,
        get_pending_batches_fn: Callable[[], List[Any]],
        result_processor: BatchResultProcessor,
        update_status_fn: Callable[..., None],
        increment_retry_fn: Callable[[Any, str], None],
        delete_batch_fn: Callable[[Any, str], None],
        claim_batch_fn: Callable[[Any, str], bool] = None
    ) -> Dict[str, int]:
        """
        Extraction batch polling loop.

        Polls Anthropic for batch completion and delegates result processing.

        Args:
            get_pending_batches_fn: Function to get pending batches from DB
            result_processor: Processor for completed batch results
            update_status_fn: Function to update batch status.
                Called as: update_status_fn(batch_id, status, error_message=..., user_id=...)
            increment_retry_fn: Function to increment retry count (batch_id, user_id)
            delete_batch_fn: Function to delete batch after success (batch_id, user_id)
            claim_batch_fn: Atomically claim batch before processing (batch_id, user_id).
                Returns True if claimed, False if already claimed by another thread.

        Returns:
            Stats dict (checked, completed, failed, expired)
        """
        stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        # Get pending batches
        pending_batches = get_pending_batches_fn()

        if not pending_batches:
            return stats

        logger.info(f"Polling extraction batches: {len(pending_batches)} pending")

        # Filter out batches older than max age (Anthropic results expire after 24h)
        batch_age_cutoff = utc_now() - timedelta(hours=BATCH_MAX_AGE_HOURS)
        fresh_batches = []
        for batch in pending_batches:
            if batch.created_at < batch_age_cutoff:
                # Mark as expired - results no longer available
                update_status_fn(
                    batch.id,
                    "expired",
                    error_message=f"Batch older than {BATCH_MAX_AGE_HOURS}h - Anthropic results expired",
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

        groups_processed = 0
        for batch_id, batches in batch_groups.items():
            if groups_processed >= MAX_BATCHES_PER_POLL:
                logger.info(
                    f"Reached per-cycle limit ({MAX_BATCHES_PER_POLL}), "
                    f"deferring {len(batch_groups) - groups_processed} remaining batch groups"
                )
                break
            groups_processed += 1

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
                    if b.retry_count + 1 >= MAX_RETRY_COUNT:
                        update_status_fn(
                            b.id,
                            "failed",
                            error_message=f"Failed after {MAX_RETRY_COUNT} retries: {str(e)}",
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
                    # Atomically claim batch before processing to prevent
                    # concurrent threads from processing the same batch
                    # (happens when poll timeout fires but thread continues)
                    if claim_batch_fn and not claim_batch_fn(b.id, b.user_id):
                        logger.debug(f"Batch {b.id} already claimed by another thread, skipping")
                        continue

                    try:
                        # Process with timeout to prevent infinite hangs
                        # CRITICAL: Don't use 'with' context manager - it calls
                        # shutdown(wait=True) which blocks forever if thread is stuck
                        executor = ThreadPoolExecutor(max_workers=1)
                        future = executor.submit(result_processor.process_result, batch_id, b)
                        try:
                            if future.result(timeout=BATCH_PROCESSING_TIMEOUT_SECONDS):
                                stats["completed"] += 1
                                delete_batch_fn(b.id, b.user_id)
                            else:
                                raise RuntimeError("Processing returned False")
                        except FuturesTimeoutError:
                            raise TimeoutError(
                                f"Batch processing exceeded {BATCH_PROCESSING_TIMEOUT_SECONDS}s timeout"
                            )
                        finally:
                            # Shutdown WITHOUT waiting - don't block if thread is stuck
                            executor.shutdown(wait=False, cancel_futures=True)

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
                        if b.retry_count + 1 >= MAX_RETRY_COUNT:
                            update_status_fn(
                                b.id,
                                "failed",
                                error_message=f"Failed after {MAX_RETRY_COUNT} retries: {str(e)}",
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
            f"Extraction polling: {stats['completed']} completed, "
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
        users_with_pending = self.db.get_users_with_pending_batches()

        all_stats = {"checked": 0, "completed": 0, "failed": 0, "expired": 0}

        for user_id in users_with_pending:
            stats = self.poll_batches(
                get_pending_batches_fn=lambda uid=user_id: self.db.get_pending_batches_for_user(uid),
                result_processor=result_processor,
                update_status_fn=lambda bid, status, **kw: self.db.update_batch_status(bid, status, **kw),
                increment_retry_fn=lambda bid, uid: self.db.increment_batch_retry(bid, uid),
                delete_batch_fn=lambda bid, uid: self.db.delete_batch(bid, uid),
                claim_batch_fn=lambda bid, uid: self.db.claim_batch(bid, uid)
            )

            for key in all_stats:
                all_stats[key] += stats[key]

        return all_stats
