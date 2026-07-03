"""
Scheduled task registration for LT_Memory system.

Registers all periodic jobs with APScheduler for memory extraction,
batch processing, and maintenance operations. Day-interval jobs use
modular arithmetic on cumulative_activity_days for stateless, use-day-based
scheduling via get_users_due_for_job().
"""
import logging
from apscheduler.triggers.interval import IntervalTrigger
from utils.scheduled_task_monitor import ScheduledTaskMonitor

logger = logging.getLogger(__name__)


def register_lt_memory_jobs(scheduler_service, lt_memory_factory) -> None:
    """
    Register all LT_Memory scheduled jobs with the scheduler.

    Jobs registered:
    - Extraction retry sweep (6-hour intervals, calendar-based)
    - Extraction batch polling (1-minute intervals, calendar-based)
    - Post-processing batch polling (1-minute intervals, calendar-based)
    - Consolidation (deadheaded pending redesign)
    - Temporal score recalculation (daily tick, use-day gated)
    - Bulk score recalculation (daily tick, use-day gated)
    - Entity garbage collection (deadheaded pending redesign)
    - Batch cleanup (daily tick, use-day gated)

    Args:
        scheduler_service: System scheduler service instance
        lt_memory_factory: LTMemoryFactory instance with all services

    Raises:
        RuntimeError: If job registration fails
    """
    from config import config
    extraction_orchestrator = lt_memory_factory.extraction_orchestrator
    batch_coordinator = lt_memory_factory.batch_coordinator
    post_processing = lt_memory_factory.post_processing_orchestrator
    jobs_config = config.scheduled_jobs

    # ================================================================
    # Calendar-based jobs (not use-day gated)
    # ================================================================

    # Extraction retry sweep (6-hour intervals)
    scheduler_service.register_job(
        job_id="lt_memory_extract_unprocessed_segments",
        func=extraction_orchestrator.extract_unprocessed_segments,
        trigger=IntervalTrigger(hours=jobs_config.extraction_retry_hours),
        component="lt_memory",
        description=f"Extract unprocessed collapsed segments every {jobs_config.extraction_retry_hours} hours (safety net)"
    )
    logger.info("Registered extraction retry sweep (%dh interval)", jobs_config.extraction_retry_hours)

    # Extraction batch polling (1-minute intervals)
    def poll_extraction_batches_with_handler():
        return batch_coordinator.poll_extraction_batches(
            result_processor=lt_memory_factory.extraction_result_handler
        )

    monitored_extraction_poll = ScheduledTaskMonitor.wrap_scheduled_job(
        job_id="lt_memory_extraction_batch_polling",
        func=poll_extraction_batches_with_handler,
        timeout_seconds=jobs_config.job_timeout_seconds,
        kill_on_timeout=True
    )

    scheduler_service.register_job(
        job_id="lt_memory_extraction_batch_polling",
        func=monitored_extraction_poll,
        trigger=IntervalTrigger(minutes=jobs_config.batch_poll_minutes),
        component="lt_memory",
        description=f"Poll Anthropic Batch API for extraction results every {jobs_config.batch_poll_minutes} minute(s)"
    )
    logger.info("Registered extraction batch polling (%dmin interval)", jobs_config.batch_poll_minutes)

    # Post-processing batch polling (1-minute intervals)
    def poll_post_processing_batches_with_handler():
        return batch_coordinator.poll_post_processing_batches(
            result_processor=lt_memory_factory.post_processing_dispatcher
        )

    monitored_post_processing_poll = ScheduledTaskMonitor.wrap_scheduled_job(
        job_id="lt_memory_post_processing_batch_polling",
        func=poll_post_processing_batches_with_handler,
        timeout_seconds=jobs_config.job_timeout_seconds,
        kill_on_timeout=True
    )

    scheduler_service.register_job(
        job_id="lt_memory_post_processing_batch_polling",
        func=monitored_post_processing_poll,
        trigger=IntervalTrigger(minutes=jobs_config.batch_poll_minutes),
        component="lt_memory",
        description=f"Poll Anthropic Batch API for post-processing results every {jobs_config.batch_poll_minutes} minute(s)"
    )
    logger.info("Registered post-processing batch polling (%dmin interval)", jobs_config.batch_poll_minutes)

    # ================================================================
    # Use-day-gated jobs (daily tick, filtered by modular arithmetic)
    # ================================================================

    # Consolidation
    def run_consolidation_for_due_users():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.consolidation_use_days)
        total_submitted = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                batch_id = post_processing.submit_consolidation_batch(user_id)
                if batch_id:
                    total_submitted += 1
            finally:
                clear_user_context()

        logger.info("Consolidation sweep: submitted batches for %d/%d due users", total_submitted, len(users))
        return {"users_processed": total_submitted}

    logger.info("Consolidation job is deadheaded pending redesign")

    # Temporal score recalculation
    def run_temporal_score_recalculation():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.temporal_score_recalc_use_days)
        total_updated = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                db = lt_memory_factory.db
                updated = db.recalculate_temporal_scores(user_id=user_id, batch_size=1000)
                total_updated += updated
            finally:
                clear_user_context()

        logger.info("Temporal score sweep: updated %d memories across %d due users", total_updated, len(users))
        return {"memories_updated": total_updated}

    scheduler_service.register_job(
        job_id="lt_memory_temporal_score_recalculation",
        func=run_temporal_score_recalculation,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description=f"Recalculate temporal memory scores (every {jobs_config.temporal_score_recalc_use_days} use-days)"
    )
    logger.info("Registered temporal score recalculation (every %d use-days)", jobs_config.temporal_score_recalc_use_days)

    # Bulk score recalculation
    def run_bulk_score_recalculation():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.bulk_score_recalc_use_days)
        total_updated = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                db = lt_memory_factory.db
                updated = db.bulk_recalculate_scores(user_id=user_id, batch_size=1000)
                total_updated += updated
            finally:
                clear_user_context()

        logger.info("Bulk score recalculation sweep: updated %d stale memories across %d due users", total_updated, len(users))
        return {"memories_updated": total_updated}

    scheduler_service.register_job(
        job_id="lt_memory_bulk_score_recalculation",
        func=run_bulk_score_recalculation,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description=f"Recalculate stale memory scores (every {jobs_config.bulk_score_recalc_use_days} use-days)"
    )
    logger.info("Registered bulk score recalculation (every %d use-days)", jobs_config.bulk_score_recalc_use_days)

    # Entity garbage collection
    def submit_entity_gc_for_due_users():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.entity_gc_use_days)
        total_submitted = 0
        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                entity_gc = lt_memory_factory.entity_gc
                batch_id = entity_gc.submit_entity_gc_batch()
                if batch_id:
                    total_submitted += 1
            finally:
                clear_user_context()

        logger.info("Entity GC sweep: submitted batches for %d/%d due users", total_submitted, len(users))
        return {"users_submitted": total_submitted}

    logger.info("Entity GC job is deadheaded pending redesign")

    # Batch cleanup
    def run_batch_cleanup_for_due_users():
        from utils.user_context import set_current_user_id, clear_user_context
        from utils.scheduled_tasks import get_users_due_for_job

        users = get_users_due_for_job(jobs_config.batch_cleanup_use_days)
        retention_hours = lt_memory_factory.config.batching.batch_max_age_hours

        total_extraction_deleted = 0
        total_relationship_deleted = 0

        for user in users:
            user_id = str(user["id"])
            set_current_user_id(user_id)
            try:
                db = lt_memory_factory.db
                extraction_deleted = db.cleanup_old_batches(
                    "extraction",
                    retention_hours=retention_hours,
                    user_id=user_id
                )
                relationship_deleted = db.cleanup_old_batches(
                    "post_processing",
                    retention_hours=retention_hours,
                    user_id=user_id
                )
                total_extraction_deleted += extraction_deleted
                total_relationship_deleted += relationship_deleted
            finally:
                clear_user_context()

        logger.info(
            "Batch cleanup: deleted %d extraction, %d relationship batches across %d due users (retention: %dh)",
            total_extraction_deleted, total_relationship_deleted, len(users), retention_hours
        )
        return {
            "extraction_deleted": total_extraction_deleted,
            "relationship_deleted": total_relationship_deleted
        }

    scheduler_service.register_job(
        job_id="lt_memory_batch_cleanup",
        func=run_batch_cleanup_for_due_users,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description=f"Clean up old failed/expired/cancelled batches (every {jobs_config.batch_cleanup_use_days} use-days)"
    )
    logger.info("Registered batch cleanup (every %d use-days)", jobs_config.batch_cleanup_use_days)

    logger.info("All LT_Memory scheduled jobs registered successfully")
