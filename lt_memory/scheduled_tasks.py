"""
Scheduled task registration for LT_Memory system.

Registers all periodic jobs with APScheduler for memory extraction,
batch processing, and refinement operations.
"""
import logging
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


def register_lt_memory_jobs(scheduler_service, lt_memory_factory) -> bool:
    """
    Register all LT_Memory scheduled jobs with the scheduler.

    Jobs registered:
    - Daily extraction (6-hour intervals)
    - Extraction batch polling (1-minute intervals)
    - Relationship batch polling (1-minute intervals)
    - Memory refinement (7-day intervals)

    Args:
        scheduler_service: System scheduler service instance
        lt_memory_factory: LTMemoryFactory instance with all services

    Returns:
        True if all jobs registered successfully

    Raises:
        Exception: If job registration fails (job registration is critical for system operation)
    """
    # Get services from factory
    batching = lt_memory_factory.batching
    refinement = lt_memory_factory.refinement

    # Register failed extraction retry job (6-hour intervals)
    success_extraction = scheduler_service.register_job(
        job_id="lt_memory_retry_failed_extractions",
        func=batching.retry_failed_extractions,
        trigger=IntervalTrigger(hours=6),
        component="lt_memory",
        description="Retry failed segment extractions every 6 hours (safety net)"
    )

    if not success_extraction:
        raise RuntimeError("Failed to register failed extraction retry job - scheduler job registration is critical")
    logger.info("Successfully registered failed extraction retry job (6-hour interval)")

    # Register extraction batch polling job (1-minute intervals)
    success_extraction_poll = scheduler_service.register_job(
        job_id="lt_memory_extraction_batch_polling",
        func=batching.poll_extraction_batches,
        trigger=IntervalTrigger(minutes=1),
        component="lt_memory",
        description="Poll Anthropic Batch API for extraction results every 1 minute"
    )

    if not success_extraction_poll:
        raise RuntimeError("Failed to register extraction batch polling job - scheduler job registration is critical")
    logger.info("Successfully registered extraction batch polling job (1-minute interval)")

    # Register relationship batch polling job (1-minute intervals)
    success_linking_poll = scheduler_service.register_job(
        job_id="lt_memory_linking_batch_polling",
        func=batching.poll_linking_batches,
        trigger=IntervalTrigger(minutes=1),
        component="lt_memory",
        description="Poll Anthropic Batch API for relationship results every 1 minute"
    )

    if not success_linking_poll:
        raise RuntimeError("Failed to register linking batch polling job - scheduler job registration is critical")
    logger.info("Successfully registered linking batch polling job (1-minute interval)")

    # Register refinement job (7-day intervals)
    success_refinement = scheduler_service.register_job(
        job_id="lt_memory_refinement",
        func=refinement.run_full_refinement,
        trigger=IntervalTrigger(days=7),
        component="lt_memory",
        description="Run memory refinement (verbose trimming) every 7 days"
    )

    if not success_refinement:
        raise RuntimeError("Failed to register memory refinement job - scheduler job registration is critical")
    logger.info("Successfully registered memory refinement job (7-day interval)")

    # Register consolidation job (7-day intervals)
    def run_consolidation_for_all_users():
        """Run consolidation for all users with consolidation enabled."""
        try:
            from auth.database import AuthDatabase
            from utils.user_context import set_current_user_id, clear_user_context
            auth_db = AuthDatabase()
            users = auth_db.get_users_with_memory_enabled()

            total_submitted = 0
            for user in users:
                user_id = str(user["id"])
                set_current_user_id(user_id)
                try:
                    batch_id = batching.submit_consolidation_batch(user_id)
                    if batch_id:
                        total_submitted += 1
                finally:
                    clear_user_context()

            logger.info(f"Consolidation sweep: submitted batches for {total_submitted} users")
            return {"users_processed": total_submitted}
        except Exception as e:
            logger.error(f"Error in consolidation sweep: {e}", exc_info=True)
            return {"error": str(e)}

    success_consolidation = scheduler_service.register_job(
        job_id="lt_memory_consolidation",
        func=run_consolidation_for_all_users,
        trigger=IntervalTrigger(days=7),
        component="lt_memory",
        description="Submit consolidation batches for all users every 7 days"
    )

    if not success_consolidation:
        raise RuntimeError("Failed to register consolidation job - scheduler job registration is critical")
    logger.info("Successfully registered consolidation job (7-day interval)")

    # Register temporal score recalculation job (daily intervals)
    def run_temporal_score_recalculation():
        """Recalculate scores for temporal memories across all users."""
        try:
            from auth.database import AuthDatabase
            from utils.user_context import set_current_user_id, clear_user_context
            auth_db = AuthDatabase()
            users = auth_db.get_users_with_memory_enabled()

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

            logger.info(f"Temporal score sweep: updated {total_updated} memories across all users")
            return {"memories_updated": total_updated}
        except Exception as e:
            logger.error(f"Error in temporal score recalculation: {e}", exc_info=True)
            return {"error": str(e)}

    success_temporal = scheduler_service.register_job(
        job_id="lt_memory_temporal_score_recalculation",
        func=run_temporal_score_recalculation,
        trigger=IntervalTrigger(days=1),
        component="lt_memory",
        description="Recalculate scores for temporal memories (happens_at/expires_at) daily"
    )

    if not success_temporal:
        raise RuntimeError("Failed to register temporal score recalculation job - scheduler job registration is critical")
    logger.info("Successfully registered temporal score recalculation job (daily interval)")

    # Register entity garbage collection job (monthly intervals)
    def run_entity_gc_for_all_users():
        """Run entity garbage collection for all users."""
        try:
            from auth.database import AuthDatabase
            from utils.user_context import set_current_user_id, clear_user_context
            auth_db = AuthDatabase()
            users = auth_db.get_users_with_memory_enabled()

            total_merged = 0
            total_deleted = 0
            total_kept = 0

            for user in users:
                user_id = str(user["id"])
                set_current_user_id(user_id)
                try:
                    entity_gc = lt_memory_factory.entity_gc
                    results = entity_gc.run_entity_gc_for_user()
                    total_merged += results.get("merged", 0)
                    total_deleted += results.get("deleted", 0)
                    total_kept += results.get("kept", 0)
                finally:
                    clear_user_context()

            logger.info(
                f"Entity GC sweep: {total_merged} merged, {total_deleted} deleted, "
                f"{total_kept} kept across {len(users)} users"
            )
            return {"merged": total_merged, "deleted": total_deleted, "kept": total_kept}
        except Exception as e:
            logger.error(f"Error in entity GC sweep: {e}", exc_info=True)
            return {"error": str(e)}

    success_entity_gc = scheduler_service.register_job(
        job_id="lt_memory_entity_gc",
        func=run_entity_gc_for_all_users,
        trigger=IntervalTrigger(days=30),  # Monthly
        component="lt_memory",
        description="Run entity garbage collection (merge/delete dormant entities) monthly"
    )

    if not success_entity_gc:
        raise RuntimeError("Failed to register entity GC job - scheduler job registration is critical")
    logger.info("Successfully registered entity GC job (monthly interval)")

    logger.info("All LT_Memory scheduled jobs registered successfully")
    return True
