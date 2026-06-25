"""
Health API endpoint - basic system health checks.

Simple health monitoring for database, basic system status.
"""
import logging
import secrets
import time

import psycopg

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse

from .base import BaseHandler, ErrorResponse, SuccessResponse, create_success_response, create_error_response
from clients.postgres_client import PostgresClient
from utils.timezone_utils import utc_now, format_utc_iso
from utils.thread_monitor import ThreadMonitor
from utils.scheduled_task_monitor import ScheduledTaskMonitor

logger = logging.getLogger(__name__)

router = APIRouter()

DIAGNOSTICS_TOKEN_HEADER = "X-MIRA-Diagnostics-Token"


class HealthEndpoint(BaseHandler):
    """Health endpoint handler with basic system checks."""

    def process_request(self, **params) -> SuccessResponse | ErrorResponse:
        """Check system health components."""
        start_time = time.time()
        components: dict[str, dict] = {}
        overall_status = "healthy"

        # Check database connectivity (unified mira_service database)
        try:
            db = PostgresClient("mira_service")
            db.execute_single("SELECT 1")
            components["database"] = {"status": "healthy", "latency_ms": round((time.time() - start_time) * 1000, 1)}
        except (psycopg.OperationalError, psycopg.DatabaseError) as e:
            components["database"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"

        # Basic system info
        components["system"] = {
            "status": "healthy",
            "uptime_seconds": int(time.time()),  # Placeholder - actual uptime would need process start tracking
            "version": "1.0.0"
        }

        total_time = round((time.time() - start_time) * 1000, 1)

        health_data: dict = {
            "status": overall_status,
            "timestamp": format_utc_iso(utc_now()),
            "components": components,
            "meta": {
                "check_duration_ms": total_time,
                "checks_run": len(components)
            }
        }

        # Return appropriate response based on health status
        if overall_status == "unhealthy":
            return ErrorResponse(
                error={
                    "code": "SYSTEM_UNHEALTHY",
                    "message": "One or more system components are unhealthy",
                    "details": health_data
                },
                meta={}
            )

        return create_success_response(health_data)


def get_health_handler() -> HealthEndpoint:
    """Get health endpoint handler instance."""
    return HealthEndpoint()


@router.get("/health")
def health_endpoint():
    """System health check endpoint (no authentication required)."""
    try:
        handler = get_health_handler()
        response = handler.handle_request()

        if isinstance(response, ErrorResponse):
            return JSONResponse(status_code=503, content=response.to_dict())

        return response.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health endpoint error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content=create_error_response(e).to_dict())


def _require_diagnostics_token(x_mira_diagnostics_token: str | None = Header(default=None)) -> None:
    """Require a Vault-backed diagnostics token for detailed health endpoints."""
    from clients.vault_client import get_service_config

    try:
        expected_token = get_service_config("diagnostics_token")
    except Exception as e:
        logger.error(f"Diagnostics token is not configured: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Diagnostics endpoint not configured")

    if not x_mira_diagnostics_token:
        raise HTTPException(status_code=401, detail=f"Missing {DIAGNOSTICS_TOKEN_HEADER} header")
    if not secrets.compare_digest(x_mira_diagnostics_token, expected_token):
        raise HTTPException(status_code=403, detail="Invalid diagnostics token")


@router.get("/health/threads")
async def thread_health_endpoint(_: None = Depends(_require_diagnostics_token)):
    """
    Thread monitoring endpoint - shows active and stuck operations.

    Returns information about thread pool usage and potentially stuck operations.
    """
    try:
        from anyio import to_thread
        limiter = to_thread.current_default_thread_limiter()

        # Get thread pool stats
        available_threads = limiter.available_tokens
        total_threads = limiter.total_tokens
        used_threads = total_threads - available_threads
        usage_percent = (used_threads / total_threads) * 100 if total_threads > 0 else 0

        # Get stuck operations
        stuck_ops = ThreadMonitor.get_stuck_operations()
        active_ops = ThreadMonitor.get_active_operations()

        # Get scheduled job stats
        job_stats = ScheduledTaskMonitor.get_job_stats()
        from utils.scheduler_service import scheduler_service
        scheduler_jobs = scheduler_service.scheduler.get_jobs() if scheduler_service._running else []
        scheduler_state = {
            "running": scheduler_service._running,
            "registered_jobs": sorted(scheduler_service._registered_jobs.keys()),
            "apscheduler_jobs": sorted(job.id for job in scheduler_jobs),
        }

        # Determine health status
        status = "healthy"
        if usage_percent > 90:
            status = "critical"
        elif usage_percent > 70:
            status = "warning"
        if stuck_ops:
            status = "critical"

        data = {
            "status": status,
            "thread_pool": {
                "available": available_threads,
                "total": total_threads,
                "used": used_threads,
                "usage_percent": round(usage_percent, 1)
            },
            "operations": {
                "active": len(active_ops),
                "stuck": len(stuck_ops),
                "active_operations": active_ops[:10],
                "stuck_operations": stuck_ops
            },
            "scheduled_jobs": job_stats,
            "scheduler": scheduler_state,
            "timestamp": format_utc_iso(utc_now())
        }

        response = create_success_response(data)

        # Return 503 if critical
        if status == "critical":
            return JSONResponse(status_code=503, content=response.to_dict())

        return response.to_dict()

    except Exception as e:
        logger.error(f"Thread health endpoint error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content=create_error_response(e).to_dict())


@router.get("/health/thread-dump")
def thread_dump_endpoint(_: None = Depends(_require_diagnostics_token)):
    """
    Generate a detailed thread dump for debugging.

    Returns complete thread state information for all threads.
    """
    try:
        dump = ThreadMonitor.dump_thread_states()

        # Also save to file for later analysis
        import time
        dump_file = f"/tmp/thread_dump_api_{int(time.time())}.txt"
        with open(dump_file, 'w') as f:
            f.write(dump)

        return create_success_response({
            "thread_dump": dump,
            "saved_to": dump_file,
            "timestamp": format_utc_iso(utc_now())
        }).to_dict()

    except Exception as e:
        logger.error(f"Thread dump endpoint error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content=create_error_response(e).to_dict())
