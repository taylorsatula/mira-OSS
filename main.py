#!/usr/bin/env python3
"""
MIRA - Main Application Entry Point
FastAPI server that wires together the CNS architecture and handles startup/shutdown.
"""

import argparse
import asyncio
import logging
import sys
import os
import signal
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from config.config_manager import config
from cns.api import data, actions, health
from cns.api import chat as chat_api
from cns.api.base import APIError, create_error_response, generate_request_id
from utils.scheduler_service import scheduler_service
from utils.scheduled_tasks import initialize_all_scheduled_tasks
from utils.colored_logging import setup_colored_root_logging

setup_colored_root_logging(log_level=logging.INFO, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set APScheduler loggers to DEBUG to suppress routine job execution logs
logging.getLogger('apscheduler.executors.default').setLevel(logging.DEBUG)
logging.getLogger('apscheduler.scheduler').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)


def ensure_single_user(app: FastAPI) -> str:
    """
    Ensure exactly one user exists in the database and set global user context.

    Creates a default user if none exist, returns the user ID if exactly one exists,
    and exits with error if multiple users exist (single-user mode violation).

    Args:
        app: FastAPI application instance

    Returns:
        User ID of the single user

    Raises:
        SystemExit: If multiple users exist
    """
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id

    db = PostgresClient('mira_service')

    # Count users
    result = db.execute_single("SELECT COUNT(*) as count FROM users")
    user_count = result['count']

    if user_count == 0:
        print("\n" + "="*60)
        print("🚀 MIRA Single-User Setup")
        print("="*60)
        print("No user found. Creating default user...")

        # Create default user
        user_result = db.execute_single("""
            INSERT INTO users (email, is_active, created_at, memory_enabled)
            VALUES ('user@localhost', true, NOW(), true)
            RETURNING id, email
        """)
        user_id = str(user_result['id'])
        email = user_result['email']

        print(f"✅ Created user: {email}")
        print(f"User ID: {user_id}")
        print("="*60 + "\n")

    elif user_count > 1:
        print(f"\n❌ ERROR: Found {user_count} users")
        print("MIRA operates in single-user mode only.")
        print("Please keep only one user in the database.")
        print("\nTo fix: Connect to database and delete extra users:")
        print("  psql -U taylut -h localhost -d mira_service")
        print("  DELETE FROM users WHERE id != '<desired-user-id>';")
        sys.exit(1)

    else:
        # Exactly one user exists
        user = db.execute_single("SELECT id, email FROM users LIMIT 1")
        user_id = str(user['id'])
        email = user['email']

        print(f"\n✅ MIRA Ready - Single-User Mode")
        print(f"User: {email}")
        print(f"User ID: {user_id}\n")

    # Set global user context (persists for entire application lifecycle)
    set_current_user_id(user_id)
    app.state.user_id = user_id

    return user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    
    # Startup
    logger.info("  Starting MIRA...\n\n\n")
    logger.info("====================")

    # Ensure single-user mode and set global context
    ensure_single_user(app)

    # Configure FastAPI thread pool for synchronous endpoints
    from anyio import to_thread
    to_thread.current_default_thread_limiter().total_tokens = 100
    logger.info("FastAPI thread pool configured for 100 concurrent threads")
    
    # Pre-initialize expensive singleton resources at startup
    logger.info("Pre-initializing singleton resources...")
    
    # Initialize embeddings provider (loads AllMiniLM + OpenAI embeddings)
    from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
    embeddings_provider = get_hybrid_embeddings_provider()
    logger.info(f"Embeddings provider initialized: {type(embeddings_provider).__name__}")
    
    # Initialize continuum repository (creates DB connection pool)
    from cns.infrastructure.continuum_repository import get_continuum_repository
    continuum_repo = get_continuum_repository()
    logger.info("Continuum repository initialized with connection pool")
    


    # Initialize lt_memory factory following MIRA's singleton pattern
    logger.info("Initializing lt_memory factory...")
    try:
        from clients.llm_provider import LLMProvider
        from utils.database_session_manager import get_shared_session_manager
        from lt_memory.factory import get_lt_memory_factory

        # Create LLM provider for lt_memory (no tools needed for memory extraction)
        lt_memory_llm_provider = LLMProvider(tool_repo=None)

        # Initialize lt_memory factory as singleton
        lt_memory_factory = get_lt_memory_factory(
            config=config.lt_memory,
            session_manager=get_shared_session_manager(),
            embeddings_provider=embeddings_provider,  # Reuse from above
            llm_provider=lt_memory_llm_provider,
            anthropic_client=lt_memory_llm_provider.anthropic_client,  # Reuse from LLMProvider
            conversation_repo=continuum_repo  # Reuse from above
        )
        logger.info("lt_memory factory initialized as singleton")
    except Exception as e:
        logger.critical(f"Failed to initialize lt_memory factory: {e}")
        raise RuntimeError(f"lt_memory initialization failed - cannot start MIRA: {e}") from e

    # Initialize orchestrator as singleton
    logger.info("Initializing continuum orchestrator...")
    from cns.integration.factory import create_cns_orchestrator
    from cns.services.orchestrator import initialize_orchestrator

    orchestrator = create_cns_orchestrator()
    initialize_orchestrator(orchestrator)
    logger.info("CNS Orchastrator initialized as global singleton")

    # Flush Valkey caches on startup except auth sessions and rate limiting
    logger.info("Flushing Valkey caches (preserving sessions and rate limits)...")
    from clients.valkey_client import get_valkey_client
    valkey_client = get_valkey_client()
    flushed_count = valkey_client.flush_except_whitelist(
        preserve_prefixes=["session:", "rate_limit:"]
    )
    logger.info(f"Flushed {flushed_count} cache keys from Valkey")

    # Initialize PlaywrightService for JavaScript-rendered webpages
    try:
        from utils.playwright_service import PlaywrightService
        playwright_service = PlaywrightService.get_instance()
        logger.info("PlaywrightService initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize PlaywrightService: {e}")
        logger.warning("webaccess_tool will not be able to render JavaScript-heavy pages")

    # Event bus is synchronous
    logger.info("Event bus initialized (synchronous)")

    # Initialize all scheduled tasks through central registry
    initialize_all_scheduled_tasks(scheduler_service)

    # Register segment timeout detection job (needs event_bus from orchestrator)
    from utils.scheduled_tasks import register_segment_timeout_job
    register_segment_timeout_job(scheduler_service, orchestrator.event_bus)

    scheduler_service.start()

    # Run on-boot memory extraction sweep in background thread to avoid blocking startup
    def run_boot_extraction():
        """Run boot extraction in background thread - one-time task on startup."""
        logger.info("Running on-boot memory extraction sweep...")
        try:
            boot_results = lt_memory_factory.batching.run_boot_extraction()
            logger.info(
                f"Boot extraction complete: {boot_results['batches_submitted']} batches submitted "
                f"for {boot_results['users_checked']} users checked, "
                f"{boot_results['users_skipped']} users skipped"
            )
        except Exception as e:
            logger.error(f"Boot extraction failed: {e}")
            logger.warning("Memory extraction may be delayed")

    # Start boot extraction in daemon thread (terminates automatically when function completes)
    boot_thread = threading.Thread(target=run_boot_extraction, daemon=True, name="boot-extraction")
    boot_thread.start()
    logger.info("Boot extraction started in background thread (non-blocking, will auto-terminate)")

    # Verify Vault connection (non-blocking)
    from clients.vault_client import test_vault_connection
    vault_status = test_vault_connection()
    if vault_status["status"] != "success":
        logger.warning(f"Vault connection issue: {vault_status['message']}")

    # Federation services moved to separate repository
    # See https://github.com/taylorsatula/gossip-federation for federation support

    logger.info("MIRA startup complete")
    
    yield
    ## @CLAUDE what is this for? ^^
    
    # Shutdown
    logger.info("Shutting down MIRA...")
    scheduler_service.stop()

    # Shutdown event bus
    if orchestrator.event_bus:
        orchestrator.event_bus.shutdown()
        logger.info("Event bus shutdown complete")
    
    # Shutdown Valkey TTL monitoring
    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()
    if valkey:
        valkey.shutdown()
        logger.info("Valkey TTL monitoring shutdown complete")
    
    # Clean up singleton resources
    logger.info("Cleaning up singleton resources...")
    # Clean up embeddings provider
    if embeddings_provider:
        embeddings_provider.close()
        logger.info("Embeddings provider closed")

    # Clean up lt_memory factory
    try:
        lt_memory_factory = get_lt_memory_factory()
        if lt_memory_factory:
            lt_memory_factory.cleanup()
            logger.info("LT_Memory factory cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up LT_Memory factory: {e}")

    # Shutdown PlaywrightService
    try:
        from utils.playwright_service import PlaywrightService
        if PlaywrightService._instance:
            PlaywrightService._instance.shutdown()
            logger.info("PlaywrightService shutdown complete")
    except Exception as e:
        logger.warning(f"Error shutting down PlaywrightService: {e}")

    # Clean up database connections
    from clients.postgres_client import PostgresClient
    PostgresClient.close_all_pools()
    logger.info("PostgreSQL connection pools closed")
    
    from utils.database_session_manager import get_shared_session_manager
    get_shared_session_manager().cleanup()
    
    logger.info("MIRA shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title="MIRA",
        description="A lil Brain-in-a-Box",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Global exception handlers for consistent error responses
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        request_id = generate_request_id()
        errors = exc.errors()
        
        # Format validation errors consistently
        formatted_errors = []
        for error in errors:
            formatted_errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })
        
        response = create_error_response(
            APIError("VALIDATION_ERROR", "Request validation failed", {"errors": formatted_errors}),
            request_id
        )
        return JSONResponse(
            status_code=422,
            content=response.to_dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle FastAPI request validation errors."""
        request_id = generate_request_id()
        errors = exc.errors()
        
        # Format validation errors with field details
        formatted_errors = []
        for error in errors:
            formatted_errors.append({
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"]
            })
        
        response = create_error_response(
            APIError("REQUEST_VALIDATION_ERROR", "Invalid request format", {"detail": formatted_errors}),
            request_id
        )
        # Keep FastAPI's standard validation error format for compatibility
        return JSONResponse(
            status_code=422,
            content={"detail": formatted_errors}
        )
    
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        """Handle custom API errors."""
        request_id = generate_request_id()
        response = create_error_response(exc, request_id)
        
        # Determine status code based on error code
        status_code = 400  # Default to bad request
        if exc.code == "NOT_FOUND":
            status_code = 404
        elif exc.code == "UNAUTHORIZED":
            status_code = 401
        elif exc.code == "FORBIDDEN":
            status_code = 403
        elif exc.code == "SERVICE_UNAVAILABLE":
            status_code = 503
        elif exc.code == "INTERNAL_ERROR":
            status_code = 500
        elif exc.code == "RATE_LIMIT_EXCEEDED":
            status_code = 429
        
        return JSONResponse(
            status_code=status_code,
            content=response.to_dict()
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions."""
        request_id = generate_request_id()
        
        # Log the actual error for debugging
        logger.error(f"Unhandled exception (request_id: {request_id}): {exc}", exc_info=True)
        
        # Return safe error message to client
        response = create_error_response(
            APIError("INTERNAL_ERROR", "An unexpected error occurred", {"request_id": request_id}),
            request_id
        )
        return JSONResponse(
            status_code=500,
            content=response.to_dict()
        )
    
    # Middleware stack (order matters)
    if config.api_server.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api_server.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
    
    # API routes - v0 versioning (beta signal)
    app.include_router(health.router, prefix="/v0/api", tags=["health"])
    app.include_router(chat_api.router, prefix="/v0/api", tags=["chat"])
    app.include_router(data.router, prefix="/v0/api", tags=["data"])
    app.include_router(actions.router, prefix="/v0/api", tags=["actions"])

    return app


async def shutdown_handler(loop, signal=None):
    """Handle shutdown gracefully."""
    if signal:
        logger.info(f"Received exit signal {signal.name}...")
    else:
        logger.info("Shutdown requested...")
    
    # Cancel all running tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    
    for task in tasks:
        task.cancel()
    
    # Wait for all tasks to complete with timeout
    await asyncio.gather(*tasks, return_exceptions=True)
    
    # Stop the event loop
    loop.stop()


def main():
    """Main entry point."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MIRA - AI Assistant with persistent memory')
    parser.add_argument('--firehose', action='store_true',
                       help='Enable firehose mode: log all LLM API calls to firehose_output.json for debugging')
    args = parser.parse_args()

    # Store firehose flag in config for LLMProvider to access
    config.system._firehose_enabled = args.firehose
    if args.firehose:
        logger.info("Firehose mode enabled - LLM API calls will be logged to firehose_output.json")

    try:
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, config.system.log_level.upper(), logging.INFO))
        
        logger.info(f"Starting MIRA on {config.api_server.host}:{config.api_server.port}")
        
        # HTTP/2 is required to prevent connection blocking during streaming
        import hypercorn.asyncio
        from hypercorn import Config
        
        logger.info("Starting with Hypercorn (HTTP/2 enabled)")
        
        # Check for development mode
        dev_mode = os.getenv("MIRA_DEV", "false").lower() in ["true", "1", "yes"]
        
        hypercorn_config = Config()
        hypercorn_config.bind = [f"{config.api_server.host}:{config.api_server.port}"]
        hypercorn_config.alpn_protocols = ["h2", "http/1.1"]  # Prefer HTTP/2, fallback to HTTP/1.1
        hypercorn_config.log_level = config.api_server.log_level

        # Trust proxy headers from nginx (localhost only)
        # This allows proper client IP logging from X-Forwarded-For header
        hypercorn_config.forwarded_allow_ips = ["127.0.0.1", "::1"]
        
        if dev_mode:
            logger.info("Development mode enabled")
            hypercorn_config.use_reloader = True
            hypercorn_config.reload_dirs = [".", "cns", "utils", "tools", "config", "clients"]
            hypercorn_config.workers = 1  # Single worker for development
        else:
            hypercorn_config.workers = config.api_server.workers
        
        # Create event loop and set up signal handlers
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Install signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, 
                lambda s=sig: asyncio.create_task(shutdown_handler(loop, signal=s))
            )
        
        # Run the server
        try:
            loop.run_until_complete(hypercorn.asyncio.serve(create_app(), hypercorn_config))
        except asyncio.CancelledError:
            logger.info("Server task cancelled")
        finally:
            loop.close()
            logger.info("Event loop closed")
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
