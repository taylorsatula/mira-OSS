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
from contextlib import asynccontextmanager
from pathlib import Path

from utils.logging_config import setup_colored_root_logging, setup_anthropic_sdk_logging
setup_colored_root_logging(log_level=logging.WARNING, fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
setup_anthropic_sdk_logging(log_dir="/opt/mira/logs")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from config.config_manager import config
from config.announcement import load_announcement
from cns.api import data, actions, health, websocket_chat, tool_config, trigger_rules, update, federation as federation_api
from cns.api import chat as chat_api
from cns.api import files as files_api
from cns.api import location
from cns.api.base import APIError, create_error_response, generate_request_id
from utils.scheduler_service import scheduler_service
from utils.scheduled_tasks import initialize_all_scheduled_tasks

# Suppress routine APScheduler job execution logs (running/success/debug chatter)
logging.getLogger('apscheduler.executors.default').setLevel(logging.WARNING)
logging.getLogger('apscheduler.scheduler').setLevel(logging.WARNING)
# logging.getLogger('tools.implementations.imagegen_tool').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)



def ensure_single_user(app: FastAPI) -> None:
    """Ensure exactly one user exists for single-user mode."""
    import sys
    from utils.database_session_manager import get_shared_session_manager

    session_manager = get_shared_session_manager()

    # Check user count and create if needed (commits on block exit)
    user_id = None
    default_email = "user@localhost"

    with session_manager.get_admin_session() as session:
        result = session.execute_single("SELECT COUNT(*) as count FROM users")
        user_count = result['count']

        # Detect offline tier (created by deploy when user chose offline mode)
        offline_tier = session.execute_single(
            "SELECT name FROM conversation_llm WHERE name = 'offline'"
        )
        oss_default_tier = 'offline' if offline_tier else 'primary'

        if user_count > 1:
            print(f"\nERROR: Found {user_count} users")
            print("MIRA OSS operates in single-user mode only.")
            sys.exit(1)

        if user_count == 1:
            user = session.execute_single("SELECT id, email FROM users LIMIT 1")
            app.state.single_user_id = str(user['id'])
            app.state.user_email = user['email']

            # OSS: user brings own API key, set balance high and default to correct tier
            session.execute_update(
                """UPDATE users SET balance_usd = 999999.00,
                   conversation_llm = CASE WHEN conversation_llm NOT IN (SELECT name FROM conversation_llm) THEN %(tier)s ELSE conversation_llm END
                   WHERE id = %(id)s""",
                {'id': str(user['id']), 'tier': oss_default_tier}
            )

            try:
                from clients.vault_client import _ensure_vault_client
                vault_client = _ensure_vault_client()
                secret_data = vault_client.client.secrets.kv.v2.read_secret_version(
                    path='mira/api_keys'
                )
                api_key = secret_data['data']['data'].get('mira_api')
                app.state.api_key = api_key

                print(f"\nMIRA Ready - User: {user['email']}\n")
            except Exception as e:
                logger.error(f"Failed to retrieve API key from Vault: {e}")
                print("\nERROR: Could not retrieve API key from Vault")
                sys.exit(1)
            return

        # user_count == 0: create the single user
        import uuid

        user_id = str(uuid.uuid4())

        session.execute_update("""
            INSERT INTO users (id, email, is_active, memory_manipulation_enabled, balance_usd, conversation_llm)
            VALUES (%(id)s, %(email)s, true, true, 999999.00, %(tier)s)
        """, {'id': user_id, 'email': default_email, 'tier': oss_default_tier})

        # Create the continuum (normally done during signup flow)
        continuum_id = str(uuid.uuid4())
        session.execute_update("""
            INSERT INTO continuums (id, user_id, metadata, created_at, updated_at)
            VALUES (%(id)s, %(user_id)s, '{}'::jsonb, NOW(), NOW())
        """, {'id': continuum_id, 'user_id': user_id})

        # Prepopulate with starter messages (ported from auth.database.prepopulate_new_user)
        import json
        from utils.timezone_utils import utc_now

        # Message 1: Beginning marker
        msg1_id = str(uuid.uuid4())
        session.execute_update("""
            INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at)
            VALUES (%(id)s, %(continuum_id)s, %(user_id)s, 'user', %(content)s, %(metadata)s, NOW())
        """, {
            'id': msg1_id,
            'continuum_id': continuum_id,
            'user_id': user_id,
            'content': '.. this is the beginning of the conversation. there are no messages older than this one ..',
            'metadata': json.dumps({'system_generated': True})
        })

        # Message 2: Active segment sentinel
        segment_id = str(uuid.uuid4())
        segment_metadata = {
            'is_segment_boundary': True,
            'status': 'active',
            'segment_id': segment_id,
            'segment_start_time': utc_now().isoformat(),
            'segment_end_time': utc_now().isoformat(),
            'segment_turn_count': 1,  # Required for increment_segment_turn()
            'tools_used': [],
            'memories_extracted': False,
            'domain_blocks_updated': False
        }
        msg2_id = str(uuid.uuid4())
        session.execute_update("""
            INSERT INTO messages (id, continuum_id, user_id, role, content, metadata, created_at)
            VALUES (%(id)s, %(continuum_id)s, %(user_id)s, 'assistant', %(content)s, %(metadata)s, NOW() + interval '100 milliseconds')
        """, {
            'id': msg2_id,
            'continuum_id': continuum_id,
            'user_id': user_id,
            'content': '[Segment in progress]',
            'metadata': json.dumps(segment_metadata)
        })

        logger.info(f"Created user {user_id} with continuum {continuum_id} and starter messages")

    # Admin session committed — user row now visible to other connections
    # Initialize feedback tracking (uses its own session via get_session)
    from auth.seed_lora import seed_lora_postgres
    seed_lora_postgres(user_id)
    logger.info(f"Initialized feedback tracking for user {user_id}")

    import secrets
    api_key = f"mira_{secrets.token_urlsafe(32)}"

    try:
        from clients.vault_client import _ensure_vault_client
        vault_client = _ensure_vault_client()
        # Use patch to add mira_api without overwriting anthropic_key/provider_key
        vault_client.client.secrets.kv.v2.patch(
            path='mira/api_keys',
            secret=dict(mira_api=api_key)
        )
    except Exception as e:
        logger.warning(f"Could not store key in Vault: {e}")

    app.state.single_user_id = user_id
    app.state.user_email = default_email
    app.state.api_key = api_key

    print(f"\n{'='*60}")
    print("MIRA Ready - Single-User OSS Mode")
    print(f"{'='*60}")
    print(f"User: {default_email}")
    print(f"API Key: {api_key}")
    print(f"{'='*60}\n")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    
    # Startup
    logger.info("  Starting MIRA...\n\n\n")
    logger.info("====================")

    # Ensure single user exists and load credentials
    ensure_single_user(app)


    # Configure FastAPI thread pool for synchronous endpoints
    from anyio import to_thread
    to_thread.current_default_thread_limiter().total_tokens = 100
    logger.info("FastAPI thread pool configured for 100 concurrent threads")
    
    # Pre-initialize expensive singleton resources at startup
    logger.info("Pre-initializing singleton resources...")

    # Preload all Vault secrets into memory cache (prevents token expiration issues)
    from clients.vault_client import preload_secrets
    preload_secrets()

    # Load announcement config (cached for lifetime of process)
    load_announcement()

    # Initialize embeddings provider (loads mdbr-leaf-ir-asym 768d model)
    from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
    embeddings_provider = get_hybrid_embeddings_provider()
    logger.info(f"Embeddings provider initialized: {type(embeddings_provider).__name__}")
    
    # Initialize continuum repository (creates DB connection pool)
    from cns.infrastructure.continuum_repository import get_continuum_repository
    continuum_repo = get_continuum_repository()
    logger.info("Continuum repository initialized with connection pool")

    # Load internal LLM configs from database (fail-fast at startup)
    from utils.user_context import load_internal_llm_configs
    load_internal_llm_configs()
    logger.info("Internal LLM configs loaded from database")

    # Load billing pricing cache and validate prices (skipped in OSS mode)
    try:
        from billing.pricing import load_pricing_cache, build_config_lookup, ensure_pricing_keys

        # 1. Seed: ensure every conversation_llm + internal_llm key has a usage_pricing row (NULL prices)
        ensure_pricing_keys()
        # 2. Load: read all pricing rows into memory
        load_pricing_cache()
        logger.info("Billing pricing cache loaded from database")
        # 3. Resolve and validate: startup must fail if any pricing remains unresolved
        from billing.price_validator import validate_prices_against_openrouter
        validate_prices_against_openrouter()
        # 4. Lookup: build reverse map for runtime pricing_key resolution
        build_config_lookup()
    except ImportError:
        logger.info("Billing module not available (OSS mode)")

    # Initialize lt_memory factory following MIRA's singleton pattern
    logger.info("Initializing lt_memory factory...")
    try:
        from clients.llm_provider import LLMProvider
        from utils.database_session_manager import get_shared_session_manager
        from lt_memory.factory import get_lt_memory_factory

        lt_memory_llm_provider = LLMProvider()

        lt_memory_factory = get_lt_memory_factory(
            session_manager=get_shared_session_manager(),
            embeddings_provider=embeddings_provider,
            llm_provider=lt_memory_llm_provider,
            conversation_repo=continuum_repo
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
    logger.info("CNS Orchestrator initialized as global singleton")

    # Flush Valkey caches on startup except auth sessions and rate limiting
    logger.info("Flushing Valkey caches (preserving sessions and rate limits)...")
    from clients.valkey_client import get_valkey_client
    valkey_client = get_valkey_client()
    flushed_count = valkey_client.flush_except_whitelist(
        preserve_prefixes=["session:", "rate_limit:"]
    )
    logger.info(f"Flushed {flushed_count} cache keys from Valkey")

    # PlaywrightService is lazy — Chromium launches on first web_tool fetch,
    # auto-shuts down after idle timeout. Just verify the import works.
    try:
        from utils.playwright_service import PlaywrightService
        PlaywrightService.get_instance()  # Creates singleton, does NOT launch Chromium
        logger.info("PlaywrightService ready (Chromium launches on first use)")
    except ImportError as e:
        logger.warning(f"Playwright not available: {e}")
        logger.warning("web_tool will not be able to render JavaScript-heavy pages")

    # Event bus is synchronous
    logger.info("Event bus initialized (synchronous)")

    # Initialize all scheduled tasks through central registry
    initialize_all_scheduled_tasks(scheduler_service)

    # Register segment timeout detection job (needs event_bus from orchestrator)
    from utils.scheduled_tasks import register_segment_timeout_job, register_sidebar_dispatcher_job
    register_segment_timeout_job(scheduler_service, orchestrator.event_bus)

    # Register sidebar dispatcher (needs tool_repo + event_bus)
    register_sidebar_dispatcher_job(
        scheduler_service, orchestrator.tool_repo, orchestrator.event_bus
    )

    # Register billing daily drip job (skipped in OSS mode)
    try:
        from billing.drip import DailyDripService
        drip_service = DailyDripService()
        drip_service.register_jobs(scheduler_service)
    except ImportError:
        pass  # OSS mode - no billing

    scheduler_service.start()

    # Collapse any segments stale during downtime through the existing event pipeline.
    # check_timeouts() publishes SegmentTimeoutEvent for stale segments, which the
    # collapse handler processes (summary + extraction). The 6-hour extract_unprocessed_segments
    # sweep catches any that fail.
    from cns.services.segment_timeout_service import get_timeout_service
    timeout_service = get_timeout_service(orchestrator.event_bus)
    timeout_service.check_timeouts()
    logger.info("Startup timeout check complete (stale segments will collapse via event pipeline)")

    # Verify Vault connection (non-blocking)
    from clients.vault_client import test_vault_connection
    vault_status = test_vault_connection()
    if vault_status["status"] != "success":
        logger.warning(f"Vault connection issue: {vault_status['message']}")

    # Register Lattice username resolver for federation
    # This allows Lattice to resolve usernames to user_ids for inbound message delivery
    try:
        from lattice.username_resolver import set_username_resolver
        from clients.postgres_client import PostgresClient
        from typing import Optional

        def mira_resolve_username(username: str) -> Optional[str]:
            """Resolve username to user_id for Lattice federation."""
            db = PostgresClient("mira_service")
            result = db.execute_single(
                "SELECT user_id FROM global_usernames WHERE username = %(username)s AND active = true",
                {"username": username.lower()}
            )
            return str(result["user_id"]) if result else None

        set_username_resolver(mira_resolve_username)
        logger.info("Lattice username resolver registered")
    except ImportError:
        logger.warning("Lattice package not available - federation disabled")
    except Exception as e:
        logger.warning(f"Failed to register Lattice username resolver: {e}")


    logger.info("MIRA startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MIRA...")
    scheduler_service.stop()

    # Close all active WebSocket connections
    from cns.api.websocket_chat import close_all_connections
    await close_all_connections()
    logger.info("WebSocket connections closed")
    
    # Shutdown event bus
    if orchestrator.event_bus:
        orchestrator.event_bus.shutdown()
        logger.info("Event bus shutdown complete")
    
    # Shutdown Valkey client
    from clients.valkey_client import get_valkey_client
    valkey = get_valkey_client()
    if valkey:
        valkey.shutdown()
        logger.info("Valkey client shutdown complete")
    
    # Clean up singleton resources
    logger.info("Cleaning up singleton resources...")

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

    # Clean up UserDataManager SQLite connections
    from utils.userdata_manager import clear_manager_cache
    clear_manager_cache()
    logger.info("UserDataManager cache cleared (SQLite connections closed)")

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
        version="2026.03.07-major",
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
    
    # Middleware stack (order matters — applied in reverse registration order)
    from utils.perf import PerfMiddleware
    app.add_middleware(PerfMiddleware)

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
    app.include_router(update.router, prefix="/v0/api", tags=["update"])  # Public update check
    app.include_router(chat_api.router, prefix="/v0/api", tags=["chat"])
    app.include_router(data.router, prefix="/v0/api", tags=["data"])
    app.include_router(actions.router, prefix="/v0/api", tags=["actions"])
    app.include_router(tool_config.router, prefix="/v0/api", tags=["tool_config"])
    app.include_router(trigger_rules.router, prefix="/v0/api", tags=["trigger_rules"])
    app.include_router(files_api.router, prefix="/v0/api", tags=["files"])
    app.include_router(location.router, prefix="/v0/api", tags=["location"])
    app.include_router(websocket_chat.router, prefix="/v0", tags=["websocket"])  # /v0/ws/chat
    app.include_router(federation_api.router, prefix="/v0/api", tags=["federation"])

    # OSS browser-auth + asset routes (always-on; /oss-auth/token feeds the web UI's Bearer-key login)
    from cns.api import oss_ui
    app.include_router(oss_ui.router, tags=["oss-ui"])

    # Billing routes (skipped in OSS mode)
    try:
        from billing import api as billing_api
        from billing import stripe_webhooks
        app.include_router(billing_api.router, prefix="/v0/api", tags=["billing"])
        app.include_router(stripe_webhooks.router, prefix="/v0/api", tags=["billing-webhooks"])
    except ImportError:
        pass  # OSS mode - no billing

    # Performance monitoring (gated by mira.perf logger level)
    from utils.perf import register_perf_routes, install_db_instrumentation
    register_perf_routes(app)
    install_db_instrumentation()

    # Full web UI — de-auth-gated page routes (OSS single-user: no session dependency).
    # Serves the ported web/ bundle (chat, memories, domaindocs, settings) plus
    # root meta files and the /assets static mount.
    if Path("web").exists():
        @app.get("/", include_in_schema=False)
        async def serve_root():
            return RedirectResponse(url="/chat")

        @app.get("/chat", include_in_schema=False)
        @app.get("/chat/", include_in_schema=False)
        async def serve_chat():
            return FileResponse("web/chat/index.html")

        @app.get("/memories", include_in_schema=False)
        @app.get("/memories/", include_in_schema=False)
        async def serve_memories():
            return FileResponse("web/memories/index.html")

        @app.get("/domaindocs", include_in_schema=False)
        @app.get("/domaindocs/", include_in_schema=False)
        async def serve_domaindocs():
            return FileResponse("web/domaindocs/index.html")

        @app.get("/settings", include_in_schema=False)
        @app.get("/settings/", include_in_schema=False)
        async def serve_settings():
            return FileResponse("web/settings/index.html")

        # Browser-expected static files from root
        @app.get("/apple-touch-icon.png", include_in_schema=False)
        async def serve_apple_touch_icon():
            return FileResponse("web/apple-touch-icon.png")

        @app.get("/favicon.ico", include_in_schema=False)
        async def serve_favicon():
            return FileResponse("web/favicon.ico")

        @app.get("/manifest.json", include_in_schema=False)
        async def serve_manifest():
            return FileResponse("web/manifest.json")

        # Static assets (JS/CSS/fonts/images) — mounted after page routes
        app.mount("/assets", StaticFiles(directory="web/assets"), name="assets")

    return app


def main():
    """Main entry point."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MIRA - AI Assistant with persistent memory')
    parser.add_argument('--firehose', action='store_true',
                       help='Enable firehose mode: log all LLM API calls to firehose_output.json for debugging')
    args = parser.parse_args()

    # Firehose: toggle live with kill -USR1 $(systemctl show mira -p MainPID --value)
    if args.firehose:
        from utils.llm_tap import toggle as _toggle_traffic_tap
        _toggle_traffic_tap(None, None)

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

        from utils.power_on_self_test import run_pre_server_post_gate
        run_pre_server_post_gate()
        
        # Run the server — Hypercorn manages SIGTERM/SIGINT natively
        asyncio.run(hypercorn.asyncio.serve(create_app(), hypercorn_config))
        
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
