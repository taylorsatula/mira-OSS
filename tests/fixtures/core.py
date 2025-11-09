"""
Core test fixtures using real services and authentication.

No mocks - uses real infrastructure with dedicated test user.
"""
import pytest
import pytest_asyncio
import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

# Use the dedicated test user for consistency
TEST_USER_ID = "443a898d-ed56-495a-b9de-0551c80169fe"
TEST_USER_EMAIL = "test@example.com"

# Second test user for RLS testing
SECOND_TEST_USER_ID = "7b8e4c2a-f3d1-4a5b-9e6c-1d2f3a4b5c6d"
SECOND_TEST_USER_EMAIL = "test2@example.com"


def ensure_test_user_exists() -> dict:
    """
    Ensure the test user exists in the database, creating if necessary.
    Returns the user record with actual user ID.

    This function implements a get-or-create pattern using email as the
    consistent identifier since database-generated UUIDs are not predictable.
    """
    from auth.database import AuthDatabase

    auth_db = AuthDatabase()

    # Try to get user by email (reliable identifier)
    user_record = auth_db.get_user_by_email(TEST_USER_EMAIL)
    if user_record:
        return user_record

    # User doesn't exist, create new one
    try:
        created_user_id = auth_db.create_user(TEST_USER_EMAIL)
        user_record = auth_db.get_user_by_id(created_user_id)
        logger.info(f"Created test user with ID {created_user_id}")
        return user_record
    except Exception as e:
        # If creation fails (user might already exist due to race condition), try by email again
        user_record = auth_db.get_user_by_email(TEST_USER_EMAIL)
        if user_record:
            return user_record
        raise RuntimeError(f"Failed to ensure test user exists: {e}")


def ensure_test_user_ready() -> dict:
    """
    Ensure test user is fully set up and ready: user record + continuum + context.

    This is the comprehensive setup function that prepares everything needed for tests.
    Call this when you need a fully initialized test user ready to add messages.

    Returns:
        dict with 'user_id', 'email', 'continuum_id' - everything needed for testing
    """
    from cns.infrastructure.continuum_repository import get_continuum_repository
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id

    # Step 1: Ensure user exists
    user_record = ensure_test_user_exists()
    user_id = user_record["id"]

    # Step 2: Set user context for RLS
    set_current_user_id(user_id)

    # Step 3: Ensure continuum exists (each user has exactly one)
    db = PostgresClient("mira_service", user_id=user_id)
    result = db.execute_query(
        "SELECT id FROM continuums WHERE user_id = %s LIMIT 1",
        (user_id,)
    )

    if result and len(result) > 0:
        # Continuum exists, use it
        continuum_id = str(result[0][0])
        logger.debug(f"Using existing continuum {continuum_id} for test user {user_id}")
    else:
        # Create continuum for test user
        repo = get_continuum_repository()
        continuum = repo.create_continuum(user_id)
        continuum_id = str(continuum.id)
        logger.info(f"Created continuum {continuum_id} for test user {user_id}")

    # Step 4: Ensure tool schemas exist
    from pathlib import Path
    user_db_path = Path(f"data/users/{user_id}/userdata.db")
    if user_db_path.exists():
        # Check if schemas are already loaded
        import sqlite3
        conn = sqlite3.connect(str(user_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contacts'")
        has_schema = cursor.fetchone() is not None
        conn.close()

        if not has_schema:
            # Load schemas for existing database
            from tools.schema_distribution import initialize_user_database
            initialize_user_database(str(user_id))
            logger.info(f"Initialized schemas for test user {user_id}")

    return {
        "user_id": user_id,
        "email": user_record["email"],
        "continuum_id": continuum_id,
        "is_active": user_record["is_active"],
        "created_at": user_record["created_at"]
    }


def ensure_second_test_user_exists() -> dict:
    """
    Ensure the second test user exists in the database, creating if necessary.
    Returns the user record with actual user ID.

    This is used for RLS testing to verify user isolation.
    """
    from auth.database import AuthDatabase

    auth_db = AuthDatabase()

    # Try to get user by email (reliable identifier)
    user_record = auth_db.get_user_by_email(SECOND_TEST_USER_EMAIL)
    if user_record:
        return user_record

    # User doesn't exist, create new one
    try:
        created_user_id = auth_db.create_user(SECOND_TEST_USER_EMAIL)
        user_record = auth_db.get_user_by_id(created_user_id)
        logger.info(f"Created second test user with ID {created_user_id}")
        return user_record
    except Exception as e:
        # If creation fails (user might already exist due to race condition), try by email again
        user_record = auth_db.get_user_by_email(SECOND_TEST_USER_EMAIL)
        if user_record:
            return user_record
        raise RuntimeError(f"Failed to ensure second test user exists: {e}")


def ensure_second_test_user_ready() -> dict:
    """
    Ensure second test user is fully set up and ready: user record + continuum + context.

    This is used for RLS testing to verify user isolation.

    Returns:
        dict with 'user_id', 'email', 'continuum_id' - everything needed for testing
    """
    from cns.infrastructure.continuum_repository import get_continuum_repository
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id

    # Step 1: Ensure user exists
    user_record = ensure_second_test_user_exists()
    user_id = user_record["id"]

    # Step 2: Set user context for RLS
    set_current_user_id(user_id)

    # Step 3: Ensure continuum exists (each user has exactly one)
    db = PostgresClient("mira_service", user_id=user_id)
    result = db.execute_query(
        "SELECT id FROM continuums WHERE user_id = %s LIMIT 1",
        (user_id,)
    )

    if result and len(result) > 0:
        # Continuum exists, use it
        continuum_id = str(result[0][0])
        logger.debug(f"Using existing continuum {continuum_id} for second test user {user_id}")
    else:
        # Create continuum for test user
        repo = get_continuum_repository()
        continuum = repo.create_continuum(user_id)
        continuum_id = str(continuum.id)
        logger.info(f"Created continuum {continuum_id} for second test user {user_id}")

    # Step 4: Ensure tool schemas exist
    from pathlib import Path
    user_db_path = Path(f"data/users/{user_id}/userdata.db")
    if user_db_path.exists():
        # Check if schemas are already loaded
        import sqlite3
        conn = sqlite3.connect(str(user_db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contacts'")
        has_schema = cursor.fetchone() is not None
        conn.close()

        if not has_schema:
            # Load schemas for existing database
            from tools.schema_distribution import initialize_user_database
            initialize_user_database(str(user_id))
            logger.info(f"Initialized schemas for second test user {user_id}")

    return {
        "user_id": user_id,
        "email": user_record["email"],
        "continuum_id": continuum_id,
        "is_active": user_record["is_active"],
        "created_at": user_record["created_at"]
    }


# ================== DATABASE FIXTURES ==================

@pytest_asyncio.fixture
async def test_db():
    """
    Provide real database access with the test user context.
    Uses actual database with dedicated test user for RLS isolation.
    """
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id, clear_user_context
    
    # Get actual test user and use their real ID
    user_record = ensure_test_user_exists()
    actual_user_id = user_record["id"]
    
    set_current_user_id(actual_user_id)
    db = PostgresClient("mira_service", user_id=actual_user_id)

    try:
        yield db
    finally:
        clear_user_context()


@pytest_asyncio.fixture
async def test_memory_db():
    """Provide real database access with test user context (unified mira_service)."""
    from clients.postgres_client import PostgresClient
    from utils.user_context import set_current_user_id, clear_user_context

    # Get actual test user and use their real ID
    user_record = ensure_test_user_exists()
    actual_user_id = user_record["id"]

    set_current_user_id(actual_user_id)
    db = PostgresClient("mira_service", user_id=actual_user_id)

    try:
        yield db
    finally:
        clear_user_context()


# ================== SERVICE FIXTURES ==================

@pytest_asyncio.fixture
async def conversation_repository():
    """
    Provide the real continuum repository singleton.
    Uses actual singleton with test user isolation for data separation.
    """
    from cns.infrastructure.continuum_repository import get_continuum_repository
    return get_continuum_repository()


@pytest_asyncio.fixture
async def auth_service():
    """
    Provide real auth service instance.
    Uses actual auth infrastructure for realistic testing.
    """
    from auth.service import AuthService
    return AuthService()


@pytest_asyncio.fixture
async def vault_client():
    """Provide real vault client for testing."""
    from clients.vault_client import VaultClient
    try:
        client = VaultClient()
        client.get_secret('mira/auth', 'jwt_secret_key')
        return client
    except Exception as e:
        pytest.skip(f"Vault unavailable: {e}")


@pytest_asyncio.fixture
async def valkey_client():
    """
    Provide real Valkey client for testing.
    """
    from clients.valkey_client import get_valkey_client
    client = get_valkey_client()
    
    if not client.valkey_available:
        pytest.skip("Valkey unavailable")
    
    return client


# ================== USER & AUTH FIXTURES ==================

@pytest_asyncio.fixture
async def authenticated_user(auth_service):
    """
    Provide authenticated test user with valid session token AND continuum ready.

    This fixture ensures complete test user setup:
    - User record exists
    - User's continuum exists
    - User context is set
    - Session token is created

    Tests can immediately add messages without additional setup.
    """
    from utils.user_context import set_current_user_data, clear_user_context
    from auth.database import AuthDatabase

    # Ensure test user is fully ready (user + continuum + context)
    setup = ensure_test_user_ready()

    # Create real session token
    user_record = {
        "id": setup["user_id"],
        "email": setup["email"],
        "is_active": setup["is_active"],
        "created_at": setup["created_at"]
    }
    session_token = auth_service.create_session(setup["user_id"], user_record)

    # Set user context for the test
    user_data = {
        "user_id": setup["user_id"],
        "email": setup["email"],
        "continuum_id": setup["continuum_id"],  # Include continuum_id
        "is_active": setup["is_active"]
    }
    set_current_user_data(user_data)

    user_with_auth = user_data.copy()
    user_with_auth["access_token"] = session_token  # Use session token as Bearer token

    try:
        yield user_with_auth
    finally:
        clear_user_context()


@pytest_asyncio.fixture
async def second_authenticated_user(auth_service):
    """
    Provide second authenticated test user for RLS testing.

    This fixture creates a completely separate user with their own:
    - User record
    - User's continuum
    - User context
    - Session token

    Use this fixture alongside `authenticated_user` to verify that RLS
    correctly isolates data between users.
    """
    from utils.user_context import set_current_user_data, clear_user_context
    from auth.database import AuthDatabase

    # Ensure second test user is fully ready (user + continuum + context)
    setup = ensure_second_test_user_ready()

    # Create real session token
    user_record = {
        "id": setup["user_id"],
        "email": setup["email"],
        "is_active": setup["is_active"],
        "created_at": setup["created_at"]
    }
    session_token = auth_service.create_session(setup["user_id"], user_record)

    # Set user context for the test
    user_data = {
        "user_id": setup["user_id"],
        "email": setup["email"],
        "continuum_id": setup["continuum_id"],
        "is_active": setup["is_active"]
    }
    set_current_user_data(user_data)

    user_with_auth = user_data.copy()
    user_with_auth["access_token"] = session_token

    try:
        yield user_with_auth
    finally:
        clear_user_context()


# ================== CONTINUUM FIXTURES ==================

@pytest_asyncio.fixture
async def realistic_conversation(conversation_repository, authenticated_user):
    """
    Load the realistic continuum from JSON fixture.
    Provides continuum with rich history for testing.
    """
    from tests.fixtures.conversation_data import load_realistic_conversation_data
    
    # Load realistic continuum (handles cleanup internally)
    continuum = await load_realistic_conversation_data()
    return continuum


@pytest_asyncio.fixture
async def realistic_messages():
    """
    Provide just the realistic messages without creating a continuum.
    """
    import json
    from pathlib import Path
    from datetime import timedelta
    from cns.core.message import Message
    from utils.timezone_utils import utc_now
    
    # Load continuum data from JSON
    fixture_path = Path(__file__).parent / "realistic_conversation.json"
    with open(fixture_path, 'r') as f:
        data = json.load(f)
    
    # Create message objects
    messages = []
    base_time = utc_now()
    
    for msg_data in data['messages']:
        timestamp_offset = timedelta(hours=msg_data.get('timestamp_offset_hours', 0))
        message = Message(
            content=msg_data['content'],
            role=msg_data['role'],
            created_at=base_time + timestamp_offset,
            metadata=msg_data.get('metadata', {})
        )
        messages.append(message)
    
    return messages


# ================== CLEANUP FIXTURES ==================

@pytest.fixture(autouse=True)
def cleanup_test_data():
    """
    Automatically clean up test data before and after each test.
    
    This ensures complete test isolation by removing all user-scoped data
    while preserving the test user record for reuse.
    """
    from tests.fixtures.conversation_data import cleanup_test_user_data
    
    # Clean before test - ensure clean state
    try:
        cleanup_test_user_data()
        logger.debug("Pre-test cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Pre-test cleanup failed: {e}")
        # Continue with test - don't fail on cleanup issues
    
    yield
    
    # Clean after test - ensure no test artifacts remain
    try:
        cleanup_test_user_data()
        logger.debug("Post-test cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Post-test cleanup failed: {e}")
        # Don't fail the test due to cleanup issues


@pytest.fixture(autouse=True)
def reset_user_context():
    """
    Ensure user context is clean for each test.
    """
    from utils.user_context import clear_user_context
    
    clear_user_context()
    yield
    clear_user_context()


# ================== INTEGRATION TEST FIXTURES ==================

@pytest_asyncio.fixture
async def chat_handler(conversation_repository, authenticated_user):
    """
    Provide real ChatHandler with all dependencies.
    Uses real orchestrator and repository for integration testing.
    """
    from cns.api.chat import ChatHandler
    from cns.integration.factory import create_cns_orchestrator
    
    orchestrator = await create_cns_orchestrator()
    return ChatHandler(orchestrator, conversation_repository)


@pytest.fixture
def orchestrator():
    """Provide real CNS orchestrator for integration tests."""
    from cns.integration.factory import create_cns_orchestrator
    from utils.user_context import set_current_user_id, clear_user_context
    
    # Get actual test user and use their real ID
    user_record = ensure_test_user_exists()
    actual_user_id = user_record["id"]
    
    # Set test user context before creating orchestrator
    # This prevents tools from failing during initialization
    set_current_user_id(actual_user_id)
    
    try:
        orchestrator_instance = create_cns_orchestrator()
        yield orchestrator_instance
    finally:
        clear_user_context()


# ================== FASTAPI TEST CLIENT ==================

@pytest.fixture
def test_client():
    """
    Create FastAPI TestClient with real app configuration.
    """
    from fastapi.testclient import TestClient
    from main import create_app
    
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def authenticated_client(test_client, authenticated_user):
    """
    Create authenticated test client with headers set.
    
    Note: This is for API testing only. For testing sync code directly,
    use the authenticated_user fixture and set context manually.
    """
    # Just set the headers - don't try to override dependencies
    test_client.headers = {"Authorization": f"Bearer {authenticated_user['access_token']}"}
    return test_client


@pytest_asyncio.fixture
async def async_client():
    """
    Create async HTTP client for testing async endpoints.
    """
    import httpx
    from main import create_app
    
    app = create_app()
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def authenticated_async_client(authenticated_user):
    """
    Create authenticated async HTTP client for testing streaming endpoints.
    
    Note: This is for API testing only. For testing sync code directly,
    use the authenticated_user fixture and set context manually.
    """
    import httpx
    from main import create_app
    
    app = create_app()
    
    # Use ASGITransport for proper async support
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Set headers for authentication
        client.headers = {"Authorization": f"Bearer {authenticated_user['access_token']}"}
        yield client


# ================== REQUEST DATA FIXTURES ==================

@pytest.fixture
def sample_chat_request():
    """Provide sample chat request data."""
    return {
        "message": "Hello, this is a test message",
        "continuum_id": None,
        "stream": False
    }


# ================== PERFORMANCE FIXTURES ==================

@pytest.fixture
def timer():
    """Simple timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed()
        
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()
