"""
User data management module for per-user SQLite databases with session-based encryption.

Connection management: Each UserDataManager owns a single persistent SQLite connection
(created lazily on first access). Instances are cached per-user to avoid redundant
connection creation. Connections are cleaned up on session collapse or process shutdown.
"""

import base64
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cns.integration.event_bus import EventBus

# Module-level cache for UserDataManager instances
_manager_cache: Dict[str, "UserDataManager"] = {}


class UserDataManager:
    """
    Manages per-user SQLite databases with session-based encryption.

    Connection ownership: Each instance owns a single persistent SQLite connection,
    created lazily on first database access. Use `close()` for explicit cleanup or
    context manager pattern (`with get_user_data_manager(user_id) as db:`).
    """

    def __init__(self, user_id: UUID, session_key: bytes):
        self.user_id = user_id
        self.session_key = session_key
        self.fernet = self._create_fernet()
        self.db_path = self._get_user_db_path()
        self._conn: Optional[sqlite3.Connection] = None  # Lazy persistent connection
        self._ensure_database()

    @property
    def connection(self) -> sqlite3.Connection:
        """
        Lazy persistent connection (thread-safe for cross-thread reuse).

        Connection is created on first access and reused for all subsequent operations.
        Uses check_same_thread=False to allow reuse across ThreadPoolExecutor workers.
        WAL mode enables concurrent reads/writes from multiple threads.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False  # Allow cross-thread usage
            )
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for concurrent read/write support
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s for locks
        return self._conn

    def close(self) -> None:
        """Close the persistent connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug(f"Closed SQLite connection for user {self.user_id}")

    def __enter__(self) -> "UserDataManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher from session key."""
        key = base64.urlsafe_b64encode(self.session_key[:32])  # Ensure 32 bytes
        return Fernet(key)
    
    @property
    def base_dir(self) -> Path:
        from utils.instance import user_data_base
        user_dir = user_data_base() / str(self.user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _get_user_db_path(self) -> Path:
        return self.base_dir / "userdata.db"
    
    def _ensure_database(self):
        """Ensure database exists and schemas are current.

        Always runs schema initialization since CREATE TABLE IF NOT EXISTS
        is idempotent. This ensures new tables are added to existing databases
        when features are deployed.
        """
        if not self.db_path.exists():
            self.db_path.touch()
            logger.info(f"Created user database: {self.db_path}")
        self._initialize_tool_schemas()
    
    def _initialize_tool_schemas(self):
        """Initialize database schemas for all tools.

        TODO: Move schema migrations out of this hot path into standalone
        scripts (scripts/migrations/) with a version table per user DB.
        Running migrations on every UserDataManager instantiation is fragile —
        migration ordering bugs caused a production outage (archived index
        before column existed).
        """
        logger.info(f"Initializing tool schemas for user {self.user_id}")

        # Initialize PagerTool schema
        self._init_pager_schema()

        # Initialize Domaindoc schema
        self._init_domaindoc_schema()

        # Initialize trigger rules schema (sidebar agent trigger filters)
        self._init_trigger_rules_schema()

        logger.info("Tool schemas initialized successfully")
    
    def _init_pager_schema(self):
        """Initialize PagerTool database schema."""
        cursor = self.connection.cursor()

        # Pager devices table
        devices_sql = """
        CREATE TABLE IF NOT EXISTS pager_devices (
            id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            device_secret TEXT NOT NULL,
            device_fingerprint TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE
        )
        """

        # Pager trust table
        trust_sql = """
        CREATE TABLE IF NOT EXISTS pager_trust (
            id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            trusting_device_id TEXT NOT NULL,
            trusted_device_id TEXT NOT NULL,
            trusted_fingerprint TEXT NOT NULL,
            trusted_name TEXT,
            first_seen TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_verified TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            trust_status TEXT NOT NULL DEFAULT 'trusted',
            UNIQUE(trusting_device_id, trusted_device_id)
        )
        """

        # Pager messages table
        messages_sql = """
        CREATE TABLE IF NOT EXISTS pager_messages (
            id TEXT PRIMARY KEY,
            user_id UUID NOT NULL,
            sender_id TEXT NOT NULL,
            recipient_id TEXT NOT NULL,
            content TEXT NOT NULL,
            original_content TEXT,
            ai_distilled BOOLEAN NOT NULL DEFAULT FALSE,
            priority INTEGER NOT NULL DEFAULT 0,
            location TEXT,
            sent_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP WITH TIME ZONE,
            read_at TIMESTAMP WITH TIME ZONE,
            delivered BOOLEAN NOT NULL DEFAULT TRUE,
            read BOOLEAN NOT NULL DEFAULT FALSE,
            message_signature TEXT,
            sender_fingerprint TEXT
        )
        """

        # Create indexes
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_pager_devices_user_id ON pager_devices(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_pager_devices_active ON pager_devices(active)",
            "CREATE INDEX IF NOT EXISTS idx_pager_trust_user_id ON pager_trust(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_pager_trust_trusting_device ON pager_trust(trusting_device_id)",
            "CREATE INDEX IF NOT EXISTS idx_pager_messages_user_id ON pager_messages(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_pager_messages_sender ON pager_messages(sender_id)",
            "CREATE INDEX IF NOT EXISTS idx_pager_messages_recipient ON pager_messages(recipient_id)",
            "CREATE INDEX IF NOT EXISTS idx_pager_messages_expires ON pager_messages(expires_at)"
        ]

        # Execute schema creation
        cursor.execute(devices_sql)
        cursor.execute(trust_sql)
        cursor.execute(messages_sql)

        # Create indexes
        for index_sql in indexes_sql:
            cursor.execute(index_sql)

        self.connection.commit()

    def _init_domaindoc_schema(self):
        """Initialize Domaindoc database schema for section-aware storage."""
        cursor = self.connection.cursor()

        # Domain metadata (replaces manifest.json)
        # Note: 'label' is the single identifier - used for both lookups and display
        domaindocs_sql = """
        CREATE TABLE IF NOT EXISTS domaindocs (
            id INTEGER PRIMARY KEY,
            label TEXT UNIQUE NOT NULL,
            encrypted__description TEXT,
            enabled BOOLEAN DEFAULT TRUE,
            archived BOOLEAN DEFAULT FALSE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """

        # Section-level storage (supports one level of nesting via parent_section_id)
        sections_sql = """
        CREATE TABLE IF NOT EXISTS domaindoc_sections (
            id INTEGER PRIMARY KEY,
            domaindoc_id INTEGER NOT NULL REFERENCES domaindocs(id) ON DELETE CASCADE,
            parent_section_id INTEGER DEFAULT NULL REFERENCES domaindoc_sections(id) ON DELETE CASCADE,
            header TEXT NOT NULL,
            encrypted__content TEXT NOT NULL,
            encrypted__summary TEXT DEFAULT NULL,
            sort_order INTEGER NOT NULL,
            collapsed BOOLEAN DEFAULT FALSE,
            expanded_by_default BOOLEAN DEFAULT FALSE,
            pinned BOOLEAN DEFAULT FALSE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(domaindoc_id, parent_section_id, header)
        )
        """

        # Version history (operation diffs)
        versions_sql = """
        CREATE TABLE IF NOT EXISTS domaindoc_versions (
            id INTEGER PRIMARY KEY,
            domaindoc_id INTEGER NOT NULL REFERENCES domaindocs(id) ON DELETE CASCADE,
            section_id INTEGER REFERENCES domaindoc_sections(id) ON DELETE SET NULL,
            version_num INTEGER NOT NULL,
            operation TEXT NOT NULL,
            encrypted__diff_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(domaindoc_id, version_num)
        )
        """

        # Create indexes
        indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_domaindocs_label ON domaindocs(label)",
            "CREATE INDEX IF NOT EXISTS idx_domaindocs_enabled ON domaindocs(enabled)",
            "CREATE INDEX IF NOT EXISTS idx_domaindocs_archived ON domaindocs(archived)",
            "CREATE INDEX IF NOT EXISTS idx_sections_domaindoc ON domaindoc_sections(domaindoc_id)",
            "CREATE INDEX IF NOT EXISTS idx_sections_collapsed ON domaindoc_sections(domaindoc_id, collapsed)",
            "CREATE INDEX IF NOT EXISTS idx_sections_parent ON domaindoc_sections(parent_section_id)",
            "CREATE INDEX IF NOT EXISTS idx_versions_domaindoc ON domaindoc_versions(domaindoc_id)",
        ]

        # Execute schema creation
        cursor.execute(domaindocs_sql)
        cursor.execute(sections_sql)
        cursor.execute(versions_sql)

        # Run migrations for existing databases (must precede index creation
        # since indexes may reference columns added by migrations)
        self._migrate_domaindoc_schema(cursor)

        # Create indexes
        for index_sql in indexes_sql:
            cursor.execute(index_sql)

        self.connection.commit()

    def _migrate_domaindoc_schema(self, cursor: sqlite3.Cursor) -> None:
        """Run schema migrations for domaindoc tables.

        Handles adding new columns to existing databases and setting
        appropriate defaults for existing data. Must run BEFORE index
        creation since indexes may reference columns added here.
        """
        # Migrate domaindocs table
        cursor.execute("PRAGMA table_info(domaindocs)")
        domaindoc_columns = {row[1] for row in cursor.fetchall()}

        if "archived" not in domaindoc_columns:
            cursor.execute(
                "ALTER TABLE domaindocs ADD COLUMN archived BOOLEAN DEFAULT FALSE"
            )
            self.connection.commit()
            logger.info("Migrated domaindocs: added archived column")

        # Migrate domaindoc_sections table
        cursor.execute("PRAGMA table_info(domaindoc_sections)")
        section_columns = {row[1] for row in cursor.fetchall()}

        if "pinned" not in section_columns:
            cursor.execute(
                "ALTER TABLE domaindoc_sections ADD COLUMN pinned BOOLEAN DEFAULT FALSE"
            )
            # Migrate: set pinned=TRUE for top-level sections that were previously
            # always-expanded (sort_order=0 with no parent)
            cursor.execute(
                "UPDATE domaindoc_sections SET pinned = TRUE "
                "WHERE sort_order = 0 AND parent_section_id IS NULL"
            )
            self.connection.commit()
            logger.info("Migrated domaindoc_sections: added pinned column")

    def _init_trigger_rules_schema(self):
        """Initialize trigger_rules table for per-user sidebar trigger filters.

        Each row is a filter rule scoped to a specific trigger type. The trigger
        reads its own rules by filtering on trigger_id. Field names and scope
        semantics are trigger-specific (e.g. IMAP: scope=folder, field=from/subject/body;
        a future Slack trigger: scope=channel, field=author/text).
        """
        cursor = self.connection.cursor()

        rules_sql = """
        CREATE TABLE IF NOT EXISTS trigger_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger_id TEXT NOT NULL,
            scope TEXT NOT NULL DEFAULT 'INBOX',
            field TEXT NOT NULL,
            pattern TEXT NOT NULL,
            prompt TEXT,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
        cursor.execute(rules_sql)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trigger_rules_trigger "
            "ON trigger_rules(trigger_id, enabled)"
        )
        self.connection.commit()

    def _encrypt_value(self, value: Any) -> str:
        """Encrypt a value for storage."""
        json_str = json.dumps(value)
        encrypted_token = self.fernet.encrypt(json_str.encode())
        return encrypted_token.decode()  # Fernet returns base64-encoded bytes
    
    def _decrypt_value(self, encrypted_str: str) -> Any:
        """Decrypt a value from storage."""
        try:
            decrypted_bytes = self.fernet.decrypt(encrypted_str.encode())
            return json.loads(decrypted_bytes.decode())
        except Exception:
            # Fallback for unencrypted data (migration scenario)
            try:
                return json.loads(encrypted_str)
            except json.JSONDecodeError:
                return encrypted_str
    
    def _encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt fields with encrypted__ prefix (prefix is kept in column name)."""
        result = {}
        for key, value in data.items():
            if key.startswith('encrypted__'):
                # Encrypt value but keep the encrypted__ prefix in the column name
                result[key] = self._encrypt_value(value) if value is not None else None
            elif value is None:
                result[key] = None
            elif isinstance(value, bool):
                # SQLite stores booleans as integers (1/0)
                result[key] = 1 if value else 0
            elif isinstance(value, (int, float)):
                # Keep numeric types as-is for proper SQL comparisons
                result[key] = value
            else:
                # Convert other types to string
                result[key] = str(value)
        return result
    
    def _decrypt_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Decrypt fields prefixed with encrypted__."""
        result = {}
        for key, value in data.items():
            if key.startswith("encrypted__") and value is not None:
                try:
                    result[key] = self._decrypt_value(value)
                except Exception:
                    # Decryption failed, use as-is
                    result[key] = value
            else:
                result[key] = value
        return result
    
    def execute(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute SQL query and return results."""
        cursor = self.connection.cursor()
        cursor.execute(query, params or {})

        if cursor.description:
            # SELECT query - return rows as dicts
            results = [dict(row) for row in cursor.fetchall()]
            # Close any implicit read transaction to ensure fresh reads
            self.connection.commit()
            return results
        else:
            # INSERT/UPDATE/DELETE - commit and return empty list
            self.connection.commit()
            return []

    def fetchone(self, query: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Execute query and return single row."""
        results = self.execute(query, params)
        return results[0] if results else None

    def fetchall(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute query and return all rows."""
        return self.execute(query, params)

    def create_table(self, table_name: str, schema: str):
        """Create table with given schema."""
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        cursor = self.connection.cursor()
        cursor.execute(query)
        self.connection.commit()

    def insert(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert encrypted data and return row ID."""
        encrypted_data = self._encrypt_dict(data)
        columns = list(encrypted_data.keys())
        placeholders = [f":{col}" for col in columns]
        query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        cursor = self.connection.cursor()
        cursor.execute(query, encrypted_data)
        self.connection.commit()
        return str(cursor.lastrowid)

    def select(self, table_name: str, where: str = None, params: Optional[Dict] = None, order_by: str = None) -> List[Dict[str, Any]]:
        """Select and decrypt rows from table."""
        query = f"SELECT * FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        rows = self.fetchall(query, params)
        return [self._decrypt_dict(row) for row in rows]

    def update(self, table_name: str, data: Dict[str, Any], where: str, params: Optional[Dict] = None) -> int:
        """Update rows with encrypted data."""
        encrypted_data = self._encrypt_dict(data)
        set_clauses = [f"{col} = :{col}" for col in encrypted_data.keys()]
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where}"

        all_params = encrypted_data.copy()
        if params:
            all_params.update(params)

        cursor = self.connection.cursor()
        cursor.execute(query, all_params)
        self.connection.commit()
        return cursor.rowcount

    def delete(self, table_name: str, where: str, params: Optional[Dict] = None) -> int:
        """Delete rows from table."""
        query = f"DELETE FROM {table_name} WHERE {where}"
        cursor = self.connection.cursor()
        cursor.execute(query, params or {})
        self.connection.commit()
        return cursor.rowcount
    
    def get_tool_data_dir(self, tool_name: str) -> Path:
        path = self.base_dir / "tools" / tool_name
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _ensure_credentials_table(self):
        """Helper method to ensure credentials table exists (used by UserCredentialService)."""
        schema = """
            id TEXT PRIMARY KEY,
            credential_type TEXT NOT NULL,
            service_name TEXT NOT NULL,
            encrypted__credential_value TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(credential_type, service_name)
        """
        self.create_table('credentials', schema)


def derive_session_key(user_id: str) -> bytes:
    """Derive persistent encryption key from user ID."""
    import hashlib
    # Create deterministic key from user UUID
    # This key remains constant for the user's lifetime
    key_material = f"userdata_encryption_{user_id}".encode()
    return hashlib.sha256(key_material).digest()


def get_user_data_manager(user_id: UUID) -> UserDataManager:
    """
    Get a cached UserDataManager instance with automatic encryption key derivation.

    Instances are cached per-user for connection reuse across operations.
    Use clear_manager_cache() to explicitly close connections when needed.
    """
    cache_key = str(user_id)

    if cache_key not in _manager_cache:
        session_key = derive_session_key(cache_key)
        _manager_cache[cache_key] = UserDataManager(user_id, session_key)
        logger.debug(f"Created new UserDataManager for user {user_id}")

    return _manager_cache[cache_key]


def clear_manager_cache(user_id: Optional[UUID] = None) -> None:
    """
    Clear cached UserDataManager instances and close their connections.

    Args:
        user_id: If provided, only clear the cache for this user.
                 If None, clear all cached managers.
    """
    global _manager_cache

    if user_id is not None:
        cache_key = str(user_id)
        if cache_key in _manager_cache:
            _manager_cache[cache_key].close()
            del _manager_cache[cache_key]
            logger.debug(f"Cleared UserDataManager cache for user {user_id}")
    else:
        for manager in _manager_cache.values():
            manager.close()
        _manager_cache.clear()
        logger.info("Cleared all UserDataManager caches")


class UserDataManagerCleanupHandler:
    """
    Handles cleanup of cached UserDataManager instances on session collapse.

    Subscribes to SegmentCollapsedEvent and closes the user's SQLite connection
    when their conversation session collapses, freeing resources.
    """

    def __init__(self, event_bus: 'EventBus'):
        self.event_bus = event_bus
        self.event_bus.subscribe('SegmentCollapsedEvent', self._handle_segment_collapsed)
        logger.info("UserDataManagerCleanupHandler subscribed to SegmentCollapsedEvent")

    def _handle_segment_collapsed(self, event) -> None:
        """Close user's SQLite connection when their session collapses."""
        try:
            user_id = UUID(event.user_id)
            clear_manager_cache(user_id)
            logger.debug(f"Closed UserDataManager for user {event.user_id} after segment collapse")
        except Exception as e:
            # Don't let cleanup failures break the collapse pipeline
            logger.warning(f"Failed to clean up UserDataManager for user {event.user_id}: {e}")

