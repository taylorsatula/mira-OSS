"""
PostgreSQL client with explicit connection management and dict-style results.

General-purpose Postgres client that can be used anywhere in the application.
Replaces SQLAlchemy with raw SQL for performance and explicit memory management.
"""

from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout
from pgvector.psycopg import register_vector
import logging
import threading
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, TypedDict, Union, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PoolStats(TypedDict):
    """Stats for a single connection pool."""
    min_size: int
    max_size: int
    closed: bool


class PostgresClient:
    """Raw SQL client with connection pooling, user isolation via RLS, and automatic JSON serialization."""

    # Class-level connection pools shared across all instances
    _connection_pools: Dict[str, ConnectionPool] = {}
    _pools_lock = threading.RLock()  # Thread safety for pool creation

    @classmethod
    def reset_all_pools(cls):
        """
        Reset all connection pools. Used primarily for testing.

        Properly closes all connections before clearing pools to avoid
        leaving dangling database connections.
        """
        with cls._pools_lock:
            # Close sync pools
            for pool in cls._connection_pools.values():
                try:
                    pool.close(timeout=0)  # Workers are daemon threads — no need to wait
                except Exception as e:
                    logger.warning(f"Error closing sync pool: {e}", exc_info=True)
            cls._connection_pools.clear()

            logger.debug("All PostgresClient connection pools reset")

    def __init__(self, database_name: str, user_id: Optional[str] = None, admin: bool = False):
        self.database_name = database_name
        self.user_id = user_id
        self._admin = admin
        # Admin connections use a separate pool with BYPASSRLS role
        self._pool_key = f"{database_name}_admin" if admin else database_name
        from clients.vault_client import get_database_url
        self._conninfo = self._parse_database_url(get_database_url(database_name, admin=admin))
        self._ensure_connection_pool()

    @staticmethod
    def _parse_database_url(url: str) -> str:
        """Parse a database URL into a properly escaped psycopg3 conninfo string.

        psycopg3 is stricter than psycopg2 about special characters (spaces,
        parens, etc.) in connection URIs. Decomposing into keyword args and
        rebuilding via make_conninfo() lets psycopg3 handle its own escaping.
        """
        parsed = urlparse(url)
        return make_conninfo(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            dbname=parsed.path.lstrip('/'),
            user=parsed.username,
            password=parsed.password,
        )

    def _needs_vector(self) -> bool:
        """mira_service (any instance) stores embeddings/vectors."""
        from utils.instance import database_name
        return self.database_name == database_name()

    def _ensure_connection_pool(self):
        """Ensure connection pool exists for this database."""
        with self._pools_lock:
            if self._pool_key not in self._connection_pools:
                try:
                    pool = ConnectionPool(
                        conninfo=self._conninfo,
                        min_size=3,
                        max_size=30,
                        timeout=30,
                        max_lifetime=3600,  # Recycle connections after 1 hour
                        max_idle=300        # Close idle connections after 5 minutes
                    )

                    # Register pgvector type only for databases that need it
                    if self._needs_vector():
                        for i in range(pool.min_size):
                            conn = pool.getconn()
                            register_vector(conn)
                            pool.putconn(conn)

                    self._connection_pools[self._pool_key] = pool
                    logger.toast(f"Connection pool created: {self._pool_key} (min={pool.min_size}, max={pool.max_size})")

                    if self._needs_vector():
                        logger.toast(f"pgvector registered for {self._pool_key}")

                except Exception as e:
                    logger.error(f"Connection pool creation failed for {self._pool_key}: {e}", exc_info=True)
                    raise

    @contextmanager
    def get_connection(self):
        """Gets pooled connection and sets app.current_user_id for Row Level Security.

        Admin connections (admin=True) bypass RLS entirely via the mira_admin role
        and never set user context. Used for cross-user billing/admin operations.
        """
        # Ensure pool exists (thread-safe check)
        if self._pool_key not in self._connection_pools:
            self._ensure_connection_pool()
        pool = self._connection_pools[self._pool_key]
        conn = None
        try:
            conn = pool.getconn()
            if conn is None:
                raise Exception(f"Could not get connection from pool for {self._pool_key}")

            conn.autocommit = True

            if self._needs_vector():
                register_vector(conn)

            # Admin connections bypass RLS via role — no user context needed
            if not self._admin:
                # ALWAYS set or clear the user context to prevent inheriting from pooled connections
                with conn.cursor() as cur:
                    if self.user_id:
                        cur.execute("SELECT set_config('app.current_user_id', %s, false)", (str(self.user_id),))
                    else:
                        # Clear any previous user context to prevent data leaks
                        cur.execute("SELECT set_config('app.current_user_id', '', false)")
            yield conn
        except PoolTimeout as e:
            logger.error(f"Connection pool exhausted for {self._pool_key}: {e}", exc_info=True)
            raise Exception(f"Database connection pool exhausted for {self._pool_key}")
        finally:
            if conn:
                pool.putconn(conn)




    def execute_query(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> List[Dict]:
        """Execute a SELECT query and return rows as list of dictionaries."""
        with self.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                if cur.description:
                    return cur.fetchall()
                return []

    def execute_returning(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> List[Dict]:
        with self.get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def execute_single(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Optional[Dict]:
        """Execute a query and return the first row as a dictionary or None."""
        results = self.execute_query(query, params)
        return results[0] if results else None

    def execute_scalar(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> Any:
        """Execute a query and return the first value of the first row."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                result = cur.fetchone()
                return result[0] if result else None

    def execute_insert(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> None:
        """Execute a single INSERT query."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

    def execute_update(self, query: str, params: Optional[Union[Dict, Tuple]] = None) -> int:
        """Execute an UPDATE query and return the number of affected rows."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount

    def execute_transaction(self, operations: List[Tuple[str, Optional[Union[Dict, Tuple]]]]) -> List[Any]:
        """Executes multiple operations atomically - all succeed or all rollback."""
        with self.get_connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:
                    results = []
                    for query, params in operations:
                        cur.execute(query, params)
                        if cur.description:
                            results.append(cur.fetchall())
                        else:
                            results.append(cur.rowcount)
                    return results

    @classmethod
    def get_pool_stats(cls) -> Dict[str, PoolStats]:
        """Get connection pool configuration stats for diagnostics."""
        stats: Dict[str, PoolStats] = {}
        with cls._pools_lock:
            for pool_key, pool in cls._connection_pools.items():
                stats[pool_key] = PoolStats(
                    min_size=pool.min_size,
                    max_size=pool.max_size,
                    closed=pool.closed,
                )
        return stats

    @classmethod
    def close_all_pools(cls):
        for db_name, pool in cls._connection_pools.items():
            pool.close(timeout=0)  # Workers are daemon threads — no need to wait
            logger.toast(f"Connection pool closed: {db_name}")
        cls._connection_pools.clear()
