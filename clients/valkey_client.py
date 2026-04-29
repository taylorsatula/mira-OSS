"""
Unified Valkey client for system-wide use.

Combines auth operations (async) and working memory operations (sync).
Uses Vault for configuration with fail-fast semantics.
"""

import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Global connection pool (singleton like PostgresClient)
_valkey_pool = None


class ValkeyClient:
    """
    Production Valkey client with fail-fast semantics for required infrastructure.

    System fails to start if Valkey is unreachable. Operations fail loudly if
    Valkey goes down during runtime. Features connection pooling and retry
    patterns for transient failures.
    """

    def __init__(self):
        # Get configuration and initialize connections
        # System fails to start if Valkey is unreachable (fail-fast)
        self._load_config()
        self._init_connections()

        from utils.instance import valkey_prefix
        self._prefix = valkey_prefix()

    def _key(self, key: str) -> str:
        """Apply instance prefix to key. No-op for default instance."""
        return f"{self._prefix}{key}" if self._prefix else key

    def _load_config(self):
        """Load Valkey URL and password from Vault. Raises if Vault is unavailable."""
        from urllib.parse import urlparse
        from clients.vault_client import get_service_config
        self.valkey_url = get_service_config('valkey_url')
        try:
            self.password = get_service_config('valkey_password')
        except KeyError:
            self.password = None
            logger.toast("Valkey running without authentication")

        parsed = urlparse(self.valkey_url)
        self.host = parsed.hostname or 'localhost'
        self.port = parsed.port or 6379
        logger.debug(f"Valkey config loaded from Vault: {self.host}:{self.port}")

    def _init_connections(self):
        """
        Initialize Valkey connections with fail-fast semantics.

        Raises if Valkey is unreachable - system should not start without Valkey.
        """
        global _valkey_pool

        import valkey
        import valkey.asyncio as async_valkey
        from valkey.connection import ConnectionPool

        conn_params = {
            'host': self.host,
            'port': self.port,
            'password': self.password,
            'decode_responses': True,
            'health_check_interval': 30,
            'socket_keepalive': True,
            'socket_keepalive_options': {},
            'retry_on_timeout': True,
            'socket_connect_timeout': 5,
            'socket_timeout': 5
        }

        # Create thread-safe connection pool (like PostgresClient)
        if _valkey_pool is None:
            _valkey_pool = ConnectionPool(
                max_connections=20,
                **conn_params
            )
            logger.toast(f"Valkey connection pool created: {self.host}:{self.port} (max=20)")

        # Use pooled connections
        self._client = valkey.Valkey(connection_pool=_valkey_pool)
        self._client.ping()  # Raises if unreachable - system fails to start

        # Create binary client using same connection params but without decode_responses
        binary_conn_params = conn_params.copy()
        binary_conn_params['decode_responses'] = False
        self._binary_client = valkey.Valkey(**binary_conn_params)

        # Build URL for async client (with auth if password is set)
        if self.password:
            from urllib.parse import quote
            auth_url = f"valkey://:{quote(self.password, safe='')}@{self.host}:{self.port}"
        else:
            auth_url = f"valkey://{self.host}:{self.port}"
        self.valkey = async_valkey.from_url(
            auth_url,
            encoding="utf-8",
            **{k: v for k, v in conn_params.items() if k not in ['host', 'port', 'password']}
        )

        logger.toast(f"Valkey client initialized: {self.host}:{self.port}")

    @property
    def valkey_available(self) -> bool:
        """
        Check if Valkey is available.

        Always returns True - if ValkeyClient initialized successfully, Valkey is available.
        ValkeyClient fails-fast at initialization if Valkey is unreachable.

        This property exists for test compatibility.
        """
        return True

    @property
    def valkey_binary(self) -> "valkey.Valkey":
        """
        Access the binary client for storing raw bytes (e.g., embeddings).

        The binary client has decode_responses=False, preserving bytes as-is
        without UTF-8 decoding. Used for numpy arrays and other binary data.
        """
        return self._binary_client

    def get(self, key: str) -> Optional[str]:
        """
        Get value by key.

        Returns None if key doesn't exist.
        Raises if Valkey connection fails.
        """
        return self._client.get(self._key(key))

    def delete(self, key: str) -> bool:
        """
        Delete key.

        Returns True if key was deleted, False if key didn't exist.
        Raises if Valkey connection fails.
        """
        result = self._client.delete(self._key(key))
        return result > 0

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Returns True if key exists, False if not.
        Raises if Valkey connection fails.
        """
        result = self._client.exists(self._key(key))
        return result > 0

    def ttl(self, key: str) -> int:
        """
        Get TTL of a key in seconds.

        Returns -1 if key has no TTL, -2 if key doesn't exist.
        Raises if Valkey connection fails.
        """
        return self._client.ttl(self._key(key))

    def scan(self, cursor: int, match: Optional[str] = None, count: int = 100) -> tuple[int, list[str]]:
        """
        Scan keys matching pattern.

        Returns (next_cursor, keys) tuple.
        Raises if Valkey connection fails.
        """
        if match and self._prefix:
            match = f"{self._prefix}{match}"
        elif self._prefix:
            match = f"{self._prefix}*"
        next_cursor, keys = self._client.scan(cursor, match=match, count=count)
        if self._prefix:
            keys = [k[len(self._prefix):] for k in keys]
        return next_cursor, keys

    def increment_with_expiry(self, key: str, expiry_seconds: int) -> int:
        """Atomic increment with expiration - ideal for rate limiting."""
        prefixed = self._key(key)
        count = self._client.incr(prefixed)

        # Only set expiry if this is the first increment (count == 1)
        # This prevents resetting the TTL window on every request
        if count == 1:
            self._client.expire(prefixed, expiry_seconds)

        return count

    def json_set(self, key: str, path: str, value: Dict[str, Any], ex: Optional[int] = None) -> bool:
        """Set JSON data, optionally with expiration.

        When path is "$", replaces entire value. When path is "$.field_name",
        updates just that field in existing JSON (read-modify-write).

        Args:
            key: The key to store JSON data
            path: "$" for full replacement, "$.field_name" for field update
            value: Value to set
            ex: Expiration in seconds. If None, preserves existing TTL (for updates)
                or uses no expiration (for new keys)

        Returns:
            True if successful, False if key doesn't exist (for field updates)
        """
        prefixed = self._key(key)
        if path == "$":
            # Full replacement
            json_data = json.dumps(value)
            if ex is not None:
                return self._client.setex(prefixed, ex, json_data)
            else:
                return self._client.set(prefixed, json_data)

        # Field update: read-modify-write
        if not path.startswith("$."):
            raise ValueError(f"Unsupported path: {path}. Use '$' or '$.field_name'")

        # self.json_get and self.ttl apply prefix internally — pass bare key
        current = self.json_get(key, "$")
        if current is None:
            return False

        data = current[0]
        field = path[2:]
        data[field] = value

        # Preserve existing TTL if no expiry specified
        if ex is None:
            remaining_ttl = self.ttl(key)
            ex = remaining_ttl if remaining_ttl > 0 else None

        json_data = json.dumps(data)
        if ex is not None:
            return self._client.setex(prefixed, ex, json_data)
        else:
            return self._client.set(prefixed, json_data)

    def json_set_with_expiry(self, key: str, path: str, value: Dict[str, Any], ex: int) -> bool:
        """Set JSON data with expiration. Alias for json_set with required ex."""
        return self.json_set(key, path, value, ex=ex)

    def json_get(self, key: str, path: str) -> Optional[list[Dict[str, Any]]]:
        """Get JSON data (returns list format for JSONPath compatibility)."""
        json_str = self._client.get(self._key(key))
        if json_str is None:
            return None

        data = json.loads(json_str)
        return [data]  # Wrap in list to match expected JSONPath result format

    def hset_with_retry(self, hash_key: str, field: str, value: str) -> int:
        """Hash set with retry pattern for transient failures."""
        prefixed = self._key(hash_key)
        try:
            return self._client.hset(prefixed, field, value)
        except Exception as e:
            logger.warning(f"Valkey write failed, retrying: {e}", exc_info=True)

        time.sleep(0.1)
        return self._client.hset(prefixed, field, value)  # Raises on failure

    def hget_with_retry(self, hash_key: str, field: str) -> Optional[str]:
        """Hash get with retry pattern for transient failures."""
        prefixed = self._key(hash_key)
        try:
            return self._client.hget(prefixed, field)
        except Exception as e:
            logger.warning(f"Valkey read failed, retrying: {e}", exc_info=True)

        time.sleep(0.1)
        return self._client.hget(prefixed, field)

    def hdel_with_retry(self, hash_key: str, *fields) -> int:
        """Hash delete with retry pattern for transient failures."""
        prefixed = self._key(hash_key)
        try:
            return self._client.hdel(prefixed, *fields)
        except Exception as e:
            logger.warning(f"Valkey delete failed, retrying: {e}", exc_info=True)

        time.sleep(0.1)
        return self._client.hdel(prefixed, *fields)

    def set(self, key: str, value: str, nx: Optional[bool] = None, ex: Optional[int] = None) -> bool:
        """
        Set key to value.

        Args:
            key: Key to set
            value: Value to store
            nx: If True, only set if key doesn't exist
            ex: Expiration time in seconds

        Returns:
            True if key was set, False if NX was specified and key already exists

        Raises if Valkey connection fails.
        """
        kwargs = {}
        if nx is not None:
            kwargs['nx'] = nx
        if ex is not None:
            kwargs['ex'] = ex
        return self._client.set(self._key(key), value, **kwargs)

    def scan_iter(self, match: Optional[str] = None) -> Iterator[str]:
        """Scan iterator for keys matching pattern."""
        if match and self._prefix:
            match = f"{self._prefix}{match}"
        elif self._prefix:
            match = f"{self._prefix}*"
        for key in self._client.scan_iter(match=match):
            if self._prefix:
                yield key[len(self._prefix):]
            else:
                yield key

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on key."""
        return self._client.expire(self._key(key), seconds)

    def setex(self, key: str, seconds: int, value: str) -> bool:
        """Set key with expiration."""
        return self._client.setex(self._key(key), seconds, value)

    def rpush(self, key: str, *values: str) -> int:
        """
        Append values to the end of a list.

        Creates the list if it doesn't exist.

        Args:
            key: List key
            *values: Values to append

        Returns:
            Length of list after push

        Raises if Valkey connection fails.
        """
        return self._client.rpush(self._key(key), *values)

    def lrange(self, key: str, start: int, stop: int) -> list[str]:
        """
        Get a range of elements from a list.

        Args:
            key: List key
            start: Start index (0-based, negative from end)
            stop: Stop index (inclusive, -1 means end)

        Returns:
            List of elements in range, empty list if key doesn't exist

        Raises if Valkey connection fails.
        """
        return self._client.lrange(self._key(key), start, stop)

    def flush_except_whitelist(self, preserve_prefixes: list[str]) -> int:
        """
        Delete all keys from Valkey except those matching whitelist prefixes.

        Used during system startup to clear all caches while preserving
        critical data like auth sessions and rate limiting.

        Args:
            preserve_prefixes: List of key prefixes to preserve (e.g., ["session:", "rate_limit:"])

        Returns:
            Number of keys deleted
        """
        if any(prefix == "" for prefix in preserve_prefixes):
            raise ValueError("Empty string in preserve_prefixes would preserve all keys")

        deleted_count = 0
        preserved_count = 0

        # When prefixed, only scan our instance's keys
        scan_pattern = f"{self._prefix}*" if self._prefix else "*"

        for key in self._client.scan_iter(match=scan_pattern):
            # Strip prefix to check against preserve list
            bare_key = key[len(self._prefix):] if self._prefix else key
            should_preserve = any(bare_key.startswith(prefix) for prefix in preserve_prefixes)

            if should_preserve:
                preserved_count += 1
            else:
                if self._client.delete(key):
                    deleted_count += 1

        logger.toast(
            f"Flushed {deleted_count} keys from Valkey, "
            f"preserved {preserved_count} keys matching whitelist: {preserve_prefixes}"
        )
        return deleted_count

    def shutdown(self):
        """Clean shutdown of Valkey client."""
        logger.toast("Valkey client shutdown complete")


_valkey_client: Optional[ValkeyClient] = None


def get_valkey_client() -> ValkeyClient:
    global _valkey_client
    if _valkey_client is None:
        _valkey_client = ValkeyClient()
    return _valkey_client


def get_valkey() -> ValkeyClient:
    """Get Valkey client instance."""
    return get_valkey_client()
