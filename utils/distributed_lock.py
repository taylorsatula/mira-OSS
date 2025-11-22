"""
Distributed lock implementation using Valkey for multi-process concurrency control.

Provides atomic distributed locks that work across multiple worker processes,
replacing in-memory locks that only work within a single process.
"""

import logging
import uuid
from typing import Optional
from contextlib import contextmanager

from clients.valkey_client import get_valkey

logger = logging.getLogger(__name__)


class DistributedLock:
    """
    Distributed lock using Valkey's atomic SET NX operation.
    
    Ensures only one process can hold a lock for a given resource at a time,
    with automatic expiration to prevent deadlocks from crashed processes.
    """
    
    def __init__(self, lock_prefix: str = "lock:", default_ttl: int = 60):
        """
        Initialize distributed lock manager.
        
        Args:
            lock_prefix: Prefix for lock keys in Valkey
            default_ttl: Default TTL in seconds for locks (prevents deadlocks)
        """
        self.lock_prefix = lock_prefix
        self.default_ttl = default_ttl
        self.valkey = get_valkey()
    
    def acquire(self, resource_id: str, ttl: Optional[int] = None, lock_value: Optional[str] = None) -> bool:
        """
        Attempt to acquire a distributed lock.

        Uses Valkey's atomic SET NX (set if not exists) operation to ensure
        only one process can acquire the lock.

        Args:
            resource_id: Unique identifier for the resource to lock
            ttl: Time-to-live in seconds (uses default if not specified)
            lock_value: Optional value to store with lock (for debugging)

        Returns:
            True if lock was acquired, False if already locked

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        ttl = ttl or self.default_ttl
        lock_value = lock_value or str(uuid.uuid4())

        # SET NX (set if not exists) with EX (expiration)
        # This is atomic - either we get the lock or we don't
        success = self.valkey.set(
            key,
            lock_value,
            nx=True,  # Only set if key doesn't exist
            ex=ttl    # Set expiration time
        )

        if success:
            logger.debug(f"Acquired lock for {resource_id} with TTL {ttl}s")
        else:
            logger.debug(f"Failed to acquire lock for {resource_id} - already locked")

        return bool(success)
    
    def get_lock_owner(self, resource_id: str) -> Optional[str]:
        """
        Get the current owner (value) of a lock.

        Args:
            resource_id: Unique identifier for the resource

        Returns:
            Lock owner value if locked, None if not locked

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        value = self.valkey.get(key)
        return value
    
    def force_release(self, resource_id: str) -> bool:
        """
        Force release a lock regardless of owner.
        
        Use with caution - only for cleaning up stale locks.
        
        Args:
            resource_id: Unique identifier for the resource
            
        Returns:
            True if lock was released, False if lock didn't exist
        """
        logger.warning(f"Force releasing lock for {resource_id}")
        return self.release(resource_id)
    
    def release(self, resource_id: str) -> bool:
        """
        Release a distributed lock.

        Args:
            resource_id: Unique identifier for the resource to unlock

        Returns:
            True if lock was released, False if lock didn't exist

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"

        deleted = self.valkey.delete(key)

        if deleted:
            logger.debug(f"Released lock for {resource_id}")
        else:
            logger.debug(f"No lock to release for {resource_id}")

        return bool(deleted)
    
    def is_locked(self, resource_id: str) -> bool:
        """
        Check if a resource is currently locked.

        Args:
            resource_id: Unique identifier for the resource

        Returns:
            True if resource is locked, False otherwise

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        return self.valkey.exists(key)
    
    def get_ttl(self, resource_id: str) -> int:
        """
        Get remaining TTL for a lock.

        Args:
            resource_id: Unique identifier for the resource

        Returns:
            TTL in seconds, -2 if key doesn't exist, -1 if no TTL set

        Raises:
            Exception: If Valkey is unavailable (infrastructure failure)
        """
        key = f"{self.lock_prefix}{resource_id}"
        return self.valkey.ttl(key)
    
    @contextmanager
    def lock(self, resource_id: str, ttl: Optional[int] = None):
        """
        Context manager for distributed locks.

        Usage:
            with distributed_lock.lock("user_123"):
                # Critical section - only one process can be here
                process_user_request()

        Args:
            resource_id: Unique identifier for the resource to lock
            ttl: Time-to-live in seconds

        Raises:
            LockAcquisitionError: If lock can't be acquired

        Yields:
            None if lock acquired successfully
        """
        acquired = False
        try:
            acquired = self.acquire(resource_id, ttl)
            if not acquired:
                raise LockAcquisitionError(f"Could not acquire lock for {resource_id}")
            yield
        finally:
            if acquired:
                self.release(resource_id)


class LockAcquisitionError(Exception):
    """Raised when a distributed lock cannot be acquired."""
    pass


class UserRequestLock:
    """
    Specialized distributed lock for per-user request concurrency control.
    
    Ensures a user can only have one active chat request at a time across
    all worker processes.
    """
    
    def __init__(self, ttl: int = 60):
        """
        Initialize user request lock.
        
        Args:
            ttl: Lock timeout in seconds (protects against crashes)
        """
        self.lock = DistributedLock(lock_prefix="user_lock:", default_ttl=ttl)
        self.default_ttl = ttl
    
    
    
    def acquire(self, user_id: str) -> bool:
        """
        Attempt to acquire lock for user.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if lock acquired, False if user has concurrent request
        """
        success = self.lock.acquire(user_id, ttl=self.default_ttl)
        if success:
            logger.debug(f"Acquired lock for user {user_id} (TTL: {self.default_ttl}s)")
        else:
            logger.debug(f"Failed to acquire lock for user {user_id} - concurrent request in progress")
        return success
    
    
    def release(self, user_id: str) -> bool:
        """
        Release lock for user.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if lock was released
        """
        return self.lock.release(user_id)
    
    def is_locked(self, user_id: str) -> bool:
        """
        Check if user currently has an active request.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if user has active request
        """
        return self.lock.is_locked(user_id)
    
    @contextmanager
    def lock_user(self, user_id: str):
        """
        Context manager for user request locks.

        Args:
            user_id: User identifier

        Raises:
            LockAcquisitionError: If lock can't be acquired

        Yields:
            None if lock acquired successfully
        """
        with self.lock.lock(user_id):
            yield