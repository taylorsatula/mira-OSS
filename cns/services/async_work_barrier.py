"""
Global async work barrier for cross-turn coordination.

Provides a shared mechanism for background tasks to signal pending work
per user, and a single gate that blocks until ALL registered tasks complete.

Usage:
    # At startup, initialize once via initialize_async_work_barrier()

    # Async producer (e.g., TurnCompletedEvent handler):
    barrier = get_async_work_barrier()
    done = barrier.register_work(user_id)
    threading.Thread(target=lambda: (do_work(), done())).start()

    # Consumer (e.g., API endpoint before next turn):
    barrier = get_async_work_barrier()
    barrier.wait_for_user(user_id)  # blocks until all producers call done()
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class _UserLatch:
    """Per-user counter + event. Counts down from N registered tasks to 0."""

    __slots__ = ("_count", "_event", "_lock")

    def __init__(self) -> None:
        self._count = 0
        self._event = threading.Event()
        self._event.set()  # start in "clear" state
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._count += 1
            if self._count == 1:
                # First task — clear the event so waiters block
                self._event.clear()

    def decrement(self) -> None:
        with self._lock:
            self._count -= 1
            if self._count <= 0:
                self._count = 0
                self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout=timeout)


class AsyncWorkBarrier:
    """
    Global registry of per-user async work with a unified wait gate.

    Thread-safe. Designed for single-process deployment where all turns
    share the same Python process.
    """

    def __init__(self) -> None:
        self._latches: dict[str, _UserLatch] = {}
        self._lock = threading.Lock()

    def _get_latch(self, user_id: str) -> _UserLatch:
        with self._lock:
            if user_id not in self._latches:
                self._latches[user_id] = _UserLatch()
            return self._latches[user_id]

    def register_work(self, user_id: str) -> Callable[[], None]:
        """
        Register a unit of async work for a user.

        Returns a callback that the caller must invoke exactly once when
        the work completes (success or failure). Typical pattern:

            done = barrier.register_work(user_id)
            try:
                do_the_work()
            finally:
                done()

        Args:
            user_id: The user this work belongs to.

        Returns:
            A zero-argument callable to signal completion.
        """
        latch = self._get_latch(user_id)
        latch.increment()
        done_lock = threading.Lock()
        completed = False

        def done() -> None:
            nonlocal completed
            with done_lock:
                if completed:
                    return
                completed = True
            latch.decrement()

        return done

    def wait_for_user(
        self,
        user_id: str,
        timeout: float | None = None,
        source: str = "",
    ) -> bool:
        """
        Block until all registered async work for this user completes.

        Args:
            user_id: The user to wait for.
            timeout: Max seconds to wait. None = wait forever.
                     Recommended: 30s with fail-open logging.
            source: Label for log messages (e.g., "tool_summarizer", "portrait_synthesis").

        Returns:
            True if all work completed, False if timed out.
        """
        with self._lock:
            latch = self._latches.get(user_id)

        if latch is None:
            return True  # no work ever registered for this user

        label = f" ({source})" if source else ""
        completed = latch.wait(timeout=timeout)
        if not completed and timeout is not None:
            logger.warning(
                "Async work barrier timed out for user %s after %.1fs%s — proceeding",
                user_id, timeout, label,
            )
        return completed


# Global instance
_barrier: AsyncWorkBarrier | None = None


def initialize_async_work_barrier() -> AsyncWorkBarrier:
    """Initialize the global async work barrier. Call once at startup."""
    global _barrier
    _barrier = AsyncWorkBarrier()
    logger.info("Async work barrier initialized")
    return _barrier


def get_async_work_barrier() -> AsyncWorkBarrier:
    """Get the global async work barrier instance."""
    global _barrier
    if _barrier is None:
        raise RuntimeError(
            "Async work barrier not initialized. "
            "Call initialize_async_work_barrier() during startup."
        )
    return _barrier
