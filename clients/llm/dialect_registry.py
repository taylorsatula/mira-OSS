"""Filesystem-as-contract registry for provider-specific Dialect classes.

The registry walks `clients.llm.dialects/`, imports each module, and registers
any concrete Dialect subclass whose `dialect_name` is a non-base member of
DialectName. Adding a new provider becomes a single-file operation: drop a
module into the package, declare the class, and the registry picks it up at
startup.

Discovery validates each candidate (fail-loud) — a malformed dialect class
raises at startup rather than at first request, so misconfiguration is caught
during boot rather than under load.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import threading
from typing import get_args

from clients.llm.dialects.base import Dialect
from clients.llm.types import DialectName, NativeField

logger = logging.getLogger(__name__)


def _validate_dialect_class(cls: type[Dialect]) -> None:
    """Raise loudly if a Dialect subclass is malformed.

    Called by DialectRegistry.discover() for each candidate. The intent is to
    catch dialect-author mistakes at startup rather than at first request.
    """
    name = cls.dialect_name
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Dialect {cls.__name__}: dialect_name must be a non-empty string"
        )
    if name == "base":
        raise ValueError(
            f"Dialect {cls.__name__}: dialect_name 'base' is reserved"
        )
    if name not in get_args(DialectName):
        raise ValueError(
            f"Dialect {cls.__name__}: dialect_name {name!r} is not a member "
            f"of DialectName literal {get_args(DialectName)!r}. Extend the "
            f"DialectName literal in types.py before registering this dialect."
        )
    if not isinstance(cls.native_thinking_fields, tuple):
        raise TypeError(
            f"Dialect {cls.__name__}: native_thinking_fields must be a tuple, "
            f"got {type(cls.native_thinking_fields).__name__}"
        )
    valid_fields = set(get_args(NativeField))
    invalid = [f for f in cls.native_thinking_fields if f not in valid_fields]
    if invalid:
        raise ValueError(
            f"Dialect {cls.__name__}: unknown native_thinking_fields {invalid}. "
            f"Valid values: {sorted(valid_fields)!r}"
        )
    if not isinstance(inspect.getattr_static(cls, "from_selection"), classmethod):
        raise TypeError(
            f"Dialect {cls.__name__}: from_selection must be declared with "
            f"@classmethod, not as an instance method"
        )


class DialectRegistry:
    """In-process registry mapping dialect_name → Dialect subclass."""

    def __init__(self) -> None:
        self._classes: dict[str, type[Dialect]] = {}
        self._discovered = False
        self._lock = threading.Lock()

    def discover(self, package_path: str = "clients.llm.dialects") -> None:
        """Walk the dialects package and register every concrete Dialect subclass.

        Idempotent — re-entry after first discovery is a no-op. Subsequent
        invocations from tests can monkeypatch `_classes` directly for
        isolation.
        """
        if self._discovered:
            return
        with self._lock:
            if self._discovered:
                return
            package = importlib.import_module(package_path)
            for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
                if is_pkg:
                    continue
                module = importlib.import_module(f"{package_path}.{module_name}")
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if not issubclass(obj, Dialect):
                        continue
                    if obj is Dialect:
                        continue
                    if obj.dialect_name == "base":
                        continue
                    if obj.is_abstract:
                        continue
                    if inspect.isabstract(obj):
                        continue
                    if obj.__module__ != module.__name__:
                        # Skip re-exports; only register classes whose home
                        # module is this dialect file.
                        continue
                    _validate_dialect_class(obj)
                    if obj.dialect_name in self._classes:
                        existing = self._classes[obj.dialect_name]
                        if existing is obj:
                            continue
                        raise ValueError(
                            f"Duplicate dialect_name {obj.dialect_name!r}: "
                            f"{existing.__module__}.{existing.__name__} vs "
                            f"{obj.__module__}.{obj.__name__}"
                        )
                    self._classes[obj.dialect_name] = obj
                    logger.debug(
                        "Registered dialect %r → %s.%s",
                        obj.dialect_name, obj.__module__, obj.__name__,
                    )
            self._discovered = True

    def get(self, dialect_name: str) -> type[Dialect]:
        self.discover()
        if dialect_name not in self._classes:
            raise ValueError(
                f"Unknown dialect: {dialect_name!r}. Registered: {sorted(self._classes)!r}"
            )
        return self._classes[dialect_name]


_registry: DialectRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> DialectRegistry:
    """Return the module-level DialectRegistry, lazy-initializing on first call."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DialectRegistry()
    return _registry
