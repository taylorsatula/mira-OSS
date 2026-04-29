"""
MIRA instance identity.

Reads MIRA_INSTANCE from environment once at import time.
All instance-specific names (database, Vault paths, data dir, Valkey prefix)
derive from this single value. When unset or "default", all names match
the original hardcoded values for backwards compatibility.
"""
import os
import re
from pathlib import Path


def _read_instance_name() -> str:
    raw = os.environ.get("MIRA_INSTANCE", "default").strip().lower()
    if not raw:
        return "default"
    if raw != "default" and not re.fullmatch(r'[a-z][a-z0-9_]*', raw):
        raise ValueError(
            f"MIRA_INSTANCE={raw!r} is invalid. "
            "Must start with a letter and contain only lowercase letters, digits, and underscores."
        )
    return raw


INSTANCE_NAME: str = _read_instance_name()


def vault_prefix() -> str:
    if INSTANCE_NAME == "default":
        return "mira"
    return f"mira-{INSTANCE_NAME}"


def database_name() -> str:
    if INSTANCE_NAME == "default":
        return "mira_service"
    return f"mira_service_{INSTANCE_NAME}"


def user_data_base() -> Path:
    if INSTANCE_NAME == "default":
        return Path("data/users")
    return Path(f"data-{INSTANCE_NAME}/users")


def valkey_prefix() -> str:
    if INSTANCE_NAME == "default":
        return ""
    return f"{INSTANCE_NAME}:"
