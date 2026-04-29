"""
MIRA instance identity.

Reads MIRA_INSTANCE from environment once at import time.
All instance-specific names (database, Vault paths, data dir, Valkey prefix)
derive from this single value. When unset or "default", all names match
the original hardcoded values for backwards compatibility.
"""
import os
from pathlib import Path


def _read_instance_name() -> str:
    raw = os.environ.get("MIRA_INSTANCE", "default")
    return raw.strip().lower()


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
