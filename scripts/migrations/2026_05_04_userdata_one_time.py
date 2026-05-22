#!/usr/bin/env python3
"""One-time migration: re-encrypt legacy SQLite cells with the Vault-keyed Fernet,
drain plaintext pager columns into their encrypted__* counterparts, and add the
final post-hardening columns.

Replaces the runtime hedges previously living in UserDataManager:
  * legacy-key fallback in `_decrypt_value`
  * `_reencrypt_legacy_encrypted_rows` on every init
  * `_migrate_pager_schema` ALTERs on every init
  * `_ensure_credentials_table` metadata ALTER on every credentials lookup

Idempotent: a second run reports zero work and does not overwrite the .bak file.
Aborts the entire script on the first unrecoverable ciphertext so an operator
investigates the offending cell before any other database is partially mutated.

Usage:
    python scripts/migrations/2026_05_04_userdata_one_time.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import logging
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

# Allow running as a standalone script.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clients.vault_client import get_service_config


BACKUP_SUFFIX = ".bak.20260504"
DATA_USERS_DIR = Path("data/users")


logger = logging.getLogger("migration_2026_05_04")


@dataclass
class UserStats:
    user_id: str
    rows_reencrypted: int = 0
    pager_plaintext_drained: int = 0
    columns_dropped: int = 0
    columns_added: int = 0
    aborted: Optional[str] = None


@dataclass
class Summary:
    users_processed: int = 0
    rows_reencrypted: int = 0
    pager_plaintext_drained: int = 0
    columns_dropped: int = 0
    columns_added: int = 0
    errors: list[str] = field(default_factory=list)


def _legacy_key(user_id: str) -> bytes:
    """Pre-Vault deterministic key derivation. Migration-only; never used at runtime."""
    return hashlib.sha256(f"userdata_encryption_{user_id}".encode()).digest()


def _vault_key(user_id: str, master_key: bytes) -> bytes:
    return hmac.new(master_key, user_id.encode(), hashlib.sha256).digest()


def _build_fernet(raw_key: bytes) -> Fernet:
    return Fernet(base64.urlsafe_b64encode(raw_key[:32]))


def _encrypt(fernet: Fernet, value) -> str:
    return fernet.encrypt(json.dumps(value).encode()).decode()


def _try_decrypt(fernet: Fernet, ciphertext: str):
    return json.loads(fernet.decrypt(ciphertext.encode()).decode())


def _column_names(cursor: sqlite3.Cursor, table: str) -> list[str]:
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _table_exists(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = :name",
        {"name": table},
    )
    return cursor.fetchone() is not None


def _scan_and_reencrypt(
    cursor: sqlite3.Cursor,
    current: Fernet,
    legacy: Fernet,
    user_id: str,
    dry_run: bool,
    stats: UserStats,
) -> None:
    cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    table_names = [row[0] for row in cursor.fetchall()]

    for table_name in table_names:
        encrypted_columns = [
            col for col in _column_names(cursor, table_name)
            if col.startswith("encrypted__")
        ]
        if not encrypted_columns:
            continue

        select_cols = ", ".join(["rowid", *encrypted_columns])
        cursor.execute(f"SELECT {select_cols} FROM {table_name}")
        rows = cursor.fetchall()

        for row in rows:
            updates: dict[str, str] = {}
            for column in encrypted_columns:
                value = row[column]
                if value is None:
                    continue
                # Already migrated.
                try:
                    _try_decrypt(current, value)
                    continue
                except (InvalidToken, ValueError, json.JSONDecodeError):
                    pass

                # Legacy ciphertext.
                try:
                    decrypted = _try_decrypt(legacy, value)
                    updates[column] = _encrypt(current, decrypted)
                    continue
                except (InvalidToken, ValueError, json.JSONDecodeError):
                    pass

                # Plaintext copied from pre-encryption schema (pager content).
                try:
                    decrypted = json.loads(value)
                    updates[column] = _encrypt(current, decrypted)
                    continue
                except json.JSONDecodeError:
                    pass

                raise RuntimeError(
                    f"Unrecoverable ciphertext at {table_name}.{column} "
                    f"rowid={row['rowid']} (user {user_id}): cell is neither "
                    "current-key, legacy-key, nor JSON-parseable plaintext."
                )

            if updates and not dry_run:
                set_clause = ", ".join(f"{col} = :{col}" for col in updates)
                params = dict(updates)
                params["rowid"] = row["rowid"]
                cursor.execute(
                    f"UPDATE {table_name} SET {set_clause} WHERE rowid = :rowid",
                    params,
                )
            stats.rows_reencrypted += len(updates)


def _drain_pager_plaintext(
    cursor: sqlite3.Cursor,
    current: Fernet,
    dry_run: bool,
    stats: UserStats,
) -> None:
    if not _table_exists(cursor, "pager_messages"):
        return

    columns = set(_column_names(cursor, "pager_messages"))
    pairs = [
        ("content", "encrypted__content"),
        ("original_content", "encrypted__original_content"),
    ]

    for plain_col, enc_col in pairs:
        if plain_col not in columns:
            continue
        if enc_col not in columns:
            # Encrypted column missing — schema older than the hardening pass.
            if not dry_run:
                cursor.execute(
                    f"ALTER TABLE pager_messages ADD COLUMN {enc_col} TEXT"
                )
                stats.columns_added += 1
            columns.add(enc_col)

        cursor.execute(
            f"SELECT rowid, {plain_col} FROM pager_messages "
            f"WHERE {plain_col} IS NOT NULL AND {enc_col} IS NULL"
        )
        rows_to_drain = cursor.fetchall()
        for row in rows_to_drain:
            ciphertext = _encrypt(current, row[plain_col])
            if not dry_run:
                cursor.execute(
                    f"UPDATE pager_messages SET {enc_col} = :enc WHERE rowid = :rowid",
                    {"enc": ciphertext, "rowid": row["rowid"]},
                )
        stats.pager_plaintext_drained += len(rows_to_drain)

    # Drop the plaintext columns once their data has been drained.
    columns = set(_column_names(cursor, "pager_messages"))
    for plain_col, _ in pairs:
        if plain_col in columns:
            if not dry_run:
                cursor.execute(f"ALTER TABLE pager_messages DROP COLUMN {plain_col}")
                stats.columns_dropped += 1


def _add_external_message_id(
    cursor: sqlite3.Cursor,
    dry_run: bool,
    stats: UserStats,
) -> None:
    if not _table_exists(cursor, "pager_messages"):
        return
    columns = set(_column_names(cursor, "pager_messages"))
    if "external_message_id" not in columns:
        if not dry_run:
            cursor.execute(
                "ALTER TABLE pager_messages ADD COLUMN external_message_id TEXT"
            )
            stats.columns_added += 1


def _ensure_external_id_index(cursor: sqlite3.Cursor, dry_run: bool) -> None:
    if not _table_exists(cursor, "pager_messages"):
        return
    if dry_run:
        return
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_pager_messages_external_id "
        "ON pager_messages(user_id, external_message_id) "
        "WHERE external_message_id IS NOT NULL"
    )


def _add_credentials_metadata(
    cursor: sqlite3.Cursor,
    dry_run: bool,
    stats: UserStats,
) -> None:
    if not _table_exists(cursor, "credentials"):
        return
    columns = set(_column_names(cursor, "credentials"))
    if "metadata" not in columns:
        if not dry_run:
            cursor.execute(
                "ALTER TABLE credentials "
                "ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'"
            )
            stats.columns_added += 1


def migrate_user_db(
    db_path: Path,
    user_id: str,
    master_key: bytes,
    dry_run: bool,
) -> UserStats:
    """Run all migration steps for one user database under a single transaction."""
    stats = UserStats(user_id=user_id)

    backup_path = db_path.with_suffix(db_path.suffix + BACKUP_SUFFIX)
    if not dry_run and not backup_path.exists():
        shutil.copy2(db_path, backup_path)

    current = _build_fernet(_vault_key(user_id, master_key))
    legacy = _build_fernet(_legacy_key(user_id))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN")
        try:
            _scan_and_reencrypt(cursor, current, legacy, user_id, dry_run, stats)
            _drain_pager_plaintext(cursor, current, dry_run, stats)
            _add_external_message_id(cursor, dry_run, stats)
            _ensure_external_id_index(cursor, dry_run)
            _add_credentials_metadata(cursor, dry_run, stats)
            if dry_run:
                conn.rollback()
            else:
                conn.commit()
        except Exception as exc:
            conn.rollback()
            stats.aborted = str(exc)
            raise
    finally:
        conn.close()

    return stats


def _user_db_paths() -> list[tuple[str, Path]]:
    if not DATA_USERS_DIR.exists():
        return []
    out: list[tuple[str, Path]] = []
    for user_dir in sorted(DATA_USERS_DIR.iterdir()):
        if not user_dir.is_dir():
            continue
        db_path = user_dir / "userdata.db"
        if db_path.exists():
            out.append((user_dir.name, db_path))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan for rows needing migration but write nothing to disk.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data/users root (for tests).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.data_dir is not None:
        global DATA_USERS_DIR
        DATA_USERS_DIR = args.data_dir

    # Fail fast on Vault unreachable — never silently fall back to the legacy key.
    master_key = get_service_config("userdata_encryption_key").encode()

    summary = Summary()
    user_dbs = _user_db_paths()
    if not user_dbs:
        logger.info("No user databases found at %s — nothing to migrate.", DATA_USERS_DIR)
        return 0

    for user_id, db_path in user_dbs:
        try:
            stats = migrate_user_db(db_path, user_id, master_key, args.dry_run)
        except Exception as exc:
            summary.errors.append(f"{user_id}: {exc}")
            logger.error("ABORT for %s: %s", user_id, exc)
            break

        summary.users_processed += 1
        summary.rows_reencrypted += stats.rows_reencrypted
        summary.pager_plaintext_drained += stats.pager_plaintext_drained
        summary.columns_dropped += stats.columns_dropped
        summary.columns_added += stats.columns_added

        if (
            stats.rows_reencrypted
            or stats.pager_plaintext_drained
            or stats.columns_dropped
            or stats.columns_added
        ):
            logger.info(
                "%s: reencrypted=%d drained=%d cols_added=%d cols_dropped=%d",
                user_id[:8],
                stats.rows_reencrypted,
                stats.pager_plaintext_drained,
                stats.columns_added,
                stats.columns_dropped,
            )

    mode = "DRY-RUN " if args.dry_run else ""
    logger.info(
        "%sSummary: users=%d rows_reencrypted=%d pager_drained=%d "
        "cols_added=%d cols_dropped=%d errors=%d",
        mode,
        summary.users_processed,
        summary.rows_reencrypted,
        summary.pager_plaintext_drained,
        summary.columns_added,
        summary.columns_dropped,
        len(summary.errors),
    )
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
