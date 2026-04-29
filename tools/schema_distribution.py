"""
Schema distribution system for tool databases.

Handles initialization of user databases with tool schemas and provides
migration utilities for schema updates.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def initialize_user_database(user_id: str) -> None:
    """
    Initialize a new user's database with all tool schemas.

    Called during user creation to set up their userdata.db with schemas
    from all registered tools.

    Args:
        user_id: The user's UUID

    Raises:
        RuntimeError: If database initialization fails
    """
    from utils.instance import user_data_base
    db_path = user_data_base() / user_id / "userdata.db"

    # Ensure user directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = None
    try:
        conn = sqlite3.connect(str(db_path))

        # Load and execute all tool schemas
        schema_dir = Path("tools/implementations/schemas")

        if not schema_dir.exists():
            raise RuntimeError(f"Schema directory not found: {schema_dir}")

        schema_files = sorted(schema_dir.glob("*.sql"))

        if not schema_files:
            raise RuntimeError("No tool schemas found to initialize")

        for schema_file in schema_files:
            logger.info(f"Loading schema for user {user_id}: {schema_file.name}")

            with open(schema_file, 'r') as f:
                schema_sql = f.read()

            conn.executescript(schema_sql)

        conn.commit()
        conn.close()
        conn = None

        logger.info(f"Initialized database for user {user_id} with {len(schema_files)} tool schemas")

    except Exception as e:
        if conn:
            conn.close()
        logger.error(f"Failed to initialize database for user {user_id}: {e}")
        raise RuntimeError(f"Database initialization failed: {e}")


def apply_schema_to_all_users(schema_name: str) -> Dict[str, List]:
    """
    Apply a specific schema to all existing user databases.

    Used when a tool schema is updated and needs to be distributed
    to all existing users.

    Args:
        schema_name: Name of schema file without .sql extension (e.g., "contacts_tool")

    Returns:
        Dict with "success" and "failed" lists containing user IDs

    Raises:
        ValueError: If schema file doesn't exist
    """
    schema_path = Path(f"tools/implementations/schemas/{schema_name}.sql")

    if not schema_path.exists():
        raise ValueError(f"Schema file not found: {schema_path}")

    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    from utils.instance import user_data_base
    users_dir = user_data_base()
    results = {"success": [], "failed": []}

    if not users_dir.exists():
        logger.warning("No users directory found")
        return results

    for user_dir in users_dir.iterdir():
        if not user_dir.is_dir():
            continue

        user_id = user_dir.name
        db_path = user_dir / "userdata.db"

        if not db_path.exists():
            logger.warning(f"No database found for user {user_id}")
            continue

        try:
            conn = sqlite3.connect(str(db_path))
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()

            results["success"].append(user_id)
            logger.info(f"Applied {schema_name} schema to user {user_id}")

        except Exception as e:
            results["failed"].append({"user_id": user_id, "error": str(e)})
            logger.error(f"Failed to apply schema to user {user_id}: {e}")

    return results


def apply_all_schemas_to_all_users() -> Dict[str, int]:
    """
    Re-apply all tool schemas to all existing user databases.

    Nuclear option for development or major schema migrations.
    Assumes all schema files are idempotent.

    Returns:
        Dict with counts of users updated and failed
    """
    from utils.instance import user_data_base
    schema_dir = Path("tools/implementations/schemas")
    users_dir = user_data_base()

    if not schema_dir.exists():
        logger.error(f"Schema directory not found: {schema_dir}")
        return {"updated": 0, "failed": 0}

    if not users_dir.exists():
        logger.error(f"Users directory not found: {users_dir}")
        return {"updated": 0, "failed": 0}

    schema_files = sorted(schema_dir.glob("*.sql"))

    if not schema_files:
        logger.warning("No schemas found to apply")
        return {"updated": 0, "failed": 0}

    updated_count = 0
    failed_count = 0

    for user_dir in users_dir.iterdir():
        if not user_dir.is_dir():
            continue

        user_id = user_dir.name
        db_path = user_dir / "userdata.db"

        if not db_path.exists():
            continue

        try:
            logger.info(f"Updating schemas for user {user_id}")

            conn = sqlite3.connect(str(db_path))

            for schema_file in schema_files:
                logger.debug(f"  Applying {schema_file.name} to user {user_id}")
                with open(schema_file, 'r') as f:
                    conn.executescript(f.read())

            conn.commit()
            conn.close()

            updated_count += 1
            logger.info(f"Updated {len(schema_files)} schemas for user {user_id}")

        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to update user {user_id}: {e}")

    return {"updated": updated_count, "failed": failed_count}
