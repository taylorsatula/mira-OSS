#!/usr/bin/env python3
"""
Query user messages to debug prepopulation issues.

Usage:
    python scripts/query_user_messages.py <email>
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auth.database import AuthDatabase
from utils.database_session_manager import DatabaseSessionManager
from clients.vault_client import get_database_url


def query_user_messages(email: str):
    """Query user and their messages."""
    auth_db = AuthDatabase()

    print(f"Looking up user account for {email}...")
    user = auth_db.get_user_by_email(email)

    if not user:
        print(f"✗ No user found with email: {email}")
        sys.exit(1)

    user_id = str(user['id'])
    print(f"✓ User found with ID: {user_id}")
    print(f"  First name: {user['first_name']}")
    print(f"  Last name: {user['last_name']}")
    print(f"  Created: {user['created_at']}")
    print()

    # Query messages
    database_url = get_database_url('mira_service', admin=True)
    session_manager = DatabaseSessionManager(database_url)

    with session_manager.get_admin_session() as session:
        messages = session.execute_query(
            "SELECT id, role, content, metadata, created_at FROM messages WHERE user_id = %(user_id)s ORDER BY created_at",
            {'user_id': user_id}
        )

        print(f"Messages count: {len(messages)}")
        print()

        if messages:
            for i, msg in enumerate(messages, 1):
                print(f"--- Message {i} ---")
                print(f"ID: {msg['id']}")
                print(f"Role: {msg['role']}")
                print(f"Content: {msg['content'][:100]}...")
                print(f"Metadata: {msg['metadata']}")
                print(f"Created: {msg['created_at']}")
                print()
        else:
            print("No messages found for this user!")
            print()

        # Check continuums
        continuums = session.execute_query(
            "SELECT id, created_at FROM continuums WHERE user_id = %(user_id)s",
            {'user_id': user_id}
        )

        print(f"Continuums count: {len(continuums)}")
        if continuums:
            for cont in continuums:
                print(f"  Continuum ID: {cont['id']}, Created: {cont['created_at']}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/query_user_messages.py <email>")
        sys.exit(1)

    email = sys.argv[1]
    query_user_messages(email)
