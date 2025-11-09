#!/usr/bin/env python3
"""
Emergency script to create a user account and generate a magic link when email is down.

Usage:
    python scripts/create_user_with_magic_link.py <email>

This script:
1. Creates a new user account
2. Generates a magic link token
3. Stores it in the database
4. Outputs the token you can use to verify: POST /verify with {"email": "...", "token": "..."}
"""

import sys
import secrets
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auth.database import AuthDatabase
from auth.config import config
from utils.timezone_utils import utc_now


def create_user_and_magic_link(email: str) -> tuple[str, str]:
    """
    Create a user account and generate a magic link token.

    Args:
        email: Email address for the new user

    Returns:
        Tuple of (user_id, token) where token can be used to verify the magic link
    """
    db = AuthDatabase()

    print(f"Creating user account for {email}...")
    try:
        user_id = db.create_user(email)
        print(f"✓ User created with ID: {user_id}")
    except Exception as e:
        print(f"✗ Failed to create user: {e}")
        sys.exit(1)

    print(f"Generating magic link token...")

    # Generate the token (same as auth/service.py _generate_secure_token)
    token = secrets.token_urlsafe(32)

    # Hash it (same as auth/service.py _hash_token)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # Calculate expiry (10 minutes from now, same as auth/service.py)
    expires_at = utc_now() + timedelta(seconds=config.MAGIC_LINK_EXPIRY)

    print(f"Storing magic link in database...")
    try:
        magic_link_id = db.create_magic_link(
            user_id=user_id,
            email=email,
            token_hash=token_hash,
            expires_at=expires_at
        )
        print(f"✓ Magic link created with ID: {magic_link_id}")
    except Exception as e:
        print(f"✗ Failed to create magic link: {e}")
        sys.exit(1)

    print("\n" + "="*70)
    print("✓ SUCCESS! Your account is ready to log in")
    print("="*70)
    print(f"\nEmail:  {email}")
    print(f"User ID: {user_id}")
    print(f"\nMagic Link Token (expires in 10 minutes):")
    print(f"  {token}")
    print(f"\nTo log in, make a POST request to /verify:")
    print(f'  curl -X POST http://localhost:8000/auth/verify \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"email": "{email}", "token": "{token}"}}\'')
    print("\n" + "="*70)

    return user_id, token


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/create_user_with_magic_link.py <email>")
        sys.exit(1)

    email = sys.argv[1]

    # Basic email validation
    if "@" not in email or "." not in email:
        print(f"✗ Invalid email address: {email}")
        sys.exit(1)

    create_user_and_magic_link(email)
