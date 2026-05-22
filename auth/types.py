"""
Type definitions for the lean authentication system.
"""

from typing import Optional, Literal, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class UserRecord(BaseModel):
    """Database user record structure."""
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None
    webauthn_credentials: Dict[str, Any] = Field(default_factory=dict)
    memory_manipulation_enabled: bool
    daily_manipulation_last_run: Optional[datetime] = None
    timezone: str


class UserProfile(BaseModel):
    """User profile data returned to client."""
    id: str
    email: str
    is_active: bool
    created_at: datetime


class SessionData(BaseModel):
    """Session data stored in Valkey."""
    user_id: str
    email: str
    created_at: str  # ISO format for JSON serialization
    last_activity: str  # ISO format for JSON serialization
    max_expiry: str  # ISO format for JSON serialization


class MagicLinkRecord(BaseModel):
    """Database magic link record structure."""
    id: str
    user_id: str
    email: str
    token_hash: str
    expires_at: datetime
    used_at: Optional[datetime] = None
    created_at: datetime


class CookieSettings(BaseModel):
    """Secure cookie configuration."""
    samesite: Literal["Strict", "Lax", "None"]
    httponly: bool
    secure: bool
    max_age: int


class APITokenContext(BaseModel):
    """Context for API token-authenticated requests (not session-based)."""
    user_id: str
    token_type: str
    token_id: str
