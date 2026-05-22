"""
User credential management bridge to UserDataManager.

This module provides the expected interface for tools while leveraging
the existing UserDataManager's SQLite-based credential storage with
automatic encryption in user-specific databases.
"""

import json
from typing import Any, Optional, Dict

from typing_extensions import TypedDict
from utils.user_context import get_current_user_id
from utils.userdata_manager import get_user_data_manager


class CredentialMetadata(TypedDict):
    """Metadata for a stored credential."""
    created_at: Optional[str]
    updated_at: Optional[str]
    metadata: Dict[str, Any]


class UserCredentialService:
    """
    Bridge class that provides the expected credential interface
    while using the existing UserDataManager infrastructure.
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """Initialize with optional user_id, defaults to current user."""
        if user_id is not None:
            self.user_id = user_id
        else:
            try:
                self.user_id = get_current_user_id()
            except RuntimeError:
                raise RuntimeError("No user context set. Ensure authentication is properly initialized.")
        self.data_manager = get_user_data_manager(self.user_id)
    
    def store_credential(
        self,
        credential_type: str,
        service_name: str,
        credential_value: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store an encrypted credential using UserDataManager."""
        dm = get_user_data_manager(self.user_id)
        dm._ensure_credentials_table()

        existing = dm.select(
            'credentials',
            'credential_type = :ctype AND service_name = :service',
            {'ctype': credential_type, 'service': service_name}
        )

        from utils.timezone_utils import utc_now, format_utc_iso
        now = format_utc_iso(utc_now())

        credential_data = {
            'credential_type': credential_type,
            'service_name': service_name,
            'encrypted__credential_value': credential_value,
            'metadata': json.dumps(metadata or {}, sort_keys=True),
            'updated_at': now
        }

        if existing:
            dm.update(
                'credentials',
                credential_data,
                'credential_type = :ctype AND service_name = :service',
                {'ctype': credential_type, 'service': service_name}
            )
        else:
            import uuid
            credential_data['id'] = str(uuid.uuid4())
            credential_data['created_at'] = now
            dm.insert('credentials', credential_data)
    
    def get_credential(
        self,
        credential_type: str,
        service_name: str
    ) -> Optional[str]:
        """Retrieve a credential using UserDataManager."""
        dm = get_user_data_manager(self.user_id)
        dm._ensure_credentials_table()

        results = dm.select(
            'credentials',
            'credential_type = :ctype AND service_name = :service',
            {'ctype': credential_type, 'service': service_name}
        )

        return results[0]['encrypted__credential_value'] if results else None

    def get_credential_metadata(
        self,
        credential_type: str,
        service_name: str
    ) -> Dict[str, Any]:
        """Retrieve credential metadata without returning the secret value."""
        dm = get_user_data_manager(self.user_id)
        dm._ensure_credentials_table()

        results = dm.select(
            'credentials',
            'credential_type = :ctype AND service_name = :service',
            {'ctype': credential_type, 'service': service_name}
        )
        if not results:
            return {}
        return json.loads(results[0].get('metadata') or "{}")
    
    def delete_credential(
        self,
        credential_type: str,
        service_name: str
    ) -> bool:
        """Delete a credential using UserDataManager."""
        dm = get_user_data_manager(self.user_id)
        dm._ensure_credentials_table()

        rows_deleted = dm.delete(
            'credentials',
            'credential_type = :ctype AND service_name = :service',
            {'ctype': credential_type, 'service': service_name}
        )

        return rows_deleted > 0

    def list_user_credentials(self) -> Dict[str, Dict[str, CredentialMetadata]]:
        """List all credentials for a user."""
        dm = get_user_data_manager(self.user_id)
        dm._ensure_credentials_table()

        results = dm.select('credentials')

        credentials = {}
        for row in results:
            ctype = row['credential_type']
            service = row['service_name']

            if ctype not in credentials:
                credentials[ctype] = {}

            credentials[ctype][service] = {
                'created_at': row.get('created_at'),
                'updated_at': row.get('updated_at'),
                'metadata': json.loads(row.get('metadata') or "{}")
            }

        return credentials
