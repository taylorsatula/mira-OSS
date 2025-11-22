"""
User credential management bridge to UserDataManager.

This module provides the expected interface for tools while leveraging
the existing UserDataManager's SQLite-based credential storage with
automatic encryption in user-specific databases.
"""

import json
from typing import Optional, Dict, Any
from utils.user_context import get_current_user_id
from utils.userdata_manager import get_user_data_manager


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
        credential_value: str
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

        return results[0]['credential_value'] if results else None
    
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

    def list_user_credentials(self) -> Dict[str, Dict[str, Any]]:
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
                'updated_at': row.get('updated_at')
            }

        return credentials


# Convenience functions for email configuration
def store_email_config_for_current_user(config: dict) -> None:
    """Store complete email configuration in the current user's encrypted SQLite database."""
    credential_service = UserCredentialService()
    credential_service.store_credential(
        credential_type="email_config",
        service_name="email",
        credential_value=json.dumps(config)
    )


def get_email_config_for_current_user() -> Optional[dict]:
    """Get complete email configuration from the current user's encrypted SQLite database."""
    credential_service = UserCredentialService()
    config_json = credential_service.get_credential(
        credential_type="email_config",
        service_name="email"
    )

    if config_json:
        return json.loads(config_json)
    return None