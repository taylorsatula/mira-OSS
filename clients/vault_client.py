"""
Production-ready Vault client for MIRA secret management.

Core principles:
- Never manages vault server lifecycle
- Fails fast with clear errors
- Uses environment variables for configuration
- Individual secret retrieval only
- IAM-ready auth abstraction
"""

import os
import logging
from typing import Optional, Dict, Any, TypedDict
import hvac
from hvac.exceptions import VaultError, InvalidPath, Unauthorized, Forbidden

logger = logging.getLogger(__name__)

# Global singleton instance and cache
_vault_client_instance: Optional['VaultClient'] = None
_secret_cache: Dict[str, str] = {}


def _ensure_vault_client() -> 'VaultClient':
    global _vault_client_instance
    if _vault_client_instance is None:
        _vault_client_instance = VaultClient()
    return _vault_client_instance


class VaultClient:
    """Production client with AppRole auth, env-based config, fail-fast validation, and KV v2 API."""

    def __init__(self, vault_addr: Optional[str] = None,
                 vault_namespace: Optional[str] = None):
        """Initializes with env vars, validates config, authenticates via AppRole, fails fast on errors."""
        try:

            # Get configuration from environment
            self.vault_addr = vault_addr or os.getenv('VAULT_ADDR')
            self.vault_namespace = vault_namespace or os.getenv('VAULT_NAMESPACE')
            
            # AppRole credentials
            self.vault_role_id = os.getenv('VAULT_ROLE_ID')
            self.vault_secret_id = os.getenv('VAULT_SECRET_ID')
            
            if not self.vault_addr:
                raise ValueError("VAULT_ADDR environment variable is required")
            
            if not self.vault_role_id or not self.vault_secret_id:
                raise ValueError("VAULT_ROLE_ID and VAULT_SECRET_ID environment variables are required")
            
            client_kwargs = {'url': self.vault_addr}
            
            if self.vault_namespace:
                client_kwargs['namespace'] = self.vault_namespace
            
            self.client = hvac.Client(**client_kwargs)
            self._authenticate_approle()
            
            if not self.client.is_authenticated():
                raise PermissionError("Vault authentication failed")

            logger.toast(f"Vault client initialized: {self.vault_addr}")

        except Exception as e:
            logger.error(f"Vault client initialization failed: {e}", exc_info=True)
            raise
    
    def _authenticate_approle(self):
        try:
            auth_response = self.client.auth.approle.login(
                role_id=self.vault_role_id,
                secret_id=self.vault_secret_id
            )
            
            self.client.token = auth_response['auth']['client_token']
            logger.toast("AppRole authentication successful")

        except Exception as e:
            logger.error(f"AppRole authentication failed: {e}", exc_info=True)
            raise PermissionError(f"AppRole authentication failed: {str(e)}")
    
    def get_secret(self, path: str, field: str) -> str:
        """Retrieves single field from KV v2 API with structured error handling."""
        try:

            response = self.client.secrets.kv.v2.read_secret_version(path=path, raise_on_deleted_version=True)
            secret_data = response['data']['data']

            if field not in secret_data:
                available_fields = list(secret_data.keys())
                raise KeyError(f"Field '{field}' not found in secret '{path}'. Available: {', '.join(available_fields)}")

            return secret_data[field]

        except InvalidPath:
            logger.error(f"Secret path not found: {path}", exc_info=True)
            raise FileNotFoundError(f"Secret path '{path}' not found in Vault")

        except (Unauthorized, Forbidden) as e:
            logger.error(f"Access denied to secret {path}/{field}: {e}", exc_info=True)
            raise PermissionError(f"Access denied to secret '{path}': {str(e)}")

        except VaultError as e:
            logger.error(f"Vault API error for {path}/{field}: {e}", exc_info=True)
            raise RuntimeError(f"Vault API error retrieving '{path}/{field}': {str(e)}")

# Individual secret retrieval functions
def get_database_url(service: str, admin: bool = False) -> str:
    """
    Get database URL from Vault for the current instance's database.

    Args:
        service: Database service name (must match instance database name)
        admin: If True, returns admin connection string (mira_admin role with BYPASSRLS)

    Returns:
        PostgreSQL connection URL
    """
    from utils.instance import database_name, vault_prefix
    expected = database_name()
    if service != expected:
        raise ValueError(f"Unknown database service: '{service}'. Expected '{expected}'.")

    field = 'admin_url' if admin else 'service_url'
    prefix = vault_prefix()
    cache_key = f"{prefix}/database/{field}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret(f'{prefix}/database', field)
    _secret_cache[cache_key] = value
    return value


def get_api_key(key_name: str) -> str:
    from utils.instance import vault_prefix
    prefix = vault_prefix()
    cache_key = f"{prefix}/api_keys/{key_name}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret(f'{prefix}/api_keys', key_name)
    _secret_cache[cache_key] = value
    return value


def get_auth_secret(secret_name: str) -> str:
    from utils.instance import vault_prefix
    prefix = vault_prefix()
    cache_key = f"{prefix}/auth/{secret_name}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret(f'{prefix}/auth', secret_name)
    _secret_cache[cache_key] = value
    return value


def get_service_config(field: str) -> str:
    from utils.instance import vault_prefix
    prefix = vault_prefix()
    cache_key = f"{prefix}/services/{field}"

    if cache_key in _secret_cache:
        return _secret_cache[cache_key]

    vault_client = _ensure_vault_client()
    value = vault_client.get_secret(f'{prefix}/services', field)
    _secret_cache[cache_key] = value
    return value


def preload_secrets() -> None:
    """
    Load all secrets into memory cache at startup.

    This prevents token expiration issues by caching everything
    while the AppRole token is still valid. Raises on any failure —
    missing secrets at startup means scattered failures at runtime.
    """
    from utils.instance import vault_prefix
    prefix = vault_prefix()
    vault_client = _ensure_vault_client()

    secret_groups = [
        (f'{prefix}/api_keys', 'API keys'),
        (f'{prefix}/database', 'database secrets'),
    ]

    successes = []
    failures = []

    for path, label in secret_groups:
        try:
            response = vault_client.client.secrets.kv.v2.read_secret_version(
                path=path,
                raise_on_deleted_version=True
            )
            secrets = response['data']['data']
            for field, value in secrets.items():
                _secret_cache[f"{path}/{field}"] = value
            successes.append(f"{len(secrets)} {label}")
        except Exception as e:
            failures.append(f"{label}: {e}")

    if successes:
        logger.info(f"Preloaded secrets: {', '.join(successes)}")

    if failures:
        raise RuntimeError(
            f"Failed to preload {len(failures)} secret group(s): {'; '.join(failures)}"
        )


class VaultHealthCheck(TypedDict, total=False):
    """Health check result from test_vault_connection."""
    status: str
    vault_addr: str
    namespace: Optional[str]
    authenticated: bool
    message: str


# Health check function
def test_vault_connection() -> VaultHealthCheck:
    try:
        from utils.instance import vault_prefix
        vault_client = _ensure_vault_client()
        vault_client.get_secret(f'{vault_prefix()}/database', 'username')

        logger.info("Vault connection test successful")
        return {
            "status": "success",
            "vault_addr": vault_client.vault_addr,
            "namespace": vault_client.vault_namespace,
            "authenticated": True,
            "message": "Vault connection successful"
        }

    except Exception as e:
        logger.error(f"Vault connection test failed: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "authenticated": False
        }