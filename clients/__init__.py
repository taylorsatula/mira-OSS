"""
Client modules for external service integrations.
"""

from .hybrid_embeddings_provider import get_hybrid_embeddings_provider, HybridEmbeddingsProvider
from .postgres_client import PostgresClient
from .sqlite_client import SQLiteClient
from .valkey_client import ValkeyClient, get_valkey, get_valkey_client
from .vault_client import get_auth_secret, get_database_url, get_service_config


def __getattr__(name):
    if name == "LLMProvider":
        from .llm_provider import LLMProvider
        return LLMProvider
    raise AttributeError(name)

__all__ = [
    'HybridEmbeddingsProvider',
    'get_hybrid_embeddings_provider',
    'LLMProvider',
    'PostgresClient',
    'SQLiteClient',
    'ValkeyClient',
    'get_valkey',
    'get_valkey_client',
    'get_auth_secret',
    'get_database_url',
    'get_service_config'
]
