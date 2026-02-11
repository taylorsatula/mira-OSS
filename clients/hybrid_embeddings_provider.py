"""
Hybrid embeddings provider using mdbr-leaf-ir-asym for asymmetric retrieval.

Architecture (bundled in mdbr-leaf-ir-asym):
- Query encoding: mdbr-leaf-ir (23M params) - fast, lightweight
- Document encoding: snowflake-arctic-embed-m-v1.5 - higher quality

Methods:
- encode_realtime(): Query encoding via model.encode_query() for fingerprints
- encode_deep(): Document encoding via model.encode_document() for memories/summaries
"""
import logging
import hashlib
from typing import List, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton instance
_hybrid_provider_instance = None


class EmbeddingCache:
    """
    Valkey-backed embedding cache with 15-minute TTL.

    Raises if Valkey is unreachable - embedding cache requires Valkey.
    """

    def __init__(self, key_prefix: str = "embedding"):
        self.logger = logging.getLogger("embedding_cache")
        self.key_prefix = key_prefix
        from clients.valkey_client import get_valkey_client
        self.valkey = get_valkey_client()  # Raises if Valkey unreachable
        self.logger.info(f"Embedding cache initialized with Valkey backend (prefix: {key_prefix})")

    def _get_cache_key(self, text: str) -> str:
        return f"{self.key_prefix}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding.

        Returns None if key not found (cache miss).
        Raises if Valkey operation fails.
        """
        cache_key = self._get_cache_key(text)
        cached_data = self.valkey.valkey_binary.get(cache_key)
        if cached_data:
            return np.frombuffer(cached_data, dtype=np.float16)

        return None  # Cache miss

    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache embedding with 15-minute TTL.

        Raises if Valkey operation fails.
        """
        cache_key = self._get_cache_key(text)
        embedding_bytes = embedding.astype(np.float16).tobytes()
        self.valkey.valkey_binary.setex(cache_key, 900, embedding_bytes)


def get_hybrid_embeddings_provider(cache_enabled: bool = True) -> 'HybridEmbeddingsProvider':
    """
    Get or create singleton HybridEmbeddingsProvider instance.

    Args:
        cache_enabled: Whether to enable embedding caching

    Returns:
        Singleton HybridEmbeddingsProvider instance
    """
    global _hybrid_provider_instance
    if _hybrid_provider_instance is None:
        logger.info("Creating singleton HybridEmbeddingsProvider instance")
        _hybrid_provider_instance = HybridEmbeddingsProvider(cache_enabled=cache_enabled)
    return _hybrid_provider_instance


class HybridEmbeddingsProvider:
    """
    Manages asymmetric embeddings using mdbr-leaf-ir-asym (768-dim).

    - encode_realtime(): Lightweight query encoding for fingerprints
    - encode_deep(): Higher quality document encoding for memories/summaries
    """

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize the hybrid provider with mdbr-leaf-ir-asym model.

        Args:
            cache_enabled: Whether to enable embedding caching
        """
        self.logger = logging.getLogger("hybrid_embeddings")
        self.cache_enabled = cache_enabled

        from config.config_manager import config
        from sentence_transformers import SentenceTransformer

        # Load mdbr-leaf-ir-asym for asymmetric retrieval
        self.logger.info("Loading mdbr-leaf-ir-asym model for asymmetric retrieval")
        self.model = SentenceTransformer(
            "MongoDB/mdbr-leaf-ir-asym",
            cache_folder=config.embeddings.fast_model.cache_dir
        )

        # Initialize caches for query and document embeddings
        if cache_enabled:
            self.query_cache = EmbeddingCache(key_prefix="embedding_768_query")
            self.doc_cache = EmbeddingCache(key_prefix="embedding_768_doc")
        else:
            self.query_cache = None
            self.doc_cache = None

        self.logger.info("HybridEmbeddingsProvider initialized")

    def encode_realtime(self,
                        texts: Union[str, List[str]],
                        batch_size: Optional[int] = None) -> np.ndarray:
        """
        Lightweight query encoding for fingerprints (768-dim).

        Used for retrieval queries where speed matters.

        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding

        Returns:
            768-dimensional normalized embeddings
        """
        if batch_size is None:
            from config.config_manager import config
            batch_size = config.embeddings.fast_model.batch_size

        # Handle caching for single text
        if self.cache_enabled and isinstance(texts, str) and self.query_cache:
            cached = self.query_cache.get(texts)
            if cached is not None:
                return cached

        # Generate query embeddings (uses mdbr-leaf-ir internally)
        embeddings = self.model.encode_query(texts, batch_size=batch_size)

        embeddings = embeddings.astype(np.float16)

        # Cache single embeddings
        if self.cache_enabled and isinstance(texts, str) and self.query_cache:
            self.query_cache.set(texts, embeddings)

        return embeddings

    def encode_deep(self,
                    texts: Union[str, List[str]],
                    batch_size: Optional[int] = None) -> np.ndarray:
        """
        Higher quality document encoding for memories and summaries (768-dim).

        Used for storing memories and segment summaries where quality matters.

        Args:
            texts: Text or list of texts to encode
            batch_size: Batch size for encoding

        Returns:
            768-dimensional normalized embeddings
        """
        if batch_size is None:
            from config.config_manager import config
            batch_size = config.embeddings.fast_model.batch_size

        # Handle caching for single text
        if self.cache_enabled and isinstance(texts, str) and self.doc_cache:
            cached = self.doc_cache.get(texts)
            if cached is not None:
                return cached

        # Generate document embeddings (uses snowflake-arctic-embed internally)
        embeddings = self.model.encode_document(texts, batch_size=batch_size)

        embeddings = embeddings.astype(np.float16)

        # Cache single embeddings
        if self.cache_enabled and isinstance(texts, str) and self.doc_cache:
            self.doc_cache.set(texts, embeddings)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text for compatibility with orchestrator.

        This is a wrapper around encode_realtime() for single queries.

        Args:
            text: Single text string to embed

        Returns:
            768-dimensional embedding as a list of floats
        """
        embedding = self.encode_realtime(text)
        return embedding.tolist()

    def close(self):
        """Clean up resources."""
        if hasattr(self.model, 'close'):
            self.model.close()
        self.logger.info("HybridEmbeddingsProvider closed")
