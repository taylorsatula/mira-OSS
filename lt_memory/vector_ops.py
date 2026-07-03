"""
Vector operations for LT_Memory system.

Handles embedding generation and storage using mdbr-leaf-ir-asym (768d) embeddings.
Singleton service that wraps the embeddings provider and database access.
"""
import logging
import numpy as np
from typing import List, Optional, Union, TYPE_CHECKING
from uuid import UUID

from lt_memory.models import Memory, ExtractedMemory
from lt_memory.db_access import LTMemoryDB
from lt_memory.hybrid_search import HybridSearcher

if TYPE_CHECKING:
    from clients.hybrid_embeddings_provider import HybridEmbeddingsProvider

logger = logging.getLogger(__name__)

# Vector search defaults
DEFAULT_VECTOR_SIMILARITY_THRESHOLD = 0.7
DEFAULT_VECTOR_SEARCH_LIMIT = 10


class VectorOps:
    """
    Vector operations service for embedding generation and similarity search.

    Uses mdbr-leaf-ir-asym (768d) for document embeddings.
    """

    def __init__(
        self,
        embeddings_provider: 'HybridEmbeddingsProvider',
        db: LTMemoryDB,
    ):
        self.embeddings_provider = embeddings_provider
        self.db = db
        self.hybrid_searcher = HybridSearcher(db)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate document embedding (768d) for memory storage.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embeddings_provider.encode_deep([text])[0]

        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate document embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = self.embeddings_provider.encode_deep(texts)

        result = []
        for embedding in embeddings:
            if isinstance(embedding, np.ndarray):
                result.append(embedding.tolist())
            else:
                result.append(embedding)

        return result

    def store_memories_with_embeddings(
        self,
        memories: List[ExtractedMemory]
    ) -> List[UUID]:
        """
        Generate embeddings and store multiple memories.

        Args:
            memories: List of ExtractedMemory objects

        Returns:
            List of created memory UUIDs
        """
        if not memories:
            return []

        texts = [memory.text for memory in memories]
        embeddings = self.generate_embeddings_batch(texts)

        return self.db.store_memories(
            memories=memories,
            embeddings=embeddings
        )

    def _search_with_embedding(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        min_importance: float
    ) -> List[Memory]:
        """
        Internal method that performs vector similarity search.

        Args:
            query_embedding: Embedding vector (768d)
            limit: Maximum results to return
            similarity_threshold: Minimum cosine similarity (0-1)
            min_importance: Minimum importance score filter

        Returns:
            List of Memory models sorted by similarity
        """
        return self.db.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

    def find_similar_memories(
        self,
        query: str,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_importance: Optional[float] = None
    ) -> List[Memory]:
        """
        Find similar memories using vector similarity search from text query.

        Args:
            query: Query text to search for
            limit: Maximum results to return (uses config default if None)
            similarity_threshold: Minimum cosine similarity (uses config default if None)
            min_importance: Minimum importance score filter (default: 0.1)

        Returns:
            List of Memory models sorted by similarity
        """
        # Apply config defaults if values not provided
        if self.vector_search_config:
            limit = limit if limit is not None else DEFAULT_VECTOR_SEARCH_LIMIT
            similarity_threshold = similarity_threshold if similarity_threshold is not None else DEFAULT_VECTOR_SIMILARITY_THRESHOLD
        else:
            limit = limit if limit is not None else 10
            similarity_threshold = similarity_threshold if similarity_threshold is not None else 0.7
        min_importance = min_importance if min_importance is not None else 0.1

        # Use realtime (query) encoding for search queries
        embedding = self.embeddings_provider.encode_realtime([query])[0]
        if isinstance(embedding, np.ndarray):
            query_embedding = embedding.tolist()
        else:
            query_embedding = embedding

        return self._search_with_embedding(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

    def find_similar_for_dedup(
        self,
        query_text: str,
        limit: int = 5,
        similarity_threshold: float = 0.92,
        min_importance: float = 0.001
    ) -> List[Memory]:
        """
        Find similar memories using document encoding for dedup comparison.

        Unlike find_similar_memories() which uses encode_realtime() (query encoder),
        this uses encode_deep() (document encoder) for document-to-document comparison.
        The asymmetric embedding model produces different vector spaces for queries vs
        documents — dedup compares documents against documents so must use the same encoder.
        """
        embedding = self.embeddings_provider.encode_deep([query_text])[0]
        if isinstance(embedding, np.ndarray):
            query_embedding = embedding.tolist()
        else:
            query_embedding = embedding

        return self._search_with_embedding(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

    def find_similar_by_embedding(
        self,
        query_embedding: Union[List[float], np.ndarray],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        min_importance: float = 0.1
    ) -> List[Memory]:
        """
        Find similar memories using pre-computed embedding.

        Args:
            query_embedding: Pre-computed embedding vector (768d)
            limit: Maximum results to return
            similarity_threshold: Minimum cosine similarity (0-1)
            min_importance: Minimum importance score filter

        Returns:
            List of Memory models sorted by similarity
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        # Validate dimensions
        if len(query_embedding) != 768:
            raise ValueError(
                f"Expected 768-dimensional embedding, got {len(query_embedding)}"
            )

        return self._search_with_embedding(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

    def find_similar_to_memory(
        self,
        memory_id: UUID,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        min_importance: float = 0.1
    ) -> List[Memory]:
        """
        Find memories similar to an existing memory using its embedding.

        Args:
            memory_id: Reference memory UUID to find neighbors for
            limit: Maximum results to return
            similarity_threshold: Minimum cosine similarity (0-1)
            min_importance: Minimum importance score filter

        Returns:
            List of Memory models sorted by similarity (excludes reference memory)
        """
        reference_memory = self.db.get_memory(memory_id)

        if not reference_memory or not reference_memory.embedding:
            logger.warning(f"Memory {memory_id} not found or has no embedding")
            return []

        results = self.db.search_similar(
            query_embedding=reference_memory.embedding,
            limit=limit + 1,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        return [m for m in results if m.id != memory_id][:limit]

    def update_memory_embedding(
        self,
        memory_id: UUID,
        new_text: str
    ) -> Memory:
        """
        Update memory text and regenerate embedding.

        Args:
            memory_id: Memory UUID
            new_text: New text content

        Returns:
            Updated Memory model

        Raises:
            ValueError: If memory not found
        """
        new_embedding = self.generate_embedding(new_text)

        updated_memory = self.db.update_memory(
            memory_id,
            updates={
                'text': new_text,
                'embedding': new_embedding
            }
        )

        logger.info(f"Updated memory {memory_id} with new text and embedding")
        return updated_memory

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: Union[List[float], np.ndarray],
        search_intent: str = "general",
        limit: int = 10,
        similarity_threshold: float = 0.7,
        min_importance: float = 0.1
    ) -> List[Memory]:
        """
        Perform hybrid BM25 + vector search for optimal memory retrieval.

        Args:
            query_text: Text query for BM25 search
            query_embedding: Pre-computed embedding for vector search
            search_intent: Search strategy (recall/explore/exact/general)
            limit: Maximum results to return
            similarity_threshold: Minimum similarity for vector search
            min_importance: Minimum importance score

        Returns:
            List of Memory models ranked by hybrid score
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        return self.hybrid_searcher.hybrid_search(
            query_text=query_text,
            query_embedding=query_embedding,
            search_intent=search_intent,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

    def cleanup(self) -> None:
        """
        Clean up resources.

        No-op: Dependencies managed by factory lifecycle.
        """
        logger.debug("VectorOps cleanup completed (no-op)")
