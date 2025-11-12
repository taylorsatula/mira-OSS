"""
Hybrid search implementation combining BM25 text search with vector similarity.

This module provides hybrid retrieval that leverages both lexical matching
(for exact phrases) and semantic similarity (for related concepts).
"""
import logging
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Combines BM25 text search with vector similarity for optimal retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine results from both methods,
    with intent-aware weighting to optimize for different query types.
    """

    def __init__(self, db_access):
        """
        Initialize hybrid searcher.

        Args:
            db_access: LTMemoryDB instance for database operations
        """
        self.db = db_access

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        search_intent: str = "general",
        limit: int = 20,
        similarity_threshold: float = 0.5,
        min_importance: float = 0.1
    ) -> List[Any]:
        """
        Perform hybrid search combining BM25 and vector similarity.

        Args:
            query_text: Text query for BM25 search
            query_embedding: Embedding for vector search
            search_intent: Intent type (recall/explore/exact/general)
            limit: Maximum results to return
            similarity_threshold: Minimum similarity for vector search
            min_importance: Minimum importance score

        Returns:
            List of Memory objects ranked by hybrid score
        """
        # Run searches in parallel (would be async in production)
        bm25_results = self._bm25_search(
            query_text,
            limit=limit * 2,  # Oversample for fusion
            min_importance=min_importance
        )

        vector_results = self._vector_search(
            query_embedding,
            limit=limit * 2,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        # Apply intent-based weighting
        weights = {
            "recall": (0.6, 0.4),    # User trying to remember - favor exact matches
            "explore": (0.3, 0.7),   # User exploring concepts - favor semantic similarity
            "exact": (0.8, 0.2),     # User used specific phrases - strong BM25 preference
            "general": (0.4, 0.6)    # Balanced approach for ambient understanding
        }

        bm25_weight, vector_weight = weights.get(search_intent, weights["general"])

        # Combine using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            vector_results,
            bm25_weight,
            vector_weight,
            limit
        )

        logger.info(
            f"Hybrid search: {len(bm25_results)} BM25 + {len(vector_results)} vector "
            f"-> {len(fused_results)} fused results (intent: {search_intent})"
        )

        return fused_results

    def _bm25_search(
        self,
        query_text: str,
        limit: int,
        min_importance: float
    ) -> List[Tuple[Any, float]]:
        """
        Perform BM25 text search using PostgreSQL full-text search.

        Returns list of (Memory, score) tuples.
        """
        resolved_user_id = self.db._resolve_user_id()

        with self.db.session_manager.get_session(resolved_user_id) as session:
            # Use plainto_tsquery for user-friendly query parsing
            query = """
            SELECT m.*,
                   ts_rank(m.search_vector, plainto_tsquery('english', %(query)s)) as rank
            FROM memories m
            WHERE m.search_vector @@ plainto_tsquery('english', %(query)s)
              AND m.importance_score >= %(min_importance)s
              AND (m.expires_at IS NULL OR m.expires_at > NOW())
              AND m.is_archived = FALSE
            ORDER BY rank DESC
            LIMIT %(limit)s
            """

            results = session.execute_query(query, {
                'query': query_text,
                'limit': limit,
                'min_importance': min_importance
            })

            # Convert to Memory objects with scores
            from lt_memory.models import Memory
            return [(Memory(**row), row['rank']) for row in results]

    def _vector_search(
        self,
        query_embedding: List[float],
        limit: int,
        similarity_threshold: float,
        min_importance: float
    ) -> List[Tuple[Any, float]]:
        """
        Perform vector similarity search.

        Returns list of (Memory, score) tuples.
        """
        # Reuse existing vector search
        memories = self.db.search_similar(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            min_importance=min_importance
        )

        # Use similarity scores calculated by database
        results = []
        for memory in memories:
            if memory.similarity_score is None:
                raise RuntimeError(
                    f"Memory {memory.id} missing similarity_score - "
                    f"this indicates db.search_similar() did not populate the transient field"
                )
            results.append((memory, memory.similarity_score))

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Any, float]],
        vector_results: List[Tuple[Any, float]],
        bm25_weight: float,
        vector_weight: float,
        limit: int
    ) -> List[Any]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF formula: score(d) = Σ(1 / (k + rank(d)))
        where k is a constant (typically 60) that determines how quickly scores decay.
        """
        k = 60  # RRF constant

        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        memory_map = {}

        # Process BM25 results
        for rank, (memory, _) in enumerate(bm25_results, 1):
            memory_id = str(memory.id)
            rrf_scores[memory_id] += bm25_weight * (1.0 / (k + rank))
            memory_map[memory_id] = memory

        # Process vector results
        for rank, (memory, _) in enumerate(vector_results, 1):
            memory_id = str(memory.id)
            rrf_scores[memory_id] += vector_weight * (1.0 / (k + rank))
            memory_map[memory_id] = memory

        # Sort by combined RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top memories
        return [memory_map[memory_id] for memory_id, _ in sorted_ids[:limit]]