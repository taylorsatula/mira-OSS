"""
Continuum search tool for finding relevant messages in continuum history.

This tool provides multi-tier search capability:
1. Default: Hybrid vector + BM25 search on segment summaries for efficient retrieval
2. Scoped message search within specific time boundaries from segment results
3. Legacy: Direct BM25 full-text search on messages (requires timescope)

Results use progressive disclosure - segment summaries contain synthesized information
from entire conversation segments. MIRA can then search within specific segments
using the provided time boundaries for detailed information.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from cns.infrastructure.continuum_repository import get_continuum_repository
from utils.timezone_utils import format_utc_iso, parse_utc_time_string
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
from lt_memory.hybrid_search import HybridSearcher
from lt_memory.db_access import LTMemoryDB


class ContinuumSearchToolConfig(BaseModel):
    """Configuration for the continuum search tool."""

    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled by default"
    )
    default_results_per_page: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Default number of results per page"
    )
    max_results_per_page: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results that can be requested per page"
    )
    preview_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Character length for message previews"
    )
    min_rank_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum BM25 rank score to include in results"
    )
    high_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Threshold for considering results high confidence"
    )
    default_context_window: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Default number of messages to include before/after when expanding"
    )
    # Hybrid search configuration
    hybrid_vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity in hybrid search (0.0-1.0). "
                    "Remaining weight goes to BM25 text ranking. "
                    "Higher values favor semantic similarity over keyword matching."
    )
    # Confidence clustering configuration
    confidence_cluster_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Confidence score difference threshold for result clustering. "
                    "Results within this threshold of the top result are considered similar."
    )
    max_clustered_results: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of results to return even if more are within cluster threshold"
    )
    # Search behavior configuration
    vector_search_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Multiplier for vector search candidates (e.g., 2.0 means fetch 2x the limit)"
    )
    max_vector_candidates: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Maximum candidates to fetch in vector search regardless of multiplier"
    )


registry.register("continuumsearch_tool", ContinuumSearchToolConfig)


class ContinuumSearchTool(Tool):
    """
    Search continuum history using BM25 full-text search with progressive disclosure.

    Supports two modes:
    1. search: Find messages matching a query, return truncated previews with confidence
    2. expand: Retrieve full content of a message plus surrounding context

    For ambiguous or low-confidence searches, the tool can trigger agentic deep search
    where an LLM agent reads full continuum segments to find synthesized information.
    """

    name = "continuumsearch_tool"

    simple_description = "Search past conversations and extracted memories with immediate results. Hybrid vector+BM25 search finds relevant segments, messages, or memories. Use when you need synchronous search of conversation history."

    anthropic_schema = {
        "name": "continuumsearch_tool",
        "description": (
            "Search conversation history or long-term memories using hybrid vector+BM25 search. "
            "For conversations: start with 'search' on summaries (default) to find relevant segments. "
            "For memories: use search_mode='memories' to search your extracted knowledge and insights. "
            "Extract entities/proper nouns from queries for better matching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "search_within_segment", "expand_message"],
                    "description": (
                        "Operation: 'search' for summaries/messages, 'search_within_segment' "
                        "to explore a specific segment, 'expand_message' for full content"
                    )
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["summaries", "messages", "memories"],
                    "description": (
                        "Search mode for 'search' operation. 'summaries' (default) searches "
                        "segment summaries. 'messages' requires start_time and end_time. "
                        "'memories' searches long-term memories using hybrid vector+BM25 search."
                    )
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query. Required for 'search' operation. "
                        "Example: 'Mark and his XFS system', 'database migration discussion'"
                    )
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Important entities/proper nouns to boost in search. "
                        "Example: ['Mark', 'XFS'] for 'Mark's XFS system'. "
                        "Include: names, places, products, technical terms."
                    )
                },
                "start_time": {
                    "type": "string",
                    "description": (
                        "ISO timestamp for search start boundary. REQUIRED when search_mode='messages'. "
                        "Get from segment summary results. Example: '2024-10-15T14:00:00Z'"
                    )
                },
                "end_time": {
                    "type": "string",
                    "description": (
                        "ISO timestamp for search end boundary. REQUIRED when search_mode='messages'. "
                        "Get from segment summary results. Example: '2024-10-15T16:30:00Z'"
                    )
                },
                "temporal_direction": {
                    "type": "string",
                    "enum": ["before", "after", "around"],
                    "description": (
                        "Temporal search direction relative to reference_time. 'before' finds earlier segments, "
                        "'after' finds later segments, 'around' finds segments near the reference time. "
                        "Use with reference_time to walk through conversation history."
                    )
                },
                "reference_time": {
                    "type": "string",
                    "description": (
                        "ISO timestamp as anchor for temporal_direction search. "
                        "Example: '2024-10-15T14:00:00Z'. When set, limits search relative to this time."
                    )
                },
                "segment_id": {
                    "type": "string",
                    "description": (
                        "8-character segment ID from summary search results. Required for "
                        "'search_within_segment' operation. Example: 'abc12345'"
                    )
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Number of results per page (default: 10 for summaries, 20 for messages)"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (default: 1)"
                },
                "message_id": {
                    "type": "string",
                    "description": (
                        "First 8 characters of message UUID. Required for 'expand_message'. "
                        "Example: 'a7b3c4d5'"
                    )
                },
                "direction": {
                    "type": "string",
                    "enum": ["before", "after", "both"],
                    "description": "Direction for context messages (default: 'both')"
                },
                "context_count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Number of context messages per direction (default: 2)"
                }
            },
            "required": ["operation"]
        }
    }

    def __init__(self):
        """Initialize the continuum search tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Load configuration
        config_cls = registry.get("continuumsearch_tool") or ContinuumSearchToolConfig
        self._config = config_cls()

        # Get continuum repository for database access
        self._conversation_repo = get_continuum_repository()

        # Get embeddings provider for query embeddings
        self._embeddings_provider = get_hybrid_embeddings_provider()

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a continuum search operation.

        Args:
            operation: Operation to perform ("search", "search_within_segment", or "expand_message")
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ValueError: If operation fails or parameters are invalid
        """
        try:
            if operation == "search":
                return self._search_messages(**kwargs)
            elif operation == "search_within_segment":
                return self._search_within_segment(**kwargs)
            elif operation == "expand_message":
                return self._expand_message(**kwargs)
            else:
                raise ValueError(
                    f"Unknown operation: {operation}. "
                    f"Valid operations are: search, search_within_segment, expand_message"
                )
        except Exception as e:
            self.logger.error(f"Error executing {operation} in conversationsearch_tool: {e}")
            raise

    def _search_messages(
        self,
        query: str,
        search_mode: str = "summaries",
        entities: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        page: int = 1,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        temporal_direction: Optional[str] = None,
        reference_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search continuum using summaries (default) or messages.

        Args:
            query: Natural language search query
            search_mode: "summaries" (default) or "messages" (requires timescope)
            entities: Optional list of entities/proper nouns to boost
            max_results: Number of results per page
            page: Page number for pagination
            start_time: Required for search_mode="messages"
            end_time: Required for search_mode="messages"
            temporal_direction: "before", "after", or "around" for summary search
            reference_time: ISO timestamp for temporal direction

        Returns:
            Dict containing search results with confidence score
        """
        if not query or not query.strip():
            raise ValueError("Query must be provided for search operation")

        query = query.strip()
        entities = entities or []

        # Validate search mode
        if search_mode not in ["summaries", "messages", "memories"]:
            raise ValueError(f"search_mode must be 'summaries', 'messages', or 'memories', got: {search_mode}")

        # Message mode requires timescope
        if search_mode == "messages":
            if not start_time or not end_time:
                raise ValueError(
                    "Message search requires both start_time and end_time parameters. "
                    "Use summary search first to find relevant segments, then search within "
                    "their time boundaries."
                )
            return self._search_messages_in_timeframe(
                query=query,
                start_time=start_time,
                end_time=end_time,
                entities=entities,
                max_results=max_results,
                page=page
            )

        # Memory mode - search long-term memories
        if search_mode == "memories":
            return self._search_memories_hybrid(
                query=query,
                entities=entities,
                max_results=max_results,
                page=page
            )

        # Summary mode (default)
        return self._search_summaries(
            query=query,
            entities=entities,
            max_results=max_results,
            page=page,
            temporal_direction=temporal_direction,
            reference_time=reference_time
        )

    def _search_summaries(
        self,
        query: str,
        entities: List[str],
        max_results: Optional[int] = None,
        page: int = 1,
        temporal_direction: Optional[str] = None,
        reference_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search segment summaries using hybrid vector + BM25 search.

        Combines semantic similarity from embeddings with keyword relevance from
        BM25 text search. Supports temporal filtering with direction and reference time.

        Args:
            query: Search query text
            entities: List of entities to boost
            max_results: Number of results per page
            page: Page number for pagination
            temporal_direction: Optional "before", "after", or "around"
            reference_time: Optional ISO timestamp for temporal filtering

        Returns:
            Dict with segment summaries, confidence scores, and pagination info
        """
        # Set defaults
        limit = max_results or 10  # More summaries by default
        offset = (page - 1) * limit

        # Generate query embedding
        try:
            query_embedding = self._embeddings_provider.encode_realtime(query)
            # Convert to list for PostgreSQL
            embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
            # Format as PostgreSQL vector string
            embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {e}")
            raise ValueError(f"Cannot perform summary search: embedding generation failed - {str(e)}")

        # Build temporal filter if specified
        temporal_clause = ""  # SQL structure component
        temporal_params = []  # Data parameters

        if temporal_direction and reference_time:
            ref_time = parse_utc_time_string(reference_time)

            if temporal_direction == "before":
                temporal_clause = "AND m.created_at < %s"
                temporal_params = [ref_time]
            elif temporal_direction == "after":
                temporal_clause = "AND m.created_at > %s"
                temporal_params = [ref_time]
            elif temporal_direction == "around":
                # ±7 days from reference time
                from datetime import timedelta
                start_time = ref_time - timedelta(days=7)
                end_time = ref_time + timedelta(days=7)
                temporal_clause = "AND m.created_at BETWEEN %s AND %s"
                temporal_params = [start_time, end_time]

        db = self._conversation_repo._get_client(self.user_id)

        # Hybrid search query - temporal_clause is SQL structure, not data
        search_sql = f"""
            WITH vector_search AS (
                -- Vector similarity search on segment embeddings
                SELECT
                    m.id,
                    m.continuum_id,
                    m.created_at,
                    m.metadata,
                    1 - (m.segment_embedding <=> %s::vector) AS vector_score
                FROM messages m
                WHERE m.metadata->>'is_segment_boundary' = 'true'
                  AND m.metadata->>'status' = 'collapsed'
                  AND m.segment_embedding IS NOT NULL
                  {temporal_clause}
                ORDER BY m.segment_embedding <=> %s::vector
                LIMIT %s
            ),
            text_search AS (
                -- BM25 text search on segment summaries
                SELECT
                    m.id,
                    ts_rank_cd(
                        to_tsvector('english', m.metadata->>'summary'),
                        plainto_tsquery('english', %s)
                    ) AS text_rank
                FROM messages m
                WHERE m.id IN (SELECT id FROM vector_search)
                  AND m.metadata->>'summary' IS NOT NULL
                  AND to_tsvector('english', m.metadata->>'summary') @@ plainto_tsquery('english', %s)
            )
            SELECT
                v.id,
                v.continuum_id,
                v.created_at,
                v.metadata,
                v.vector_score,
                COALESCE(t.text_rank, 0.0) AS text_rank,
                -- Hybrid score: configurable vector/text weighting
                ({self._config.hybrid_vector_weight} * v.vector_score + {1.0 - self._config.hybrid_vector_weight} * COALESCE(t.text_rank, 0.0)) AS hybrid_score
            FROM vector_search v
            LEFT JOIN text_search t ON v.id = t.id
            ORDER BY hybrid_score DESC
            OFFSET %s
            LIMIT %s
        """

        # Execute with all parameters
        # Fetch extra candidates for BM25 reranking, but cap to avoid excessive vector ops
        vector_limit = min(
            int(limit * self._config.vector_search_multiplier),
            self._config.max_vector_candidates
        )
        # Build params in SQL placeholder order:
        # 1. embedding for vector score (line 562)
        # 2. temporal_params for WHERE clause (line 567) - if present
        # 3. embedding for ORDER BY (line 568)
        # 4. vector_limit for LIMIT (line 569)
        # 5. query params for text search (lines 577, 582)
        # 6. offset and limit for final pagination (lines 596, 597)
        params = [embedding_str]
        params.extend(temporal_params)  # Add temporal filter params if any
        params.extend([embedding_str, vector_limit])
        params.extend([query, query])  # Text search params
        params.extend([offset, limit])  # Pagination params

        try:
            rows = db.execute_query(search_sql, tuple(params))
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            raise ValueError(f"Summary search failed: {str(e)}")

        # Process results
        results = []
        for row in rows:
            # Defensive metadata parsing
            metadata = row.get("metadata") or {}
            if not isinstance(metadata, dict):
                self.logger.warning(f"Malformed metadata in segment {row.get('id')}: {type(metadata)}")
                metadata = {}

            # Entity boosting
            boost = 1.0
            matched_entities = []
            if entities and metadata.get("summary"):
                summary_lower = metadata["summary"].lower()
                for entity in entities:
                    if entity.lower() in summary_lower:
                        matched_entities.append(entity)
                # Boost by 10% per matched entity
                boost = 1.0 + (0.1 * len(matched_entities))

            results.append({
                "result_type": "segment_summary",
                "segment_id": str(row["id"])[:8],
                "display_title": metadata.get("display_title", "Conversation segment"),
                "summary": metadata.get("summary", "No summary available"),
                "confidence_score": min(row["hybrid_score"] * boost, 1.0),
                "time_boundaries": {
                    "start": metadata.get("segment_start_time"),
                    "end": metadata.get("segment_end_time")
                },
                "tools_used": metadata.get("tools_used", []),
                "matched_entities": matched_entities,
                "created_at": format_utc_iso(row["created_at"])
            })

        # Apply smart filtering based on confidence clustering
        filtered_results = self._filter_results_by_confidence(results)

        # Calculate overall confidence
        if not filtered_results:
            confidence = 0.0
            status = "no_results"
        else:
            top_score = filtered_results[0]["confidence_score"]
            if top_score >= self._config.high_confidence_threshold:
                status = "high_confidence"
            elif top_score >= 0.40:
                status = "medium_confidence"
            else:
                status = "low_confidence"
            confidence = top_score

        return {
            "status": status,
            "confidence": round(confidence, 3),
            "query": query,
            "entities": entities,
            "results": filtered_results,
            "result_count": len(filtered_results),
            "page": page,
            "has_more_pages": len(results) == limit,  # Use original results for pagination check
            "search_mode": "summaries",
            "temporal_filter": {
                "direction": temporal_direction,
                "reference_time": reference_time
            } if temporal_direction else None,
            "meta": {
                "search_tier": "hybrid_vector_bm25",
                "vector_weight": self._config.hybrid_vector_weight,
                "text_weight": 1.0 - self._config.hybrid_vector_weight
            }
        }

    def _execute_bm25_search(
        self,
        query: str,
        entities: List[str],
        limit: int,
        offset: int
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 full-text search against messages table.

        Args:
            query: Search query text
            entities: List of entities to boost
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of matching messages with rank scores
        """
        try:
            db = self._conversation_repo._get_client(self.user_id)

            # Primary BM25 search using PostgreSQL full-text search
            search_sql = """
                SELECT
                    m.id,
                    m.continuum_id,
                    m.role,
                    m.content,
                    m.created_at,
                    m.metadata,
                    ts_rank_cd(
                        to_tsvector('english', m.content),
                        plainto_tsquery('english', %s)
                    ) AS rank
                FROM messages m
                WHERE m.content IS NOT NULL
                  AND m.content <> ''
                  AND to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s OFFSET %s
            """

            rows = db.execute_query(search_sql, (query, query, limit, offset))

            results = []
            for row in rows:
                rank_value = row.get("rank")
                if rank_value is None or float(rank_value) < self._config.min_rank_threshold:
                    continue

                # Boost rank if entities are matched
                final_rank = float(rank_value)
                matched_entities = []
                if entities:
                    content_lower = row.get("content", "").lower()
                    for entity in entities:
                        if entity.lower() in content_lower:
                            matched_entities.append(entity)

                    # Boost rank by 20% for each matched entity
                    if matched_entities:
                        boost_factor = 1.0 + (0.2 * len(matched_entities))
                        final_rank = min(final_rank * boost_factor, 1.0)

                results.append({
                    "id": str(row["id"]),
                    "continuum_id": str(row["continuum_id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "metadata": row.get("metadata", {}),
                    "rank": round(final_rank, 4),
                    "matched_entities": matched_entities
                })

            return results

        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            return []

    def _calculate_confidence(
        self,
        results: List[Dict[str, Any]],
        query: str,
        entities: List[str]
    ) -> float:
        """
        Calculate confidence score for search results.

        Confidence is based on:
        - Top result rank score (50%)
        - Entity match coverage (30%)
        - Result depth - multiple high-quality results (20%)

        Args:
            results: Search results with rank scores
            query: Original query
            entities: Expected entities

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results:
            return 0.0

        # Top rank quality (50%)
        top_rank = results[0].get("rank", 0.0)
        rank_component = top_rank * 0.5

        # Entity coverage (30%)
        entity_component = 0.0
        if entities:
            matched = results[0].get("matched_entities", [])
            coverage = len(matched) / len(entities)
            entity_component = coverage * 0.3

        # Result depth (20%)
        high_quality_count = sum(1 for r in results[:5] if r.get("rank", 0) > 0.5)
        depth_component = min(high_quality_count / 3, 1.0) * 0.2

        confidence = rank_component + entity_component + depth_component
        return round(min(confidence, 1.0), 3)

    def _format_message_preview(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a message as a preview with truncated content.

        Args:
            message: Raw message data from database

        Returns:
            Formatted preview with short UUID and truncation markers
        """
        content = message.get("content", "")
        is_truncated = len(content) > self._config.preview_length

        if is_truncated:
            preview = self._truncate_smart(content, self._config.preview_length)
        else:
            preview = content

        # Extract first 8 chars of UUID
        full_uuid = message.get("id", "")
        short_id = full_uuid[:8] if full_uuid else ""

        return {
            "message_id": short_id,
            "full_uuid": full_uuid,  # Include but don't show to user
            "continuum_id": message.get("continuum_id", ""),
            "role": message.get("role", ""),
            "timestamp": format_utc_iso(message.get("created_at")) if message.get("created_at") else None,
            "preview": preview,
            "is_truncated": is_truncated,
            "full_length": len(content),
            "match_score": message.get("rank", 0.0),
            "matched_entities": message.get("matched_entities", [])
        }

    def _truncate_smart(self, text: str, max_chars: int) -> str:
        """
        Truncate text at sentence boundary, avoiding mid-sentence cuts.

        Args:
            text: Text to truncate
            max_chars: Maximum character length (including ellipsis)

        Returns:
            Truncated text ending at sentence boundary with ellipsis
        """
        if len(text) <= max_chars:
            return text

        # Reserve space for ellipsis
        truncated = text[:max_chars - 2]  # Reserve 2 chars for ".."

        # Look for sentence endings
        boundaries = [
            truncated.rfind('. '),
            truncated.rfind('.\n'),
            truncated.rfind('? '),
            truncated.rfind('! ')
        ]

        last_boundary = max(boundaries)

        # Use boundary if it's at least 60% of target length
        if last_boundary > (max_chars - 2) * 0.6:
            return text[:last_boundary + 1] + ".."

        # Otherwise just cut at max length
        return truncated.rstrip() + ".."

    def _filter_results_by_confidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply smart filtering to results based on confidence clustering.

        Uses configuration to determine:
        - When to return only a clear winner
        - How many similar results to group together
        - Maximum results to return

        Args:
            results: List of results sorted by confidence score

        Returns:
            Filtered list of results based on confidence clustering
        """
        if not results:
            return []

        if len(results) == 1:
            return results

        # Start with the top result
        filtered = [results[0]]
        top_score = results[0]["confidence_score"]

        # Check if there's a clear winner (>threshold ahead of second place)
        if len(results) > 1:
            second_score = results[1]["confidence_score"]
            if top_score - second_score > self._config.confidence_cluster_threshold:
                # Clear winner, return only the top result
                self.logger.debug(f"Clear winner with {top_score:.3f} confidence, {top_score - second_score:.3f} ahead")
                return filtered

        # Check how many results are within threshold of the top
        cluster_threshold = top_score - self._config.confidence_cluster_threshold
        clustered_count = 1  # Start with top result

        for i in range(1, min(len(results), self._config.max_clustered_results)):  # Check up to max configured
            if results[i]["confidence_score"] >= cluster_threshold:
                clustered_count += 1
            else:
                break

        # Decide how many to return
        if clustered_count >= 3:
            # Top 3 are close, return all of them
            filtered = results[:clustered_count]
            self.logger.debug(f"Returning {clustered_count} clustered results within {self._config.confidence_cluster_threshold:.1%} of top score {top_score:.3f}")
        else:
            # Default: return top 2
            filtered = results[:min(len(results), 2)]
            second_score = f"{filtered[1]['confidence_score']:.3f}" if len(filtered) > 1 else "N/A"
            self.logger.debug(f"Returning top 2 results (scores: {filtered[0]['confidence_score']:.3f}, {second_score})")

        return filtered

    def _search_messages_in_timeframe(
        self,
        query: str,
        start_time: str,
        end_time: str,
        entities: List[str],
        max_results: Optional[int] = None,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search messages within a specific time window using BM25.

        This is used when MIRA needs detailed messages from a specific segment
        or time range. Requires explicit time boundaries to prevent unbounded searches.

        Args:
            query: Search query
            start_time: ISO timestamp for start boundary
            end_time: ISO timestamp for end boundary
            entities: List of entities to boost
            max_results: Number of results per page
            page: Page number

        Returns:
            Dict with message results within the timeframe
        """
        # Parse timestamps
        start_ts = parse_utc_time_string(start_time)
        end_ts = parse_utc_time_string(end_time)

        # Validate time range
        if start_ts >= end_ts:
            raise ValueError("start_time must be before end_time")

        # Set pagination
        limit = max_results or 20  # More messages for detail search
        offset = (page - 1) * limit

        db = self._conversation_repo._get_client(self.user_id)

        # BM25 search within time boundaries
        search_sql = """
            SELECT
                m.id,
                m.continuum_id,
                m.role,
                m.content,
                m.created_at,
                m.metadata,
                ts_rank_cd(
                    to_tsvector('english', m.content),
                    plainto_tsquery('english', %s)
                ) AS rank
            FROM messages m
            WHERE m.created_at >= %s
              AND m.created_at <= %s
              AND m.content IS NOT NULL
              AND m.content <> ''
              AND to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
              AND (m.metadata->>'is_segment_boundary' IS NULL
                   OR m.metadata->>'is_segment_boundary' = 'false')
            ORDER BY rank DESC, m.created_at ASC
            OFFSET %s
            LIMIT %s
        """

        rows = db.execute_query(search_sql, (query, start_ts, end_ts, query, offset, limit))

        # Process results with entity boosting
        results = []
        for row in rows:
            rank = float(row.get("rank", 0))

            # Entity boosting
            matched_entities = []
            boost = 1.0
            if entities:
                content_lower = row.get("content", "").lower()
                for entity in entities:
                    if entity.lower() in content_lower:
                        matched_entities.append(entity)
                # Boost by 20% per matched entity for messages
                boost = 1.0 + (0.2 * len(matched_entities))

            final_rank = min(rank * boost, 1.0)

            results.append({
                "id": str(row["id"]),
                "continuum_id": str(row["continuum_id"]),
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "metadata": row.get("metadata", {}),
                "rank": final_rank,
                "matched_entities": matched_entities
            })

        # Calculate confidence
        confidence = self._calculate_confidence(results, query, entities)

        # Format as message previews
        formatted_results = [self._format_message_preview(msg) for msg in results]

        # Determine status
        if confidence >= self._config.high_confidence_threshold:
            status = "high_confidence"
        elif confidence >= 0.40:
            status = "medium_confidence"
        else:
            status = "low_confidence"

        return {
            "status": status,
            "confidence": confidence,
            "query": query,
            "entities": entities,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "page": page,
            "has_more_pages": len(results) == limit,
            "search_mode": "messages",
            "time_boundaries": {
                "start": start_time,
                "end": end_time
            },
            "meta": {
                "search_tier": "bm25_timeframe",
                "message_count": len(results)
            }
        }

    def _search_memories_hybrid(
        self,
        query: str,
        entities: List[str],
        max_results: Optional[int] = None,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search long-term memories using hybrid vector + BM25 search.

        Args:
            query: Natural language search query
            entities: List of entities to potentially boost (not used in memory search)
            max_results: Number of results per page
            page: Page number for pagination

        Returns:
            Dict with memory search results including confidence scores
        """
        # Set defaults
        limit = max_results or 10
        offset = (page - 1) * limit

        # Initialize memory database and hybrid searcher
        from utils.database_session_manager import get_shared_session_manager
        session_manager = get_shared_session_manager()
        memory_db = LTMemoryDB(session_manager)
        searcher = HybridSearcher(memory_db)

        try:
            # Generate query embedding
            query_embedding = self._embeddings_provider.encode_realtime(query)

            # Perform hybrid search on memories
            # Using 'general' intent for balanced text/vector weights
            # Note: searcher uses ambient context internally via db._resolve_user_id()
            results = searcher.hybrid_search(
                query_text=query,
                query_embedding=query_embedding.tolist(),  # Convert to list for database
                search_intent="general",  # String, not enum
                limit=limit + offset,  # Fetch extra for pagination
                similarity_threshold=0.5,  # Lower threshold for memories
                min_importance=0.1
            )

            # Apply pagination
            paginated_results = results[offset:offset + limit]

            # Format results
            formatted_results = []
            for memory in paginated_results:
                formatted_results.append({
                    "result_type": "memory",
                    "memory_id": str(memory.id)[:8],
                    "full_uuid": str(memory.id),
                    "text": memory.text,
                    "importance_score": round(memory.importance_score, 3),
                    "confidence": round(memory.confidence, 3) if memory.confidence else 1.0,
                    "created_at": format_utc_iso(memory.created_at),
                    "happens_at": format_utc_iso(memory.happens_at) if memory.happens_at else None,
                    "expires_at": format_utc_iso(memory.expires_at) if memory.expires_at else None,
                    "is_refined": memory.is_refined,
                    "access_count": memory.access_count,
                    "entity_links": memory.entity_links or [],
                    "inbound_links": len(memory.inbound_links) if memory.inbound_links else 0,
                    "outbound_links": len(memory.outbound_links) if memory.outbound_links else 0
                })

            # Calculate overall confidence
            if not formatted_results:
                confidence = 0.0
                status = "no_results"
            else:
                # Use importance score of top result as confidence
                top_importance = formatted_results[0]["importance_score"]
                if top_importance >= 0.7:
                    status = "high_confidence"
                elif top_importance >= 0.4:
                    status = "medium_confidence"
                else:
                    status = "low_confidence"
                confidence = top_importance

            return {
                "status": status,
                "confidence": round(confidence, 3),
                "query": query,
                "entities": entities,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "page": page,
                "has_more_pages": len(results) > (offset + limit),
                "search_mode": "memories",
                "meta": {
                    "search_tier": "hybrid_vector_bm25_memories",
                    "total_memories_found": len(results),
                    "vector_weight": 0.6,  # Default for GENERAL intent
                    "text_weight": 0.4
                }
            }

        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            raise ValueError(f"Memory search failed: {str(e)}")

    def _search_within_segment(
        self,
        segment_id: str,
        query: str,
        max_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for specific messages within a segment using its time boundaries.

        This operation is used after finding relevant segments through summary search.
        It performs a BM25 search within the segment's time boundaries to find
        specific messages matching the query.

        Args:
            segment_id: 8-character segment ID from summary search
            query: Query to search for within the segment (required)
            max_results: Maximum messages to return

        Returns:
            Messages from the segment matching the query
        """
        if not query or not query.strip():
            raise ValueError("Query is required for search_within_segment operation")

        # Find the full segment sentinel
        db = self._conversation_repo._get_client(self.user_id)

        # Find segment by short ID
        segment_sql = """
            SELECT id, metadata, created_at
            FROM messages
            WHERE CAST(id AS TEXT) LIKE %s
              AND metadata->>'is_segment_boundary' = 'true'
            LIMIT 1
        """

        rows = db.execute_query(segment_sql, (f"{segment_id}%",))

        if not rows:
            raise ValueError(f"No segment found with ID starting with '{segment_id}'")

        segment = rows[0]
        # Defensive metadata parsing
        metadata = segment.get("metadata") or {}
        if not isinstance(metadata, dict):
            self.logger.warning(f"Malformed metadata in segment {segment.get('id')}: {type(metadata)}")
            metadata = {}

        # Get time boundaries
        start_time = metadata.get("segment_start_time")
        end_time = metadata.get("segment_end_time")

        if not start_time:
            # Fallback: use segment creation time as start
            start_time = format_utc_iso(segment["created_at"])

        if not end_time:
            # Fallback: find next segment or use current time
            next_segment_sql = """
                SELECT created_at
                FROM messages
                WHERE created_at > %s
                  AND metadata->>'is_segment_boundary' = 'true'
                ORDER BY created_at ASC
                LIMIT 1
            """
            next_rows = db.execute_query(next_segment_sql, (segment["created_at"],))
            if next_rows:
                end_time = format_utc_iso(next_rows[0]["created_at"])
            else:
                from utils.timezone_utils import utc_now
                end_time = format_utc_iso(utc_now())

        # Use the time-bounded message search
        result = self._search_messages_in_timeframe(
            query=query,
            start_time=start_time,
            end_time=end_time,
            entities=[],  # No entity extraction for within-segment search
            max_results=max_results,
            page=1
        )

        # Enhance result with segment information
        result["segment_info"] = {
            "segment_id": segment_id,
            "display_title": metadata.get("display_title", "Conversation segment"),
            "summary": metadata.get("summary", "")
        }

        return result

    def _expand_message(
        self,
        message_id: str,
        direction: str = "both",
        context_count: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Expand a message to show full content with surrounding context.

        Args:
            message_id: First 8 characters of message UUID
            direction: Which direction to fetch context ("before", "after", "both")
            context_count: Number of messages to fetch in each direction

        Returns:
            Dict with full message content and context messages
        """
        if not message_id or len(message_id) < 8:
            raise ValueError("message_id must be at least 8 characters")

        # Validate direction
        if direction not in ["before", "after", "both"]:
            raise ValueError(f"direction must be 'before', 'after', or 'both', got: {direction}")

        # Set context count
        if context_count is None:
            context_count = self._config.default_context_window
        context_count = max(0, min(context_count, 10))

        # Find the full UUID and message
        origin_message = self._find_message_by_short_id(message_id)
        if not origin_message:
            raise ValueError(f"No message found with ID starting with '{message_id}'")

        result = {
            "status": "expanded",
            "origin_message": {
                "message_id": message_id,
                "full_uuid": origin_message["id"],
                "continuum_id": origin_message["continuum_id"],
                "role": origin_message["role"],
                "content": origin_message["content"],
                "timestamp": format_utc_iso(origin_message["created_at"]) if origin_message.get("created_at") else None,
                "is_truncated": False
            }
        }

        # Fetch context messages if requested
        if context_count > 0:
            if direction in ["before", "both"]:
                result["context_before"] = self._fetch_context_messages(
                    origin_message,
                    direction="before",
                    count=context_count
                )

            if direction in ["after", "both"]:
                result["context_after"] = self._fetch_context_messages(
                    origin_message,
                    direction="after",
                    count=context_count
                )

        return result

    def _find_message_by_short_id(self, short_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a message by the first 8 characters of its UUID.

        Args:
            short_id: First 8 characters of UUID

        Returns:
            Message data or None if not found
        """
        try:
            db = self._conversation_repo._get_client(self.user_id)

            # Use LIKE to match UUID prefix (PostgreSQL UUID is stored as string in query)
            query = """
                SELECT id, continuum_id, role, content, created_at, metadata
                FROM messages
                WHERE CAST(id AS TEXT) LIKE %s
                LIMIT 1
            """

            pattern = f"{short_id}%"
            rows = db.execute_query(query, (pattern,))

            if rows:
                row = rows[0]
                return {
                    "id": str(row["id"]),
                    "continuum_id": str(row["continuum_id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "metadata": row.get("metadata", {})
                }

            return None

        except Exception as e:
            self.logger.error(f"Failed to find message by short ID {short_id}: {e}")
            return None

    def _fetch_context_messages(
        self,
        origin_message: Dict[str, Any],
        direction: str,
        count: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch context messages before or after the origin message.

        Args:
            origin_message: The reference message
            direction: "before" or "after"
            count: Number of messages to fetch

        Returns:
            List of context messages with relation indicators
        """
        try:
            db = self._conversation_repo._get_client(self.user_id)

            origin_time = origin_message["created_at"]
            continuum_id = origin_message["continuum_id"]

            if direction == "before":
                query = """
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE continuum_id = %s
                      AND created_at < %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """
                rows = db.execute_query(query, (continuum_id, origin_time, count))
                # Reverse to get chronological order
                rows.reverse()
            else:  # after
                query = """
                    SELECT id, role, content, created_at
                    FROM messages
                    WHERE continuum_id = %s
                      AND created_at > %s
                    ORDER BY created_at ASC
                    LIMIT %s
                """
                rows = db.execute_query(query, (continuum_id, origin_time, count))

            context = []
            for idx, row in enumerate(rows, 1):
                context.append({
                    "message_id": str(row["id"])[:8],
                    "full_uuid": str(row["id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": format_utc_iso(row["created_at"]) if row.get("created_at") else None,
                    "relation": f"[{idx} message{'s' if idx > 1 else ''} {direction} origin]"
                })

            return context

        except Exception as e:
            self.logger.error(f"Failed to fetch context messages: {e}")
            return []
