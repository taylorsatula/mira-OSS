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

import logging
import math
from datetime import timedelta
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from tools.repo import Tool, coerce_to_int
from tools.registry import registry
from cns.infrastructure.continuum_repository import get_continuum_repository
from utils.timezone_utils import format_utc_iso, parse_utc_time_string, utc_now
from utils.user_context import get_current_segment_id
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider


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
    high_confidence_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Threshold for considering results high confidence"
    )
    medium_confidence_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Threshold for considering results medium confidence"
    )
    temporal_around_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Number of days before/after reference_time for 'around' temporal searches"
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


registry.register("continuum_tool", ContinuumSearchToolConfig)


class ContinuumSearchTool(Tool):
    """
    Search continuum history using BM25 full-text search with progressive disclosure.

    Supports two modes:
    1. search: Find messages matching a query, return truncated previews with confidence
    2. expand_message: Retrieve full content of a message plus surrounding context

    For ambiguous or low-confidence searches, the tool can trigger agentic deep search
    where an LLM agent reads full continuum segments to find synthesized information.
    """

    name = "continuum_tool"

    simple_description = "Search past conversations with immediate results. Hybrid vector+BM25 search finds relevant segments or messages. Use when you need synchronous search of conversation history."

    tool_schema = {
        "name": "continuum_tool",
        "description": "Search past conversation history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "search",
                        "search_within_segment",
                        "expand_message",
                        "get_tool_result",
                    ],
                    "default": "search",
                    "description": (
                        "'search' (default): query segment summaries or time-bounded messages. "
                        "'search_within_segment': query messages inside a specific segment (requires segment_id, query). "
                        "'expand_message': get full message text plus context (requires message_id). "
                        "'get_tool_result': retrieve an exact collapsed tool result (requires tool_result_id)"
                    )
                },
                "tool_result_id": {
                    "type": "string",
                    "description": (
                        "Session-scoped tool result identifier copied from a collapsed-result "
                        "breadcrumb. Required for 'get_tool_result'."
                    ),
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["summaries", "precis", "messages"],
                    "description": (
                        "For operation='search' only. 'summaries' (default): search segment summaries. "
                        "'precis': search segment precis (2-sentence compressed summaries). "
                        "Results are compact so larger result sets are inexpensive; use for broad historical scans. "
                        "'messages': search individual messages (start_time and end_time required). "
                        "Ignored by other operations"
                    )
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language search query. Required for 'search' and 'search_within_segment' operations. "
                        "Ignored by 'expand_message'. Example: 'database migration discussion'"
                    )
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Proper nouns extracted from the query to boost in ranking. "
                        "Case-insensitive substring match against results — matched entities increase confidence score. "
                        "Example: ['Mark', 'XFS']. Used by 'search' operation only"
                    )
                },
                "start_time": {
                    "type": "string",
                    "description": (
                        "ISO 8601 datetime for start of search window. Required with search_mode='messages'. "
                        "Must be before end_time. Copy from segment summary time_boundaries.start. "
                        "Example: '2024-10-15T14:00:00'"
                    )
                },
                "end_time": {
                    "type": "string",
                    "description": (
                        "ISO 8601 datetime for end of search window. Required when search_mode='messages'. "
                        "Must be after start_time. Copy from segment summary time_boundaries.end. "
                        "Example: '2024-10-15T16:30:00'"
                    )
                },
                "temporal_direction": {
                    "type": "string",
                    "enum": ["before", "after", "around"],
                    "description": (
                        "Temporal filter for segment summary search. 'before': segments before reference_time. "
                        "'after': segments after reference_time. 'around': segments within a window centered on reference_time. "
                        "Ignored unless reference_time is also set"
                    )
                },
                "reference_time": {
                    "type": "string",
                    "description": (
                        "ISO 8601 datetime anchor for temporal_direction filtering. "
                        "Ignored unless temporal_direction is also set. Only applies to search_mode='summaries'. "
                        "Example: '2024-10-15T14:00:00'"
                    )
                },
                "segment_id": {
                    "type": "string",
                    "description": (
                        "First 8 characters of a segment UUID, as returned in segment_id from a prior summary search. "
                        "Required for 'search_within_segment'. Example: 'abc12345'"
                    )
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Results per page. Defaults to 10 (summaries) or 20 (messages/search_within_segment). Integer, 1-20"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number, starting from 1. Default: 1. Check has_more_pages in the response to determine if more pages exist"
                },
                "message_id": {
                    "type": "string",
                    "description": (
                        "message_id value from a prior search result (8-char UUID prefix). "
                        "Required for 'expand_message'. Example: 'a7b3c4d5'"
                    )
                },
                "direction": {
                    "type": "string",
                    "enum": ["before", "after", "both"],
                    "description": (
                        "For 'expand_message': which neighboring messages to include. "
                        "'before': earlier messages only. 'after': later messages only. "
                        "'both': messages in both directions. Default: 'both'"
                    )
                },
                "context_count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Messages to include per direction around the expanded message. 0 returns only the target message. Default: 2, max: 10. Only for 'expand_message'"
                },
                "include_thinking": {
                    "type": "boolean",
                    "description": (
                        "When true, assistant messages in results include a thinking_trace field containing "
                        "the reasoning from when the response was generated. Default: false. Applies to all operations"
                    )
                }
            },
            "required": []
        }
    }

    def __init__(self):
        """Initialize the continuum search tool."""
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Load configuration
        config_cls = registry.get("continuum_tool") or ContinuumSearchToolConfig
        self._config = config_cls()

        # Get continuum repository for database access
        self._conversation_repo = get_continuum_repository()

        # Get embeddings provider for query embeddings
        self._embeddings_provider = get_hybrid_embeddings_provider()

    def _escape_like_pattern(self, value: str) -> str:
        """Escape SQL LIKE special characters to prevent wildcard injection."""
        return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

    # Parameter sets for each operation to filter kwargs
    _SEARCH_PARAMS = {
        'query', 'search_mode', 'entities', 'max_results', 'page',
        'start_time', 'end_time', 'temporal_direction', 'reference_time',
        'include_thinking'
    }
    _SEARCH_WITHIN_SEGMENT_PARAMS = {'segment_id', 'query', 'max_results', 'include_thinking'}
    _EXPAND_MESSAGE_PARAMS = {'message_id', 'direction', 'context_count', 'include_thinking'}
    _GET_TOOL_RESULT_PARAMS = {'tool_result_id'}

    def run(self, operation: str = "search", **kwargs) -> Any:
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
                # Filter to only accepted parameters to avoid unexpected keyword argument errors
                filtered = {k: v for k, v in kwargs.items() if k in self._SEARCH_PARAMS}
                return self._search_messages(**filtered)
            elif operation == "search_within_segment":
                filtered = {k: v for k, v in kwargs.items() if k in self._SEARCH_WITHIN_SEGMENT_PARAMS}
                return self._search_within_segment(**filtered)
            elif operation == "expand_message":
                filtered = {k: v for k, v in kwargs.items() if k in self._EXPAND_MESSAGE_PARAMS}
                return self._expand_message(**filtered)
            elif operation == "get_tool_result":
                filtered = {k: v for k, v in kwargs.items() if k in self._GET_TOOL_RESULT_PARAMS}
                return self._get_tool_result(**filtered)
            else:
                raise ValueError(
                    f"Unknown operation: {operation}. "
                    "Valid operations are: search, search_within_segment, "
                    "expand_message, get_tool_result"
                )
        except Exception as e:
            self.logger.error(f"Error executing {operation} in continuum_tool: {e}")
            raise

    def _get_tool_result(self, tool_result_id: str | None = None) -> str:
        """Return the exact persisted tool result referenced by a breadcrumb."""
        if not tool_result_id or not tool_result_id.startswith("tr_"):
            raise ValueError("tool_result_id must be copied from a collapse breadcrumb")

        session_id = get_current_segment_id()
        if not session_id:
            raise ValueError("No active conversation session for tool result lookup")

        content = self._conversation_repo.load_tool_result_by_id(
            user_id=self.user_id,
            session_id=session_id,
            tool_result_id=tool_result_id,
        )
        if content is None:
            raise ValueError(
                "No tool result found for that identifier in the active session"
            )
        return content

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
        reference_time: Optional[str] = None,
        include_thinking: bool = False
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

        # Validate numeric parameters - LLM sometimes sends lists instead of scalars
        max_results = coerce_to_int(max_results, "max_results")
        page = coerce_to_int(page, "page") or 1

        # Validate search mode
        if search_mode not in ["summaries", "precis", "messages"]:
            raise ValueError(f"search_mode must be 'summaries', 'precis', or 'messages', got: {search_mode}")

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
                page=page,
                include_thinking=include_thinking
            )

        # Summary or precis mode
        return self._search_summaries(
            query=query,
            entities=entities,
            max_results=max_results,
            page=page,
            temporal_direction=temporal_direction,
            reference_time=reference_time,
            precis_mode=(search_mode == "precis")
        )

    def _search_summaries(
        self,
        query: str,
        entities: List[str],
        max_results: Optional[int] = None,
        page: int = 1,
        temporal_direction: Optional[str] = None,
        reference_time: Optional[str] = None,
        precis_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Search segment summaries using hybrid vector + BM25 search.

        Combines semantic similarity from embeddings with keyword relevance from
        BM25 text search. Supports temporal filtering with direction and reference time.

        When precis_mode=True, returns the 2-sentence precis instead of the full
        summary and skips segments without a precis. Higher default result limit.

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
        # Set defaults — precis results are compact so default to more
        limit = max_results or (20 if precis_mode else 10)
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

        precis_clause = "AND m.metadata->>'precis' IS NOT NULL AND m.metadata->>'precis' != ''" if precis_mode else ""

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
                # ± configured days from reference time
                days = self._config.temporal_around_days
                start_time = ref_time - timedelta(days=days)
                end_time = ref_time + timedelta(days=days)
                temporal_clause = "AND m.created_at BETWEEN %s AND %s"
                temporal_params = [start_time, end_time]

        db = self._conversation_repo._get_client(self.user_id)

        # Hybrid search query - temporal_clause is SQL structure, not data
        search_sql = f"""
            WITH vector_search AS (
                -- Vector similarity search on segment embeddings (uses HNSW index)
                SELECT
                    m.id,
                    m.continuum_id,
                    m.created_at,
                    m.metadata,
                    m.content,
                    1 - (m.segment_embedding <=> %s::vector) AS vector_score
                FROM messages m
                WHERE m.metadata->>'is_segment_boundary' = 'true'
                  AND m.metadata->>'status' = 'collapsed'
                  AND m.segment_embedding IS NOT NULL
                  {precis_clause}
                  {temporal_clause}
                ORDER BY m.segment_embedding <=> %s::vector
                LIMIT %s
            ),
            text_search AS (
                -- BM25 text search on segment summaries (stored in content field)
                SELECT
                    m.id,
                    ts_rank_cd(
                        to_tsvector('english', m.content),
                        plainto_tsquery('english', %s)
                    ) AS text_rank
                FROM messages m
                WHERE m.id IN (SELECT id FROM vector_search)
                  AND m.content IS NOT NULL
                  AND to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
            )
            SELECT
                v.id,
                v.continuum_id,
                v.created_at,
                v.metadata,
                v.content,
                v.vector_score,
                COALESCE(t.text_rank, 0.0) AS text_rank,
                -- Hybrid score: tanh normalizes BM25 (unbounded) to [0,1] to match vector_score range
                ({self._config.hybrid_vector_weight} * v.vector_score + {1.0 - self._config.hybrid_vector_weight} * TANH(COALESCE(t.text_rank, 0.0))) AS hybrid_score
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
        # 1. embedding for vector_score calculation
        # 2. temporal_params for WHERE clause (if present)
        # 3. embedding for ORDER BY (HNSW index)
        # 4. vector_limit for LIMIT
        # 5. query params for text search (x2)
        # 6. offset and limit for final pagination
        params = [embedding_str]
        params.extend(temporal_params)  # Add temporal filter params if any
        params.extend([embedding_str])  # ORDER BY uses HNSW index
        params.extend([vector_limit])
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

            # Get summary from content field (not metadata)
            summary = row.get("content") or ""

            precis = metadata.get("precis", "")

            # Entity boosting — match against synopsis (full keyword density) regardless of mode
            boost = 1.0
            matched_entities = []
            if entities and summary:
                summary_lower = summary.lower()
                for entity in entities:
                    if entity.lower() in summary_lower:
                        matched_entities.append(entity)
                # Boost by 10% per matched entity
                boost = 1.0 + (0.1 * len(matched_entities))

            result_entry = {
                "result_type": "segment_precis" if precis_mode else "segment_summary",
                "segment_id": str(row["id"])[:8],
                "display_title": metadata.get("display_title", "Conversation segment"),
                "confidence_score": min(row["hybrid_score"] * boost, 1.0),
                "time_boundaries": {
                    "start": metadata.get("segment_start_time"),
                    "end": metadata.get("segment_end_time")
                },
                "tools_used": metadata.get("tools_used", []),
                "matched_entities": matched_entities,
                "created_at": format_utc_iso(row["created_at"])
            }

            if precis_mode:
                result_entry["precis"] = precis
            else:
                result_entry["summary"] = summary or "No summary available"

            results.append(result_entry)

        # Apply smart filtering — skip for precis mode (designed for large result sets)
        unfiltered_count = len(results)
        filtered_results = results if precis_mode else self._filter_results_by_confidence(results)

        # Calculate overall confidence
        if not filtered_results:
            confidence = 0.0
            status = "no_results"
        else:
            top_score = filtered_results[0]["confidence_score"]
            if top_score >= self._config.high_confidence_threshold:
                status = "high_confidence"
            elif top_score >= self._config.medium_confidence_threshold:
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
            "filtered_from": unfiltered_count,
            "page": page,
            "has_more_pages": unfiltered_count == limit,
            "search_mode": "precis" if precis_mode else "summaries",
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

    def _format_message_preview(
        self, message: Dict[str, Any], include_thinking: bool = False
    ) -> Dict[str, Any]:
        """
        Format a message as a preview with truncated content.

        Args:
            message: Raw message data from database
            include_thinking: Whether to include thinking traces for assistant messages

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

        rank = message.get("rank", 0.0)
        result = {
            "message_id": short_id,
            "full_uuid": full_uuid,  # Include but don't show to user
            "continuum_id": message.get("continuum_id", ""),
            "role": message.get("role", ""),
            "timestamp": format_utc_iso(message.get("created_at")) if message.get("created_at") else None,
            "preview": preview,
            "is_truncated": is_truncated,
            "full_length": len(content),
            "match_score": rank,
            "confidence_score": rank,  # Alias for confidence filtering compatibility
            "matched_entities": message.get("matched_entities", [])
        }

        if include_thinking and message.get("role") == "assistant":
            metadata = message.get("metadata") or {}
            thinking = metadata.get("thinking")
            result["thinking_trace"] = thinking
            if thinking is None:
                result["thinking_note"] = "No thinking trace stored for this response"

        return result

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
        page: int = 1,
        include_thinking: bool = False
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

        # Hybrid BM25 + trigram ranking within time boundaries
        # No hard text filter - time scope from segment search is sufficient
        # BM25 handles stemming (run/running), trigrams handle typos (Talyor/Taylor)
        search_sql = """
            SELECT
                m.id,
                m.continuum_id,
                m.role,
                m.content,
                m.created_at,
                m.metadata,
                GREATEST(
                    ts_rank_cd(
                        to_tsvector('english', m.content),
                        plainto_tsquery('english', %s)
                    ),
                    word_similarity(%s, m.content) * 0.5
                ) AS rank
            FROM messages m
            WHERE m.created_at >= %s
              AND m.created_at <= %s
              AND m.content IS NOT NULL
              AND m.content <> ''
              AND (m.metadata->>'is_segment_boundary' IS NULL
                   OR m.metadata->>'is_segment_boundary' = 'false')
            ORDER BY rank DESC, m.created_at ASC
            OFFSET %s
            LIMIT %s
        """

        rows = db.execute_query(search_sql, (query, query, start_ts, end_ts, offset, limit))

        # Process results with entity boosting
        results = []
        for row in rows:
            # Normalize unbounded BM25 score to 0-1 range (matches summary search tanh normalization)
            rank = math.tanh(float(row.get("rank", 0)))

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

        # Format as message previews (includes confidence_score for filtering)
        formatted_results = [self._format_message_preview(msg, include_thinking=include_thinking) for msg in results]

        # Apply smart filtering based on confidence clustering
        unfiltered_count = len(formatted_results)
        filtered_results = self._filter_results_by_confidence(formatted_results)

        # Calculate overall confidence from filtered results
        if not filtered_results:
            confidence = 0.0
            status = "no_results"
        else:
            confidence = filtered_results[0]["confidence_score"]
            if confidence >= self._config.high_confidence_threshold:
                status = "high_confidence"
            elif confidence >= self._config.medium_confidence_threshold:
                status = "medium_confidence"
            else:
                status = "low_confidence"

        return {
            "status": status,
            "confidence": confidence,
            "query": query,
            "entities": entities,
            "results": filtered_results,
            "result_count": len(filtered_results),
            "filtered_from": unfiltered_count,
            "page": page,
            "has_more_pages": unfiltered_count == limit,
            "search_mode": "messages",
            "time_boundaries": {
                "start": start_time,
                "end": end_time
            },
            "meta": {
                "search_tier": "bm25_timeframe",
                "message_count": unfiltered_count
            }
        }

    def _search_within_segment(
        self,
        segment_id: str,
        query: str,
        max_results: Optional[int] = None,
        include_thinking: bool = False
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

        # Validate numeric parameter
        max_results = coerce_to_int(max_results, "max_results")

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

        rows = db.execute_query(segment_sql, (f"{self._escape_like_pattern(segment_id)}%",))

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
                end_time = format_utc_iso(utc_now())

        # Use the time-bounded message search
        result = self._search_messages_in_timeframe(
            query=query,
            start_time=start_time,
            end_time=end_time,
            entities=[],  # No entity extraction for within-segment search
            max_results=max_results,
            page=1,
            include_thinking=include_thinking
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
        context_count: Optional[int] = None,
        include_thinking: bool = False
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

        # Validate and set context count
        context_count = coerce_to_int(context_count, "context_count")
        if context_count is None:
            context_count = self._config.default_context_window
        context_count = max(0, min(context_count, 10))

        # Find the full UUID and message
        origin_message = self._find_message_by_short_id(message_id)
        if not origin_message:
            raise ValueError(f"No message found with ID starting with '{message_id}'")

        origin_dict = {
            "message_id": message_id,
            "full_uuid": origin_message["id"],
            "continuum_id": origin_message["continuum_id"],
            "role": origin_message["role"],
            "content": origin_message["content"],
            "timestamp": format_utc_iso(origin_message["created_at"]) if origin_message.get("created_at") else None,
            "is_truncated": False
        }

        if include_thinking and origin_message["role"] == "assistant":
            metadata = origin_message.get("metadata") or {}
            thinking = metadata.get("thinking")
            origin_dict["thinking_trace"] = thinking
            if thinking is None:
                origin_dict["thinking_note"] = "No thinking trace stored for this response"

        result = {
            "status": "expanded",
            "origin_message": origin_dict
        }

        # Fetch context messages if requested
        if context_count > 0:
            if direction in ["before", "both"]:
                result["context_before"] = self._fetch_context_messages(
                    origin_message,
                    direction="before",
                    count=context_count,
                    include_thinking=include_thinking
                )

            if direction in ["after", "both"]:
                result["context_after"] = self._fetch_context_messages(
                    origin_message,
                    direction="after",
                    count=context_count,
                    include_thinking=include_thinking
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

            pattern = f"{self._escape_like_pattern(short_id)}%"
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
        count: int,
        include_thinking: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch context messages before or after the origin message.

        Args:
            origin_message: The reference message
            direction: "before" or "after"
            count: Number of messages to fetch
            include_thinking: Whether to include thinking traces for assistant messages

        Returns:
            List of context messages with relation indicators
        """
        try:
            db = self._conversation_repo._get_client(self.user_id)

            origin_time = origin_message["created_at"]
            continuum_id = origin_message["continuum_id"]

            if direction == "before":
                query = """
                    SELECT id, role, content, created_at, metadata
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
                    SELECT id, role, content, created_at, metadata
                    FROM messages
                    WHERE continuum_id = %s
                      AND created_at > %s
                    ORDER BY created_at ASC
                    LIMIT %s
                """
                rows = db.execute_query(query, (continuum_id, origin_time, count))

            context = []
            for idx, row in enumerate(rows, 1):
                msg = {
                    "message_id": str(row["id"])[:8],
                    "full_uuid": str(row["id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "timestamp": format_utc_iso(row["created_at"]) if row.get("created_at") else None,
                    "relation": f"[{idx} message{'s' if idx > 1 else ''} {direction} origin]"
                }

                if include_thinking and row["role"] == "assistant":
                    metadata = row.get("metadata") or {}
                    thinking = metadata.get("thinking")
                    msg["thinking_trace"] = thinking
                    if thinking is None:
                        msg["thinking_note"] = "No thinking trace stored for this response"

                context.append(msg)

            return context

        except Exception as e:
            # WARNING not ERROR: empty context is non-fatal - user still gets main message
            self.logger.warning(f"Failed to fetch context messages: {e}")
            return []
