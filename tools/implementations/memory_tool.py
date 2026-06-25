"""
Memory management tool for manual memory operations.

Provides search, create, link, and annotate operations for long-term memories.
This tool gives full domain ownership over memory operations, replacing the
memory search functionality previously in continuum_tool.

Memory creation is deferred: create_memory() queues memories to Valkey for
processing at segment collapse. This keeps tool initialization lightweight
(no spaCy model loading) and moves heavy operations (embedding generation,
entity extraction) to the existing extraction pipeline.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Dict, Any, Optional, List
from uuid import uuid4

from pydantic import BaseModel, Field

from tools.repo import Tool, coerce_to_int
from tools.registry import registry
from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
from clients.valkey_client import get_valkey_client
from lt_memory.db_access import LTMemoryDB
from lt_memory.hybrid_search import HybridSearcher
from lt_memory.models import Memory, MemoryLink, TraversalResult, VALID_RELATIONSHIP_TYPES, PendingManualMemory
from utils.database_session_manager import get_shared_session_manager
from utils.timezone_utils import utc_now, format_utc_iso
from utils.tag_parser import format_memory_id, parse_memory_id
from utils.user_context import get_current_segment_id


logger = logging.getLogger(__name__)

# Valkey key pattern for pending memories
PENDING_MEMORIES_KEY_PREFIX = "pending_memories"
PENDING_MEMORIES_TTL = 86400  # 24 hours


class MemoryToolConfig(BaseModel):
    """Configuration for the memory_tool."""

    min_text_length: int = Field(
        default=10,
        description="Minimum characters for memory text"
    )
    manual_link_confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for manually created links"
    )
    default_importance_user_requested: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default importance for user-requested memories"
    )
    default_importance_self_directed: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default importance for MIRA-initiated memories"
    )
    confidence_cluster_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description="Score threshold for clustering similar results"
    )
    max_search_results: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum results to return from search"
    )


# Register with registry
registry.register("memory_tool", MemoryToolConfig)


class MemoryTool(Tool):
    """
    Memory management tool for search, creation, linking, and annotation.

    Provides manual control over long-term memories with graph-aware search
    and relationship management capabilities.

    Architecture note: Memory creation is queued (not immediate). The tool
    stores pending memories in Valkey, which are processed at segment collapse
    by the existing LT_Memory pipeline. This keeps initialization lightweight
    (no EntityExtractor/spaCy) and defers heavy operations appropriately.
    """

    name = "memory_tool"

    # Operations without ordering dependencies — safe for concurrent execution.
    # create_memory only queues to Valkey, touch applies an independent boost.
    _parallel_safe_operations = frozenset({"search", "touch", "create_memory"})

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    simple_description = (
        "Manage long-term memories: search with graph traversal, create memories, "
        "link related memories, add annotations, touch referenced memories."
    )

    tool_schema = {
        "name": "memory_tool",
        "description": "Search and manage long-term memories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "create_memory", "link_memories", "annotate_memory", "touch"],
                    "description": "Operation to perform"
                },
                # Search parameters
                "query": {
                    "type": "string",
                    "description": "Natural language search query (required for 'search')"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum primary results per page, 1-20, default 10"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination, default 1"
                },
                # Create memory parameters
                "content": {
                    "type": "string",
                    "description": "The memory content to store. Minimum 10 characters after whitespace trimming. Required for 'create_memory'"
                },
                "user_requested": {
                    "type": "boolean",
                    "description": "True if user explicitly asked to remember this. Sets importance to 0.7 if true, 0.5 if false. Default false"
                },
                "happens_at": {
                    "type": "string",
                    "description": "ISO 8601 datetime or past-relative phrase (e.g. '6 months ago', 'last Tuesday') for when the remembered event occurred"
                },
                "expires_at": {
                    "type": "string",
                    "description": "ISO 8601 datetime or future-relative phrase (e.g. 'in 3 months') after which this memory should be considered expired"
                },
                "supersedes_memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "mem_XXXXXXXX IDs of existing memories this new memory replaces. Superseded memories receive a 70% score penalty in search results"
                },
                # Link memories parameters
                "source_memory_id": {
                    "type": "string",
                    "description": "mem_XXXXXXXX ID of the source memory in the link. The link reads as: source [link_type] target"
                },
                "target_memory_id": {
                    "type": "string",
                    "description": "mem_XXXXXXXX ID of the memory the relationship points to. Required for 'link_memories'"
                },
                "link_type": {
                    "type": "string",
                    "enum": ["corroborates", "conflicts", "supersedes", "refines", "precedes", "contextualizes", "exemplifies"],
                    "description": "Directional relationship from source to target. The link reads: source [link_type] target"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation for why this link exists. Min 5 chars. Stored with the link for future reference. Required for 'link_memories'"
                },
                # Annotate memory parameters
                "memory_id": {
                    "type": "string",
                    "description": "mem_XXXXXXXX ID of the memory to annotate. Required for 'annotate_memory'"
                },
                "annotation": {
                    "type": "string",
                    "description": "Annotation content appended to the memory's annotation list. Min 3 chars. Required for 'annotate_memory'"
                },
                # Touch parameters
                "memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of mem_XXXXXXXX IDs that were actually used in your response. Must be non-empty. Required for 'touch'"
                }
            },
            "required": ["operation"]
        }
    }

    def __init__(self):
        """
        Initialize the memory tool with lightweight services only.

        No EntityExtractor (spaCy) loading here - entity extraction happens
        at segment collapse via the existing LT_Memory pipeline.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Load configuration
        config_cls = registry.get("memory_tool") or MemoryToolConfig
        self._config = config_cls()

        # Lightweight service initialization (no spaCy)
        self._embeddings_provider = get_hybrid_embeddings_provider()
        session_manager = get_shared_session_manager()
        self._memory_db = LTMemoryDB(session_manager)
        self._hybrid_searcher = HybridSearcher(self._memory_db)
        self._valkey = get_valkey_client()

        # HubDiscoveryService is lazy-loaded only when search needs entity discovery
        # It has its own EntityExtractor, but only loads when actually used
        self._hub_discovery = None

    def _get_hub_discovery(self):
        """Lazy-load HubDiscoveryService for search operations."""
        if self._hub_discovery is None:
            from lt_memory.hub_discovery import HubDiscoveryService
            from lt_memory.entity_extraction import EntityExtractor

            self._entity_extractor = EntityExtractor()
            self._hub_discovery = HubDiscoveryService(
                db=self._memory_db,
            )
        return self._hub_discovery

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a memory operation.

        Args:
            operation: Operation to perform (search, create_memory, link_memories, annotate_memory)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ValueError: If operation fails or parameters are invalid
        """
        try:
            if operation == "search":
                return self._search(**kwargs)
            elif operation == "create_memory":
                return self._create_memory(**kwargs)
            elif operation == "link_memories":
                return self._link_memories(**kwargs)
            elif operation == "annotate_memory":
                return self._annotate_memory(**kwargs)
            elif operation == "touch":
                return self._touch(**kwargs)
            else:
                raise ValueError(
                    f"Unknown operation: {operation}. "
                    f"Valid operations are: search, create_memory, link_memories, annotate_memory, touch"
                )
        except Exception as e:
            self.logger.error(f"Error executing {operation} in memory_tool: {e}")
            raise

    def _search(
        self,
        query: str,
        max_results: int = 10,
        page: int = 1,
        include_hub_discovery: bool = True,
        include_link_traversal: bool = True,
        traversal_depth: int = 1,
        **kwargs  # Accept extra params gracefully
    ) -> Dict[str, Any]:
        """
        Comprehensive graph-aware memory search.

        Combines hybrid search, entity hub discovery, and link traversal
        to find the most relevant memories.

        Args:
            query: Natural language search query
            max_results: Number of primary results (default 10)
            page: Page number for pagination (default 1)
            include_hub_discovery: Include entity-linked memories (default true)
            include_link_traversal: Traverse links on top results (default true)
            traversal_depth: How many link hops to follow (default 1)

        Returns:
            Search results with primary and related memories
        """
        if not query or not query.strip():
            raise ValueError("Query is required for search operation")

        query = query.strip()
        # Coerce numeric parameters (tool inputs may come as strings/lists from JSON)
        max_results = coerce_to_int(max_results, "max_results") or 10
        page = coerce_to_int(page, "page") or 1
        traversal_depth = coerce_to_int(traversal_depth, "traversal_depth") or 1
        limit = min(max_results, self._config.max_search_results)
        offset = (page - 1) * limit

        # Generate query embedding (768d realtime encoder)
        query_embedding = self._embeddings_provider.encode_realtime(query)

        # Extract entities from query for hub discovery (lazy-load EntityExtractor)
        extracted_entities = []
        if include_hub_discovery:
            hub_discovery = self._get_hub_discovery()
            extracted_entities = list(self._entity_extractor.extract_entities(query))

        # Run parallel retrieval
        def _fetch_similarity_pool():
            """Fetch similarity-based memories via hybrid search."""
            return self._hybrid_searcher.hybrid_search(
                query_text=query,
                query_embedding=query_embedding.tolist(),
                search_intent="general",
                limit=limit * 2,  # Oversample for filtering
                similarity_threshold=0.5,
                min_importance=0.1
            )

        def _fetch_hub_pool():
            """Fetch hub-derived memories via entity navigation."""
            if not include_hub_discovery or not extracted_entities:
                return []
            hub_discovery = self._get_hub_discovery()
            return hub_discovery.discover_hub_memories(
                extracted_entities=extracted_entities,
                expansion_embedding=query_embedding,
                top_n=limit * 2
            )

        # Execute searches in parallel with context propagation
        with ThreadPoolExecutor(max_workers=2) as executor:
            ctx_similarity = copy_context()
            ctx_hub = copy_context()
            similarity_future = executor.submit(ctx_similarity.run, _fetch_similarity_pool)
            hub_future = executor.submit(ctx_hub.run, _fetch_hub_pool)

            similarity_pool: List[Memory] = similarity_future.result()
            hub_pool: List[Memory] = hub_future.result()

        # Merge pools (deduplicate, similarity pool takes precedence)
        merged_results = self._merge_memory_pools(similarity_pool, hub_pool)

        # Apply pagination
        paginated_results = merged_results[offset:offset + limit]

        if not paginated_results:
            return {
                "status": "no_results",
                "confidence": 0.0,
                "query": query,
                "results": [],
                "result_count": 0,
                "page": page,
                "has_more_pages": False
            }

        # Link traversal on top results
        if include_link_traversal:
            from lt_memory.linking import LinkingService
            from lt_memory.vector_ops import VectorOps

            vector_ops = VectorOps(self._embeddings_provider, self._memory_db)
            linking_service = LinkingService(
                vector_ops=vector_ops,
                db=self._memory_db
            )

            for memory in paginated_results[:5]:  # Traverse top 5
                related: List[TraversalResult] = linking_service.traverse_related(memory.id, depth=traversal_depth)
                memory.linked_memories = [r["memory"] for r in related]
                # Store metadata for each linked memory
                for i, linked in enumerate(memory.linked_memories):
                    linked.link_metadata = {
                        "link_type": related[i]["link_type"],
                        "confidence": related[i]["confidence"],
                        "reasoning": related[i]["reasoning"],
                        "depth": related[i]["depth"],
                        "linked_from_id": related[i]["linked_from_id"]
                    }

        # Format results
        formatted_results = []
        for memory in paginated_results:
            result = self._format_memory_for_output(memory)
            if include_link_traversal and hasattr(memory, 'linked_memories') and memory.linked_memories:
                result["related_memories"] = [
                    self._format_memory_for_output(linked)
                    for linked in memory.linked_memories
                ]
            formatted_results.append(result)

        # Calculate confidence from top result
        top_score = paginated_results[0].importance_score if paginated_results else 0.0
        if top_score >= 0.7:
            status = "high_confidence"
        elif top_score >= 0.4:
            status = "medium_confidence"
        else:
            status = "low_confidence"

        return {
            "status": status,
            "confidence": round(top_score, 3),
            "query": query,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "page": page,
            "has_more_pages": len(merged_results) > offset + limit,
            "meta": {
                "similarity_pool_size": len(similarity_pool),
                "hub_pool_size": len(hub_pool),
                "entities_extracted": extracted_entities
            }
        }

    def _merge_memory_pools(self, similarity_pool: List, hub_pool: List) -> List:
        """
        Merge similarity and hub-derived pools with supersedes penalty.

        Args:
            similarity_pool: Memories from hybrid search
            hub_pool: Memories from hub discovery

        Returns:
            Merged, deduplicated list sorted by effective score
        """
        merged = list(similarity_pool)
        seen_ids = {m.id for m in similarity_pool}

        for memory in hub_pool:
            if memory.id not in seen_ids:
                merged.append(memory)
                seen_ids.add(memory.id)

        def effective_score(memory) -> float:
            score = memory.importance_score

            # Apply supersedes penalty for soft demotion
            if memory.inbound_links:
                has_supersedes = any(
                    link.get('type') == 'supersedes'
                    for link in memory.inbound_links
                )
                if has_supersedes:
                    score *= 0.3  # 70% reduction
                    self.logger.debug(f"Supersedes penalty for {memory.id}")

            return score

        merged.sort(key=effective_score, reverse=True)
        return merged

    def _format_memory_for_output(self, memory) -> Dict[str, Any]:
        """Format a Memory model for API output."""
        result = {
            "memory_id": format_memory_id(str(memory.id)),
            "full_uuid": str(memory.id),
            "text": memory.text,
            "importance_score": round(memory.importance_score, 3),
            "similarity_score": round(memory.similarity_score, 3) if memory.similarity_score else None,
            "created_at": format_utc_iso(memory.created_at),
            "happens_at": format_utc_iso(memory.happens_at) if memory.happens_at else None,
            "expires_at": format_utc_iso(memory.expires_at) if memory.expires_at else None,
            "entity_links": memory.entity_links or [],
            "inbound_links_count": len(memory.inbound_links) if memory.inbound_links else 0,
            "outbound_links_count": len(memory.outbound_links) if memory.outbound_links else 0,
            "annotations": memory.annotations if hasattr(memory, 'annotations') and memory.annotations else [],
            # source_segment_id enables tracing a memory back to its origin conversation.
            # A future "investigate" operation could use this + continuum_tool to pull
            # the source segment's messages, but deferred — rarely needed and adds a
            # new tool parameter for a niche use case.
            "source_segment_id": str(memory.source_segment_id) if hasattr(memory, 'source_segment_id') and memory.source_segment_id else None
        }

        if hasattr(memory, 'link_metadata') and memory.link_metadata:
            result["link_metadata"] = memory.link_metadata

        return result

    def _create_memory(
        self,
        content: str,
        user_requested: bool = False,
        importance_score: Optional[float] = None,
        happens_at: Optional[str] = None,
        expires_at: Optional[str] = None,
        supersedes_memory_ids: Optional[List[str]] = None,
        **kwargs  # Accept extra params gracefully
    ) -> Dict[str, Any]:
        """
        Queue a new memory for processing at segment collapse.

        Memory is stored in Valkey and will be processed (embedding generation,
        entity extraction, linking) when the conversation segment closes.
        This keeps tool invocation fast and defers heavy operations.

        Args:
            content: Memory content (min 10 chars)
            user_requested: True if user explicitly said "remember this"
            importance_score: Override importance (0.0-1.0)
            happens_at: ISO timestamp for event
            expires_at: ISO timestamp for expiration
            supersedes_memory_ids: List of 8-char IDs to supersede

        Returns:
            Queue confirmation with pending_id for tracking
        """
        # Validate content
        if not content or len(content.strip()) < self._config.min_text_length:
            raise ValueError(
                f"Content must be at least {self._config.min_text_length} characters"
            )
        content = content.strip()

        # Determine importance score (cast to float - tool inputs may be strings from JSON)
        if importance_score is not None:
            score = max(0.0, min(1.0, float(importance_score)))
        elif user_requested:
            score = self._config.default_importance_user_requested
        else:
            score = self._config.default_importance_self_directed

        # Get current segment ID from ambient context
        segment_id = get_current_segment_id()

        # Build pending memory
        pending = PendingManualMemory(
            text=content,
            importance_score=score,
            user_requested=user_requested,
            happens_at=happens_at,
            expires_at=expires_at,
            supersedes_memory_ids=supersedes_memory_ids or [],
            queued_at=format_utc_iso(utc_now()),
            pending_id=str(uuid4())
        )

        # Queue to Valkey
        if segment_id:
            queue_key = f"{PENDING_MEMORIES_KEY_PREFIX}:{self.user_id}:{segment_id}"
        else:
            # No segment yet (first message) - use user-level queue
            # Will be processed when first segment collapses
            queue_key = f"{PENDING_MEMORIES_KEY_PREFIX}:{self.user_id}:presegment"

        self._valkey.rpush(queue_key, pending.to_json())
        self._valkey.expire(queue_key, PENDING_MEMORIES_TTL)

        self.logger.info(
            f"Queued memory {pending.pending_id} for user {self.user_id}, "
            f"segment {segment_id or 'presegment'}"
        )

        return {
            "status": "queued",
            "pending_id": pending.pending_id,
            "will_process_at": "segment_collapse",
            "text_preview": content[:50] + "..." if len(content) > 50 else content,
            "importance_score": score,
            "message": "Memory queued for processing when conversation segment closes"
        }

    def _link_memories(
        self,
        source_memory_id: str,
        target_memory_id: str,
        link_type: str,
        reasoning: str,
        **kwargs  # Accept extra params gracefully
    ) -> Dict[str, Any]:
        """
        Create a relationship link between two memories.

        Args:
            source_memory_id: 8-char ID of source memory
            target_memory_id: 8-char ID of target memory
            link_type: Relationship type (supports, conflicts, supersedes, refines, precedes, contextualizes)
            reasoning: Explanation for the link (min 5 chars)

        Returns:
            Link creation confirmation
        """
        # Validate link_type
        allowed_types = VALID_RELATIONSHIP_TYPES - {"null"}  # null means no relationship
        if link_type not in allowed_types:
            raise ValueError(f"link_type must be one of {allowed_types}, got '{link_type}'")

        # Validate reasoning
        if not reasoning or len(reasoning.strip()) < 5:
            raise ValueError("Reasoning must be at least 5 characters")
        reasoning = reasoning.strip()

        # Resolve short IDs to full UUIDs
        source = self._find_memory_by_short_id(source_memory_id)
        target = self._find_memory_by_short_id(target_memory_id)

        if not source:
            raise ValueError(f"Source memory '{source_memory_id}' not found")
        if not target:
            raise ValueError(f"Target memory '{target_memory_id}' not found")
        if source.id == target.id:
            raise ValueError("Cannot link memory to itself")

        # Create MemoryLink with confidence=1.0 (manual = high confidence)
        link = MemoryLink(
            source_id=source.id,
            target_id=target.id,
            link_type=link_type,
            confidence=self._config.manual_link_confidence,
            reasoning=reasoning,
            created_at=utc_now()
        )

        # Store bidirectionally
        self._memory_db.create_links([link])

        self.logger.info(f"Created {link_type} link: {source.id} <-> {target.id}")

        return {
            "status": "linked",
            "source_memory_id": format_memory_id(str(source.id)),
            "target_memory_id": format_memory_id(str(target.id)),
            "link_type": link_type,
            "confidence": self._config.manual_link_confidence,
            "message": f"Created '{link_type}' link between {format_memory_id(str(source.id))} and {format_memory_id(str(target.id))}"
        }

    def _annotate_memory(
        self,
        memory_id: str,
        annotation: str,
        **kwargs  # Accept extra params gracefully
    ) -> Dict[str, Any]:
        """
        Add an annotation to an existing memory.

        Args:
            memory_id: 8-char ID of memory to annotate
            annotation: Note to add (min 3 chars)

        Returns:
            Annotation confirmation
        """
        # Validate annotation
        if not annotation or len(annotation.strip()) < 3:
            raise ValueError("Annotation must be at least 3 characters")
        annotation = annotation.strip()

        # Resolve short ID
        memory = self._find_memory_by_short_id(memory_id)
        if not memory:
            raise ValueError(f"Memory '{memory_id}' not found")

        # Build annotation object
        annotation_entry = {
            "text": annotation,
            "created_at": format_utc_iso(utc_now()),
            "source": "mira"
        }

        # Append to existing annotations
        existing = memory.annotations if hasattr(memory, 'annotations') and memory.annotations else []
        updated = existing + [annotation_entry]

        # Update via db_access (serialize to JSON string for JSONB column)
        self._memory_db.update_memory(memory.id, {"annotations": json.dumps(updated)})

        self.logger.info(f"Added annotation to memory {memory.id}")

        return {
            "status": "annotated",
            "memory_id": format_memory_id(str(memory.id)),
            "annotation_count": len(updated),
            "annotation": annotation,
            "message": f"Added annotation to {format_memory_id(str(memory.id))} ({len(updated)} total annotations)"
        }

    def _find_memory_by_short_id(self, short_id: str):
        """
        Find a memory by the first 8 characters of its UUID.

        Args:
            short_id: Either "mem_XXXXXXXX" or raw "XXXXXXXX"

        Returns:
            Memory model or None if not found
        """
        clean_id = parse_memory_id(short_id)
        if not clean_id or len(clean_id) < 8:
            return None

        # Query by UUID prefix
        with self._memory_db.session_manager.get_session(self.user_id) as session:
            query = """
            SELECT * FROM memories
            WHERE REPLACE(id::text, '-', '') LIKE %(pattern)s
              AND is_archived = FALSE
            LIMIT 1
            """
            result = session.execute_single(query, {'pattern': f"{clean_id.lower()}%"})

            if result:
                return Memory(**result)
            return None

    def _touch(
        self,
        memory_ids: List[str],
        **kwargs  # Accept extra params gracefully
    ) -> Dict[str, Any]:
        """
        Record that surfaced memories were referenced in the response.

        Resolves short IDs to full UUIDs and applies mention_count boost
        immediately. Called as a post-response tool call instead of inline
        XML tags, which have low LLM compliance.

        Args:
            memory_ids: List of mem_XXXXXXXX short IDs that were referenced

        Returns:
            Touch confirmation with resolved UUIDs and any failed IDs

        Raises:
            ValueError: If memory_ids is empty or missing
        """
        if not memory_ids:
            raise ValueError("memory_ids is required and must be non-empty")

        resolved_uuids: List[str] = []
        failed_ids: List[str] = []

        for short_id in memory_ids:
            memory = self._find_memory_by_short_id(short_id)
            if memory:
                resolved_uuids.append(str(memory.id))
            else:
                failed_ids.append(short_id)

        if failed_ids:
            self.logger.warning(f"Touch: could not resolve IDs: {failed_ids}")

        if resolved_uuids:
            self._memory_db.apply_mention_boost(resolved_uuids)

        return {
            "status": "touched",
            "resolved_uuids": resolved_uuids,
            "boosted_count": len(resolved_uuids),
            "failed_ids": failed_ids,
        }
