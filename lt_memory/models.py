"""
Pydantic models for LT_Memory system.

All data structures for memories, links, batches, and processing chunks.
"""
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import List, Dict, Any, Literal, NamedTuple, NotRequired, Optional, TypedDict
from uuid import UUID

# Valid relationship types for memory linking
# Used across extraction, linking, and processing modules
VALID_RELATIONSHIP_TYPES = frozenset({
    "corroborates", "conflicts", "supersedes", "refines",
    "precedes", "contextualizes", "exemplifies", "null"
})

# Type aliases for Literal-validated fields
RelationshipType = Literal[
    "corroborates", "conflicts", "supersedes", "refines",
    "precedes", "contextualizes", "exemplifies", "null"
]
BatchStatus = Literal[
    "submitted", "processing", "result_processing", "completed", "failed", "expired", "cancelled"
]


# ============================================================================
# Shared TypedDicts — cross-file structures stored in JSONB or returned by
# multiple modules. File-local types stay in their respective files.
# ============================================================================

class MemoryLinkEntry(TypedDict):
    """Entry in Memory.inbound_links / outbound_links JSONB arrays."""
    uuid: str
    type: str
    reasoning: str
    created_at: str
    extraction_bond: NotRequired[str]



class EntityLinkEntry(TypedDict):
    """Entry in Memory.entity_links JSONB array."""
    uuid: str
    type: str
    name: str


class AnnotationEntry(TypedDict):
    """Entry in Memory.annotations JSONB array."""
    text: str
    created_at: str
    source: str
    archived_source_ids: NotRequired[list[str]]
    source_segment_ids: NotRequired[list[str]]  # Segment provenance preserved during consolidation


class LinkMetadata(TypedDict):
    """Transient link context attached during proactive traversal."""
    link_type: str
    reasoning: str
    depth: int
    linked_from_id: UUID


class TraversalResult(TypedDict):
    """Single entry returned by LinkingService.traverse_related()."""
    memory: 'Memory'
    link_type: str | None
    reasoning: str | None
    depth: int
    linked_from_id: UUID | None


class UserMemorySettings(TypedDict):
    """Row from get_users_with_memory_enabled()."""
    id: UUID
    email: str
    memory_manipulation_enabled: bool
    daily_manipulation_last_run: datetime | None
    timezone: str | None


class MemoryPageResult(TypedDict):
    """Paginated memory query result."""
    memories: list[dict[str, Any]]
    has_more: bool
    next_offset: int | None


class NamedEntity(TypedDict):
    """Entity extracted by spaCy with type label."""
    name: str
    entity_type: str


class MemoryContext(TypedDict):
    """Memory context built by ExtractionEngine for extraction prompts."""
    memory_ids: list[str]
    memory_texts: dict[str, str]
    snapshot_timestamp: NotRequired[str]
    uuid_to_short: NotRequired[dict[str, str]]
    short_to_uuid: NotRequired[dict[str, str]]


class MemoryContextSnapshot(TypedDict):
    """Memory context snapshot built by ExtractionOrchestrator for chunks."""
    memory_ids: list[str]
    referenced_memory_ids: list[str]
    memory_texts: dict[str, str]
    pinned_short_ids: list[str]


class EntityPairRow(TypedDict):
    """Row from find_similar_entity_pairs() pg_trgm self-join."""
    id_a: UUID
    name_a: str
    type_a: str
    links_a: int
    id_b: UUID
    name_b: str
    type_b: str
    links_b: int
    sim: float


class ChunkMetadata(TypedDict):
    """Metadata stored with ExtractionBatch for result processing."""
    message_count: int
    short_to_uuid: dict[str, str]
    segment_id: str | None


class MemoryDict(TypedDict):
    """Dictionary representation of a Memory for proactive surfacing."""
    id: str
    text: str
    importance_score: float
    similarity_score: float | None
    source: str
    created_at: str | None
    last_accessed: str | None
    access_count: int
    happens_at: str | None
    expires_at: str | None
    inbound_links: list[MemoryLinkEntry]
    outbound_links: list[MemoryLinkEntry]
    entity_links: list[EntityLinkEntry]
    annotations: list[AnnotationEntry]
    link_metadata: NotRequired[LinkMetadata]
    linked_memories: NotRequired[list['MemoryDict']]


class Memory(BaseModel):
    """
    Represents a stored memory from database.

    Returned by db_access methods for type-safe memory operations.
    Can represent either personal memories (user_id set) or global memories (user_id None).
    """
    id: UUID
    user_id: Optional[UUID] = None  # None for global memories
    text: str
    embedding: Optional[List[float]] = None  # mdbr-leaf-ir-asym (768d)
    importance_score: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    access_count: int = 0
    mention_count: int = 0  # Explicit LLM references (strongest importance signal)
    last_accessed: Optional[datetime] = None
    happens_at: Optional[datetime] = None
    inbound_links: List[MemoryLinkEntry] = Field(default_factory=list)
    outbound_links: List[MemoryLinkEntry] = Field(default_factory=list)
    entity_links: List[EntityLinkEntry] = Field(default_factory=list)
    is_archived: bool = False
    archived_at: Optional[datetime] = None

    # Last timestamp the MemoryCuratorAgent tended this memory (linked/merged/
    # archived/salvaged). NULL = never tended (integration not yet run).
    # The floor trigger samples "memories not tended in N days" via this field.
    last_tended_at: Optional[datetime] = None

    # Activity day snapshots for vacation-proof scoring
    activity_days_at_creation: Optional[int] = None
    activity_days_at_last_access: Optional[int] = None

    # Annotations for manual notes/context
    annotations: List[AnnotationEntry] = Field(
        default_factory=list,
        description="Contextual notes: [{text, created_at, source}]"
    )

    # Source segment for context exploration
    source_segment_id: Optional[UUID] = None  # Segment this memory was extracted from

    # Transient field populated by similarity search queries
    similarity_score: Optional[float] = None

    # Source indicator for hybrid search results (personal vs global)
    source: str = Field(default="personal", description="Memory source: 'personal' or 'global'")

    # Transient fields populated by proactive service during link traversal
    linked_memories: Optional[List['Memory']] = Field(default=None, exclude=True)
    link_metadata: Optional[LinkMetadata] = Field(default=None, exclude=True)



class ExtractionRef(TypedDict):
    """Reference to existing memory with extraction-time bond descriptor."""
    id: str       # Full UUID (after short-ID remapping)
    bond: str     # 3-word relationship descriptor


class ExtractedMemory(BaseModel):
    """
    Memory extracted from continuum chunk.

    Used during extraction pipeline before persistence.
    """
    text: str
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    expires_at: Optional[datetime] = None
    happens_at: Optional[datetime] = None
    related_memory_ids: List[ExtractionRef] = Field(default_factory=list)
    entities: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Entities extracted by LLM: [{'name': str, 'type': PERSON|ORG|PRODUCT|PLACE}]"
    )

    # Source segment for context exploration
    source_segment_id: Optional[UUID] = None  # Segment this memory was extracted from

    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate entities list structure."""
        if not isinstance(v, list):
            return []

        valid_entities = []
        for entity in v:
            if not isinstance(entity, dict):
                continue
            if 'name' not in entity or 'type' not in entity:
                continue
            if not isinstance(entity['name'], str) or not isinstance(entity['type'], str):
                continue
            valid_entities.append(entity)

        return valid_entities

    @field_validator('importance_score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v


class ExtractionResult(BaseModel):
    """
    Result of memory extraction containing extracted memories.
    """
    memories: List['ExtractedMemory']


class MemoryLink(BaseModel):
    """
    Relationship link between memories.

    Stored bidirectionally in memory inbound_links/outbound_links JSONB arrays.
    """
    source_id: UUID
    target_id: UUID
    link_type: RelationshipType
    reasoning: str
    extraction_bond: str = ""  # 3-word relationship descriptor carried from extraction
    created_at: datetime


class Entity(BaseModel):
    """
    Persistent knowledge anchor (entity) that memories link to.

    Entities represent named entities (people, organizations, products, events, etc.)
    extracted from memory text. They serve as knowledge graph nodes enabling
    entity-based memory retrieval and relationship discovery.

    Entity matching uses PostgreSQL trigram fuzzy matching (pg_trgm) on names,
    not vector similarity - appropriate for handling LLM naming variations.
    """
    id: UUID
    user_id: UUID
    name: str = Field(description="Canonical normalized entity name")
    entity_type: str = Field(description="LLM-extracted entity type: PERSON, ORG, GPE, PRODUCT, EVENT, etc.")
    link_count: int = Field(default=0, description="Number of memories linking to this entity")
    last_linked_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of most recent memory link (for dormancy detection)"
    )
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_archived: bool = False
    archived_at: Optional[datetime] = None

    # Transient field populated by similarity search queries
    similarity_score: Optional[float] = None


class ProcessingChunk(BaseModel):
    """
    Ephemeral continuum chunk for batch extraction processing.

    Temporary container that holds messages and metadata during batch
    submission orchestration. Discarded after batch request is built.
    Holds Message objects directly (no conversion to dict).
    """
    messages: List[Any]  # Message objects from cns.core.message
    temporal_start: datetime
    temporal_end: datetime
    chunk_index: int
    memory_context_snapshot: MemoryContextSnapshot | None = None
    segment_id: Optional[UUID] = None  # Source segment for context exploration

    model_config = {"arbitrary_types_allowed": True}

    @field_validator('messages')
    @classmethod
    def validate_messages_not_empty(cls, v: List[Any]) -> List[Any]:
        """Ensure chunk has at least one message."""
        if not v:
            raise ValueError("ProcessingChunk must contain at least one message")
        return v

    @classmethod
    def from_conversation_messages(
        cls,
        messages: List[Any],  # Message objects
        chunk_index: int,
        segment_id: Optional[UUID] = None
    ) -> 'ProcessingChunk':
        """
        Create ProcessingChunk from continuum Message objects.

        Holds Message objects directly without conversion to preserve
        all attributes and methods during batch payload construction.

        Args:
            messages: List of Message objects from continuum
            chunk_index: Index of this chunk in sequence
            segment_id: Optional source segment UUID for context exploration

        Returns:
            ProcessingChunk instance

        Raises:
            ValueError: If messages list is empty
        """
        if not messages:
            raise ValueError("Cannot create chunk from empty message list")

        return cls(
            messages=messages,
            temporal_start=messages[0].created_at,
            temporal_end=messages[-1].created_at,
            chunk_index=chunk_index,
            memory_context_snapshot=None,
            segment_id=segment_id
        )


class ExtractionBatch(BaseModel):
    """
    Batch extraction tracking.

    Represents a row in extraction_batches table.
    """
    id: Optional[UUID] = None  # Generated by database
    batch_id: str  # Anthropic batch ID
    custom_id: str
    user_id: UUID
    chunk_index: int
    request_payload: Dict[str, Any]
    chunk_metadata: Optional[ChunkMetadata] = None
    memory_context: Optional[MemoryContext] = None
    status: BatchStatus
    created_at: datetime
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    result_url: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None
    extracted_memories: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None



class PendingManualMemory(BaseModel):
    """
    Queued manual memory awaiting processing at segment collapse.

    Created by memory_tool.create_memory() and stored in Valkey. Processing
    (embedding generation, entity extraction, linking) is deferred until
    segment collapse to avoid blocking tool execution with heavy operations.
    """
    text: str = Field(description="Memory content text")
    importance_score: float = Field(
        ge=0.0, le=1.0,
        description="Importance score (0.0-1.0)"
    )
    user_requested: bool = Field(
        default=False,
        description="True if user explicitly said 'remember this'"
    )
    happens_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp for when the event occurs"
    )
    expires_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp for when memory expires"
    )
    supersedes_memory_ids: List[str] = Field(
        default_factory=list,
        description="List of 8-char memory IDs this memory supersedes"
    )
    queued_at: str = Field(description="ISO timestamp when queued")
    pending_id: str = Field(description="UUID for tracking this pending memory")

    def to_json(self) -> str:
        """Serialize to JSON for Valkey storage."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> 'PendingManualMemory':
        """Deserialize from JSON retrieved from Valkey."""
        return cls.model_validate_json(json_str)
