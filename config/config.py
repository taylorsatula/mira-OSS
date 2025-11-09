"""
Base schema definitions for configuration models.

These are the core schema models used throughout the application to
ensure type safety and validation of configuration values.
"""

import os
from typing import Optional, Dict, Any, List, Tuple

from pydantic import BaseModel, Field
import multiprocessing

class ApiConfig(BaseModel):
    """Anthropic API configuration settings."""

    # Model configuration
    model: str = Field(default="claude-sonnet-4-5-20250929", description="Anthropic reasoning model for complex operations")
    execution_model: str = Field(default="openai/gpt-oss-20b", description="Faster model for simple tool operations (dynamic routing)")
    execution_endpoint: str = Field(default="https://api.groq.com/openai/v1/chat/completions", description="OpenAI-compatible endpoint for execution model")
    execution_api_key_name: str = Field(default="groq_key", description="Vault key name for execution model API key")
    simple_tools: List[str] = Field(default=["reminder_tool", "punchclock_tool", "weather_tool"], description="Tools that don't require reasoning model capabilities")

    # API key configuration
    api_key_name: str = Field(default="anthropic_key", description="Name of the Anthropic API key to retrieve from Vault")

    # Generation settings
    max_tokens: int = Field(default=10000, description="Maximum number of tokens to generate in responses")
    context_window_tokens: int = Field(default=200000, description="Total context window size in tokens for the model")
    temperature: float = Field(default=1.0, description="Temperature setting for response generation (Anthropic default: 1.0)")

    # Request settings
    max_retries: int = Field(default=3, description="Maximum number of retries for failed API requests")
    timeout: int = Field(default=60, description="Request timeout in seconds")

    # Prompt caching (Anthropic-specific)
    enable_prompt_caching: bool = Field(default=True, description="Enable prompt caching for system prompts and tools (reduces costs by ~90% on cached content)")

    # Emergency fallback settings
    emergency_fallback_enabled: bool = Field(default=True, description="Enable automatic failover to emergency provider on Anthropic errors")
    emergency_fallback_endpoint: str = Field(default="https://api.openai.com/v1/chat/completions", description="OpenAI-compatible endpoint for emergency fallback")
    emergency_fallback_api_key_name: str = Field(default="openai_key", description="Vault key name for emergency fallback API key")
    emergency_fallback_model: str = Field(default="gpt-5-mini", description="Model to use during emergency fallback")
    emergency_fallback_recovery_minutes: int = Field(default=5, description="Minutes to wait before testing Anthropic recovery")

    # Pre-processing analysis settings (subconscious reasoning layer)
    analysis_enabled: bool = Field(default=True, description="Enable pre-processing analysis call for touchstone generation")
    analysis_endpoint: str = Field(default="https://api.groq.com/openai/v1/chat/completions", description="OpenAI-compatible endpoint for analysis calls")
    analysis_api_key_name: str = Field(default="groq_key", description="Vault key name for analysis API key")
    analysis_model: str = Field(default="openai/gpt-oss-20b", description="Fast model for analysis generation")
    analysis_max_tokens: int = Field(default=500, description="Maximum tokens for analysis response")
    analysis_temperature: float = Field(default=1.0, description="Temperature for analysis generation")
    analysis_timeout: int = Field(default=10, description="Timeout for analysis requests in seconds")
    analysis_context_pairs: int = Field(default=5, description="Number of recent user/assistant turn pairs to include in analysis context")

class ApiServerConfig(BaseModel):
    """FastAPI server configuration settings."""
    
    host: str = Field(default="0.0.0.0", description="Host address for the FastAPI server")
    port: int = Field(default=1993, description="Port for the FastAPI server")
    workers: int = Field(default=1, description="Number of uvicorn workers")
    log_level: str = Field(default="warning", description="Log level for uvicorn server")
    enable_cors: bool = Field(default=True, description="Enable CORS middleware")
    cors_origins: List[str] = Field(
        default=["https://miraos.org", "http://localhost:1993", "http://127.0.0.1:1993"],
        description="Allowed CORS origins (production and local development)"
    )
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    rate_limit_rpm: int = Field(default=10, description="Maximum API requests per minute")
    burst_limit: int = Field(default=5, description="Maximum number of requests allowed in a burst")
    extended_thinking: bool = Field(default=True, description="Whether to enable extended thinking capability")
    extended_thinking_budget: int = Field(default=1024, description="Token budget for extended thinking when enabled (min: 1024)")


class PathConfig(BaseModel):
    """Path configuration settings."""
    
    data_dir: str = Field(default="data", description="Directory for data storage")
    persistent_dir: str = Field(default="persistent", description="Directory for persistent storage")
    # Continuum history is now stored per-user in data/users/{user_id}/conversations/
    # conversation_history_dir removed - use get_user_conversation_dir() helper instead
    prompts_dir: str = Field(default="config/prompts", description="Directory containing prompt templates")
    tools_dir: str = Field(default="tools/implementations", description="Directory containing tools")



class InvokeOtherToolConfig(BaseModel):
    """Configuration for invokeother_tool dynamic tool loader."""
    enabled: bool = Field(default=True, description="Whether invokeother_tool is enabled")
    idle_threshold: int = Field(default=5, description="Number of turns before an unused tool is automatically unloaded")


class ToolConfig(BaseModel):
    """Tool-related configuration settings."""

    enabled: bool = Field(default=True, description="Whether tools are enabled")
    auto_discovery: bool = Field(default=True, description="Whether automatic tool discovery is enabled")
    timeout: int = Field(default=30, description="Default timeout in seconds for tool operations")
    essential_tools: List[str] = Field(default=["webaccess_tool", "reminder_tool", "invokeother_tool"], description="List of essential tools to always load")
    invokeother_tool: InvokeOtherToolConfig = Field(default_factory=InvokeOtherToolConfig, description="Configuration for the invokeother_tool dynamic loader")
    # Synthetic data generator settings
    synthetic_data_analysis_model: str = Field(default="claude-3-7-sonnet-20250219", description="LLM model to use for code analysis and example review in synthetic data analysis")
    synthetic_data_generation_model: str = Field(default="claude-3-5-haiku-20241022", description="LLM model to use for example generation in synthetic data generation")
    # Synthetic data now uses unified BGE-M3 for deduplication

class EmbeddingsFastModelConfig(BaseModel):
    """Fast model configuration for real-time operations."""
    
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="AllMiniLM model for 384-dim real-time embeddings")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory for model files")
    thread_limit: int = Field(default=2, description="Thread limit for fast inference")
    batch_size: int = Field(default=32, description="Batch size for fast model")

class EmbeddingsDeepModelConfig(BaseModel):
    """Deep model configuration for advanced features."""
    
    model_name: str = Field(default="BAAI/bge-m3", description="Deep embeddings model (not used when OpenAI backend is enabled)")
    cache_dir: Optional[str] = Field(default=None, description="Cache directory for model files")
    thread_limit: int = Field(default=4, description="Thread limit for deep inference")
    batch_size: int = Field(default=16, description="Batch size for deep model")


class EmbeddingsRemoteConfig(BaseModel):
    """Remote embeddings provider settings."""
    
    model: str = Field(default="text-embedding-3-small", description="Remote embedding model name")


class EmbeddingsConfig(BaseModel):
    """Embeddings provider configuration settings."""
    
    provider: str = Field(default="hybrid", description="Embeddings provider: 'hybrid' for dual-model system")
    
    # Model configurations
    fast_model: EmbeddingsFastModelConfig = Field(default_factory=EmbeddingsFastModelConfig, description="Fast model configuration (AllMiniLM)")
    deep_model: EmbeddingsDeepModelConfig = Field(default_factory=EmbeddingsDeepModelConfig, description="Deep model configuration (BGE-M3)")
    remote: EmbeddingsRemoteConfig = Field(default_factory=EmbeddingsRemoteConfig, description="Remote provider configuration (legacy)")
    
    # Common settings
    cache_enabled: bool = Field(default=True, description="Enable embedding caching")
    reranker_pool_size: int = Field(
        default=1,
        description=(
            "BGE reranker pool size for concurrent requests. "
            "pool_size=1: Thread-safe single instance (low overhead, no parallelism). "
            "pool_size>1: Process pool with true parallelism (higher memory usage). "
            "Set to 1 for resource-constrained systems, 4-8 for production."
        )
    )

class DomainKnowledgeConfig(BaseModel):
    """Domain knowledge service configuration."""

    message_batch_size: int = Field(default=10, description="Number of messages to batch before sending to Letta")
    block_cache_ttl: int = Field(default=300, description="Cache TTL for domain block content in seconds (5 minutes)")
    sleeptime_agent_model: str = Field(default="openai/gpt-4o-mini", description="LLM model for Letta sleeptime agents (fast, cheap model for block updates)")


class SystemConfig(BaseModel):
    """System-level configuration settings."""

    log_level: str = Field(default="WARNING", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    timezone: str = Field(default="America/Chicago", description="Default timezone for date/time operations (must be a valid IANA timezone name like 'America/New_York', 'Europe/London')")
    streaming: bool = Field(default=True, description="Whether to stream responses from the API")
    json_indent: int = Field(default=2, description="Indentation level for JSON output")
    max_tool_iterations: int = Field(default=10, description="Maximum number of tool iterations for a single request")
    continuum_pool_size: int = Field(default=100, description="Maximum number of conversations to keep in memory pool for performance")

    # Segment Timeout Threshold (minutes)
    # NOTE: Can be made context-aware by time of day if needed in the future
    # (e.g., shorter timeout during morning hours, longer during late night)
    segment_timeout: int = Field(default=60, description="Segment collapse timeout in minutes (60 minutes)")

    # Manifest Display Settings
    manifest_depth: int = Field(default=30, description="Number of recent segments to include in manifest display")
    manifest_cache_ttl: int = Field(default=3600, description="TTL for manifest cache in seconds (1 hour default)")
    manifest_summary_truncate_length: int = Field(default=60, description="Maximum characters for segment summary in manifest display")

    # Session Cache Settings (Complexity-Based Loading)
    session_summary_complexity_limit: int = Field(
        default=8,
        description="Maximum total complexity score for loaded segment summaries (accumulates until limit reached)"
    )
    session_summary_max_count: int = Field(
        default=5,
        description="Maximum number of segment summaries to load regardless of complexity"
    )
    session_summary_query_window: int = Field(
        default=9,
        description="Number of recent segments to query for complexity-based selection"
    )


# ============================================================================
# LT_Memory Configuration
# ============================================================================

class ExtractionConfig(BaseModel):
    """
    Extraction service configuration.

    Controls memory extraction behavior and LLM parameters.
    """
    extraction_model: str = Field(
        default="claude-haiku-4-5",
        description="Model for memory extraction (used by both sync and batch)"
    )
    extraction_thinking_enabled: bool = Field(
        default=True,
        description="Enable extended thinking for memory extraction"
    )
    extraction_thinking_budget: int = Field(
        default=1024,
        description="Token budget for extended thinking during extraction"
    )
    max_extraction_tokens: int = Field(
        default=16000,
        description="Maximum tokens for extraction response"
    )
    extraction_temperature: float = Field(
        default=1.0,
        description="Temperature for extraction LLM calls"
    )
    dedup_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for duplicate detection"
    )
    default_importance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default importance score for newly extracted memories"
    )
    retry_attempts: int = Field(
        default=2,
        description="Number of retry attempts for failed extractions"
    )


class BatchingConfig(BaseModel):
    """
    Batching orchestration configuration.

    Controls batch submission and processing behavior.
    """
    batch_expiry_hours: int = Field(
        default=24,
        description="Hours before Anthropic batch expires"
    )
    max_chunk_size: int = Field(
        default=100,
        description="Maximum messages per processing chunk for boot extraction"
    )
    segment_chunk_size: int = Field(
        default=40,
        description="Maximum messages per chunk for segment-based extraction (smaller chunks for better quality)"
    )
    boot_check_enabled: bool = Field(
        default=True,
        description="Whether to run extraction sweep on application boot"
    )
    min_messages_for_boot_extraction: int = Field(
        default=20,
        description="Minimum messages required before running boot extraction (segment extraction has no minimum)"
    )
    relationship_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Model for relationship classification via Batch API"
    )
    relationship_max_tokens: int = Field(
        default=500,
        description="Maximum tokens for relationship classification"
    )
    relationship_temperature: float = Field(
        default=0.2,
        description="Temperature for relationship classification"
    )


class LinkingConfig(BaseModel):
    """
    Linking service configuration.

    Controls relationship classification and link management.
    """
    similarity_threshold_for_linking: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity to consider for relationship classification"
    )
    max_candidates_per_memory: int = Field(
        default=20,
        description="Maximum candidate memories to evaluate for links"
    )
    link_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to create a link"
    )
    max_link_traversal_depth: int = Field(
        default=3,
        description="Maximum depth for link traversal when navigating memory graph"
    )
    classification_max_tokens: int = Field(
        default=500,
        description="Maximum tokens for relationship classification LLM calls"
    )
    entity_hub_importance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for memory to participate in entity linking (hub-only topology)"
    )


class RefinementConfig(BaseModel):
    """
    Refinement service configuration.

    Controls memory consolidation and verbose trimming behavior.
    """
    verbose_threshold_chars: int = Field(
        default=70,
        description="Character count above which memory is considered verbose"
    )
    consolidation_similarity_threshold: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for considering consolidation"
    )
    min_cluster_size: int = Field(
        default=2,
        description="Minimum memories in cluster for consolidation"
    )
    max_cluster_size: int = Field(
        default=5,
        description="Maximum memories in cluster for consolidation"
    )
    consolidation_confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to perform consolidation"
    )
    refinement_cooldown_days: int = Field(
        default=30,
        description="Days to wait before re-refining a memory"
    )
    refinement_max_tokens: int = Field(
        default=1000,
        description="Maximum tokens for refinement LLM calls"
    )
    verbose_candidates_limit: int = Field(
        default=20,
        description="Maximum verbose memories to identify per run"
    )
    min_age_for_refinement_days: int = Field(
        default=7,
        description="Minimum age in days before memory is eligible for refinement"
    )
    min_access_count_for_refinement: int = Field(
        default=3,
        description="Minimum access count before verbose memory is considered stable enough to refine"
    )
    max_rejection_count: int = Field(
        default=3,
        description="Number of do_nothing rejections before memory is excluded from future refinement"
    )


class ProactiveConfig(BaseModel):
    """
    Proactive surfacing configuration.

    Controls memory surfacing behavior for CNS integration.
    """
    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for surfacing memories"
    )
    max_link_traversal_depth: int = Field(
        default=3,
        description="Maximum depth for link traversal when expanding context"
    )
    max_memories: int = Field(
        default=10,
        description="Maximum memories to return per search"
    )
    min_importance_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum importance score for surfacing"
    )


class EntityGarbageCollectionConfig(BaseModel):
    """
    Entity garbage collection configuration.

    Controls entity dormancy detection, merge candidate scoring, and LLM review behavior.
    """
    dormancy_days: int = Field(
        default=45,
        description="Days without new links before entity is considered dormant"
    )
    max_link_count_for_gc: int = Field(
        default=5,
        description="Only review entities with this many or fewer links"
    )
    min_link_count_for_gc: int = Field(
        default=1,
        description="Entities with fewer links are immediately deleted without LLM review"
    )
    vector_similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Vector similarity threshold for merge candidates (semantic equivalence)"
    )
    string_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="String similarity threshold for merge candidates (spelling variations)"
    )
    co_occurrence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Co-occurrence score threshold (ratio of shared memories)"
    )
    max_merge_candidates: int = Field(
        default=5,
        description="Maximum merge candidates to present per entity"
    )
    gc_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Model for entity GC review"
    )
    gc_max_tokens: int = Field(
        default=1000,
        description="Maximum tokens for GC review response"
    )
    gc_temperature: float = Field(
        default=0.2,
        description="Temperature for GC review LLM calls"
    )


class LTMemoryConfig(BaseModel):
    """
    Complete LT_Memory system configuration.

    Aggregates all module-specific configs into single source of truth.
    Note: Scoring constants are hardcoded in lt_memory/scoring_formula.sql
    """
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    linking: LinkingConfig = Field(default_factory=LinkingConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    entity_gc: EntityGarbageCollectionConfig = Field(default_factory=EntityGarbageCollectionConfig)

    # Global settings
    temporal_rag_enabled: bool = Field(
        default=True,
        description="Whether temporal RAG features are enabled"
    )
    ivfflat_lists: int = Field(
        default=0,
        description="Number of lists for IVFFlat vector index (0 = no index, better for small datasets)"
    )

