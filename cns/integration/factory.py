"""
CNS Integration Factory

Provides service initialization and dependency injection for connecting CNS
with existing MIRA components (tool repository, working memory, workflow manager, etc.)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config.config_manager import config
from clients.llm_provider import LLMProvider
from working_memory.core import WorkingMemory
from tools.repo import ToolRepository
from utils.tag_parser import TagParser

from ..services.orchestrator import ContinuumOrchestrator
from .event_bus import EventBus
from ..services.summary_generator import SummaryGenerator
from ..core.segment_cache_loader import SegmentCacheLoader
from ..services.segment_collapse_handler import SegmentCollapseHandler
from ..infrastructure.continuum_pool import get_continuum_pool, initialize_continuum_pool
from ..infrastructure.valkey_message_cache import ValkeyMessageCache

if TYPE_CHECKING:
    from clients.hybrid_embeddings_provider import HybridEmbeddingsProvider
    from cns.services.memory_relevance_service import MemoryRelevanceService
    from cns.services.subcortical import SubcorticalLayer
    from cns.services.peanutgallery_service import PeanutGalleryService
    from cns.services.live_context_compaction_service import LiveContextCompactionService

logger = logging.getLogger(__name__)


class CNSIntegrationFactory:
    """
    Factory for initializing CNS with proper integration to existing MIRA components.
    
    Follows the same initialization patterns as system_initializer.py while providing
    clean dependency injection for the CNS architecture.
    """
    
    def __init__(self, config_instance: object = None) -> None:
        """
        Initialize the factory with configuration.

        Args:
            config_instance: MIRA configuration instance. If None, will use global config.
        """
        self.config = config_instance or config
        self._embedding_model: HybridEmbeddingsProvider | None = None
        self._llm_provider: LLMProvider | None = None
        self._working_memory: WorkingMemory | None = None
        self._tool_repo: ToolRepository | None = None
        self._tag_parser: TagParser | None = None
        self._event_bus: EventBus | None = None
        self._summary_generator: SummaryGenerator | None = None
        self._session_cache_loader: SegmentCacheLoader | None = None
        self._subcortical_layer: SubcorticalLayer | None = None
        self._peanutgallery_service: PeanutGalleryService | None = None
        self._inbox_poller: object | None = None
        self._valkey_cache: ValkeyMessageCache | None = None
        self._memory_relevance_service: MemoryRelevanceService | None = None
        self._live_context_compaction_service: LiveContextCompactionService | None = None
        
    def create_orchestrator(self) -> ContinuumOrchestrator:
        """
        Create a fully configured ContinuumOrchestrator with all integrations.
        
        Returns:
            ContinuumOrchestrator with all dependencies properly injected
        """
        logger.info("Initializing CNS with full MIRA component integration")
        
        # Initialize core services in dependency order
        embedding_model = self._get_embedding_model()
        
        # Create event bus early as it's needed by working memory
        event_bus = self._get_event_bus()
        
        # Create working memory with event bus
        working_memory = self._get_working_memory(event_bus)

        tool_repo = self._get_tool_repository(working_memory)

        # Wire ephemeral tool cleanup on turn end
        from tools.repo import ESSENTIAL_TOOLS
        essential_set = set(ESSENTIAL_TOOLS)
        event_bus.subscribe('TurnCompletedEvent',
            lambda e: tool_repo.cleanup_ephemeral_tools(essential_set))

        llm_provider = self._get_llm_provider()
        tag_parser = self._get_tag_parser()

        # Create CNS services - first create repo without cache manager
        from ..infrastructure.continuum_repository import get_continuum_repository
        continuum_repo = get_continuum_repository()  # Use singleton

        live_context_compaction_service = self._get_live_context_compaction_service(
            continuum_repo,
            llm_provider,
        )

        # Create memory relevance service for surfacing memories
        memory_relevance_service = self._get_memory_relevance_service()

        # Create subcortical layer for retrieval query expansion
        subcortical_layer = self._get_subcortical_layer(llm_provider)

        # Initialize session cache loader
        self._initialize_session_cache(continuum_repo, event_bus)

        # Initialize domain knowledge service with event bus and continuum pool
        # Must be done after session cache initialization so continuum_pool exists
        self._initialize_domain_knowledge_service(event_bus)

        # Initialize segment collapse handler with event bus
        self._initialize_segment_collapse_handler(event_bus)

        # Initialize UserDataManager cleanup handler for session collapse cleanup
        self._initialize_userdata_cleanup_handler(event_bus)

        # Initialize manifest query service with event bus
        self._initialize_manifest_query_service(event_bus)

        # Initialize Peanut Gallery metacognitive observer service with event bus
        self._initialize_peanutgallery_service(event_bus, llm_provider)

        # Initialize inbox poller for email awareness during active segments
        self._initialize_inbox_poller(event_bus)

        # Create orchestrator with all dependencies
        orchestrator = ContinuumOrchestrator(
            llm_provider=llm_provider,
            continuum_repo=continuum_repo,
            working_memory=working_memory,
            tool_repo=tool_repo,
            tag_parser=tag_parser,
            subcortical_layer=subcortical_layer,
            memory_relevance_service=memory_relevance_service,
            event_bus=event_bus,
            live_context_compaction_service=live_context_compaction_service,
        )

        # Initialize async work barrier (shared gate for cross-turn async tasks)
        from cns.services.async_work_barrier import initialize_async_work_barrier
        initialize_async_work_barrier()

        # Initialize retrieval-backed tool result compaction for the hot cache.
        from cns.services.tool_result_summarizer import initialize_tool_result_summarizer
        initialize_tool_result_summarizer(
            llm_provider,
            event_bus,
            get_continuum_pool().valkey_cache,
        )

        logger.info("CNS orchestrator initialized successfully with full integration")
        return orchestrator
        
    def _get_embedding_model(self) -> HybridEmbeddingsProvider:
        """Get or create hybrid embedding provider instance."""
        if self._embedding_model is None:
            logger.info("Initializing hybrid embedding provider")
            from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
            self._embedding_model = get_hybrid_embeddings_provider()  # Use singleton
            logger.info("Hybrid embedding provider initialized")
        return self._embedding_model
        
    def _get_llm_provider(self) -> LLMProvider:
        """Get or create LLM provider instance."""
        if self._llm_provider is None:
            logger.info("Initializing LLM provider")
            self._llm_provider = LLMProvider()
            logger.info("LLM provider initialized")
        return self._llm_provider
        
    def _get_working_memory(self, event_bus: 'EventBus') -> WorkingMemory:
        """Get or create event-driven working memory instance."""
        if self._working_memory is None:
            logger.info("Initializing event-driven working memory")
            
            from working_memory import WorkingMemory
            self._working_memory = WorkingMemory(event_bus)
            
            # Create and register trinkets with event bus
            from working_memory.trinkets.time_manager import TimeManager
            from working_memory.trinkets.reminder_manager import ReminderManager
            from working_memory.trinkets.manifest_trinket import ManifestTrinket
            from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket
            from working_memory.trinkets.domaindoc_trinket import DomaindocTrinket
            from working_memory.trinkets.forage_trinket import ForageTrinket
            from working_memory.trinkets.whilethecatsaway_trinket import WhileTheCatsAwayTrinket
            from working_memory.trinkets.lora_trinket import LoraTrinket
            from working_memory.trinkets.location_trinket import LocationTrinket
            from working_memory.trinkets.asyncactivity_trinket import AsyncActivityTrinket
            from working_memory.trinkets.live_context_compaction_trinket import LiveContextCompactionTrinket
            from working_memory.trinkets.memory_curator_trinket import MemoryCuratorTrinket

            # @CODEX: I feel like we could do away with these manual registrations if they're all going to do identical asks for ```event_bus, self._working_memory```. Thoughts?
            # Trinkets self-register with working memory
            TimeManager(event_bus, self._working_memory)
            ReminderManager(event_bus, self._working_memory)
            LiveContextCompactionTrinket(
                event_bus,
                self._working_memory,
            )
            ManifestTrinket(event_bus, self._working_memory)
            ProactiveMemoryTrinket(event_bus, self._working_memory)
            DomaindocTrinket(event_bus, self._working_memory)
            ForageTrinket(event_bus, self._working_memory)
            WhileTheCatsAwayTrinket(event_bus, self._working_memory)
            LoraTrinket(event_bus, self._working_memory)
            LocationTrinket(event_bus, self._working_memory)
            AsyncActivityTrinket(event_bus, self._working_memory)
            MemoryCuratorTrinket(event_bus, self._working_memory)

            logger.info("Event-driven working memory initialized with trinkets")
        return self._working_memory

    def _get_tool_repository(self, working_memory: WorkingMemory) -> ToolRepository:
        """Get or create tool repository instance."""
        if self._tool_repo is None:
            logger.info("Initializing tool repository")
            self._tool_repo = ToolRepository(working_memory=working_memory)

            # Follow original initialization pattern
            self._tool_repo.discover_tools()
            self._tool_repo.enable_tools_from_config()

            # Register gated tools - tools that self-determine availability
            self._register_gated_tools()

            logger.info(f"Tool repository initialized with {len(self._tool_repo.get_enabled_tools())} enabled tools")
        return self._tool_repo

    def _register_gated_tools(self) -> None:
        """Register tools that self-determine their availability via is_available().

        Previously registered domaindoc_tool here — moved to essential_tools.
        Preserved for future gated tools.
        """
        pass

    def _get_tag_parser(self) -> TagParser:
        """Get or create tag parser instance."""
        if self._tag_parser is None:
            logger.info("Initializing tag parser")
            self._tag_parser = TagParser()
            logger.info("Tag parser initialized")
        return self._tag_parser
        
    def _get_event_bus(self) -> EventBus:
        """Get or create event bus instance."""
        if self._event_bus is None:
            logger.info("Initializing event bus")
            self._event_bus = EventBus()
            logger.info("Event bus initialized")
        return self._event_bus

    def _get_summary_generator(self) -> SummaryGenerator:
        """Get or create summary generator instance."""
        if self._summary_generator is None:
            logger.info("Initializing summary generator")
            from ..infrastructure.continuum_repository import get_continuum_repository
            repository = get_continuum_repository()
            self._summary_generator = SummaryGenerator(
                repository=repository,
                llm_provider=None  # Will create its own with summary-specific settings
            )
            logger.info("Summary generator initialized")
        return self._summary_generator
    
    def _initialize_session_cache(self, continuum_repo: object, event_bus: EventBus) -> None:
        """Initialize session cache loader."""
        logger.info("Initializing session cache loader")

        # Create summary generator
        summary_generator = self._get_summary_generator()

        # Create session cache loader
        self._session_cache_loader = SegmentCacheLoader(repository=continuum_repo)
        logger.info("Session cache loader initialized")

        # Initialize continuum pool with session loader
        initialize_continuum_pool(continuum_repo, self._session_cache_loader)

    def _get_memory_relevance_service(self) -> MemoryRelevanceService:
        """Get or create memory relevance service for surfacing memories."""
        if self._memory_relevance_service is None:
            logger.info("Initializing Memory Relevance Service")

            # Import new CNS service and lt_memory factory
            from cns.services.memory_relevance_service import MemoryRelevanceService
            from lt_memory.factory import get_lt_memory_factory

            # Get lt_memory factory singleton with all services initialized
            lt_memory_factory = get_lt_memory_factory()

            # Create CNS memory relevance service wrapping ProactiveService
            self._memory_relevance_service = MemoryRelevanceService(
                proactive_service=lt_memory_factory.proactive
            )

            logger.info("Memory Relevance Service initialized (wraps lt_memory.proactive)")

        return self._memory_relevance_service
        
    
    def _get_subcortical_layer(self, llm_provider: LLMProvider) -> SubcorticalLayer:
        """Get or create subcortical layer instance."""
        if self._subcortical_layer is None:
            logger.info("Initializing SubcorticalLayer")
            from ..services.subcortical import SubcorticalLayer
            self._subcortical_layer = SubcorticalLayer(
                analysis_enabled=self.config.api.analysis_enabled,
                llm_provider=llm_provider,
                prefill_warmup_enabled=self.config.api.subcortical_prefill_warmup,
            )
            logger.info("SubcorticalLayer initialized")
        return self._subcortical_layer

    def _get_live_context_compaction_service(
        self,
        continuum_repo: object,
        llm_provider: LLMProvider,
    ) -> LiveContextCompactionService:
        """Get or create the background live context compaction service."""
        if self._live_context_compaction_service is None:
            from cns.services.live_context_compaction_service import (
                LiveContextCompactionService,
                LiveContextCompactionStore,
            )
            self._live_context_compaction_service = LiveContextCompactionService(
                continuum_repo=continuum_repo,
                store=LiveContextCompactionStore(),
                llm_provider=llm_provider,
            )
        return self._live_context_compaction_service

    def _initialize_domain_knowledge_service(self, event_bus: EventBus) -> None:
        """
        Initialize domain knowledge (domaindoc) system.

        The domaindoc system uses:
        - SQLite storage via UserDataManager (domaindocs + domaindoc_sections tables)
        - Section-aware editing with expand/collapse and one-level subsection nesting
        - Gated tool pattern (domaindoc_tool appears when domains are enabled)
        - DomaindocTrinket for content injection
        - API endpoint for lifecycle management (create/enable/disable/delete)

        No service initialization needed - state is in per-user SQLite.
        """
        logger.info("Domaindoc system uses SQLite storage (no service initialization needed)")

    def _initialize_segment_collapse_handler(self, event_bus: EventBus) -> None:
        """
        Initialize segment collapse handler with event bus.

        The handler subscribes to SegmentTimeoutEvent and orchestrates the
        collapse pipeline: summary generation, embedding, sentinel update,
        and downstream processing triggers.
        """
        logger.info("Initializing segment collapse handler with event subscriptions")

        from ..infrastructure.continuum_repository import get_continuum_repository
        from ..infrastructure.continuum_pool import get_continuum_pool
        from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
        from lt_memory.factory import get_lt_memory_factory

        continuum_repo = get_continuum_repository()
        continuum_pool = get_continuum_pool()
        summary_generator = self._get_summary_generator()
        embeddings_provider = get_hybrid_embeddings_provider()

        # Get lt_memory factory for downstream processing (required)
        lt_memory_factory = get_lt_memory_factory()

        # Create and register collapse handler
        collapse_handler = SegmentCollapseHandler(
            continuum_repo=continuum_repo,
            summary_generator=summary_generator,
            embeddings_provider=embeddings_provider,
            event_bus=event_bus,
            continuum_pool=continuum_pool,
            lt_memory_factory=lt_memory_factory,
            tool_repo=self._tool_repo,
        )

        # Store singleton for API access
        from ..services.segment_collapse_handler import initialize_segment_collapse_handler
        initialize_segment_collapse_handler(collapse_handler)

        logger.info("Segment collapse handler initialized and subscribed to SegmentTimeoutEvent")

    def _initialize_userdata_cleanup_handler(self, event_bus: EventBus) -> None:
        """
        Initialize UserDataManager cleanup handler with event bus.

        The handler subscribes to SegmentCollapsedEvent and closes the user's
        SQLite connection when their session collapses, freeing resources.
        """
        from utils.userdata_manager import UserDataManagerCleanupHandler

        self._userdata_cleanup_handler = UserDataManagerCleanupHandler(event_bus)
        logger.info("UserDataManager cleanup handler initialized")

    def _initialize_manifest_query_service(self, event_bus: EventBus) -> None:
        """
        Initialize manifest query service with event bus.

        The service subscribes to ManifestUpdatedEvent for cache invalidation.
        """
        logger.info("Initializing manifest query service with event subscriptions")

        from cns.services.manifest_query_service import initialize_manifest_query_service

        # Initialize service with event bus (singleton pattern)
        manifest_service = initialize_manifest_query_service(event_bus=event_bus)

        logger.info("Manifest query service initialized and subscribed to ManifestUpdatedEvent")

    def _initialize_peanutgallery_service(self, event_bus: EventBus, llm_provider: LLMProvider) -> None:
        """
        Initialize Peanut Gallery metacognitive observer service with event bus.

        The service subscribes to TurnCompletedEvent and periodically evaluates
        the conversation for evidence-backed concerns and coaching directives.
        """
        if self._peanutgallery_service is not None:
            return

        if not self.config.system.peanutgallery_enabled:
            logger.info("Peanut Gallery observer disabled in config")
            return

        logger.info("Initializing Peanut Gallery observer service")

        from cns.services.peanutgallery_model import PeanutGalleryModel
        from cns.services.peanutgallery_service import PeanutGalleryService
        from working_memory.trinkets.peanutgallery_trinket import PeanutGalleryTrinket

        from cns.services.peanutgallery_service import PG_TRIGGER_INTERVAL, PG_GUIDANCE_TTL_TURNS

        model = PeanutGalleryModel(
            llm_provider=llm_provider,
            analysis_interval_turns=PG_TRIGGER_INTERVAL,
        )

        PeanutGalleryTrinket(
            event_bus,
            self._working_memory,
            default_ttl=PG_GUIDANCE_TTL_TURNS
        )

        self._peanutgallery_service = PeanutGalleryService(
            model=model,
            event_bus=event_bus,
        )

        logger.info(
            f"Peanut Gallery service initialized "
            f"(trigger_interval={PG_TRIGGER_INTERVAL})"
        )

    def _initialize_inbox_poller(self, event_bus: EventBus) -> None:
        """Initialize inbox poller for email awareness during active segments.

        The service polls IMAP for unread emails and publishes to EmailTrinket.
        Gracefully no-ops if the user has no email credentials configured.
        """
        from cns.services.pollers.inbox_poller import InboxPollerService
        from working_memory.trinkets.email_trinket import EmailTrinket

        # Trinket self-registers with working memory
        EmailTrinket(event_bus, self._working_memory)

        # Service subscribes to lifecycle events in constructor
        self._inbox_poller = InboxPollerService(event_bus)

        logger.info("Inbox poller service initialized")


def create_cns_orchestrator(config_instance: object = None) -> ContinuumOrchestrator:
    """
    Convenience function to create a fully configured CNS orchestrator.
    
    Args:
        config_instance: MIRA configuration instance. If None, will use global config.
        
    Returns:
        ContinuumOrchestrator with all integrations
    """
    factory = CNSIntegrationFactory(config_instance)
    return factory.create_orchestrator()
