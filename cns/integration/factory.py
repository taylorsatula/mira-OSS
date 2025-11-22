"""
CNS Integration Factory

Provides service initialization and dependency injection for connecting CNS
with existing MIRA components (tool repository, working memory, workflow manager, etc.)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from config.config_manager import config
from clients.llm_provider import LLMProvider
from working_memory.core import WorkingMemory
from tools.repo import ToolRepository
from utils.tag_parser import TagParser

from ..services.orchestrator import ContinuumOrchestrator
from ..infrastructure.continuum_repository import ContinuumRepository
from .event_bus import EventBus
from ..services.summary_generator import SummaryGenerator
from ..core.segment_cache_loader import SegmentCacheLoader
from ..services.segment_collapse_handler import SegmentCollapseHandler
from ..infrastructure.continuum_pool import get_continuum_pool, initialize_continuum_pool

logger = logging.getLogger(__name__)


class CNSIntegrationFactory:
    """
    Factory for initializing CNS with proper integration to existing MIRA components.
    
    Follows the same initialization patterns as system_initializer.py while providing
    clean dependency injection for the CNS architecture.
    """
    
    def __init__(self, config_instance = None):
        """
        Initialize the factory with configuration.
        
        Args:
            config_instance: MIRA configuration instance. If None, will use global config.
        """
        self.config = config_instance or config
        self._embedding_model = None
        self._llm_provider = None
        self._working_memory = None
        self._tool_repo = None
        self._tag_parser = None
        self._event_bus = None
        self._summary_generator = None
        self._session_cache_loader = None
        self._pointer_summary_coordinator = None
        self._analysis_generator = None
        
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

        # Create ToolLoaderTrinket after tool_repo is available
        # This must be done after tool_repo creation but before LLM provider
        self._create_tool_loader_trinket(event_bus, working_memory, tool_repo)

        llm_provider = self._get_llm_provider(tool_repo)
        tag_parser = self._get_tag_parser()

        # Create CNS services - first create repo without cache manager
        from ..infrastructure.continuum_repository import get_continuum_repository
        continuum_repo = get_continuum_repository()  # Use singleton
        

        # Create memory relevance service for surfacing memories
        memory_relevance_service = self._get_memory_relevance_service()

        # Create analysis generator for pre-processing touchstone generation
        analysis_generator = self._get_analysis_generator(tag_parser, llm_provider)

        # Initialize session cache loader
        self._initialize_session_cache(continuum_repo, event_bus)

        # Initialize domain knowledge service with event bus and continuum pool
        # Must be done after session cache initialization so continuum_pool exists
        self._initialize_domain_knowledge_service(event_bus)

        # Initialize segment collapse handler with event bus
        self._initialize_segment_collapse_handler(event_bus)

        # Initialize manifest query service with event bus
        self._initialize_manifest_query_service(event_bus)

        # Create orchestrator with all dependencies
        orchestrator = ContinuumOrchestrator(
            llm_provider=llm_provider,
            continuum_repo=continuum_repo,
            working_memory=working_memory,
            tool_repo=tool_repo,
            tag_parser=tag_parser,
            analysis_generator=analysis_generator,
            memory_relevance_service=memory_relevance_service,
            event_bus=event_bus
        )

        logger.info("CNS orchestrator initialized successfully with full integration")
        return orchestrator
        
    def _get_embedding_model(self):
        """Get or create hybrid embedding provider instance."""
        if self._embedding_model is None:
            logger.info("Initializing hybrid embedding provider")
            from clients.hybrid_embeddings_provider import get_hybrid_embeddings_provider
            self._embedding_model = get_hybrid_embeddings_provider()  # Use singleton
            logger.info("Hybrid embedding provider initialized")
        return self._embedding_model
        
    def _get_llm_provider(self, tool_repo=None) -> LLMProvider:
        """Get or create LLM provider instance."""
        if self._llm_provider is None:
            logger.info("Initializing LLM provider")
            self._llm_provider = LLMProvider(tool_repo=tool_repo)
            if tool_repo:
                logger.info("LLM provider initialized with tool execution capability")
            else:
                logger.info("LLM provider initialized without tool execution")
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
            from working_memory.trinkets.user_info_trinket import UserInfoTrinket
            from working_memory.trinkets.manifest_trinket import ManifestTrinket
            from working_memory.trinkets.proactive_memory_trinket import ProactiveMemoryTrinket
            from working_memory.trinkets.tool_guidance_trinket import ToolGuidanceTrinket
            from working_memory.trinkets.punchclock_trinket import PunchclockTrinket
            from working_memory.trinkets.domain_knowledge_trinket import DomainKnowledgeTrinket
            from working_memory.trinkets.getcontext_trinket import GetContextTrinket

            # Trinkets self-register with working memory
            TimeManager(event_bus, self._working_memory)
            ReminderManager(event_bus, self._working_memory)
            UserInfoTrinket(event_bus, self._working_memory)
            ManifestTrinket(event_bus, self._working_memory)
            ProactiveMemoryTrinket(event_bus, self._working_memory)
            ToolGuidanceTrinket(event_bus, self._working_memory)
            PunchclockTrinket(event_bus, self._working_memory)
            DomainKnowledgeTrinket(event_bus, self._working_memory)
            GetContextTrinket(event_bus, self._working_memory)
            
            logger.info("Event-driven working memory initialized with trinkets")
        return self._working_memory

    def _create_tool_loader_trinket(self, event_bus: EventBus, working_memory: WorkingMemory, tool_repo: ToolRepository) -> None:
        """Create ToolLoaderTrinket after tool repository is available."""
        from working_memory.trinkets.tool_loader_trinket import ToolLoaderTrinket

        # Create trinket with tool_repo for cleanup operations
        ToolLoaderTrinket(event_bus, working_memory, tool_repo)
        logger.info("ToolLoaderTrinket initialized for dynamic tool loading")

    def _get_tool_repository(self, working_memory: WorkingMemory) -> ToolRepository:
        """Get or create tool repository instance."""
        if self._tool_repo is None:
            logger.info("Initializing tool repository")
            self._tool_repo = ToolRepository(working_memory=working_memory)

            # Follow original initialization pattern
            self._tool_repo.discover_tools()
            self._tool_repo.enable_tools_from_config()

            logger.info(f"Tool repository initialized with {len(self._tool_repo.get_enabled_tools())} enabled tools")
        return self._tool_repo

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
    
    def _initialize_session_cache(self, continuum_repo, event_bus: EventBus):
        """Initialize session cache loader."""
        logger.info("Initializing session cache loader")

        # Create summary generator
        summary_generator = self._get_summary_generator()

        # Create session cache loader
        self._session_cache_loader = SegmentCacheLoader(repository=continuum_repo)
        logger.info("Session cache loader initialized")

        # Initialize continuum pool with session loader
        initialize_continuum_pool(continuum_repo, self._session_cache_loader)

        # Get continuum pool
        continuum_pool = get_continuum_pool()

        # NOTE: Memory extraction now uses segment-based APScheduler jobs
        # Segments collapse on timeout and trigger batch memory extraction automatically
        # if self._pointer_summary_coordinator is None:
        #     logger.info("Initializing pointer summary extraction coordinator")
        #     from lt_memory.pointer_summary_extraction import PointerSummaryExtractionCoordinator
        #     from lt_memory.memory_extraction_service import memory_extraction_service
        #
        #     self._pointer_summary_coordinator = PointerSummaryExtractionCoordinator(
        #         event_bus=event_bus,
        #         continuum_repository=continuum_repo,
        #         memory_service=memory_extraction_service,
        #     )
        #     logger.info("Pointer summary extraction coordinator initialized")

    def _get_memory_relevance_service(self):
        """Get or create memory relevance service for surfacing memories."""
        if not hasattr(self, '_memory_relevance_service') or self._memory_relevance_service is None:
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
        
    
    def _get_analysis_generator(self, tag_parser: TagParser, llm_provider: LLMProvider):
        """Get or create analysis generator instance."""
        if self._analysis_generator is None:
            logger.info("Initializing AnalysisGenerator")
            from ..services.analysis_generator import AnalysisGenerator
            self._analysis_generator = AnalysisGenerator(
                config=self.config.api,
                tag_parser=tag_parser,
                llm_provider=llm_provider
            )
            logger.info("AnalysisGenerator initialized")
        return self._analysis_generator

    def _initialize_domain_knowledge_service(self, event_bus: EventBus):
        """
        Initialize domain knowledge service with event bus.

        The service subscribes to TurnCompletedEvent and automatically buffers
        continuum messages to Letta domain knowledge blocks. The continuum
        object is passed directly in the event, eliminating the need for the
        service to fetch from the continuum pool.
        """
        logger.info("Initializing domain knowledge service with event subscriptions")

        # Initialize domain knowledge service with event bus
        from cns.services.domain_knowledge_service import get_domain_knowledge_service
        domain_service = get_domain_knowledge_service(event_bus=event_bus)

        if domain_service:
            logger.info("Domain knowledge service initialized and subscribed to TurnCompletedEvent")
        else:
            logger.info("Domain knowledge service not available (Letta API key not configured)")

    def _initialize_segment_collapse_handler(self, event_bus: EventBus):
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
            lt_memory_factory=lt_memory_factory
        )

        logger.info("Segment collapse handler initialized and subscribed to SegmentTimeoutEvent")

    def _initialize_manifest_query_service(self, event_bus: EventBus):
        """
        Initialize manifest query service with event bus.

        The service subscribes to ManifestUpdatedEvent for cache invalidation.
        """
        logger.info("Initializing manifest query service with event subscriptions")

        from cns.services.manifest_query_service import get_manifest_query_service

        # Initialize service with event bus (singleton pattern)
        manifest_service = get_manifest_query_service(event_bus=event_bus)

        logger.info("Manifest query service initialized and subscribed to ManifestUpdatedEvent")

    def cleanup(self):
        """Clean up all initialized services."""
        logger.info("Cleaning up CNS integration factory")

        if self._working_memory:
            self._working_memory.cleanup_all_managers()

        # Pointer summary coordinator cleanup disabled while using APScheduler
        # if self._pointer_summary_coordinator:
        #     self._pointer_summary_coordinator.cleanup()

        logger.info("CNS integration factory cleanup complete")



def create_cns_orchestrator(config_instance = None) -> ContinuumOrchestrator:
    """
    Convenience function to create a fully configured CNS orchestrator.
    
    Args:
        config_instance: MIRA configuration instance. If None, will use global config.
        
    Returns:
        ContinuumOrchestrator with all integrations
    """
    factory = CNSIntegrationFactory(config_instance)
    return factory.create_orchestrator()
