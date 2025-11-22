"""
Asynchronous context search tool for MIRA.

This tool performs multi-turn agentic search across conversation history,
long-term memory, and the web. It returns immediately while a background
agent iteratively searches until sufficient context is found.
"""
import json
import logging
import threading
import uuid
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from clients.valkey_client import get_valkey_client
from utils.user_context import get_current_user_id, get_current_user
from utils.timezone_utils import utc_now, format_utc_iso

if TYPE_CHECKING:
    from tools.repo import ToolRepository
    from working_memory.core import WorkingMemory
    from cns.integration.event_bus import EventBus


# -------------------- CONFIGURATION --------------------

class GetContextToolConfig(BaseModel):
    """Configuration for getcontext_tool."""
    enabled: bool = Field(
        default=True,
        description="Whether this tool is enabled"
    )
    max_iterations: int = Field(
        default=8,
        description="Maximum search iterations before forcing completion"
    )
    standard_completion_threshold: int = Field(
        default=3,
        description="Minimum iterations for standard mode before checking completion"
    )
    deep_completion_threshold: int = Field(
        default=5,
        description="Minimum iterations for deep mode before checking completion"
    )
    search_result_ttl: int = Field(
        default=300,
        description="TTL for search results in Valkey (seconds)"
    )


# Register configuration
registry.register("getcontext_tool", GetContextToolConfig)


# -------------------- SEARCH AGENT --------------------

class ContextSearchAgent:
    """Agent that orchestrates iterative search with completion criteria."""

    SEARCH_SYSTEM_PROMPT = """You are a context search agent that determines when searches have found sufficient information.

Your tasks:
1. Plan search strategies based on queries
2. Determine what to search next based on current findings
3. Evaluate if search is complete based on criteria
4. Summarize findings effectively

Search modes:
- standard: Find key information that answers the query
- deep: Find comprehensive context from multiple perspectives

Be systematic and thorough."""

    def __init__(self, search_mode: str, tool_repo: Optional['ToolRepository'], config: GetContextToolConfig):
        self.search_mode = search_mode
        self.tool_repo = tool_repo
        self.config = config
        self.scratchpad = []
        self.search_history = []
        self.logger = logging.getLogger(__name__)

    def _get_llm_client(self):
        """Get LLM client using execution model from config."""
        from config.config_manager import config
        from clients.llm_provider import GenericProviderClient
        from clients.vault_client import get_api_key

        api_key = get_api_key(config.api.execution_api_key_name)

        return GenericProviderClient(
            api_key=api_key,
            model=config.api.execution_model,
            api_endpoint=config.api.execution_endpoint,
            temperature=0.3,
            max_tokens=1000
        )

    def plan_search(self, query: str, search_scope: List[str]) -> Dict[str, Any]:
        """Create initial search plan."""
        prompt = f"""
Analyze this query and create a search plan.
Query: "{query}"
Available sources: {', '.join(search_scope)}

Identify:
1. Key concepts and entities to search for
2. Which sources are most likely to have relevant information
3. Search priority order

Return JSON:
{{
    "entities": ["list", "of", "key", "terms"],
    "concepts": ["broader", "concepts"],
    "source_priority": ["ordered", "list", "of", "sources"],
    "strategy": "brief strategy description"
}}"""

        client = self._get_llm_client()
        response = client.generate_response(
            messages=[
                {"role": "system", "content": self.SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        content = client.extract_text_content(response)
        return json.loads(content)

    def determine_next_search(self, original_query: str, plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine what to search next."""
        if not self.scratchpad and not self.search_history:
            # First search
            priority_source = plan['source_priority'][0] if plan['source_priority'] else 'conversation'
            return {
                'source': priority_source,
                'query': original_query
            }

        prompt = f"""
Original query: "{original_query}"
Search plan: {json.dumps(plan, indent=2)}
Searches performed: {json.dumps(self.search_history, indent=2)}
Current findings: {len(self.scratchpad)} items found

What information is still missing? What should we search next?

Return JSON or null if search is complete:
{{
    "source": "conversation|memory|web",
    "query": "specific search query",
    "reason": "why this search is needed"
}}"""

        client = self._get_llm_client()
        response = client.generate_response(
            messages=[
                {"role": "system", "content": self.SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        content = client.extract_text_content(response).strip()
        if content == "null" or not content:
            return None

        next_search = json.loads(content)
        self.search_history.append(next_search)
        return next_search

    def is_search_complete(self, query: str) -> tuple[bool, str]:
        """Check if we have sufficient context based on search mode."""
        prompt = f"""
Search mode: {self.search_mode}
Original query: "{query}"
Findings collected: {len(self.scratchpad)} items

Completion criteria:
- standard mode: Found specific answer OR covered main aspects of the query
- deep mode: Found comprehensive context from multiple sources and perspectives

Current findings summary:
{self._summarize_scratchpad_for_evaluation()}

Evaluate if the search has found sufficient information.

Return JSON:
{{
    "complete": true/false,
    "confidence": 0.0-1.0,
    "reason": "explanation",
    "missing": "what's still needed" (only if not complete)
}}"""

        client = self._get_llm_client()
        response = client.generate_response(
            messages=[
                {"role": "system", "content": self.SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        content = client.extract_text_content(response)
        result = json.loads(content)
        return result['complete'], result['reason']

    def add_to_scratchpad(self, source: str, findings: List[Dict[str, Any]]):
        """Add findings to scratchpad progressively."""
        for finding in findings:
            # Handle different result types (conversation, memory, web)
            if source == 'memory':
                content = finding.get('text', '')
                title = f"Memory (importance: {finding.get('importance_score', 0):.2f})"
            else:
                content = finding.get('summary', finding.get('content', finding.get('text', '')))
                title = finding.get('title', finding.get('display_title', ''))

            self.scratchpad.append({
                'source': source,
                'content': content,
                'metadata': {
                    'title': title,
                    'url': finding.get('url', ''),
                    'timestamp': finding.get('timestamp', finding.get('created_at', '')),
                    'confidence': finding.get('confidence_score', finding.get('confidence', finding.get('importance_score', 0))),
                    'entity_links': finding.get('entity_links', [])
                },
                'added_at': format_utc_iso(utc_now())
            })

    def summarize_findings(self, query: str) -> Dict[str, Any]:
        """Create final summary of all findings."""
        prompt = f"""
Create a focused summary of search findings for this query:
"{query}"

Search mode: {self.search_mode}
Total findings: {len(self.scratchpad)}

Findings:
{self._format_scratchpad_for_summary()}

Create a summary that:
1. Directly addresses the original query
2. Groups related findings
3. Highlights most relevant information
4. Notes any gaps or limitations

Return JSON:
{{
    "query": "original query",
    "summary": "executive summary answering the query",
    "key_findings": [
        {{
            "point": "key finding",
            "source": "where found",
            "details": "relevant details",
            "confidence": 0.0-1.0
        }}
    ],
    "sources_searched": ["list of sources used"],
    "confidence": 0.0-1.0,
    "limitations": "any gaps or caveats"
}}"""

        client = self._get_llm_client()
        response = client.generate_response(
            messages=[
                {"role": "system", "content": self.SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        content = client.extract_text_content(response)
        return json.loads(content)

    def _summarize_scratchpad_for_evaluation(self) -> str:
        """Summarize scratchpad for completion evaluation."""
        by_source = {}
        for item in self.scratchpad:
            source = item['source']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item['content'][:200] + '...')

        summary_parts = []
        for source, items in by_source.items():
            summary_parts.append(f"\n{source.upper()} ({len(items)} findings):")
            for item in items[:3]:
                summary_parts.append(f"- {item}")

        return '\n'.join(summary_parts)

    def _format_scratchpad_for_summary(self) -> str:
        """Format full scratchpad for final summary."""
        formatted_parts = []

        for idx, item in enumerate(self.scratchpad, 1):
            formatted_parts.append(f"\n[Finding {idx}]")
            formatted_parts.append(f"Source: {item['source']}")
            formatted_parts.append(f"Content: {item['content']}")

            meta = item['metadata']
            if meta.get('title'):
                formatted_parts.append(f"Title: {meta['title']}")
            if meta.get('timestamp'):
                formatted_parts.append(f"Time: {meta['timestamp']}")
            if meta.get('confidence'):
                formatted_parts.append(f"Confidence: {meta['confidence']}")

        return '\n'.join(formatted_parts)


# -------------------- MAIN TOOL CLASS --------------------

class GetContextTool(Tool):
    """Asynchronous agentic context search across memory, conversations, and web."""

    name = "getcontext_tool"

    simple_description = """
    Launches an autonomous search agent that gathers context from conversations, memories, and web.
    Returns instantly - search runs in background. Use liberally whenever additional context might exist.
    Agent handles search planning and synthesis automatically. Results appear in context window when ready.
    """

    anthropic_schema = {
        "name": "getcontext_tool",
        "description": "Launches background search agent across conversations, memories, and web. Returns instantly - agent autonomously plans searches, gathers context, synthesizes findings. Use liberally when additional context might exist. Results appear in context window asynchronously.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The context or information to search for"
                },
                "search_scope": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["conversation", "memory", "web"]
                    },
                    "description": "Which sources to search (default: all)"
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["standard", "deep"],
                    "description": "Standard: stops when key info found. Deep: exhaustive search."
                }
            },
            "required": ["query"]
        }
    }

    def __init__(self,
                 tool_repo: Optional['ToolRepository'] = None,
                 working_memory: Optional['WorkingMemory'] = None):
        """Initialize with dependency injection for tool interoperability."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tool_repo = tool_repo
        self.working_memory = working_memory
        self.valkey = get_valkey_client()

        # Load config
        from config.config_manager import config
        self.config = config.getcontext_tool

        # Get event bus from working memory if available
        self.event_bus = None
        if self.working_memory:
            self.event_bus = self.working_memory.event_bus

    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """Start async search and return immediately."""
        if not query:
            raise ValueError("Query is required")

        # Create task ID
        task_id = str(uuid.uuid4())

        # Get continuum_id from user context
        user_context = get_current_user()
        user_id = get_current_user_id()
        continuum_id = user_context.get('continuum_id', user_id)

        # Parse options
        search_scope = kwargs.get('search_scope', ['conversation', 'memory', 'web'])
        search_mode = kwargs.get('search_mode', 'standard')

        # Validate search_scope
        valid_scopes = {'conversation', 'memory', 'web'}
        search_scope = [s for s in search_scope if s in valid_scopes]
        if not search_scope:
            search_scope = ['conversation', 'memory', 'web']

        # Publish pending state immediately
        self._publish_pending_result(
            continuum_id=continuum_id,
            task_id=task_id,
            query=query,
            search_scope=search_scope,
            search_mode=search_mode
        )

        # Capture ambient context to propagate to background thread
        from contextvars import copy_context
        ctx = copy_context()

        # Start background thread with captured context
        search_thread = threading.Thread(
            target=ctx.run,
            args=(self._async_search_worker, task_id, continuum_id, query, search_scope, search_mode),
            name=f"getcontext-{task_id[:8]}"
        )
        search_thread.daemon = True
        search_thread.start()

        return {
            "success": True,
            "task_id": task_id,
            "message": f"Context search initiated for: '{query}'. Results will appear when ready.",
            "status": "searching",
            "search_scope": search_scope,
            "search_mode": search_mode
        }

    def _async_search_worker(self, task_id: str, continuum_id: str,
                           query: str, search_scope: List[str], search_mode: str):
        """Background worker that performs iterative search with timeout enforcement.

        Note: Runs within copied context from parent thread, so ambient context
        (user_id, etc.) is available via get_current_user_id().

        Thread owns its outcome - publishes success, timeout, or failure events directly.
        """
        start_time = utc_now()
        timeout_seconds = self.config.search_result_ttl  # 300 seconds (5 minutes)

        try:
            self.logger.info(f"Starting context search {task_id[:8]} for query: {query}")

            # Initialize search agent
            agent = ContextSearchAgent(
                search_mode=search_mode,
                tool_repo=self.tool_repo,
                config=self.config
            )

            # Decompose query
            search_plan = agent.plan_search(query, search_scope)
            self.logger.debug(f"Search plan: {search_plan}")

            # Iterative search loop with timeout checking
            iteration = 0
            min_iterations = (self.config.deep_completion_threshold
                            if search_mode == 'deep'
                            else self.config.standard_completion_threshold)

            while iteration < self.config.max_iterations:
                iteration += 1

                # CHECK TIMEOUT EACH ITERATION
                elapsed = (utc_now() - start_time).total_seconds()
                if elapsed > timeout_seconds:
                    self.logger.warning(
                        f"Search {task_id[:8]} timed out after {elapsed:.1f}s at iteration {iteration}"
                    )
                    self._publish_timeout_result(
                        continuum_id=continuum_id,
                        task_id=task_id,
                        query=query,
                        iteration=iteration,
                        elapsed=elapsed,
                        search_mode=search_mode,
                        findings_count=len(agent.scratchpad)
                    )
                    return  # Exit early on timeout

                # Determine next search
                next_search = agent.determine_next_search(query, search_plan)

                if not next_search:
                    self.logger.info(f"No more searches needed after {iteration} iterations")
                    break

                self.logger.debug(f"Iteration {iteration}: Searching {next_search['source']} for: {next_search['query']}")

                # Execute search based on source
                if next_search['source'] == 'conversation' and 'conversation' in search_scope:
                    results = self._search_conversation(next_search['query'])
                    agent.add_to_scratchpad('conversation', results)

                elif next_search['source'] == 'memory' and 'memory' in search_scope:
                    results = self._search_memory(next_search['query'])
                    agent.add_to_scratchpad('memory', results)

                elif next_search['source'] == 'web' and 'web' in search_scope:
                    results = self._search_web(next_search['query'])
                    agent.add_to_scratchpad('web', results)

                # Check completion after minimum iterations
                if iteration >= min_iterations:
                    is_complete, reason = agent.is_search_complete(query)
                    if is_complete:
                        self.logger.info(f"Search complete after {iteration} iterations: {reason}")
                        break

            # Log if we hit the iteration limit
            if iteration >= self.config.max_iterations:
                self.logger.warning(f"Search hit iteration limit of {self.config.max_iterations}")

            # SUCCESS PATH - Summarize findings
            final_summary = agent.summarize_findings(query)
            final_summary['iterations'] = iteration
            final_summary['search_mode'] = search_mode
            final_summary['task_id'] = task_id

            self.logger.info(f"Context search {task_id[:8]} complete with {len(agent.scratchpad)} findings")

            # Publish success result
            self._publish_success_result(
                continuum_id=continuum_id,
                task_id=task_id,
                summary=final_summary
            )

        except Exception as e:
            # FAILURE PATH - Thread crashed
            self.logger.error(f"Context search {task_id[:8]} failed: {e}", exc_info=True)
            self._publish_failure_result(
                continuum_id=continuum_id,
                task_id=task_id,
                query=query,
                error=str(e),
                error_type=type(e).__name__
            )

    def _search_conversation(self, query: str) -> List[Dict[str, Any]]:
        """Use continuumsearch_tool to search conversation history."""
        if not self.tool_repo:
            self.logger.warning("No tool repository available for conversation search")
            return []

        tool = self.tool_repo.get_tool('continuumsearch_tool')
        result = tool.run(
            operation='search',
            query=query,
            search_mode='summaries',
            max_results=5
        )
        return result.get('results', [])

    def _search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search long-term memory using continuumsearch_tool."""
        if not self.tool_repo:
            self.logger.warning("No tool repository available for memory search")
            return []

        tool = self.tool_repo.get_tool('continuumsearch_tool')
        result = tool.run(
            operation='search',
            query=query,
            search_mode='memories',
            max_results=5
        )
        return result.get('results', [])

    def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """Use webaccess_tool to search the web."""
        if not self.tool_repo:
            self.logger.warning("No tool repository available for web search")
            return []

        tool = self.tool_repo.get_tool('webaccess_tool')
        result = tool.run(
            operation='web_search',
            query=query,
            max_results=3
        )
        return result.get('results', [])

    def _publish_success_result(self, continuum_id: str, task_id: str,
                                summary: Dict[str, Any]) -> None:
        """Publish successful search results to trinket."""
        if self.event_bus is None:
            return
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=continuum_id,
            target_trinket='GetContextTrinket',
            context={
                'task_id': task_id,
                'status': 'success',
                'summary': summary
            }
        ))

    def _publish_timeout_result(self, continuum_id: str, task_id: str,
                                query: str, iteration: int, elapsed: float,
                                search_mode: str, findings_count: int) -> None:
        """Publish timeout notification to trinket."""
        if self.event_bus is None:
            return
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=continuum_id,
            target_trinket='GetContextTrinket',
            context={
                'task_id': task_id,
                'status': 'timeout',
                'query': query,
                'iteration': iteration,
                'elapsed': elapsed,
                'search_mode': search_mode,
                'findings_count': findings_count
            }
        ))

    def _publish_failure_result(self, continuum_id: str, task_id: str,
                                query: str, error: str, error_type: str) -> None:
        """Publish failure notification to trinket."""
        if self.event_bus is None:
            return
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=continuum_id,
            target_trinket='GetContextTrinket',
            context={
                'task_id': task_id,
                'status': 'failed',
                'query': query,
                'error': error,
                'error_type': error_type
            }
        ))

    def _publish_pending_result(self, continuum_id: str, task_id: str,
                                 query: str, search_scope: List[str],
                                 search_mode: str) -> None:
        """Publish pending/searching notification to trinket."""
        if self.event_bus is None:
            return
        from cns.core.events import UpdateTrinketEvent
        self.event_bus.publish(UpdateTrinketEvent.create(
            continuum_id=continuum_id,
            target_trinket='GetContextTrinket',
            context={
                'task_id': task_id,
                'status': 'pending',
                'query': query,
                'search_scope': search_scope,
                'search_mode': search_mode
            }
        ))

