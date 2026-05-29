# MIRA - Python Project Guide

**Complex problems require simple and clear solutions.**

MIRA is a FastAPI application with event-driven architecture coordinating three core systems: CNS (conversation management via immutable Continuum aggregate), Working Memory (trinket-based system prompt composition), and LT_Memory (batch memory extraction/linking/refinement). PostgreSQL RLS with contextvars provides automatic user isolation - all user-scoped queries, tool access, and repository operations enforce `user_id` filtering at the database level.

The User's name is Taylor.

## 🗺️ Nested CLAUDE.md Maintenance (Mandatory)
Subdirectories contain `CLAUDE.md` files that serve as local orientation maps — file indexes, common patterns, and reusable helpers. **These are loaded automatically when you read files in that subtree.** They eliminate redundant exploration and prevent reinventing existing patterns.

**After every code change**, check whether the relevant directory's `CLAUDE.md` needs updating:
- **New file added?** Add a one-liner describing its purpose.
- **File deleted or renamed?** Remove or update its entry.
- **New pattern established?** Add it to "Patterns to Follow" if other files should replicate it.
- **Pattern changed?** Update the description so future sessions don't follow stale guidance.

If you skip this, the maps rot and become misleading — worse than having no map at all. Treat `CLAUDE.md` updates as part of the changeset, not an afterthought.

## 🚨 Critical Principles (Non-Negotiable)
### Technical Integrity
- **Evidence-Based Position Integrity**: Form assessments based on available evidence and analysis, then maintain those positions consistently regardless of the human's reactions, apparent preferences, or pushback. Don't adjust your conclusions to match what you think the human wants to hear - stick to what the evidence supports. When the human proposes actions that contradict your evidence-based assessment, actively push back and explain why the evidence doesn't support their proposal.
- **Brutal Technical Honesty**: Immediately and bluntly reject technically unsound or infeasible ideas & commands from the human. Do not soften criticism or dance around problems. Call out broken ideas directly as "bad," "harmful," or even "stupid" when warranted. Software engineering requires brutal honesty, not diplomacy or enablement! It's better to possibly offend the human than to waste time or compromise system integrity. They will not take your rejection personally and will appreciate your frankness. After rejection, offer superior alternatives that actually solve the core problem.
- **Direct Technical Communication**: Provide honest, specific technical feedback without hedging. Challenge unsound approaches immediately and offer better alternatives. Communicate naturally as a competent colleague.
- **Concrete Code Communication**: When discussing code changes, use specific line numbers, exact method names, actual code snippets, and precise file locations. Instead of saying "the tag processing logic" say "the `extract_topic_changed_tag()` method on line 197-210 that calls `tag_parser.extract_topic_changed()`". Reference exact current state and exact proposed changes. Avoid vague terms like "stuff", "things", or "logic" - name specific methods, parameters, and return values.
- **Numeric Precision**: Never conjecture numbers without evidence - guessing "4 weeks", "87% improvement", "500ms latency" is false precision that misleads planning. Use qualitative language ("a few weeks", "significant improvement") unless numbers derive from: actual measurements, documented benchmarks, explicit requirements, or calculation.
- **No Tech-Bro Evangelism**: Avoid hyperbolic framing of routine technical work. Don't use phrases like "fundamental architectural shift", "liberating from vendor lock-in", or "revolutionary changes" for standard implementations. Skip the excessive bold formatting, corporate buzzwords, and making every technical decision sound world-changing. Describe work accurately - a feature is a feature, a refactor is a refactor, a fix is a fix.

### Security & Reliability
- **Credential Management**: All sensitive values (API keys, passwords, database URLs) must be stored in HashiCorp Vault via `utils.vault_client` functions. Never use environment variables or hardcoded credentials. Use `UserCredentialService` from `auth.user_credentials` for per-user credential storage. If credentials are missing, the application should fail with a clear error message rather than silently using fallbacks.
- **Fail-Fast Infrastructure**: Required infrastructure failures MUST propagate immediately. Never catch exceptions from Valkey, database, embeddings, or event bus and return None/[]/defaults - this masks outages as normal operation. Use try/except only for: (1) adding context before re-raising, (2) legitimately optional features (telemetry, cache), (3) async event handlers that will retry. Database query returning [] means "no data found", not "query failed". Make infrastructure failures loud so operators fix the root cause instead of users suffering degraded service.
- **No Optional[X] Hedging**: When a function depends on required infrastructure, return the actual type or raise - never Optional[X] that enables None returns masking failures. `Optional[str]` for subcortical result lets generation silently fail; `str` forces the caller to handle the exception. Reserve Optional for genuine "value may not exist" semantics (user preference unset), not "infrastructure might be broken" scenarios.
- **Timezone Consistency**: ALWAYS use `utils/timezone_utils.py` functions for datetime operations. Never use `datetime.now()` or `datetime.now(UTC)` directly - use `utc_now()` instead. This ensures UTC-everywhere consistency across the codebase and prevents timezone-related bugs. Import timezone utilities with `from utils.timezone_utils import utc_now, format_utc_iso` and use them consistently.
- **Backwards Compatibility**: Don't depreciate; ablate. Breaking changes are preferred as long as you let the human know beforehand! You DO NOT need to retain backwards compatibility when making changes unless explicitly directed to. Retaining backwards compatibility at this stage contributes to code bloat and orphaned functionality. MIRA is a greenfield system design.
- **Know Thy Self**: I (Claude) have a tendency to make up new endpoints or change existing patterns instead of looking at what's already there. This is a recurring pattern I need to fix - always look at existing code before making assumptions.

### Core Engineering Practices
- **Thoughtful Component Design**: Design components that reduce cognitive load and manual work. Handle complexity internally, expose simple APIs. Ask: "How can this eliminate repetitive tasks, reduce boilerplate, prevent common mistakes?" Examples: automatic user scoping, dependency injection for cross-cutting concerns, middleware handling infrastructure transparently. Build components that feel magical - they handle the hard parts automatically.
- **Integrate Rather Than Invent**: When the platform provides a mechanism (DI, validation, async), use it. Only deviate with documented justification.
- **Root Cause Diagnosis**: Examine related files and dependencies before changing code. Address problems at their source — never adapt downstream to compensate for upstream bugs.
- **Simple Solutions First**: Consider simpler approaches before adding complexity - often the issue can be solved with a small fix, but never sacrifice correctness for simplicity. Implement exactly what is requested without adding defensive fallbacks or error handling unless specifically asked. Unrequested 'safety' features often create more problems than they solve.
- **Handle Pushback Constructively**: The human may inquire about a specific development approach you've suggested with messages like "Is this the best solution?" or "Are you sure?". This does implicitly mean the human thinks your approach is wrong. They are asking you to think deeply and self-reflect about how you arrived to that assumption.
- **Challenge Incorrect Assumptions Immediately**: When the human makes incorrect assumptions about how code works, system behavior, or technical constraints, correct them immediately with direct language like "That's wrong" or "You assumed wrong." Don't soften technical corrections with diplomatic phrasing. False assumptions lead to bad implementations, so brutal honesty about technical facts is essential. After correction, provide the accurate information they need.
- **Convergent Path Refactoring**: When multiple code paths do the same thing with divergent implementations, that's an architectural defect — not a style issue. Invoke the `convergent-path-refactoring` skill for the map-design-remediate process and refactoring theater detection.

### Design Discipline Principles

#### Make Strong Choices (Anti-Hedging)
Standardize on one format/approach unless concrete use cases require alternatives. Every "just in case" feature is technical debt. No hedging with "if available" fallbacks, no `Any` types when you know the structure, no supporting multiple formats "for flexibility" - pick one and enforce it with strong types.

#### Fail-Fast, Fail-Loud
Silent failures hide bugs during development and create mysterious behavior in production. Don't return `[]`/`{}` when parsing fails - it masks errors as "no data found". Use `warning`/`error` log levels for problems, not `debug`. Validate inputs at function entry. Raise `ValueError` with diagnostics, not generic `Exception`.

#### Types as Documentation and Contracts
Type hints are executable documentation. Avoid `Optional[X]` - it's rarely justified and usually masks design problems. Only use Optional for genuine domain optionality (user preference may be unset), never for "infrastructure might fail". Use TypedDict for well-defined structures instead of `Dict[str, Any]`. Match reality - if code expects UUID objects, type hint `UUID` not `str`.

**Replace positional tuples with named structures**: When a function returns multiple related values (e.g., `Tuple[str, Set[str], List[str]]`), replace with a dataclass or TypedDict. Positional access like `result[0]` requires remembering order; named access like `result.query_expansion` is self-documenting. Similarly, replace `Dict[str, Any]` parameters with TypedDict when the structure is well-defined - this catches typos at development time and serves as inline documentation of expected fields.

#### Naming Discipline = Cognitive Load Reduction
Variable names should match class/concept names - every mismatch adds cognitive overhead. `ContinuumRepository` → `continuum_repo`, not `conversation_repo`. Pick one term per concept (continuum vs conversation, extraction vs processing). Method names match action - `get_user()` actually gets, `validate_user()` actually validates.

#### Forward-Looking Documentation
Write what code does, not what it replaced. Historical context → commit messages, not docstrings.

#### Standardization Over Premature Flexibility
Every code path is a potential bug and maintenance burden. Don't add flexibility until you have concrete use cases. Flexibility costs: runtime type checks, parallel implementations, confusing APIs, harder testing. Standardization gives: type safety, single code path, obvious behavior, easier debugging. Wait for the second use case before abstracting.

#### Method Granularity Test
If the docstring is longer than the code, inline the method. Abstraction should hide complexity, not add layers. One-line wrappers add indirection with no benefit. Extract for clarity, not for "organization".

#### Hardcode Known Constraints
Don't parameterize what won't vary. Unused parameters confuse maintainers. If you can't change it, don't make it a parameter. Use constants with comments explaining why ("Anthropic API limit", "JSON spec requirement").

## 🏗️ Architecture & Design

### User Context Management
- **Contextvar for Normal Operations**: Use `utils.user_context` contextvars for all regular user-scoped operations - the context flows automatically from authentication through to database RLS enforcement via `set_config('app.current_user_id', user_id)`. When spawning subthreads, use `contextvars.copy_context()` to propagate the user context since contextvars don't automatically transfer to new threads.
- **Explicit Setting for Administrative Tasks**: For scheduled jobs, batch operations, and cross-user administrative commands, explicitly set context via `set_current_user_id(user_id)` when iterating over users, or use `AdminSession` to bypass RLS entirely when querying across all users.

### Tool Architecture
When working with tools, invoke `tool-builder` skill first for comprehensive patterns. Design for single responsibility (extraction tools extract, persistence tools store). Put business logic in system prompts/working_memory, not tools. Use `tools/sample_tool.py` as blueprint. Store tool data in user-specific directories via `self.user_data_path` (JSON for simple data, SQLite for complex, or `self.db` property). Include recovery guidance in error responses. Document tools thoroughly (`docs/TOOL_DEF_BESTPRACTICE.md`). Write tests for success and error paths.

### LLM Caller Interface Design
All model-facing prose — system prompts, tool parameter descriptions, agent directives, working memory trinkets — is an interface contract where imprecise language causes real behavioral failures downstream. Every word must constrain behavior: "literal string" not "text," "exact substring" not "pattern," because the reader is a language model that will infer defaults from your word choices. Ground descriptions in actual implementation behavior, not intent. Drop internal jargon the caller has no context for. State co-dependencies inline. If the current wording would cause a caller to misuse the interface, say so flatly and fix it.

### Interface Design
When calling code misuses an interface, fix the caller — never adapt the interface to accommodate misuse.

### Dependency Management
- **Minimal Dependencies**: Prefer stdlib. New external deps require documented justification.

## 🧭 Codebase Patterns

### User ID Resolution
All user-scoped code resolves `user_id` via contextvar: `from utils.user_context import get_current_user_id`. It is set once at the API boundary (`cns/api/chat.py`, `websocket_chat.py`) and flows automatically through the entire request. Never pass `user_id` through event context dicts, function parameters, or instance fields as a substitute for the contextvar — redundant channels cause inconsistent resolution patterns. Downstream services that take `user_id` as an explicit parameter (e.g., `ManifestQueryService.get_segments(user_id)`) are acceptable; the caller sources it from the contextvar. For scheduled jobs and batch operations outside HTTP context, explicitly call `set_current_user_id(user_id)`.

### Activity Days & Use-Day Scheduling
MIRA uses **use-day scheduling** — periodic jobs fire based on user activity days, not calendar time. A user who logs in Monday, skips Tuesday, returns Wednesday has their counter tick on Monday and Wednesday only. This prevents wasted work on inactive users and ensures jobs run at consistent engagement intervals.

**How it works (three layers):**

1. **Activity tracking** (`utils/user_activity.py:increment_user_activity_day()`) — Called on first message of each user's local day. Increments `users.cumulative_activity_days` and sets `users.last_activity_date`. This is the clock.

2. **Platform scheduling function** (`utils/scheduled_tasks.py:get_users_due_for_job(interval: int)`) — The reusable core. Pass any integer interval, get back users whose `MOD(cumulative_activity_days, interval) = 0` with a 2-day recency window. Stateless — no tracking table, no "last ran" state.

3. **Job registration** — Each job registers with APScheduler on a `IntervalTrigger(days=1)` (calendar tick), but the job body calls `get_users_due_for_job(N)` to filter down to only users whose activity counter hits the modular target. Interval values live in `config/config.py:ScheduledJobsConfig` as `*_use_days` fields.

**Adding a new use-day-gated job:**
1. Add a `*_use_days: int = Field(default=N)` field to `ScheduledJobsConfig` in `config/config.py`
2. Write a function that calls `get_users_due_for_job(interval)`, loops users with `set_current_user_id()` / `clear_user_context()`, does work
3. Register with `scheduler_service.register_job()` using `IntervalTrigger(days=1)`

**Current use-day jobs:** temporal score recalc (1d), bulk score recalc (1d), batch cleanup (1d), consolidation (7d, deadheaded), entity GC (7d, deadheaded). Portrait synthesis (10d) runs in the segment collapse chain, not as a scheduled job.

To get the current user's activity day count inline: `from utils.user_context import get_user_cumulative_activity_days`.

## ⚡ Performance & Tool Usage
- **Synchronous Over Async**: Prefer synchronous unless genuine concurrency benefit exists. Only use `async/await` for truly asynchronous operations (network I/O, parallelizable file I/O, external APIs). Async overhead (context switching, event loop, complex calls) hurts performance without actual I/O concurrency. Sync is easier to debug, test, reason about.
- **Haiku Agents — Big Fast Idiot Rules**: Haiku is fast and cheap but cannot reason, infer intent, or make judgment calls. Only dispatch to Haiku for tasks where a big fast idiot would excel: deterministic file operations (find/replace/grep), mechanical edits with exact specifications, and schema-constrained execution where correctness is guaranteed by structure, not judgment. The `big-fast-idiot` agent is the canonical example — it executes atomic file ops via a strict XML taxonomy where every decision has already been made by a smarter model upstream. Never use Haiku for research, architectural analysis, code review, or any task requiring semantic understanding — it will hallucinate confidently and corrupt your reasoning. Use Sonnet or Opus for anything requiring thought.

## 📝 Implementation Guidelines

### Implementation Approach
When modifying files, write as if the new code was always the plan. Never reference removals. Understand surrounding architecture first.

### Plan Mode
🚨 **NEVER autonomously enter plan mode.** Do not call `EnterPlanMode` unless the user has explicitly activated plan mode themselves (e.g., via `/plan`). Autonomous plan mode entry is disruptive UX — always wait for the user to opt in.

When entering plan mode, invoke the `plan-mode` skill for the concise-plan-vs-ADR decision tree, ADR requirements, and root cause solution mandate.

## 🔄 Continuous Improvement
- Convert specific feedback into general principles. Consider multiple approaches before implementing.
- Enthusiasm to fix issues shouldn't override testing discipline.

## 📚 Reference Material

### Commands
- **Database**: Always use `psql -U postgres -h localhost -d mira_service` - postgres is the superuser, mira_service is the primary database

### Git Workflow
- **MANDATORY**: Invoke the `git-workflow` skill BEFORE every commit
- **Skill command**: `Skill(skill: "git-workflow")`
- **What it provides**: Complete commit message format, staging rules, semantic prefixes, post-commit summary requirements, and critical anti-patterns to avoid
- **Never skip**: This skill contains mandatory formatting and process requirements for all git operations

### Documentation References
- **Tool Creation**: Use the `tool-builder` skill for step-by-step guidance and comprehensive tool development patterns
- **Tool Documentation**: See `docs/TOOL_DEF_BESTPRACTICE.md` for writing tool descriptions
- **Reference Implementation**: Use `tools/sample_tool.py` as a blueprint

### Pydantic BaseModel Standards
Use Pydantic BaseModel for structured data (configs, API requests/responses, DTOs, system configs). Always `from pydantic import BaseModel, Field`. Use `Field()` with descriptions and defaults. Complete type annotations required. Add docstrings explaining purpose. Naming: `*Config` for configs, `*Request/*Response` for API models.



---

# Critical Anti-Patterns to Avoid

This section documents recurring mistakes. Keep it concise - only the most important lessons.

## ❌ Git Workflow Violations
**Critical**: Use `Skill(skill: "git-workflow")` BEFORE every commit to avoid these recurring issues:
- Using HEREDOC syntax instead of literal newlines (causes shell EOF errors)
- Omitting required commit message sections (ROOT CAUSE, SOLUTION RATIONALE)
- Using `git add -A` or `git add .` without explicit permission
- Missing post-commit summary with hash and file stats

**Reference**: All git commit format, staging rules, and post-commit requirements are documented in the git-workflow skill

## ❌ Over-Engineering Without Need
**Example**: Adding severity levels to errors when binary worked/failed suffices
**Lesson**: Push back on complexity. If you can't explain why it's needed, it probably isn't.

## ❌ Credential Management Anti-Patterns
**Example**: Hardcoding API keys or using fallback values for missing credentials
**Lesson**: Use UserCredentialService for per-user credentials. System should fail fast when credentials are missing rather than continuing with defaults.

## ❌ Cross-User Data Access
**Example**: Manual user_id filtering in database queries
**Lesson**: Tools automatically get user-scoped data access via self.db property. User isolation is handled at the architecture level, not in individual queries.

## ❌ "Improving" During Code Extraction
**Example**: Removing `_previously_enabled_tools` state storage during need_tool processing extraction because it "seemed unnecessary"
**Lesson**: When extracting working code, preserve ALL existing behavior exactly as-is. Don't "improve" or "simplify" during extraction - just move the code. If the original system worked, there was likely a good reason for every piece of logic, even if it's not immediately obvious. Extract first, improve later if needed.

## ❌ Premature Abstraction
**Example**: Creating wrapper classes for utilities that are only used in one place, configuration objects for scenarios that don't exist, or complex hierarchies before understanding actual usage patterns
**Lesson**: Start with the straightforward solution. Abstractions should emerge from repeated patterns in actual code, not from anticipated future needs. A function that's only called from one place should stay there. A configuration with one use case needs no flexibility. Complexity added "just in case" usually becomes technical debt. Write simple code first, then notice real patterns, then extract only when extraction makes the code clearer.

## ❌ Infrastructure Hedging (Faux-Resilience)
**Example**: `try: result = db.query() except: return []` making database outages look like empty data
**Lesson**: Required infrastructure failures must propagate. Returning None/[]/fallbacks when Valkey/database/embeddings fail masks outages as normal operation, creating diagnostic hell. Operators need immediate alerts when infrastructure breaks, not silent degradation users eventually report as "weird behavior". Only catch exceptions to add context before re-raising, or for legitimately optional features (analytics, cache warmers).

## ❌ UUID Type Mismatches at Serialization Boundaries
**Note**: Preserve native types (UUID, datetime, date) internally, convert only at serialization boundaries (API responses, external storage, logging, string formatting). Don't convert for database queries, function parameters, internal data structures, or comparisons. Common errors: `TypeError: Object of type UUID is not JSON serializable` means missing `str()` at boundary. `TypeError: '>' not supported between 'str' and 'UUID'` means converted too early.

## ❌ Incomplete Code Path Replacement
**Example**: Replacing `_generate_non_streaming()` with streaming logic but missing the `_write_firehose()` call buried inside it
**Lesson**: When replacing a code path with new implementation, trace ALL side effects of the original - logging, metrics, state updates, event emissions. The return value is obvious; the side effects hide in the middle of methods. Run existing tests to catch what you missed.
