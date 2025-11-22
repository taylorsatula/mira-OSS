# How to Build a Tool

*Technical guide based on successful tool development patterns*

## 🚀 START HERE

Tools in MIRA follow consistent patterns. This guide shows you where to find each pattern in existing, battle-tested implementations. Use the **Pattern Index** below to jump directly to the code that demonstrates what you need.

**Learning approach:**
1. Scan the Pattern Index to see what's available
2. Find the pattern you need in an existing tool
3. Read that specific implementation with line numbers provided
4. Copy the pattern and adapt it to your use case

## Core Concepts

Effective tool building leverages two different types of pattern recognition:
- **Human perspective**: Real-world usage patterns, social dynamics, UX intuition
- **AI perspective**: System architecture, technical constraints, code consistency

Neither perspective alone produces optimal tools. The best solutions emerge from the intersection.

## 📋 Pattern Index

### Essential Patterns

Every tool needs these core patterns:

| Pattern | Where to Find | What It Shows |
|---------|---------------|---------------|
| **Tool Structure** | reminder_tool.py:38-96 | Required metadata (name, descriptions, schema) |
| **Configuration** | weather_tool.py:48-104 | Pydantic config with Field descriptions, multiple options |
| | reminder_tool.py:38-63 | Simpler config example |
| **Initialization** | reminder_tool.py:179-233 | Deferred table creation with has_user_context() check |
| | contacts_tool.py:127-150 | Another initialization example |
| **Database Schema** | reminder_tool.py:199-233 | Tables with encrypted__ fields, proper indexes |
| | contacts_tool.py:132-150 | Schema with foreign keys |
| **Operation Routing** | reminder_tool.py:303-335 | JSON kwargs parsing, operation dispatch, error handling |
| | contacts_tool.py:152-199 | Clean operation routing pattern |
| **Input Validation** | weather_tool.py:211-456 | Collecting all errors before raising, specific messages |
| | reminder_tool.py:362-390 | Required field validation |
| **CRUD Operations** | contacts_tool.py:201-579 | Complete CRUD with UUID generation, encryption |
| | reminder_tool.py:337-582 | CRUD with cross-tool linking |
| **Timezone Handling** | reminder_tool.py:849-917 | UTC storage, natural language parsing, display conversion |
| | punchclock_tool.py:180-250 | Timezone conversion for display |
| **Encryption** | contacts_tool.py:201-260 | encrypted__ prefix for automatic encryption/decryption |
| | reminder_tool.py:362-430 | Encrypted fields in database operations |
| **Search & Scoring** | contacts_tool.py:314-400 | Score-based fuzzy matching with relevance |
| | kasa_tool.py:1234-1401 | Device name matching with similarity scoring |
| **File Operations** | weather_tool.py:107-200 | Cache file management, TTL checking |
| | kasa_tool.py:1214-1232 | JSON file read/write for config |
| **Response Formatting** | reminder_tool.py:805-847 | Consistent {"success": bool, ...} format with display data |
| | contacts_tool.py:580-650 | Formatting with linked entity data |
| **Error Handling** | reminder_tool.py:584-614 | Helpful errors with available options |
| | weather_tool.py:211-280 | Validation with collected error messages |

### Specialized Patterns

For advanced functionality, reference these specific implementations:

| Pattern | Tool | Lines | When You Need It |
|---------|------|-------|------------------|
| **Async Operations** | kasa_tool.py | 567-622 | Async device discovery, network calls |
| **API Retry Logic** | webaccess_tool.py | 469-675 | Exponential backoff, circuit breaker |
| **Fuzzy Name Matching** | contacts_tool.py | 314-400 | Scoring partial matches, handling ambiguity |
| | kasa_tool.py | 1234-1401 | Device name matching with suggestions |
| **Working Memory Integration** | punchclock_tool.py | 600-610 | Publishing trinket updates, UI refresh |
| **Complex API Caching** | weather_tool.py | 107-200 | File-based cache with TTL, cache invalidation |
| **Credential Management** | UserCredentialService | - | Per-user API keys, OAuth tokens |
| **UUID Cross-Tool Linking** | reminder_tool.py | 805-847 | Linking to contacts, fetching fresh data |
| | contacts_tool.py | 314-579 | Managing linked entities |
| **Natural Language Dates** | reminder_tool.py | 849-917 | Parsing "tomorrow", "next week", etc. |
| **LLM Integration** | webaccess_tool.py | 800-950 | Using LLM for content analysis and processing |
| **Batch Operations** | - | - | *Not yet implemented - good opportunity!* |
| **Rate Limiting** | - | - | *Not yet implemented - good opportunity!* |

## Development Process

### Phase 1: Requirements Discovery

Initial descriptions often use metaphors or analogies. Extract concrete requirements:

```
Example: "Like 90s pagers"
Extracted requirements:
- High urgency messaging only
- Minimal UI complexity  
- Respects user attention
- No feature creep
```

Essential questions:
- "Can you walk me through a typical usage scenario?"
- "What problem does this solve that existing tools don't?"
- "What should this tool explicitly NOT do?"
- "What would indicate success for users?"

### Phase 2: Specification Analysis

Detailed specifications often contain both explicit features and implicit design philosophy. Minor details frequently encode critical constraints.

Example: If a spec mentions "no notification fatigue," this implies rate limiting, priority systems, or other attention-management features.

### Phase 3: Codebase Pattern Study

**Use the Pattern Index above** to find exactly what you need. Each entry shows the tool and line numbers where the pattern is implemented.

**Recommended reading order:**
1. **Start simple**: `reminder_tool.py` - Clean CRUD operations, timezone handling, basic patterns
2. **Add complexity**: `contacts_tool.py` - Fuzzy search, encryption, UUID linking
3. **Learn caching**: `weather_tool.py` - API integration, file-based caching, validation
4. **Go async**: `kasa_tool.py` - Async operations, device management
5. **Advanced features**: `webaccess_tool.py` - Retry logic, LLM integration

**Infrastructure references:**
```python
tools/repo.py                   # Base Tool class, ToolRepository
utils/userdata_manager.py       # Database API (self.db operations)
utils/timezone_utils.py         # utc_now(), format_utc_iso(), etc.
utils/user_context.py           # get_current_user_id(), get_user_timezone()
```

**Critical:** Deviating from established patterns causes integration issues and maintenance debt. Always check the Pattern Index first.

### Phase 4: Incremental Implementation

Build order matters:
1. **Tool structure** - Define class, metadata, schema (see reminder_tool.py:38-96)
2. **Database schema** - Create tables with indexes (see reminder_tool.py:199-233)
3. **Basic CRUD** - Add/get/update/delete operations (see contacts_tool.py:201-579)
4. **Core business logic** - Your domain-specific code
5. **Advanced features** - Search, export, etc. as needed

**Key implementation principles:**

- **Track progress**: Use TodoWrite to break down complex implementations
- **Validate inputs**: Collect ALL errors before raising (see weather_tool.py:211-280)
- **Log before raising**: Always log errors before propagating (see reminder_tool.py:303-335)
- **Consistent responses**: Use `{"success": bool, "message": str, ...}` format (see reminder_tool.py:805-847)
- **Timezone everywhere**: Use `utc_now()` and `format_utc_iso()` for all timestamps (see reminder_tool.py:849-917)
- **Encrypted fields**: Prefix sensitive data with `encrypted__` (see contacts_tool.py:201-260)
- **Helpful errors**: Include suggestions and available options (see reminder_tool.py:584-614)

### Phase 5: Collaborative Solution Design

Real workshopping between skilled practitioners - direct but not theatrical:

```
H: "Need a high-urgency notification tool. Think 90s pagers - when it goes off,
    it matters. No spam, no noise."

AI: "So restricted senders? Like an allowlist of who can page you?"

H: "Allowlists are a pain to manage upfront. What if we make people solve a
    captcha before they can send?"

AI: "That won't work - you'd need an external captcha service, and if someone has
    an actual emergency they shouldn't be solving puzzles."

H: "Fair point. What about a time delay? Message sits for 30 seconds before
    sending, gives them time to reconsider?"

AI: "Same problem - real emergencies need immediate delivery. And a delay doesn't
    actually filter importance, it just annoys people."

H: "Okay, back to allowlist but somehow less annoying?"

AI: "What if we flip it - first message from a new sender goes to a review queue.
    You see it, approve or block them once, then future messages are automatic."

H: "So it's still an allowlist, just built retroactively instead of upfront?"

AI: "Right. You're not maintaining a list ahead of time. Someone sends, you review
    that specific message, make a decision, done."

H: "I like that better. But what happens when someone gets a new phone? Their
    device changed, so..."

AI: "Yeah we'd need device fingerprinting. Store a hash of their browser/device,
    check it when they send. If the hash changes, route back to review queue."

H: "That's gonna annoy people though. People upgrade phones all the time."

AI: "True. We could show context when it happens - 'New device detected for Mom,
    last device was used 2 weeks ago' so you can decide if that timing makes sense."

H: "Okay yeah, that way it's not just silently blocking them. What about message
    length? I want them short."

AI: "How short are we talking?"

H: "I don't know, maybe 300 characters? That's long enough for 'stuck in traffic,
    start the meeting without me' but short enough to force brevity."

AI: "A hard 300 char limit is pretty restrictive. What if someone has a real
    reason to send more detail?"

H: "Then it's not urgent enough for this tool, use regular messaging."

AI: "Sure, but people will push back on that. What if we auto-summarize messages
    over 300 chars instead of blocking them?"

H: "Using the LLM?"

AI: "Yeah, run it through summarization, show the summary to you by default, but
    the full message is still there if you click to expand."

H: "Does that defeat the whole brevity thing though?"

AI: "Not really - you still see brevity on your end. The sender just doesn't get
    blocked from expressing themselves. We're translating verbose to concise
    instead of punishing verbose."

H: "Okay I can see that working. So schema-wise we need operations for sending,
    approving, and blocking. Device hash field. Message field and summary field."

AI: "Standard user-scoped table. Status enum for pending, approved, blocked.
    Timestamps for created_at and maybe last_device_seen."

H: "How long should messages stick around? I don't want this to become a chat app."

AI: "24 hours? Keeps the ephemeral pager feel."

H: "Do we need a scheduled cleanup job for that?"

AI: "Could just check on read - delete anything older than 24h whenever someone
    queries their messages. Simpler than scheduling."

H: "Alright, let's build it."
```

Key observations from this exchange:
- Initial requirement was too vague, needed clarification
- Human's first two solutions (captcha, delay) were technically flawed
- AI had to directly reject bad ideas rather than being diplomatic
- Solution emerged through iteration, not a single insight
- Neither party had the answer immediately - they built it together
- Conversation included uncertainty ("I don't know", "Can we?")
- Real decision-making sounds tentative ("Let's try it") not triumphant
- Technical patterns emerged naturally from problem-solving, not name-dropped

### Phase 6: Handling Mid-Implementation Feedback

Interruptions during tool use contain valuable course corrections:

```
[Request interrupted by user for tool use]
"You're setting the default expiry to 48 hours but that's too long 
for a pager metaphor. These should be ephemeral - 24 hours max."
```

This feedback indicates design misalignment. Parse for:
- Specific parameter corrections
- Underlying philosophy mismatches  
- Missing requirements

## Technical Requirements

### User Scoping

**Every tool automatically gets user-scoped access via `self.db`** - no manual filtering needed.

**See reminder_tool.py:199-233** for complete database patterns including:
- Table creation with proper schema
- Encrypted fields (use `encrypted__` prefix)
- Indexes for performance
- CRUD operations

**Key Points:**

1. **Automatic User Scoping**: All `self.db` operations are scoped to the current user
2. **Automatic Encryption**: Fields prefixed with `encrypted__` are encrypted on write, decrypted on read
3. **Prefix Handling**: On read, the `encrypted__` prefix is automatically stripped from field names

**Example:**
```python
# Creating a table with encrypted fields
schema = """
    id TEXT PRIMARY KEY,
    encrypted__title TEXT NOT NULL,
    encrypted__notes TEXT,
    created_at TEXT NOT NULL
"""
self.db.create_table('my_items', schema)

# Insert - encryption happens automatically
self.db.insert('my_items', {
    'id': item_id,
    'encrypted__title': 'Secret Meeting',  # Will be encrypted
    'encrypted__notes': 'Confidential',     # Will be encrypted
    'created_at': timestamp
})

# Select - decryption happens automatically, prefix stripped
items = self.db.select('my_items')
# Returns: [{'id': '...', 'title': 'Secret Meeting', 'notes': 'Confidential', ...}]
#          ^^^^ Note: 'title' not 'encrypted__title'
```

### Credential Storage

For API keys and sensitive credentials, use `UserCredentialService`:

```python
from utils.user_credentials import UserCredentialService

cred_service = UserCredentialService()
api_key = cred_service.get_credential(user_id, 'api_key', 'service_name')
cred_service.store_credential(user_id, 'api_key', 'service_name', key_value)
```

### File Operations

**See weather_tool.py:107-200** for cache file management and kasa_tool.py:1214-1232 for JSON config files.

Tools get automatic file methods:
- `self.open_file(filename, mode)` - Open file in tool's directory
- `self.get_file_path(filename)` - Get full path to file
- `self.file_exists(filename)` - Check if file exists
- `self.make_dir(path)` - Create subdirectory

**Example:**
```python
# Export data to JSON
filename = f"export_{utc_now().strftime('%Y%m%d_%H%M%S')}.json"
with self.open_file(filename, 'w') as f:
    json.dump(data, f, indent=2)

full_path = self.get_file_path(filename)
return {"success": True, "file_path": str(full_path)}
```

### Timezone Handling

**See reminder_tool.py:849-917** for natural language date parsing and **punchclock_tool.py:171-176** for display conversion.

**CRITICAL: Always use UTC internally, convert only for display**

```python
from utils.timezone_utils import (
    utc_now, format_utc_iso, parse_utc_time_string,
    get_user_timezone, convert_from_utc
)

# Store as UTC ISO strings (ALWAYS)
timestamp = format_utc_iso(utc_now())
self.db.insert('items', {'created_at': timestamp})

# Parse stored UTC strings
stored_dt = parse_utc_time_string(item['created_at'])

# Convert to user's timezone ONLY for display
user_tz = get_user_timezone()
local_dt = convert_from_utc(stored_dt, user_tz)
display_string = format_datetime(local_dt, "date_time_short")
```

### Code Organization

Tools don't require specific section markers, but consistency helps. Look at existing tools for organization patterns:

```python
# Standard tool structure
import statements
logging setup
configuration class (if needed)
tool class with:
    - metadata (name, descriptions)
    - anthropic_schema
    - __init__
    - run() method
    - operation handlers (_add_item, _get_items, etc.)
    - helper methods
```

### Tool Registration and Auto-Discovery

**See reminder_tool.py:38-63** and **weather_tool.py:48-104** for configuration examples.

**Auto-Discovery**: Place your tool in `tools/implementations/` and restart MIRA. Done.

**Configuration (Optional)**:
```python
from pydantic import BaseModel, Field
from tools.registry import registry

class MyToolConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether enabled")
    max_items: int = Field(default=10, description="Max items to return")

# Only register if you have custom config beyond 'enabled'
registry.register("my_tool", MyToolConfig)
```

### Tool Descriptions

**See reminder_tool.py:38-96** for complete metadata examples.

Two required fields:
- `simple_description`: Ultra-concise action phrase ("manages reminders")
- `description`: Detailed explanation with features and use cases

### Anthropic Schema

**See reminder_tool.py:97-158** for complete schema with operations.

**Critical points:**
- Set `"additionalProperties": false` - prevents unexpected params
- Use `enum` for fixed options
- Clear descriptions - Claude uses these
- Mark required fields in `"required"` array

### Database Operations

**See reminder_tool.py:199-233** for schema creation and **contacts_tool.py:201-579** for CRUD patterns.

**Quick reference:**
```python
# Create table
schema = """
    id TEXT PRIMARY KEY,
    encrypted__title TEXT NOT NULL,
    created_at TEXT NOT NULL
"""
self.db.create_table('items', schema)

# Create indexes
self.db.execute("CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at)")

# CRUD operations
self.db.insert('items', {'id': id, 'encrypted__title': title, 'created_at': timestamp})
items = self.db.select('items', 'status = :status', {'status': 'active'})
self.db.update('items', {'encrypted__title': new_title}, 'id = :id', {'id': id})
self.db.delete('items', 'id = :id', {'id': id})
```

### Quick Start Template

Start with this minimal structure, then add patterns from the index as needed:

```python
import logging
import uuid
from typing import Dict, Any
from tools.repo import Tool
from utils.timezone_utils import utc_now, format_utc_iso

logger = logging.getLogger(__name__)

class MyTool(Tool):
    name = "my_tool"
    simple_description = "does something useful"
    description = "Detailed explanation of what this tool does"

    anthropic_schema = {
        "name": "my_tool",
        "description": "What this tool does and when to use it",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "get", "delete"],
                    "description": "The operation to perform"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._ensure_tables()

    def _ensure_tables(self):
        schema = """
            id TEXT PRIMARY KEY,
            encrypted__data TEXT,
            created_at TEXT NOT NULL
        """
        self.db.create_table('my_data', schema)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        try:
            if operation == "add":
                return self._add(**kwargs)
            elif operation == "get":
                return self._get(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error in {operation}: {e}")
            raise

    def _add(self, **kwargs) -> Dict[str, Any]:
        # Your implementation
        return {"success": True, "message": "Added"}
```

### Error Handling

**See reminder_tool.py:584-614** and **weather_tool.py:211-280** for error patterns.

**Key principles:**
- Always log before raising: `self.logger.error(f"Error: {e}")` then `raise`
- Use `ValueError` for invalid input, `RuntimeError` for operation failures
- Provide helpful messages with suggestions
- Collect all validation errors before raising

### Dependency Injection

**See punchclock_tool.py:231-233** for working memory integration.

Tools can request dependencies in `__init__`:
```python
def __init__(self, working_memory: Optional["WorkingMemory"] = None):
    super().__init__()
    self._working_memory = working_memory
```

Supported types: `LLMProvider`, `LLMBridge`, `ToolRepository`, `WorkingMemory`

ToolRepository automatically injects these when creating instances.

## Common Failure Patterns

| Pattern | Indicator | Resolution |
|---------|-----------|------------|
| Building without understanding | No clarifying questions asked | Stop coding, explore use case |
| Feature creep | Adding unrequested functionality | Return to core requirements |
| Over-engineering | Complex solution to simple problem | Discuss simpler alternatives |
| Under-communication | Assumptions instead of questions | Increase dialogue frequency |
| Misaligned mental models | "That's not quite right" feedback | Deep dive into user's vision |

## Best Practices

1. **Question assumptions**: Initial requirements are rarely complete
2. **Propose alternatives**: When you see technical issues, suggest better approaches
3. **Embrace iteration**: First implementations reveal hidden requirements
4. **Respect both perspectives**: Technical elegance without usability fails; user-friendly but broken also fails
5. **Document decisions**: When choosing between approaches, note why

## Commit Message Format

**Required Structure**: All commits must follow this detailed format with semantic prefixes:

```bash
# REQUIRED FORMAT - Use literal newlines, never HEREDOC
git commit -m "prefix: brief summary (50 chars max)

PROBLEM SOLVED:
Clear description of what issue this commit addresses

ROOT CAUSE: (if applicable)
Technical explanation of why the problem occurred

SOLUTION:
Detailed explanation of the approach taken

CHANGES:
- Bulleted list of specific code changes
- File modifications, method additions/removals
- API or interface changes

FUNCTIONALITY PRESERVED: (if applicable)
- What existing behavior remains unchanged
- Backward compatibility notes

IMPACT: (choose relevant sections)
- SECURITY: Security implications
- PERFORMANCE: Performance impact analysis
- BREAKING: Breaking changes and migration notes
- TESTING: Test coverage changes

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Semantic Prefixes** (use exactly these):
- `feat:` - New feature or functionality
- `fix:` - Bug fix or error correction
- `refactor:` - Code restructuring without functional changes
- `perf:` - Performance improvements
- `security:` - Security-related changes
- `test:` - Adding or modifying tests
- `docs:` - Documentation updates
- `chore:` - Maintenance tasks, dependency updates
- `style:` - Code formatting, whitespace fixes
- `revert:` - Reverting previous commits

## Summary

Tool building is a discovery process. The human's initial vision and the final implementation often differ significantly - not due to miscommunication, but because dialogue reveals better solutions. Success requires engaged technical discussion where both perspectives challenge and refine each other.

The extended pager example above isn't just documentation - it's a template for how these discoveries happen. Study the pattern: question, propose, identify issues, iterate, converge on solution neither party initially envisioned.