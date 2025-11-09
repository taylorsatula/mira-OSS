# How to Build a Tool

*Technical guide based on successful tool development patterns*

## Core Concepts

Effective tool building leverages two different types of pattern recognition:
- **Human perspective**: Real-world usage patterns, social dynamics, UX intuition
- **AI perspective**: System architecture, technical constraints, code consistency

Neither perspective alone produces optimal tools. The best solutions emerge from the intersection.

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

```python
# Required reading before implementation
tools/implementations/reminder_tool.py    # Complete tool example
tools/implementations/contacts_tool.py    # Tool with UUID linking
tools/implementations/punchclock_tool.py  # Tool with dependency injection
tools/repo.py                             # Base class and ToolRepository
utils/userdata_manager.py                 # Database API reference
```

Deviating from established patterns causes integration issues and maintenance debt.

### Phase 4: Incremental Implementation

Build order matters:
1. Data models (foundational structure)
2. Basic CRUD operations  
3. Core business logic
4. Complex features
5. Security/validation layers

```python
# Progress tracking
TodoWrite({
    "todos": [
        {"content": "Create data models", "activeForm": "Creating data models", "status": "pending"},
        {"content": "Implement basic operations", "activeForm": "Implementing basic operations", "status": "pending"},
        {"content": "Add validation", "activeForm": "Adding validation", "status": "pending"}
    ]
})

# Error handling pattern
def _add_item(self, title: str, **kwargs) -> Dict[str, Any]:
    if not title:
        self.logger.error("Title is required")
        raise ValueError("Title is required for this operation")

    try:
        # Implementation
        item_id = str(uuid.uuid4())
        timestamp = format_utc_iso(utc_now())

        self.db.insert('items', {
            'id': item_id,
            'title': title,
            'created_at': timestamp
        })

        return {"success": True, "item_id": item_id}

    except Exception as e:
        self.logger.error(f"Failed to add item: {e}")
        raise
```

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

Each tool automatically gets access to comprehensive user-scoped resources via `self.db`:

**1. User-Scoped SQLite Database**
Dedicated SQLite database per user with automatic encryption for fields marked with `encrypted__` prefix:

```python
# All database operations are automatically user-scoped
reminders = self.db.select('reminders', 'completed = 0')
reminder = self.db.select('reminders', 'id = :id', {'id': reminder_id})

# Fields with encrypted__ prefix are automatically encrypted/decrypted
self.db.insert('contacts', {
    'id': contact_id,
    'encrypted__name': 'John Doe',  # Encrypted on write, decrypted on read
    'encrypted__email': 'john@example.com',  # Encrypted
    'encrypted__phone': '555-1234',  # Encrypted
    'created_at': timestamp,  # Not encrypted
    'updated_at': timestamp  # Not encrypted
})

# On read, encrypted__ prefix is stripped and values are decrypted
contacts = self.db.select('contacts')  # Returns {'id': ..., 'name': 'John Doe', 'email': ..., ...}
```

**2. Credential Storage**
Store and retrieve user API keys, passwords, and other credentials securely using UserCredentialService:

```python
from utils.user_credentials import UserCredentialService

# Initialize service (auto-detects current user)
cred_service = UserCredentialService()

# Store credentials
cred_service.store_credential(user_id, 'api_key', 'openweather', user_api_key)
cred_service.store_credential(user_id, 'oauth_token', 'spotify', token)

# Retrieve credentials
api_key = cred_service.get_credential(user_id, 'api_key', 'openweather')
token = cred_service.get_credential(user_id, 'oauth_token', 'spotify')

# List all user credentials
creds = cred_service.list_user_credentials(user_id)

# Delete credentials
cred_service.delete_credential(user_id, 'api_key', 'openweather')
```

**3. Tool-Specific File Storage**
Each tool gets a dedicated directory for JSON files, downloads, or other file-based data:

```python
# Get tool's data directory (automatically created)
data_dir = self.db.get_tool_data_dir(self.name)

# Store files
config_file = data_dir / "settings.json"
with open(config_file, 'w') as f:
    json.dump(settings, f)

# Access other user directories
conversations_dir = self.db.conversations_dir
config_path = self.db.config_path
```

**4. User Context Access**
Access user preferences and metadata:

```python
from utils.user_context import get_current_user_id, get_user_timezone

user_id = get_current_user_id()  # Current user's ID
timezone = get_user_timezone()  # User's timezone preference
```

### Time Handling
```python
from utils.timezone_utils import utc_now, format_utc_iso, parse_time_string
from utils.user_context import get_user_timezone

# Never use datetime.now() or datetime.now(UTC) directly
# All timestamps stored as UTC ISO format strings
timestamp = format_utc_iso(utc_now())  # "2025-01-15T10:30:00Z"

# All database datetime columns store ISO strings, not datetime objects
```

### Tool Registration and Auto-Discovery

Tools are automatically discovered and loaded - no manual registration required:

**How It Works**:
1. Place your tool file in `tools/implementations/`
2. Restart the MIRA process

That's it. The system automatically:
- Scans `tools/implementations/` for Tool subclasses
- Creates a default configuration (with `enabled=True`)
- Creates user data directories
- Makes the tool available to the AI

**Optional Custom Configuration**:

If your tool needs custom configuration options beyond `enabled`, define and register a config class:

```python
from pydantic import BaseModel, Field
from tools.registry import registry

class PagerToolConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    max_message_length: int = Field(default=300, description="Maximum message length")
    ttl_hours: int = Field(default=24, description="Message time-to-live in hours")

# Registration is OPTIONAL - only needed if you want custom config options
registry.register("pager_tool", PagerToolConfig)
```

**Without registration**: Tool gets a default config with just `enabled=True`
**With registration**: Tool gets your custom config with all your defined options

No setup files, no environment variables, no manual wiring - the system handles everything.

### Tool Descriptions

Tools require two description fields with different purposes:

```python
# Simple description - MUST be extremely concise action phrase
simple_description = "checks the weather"  # Good: short, actionable
simple_description = "sends messages to other people"  # Good: clear, brief
simple_description = "controls Kasa smart home devices"  # Good: specific, concise

# Bad examples - too wordy or explanatory
simple_description = "This tool allows you to check weather conditions"  # Too verbose
simple_description = "Weather checking functionality"  # Too formal
```

### Anthropic Schema
```python
# Required class attribute defining tool interface
anthropic_schema = {
    "name": "tool_name",
    "description": "Clear description of what this tool does and when to use it",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add_item", "get_items", "update_item", "delete_item"],
                "description": "The operation to perform"
            },
            "item_id": {
                "type": "string",
                "description": "Unique identifier for the item (required for update/delete)"
            }
        },
        "required": ["operation"],
        "additionalProperties": False
    }
}
```

### Database Operations
```python
# UserDataManager provides these methods via self.db property

# Create table (call once in __init__ or on first use)
schema = """
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL
"""
self.db.create_table('items', schema)

# Insert record
self.db.insert('items', {
    'id': item_id,
    'title': 'Example',
    'created_at': format_utc_iso(utc_now())
})

# Select records
all_items = self.db.select('items')  # All records
active = self.db.select('items', 'status = :status', {'status': 'active'})
single = self.db.select('items', 'id = :id', {'id': item_id})

# Update record
self.db.update('items',
    {'title': 'New Title', 'updated_at': format_utc_iso(utc_now())},
    'id = :id',
    {'id': item_id}
)

# Delete record
self.db.delete('items', 'id = :id', {'id': item_id})

# Execute raw SQL
results = self.db.execute('SELECT * FROM items WHERE created_at > :date',
    {'date': cutoff_date}
)

# Automatic encryption: title, description, name, email, phone, content fields
# are automatically encrypted on insert/update and decrypted on select
```

### Standard Imports
```python
# Every tool should import these
import json
import logging
import uuid
from typing import Dict, Any, Optional, List

from tools.repo import Tool
from utils.timezone_utils import utc_now, format_utc_iso, parse_time_string
from utils.user_context import get_current_user_id, get_user_timezone

# Optional imports
from pydantic import BaseModel, Field  # Only if defining custom config
from tools.registry import registry  # Only if registering custom config
from datetime import datetime, timedelta  # Common but not always needed
```

### Tool Structure Template
```python
from typing import Dict, Any, Optional
import logging
import uuid

from pydantic import BaseModel, Field
from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import utc_now, format_utc_iso

# Configuration (OPTIONAL - only if you need custom config beyond enabled=True)
class MyToolConfig(BaseModel):
    """Configuration for my_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled")
    # Add custom configuration options here

# Registration is OPTIONAL - only needed if you defined a custom config above
registry.register("my_tool", MyToolConfig)

# Tool implementation
class MyTool(Tool):
    name = "my_tool"
    description = "Brief description of what this tool does"
    simple_description = "concise action phrase"  # KEEP SUPER CONCISE: "checks the weather", "sends messages", "controls smart devices"

    anthropic_schema = {
        "name": "my_tool",
        "description": "Detailed description for the AI",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["operation1", "operation2"],
                    "description": "The operation to perform"
                }
            },
            "required": ["operation"],
            "additionalProperties": False
        }
    }

    def __init__(self):
        """Initialize the tool and create database tables."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._ensure_table()

    def _ensure_table(self):
        """Create database table if it doesn't exist."""
        schema = """
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        """
        self.db.create_table('my_data', schema)

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Execute the requested operation."""
        try:
            # Handle kwargs JSON string if needed
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in kwargs: {e}")

            # Route to operation handlers
            if operation == "operation1":
                return self._operation1(**kwargs)
            elif operation == "operation2":
                return self._operation2(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        except Exception as e:
            self.logger.error(f"Error executing {operation}: {e}")
            raise

    def _operation1(self, **kwargs) -> Dict[str, Any]:
        """Implementation of operation1."""
        # Validate inputs
        # Perform operation
        # Return structured response
        return {"success": True, "message": "Operation completed"}
```

### Error Handling
```python
# Use standard Python exceptions
# ValueError: Invalid input parameters
if not required_param:
    self.logger.error("Required parameter missing")
    raise ValueError("Parameter 'required_param' is required")

# RuntimeError: Failed operations
if api_call_failed:
    self.logger.error(f"API call failed: {error_details}")
    raise RuntimeError(f"Failed to complete operation: {error_details}")

# TimeoutError: Operation exceeded time limits
if elapsed > timeout:
    raise TimeoutError(f"Operation timed out after {timeout}s")

# ConnectionError: Network issues
if not connected:
    raise ConnectionError("Unable to connect to service")

# Always log errors before raising
self.logger.error(f"Operation failed: {error_message}")
raise

# Structured error responses
return {
    "success": False,
    "message": "Clear error message for the AI",
    "details": {"additional": "context"}
}
```

### Dependency Injection
```python
# Tools can declare dependencies in __init__ signature
# ToolRepository automatically injects known types

from typing import Optional

class MyTool(Tool):
    def __init__(self,
                 tool_repo: Optional['ToolRepository'] = None,
                 working_memory: Optional['WorkingMemory'] = None):
        """
        Args:
            tool_repo: Injected by ToolRepository
            working_memory: Injected by ToolRepository
        """
        super().__init__()
        self.tool_repo = tool_repo
        self.working_memory = working_memory

# Supported injection types:
# - LLMProvider / LLMBridge
# - ToolRepository
# - WorkingMemory
```

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