# How to Build a Trinket

*Guide for creating event-driven system prompt components*

## What is a Trinket?

A **trinket** is an event-driven component that dynamically generates sections of the system prompt. Unlike tools (which execute user requests), trinkets **passively reflect system state** in Claude's context.

**Key Differences from Tools:**

| Aspect | Tools | Trinkets |
|--------|-------|----------|
| Purpose | Execute user operations | Display system state in prompt |
| Trigger | Claude invokes via function call | Events (turn completion, state changes) |
| Interaction | Active command execution | Passive observation and reporting |
| Output | Results returned to Claude | Content injected into system prompt |
| Examples | Send email, check weather | Show active sessions, list available tools |

## When to Build a Trinket

Build a trinket when you need to:

✅ **Display dynamic state** in the system prompt (active sessions, loaded tools, current context)
✅ **Update prompt based on events** (turn completion, state changes, tool loading)
✅ **Provide ambient awareness** (what's running, what's available, recent activity)
✅ **Cache semi-static information** (user preferences, tool guidance)

❌ **Don't build a trinket** for:
- Executing user commands (use a tool instead)
- One-time data retrieval (use a tool)
- Processing user input (use a tool)
- Persistent storage (trinkets are ephemeral prompt generators)

## Essential Patterns

### Pattern 1: Basic Trinket Structure

**See:** `user_info_trinket.py` for the simplest example

```python
from typing import Dict, Any
from .base import EventAwareTrinket

class MyTrinket(EventAwareTrinket):
    """Brief description of what this trinket displays."""

    # Cache policy: True if content rarely changes
    cache_policy = False

    def _get_variable_name(self) -> str:
        """The variable name in the system prompt."""
        return "my_section"

    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate the content for this trinket.

        Args:
            context: Update context with any relevant data

        Returns:
            Formatted string for system prompt, or "" if no content
        """
        # Your content generation logic here
        return "=== MY SECTION ===\nContent here"
```

### Pattern 2: Event Subscription

**See:** `tool_loader_trinket.py:59-60` for turn completion subscription

Trinkets can subscribe to events to update state or trigger refreshes:

```python
def __init__(self, event_bus, working_memory):
    super().__init__(event_bus, working_memory)

    # Subscribe to events
    self.event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)

def _handle_turn_completed(self, event):
    """Handle turn completion - update state or trigger refresh."""
    # Update internal state based on event
    self.current_turn = event.turn_number

    # Trigger refresh if needed
    self.working_memory.publish_trinket_update(
        target_trinket="MyTrinket",
        context={"action": "refresh"}
    )
```

**Common Events:**
- `TurnCompletedEvent` - After each conversation turn
- `UpdateTrinketEvent` - Explicit update request from tools or system
- `TrinketContentEvent` - Content publication (you publish this)

### Pattern 3: Custom Update Handling

**See:** `tool_loader_trinket.py:66-90` for action-based updates

For trinkets that respond to specific actions:

```python
def handle_update_request(self, event) -> None:
    """Handle custom update actions before generating content."""
    context = event.context
    action = context.get('action')

    # Process action-specific logic
    if action == 'add_item':
        self._handle_add_item(context)
    elif action == 'remove_item':
        self._handle_remove_item(context)

    # Always call parent to generate and publish content
    super().handle_update_request(event)
```

### Pattern 4: State Management

**See:** `tool_loader_trinket.py:48-52` for state tracking

Trinkets maintain ephemeral state between updates:

```python
def __init__(self, event_bus, working_memory):
    super().__init__(event_bus, working_memory)

    # Track state relevant to content generation
    self.active_items: Dict[str, Any] = {}
    self.current_turn: int = 0
    self.last_update: Optional[datetime] = None
```

**Important:** Trinket state is **in-memory only**. For persistent data, use tools with database storage.

### Pattern 5: Content Formatting

**See:** `punchclock_trinket.py:43-72` for multi-section formatting

Use clear headers and consistent formatting:

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    """Generate formatted content with clear sections."""
    if not self.has_content():
        return ""  # Return empty if nothing to show

    parts = ["=== MAIN HEADER ==="]

    # Section 1
    if self.section1_data:
        parts.append("= Section 1 =")
        for item in self.section1_data:
            parts.append(f"- {item}")

    # Section 2
    if self.section2_data:
        parts.append("= Section 2 =")
        for item in self.section2_data:
            parts.append(f"- {item}")

    return "\n".join(parts)
```

**Formatting Guidelines:**
- Use `=== HEADER ===` for main sections
- Use `= Subsection =` for subsections
- Use `- ` for list items
- Return empty string `""` if no content (not placeholder text)
- Keep formatting consistent across trinkets

### Pattern 6: Caching

**See:** `user_info_trinket.py:23` for cache policy

Set `cache_policy = True` for content that rarely changes:

```python
class MyTrinket(EventAwareTrinket):
    # Enable caching for semi-static content
    cache_policy = True
```

**When to cache:**
- User preferences and metadata
- Static guidance or documentation
- Tool descriptions and capabilities
- Configuration information

**When NOT to cache:**
- Active sessions or running state
- Turn-specific information
- Frequently changing data
- Time-sensitive information

### Pattern 7: Error Handling

**See:** `user_info_trinket.py:46-50` for user context checks

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    """Generate content with proper error handling."""
    from utils.user_context import get_current_user_id

    # Check prerequisites
    user_id = get_current_user_id()
    if not user_id:
        logger.warning("MyTrinket called without user context")
        return ""  # Gracefully return empty

    try:
        # Generate content
        data = self._fetch_data()
        return self._format_data(data)

    except Exception as e:
        # Log error but don't crash
        logger.error(f"Error generating content: {e}")
        return ""  # Return empty on error
```

**Error Handling Principles:**
- **Gracefully degrade** - return empty string on errors
- **Log problems** - use logger.warning or logger.error
- **Don't propagate** - trinket failures shouldn't break the conversation
- **Optional features** - if trinket can't load, conversation continues

### Pattern 8: Database Queries

**See:** `user_info_trinket.py:59-70` for database access

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    """Query database for content."""
    from utils.database_session_manager import get_shared_session_manager
    from utils.user_context import get_current_user_id

    user_id = get_current_user_id()
    session_manager = get_shared_session_manager()

    with session_manager.get_session(user_id) as session:
        result = session.execute_single(
            "SELECT data FROM my_table WHERE user_id = %(user_id)s",
            {'user_id': user_id}
        )

        if not result:
            return ""

        return f"=== MY DATA ===\n{result['data']}"
```

### Pattern 9: Timezone Handling

**See:** `punchclock_trinket.py:171-176` for timezone conversion

```python
from utils.timezone_utils import convert_from_utc, format_datetime
from utils.user_context import get_user_timezone

def _format_timestamp(self, utc_timestamp: str) -> str:
    """Convert UTC timestamp to user's timezone."""
    try:
        user_tz = get_user_timezone()
    except RuntimeError:
        user_tz = "UTC"  # Fallback if not configured

    dt = datetime.fromisoformat(utc_timestamp)
    local_dt = convert_from_utc(dt, user_tz)
    return format_datetime(local_dt, "date_time_short")
```

### Pattern 10: Triggering Updates from Tools

**See:** `punchclock_tool.py:600-610` for working memory integration

Tools can trigger trinket updates:

```python
# In a tool that changes state
def _update_session_state(self):
    """Update state and notify trinket."""
    # Perform state change
    self._save_to_database()

    # Trigger trinket update
    if self._working_memory:
        self._working_memory.publish_trinket_update(
            target_trinket="PunchclockTrinket",
            context={"action": "session_updated"}
        )
```

## Complete Example: Session Monitor Trinket

Here's a complete example showing all essential patterns:

```python
"""
Session monitor trinket - displays active user sessions.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from .base import EventAwareTrinket
from utils.timezone_utils import utc_now, format_datetime, convert_from_utc
from utils.user_context import get_current_user_id, get_user_timezone

logger = logging.getLogger(__name__)


class SessionMonitorTrinket(EventAwareTrinket):
    """
    Displays active sessions in the system prompt.

    Subscribes to TurnCompletedEvent to check for stale sessions
    and updates when tools notify of session changes.
    """

    # Sessions change frequently, don't cache
    cache_policy = False

    def __init__(self, event_bus, working_memory):
        """Initialize and subscribe to events."""
        super().__init__(event_bus, working_memory)

        # Track active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.current_turn: int = 0

        # Subscribe to turn completion for cleanup
        self.event_bus.subscribe('TurnCompletedEvent', self._handle_turn_completed)

        logger.info("SessionMonitorTrinket initialized")

    def _get_variable_name(self) -> str:
        """Publishes to 'active_sessions'."""
        return "active_sessions"

    def handle_update_request(self, event) -> None:
        """Handle session-specific updates."""
        context = event.context
        action = context.get('action')

        if action == 'session_started':
            session_id = context.get('session_id')
            self.active_sessions[session_id] = {
                'started_at': utc_now().isoformat(),
                'name': context.get('name', 'Unnamed'),
                'last_activity': utc_now().isoformat()
            }

        elif action == 'session_ended':
            session_id = context.get('session_id')
            self.active_sessions.pop(session_id, None)

        # Generate and publish content
        super().handle_update_request(event)

    def _handle_turn_completed(self, event):
        """Check for stale sessions on turn completion."""
        self.current_turn = event.turn_number

        # Remove sessions with no activity in 1 hour
        cutoff = utc_now().timestamp() - 3600
        stale = []

        for session_id, session in self.active_sessions.items():
            last_activity = datetime.fromisoformat(session['last_activity'])
            if last_activity.timestamp() < cutoff:
                stale.append(session_id)

        for session_id in stale:
            self.active_sessions.pop(session_id)
            logger.info(f"Removed stale session {session_id}")

        # Trigger refresh if we removed anything
        if stale:
            self.working_memory.publish_trinket_update(
                target_trinket="SessionMonitorTrinket",
                context={"action": "cleanup_completed"}
            )

    def generate_content(self, context: Dict[str, Any]) -> str:
        """Generate session list for system prompt."""
        if not self.active_sessions:
            return ""  # No content if no sessions

        try:
            user_tz = get_user_timezone()
        except RuntimeError:
            user_tz = "UTC"

        parts = ["=== ACTIVE SESSIONS ==="]

        for session_id, session in sorted(
            self.active_sessions.items(),
            key=lambda x: x[1]['started_at'],
            reverse=True
        ):
            name = session['name']
            started = datetime.fromisoformat(session['started_at'])
            local_time = convert_from_utc(started, user_tz)
            time_str = format_datetime(local_time, "time_short")

            # Calculate duration
            duration = utc_now() - started
            duration_str = self._format_duration(duration.total_seconds())

            parts.append(f"- [{session_id[:6]}] {name} — started {time_str} ({duration_str})")

        return "\n".join(parts)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return "just now"
```

## Development Workflow

### 1. Define Purpose

What state does this trinket display? When should it update?

```python
"""
Purpose: Display active punchclock sessions
Updates: When sessions start/stop, on turn completion
Cache: No - state changes frequently
"""
```

### 2. Identify Events

What events should trigger updates?

- `TurnCompletedEvent` for periodic checks
- Custom actions from tools for immediate updates

### 3. Design Content Format

How will this appear in the system prompt?

```
=== MAIN HEADER ===
= Subsection =
- Item 1
- Item 2
```

### 4. Implement Core Pattern

Start with basic structure:

```python
class MyTrinket(EventAwareTrinket):
    cache_policy = False

    def _get_variable_name(self) -> str:
        return "my_section"

    def generate_content(self, context: Dict[str, Any]) -> str:
        return "=== MY SECTION ===\nContent"
```

### 5. Add State and Events

Add state tracking and event subscriptions as needed.

### 6. Register in Working Memory

**File:** `working_memory/core.py`

Add to trinket initialization:

```python
self.my_trinket = MyTrinket(self.event_bus, self)
```

## Common Patterns

### Pattern: Cleanup on Turn Completion

```python
def _handle_turn_completed(self, event):
    """Clean up stale data."""
    self.current_turn = event.turn_number

    # Remove old items
    items_to_remove = [
        item_id for item_id, item in self.items.items()
        if self._is_stale(item)
    ]

    for item_id in items_to_remove:
        self.items.pop(item_id)
```

### Pattern: Conditional Content

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    """Only show content when relevant."""
    if not self.has_relevant_data():
        return ""  # Don't add empty sections

    return self._format_content()
```

### Pattern: Multi-Section Output

```python
def generate_content(self, context: Dict[str, Any]) -> str:
    """Generate multi-section output."""
    parts = ["=== MAIN HEADER ==="]

    # Section 1
    if self.section1_data:
        parts.append("= Section 1 =")
        parts.extend(self._format_section1())

    # Section 2
    if self.section2_data:
        parts.append("= Section 2 =")
        parts.extend(self._format_section2())

    return "\n".join(parts)
```

## Testing Your Trinket

### Manual Testing

1. **Check initialization:**
   ```python
   # In working_memory/core.py
   logger.info(f"MyTrinket initialized: {self.my_trinket._variable_name}")
   ```

2. **Trigger updates:**
   ```python
   # From a tool or test
   working_memory.publish_trinket_update(
       target_trinket="MyTrinket",
       context={"action": "test"}
   )
   ```

3. **Verify content:**
   - Check system prompt composition
   - Verify formatting and headers
   - Confirm empty content returns ""

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Content not appearing | Not registered in core.py | Add to trinket initialization |
| Updates not triggering | No event subscription | Subscribe in __init__ |
| Formatting broken | Inconsistent headers | Use === and = patterns |
| Crashes on error | Uncaught exceptions | Add try/except, return "" |
| Stale data shown | No cleanup logic | Add turn completion handler |

## Best Practices

1. **Return empty string when no content** - Don't add placeholder text
2. **Gracefully handle errors** - Log but don't crash
3. **Keep formatting consistent** - Use standard header patterns
4. **Cache when appropriate** - Set cache_policy for static content
5. **Clean up stale data** - Use TurnCompletedEvent for cleanup
6. **Log important events** - Use logger.info for state changes
7. **Document your trinket** - Explain what it shows and when it updates

## Reference Implementations

| Trinket | Purpose | Key Patterns |
|---------|---------|--------------|
| `user_info_trinket.py` | Display user metadata | Caching, database query, simple structure |
| `punchclock_trinket.py` | Show active sessions | Formatting, timezone conversion, partitioning |
| `tool_loader_trinket.py` | List available/loaded tools | Event subscription, state management, cleanup |
| `manifest_trinket.py` | Show conversation metadata | Simple static content |

## Summary

Trinkets are **passive prompt composers** that:
- Display system state in the system prompt
- Update based on events (not user commands)
- Gracefully degrade on errors
- Return empty string when no content
- Use consistent formatting patterns

For user-facing operations, build a **tool** instead. For system state reflection in the prompt, build a **trinket**.
