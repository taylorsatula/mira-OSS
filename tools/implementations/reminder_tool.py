"""
SQLite-based reminder tool for managing scheduled reminders with complete user isolation.

This tool stores all reminder data in user-specific SQLite databases, ensuring perfect
multi-tenant isolation through automatic user-scoped queries. Integrates with
the user's contacts by storing contact UUID references.
"""

import inspect
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import (
    convert_to_timezone,
    format_datetime, parse_time_string, utc_now, ensure_utc, format_utc_iso,
    parse_utc_time_string, convert_from_utc
)
from utils.user_context import get_user_preferences


class ReminderToolConfig(BaseModel):
    """Configuration for the reminder_tool."""
    enabled: bool = Field(default=True, description="Whether this tool is enabled by default")


# Register with registry
registry.register("reminder_tool", ReminderToolConfig)


class ReminderTool(Tool):
    """
    SQLite-based reminder tool with complete user isolation and contacts integration.

    All reminder data is stored in user-specific SQLite databases with automatic
    user isolation through the db property inherited from the Tool base class.

    Integrates with user's contacts by storing contact UUID references.
    """

    name = "reminder_tool"

    # Operations without ordering dependencies — safe for concurrent execution
    _parallel_safe_operations = frozenset({"get_reminders"})

    @classmethod
    def is_call_parallel_safe(cls, tool_input: Dict[str, Any]) -> bool:
        return tool_input.get("operation") in cls._parallel_safe_operations

    simple_description = "Create and manage scheduled reminders. Link reminders to contacts. Query by date (today, tomorrow, upcoming, overdue). Supports both user-facing and internal (MIRA's own) reminders."

    tool_schema = {
        "name": "reminder_tool",
        "description": "Create and manage scheduled reminders with contact linking.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add_reminder", "get_reminders", "mark_completed", "update_reminder", "delete_reminder", "snooze_reminder", "batch"],
                        "description": "Which operation to execute. snooze_reminder pushes due time forward from now (default 1 hour, set snooze_duration for custom)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Brief title for the reminder (required for add_reminder; if provided for update_reminder, replaces existing title)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Reminder datetime. Accepts ISO 8601 (YYYY-MM-DDTHH:MM:SS) or relative phrases like 'tomorrow', 'in 2 days', 'next week'. Required for add_reminder, optional for update_reminder"
                    },
                    "description": {
                        "type": "string",
                        "description": "Free-text description of the reminder (optional). If provided for update_reminder, replaces existing description"
                    },
                    "contact_name": {
                        "type": "string",
                        "description": "Contact name to link. Performs case-insensitive substring match against user's contacts. If no match found, reminder is created without a contact link. Optional"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["user", "internal"],
                        "description": "'user' = user-requested (default). 'internal' = proactive follow-up reminders YOU create when you notice something with a future resolution point: a pending result, something ordered, a decision deferred, a question you'd naturally circle back to — like a friend who just remembers to ask \"how'd it go?\" Set date to when your future self should notice. Write notes with enough context to act on cold. Bias toward creation; a false positive costs nothing, a missed observation is gone forever."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Extra context to store with the reminder. For 'internal' category reminders, include enough detail to act without conversation history. Optional"
                    },
                    "reminder_id": {
                        "type": "string",
                        "description": "Reminder ID (e.g., 'rem_a1b2c3d4'). Required for mark_completed, update_reminder, delete_reminder. For snooze_reminder, provide this, reminder_ids, or both — they are combined, not mutually exclusive"
                    },
                    "reminder_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of reminder ID strings to process. Required when operation='batch'. Optional for snooze_reminder (can combine with reminder_id). Duplicates are deduplicated automatically"
                    },
                    "date_filter": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "upcoming", "past", "all", "date", "range", "overdue"],
                        "description": "Date filter mode for get_reminders (required). today/tomorrow/upcoming/overdue return active reminders only. past/all include completed. 'date' needs specific_date, 'range' needs start_date + end_date"
                    },
                    "specific_date": {
                        "type": "string",
                        "description": "Target date for single-day query. Accepts ISO 8601 or relative phrases. Required when date_filter='date'. Matches reminders falling on that day (midnight to midnight)"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Range start (inclusive). Accepts ISO 8601 or relative phrases. Required when date_filter='range'"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End of date range. Accepts ISO 8601 or relative phrases. Required when date_filter='range'. Inclusive — reminders on this date are returned"
                    },
                    "batch_action": {
                        "type": "string",
                        "enum": ["complete", "delete"],
                        "description": "Action to apply to each reminder in reminder_ids. 'complete' = mark as completed, 'delete' = permanently remove. Required when operation='batch'"
                    },
                    "snooze_duration": {
                        "type": "string",
                        "description": "How long to snooze forward from now. Accepts '30 minutes', '2 hours', '1 day', '1 week', or a plain number (interpreted as hours). Default: '1 hour'. For snooze_reminder only"
                    },
                    "resolution_note": {
                        "type": "string",
                        "description": "Brief note recorded at completion time describing the outcome or what happened. For mark_completed and batch (with batch_action='complete') only. Especially useful for 'internal' reminders to close the loop on what you learned when following up"
                    }
                },
                "required": ["operation"],
                "additionalProperties": False
            }
        }

    def __init__(self):
        """Initialize the reminder tool with SQLite storage."""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Only create tables if user context is available (not during startup/discovery)
        from utils.user_context import has_user_context
        if has_user_context():
            self._ensure_reminders_table()
    
    def _ensure_reminders_table(self):
        """Create reminders table if it doesn't exist."""
        schema = """
            id TEXT PRIMARY KEY,
            encrypted__title TEXT NOT NULL,
            encrypted__description TEXT,
            reminder_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed INTEGER DEFAULT 0,
            completed_at TEXT,
            contact_uuid TEXT,
            encrypted__additional_notes TEXT,
            category TEXT DEFAULT 'user',
            encrypted__resolution_note TEXT
        """
        self.db.create_table('reminders', schema)

        # Create indexes for faster queries
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_date ON reminders(reminder_date)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_completed ON reminders(completed)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_contact ON reminders(contact_uuid)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_category ON reminders(category)")

        # Migrate: add resolution_note column for existing databases
        try:
            self.db.execute("ALTER TABLE reminders ADD COLUMN encrypted__resolution_note TEXT")
        except Exception:
            pass  # Column already exists

    def _load_reminders(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        """Load reminders from SQLite database."""
        if include_completed:
            # Load all reminders
            return self.db.select('reminders')
        else:
            # Load only active (not completed) reminders
            return self.db.select('reminders', 'completed = 0')

    def _save_reminder(self, reminder: Dict[str, Any]) -> str:
        """Save a reminder to SQLite database."""
        # Insert new reminder
        return self.db.insert('reminders', reminder)

    def _check_duplicate_reminder(self, title: str, reminder_date: datetime, window_seconds: int = 60) -> Optional[Dict[str, Any]]:
        """
        Check for duplicate reminders created within a time window.

        Prevents duplicate creation from retries, race conditions, or streaming artifacts.

        Args:
            title: Reminder title to check
            reminder_date: Reminder date to check
            window_seconds: Time window in seconds to check for duplicates

        Returns:
            Existing reminder dict if duplicate found, None otherwise
        """
        try:
            # Get all reminders and check for recent duplicates
            all_reminders = self.db.select('reminders')
            now = utc_now()
            window_start = now - timedelta(seconds=window_seconds)

            for existing in all_reminders:
                # Check if title matches (case-insensitive)
                existing_title = existing.get('encrypted__title', '')
                if existing_title.lower() != title.lower():
                    continue

                # Check if reminder_date matches (within 1 minute tolerance)
                existing_date_str = existing.get('reminder_date')
                if not existing_date_str:
                    continue
                try:
                    existing_date = parse_utc_time_string(existing_date_str)
                    if abs((existing_date - reminder_date).total_seconds()) > 60:
                        continue
                except Exception:
                    continue

                # Check if created recently (within window)
                created_at_str = existing.get('created_at')
                if not created_at_str:
                    continue
                try:
                    created_at = parse_utc_time_string(created_at_str)
                    if created_at >= window_start:
                        return existing
                except Exception:
                    continue

            return None
        except Exception as e:
            self.logger.warning(f"Duplicate check failed, proceeding with creation: {e}")
            return None

    def _load_contacts(self) -> List[Dict[str, Any]]:
        """Load user's contacts from SQLite database."""
        try:
            # UserDataManager does not expose table_exists; attempt select directly
            return self.db.select('contacts')
        except Exception as e:
            self.logger.warning(f"Failed to load contacts: {e}")
            return []

    def _get_contact_by_uuid(self, contact_uuid: str) -> Optional[Dict[str, Any]]:
        """Get a contact by UUID."""
        try:
            contacts = self.db.select('contacts', 'id = :uuid', {'uuid': contact_uuid})
            return contacts[0] if contacts else None
        except Exception as e:
            self.logger.warning(f"Failed to get contact {contact_uuid}: {e}")
            return None

    def run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a reminder operation.

        Args:
            operation: Operation to perform (see below for valid operations)
            **kwargs: Parameters for the specific operation

        Returns:
            Response data for the operation

        Raises:
            ValueError: If operation fails or parameters are invalid

        Valid Operations:

        1. add_reminder: Create a new reminder
           - Required: title, date
           - Optional: description, contact_name, notes
           - Returns: Dict with created reminder

        2. get_reminders: Retrieve reminders
           - Required: date_filter ("today", "tomorrow", "upcoming", "past", "all", "date" or
            "range")
           - If date_filter is "date", requires specific_date parameter
           - If date_filter is "range", requires start_date and end_date parameters
           - Returns: Dict with list of reminders

        3. mark_completed: Mark a reminder as completed
           - Required: reminder_id
           - Optional: resolution_note
           - Returns: Dict with updated reminder

        4. update_reminder: Update an existing reminder
           - Required: reminder_id
           - Optional: Any fields to update (title, description, date, contact_name)
           - Returns: Dict with updated reminder

        5. delete_reminder: Delete a reminder
           - Required: reminder_id
           - Returns: Dict with deletion confirmation

        6. snooze_reminder: Snooze reminder(s) by 1 hour
           - Required: reminder_id OR reminder_ids (at least one)
           - Returns: Dict with snoozed reminder(s) and new times
        """
        try:
            # Ensure reminders table exists on first use
            self._ensure_reminders_table()
            # Parse kwargs JSON string if provided that way
            if "kwargs" in kwargs and isinstance(kwargs["kwargs"], str):
                try:
                    params = json.loads(kwargs["kwargs"])
                    kwargs = params
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in kwargs for {operation}: {e}")
                    raise ValueError(f"Invalid JSON in kwargs: {e}")
            
            # Route to the appropriate operation, filtering kwargs to params
            # the target method actually accepts. The repo filters unknown schema
            # properties, but can't filter per-operation — schema-valid params like
            # 'title' pass through and crash methods that don't accept them.
            operation_map = {
                "add_reminder": self._add_reminder,
                "get_reminders": self._get_reminders,
                "mark_completed": self._mark_completed,
                "update_reminder": self._update_reminder,
                "delete_reminder": self._delete_reminder,
                "snooze_reminder": self._snooze_reminder,
                "batch": self._batch,
            }
            method = operation_map.get(operation)
            if method is None:
                self.logger.error(f"Unknown operation: {operation}")
                raise ValueError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "add_reminder, get_reminders, mark_completed, "
                    "update_reminder, delete_reminder, snooze_reminder, batch"
                )
            accepted = set(inspect.signature(method).parameters)
            filtered = {k: v for k, v in kwargs.items() if k in accepted}
            return method(**filtered)
        except Exception as e:
            self.logger.error(f"Error executing {operation} in reminder_tool: {e}")
            raise

    def _add_reminder(
        self,
        title: str,
        date: str,
        description: Optional[str] = None,
        contact_name: Optional[str] = None,
        notes: Optional[str] = None,
        category: str = "user",
    ) -> Dict[str, Any]:
        """
        Add a new reminder with optional contact linkage.
        
        Args:
            title: Brief title or subject of the reminder
            date: When the reminder should occur (can be natural language like
                "tomorrow" or "in 3 weeks")
            description: Detailed description of the reminder
            contact_name: Name of the contact to link with this reminder
            notes: Any additional information to store with the reminder
            category: Category of reminder ('user' or 'internal', default 'user')
            
        Returns:
            Dict containing the created reminder
            
        Raises:
            ValueError: If required fields are missing or date parsing fails
        """
        
        # Validate required parameters
        if not title:
            self.logger.error("Title is required for adding a reminder")
            raise ValueError("Title is required for adding a reminder")
            
        if not date:
            self.logger.error("Date is required for adding a reminder")
            raise ValueError("Date is required for adding a reminder")
            
        # Validate category
        if category not in ["user", "internal"]:
            self.logger.error(f"Invalid category '{category}'. Must be 'user' or 'internal'")
            raise ValueError(f"Invalid category '{category}'. Must be 'user' or 'internal'")
            
        # Parse the date from natural language
        try:
            reminder_date = self._parse_date(date)
        except Exception as e:
            self.logger.error(f"Failed to parse date '{date}': {str(e)}")
            raise ValueError(f"Failed to parse date '{date}': {str(e)}")
            
        # Generate a unique ID for the reminder
        reminder_id = f"rem_{uuid.uuid4().hex[:8]}"
        
        # Check if contact name exists in user's contacts
        contact_info = None
        contact_uuid = None
        if contact_name:
            contact_info = self._lookup_contact(contact_name)
            if contact_info:
                contact_uuid = contact_info["contact"]["id"]
            
        # Check for duplicate reminder (prevents retries/race conditions from creating duplicates)
        existing = self._check_duplicate_reminder(title, reminder_date)
        if existing:
            self.logger.info(f"Duplicate reminder detected, returning existing: {existing.get('id')}")
            response_reminder = self._format_reminder_for_display(existing)
            user_tz = get_user_preferences().timezone
            existing_date = parse_utc_time_string(existing.get('reminder_date'))
            local_reminder_time = convert_from_utc(existing_date, user_tz)
            formatted_local_time = format_datetime(local_reminder_time, 'date_time', include_timezone=True)
            return {
                "reminder": response_reminder,
                "message": f"Reminder already exists for {formatted_local_time}",
                "duplicate_detected": True
            }

        # Create the reminder object
        reminder = {
            "id": reminder_id,
            "encrypted__title": title,
            "encrypted__description": description,
            "reminder_date": format_utc_iso(reminder_date),
            "created_at": format_utc_iso(utc_now()),
            "updated_at": format_utc_iso(utc_now()),
            "completed": 0,
            "completed_at": None,
            "contact_uuid": contact_uuid,
            "encrypted__additional_notes": notes,
            "category": category
        }

        # Save reminder
        self._save_reminder(reminder)
            
        # Prepare response with contact details if linked
        response_reminder = self._format_reminder_for_display(reminder)
        
        # Convert reminder time to user's local timezone for the message
        user_tz = get_user_preferences().timezone
        local_reminder_time = convert_from_utc(reminder_date, user_tz)
        formatted_local_time = format_datetime(local_reminder_time, 'date_time', include_timezone=True)
        
        result = {
            "reminder": response_reminder,
            "message": f"Reminder added for {formatted_local_time}"
        }
        
        # Add contact details to response if found
        if contact_info:
            result["contact_found"] = True
            result["contact_info"] = contact_info.get("contact", {})
            result["message"] += f" linked to contact {contact_name}"
        elif contact_name:
            result["contact_found"] = False
            result["message"] += f". No contact record found for {contact_name}."
            
        return result

    def _get_reminders(
        self,
        date_filter: str,
        specific_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: str = "all"
    ) -> Dict[str, Any]:
        """
        Get reminders based on date criteria.
        
        Args:
            date_filter: Type of date query ("today", "tomorrow", "upcoming", "past",
                "all", "date", "range")
            specific_date: Specific date string (required if date_filter is "date")
            start_date: Start date string (required if date_filter is "range")
            end_date: End date string (required if date_filter is "range")
            category: Filter by category ("user", "internal", or "all", default "all")
            
        Returns:
            Dict containing list of reminders matching the criteria
            
        Raises:
            ValueError: If parameters are invalid or missing required fields
        """
        
        # Validate date_filter
        valid_date_filters = ["today", "tomorrow", "upcoming", "past", "all", "date", "range", "overdue"]
        if date_filter not in valid_date_filters:
            self.logger.error(f"Invalid date_filter: {date_filter}. Must be one of {valid_date_filters}")
            raise ValueError(f"Invalid date_filter: {date_filter}. Must be one of {valid_date_filters}")
            
        # Use user's local timezone for day boundaries, then convert to UTC
        user_tz = get_user_preferences().timezone
        today_local = convert_to_timezone(utc_now(), user_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        today = convert_to_timezone(today_local, "UTC")
        
        # Load reminders (include completed for "all" and "past")
        include_completed = date_filter in ["all", "past"]
        reminders = self._load_reminders(include_completed)
        
        # Validate category parameter
        valid_categories = ["user", "internal", "all"]
        if category not in valid_categories:
            self.logger.error(f"Invalid category: {category}. Must be one of {valid_categories}")
            raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
        
        # Filter by category first
        if category != "all":
            reminders = [r for r in reminders if r.get("category", "user") == category]
        
        # Filter based on date_filter
        filtered_reminders = []
        date_description = ""
        
        for reminder in reminders:
            try:
                # Parse stored UTC timestamp
                reminder_date = parse_utc_time_string(reminder["reminder_date"])
            except (ValueError, KeyError):
                continue  # Skip invalid reminders
            
            # Apply date filters based on date_filter
            if date_filter == "today":
                tomorrow = today + timedelta(days=1)
                if today <= reminder_date < tomorrow:
                    filtered_reminders.append(reminder)
                date_description = "today"
                
            elif date_filter == "tomorrow":
                tomorrow = today + timedelta(days=1)
                day_after = tomorrow + timedelta(days=1)
                if tomorrow <= reminder_date < day_after:
                    filtered_reminders.append(reminder)
                date_description = "tomorrow"
                
            elif date_filter == "upcoming":
                if reminder_date >= today and not reminder.get("completed", False):
                    filtered_reminders.append(reminder)
                date_description = "upcoming"
                
            elif date_filter == "overdue":
                # Overdue reminders: past due date and not completed
                if reminder_date < today and not reminder.get("completed", False):
                    filtered_reminders.append(reminder)
                date_description = "overdue"

            elif date_filter == "past":
                if reminder_date < today:
                    filtered_reminders.append(reminder)
                date_description = "past"

            elif date_filter == "all":
                # No filter needed
                filtered_reminders.append(reminder)
                date_description = "all"
                
            elif date_filter == "date":
                if not specific_date:
                    self.logger.error("specific_date is required when date_filter is 'date'")
                    raise ValueError("specific_date is required when date_filter is 'date'")
                
                try:
                    # Use our timezone-aware date parser
                    parsed_date = self._parse_date(specific_date)
                    next_date = parsed_date + timedelta(days=1)
                    if parsed_date <= reminder_date < next_date:
                        filtered_reminders.append(reminder)
                    # Format the date in UTC ISO
                    date_description = f"on {format_utc_iso(parsed_date)}"
                except Exception as e:
                    self.logger.error(f"Failed to parse specific_date '{specific_date}': {str(e)}")
                    raise ValueError(f"Failed to parse specific_date '{specific_date}': {str(e)}")
                
            elif date_filter == "range":
                if not start_date or not end_date:
                    self.logger.error("start_date and end_date are required when date_filter is 'range'")
                    raise ValueError("start_date and end_date are required when date_filter is 'range'")
                
                try:
                    # Use our timezone-aware date parser for both dates
                    parsed_start = self._parse_date(start_date)
                    # Include end date fully
                    parsed_end = self._parse_date(end_date) + timedelta(days=1)
                    if parsed_start <= reminder_date < parsed_end:
                        filtered_reminders.append(reminder)
                    # Format the dates in UTC ISO
                    date_description = f"from {format_utc_iso(parsed_start)} to {format_utc_iso(parsed_end - timedelta(days=1))}"
                except Exception as e:
                    self.logger.error(f"Failed to parse date range: {str(e)}")
                    raise ValueError(f"Failed to parse date range: {str(e)}")
        
        # Sort by reminder date
        filtered_reminders.sort(key=lambda r: r["reminder_date"])
        
        # Format for display with contact information
        reminder_list = [self._format_reminder_for_display(r) for r in filtered_reminders]
        
        return {
            "reminders": reminder_list,
            "count": len(reminder_list),
            "date_filter": date_filter,
            "message": f"Found {len(reminder_list)} reminder(s) {date_description}"
        }

    def _get_reminder_not_found_error(self, reminder_id: str) -> str:
        """
        Generate a helpful error message when a reminder is not found.

        Args:
            reminder_id: The ID that was not found

        Returns:
            Error message string with available reminder IDs
        """
        # Get active reminders
        active_reminders = self.db.select('reminders', 'completed = 0')

        error_msg = f"Reminder with ID '{reminder_id}' not found."

        if not active_reminders:
            error_msg += " No active reminders available."
        else:
            # Format available reminders
            available = []
            for r in active_reminders[:5]:  # Show first 5
                title = r.get('encrypted__title', 'Untitled')
                available.append(f"  - {r['id']}: {title}")

            error_msg += f"\n\nAvailable active reminders ({len(active_reminders)} total):\n"
            error_msg += "\n".join(available)

            if len(active_reminders) > 5:
                error_msg += f"\n  ... and {len(active_reminders) - 5} more"

        return error_msg

    def _mark_completed(self, reminder_id: str, resolution_note: str | None = None) -> Dict[str, Any]:
        """
        Mark a reminder as completed with an optional resolution note.

        Args:
            reminder_id: ID of the reminder to mark as completed
            resolution_note: Brief note describing the outcome at completion time

        Returns:
            Dict containing the updated reminder

        Raises:
            ValueError: If reminder_id is invalid or not found
        """

        # Find the reminder
        reminders = self.db.select('reminders', 'id = :id', {'id': reminder_id})

        if not reminders:
            self.logger.error(f"Reminder with ID '{reminder_id}' not found")
            return {
                "success": False,
                "error": "reminder_not_found",
                "message": f"Reminder '{reminder_id}' not found. Valid reminder IDs start with 'rem_' followed by 8 characters (e.g., 'rem_a1b2c3d4'). You can list all reminders to see valid IDs."
            }
        # Update reminder
        update_data = {
            'completed': 1,
            'completed_at': format_utc_iso(utc_now())
        }
        if resolution_note:
            update_data['encrypted__resolution_note'] = resolution_note
        
        rows_updated = self.db.update(
            'reminders',
            update_data,
            'id = :id',
            {'id': reminder_id}
        )

        # Verify the update actually happened
        if rows_updated == 0:
            self.logger.error(
                f"UPDATE affected 0 rows for reminder '{reminder_id}' - "
                f"SELECT found it but UPDATE didn't match. This indicates a persistence bug."
            )
            raise ValueError(f"Failed to update reminder '{reminder_id}' - no rows affected")

        # Get updated reminder and verify completion state
        updated_reminders = self.db.select('reminders', 'id = :id', {'id': reminder_id})
        if updated_reminders:
            updated_reminder = updated_reminders[0]

            # Verify the completed flag is actually set
            completed_value = updated_reminder.get('completed')
            if not completed_value or str(completed_value) == '0':
                self.logger.error(
                    f"PERSISTENCE BUG: Reminder '{reminder_id}' still shows completed={completed_value} "
                    f"after UPDATE with rows_updated={rows_updated}. "
                    f"DB path: {self.db.db_path}"
                )
                raise ValueError(
                    f"Reminder update failed to persist - completed={completed_value} after commit"
                )

            self.logger.info(f"Reminder '{reminder_id}' successfully marked complete (rows_updated={rows_updated})")
            return {
                "reminder": self._format_reminder_for_display(updated_reminder),
                "message": f"Reminder '{updated_reminder['encrypted__title']}' marked as completed"
            }
        
        raise ValueError("Failed to retrieve updated reminder")

    def _update_reminder(
        self,
        reminder_id: str,
        title: Optional[str] = None,
        date: Optional[str] = None,
        description: Optional[str] = None,
        contact_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing reminder.

        Args:
            reminder_id: ID of the reminder to update
            title: New title (optional)
            date: New date (optional)
            description: New description (optional)
            contact_name: New contact name to link (optional)
            notes: New additional notes (optional)
            
        Returns:
            Dict containing the updated reminder
            
        Raises:
            ValueError: If reminder_id is invalid or not found
        """

        # Find the reminder
        reminders = self.db.select('reminders', 'id = :id', {'id': reminder_id})

        if not reminders:
            self.logger.error(f"Reminder with ID '{reminder_id}' not found")
            return {
                "success": False,
                "error": "reminder_not_found",
                "message": f"Reminder '{reminder_id}' not found. Valid reminder IDs start with 'rem_' followed by 8 characters (e.g., 'rem_a1b2c3d4'). You can list all reminders to see valid IDs."
            }

        # Build update data
        update_data = {}
        changes = []

        if title is not None:
            update_data['encrypted__title'] = title
            changes.append("title")

        if date is not None:
            try:
                update_data['reminder_date'] = format_utc_iso(self._parse_date(date))
                changes.append("date")
            except Exception as e:
                self.logger.error(f"Failed to parse date '{date}': {str(e)}")
                raise ValueError(f"Failed to parse date '{date}': {str(e)}")

        if description is not None:
            update_data['encrypted__description'] = description
            changes.append("description")

        # Contact linkage update
        if contact_name is not None:
            contact_info = self._lookup_contact(contact_name)
            if contact_info:
                update_data['contact_uuid'] = contact_info["contact"]["id"]
                changes.append("contact_uuid")
            else:
                # Clear contact if name provided but not found
                update_data['contact_uuid'] = None
                changes.append("contact_uuid (cleared)")

        if notes is not None:
            update_data['encrypted__additional_notes'] = notes
            changes.append("notes")
        
        # Always bump updated_at to reflect modification
        update_data['updated_at'] = format_utc_iso(utc_now())
        if 'updated_at' not in changes:
            changes.append('updated_at')

        # Update if there are changes
        if update_data:
            rows_updated = self.db.update(
                'reminders',
                update_data,
                'id = :id',
                {'id': reminder_id}
            )
        
        # Get updated reminder
        updated_reminders = self.db.select('reminders', 'id = :id', {'id': reminder_id})
        if updated_reminders:
            updated_reminder = updated_reminders[0]
            return {
                "reminder": self._format_reminder_for_display(updated_reminder),
                "updated_fields": changes,
                "message": (
                    f"Reminder updated: {', '.join(changes)}" if changes
                    else "No changes made to reminder"
                )
            }
        
        raise ValueError("Failed to retrieve updated reminder")

    def _delete_reminder(self, reminder_id: str) -> Dict[str, Any]:
        """
        Delete a reminder.
        
        Args:
            reminder_id: ID of the reminder to delete
            
        Returns:
            Dict containing deletion confirmation
            
        Raises:
            ValueError: If reminder_id is invalid or not found
        """

        # Find the reminder first to get its title for confirmation
        reminders = self.db.select('reminders', 'id = :id', {'id': reminder_id})

        if not reminders:
            self.logger.error(f"Reminder with ID '{reminder_id}' not found")
            return {
                "success": False,
                "error": "reminder_not_found",
                "message": f"Reminder '{reminder_id}' not found. Valid reminder IDs start with 'rem_' followed by 8 characters (e.g., 'rem_a1b2c3d4'). You can list all reminders to see valid IDs."
            }

        reminder = reminders[0]

        # Delete from database
        rows_deleted = self.db.delete(
            'reminders',
            'id = :id',
            {'id': reminder_id}
        )
        
        return {
            "id": reminder_id,
            "message": f"Reminder '{reminder['encrypted__title']}' deleted successfully"
        }

    def _snooze_reminder(
        self,
        reminder_id: Optional[str] = None,
        reminder_ids: Optional[List[str]] = None,
        snooze_duration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Snooze one or more reminders by rescheduling forward from now.

        Args:
            reminder_id: Single reminder ID to snooze
            reminder_ids: List of reminder IDs to snooze (for bulk operations)
            snooze_duration: How far forward from now. Accepts '1 hour',
                '30 minutes', '2 days', '1 week'. Default: 1 hour.

        Returns:
            Dict containing the snoozed reminder(s) and new times

        Raises:
            ValueError: If no reminder IDs provided or reminders not found
        """
        # Normalize to list of IDs
        ids_to_snooze: List[str] = []
        if reminder_ids:
            ids_to_snooze.extend(reminder_ids)
        if reminder_id:
            ids_to_snooze.append(reminder_id)

        if not ids_to_snooze:
            raise ValueError("Either reminder_id or reminder_ids must be provided for snooze_reminder")

        # Remove duplicates while preserving order
        ids_to_snooze = list(dict.fromkeys(ids_to_snooze))

        # Parse snooze duration (default: 1 hour)
        snooze_delta = self._parse_snooze_duration(snooze_duration)
        duration_label = snooze_duration or "1 hour"

        snoozed = []
        not_found = []

        for rid in ids_to_snooze:
            reminders = self.db.select('reminders', 'id = :id', {'id': rid})

            if not reminders:
                not_found.append(rid)
                continue

            reminder = reminders[0]

            new_date = utc_now() + snooze_delta

            # Update the reminder
            update_data = {
                'reminder_date': format_utc_iso(new_date),
                'updated_at': format_utc_iso(utc_now())
            }

            self.db.update(
                'reminders',
                update_data,
                'id = :id',
                {'id': rid}
            )

            # Get updated reminder for response
            updated_reminders = self.db.select('reminders', 'id = :id', {'id': rid})
            if updated_reminders:
                snoozed.append(self._format_reminder_for_display(updated_reminders[0]))

        # Build response
        user_tz = get_user_preferences().timezone

        result: Dict[str, Any] = {
            "snoozed_count": len(snoozed),
            "snoozed_reminders": snoozed
        }

        if len(snoozed) == 1:
            new_time = convert_from_utc(parse_utc_time_string(snoozed[0]['reminder_date']), user_tz)
            result["message"] = f"Reminder snoozed for {duration_label} (new time: {format_datetime(new_time, 'date_time', include_timezone=True)})"
        elif snoozed:
            result["message"] = f"Snoozed {len(snoozed)} reminder(s) for {duration_label}"
        else:
            result["message"] = "No reminders were snoozed"

        if not_found:
            result["not_found"] = not_found
            result["message"] += f". {len(not_found)} reminder(s) not found: {', '.join(not_found)}"

        return result

    def _batch(self, batch_action: str, reminder_ids: List[str], resolution_note: str | None = None) -> Dict[str, Any]:
        """
        Perform batch action on multiple reminders.

        Args:
            batch_action: Action to perform - 'complete' or 'delete'
            reminder_ids: List of reminder IDs to process
            resolution_note: Brief note describing the outcome (applied to all completed reminders)

        Returns:
            Dict with succeeded/not_found/failed lists and summary message
        """
        if batch_action not in ("complete", "delete"):
            raise ValueError(f"batch_action must be 'complete' or 'delete', got '{batch_action}'")

        if not reminder_ids:
            raise ValueError("reminder_ids array cannot be empty")

        results: Dict[str, List] = {"succeeded": [], "not_found": [], "failed": []}

        for rid in reminder_ids:
            try:
                if batch_action == "complete":
                    result = self._mark_completed(rid, resolution_note=resolution_note)
                else:
                    result = self._delete_reminder(rid)
                if result.get("success") is False:
                    results["not_found"].append(rid)
                else:
                    results["succeeded"].append(rid)
            except Exception as e:
                results["failed"].append({"id": rid, "error": str(e)})

        total = len(reminder_ids)
        succeeded = len(results["succeeded"])
        verb = "Completed" if batch_action == "complete" else "Deleted"

        return {
            "success": succeeded == total,
            "batch_action": batch_action,
            "succeeded": results["succeeded"],
            "not_found": results["not_found"],
            "failed": results["failed"],
            "message": f"{verb} {succeeded} of {total} reminders"
        }

    def _format_reminder_for_display(self, reminder: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the reminder to a dictionary for display with contact information.
        
        All timestamps are returned in UTC ISO format for frontend parsing.
        
        Returns:
            Dict representation of the reminder with UTC timestamps
        """
        # Function to ensure UTC ISO format
        def format_dt(dt_str: Optional[str]) -> Optional[str]:
            if not dt_str:
                return None
            try:
                # Parse and ensure UTC
                dt = parse_utc_time_string(dt_str)
                return format_utc_iso(dt)
            except:
                return dt_str
        
        formatted = {
            "id": reminder["id"],
            "encrypted__title": reminder["encrypted__title"],
            "encrypted__description": reminder.get("encrypted__description"),
            "reminder_date": format_dt(reminder.get("reminder_date")),
            "created_at": format_dt(reminder.get("created_at")),
            "updated_at": format_dt(reminder.get("updated_at")),
            "completed": bool(reminder.get("completed", 0)),
            "completed_at": format_dt(reminder.get("completed_at")),
            "encrypted__additional_notes": reminder.get("encrypted__additional_notes"),
            "category": reminder.get("category", "user"),
            "encrypted__resolution_note": reminder.get("encrypted__resolution_note")
        }
        
        # Add contact information if linked
        if reminder.get("contact_uuid"):
            contact = self._get_contact_by_uuid(reminder["contact_uuid"])
            if contact:
                formatted["contact_encrypted__name"] = contact["encrypted__name"]
                formatted["contact_encrypted__email"] = contact.get("encrypted__email")
                formatted["contact_encrypted__phone"] = contact.get("encrypted__phone")
                formatted["contact_uuid"] = contact["id"]
        
        return formatted

    def _parse_snooze_duration(self, duration_str: str | None) -> timedelta:
        """
        Parse a snooze duration string into a timedelta.

        Accepts:
        - None or empty → 1 hour (default)
        - "N hour(s)", "N minute(s)", "N day(s)", "N week(s)"
        - Plain number → interpreted as hours

        Returns:
            timedelta representing the snooze duration

        Raises:
            ValueError: If the duration string cannot be parsed
        """
        if not duration_str or not duration_str.strip():
            return timedelta(hours=1)

        import re
        ds = duration_str.strip().lower()

        # Match "N unit" pattern
        m = re.match(r"^(\d+)\s*(hour|hours|minute|minutes|min|mins|day|days|week|weeks)$", ds)
        if m:
            qty = int(m.group(1))
            unit = m.group(2)
            if unit.startswith("hour"):
                return timedelta(hours=qty)
            elif unit.startswith("min"):
                return timedelta(minutes=qty)
            elif unit.startswith("day"):
                return timedelta(days=qty)
            elif unit.startswith("week"):
                return timedelta(weeks=qty)

        # Plain number → hours
        try:
            return timedelta(hours=int(ds))
        except ValueError:
            pass

        raise ValueError(
            f"Cannot parse snooze duration: '{duration_str}'. "
            f"Use formats like '30 minutes', '2 hours', '1 day'."
        )

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse a date string, supporting ISO 8601, date-only, time-only (with today's date),
        and a minimal set of natural-language phrases:
        - today, tomorrow, yesterday
        - in N day(s)/week(s)/month(s)/hour(s)/minute(s)
        - next day/week/month

        Returns a timezone-aware datetime object in UTC.
        """
        user_tz = get_user_preferences().timezone
        ds = (date_str or "").strip().lower()

        # Handle simple natural language
        try:
            base_local = convert_to_timezone(utc_now(), user_tz)
            if ds in ("today",):
                return ensure_utc(base_local)
            if ds in ("tomorrow",):
                return ensure_utc(base_local + timedelta(days=1))
            if ds in ("yesterday",):
                return ensure_utc(base_local - timedelta(days=1))

            import re
            m = re.match(r"^in\s+(\d+)\s+(day|days|week|weeks|month|months|hour|hours|minute|minutes)$", ds)
            if m:
                qty = int(m.group(1))
                unit = m.group(2)
                delta = None
                if unit.startswith("day"):
                    delta = relativedelta(days=qty)
                elif unit.startswith("week"):
                    delta = relativedelta(weeks=qty)
                elif unit.startswith("month"):
                    delta = relativedelta(months=qty)
                elif unit.startswith("hour"):
                    delta = relativedelta(hours=qty)
                elif unit.startswith("minute"):
                    delta = relativedelta(minutes=qty)
                if delta:
                    return ensure_utc(base_local + delta)

            if ds in ("next day",):
                return ensure_utc(base_local + relativedelta(days=1))
            if ds in ("next week",):
                return ensure_utc(base_local + relativedelta(weeks=1))
            if ds in ("next month",):
                return ensure_utc(base_local + relativedelta(months=1))
        except Exception:
            # Fall through to structured parsing
            pass

        # Try structured parsing with timezone_utils (handles ISO, date-only, time-only)
        try:
            dt = parse_time_string(date_str, tz_name=user_tz)
            return ensure_utc(dt)
        except Exception:
            # Final attempt: strict ISO parsing
            try:
                return parse_utc_time_string(date_str)
            except Exception:
                raise ValueError(
                    f"Invalid date format: '{date_str}'. Please use ISO 8601 (YYYY-MM-DDTHH:MM:SS) "
                    f"or supported phrases like 'tomorrow', 'in 3 weeks'."
                )

    def _lookup_contact(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Lookup a contact by name in the user's contacts.
        
        Args:
            name: Contact name to search for
            
        Returns:
            Dict with contact info or None if not found
        """
        try:
            contacts = self._load_contacts()
            
            # Search for contact by name (case-insensitive)
            name_lower = name.lower()
            for contact in contacts:
                contact_name = (contact.get("encrypted__name") or "").lower()
                if name_lower in contact_name or contact_name in name_lower:
                    return {
                        "contact": contact,
                        "matched_field": "name"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Contact lookup failed for {name}: {e}")
            return None
