"""
SQLite-based reminder tool for managing scheduled reminders with complete user isolation.

This tool stores all reminder data in user-specific SQLite databases, ensuring perfect
multi-tenant isolation through automatic user-scoped queries. Integrates with
the user's contacts by storing contact UUID references.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel, Field

from tools.repo import Tool
from tools.registry import registry
from utils.timezone_utils import (
    validate_timezone, convert_to_timezone,
    format_datetime, parse_time_string, utc_now, ensure_utc, format_utc_iso,
    parse_utc_time_string, convert_from_utc, get_timezone_instance, convert_to_utc
)
from utils.user_context import get_user_timezone


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
    simple_description = "Create and manage scheduled reminders. Link reminders to contacts. Query by date (today, tomorrow, upcoming, overdue). Supports both user-facing and internal (MIRA's own) reminders."

    anthropic_schema = {
        "name": "reminder_tool",
        "description": "Manages scheduled reminders with contact information integration. Use this tool when the user wants to create, view, or manage reminders about tasks, follow-ups, or appointments. You can also create internal reminders for yourself using category='internal' to track things you need to remember or follow up on later.",
        "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add_reminder", "get_reminders", "mark_completed", "update_reminder", "delete_reminder"],
                        "description": "The operation to perform"
                    },
                    "title": {
                        "type": "string",
                        "description": "Brief title or subject of the reminder (required for add_reminder, optional for update_reminder)"
                    },
                    "date": {
                        "type": "string",
                        "description": "When the reminder should occur in ISO 8601 format (YYYY-MM-DDTHH:MM:SS). Example: 2025-09-14T14:30:00. Time zone will be interpreted as user's local time. (required for add_reminder, optional for update_reminder)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the reminder (optional)"
                    },
                    "contact_name": {
                        "type": "string",
                        "description": "Name of the contact to link with this reminder (optional - will search existing contacts)"
                    },
                    "category": {
                        "type": "string",
                        "enum": ["user", "internal"],
                        "description": "Category of reminder: 'user' for user-facing reminders, 'internal' for MIRA's internal reminders (default: 'user')"
                    },
                    "additional_notes": {
                        "type": "string",
                        "description": "Any additional information to store with the reminder (optional)"
                    },
                    "reminder_id": {
                        "type": "string",
                        "description": "ID of the reminder to mark completed, update, or delete (required for mark_completed, update_reminder, delete_reminder)"
                    },
                    "date_type": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "upcoming", "past", "all", "date", "range", "overdue"],
                        "description": "Type of date query for get_reminders operation (required for get_reminders)"
                    },
                    "specific_date": {
                        "type": "string",
                        "description": "Specific date in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) (required when date_type is 'date')"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) (required when date_type is 'range')"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) (required when date_type is 'range')"
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
            category TEXT DEFAULT 'user'
        """
        self.db.create_table('reminders', schema)
        
        # Create indexes for faster queries
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_date ON reminders(reminder_date)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_completed ON reminders(completed)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_contact ON reminders(contact_uuid)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_reminders_category ON reminders(category)")

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
           - Optional: description, contact_name, additional_notes
           - Returns: Dict with created reminder

        2. get_reminders: Retrieve reminders
           - Required: date_type ("today", "tomorrow", "upcoming", "past", "all", "date" or
            "range")
           - If date_type is "date", requires specific_date parameter
           - If date_type is "range", requires start_date and end_date parameters
           - Returns: Dict with list of reminders

        3. mark_completed: Mark a reminder as completed
           - Required: reminder_id
           - Returns: Dict with updated reminder

        4. update_reminder: Update an existing reminder
           - Required: reminder_id
           - Optional: Any fields to update (title, description, date, contact_name)
           - Returns: Dict with updated reminder

        5. delete_reminder: Delete a reminder
           - Required: reminder_id
           - Returns: Dict with deletion confirmation
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
            
            # Route to the appropriate operation
            if operation == "add_reminder":
                return self._add_reminder(**kwargs)
            elif operation == "get_reminders":
                return self._get_reminders(**kwargs)
            elif operation == "mark_completed":
                return self._mark_completed(**kwargs)
            elif operation == "update_reminder":
                return self._update_reminder(**kwargs)
            elif operation == "delete_reminder":
                return self._delete_reminder(**kwargs)
            else:
                self.logger.error(f"Unknown operation: {operation}")
                raise ValueError(
                    f"Unknown operation: {operation}. Valid operations are: "
                    "add_reminder, get_reminders, mark_completed, "
                    "update_reminder, delete_reminder"
                )
        except Exception as e:
            self.logger.error(f"Error executing {operation} in reminder_tool: {e}")
            raise

    def _add_reminder(
        self,
        title: str,
        date: str,
        description: Optional[str] = None,
        contact_name: Optional[str] = None,
        additional_notes: Optional[str] = None,
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
            additional_notes: Any additional information to store with the reminder
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
            "encrypted__additional_notes": additional_notes,
            "category": category
        }
        
        # Save reminder
        self._save_reminder(reminder)
            
        # Prepare response with contact details if linked
        response_reminder = self._format_reminder_for_display(reminder)
        
        # Convert reminder time to user's local timezone for the message
        user_tz = get_user_timezone()
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
        date_type: str,
        specific_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: str = "all"
    ) -> Dict[str, Any]:
        """
        Get reminders based on date criteria.
        
        Args:
            date_type: Type of date query ("today", "tomorrow", "upcoming", "past",
                "all", "date", "range")
            specific_date: Specific date string (required if date_type is "date")
            start_date: Start date string (required if date_type is "range")
            end_date: End date string (required if date_type is "range")
            category: Filter by category ("user", "internal", or "all", default "all")
            
        Returns:
            Dict containing list of reminders matching the criteria
            
        Raises:
            ValueError: If parameters are invalid or missing required fields
        """
        
        # Validate date_type
        valid_date_types = ["today", "tomorrow", "upcoming", "past", "all", "date", "range", "overdue"]
        if date_type not in valid_date_types:
            self.logger.error(f"Invalid date_type: {date_type}. Must be one of {valid_date_types}")
            raise ValueError(f"Invalid date_type: {date_type}. Must be one of {valid_date_types}")
            
        # Use user's local timezone for day boundaries, then convert to UTC
        user_tz = get_user_timezone()
        today_local = convert_to_timezone(utc_now(), user_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        today = convert_to_timezone(today_local, "UTC")
        
        # Load reminders (include completed for "all" and "past")
        include_completed = date_type in ["all", "past"]
        reminders = self._load_reminders(include_completed)
        
        # Validate category parameter
        valid_categories = ["user", "internal", "all"]
        if category not in valid_categories:
            self.logger.error(f"Invalid category: {category}. Must be one of {valid_categories}")
            raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
        
        # Filter by category first
        if category != "all":
            reminders = [r for r in reminders if r.get("category", "user") == category]
        
        # Filter based on date_type
        filtered_reminders = []
        date_description = ""
        
        for reminder in reminders:
            try:
                # Parse stored UTC timestamp
                reminder_date = parse_utc_time_string(reminder["reminder_date"])
            except (ValueError, KeyError):
                continue  # Skip invalid reminders
            
            # Apply date filters based on date_type
            if date_type == "today":
                tomorrow = today + timedelta(days=1)
                if today <= reminder_date < tomorrow:
                    filtered_reminders.append(reminder)
                date_description = "today"
                
            elif date_type == "tomorrow":
                tomorrow = today + timedelta(days=1)
                day_after = tomorrow + timedelta(days=1)
                if tomorrow <= reminder_date < day_after:
                    filtered_reminders.append(reminder)
                date_description = "tomorrow"
                
            elif date_type == "upcoming":
                if reminder_date >= today and not reminder.get("completed", False):
                    filtered_reminders.append(reminder)
                date_description = "upcoming"
                
            elif date_type == "overdue":
                # Overdue reminders: past due date and not completed
                if reminder_date < today and not reminder.get("completed", False):
                    filtered_reminders.append(reminder)
                date_description = "overdue"

            elif date_type == "past":
                if reminder_date < today:
                    filtered_reminders.append(reminder)
                date_description = "past"

            elif date_type == "all":
                # No filter needed
                filtered_reminders.append(reminder)
                date_description = "all"
                
            elif date_type == "date":
                if not specific_date:
                    self.logger.error("specific_date is required when date_type is 'date'")
                    raise ValueError("specific_date is required when date_type is 'date'")
                
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
                
            elif date_type == "range":
                if not start_date or not end_date:
                    self.logger.error("start_date and end_date are required when date_type is 'range'")
                    raise ValueError("start_date and end_date are required when date_type is 'range'")
                
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
            "date_type": date_type,
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

    def _mark_completed(self, reminder_id: str) -> Dict[str, Any]:
        """
        Mark a reminder as completed.

        Args:
            reminder_id: ID of the reminder to mark as completed

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
        additional_notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update an existing reminder.
        
        Args:
            reminder_id: ID of the reminder to update
            title: New title (optional)
            date: New date (optional)
            description: New description (optional)
            contact_name: New contact name to link (optional)
            additional_notes: New additional notes (optional)
            
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

        if additional_notes is not None:
            update_data['encrypted__additional_notes'] = additional_notes
            changes.append("additional_notes")
        
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
            "category": reminder.get("category", "user")
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

    def _parse_date(self, date_str: str) -> datetime:
        """
        Parse a date string, supporting ISO 8601, date-only, time-only (with today's date),
        and a minimal set of natural-language phrases:
        - today, tomorrow, yesterday
        - in N day(s)/week(s)/month(s)/hour(s)/minute(s)
        - next day/week/month

        Returns a timezone-aware datetime object in UTC.
        """
        user_tz = get_user_timezone()
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
            # Final attempt: strict ISO then localize if naive
            try:
                parsed_dt = datetime.fromisoformat(date_str)
                if parsed_dt.tzinfo is None:
                    tz_instance = get_timezone_instance(user_tz)
                    parsed_dt = parsed_dt.replace(tzinfo=tz_instance)
                return ensure_utc(parsed_dt)
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
                contact_name = contact.get("name", "").lower()
                if name_lower in contact_name or contact_name in name_lower:
                    return {
                        "contact": contact,
                        "matched_field": "name"
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Contact lookup failed for {name}: {e}")
            return None
