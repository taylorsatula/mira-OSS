"""Reminder manager trinket for system prompt injection."""
import logging
import datetime
from typing import Dict, Any, List

from utils.timezone_utils import convert_from_utc, format_datetime
from utils.user_context import get_user_timezone
from .base import EventAwareTrinket

logger = logging.getLogger(__name__)


class ReminderManager(EventAwareTrinket):
    """
    Manages reminder information for the system prompt.
    
    Fetches active reminders from the reminder tool when requested.
    """
    
    def _get_variable_name(self) -> str:
        """Reminder manager publishes to 'active_reminders'."""
        return "active_reminders"
    
    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate reminder content by fetching from reminder tool.

        Args:
            context: Update context (unused for reminder manager)

        Returns:
            Formatted reminders section or empty string if no reminders

        Raises:
            Exception: If ReminderTool operations fail (infrastructure/filesystem issues)
        """
        from tools.implementations.reminder_tool import ReminderTool
        reminder_tool = ReminderTool()

        # Let infrastructure failures propagate
        overdue_result = reminder_tool.run(
            operation="get_reminders",
            date_type="overdue",
            category="user"
        )

        today_result = reminder_tool.run(
            operation="get_reminders",
            date_type="today",
            category="user"
        )

        # Get internal reminders separately
        internal_overdue = reminder_tool.run(
            operation="get_reminders",
            date_type="overdue",
            category="internal"
        )

        internal_today = reminder_tool.run(
            operation="get_reminders",
            date_type="today",
            category="internal"
        )

        # Collect reminders, keeping overdue and today separate
        user_overdue = self._collect_reminders([overdue_result])
        user_today = self._collect_reminders([today_result])
        internal_overdue_list = self._collect_reminders([internal_overdue])
        internal_today_list = self._collect_reminders([internal_today])

        if not user_overdue and not user_today and not internal_overdue_list and not internal_today_list:
            logger.debug("No active reminders")
            return ""  # Legitimately empty - user has no reminders set

        # Format reminder content with separate overdue/today sections
        reminder_info = self._format_reminders(
            user_overdue, user_today,
            internal_overdue_list, internal_today_list
        )
        total_user = len(user_overdue) + len(user_today)
        total_internal = len(internal_overdue_list) + len(internal_today_list)
        logger.debug(f"Generated reminder info with {total_user} user ({len(user_overdue)} overdue) and {total_internal} internal ({len(internal_overdue_list)} overdue) reminders")
        return reminder_info
    
    def _collect_reminders(self, results: List[Dict]) -> List[Dict]:
        """Collect non-completed reminders from multiple results."""
        reminders = []
        for result in results:
            if result.get("count", 0) > 0:
                for reminder in result.get("reminders", []):
                    if not reminder.get('completed', False):
                        reminders.append(reminder)
        return reminders
    
    def _format_reminders(
        self,
        user_overdue: List[Dict],
        user_today: List[Dict],
        internal_overdue: List[Dict],
        internal_today: List[Dict]
    ) -> str:
        """
        Format reminders with urgent overdue section and relative time.

        Overdue reminders are displayed with visual emphasis (emojis, bold, caps)
        and relative time ("DUE 3 days ago"). Today's reminders use hybrid format
        with both relative and absolute time for context.
        """
        from utils.timezone_utils import format_relative_time, utc_now

        user_tz = get_user_timezone()
        reminder_info = "# Active Reminders\n\n"

        # OVERDUE USER REMINDERS - LOUD AND URGENT
        if user_overdue:
            reminder_info += "⚠️⚠️⚠️ OVERDUE REMINDERS - REQUIRE IMMEDIATE ATTENTION ⚠️⚠️⚠️\n"
            reminder_info += "The following reminders are PAST DUE and MUST be addressed NOW:\n\n"
            for reminder in user_overdue:
                date_obj = datetime.datetime.fromisoformat(reminder["reminder_date"])
                relative_time = format_relative_time(date_obj)
                reminder_info += f"* **{reminder['encrypted__title']}** - DUE {relative_time}\n"
                if reminder.get('encrypted__description'):
                    reminder_info += f"  Details: {reminder['encrypted__description']}\n"
            reminder_info += "\nYOU MUST notify the user about these overdue reminders IMMEDIATELY, even if discussing an unrelated topic. Keep them front-of-mind until resolved.\n"

        # Visual separator between sections
        if user_overdue and user_today:
            reminder_info += "\n" + ("━" * 70) + "\n\n"

        # TODAY'S UPCOMING USER REMINDERS
        if user_today:
            reminder_info += "TODAY'S UPCOMING REMINDERS:\n"
            reminder_info += "The user has the following reminders for today:\n\n"
            now = utc_now()
            for reminder in user_today:
                date_obj = datetime.datetime.fromisoformat(reminder["reminder_date"])
                local_time = convert_from_utc(date_obj, user_tz)

                # Hybrid format: relative + absolute time for context
                if date_obj > now:
                    relative_time = format_relative_time(date_obj)
                    time_str = format_datetime(local_time, 'short')  # HH:MM format
                    reminder_info += f"* {reminder['encrypted__title']} - {relative_time} ({time_str})\n"
                else:
                    # Already passed but still today (edge case)
                    formatted_time = format_datetime(local_time, 'date_time')
                    reminder_info += f"* {reminder['encrypted__title']} - {formatted_time}\n"

                if reminder.get('encrypted__description'):
                    reminder_info += f"  Details: {reminder['encrypted__description']}\n"
            reminder_info += "\nPlease remind the user about these during the continuum when appropriate.\n"

        # INTERNAL REMINDERS SECTION
        if internal_overdue or internal_today:
            if user_overdue or user_today:
                reminder_info += "\n" + ("─" * 70) + "\n\n"

            reminder_info += "## Internal Reminders (MIRA's notes)\n"

            # Overdue internal reminders
            if internal_overdue:
                reminder_info += "\n⚠️ OVERDUE INTERNAL REMINDERS:\n\n"
                for reminder in internal_overdue:
                    date_obj = datetime.datetime.fromisoformat(reminder["reminder_date"])
                    relative_time = format_relative_time(date_obj)
                    reminder_info += f"* **{reminder['encrypted__title']}** - DUE {relative_time}\n"
                    if reminder.get('encrypted__description'):
                        reminder_info += f"  Details: {reminder['encrypted__description']}\n"

            # Today's internal reminders
            if internal_today:
                if internal_overdue:
                    reminder_info += "\n"
                reminder_info += "Today's internal reminders:\n\n"
                now = utc_now()
                for reminder in internal_today:
                    date_obj = datetime.datetime.fromisoformat(reminder["reminder_date"])
                    local_time = convert_from_utc(date_obj, user_tz)

                    if date_obj > now:
                        relative_time = format_relative_time(date_obj)
                        time_str = format_datetime(local_time, 'short')
                        reminder_info += f"* {reminder['encrypted__title']} - {relative_time} ({time_str})\n"
                    else:
                        formatted_time = format_datetime(local_time, 'date_time')
                        reminder_info += f"* {reminder['encrypted__title']} - {formatted_time}\n"

                    if reminder.get('encrypted__description'):
                        reminder_info += f"  Details: {reminder['encrypted__description']}\n"

        return reminder_info.rstrip()