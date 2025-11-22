"""Time manager trinket for current date/time injection."""
import logging
from typing import Dict, Any

from utils.timezone_utils import (
    utc_now, convert_from_utc, format_datetime
)
from utils.user_context import get_user_timezone
from .base import EventAwareTrinket

logger = logging.getLogger(__name__)


class TimeManager(EventAwareTrinket):
    """
    Manages current date/time information for the system prompt.
    
    Always generates fresh timestamp when requested.
    """
    
    def _get_variable_name(self) -> str:
        """Time manager publishes to 'datetime_section'."""
        return "datetime_section"
    
    def generate_content(self, context: Dict[str, Any]) -> str:
        """
        Generate current date/time content.
        
        Args:
            context: Update context (unused for time manager)
            
        Returns:
            Formatted date/time section
        """
        current_time = utc_now()
        user_tz = get_user_timezone()
        local_time = convert_from_utc(current_time, user_tz)
        
        # Format with day of week and prettier display
        day_of_week = local_time.strftime('%A').upper()
        date_part = local_time.strftime('%B %d, %Y').upper()
        time_part = local_time.strftime('%-I:%M %p').upper()
        timezone_name = local_time.strftime('%Z')

        datetime_info = f"TODAY IS {day_of_week}, {date_part} AT {time_part} {timezone_name}."
        
        logger.debug(f"Generated datetime information for {day_of_week}, {date_part} at {time_part}")
        return datetime_info