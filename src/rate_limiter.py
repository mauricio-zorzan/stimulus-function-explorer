"""
Rate limiting utilities for OpenAI API calls.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional
import streamlit as st


class RateLimiter:
    """Simple rate limiter using Streamlit session state."""

    def __init__(
        self,
        max_calls_per_minute: int = 10,
        max_calls_per_hour: int = 100,
        max_calls_per_day: int = 500,
    ):
        """Initialize rate limiter.

        Args:
            max_calls_per_minute: Maximum API calls per minute per session
            max_calls_per_hour: Maximum API calls per hour per session
            max_calls_per_day: Maximum API calls per day per session
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_hour = max_calls_per_hour
        self.max_calls_per_day = max_calls_per_day

        # Initialize session state
        if "api_call_history" not in st.session_state:
            st.session_state.api_call_history = []

    def _clean_old_calls(self):
        """Remove API calls older than 24 hours."""
        now = datetime.now()
        cutoff = now - timedelta(days=1)

        st.session_state.api_call_history = [
            call_time
            for call_time in st.session_state.api_call_history
            if call_time > cutoff
        ]

    def _get_call_counts(self) -> Dict[str, int]:
        """Get call counts for different time windows."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        calls = st.session_state.api_call_history

        return {
            "minute": sum(1 for call in calls if call > minute_ago),
            "hour": sum(1 for call in calls if call > hour_ago),
            "day": sum(1 for call in calls if call > day_ago),
        }

    def can_make_call(self) -> tuple[bool, Optional[str]]:
        """Check if a new API call can be made.

        Returns:
            Tuple of (can_call, reason_if_not)
        """
        self._clean_old_calls()
        counts = self._get_call_counts()

        # Check minute limit
        if counts["minute"] >= self.max_calls_per_minute:
            return False, f"Rate limit: {self.max_calls_per_minute} calls/minute"

        # Check hour limit
        if counts["hour"] >= self.max_calls_per_hour:
            return False, f"Rate limit: {self.max_calls_per_hour} calls/hour"

        # Check day limit
        if counts["day"] >= self.max_calls_per_day:
            return False, f"Rate limit: {self.max_calls_per_day} calls/day"

        return True, None

    def record_call(self):
        """Record a new API call."""
        st.session_state.api_call_history.append(datetime.now())

    def get_remaining_calls(self) -> Dict[str, int]:
        """Get remaining calls for different time windows."""
        counts = self._get_call_counts()

        return {
            "minute": max(0, self.max_calls_per_minute - counts["minute"]),
            "hour": max(0, self.max_calls_per_hour - counts["hour"]),
            "day": max(0, self.max_calls_per_day - counts["day"]),
        }

    def get_usage_stats(self) -> str:
        """Get human-readable usage statistics."""
        counts = self._get_call_counts()
        remaining = self.get_remaining_calls()

        return f"""**API Usage:**
- Last minute: {counts["minute"]}/{self.max_calls_per_minute} ({remaining["minute"]} remaining)
- Last hour: {counts["hour"]}/{self.max_calls_per_hour} ({remaining["hour"]} remaining)
- Last 24 hours: {counts["day"]}/{self.max_calls_per_day} ({remaining["day"]} remaining)
"""


