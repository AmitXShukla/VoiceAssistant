"""
tools/datetime_tool.py — Date and time tool

Python 3.12 fix:
- %-I format (Linux strip-zero) replaced with explicit int() strip
  to work on both Linux and Windows.
"""

from __future__ import annotations

from datetime import datetime


class DateTimeTool:
    def get_datetime(self) -> str:
        now = datetime.now()
        hour = now.strftime("%I").lstrip("0") or "12"   # "%-I" not portable
        return now.strftime(f"Today is %A, %B %d %Y. The time is {hour}:%M %p.")
