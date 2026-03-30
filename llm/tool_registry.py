"""
llm/tool_registry.py — Registers and dispatches tool calls from the LLM

Python 3.12: no changes needed here, but added __future__ annotations
and cleaned up Any import.
"""

from __future__ import annotations

import logging

from config import Config
from memory.store import MemoryStore

log = logging.getLogger("atlas.tools")


class ToolRegistry:
    def __init__(self, cfg: Config, memory: MemoryStore):
        self.cfg = cfg
        self.memory = memory
        self._tools: dict = {}

    def _load(self, name: str):
        """Lazy-load tool instances on first use."""
        if name in self._tools:
            return self._tools[name]

        if name == "web_search":
            from tools.web_search import WebSearchTool
            self._tools[name] = WebSearchTool(self.cfg)

        elif name == "get_datetime":
            from tools.datetime_tool import DateTimeTool
            self._tools[name] = DateTimeTool()

        elif name in ("set_reminder", "list_reminders", "cancel_reminder"):
            from tools.reminders import ReminderTool
            tool = ReminderTool(self.cfg, self.memory)
            self._tools["set_reminder"]   = tool
            self._tools["list_reminders"] = tool
            self._tools["cancel_reminder"] = tool

        return self._tools.get(name)

    def call(self, tool_name: str, args: dict) -> str:
        tool = self._load(tool_name)
        if tool is None:
            return f"Unknown tool: '{tool_name}'"

        method = getattr(tool, tool_name, None)
        if method is None:
            return f"Tool '{tool_name}' has no method '{tool_name}'"

        try:
            result = method(**args)
            return str(result)
        except TypeError as exc:
            return f"Tool '{tool_name}' called with wrong arguments: {exc}"
        except Exception as exc:
            log.error(f"Tool '{tool_name}' error: {exc}", exc_info=True)
            return f"Tool error: {exc}"
