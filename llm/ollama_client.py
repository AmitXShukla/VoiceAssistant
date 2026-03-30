"""
llm/ollama_client.py — Streaming Ollama LLM client with tool calling

Python 3.12 fixes:
- iter_lines() can return bytes in some requests versions — decode explicitly
- tuple[list[str], str] return annotation style (valid in 3.9+, kept as-is)
- Optional[X] → X | None style via __future__ annotations
- Removed unused _SENTENCE_END compiled regex (the split regex is inline)
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Callable

import requests

from config import Config
from llm.tool_registry import ToolRegistry
from memory.store import MemoryStore

log = logging.getLogger("atlas.llm")


class OllamaClient:
    def __init__(self, cfg: Config, memory: MemoryStore):
        self.cfg = cfg
        self.memory = memory
        self.tools = ToolRegistry(cfg, memory)
        self._lock = threading.Lock()

    def stream_response(
        self,
        user_text: str,
        sentence_callback: Callable[[str], None],
        stop_event: threading.Event | None = None,
    ) -> str:
        """
        Stream the LLM response. Calls sentence_callback for each complete
        sentence as it arrives (enables streaming TTS).
        Returns the full response text.
        """
        self.memory.add_message("user", user_text)

        messages = [
            {"role": "system", "content": self.cfg.ollama_system_prompt},
            *self.memory.get_recent_messages(self.cfg.context_window),
        ]

        full_response = self._stream_ollama(messages, sentence_callback, stop_event)

        # Check for tool call
        tool_result = self._maybe_execute_tool(full_response)
        if tool_result is not None:
            log.info(f"🔧 Tool result: {tool_result[:120]}")
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": (
                    f"Tool result: {tool_result}\n"
                    "Now give a natural spoken response in 1-3 sentences. "
                    "No lists or markdown."
                ),
            })
            full_response = self._stream_ollama(messages, sentence_callback, stop_event)

        self.memory.add_message("assistant", full_response)
        return full_response

    def _stream_ollama(
        self,
        messages: list,
        sentence_callback: Callable[[str], None],
        stop_event: threading.Event | None,
    ) -> str:
        url = f"{self.cfg.ollama_host}/api/chat"
        payload = {
            "model": self.cfg.ollama_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.cfg.ollama_temperature,
                "num_predict": self.cfg.ollama_max_tokens,
            },
        }

        full_text = ""
        buffer = ""

        for attempt in range(3):
            try:
                with requests.post(url, json=payload, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        if stop_event and stop_event.is_set():
                            log.debug("LLM stream interrupted.")
                            return full_text

                        if not raw_line:
                            continue

                        # iter_lines() may return bytes or str depending on requests version
                        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line

                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        delta = chunk.get("message", {}).get("content", "")
                        full_text += delta
                        buffer += delta

                        # Emit complete sentences immediately (for streaming TTS)
                        sentences, buffer = self._extract_sentences(buffer)
                        for s in sentences:
                            if s.strip():
                                sentence_callback(s.strip())

                        if chunk.get("done"):
                            if buffer.strip():
                                sentence_callback(buffer.strip())
                            buffer = ""
                            break

                return full_text

            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise RuntimeError(
                        f"Cannot connect to Ollama at {self.cfg.ollama_host}. "
                        "Is Ollama running?  Run: ollama serve"
                    )
                log.warning(f"Ollama connection error — retry {attempt + 1}/3…")
                time.sleep(1.5)

            except requests.exceptions.HTTPError as exc:
                raise RuntimeError(f"Ollama HTTP error: {exc}") from exc

        return full_text

    def _extract_sentences(self, text: str) -> tuple[list[str], str]:
        """
        Split text into complete sentences + a remaining (incomplete) buffer.
        Splits on sentence-ending punctuation followed by whitespace.
        """
        parts = re.split(r'(?<=[.!?])\s+', text)
        if len(parts) <= 1:
            return [], text
        return [s for s in parts[:-1] if s.strip()], parts[-1]

    def _maybe_execute_tool(self, response: str) -> str | None:
        """Detect and execute a JSON tool call in the LLM response."""
        response = response.strip()
        if not (response.startswith("{") and '"tool"' in response):
            return None
        try:
            call = json.loads(response)
            tool_name = call.get("tool")
            args = call.get("args", {})
            if tool_name:
                log.info(f"🔧 Tool call: {tool_name}({args})")
                return self.tools.call(tool_name, args)
        except json.JSONDecodeError:
            pass
        return None
