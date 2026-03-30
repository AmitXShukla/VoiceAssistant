"""
tools/web_search.py — Web search via DuckDuckGo (no API key needed)

Python 3.12: no breaking changes, but added __future__ annotations
and tightened exception handling.
"""

from __future__ import annotations

import logging
from config import Config

log = logging.getLogger("atlas.tools.search")


class WebSearchTool:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def web_search(self, query: str) -> str:
        log.info(f"🔍 Searching: {query!r}")
        return self._serpapi(query) if self.cfg.serpapi_key else self._ddg(query)

    def _ddg(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query,
                    max_results=self.cfg.search_max_results,
                    safesearch="moderate",
                ))
            if not results:
                return "No results found for that query."
            parts = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body", "")
                if body:
                    parts.append(f"{title}: {body}")
            return " | ".join(parts) or "No usable results found."

        except ImportError:
            return "Web search unavailable. Run: pip install duckduckgo-search"
        except Exception as exc:
            log.error(f"DuckDuckGo error: {exc}")
            return f"Search failed: {exc}"

    def _serpapi(self, query: str) -> str:
        try:
            import requests
            resp = requests.get(
                "https://serpapi.com/search",
                params={"q": query, "api_key": self.cfg.serpapi_key,
                        "num": self.cfg.search_max_results},
                timeout=10,
            )
            data = resp.json()
            snippets = [
                r.get("snippet", "")
                for r in data.get("organic_results", [])
                if r.get("snippet")
            ]
            return " | ".join(snippets) or "No results."
        except Exception as exc:
            log.error(f"SerpAPI error: {exc}")
            return self._ddg(query)
