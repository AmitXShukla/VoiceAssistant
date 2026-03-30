"""
ui/server.py — FastAPI web dashboard for Atlas

Python 3.12 fixes:
- asyncio.get_running_loop() instead of deprecated get_event_loop()
- Thread-safe broadcast queue instead of direct coroutine scheduling
- Removed deprecated typing.Set / typing.Optional in favour of builtins
- Guard _loop for None before broadcast calls
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

log = logging.getLogger("atlas.ui")

# ── Globals ───────────────────────────────────────────────────────────────────
_ws_connections: set = set()
_connections_lock = threading.Lock()
_broadcast_queue: queue.Queue = queue.Queue()   # thread-safe bridge → async loop
_loop: asyncio.AbstractEventLoop | None = None
_memory = None
_app = None


# ── Public broadcast API (called safely from any thread) ─────────────────────

def broadcast(message: dict) -> None:
    """Enqueue a message for all WebSocket clients. Safe to call from any thread."""
    if _loop is None:
        return  # UI not started yet — silently drop
    _broadcast_queue.put_nowait(json.dumps(message))


def broadcast_state(state: str, text: str | None = None) -> None:
    broadcast({"type": "state", "state": state, "text": text, "ts": time.time()})


def broadcast_reminder(text: str) -> None:
    broadcast({"type": "reminder", "text": text, "ts": time.time()})


# ── Server ────────────────────────────────────────────────────────────────────

def start_ui_server(cfg: "Config") -> None:
    """
    Blocking — run this in a daemon thread.
    Starts FastAPI + uvicorn and drains broadcast_queue into WebSocket sends.
    """
    global _app, _memory, _loop

    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse
        import uvicorn
    except ImportError:
        log.error("FastAPI/uvicorn not installed. Run: pip install fastapi 'uvicorn[standard]'")
        return

    from memory.store import MemoryStore
    _memory = MemoryStore(cfg)
    _app = FastAPI(title="Atlas Dashboard")

    # ── Routes ────────────────────────────────────────────────────────────────

    @_app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        with open(html_path, encoding="utf-8") as f:
            return f.read()

    @_app.get("/api/history")
    async def get_history():
        return _memory.get_all_messages(limit=100)

    @_app.get("/api/reminders")
    async def get_reminders():
        return _memory.get_active_reminders()

    @_app.get("/api/stats")
    async def get_stats():
        return _memory.stats()

    @_app.post("/api/clear")
    async def clear_history():
        _memory.clear_history()
        return {"status": "cleared"}

    @_app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        with _connections_lock:
            _ws_connections.add(websocket)
        log.debug("WebSocket client connected.")
        try:
            await websocket.send_text(json.dumps({"type": "state", "state": "IDLE"}))
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            with _connections_lock:
                _ws_connections.discard(websocket)

    # ── Background task: drain broadcast_queue → WebSocket sends ─────────────

    async def _drain_broadcast_queue():
        """Asyncio task that drains the thread-safe queue and fans out to clients."""
        while True:
            try:
                try:
                    data = _broadcast_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue

                dead: set = set()
                with _connections_lock:
                    connections = list(_ws_connections)

                for ws in connections:
                    try:
                        await ws.send_text(data)
                    except Exception:
                        dead.add(ws)

                with _connections_lock:
                    for ws in dead:
                        _ws_connections.discard(ws)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug(f"Broadcast drain error: {exc}")

    # ── Startup: capture running loop (Python 3.12 correct API) ──────────────

    @_app.on_event("startup")
    async def _on_startup():
        global _loop
        _loop = asyncio.get_running_loop()   # get_running_loop() is correct in 3.10+
        asyncio.create_task(_drain_broadcast_queue(), name="broadcast-drain")
        log.info(f"🌐 Atlas dashboard → http://localhost:{cfg.ui_port}")

    uvicorn.run(_app, host=cfg.ui_host, port=cfg.ui_port, log_level="error")
