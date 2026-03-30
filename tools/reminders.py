"""
tools/reminders.py — Reminder and timer tool

Python 3.12 fixes:
- Replaced XTTSBackend.__new__() hack with a proper module-level
  get_tts_instance() that reuses the cached model safely
- broadcast_reminder guarded against UI not running (broadcast handles None loop)
- asyncio usage removed from fire path (timer runs in a plain thread)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Optional

from config import Config
from memory.store import MemoryStore

log = logging.getLogger("atlas.tools.reminders")

_active_reminders: dict[str, dict] = {}
_lock = threading.Lock()


def _get_tts(cfg: Config):
    """
    Return a working TTS backend instance that reuses already-loaded models.
    For XTTS-v2: the model and latents are module-level singletons in
    tts.xtts_backend — simply constructing a new XTTSBackend reuses them.
    """
    try:
        if cfg.tts_backend == "xtts":
            from tts.xtts_backend import XTTSBackend
            return XTTSBackend(cfg)   # __init__ reuses cached _model / _latents
        else:
            from tts.rvc_backend import RVCBackend
            return RVCBackend(cfg)    # same singleton pattern
    except Exception as exc:
        log.warning(f"Could not get TTS instance for reminder: {exc}")
        return None


class ReminderTool:
    def __init__(self, cfg: Config, memory: MemoryStore):
        self.cfg = cfg
        self.memory = memory

    def set_reminder(self, text: str, seconds: int) -> str:
        try:
            seconds = int(seconds)
        except (TypeError, ValueError):
            return "Invalid duration. Please specify seconds as a number."

        if seconds <= 0:
            return "Reminder duration must be positive."
        if seconds > 86400:
            return "Maximum reminder duration is 24 hours (86400 seconds)."

        rid = str(uuid.uuid4())[:8]
        due_ts = time.time() + seconds

        with _lock:
            _active_reminders[rid] = {
                "id": rid, "text": text, "due": due_ts, "seconds": seconds
            }
        self.memory.save_reminder(rid, text, due_ts)

        timer = threading.Timer(seconds, self._fire_reminder, args=(rid, text))
        timer.daemon = True
        timer.start()

        if seconds < 60:
            dur = f"{seconds} seconds"
        elif seconds < 3600:
            dur = f"{seconds // 60} minutes"
        else:
            dur = f"{seconds // 3600} hours"

        log.info(f"⏰ Reminder set: '{text}' in {dur} (id={rid})")
        return f"Reminder set for {dur} from now. ID: {rid}"

    def list_reminders(self) -> str:
        with _lock:
            if not _active_reminders:
                return "No active reminders."
            lines = []
            for r in _active_reminders.values():
                left = max(0, int(r["due"] - time.time()))
                if left < 60:
                    t = f"{left}s"
                elif left < 3600:
                    t = f"{left // 60}m"
                else:
                    t = f"{left // 3600}h"
                lines.append(f"[{r['id']}] '{r['text']}' in {t}")
            return "; ".join(lines)

    def cancel_reminder(self, id: str) -> str:
        with _lock:
            if id in _active_reminders:
                del _active_reminders[id]
                self.memory.delete_reminder(id)
                return f"Reminder {id} cancelled."
        return f"No reminder with id '{id}'."

    def _fire_reminder(self, rid: str, text: str):
        log.info(f"🔔 Reminder firing: '{text}'")
        with _lock:
            _active_reminders.pop(rid, None)
        self.memory.delete_reminder(rid)

        # Push WebSocket notification (safe even if UI is not running)
        try:
            from ui.server import broadcast_reminder
            broadcast_reminder(text)
        except Exception:
            pass

        # Speak the reminder using TTS
        try:
            import sounddevice as sd
            tts = _get_tts(self.cfg)
            if tts is not None:
                audio, sr = tts.synthesize(f"Reminder: {text}")
                sd.play(audio, samplerate=sr)
                sd.wait()
        except Exception as exc:
            log.error(f"Reminder TTS playback failed: {exc}")
