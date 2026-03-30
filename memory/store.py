"""
memory/store.py — Persistent SQLite memory for Atlas

Python 3.12 fixes:
- Replaced deprecated typing.List / typing.Dict with builtin list / dict
- Added explicit conn.close() in thread-local cleanup
- sqlite3 WAL mode for better concurrent reads (multiple threads)
- Type annotations use X | None instead of Optional[X]
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time

from config import Config

log = logging.getLogger("atlas.memory")


class MemoryStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._db_path = cfg.db_path
        self._local = threading.local()
        self._init_db()
        log.info(f"💾 Memory store: {self._db_path}")

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")   # better concurrent access
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    role    TEXT    NOT NULL,
                    content TEXT    NOT NULL,
                    ts      REAL    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    id      TEXT PRIMARY KEY,
                    text    TEXT NOT NULL,
                    due_ts  REAL NOT NULL,
                    created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_facts (
                    id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL UNIQUE,
                    ts   REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
            """)

    # ── Messages ──────────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO messages (role, content, ts) VALUES (?, ?, ?)",
                (role, content, time.time()),
            )

    def get_recent_messages(self, n_turns: int = 10) -> list[dict]:
        """Return last n_turns*2 messages as list of dicts for Ollama API."""
        rows = self._conn().execute(
            "SELECT role, content FROM messages ORDER BY ts DESC LIMIT ?",
            (n_turns * 2,),
        ).fetchall()
        rows.reverse()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def get_all_messages(self, limit: int = 100) -> list[dict]:
        rows = self._conn().execute(
            "SELECT role, content, ts FROM messages ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        rows.reverse()
        return [{"role": r["role"], "content": r["content"], "ts": r["ts"]} for r in rows]

    def clear_history(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages")
        log.info("Conversation history cleared.")

    # ── Reminders ─────────────────────────────────────────────────────────────

    def save_reminder(self, rid: str, text: str, due_ts: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reminders (id, text, due_ts, created) VALUES (?, ?, ?, ?)",
                (rid, text, due_ts, time.time()),
            )

    def delete_reminder(self, rid: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM reminders WHERE id = ?", (rid,))

    def get_active_reminders(self) -> list[dict]:
        rows = self._conn().execute(
            "SELECT * FROM reminders WHERE due_ts > ? ORDER BY due_ts",
            (time.time(),),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── User facts ────────────────────────────────────────────────────────────

    def save_fact(self, fact: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO user_facts (fact, ts) VALUES (?, ?)",
                (fact, time.time()),
            )

    def get_facts(self) -> list[str]:
        rows = self._conn().execute(
            "SELECT fact FROM user_facts ORDER BY ts DESC LIMIT 20"
        ).fetchall()
        return [r["fact"] for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        conn = self._conn()
        total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        active_rem = conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE due_ts > ?", (time.time(),)
        ).fetchone()[0]
        return {"total_messages": total_msgs, "active_reminders": active_rem}

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
