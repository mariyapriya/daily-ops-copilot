"""
The agent's single source of truth for structured facts (emails, calendar
events, tasks, notes). The agent is required to reach every fact through
this module's read functions rather than recalling it from LLM context —
that's the core hallucination-mitigation mechanism for this project: if a
claim isn't retrievable here, the agent must not assert it.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DATA_DIR = Path(__file__).parent.parent / "data"

SCHEMA = """
CREATE TABLE IF NOT EXISTS emails (
    id TEXT PRIMARY KEY,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    received_at TEXT NOT NULL,
    thread_id TEXT
);

CREATE TABLE IF NOT EXISTS calendar_events (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start TEXT NOT NULL,
    end TEXT NOT NULL,
    attendees TEXT NOT NULL,   -- JSON-encoded list
    location TEXT,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,       -- open | done | blocked
    due_date TEXT,
    created_at TEXT NOT NULL,
    source_email_id TEXT,
    tags TEXT NOT NULL          -- JSON-encoded list
);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    tags TEXT NOT NULL          -- JSON-encoded list
);

CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    email_id TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


@dataclass
class Store:
    """Thin repository over a SQLite file. One instance per DB path."""

    db_path: str = ":memory:"
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @contextmanager
    def _cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        finally:
            cur.close()

    # ---- seeding -----------------------------------------------------

    def seed_from_fixtures(self, data_dir: Path = DATA_DIR) -> None:
        """Loads the JSON fixtures in `data/` into the store, replacing any
        existing rows with the same ids (safe to call repeatedly)."""
        emails = json.loads((data_dir / "emails.json").read_text())
        events = json.loads((data_dir / "calendar.json").read_text())
        tasks = json.loads((data_dir / "tasks.json").read_text())
        notes = json.loads((data_dir / "notes.json").read_text())

        with self._cursor() as cur:
            for e in emails:
                cur.execute(
                    "INSERT OR REPLACE INTO emails VALUES (?,?,?,?,?,?,?)",
                    (e["id"], e["from"], e["to"], e["subject"], e["body"], e["received_at"], e.get("thread_id")),
                )
            for ev in events:
                cur.execute(
                    "INSERT OR REPLACE INTO calendar_events VALUES (?,?,?,?,?,?,?)",
                    (
                        ev["id"],
                        ev["title"],
                        ev["start"],
                        ev["end"],
                        json.dumps(ev["attendees"]),
                        ev.get("location"),
                        ev["status"],
                    ),
                )
            for t in tasks:
                cur.execute(
                    "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?)",
                    (
                        t["id"],
                        t["title"],
                        t["status"],
                        t.get("due_date"),
                        t["created_at"],
                        t.get("source_email_id"),
                        json.dumps(t.get("tags", [])),
                    ),
                )
            for n in notes:
                cur.execute(
                    "INSERT OR REPLACE INTO notes VALUES (?,?,?,?,?)",
                    (n["id"], n["title"], n["body"], n["created_at"], json.dumps(n.get("tags", []))),
                )

    # ---- reads ---------------------------------------------------------

    def list_emails(self, since: str | None = None) -> list[dict]:
        query = "SELECT * FROM emails"
        params: tuple = ()
        if since:
            query += " WHERE received_at >= ?"
            params = (since,)
        query += " ORDER BY received_at ASC"
        with self._cursor() as cur:
            rows = cur.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_calendar_events(self, start: str | None = None, end: str | None = None) -> list[dict]:
        query = "SELECT * FROM calendar_events"
        clauses, params = [], []
        if start:
            clauses.append("start >= ?")
            params.append(start)
        if end:
            clauses.append("start <= ?")
            params.append(end)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY start ASC"
        with self._cursor() as cur:
            rows = cur.execute(query, tuple(params)).fetchall()
        return [self._row_with_json(r, ["attendees"]) for r in rows]

    def list_tasks(self, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM tasks"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY due_date IS NULL, due_date ASC"
        with self._cursor() as cur:
            rows = cur.execute(query, params).fetchall()
        return [self._row_with_json(r, ["tags"]) for r in rows]

    def get_task(self, task_id: str) -> dict | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_with_json(row, ["tags"]) if row else None

    def search_notes(self, query: str) -> list[dict]:
        """Naive substring search — the real semantic search layer (vector
        store) replaces this in a later milestone; this keeps week 1 usable
        without an embeddings dependency."""
        like = f"%{query}%"
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM notes WHERE body LIKE ? OR title LIKE ? ORDER BY created_at DESC",
                (like, like),
            ).fetchall()
        return [self._row_with_json(r, ["tags"]) for r in rows]

    # ---- writes ---------------------------------------------------------

    def create_task(self, id: str, title: str, due_date: str | None, tags: list[str] | None = None,
                     source_email_id: str | None = None, created_at: str = "") -> dict:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?)",
                (id, title, "open", due_date, created_at, source_email_id, json.dumps(tags or [])),
            )
        return self.get_task(id)  # type: ignore[return-value]

    def update_task_status(self, task_id: str, status: str) -> dict | None:
        if status not in ("open", "done", "blocked"):
            raise ValueError(f"invalid task status: {status!r}")
        with self._cursor() as cur:
            cur.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        return self.get_task(task_id)

    def schedule_event(self, id: str, title: str, start: str, end: str,
                        attendees: list[str] | None = None, location: str | None = None) -> dict:
        with self._cursor() as cur:
            cur.execute(
                "INSERT OR REPLACE INTO calendar_events VALUES (?,?,?,?,?,?,?)",
                (id, title, start, end, json.dumps(attendees or []), location, "confirmed"),
            )
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM calendar_events WHERE id = ?", (id,)).fetchone()
        return self._row_with_json(row, ["attendees"])

    def create_draft(self, id: str, email_id: str, body: str, created_at: str = "") -> dict:
        with self._cursor() as cur:
            cur.execute("INSERT OR REPLACE INTO drafts VALUES (?,?,?,?)", (id, email_id, body, created_at))
            row = cur.execute("SELECT * FROM drafts WHERE id = ?", (id,)).fetchone()
        return dict(row)

    def get_email(self, email_id: str) -> dict | None:
        with self._cursor() as cur:
            row = cur.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()
        return dict(row) if row else None

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _row_with_json(row: sqlite3.Row, json_cols: list[str]) -> dict:
        d = dict(row)
        for col in json_cols:
            d[col] = json.loads(d[col]) if d.get(col) else []
        return d
