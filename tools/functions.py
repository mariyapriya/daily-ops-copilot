"""
Tool implementations. Each function is a thin, deterministic wrapper around
`store.db.Store` — no LLM involved anywhere in this file, which is what
makes these independently unit-testable (see tests/test_tools.py) and
trustworthy as the agent's only path to ground truth.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from store.db import Store
from tools import schemas


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_emails(store: Store, args: schemas.GetEmails) -> list[dict]:
    return store.list_emails(since=args.since)


def get_calendar_events(store: Store, args: schemas.GetCalendarEvents) -> list[dict]:
    return store.list_calendar_events(start=args.start, end=args.end)


def get_tasks(store: Store, args: schemas.GetTasks) -> list[dict]:
    return store.list_tasks(status=args.status)


def create_task(store: Store, args: schemas.CreateTask) -> dict:
    return store.create_task(
        id=_new_id("tk"),
        title=args.title,
        due_date=args.due_date,
        tags=args.tags,
        source_email_id=args.source_email_id,
        created_at=_now(),
    )


def update_task_status(store: Store, args: schemas.UpdateTaskStatus) -> dict:
    result = store.update_task_status(args.task_id, args.status)
    if result is None:
        return {"error": f"no task with id {args.task_id!r}"}
    return result


def schedule_event(store: Store, args: schemas.ScheduleEvent) -> dict:
    return store.schedule_event(
        id=_new_id("ev"),
        title=args.title,
        start=args.start,
        end=args.end,
        attendees=args.attendees,
        location=args.location,
    )


def search_notes(store: Store, args: schemas.SearchNotes) -> list[dict]:
    return store.search_notes(args.query)


def draft_reply(store: Store, args: schemas.DraftReply) -> dict:
    email = store.get_email(args.email_id)
    if email is None:
        return {"error": f"no email with id {args.email_id!r}; cannot draft a reply to it"}
    return store.create_draft(id=_new_id("dr"), email_id=args.email_id, body=args.body, created_at=_now())
