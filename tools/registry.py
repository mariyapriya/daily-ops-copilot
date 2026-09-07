"""
Central registry tying each tool's Pydantic input schema to its
implementation, and generating the OpenAI-style `tools=[...]` function specs
from those same schemas so the contract can't drift between what the LLM is
told and what actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from store.db import Store
from tools import functions, schemas


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    func: Callable[[Store, BaseModel], Any]


TOOLS: dict[str, ToolSpec] = {
    "get_emails": ToolSpec(
        "get_emails",
        "List emails, optionally filtered to those received since a given ISO datetime.",
        schemas.GetEmails,
        functions.get_emails,
    ),
    "get_calendar_events": ToolSpec(
        "get_calendar_events",
        "List calendar events, optionally filtered to a start/end ISO datetime range.",
        schemas.GetCalendarEvents,
        functions.get_calendar_events,
    ),
    "get_tasks": ToolSpec(
        "get_tasks",
        "List tasks, optionally filtered by status (open, done, blocked).",
        schemas.GetTasks,
        functions.get_tasks,
    ),
    "create_task": ToolSpec(
        "create_task",
        "Create a new task. Use this for action items found in emails/notes that don't already exist as a task.",
        schemas.CreateTask,
        functions.create_task,
    ),
    "update_task_status": ToolSpec(
        "update_task_status",
        "Update an existing task's status by its id.",
        schemas.UpdateTaskStatus,
        functions.update_task_status,
    ),
    "schedule_event": ToolSpec(
        "schedule_event",
        "Create a new calendar event.",
        schemas.ScheduleEvent,
        functions.schedule_event,
    ),
    "search_notes": ToolSpec(
        "search_notes",
        "Search personal notes by keyword for background context (e.g. project details, people's preferences).",
        schemas.SearchNotes,
        functions.search_notes,
    ),
    "draft_reply": ToolSpec(
        "draft_reply",
        "Save a draft reply to an email. This never actually sends anything.",
        schemas.DraftReply,
        functions.draft_reply,
    ),
}


def build_openai_tools() -> list[dict]:
    """Returns the `tools=[...]` list in OpenAI's function-calling format."""
    specs = []
    for tool in TOOLS.values():
        schema = tool.model.model_json_schema()
        schema.pop("title", None)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            }
        )
    return specs


# Smaller/local models frequently emit the literal string "null" (or "none",
# "N/A", "") for an omitted optional argument instead of actually omitting the
# key or emitting JSON null. Left unhandled, this is silent and dangerous: a
# str | None field happily accepts the *string* "null" as a valid value, so
# e.g. `get_emails(since="null")` doesn't error — it just filters on the
# literal string "null" and returns zero rows with no indication anything
# went wrong. Normalizing these sentinel strings to real None before
# validation turns a silent wrong-answer failure mode into correct behavior.
_NULL_SENTINELS = {"null", "none", "n/a", ""}


def _normalize_nulls(arguments: dict) -> dict:
    return {
        key: (None if isinstance(value, str) and value.strip().lower() in _NULL_SENTINELS else value)
        for key, value in arguments.items()
    }


def dispatch(store: Store, name: str, arguments: dict) -> Any:
    """Validates arguments against the tool's schema and runs it.

    Never raises for bad input or unknown tool names — instead returns an
    `{"error": ...}` dict, since a malformed tool call is exactly the kind of
    thing the agent loop should be able to see and recover from (e.g. by
    retrying with corrected arguments) rather than crashing the whole run.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": f"unknown tool {name!r}. Available tools: {sorted(TOOLS)}"}
    try:
        validated = tool.model.model_validate(_normalize_nulls(arguments))
    except ValidationError as exc:
        return {"error": f"invalid arguments for {name!r}: {exc.errors()}"}
    return tool.func(store, validated)
