"""Pydantic input schemas for every tool the agent can call.

Keeping these separate from the implementations means each tool's argument
contract is explicit, validated, and independently reusable when generating
the JSON-schema tool specs handed to the LLM (see tools/registry.py).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tools.coercion import OptionalStr, StrList


class GetEmails(BaseModel):
    since: OptionalStr = Field(default=None, description="ISO datetime; only return emails received at or after this.")


class GetCalendarEvents(BaseModel):
    start: OptionalStr = Field(default=None, description="ISO datetime lower bound (inclusive).")
    end: OptionalStr = Field(default=None, description="ISO datetime upper bound (inclusive).")


class GetTasks(BaseModel):
    status: OptionalStr = Field(default=None, description="Filter by status: 'open', 'done', or 'blocked'.")


class CreateTask(BaseModel):
    title: str = Field(description="Short, human-readable task title.")
    due_date: str | None = Field(default=None, description="ISO date (YYYY-MM-DD), if known.")
    tags: StrList = Field(default_factory=list)
    source_email_id: str | None = Field(default=None, description="Email id this task was derived from, if any.")


class UpdateTaskStatus(BaseModel):
    task_id: str
    status: str = Field(description="One of: open, done, blocked.")


class ScheduleEvent(BaseModel):
    title: str
    start: str = Field(description="ISO datetime.")
    end: str = Field(description="ISO datetime.")
    attendees: StrList = Field(default_factory=list)
    location: str | None = None


class SearchNotes(BaseModel):
    query: str = Field(description="Keyword or phrase to search for in note titles/bodies.")


class DraftReply(BaseModel):
    email_id: str = Field(description="Id of the email being replied to. Must be a real id from get_emails.")
    body: str = Field(description="Draft reply text. This is saved as a draft only — never actually sent.")
