"""
Generates the synthetic personal-ops dataset (emails, calendar, tasks, notes)
used both to seed the agent's store and as the basis for eval/scenarios.

The data is hand-curated, not randomly generated, and deliberately includes
edge cases the agent must handle correctly:

- an email/calendar factual contradiction (hallucination trap)
- a double-booked pair of calendar events (conflict detection)
- an ambiguous action item with no concrete date ("sometime this week")
- a task that's already done but an email follow-up implies it's still open
- a near-duplicate task (same intent, different wording)
- a low-signal newsletter email that should be triaged as noise

Run directly to (re)write the JSON fixture files in this directory.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

EMAILS = [
    {
        "id": "em-001",
        "from": "priya@acme.com",
        "to": "me@acme.com",
        "subject": "Q3 roadmap deck",
        "body": "Hey, can you send me the Q3 roadmap deck by Friday? Want to review before the board meeting.",
        "received_at": "2026-09-01T09:12:00",
        "thread_id": "th-001",
    },
    {
        "id": "em-002",
        "from": "priya@acme.com",
        "to": "me@acme.com",
        "subject": "Re: Q3 roadmap deck",
        "body": "Following up — did you get a chance to send the Q3 deck over yet?",
        "received_at": "2026-09-05T14:03:00",
        "thread_id": "th-001",
    },
    {
        "id": "em-003",
        "from": "dev@acme.com",
        "to": "me@acme.com",
        "subject": "Sync this week?",
        "body": "Let's sync sometime this week about the migration timeline. Let me know what works.",
        "received_at": "2026-09-02T11:30:00",
        "thread_id": "th-002",
    },
    {
        "id": "em-004",
        "from": "sam@acme.com",
        "to": "me@acme.com",
        "subject": "Confirmed: Design review Thursday 2pm",
        "body": "Confirming our design review is set for Thursday at 2pm in Room B. See you there!",
        "received_at": "2026-09-03T08:45:00",
        "thread_id": "th-003",
    },
    {
        "id": "em-005",
        "from": "newsletter@saastools.io",
        "to": "me@acme.com",
        "subject": "5 productivity hacks you need in 2026",
        "body": "Discover the top 5 tools power users swear by this year...",
        "received_at": "2026-09-04T06:00:00",
        "thread_id": "th-004",
    },
]

# NOTE: em-004 says the design review is "Thursday at 2pm" but the calendar
# event below has it at 3pm — an intentional email/calendar contradiction.
# The agent must trust the calendar (source of truth) or flag the conflict,
# never just restate whatever the email claims.
CALENDAR_EVENTS = [
    {
        "id": "ev-001",
        "title": "Design review",
        "start": "2026-09-10T15:00:00",
        "end": "2026-09-10T16:00:00",
        "attendees": ["sam@acme.com", "me@acme.com"],
        "location": "Room B",
        "status": "confirmed",
    },
    {
        "id": "ev-002",
        "title": "1:1 with manager",
        "start": "2026-09-10T15:30:00",
        "end": "2026-09-10T16:00:00",
        "attendees": ["manager@acme.com", "me@acme.com"],
        "location": "Zoom",
        "status": "confirmed",
    },
    {
        "id": "ev-003",
        "title": "Daily standup",
        "start": "2026-09-08T09:00:00",
        "end": "2026-09-08T09:15:00",
        "attendees": ["team@acme.com"],
        "location": "Zoom",
        "status": "confirmed",
    },
]

TASKS = [
    {
        "id": "tk-001",
        "title": "Send Q3 roadmap deck to Priya",
        "status": "open",
        "due_date": "2026-09-05",
        "created_at": "2026-09-01T09:15:00",
        "source_email_id": "em-001",
        "tags": ["reporting"],
    },
    {
        "id": "tk-002",
        "title": "Share Q3 deck with Priya before board mtg",
        "status": "done",
        "due_date": "2026-09-04",
        "created_at": "2026-09-01T18:00:00",
        "source_email_id": None,
        "tags": ["reporting"],
    },
    {
        "id": "tk-003",
        "title": "Review migration timeline with dev team",
        "status": "open",
        "due_date": None,
        "created_at": "2026-09-02T11:35:00",
        "source_email_id": "em-003",
        "tags": ["engineering"],
    },
    {
        "id": "tk-004",
        "title": "Renew parking permit",
        "status": "open",
        "due_date": "2026-08-20",
        "created_at": "2026-08-10T10:00:00",
        "source_email_id": None,
        "tags": ["admin"],
    },
]

NOTES = [
    {
        "id": "nt-001",
        "title": "Priya - working prefs",
        "body": "Priya prefers async written updates over calls when possible. Based in a UTC+1 timezone.",
        "created_at": "2026-07-15T10:00:00",
        "tags": ["people"],
    },
    {
        "id": "nt-002",
        "title": "Migration project - context",
        "body": "The migration project moves the legacy billing service off the old DB. Dev team owns the timeline; "
        "current target is end of Q3 but has slipped twice already.",
        "created_at": "2026-08-01T09:00:00",
        "tags": ["engineering", "migration"],
    },
]


def main() -> None:
    (DATA_DIR / "emails.json").write_text(json.dumps(EMAILS, indent=2))
    (DATA_DIR / "calendar.json").write_text(json.dumps(CALENDAR_EVENTS, indent=2))
    (DATA_DIR / "tasks.json").write_text(json.dumps(TASKS, indent=2))
    (DATA_DIR / "notes.json").write_text(json.dumps(NOTES, indent=2))
    print(f"Wrote fixtures to {DATA_DIR}")


if __name__ == "__main__":
    main()
