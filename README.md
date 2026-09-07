# Daily Ops Copilot

A proactive personal-ops agent that reads a synthetic inbox, calendar, task
list, and notes, and produces a **morning briefing** — without the user
having to prompt-engineer anything. Built as a portfolio project targeting
LLM Application Engineer roles (agentic workflows, reliability, evaluation,
observability), not a toy chatbot demo.

## Why this project

Everyday tools (email, notes, tasks, calendar) aren't AI-native — using them
well still means reading everything yourself and reasoning across scattered
sources of truth. This project simulates that exact problem: an agent that
must **reason over persistent context, plan multi-step actions, call tools
instead of guessing, and never assert a fact it can't verify** against its
own data store. That last point is the whole point — LLMs are probabilistic;
the engineering problem is making their outputs predictable, observable, and
safe anyway.

## Current status: Week 1 — grounded baseline agent

What's built so far:

- **Synthetic dataset** (`data/`) — hand-curated emails, calendar events,
  tasks, and notes, with intentional edge cases: a double-booked pair of
  meetings, an email that states a meeting time contradicting the calendar,
  an ambiguous "let's sync sometime" action item, and a task marked done
  that a stale follow-up email implies is still pending.
- **SQLite store** (`store/db.py`) — the single source of truth. The agent
  is never allowed to answer from memory; every fact must come from a tool
  call against this store.
- **Tool layer** (`tools/`) — 8 Pydantic-schema'd, deterministic, unit-tested
  functions (`get_emails`, `get_calendar_events`, `get_tasks`, `create_task`,
  `update_task_status`, `schedule_event`, `search_notes`, `draft_reply`).
  None of them touch an LLM, which is what makes them independently
  testable (`tests/test_tools.py`, `tests/test_store.py` — 19 tests).
- **OpenAI-compatible LLM client** (`agent/llm.py`) — talks to any
  `/v1/chat/completions` backend. Defaults to a local Ollama model
  (`llama3.2`, free, no API key) purely via a base-URL swap; pointing it at
  OpenAI/Anthropic-compatible hosted APIs is a config change, not a rewrite.
- **Naive tool-calling agent** (`agent/naive_agent.py`) — a plain ReAct-style
  loop with no orchestration graph, no memory, and no verification step yet.
  This is intentional: it's the "before" baseline that the LangGraph +
  guardrail version (Week 2) gets measured against, so the improvement is
  demonstrable rather than asserted.

### A real reliability bug found and fixed in Week 1

Running the naive agent against the local model surfaced a genuine failure
mode, not a hypothetical one: the model frequently emitted the *literal
string* `"null"` for an omitted optional tool argument instead of actually
omitting it. Since the argument's schema is `str | None`, Pydantic happily
accepted `"null"` as a valid string — so `get_tasks(status="null")` silently
filtered on the literal text "null" and returned zero rows, with no error
anywhere. Every tool call "succeeded" and every result was wrong. Fixed in
`tools/registry.py` by normalizing null-sentinel strings (`"null"`, `"none"`,
`""`) to real `None` before validation — with a regression test
(`test_dispatch_normalizes_literal_null_string_to_none`) pinning the
behavior. This is exactly the "probabilistic output → predictable action"
problem the role is about, caught by actually running the system rather than
eyeballing the code.

### Known gaps in the Week-1 baseline (by design — this is what Week 2 fixes)

Running `python main.py --today 2026-09-10 --show-trace` today shows the
naive agent:

1. **Hallucinating a meeting time from a stale email over the calendar.**
   The fixture email `em-004` claims the design review is "Thursday at 2pm";
   the calendar's `ev-001` says 3:00 PM. The naive agent's briefing reported
   "2:00 PM" — it trusted the email's claim instead of the calendar tool
   result it had just retrieved. This is the exact hallucination trap the
   fixtures were built to catch, and the naive baseline fails it.
2. **Missing the double-booking.** `ev-001` (3:00-4:00 PM) and `ev-002`
   (3:30-4:00 PM) overlap. The system prompt explicitly instructs the agent
   to flag overlapping events, and it still listed both with no conflict
   called out.

Both are the intended job for the Week 2 **verify** node: cross-check every
factual claim in the draft output against the SQLite store before it's
allowed through, and loop back to gather more information (or flag
uncertainty) on mismatch — rather than relying on prompt instructions alone,
which this run shows are not sufficient on their own.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python data/generate_fixtures.py     # (re)writes data/*.json
ollama pull llama3.2                 # or point LLM_BASE_URL/LLM_MODEL at a hosted API
python main.py --today 2026-09-10 --show-trace
```

Environment variables (all optional, default to local Ollama):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | `ollama` | Ignored by Ollama; required for hosted APIs |
| `LLM_MODEL` | `llama3.2` | Model name for the configured backend |

Run tests: `.venv/bin/python -m pytest tests/ -v`

## Roadmap

- **Week 2**: LangGraph orchestration (`plan → gather → reason/draft →
  verify → act → summarize`) with the verify node closing the two gaps
  above, plus run tracing.
- **Week 3**: labeled eval dataset (including the hallucination/conflict
  traps as scored scenarios) and an automated harness measuring
  task-completion accuracy, factual-grounding rate, latency, and cost —
  re-run on every prompt/graph change to catch regressions.
- **Week 4**: Streamlit demo (briefing / trace timeline / eval dashboard),
  architecture diagram, short demo recording. Stretch: swap the synthetic
  fixtures for a real Google Calendar/Gmail integration behind the same
  tool interface.
