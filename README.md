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

## Current status: Week 2 — LangGraph orchestration with a verify/guardrail node

### Week 1: grounded baseline agent

What's built:

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

### Known gaps in the Week-1 baseline (motivated Week 2)

Running `python main.py --engine naive --today 2026-09-10` showed the naive
agent, across different runs:

1. **Hallucinating a meeting time from a stale email over the calendar** —
   reporting the design review at "2:00 PM" (from the fixture email
   `em-004`'s stale claim) instead of the calendar's actual 3:00 PM
   (`ev-001`).
2. **Missing the double-booking** — `ev-001` (3:00-4:00 PM) and `ev-002`
   (3:30-4:00 PM) overlap; the system prompt explicitly says to flag
   overlaps, and every naive run still listed both events with no conflict
   called out.
3. On a different run (same fixtures, same prompt — LLMs are stochastic),
   the time hallucination didn't recur, but a *new* one did: a task marked
   `done` in the store (`tk-002`) was reported as "overdue."

That third point matters as much as the first two: which hallucination
shows up is not consistent run to run, so a single demo run proves very
little either way. That's the real argument for Week 2's structural fix
over further prompt-tweaking, and for Week 3's eval suite over spot-checking
by hand.

## Week 2: LangGraph orchestration + verify/guardrail node

### Architecture

```
plan → agent ⇄ tools → verify ─┬─(ok, or attempts exhausted)─→ finalize
                    ▲            │
                    └─(issues, attempts remain)┘
```

(`agent/graph.py`). `plan` and `agent`/`tools` follow the same message
format and tool-dispatch path as the Week 1 naive agent — LangGraph is used
purely for the state graph, conditional routing, and checkpointing, not to
replace `agent/llm.py` or `tools/registry.py`. One simplification versus the
original 6-node sketch: `act`/`summarize` were dropped as separate nodes
since, for a read-mostly briefing workflow, the draft produced by `agent`
already *is* the summary — a dedicated no-op node would have padded the
graph without adding anything to explain. (Actual write actions — drafting
a reply, creating a task — are deferred to Week 4, paired with the
Streamlit UI's approve/reject button, which is a more honest home for a
human-in-the-loop gate than a CLI `y/n` prompt.)

The **verify** node is two independent checks, deliberately combined:

1. An **LLM fact-checker**, forced via `tool_choice` to call a
   `report_verification(ok, issues)` tool (`tools/verification_schema.py`)
   given the draft plus the actual retrieved tool outputs — it checks
   date/time/status claims and overlap-flagging.
2. A **deterministic check** (`_find_tool_error_issues` in `agent/graph.py`)
   that scans the run's tool-call history for any that actually failed and
   forces `ok=False` if the draft doesn't account for it — regardless of
   what the LLM judge says. Not everything needs an LLM: whether a tool call
   errored is a fact, not a judgment call, and checking it in code is
   strictly more reliable than hoping a judge model notices.

On failure, the graph loops back to `agent` with the issues as feedback (capped at 2 attempts); if still unresolved, it finalizes anyway with a visible `⚠️` caveat rather than looping forever or silently shipping a wrong answer.

### Two more real bugs found by running it (same pattern as Week 1's null-sentinel fix)

- **A stringified array instead of a native one.** The verifier's first live
  test came back with `issues` as the *string* `'["Meeting time discrepancy"]'`
  instead of a JSON array — Pydantic rejects a bare string for `list[str]`
  outright. Fixed once, generally, in `tools/coercion.py` (`StrList`) and
  applied to every `list[str]` tool argument (this field, plus `tags` and
  `attendees` from Week 1, which had the same latent risk untested until now).
- **The mirror-image quirk**: `get_tasks(status=["open"])` — a plain
  `str | None` filter wrapped in a one-element list. Fixed the same way
  (`OptionalStr` in `tools/coercion.py`), applied to every optional string
  filter argument.

Both are pinned with regression tests (`test_dispatch_coerces_json_stringified_list_argument`, `test_dispatch_coerces_single_element_list_to_scalar_string`).

### The guardrail generalized to a failure it wasn't built for

While confirming the fix for the two Week-1 gaps, the LLM also called
`search_notes(query=None)` — a required field, no valid default — which
genuinely failed. The drafting model quietly reported "no results" anyway,
and the LLM verifier's first pass didn't catch it. The **deterministic**
check did, immediately, with no code change: it doesn't care which tool
failed, only that one did. The run correctly finalized with a caveat
instead of a false "verified" claim:

> ⚠️ Automatic fact-check could not fully confirm this briefing after 2
> attempt(s): Design review time conflict: 1:1 with manager overlaps with
> Design review; The search_notes call failed (...) — the draft must not
> present this as 'no data found'; it must say this check could not be
> completed.

Meanwhile the two original Week-1 failures were both actually fixed in that
same run: the design review appeared under "Today's Calendar Events" at the
*correct* 15:00-16:00 (not the email's stale "2pm"), and the overlap with
the 1:1 was identified as an issue on both verify passes (the model never
fully wrote the conflict into the visible draft text within the 2-attempt
budget — which is exactly why the run ended in an honest caveat rather than
a falsely "verified" success). `search_notes`'s `None`-query case is left
unfixed on purpose for now and logged as a Week 3 eval-scenario candidate,
rather than chased down ad hoc — the point of this section is that the
*architecture* catches unanticipated failures, not that every individual
model quirk gets patched reactively.

### Tracing and cost

Every graph run writes a full JSONL trace to `traces/<run_id>.jsonl`
(`agent/tracing.py`) — every LLM call and tool call, with latency, token
usage, and an estimated cost (`agent/cost.py`; $0 for local Ollama, a real
number if `LLM_MODEL` points at a hosted model) — so a run is replayable
after the fact without an external tracing service.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
python data/generate_fixtures.py     # (re)writes data/*.json
ollama pull llama3.2                 # or point LLM_BASE_URL/LLM_MODEL at a hosted API

python main.py --engine graph --today 2026-09-10 --show-trace   # Week 2 agent (default)
python main.py --engine naive --today 2026-09-10                # Week 1 baseline, for comparison
```

Environment variables (all optional, default to local Ollama):

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | `ollama` | Ignored by Ollama; required for hosted APIs |
| `LLM_MODEL` | `llama3.2` | Model name for the configured backend |

Run tests: `.venv/bin/python -m pytest tests/ -v` (30 tests: store, tools,
coercion regressions, and graph routing/guardrail logic against a stubbed
LLM client — no live Ollama server required to run the suite).

## Roadmap

- **Week 3**: labeled eval dataset (including the hallucination/conflict
  traps as scored scenarios) and an automated harness measuring
  task-completion accuracy, factual-grounding rate, latency, and cost —
  re-run on every prompt/graph change to catch regressions.
- **Week 4**: Streamlit demo (briefing / trace timeline / eval dashboard),
  architecture diagram, short demo recording. Stretch: swap the synthetic
  fixtures for a real Google Calendar/Gmail integration behind the same
  tool interface.
