"""
Week-1 baseline agent: a plain ReAct-style tool-calling loop with no graph,
no verification step, and no memory beyond the current conversation. It
exists to be compared against the LangGraph version built in a later
milestone — a controlled "before" to make the orchestration/guardrail
improvements measurable rather than just asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from agent.llm import ChatResult, LLMClient
from store.db import Store
from tools.registry import build_openai_tools, dispatch

MAX_TURNS = 6

SYSTEM_PROMPT = """You are a proactive daily-ops assistant. Your job is to produce a short \
morning briefing covering:
1. Anything from recent email that needs action today (replies owed, deadlines).
2. Calendar conflicts or notable events today or tomorrow.
3. Open tasks that are overdue or otherwise need attention.

Hard rules:
- You MUST use the available tools to look up every fact (emails, calendar events, \
tasks, notes). Never state a date, meeting time, task status, or email content that \
you have not just retrieved via a tool call in this conversation.
- If two calendar events overlap, call this out explicitly as a conflict rather than \
mentioning only one of them.
- If something is ambiguous or you're not confident, say so plainly instead of guessing.
- When you have gathered enough information, respond with the final briefing as plain \
text and do not call any more tools.
"""


@dataclass
class AgentRun:
    final_message: str
    trace: list[dict] = field(default_factory=list)
    turns_used: int = 0
    total_usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


def _accumulate_usage(total: dict, usage: dict | None) -> None:
    if not usage:
        return
    for key in total:
        total[key] += usage.get(key, 0)


def run_daily_briefing(
    store: Store,
    llm: LLMClient | None = None,
    max_turns: int = MAX_TURNS,
    today: str | None = None,
) -> AgentRun:
    llm = llm or LLMClient()
    tools = build_openai_tools()
    today = today or date.today().isoformat()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Today's date is {today}. Give me my morning briefing."},
    ]
    trace: list[dict] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for turn in range(1, max_turns + 1):
        result: ChatResult = llm.chat(messages, tools=tools)
        _accumulate_usage(total_usage, result.usage)

        if not result.tool_calls:
            trace.append({"turn": turn, "type": "final", "content": result.content})
            return AgentRun(final_message=result.content or "", trace=trace, turns_used=turn, total_usage=total_usage)

        messages.append(
            {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in result.tool_calls
                ],
            }
        )
        for tc in result.tool_calls:
            output = dispatch(store, tc.name, tc.arguments)
            trace.append({"turn": turn, "type": "tool_call", "tool": tc.name, "arguments": tc.arguments, "output": output})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output, default=str)})

    # Ran out of turns: force one final answer with tools disabled rather than
    # returning nothing, and flag it clearly as a truncated run.
    forced = llm.chat(messages, tools=None)
    _accumulate_usage(total_usage, forced.usage)
    trace.append({"turn": max_turns + 1, "type": "final_forced", "content": forced.content})
    return AgentRun(
        final_message=forced.content or "(agent did not produce a final answer within the turn budget)",
        trace=trace,
        turns_used=max_turns + 1,
        total_usage=total_usage,
    )
