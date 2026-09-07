"""
Week 2's LangGraph orchestration: plan -> agent (tool-calling loop) -> verify
-> finalize, with a bounded retry loop back to `agent` when verification
finds problems. This is what fixes the two concrete failures the naive
agent (agent/naive_agent.py, kept unchanged as the "before" baseline)
produced: trusting a stale email's claimed meeting time over the calendar,
and missing an explicit double-booking.

LangGraph is used purely for the state-graph/conditional-routing/
checkpointing mechanics here. Messages stay plain OpenAI-format dicts (the
same shape naive_agent.py already uses) rather than LangChain message
objects, and tool execution goes through the exact same tools/registry.py
dispatch used in Week 1 — no rework of already-tested code.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.cost import estimate_cost_usd
from agent.llm import LLMClient
from agent.tracing import Tracer
from store.db import Store
from tools.registry import build_openai_tools, dispatch
from tools.verification_schema import VERIFICATION_TOOL_NAME, VERIFICATION_TOOL_SPEC, VerificationReport

MAX_AGENT_TURNS = 6
MAX_VERIFY_ATTEMPTS = 2

PLAN_PROMPT = """You are a proactive daily-ops assistant about to produce a morning briefing. \
Before looking anything up, write a short checklist (3-5 bullet points) of exactly what you \
will check: new/unread-worthy emails, today's and tomorrow's calendar events (including any \
overlaps), and open tasks that are overdue or need attention. Do not look anything up yet — \
just state the plan."""

AGENT_SYSTEM_PROMPT = """You are a proactive daily-ops assistant producing a morning briefing.

Hard rules:
- You MUST use the available tools to look up every fact (emails, calendar events, tasks, \
notes). Never state a date, meeting time, task status, or email content you have not just \
retrieved via a tool call in this conversation.
- Trust calendar tool results over anything an email merely claims — emails can be stale or wrong.
- If two calendar events overlap, call this out explicitly as a conflict.
- If something is ambiguous, say so plainly instead of guessing.
- When you have gathered enough information, respond with the final briefing as plain text \
and do not call any more tools.
"""

VERIFY_SYSTEM_PROMPT = """You are a strict fact-checker reviewing a draft morning briefing.

You will be given the draft text and the ground-truth data actually retrieved this run \
(emails, calendar events, tasks). Check:
1. Every date, time, and status claim in the draft must exactly match the ground truth. Flag \
anything the draft asserts that isn't directly supported by it, and anything where the draft \
sided with an email's claim over the calendar's actual data (the calendar always wins).
2. If two or more calendar events in the ground truth overlap in time, the draft must \
explicitly call that out as a conflict; flag it as an issue if it doesn't.

Call report_verification with ok=true only if you find zero issues.
"""


class BriefingState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    today: str
    retrieved_facts: Annotated[list[dict], operator.add]
    agent_turns: int
    verify_attempts: int
    verified_ok: bool | None
    verification_issues: list[str]
    final_message: str | None


def _last_assistant_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def _find_tool_error_issues(retrieved_facts: list[dict]) -> list[str]:
    """Deterministic (non-LLM) guardrail: if any tool call this run actually
    failed, that's checkable with certainty from the trace — no need to rely
    on the LLM verifier noticing it. Observed live: a failed get_tasks call
    (bad arguments) was silently reported by the drafting model as "no open
    tasks," which is a materially different, false claim ("the check failed"
    vs. "the check succeeded and found nothing"). Catching this class of
    mistake deterministically is strictly more reliable than hoping an LLM
    judge happens to notice it — exactly the "not everything needs an LLM"
    principle for reliable orchestration."""
    issues = []
    for fact in retrieved_facts:
        output = fact.get("output")
        if isinstance(output, dict) and "error" in output:
            issues.append(
                f"The {fact['tool']} call failed ({output['error']}) — the draft must not present this as "
                "'no data found'; it must say this check could not be completed."
            )
    return issues


def build_graph(store: Store, llm: LLMClient, tracer: Tracer):
    tools_spec = build_openai_tools()

    def plan_node(state: BriefingState) -> dict:
        with tracer.span("plan", "llm_call") as ev:
            result = llm.chat(
                messages=[
                    {"role": "system", "content": PLAN_PROMPT},
                    {"role": "user", "content": f"Today is {state['today']}."},
                ]
            )
            ev["usage"] = result.usage
            ev["cost_usd"] = estimate_cost_usd(llm.model, result.usage)
            ev["content"] = result.content

        return {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Today's date is {state['today']}. Plan:\n{result.content}\n\n"
                        "Now gather the facts and produce the morning briefing."
                    ),
                },
            ]
        }

    def agent_node(state: BriefingState) -> dict:
        turns = state.get("agent_turns", 0) + 1
        force_final = turns > MAX_AGENT_TURNS

        with tracer.span("agent", "llm_call") as ev:
            result = llm.chat(state["messages"], tools=None if force_final else tools_spec)
            ev["usage"] = result.usage
            ev["cost_usd"] = estimate_cost_usd(llm.model, result.usage)
            ev["tool_call_count"] = len(result.tool_calls)
            ev["forced_final"] = force_final

        assistant_message: dict = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            assistant_message["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in result.tool_calls
            ]
        return {"messages": [assistant_message], "agent_turns": turns}

    def route_after_agent(state: BriefingState) -> str:
        last = state["messages"][-1]
        return "tools" if last.get("tool_calls") else "verify"

    def tools_node(state: BriefingState) -> dict:
        last = state["messages"][-1]
        tool_calls = last.get("tool_calls") or []
        tool_messages, facts = [], []

        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            with tracer.span("tools", "tool_call") as ev:
                output = dispatch(store, name, args)
                ev["tool"], ev["arguments"], ev["output"] = name, args, output
            tool_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(output, default=str)})
            facts.append({"tool": name, "arguments": args, "output": output})

        return {"messages": tool_messages, "retrieved_facts": facts}

    def verify_node(state: BriefingState) -> dict:
        draft = _last_assistant_text(state["messages"])
        deterministic_issues = _find_tool_error_issues(state["retrieved_facts"])
        ground_truth = json.dumps(state["retrieved_facts"], default=str)
        verify_messages = [
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Ground truth (JSON):\n{ground_truth}\n\nDraft briefing:\n{draft}"},
        ]

        with tracer.span("verify", "llm_call") as ev:
            result = llm.chat(
                verify_messages,
                tools=[VERIFICATION_TOOL_SPEC],
                tool_choice={"type": "function", "function": {"name": VERIFICATION_TOOL_NAME}},
            )
            ev["usage"] = result.usage
            ev["cost_usd"] = estimate_cost_usd(llm.model, result.usage)

        attempts = state["verify_attempts"] + 1

        if not result.tool_calls:
            # The model didn't comply even with a forced tool call. Fail safe:
            # treat as unverified rather than assuming the draft is fine.
            issues = ["verifier did not return a structured report"] + deterministic_issues
            tracer.log("verify", "report", ok=False, issues=issues)
            return {
                "verified_ok": False,
                "verification_issues": issues,
                "verify_attempts": attempts,
                "messages": [{"role": "user", "content": "Verification could not be completed; please restate the briefing carefully, double-checking every fact against the tool results above."}],
            }

        report = VerificationReport.model_validate(result.tool_calls[0].arguments)
        ok = report.ok and not deterministic_issues
        issues = report.issues + deterministic_issues
        tracer.log("verify", "report", ok=ok, issues=issues, llm_ok=report.ok, deterministic_issues=deterministic_issues)

        feedback: list[dict] = []
        if not ok:
            feedback.append(
                {
                    "role": "user",
                    "content": "Verification found problems: " + "; ".join(issues)
                    + ". Revise the briefing, re-checking the tool results already retrieved above.",
                }
            )

        return {
            "verified_ok": ok,
            "verification_issues": issues,
            "verify_attempts": attempts,
            "messages": feedback,
        }

    def route_after_verify(state: BriefingState) -> str:
        if state["verified_ok"] or state["verify_attempts"] >= MAX_VERIFY_ATTEMPTS:
            return "finalize"
        return "agent"

    def finalize_node(state: BriefingState) -> dict:
        draft = _last_assistant_text(state["messages"])
        if state["verified_ok"]:
            final = draft
        else:
            caveat = (
                f"⚠️ Automatic fact-check could not fully confirm this briefing after "
                f"{state['verify_attempts']} attempt(s): {'; '.join(state['verification_issues'])}\n\n"
            )
            final = caveat + draft
        tracer.log("finalize", "final", final_message=final, verified_ok=state["verified_ok"])
        return {"final_message": final}

    graph = StateGraph(BriefingState)
    graph.add_node("plan", plan_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("verify", verify_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "verify": "verify"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", route_after_verify, {"finalize": "finalize", "agent": "agent"})
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=MemorySaver())


@dataclass
class GraphRun:
    final_message: str
    trace: list[dict]
    verify_attempts: int
    verified_ok: bool | None
    total_usage: dict
    total_cost_usd: float


def run_daily_briefing(
    store: Store,
    llm: LLMClient | None = None,
    tracer: Tracer | None = None,
    today: str | None = None,
) -> GraphRun:
    llm = llm or LLMClient()
    tracer = tracer or Tracer()
    today = today or date.today().isoformat()

    compiled = build_graph(store, llm, tracer)
    initial_state: BriefingState = {
        "messages": [],
        "today": today,
        "retrieved_facts": [],
        "agent_turns": 0,
        "verify_attempts": 0,
        "verified_ok": None,
        "verification_issues": [],
        "final_message": None,
    }
    config = {"configurable": {"thread_id": tracer.run_id}}
    result_state = compiled.invoke(initial_state, config=config)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_cost = 0.0
    for event in tracer.events:
        usage = event.get("usage")
        if usage:
            for key in total_usage:
                total_usage[key] += usage.get(key, 0)
        total_cost += event.get("cost_usd") or 0.0

    return GraphRun(
        final_message=result_state["final_message"] or "",
        trace=tracer.events,
        verify_attempts=result_state["verify_attempts"],
        verified_ok=result_state["verified_ok"],
        total_usage=total_usage,
        total_cost_usd=round(total_cost, 6),
    )
