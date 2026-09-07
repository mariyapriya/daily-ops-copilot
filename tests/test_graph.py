from agent.graph import MAX_VERIFY_ATTEMPTS, _find_tool_error_issues, run_daily_briefing
from agent.llm import ChatResult, ToolCall
from agent.tracing import Tracer
from store.db import Store
from tests.fakes import FakeLLMClient
from tools.verification_schema import VERIFICATION_TOOL_NAME, VERIFICATION_TOOL_SPEC


def test_verification_tool_spec_shape():
    fn = VERIFICATION_TOOL_SPEC["function"]
    assert fn["name"] == VERIFICATION_TOOL_NAME
    params = fn["parameters"]
    assert params["type"] == "object"
    assert set(params["properties"]) == {"ok", "issues"}
    assert params["required"] == ["ok"]  # issues has a default, ok does not


def _verify_result(ok: bool, issues: list[str]) -> ChatResult:
    return ChatResult(
        content=None,
        tool_calls=[ToolCall(id="v1", name=VERIFICATION_TOOL_NAME, arguments={"ok": ok, "issues": issues})],
    )


def test_happy_path_no_retry_needed(store: Store):
    llm = FakeLLMClient(
        responses=[
            ChatResult(content="- check emails\n- check calendar\n- check tasks"),  # plan
            ChatResult(content="Here is your briefing: all clear."),  # agent (no tool calls)
            _verify_result(ok=True, issues=[]),  # verify
        ]
    )
    run = run_daily_briefing(store, llm=llm, tracer=Tracer(), today="2026-09-10")

    assert run.verified_ok is True
    assert run.verify_attempts == 1
    assert run.final_message == "Here is your briefing: all clear."
    assert not run.final_message.startswith("⚠️")
    assert len(llm.calls) == 3


def test_retry_then_success(store: Store):
    llm = FakeLLMClient(
        responses=[
            ChatResult(content="plan"),  # plan
            ChatResult(content="Draft v1 (wrong time)"),  # agent attempt 1
            _verify_result(ok=False, issues=["meeting time is wrong"]),  # verify attempt 1: fails
            ChatResult(content="Draft v2 (fixed)"),  # agent attempt 2, after feedback
            _verify_result(ok=True, issues=[]),  # verify attempt 2: passes
        ]
    )
    run = run_daily_briefing(store, llm=llm, tracer=Tracer(), today="2026-09-10")

    assert run.verified_ok is True
    assert run.verify_attempts == 2
    assert run.final_message == "Draft v2 (fixed)"
    assert not run.final_message.startswith("⚠️")

    # the feedback from the failed attempt must actually have reached the second agent call
    second_agent_call_messages = llm.calls[3]["messages"]
    assert any("meeting time is wrong" in m.get("content", "") for m in second_agent_call_messages)


def test_exhausted_retries_finalizes_with_caveat(store: Store):
    responses = [ChatResult(content="plan")]
    for i in range(MAX_VERIFY_ATTEMPTS):
        responses.append(ChatResult(content=f"Draft v{i + 1}"))
        responses.append(_verify_result(ok=False, issues=[f"issue {i + 1}"]))
    llm = FakeLLMClient(responses=responses)

    run = run_daily_briefing(store, llm=llm, tracer=Tracer(), today="2026-09-10")

    assert run.verified_ok is False
    assert run.verify_attempts == MAX_VERIFY_ATTEMPTS
    assert run.final_message.startswith("⚠️")
    assert f"Draft v{MAX_VERIFY_ATTEMPTS}" in run.final_message
    assert "could not fully confirm" in run.final_message


def test_tool_calling_loop_reaches_agent_again_before_verify(store: Store):
    llm = FakeLLMClient(
        responses=[
            ChatResult(content="plan"),  # plan
            ChatResult(content=None, tool_calls=[ToolCall(id="t1", name="get_tasks", arguments={"status": None})]),  # agent calls a tool
            ChatResult(content="Briefing using real task data."),  # agent, now with tool results in context
            _verify_result(ok=True, issues=[]),  # verify
        ]
    )
    run = run_daily_briefing(store, llm=llm, tracer=Tracer(), today="2026-09-10")

    assert run.final_message == "Briefing using real task data."
    assert run.verified_ok is True
    # the tool result must have actually been dispatched against the real store and fed back in
    third_call_messages = llm.calls[2]["messages"]
    assert any(m.get("role") == "tool" for m in third_call_messages)


def test_find_tool_error_issues_detects_failed_calls():
    facts = [
        {"tool": "get_emails", "arguments": {}, "output": [{"id": "em-001"}]},
        {"tool": "get_tasks", "arguments": {"status": 123}, "output": {"error": "invalid arguments"}},
    ]
    issues = _find_tool_error_issues(facts)
    assert len(issues) == 1
    assert "get_tasks" in issues[0]
    assert "no data found" in issues[0]


def test_find_tool_error_issues_empty_when_no_errors():
    facts = [{"tool": "get_emails", "arguments": {}, "output": []}]
    assert _find_tool_error_issues(facts) == []


def test_deterministic_guardrail_overrides_an_llm_verifier_that_misses_a_tool_error(store: Store):
    """Regression test for a real failure observed live: a tool call failed
    (bad arguments), the drafting model silently reported it as "no data,"
    and the LLM verifier said ok=True anyway — missing it. The deterministic
    check in verify_node must force ok=False regardless of what the LLM
    verifier says, since a failed tool call is checkable with certainty.

    The failed fact stays in `retrieved_facts` for the rest of the run (it's
    never re-fetched in this script), so the deterministic check keeps
    firing on both verify attempts and the run ends up exhausting retries —
    that's the correct, honest outcome: a caveated answer, not a silently
    wrong one and not a false "verified" claim either.
    """
    llm = FakeLLMClient(
        responses=[
            ChatResult(content="plan"),
            # agent calls get_tasks with a bad argument type -> dispatch will genuinely error
            ChatResult(content=None, tool_calls=[ToolCall(id="t1", name="get_tasks", arguments={"status": 123})]),
            ChatResult(content="Briefing: there are no open tasks to report."),  # false claim from a failed call
            _verify_result(ok=True, issues=[]),  # LLM verifier misses it, as observed live — attempt 1
            ChatResult(content="Briefing: there are no open tasks to report."),  # unchanged on retry
            _verify_result(ok=True, issues=[]),  # LLM verifier misses it again — attempt 2
        ]
    )
    run = run_daily_briefing(store, llm=llm, tracer=Tracer(), today="2026-09-10")

    assert run.verified_ok is False  # deterministic check must override the LLM's ok=True
    assert run.final_message.startswith("⚠️")
    assert "get_tasks" in run.final_message


def test_verifier_non_compliance_is_treated_as_unverified_not_crash(store: Store):
    """If the model doesn't return a tool call even when forced (still possible with
    a weak local model), the graph must fail safe rather than assume success or crash."""
    responses = [ChatResult(content="plan")]
    for _ in range(MAX_VERIFY_ATTEMPTS):
        responses.append(ChatResult(content="Draft"))
        responses.append(ChatResult(content="I refuse to call a tool.", tool_calls=[]))  # non-compliant verifier
    llm = FakeLLMClient(responses=responses)

    run = run_daily_briefing(store, llm=llm, tracer=Tracer(), today="2026-09-10")

    assert run.verified_ok is False
    assert run.final_message.startswith("⚠️")
