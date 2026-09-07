from store.db import Store
from tools.registry import build_openai_tools, dispatch


def test_build_openai_tools_covers_all_registered_tools():
    specs = build_openai_tools()
    names = {s["function"]["name"] for s in specs}
    assert names == {
        "get_emails",
        "get_calendar_events",
        "get_tasks",
        "create_task",
        "update_task_status",
        "schedule_event",
        "search_notes",
        "draft_reply",
    }
    # every spec must carry a JSON schema the LLM can use to construct arguments
    for spec in specs:
        assert "parameters" in spec["function"]
        assert spec["function"]["parameters"]["type"] == "object"


def test_dispatch_unknown_tool_returns_error_not_exception(store: Store):
    result = dispatch(store, "not_a_real_tool", {})
    assert "error" in result


def test_dispatch_invalid_arguments_returns_error_not_exception(store: Store):
    result = dispatch(store, "update_task_status", {"task_id": "tk-001"})  # missing required 'status'
    assert "error" in result


def test_dispatch_get_tasks(store: Store):
    result = dispatch(store, "get_tasks", {"status": "open"})
    assert isinstance(result, list)
    assert all(t["status"] == "open" for t in result)


def test_dispatch_create_task_then_update(store: Store):
    created = dispatch(store, "create_task", {"title": "Follow up with Sam", "due_date": "2026-09-12"})
    assert "error" not in created
    task_id = created["id"]

    updated = dispatch(store, "update_task_status", {"task_id": task_id, "status": "done"})
    assert updated["status"] == "done"


def test_dispatch_update_unknown_task_id_is_reported_as_error(store: Store):
    result = dispatch(store, "update_task_status", {"task_id": "tk-nope", "status": "done"})
    assert "error" in result


def test_dispatch_draft_reply_requires_real_email_id(store: Store):
    ok = dispatch(store, "draft_reply", {"email_id": "em-001", "body": "Sending the deck now."})
    assert "error" not in ok

    bad = dispatch(store, "draft_reply", {"email_id": "em-does-not-exist", "body": "..."})
    assert "error" in bad


def test_dispatch_schedule_event(store: Store):
    ev = dispatch(
        store,
        "schedule_event",
        {"title": "Sync w/ dev team", "start": "2026-09-09T14:00:00", "end": "2026-09-09T14:30:00"},
    )
    assert "error" not in ev
    assert ev["title"] == "Sync w/ dev team"


def test_dispatch_normalizes_literal_null_string_to_none(store: Store):
    """Regression test for a real failure observed with a local model: it
    passed the string "null" instead of omitting an optional argument, which
    silently filtered out every row (str | None happily accepts "null" as a
    valid string) instead of erroring. Every open task must come back, not
    zero — before the fix this returned []."""
    result = dispatch(store, "get_tasks", {"status": "null"})
    assert isinstance(result, list)
    assert len(result) == 4  # same as get_tasks with status omitted entirely
    assert result == dispatch(store, "get_tasks", {})


def test_dispatch_coerces_json_stringified_list_argument(store: Store):
    """Regression test for another real small-model quirk (distinct from the
    null-sentinel one): a forced tool call for report_verification came back
    with `issues` as the *string* '["Meeting time discrepancy"]' instead of a
    native JSON array. Pydantic rejects a bare string for list[str] outright,
    which would otherwise turn an entirely correct tool call into a hard
    validation error. create_task's `tags` field uses the same StrList type,
    so this is tested here directly against dispatch."""
    result = dispatch(store, "create_task", {"title": "Test", "tags": '["reporting", "urgent"]'})
    assert "error" not in result
    assert result["tags"] == ["reporting", "urgent"]


def test_dispatch_coerces_single_element_list_to_scalar_string(store: Store):
    """Regression test for a third real quirk, the mirror image of the
    stringified-list one: observed live from the local model calling
    get_tasks(status=["open"]) instead of status="open" for a plain
    `str | None` filter field. Before this fix, Pydantic raised a hard
    validation error on an otherwise perfectly clear tool call."""
    result = dispatch(store, "get_tasks", {"status": ["open"]})
    assert "error" not in result
    assert isinstance(result, list)
    assert all(t["status"] == "open" for t in result)


def test_dispatch_search_notes(store: Store):
    result = dispatch(store, "search_notes", {"query": "migration"})
    assert any("migration" in n["body"].lower() or "migration" in n["title"].lower() for n in result)
