from store.db import Store


def test_seed_loads_all_fixtures(store: Store):
    assert len(store.list_emails()) == 5
    assert len(store.list_calendar_events()) == 3
    assert len(store.list_tasks()) == 4
    assert len(store.search_notes("")) == 2  # empty LIKE '%%' matches all


def test_list_tasks_filters_by_status(store: Store):
    open_tasks = store.list_tasks(status="open")
    assert all(t["status"] == "open" for t in open_tasks)
    assert len(open_tasks) == 3  # tk-001, tk-003, tk-004 (tk-002 is done)


def test_update_task_status_rejects_invalid_status(store: Store):
    import pytest

    with pytest.raises(ValueError):
        store.update_task_status("tk-001", "not-a-real-status")


def test_update_task_status_roundtrip(store: Store):
    updated = store.update_task_status("tk-004", "done")
    assert updated is not None
    assert updated["status"] == "done"
    assert store.get_task("tk-004")["status"] == "done"


def test_update_unknown_task_returns_none(store: Store):
    assert store.update_task_status("tk-does-not-exist", "done") is None


def test_create_task_persists(store: Store):
    created = store.create_task(id="tk-new", title="New task", due_date="2026-09-20", created_at="2026-09-07T00:00:00")
    assert created["id"] == "tk-new"
    assert store.get_task("tk-new")["title"] == "New task"


def test_schedule_event_persists(store: Store):
    ev = store.schedule_event(id="ev-new", title="New meeting", start="2026-09-15T10:00:00", end="2026-09-15T11:00:00")
    found = [e for e in store.list_calendar_events() if e["id"] == "ev-new"]
    assert len(found) == 1
    assert found[0]["title"] == "New meeting"
    assert ev["status"] == "confirmed"


def test_search_notes_finds_by_keyword(store: Store):
    results = store.search_notes("Priya")
    assert any("Priya" in n["title"] for n in results)
    assert store.search_notes("no-such-keyword-xyz") == []


def test_double_booked_events_both_present(store: Store):
    """Fixture data intentionally double-books ev-001 and ev-002 at overlapping
    times on 2026-09-10 — the store must surface both, not silently dedupe or
    hide the conflict. Conflict *detection* is the agent's job, not the store's."""
    events = store.list_calendar_events(start="2026-09-10T00:00:00", end="2026-09-10T23:59:59")
    ids = {e["id"] for e in events}
    assert {"ev-001", "ev-002"}.issubset(ids)
