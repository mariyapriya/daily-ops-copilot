"""Validates the eval scenario files themselves — fast, no LLM involved.
Catches a malformed scenario (typo'd category, missing required key, checks
with no way to ever fail or ever pass) before it wastes a slow live run."""

from eval.run_eval import load_scenarios

VALID_CATEGORIES = {"extraction", "conflict_detection", "hallucination_trap", "ambiguous_instruction"}
VALID_CHECK_KEYS = {"contains_all", "contains_none", "contains_any_of", "expect_verified_ok"}


def test_at_least_a_dozen_scenarios_exist():
    scenarios = load_scenarios()
    assert len(scenarios) >= 12


def test_scenario_ids_are_unique():
    scenarios = load_scenarios()
    ids = [s["id"] for s in scenarios]
    assert len(ids) == len(set(ids))


def test_every_scenario_has_required_top_level_keys():
    for s in load_scenarios():
        assert set(s.keys()) >= {"id", "category", "description", "today", "data", "checks"}, s.get("id")


def test_every_scenario_has_a_valid_category():
    for s in load_scenarios():
        assert s["category"] in VALID_CATEGORIES, f"{s['id']}: invalid category {s['category']!r}"


def test_every_scenario_data_has_all_four_lists():
    for s in load_scenarios():
        data = s["data"]
        assert set(data.keys()) >= {"emails", "calendar_events", "tasks", "notes"}, s["id"]
        for key in ("emails", "calendar_events", "tasks", "notes"):
            assert isinstance(data[key], list), f"{s['id']}.{key} must be a list"


def test_every_scenario_has_at_least_one_check():
    for s in load_scenarios():
        checks = s["checks"]
        assert checks, f"{s['id']} has no checks — it could never fail"
        assert set(checks.keys()) <= VALID_CHECK_KEYS, f"{s['id']} has an unrecognized check key"


def test_every_category_has_at_least_two_scenarios():
    scenarios = load_scenarios()
    for category in VALID_CATEGORIES:
        count = sum(1 for s in scenarios if s["category"] == category)
        assert count >= 2, f"category {category!r} has only {count} scenario(s)"
