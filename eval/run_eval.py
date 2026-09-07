"""
Automated eval harness: runs the Week-2 graph agent against every labeled
scenario in eval/scenarios/, scores the output, and reports a 3-way outcome
per scenario rather than a flat pass/fail:

- PASS: content checks passed and the guardrail's own `verified_ok` matched
  what was expected.
- SAFE_CAVEAT: the guardrail itself flagged the run as unverified (honest
  uncertainty) instead of confidently asserting something wrong. Treated as
  distinct from FAIL because "I'm not sure" is categorically safer than
  "I'm sure" + wrong — the JD's "predictable, observable, and safe actions."
- FAIL: `verified_ok` was True but the content checks failed anyway — a
  confidently wrong answer, exactly what the guardrail exists to prevent.

Re-run after any prompt/graph change to catch regressions:
    python -m eval.run_eval
    python -m eval.run_eval --scenario hallucination-stale-email-time
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.graph import run_daily_briefing
from agent.llm import LLMClient
from agent.tracing import Tracer
from store.db import Store

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class ScenarioResult:
    id: str
    category: str
    outcome: str  # PASS | SAFE_CAVEAT | FAIL
    verified_ok: bool | None
    verify_attempts: int
    failed_checks: list[str]
    latency_s: float
    total_tokens: int
    cost_usd: float
    final_message: str


def _check_text(text: str, checks: dict) -> list[str]:
    """Human-readable failure descriptions; empty if every content check
    passes. Case-insensitive substring matching — only reliable because
    each scenario's fixture is minimal/isolated, which narrows the range of
    reasonable correct phrasing enough to check deterministically."""
    lower = text.lower()
    failures = []

    for needle in checks.get("contains_all", []):
        if needle.lower() not in lower:
            failures.append(f"missing required text: {needle!r}")

    for needle in checks.get("contains_none", []):
        if needle.lower() in lower:
            failures.append(f"contains forbidden text: {needle!r}")

    any_of = checks.get("contains_any_of", [])
    if any_of and not any(needle.lower() in lower for needle in any_of):
        failures.append(f"none of the expected alternatives present: {any_of!r}")

    return failures


def load_scenarios() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(SCENARIOS_DIR.glob("*.json"))]


def run_scenario(scenario: dict, llm: LLMClient) -> ScenarioResult:
    data = scenario["data"]
    store = Store(db_path=":memory:")
    store.seed_from_data(
        emails=data.get("emails", []),
        events=data.get("calendar_events", []),
        tasks=data.get("tasks", []),
        notes=data.get("notes", []),
    )

    tracer = Tracer()
    start = time.perf_counter()
    run = run_daily_briefing(store, llm=llm, tracer=tracer, today=scenario["today"])
    latency = time.perf_counter() - start

    checks = scenario.get("checks", {})
    failures = _check_text(run.final_message, checks)

    expected_verified = checks.get("expect_verified_ok")
    if expected_verified is not None and run.verified_ok != expected_verified:
        failures.append(f"expected verified_ok={expected_verified}, got {run.verified_ok}")

    if not failures:
        outcome = "PASS"
    elif run.verified_ok is False:
        outcome = "SAFE_CAVEAT"
    else:
        outcome = "FAIL"

    return ScenarioResult(
        id=scenario["id"],
        category=scenario["category"],
        outcome=outcome,
        verified_ok=run.verified_ok,
        verify_attempts=run.verify_attempts,
        failed_checks=failures,
        latency_s=round(latency, 2),
        total_tokens=run.total_usage["total_tokens"],
        cost_usd=run.total_cost_usd,
        final_message=run.final_message,
    )


def print_summary(results: list[ScenarioResult]) -> None:
    total = len(results)
    by_outcome = {"PASS": 0, "SAFE_CAVEAT": 0, "FAIL": 0}
    for r in results:
        by_outcome[r.outcome] += 1

    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    print(f"total scenarios: {total}")
    for outcome, count in by_outcome.items():
        pct = 100 * count / total if total else 0
        print(f"  {outcome:12s}: {count:3d}  ({pct:.0f}%)")

    print("\nby category:")
    for cat in sorted({r.category for r in results}):
        cat_results = [r for r in results if r.category == cat]
        passes = sum(1 for r in cat_results if r.outcome == "PASS")
        print(f"  {cat:22s}: {passes}/{len(cat_results)} passed")

    total_cost = sum(r.cost_usd for r in results)
    total_latency = sum(r.latency_s for r in results)
    print(f"\ntotal latency: {total_latency:.1f}s   total est. cost: ${total_cost:.4f}")

    fails = [r for r in results if r.outcome == "FAIL"]
    if fails:
        print("\nFAIL (confidently wrong — highest priority to fix):")
        for r in fails:
            print(f"  - {r.id}: {r.failed_checks}")

    caveats = [r for r in results if r.outcome == "SAFE_CAVEAT"]
    if caveats:
        print("\nSAFE_CAVEAT (honest uncertainty, not silently wrong):")
        for r in caveats:
            print(f"  - {r.id}: {r.failed_checks}")


def write_results(results: list[ScenarioResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{int(time.time())}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nresults written to: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily-ops-copilot eval suite.")
    parser.add_argument("--scenario", default=None, help="Run only this scenario id.")
    parser.add_argument("--model", default=None, help="Override the LLM model name.")
    args = parser.parse_args()

    scenarios = load_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            raise SystemExit(f"no scenario found with id {args.scenario!r}")

    llm = LLMClient(model=args.model)
    results: list[ScenarioResult] = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] running {scenario['id']} ...", flush=True)
        result = run_scenario(scenario, llm)
        results.append(result)
        print(f"    -> {result.outcome} ({result.latency_s}s, {result.total_tokens} tokens)")
        for f in result.failed_checks:
            print(f"       - {f}")

    print_summary(results)
    write_results(results)

    if any(r.outcome == "FAIL" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
