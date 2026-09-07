"""
CLI entrypoint: seeds the store from the synthetic fixtures and runs the
Week-1 naive agent to produce a morning briefing, printing both the final
answer and the full tool-call trace (a stand-in for real tracing/observability,
built out properly in a later milestone).
"""

from __future__ import annotations

import argparse
import json

from agent.llm import LLMClient
from agent.naive_agent import run_daily_briefing
from store.db import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily-ops-copilot morning briefing agent.")
    parser.add_argument("--today", default=None, help="Override today's date (ISO), e.g. 2026-09-10.")
    parser.add_argument("--model", default=None, help="Override the LLM model name (env LLM_MODEL otherwise).")
    parser.add_argument("--show-trace", action="store_true", help="Print the full tool-call trace.")
    args = parser.parse_args()

    store = Store(db_path=":memory:")
    store.seed_from_fixtures()

    llm = LLMClient(model=args.model)
    run = run_daily_briefing(store, llm=llm, today=args.today)

    print("=" * 70)
    print("MORNING BRIEFING")
    print("=" * 70)
    print(run.final_message)
    print()
    print("-" * 70)
    print(f"turns used: {run.turns_used}   token usage: {run.total_usage}")

    if args.show_trace:
        print("-" * 70)
        print("TRACE")
        for step in run.trace:
            print(json.dumps(step, indent=2, default=str))


if __name__ == "__main__":
    main()
