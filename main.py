"""
CLI entrypoint: seeds the store from the synthetic fixtures and runs either
the Week-1 naive agent or the Week-2 LangGraph agent to produce a morning
briefing. Keeping both selectable via --engine is what makes the before/after
improvement from the verify node a concrete, runnable comparison rather than
an assertion in a README.
"""

from __future__ import annotations

import argparse
import json

from agent.llm import LLMClient
from store.db import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily-ops-copilot morning briefing agent.")
    parser.add_argument("--engine", choices=["naive", "graph"], default="graph", help="Which agent implementation to run.")
    parser.add_argument("--today", default=None, help="Override today's date (ISO), e.g. 2026-09-10.")
    parser.add_argument("--model", default=None, help="Override the LLM model name (env LLM_MODEL otherwise).")
    parser.add_argument("--show-trace", action="store_true", help="Print the full run trace.")
    args = parser.parse_args()

    store = Store(db_path=":memory:")
    store.seed_from_fixtures()
    llm = LLMClient(model=args.model)

    print("=" * 70)
    print(f"MORNING BRIEFING  (engine: {args.engine})")
    print("=" * 70)

    if args.engine == "naive":
        from agent.naive_agent import run_daily_briefing

        run = run_daily_briefing(store, llm=llm, today=args.today)
        print(run.final_message)
        print()
        print("-" * 70)
        print(f"turns used: {run.turns_used}   token usage: {run.total_usage}")
        trace = run.trace
    else:
        from agent.graph import run_daily_briefing
        from agent.tracing import Tracer

        tracer = Tracer()
        run = run_daily_briefing(store, llm=llm, tracer=tracer, today=args.today)
        print(run.final_message)
        print()
        print("-" * 70)
        print(
            f"verified_ok: {run.verified_ok}   verify attempts: {run.verify_attempts}   "
            f"token usage: {run.total_usage}   est. cost: ${run.total_cost_usd}"
        )
        trace_path = tracer.flush()
        print(f"trace written to: {trace_path}")
        trace = run.trace

    if args.show_trace:
        print("-" * 70)
        print("TRACE")
        for step in trace:
            print(json.dumps(step, indent=2, default=str))


if __name__ == "__main__":
    main()
