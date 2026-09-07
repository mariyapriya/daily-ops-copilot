"""
Streamlit dashboard for Daily Ops Copilot — the click-through demo layer on
top of the CLI. Three tabs: trigger a live agent run, explore any run's
trace, and browse eval-suite results. Reuses the exact same store/tools/
agent/eval code the CLI uses — this is a viewer and trigger, not a parallel
implementation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from agent.cost import estimate_cost_usd
from agent.graph import run_daily_briefing as run_graph
from agent.llm import LLMClient
from agent.naive_agent import run_daily_briefing as run_naive
from agent.tracing import Tracer
from eval.run_eval import load_scenarios, run_scenario, write_results
from store.db import Store

ROOT = Path(__file__).parent
TRACES_DIR = ROOT / "traces"
RESULTS_DIR = ROOT / "eval" / "results"
DATA_DIR = ROOT / "data"

# Status palette (validated categorical/status colors — see the project's
# dataviz-skill pass): PASS/good, SAFE_CAVEAT/warning, FAIL/critical. Fixed
# order, never cycled or reassigned by a filter.
STATUS_COLORS = {"PASS": "#0ca30c", "SAFE_CAVEAT": "#fab219", "FAIL": "#d03b3b"}
STATUS_ORDER = ["PASS", "SAFE_CAVEAT", "FAIL"]

# Categorical palette, slots 1-4, fixed order.
CATEGORY_COLORS = {
    "extraction": "#2a78d6",
    "conflict_detection": "#eb6834",
    "hallucination_trap": "#1baf7a",
    "ambiguous_instruction": "#eda100",
}

st.set_page_config(page_title="Daily Ops Copilot", layout="wide")
st.title("Daily Ops Copilot")
st.caption("Portfolio project — proactive daily-ops agent with a verify/guardrail node and a systematic eval suite.")


def load_data_sources() -> dict[str, dict]:
    """Every runnable dataset: the flagship Week 1/2 fixtures (the
    double-booking + stale-email story) plus every isolated eval scenario,
    so a visitor can pick any specific edge case to run live."""
    sources: dict[str, dict] = {
        "Default fixtures (double-booking + stale email)": {
            "today": "2026-09-10",
            "data": {
                "emails": json.loads((DATA_DIR / "emails.json").read_text()),
                "calendar_events": json.loads((DATA_DIR / "calendar.json").read_text()),
                "tasks": json.loads((DATA_DIR / "tasks.json").read_text()),
                "notes": json.loads((DATA_DIR / "notes.json").read_text()),
            },
        }
    }
    for scenario in load_scenarios():
        sources[f"[eval] {scenario['id']}"] = {"today": scenario["today"], "data": scenario["data"], "description": scenario.get("description")}
    return sources


def seed_store(data: dict) -> Store:
    store = Store(db_path=":memory:")
    store.seed_from_data(
        emails=data.get("emails", []),
        events=data.get("calendar_events", []),
        tasks=data.get("tasks", []),
        notes=data.get("notes", []),
    )
    return store


tab_run, tab_trace, tab_eval = st.tabs(["Run Briefing", "Trace Explorer", "Eval Dashboard"])

# ---------------------------------------------------------------------------
# Tab 1: Run Briefing (live)
# ---------------------------------------------------------------------------
with tab_run:
    st.subheader("Run a briefing live")
    st.write("Runs the real agent against a chosen dataset — this calls a live LLM backend and can take 15s-3min depending on the engine and model.")

    sources = load_data_sources()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        source_label = st.selectbox("Dataset", list(sources.keys()))
    with col2:
        engine = st.radio("Engine", ["graph", "naive"], help="graph = Week 2 orchestration + verify node. naive = Week 1 baseline, no guardrail.")
    with col3:
        model = st.text_input("Model", value="", placeholder="env default (llama3.2)")

    source = sources[source_label]
    if source.get("description"):
        st.caption(source["description"])
    today = st.text_input("Today's date (ISO)", value=source["today"])

    if st.button("Run briefing", type="primary"):
        store = seed_store(source["data"])
        llm = LLMClient(model=model or None)
        tracer = Tracer()

        with st.spinner(f"Running the {engine} engine... this can take a couple of minutes locally."):
            start = time.perf_counter()
            if engine == "graph":
                run = run_graph(store, llm=llm, tracer=tracer, today=today)
                verified_ok, verify_attempts, cost = run.verified_ok, run.verify_attempts, run.total_cost_usd
            else:
                # The Week 1 naive agent predates agent/tracing.py and keeps its own
                # simpler inline trace shape — it's genuinely not instrumented the
                # way the graph engine is, which is itself part of the Week 1→2
                # story, so it isn't forced into the Tracer/JSONL format here.
                run = run_naive(store, llm=llm, today=today)
                verified_ok, verify_attempts = None, None
                cost = estimate_cost_usd(llm.model, run.total_usage)
            elapsed = time.perf_counter() - start

        st.success(f"Done in {elapsed:.1f}s")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Verified", "—" if verified_ok is None else ("Yes" if verified_ok else "No, caveated"))
        m2.metric("Verify attempts", verify_attempts if verify_attempts is not None else "—")
        m3.metric("Tokens", run.total_usage["total_tokens"])
        m4.metric("Est. cost", f"${cost:.4f}")

        st.markdown("#### Briefing")
        st.markdown(run.final_message)

        if engine == "graph":
            trace_path = tracer.flush()
            st.session_state["last_trace_path"] = str(trace_path)
            st.caption(f"Trace written to `{trace_path.relative_to(ROOT)}` — open the Trace Explorer tab to inspect it.")
        else:
            with st.expander("Raw trace (naive agent's own format — no cost/latency tracking in Week 1)"):
                st.json(run.trace)

# ---------------------------------------------------------------------------
# Tab 2: Trace Explorer
# ---------------------------------------------------------------------------
with tab_trace:
    st.subheader("Inspect a run trace")
    st.write("Every LLM call and tool call in a run, with latency, tokens, and cost — replayable after the fact without an external tracing service.")

    trace_files = sorted(TRACES_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if TRACES_DIR.exists() else []
    default_idx = 0
    options = [str(p.relative_to(ROOT)) for p in trace_files]
    if "last_trace_path" in st.session_state and st.session_state["last_trace_path"] in [str(p) for p in trace_files]:
        default_idx = options.index(str(Path(st.session_state["last_trace_path"]).relative_to(ROOT)))

    if not options:
        st.info("No traces yet — run a briefing in the first tab to generate one.")
    else:
        chosen = st.selectbox("Trace file", options, index=default_idx)
        events = [json.loads(line) for line in (ROOT / chosen).read_text().splitlines() if line.strip()]

        if events:
            t0 = events[0]["ts"]
            rows = []
            for e in events:
                rows.append(
                    {
                        "t+": f"{e['ts'] - t0:6.1f}s",
                        "node": e.get("node"),
                        "type": e.get("type"),
                        "latency_s": e.get("latency_s"),
                        "tokens": (e.get("usage") or {}).get("total_tokens"),
                        "cost_usd": e.get("cost_usd"),
                        "detail": e.get("tool") or e.get("content") or "",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            with st.expander("Raw events (full JSON per step)"):
                for e in events:
                    st.json(e)

# ---------------------------------------------------------------------------
# Tab 3: Eval Dashboard
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("Eval suite results")
    st.write("Scores a 3-way outcome per scenario: PASS, SAFE_CAVEAT (the guardrail honestly refused to certify), or FAIL (confidently wrong).")

    result_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if RESULTS_DIR.exists() else []

    with st.expander("Run a single scenario live and add it to the results"):
        scenario_ids = [s["id"] for s in load_scenarios()]
        pick = st.selectbox("Scenario", scenario_ids, key="eval_scenario_pick")
        if st.button("Run this scenario"):
            scenario = next(s for s in load_scenarios() if s["id"] == pick)
            with st.spinner(f"Running {pick}..."):
                result = run_scenario(scenario, LLMClient())
                path = write_results([result])
            st.success(f"{result.outcome} — written to {path.relative_to(ROOT)}")
            result_files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not result_files:
        st.info("No eval results yet — run `python -m eval.run_eval` or use the button above.")
    else:
        options = [str(p.relative_to(ROOT)) for p in result_files]
        chosen = st.selectbox("Results file", options)
        results = json.loads((ROOT / chosen).read_text())
        df = pd.DataFrame(results)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Outcome breakdown")
            outcome_counts = df["outcome"].value_counts().reindex(STATUS_ORDER, fill_value=0).reset_index()
            outcome_counts.columns = ["outcome", "count"]
            chart = (
                alt.Chart(outcome_counts)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("outcome:N", sort=STATUS_ORDER, title=None),
                    y=alt.Y("count:Q", title="scenarios"),
                    color=alt.Color("outcome:N", scale=alt.Scale(domain=STATUS_ORDER, range=[STATUS_COLORS[o] for o in STATUS_ORDER]), legend=None),
                    tooltip=["outcome", "count"],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.markdown("##### Pass rate by category")
            cat_summary = (
                df.groupby("category")["outcome"]
                .apply(lambda s: (s == "PASS").sum() / len(s))
                .reset_index(name="pass_rate")
            )
            categories = list(CATEGORY_COLORS.keys())
            chart = (
                alt.Chart(cat_summary)
                .mark_bar(cornerRadius=4)
                .encode(
                    x=alt.X("category:N", sort=categories, title=None),
                    y=alt.Y("pass_rate:Q", title="pass rate", axis=alt.Axis(format="%")),
                    color=alt.Color("category:N", scale=alt.Scale(domain=categories, range=[CATEGORY_COLORS[c] for c in categories]), legend=None),
                    tooltip=["category", alt.Tooltip("pass_rate:Q", format=".0%")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total scenarios", len(df))
        m2.metric("Total latency", f"{df['latency_s'].sum():.0f}s")
        m3.metric("Total est. cost", f"${df['cost_usd'].sum():.4f}")

        st.markdown("##### Scenario detail")
        for _, row in df.sort_values("outcome", key=lambda s: s.map({o: i for i, o in enumerate(STATUS_ORDER)})).iterrows():
            color = STATUS_COLORS[row["outcome"]]
            with st.expander(f":{'green' if row['outcome']=='PASS' else 'orange' if row['outcome']=='SAFE_CAVEAT' else 'red'}[{row['outcome']}] {row['id']}"):
                st.caption(f"category: {row['category']}  ·  verified_ok: {row['verified_ok']}  ·  {row['latency_s']}s  ·  {row['total_tokens']} tokens")
                if row["failed_checks"]:
                    st.write("**Failed checks:**")
                    for c in row["failed_checks"]:
                        st.write(f"- {c}")
                st.write("**Final message:**")
                st.text(row["final_message"])
