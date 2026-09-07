"""
A minimal $/token lookup so every run can report an estimated cost, even
though the default local Ollama backend is free. Swapping `LLM_MODEL` to a
hosted model gets a real cost estimate with zero code changes — the point
isn't pricing precision (rates drift), it's that cost is tracked as a
first-class metric per run, per the JD's "optimise ... for quality, latency,
and cost."
"""

from __future__ import annotations

# (input $ per 1K tokens, output $ per 1K tokens). Approximate, illustrative —
# not guaranteed current; the point is the mechanism, not the exact numbers.
_RATES_PER_1K: dict[str, tuple[float, float]] = {
    "llama3.2": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-5-sonnet": (0.003, 0.015),
}

_DEFAULT_RATE = (0.0, 0.0)  # unknown/local models are treated as free rather than guessed


def estimate_cost_usd(model: str, usage: dict | None) -> float:
    if not usage:
        return 0.0
    input_rate, output_rate = _RATES_PER_1K.get(model, _DEFAULT_RATE)
    prompt_cost = (usage.get("prompt_tokens", 0) / 1000) * input_rate
    completion_cost = (usage.get("completion_tokens", 0) / 1000) * output_rate
    return round(prompt_cost + completion_cost, 6)
