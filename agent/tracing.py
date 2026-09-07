"""
Local, zero-dependency run tracing: every LLM call and tool call gets one
JSON line in `traces/<run_id>.jsonl`, with timing/token/cost attached, so a
run is fully replayable/debuggable after the fact without an external
tracing service. This mirrors what a hosted tool like LangSmith gives you —
kept local here to match the project's "runs free, no external account
required" design.
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

TRACES_DIR = Path(__file__).parent.parent / "traces"


@dataclass
class Tracer:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    _events: list[dict] = field(default_factory=list, repr=False)

    def log(self, node: str, event_type: str, **fields) -> None:
        self._events.append({"ts": time.time(), "run_id": self.run_id, "node": node, "type": event_type, **fields})

    @contextmanager
    def span(self, node: str, event_type: str) -> Iterator[dict]:
        """Times a block, yielding a dict the caller fills in with whatever's
        only known after the call (usage, output, tool name...); logged as
        one event with an added `latency_s` field once the block exits."""
        start = time.perf_counter()
        fields: dict = {}
        try:
            yield fields
        finally:
            fields["latency_s"] = round(time.perf_counter() - start, 4)
            self.log(node, event_type, **fields)

    def flush(self, out_dir: Path = TRACES_DIR) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.run_id}.jsonl"
        with path.open("w") as f:
            for event in self._events:
                f.write(json.dumps(event, default=str) + "\n")
        return path

    @property
    def events(self) -> list[dict]:
        return list(self._events)
