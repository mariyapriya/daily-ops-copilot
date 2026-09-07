"""
Shared Pydantic coercion helpers for small-model tool-calling quirks.

This started as a one-off fix (see `tools/registry.py`'s null-sentinel
normalization for `"null"`-as-a-string) but the same failure shape showed up
again independently: the local model, when asked to call a tool with a
`list[str]` argument, sometimes serializes it as a *JSON-encoded string*
(e.g. `"[\"issue one\"]"`) instead of a native JSON array. Pydantic's default
behavior for `list[str]` rejects a plain string outright — which is at least
loud (a clear validation error) rather than the null-sentinel bug's silent
wrong-answer failure mode, but it still needlessly breaks otherwise-correct
tool calls from smaller/local models. Centralizing the coercion here means
every `list[str]` tool argument gets the same tolerant handling instead of
each schema growing its own ad hoc validator.
"""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BeforeValidator


def _coerce_str_list(value: object) -> object:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # Not JSON at all — treat the whole string as a single item
            # rather than rejecting an otherwise-reasonable tool call.
            return [value] if value.strip() else []
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    return value  # let Pydantic raise its normal error for anything else


StrList = Annotated[list[str], BeforeValidator(_coerce_str_list)]


def _coerce_optional_scalar_str(value: object) -> object:
    """The mirror-image quirk: a field typed `str | None` (e.g. a filter
    like `status`) coming back wrapped in a single-element list, e.g.
    `["open"]` instead of `"open"`. Observed live from the local model on
    `get_tasks(status=...)` — it correctly omitted a *filter* concept but
    encoded it as a list rather than a bare string. Unwrap a one-element
    list; for anything longer, best-effort join rather than reject a call
    that's still semantically clear."""
    if isinstance(value, list):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return value[0]
        return ", ".join(str(v) for v in value)
    return value


OptionalStr = Annotated[str | None, BeforeValidator(_coerce_optional_scalar_str)]
