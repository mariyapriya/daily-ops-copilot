"""
The verify node's structured-output contract. This is deliberately separate
from tools/schemas.py: it isn't a data-access tool the agent chooses to call
mid-reasoning, it's the forced output shape of a dedicated fact-checking LLM
call (see agent/graph.py's `verify` node).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tools.coercion import StrList


class VerificationReport(BaseModel):
    ok: bool = Field(description="True only if every factual claim in the draft is fully supported by the provided facts.")
    issues: StrList = Field(
        default_factory=list,
        description="One entry per problem found: an unsupported/incorrect claim, or a missed conflict. Empty if ok=True.",
    )


VERIFICATION_TOOL_NAME = "report_verification"

VERIFICATION_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": VERIFICATION_TOOL_NAME,
        "description": "Report the result of fact-checking a draft briefing against ground-truth data.",
        "parameters": VerificationReport.model_json_schema(),
    },
}
