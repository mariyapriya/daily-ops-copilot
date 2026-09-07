"""A scripted stand-in for LLMClient so graph/routing tests don't depend on
a live Ollama server — each test pre-scripts the exact sequence of
responses the graph is expected to request, in order."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.llm import ChatResult


@dataclass
class FakeLLMClient:
    responses: list[ChatResult]
    model: str = "fake-model"
    calls: list[dict] = field(default_factory=list)

    def chat(self, messages, tools=None, tool_choice="auto", temperature=0.2) -> ChatResult:
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if not self.responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self.responses.pop(0)
