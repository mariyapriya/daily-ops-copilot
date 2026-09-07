"""
A single OpenAI-compatible LLM client. Because Ollama, OpenAI, and most other
providers all speak the same `/v1/chat/completions` shape, the agent code
never needs to know which backend it's talking to — only the base URL,
key, and model name change. Defaults to a local Ollama model so the whole
project runs for free with no API key; point it at OpenAI/Anthropic-
compatible endpoints via env vars for a hosted run.

This mirrors (and generalizes) the hand-rolled Ollama HTTP client in
ai-qa-agent/qa_agent/llm.py, but speaks the standard OpenAI wire format
instead of Ollama's native /api/generate, which is what makes swapping
providers a config change rather than a rewrite.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict | None = None


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")  # Ollama ignores the key but the SDK requires one
        self.model = model or os.getenv("LLM_MODEL", "llama3.2")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict = "auto",
        temperature: float = 0.2,
    ) -> ChatResult:
        """`tool_choice` follows the OpenAI wire format: "auto" (default),
        "none", "required", or a forced-call dict like
        `{"type": "function", "function": {"name": "some_tool"}}` — the last
        form is what the verify node uses to guarantee structured output
        instead of hoping the model calls the right tool on its own."""
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for tc in message.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                arguments = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ChatResult(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
        )
