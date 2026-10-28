"""xAI Grok API adapter using the official OpenAI-compatible Python SDK."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from demo.providers.base import GenerationResult


class GrokProvider:
    """Thin Grok adapter; provider usage stays separate from local token estimates."""

    def __init__(self, *, api_key: str, model_id: str) -> None:
        if not api_key.strip():
            raise ValueError("An xAI API key is required.")
        if not model_id.strip():
            raise ValueError("A Grok model ID is required.")

        # The OpenAI SDK is used with xAI's documented API-compatible endpoint.
        # A small SDK retry budget handles transient 429/5xx responses.
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=180.0,
            max_retries=2,
        )
        self.model_id = model_id

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationResult:
        api_messages: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role", "")
            content: Any = message.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if role == "model":
                role = "assistant"
            if role in {"system", "developer", "user", "assistant"}:
                api_messages.append({"role": role, "content": content})

        if not any(message["role"] == "user" for message in api_messages):
            raise ValueError("No user message was available to send to Grok.")

        started = time.perf_counter()
        response = self._client.responses.create(
            model=self.model_id,
            input=api_messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            # The demo maintains its own history and uses synthetic test data.
            # Avoid server-side response storage when the API supports this option.
            store=False,
        )
        latency = time.perf_counter() - started

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(
                "Grok returned no text. Check the model's response status and API diagnostics."
            )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        status = getattr(response, "status", None)

        return GenerationResult(
            text=text,
            provider="xai",
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_latency_seconds=latency,
            finish_reason=str(status) if status is not None else None,
        )
