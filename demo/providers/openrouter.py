"""OpenRouter chat-completions adapter using the OpenAI-compatible API."""

from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from demo.providers.base import GenerationResult


class OpenRouterProvider:
    """Call an OpenRouter model and preserve provider-reported usage."""

    def __init__(self, *, api_key: str, model_id: str) -> None:
        if not api_key.strip():
            raise ValueError("An OpenRouter API key is required.")
        if not model_id.strip():
            raise ValueError("An OpenRouter model ID is required.")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=120.0,
            max_retries=2,
            default_headers={
                "HTTP-Referer": "https://github.com/23-mid-0347/PromptVault-demo",
                "X-OpenRouter-Title": "PromptVault Demo",
            },
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
            raise ValueError("No user message was available to send to OpenRouter.")

        started = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        latency = time.perf_counter() - started

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenRouter returned no completion choices.")

        choice = choices[0]
        message = getattr(choice, "message", None)
        response_text = getattr(message, "content", None) if message else None
        if not isinstance(response_text, str) or not response_text.strip():
            raise RuntimeError(
                "OpenRouter returned no text. Check the selected model and response."
            )

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        finish_reason = getattr(choice, "finish_reason", None)

        return GenerationResult(
            text=response_text,
            provider="openrouter",
            model_id=self.model_id,
            input_tokens=(
                input_tokens if isinstance(input_tokens, int) else None
            ),
            output_tokens=(
                output_tokens if isinstance(output_tokens, int) else None
            ),
            request_latency_seconds=latency,
            finish_reason=(
                str(finish_reason) if finish_reason is not None else None
            ),
        )
