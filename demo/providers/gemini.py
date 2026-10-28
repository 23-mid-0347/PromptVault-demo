"""Gemini API adapter using Google's official google-genai SDK."""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types

from demo.providers.base import GenerationResult


class GeminiProvider:
    """Thin Gemini adapter. Provider usage remains separate from local token estimates."""

    def __init__(self, *, api_key: str, model_id: str) -> None:
        if not api_key.strip():
            raise ValueError("A Gemini API key is required.")
        self._client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationResult:
        contents: list[types.Content] = []
        system_text: list[str] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if not isinstance(content, str):
                continue
            if role == "system":
                system_text.append(content)
            elif role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role in {"assistant", "model"}:
                contents.append(types.Content(role="model", parts=[types.Part(text=content)]))

        if not contents:
            raise ValueError("No user/assistant conversation content was available to send.")

        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if system_text:
            config["system_instruction"] = "\n\n".join(system_text)

        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        )
        latency = time.perf_counter() - started

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = str(reason) if reason is not None else None

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Gemini returned no text. Check response safety/finish metadata.")

        return GenerationResult(
            text=text,
            provider="gemini",
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            request_latency_seconds=latency,
            finish_reason=finish_reason,
        )
