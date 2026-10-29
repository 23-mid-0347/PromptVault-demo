"""Shared response contract for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Normalized result returned by a provider adapter.

    Provider usage fields are ``None`` when the upstream API does not report usage.
    They must not be populated with locally estimated token counts.
    """

    text: str
    provider: str
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    request_latency_seconds: float
    finish_reason: str | None = None
