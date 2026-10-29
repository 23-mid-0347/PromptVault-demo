"""Shared result type returned by model provider adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Model response plus provider-reported usage and timing metadata."""

    text: str
    provider: str
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    request_latency_seconds: float
    finish_reason: str | None = None
