"""Provider-neutral interfaces and response model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    provider: str
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    request_latency_seconds: float
    finish_reason: str | None = None


class ChatProvider(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> GenerationResult:
        """Generate an assistant response from role/content messages."""
