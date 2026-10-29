"""Small, deterministic metrics that do not require an API call."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def normalized_text(text: str) -> str:
    """Normalize text for strict gold-answer matching.

    This deliberately does not use substring matching: a long response should not be marked
    correct merely because it happens to contain an accepted answer somewhere inside it.
    """
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    return normalized.strip(" \t\n\r.,!?;:\"'`()[]{}")


def answer_matches(answer: str, accepted_answers: Iterable[str]) -> bool:
    """Return whether an answer exactly matches one accepted answer after normalization."""
    candidate = normalized_text(answer)
    if not candidate:
        return False
    return any(candidate == normalized_text(expected) for expected in accepted_answers)


def token_savings(
    full_input_tokens: int | None,
    strategy_input_tokens: int | None,
) -> dict[str, int | float | None]:
    """Compare two provider-reported input-token counts, if both are available."""
    if full_input_tokens is None or strategy_input_tokens is None:
        return {"tokens_saved": None, "savings_pct": None}
    if full_input_tokens < 0 or strategy_input_tokens < 0:
        raise ValueError("Token counts must be non-negative.")

    saved = full_input_tokens - strategy_input_tokens
    pct = (saved / full_input_tokens * 100.0) if full_input_tokens else None
    return {
        "tokens_saved": saved,
        "savings_pct": round(pct, 2) if pct is not None else None,
    }
