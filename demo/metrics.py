"""Pure metric helpers with explicit handling for unavailable measurements."""

from __future__ import annotations

from typing import Any


def token_savings(full_tokens: int | None, strategy_tokens: int | None) -> dict[str, float | int | None]:
    if full_tokens is None or strategy_tokens is None:
        return {"tokens_saved": None, "savings_pct": None}
    saved = full_tokens - strategy_tokens
    pct = (100.0 * saved / full_tokens) if full_tokens > 0 else None
    return {"tokens_saved": saved, "savings_pct": pct}


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def answer_matches(answer: str, accepted_answers: list[str]) -> bool:
    normalized = normalized_text(answer)
    return any(normalized == normalized_text(expected) for expected in accepted_answers)


def score_case(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    accepted = case.get("accepted_answers", [])
    if not isinstance(accepted, list) or not all(isinstance(item, str) for item in accepted):
        raise ValueError("accepted_answers must be a list of strings.")
    expected_abstention = bool(case.get("unanswerable", False))
    correct = (
        not normalized_text(answer)
        or any(
            phrase in normalized_text(answer)
            for phrase in ("not provided", "don't know", "do not know", "cannot determine")
        )
    ) if expected_abstention else answer_matches(answer, accepted)
    return {"correct": bool(correct), "unanswerable": expected_abstention}
