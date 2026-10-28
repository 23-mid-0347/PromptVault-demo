"""Small evaluation utilities; network execution can be layered on after API validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from demo.metrics import score_case, token_savings


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Synthetic QA dataset root must be a JSON list.")
    for case in payload:
        for field in ("case_id", "messages", "question", "accepted_answers", "category"):
            if field not in case:
                raise ValueError(f"Case is missing required field: {field}")
    return payload


def make_run_record(
    *,
    case_id: str,
    strategy: str,
    model_id: str,
    answer: str,
    accepted_answers: list[str],
    full_input_tokens: int | None = None,
    strategy_input_tokens: int | None = None,
    estimated_tokens: int | None = None,
    latency_seconds: float | None = None,
    compression_seconds: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    score = score_case(answer, {"accepted_answers": accepted_answers})
    savings = token_savings(full_input_tokens, strategy_input_tokens)
    return {
        "case_id": case_id,
        "strategy": strategy,
        "model_id": model_id,
        "answer": answer,
        "correct": score["correct"],
        "full_provider_input_tokens": full_input_tokens,
        "strategy_provider_input_tokens": strategy_input_tokens,
        "provider_tokens_saved": savings["tokens_saved"],
        "provider_savings_pct": savings["savings_pct"],
        "estimated_prepared_tokens": estimated_tokens,
        "latency_seconds": latency_seconds,
        "compression_seconds": compression_seconds,
        "error": error,
    }


def export_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    if not records:
        raise ValueError("Cannot export an empty result set.")
    fieldnames = list(records[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
