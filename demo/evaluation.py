"""Dataset loading and serializable evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from demo.metrics import answer_matches, token_savings


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate a JSON list of synthetic QA cases.

    The file may be either a list of cases or an object containing a ``cases`` list.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Synthetic QA dataset not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Synthetic QA dataset is not valid JSON: {source}") from exc

    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("Synthetic QA dataset must be a list or an object with a 'cases' list.")

    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Dataset case at index {index} must be a JSON object.")
        case_id = case.get("case_id")
        messages = case.get("messages")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Dataset case at index {index} needs a non-empty case_id.")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case_id in synthetic QA dataset: {case_id}")
        if not isinstance(messages, list):
            raise ValueError(f"Dataset case {case_id!r} needs a messages list.")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Dataset case {case_id!r} needs a non-empty question.")
        if not isinstance(case.get("accepted_answers"), list):
            raise ValueError(f"Dataset case {case_id!r} needs an accepted_answers list.")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError(
                    f"Every message in case {case_id!r} must contain "
                    "a user/assistant role and text content."
                )
        seen_ids.add(case_id)

    return cases


def make_run_record(
    *,
    case_id: str,
    strategy: str,
    model_id: str,
    answer: str,
    accepted_answers: list[str],
    full_input_tokens: int | None,
    strategy_input_tokens: int | None,
    estimated_tokens: int | None,
    latency_seconds: float | None,
    compression_seconds: float | None,
    provider: str = "openrouter",
    category: str | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build a flat record suitable for JSONL/CSV export.

    Provider-reported prompt-token counts and local estimates remain separate fields.
    ``full_input_tokens`` and ``strategy_input_tokens`` must be values reported by the provider,
    not rough local estimates.
    """
    savings = token_savings(full_input_tokens, strategy_input_tokens)
    return {
        "case_id": case_id,
        "category": category,
        "strategy": strategy,
        "provider": provider,
        "model_id": model_id,
        "answer": answer,
        "accepted_answers": list(accepted_answers),
        "correct": answer_matches(answer, accepted_answers),
        "full_provider_input_tokens": full_input_tokens,
        "strategy_provider_input_tokens": strategy_input_tokens,
        "provider_tokens_saved": savings["tokens_saved"],
        "provider_savings_pct": savings["savings_pct"],
        "estimated_prepared_tokens": estimated_tokens,
        "latency_seconds": latency_seconds,
        "compression_seconds": compression_seconds,
        "notes": list(notes or []),
    }
