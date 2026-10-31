"""Conversation-history preparation strategies.

PromptVault integration follows the public API documented in
``PromptVault_demo.md`` V2. The exact core-library API still needs to be
checked against the local PromptVault checkout before this strategy can be
considered verified end-to-end.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class StrategyName(StrEnum):
    """Available history-preparation strategies."""

    FULL_HISTORY = "full_history"
    LAST_N = "last_n"
    SLIDING_WINDOW = "sliding_window"
    PROMPTVAULT = "promptvault"


@dataclass(slots=True)
class PreparedHistory:
    """Prepared messages and diagnostics consumed by the Streamlit app."""

    messages: list[dict[str, str]]
    estimated_tokens: int
    compression_seconds: float
    original_tokens: int | None
    compressed_tokens: int | None
    compression_ratio: float | None
    notes: list[str] = field(default_factory=list)


def _supported_messages(
    history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Keep supported text user/assistant messages without mutating input data."""
    messages: list[dict[str, str]] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue

        if not content.strip():
            continue

        messages.append({"role": role, "content": content})

    return messages


def rough_token_estimate(messages: list[dict[str, str]]) -> int:
    """Estimate text tokens using roughly one token per four characters.

    This lightweight diagnostic is not a tokenizer and does not represent
    provider-reported usage. Each non-empty message counts as at least one
    token.
    """
    total = 0

    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str) and content:
            total += max(1, math.ceil(len(content) / 4))

    return total


def _ratio(original: int, compressed: int) -> float | None:
    """Return compressed/original, or None when the original is empty."""
    return compressed / original if original > 0 else None


def _last_user_query(
    messages: list[dict[str, str]],
    query: str | None,
) -> str:
    """Resolve the query explicitly or from the latest user message."""
    if isinstance(query, str) and query.strip():
        return query.strip()

    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]

    raise ValueError(
        "PromptVault needs the current question. Add a user message "
        "or pass query= explicitly."
    )


def _promptvault_history(
    messages: list[dict[str, str]],
    *,
    token_budget: int,
    preserve_recent_n: int,
    query: str | None,
    semantic_weight: float,
    lexical_weight: float,
) -> PreparedHistory:
    """Call the PromptVault library and normalize its result for the app."""
    try:
        # pyrefly: ignore[missing-import]
        from promptvault import PromptVault
    except ImportError as exc:
        raise RuntimeError(
            "PromptVault is not installed in this Python environment. Activate "
            "the demo .venv, then install the core PromptVault package from its "
            "actual local checkout and verify its public API before retrying. "
            "No compression was performed."
        ) from exc

    resolved_query = _last_user_query(messages, query)
    original_estimate = rough_token_estimate(messages)
    started = time.perf_counter()

    try:
        vault = PromptVault(
            token_budget=token_budget,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            preserve_recent_n=preserve_recent_n,
        )
        result = vault.compress(deepcopy(messages), resolved_query)
    except Exception as exc:
        raise RuntimeError(
            "PromptVault could not compress this history. Check the installed "
            "core library's constructor, compress(history, query) signature, "
            "and dependency versions. The app will not silently fall back to "
            "uncompressed history."
        ) from exc

    elapsed = time.perf_counter() - started
    compressed_raw = getattr(result, "messages", None)

    if not isinstance(compressed_raw, list):
        raise RuntimeError(
            "PromptVault returned no list in result.messages. Compare the "
            "installed core API with the API described in PromptVault_demo.md "
            "before continuing."
        )

    compressed_messages = _supported_messages(compressed_raw)

    if not compressed_messages and messages:
        raise RuntimeError(
            "PromptVault returned no supported text user/assistant messages. "
            "This demo currently supports plain-text user/assistant history only."
        )

    notes: list[str] = [
        "PromptVault core compression was invoked; confirm this API "
        "against the checked-out core library.",
        "estimated_tokens is a local character-based estimate, not provider usage.",
    ]

    original_tokens = getattr(result, "original_tokens", None)
    compressed_tokens = getattr(result, "compressed_tokens", None)
    compression_ratio = getattr(result, "compression_ratio", None)

    if not isinstance(original_tokens, int):
        original_tokens = original_estimate
        notes.append(
            "Original token count was not exposed as an integer; "
            "using a rough estimate."
        )

    if not isinstance(compressed_tokens, int):
        compressed_tokens = rough_token_estimate(compressed_messages)
        notes.append(
            "Compressed token count was not exposed as an integer; "
            "using a rough estimate."
        )

    if not isinstance(compression_ratio, (int, float)):
        compression_ratio = None
        notes.append(
            "PromptVault did not expose a numeric compression_ratio; "
            "left unavailable."
        )
    else:
        compression_ratio = float(compression_ratio)

    return PreparedHistory(
        messages=compressed_messages,
        estimated_tokens=rough_token_estimate(compressed_messages),
        compression_seconds=elapsed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=compression_ratio,
        notes=notes,
    )


def prepare_history(
    history: list[dict[str, Any]],
    strategy: StrategyName | str,
    *,
    token_budget: int = 2000,
    preserve_recent_n: int = 2,
    last_n: int = 6,
    query: str | None = None,
    semantic_weight: float = 0.6,
    lexical_weight: float = 0.4,
) -> PreparedHistory:
    """Prepare history using one of the supported strategies.

    ``last_n`` means messages, not user/assistant exchanges. The PromptVault
    strategy delegates to the core library and raises an actionable error if
    the library is unavailable or fails.
    """
    try:
        selected = (
            strategy
            if isinstance(strategy, StrategyName)
            else StrategyName(strategy)
        )
    except ValueError as exc:
        allowed = ", ".join(item.value for item in StrategyName)
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose one of: {allowed}."
        ) from exc

    if token_budget < 0:
        raise ValueError("token_budget must be non-negative.")

    if preserve_recent_n < 0:
        raise ValueError("preserve_recent_n must be non-negative.")

    if last_n < 0:
        raise ValueError("last_n must be non-negative.")

    if (
        not 0.0 <= semantic_weight <= 1.0
        or not 0.0 <= lexical_weight <= 1.0
    ):
        raise ValueError(
            "semantic_weight and lexical_weight must each be between 0 and 1."
        )

    messages = _supported_messages(history)
    original_estimate = rough_token_estimate(messages)

    if selected is StrategyName.PROMPTVAULT:
        return _promptvault_history(
            messages,
            token_budget=token_budget,
            preserve_recent_n=preserve_recent_n,
            query=query,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

    started = time.perf_counter()
    notes: list[str] = [
        "Token counts are rough character-based estimates, not provider usage."
    ]

    if selected is StrategyName.FULL_HISTORY:
        prepared = messages
        notes.append(
            "Full history selected; no history compression was applied."
        )

    elif selected is StrategyName.LAST_N:
        prepared = messages[-last_n:] if last_n else []
        notes.append(
            f"Last-N keeps at most {last_n} messages; N does not mean exchanges."
        )

    else:
        if token_budget == 0:
            prepared = []
            notes.append("A zero token budget leaves no messages to send.")

        else:
            prepared_reversed: list[dict[str, str]] = []
            used = 0

            for message in reversed(messages):
                cost = rough_token_estimate([message])

                if used + cost <= token_budget:
                    prepared_reversed.append(message)
                    used += cost
                    continue

                if not prepared_reversed:
                    # Keep the tail of an oversized newest message within budget.
                    max_chars = token_budget * 4
                    clipped = {
                        "role": message["role"],
                        "content": message["content"][-max_chars:],
                    }
                    prepared_reversed.append(clipped)
                    notes.append(
                        "The newest message exceeded the rough budget and was "
                        "truncated from the left."
                    )

                break

            prepared = list(reversed(prepared_reversed))

        notes.append(
            f"Sliding window uses a rough budget of {token_budget} tokens."
        )

    elapsed = time.perf_counter() - started
    prepared_estimate = rough_token_estimate(prepared)

    return PreparedHistory(
        messages=prepared,
        estimated_tokens=prepared_estimate,
        compression_seconds=elapsed,
        original_tokens=original_estimate,
        compressed_tokens=prepared_estimate,
        compression_ratio=_ratio(original_estimate, prepared_estimate),
        notes=notes,
    )