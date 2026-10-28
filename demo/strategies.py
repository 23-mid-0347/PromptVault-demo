"""History preparation strategies.

PromptVault is imported lazily so the rest of the scaffold and tests remain usable
without the core library installed. Verify the assumed API against the actual package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum


class StrategyName(StrEnum):
    FULL_HISTORY = "full_history"
    LAST_N = "last_n"
    SLIDING_WINDOW = "sliding_window"
    PROMPTVAULT = "promptvault"


def rough_token_estimate(messages: list[dict[str, str]]) -> int:
    """A stable, rough token estimate based on text length, not a real tokenizer."""
    chars = sum(len(message.get("content", "")) for message in messages)
    return max(1, (chars + 3) // 4) if messages else 0


@dataclass(frozen=True, slots=True)
class PreparedHistory:
    messages: list[dict[str, str]]
    estimated_tokens: int
    compression_seconds: float
    original_tokens: int | None = None
    compressed_tokens: int | None = None
    compression_ratio: float | None = None
    notes: str = ""


def prepare_history(
    history: list[dict[str, str]],
    strategy: StrategyName,
    *,
    token_budget: int = 2000,
    preserve_recent_n: int = 2,
    last_n: int = 6,
) -> PreparedHistory:
    """Prepare role/content text messages according to one history strategy."""
    safe_history = [
        {"role": item["role"], "content": item["content"]}
        for item in history
        if item.get("role") in {"user", "assistant", "system"}
        and isinstance(item.get("content"), str)
    ]
    original_estimate = rough_token_estimate(safe_history)
    started = time.perf_counter()
    notes = ""

    if strategy == StrategyName.FULL_HISTORY:
        messages = safe_history
    elif strategy == StrategyName.LAST_N:
        messages = safe_history[-max(1, last_n):]
    elif strategy == StrategyName.SLIDING_WINDOW:
        messages = []
        for message in reversed(safe_history):
            candidate = [message, *messages]
            if rough_token_estimate(candidate) > token_budget and messages:
                break
            messages = candidate
        notes = "Sliding window uses a rough character-based token estimate."
    elif strategy == StrategyName.PROMPTVAULT:
        try:
            from promptvault import PromptVault
        except ImportError as exc:
            raise RuntimeError(
                "PromptVault is not installed in this environment. Install the core library "
                "and verify its public API before using PromptVault mode."
            ) from exc

        # The current plan proposes this API, but the actual checkout must be authoritative.
        vault = PromptVault(
            token_budget=token_budget,
            semantic_weight=0.6,
            lexical_weight=0.4,
            preserve_recent_n=preserve_recent_n,
        )
        if not safe_history or safe_history[-1]["role"] != "user":
            raise ValueError("PromptVault mode expects the current user question as the last message.")
        query = safe_history[-1]["content"]
        prior_history = safe_history[:-1]
        result = vault.compress(prior_history, query)
        compressed = getattr(result, "messages", None)
        if not isinstance(compressed, list):
            raise RuntimeError(
                "PromptVault result did not expose a list-valued `.messages`. "
                "Confirm the core library's current API."
            )
        messages = [
            {"role": item["role"], "content": item["content"]}
            for item in compressed
            if isinstance(item, dict)
            and item.get("role") in {"user", "assistant", "system"}
            and isinstance(item.get("content"), str)
        ]
        # Preserve the current question explicitly unless the library returns it itself.
        if not messages or messages[-1] != safe_history[-1]:
            messages.append(safe_history[-1])
        notes = "PromptVault fields are read only when present; verify their semantics in the core package."
        original_tokens = getattr(result, "original_tokens", None)
        compressed_tokens = getattr(result, "compressed_tokens", None)
        ratio = getattr(result, "compression_ratio", None)
        elapsed = time.perf_counter() - started
        return PreparedHistory(
            messages=messages,
            estimated_tokens=rough_token_estimate(messages),
            compression_seconds=elapsed,
            original_tokens=original_tokens if isinstance(original_tokens, int) else None,
            compressed_tokens=compressed_tokens if isinstance(compressed_tokens, int) else None,
            compression_ratio=ratio if isinstance(ratio, (int, float)) else None,
            notes=notes,
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    elapsed = time.perf_counter() - started
    return PreparedHistory(
        messages=messages,
        estimated_tokens=rough_token_estimate(messages),
        compression_seconds=elapsed,
        original_tokens=original_estimate,
        compressed_tokens=rough_token_estimate(messages),
        compression_ratio=(
            rough_token_estimate(messages) / original_estimate if original_estimate else 1.0
        ),
        notes=notes,
    )
