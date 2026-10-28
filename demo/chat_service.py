"""Chat history preparation helpers."""

from __future__ import annotations

from demo.strategies import PreparedHistory, StrategyName, prepare_history as _prepare_history


def prepare_history(
    history: list[dict[str, str]],
    strategy: StrategyName,
    *,
    token_budget: int = 2000,
    preserve_recent_n: int = 2,
) -> PreparedHistory:
    return _prepare_history(
        history,
        strategy,
        token_budget=token_budget,
        preserve_recent_n=preserve_recent_n,
    )
