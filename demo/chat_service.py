"""Chat-level orchestration separated from the Streamlit user interface."""

from __future__ import annotations

from typing import Any

from demo.strategies import PreparedHistory, StrategyName
from demo.strategies import prepare_history as _prepare_history


def prepare_history(
    history: list[dict[str, Any]],
    strategy: StrategyName | str,
    *,
    token_budget: int = 2000,
    preserve_recent_n: int = 2,
    last_n: int = 6,
    query: str | None = None,
) -> PreparedHistory:
    """Prepare conversation context for the selected history strategy.

    This wrapper is kept as the stable import used by ``app.py``. The underlying strategy module
    contains the token-budget logic and the explicitly marked PromptVault adapter.
    """
    return _prepare_history(
        history,
        strategy,
        token_budget=token_budget,
        preserve_recent_n=preserve_recent_n,
        last_n=last_n,
        query=query,
    )
