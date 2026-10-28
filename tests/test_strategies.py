from demo.strategies import StrategyName, prepare_history, rough_token_estimate


def test_full_history_keeps_all_supported_messages():
    history = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "tool", "content": "should be omitted"},
    ]
    result = prepare_history(history, StrategyName.FULL_HISTORY)
    assert result.messages == history[:2]


def test_last_n_keeps_recent_messages():
    history = [{"role": "user", "content": str(i)} for i in range(8)]
    result = prepare_history(history, StrategyName.LAST_N, last_n=3)
    assert [item["content"] for item in result.messages] == ["5", "6", "7"]


def test_sliding_window_respects_rough_budget_when_possible():
    history = [{"role": "user", "content": "x" * 40} for _ in range(6)]
    result = prepare_history(history, StrategyName.SLIDING_WINDOW, token_budget=12)
    assert result.messages
    assert rough_token_estimate(result.messages) <= 12


def test_promptvault_mode_has_actionable_error_if_library_missing(monkeypatch):
    # This test is environment dependent; if PromptVault is installed it is skipped.
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "promptvault":
            raise ImportError("simulated missing package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    history = [{"role": "user", "content": "What was the fact?"}]
    try:
        prepare_history(history, StrategyName.PROMPTVAULT)
    except RuntimeError as exc:
        assert "PromptVault is not installed" in str(exc)
    else:
        raise AssertionError("Expected an actionable RuntimeError.")
