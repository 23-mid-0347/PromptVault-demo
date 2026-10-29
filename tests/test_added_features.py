from pathlib import Path

from demo.evaluation import load_cases, make_run_record
from demo.metrics import token_savings
from demo.strategies import StrategyName, prepare_history, rough_token_estimate


def test_dataset_loads_six_synthetic_cases():
    path = Path(__file__).parents[1] / "demo" / "datasets" / "synthetic_qa.json"
    cases = load_cases(path)
    assert len(cases) == 6
    assert all(case["question"] and case["accepted_answers"] for case in cases)


def test_provider_counts_stay_distinct_from_local_estimate():
    record = make_run_record(
        case_id="test-case",
        strategy="sliding_window",
        model_id="example/model",
        answer="Silverstone",
        accepted_answers=["Silverstone"],
        full_input_tokens=100,
        strategy_input_tokens=70,
        estimated_tokens=18,
        latency_seconds=0.5,
        compression_seconds=0.01,
    )
    assert record["provider_tokens_saved"] == 30
    assert record["estimated_prepared_tokens"] == 18
    assert record["strategy_provider_input_tokens"] == 70


def test_missing_provider_counts_are_not_guessed_from_estimates():
    assert token_savings(None, 70) == {"tokens_saved": None, "savings_pct": None}


def test_sliding_window_output_is_roughly_within_budget():
    history = [{"role": "user", "content": f"message {i} " + "x" * 35} for i in range(8)]
    result = prepare_history(history, StrategyName.SLIDING_WINDOW, token_budget=12)
    assert result.messages
    assert rough_token_estimate(result.messages) <= 12


def test_last_n_does_not_change_the_original_history():
    history = [{"role": "user", "content": str(i)} for i in range(8)]
    before = [dict(item) for item in history]
    prepare_history(history, StrategyName.LAST_N, last_n=2)
    assert history == before
