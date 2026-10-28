from pathlib import Path

from demo.evaluation import load_cases, make_run_record


def test_included_synthetic_dataset_has_required_fields():
    path = Path(__file__).parents[1] / "demo" / "datasets" / "synthetic_qa.json"
    cases = load_cases(path)
    assert len(cases) >= 4
    assert all(case["case_id"] and case["messages"] for case in cases)


def test_run_record_separates_provider_usage_from_estimates():
    record = make_run_record(
        case_id="case-1",
        strategy="full_history",
        model_id="test-model",
        answer="Silverstone",
        accepted_answers=["Silverstone"],
        full_input_tokens=100,
        strategy_input_tokens=100,
        estimated_tokens=95,
        latency_seconds=0.2,
        compression_seconds=0.0,
    )
    assert record["correct"] is True
    assert record["provider_tokens_saved"] == 0
    assert record["estimated_prepared_tokens"] == 95
