from demo.metrics import answer_matches, normalized_text, token_savings


def test_token_savings_uses_provider_counts_when_available():
    assert token_savings(100, 70) == {"tokens_saved": 30, "savings_pct": 30.0}


def test_token_savings_marks_missing_usage_unavailable():
    assert token_savings(None, 70) == {"tokens_saved": None, "savings_pct": None}


def test_token_savings_handles_zero_baseline():
    assert token_savings(0, 0) == {"tokens_saved": 0, "savings_pct": None}


def test_normalized_answer_match_ignores_case_and_extra_whitespace():
    assert normalized_text("  SilverSTONE   ") == "silverstone"
    assert answer_matches(" Silverstone ", ["silverstone", "Spa"])
