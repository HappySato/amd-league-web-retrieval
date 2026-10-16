from __future__ import annotations

from eval.evaluate import main as evaluate_main
from eval.evaluate import percentile, summarize
from eval.scoring import score_answer


def test_scoring_exact_contains_and_refusal() -> None:
    assert score_answer("Paris", ["Paris"]) == {"exact": True, "contains": True}
    assert score_answer("paris.", ["Paris"])["exact"] is True
    assert score_answer("The answer is Paris", ["Paris"]) == {"exact": False, "contains": True}
    assert score_answer("8", ["8", "eight"])["exact"] is True
    assert score_answer("eight", ["8", "eight"])["exact"] is True
    assert score_answer("red, green, blue", ["red, green, and blue"])["contains"] is True
    assert score_answer("", [""]) == {"exact": True, "contains": True}
    assert score_answer("12", [""]) == {"exact": False, "contains": False}


def test_fixture_evaluation_passes(tmp_path) -> None:
    code = evaluate_main(
        ["--mode", "fixture", "--output", str(tmp_path), "--fail-under", "1.0"]
    )
    assert code == 0
    report = (tmp_path / "report.json").read_text(encoding="utf-8")
    assert '"primary_pass": 11' in report


def test_latency_percentiles_and_abstain_rate() -> None:
    rows = [
        {"pass": True, "exact": True, "contains": True, "answer": "Paris", "seconds": 1.0, "abstain": False},
        {"pass": False, "exact": False, "contains": False, "answer": "", "seconds": 2.0, "abstain": True},
        {"pass": True, "exact": True, "contains": True, "answer": "", "seconds": 10.0, "abstain": True},
        {"pass": True, "exact": False, "contains": True, "answer": "yen", "seconds": 24.0, "abstain": False},
    ]
    summary = summarize(rows, "live", 24.0)
    assert summary["total"] == 4
    assert summary["primary_pass"] == 3
    assert summary["abstain"] == 2
    assert summary["abstain_rate"] == 0.5
    assert summary["p50_seconds"] == 6.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert summary["within_budget"] == 4
