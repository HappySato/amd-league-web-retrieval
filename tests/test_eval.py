from __future__ import annotations

from eval.evaluate import main as evaluate_main
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
