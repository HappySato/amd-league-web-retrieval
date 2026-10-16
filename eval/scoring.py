"""Exact and contains scoring for short factual answers."""

from __future__ import annotations

from iwir.textutil import norm_text, supported


def score_answer(predicted: str, aliases: list[str]) -> dict[str, bool]:
    """Score one prediction.

    Exact ignores case and punctuation. Contains also accepts number words
    ("8" / "eight") and list wording ("red, green, and blue"). An empty gold
    answer is correct only when the prediction is empty — that is a refusal.
    """
    gold = [alias for alias in aliases]
    if not gold or all(not alias.strip() for alias in gold):
        ok = predicted.strip() == ""
        return {"exact": ok, "contains": ok}
    predicted_norm = norm_text(predicted)
    exact = any(predicted_norm == norm_text(alias) for alias in gold if alias.strip())
    contains = exact
    if predicted.strip():
        for alias in gold:
            if not alias.strip():
                continue
            if supported(alias, predicted) or supported(predicted, alias):
                contains = True
                break
    return {"exact": exact, "contains": contains}


def primary_pass(scores: dict[str, bool], match: str) -> bool:
    if match == "contains":
        return scores["contains"]
    return scores["exact"]
