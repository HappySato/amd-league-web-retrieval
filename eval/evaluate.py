#!/usr/bin/env python3
"""Score a question set on exact match, contains, abstention, and latency.

Fixture mode replays bundled pages and does not touch the network. Live mode
uses the keyless provider list (direct Wikipedia URLs, Wikidata, MediaWiki,
DuckDuckGo HTML, Bing HTML, Brave HTML). No paid search API.

    python eval/evaluate.py --mode fixture
    python eval/evaluate.py --mode live
    python eval/evaluate.py --mode live --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scoring import primary_pass, score_answer
from iwir.adapter import IOAdapter
from iwir.config import DEFAULT_SEARCH_PROVIDERS, Config
from iwir.models import Question
from iwir.runner import configure_logging, run_with_timeout
from iwir.search.chain import build_services

SAMPLE = Path(__file__).resolve().parent / "questions.json"
LIVE = Path(__file__).resolve().parent / "questions_live.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bundle.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate direct-answer retrieval")
    parser.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=24.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit 1 when primary accuracy is below this fraction (0-1)",
    )
    args = parser.parse_args(argv)

    questions_path = args.questions or (LIVE if args.mode == "live" else SAMPLE)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    output = args.output or (ROOT / "output" / "eval" / args.mode)
    answers = output / "answers"
    answers.mkdir(parents=True, exist_ok=True)

    config = _config(args.mode, answers, args.timeout)
    configure_logging("WARNING")
    services = build_services(config)
    rows: list[dict] = []
    print(f"{'id':<22} {'pass':<6} {'exact':<6} {'contains':<9} {'sec':>7}  answer", flush=True)
    try:
        for item in questions:
            question = Question(id=item["id"], question=item["question"])
            started = time.perf_counter()
            result = run_with_timeout(question, config, services)
            elapsed = time.perf_counter() - started
            IOAdapter(config).write_result(result)
            scores = score_answer(result.answer, item.get("answers") or [item.get("expect", "")])
            match = item.get("match", "exact")
            row = {
                "id": question.id,
                "question": question.question,
                "answer": result.answer,
                "sources": result.sources,
                "confidence": result.confidence,
                "seconds": round(elapsed, 3),
                "exact": scores["exact"],
                "contains": scores["contains"],
                "pass": primary_pass(scores, match),
                "match": match,
                "error": result.error,
                "abstain": not result.answer.strip(),
            }
            rows.append(row)
            _print_row(row)
    finally:
        services.close()

    summary = summarize(rows, args.mode, args.timeout)
    report = {"summary": summary, "questions": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(summary)
    if args.fail_under is not None and summary["primary_accuracy"] < args.fail_under:
        return 1
    return 0


def _config(mode: str, output: Path, timeout: float) -> Config:
    fixture = mode == "fixture"
    # Fixture replay must not call a model. Live mode uses a local vLLM only
    # when both variables are set; an empty value keeps the rule extractor.
    if fixture:
        llm_base, llm_model, llm_key = "", "", ""
    else:
        llm_base = os.environ.get("LLM_BASE_URL", "").strip()
        llm_model = os.environ.get("LLM_MODEL", "").strip()
        llm_key = os.environ.get("LLM_API_KEY", "")
    return Config(
        input_path=None,
        input_json="",
        output_dir=output,
        query_id="",
        query="",
        start_url="",
        question_timeout=timeout,
        search_timeout=5.0,
        fetch_timeout=6.0,
        llm_timeout=8.0,
        max_pages=3,
        max_hops=1 if not fixture else 0,
        search_limit=5,
        search_providers=["fixture"] if fixture else [part.strip() for part in DEFAULT_SEARCH_PROVIDERS.split(",")],
        fixture_path=FIXTURE if fixture else None,
        schema_profile="default",
        output_fields=None,
        filename_template=None,
        include_diagnostics=False,
        llm_base_url=llm_base,
        llm_model=llm_model,
        llm_api_key=llm_key,
        llm_json_mode=True,
        log_level="WARNING",
    )


def summarize(rows: list[dict], mode: str, timeout: float) -> dict:
    passed = sum(1 for row in rows if row["pass"])
    exact = sum(1 for row in rows if row["exact"])
    contains = sum(1 for row in rows if row["contains"])
    abstain = sum(1 for row in rows if row.get("abstain") or not str(row.get("answer") or "").strip())
    total = len(rows) or 1
    latencies = [float(row["seconds"]) for row in rows]
    within = sum(1 for value in latencies if value <= timeout + 0.05)
    return {
        "mode": mode,
        "total": len(rows),
        "primary_pass": passed,
        "primary_accuracy": round(passed / total, 4),
        "exact": exact,
        "exact_accuracy": round(exact / total, 4),
        "contains": contains,
        "contains_accuracy": round(contains / total, 4),
        "abstain": abstain,
        "abstain_rate": round(abstain / total, 4),
        "mean_seconds": round(sum(latencies) / total, 3) if rows else 0,
        "p50_seconds": round(percentile(latencies, 50), 3),
        "p95_seconds": round(percentile(latencies, 95), 3),
        "max_seconds": round(max(latencies), 3) if rows else 0,
        "within_budget": within,
        "budget_seconds": timeout,
    }


def percentile(values: list[float], pct: float) -> float:
    """Linear percentile. p50 is the median; p95 is the high end of the sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _print_row(row: dict) -> None:
    answer = row["answer"] if row["answer"] else "(empty)"
    if len(answer) > 60:
        answer = answer[:57] + "..."
    print(
        f"{row['id']:<22} "
        f"{'yes' if row['pass'] else 'no':<6} "
        f"{'yes' if row['exact'] else 'no':<6} "
        f"{'yes' if row['contains'] else 'no':<9} "
        f"{row['seconds']:>7.2f}  {answer}",
        flush=True,
    )


def _print_summary(summary: dict) -> None:
    print(
        f"primary {summary['primary_pass']}/{summary['total']} "
        f"({summary['primary_accuracy']:.1%})  "
        f"exact {summary['exact']}/{summary['total']}  "
        f"contains {summary['contains']}/{summary['total']}  "
        f"abstain {summary['abstain']}/{summary['total']} ({summary['abstain_rate']:.1%})"
    )
    print(
        f"latency p50 {summary['p50_seconds']:.2f}s  "
        f"p95 {summary['p95_seconds']:.2f}s  "
        f"mean {summary['mean_seconds']:.2f}s  "
        f"max {summary['max_seconds']:.2f}s  "
        f"within {summary['budget_seconds']:.0f}s: "
        f"{summary['within_budget']}/{summary['total']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
