#!/usr/bin/env python3
"""Score a small question set on exact match, contains, and latency.

Fixture mode replays bundled pages and does not touch the network. Live mode
searches DuckDuckGo HTML and Wikipedia, then fetches the pages it cites.

    python eval/evaluate.py --mode fixture
    python eval/evaluate.py --mode live
    python eval/evaluate.py --mode live --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.scoring import primary_pass, score_answer
from iwir.adapter import IOAdapter
from iwir.config import Config
from iwir.models import Question
from iwir.runner import configure_logging, run_with_timeout
from iwir.search.chain import build_services

SAMPLE = Path(__file__).resolve().parent / "questions.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bundle.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate direct-answer retrieval")
    parser.add_argument("--mode", choices=("fixture", "live"), default="fixture")
    parser.add_argument("--questions", type=Path, default=SAMPLE)
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

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    output = args.output or (ROOT / "output" / "eval" / args.mode)
    answers = output / "answers"
    answers.mkdir(parents=True, exist_ok=True)

    config = _config(args.mode, answers, args.timeout)
    configure_logging("WARNING")
    services = build_services(config)
    rows: list[dict] = []
    try:
        for item in questions:
            question = Question(id=item["id"], question=item["question"])
            started = time.perf_counter()
            result = run_with_timeout(question, config, services)
            elapsed = time.perf_counter() - started
            IOAdapter(config).write_result(result)
            scores = score_answer(result.answer, item.get("answers") or [item.get("expect", "")])
            match = item.get("match", "exact")
            rows.append(
                {
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
                }
            )
    finally:
        services.close()

    passed = sum(1 for row in rows if row["pass"])
    exact = sum(1 for row in rows if row["exact"])
    contains = sum(1 for row in rows if row["contains"])
    total = len(rows) or 1
    latencies = [row["seconds"] for row in rows]
    summary = {
        "mode": args.mode,
        "total": len(rows),
        "primary_pass": passed,
        "primary_accuracy": round(passed / total, 4),
        "exact": exact,
        "exact_accuracy": round(exact / total, 4),
        "contains": contains,
        "contains_accuracy": round(contains / total, 4),
        "mean_seconds": round(sum(latencies) / total, 3) if rows else 0,
        "max_seconds": round(max(latencies), 3) if rows else 0,
    }
    report = {"summary": summary, "questions": rows}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print(summary, rows)
    if args.fail_under is not None and summary["primary_accuracy"] < args.fail_under:
        return 1
    return 0


def _config(mode: str, output: Path, timeout: float) -> Config:
    fixture = mode == "fixture"
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
        search_providers=["fixture"] if fixture else ["duckduckgo", "wikipedia"],
        fixture_path=FIXTURE if fixture else None,
        schema_profile="default",
        output_fields=None,
        filename_template=None,
        include_diagnostics=False,
        llm_base_url="",
        llm_model="",
        llm_api_key="",
        llm_json_mode=True,
        log_level="WARNING",
    )


def _print(summary: dict, rows: list[dict]) -> None:
    print(
        f"{'id':<20} {'pass':<6} {'exact':<6} {'contains':<9} {'sec':>7}  answer"
    )
    for row in rows:
        answer = row["answer"] if row["answer"] else "(empty)"
        if len(answer) > 60:
            answer = answer[:57] + "..."
        print(
            f"{row['id']:<20} "
            f"{'yes' if row['pass'] else 'no':<6} "
            f"{'yes' if row['exact'] else 'no':<6} "
            f"{'yes' if row['contains'] else 'no':<9} "
            f"{row['seconds']:>7.2f}  {answer}"
        )
    print(
        f"primary {summary['primary_pass']}/{summary['total']}  "
        f"exact {summary['exact']}/{summary['total']}  "
        f"contains {summary['contains']}/{summary['total']}  "
        f"mean {summary['mean_seconds']:.2f}s  max {summary['max_seconds']:.2f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())
