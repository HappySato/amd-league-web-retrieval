"""Per-question timeout wrapper and the process entry point."""

from __future__ import annotations

import logging
import sys
import threading
import time

from iwir.adapter import IOAdapter
from iwir.config import Config, ConfigError
from iwir.models import AnswerResult, Question, empty_result
from iwir.pipeline import answer_question
from iwir.search.chain import Services, build_services

log = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )


def run_with_timeout(question: Question, config: Config, services: Services) -> AnswerResult:
    """Always return a result. The worker is a daemon so a stuck fetch cannot wedge the process."""
    deadline = time.monotonic() + config.question_timeout
    holder: dict[str, AnswerResult | BaseException] = {}

    def target() -> None:
        try:
            holder["result"] = answer_question(question, config, services, deadline)
        except Exception as exc:
            log.exception("question %s failed", question.id)
            holder["error"] = exc

    thread = threading.Thread(target=target, name=f"iwir-{question.id}", daemon=True)
    thread.start()
    thread.join(config.question_timeout + 0.5)
    result = holder.get("result")
    if isinstance(result, AnswerResult):
        return result
    if thread.is_alive():
        log.error("question %s exceeded %.1fs", question.id, config.question_timeout)
        return empty_result(question, error="timeout")
    error = holder.get("error")
    if isinstance(error, BaseException):
        return empty_result(question, error=type(error).__name__)
    return empty_result(question, error="unknown")


def main(argv: list[str] | None = None) -> int:
    try:
        config = Config.from_argv(argv)
    except ConfigError as exc:
        configure_logging("INFO")
        log.error("%s", exc)
        return 2
    configure_logging(config.log_level)
    try:
        adapter = IOAdapter(config)
        questions = adapter.load_questions()
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    except Exception:
        log.exception("could not read questions")
        return 2
    if not questions:
        log.warning("no questions found; nothing written")
        return 0

    if config.llm_enabled:
        log.info("LLM enabled model=%s", config.llm_model)
    else:
        log.info("LLM disabled (set LLM_BASE_URL and LLM_MODEL to use a local vLLM server)")

    try:
        services = build_services(config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    status = 0
    try:
        for question in questions:
            started = time.perf_counter()
            result = run_with_timeout(question, config, services)
            try:
                path = adapter.write_result(result)
            except Exception:
                log.exception("failed to write result for %s", question.id)
                status = 1
                continue
            elapsed = time.perf_counter() - started
            log.info(
                "wrote %s in %.2fs answer=%r confidence=%.2f",
                path,
                elapsed,
                result.answer,
                result.confidence,
            )
    finally:
        services.close()
    return status
