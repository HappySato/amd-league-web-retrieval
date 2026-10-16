from __future__ import annotations

import time
from pathlib import Path

from iwir.llm import chat_completions_url, parse_json_object, parse_llm_payload
from iwir.models import LLMAnswer, Page, Question, SearchHit
from iwir.pipeline import _candidate_from_llm, _prefer, answer_question
from iwir.runner import run_with_timeout
from iwir.search.chain import Services
from tests.conftest import make_config

FRANCE = Question(id="capital-france", question="What is the capital of France?")


class StubLLM:
    def __init__(self, answer: LLMAnswer | None) -> None:
        self.enabled = True
        self.answer = answer
        self.calls = 0

    def extract(self, question: str, pages: list[Page], deadline: float):
        del question, deadline
        self.calls += 1
        return self.answer


def test_fixture_provider_answers_from_the_bundle(tmp_path: Path) -> None:
    from iwir.search.chain import build_services

    config = make_config(tmp_path, max_hops=1)
    services = build_services(config)
    try:
        result = answer_question(FRANCE, config, services, time.monotonic() + 5)
    finally:
        services.close()
    assert result.answer == "Paris"
    assert result.sources == ["https://example.test/france"]
    assert result.confidence > 0.8


def test_past_deadline_does_not_search(tmp_path: Path) -> None:
    calls: list[int] = []

    def search(request):
        calls.append(1)
        return []

    services = Services(search=search, fetch=lambda url, deadline: None, llm=None, close=lambda: None)
    result = answer_question(FRANCE, make_config(tmp_path), services, time.monotonic() - 1)
    assert result.answer == ""
    assert result.error == "timeout"
    assert calls == []


def test_runner_timeout_still_returns_a_result(tmp_path: Path) -> None:
    def search(request):
        time.sleep(2)
        return []

    services = Services(search=search, fetch=lambda url, deadline: None, llm=None, close=lambda: None)
    config = make_config(tmp_path, question_timeout=0.3)
    started = time.perf_counter()
    result = run_with_timeout(FRANCE, config, services)
    assert time.perf_counter() - started < 1.5
    assert result.answer == ""
    assert result.error == "timeout"
    assert result.confidence == 0.0


def test_search_failure_abstains(tmp_path: Path) -> None:
    def search(request):
        raise RuntimeError("blocked")

    def fetch(url, deadline):
        raise RuntimeError("nope")

    services = Services(search=search, fetch=fetch, llm=None, close=lambda: None)
    result = answer_question(FRANCE, make_config(tmp_path), services, time.monotonic() + 3)
    assert result.answer == ""
    assert result.sources == []


def test_llm_hallucination_is_dropped(tmp_path: Path) -> None:
    page = Page(
        url="https://example.test/brochure",
        title="",
        text="Its capital is Paris.",
    )
    hallucinated = LLMAnswer(
        answer="Atlantis",
        quote="Atlantis",
        source_url=page.url,
        confidence=0.99,
    )
    assert _candidate_from_llm(FRANCE.question, hallucinated, [page]) is None

    llm = StubLLM(hallucinated)
    services = Services(
        search=lambda request: [],
        fetch=lambda url, deadline: page if url == page.url else None,
        llm=llm,
        close=lambda: None,
    )
    question = Question(
        id="seeded",
        question=FRANCE.question,
        start_url="https://example.test/brochure",
    )
    result = answer_question(question, make_config(tmp_path, max_hops=0), services, time.monotonic() + 3)
    assert llm.calls == 1
    assert result.answer == "Paris"
    assert result.sources == [page.url]


def test_llm_answer_is_tightened_to_the_value(tmp_path: Path) -> None:
    del tmp_path
    page = Page(
        url="https://example.test/france",
        title="France",
        text="France is a country in Western Europe. Its capital is Paris and its largest city is Paris.",
    )
    extracted = LLMAnswer(
        answer="Paris is the capital of France",
        quote="Its capital is Paris",
        source_url=page.url,
        confidence=0.9,
    )
    chosen = _candidate_from_llm(FRANCE.question, extracted, [page])
    assert chosen is not None
    assert chosen.answer == "Paris"


def test_agreement_prefers_the_digit() -> None:
    from iwir.models import Candidate

    heuristic = Candidate("8", "https://example.test/a", 0.8, "pattern", "eight planets")
    model = Candidate("eight", "https://example.test/a", 0.9, "llm", "eight planets")
    chosen = _prefer(heuristic, model)
    assert chosen is not None
    assert chosen.answer == "8"
    assert chosen.method == "agreement"


def test_snippet_fallback_when_fetch_fails(tmp_path: Path) -> None:
    hit = SearchHit(
        url="https://example.test/france",
        title="France",
        snippet="Its capital is Paris and its largest city is Paris.",
    )

    def search(request):
        return [hit]

    services = Services(search=search, fetch=lambda url, deadline: None, llm=None, close=lambda: None)
    result = answer_question(FRANCE, make_config(tmp_path), services, time.monotonic() + 3)
    assert result.answer == "Paris"
    assert result.sources == ["https://example.test/france"]
    assert result.confidence <= 0.7


def test_llm_payload_parser() -> None:
    payload = parse_llm_payload(
        {"choices": [{"message": {"content": '```json\n{"answer":"Paris","quote":"capital is Paris","source_url":"https://example.test/france","confidence":0.91}\n```'}}]}
    )
    assert payload is not None
    assert payload.answer == "Paris"
    assert parse_json_object("no json here") is None
    assert chat_completions_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1/chat/completions"
    assert (
        chat_completions_url("http://127.0.0.1:8000/v1/chat/completions")
        == "http://127.0.0.1:8000/v1/chat/completions"
    )
