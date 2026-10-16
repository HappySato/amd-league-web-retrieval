"""Shared data types. The retrieval pipeline never writes files; the adapter does."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Question:
    id: str
    question: str
    start_url: str | None = None
    seed_urls: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SearchHit:
    url: str
    title: str
    snippet: str


@dataclass(slots=True)
class SearchRequest:
    query: str
    question: str
    question_id: str
    limit: int
    deadline: float


@dataclass(slots=True)
class Link:
    url: str
    text: str


@dataclass(slots=True)
class Page:
    url: str
    title: str
    text: str
    pairs: list[tuple[str, str]] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    from_snippet: bool = False


@dataclass(slots=True)
class LLMAnswer:
    answer: str
    quote: str
    source_url: str
    confidence: float


@dataclass(slots=True)
class Candidate:
    answer: str
    source_url: str
    confidence: float
    method: str
    quote: str = ""


@dataclass(slots=True)
class AnswerResult:
    id: str
    question: str
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    error: str = ""
    method: str = ""


def empty_result(question: Question, error: str = "") -> AnswerResult:
    return AnswerResult(
        id=question.id,
        question=question.question,
        answer="",
        sources=[],
        confidence=0.0,
        error=error,
        method="abstain",
    )
