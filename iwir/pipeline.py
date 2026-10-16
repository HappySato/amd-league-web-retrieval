"""Search, read, and return the value itself — or an empty answer.

Every stage checks the deadline. A missing page, a blocked search, or a model
that is down degrades to the next source of evidence. Unsupported values are
dropped, so the JSON can abstain instead of guessing.
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import urlparse

from iwir.config import Config
from iwir.extract import choose_answer
from iwir.models import (
    AnswerResult,
    Candidate,
    LLMAnswer,
    Page,
    Question,
    SearchHit,
    SearchRequest,
    empty_result,
)
from iwir.net import allowed_url, urls_in_text
from iwir.search.chain import Services
from iwir.textutil import canon_terms, content_terms, keyword_query, norm_text, same_answer, search_query, supported

log = logging.getLogger(__name__)

_BAD_ANCHOR = re.compile(
    r"^(edit|citation needed|read|jump to navigation|main page|log in|sign in|menu)$",
    re.I,
)


def answer_question(
    question: Question,
    config: Config,
    services: Services,
    deadline: float,
) -> AnswerResult:
    if not question.question.strip():
        return empty_result(question, error="empty-question")
    if time.monotonic() >= deadline:
        return empty_result(question, error="timeout")

    pages: list[Page] = []
    seen: set[str] = set()
    for url in _seed_urls(question):
        if time.monotonic() >= deadline - 0.4:
            break
        page = _take_page(url, None, services, deadline)
        if page is None or page.url in seen:
            continue
        seen.add(page.url)
        pages.append(page)

    heuristic = choose_answer(question.question, pages) if pages else None
    if not (heuristic and heuristic.confidence >= 0.9):
        hits = _prefer_reference_hosts(_search(question, config, services, deadline), question.question)
        heuristic = _read_hits(
            question, hits, pages, seen, config, services, deadline, heuristic, config.max_pages
        )
        # Keep asking the next keyless provider while the answer is weak.
        # A snippet-only value (capped near 0.7) gets one extra source, not the
        # whole list, so a rate limit does not burn the 24s budget.
        rounds = 0
        while (heuristic is None or heuristic.confidence < 0.8) and rounds < 4:
            if (
                heuristic is not None
                and heuristic.confidence >= 0.7
                and "+snippet" in heuristic.method
                and rounds >= 1
            ):
                break
            more = _search_rest(question, config, services, deadline)
            if not more:
                break
            rounds += 1
            heuristic = _read_hits(
                question,
                _prefer_reference_hosts(more, question.question),
                pages,
                seen,
                config,
                services,
                deadline,
                heuristic,
                config.max_pages + 2,
            )

    if (heuristic is None or heuristic.confidence < 0.75) and config.max_hops > 0:
        hops = 0
        for url in _rank_links(question.question, pages):
            if hops >= config.max_hops or url in seen:
                continue
            if time.monotonic() >= deadline - _reserve(config, services):
                break
            page = _take_page(url, None, services, deadline)
            hops += 1
            if page is None:
                continue
            seen.add(page.url)
            pages.append(page)
        heuristic = choose_answer(question.question, pages) if pages else None

    llm_candidate = _from_llm(question, pages, services, deadline, heuristic)
    chosen = _prefer(heuristic, llm_candidate)
    return _result(question, chosen, pages)


def _seed_urls(question: Question) -> list[str]:
    urls: list[str] = []
    if question.start_url:
        urls.append(question.start_url)
    urls.extend(question.seed_urls)
    urls.extend(urls_in_text(question.question))
    unique: list[str] = []
    for url in urls:
        if allowed_url(url) and url not in unique:
            unique.append(url)
    return unique[:2]


def _search(
    question: Question,
    config: Config,
    services: Services,
    deadline: float,
) -> list[SearchHit]:
    if time.monotonic() >= deadline - 0.2:
        return []
    queries = _search_queries(question, services, deadline)
    hits: list[SearchHit] = []
    used = queries[0] if queries else search_query(question.question)
    for query in queries:
        if time.monotonic() >= deadline - 0.2:
            break
        found = _run_search(services, question, config, deadline, query)
        used = query
        hits = [hit for hit in found if allowed_url(hit.url)]
        if hits:
            break
    if not hits and time.monotonic() < deadline - 0.2:
        keywords = keyword_query(question.question)
        tried = {query.casefold() for query in queries}
        if keywords and keywords.casefold() not in tried and keywords.casefold() != question.question.casefold():
            found = _run_search(services, question, config, deadline, keywords)
            hits = [hit for hit in found if allowed_url(hit.url)]
            if hits:
                used = keywords
    services.active_query = used
    return hits


def _search_queries(question: Question, services: Services, deadline: float) -> list[str]:
    """Model queries first, then the rule-based query. Duplicates are dropped."""
    fallback = search_query(question.question)
    rewritten: list[str] = []
    llm = services.llm
    rewrite = getattr(llm, "rewrite_queries", None) if llm is not None and getattr(llm, "enabled", False) else None
    if rewrite is not None:
        try:
            rewritten = list(rewrite(question.question, deadline) or [])
        except Exception as exc:
            log.warning("query rewrite failed; using the rule-based query: %s", exc)
            rewritten = []
    queries: list[str] = []
    seen: set[str] = set()
    for query in [*rewritten, fallback]:
        text = " ".join(str(query).split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(text)
    return queries[:3]


def _run_search(
    services: Services,
    question: Question,
    config: Config,
    deadline: float,
    query: str,
) -> list[SearchHit]:
    request = SearchRequest(
        query=query,
        question=question.question,
        question_id=question.id,
        limit=config.search_limit,
        deadline=deadline,
    )
    try:
        return list(services.search(request) or [])
    except Exception as exc:
        log.warning("search failed: %s", exc)
        return []


def _search_rest(
    question: Question,
    config: Config,
    services: Services,
    deadline: float,
) -> list[SearchHit]:
    rest = getattr(services, "search_rest", None)
    if rest is None or time.monotonic() >= deadline - 0.4:
        return []
    request = SearchRequest(
        query=getattr(services, "active_query", None) or search_query(question.question),
        question=question.question,
        question_id=question.id,
        limit=config.search_limit,
        deadline=deadline,
    )
    try:
        hits = list(rest(request) or [])
    except Exception as exc:
        log.warning("fallback search failed: %s", exc)
        hits = []
    return [hit for hit in hits if allowed_url(hit.url)]


def _read_hits(
    question: Question,
    hits: list[SearchHit],
    pages: list[Page],
    seen: set[str],
    config: Config,
    services: Services,
    deadline: float,
    heuristic: Candidate | None,
    page_limit: int,
) -> Candidate | None:
    for hit in hits:
        if len(pages) >= page_limit:
            break
        if time.monotonic() >= deadline - _reserve(config, services):
            break
        if hit.url in seen:
            continue
        page = _take_page(hit.url, hit, services, deadline)
        if page is None:
            continue
        seen.add(page.url)
        pages.append(page)
        heuristic = choose_answer(question.question, pages)
        if heuristic and heuristic.confidence >= 0.9:
            break
    return heuristic


def _prefer_reference_hosts(hits: list[SearchHit], question: str = "") -> list[SearchHit]:
    """Read Wikipedia before blogs, and the work itself before a film adaptation."""
    folded = question.casefold()
    wants_work = any(word in folded for word in ("novel", "play", "book"))

    def rank(hit: SearchHit) -> tuple[int, int]:
        host = (urlparse(hit.url).hostname or "").lower()
        blob = f"{hit.url} {hit.title}".casefold()
        adaptation = 0
        if wants_work and any(
            token in blob for token in ("film", "tv_series", "television", "miniseries", "movie")
        ):
            adaptation = 1
        wiki = 0 if host.endswith("wikipedia.org") else 1
        return (adaptation, wiki)

    return sorted(hits, key=rank)


def _take_page(
    url: str,
    hit: SearchHit | None,
    services: Services,
    deadline: float,
) -> Page | None:
    if not allowed_url(url) or time.monotonic() >= deadline:
        return None
    try:
        page = services.fetch(url, deadline)
    except Exception as exc:
        log.warning("fetch failed for %s: %s", url, exc)
        page = None
    if isinstance(page, Page) and page.text.strip():
        return page
    if hit and (hit.snippet or hit.title):
        text = "\n".join(part for part in (hit.title, hit.snippet) if part)
        return Page(url=hit.url, title=hit.title, text=text, from_snippet=True)
    return None


def _from_llm(
    question: Question,
    pages: list[Page],
    services: Services,
    deadline: float,
    heuristic: Candidate | None,
) -> Candidate | None:
    llm = services.llm
    if llm is None or not getattr(llm, "enabled", False) or not pages:
        return None
    if time.monotonic() > deadline - 1.2:
        return None
    try:
        extracted = llm.extract(question.question, pages, deadline)
    except Exception as exc:
        log.warning("LLM extract failed: %s", exc)
        return None
    if extracted is None:
        return None
    return _candidate_from_llm(question.question, extracted, pages)


def _candidate_from_llm(question: str, extracted: LLMAnswer, pages: list[Page]) -> Candidate | None:
    page = _page_for(extracted, pages)
    if page is None:
        return None
    answer = re.sub(r"\s+", " ", extracted.answer).strip().strip("\"'`")
    quote = extracted.quote.strip()
    if quote and norm_text(quote) not in norm_text(page.text):
        quote = ""
    # Tighten before the support check. Models often return a sentence; the
    # sentence is not a contiguous span of the page, but the value inside the
    # quote is.
    evidence = quote or answer
    tightened = choose_answer(question, [Page(url=page.url, title="", text=evidence)])
    if tightened and supported(tightened.answer, page.text):
        if not supported(answer, page.text) or len(norm_text(tightened.answer)) < len(norm_text(answer)):
            answer = tightened.answer
    if not answer or not supported(answer, page.text):
        log.info("dropping LLM answer %r; it is not in %s", extracted.answer, page.url)
        return None
    # Cap below a strong infobox hit so a merely-present span cannot replace it.
    # Agreement still raises confidence when the two answers match.
    confidence = max(0.8, min(0.9, extracted.confidence if extracted.confidence else 0.8))
    if not quote:
        confidence -= 0.05
    return Candidate(
        answer=answer,
        source_url=page.url,
        confidence=confidence,
        method="llm",
        quote=quote or answer,
    )


def _page_for(extracted: LLMAnswer, pages: list[Page]) -> Page | None:
    if extracted.source_url:
        for page in pages:
            if page.url == extracted.source_url:
                return page
    if extracted.quote:
        for page in pages:
            if norm_text(extracted.quote) in norm_text(page.text):
                return page
    for page in pages:
        if supported(extracted.answer, page.text):
            return page
    return None


def _prefer(heuristic: Candidate | None, llm_candidate: Candidate | None) -> Candidate | None:
    if heuristic is None:
        return llm_candidate
    if llm_candidate is None:
        return heuristic
    if same_answer(heuristic.answer, llm_candidate.answer):
        answer = heuristic.answer if _prefer_surface(heuristic.answer, llm_candidate.answer) else llm_candidate.answer
        return Candidate(
            answer=answer,
            source_url=heuristic.source_url or llm_candidate.source_url,
            confidence=min(0.97, max(heuristic.confidence, llm_candidate.confidence) + 0.02),
            method="agreement",
            quote=heuristic.quote or llm_candidate.quote,
        )
    h_norm = norm_text(heuristic.answer)
    l_norm = norm_text(llm_candidate.answer)
    if l_norm and l_norm in h_norm and len(l_norm) < len(h_norm):
        return llm_candidate
    if h_norm and h_norm in l_norm and len(h_norm) < len(l_norm):
        return heuristic
    return heuristic if heuristic.confidence >= llm_candidate.confidence else llm_candidate


def _prefer_surface(left: str, right: str) -> bool:
    if left.isdigit() and not right.isdigit():
        return True
    if right.isdigit() and not left.isdigit():
        return False
    return len(left) <= len(right)


def _result(question: Question, chosen: Candidate | None, pages: list[Page]) -> AnswerResult:
    if chosen is None or not chosen.answer.strip():
        return empty_result(question, error="no-evidence")
    answer = re.sub(r"\s+", " ", chosen.answer).strip()
    sources = _sources(answer, chosen.source_url, pages)
    if not sources:
        return empty_result(question, error="no-evidence")
    confidence = chosen.confidence
    if len(sources) >= 2:
        confidence = min(0.97, confidence + 0.04)
    supporting = [page for page in pages if page.url in sources]
    if supporting and all(page.from_snippet for page in supporting):
        confidence = min(confidence, 0.7)
    log.info(
        "answered %s via %s (%d source%s)",
        question.id,
        chosen.method,
        len(sources),
        "" if len(sources) == 1 else "s",
    )
    return AnswerResult(
        id=question.id,
        question=question.question,
        answer=answer,
        sources=sources,
        confidence=round(max(0.0, min(1.0, confidence)), 4),
        error="",
        method=chosen.method,
    )


def _sources(answer: str, primary: str, pages: list[Page]) -> list[str]:
    ordered: list[str] = []
    if primary:
        ordered.append(primary)
    for page in pages:
        if page.url not in ordered:
            ordered.append(page.url)
    kept: list[str] = []
    for url in ordered:
        page = next((item for item in pages if item.url == url), None)
        if page is None:
            continue
        if supported(answer, page.text) and allowed_url(url):
            kept.append(url)
    return kept


def _rank_links(question: str, pages: list[Page]) -> list[str]:
    terms = canon_terms(question)
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for page in pages:
        for link in page.links:
            if link.url in seen or not allowed_url(link.url):
                continue
            if _BAD_ANCHOR.match(link.text.strip()):
                continue
            if re.fullmatch(r"\d+", link.text.strip()):
                continue
            overlap = len(terms & canon_terms(link.text))
            if overlap <= 0:
                continue
            # Skip anchors that only repeat the question.
            if set(content_terms(link.text)) <= set(content_terms(question)):
                continue
            seen.add(link.url)
            scored.append((overlap, link.url))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [url for _, url in scored[:8]]


def _reserve(config: Config, services: Services) -> float:
    if services.llm is not None and services.llm.enabled:
        return min(6.0, config.llm_timeout)
    return 0.8
