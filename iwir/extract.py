"""Turn page text into the value the question asked for.

The heuristic is the offline path: it has to work with no model and it has to
refuse when the pages do not actually contain an answer. Patterns cover the
common factual shapes (infobox label, copula, year, count, symbol). A local
model, when configured, is asked only if this path is unsure.
"""

from __future__ import annotations

import re

from iwir.models import Candidate, Page
from iwir.textutil import (
    NUMBER_WORDS,
    canon_terms,
    echoes_question,
    head_term,
    intent_of,
    split_sentences,
    supported,
)

_HIGH = 0.82
_MED = 0.73
_LOW = 0.61


def choose_answer(question: str, pages: list[Page]) -> Candidate | None:
    if not question.strip() or not pages:
        return None
    intent = intent_of(question)
    candidates: list[Candidate] = []
    for rank, page in enumerate(pages):
        penalty = rank * 0.01
        for candidate in _from_page(question, page, intent):
            candidate.confidence = max(0.0, candidate.confidence - penalty)
            if page.from_snippet:
                candidate.confidence = min(candidate.confidence, 0.7)
                candidate.method = f"{candidate.method}+snippet"
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item.confidence, -len(item.answer)), reverse=True)
    return candidates[0] if candidates else None


def _from_page(question: str, page: Page, intent: str) -> list[Candidate]:
    found: list[Candidate] = []
    found.extend(_from_pairs(question, page, intent))
    terms = canon_terms(question)
    patterns = _patterns(question, intent)
    for sentence in split_sentences(page.text):
        overlap = len(terms & canon_terms(sentence))
        if not _relevant(overlap, len(terms)):
            continue
        for method, pattern, base in patterns:
            for match in list(pattern.finditer(sentence))[:4]:
                if method == "generic":
                    if intent != "fact" or overlap < 2:
                        continue
                    subject = match.group(1)
                    raw = match.group(2)
                    if not (canon_terms(subject) & terms):
                        continue
                else:
                    raw = match.group(1)
                answer = _finalize(raw, question, intent)
                if not answer:
                    continue
                if method == "generic" and len(answer.split()) > 4:
                    continue
                if not supported(answer, page.text):
                    continue
                if _skip_adaptation_credit(question, page.url, sentence):
                    continue
                confidence = min(0.93, base + 0.015 * overlap)
                # "first released" should not lose to a later "released in 2008".
                if (
                    intent == "date"
                    and "first" in question.casefold()
                    and re.fullmatch(r"[12][0-9]{3}", answer)
                    and "first" not in sentence[max(0, match.start() - 24) : match.end()].casefold()
                ):
                    confidence -= 0.12
                if confidence < 0.7:
                    continue
                # Prefer the count that sits next to the thing being counted
                # ("eight planets") over an earlier incidental number ("one of").
                if re.fullmatch(r"\d+", answer):
                    head = head_term(sentence[match.end() : match.end() + 40])
                    if head and head in terms:
                        confidence = min(0.93, confidence + 0.08)
                # "How many" wants a small integer. A measurement does not;
                # 1.0003 must not beat 299792458, and must not stand alone.
                if (
                    intent == "number"
                    and "how many" not in question.casefold()
                    and re.fullmatch(r"\d+(?:\.\d+)?", answer)
                    and len(answer.split(".", 1)[0]) < 3
                ):
                    confidence -= 0.25
                if confidence < 0.7:
                    continue
                found.append(
                    Candidate(
                        answer=answer,
                        source_url=page.url,
                        confidence=confidence,
                        method=method,
                        quote=sentence[:300],
                    )
                )
    return found


# Words that ride along on an infobox label ("Capital and largest city")
# without changing which fact the row is about.
_LABEL_MODIFIERS = {
    "largest",
    "biggest",
    "smallest",
    "major",
    "main",
    "official",
    "national",
    "current",
    "primary",
    "administrative",
    "federal",
    "city",
    "town",
    "state",
}


def _from_pairs(question: str, page: Page, intent: str) -> list[Candidate]:
    q_terms = canon_terms(question)
    found: list[Candidate] = []
    for label, value in page.pairs:
        label_terms = canon_terms(label)
        if not label_terms or not (label_terms & q_terms):
            continue
        # "Capital and largest city" still answers "capital". A label that
        # adds some other fact ("area code") does not.
        if label_terms - q_terms - _LABEL_MODIFIERS:
            continue
        focused = label_terms - _LABEL_MODIFIERS
        if not focused or not (focused & q_terms):
            continue
        coverage = len(focused & q_terms) / len(focused)
        answer = _finalize(value, question, intent)
        if not answer or not supported(answer, page.text):
            continue
        if _skip_adaptation_credit(question, page.url, f"{label}: {value}"):
            continue
        found.append(
            Candidate(
                answer=answer,
                source_url=page.url,
                confidence=min(0.93, 0.86 + 0.06 * coverage),
                method="infobox",
                quote=f"{label}: {value}",
            )
        )
    return found


def _patterns(question: str, intent: str) -> list[tuple[str, re.Pattern[str], float]]:
    q = question.casefold()
    compiled: list[tuple[str, re.Pattern[str], float]] = []

    def add(method: str, pattern: str, confidence: float, flags: int = re.I) -> None:
        compiled.append((method, re.compile(pattern, flags), confidence))

    if "capital" in q:
        add("pattern", r"capital(?:\s+city)?(?:\s+of\s+[a-z][a-z\s]{0,40})?\s+is\s+([^.,;\n]+)", _HIGH)
        add(
            "pattern",
            r"((?-i:[A-Z][A-Za-z]+(?:\s+(?-i:[A-Z][A-Za-z]+)){0,2}))\s+is\s+(?:the\s+)?capital\b",
            _HIGH,
        )
    if "symbol" in q:
        add(
            "pattern",
            r"(?:chemical\s+)?symbol\s+(?:is\s+|for\s+[a-z0-9]+\s+is\s+)?((?-i:[A-Z][a-z]?))\b",
            _HIGH,
        )
    if "currency" in q:
        add("pattern", r"currency(?:\s+of\s+[a-z][a-z\s]{0,40})?\s+is\s+(?:the\s+)?([^.,;\n]+)", _HIGH)
        add("pattern", r"((?-i:[A-Z])[^.,;\n]{1,40}?)\s+is\s+the\s+(?:official\s+)?currency", _HIGH)
    if q.startswith("who ") or "author" in q or "wrote" in q or "written" in q:
        add("pattern", r"(?:written|authored|composed|invented|created)\s+by\s+([^.,;\n]+)", _HIGH)
        add("pattern", r"(?:novel|play|book|tragedy|comedy|poem)\s+by\s+([^.,;\n]+)", _HIGH)
    if any(word in q for word in ("released", "founded", "born", "published", "what year", "which year")):
        add(
            "pattern",
            r"(?:first\s+)?(?:released|founded|born|published|created|introduced)\s+in\s+([12][0-9]{3})",
            _HIGH,
        )
        add("pattern", r"\bin\s+([12][0-9]{3})\b", _MED)
        add(
            "pattern",
            r"(?:fell|collapsed|landed|opened)\s+(?:on\s+)?(?:\d{1,2}\s+[A-Za-z]+\s+)?([12][0-9]{3})",
            _HIGH,
        )
    if "boiling" in q:
        add("pattern", r"boiling point[^\d-]{0,60}?(-?\d+(?:\.\d+)?)", _HIGH)
        add("pattern", r"boils?\s+at\s+(?:about\s+|approximately\s+|around\s+)?(\d+(?:\.\d+)?)", _HIGH)
    if "largest" in q:
        add(
            "pattern",
            r"((?-i:[A-Z][A-Za-z]+(?:\s+(?-i:[A-Z][A-Za-z]+)){0,3}))\s+is\s+(?:the\s+|earth'?s\s+)?largest",
            _HIGH,
        )
        add("pattern", r"largest\s+[a-z]+(?:\s+[a-z]+){0,4}\s+is\s+([^.,;\n]+)", _HIGH)
    if "highest" in q:
        add(
            "pattern",
            r"((?-i:[A-Z][A-Za-z]+(?:\s+(?-i:[A-Z][A-Za-z]+)){0,3}))\s+is\s+(?:the\s+|earth'?s\s+)?highest",
            _HIGH,
        )
        add("pattern", r"highest\s+[a-z]+(?:\s+[a-z]+){0,4}\s+is\s+([^.,;\n]+)", _HIGH)
    if "smallest" in q:
        add("pattern", r"((?-i:[A-Z][A-Za-z]+))\s+is\s+the\s+smallest", _HIGH)
    if "language" in q:
        add(
            "pattern",
            r"official language(?:s)?(?:\s+of\s+[a-z][a-z\s]{0,40})?\s+(?:is|are)\s+(?:the\s+)?([^.,;\n]+)",
            _HIGH,
        )
    if "painted" in q or "painter" in q:
        add("pattern", r"painted\s+by\s+([^.,;\n]+)", _HIGH)
    if intent == "number":
        add(
            "pattern",
            r"(?:is|are|was|were|equals|exactly|has|have)\s+(?:about\s+|approximately\s+|exactly\s+)?(\d[\d,]*(?:\.\d+)?)",
            _HIGH,
        )
        # "The value 299,792,458 metres per second is ..." puts the figure
        # before the verb. A comma-grouped or long integer is the measurement.
        if "how many" not in q:
            add("pattern", r"\b(\d{1,3}(?:,\d{3})+|\d{6,})\b", _HIGH)
        words = "|".join(NUMBER_WORDS)
        add(
            "pattern",
            rf"(?:is|are|was|were|has|have|of)\s+(?:about\s+|approximately\s+|exactly\s+|the\s+)?({words})\b",
            _MED,
        )
    if intent == "list":
        add("pattern", r"(?:are|:)\s+([a-z][^.]{0,140})", _MED)
    add(
        "generic",
        r"(?:^|\s)((?:(?:its|the|their)\s+)?[a-z][\w\s]{2,40}?)\s+(?:is|was|are|were)\s+([^.,;\n]{1,80})",
        _LOW,
    )
    return compiled


def _skip_adaptation_credit(question: str, url: str, sentence: str) -> bool:
    """A film or series page's "written by" line is the screenwriter, not the novelist."""
    if not any(word in question.casefold() for word in ("novel", "play", "book")):
        return False
    blob = url.casefold()
    if not any(token in blob for token in ("_film", "(film)", "tv_series", "television", "miniseries")):
        return False
    return bool(re.search(r"\b(written|directed|screenplay|teleplay|adapted)\b", sentence, re.I))


def _looks_like_name(answer: str) -> bool:
    """A name, a place, or a one/two-letter chemical symbol. Not 'typically a'."""
    if re.fullmatch(r"[A-Z][a-z]?", answer):
        return True
    parts = answer.split()
    if not parts or len(parts) > 6:
        return False
    return bool(re.match(r"[A-Z0-9]", parts[0]))


def _relevant(overlap: int, term_count: int) -> bool:
    if overlap >= 2:
        return True
    if term_count <= 1:
        return overlap >= 1
    # A single shared word is enough only for short questions ("capital of France").
    return overlap >= 1 and term_count <= 2


def _finalize(raw: str, question: str, intent: str) -> str:
    answer = clean_span(raw, as_list=intent == "list")
    if not answer:
        return ""
    if intent == "number" and answer.casefold() in NUMBER_WORDS:
        answer = NUMBER_WORDS[answer.casefold()]
    if intent == "number":
        answer = answer.replace(",", "")
    if intent == "date":
        year = re.search(r"\b([12][0-9]{3})\b", answer)
        if year:
            answer = year.group(1)
        elif not re.search(r"\d", answer):
            return ""
    if intent == "number" and not re.fullmatch(r"\d+(?:\.\d+)?", answer):
        return ""
    if intent == "name" and not _looks_like_name(answer):
        return ""
    word_count = len(answer.split())
    limit = 16 if intent == "list" else 8
    if word_count > limit or word_count == 0:
        return ""
    if echoes_question(question, answer):
        return ""
    if len(answer) < 1:
        return ""
    return answer


def clean_span(span: str, *, as_list: bool) -> str:
    text = span.strip()
    text = text.strip(" \t\"'`“”‘’.,;:")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    text = re.sub(
        r"^(?:the|a|an|about|approximately|around|exactly|roughly)\s+",
        "",
        text,
        flags=re.I,
    )
    if as_list:
        text = re.sub(r"\s+\band\b\s+", ", ", text, flags=re.I)
        parts = [part.strip(" .;") for part in text.split(",")]
        parts = [part for part in parts if part]
        text = ", ".join(parts)
    else:
        text = re.split(r"\s+\band\b\s+", text, maxsplit=1, flags=re.I)[0]
        text = re.split(r"\s*[;]\s*", text, maxsplit=1)[0]
        text = re.split(r"\s+\b(?:in|on|at|during|from)\b\s+", text, maxsplit=1, flags=re.I)[0]
    text = re.split(r"\b(?:which|whose|where|while|that)\b", text, maxsplit=1, flags=re.I)[0]
    text = re.sub(r"\s+respectively$", "", text, flags=re.I)
    return text.strip(" \t\n.,;:")
