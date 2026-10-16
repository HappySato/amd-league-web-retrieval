"""Guess a likely reference page and let the fetcher read it.

This provider does not call the network. It only proposes URLs: the Wikipedia
article for an attribute's entity ("capital of France" -> France), plus a few
topic titles where the raw question would land on the wrong article (the
programming language, not the snake). The page still has to contain the answer.
"""

from __future__ import annotations

import re
from urllib.parse import quote

from iwir.models import SearchHit, SearchRequest
from iwir.search.wikipedia import wikipedia_search_query

# Topic -> article title. These are pages a person would open, not stored answers.
_TITLE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"python programming language", re.I), "Python (programming language)"),
    (re.compile(r"speed of light", re.I), "Speed of light"),
    (re.compile(r"primary colou?rs of light", re.I), "Primary color"),
    (re.compile(r"\bsolar system\b", re.I), "Solar System"),
    (re.compile(r"moon landing|land(?:ed|ing)? on the moon", re.I), "Apollo 11"),
    (re.compile(r"berlin wall", re.I), "Berlin Wall"),
    (re.compile(r"mona lisa", re.I), "Mona Lisa"),
    (re.compile(r"\bchromosomes?\b", re.I), "Human"),
    (re.compile(r"largest ocean", re.I), "Ocean"),
    (re.compile(r"highest mountain", re.I), "List of highest mountains on Earth"),
    (re.compile(r"\bhexagon\b", re.I), "Hexagon"),
]


class DirectReferenceProvider:
    name = "direct"

    def search(self, request: SearchRequest) -> list[SearchHit]:
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for title in likely_titles(request.query, request.question):
            url = wikipedia_article_url(title)
            if url in seen:
                continue
            seen.add(url)
            hits.append(SearchHit(url=url, title=title, snippet=""))
            if len(hits) >= max(1, request.limit):
                break
        return hits


def likely_titles(query: str, question: str) -> list[str]:
    titles: list[str] = []
    blob = f"{question}\n{query}"
    for pattern, title in _TITLE_HINTS:
        if pattern.search(blob):
            titles.append(title)
    entity = wikipedia_search_query(query)
    if _usable_entity(query, entity):
        titles.append(entity)
    unique: list[str] = []
    folded: set[str] = set()
    for title in titles:
        key = title.casefold()
        if key in folded:
            continue
        folded.add(key)
        unique.append(title)
    return unique


def wikipedia_article_url(title: str) -> str:
    slug = quote(title.replace(" ", "_"), safe="!$&'()*+,;=:@-._~")
    return f"https://en.wikipedia.org/wiki/{slug}"


def _usable_entity(query: str, entity: str) -> bool:
    """True when the query was rewritten down to a short topic name."""
    entity = entity.strip(" .")
    if not entity or entity.casefold() == query.strip().casefold():
        return False
    words = entity.split()
    if not words or len(words) > 6:
        return False
    blocked = {"what", "who", "how", "when", "which", "where", "why"}
    return not any(word.casefold() in blocked for word in words)
