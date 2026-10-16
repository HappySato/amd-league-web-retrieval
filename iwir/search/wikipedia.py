"""Wikipedia search via the public MediaWiki API. No API key."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from iwir.models import SearchHit, SearchRequest
from iwir.net import WIKI_USER_AGENT, WIKIPEDIA_GATE

log = logging.getLogger(__name__)

_API = "https://en.wikipedia.org/w/api.php"


class WikipediaProvider:
    name = "wikipedia"

    def __init__(self, timeout: float, client: httpx.Client) -> None:
        self.timeout = timeout
        self.client = client
        self._down_until = 0.0

    def search(self, request: SearchRequest) -> list[SearchHit]:
        import time

        if time.monotonic() < self._down_until or WIKIPEDIA_GATE.blocked():
            log.info("wikipedia search skipped; waiting out a rate limit")
            return []
        try:
            WIKIPEDIA_GATE.wait()
            return self._search(request)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self._down_until = time.monotonic() + 20
                WIKIPEDIA_GATE.penalize(20)
                log.warning("wikipedia rate limited; skipping it for 20s")
            raise

    def _search(self, request: SearchRequest) -> list[SearchHit]:
        response = self.client.get(
            _API,
            params={
                "action": "query",
                "list": "search",
                "srsearch": wikipedia_search_query(request.query),
                "srlimit": request.limit,
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": WIKI_USER_AGENT, "Accept": "application/json"},
            timeout=_remaining_timeout(self.timeout, request.deadline),
        )
        response.raise_for_status()
        return parse_wiki_search(response.json())[: request.limit]


_ATTRIBUTE_ENTITY = re.compile(
    r"^(?:the\s+)?(?:chemical\s+)?"
    r"(?:capital|currency|population|president|symbol|boiling point)\s+"
    r"(?:of|for)\s+(?:the\s+)?(.+)$",
    re.I,
)


def wikipedia_search_query(query: str) -> str:
    """Search the entity page for attribute questions.

    ``capital of France`` is a poor MediaWiki query (it surfaces capital
    punishment). The France article's infobox already holds the capital.
    """
    query = query.strip()
    for label in ("novel", "play", "book"):
        title = _title_after(query, label)
        if title:
            return title
    match = _ATTRIBUTE_ENTITY.match(query)
    if not match:
        return query
    entity = re.split(r"\s+in\s+", match.group(1).strip(), maxsplit=1)[0].strip(" .")
    return entity or query


def _title_after(query: str, label: str) -> str | None:
    match = re.search(rf"\b{label}\s+(.+)$", query, re.I)
    if not match:
        return None
    title = match.group(1).strip(" .")
    return title or None


def parse_wiki_search(payload: dict) -> list[SearchHit]:
    items = (payload.get("query") or {}).get("search") or []
    hits: list[SearchHit] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        slug = quote(title.replace(" ", "_"), safe="!$&'()*+,;=:@-._~")
        url = f"https://en.wikipedia.org/wiki/{slug}"
        snippet_html = str(item.get("snippet") or "")
        snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)
        hits.append(SearchHit(url=url, title=title, snippet=snippet))
    return hits


def _remaining_timeout(limit: float, deadline: float) -> httpx.Timeout:
    import time

    remaining = max(0.4, deadline - time.monotonic())
    budget = min(limit, remaining)
    return httpx.Timeout(budget, connect=min(3.0, budget))
