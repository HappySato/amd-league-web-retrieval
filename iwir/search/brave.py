"""Brave HTML search. No API key and no Brave Search API."""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from iwir.http import polite_get
from iwir.models import SearchHit, SearchRequest
from iwir.net import BROWSER_USER_AGENT, allowed_url
from iwir.search.wikipedia import _remaining_timeout

log = logging.getLogger(__name__)

_SEARCH = "https://search.brave.com/search"


class BraveProvider:
    name = "brave"

    def __init__(self, timeout: float, client: httpx.Client) -> None:
        self.timeout = timeout
        self.client = client

    def search(self, request: SearchRequest) -> list[SearchHit]:
        try:
            response = polite_get(
                self.client,
                _SEARCH,
                params={"q": request.query, "source": "web"},
                headers={
                    "User-Agent": BROWSER_USER_AGENT,
                    "Accept": "text/html",
                    "Accept-Language": "en",
                },
                timeout=_remaining_timeout(self.timeout, request.deadline),
                deadline=request.deadline,
            )
        except httpx.HTTPError as exc:
            log.info("Brave HTML search failed: %s", exc)
            return []
        if response.status_code != 200:
            log.info("Brave HTML search returned HTTP %s", response.status_code)
            return []
        return parse_brave_html(response.text)[: request.limit]


def parse_brave_html(html: str) -> list[SearchHit]:
    soup = BeautifulSoup(html, "html.parser")
    nodes = soup.select("div.snippet")
    if not nodes:
        nodes = soup.select("[data-type='web']")
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for node in nodes:
        anchor = node.select_one("a[href]")
        if anchor is None:
            continue
        url = _unwrap(anchor.get("href") or "")
        if not url or url in seen or not allowed_url(url):
            continue
        host = (urlparse(url).hostname or "").lower()
        if "brave.com" in host:
            continue
        seen.add(url)
        title_node = node.select_one(".title") or anchor
        title = title_node.get_text(" ", strip=True)
        snippet_node = node.select_one(".snippet-description") or node.select_one(".snippet-content")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        hits.append(SearchHit(url=url, title=title, snippet=snippet))
    return hits


def _unwrap(href: str) -> str:
    href = href.strip()
    parsed = urlparse(href)
    if "brave.com" in (parsed.hostname or "") and parsed.path.startswith("/redirect"):
        target = parse_qs(parsed.query).get("url", [])
        if target:
            return unquote(target[0])
    return href
