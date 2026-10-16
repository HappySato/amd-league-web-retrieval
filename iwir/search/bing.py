"""Bing HTML search. No API key and no Bing Web Search API."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from iwir.http import polite_get
from iwir.models import SearchHit, SearchRequest
from iwir.net import BROWSER_USER_AGENT, allowed_url
from iwir.search.wikipedia import _remaining_timeout

log = logging.getLogger(__name__)

_SEARCH = "https://www.bing.com/search"


class BingProvider:
    name = "bing"

    def __init__(self, timeout: float, client: httpx.Client) -> None:
        self.timeout = timeout
        self.client = client

    def search(self, request: SearchRequest) -> list[SearchHit]:
        try:
            response = polite_get(
                self.client,
                _SEARCH,
                params={"q": request.query, "setlang": "en", "count": str(request.limit)},
                headers={
                    "User-Agent": BROWSER_USER_AGENT,
                    "Accept": "text/html",
                    "Accept-Language": "en",
                },
                timeout=_remaining_timeout(self.timeout, request.deadline),
                deadline=request.deadline,
            )
        except httpx.HTTPError as exc:
            log.info("Bing HTML search failed: %s", exc)
            return []
        if response.status_code != 200:
            log.info("Bing HTML search returned HTTP %s", response.status_code)
            return []
        return parse_bing_html(response.text)[: request.limit]


def parse_bing_html(html: str) -> list[SearchHit]:
    soup = BeautifulSoup(html, "html.parser")
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for item in soup.select("li.b_algo"):
        anchor = item.select_one("h2 a") or item.find("a", href=True)
        if anchor is None:
            continue
        url = (anchor.get("href") or "").strip()
        if not _keep(url) or url in seen:
            continue
        seen.add(url)
        title = anchor.get_text(" ", strip=True)
        snippet_node = item.select_one(".b_caption p") or item.find("p")
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
        hits.append(SearchHit(url=url, title=title, snippet=snippet))
    return hits


def _keep(url: str) -> bool:
    if not allowed_url(url):
        return False
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("bing.com") or host.endswith("microsoft.com"):
        return False
    lowered = url.casefold()
    return "aclick" not in lowered and "/aclk" not in lowered
