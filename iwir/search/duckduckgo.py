"""DuckDuckGo HTML search. No API key.

The HTML endpoint returns result anchors, often wrapped in a ``/l/?uddg=``
redirect. Parsing is isolated so a layout change does not touch the pipeline.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from bs4 import BeautifulSoup

from iwir.http import polite_get
from iwir.models import SearchHit, SearchRequest
from iwir.net import BROWSER_USER_AGENT, allowed_url

log = logging.getLogger(__name__)


class DuckDuckGoProvider:
    name = "duckduckgo"

    def __init__(self, timeout: float, client: httpx.Client) -> None:
        self.timeout = timeout
        self.client = client

    def search(self, request: SearchRequest) -> list[SearchHit]:
        # html.duckduckgo.com sometimes answers 202 with an interstitial and no
        # results. The lite host is the fallback and uses the same result anchors.
        headers = {
            "User-Agent": BROWSER_USER_AGENT,
            "Accept": "text/html",
            "Accept-Language": "en",
        }
        timeout = _remaining_timeout(self.timeout, request.deadline)
        for url in ("https://html.duckduckgo.com/html/", "https://lite.duckduckgo.com/lite/"):
            try:
                response = polite_get(
                    self.client,
                    url,
                    params={"q": request.query},
                    headers=headers,
                    timeout=timeout,
                    deadline=request.deadline,
                    retries=1,
                )
            except httpx.HTTPError as exc:
                log.info("DuckDuckGo %s failed: %s", url, exc)
                continue
            if response.status_code != 200:
                log.info("DuckDuckGo %s returned HTTP %s", url, response.status_code)
                continue
            hits = parse_ddg_html(response.text)
            if hits:
                return hits[: request.limit]
        return []


def parse_ddg_html(html: str) -> list[SearchHit]:
    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.select("a.result__a")
    if not anchors:
        anchors = [anchor for anchor in soup.find_all("a", href=True) if "uddg=" in (anchor.get("href") or "")]
    hits: list[SearchHit] = []
    seen: set[str] = set()
    for anchor in anchors:
        url = unwrap_ddg_href(anchor.get("href") or "")
        if not url or url in seen or not allowed_url(url) or _is_ad(url):
            continue
        seen.add(url)
        title = anchor.get_text(" ", strip=True)
        snippet = ""
        parent = anchor.find_parent(class_="result")
        if parent is None:
            parent = anchor.find_parent("tr")
        if parent is not None:
            snippet_node = parent.select_one(".result__snippet")
            if snippet_node is not None:
                snippet = snippet_node.get_text(" ", strip=True)
        hits.append(SearchHit(url=url, title=title, snippet=snippet))
    return hits


def _is_ad(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.endswith("duckduckgo.com") or host.endswith("bing.com"):
        return True
    lowered = url.casefold()
    return any(token in lowered for token in ("ad_domain=", "ad_provider=", "/y.js", "aclick?"))


def unwrap_ddg_href(href: str) -> str:
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    if not href:
        return ""
    parsed = urlparse(href)
    target = parse_qs(parsed.query).get("uddg", [])
    if target:
        return unquote(target[0])
    if href.startswith("http://") or href.startswith("https://"):
        if "duckduckgo.com" in (parsed.hostname or ""):
            return ""
        return href
    return ""


def _remaining_timeout(limit: float, deadline: float) -> httpx.Timeout:
    import time

    remaining = max(0.4, deadline - time.monotonic())
    budget = min(limit, remaining)
    return httpx.Timeout(budget, connect=min(3.0, budget))
