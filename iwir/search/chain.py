"""Provider construction. Search providers run in listed order, not all at once."""

from __future__ import annotations

import logging
import time
from typing import Callable

import httpx

from iwir.browse import fetch_page
from iwir.config import Config, ConfigError
from iwir.llm import LLMClient
from iwir.models import SearchHit, SearchRequest
from iwir.net import BROWSER_USER_AGENT
from iwir.search.duckduckgo import DuckDuckGoProvider
from iwir.search.fixture import FixtureProvider
from iwir.search.wikipedia import WikipediaProvider

log = logging.getLogger(__name__)


class Services:
    def __init__(
        self,
        search: Callable[[SearchRequest], list[SearchHit]],
        fetch: Callable[[str, float], object],
        llm: LLMClient | None,
        close: Callable[[], None],
    ) -> None:
        self.search = search
        self.fetch = fetch
        self.llm = llm
        self.close = close


def build_services(config: Config) -> Services:
    llm = LLMClient(config) if config.llm_enabled else None
    if "fixture" in config.search_providers:
        if config.fixture_path is None:
            raise ConfigError("SEARCH_PROVIDERS=fixture requires FIXTURE_PATH or --fixture")
        fixture = FixtureProvider(config.fixture_path)
        return Services(search=fixture.search, fetch=fixture.fetch, llm=llm, close=lambda: None)

    client = httpx.Client(
        follow_redirects=True,
        max_redirects=5,
        headers={"User-Agent": BROWSER_USER_AGENT, "Accept-Language": "en"},
        timeout=httpx.Timeout(10.0, connect=5.0),
    )
    providers = []
    for name in config.search_providers:
        if name == "duckduckgo":
            providers.append(DuckDuckGoProvider(config.search_timeout, client))
        elif name == "wikipedia":
            providers.append(WikipediaProvider(config.search_timeout, client))
        else:
            log.warning("unknown search provider %r (known: duckduckgo, wikipedia, fixture)", name)

    cursor: dict[str, list] = {"rest": []}

    def search(request: SearchRequest) -> list[SearchHit]:
        hits, rest = search_chain(providers, request)
        cursor["rest"] = rest
        return hits

    def search_rest(request: SearchRequest) -> list[SearchHit]:
        hits, rest = search_chain(cursor["rest"], request)
        cursor["rest"] = rest
        return hits

    def fetch(url: str, deadline: float):
        return fetch_page(url, deadline, client, timeout=config.fetch_timeout)

    def close() -> None:
        client.close()

    services = Services(search=search, fetch=fetch, llm=llm, close=close)
    services.search_rest = search_rest
    return services


def search_chain(providers: list, request: SearchRequest) -> tuple[list[SearchHit], list]:
    """Try providers in order. Stop after the first one that returns hits.

    The second value is the providers not yet called, so a weak answer can ask
    the next source without repeating the first.
    """
    if not providers or time.monotonic() >= request.deadline:
        return [], []
    hits: list[SearchHit] = []
    for index, provider in enumerate(providers):
        if time.monotonic() >= request.deadline:
            return [], providers[index:]
        try:
            found = provider.search(request) or []
        except Exception as exc:
            log.warning("search provider failed: %s", exc)
            found = []
        hits.extend(found)
        deduped = _dedupe(hits)
        if deduped:
            return deduped[: request.limit], providers[index + 1 :]
    return [], []


def _dedupe(hits: list[SearchHit]) -> list[SearchHit]:
    seen: set[str] = set()
    unique: list[SearchHit] = []
    for hit in hits:
        if hit.url in seen:
            continue
        seen.add(hit.url)
        unique.append(hit)
    return unique
