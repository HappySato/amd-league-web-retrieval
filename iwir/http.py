"""Polite HTTP: spacing, retries, and a short response cache.

Search endpoints and Wikimedia APIs share one cache for this process. A 429
is not cached. Retries cover connection failures and transient 5xx responses,
with exponential backoff and jitter, and they stop when the question deadline
is close. Retry-After is honored only when it is short enough to fit.
"""

from __future__ import annotations

import json
import random
import threading
import time
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from iwir.net import (
    BING_GATE,
    BRAVE_GATE,
    BROWSER_USER_AGENT,
    DUCKDUCKGO_GATE,
    RequestGate,
    WIKIDATA_GATE,
    WIKIPEDIA_GATE,
    WIKI_USER_AGENT,
)

_RETRY_STATUS = {500, 502, 503, 504}
_CACHE_TTL = 600.0
_CACHE_MAX = 256
_CACHE_BODY = 500_000


class CachedResponse:
    """Enough of an ``httpx.Response`` for the providers that call ``json()``."""

    def __init__(self, status_code: int, text: str, url: str, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers = {key.lower(): value for key, value in (headers or {}).items()}
        self._json: Any = None

    def json(self) -> Any:
        if self._json is None:
            self._json = json.loads(self.text)
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        request = httpx.Request("GET", self.url)
        response = httpx.Response(self.status_code, request=request, text=self.text)
        raise httpx.HTTPStatusError(
            f"HTTP {self.status_code} for {self.url}",
            request=request,
            response=response,
        )


class ResponseCache:
    def __init__(self, ttl: float = _CACHE_TTL, max_entries: int = _CACHE_MAX) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: dict[str, tuple[float, int, str, str, dict[str, str]]] = {}

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def get(self, key: str) -> CachedResponse | None:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires, status, text, url, headers = item
            if expires <= now:
                self._store.pop(key, None)
                return None
        return CachedResponse(status, text, url, headers)

    def put(self, key: str, status: int, text: str, url: str, headers: dict[str, str] | None = None) -> None:
        if status != 200 or not text or len(text) > _CACHE_BODY:
            return
        with self._lock:
            if len(self._store) >= self.max_entries:
                oldest = min(self._store, key=lambda name: self._store[name][0])
                self._store.pop(oldest, None)
            self._store[key] = (time.monotonic() + self.ttl, status, text, url, dict(headers or {}))


CACHE = ResponseCache()


def clear_cache() -> None:
    CACHE.clear()


def cache_key(url: str, params: dict | None) -> str:
    if not params:
        return url
    pairs = sorted((str(key), str(value)) for key, value in params.items())
    return url + "?" + urlencode(pairs)


def gate_for(url: str) -> RequestGate:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("wikipedia.org") or host.endswith("wikimedia.org"):
        return WIKIPEDIA_GATE
    if "wikidata.org" in host:
        return WIKIDATA_GATE
    if "duckduckgo.com" in host:
        return DUCKDUCKGO_GATE
    if host.endswith("bing.com"):
        return BING_GATE
    if "brave.com" in host:
        return BRAVE_GATE
    return _other_gate(host)


_OTHER: dict[str, RequestGate] = {}
_OTHER_LOCK = threading.Lock()


def _other_gate(host: str) -> RequestGate:
    with _OTHER_LOCK:
        gate = _OTHER.get(host)
        if gate is None:
            gate = RequestGate(0.4)
            _OTHER[host] = gate
        return gate


def _headers_for(url: str, headers: dict[str, str] | None) -> dict[str, str]:
    merged = dict(headers or {})
    host = (urlparse(url).hostname or "").lower()
    if host.endswith(("wikipedia.org", "wikimedia.org", "wikidata.org")):
        merged["User-Agent"] = WIKI_USER_AGENT
    elif not merged.get("User-Agent"):
        merged["User-Agent"] = BROWSER_USER_AGENT
    return merged


def _backoff_delay(attempt: int, retry_after: float | None) -> float:
    delay = min(3.0, 0.35 * (2**attempt)) + random.uniform(0.0, 0.25)
    if retry_after is not None and 0 < retry_after <= 2.5:
        delay = max(delay, retry_after)
    return delay


def _retry_after(headers: httpx.Headers | dict) -> float | None:
    raw = headers.get("retry-after") if headers else None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _room(deadline: float | None, needed: float) -> bool:
    if deadline is None:
        return True
    return time.monotonic() + needed < deadline


def polite_get(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | float | None = None,
    deadline: float | None = None,
    retries: int = 2,
    cache: ResponseCache | None = None,
    gate: RequestGate | None = None,
) -> httpx.Response | CachedResponse:
    """GET with cache, per-host spacing, and bounded retries."""
    store = CACHE if cache is None else cache
    key = cache_key(url, params)
    cached = store.get(key)
    if cached is not None:
        return cached

    chosen = gate if gate is not None else gate_for(url)
    if chosen.blocked():
        return CachedResponse(429, "", url)

    request_headers = _headers_for(url, headers)
    attempt = 0
    while True:
        if deadline is not None and time.monotonic() >= deadline - 0.15:
            raise httpx.TimeoutException("question deadline reached before request")
        chosen.wait()
        try:
            response = client.get(url, params=params, headers=request_headers, timeout=timeout)
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt >= retries or not _room(deadline, 0.6):
                raise
            time.sleep(min(_backoff_delay(attempt, None), _budget(deadline)))
            attempt += 1
            continue

        status = response.status_code
        if status in _RETRY_STATUS and attempt < retries and _room(deadline, 0.6):
            time.sleep(min(_backoff_delay(attempt, _retry_after(response.headers)), _budget(deadline)))
            attempt += 1
            continue
        if status == 429 and attempt == 0 and _room(deadline, 0.8):
            retry_after = _retry_after(response.headers)
            if retry_after is not None and retry_after <= 2.0:
                time.sleep(retry_after + random.uniform(0.0, 0.15))
                attempt += 1
                continue
        if status == 200:
            header_map = {key.lower(): value for key, value in response.headers.items()}
            store.put(key, status, response.text, str(response.url), header_map)
        return response


def _budget(deadline: float | None) -> float:
    if deadline is None:
        return 3.0
    return max(0.0, min(3.0, deadline - time.monotonic() - 0.25))
