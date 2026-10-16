"""URL policy and client headers.

Wikipedia asks for an identifiable user agent. General fetches use a browser
user agent because several public search HTML endpoints reject an empty one.
Literal private and link-local targets are refused so a question cannot point
the fetcher at cloud metadata or the local machine.
"""

from __future__ import annotations

import ipaddress
import re
import threading
import time
from urllib.parse import urlparse

WIKI_USER_AGENT = (
    "IWIR-Prototype/0.1 (AMD AI League match 4 educational prototype; "
    "local retrieval, no account)"
)
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

_BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "169.254.169.254",
    "metadata.google.internal",
}

_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"']+")


def allowed_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host in _BLOCKED_HOSTS:
        return False
    if host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


class RequestGate:
    """Minimum spacing between calls to one host, so a batch does not get banned."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            if delay > 0:
                time.sleep(delay)
            self._next = time.monotonic() + self.interval

    def penalize(self, seconds: float) -> None:
        with self._lock:
            self._blocked_until = time.monotonic() + seconds

    def blocked(self) -> bool:
        with self._lock:
            return time.monotonic() < getattr(self, "_blocked_until", 0.0)


DUCKDUCKGO_GATE = RequestGate(0.7)
WIKIPEDIA_GATE = RequestGate(0.4)


def urls_in_text(text: str) -> list[str]:
    found: list[str] = []
    for match in _URL_IN_TEXT.findall(text or ""):
        cleaned = match.rstrip(").,;]")
        if allowed_url(cleaned) and cleaned not in found:
            found.append(cleaned)
    return found
