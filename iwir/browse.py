"""Fetch a page and reduce it to text, label/value pairs, and links.

Wikipedia article URLs are read through the MediaWiki parse API (lead section
only, which includes the infobox). That is smaller and more stable than the
full desktop HTML. Other URLs are fetched directly, with a byte cap.
"""

from __future__ import annotations

import json
import logging
import re
import time
from urllib.parse import unquote, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from iwir.models import Link, Page
from iwir.net import BROWSER_USER_AGENT, WIKI_USER_AGENT, WIKIPEDIA_GATE, allowed_url

log = logging.getLogger(__name__)

_MAX_BYTES = 800_000
_WIKI_HOSTS = {"en.wikipedia.org", "en.m.wikipedia.org", "wikipedia.org"}
_SKIP_TAGS = ("script", "style", "noscript", "svg", "canvas", "iframe", "form")


def fetch_page(
    url: str,
    deadline: float,
    client: httpx.Client,
    *,
    timeout: float,
) -> Page | None:
    if time.monotonic() >= deadline or not allowed_url(url):
        return None
    title = wikipedia_title(url)
    if title:
        if WIKIPEDIA_GATE.blocked():
            return None
        WIKIPEDIA_GATE.wait()
        page = fetch_wikipedia_lead(title, deadline, client, timeout=timeout)
        if page is not None and page.text.strip():
            return page
        return None
    return fetch_direct(url, deadline, client, timeout=timeout)


def fetch_wikipedia_lead(
    title: str,
    deadline: float,
    client: httpx.Client,
    *,
    timeout: float,
) -> Page | None:
    try:
        response = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "parse",
                "page": title,
                "prop": "text",
                "section": "0",
                "format": "json",
                "formatversion": "2",
                "redirects": "1",
            },
            headers={"User-Agent": WIKI_USER_AGENT, "Accept": "application/json"},
            timeout=_timeout(timeout, deadline),
        )
        response.raise_for_status()
        page = page_from_wiki_parse(response.json())
        if page is not None and page.text.strip():
            return page
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        if _rate_limited(exc):
            WIKIPEDIA_GATE.penalize(20)
            log.warning("wikipedia rate limited during fetch; pausing 20s")
            return None
        log.warning("wikipedia parse failed for %s: %s", title, exc)

    slug = title.replace(" ", "_")
    summary_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + slug
    try:
        response = client.get(
            summary_url,
            headers={"User-Agent": WIKI_USER_AGENT, "Accept": "application/json"},
            timeout=_timeout(timeout, deadline),
        )
        response.raise_for_status()
        return page_from_wiki_summary(response.json())
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        log.warning("wikipedia summary failed for %s: %s", title, exc)
        return None


def _rate_limited(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def page_from_wiki_parse(payload: dict) -> Page | None:
    parse = payload.get("parse") or {}
    html = parse.get("text") or ""
    if isinstance(html, dict):
        html = html.get("*") or ""
    title = str(parse.get("title") or "").strip()
    if not html or not title:
        return None
    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    return html_to_page(url, str(html), title=title)


def page_from_wiki_summary(payload: dict) -> Page | None:
    extract = str(payload.get("extract") or "").strip()
    title = str(payload.get("title") or "").strip()
    if not extract:
        return None
    desktop = ((payload.get("content_urls") or {}).get("desktop") or {}).get("page")
    url = str(desktop or "")
    if not url and title:
        url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
    if not allowed_url(url):
        return None
    return Page(url=url, title=title, text=extract, pairs=[], links=[])


def fetch_direct(
    url: str,
    deadline: float,
    client: httpx.Client,
    *,
    timeout: float,
) -> Page | None:
    try:
        with client.stream(
            "GET",
            url,
            headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "text/html,application/json;q=0.9,*/*;q=0.5"},
            timeout=_timeout(timeout, deadline),
        ) as response:
            response.raise_for_status()
            final_url = str(response.url)
            if not allowed_url(final_url):
                log.warning("refusing redirected target %s", final_url)
                return None
            content_type = response.headers.get("content-type", "")
            raw = _read_capped(response, deadline)
    except httpx.HTTPError as exc:
        log.warning("fetch failed for %s: %s", url, exc)
        return None
    if not raw:
        return None
    text = _decode(raw, content_type)
    if "json" in content_type.lower() or text.lstrip()[:1] in "{[":
        page = page_from_json(final_url, text)
        if page is not None:
            return page
    if "html" not in content_type.lower() and "<html" not in text[:500].casefold() and "<p" not in text[:500].casefold():
        if not text.strip():
            return None
        # Plain text is still evidence.
        if "html" not in content_type.lower() and not text.lstrip().startswith("<"):
            return Page(url=final_url, title="", text=_collapse(text)[:20_000], pairs=[], links=[])
    return html_to_page(final_url, text)


def html_to_page(url: str, html: str, title: str = "") -> Page:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_SKIP_TAGS):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()

    if not title:
        heading = soup.find("title")
        if heading:
            title = heading.get_text(" ", strip=True)

    pairs = _infobox_pairs(soup)
    links = _links(url, soup)
    body = soup.get_text("\n", strip=True)
    pair_lines = [f"{label}: {value}" for label, value in pairs]
    text = _collapse("\n".join(pair_lines + [body]))
    return Page(url=url, title=title, text=text[:20_000], pairs=pairs, links=links)


def page_from_json(url: str, raw: str) -> Page | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    lines = flatten_json(payload)[:200]
    if not lines:
        return None
    pairs: list[tuple[str, str]] = []
    for line in lines:
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        label = key.split(".")[-1]
        label = re.sub(r"\[\d+\]", "", label).strip()
        value = value.strip()
        if label and value and len(value) <= 200:
            pairs.append((label, value))
    return Page(url=url, title="", text="\n".join(lines), pairs=pairs[:40], links=[])


def flatten_json(value: object, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_json(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value[:40]):
            path = f"{prefix}[{index}]"
            lines.extend(flatten_json(item, path))
    else:
        rendered = "" if value is None else str(value)
        if prefix:
            lines.append(f"{prefix}: {rendered}")
        elif rendered:
            lines.append(rendered)
    return lines


def wikipedia_title(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in _WIKI_HOSTS and not host.endswith(".wikipedia.org"):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 2 and parts[0] == "wiki":
        slug = unquote(parts[1])
        if slug in {"Special:Search", "File", "Help"} or ":" in slug:
            return None
        return slug.replace("_", " ")
    return None


def _infobox_pairs(soup: BeautifulSoup) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    tables = soup.select("table.infobox")
    if not tables:
        tables = soup.find_all("table")
    for table in tables[:2]:
        classes = " ".join(table.get("class") or [])
        if "infobox" not in classes and len(tables) > 1:
            continue
        for row in table.find_all("tr"):
            header = row.find("th")
            cell = row.find("td")
            if header is None or cell is None:
                continue
            label = re.sub(r"\s+", " ", header.get_text(" ", strip=True)).strip(":")
            value = _cell_value(cell)
            if label and value:
                pairs.append((label, value))
    return pairs[:30]


def _cell_value(cell) -> str:
    for br in cell.find_all("br"):
        br.replace_with("\n")
    lines = []
    for line in cell.get_text("\n", strip=True).splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if not cleaned:
            continue
        if re.fullmatch(r"[\d°′″'\"\sNSEW.,/−/+-]+", cleaned):
            continue
        lines.append(cleaned)
    return lines[0] if lines else ""


def _links(base_url: str, soup: BeautifulSoup) -> list[Link]:
    links: list[Link] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        text = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True))
        if not text or len(text) > 120:
            continue
        href = urljoin(base_url, anchor["href"])
        href = href.split("#", 1)[0]
        if not allowed_url(href) or href in seen:
            continue
        seen.add(href)
        links.append(Link(url=href, text=text))
        if len(links) >= 40:
            break
    return links


def _read_capped(response: httpx.Response, deadline: float) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        if time.monotonic() > deadline:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total >= _MAX_BYTES:
            break
    return b"".join(chunks)


def _decode(raw: bytes, content_type: str) -> str:
    match = re.search(r"charset=([\w.-]+)", content_type, re.I)
    encoding = match.group(1) if match else "utf-8"
    try:
        return raw.decode(encoding, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _collapse(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    kept = [line for line in lines if line]
    return "\n".join(kept)


def _timeout(limit: float, deadline: float) -> httpx.Timeout:
    remaining = max(0.4, deadline - time.monotonic())
    budget = min(limit, remaining)
    return httpx.Timeout(budget, connect=min(3.0, budget))
