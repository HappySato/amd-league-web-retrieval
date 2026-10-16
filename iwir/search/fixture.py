"""Offline provider used by tests and ``eval/evaluate.py --mode fixture``."""

from __future__ import annotations

import json
from pathlib import Path

from iwir.browse import html_to_page
from iwir.models import Page, SearchHit, SearchRequest


class FixtureProvider:
    name = "fixture"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.bundle = json.loads(path.read_text(encoding="utf-8"))

    def search(self, request: SearchRequest) -> list[SearchHit]:
        table = self.bundle.get("search") or {}
        raw = table.get(request.question_id) or table.get(request.question) or []
        hits: list[SearchHit] = []
        for item in raw[: request.limit]:
            hits.append(
                SearchHit(
                    url=item["url"],
                    title=item.get("title") or "",
                    snippet=item.get("snippet") or "",
                )
            )
        return hits

    def fetch(self, url: str, deadline: float) -> Page | None:
        del deadline
        item = (self.bundle.get("pages") or {}).get(url)
        if not item:
            return None
        title = item.get("title") or ""
        if item.get("html"):
            return html_to_page(url, item["html"], title=title)
        pairs = []
        for pair in item.get("pairs") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pairs.append((str(pair[0]), str(pair[1])))
        return Page(url=url, title=title, text=item.get("text") or "", pairs=pairs, links=[])
