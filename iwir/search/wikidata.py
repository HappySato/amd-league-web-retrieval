"""Wikidata search. No API key.

``wbsearchentities`` finds an item, ``wbgetentities`` reads a few claims, and
the provider turns those claims into short sentences the extractor already
understands. The hit URL is the English Wikipedia article when one exists, so
a normal fetch can still read the infobox. The sentences stay in the snippet
for when that fetch is rate-limited.
"""

from __future__ import annotations

import logging
import re
import time

import httpx

from iwir.http import polite_get
from iwir.models import SearchHit, SearchRequest
from iwir.net import WIKIDATA_GATE
from iwir.search.direct import wikipedia_article_url
from iwir.search.wikipedia import _remaining_timeout, wikipedia_search_query

log = logging.getLogger(__name__)

_API = "https://www.wikidata.org/w/api.php"
_CELSIUS = "Q25267"
_KELVIN = "Q11579"

# Question cue -> Wikidata property ids, in preference order.
_PROPERTY_CUES: list[tuple[str, tuple[str, ...]]] = [
    ("atomic number", ("P1086",)),
    ("boiling", ("P2102",)),
    ("chemical symbol", ("P246",)),
    ("symbol", ("P246",)),
    ("capital", ("P36",)),
    ("currency", ("P38",)),
    ("official language", ("P37",)),
    ("language", ("P37",)),
    ("painted", ("P170",)),
    ("painter", ("P170",)),
    ("author", ("P50",)),
    ("wrote", ("P50",)),
    ("written", ("P50",)),
    ("published", ("P577", "P571")),
    ("released", ("P571", "P577")),
    ("founded", ("P571",)),
    ("born", ("P569",)),
]


class WikidataProvider:
    name = "wikidata"

    def __init__(self, timeout: float, client: httpx.Client) -> None:
        self.timeout = timeout
        self.client = client

    def search(self, request: SearchRequest) -> list[SearchHit]:
        if time.monotonic() >= request.deadline - 0.4 or WIKIDATA_GATE.blocked():
            return []
        query = wikipedia_search_query(request.query)
        found = self._api(
            request.deadline,
            {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "uselang": "en",
                "format": "json",
                "limit": 1,
                "type": "item",
            },
        )
        matches = (found or {}).get("search") or []
        if not matches or not isinstance(matches[0], dict):
            return []
        qid = str(matches[0].get("id") or "")
        if not qid.startswith("Q"):
            return []
        payload = self._api(
            request.deadline,
            {
                "action": "wbgetentities",
                "ids": qid,
                "props": "claims|sitelinks|labels|descriptions",
                "languages": "en",
                "format": "json",
            },
        )
        entity = ((payload or {}).get("entities") or {}).get(qid) or {}
        if not isinstance(entity, dict) or entity.get("missing") == "":
            return []
        needed = entity_ids_for(request.question, entity)
        labels = {qid: _label(entity)}
        if needed:
            resolved = self._api(
                request.deadline,
                {
                    "action": "wbgetentities",
                    "ids": "|".join(needed[:12]),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
            )
            labels.update(labels_from(resolved or {}))
        snippet = render_wikidata_snippet(request.question, entity, labels)
        if not snippet.strip():
            return []
        title = _label(entity) or str(matches[0].get("label") or qid)
        url = _enwiki_url(entity) or f"https://www.wikidata.org/wiki/{qid}"
        return [SearchHit(url=url, title=title, snippet=snippet)]

    def _api(self, deadline: float, params: dict) -> dict | None:
        try:
            response = polite_get(
                self.client,
                _API,
                params=params,
                headers={"Accept": "application/json"},
                timeout=_remaining_timeout(self.timeout, deadline),
                deadline=deadline,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                WIKIDATA_GATE.penalize(45)
                log.warning("wikidata rate limited; skipping it for 45s")
            else:
                log.warning("wikidata request failed: %s", exc)
            return None
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("wikidata request failed: %s", exc)
            return None
        return payload if isinstance(payload, dict) else None


def wanted_properties(question: str) -> list[str]:
    folded = question.casefold()
    chosen: list[str] = []
    for cue, props in _PROPERTY_CUES:
        if cue in folded:
            for prop in props:
                if prop not in chosen:
                    chosen.append(prop)
    return chosen


def entity_ids_for(question: str, entity: dict) -> list[str]:
    claims = entity.get("claims") or {}
    found: list[str] = []
    for prop in wanted_properties(question):
        kind, raw = best_claim(claims, prop)
        if kind == "entity" and raw and raw not in found:
            found.append(raw)
    return found


def labels_from(payload: dict) -> dict[str, str]:
    labels: dict[str, str] = {}
    for qid, entity in (payload.get("entities") or {}).items():
        if isinstance(entity, dict):
            label = _label(entity)
            if label:
                labels[str(qid)] = label
    return labels


def render_wikidata_snippet(question: str, entity: dict, labels: dict[str, str]) -> str:
    label = _label(entity)
    description = _description(entity)
    lines: list[str] = []
    if label and description:
        lines.append(f"{label} is {description}.")
    claims = entity.get("claims") or {}
    for prop in wanted_properties(question):
        kind, raw = best_claim(claims, prop)
        sentence = _sentence(prop, label, kind, raw, labels)
        if sentence:
            lines.append(sentence)
    return "\n".join(lines)


def best_claim(claims: dict, prop: str) -> tuple[str, object]:
    items = claims.get(prop) or []
    if not isinstance(items, list):
        return "", ""
    ordered = sorted(items, key=lambda item: 0 if isinstance(item, dict) and item.get("rank") == "preferred" else 1)
    for item in ordered:
        if not isinstance(item, dict):
            continue
        kind, raw = _snak(item.get("mainsnak") or {})
        if kind:
            return kind, raw
    return "", ""


def _snak(snak: dict) -> tuple[str, object]:
    if not isinstance(snak, dict) or snak.get("snaktype") != "value":
        return "", ""
    data = snak.get("datavalue") or {}
    if not isinstance(data, dict):
        return "", ""
    kind = str(data.get("type") or "")
    value = data.get("value")
    if kind == "wikibase-entityid" and isinstance(value, dict):
        return "entity", str(value.get("id") or "")
    if kind == "string" and isinstance(value, str):
        return "string", value.strip()
    if kind == "monolingualtext" and isinstance(value, dict):
        return "string", str(value.get("text") or "").strip()
    if kind == "time" and isinstance(value, dict):
        return "time", str(value.get("time") or "")
    if kind == "quantity" and isinstance(value, dict):
        return "quantity", value
    return "", ""


def _sentence(prop: str, label: str, kind: str, raw: object, labels: dict[str, str]) -> str:
    if not label:
        return ""
    value = _display(prop, kind, raw, labels)
    if not value:
        return ""
    if prop == "P36":
        return f"The capital of {label} is {value}."
    if prop == "P38":
        return f"The currency of {label} is the {value}."
    if prop == "P37":
        return f"The official language of {label} is {value}."
    if prop == "P246":
        return f"The chemical symbol for {label} is {value}."
    if prop == "P50":
        return f"{label} is a work written by {value}."
    if prop == "P170":
        return f"{label} was painted by {value}."
    if prop == "P1086":
        return f"The atomic number of {label} is {value}."
    if prop == "P2102":
        return f"The boiling point of {label} is {value}."
    if prop in {"P571", "P577", "P569"}:
        year = re.search(r"[12][0-9]{3}", value)
        if not year:
            return ""
        verb = "born" if prop == "P569" else "first released"
        return f"{label} was {verb} in {year.group(0)}."
    return ""


def _display(prop: str, kind: str, raw: object, labels: dict[str, str]) -> str:
    if kind == "entity":
        return labels.get(str(raw), "")
    if kind == "string":
        return str(raw).strip()
    if kind == "time":
        match = re.search(r"([12][0-9]{3})", str(raw))
        return match.group(1) if match else ""
    if kind == "quantity" and isinstance(raw, dict):
        return _quantity(prop, raw)
    return ""


def _quantity(prop: str, raw: dict) -> str:
    try:
        amount = float(raw.get("amount"))
    except (TypeError, ValueError):
        return ""
    unit = str(raw.get("unit") or "")
    unit_id = unit.rsplit("/", 1)[-1]
    # Q25267 is degree Celsius. Q11579 is kelvin. Subtracting 273 from a
    # Celsius value turned water's 99.98 °C into 173.
    if prop == "P2102" and unit_id == _KELVIN:
        amount -= 273.15
    if abs(amount - round(amount)) < 0.2:
        return str(int(round(amount)))
    text = f"{amount:.4f}".rstrip("0").rstrip(".")
    return text


def _label(entity: dict) -> str:
    labels = entity.get("labels") or {}
    en = labels.get("en") if isinstance(labels, dict) else None
    if isinstance(en, dict):
        return str(en.get("value") or "").strip()
    return ""


def _description(entity: dict) -> str:
    descriptions = entity.get("descriptions") or {}
    en = descriptions.get("en") if isinstance(descriptions, dict) else None
    if isinstance(en, dict):
        return str(en.get("value") or "").strip().rstrip(".")
    return ""


def _enwiki_url(entity: dict) -> str:
    sitelinks = entity.get("sitelinks") or {}
    enwiki = sitelinks.get("enwiki") if isinstance(sitelinks, dict) else None
    if isinstance(enwiki, dict):
        title = str(enwiki.get("title") or "").strip()
        if title:
            return wikipedia_article_url(title)
    return ""
