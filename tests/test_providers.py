from __future__ import annotations

import time

import httpx

from iwir.http import CACHE, polite_get
from iwir.models import SearchRequest
from iwir.net import RequestGate
from iwir.search.bing import parse_bing_html
from iwir.search.brave import parse_brave_html
from iwir.search.direct import DirectReferenceProvider, likely_titles
from iwir.search.wikidata import _quantity, render_wikidata_snippet


def test_direct_titles_are_topics_not_copied_answers() -> None:
    titles = likely_titles("capital of France", "What is the capital of France?")
    assert titles == ["France"]
    titles = likely_titles(
        "Python programming language first released",
        "In what year was the Python programming language first released?",
    )
    assert "Python (programming language)" in titles
    deadline = time.monotonic() + 5
    request = SearchRequest(
        query="capital of Japan",
        question="What is the capital of Japan?",
        question_id="capital-japan",
        limit=3,
        deadline=deadline,
    )
    hits = DirectReferenceProvider().search(request)
    assert hits[0].url == "https://en.wikipedia.org/wiki/Japan"
    assert hits[0].title == "Japan"


def test_wikidata_snippet_uses_claims() -> None:
    entity = {
        "labels": {"en": {"value": "France"}},
        "descriptions": {"en": {"value": "country in Western Europe"}},
        "claims": {
            "P36": [
                {
                    "rank": "preferred",
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"type": "wikibase-entityid", "value": {"id": "Q90"}},
                    },
                }
            ]
        },
    }
    text = render_wikidata_snippet("What is the capital of France?", entity, {"Q90": "Paris"})
    assert "The capital of France is Paris." in text
    assert "country in Western Europe" in text


def test_boiling_point_units() -> None:
    celsius = {"amount": "+99.9839", "unit": "http://www.wikidata.org/entity/Q25267"}
    kelvin = {"amount": "+373.15", "unit": "http://www.wikidata.org/entity/Q11579"}
    assert _quantity("P2102", celsius) == "100"
    assert _quantity("P2102", kelvin) == "100"


def test_bing_and_brave_parsers() -> None:
    bing = """
    <li class="b_algo">
      <h2><a href="https://en.wikipedia.org/wiki/France">France - Wikipedia</a></h2>
      <div class="b_caption"><p>Its capital is Paris.</p></div>
    </li>
    <li class="b_algo">
      <h2><a href="https://www.bing.com/aclick?ld=1">Ad</a></h2>
    </li>
    """
    hits = parse_bing_html(bing)
    assert [hit.url for hit in hits] == ["https://en.wikipedia.org/wiki/France"]
    assert "Paris" in hits[0].snippet

    brave = """
    <div class="snippet" data-type="web">
      <a href="https://search.brave.com/redirect?url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FGold">
        <div class="title">Gold</div>
      </a>
      <div class="snippet-description">Gold has the symbol Au.</div>
    </div>
    """
    hits = parse_brave_html(brave)
    assert hits[0].url == "https://en.wikipedia.org/wiki/Gold"
    assert "Au" in hits[0].snippet


class _Scripted:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_polite_get_caches_and_retries(monkeypatch) -> None:
    monkeypatch.setattr("iwir.http._backoff_delay", lambda attempt, retry_after: 0.0)
    CACHE.clear()
    ok = httpx.Response(200, text='{"ok": true}', request=httpx.Request("GET", "https://example.test/a"))
    client = _Scripted(
        [
            httpx.Response(503, text="busy", request=httpx.Request("GET", "https://example.test/a")),
            ok,
        ]
    )
    gate = RequestGate(0)
    first = polite_get(
        client,
        "https://example.test/a",
        params={"q": "1"},
        timeout=1.0,
        deadline=time.monotonic() + 5,
        gate=gate,
        cache=CACHE,
    )
    assert first.status_code == 200
    assert client.calls == 2
    second = polite_get(
        client,
        "https://example.test/a",
        params={"q": "1"},
        timeout=1.0,
        deadline=time.monotonic() + 5,
        gate=gate,
        cache=CACHE,
    )
    assert second.status_code == 200
    assert client.calls == 2
    assert second.json()["ok"] is True


def test_blocked_gate_does_not_call_the_network() -> None:
    CACHE.clear()
    gate = RequestGate(0)
    gate.penalize(30)
    client = _Scripted([])
    response = polite_get(
        client,
        "https://example.test/blocked",
        deadline=time.monotonic() + 5,
        gate=gate,
        cache=CACHE,
    )
    assert response.status_code == 429
    assert client.calls == 0
