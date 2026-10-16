from __future__ import annotations

from iwir.browse import _promote_header_facts, html_to_page, page_from_json, page_from_wiki_parse
from iwir.net import allowed_url
from iwir.search.duckduckgo import parse_ddg_html, unwrap_ddg_href
from iwir.search.wikipedia import parse_wiki_search

DDG_HTML = """
<div class="result results_links">
  <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FFrance&amp;rut=abc">France</a>
  <a class="result__snippet">France is a country. Its capital is Paris.</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.test/direct">Direct</a>
  <span class="result__snippet">A direct link.</span>
</div>
"""


def test_duckduckgo_parser_unwraps_redirects() -> None:
    hits = parse_ddg_html(DDG_HTML)
    assert [hit.url for hit in hits] == [
        "https://en.wikipedia.org/wiki/France",
        "https://example.test/direct",
    ]
    assert "Paris" in hits[0].snippet
    assert unwrap_ddg_href("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Fa") == "https://example.test/a"
    assert unwrap_ddg_href("https://duckduckgo.com/html/") == ""


def test_duckduckgo_ads_are_dropped() -> None:
    html = """
    <div class="result">
      <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fduckduckgo.com%2Fy.js%3Fad_domain%3Dexample.com&rut=1">Sponsored</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://en.wikipedia.org/wiki/Paris">Paris</a>
    </div>
    """
    hits = parse_ddg_html(html)
    assert [hit.url for hit in hits] == ["https://en.wikipedia.org/wiki/Paris"]


def test_wikipedia_search_parser() -> None:
    payload = {
        "query": {
            "search": [
                {
                    "title": "Nineteen Eighty-Four",
                    "snippet": "A novel by <span class=\"searchmatch\">George Orwell</span>.",
                }
            ]
        }
    }
    hits = parse_wiki_search(payload)
    assert hits[0].title == "Nineteen Eighty-Four"
    assert hits[0].url == "https://en.wikipedia.org/wiki/Nineteen_Eighty-Four"
    assert "George Orwell" in hits[0].snippet
    assert "<span" not in hits[0].snippet


def test_infobox_drops_coordinate_line() -> None:
    html = """
    <html><body>
      <nav>Home</nav>
      <table class="infobox"><tr><th>Capital</th><td>Paris<br>48°51′N 2°21′E</td></tr></table>
      <p>Its capital is Paris.</p>
      <footer>About</footer>
    </body></html>
    """
    parsed = html_to_page("https://example.test/france", html)
    assert ("Capital", "Paris") in parsed.pairs
    assert "Home" not in parsed.text
    assert "About" not in parsed.text
    assert "Paris" in parsed.text


def test_wiki_parse_and_json_flatten() -> None:
    payload = {
        "parse": {
            "title": "France",
            "text": "<table class='infobox'><tr><th>Capital</th><td>Paris</td></tr></table>",
        }
    }
    parsed = page_from_wiki_parse(payload)
    assert parsed is not None
    assert parsed.url == "https://en.wikipedia.org/wiki/France"
    assert ("Capital", "Paris") in parsed.pairs

    flat = page_from_json("https://example.test/api", '{"item": {"lead_time_days": 12}}')
    assert flat is not None
    assert "lead_time_days: 12" in flat.text
    assert ("lead_time_days", "12") in flat.pairs


def test_element_header_lines_become_pairs() -> None:
    pairs, text = _promote_header_facts(
        'Symbol\n"Au": from Latin\nAtomic number\n6. It is nonmetallic.\n'
        "chemical symbol\nis\nAu\nBoiling point\n99.98 °C (211.96 °F)"
    )
    assert ("Symbol", "Au") in pairs
    assert ("Atomic number", "6") in pairs
    assert ("Boiling point", "99.98") in pairs
    assert "Symbol: Au" in text
    assert "Atomic number: 6" in text


def test_blocks_metadata_and_loopback() -> None:
    assert allowed_url("https://en.wikipedia.org/wiki/France")
    assert not allowed_url("http://169.254.169.254/latest/meta-data")
    assert not allowed_url("http://127.0.0.1/")
    assert not allowed_url("file:///etc/passwd")
    assert not allowed_url("http://localhost/admin")
