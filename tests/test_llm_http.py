"""LLM client against a local OpenAI-compatible server. No network beyond 127.0.0.1."""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from iwir.adapter import IOAdapter
from iwir.llm import LLMClient, models_url, parse_query_payload
from iwir.models import Page, Question, SearchHit
from iwir.pipeline import answer_question
from iwir.runner import main as runner_main
from iwir.search.chain import Services
from tests.conftest import FIXTURE, make_config

FRANCE = Question(id="capital-france", question="What is the capital of France?")
PAGE = Page(
    url="https://example.test/france",
    title="France",
    text="France is a country in Western Europe. Its capital is Paris.",
)


def _completion(content: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        self.server.events.append({"method": "GET", "path": self.path.split("?", 1)[0]})
        if self.path.split("?", 1)[0] == "/v1/models":
            body = b'{"object":"list","data":[{"id":"mock-model"}]}'
            self._send(200, body)
            return
        self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        body = json.loads(raw.decode() or "{}")
        self.server.events.append(
            {
                "method": "POST",
                "path": self.path.split("?", 1)[0],
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        system = ""
        messages = body.get("messages") or []
        if messages and isinstance(messages[0], dict):
            system = str(messages[0].get("content") or "")
        is_rewrite = "search queries" in system
        mode = self.server.mode
        if mode == "slow-extract" and not is_rewrite:
            time.sleep(2.5)
        if mode == "reject-json-mode" and "response_format" in body:
            self._send(400, b'{"error":{"message":"response_format is not supported"}}')
            return
        if mode == "invalid":
            self._send(200, _completion("The capital is Paris, but this is not JSON."))
            return
        if mode == "unsupported":
            if is_rewrite:
                self._send(200, _completion(json.dumps({"queries": ["France capital"]})))
            else:
                self._send(
                    200,
                    _completion(
                        json.dumps(
                            {
                                "answer": "Lyon",
                                "quote": "Lyon",
                                "source_url": PAGE.url,
                                "confidence": 0.99,
                            }
                        )
                    ),
                )
            return
        if is_rewrite:
            content = json.dumps({"queries": ["France capital", "France capital"]})
        else:
            content = json.dumps(
                {
                    "answer": "Paris",
                    "quote": "Its capital is Paris",
                    "source_url": PAGE.url,
                    "confidence": 0.93,
                }
            )
        self._send(200, _completion(content))

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mode = "ok"
        self.events: list[dict] = []


@pytest.fixture
def openai_server():
    server = _Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _base(server: _Server) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}/v1"


def _client(tmp_path, server: _Server, **overrides) -> LLMClient:
    config = make_config(
        tmp_path,
        llm_base_url=_base(server),
        llm_model="mock-model",
        llm_api_key="test-key",
        llm_timeout=overrides.pop("llm_timeout", 4.0),
        **overrides,
    )
    return LLMClient(config)


def _services(client: LLMClient, queries: list[str]) -> Services:
    def search(request):
        queries.append(request.query)
        return [SearchHit(url=PAGE.url, title=PAGE.title, snippet=PAGE.text)]

    return Services(
        search=search,
        fetch=lambda url, deadline: PAGE if url == PAGE.url else None,
        llm=client,
        close=lambda: None,
    )


def test_models_url_and_query_payload() -> None:
    assert models_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/models"
    queries = parse_query_payload(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "queries": [
                                    "France capital",
                                    "France capital",
                                    "x" * 200,
                                    "bad\x01query",
                                    "Paris city",
                                    "fourth query",
                                    "fifth query",
                                ]
                            }
                        )
                    }
                }
            ]
        }
    )
    assert queries == ["France capital", "Paris city", "fourth query"]
    single = parse_query_payload(
        {"choices": [{"message": {"content": json.dumps({"query": "capital of France"})}}]}
    )
    assert single == ["capital of France"]


def test_rewrite_then_extract_uses_the_cited_page(tmp_path, openai_server: _Server) -> None:
    queries: list[str] = []
    client = _client(tmp_path, openai_server)
    services = _services(client, queries)
    result = answer_question(FRANCE, make_config(tmp_path), services, time.monotonic() + 20)
    assert queries[0] == "France capital"
    assert result.answer == "Paris"
    assert result.sources == [PAGE.url]
    assert result.confidence >= 0.8
    methods = [event["method"] for event in openai_server.events]
    assert methods[0] == "GET"
    assert openai_server.events[0]["path"] == "/v1/models"
    posts = [event for event in openai_server.events if event["method"] == "POST"]
    assert posts
    assert all(event["auth"] == "Bearer test-key" for event in posts)
    assert all(event["body"]["model"] == "mock-model" for event in posts)
    assert "response_format" in posts[0]["body"]
    assert result.answer in PAGE.text


def test_answer_missing_from_the_page_is_dropped(tmp_path, openai_server: _Server) -> None:
    openai_server.mode = "unsupported"
    queries: list[str] = []
    client = _client(tmp_path, openai_server)
    result = answer_question(FRANCE, make_config(tmp_path), _services(client, queries), time.monotonic() + 20)
    assert queries[0] == "France capital"
    assert result.answer == "Paris"
    assert "Lyon" not in result.answer


def test_invalid_json_still_writes_a_file(tmp_path, openai_server: _Server, monkeypatch) -> None:
    openai_server.mode = "invalid"
    monkeypatch.setenv("LLM_BASE_URL", _base(openai_server))
    monkeypatch.setenv("LLM_MODEL", "mock-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "4")
    monkeypatch.setenv("QUESTION_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("SEARCH_PROVIDERS", "fixture")
    monkeypatch.setenv("FIXTURE_PATH", str(FIXTURE))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    for name in ("INPUT_PATH", "INPUT_JSON", "OUTPUT_FIELDS", "SCHEMA_PROFILE", "OUTPUT_FILENAME_TEMPLATE"):
        monkeypatch.delenv(name, raising=False)
    code = runner_main(["--query-id", "capital-france", "--query", "What is the capital of France?"])
    assert code == 0
    written = json.loads((tmp_path / "capital-france.json").read_text(encoding="utf-8"))
    assert written["answer"] == "Paris"
    assert written["sources"] == [PAGE.url]
    assert isinstance(written["confidence"], float)
    posts = [event for event in openai_server.events if event["method"] == "POST"]
    assert posts


def test_unreachable_server_uses_the_rule_extractor(tmp_path) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    config = make_config(
        tmp_path,
        llm_base_url=f"http://127.0.0.1:{port}/v1",
        llm_model="mock-model",
        llm_api_key="test-key",
        llm_timeout=2.0,
    )
    client = LLMClient(config)
    queries: list[str] = []
    services = _services(client, queries)
    started = time.monotonic()
    result = answer_question(FRANCE, config, services, started + 12)
    assert result.answer == "Paris"
    assert result.sources == [PAGE.url]
    assert client._down is True
    assert time.monotonic() - started < 6
    again = time.monotonic()
    result = answer_question(FRANCE, config, services, again + 5)
    assert result.answer == "Paris"
    assert time.monotonic() - again < 1.5
    payload = IOAdapter(config).write_result(result)
    body = json.loads(payload.read_text(encoding="utf-8"))
    assert body["id"] == "capital-france"
    assert body["answer"] == "Paris"


def test_slow_extract_falls_back_and_stays_up(tmp_path, openai_server: _Server) -> None:
    openai_server.mode = "slow-extract"
    client = _client(tmp_path, openai_server, llm_timeout=1.0)
    queries: list[str] = []
    started = time.monotonic()
    result = answer_question(
        FRANCE,
        make_config(tmp_path, llm_timeout=1.0),
        _services(client, queries),
        started + 20,
    )
    elapsed = time.monotonic() - started
    assert result.answer == "Paris"
    assert result.sources == [PAGE.url]
    assert elapsed < 6
    assert client._down is False
    assert client.available(time.monotonic() + 10) is True
    systems = [event["body"]["messages"][0]["content"] for event in openai_server.events if event["method"] == "POST"]
    assert any("search queries" in system for system in systems)
    assert any("extract one factual value" in system for system in systems)


def test_json_mode_400_retries_without_response_format(tmp_path, openai_server: _Server) -> None:
    openai_server.mode = "reject-json-mode"
    client = _client(tmp_path, openai_server)
    deadline = time.monotonic() + 20
    assert client.rewrite_queries("What is the capital of France?", deadline) == ["France capital"]
    extracted = client.extract(FRANCE.question, [PAGE], deadline)
    assert extracted is not None
    assert extracted.answer == "Paris"
    assert extracted.source_url == PAGE.url
    posts = [event for event in openai_server.events if event["method"] == "POST"]
    assert len(posts) == 4
    assert "response_format" in posts[0]["body"]
    assert "response_format" not in posts[1]["body"]
    assert "response_format" in posts[2]["body"]
    assert "response_format" not in posts[3]["body"]
    assert all(event["auth"] == "Bearer test-key" for event in posts)
