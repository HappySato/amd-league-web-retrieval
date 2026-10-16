from __future__ import annotations

import json
from pathlib import Path

from iwir.adapter import IOAdapter, parse_questions_payload
from iwir.models import AnswerResult
from iwir.runner import main
from tests.conftest import FIXTURE, make_config


def _result(qid: str = "capital-france", answer: str = "Paris") -> AnswerResult:
    return AnswerResult(
        id=qid,
        question="What is the capital of France?",
        answer=answer,
        sources=["https://example.test/france"] if answer else [],
        confidence=0.9 if answer else 0.0,
        method="infobox",
    )


def test_default_schema_writes_one_file(tmp_path: Path) -> None:
    adapter = IOAdapter(make_config(tmp_path))
    path = adapter.write_result(_result())
    files = list(tmp_path.glob("*.json"))
    assert files == [path]
    assert path.name == "capital-france.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"id", "question", "answer", "sources", "confidence"}
    assert payload["answer"] == "Paris"
    assert payload["sources"] == ["https://example.test/france"]
    assert isinstance(payload["confidence"], float)
    assert not list(tmp_path.glob("*.tmp"))


def test_league3_profile_and_custom_fields(tmp_path: Path) -> None:
    league = IOAdapter(make_config(tmp_path, schema_profile="league3"))
    path = league.write_result(_result("query_01"))
    assert path.name == "query_01_output.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"answer", "citations", "confidence"}
    assert payload["citations"] == ["https://example.test/france"]

    custom_dir = tmp_path / "custom"
    custom = IOAdapter(
        make_config(
            custom_dir,
            output_fields=["answer", "sources"],
            filename_template="{id}_output.json",
            include_diagnostics=True,
        )
    )
    path = custom.write_result(_result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"answer", "sources", "diagnostics"}
    assert payload["diagnostics"]["method"] == "infobox"


def test_unsafe_id_stays_inside_output_dir(tmp_path: Path) -> None:
    adapter = IOAdapter(make_config(tmp_path))
    path = adapter.write_result(_result("../etc/passwd"))
    assert path.parent == tmp_path
    assert ".." not in path.parts[len(tmp_path.parts) :]


def test_json_jsonl_and_directory(tmp_path: Path) -> None:
    batch = tmp_path / "questions.json"
    batch.write_text(
        json.dumps(
            {
                "questions": [
                    {"query_id": "a", "query": "First?"},
                    {"id": "a", "question": "duplicate"},
                    {"qid": "b", "text": "Second?", "url": "https://example.test/b"},
                ]
            }
        ),
        encoding="utf-8",
    )
    adapter = IOAdapter(make_config(tmp_path, input_path=batch))
    questions = adapter.load_questions()
    assert [item.id for item in questions] == ["a", "b"]
    assert questions[1].start_url == "https://example.test/b"

    jsonl = tmp_path / "more.jsonl"
    jsonl.write_text('{"id":"c","question":"Third?"}\n{"id":"d","question":"Fourth?"}\n', encoding="utf-8")
    loaded = IOAdapter(make_config(tmp_path, input_path=jsonl)).load_questions()
    assert [item.id for item in loaded] == ["c", "d"]

    folder = tmp_path / "inbox"
    folder.mkdir()
    (folder / "one.json").write_text('{"id":"e","question":"Fifth?"}', encoding="utf-8")
    (folder / "skip.example.json").write_text('{"id":"nope","question":"skip"}', encoding="utf-8")
    from_dir = IOAdapter(make_config(tmp_path, input_path=folder)).load_questions()
    assert [item.id for item in from_dir] == ["e"]


def test_cli_query_overrides_input_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SEARCH_PROVIDERS", "fixture")
    monkeypatch.setenv("FIXTURE_PATH", str(FIXTURE))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("QUESTION_TIMEOUT_SECONDS", "8")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("INPUT_PATH", raising=False)
    code = main(
        [
            "--query-id",
            "capital-france",
            "--query",
            "What is the capital of France?",
            "--output",
            str(tmp_path),
        ]
    )
    assert code == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["id"] == "capital-france"
    assert payload["answer"] == "Paris"
    assert payload["sources"] == ["https://example.test/france"]


def test_batch_main_writes_one_file_per_question(tmp_path: Path, monkeypatch) -> None:
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {"id": "capital-france", "question": "What is the capital of France?"},
                {
                    "id": "unanswerable-part",
                    "question": "What is the lead time in days for part ZX-9000-NO-SUCH on the supplier portal?",
                },
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    monkeypatch.setenv("SEARCH_PROVIDERS", "fixture")
    monkeypatch.setenv("FIXTURE_PATH", str(FIXTURE))
    monkeypatch.setenv("QUESTION_TIMEOUT_SECONDS", "8")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    code = main(["--input", str(questions), "--output", str(out)])
    assert code == 0
    files = sorted(path.name for path in out.glob("*.json"))
    assert files == ["capital-france.json", "unanswerable-part.json"]
    empty = json.loads((out / "unanswerable-part.json").read_text(encoding="utf-8"))
    assert empty["answer"] == ""
    assert empty["sources"] == []
    assert empty["confidence"] == 0.0


def test_inline_payload_aliases() -> None:
    questions = parse_questions_payload({"id": "z", "prompt": "Hello?"})
    assert questions[0].question == "Hello?"
