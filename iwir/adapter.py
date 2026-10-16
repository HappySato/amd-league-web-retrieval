"""Input and output adapter.

This is the only module that decides where questions come from and which keys
are written. When the official Match 4 spec settles, change this file or set
the environment variables below. Leave the retriever alone.

Profiles
--------
default
    ``{id}.json`` with ``id``, ``question``, ``answer``, ``sources``, ``confidence``.
league3
    Match-3-shaped ``{id}_output.json`` with ``answer``, ``citations``, ``confidence``.
    Useful if the revised PDF keeps that shape and only swaps file paths for URLs.

Overrides
---------
``OUTPUT_FIELDS``
    Comma-separated keys, in order. Known keys: id, query_id, question, query,
    answer, sources, citations, source, confidence.
``OUTPUT_FILENAME_TEMPLATE``
    Python format string using ``{id}``, for example ``{id}_output.json``.
``OUTPUT_INCLUDE_DIAGNOSTICS``
    When true, add a ``diagnostics`` object (error, method). Off by default so
    the file stays limited to the answer schema.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from iwir.config import Config, ConfigError, in_container_image
from iwir.models import AnswerResult, Question

log = logging.getLogger(__name__)

_PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "fields": ["id", "question", "answer", "sources", "confidence"],
        "filename": "{id}.json",
    },
    "league3": {
        "fields": ["answer", "citations", "confidence"],
        "filename": "{id}_output.json",
    },
}

_ID_ALIASES = ("id", "query_id", "qid", "uid")
_QUESTION_ALIASES = ("question", "query", "text", "prompt", "q")
_URL_ALIASES = ("start_url", "startUrl", "url", "page", "source_url")
_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


class IOAdapter:
    def __init__(self, config: Config) -> None:
        self.config = config
        profile_name = config.schema_profile or "default"
        if profile_name not in _PROFILES:
            known = ", ".join(sorted(_PROFILES))
            raise ConfigError(f"unknown SCHEMA_PROFILE {profile_name!r}; expected one of: {known}")
        profile = _PROFILES[profile_name]
        self.fields: list[str] = list(config.output_fields or profile["fields"])
        self.filename_template: str = config.filename_template or profile["filename"]
        self.output_dir = config.output_dir

    def load_questions(self) -> list[Question]:
        if self.config.query:
            qid = self.config.query_id or "q001"
            return [
                Question(
                    id=qid,
                    question=self.config.query.strip(),
                    start_url=self.config.start_url or None,
                )
            ]

        if self.config.input_json.strip() and self.config.input_path is None:
            try:
                payload = json.loads(self.config.input_json)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"INPUT_JSON is not valid JSON: {exc}") from exc
            return parse_questions_payload(payload)

        path = self._resolve_input_path()
        if path is None:
            return []
        if path.is_dir():
            return _load_directory(path)
        return _load_file(path)

    def payload(self, result: AnswerResult) -> dict[str, Any]:
        sources = list(dict.fromkeys(url for url in result.sources if url))
        confidence = _confidence(result.confidence)
        table: dict[str, Any] = {
            "id": result.id,
            "query_id": result.id,
            "question": result.question,
            "query": result.question,
            "answer": result.answer if isinstance(result.answer, str) else str(result.answer),
            "sources": sources,
            "citations": sources,
            "source": sources[0] if sources else "",
            "confidence": confidence,
        }
        body: dict[str, Any] = {}
        for key in self.fields:
            if key not in table:
                log.warning("skipping unknown output field %r", key)
                continue
            body[key] = table[key]
        if not body:
            for key in _PROFILES["default"]["fields"]:
                body[key] = table[key]
        if self.config.include_diagnostics:
            body["diagnostics"] = {
                "error": result.error,
                "method": result.method,
            }
        return body

    def write_result(self, result: AnswerResult) -> Path:
        path = self.output_path(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.payload(result), ensure_ascii=False, indent=2) + "\n"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(data, encoding="utf-8")
        temporary.replace(path)
        return path

    def output_path(self, result: AnswerResult) -> Path:
        safe = safe_id(result.id)
        try:
            name = self.filename_template.format(id=safe)
        except (KeyError, IndexError, ValueError) as exc:
            raise ConfigError(
                f"OUTPUT_FILENAME_TEMPLATE {self.filename_template!r} must use {{id}}"
            ) from exc
        if not name.endswith(".json") or "/" in name or "\\" in name or name.startswith("."):
            name = f"{safe}.json"
        return self.output_dir / name

    def _resolve_input_path(self) -> Path | None:
        if self.config.input_path is not None:
            return self.config.input_path
        local_file = Path("input/questions.json")
        if in_container_image():
            docker_input = Path("/app/input")
            if docker_input.is_dir():
                questions = docker_input / "questions.json"
                if questions.is_file():
                    return questions
                if any(_json_files(docker_input)):
                    return docker_input
        if local_file.is_file():
            return local_file
        return None


def safe_id(value: str) -> str:
    cleaned = _SAFE_ID.sub("_", value.strip())[:80].strip("._")
    return cleaned or "question"


def parse_questions_payload(payload: Any) -> list[Question]:
    if isinstance(payload, dict):
        for key in ("questions", "items", "queries", "data"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ConfigError("questions payload must be a list or an object with a questions list")
    questions: list[Question] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            log.warning("skipping question %s: expected an object", index)
            continue
        question = _question_from_object(item, index)
        if question is None:
            continue
        questions.append(question)
    return questions


def _question_from_object(item: dict[str, Any], index: int) -> Question | None:
    text = _first_str(item, _QUESTION_ALIASES)
    if not text:
        log.warning("skipping question %s: missing question text", index)
        return None
    qid = _first_str(item, _ID_ALIASES) or f"q{index:03d}"
    start = _first_str(item, _URL_ALIASES) or None
    seeds: list[str] = []
    extra = item.get("urls") or item.get("seed_urls") or []
    if isinstance(extra, str):
        extra = [extra]
    if isinstance(extra, list):
        for url in extra:
            if isinstance(url, str) and url.strip():
                seeds.append(url.strip())
    if start and start in seeds:
        seeds = [url for url in seeds if url != start]
    return Question(id=qid, question=text.strip(), start_url=start, seed_urls=seeds)


def _first_str(item: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return ""


def _load_file(path: Path) -> list[Question]:
    if not path.is_file():
        raise ConfigError(f"input file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or _looks_like_jsonl(text):
        questions: list[Question] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("skipping %s line %s: %s", path, line_no, exc)
                continue
            questions.extend(parse_questions_payload(item))
        return _dedupe_ids(questions)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc
    return _dedupe_ids(parse_questions_payload(payload))


def _looks_like_jsonl(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if text.lstrip().startswith("["):
        return False
    return all(line.lstrip().startswith("{") for line in lines)


def _load_directory(path: Path) -> list[Question]:
    named = path / "questions.json"
    if named.is_file():
        return _load_file(named)
    questions: list[Question] = []
    for file in _json_files(path):
        questions.extend(_load_file(file))
    return _dedupe_ids(questions)


def _json_files(path: Path) -> list[Path]:
    files = []
    for file in sorted(path.glob("*.json")):
        if file.name.endswith(".example.json"):
            continue
        files.append(file)
    files.extend(sorted(path.glob("*.jsonl")))
    return files


def _dedupe_ids(questions: list[Question]) -> list[Question]:
    seen: set[str] = set()
    unique: list[Question] = []
    for question in questions:
        if question.id in seen:
            log.warning("duplicate question id %s; keeping the first", question.id)
            continue
        seen.add(question.id)
        unique.append(question)
    return unique


def _confidence(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return round(max(0.0, min(1.0, number)), 4)
