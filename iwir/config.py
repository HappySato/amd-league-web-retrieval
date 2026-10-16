"""Environment and CLI configuration.

Names the grader is likely to care about (paths, timeout, schema, LLM) are
environment variables so a revised spec can be matched without code changes
when the change is only names, paths, or field order. Structural changes belong
in ``iwir.adapter``.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    pass


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value


def _env_float(name: str, default: float, *, low: float, high: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    return max(low, min(high, value))


def _env_int(name: str, default: int, *, low: int, high: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    return max(low, min(high, value))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(value: str) -> list[str]:
    return [part.strip().lower() for part in value.split(",") if part.strip()]


@dataclass(slots=True)
class Config:
    input_path: Path | None
    input_json: str
    output_dir: Path
    query_id: str
    query: str
    start_url: str
    question_timeout: float
    search_timeout: float
    fetch_timeout: float
    llm_timeout: float
    max_pages: int
    max_hops: int
    search_limit: int
    search_providers: list[str]
    fixture_path: Path | None
    schema_profile: str
    output_fields: list[str] | None
    filename_template: str | None
    include_diagnostics: bool
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_json_mode: bool
    log_level: str

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url.strip() and self.llm_model.strip())

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> Config:
        parser = argparse.ArgumentParser(
            description="Answer natural-language questions from the web and write one JSON file each.",
        )
        parser.add_argument("--input", dest="input_path", help="Questions JSON, JSONL, or directory")
        parser.add_argument("--output", dest="output_dir", help="Directory for one JSON file per question")
        parser.add_argument("--query-id", help="Single-question id (grader-style invocation)")
        parser.add_argument("--query", help="Single-question text")
        parser.add_argument("--start-url", help="Optional page to read before searching")
        parser.add_argument("--timeout", type=float, help="Per-question seconds (default 24)")
        parser.add_argument(
            "--schema-profile",
            choices=("default", "league3"),
            help="Output shape. default: id, question, answer, sources, confidence",
        )
        parser.add_argument("--fixture", help="Offline search/page bundle (SEARCH_PROVIDERS=fixture)")
        args = parser.parse_args(argv)

        if args.query_id and not args.query:
            raise ConfigError("--query-id requires --query")

        input_path = args.input_path or _env("INPUT_PATH")
        output_dir = args.output_dir or _env("OUTPUT_DIR") or _default_output_dir()
        timeout = args.timeout if args.timeout is not None else _env_float(
            "QUESTION_TIMEOUT_SECONDS", 24.0, low=1.0, high=180.0
        )
        if args.timeout is not None:
            timeout = max(1.0, min(180.0, args.timeout))

        providers = _csv(_env("SEARCH_PROVIDERS", "duckduckgo,wikipedia"))
        if providers == ["none"] or providers == ["off"]:
            providers = []

        fixture = args.fixture or _env("FIXTURE_PATH")
        fields_raw = _env("OUTPUT_FIELDS")
        template = _env("OUTPUT_FILENAME_TEMPLATE")

        return cls(
            input_path=Path(input_path) if input_path else None,
            input_json=_env("INPUT_JSON"),
            output_dir=Path(output_dir),
            query_id=args.query_id or "",
            query=args.query or "",
            start_url=args.start_url or "",
            question_timeout=timeout,
            search_timeout=_env_float("SEARCH_TIMEOUT_SECONDS", 5.0, low=0.5, high=30.0),
            fetch_timeout=_env_float("FETCH_TIMEOUT_SECONDS", 6.0, low=0.5, high=30.0),
            llm_timeout=_env_float("LLM_TIMEOUT_SECONDS", 8.0, low=0.5, high=30.0),
            max_pages=_env_int("MAX_PAGES", 3, low=1, high=8),
            max_hops=_env_int("MAX_HOPS", 1, low=0, high=3),
            search_limit=_env_int("SEARCH_LIMIT", 5, low=1, high=10),
            search_providers=providers,
            fixture_path=Path(fixture) if fixture else None,
            schema_profile=(args.schema_profile or _env("SCHEMA_PROFILE", "default")).strip().lower(),
            output_fields=_csv_keep_case(fields_raw) if fields_raw else None,
            filename_template=template or None,
            include_diagnostics=_env_bool("OUTPUT_INCLUDE_DIAGNOSTICS", False),
            llm_base_url=_env("LLM_BASE_URL").strip(),
            llm_model=_env("LLM_MODEL").strip(),
            llm_api_key=_env("LLM_API_KEY"),
            llm_json_mode=_env_bool("LLM_JSON_MODE", True),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
        )


def _csv_keep_case(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def in_container_image() -> bool:
    """True when this file tree is the /app tree the grader starts."""
    return Path("/app/app.py").is_file()


def _default_output_dir() -> str:
    if in_container_image():
        return "/app/output"
    return "output"
