from __future__ import annotations

from pathlib import Path

from iwir.config import Config

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "eval" / "fixtures" / "bundle.json"


def make_config(tmp_path: Path, **overrides) -> Config:
    config = Config(
        input_path=None,
        input_json="",
        output_dir=tmp_path,
        query_id="",
        query="",
        start_url="",
        question_timeout=8.0,
        search_timeout=5.0,
        fetch_timeout=6.0,
        llm_timeout=8.0,
        max_pages=3,
        max_hops=0,
        search_limit=5,
        search_providers=["fixture"],
        fixture_path=FIXTURE,
        schema_profile="default",
        output_fields=None,
        filename_template=None,
        include_diagnostics=False,
        llm_base_url="",
        llm_model="",
        llm_api_key="",
        llm_json_mode=True,
        log_level="WARNING",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config
