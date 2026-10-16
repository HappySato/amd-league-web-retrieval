"""The notebook helper must not crawl the web when vLLM is absent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "amd_vllm_notebook.py"
    spec = importlib.util.spec_from_file_location("amd_vllm_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_skips_live_eval_when_vllm_is_missing(tmp_path, monkeypatch) -> None:
    module = _load()
    monkeypatch.setattr(module, "_vllm_importable", lambda: False)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module, "_models_ok", lambda port, timeout=3.0: False)

    def refuse(*args, **kwargs):
        raise AssertionError(f"should not spawn a process: {args} {kwargs}")

    monkeypatch.setattr(module.subprocess, "run", refuse)
    monkeypatch.setattr(module.subprocess, "Popen", refuse)
    code = module.main(["--evidence", "--port", "9", "--evidence-dir", str(tmp_path)])
    assert code == 2
    status = (tmp_path / "status.txt").read_text(encoding="utf-8")
    assert "not installed" in status
    assert "Live evaluation was not started" in status
    assert (tmp_path / "check.log").exists()
    assert "float16" in (tmp_path / "serve-command.txt").read_text(encoding="utf-8")
    assert not (tmp_path / "eval-no-llm.log").exists()


def test_serve_command_uses_rocm_friendly_flags() -> None:
    module = _load()
    command = module.serve_command("Qwen/Qwen2.5-7B-Instruct", "0.0.0.0", 8000, 8192)
    text = " ".join(command)
    assert "Qwen/Qwen2.5-7B-Instruct" in text
    assert "--dtype" in command
    assert command[command.index("--dtype") + 1] == "float16"
    assert "--max-model-len" in command
