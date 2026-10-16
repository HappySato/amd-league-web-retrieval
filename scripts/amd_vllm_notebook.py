#!/usr/bin/env python3
"""Start a resident vLLM server on an AMD ROCm notebook, then point IWIR at it.

The retrieval app does not load weights. Model load is far longer than the
24 second question budget, so vLLM stays up for the whole session and ``app.py``
calls it over the OpenAI-compatible HTTP API.

Designed for the free AMD notebook pool: https://notebooks.amd.com/hackathon
No paid API is contacted. If vLLM is missing, this script prints the command
and exits. It does not pip-install a CUDA wheel.

Notebook cells::

    !python scripts/amd_vllm_notebook.py --check
    !python scripts/amd_vllm_notebook.py --print-serve-cmd
    # leave this running in its own cell / terminal
    !python scripts/amd_vllm_notebook.py --serve

    !LLM_BASE_URL=http://127.0.0.1:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \\
        LLM_API_KEY=EMPTY python3 app.py \\
        --query-id demo --query "What is the capital of France?"

A smaller public model (``Qwen/Qwen2.5-0.5B-Instruct``) is enough to check the
wiring. Qwen2.5-7B-Instruct is the size other Match 4 entries measured on the
Radeon Pro W7900D pool. Use ``--dtype float16``.

One command on the notebook (health check, start vLLM if needed, live eval
with and without the model, write ``output/amd-evidence/<stamp>/``)::

    python scripts/amd_vllm_notebook.py --evidence

See ``docs/AMD_NOTEBOOK.md``. This script does not pip-install vLLM.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DOCS = "https://docs.vllm.ai/en/stable/getting_started/installation/gpu/"
NOTEBOOKS = "https://notebooks.amd.com/hackathon"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resident vLLM launcher for AMD ROCm notebooks")
    parser.add_argument("--check", action="store_true", help="Print torch/ROCm/vLLM status and exit")
    parser.add_argument("--print-serve-cmd", action="store_true", help="Print the serve command and exit")
    parser.add_argument("--serve", action="store_true", help="Replace this process with the vLLM server")
    parser.add_argument("--smoke", action="store_true", help="Ask the running server one question via app.py")
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="Start vLLM if needed, then run live eval with and without the model",
    )
    parser.add_argument("--evidence-dir", default="", help="Directory for --evidence logs (default: output/amd-evidence/<stamp>)")
    parser.add_argument(
        "--health-wait",
        type=float,
        default=float(os.environ.get("VLLM_HEALTH_WAIT", "600")),
        help="Seconds to wait for /v1/models after starting vLLM",
    )
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192")))
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "0.0.0.0"))
    args = parser.parse_args(argv)

    if args.evidence:
        return run_evidence(args)

    if not any((args.check, args.print_serve_cmd, args.serve, args.smoke)):
        args.check = True
        args.print_serve_cmd = True

    if args.check or args.print_serve_cmd or args.serve:
        _print_runtime()
    command = serve_command(args.model, args.host, args.port, args.max_model_len)
    if args.print_serve_cmd or args.serve or args.check:
        print("Serve command:")
        print("  " + " ".join(command))
        print(f"Then set LLM_BASE_URL=http://127.0.0.1:{args.port}/v1")
        print(f"         LLM_MODEL={args.model}")
        print("         LLM_API_KEY=EMPTY")
    if args.serve:
        if not _vllm_importable() and shutil.which("vllm") is None:
            print(
                "\nvLLM is not installed in this environment.\n"
                "On an AMD notebook, use the image's ROCm build of vLLM. "
                "Do not install the default CUDA wheel.\n"
                f"Install notes: {DOCS}\n"
                f"Notebook pool: {NOTEBOOKS}",
                file=sys.stderr,
            )
            return 1
        env = os.environ.copy()
        env.setdefault("VLLM_TARGET_DEVICE", "rocm")
        print("Starting vLLM. This process stays in the foreground so the model remains resident.")
        os.execvpe(command[0], command, env)
    if args.smoke:
        return _smoke(args.model, args.port)
    return 0


def serve_command(model: str, host: str, port: int, max_model_len: int) -> list[str]:
    """Prefer `vllm serve` when the console script exists; otherwise the module."""
    if shutil.which("vllm"):
        return [
            "vllm",
            "serve",
            model,
            "--host",
            host,
            "--port",
            str(port),
            "--dtype",
            "float16",
            "--max-model-len",
            str(max_model_len),
        ]
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--max-model-len",
        str(max_model_len),
    ]


def _print_runtime() -> None:
    print(f"python {sys.version.split()[0]}")
    try:
        import torch
    except Exception as exc:
        print(f"torch: not importable ({exc})")
    else:
        hip = getattr(torch.version, "hip", None)
        print(f"torch {torch.__version__} hip {hip}")
        try:
            available = torch.cuda.is_available()
        except Exception as exc:
            print(f"cuda/hip available check failed: {exc}")
        else:
            print(f"torch.cuda.is_available {available} (True under ROCm means the HIP device is visible)")
            if available:
                try:
                    print(f"device {torch.cuda.get_device_name(0)}")
                except Exception as exc:
                    print(f"device name unavailable: {exc}")
    print(f"vllm importable: {_vllm_importable()}")
    print("This machine does not need a GPU for the retriever itself.")
    print("Set LLM_BASE_URL only when a local server is already running.")


def _vllm_importable() -> bool:
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


def run_evidence(args: argparse.Namespace) -> int:
    """Health-check, start vLLM when it is missing, run live eval twice.

    The server process is left running. Live eval is not started when vLLM
    cannot be launched and nothing is already listening.
    """
    root = Path(__file__).resolve().parents[1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = Path(args.evidence_dir) if args.evidence_dir else root / "output" / "amd-evidence" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    command = serve_command(args.model, args.host, args.port, args.max_model_len)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_runtime()
        print("Serve command:")
        print("  " + " ".join(command))
        print(f"LLM_BASE_URL=http://127.0.0.1:{args.port}/v1")
        print(f"LLM_MODEL={args.model}")
        print("LLM_API_KEY=EMPTY")
    (dest / "check.log").write_text(buffer.getvalue(), encoding="utf-8")
    (dest / "serve-command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(f"evidence directory: {dest}")

    base = f"http://127.0.0.1:{args.port}/v1"
    proc: subprocess.Popen[str] | None = None
    if _models_ok(args.port):
        (dest / "status.txt").write_text(
            f"existing server at {base} answered /v1/models; vLLM was not started by this command\n",
            encoding="utf-8",
        )
        (dest / "vllm.log").write_text("not started; a server was already healthy\n", encoding="utf-8")
        print(f"using the server already listening on {base}")
    elif not _vllm_importable() and shutil.which("vllm") is None:
        message = (
            "vLLM is not installed and no server is listening on "
            f"{base}. Live evaluation was not started.\n"
            "Use the notebook image's ROCm build of vLLM. Do not pip-install the CUDA wheel.\n"
            f"Install notes: {DOCS}\n"
        )
        (dest / "status.txt").write_text(message, encoding="utf-8")
        (dest / "vllm.log").write_text("not started\n", encoding="utf-8")
        print(message, file=sys.stderr)
        return 2
    else:
        env = os.environ.copy()
        env.setdefault("VLLM_TARGET_DEVICE", "rocm")
        log_path = dest / "vllm.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"starting vLLM; logging to {log_path}")
        proc = subprocess.Popen(
            command,
            env=env,
            cwd=root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_file.close()
        print(f"vLLM pid {proc.pid} (left running after this command)")
        if not _wait_until_healthy(proc, args.port, args.health_wait):
            code = proc.poll()
            (dest / "status.txt").write_text(
                f"vllm_pid={proc.pid}\n"
                f"/v1/models did not return HTTP 200 within {args.health_wait:.0f}s "
                f"(process exit {code}). Live evaluation was not started.\n",
                encoding="utf-8",
            )
            print(f"vLLM did not become healthy. See {log_path}", file=sys.stderr)
            return 2
        (dest / "status.txt").write_text(
            f"vllm_pid={proc.pid}\nserver={base}\nmodel={args.model}\n",
            encoding="utf-8",
        )

    no_llm = _run_eval(root, dest, "no-llm", args.model, args.port, use_llm=False)
    with_llm = _run_eval(root, dest, "llm", args.model, args.port, use_llm=True)
    _write_summary(dest, args, base, proc, no_llm, with_llm)
    if proc is not None and proc.poll() is None:
        print(f"vLLM is still running as pid {proc.pid}")
    if no_llm != 0 or with_llm != 0:
        return 1
    return 0


def _wait_until_healthy(proc: subprocess.Popen[str], port: int, limit: float) -> bool:
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        if _models_ok(port):
            return True
        time.sleep(5)
    return _models_ok(port)


def _models_ok(port: int, timeout: float = 3.0) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= getattr(response, "status", 0) < 300
    except Exception:
        return False


def _run_eval(root: Path, dest: Path, name: str, model: str, port: int, *, use_llm: bool) -> int:
    out = dest / name
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        env.pop(key, None)
    label = "with LLM" if use_llm else "without LLM"
    if use_llm:
        env["LLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env["LLM_MODEL"] = model
        env["LLM_API_KEY"] = "EMPTY"
        env["LLM_TIMEOUT_SECONDS"] = env.get("LLM_TIMEOUT_SECONDS") or "8"
    log_path = dest / f"eval-{name}.log"
    print(f"live eval {label}; log {log_path}")
    command = [
        sys.executable,
        str(root / "eval" / "evaluate.py"),
        "--mode",
        "live",
        "--output",
        str(out),
        "--timeout",
        "24",
    ]
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=False,
            )
        except subprocess.TimeoutExpired:
            handle.write("\neval exceeded 900s\n")
            return 1
    report = out / "report.json"
    if report.is_file():
        shutil.copyfile(report, dest / f"report-{name}.json")
    return completed.returncode


def _write_summary(
    dest: Path,
    args: argparse.Namespace,
    base: str,
    proc: subprocess.Popen[str] | None,
    no_llm: int,
    with_llm: int,
) -> None:
    pid = proc.pid if proc is not None else "not started by this command"
    lines = [
        f"model={args.model}",
        f"base={base}",
        f"vllm_pid={pid}",
        f"eval_no_llm_exit={no_llm}",
        f"eval_llm_exit={with_llm}",
        f"directory={dest}",
        "Pack this directory: tar czf amd-evidence.tar.gz " + str(dest),
        "",
    ]
    for name in ("report-no-llm.json", "report-llm.json"):
        path = dest / name
        if not path.is_file():
            lines.append(f"{name}: missing")
            continue
        try:
            import json

            summary = json.loads(path.read_text(encoding="utf-8")).get("summary") or {}
        except Exception as exc:
            lines.append(f"{name}: unreadable ({exc})")
            continue
        lines.append(
            f"{name}: primary {summary.get('primary_pass')}/{summary.get('total')} "
            f"abstain {summary.get('abstain_rate')} "
            f"p50 {summary.get('p50_seconds')}s p95 {summary.get('p95_seconds')}s"
        )
    text = "\n".join(lines) + "\n"
    (dest / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")


def _smoke(model: str, port: int) -> int:
    base = f"http://127.0.0.1:{port}/v1"
    try:
        with urllib.request.urlopen(base + "/models", timeout=3) as response:
            print(f"server responded {response.status} from {base}/models")
    except Exception as exc:
        print(f"No server at {base} ({exc}). Start it with --serve first.", file=sys.stderr)
        return 1
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env["LLM_BASE_URL"] = base
    env["LLM_MODEL"] = model
    env["LLM_API_KEY"] = env.get("LLM_API_KEY") or "EMPTY"
    env["QUESTION_TIMEOUT_SECONDS"] = env.get("QUESTION_TIMEOUT_SECONDS", "24")
    os.chdir(root)
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            os.path.join(root, "app.py"),
            "--query-id",
            "smoke",
            "--query",
            "What is the capital of France?",
        ],
        env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
