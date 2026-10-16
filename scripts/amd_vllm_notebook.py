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
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DOCS = "https://docs.vllm.ai/en/stable/getting_started/installation/gpu/"
NOTEBOOKS = "https://notebooks.amd.com/hackathon"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resident vLLM launcher for AMD ROCm notebooks")
    parser.add_argument("--check", action="store_true", help="Print torch/ROCm/vLLM status and exit")
    parser.add_argument("--print-serve-cmd", action="store_true", help="Print the serve command and exit")
    parser.add_argument("--serve", action="store_true", help="Replace this process with the vLLM server")
    parser.add_argument("--smoke", action="store_true", help="Ask the running server one question via app.py")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192")))
    parser.add_argument("--host", default=os.environ.get("VLLM_HOST", "0.0.0.0"))
    args = parser.parse_args(argv)

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
