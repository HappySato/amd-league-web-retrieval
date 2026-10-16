# AMD notebook: vLLM on ROCm, then a live eval

This is the procedure for a free GPU notebook at [https://notebooks.amd.com/hackathon](https://notebooks.amd.com/hackathon). The retriever stays a small CPU process. The model stays in a separate vLLM process that speaks the OpenAI API on `127.0.0.1`. Nothing in this repository calls a paid API.

One command does the health check, starts vLLM if it is not already up, and runs the 25-question live eval twice (rules only, then with the model):

```bash
python scripts/amd_vllm_notebook.py --evidence
```

The rest of this page is what that command does, and how to do the same steps by hand.

## 1. Open a ROCm notebook

1. Sign in at [https://notebooks.amd.com/hackathon](https://notebooks.amd.com/hackathon).
2. Start a GPU notebook. The image should already include ROCm and a ROCm build of vLLM. Other Match 4 runs used the Radeon Pro W7900D pool; any ROCm GPU with room for a 7B model in float16 is enough.
3. Open a terminal in that notebook.

Do not run `pip install vllm`. The default PyPI wheel is built for CUDA and will not use the AMD GPU. If `python -c "import vllm"` fails, use the vLLM that shipped with the image, or follow the ROCm section of the [vLLM GPU install guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). A current container alternative is `vllm/vllm-openai-rocm`. This script never installs vLLM for you.

## 2. Put the retriever on the notebook

Copy this repository onto the notebook (git clone, or unpack the archive) and install only the retriever dependencies:

```bash
cd /path/to/this/repo
pip install -r requirements.txt
```

That installs `httpx` and `beautifulsoup4`. It does not download model weights.

## 3. One command

```bash
python scripts/amd_vllm_notebook.py --evidence
```

Useful overrides, all optional:

```bash
python scripts/amd_vllm_notebook.py --evidence \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --max-model-len 8192 \
  --health-wait 600
```

What it does, in order:

1. Writes a runtime check (Python, `torch` version, HIP, whether a device is visible, whether `vllm` imports) to `output/amd-evidence/<UTC-stamp>/check.log`.
2. Probes `http://127.0.0.1:8000/v1/models`.
3. If nothing is listening and vLLM imports (or the `vllm` console script is on `PATH`), it starts a resident server and waits until `/v1/models` returns HTTP 200, or until `--health-wait` seconds pass (default 600). The process is left running. Its pid is printed and stored in `status.txt`. Logs go to `vllm.log`.
4. If vLLM is missing and no server is already up, it writes `status.txt` and exits 2. It does not start the live crawl.
5. Runs `python eval/evaluate.py --mode live` with `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` removed, and writes `eval-no-llm.log` plus `report-no-llm.json`.
6. Runs the same eval with the local server:

   ```bash
   LLM_BASE_URL=http://127.0.0.1:8000/v1
   LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
   LLM_API_KEY=EMPTY
   LLM_TIMEOUT_SECONDS=8
   ```

   Logs and the report are `eval-llm.log` and `report-llm.json`.
7. Writes `summary.txt` with both scores, the pid, and the paths.

The serve command it uses is:

```bash
VLLM_TARGET_DEVICE=rocm vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --dtype float16 --max-model-len 8192
```

`VLLM_TARGET_DEVICE` is set only when it is not already set. `--dtype float16` is required on ROCm for this model. Qwen2.5-7B-Instruct weights are about 15 GB in float16. The first start also downloads the weights, which is why the wait is several minutes rather than the 24 second question budget.

Exit codes: `0` both evals finished, `1` the server was up but an eval failed or timed out, `2` there was no model server (missing vLLM, the process exited, or `/v1/models` never answered).

## 4. The same steps by hand

Check the GPU stack:

```bash
python scripts/amd_vllm_notebook.py --check
python scripts/amd_vllm_notebook.py --print-serve-cmd
```

You want `torch.version.hip` set and `torch.cuda.is_available()` True. Under ROCm that flag means the HIP device is visible.

Start the server in its own terminal and leave it there:

```bash
python scripts/amd_vllm_notebook.py --serve
```

In a second terminal, wait until this prints a model id (a cold start can take several minutes):

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

A single question, still against the live web, with the model enabled:

```bash
python scripts/amd_vllm_notebook.py --smoke
```

Or the env vars yourself:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export LLM_API_KEY=EMPTY
export LLM_TIMEOUT_SECONDS=8
python3 app.py --query-id demo --query "What is the capital of France?"
```

`LLM_API_KEY=EMPTY` is a placeholder. vLLM does not check it. The client sends `Authorization: Bearer EMPTY`. There is no default host: if either `LLM_BASE_URL` or `LLM_MODEL` is empty, the process never opens a socket to a model.

Rule-only live eval (unset the three variables so a previous export is not picked up):

```bash
env -u LLM_BASE_URL -u LLM_MODEL -u LLM_API_KEY \
  python eval/evaluate.py --mode live
```

Same questions with the model:

```bash
LLM_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
LLM_API_KEY=EMPTY \
LLM_TIMEOUT_SECONDS=8 \
  python eval/evaluate.py --mode live
```

Fixture mode never calls the model, even if those variables are exported. Use it to confirm the tree is intact before waiting on weights:

```bash
python eval/evaluate.py --mode fixture --fail-under 1.0
```

## 5. What the model is asked to do

Inside each question's 24 second budget:

- One `GET /v1/models` health check, at most about 2 seconds. Connection refusal, timeout, or HTTP 401/403/404/5xx marks the client down for the rest of the process. Later questions stay on the rule extractor.
- Query rewrite (`POST /v1/chat/completions`) only when at least 8 seconds remain. The call itself is capped at 4 seconds and at `LLM_TIMEOUT_SECONDS`. The model must return `{"queries":["...", ...]}` (one to three short strings). It is told not to answer the question.
- After pages are fetched, one extract call, capped at `LLM_TIMEOUT_SECONDS` (default 8) and at the time still left. The model must return `{"answer","quote","source_url","confidence"}`.
- The answer is kept only when that string appears in the cited page (or a tighter value inside the quote does). A made-up string is dropped.
- `response_format=json_object` is sent by default. HTTP 400, 404, or 422 retries once without it.
- A slow call is abandoned at its own timeout. The rule extractor still runs, and the process still writes a JSON file. A timeout does not mark the server down; a refused connection does.

A confident infobox hit (about 0.9) beats a model span that merely occurs on the page. When both sides name the same value, the result records that agreement.

## 6. If 7B does not fit

Stop the server (the pid is in `status.txt`) and rerun with a smaller model or a shorter context. A 3B model is the practical step down. A 0.5B model is only a wiring check; do not treat its eval as the result you show AMD.

```bash
python scripts/amd_vllm_notebook.py --evidence \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-model-len 4096
```

```bash
python scripts/amd_vllm_notebook.py --evidence \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 4096
```

Use the same `--model` string that vLLM prints in `/v1/models`. If port 8000 is already taken by a different model, pass `--port 8001` or stop the old process first.

## 7. Evidence to keep

The stamp directory is the bundle:

```text
output/amd-evidence/20261016T160000Z/
  check.log            Python, torch, HIP, device name, vLLM import
  serve-command.txt    exact argv
  vllm.log             server log, including the ROCm startup lines
  status.txt           pid, or why the server was not started
  summary.txt          both scores and the paths
  eval-no-llm.log      rule-only live eval transcript
  eval-llm.log         live eval with LLM_BASE_URL set
  report-no-llm.json   accuracy, abstain rate, p50, p95
  report-llm.json      same, with the model enabled
  no-llm/answers/      one JSON file per question
  llm/answers/         one JSON file per question
```

Pack it from the repo root:

```bash
tar czf amd-evidence.tar.gz output/amd-evidence/20261016T160000Z
```

Replace the stamp with the directory `--evidence` printed. That archive is the artifact to attach: device name in `check.log`, a successful `/v1/models` line, both reports, and the per-question JSON. The retriever does not need a GPU of its own; the evidence that the answers used AMD hardware is the vLLM log on that notebook plus `report-llm.json` produced while `LLM_BASE_URL` pointed at it.

Live eval talks to Wikipedia, Wikidata, DuckDuckGo, Bing HTML, and Brave HTML. No API key is sent. A notebook IP that receives HTTP 429 abstains on those questions instead of guessing. That shows up as a high abstain rate in the report, not as a crash. The process always writes one JSON file per question.
