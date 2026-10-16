# IWIR — Intelligent Web Information Retrieval

Prototype for lablab AMD AI League match 4, *Intelligent Web Information Retrieval* (deadline 2026-10-27). This repository does not submit an entry.

Given a natural-language question, the program searches and opens web pages, then writes **the value itself** — a number, name, date, or comma-separated list — plus the URL of a page that contains that value. It does not return a summary or a list of links. If the pages it opened do not contain the value, the answer is empty.

No paid API is used unless you point `LLM_BASE_URL` at a server you already run. Search is keyless: a guessed reference page, Wikidata, the MediaWiki API, DuckDuckGo HTML, and Bing and Brave HTML. An optional local OpenAI-compatible model (vLLM on AMD ROCm) is used only when both the base URL and the model name are set.

## Overview

The grader-style entry point is `python3 /app/app.py`. One JSON file is written per question. The default budget is 24 seconds. A timeout, a blocked search host, or a page with no supported value still produces that file, with `answer` set to `""`.

The official task PDF was still being revised when this was written. Input and output live in `iwir/adapter.py`. Field names, the filename, and the provider list are environment variables, so a later schema can be matched without rewriting retrieval.

## Architecture

```
app.py
  iwir/runner.py        deadline thread, always writes JSON
  iwir/adapter.py       read questions, write one file
  iwir/pipeline.py      search → fetch → extract → optional model
  iwir/search/          direct, wikidata, wikipedia, duckduckgo, bing, brave, fixture
  iwir/http.py          user agent, per-host spacing, backoff, response cache
  iwir/browse.py        Wikipedia lead/infobox API, or a capped direct fetch
  iwir/extract.py       infobox and sentence patterns; drop unsupported values
  iwir/llm.py           OpenAI-compatible client, off unless configured
```

A question runs like this:

1. Seed URLs (`start_url`, extra URLs, or a URL written in the question) are fetched first.
2. Providers run in order and stop at the first one that returns hits. The rest stay queued. If the answer is still weak, the next provider is tried, up to four more rounds. A snippet-only answer gets one extra source, then the pipeline stops, so a rate limit does not spend the whole 24 seconds.
3. Wikipedia article URLs are read through the MediaWiki parse API (lead section and infobox). Other URLs are fetched directly, with an 800 KB cap. Private, loopback, and link-local addresses, including the cloud metadata host, are refused.
4. The heuristic keeps a value only when that value occurs in the page. It will not echo the question ("France" for "capital of France") or return a name for a "how many" question.
5. If `LLM_BASE_URL` and `LLM_MODEL` are both set and the heuristic is below 0.88 confidence, the best passages go to `/v1/chat/completions`. The model's value is kept only when it appears in the cited page.
6. The file is written by a temp file and rename.

The model is not loaded inside `app.py`. Weight loading does not fit in 24 seconds. Keep vLLM resident and send questions to it.

Default provider order is `direct,wikidata,wikipedia,duckduckgo,bing,brave`:

| Provider | What it calls | Key |
| --- | --- | --- |
| `direct` | No request. Proposes `https://en.wikipedia.org/wiki/{Topic}` for attribute questions and a few ambiguous topics (the programming language, not the snake). | none |
| `wikidata` | `wbsearchentities` and `wbgetentities`. Claims are rendered as short sentences so a rate-limited article fetch can still be scored. | none |
| `wikipedia` | MediaWiki search API, then the parse API for the lead and infobox. | none |
| `duckduckgo` | `html.duckduckgo.com`, then `lite.duckduckgo.com` if the first response is not a result page. Ad redirects are dropped. | none |
| `bing` | `https://www.bing.com/search` HTML. Not the Bing Web Search API. | none |
| `brave` | `https://search.brave.com/search` HTML. Not the Brave Search API. | none |
| `fixture` | Offline pages in a JSON bundle. Used by tests. | none |

Outbound calls share a process-wide response cache (10 minutes, HTTP 200 only). Each host has a minimum gap. HTTP 429, 502, 503, and 504, and connection errors, are retried with exponential backoff and jitter. `Retry-After` is followed when it is at most 2.5 seconds. A 429 is not cached. Wikimedia hosts get an identifying user agent; HTML search endpoints get a browser user agent.

## Contract and adapter

`iwir/adapter.py` is the only module that reads questions and writes JSON. Change that file, or the variables below, when the published schema settles.

**Input.** Any of:

- a JSON list, or an object with `questions`, `items`, `queries`, or `data`
- JSONL (one object per line)
- a directory of `*.json` / `*.jsonl` (`*.example.json` is ignored)
- `--query-id` and `--query` (this pair is the only question)
- `INPUT_JSON` when `INPUT_PATH` is unset

Accepted aliases: `id` / `query_id` / `qid`, `question` / `query` / `text` / `prompt`, `start_url` / `url` / `page`, `urls` / `seed_urls`.

Inside the image, the default input is `/app/input/questions.json` if that file exists, otherwise every JSON file in `/app/input`. Outside the image, the default is `input/questions.json` when that file exists. A missing default input writes nothing and exits 0. A missing *explicit* `--input` path is an error.

**Output profiles.**

| `SCHEMA_PROFILE` | File | Keys |
| --- | --- | --- |
| `default` | `{id}.json` | `id`, `question`, `answer`, `sources`, `confidence` |
| `league3` | `{id}_output.json` | `answer`, `citations`, `confidence` |

`league3` matches the Match 3 grader shape, with URLs in place of file paths. Finer renames, still without code changes:

```bash
OUTPUT_FIELDS=answer,citations,confidence
OUTPUT_FILENAME_TEMPLATE='{id}_output.json'
```

Known keys are `id`, `query_id`, `question`, `query`, `answer`, `sources`, `citations`, `source`, `confidence`. `OUTPUT_INCLUDE_DIAGNOSTICS=1` adds `diagnostics` (`error`, `method`). Leave it off for a strict grader.

`sources` are pages whose text contains the answer. `confidence` is between 0 and 1. Both are 0 and `answer` is `""` when nothing supported a value.

| Variable | Default | Meaning |
| --- | --- | --- |
| `INPUT_PATH` | image: `/app/input/questions.json`; else `input/questions.json` if present | Questions file or directory |
| `INPUT_JSON` | empty | Raw JSON, used when `INPUT_PATH` is unset |
| `OUTPUT_DIR` | `/app/output` in the image, else `output/` | One JSON file per question |
| `QUESTION_TIMEOUT_SECONDS` | `24` | Per-question budget. A file is always written. |
| `SEARCH_PROVIDERS` | `direct,wikidata,wikipedia,duckduckgo,bing,brave` | Comma-separated. `none` disables search. |
| `FIXTURE_PATH` | empty | Required when the provider list is `fixture` |
| `SEARCH_TIMEOUT_SECONDS` | `5` | Per provider call |
| `FETCH_TIMEOUT_SECONDS` | `6` | Per page |
| `MAX_PAGES` | `3` | Pages opened from the first result list |
| `MAX_HOPS` | `1` | One extra in-page link when confidence is still under 0.75 |
| `SEARCH_LIMIT` | `5` | Hits kept from a provider |
| `SCHEMA_PROFILE` | `default` | `default` or `league3` |
| `OUTPUT_FIELDS` | profile | Comma-separated output keys, in order |
| `OUTPUT_FILENAME_TEMPLATE` | profile | Format string; `{id}` only |
| `LLM_BASE_URL` | empty | OpenAI-compatible base, for example `http://127.0.0.1:8000/v1` |
| `LLM_MODEL` | empty | Served model name. Both URL and model are required. |
| `LLM_API_KEY` | empty | Sent as `Bearer`. Use `EMPTY` for a local vLLM that expects a token. |
| `LLM_TIMEOUT_SECONDS` | `8` | Generation budget, inside the question budget |
| `LLM_JSON_MODE` | `1` | Sends `response_format=json_object`, retries once without it on HTTP 400/404/422 |
| `LOG_LEVEL` | `INFO` | Logs go to stderr. stdout stays quiet. |

## Running locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 app.py \
  --query-id capital-france \
  --query "What is the capital of France?" \
  --output output/demo
```

That writes `output/demo/capital-france.json`:

```json
{
  "id": "capital-france",
  "question": "What is the capital of France?",
  "answer": "Paris",
  "sources": ["https://en.wikipedia.org/wiki/France"],
  "confidence": 0.92
}
```

A batch file works the same way:

```bash
python3 app.py --input input/questions.example.json --output output/batch
```

Tests do not use the network:

```bash
pip install -r requirements-dev.txt
pytest
python eval/evaluate.py --mode fixture --fail-under 1.0
```

## Docker

The image is `python:3.12-slim`. It is the retriever, not the GPU server. `CMD` is `["python3", "/app/app.py"]`, so a harness can replace the command.

```bash
docker build -t iwir .
docker run --rm \
  -e QUESTION_TIMEOUT_SECONDS=24 \
  --network host \
  -v "$PWD/input/questions.example.json:/app/input/questions.json:ro" \
  -v "$PWD/output:/app/output" \
  iwir
```

`--network host` is the setup used to verify this tree: the image was built and `python3 /app/app.py` wrote one JSON file per example question under `/app/output`. On a machine with a normal bridge network, drop `--network host`.

To call vLLM on the host from the container:

```bash
docker run --rm --network host \
  -e LLM_BASE_URL=http://127.0.0.1:8000/v1 \
  -e LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e LLM_API_KEY=EMPTY \
  -v "$PWD/input/questions.example.json:/app/input/questions.json:ro" \
  -v "$PWD/output:/app/output" \
  iwir
```

The harness can also run `python3 /app/app.py --query-id demo --query "What is the capital of France?"`.

## AMD vLLM

Use a free notebook from [notebooks.amd.com/hackathon](https://notebooks.amd.com/hackathon). The app container stays on CPU; the model stays in its own process.

```bash
python scripts/amd_vllm_notebook.py --check
python scripts/amd_vllm_notebook.py --print-serve-cmd
python scripts/amd_vllm_notebook.py --serve
```

`--serve` runs vLLM in the foreground with `VLLM_TARGET_DEVICE=rocm` when that variable is unset. The default model is `Qwen/Qwen2.5-7B-Instruct` at `--dtype float16` and context 8192. For a wiring check, pass `--model Qwen/Qwen2.5-0.5B-Instruct`.

In a second terminal:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export LLM_API_KEY=EMPTY
python3 app.py --query-id demo --query "What is the capital of France?"
```

`python scripts/amd_vllm_notebook.py --smoke` checks `/v1/models` once the server answers.

Install vLLM from the ROCm build that matches the notebook image. The script does not `pip install` a CUDA wheel. Upstream notes: [vLLM GPU install](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). If you would rather serve the model in Docker, a current ROCm image is `vllm/vllm-openai-rocm`. The app container only needs HTTP access to it.

The heuristic answers the bundled sample without a GPU. Turn the model on when the pattern matcher abstains or returns a low confidence. A fluent guess that is not in the cited page is discarded.

## Evaluation

```bash
python eval/evaluate.py --mode fixture --fail-under 1.0
python eval/evaluate.py --mode live
python eval/evaluate.py --mode live --limit 3 --timeout 24
```

| Mode | Questions | Network |
| --- | --- | --- |
| `fixture` | `eval/questions.json` (11), replayed from `eval/fixtures/bundle.json` | no |
| `live` | `eval/questions_live.json` (25), unless `--questions` is set | yes, keyless providers only |

`match` on each question picks the primary score. **Exact** compares the normalized value (case and punctuation ignored; `8` and `eight` match). **Contains** also accepts the gold value inside a longer answer, and the other way around. An empty gold answer is correct only when the prediction is empty.

The report is `output/eval/<mode>/report.json`. It records primary accuracy, exact and contains counts, the abstain rate (empty answers), and latency: mean, p50, p95, max, and how many questions finished inside the budget (default 24 seconds). Per-question JSON is under `output/eval/<mode>/answers/`.

Fixture mode is the regression check. Live mode depends on Wikimedia and the HTML search sites answering from the machine you run it on. A host that returns 429 is skipped for a short cooldown; the next provider still runs. From this datacenter address, Wikipedia and Wikidata started returning 429 after a handful of article fetches. On the 25-question live set the run then finished at 7/25 primary (Paris, Au, George Orwell, 299792458, 100, and two correct refusals), abstained on 20/25, and stayed inside the 24-second budget on every question (p50 1.987s, p95 5.576s, max 6.251s). Empty was preferred to a guess: that run had no wrong non-empty answer. A network that is not already throttled should get further, because the same questions resolve from the article lead when the parse API returns 200.

## Adding a provider

Implement `search(self, request) -> list[SearchHit]` and register the name in `build_services` in `iwir/search/chain.py`. Raise on failure; the chain logs it and continues. Set `SEARCH_PROVIDERS` to include the new name. Do not add a provider that needs a paid key.

## Limitations

- Datacenter addresses often get an interstitial from DuckDuckGo (HTTP 202) or HTTP 429 from Wikipedia. The cache, the cooldown, and the other providers exist for that case. They do not make a banned address look like a residential one.
- Pages that need JavaScript are not executed. A JSON body at the URL is flattened to `field: value` lines.
- The heuristic is aimed at encyclopedia-style facts: infoboxes, "X is Y", counts, years, symbols, and short lists. A supplier portal with no such sentence should abstain unless a local model is configured and the value is on the page.
- Wikidata quantities are rendered as written, except boiling point, which is converted when the unit is kelvin. Other unit conversions are not attempted.
- `direct` only proposes a Wikipedia title. It does not contain the answer. A wrong title costs one fetch, then the next provider runs.
- Lists are one comma-separated string, so `answer` stays a string.
- A second question with the same id overwrites the first file.
- Snippet-only evidence is capped at confidence 0.7, because the result blurb was not the page.
- Literal private IPs, localhost, `.local`, `.internal`, and `169.254.169.254` are not fetched.

## License

MIT. See [LICENSE](LICENSE).
