# IWIR — Intelligent Web Information Retrieval

Prototype for lablab **AMD AI League** match 4, *Intelligent Web Information Retrieval* (deadline 2026-10-27).

Ask a question in plain language. The program searches and opens web pages, then writes **the value you asked for** (a number, name, date, or comma-separated list) plus the URL that contains it. It does not return a summary or a list of links. If the pages it opened do not contain the value, the answer is empty.

The official task PDF was still changing when this was written. Input and output live in one module, `iwir/adapter.py`, and most names are environment variables, so the file format can be swapped without touching retrieval.

No paid API is called unless you set `LLM_BASE_URL` yourself. The default path is keyless search (DuckDuckGo HTML, Wikipedia, direct fetch) plus a deterministic extractor. A local OpenAI-compatible server (vLLM on AMD ROCm) is optional.

This repository does not submit anything to the league.

## English

### Layout

| Path | Role |
| --- | --- |
| `app.py` | Process entry. The container command is `python3 /app/app.py`. |
| `iwir/adapter.py` | Only place that reads questions and writes JSON. |
| `iwir/pipeline.py` | Search, fetch, extract, optional model, deadline. |
| `iwir/search/` | Pluggable providers (`duckduckgo`, `wikipedia`, `fixture`). |
| `iwir/extract.py` | Direct-answer heuristics. Refuses values that are not in the page. |
| `iwir/llm.py` | OpenAI-compatible client. Idle unless base URL and model are both set. |
| `eval/evaluate.py` | Exact / contains scores and per-question latency. |
| `scripts/amd_vllm_notebook.py` | Resident vLLM launcher for AMD notebooks. |

### Run locally

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
  "confidence": 0.9
}
```

`sources` are pages whose text actually contains the answer. `confidence` is 0 and `answer` is `""` when nothing on those pages supports a value. A timeout or a crashed fetch still writes that JSON.

A batch file works the same way. Copy `input/questions.example.json` to `input/questions.json`, or pass `--input`:

```bash
python3 app.py --input input/questions.example.json --output output/batch
```

Accepted inputs, all handled in the adapter:

- a JSON list, or `{"questions":[...]}`
- JSONL (one object per line)
- a directory of `*.json` / `*.jsonl` (`*.example.json` is ignored)
- one question on the CLI: `--query-id` and `--query`, optional `--start-url`

Field aliases already accepted on the way in: `id` / `query_id` / `qid`, `question` / `query` / `text` / `prompt`, `start_url` / `url` / `page`.

### Output profiles

`SCHEMA_PROFILE=default` (the default) writes `{id}.json` with `id`, `question`, `answer`, `sources`, `confidence`.

`SCHEMA_PROFILE=league3` writes `{id}_output.json` with `answer`, `citations`, `confidence`, which matches the Match 3 grader shape if Match 4 keeps it and only changes citations from file paths to URLs.

Anything finer is an environment override, still without code changes:

```bash
OUTPUT_FIELDS=answer,citations,confidence
OUTPUT_FILENAME_TEMPLATE='{id}_output.json'
```

`OUTPUT_INCLUDE_DIAGNOSTICS=1` adds an extra `diagnostics` object (`error`, `method`). Leave it off for a strict grader.

If the PDF changes the *structure* (not just key names), edit `iwir/adapter.py` only.

### Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `INPUT_PATH` | `/app/input/questions.json` inside the image, else `input/questions.json` if that file exists | File or directory of questions |
| `INPUT_JSON` | empty | Raw JSON, used when `INPUT_PATH` is unset |
| `OUTPUT_DIR` | `/app/output` in the image, else `output/` | One JSON file per question |
| `QUESTION_TIMEOUT_SECONDS` | `24` | Per-question budget. The process always writes a file. |
| `SEARCH_PROVIDERS` | `duckduckgo,wikipedia` | Comma-separated. `fixture` is the offline bundle. `none` disables search. |
| `FIXTURE_PATH` | empty | Required with `fixture` |
| `SEARCH_TIMEOUT_SECONDS` | `5` | Per provider |
| `FETCH_TIMEOUT_SECONDS` | `6` | Per page |
| `MAX_PAGES` | `3` | Pages opened from search hits |
| `MAX_HOPS` | `1` | Extra in-page link when the answer is still weak |
| `SCHEMA_PROFILE` | `default` | `default` or `league3` |
| `OUTPUT_FIELDS` | profile | Comma-separated output keys |
| `OUTPUT_FILENAME_TEMPLATE` | profile | Format string, `{id}` only |
| `LLM_BASE_URL` | empty | OpenAI-compatible base, for example `http://127.0.0.1:8000/v1` |
| `LLM_MODEL` | empty | Served model name. Both URL and model are required. |
| `LLM_API_KEY` | empty | Sent as `Bearer`. Use `EMPTY` for a local vLLM that expects a token. |
| `LLM_TIMEOUT_SECONDS` | `8` | Generation budget inside the question budget |
| `LLM_JSON_MODE` | `1` | Sends `response_format=json_object`, retries once without it on HTTP 400 |
| `LOG_LEVEL` | `INFO` | Logs go to stderr. stdout stays quiet. |

### What a question does

1. Optional seed URL (`start_url`, `urls`, or a URL written in the question).
2. DuckDuckGo is queried first (the HTML endpoint, then the lite endpoint if that interstitial-blocks). Wikipedia is the next provider, and for questions like "capital of France" it searches the entity article. Private addresses, metadata hosts, and ad redirects are dropped. Wikipedia pages are read before other hits.
3. The top pages are fetched. Wikipedia articles go through the MediaWiki parse API (lead section plus infobox) instead of the full desktop HTML. JSON bodies are flattened to `field: value` lines. Tables keep their rows; nav and footer chrome is removed.
4. A heuristic reads infobox labels and a few sentence patterns (capital, symbol, currency, author, year, boiling point, counts, lists). The value has to occur in that page. The question's own words are not echoed back as the answer.
5. If a local model is configured **and** the heuristic is unsure, the best passages go to `/v1/chat/completions`. The reply is JSON. The value is kept only when it appears in the cited page, then tightened to the value itself ("Paris", not "Paris is the capital of France").
6. One file is written, via a temp file and rename. On timeout, search failure, or no evidence, the file is still written with an empty answer.

The model is not loaded inside `app.py`. Loading weights does not fit in 24 seconds. Keep vLLM resident and send questions at it.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use a saved page bundle. They do not call the network or a model. In this workspace that was 42 tests, and `python eval/evaluate.py --mode fixture --fail-under 1.0` scored 11/11.

Live search is separate. DuckDuckGo and Wikipedia will answer from a normal network; this sandbox started returning HTTP 202 and HTTP 429 after a few batches. Before that limit, the capital of France came back as Paris, the symbol for gold as Au, and the invented part number was refused (empty answer, empty sources). Run `--mode live` again where those hosts are not already throttling you.

### Evaluation

```bash
python eval/evaluate.py --mode fixture --fail-under 1.0
python eval/evaluate.py --mode live
python eval/evaluate.py --mode live --limit 3 --timeout 24
```

`eval/questions.json` is eleven stable questions, including one that should be refused. Scoring is case-insensitive. **Exact** compares the normalized value to any alias. **Contains** also allows the gold value inside a longer answer, and treats `8` and `eight` as the same. The `match` field on each question picks which of those is the primary score. Latency is the in-process time for that question, including the timeout wrapper.

Fixture mode is deterministic. Live mode depends on DuckDuckGo and Wikipedia being reachable from the machine you run it on. A blocked DuckDuckGo response falls through to Wikipedia.

Reports land in `output/eval/<mode>/report.json`. Per-question files land in `output/eval/<mode>/answers/`.

### Docker

```bash
docker build -t iwir .
docker run --rm \
  -e QUESTION_TIMEOUT_SECONDS=24 \
  -v "$PWD/input/questions.example.json:/app/input/questions.json:ro" \
  -v "$PWD/output:/app/output" \
  iwir
```

The image is `python:3.12-slim`. It is the retriever, not the GPU server. `CMD` is `python3 /app/app.py`, so a harness can also run `python3 /app/app.py --query-id ... --query ...`.

To talk to vLLM on the host from inside the container:

```bash
docker run --rm --network host \
  -e LLM_BASE_URL=http://127.0.0.1:8000/v1 \
  -e LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
  -e LLM_API_KEY=EMPTY \
  -v "$PWD/output:/app/output" \
  iwir --query-id demo --query "What is the capital of France?"
```

(`docker run IMAGE --query-id ...` replaces `CMD` arguments only when `CMD` is exec form and the binary stays `python3 /app/app.py`. If your harness sets the command itself, pass the flags after `python3 /app/app.py`.)

### AMD ROCm and vLLM

Use a free notebook from [notebooks.amd.com/hackathon](https://notebooks.amd.com/hackathon). Check the image, print the serve command, then leave the server running:

```bash
python scripts/amd_vllm_notebook.py --check
python scripts/amd_vllm_notebook.py --print-serve-cmd
python scripts/amd_vllm_notebook.py --serve
```

`--serve` runs vLLM in the foreground with `VLLM_TARGET_DEVICE=rocm` when that variable is unset. The default model is `Qwen/Qwen2.5-7B-Instruct` at `--dtype float16` and context 8192, which fits the cards in that pool (other Match 4 notes measured this class of model around 15 GiB). For a wiring check, pass `--model Qwen/Qwen2.5-0.5B-Instruct`.

In a second terminal or notebook cell:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export LLM_API_KEY=EMPTY
python3 app.py --query-id demo --query "What is the capital of France?"
```

Or `python scripts/amd_vllm_notebook.py --smoke` once `/v1/models` answers.

Install vLLM from the ROCm build that matches the notebook image. The script will not `pip install` the default CUDA wheel. Upstream notes: [vLLM GPU install](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). A current ROCm image is `vllm/vllm-openai-rocm` if you are serving the model in Docker instead of inside the notebook kernel; the app container above still only needs HTTP access to it.

The heuristic already answers the sample set without a GPU. Turn the model on when the page is messy and the pattern matcher abstains or returns a low confidence. Verified answers are still checked against the page text, so a fluent guess that is not in the source is discarded.

### Adding a search provider

Subclass nothing formal. Implement `search(self, request) -> list[SearchHit]` and register the name in `iwir/search/chain.py` inside `build_services`. Raise on failure; the chain logs it and keeps the other providers. Point `SEARCH_PROVIDERS` at the new name.

### Limits

- DuckDuckGo's HTML endpoint sometimes returns a bot check from datacenter addresses. Wikipedia still runs.
- Pages that need JavaScript are not executed. A JSON API behind the URL is read.
- The heuristic covers encyclopedia-style facts (infobox, "X is Y", counts, years, short lists). Odd supplier pages need the value in the text, or the optional local model.
- Lists are a single comma-separated string, so the JSON type of `answer` stays a string.
- One file per question id. A second question with the same id overwrites the first.
- Literal private IPs, localhost, and the cloud metadata address are not fetched.

## 日本語

lablab **AMD AI League** 第4試合「Intelligent Web Information Retrieval」（締切 2026-10-27）向けの試作です。

自然文の質問を受け取り、ウェブを検索してページを開き、**聞かれた値そのもの**（数値、名前、日付、カンマ区切りの列）と、その値が書いてある URL を返します。要約やリンク集は返しません。開いたページに値が無いときは答えを空にします。

公式仕様の PDF は作成時点で改訂中でした。入出力は `iwir/adapter.py` だけが知っています。項目名やファイル名の多くは環境変数で差し替えられます。

初期状態では有料 API を呼びません。検索はキー不要の DuckDuckGo HTML と Wikipedia、ページの直接取得です。AMD GPU 上の vLLM（OpenAI 互換）は、`LLM_BASE_URL` と `LLM_MODEL` を両方置いたときだけ使います。

このリポジトリから大会への提出は行いません。

### ローカル実行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 app.py \
  --query-id capital-france \
  --query "What is the capital of France?" \
  --output output/demo
```

`output/demo/capital-france.json` が1件できます。`answer` が値、`sources` がその値を含む URL、`confidence` が 0〜1 です。根拠が無い、検索に失敗した、24秒を超えた、のいずれでも JSON は書きます。そのときの `answer` は空文字です。

まとめて聞くときは JSON / JSONL / ディレクトリを `--input` に渡します。例は `input/questions.example.json` です。

```bash
python3 app.py --input input/questions.example.json --output output/batch
```

入力の別名はアダプタが吸収します。`id` / `query_id`、`question` / `query` / `text`、`start_url` / `url`。

### 出力を仕様に合わせる

- 既定 `SCHEMA_PROFILE=default` … `{id}.json` に `id`, `question`, `answer`, `sources`, `confidence`
- `SCHEMA_PROFILE=league3` … `{id}_output.json` に `answer`, `citations`, `confidence`（第3試合の形。引用がファイルパスではなく URL）
- キー名だけ変えるなら `OUTPUT_FIELDS=answer,citations,confidence` と `OUTPUT_FILENAME_TEMPLATE={id}_output.json`
- 構造そのものが変わったら `iwir/adapter.py` だけを直します

診断用の `diagnostics` は `OUTPUT_INCLUDE_DIAGNOSTICS=1` のときだけ付きます。採点用にはオフのままにしてください。

主な環境変数は上の英語の表と同一です。質問ごとの制限時間は `QUESTION_TIMEOUT_SECONDS`（既定 24）。検索順は `SEARCH_PROVIDERS`（既定 `duckduckgo,wikipedia`）。モデルは `LLM_BASE_URL` と `LLM_MODEL` が両方あるときだけ有効で、ローカル vLLM なら `LLM_API_KEY=EMPTY` で足ります。

### 処理の流れ

1. 質問に URL や `start_url` があれば、そのページを先に開く。
2. まず DuckDuckGo（HTML。インタースティシャルなら lite）を引き、ページから値が出なければ Wikipedia を足す。「フランスの首都」のような質問は項目名ではなく France の記事を検索する。プライベート IP、メタデータ、広告リダイレクトは捨て、Wikipedia を先に読む。
3. 上位ページを取得する。Wikipedia は本文 HTML 全文ではなく、導入節と infobox の API を使う。JSON は `項目: 値` に潰す。
4. infobox と文パターンから値を抜く。ページ本文に含まれない値は捨て、質問文の繰り返し（「フランスの首都は？」に対する「フランス」）も捨てる。
5. ローカルモデルが有効で、ヒューリスティックの確信度が低いときだけ抜粋を渡す。返答の値が引用ページに実在するときだけ採用し、文ではなく値に縮める。
6. 質問ごとに JSON を1ファイル書く。一時ファイルへ書いてから置き換える。

モデルのロードは `app.py` では行いません。24秒には収まらないため、vLLM は常駐させます。

### テストと評価

```bash
pip install -r requirements-dev.txt
pytest
python eval/evaluate.py --mode fixture --fail-under 1.0
python eval/evaluate.py --mode live
```

`fixture` は同梱ページだけで採点します。ネットワークも GPU も要りません。この環境ではテスト 42 件が通り、`python eval/evaluate.py --mode fixture --fail-under 1.0` は 11/11 でした。`live` は実際に検索します。DuckDuckGo がボット判定を返しても Wikipedia にフォールバックします。このサンドボックスでは数回の実行のあと HTTP 202 と HTTP 429 になり、それより前に「フランスの首都」は Paris、「金の元素記号」は Au、実在しない部品番号は空の答えになりました。制限されていないネットワークで `--mode live` を再実行してください。正答判定は完全一致と包含の両方で、レイテンシは質問ごとの処理時間です。

### Docker

```bash
docker build -t iwir .
docker run --rm \
  -v "$PWD/input/questions.example.json:/app/input/questions.json:ro" \
  -v "$PWD/output:/app/output" \
  iwir
```

イメージは `python:3.12-slim` です。GPU ランタイムは含めません。起動コマンドは `python3 /app/app.py` です。

### AMD ノートブックで vLLM / ROCm

[notebooks.amd.com/hackathon](https://notebooks.amd.com/hackathon) のノートブック上で:

```bash
python scripts/amd_vllm_notebook.py --check
python scripts/amd_vllm_notebook.py --print-serve-cmd
python scripts/amd_vllm_notebook.py --serve
```

`--serve` はサーバをフォアグラウンドで起動し、モデルをメモリに残します。既定モデルは `Qwen/Qwen2.5-7B-Instruct`、`--dtype float16`、コンテキスト 8192 です。接続確認だけなら `--model Qwen/Qwen2.5-0.5B-Instruct` で十分です。

別のセルで:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export LLM_API_KEY=EMPTY
python3 app.py --query-id demo --query "What is the capital of France?"
```

vLLM はノートブックのイメージに合った ROCm ビルドを使ってください。このスクリプトは CUDA 用の `pip install` を勝手に走らせません。手順は [vLLM の GPU インストール](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) を参照します。モデルを別コンテナにする場合のイメージ例は `vllm/vllm-openai-rocm` です。検索アプリのコンテナは、その HTTP に届ければ足ります。

サンプル問題は GPU が無くてもヒューリスティックで解けます。モデルは、パターンが空振りしたページや確信度の低いページで使います。ページに無い答えは、モデルが返しても捨てます。

### 制限

- DuckDuckGo HTML はデータセンタ IP から弾かれることがあります。そのときは Wikipedia を使います。
- JavaScript で描画されるページは実行しません。URL 先の JSON は読みます。
- ヒューリスティックは百科事典型の事実（infobox、「X は Y」、個数、年、短い列挙）向きです。
- リストの `answer` はカンマ区切りの文字列です。
- 同じ `id` の質問はファイルを上書きします。
- プライベート IP、localhost、クラウドのメタデータアドレスへは取りに行きません。

## License

MIT. See [LICENSE](LICENSE).
