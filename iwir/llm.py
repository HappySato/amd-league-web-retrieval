"""OpenAI-compatible chat client for a local vLLM (or any /v1) server.

Nothing is called unless both LLM_BASE_URL and LLM_MODEL are set. There is no
default host, so a paid API is never contacted by accident.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from iwir.config import Config
from iwir.models import LLMAnswer, Page
from iwir.textutil import canon_terms, split_sentences

log = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """You extract one factual value from source passages.
Reply with a single JSON object and nothing else:
{"answer": string, "quote": string, "source_url": string, "confidence": number}
Rules:
- answer is the requested value itself (a number, name, date, or comma-separated list), not a sentence and not an explanation.
- quote is an exact contiguous substring of the passage you used.
- source_url is copied from one passage URL.
- If the passages do not contain the value, use empty strings for answer and quote and 0 for confidence.
- Do not invent facts and do not use outside knowledge."""

_REWRITE_SYSTEM = """You turn a question into web search queries.
Reply with a single JSON object and nothing else:
{"queries": [string, ...]}
Rules:
- Return 1 to 3 short queries a public search engine can run.
- Name the entity and the attribute. Example: "capital of France".
- Do not answer the question. Do not explain. Do not add URLs."""


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.base_url = config.llm_base_url.strip()
        self.model = config.llm_model.strip()
        self.api_key = config.llm_api_key
        self.timeout = config.llm_timeout
        self.json_mode = config.llm_json_mode
        self.enabled = bool(self.base_url and self.model)
        self._down = False
        self._probed = False

    def available(self, deadline: float) -> bool:
        """True after a short health check against the configured host.

        A refused connection, a timeout, or an auth error marks the client down
        for the rest of the process so later questions stay on the rule extractor.
        """
        if not self.enabled or self._down:
            return False
        if self._probed:
            return True
        remaining = deadline - time.monotonic()
        if remaining < 1.0:
            return False
        self._probed = True
        try:
            self._get(models_url(self.base_url), min(2.0, remaining))
        except httpx.ConnectError as exc:
            self._mark_down(f"LLM endpoint unreachable: {exc}")
            return False
        except httpx.TimeoutException as exc:
            self._mark_down(f"LLM health check timed out: {exc}")
            return False
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403, 404}:
                self._mark_down(f"LLM health check HTTP {status}")
                return False
            self._mark_down(f"LLM health check HTTP {status}")
            return False
        except httpx.HTTPError as exc:
            self._mark_down(f"LLM health check failed: {exc}")
            return False
        log.info("LLM reachable at %s model=%s", self.base_url, self.model)
        return True

    def rewrite_queries(self, question: str, deadline: float) -> list[str]:
        """Search strings for this question. Empty when the model cannot be used."""
        if not self.available(deadline):
            return []
        remaining = deadline - time.monotonic()
        # Leave room for fetch plus a later extract call inside the question budget.
        if remaining < 8:
            log.info("skipping query rewrite; %.1fs left in the question budget", remaining)
            return []
        budget = min(4.0, self.timeout, remaining - 6.0)
        if budget < 0.8:
            return []
        body = self._body(
            _REWRITE_SYSTEM,
            f"Question: {question}",
            max_tokens=128,
        )
        payload = self._chat(body, budget, deadline)
        if payload is None:
            return []
        return parse_query_payload(payload)

    def extract(self, question: str, pages: list[Page], deadline: float) -> LLMAnswer | None:
        if not self.available(deadline):
            return None
        remaining = deadline - time.monotonic()
        if remaining < 1.2 or not pages:
            return None
        passages = render_passages(question, pages)
        if not passages:
            return None
        budget = min(self.timeout, max(0.5, remaining - 0.2))
        user = f"Question: {question}\n\nPassages:\n{passages}"
        body = self._body(_EXTRACT_SYSTEM, user, max_tokens=256)
        payload = self._chat(body, budget, deadline)
        if payload is None:
            return None
        return parse_llm_payload(payload)

    def _body(self, system: str, user: str, *, max_tokens: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _chat(self, body: dict[str, Any], budget: float, deadline: float) -> dict[str, Any] | None:
        try:
            return self._post(body, budget)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {400, 404, 422} and "response_format" in body:
                log.info("retrying LLM without JSON mode after HTTP %s", status)
                body.pop("response_format", None)
                try:
                    return self._post(body, min(budget, max(0.5, deadline - time.monotonic())))
                except httpx.HTTPError as retry_exc:
                    log.warning("LLM request failed: %s", retry_exc)
                    return None
            if status in {401, 403}:
                self._mark_down(f"LLM request HTTP {status}")
            else:
                log.warning("LLM request failed: %s", exc)
            return None
        except httpx.ConnectError as exc:
            self._mark_down(f"LLM endpoint unreachable: {exc}")
            return None
        except httpx.TimeoutException as exc:
            log.warning("LLM call exceeded %.1fs; using the rule extractor: %s", budget, exc)
            return None
        except httpx.HTTPError as exc:
            log.warning("LLM request failed: %s", exc)
            return None

    def _mark_down(self, message: str) -> None:
        self._down = True
        log.warning("%s Using the rule extractor for the rest of this process.", message)

    def _get(self, url: str, budget: float) -> None:
        timeout = httpx.Timeout(max(0.4, budget), connect=min(1.5, max(0.4, budget)))
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=self._headers())
            response.raise_for_status()

    def _post(self, body: dict[str, Any], budget: float) -> dict[str, Any]:
        timeout = httpx.Timeout(max(0.5, budget), connect=min(1.5, max(0.5, budget)))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(chat_completions_url(self.base_url), headers=self._headers(), json=body)
            response.raise_for_status()
            return response.json()

    def _headers(self) -> dict[str, str]:
        token = self.api_key if self.api_key else "EMPTY"
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def models_url(base_url: str) -> str:
    """GET target for the OpenAI models list, used as a health check."""
    chat = chat_completions_url(base_url)
    suffix = "/chat/completions"
    if chat.endswith(suffix):
        return chat[: -len(suffix)] + "/models"
    return chat + "/models"


def render_passages(question: str, pages: list[Page], limit: int = 5500) -> str:
    terms = canon_terms(question)
    blocks: list[str] = []
    used = 0
    for page in pages:
        if page.from_snippet and len(pages) > 1:
            continue
        sentences = split_sentences(page.text)
        ranked = sorted(
            sentences,
            key=lambda sentence: len(terms & canon_terms(sentence)),
            reverse=True,
        )
        chosen: list[str] = []
        lead = page.text[:500].strip()
        if lead:
            chosen.append(lead)
        for sentence in ranked[:4]:
            if sentence not in lead and sentence not in chosen:
                chosen.append(sentence)
        body = "\n".join(chosen)[:1800]
        if not body:
            continue
        if used + len(body) > limit:
            break
        blocks.append(f"URL: {page.url}\n{body}")
        used += len(body)
    return "\n\n".join(blocks)


def parse_query_payload(payload: dict[str, Any]) -> list[str]:
    text = _content_of(payload)
    if text is None:
        return []
    obj = parse_json_object(text)
    if not obj:
        log.warning("LLM query rewrite was not JSON")
        return []
    raw = obj.get("queries")
    if raw is None and obj.get("query"):
        raw = [obj.get("query")]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    queries: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text_item = re.sub(r"\s+", " ", _as_text(item)).strip(" \"'`")
        if not text_item or len(text_item) > 160 or any(ord(ch) < 32 for ch in text_item):
            continue
        key = text_item.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(text_item)
        if len(queries) == 3:
            break
    return queries


def _content_of(payload: dict[str, Any]) -> str | None:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        log.warning("LLM response missing choices")
        return None
    return _message_text(content)


def parse_llm_payload(payload: dict[str, Any]) -> LLMAnswer | None:
    text = _content_of(payload)
    if text is None:
        return None
    obj = parse_json_object(text)
    if not obj:
        log.warning("LLM response was not JSON")
        return None
    answer = _as_text(obj.get("answer"))
    if not answer:
        return None
    quote = _as_text(obj.get("quote"))
    source = _as_text(obj.get("source_url") or obj.get("url") or obj.get("source"))
    try:
        confidence = float(obj.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    return LLMAnswer(
        answer=answer,
        quote=quote,
        source_url=source,
        confidence=max(0.0, min(1.0, confidence)),
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    return obj


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()
