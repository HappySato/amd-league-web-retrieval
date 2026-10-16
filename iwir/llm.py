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

_SYSTEM = """You extract one factual value from source passages.
Reply with a single JSON object and nothing else:
{"answer": string, "quote": string, "source_url": string, "confidence": number}
Rules:
- answer is the requested value itself (a number, name, date, or comma-separated list), not a sentence and not an explanation.
- quote is an exact contiguous substring of the passage you used.
- source_url is copied from one passage URL.
- If the passages do not contain the value, use empty strings for answer and quote and 0 for confidence.
- Do not invent facts and do not use outside knowledge."""


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.base_url = config.llm_base_url.strip()
        self.model = config.llm_model.strip()
        self.api_key = config.llm_api_key
        self.timeout = config.llm_timeout
        self.json_mode = config.llm_json_mode
        self.enabled = bool(self.base_url and self.model)
        self._down = False

    def extract(self, question: str, pages: list[Page], deadline: float) -> LLMAnswer | None:
        if not self.enabled or self._down:
            return None
        remaining = deadline - time.monotonic()
        if remaining < 1.2 or not pages:
            return None
        passages = render_passages(question, pages)
        if not passages:
            return None
        user = f"Question: {question}\n\nPassages:\n{passages}"
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            payload = self._post(body, min(self.timeout, remaining))
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {400, 404, 422} and "response_format" in body:
                log.info("retrying LLM without JSON mode after HTTP %s", status)
                body.pop("response_format", None)
                try:
                    payload = self._post(body, min(self.timeout, deadline - time.monotonic()))
                except httpx.HTTPError as retry_exc:
                    log.warning("LLM request failed: %s", retry_exc)
                    return None
            else:
                if status in {401, 403}:
                    self._down = True
                log.warning("LLM request failed: %s", exc)
                return None
        except httpx.ConnectError as exc:
            self._down = True
            log.warning("LLM endpoint unreachable, using the extract-only path: %s", exc)
            return None
        except httpx.HTTPError as exc:
            log.warning("LLM request failed: %s", exc)
            return None
        return parse_llm_payload(payload)

    def _post(self, body: dict[str, Any], budget: float) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        token = self.api_key if self.api_key else "EMPTY"
        headers["Authorization"] = f"Bearer {token}"
        timeout = httpx.Timeout(max(0.5, budget), connect=min(3.0, max(0.5, budget)))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(chat_completions_url(self.base_url), headers=headers, json=body)
            response.raise_for_status()
            return response.json()


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


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


def parse_llm_payload(payload: dict[str, Any]) -> LLMAnswer | None:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        log.warning("LLM response missing choices")
        return None
    text = _message_text(content)
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
