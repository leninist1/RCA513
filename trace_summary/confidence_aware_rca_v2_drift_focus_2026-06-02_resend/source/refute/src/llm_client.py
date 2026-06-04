"""
llm_client.py -- optional external LLM adapters for Layer 3.

No API keys are stored in code. Configure with environment variables:

  REFUTE_LLM_PROVIDER=anthropic_proxy | deepseek | offline

  REFUTE_ANTHROPIC_PROXY_URL=https://api.example.com/v1/messages
  REFUTE_ANTHROPIC_API_KEY=...
  REFUTE_ANTHROPIC_MODEL=claude-3-5-sonnet-latest

  REFUTE_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1/chat/completions
  REFUTE_DEEPSEEK_API_KEY=...
  REFUTE_DEEPSEEK_MODEL=deepseek-chat
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Iterable, Optional
from urllib import request, error


class LLMConfigurationError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    provider: str
    model: str
    text: str
    raw: dict


def _post_json(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMRequestError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
    except error.URLError as exc:
        raise LLMRequestError(f"LLM connection failed: {exc}") from exc


class BaseLLMClient:
    provider = "base"

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class AnthropicMessagesClient(BaseLLMClient):
    provider = "anthropic_proxy"

    def __init__(self, api_key: str, url: str, model: str, timeout: int = 60, max_tokens: int = 900):
        if not api_key:
            raise LLMConfigurationError("missing REFUTE_ANTHROPIC_API_KEY")
        self.api_key = api_key
        self.url = url
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        raw = _post_json(self.url, headers, payload, self.timeout)
        parts = raw.get("content", [])
        text = ""
        if isinstance(parts, list):
            text = "\n".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        if not text:
            text = str(raw.get("text", ""))
        return LLMResponse(self.provider, self.model, text.strip(), raw)


class OpenAIChatClient(BaseLLMClient):
    provider = "deepseek"

    def __init__(self, api_key: str, url: str, model: str, timeout: int = 60, max_tokens: int = 900):
        if not api_key:
            raise LLMConfigurationError("missing REFUTE_DEEPSEEK_API_KEY")
        self.api_key = api_key
        self.url = url
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.1,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        raw = _post_json(self.url, headers, payload, self.timeout)
        choices = raw.get("choices", [])
        text = ""
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        return LLMResponse(self.provider, self.model, str(text).strip(), raw)


def build_llm_client_from_env(provider: Optional[str] = None) -> Optional[BaseLLMClient]:
    provider = (provider or os.getenv("REFUTE_LLM_PROVIDER", "offline")).strip().lower()
    if provider in {"", "offline", "none", "disabled"}:
        return None
    timeout = int(os.getenv("REFUTE_LLM_TIMEOUT", "60"))
    max_tokens = int(os.getenv("REFUTE_LLM_MAX_TOKENS", "900"))
    if provider in {"anthropic", "anthropic_proxy", "claude"}:
        return AnthropicMessagesClient(
            api_key=os.getenv("REFUTE_ANTHROPIC_API_KEY", ""),
            url=os.getenv("REFUTE_ANTHROPIC_PROXY_URL", "https://api.anthropic.com/v1/messages"),
            model=os.getenv("REFUTE_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            timeout=timeout,
            max_tokens=max_tokens,
        )
    if provider in {"deepseek", "ds"}:
        return OpenAIChatClient(
            api_key=os.getenv("REFUTE_DEEPSEEK_API_KEY", ""),
            url=os.getenv("REFUTE_DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions"),
            model=os.getenv("REFUTE_DEEPSEEK_MODEL", "deepseek-chat"),
            timeout=timeout,
            max_tokens=max_tokens,
        )
    raise LLMConfigurationError(f"unsupported REFUTE_LLM_PROVIDER: {provider}")


def compact_reports_for_prompt(reports: Iterable[dict], limit: int = 5) -> str:
    rows = []
    for report in list(reports)[:limit]:
        candidate = report.get("candidate", {})
        rows.append({
            "candidate": candidate,
            "rebuttal_score": report.get("rebuttal_score"),
            "support_strength": report.get("support_strength"),
            "supporting": report.get("supporting", [])[:3],
            "refuting": report.get("refuting", [])[:3],
        })
    return json.dumps(rows, ensure_ascii=False, indent=2)
