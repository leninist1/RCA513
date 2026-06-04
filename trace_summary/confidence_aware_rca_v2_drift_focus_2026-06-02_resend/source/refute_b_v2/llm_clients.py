"""Real LLM clients for the Scheme B tool loop.

Secrets are never hard-coded. Configure clients with environment variables:

Claude relay:
  SHQBB_API_KEY or CLAUDE_API_KEY or ANTHROPIC_API_KEY
  REFUTE_ANTHROPIC_API_KEY is also accepted for compatibility
  CLAUDE_BASE_URL, default https://api.shqbb.com/v1/messages
  REFUTE_ANTHROPIC_PROXY_URL is also accepted
  CLAUDE_MODEL, default claude-3-5-sonnet-latest

DeepSeek:
  DEEPSEEK_API_KEY
  REFUTE_DEEPSEEK_API_KEY is also accepted for compatibility
  DEEPSEEK_BASE_URL, default https://api.deepseek.com/v1/chat/completions
  REFUTE_DEEPSEEK_BASE_URL is also accepted
  DEEPSEEK_MODEL, default deepseek-chat
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Mapping
from urllib import request, error

from refute_b_v2.llm_tool_loop import ToolCall, ToolLoopState


class LLMApiError(RuntimeError):
    pass


class MissingLLMCredentials(LLMApiError):
    pass


@dataclass(frozen=True)
class LLMDecision:
    action: str
    tool_name: str = ""
    tool_args: Mapping[str, Any] | None = None
    final: str = ""


class JSONToolLLMClient:
    """Base policy that asks an LLM for strict JSON tool decisions."""

    provider: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 900):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._last_final: dict[str, str] = {}

    def next_tool(self, state: ToolLoopState) -> ToolCall | None:
        decision = self._decide(state, purpose="tool")
        if decision.action == "tool":
            return ToolCall(decision.tool_name, dict(decision.tool_args or {}))
        if decision.final:
            self._last_final[state.case_id] = decision.final
        return None

    def finalize(self, state: ToolLoopState) -> str:
        if state.case_id in self._last_final:
            return self._last_final[state.case_id]
        decision = self._decide(state, purpose="final")
        return decision.final or "LLM returned no final recommendation."

    def _decide(self, state: ToolLoopState, purpose: str) -> LLMDecision:
        text = self.complete(_system_prompt(), _user_prompt(state, purpose))
        data = _extract_json(text)
        action = str(data.get("action", "final"))
        if action == "tool":
            tool = str(data.get("tool_name", ""))
            if tool not in state.available_tools:
                return LLMDecision("final", final=f"Requested unavailable tool: {tool}")
            return LLMDecision("tool", tool_name=tool, tool_args=dict(data.get("tool_args", {}) or {}))
        final_value = data.get("final", "")
        if isinstance(final_value, (dict, list)):
            final_text = json.dumps(final_value, ensure_ascii=False)
        else:
            final_text = str(final_value)
        return LLMDecision("final", final=final_text)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class ClaudeRelayClient(JSONToolLLMClient):
    provider = "claude"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, temperature: float = 0.0, max_tokens: int = 900):
        api_key = api_key or _first_env("SHQBB_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_API_KEY", "REFUTE_ANTHROPIC_API_KEY")
        if not api_key:
            raise MissingLLMCredentials("Set SHQBB_API_KEY, CLAUDE_API_KEY, ANTHROPIC_API_KEY, or REFUTE_ANTHROPIC_API_KEY.")
        super().__init__(model or _first_env("CLAUDE_MODEL", "REFUTE_ANTHROPIC_MODEL") or "claude-3-5-sonnet-latest", temperature, max_tokens)
        self.api_key = api_key
        self.base_url = base_url or _first_env("CLAUDE_BASE_URL", "REFUTE_ANTHROPIC_PROXY_URL") or "https://api.shqbb.com/v1/messages"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        data = _post_json(self.base_url, payload, {
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
        })
        content = data.get("content", [])
        if isinstance(content, list):
            return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content)


class DeepSeekClient(JSONToolLLMClient):
    provider = "deepseek"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, temperature: float = 0.0, max_tokens: int = 900):
        api_key = api_key or _first_env("DEEPSEEK_API_KEY", "REFUTE_DEEPSEEK_API_KEY")
        if not api_key:
            raise MissingLLMCredentials("Set DEEPSEEK_API_KEY or REFUTE_DEEPSEEK_API_KEY.")
        super().__init__(model or _first_env("DEEPSEEK_MODEL", "REFUTE_DEEPSEEK_MODEL") or "deepseek-chat", temperature, max_tokens)
        self.api_key = api_key
        self.base_url = base_url or _first_env("DEEPSEEK_BASE_URL", "REFUTE_DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1/chat/completions"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = _post_json(self.base_url, payload, {"Authorization": f"Bearer {self.api_key}"})
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return str(message.get("content", ""))


def llm_client_from_env(provider: str | None = None) -> JSONToolLLMClient:
    key = (provider or _first_env("SCHEME_B_LLM_PROVIDER", "REFUTE_LLM_PROVIDER") or "claude").strip().lower()
    if key in {"claude", "anthropic", "anthropic_proxy", "shqbb"}:
        return ClaudeRelayClient()
    if key in {"deepseek", "ds"}:
        return DeepSeekClient()
    if key in {"", "none", "offline", "disabled"}:
        raise MissingLLMCredentials("LLM provider is disabled.")
    raise ValueError(f"unknown LLM provider: {provider}")


def _post_json(url: str, payload: Mapping[str, Any], headers: Mapping[str, str]) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **dict(headers)},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LLMApiError(f"LLM HTTP {exc.code}: {detail[:500]}") from exc
    except error.URLError as exc:
        raise LLMApiError(f"LLM request failed: {exc}") from exc


def _extract_json(text: str) -> dict:
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
        raise LLMApiError(f"LLM did not return JSON: {text[:300]}")


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def _system_prompt() -> str:
    return (
        "You are an SRE RCA tool-loop controller. "
        "You must only use facts in the provided evidence matrix and tool observations. "
        "If a required_final_schema is provided, the final field must be a JSON object following that schema. "
        "Return strict JSON only."
    )


def _user_prompt(state: ToolLoopState, purpose: str) -> str:
    context = state.evidence_matrix.get("context", {}) if isinstance(state.evidence_matrix, dict) else {}
    final_schema = context.get("required_final_schema") if isinstance(context, dict) else None
    final_example: Any = "<decision, ambiguity, or blind spot explanation>"
    if final_schema:
        final_example = final_schema
    payload = {
        "case_id": state.case_id,
        "purpose": purpose,
        "available_tools": list(state.available_tools),
        "evidence_matrix": state.evidence_matrix,
        "observations": [
            {"tool": obs.call.name, "args": obs.call.args, "result": obs.result}
            for obs in state.observations
        ],
        "allowed_outputs": [
            {"action": "tool", "tool_name": "<one available tool>", "tool_args": {}},
            {"action": "final", "final": final_example},
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
