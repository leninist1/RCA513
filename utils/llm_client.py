"""Minimal LLM client supporting DeepSeek (OpenAI-compatible) and Claude."""

import os
import json
import requests
from typing import List, Dict, Optional, Any


class LLMResponse:
    def __init__(self, content: str, usage: Dict = None):
        self.content = content
        self.usage = usage or {}


class LLMClient:
    def __init__(self, provider: str = "deepseek", api_key: str = None, base_url: str = None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url or ("https://api.deepseek.com" if provider == "deepseek" else "https://api.anthropic.com")

    def call(
        self, model: str = "deepseek-chat", messages: List[Dict] = None,
        system: str = "", response_format: dict = None, max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Optional[LLMResponse]:
        """Make LLM API call with retry logic."""
        if not self.api_key:
            return None

        if self.provider == "deepseek":
            return self._call_deepseek(model, messages, system, response_format, max_tokens, temperature)
        elif self.provider == "anthropic":
            return self._call_claude(model, messages, system, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_deepseek(
        self, model: str, messages: List[Dict], system: str,
        response_format: dict = None, max_tokens: int = 1024, temperature: float = 0.1,
    ) -> Optional[LLMResponse]:
        """Call DeepSeek API (OpenAI-compatible)."""
        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = {
            "model": model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]["message"]
            content = choice.get("content", "")

            # Handle json_object response format
            if response_format and response_format.get("type") == "json_object":
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = self._extract_json(content)

            return LLMResponse(
                content=content,
                usage=data.get("usage", {}),
            )
        except Exception as e:
            print(f"  [LLM call failed: {e}]")
            return None

    def _call_claude(
        self, model: str, messages: List[Dict], system: str,
        max_tokens: int = 1024, temperature: float = 0.1,
    ) -> Optional[LLMResponse]:
        """Call Claude API (Anthropic)."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            system_msgs = [{"type": "text", "text": system}] if system else []
            formatted_msgs = []
            for m in messages:
                formatted_msgs.append({
                    "role": m["role"],
                    "content": [{"type": "text", "text": m["content"]}],
                })

            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msgs if system_msgs else None,
                messages=formatted_msgs,
            )

            content = resp.content[0].text if resp.content else ""
            usage = {
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
                "prompt_tokens": resp.usage.input_tokens,
                "completion_tokens": resp.usage.output_tokens,
            }

            return LLMResponse(content=content, usage=usage)
        except Exception as e:
            print(f"  [Claude call failed: {e}]")
            return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Try to extract JSON from LLM text response."""
        import re
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": "json_parse_failed", "raw": text[:200]}
