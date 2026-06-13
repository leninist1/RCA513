"""LLM client abstraction for PRISM-LA.

Supports real OpenAI-compatible API calls and a deterministic mock for testing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import PRISMLAConfig
from .prompts import SYSTEM_PROMPT, build_investigator_prompt, build_initial_prompt

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> Optional[str]:
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        return match.group(1).strip()
    match = re.search(r'\{[\s\S]*"?action"?[\s\S]*\}', text)
    if match:
        return match.group(0).strip()
    return text.strip()


class LLMClient:
    def __init__(self, config: PRISMLAConfig):
        self.config = config
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.api_key = config.llm_api_key
        self.base_url = config.llm_base_url
        self._messages: List[Dict[str, Any]] = []
        self._connected = False

    def connect(self) -> bool:
        if not self.api_key:
            logger.warning("No LLM API key set; using deterministic mock client")
            self._connected = False
            return False
        try:
            from openai import OpenAI  # type: ignore
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            try:
                self._client.models.list()
            except Exception:
                pass
            self._connected = True
            logger.info(f"LLM client connected: model={self.model}")
            return True
        except Exception as exc:
            logger.warning(f"LLM client connection failed: {exc}; using mock")
            self._connected = False
            return False

    def start_session(self) -> None:
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def invoke_initial(self, case_view: Dict[str, Any], top_entities: Optional[List[str]] = None) -> Dict[str, Any]:
        prompt = build_initial_prompt(case_view, AVAILABLE_TOOLS, top_entities)
        response = self._chat(prompt)
        if self.config.log_llm_messages and self.config.debug_dir:
            self._log_message("initial", prompt, response)
        return self._parse_response(response)

    def invoke_plan(
        self,
        case_view: Dict[str, Any],
        events: List[Dict[str, Any]],
        ledger_entries: List[Dict[str, Any]],
        step: int,
    ) -> Dict[str, Any]:
        prompt = build_investigator_prompt(
            case_view, events, ledger_entries, step, AVAILABLE_TOOLS
        )
        response = self._chat(prompt)
        if self.config.log_llm_messages and self.config.debug_dir:
            self._log_message(f"step_{step}", prompt, response)
        return self._parse_response(response)

    def _chat(self, user_message: str) -> str:
        if not self._connected:
            return self._mock_response(user_message)
        try:
            from openai import OpenAI  # type: ignore
            messages = list(self._messages) + [{"role": "user", "content": user_message}]
            client: Any = self._client
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "text"},
            )
            content = completion.choices[0].message.content or ""
            return content
        except Exception as exc:
            logger.error(f"LLM API error: {exc}")
            return self._mock_response(user_message)

    def _parse_response(self, text: str) -> Dict[str, Any]:
        json_str = _extract_json_block(text)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON; using mock fallback")
            result = {"action": "plan", "rationale": "parse failure, running default probe", "investigations": []}
        if not isinstance(result, dict):
            result = {"action": "plan", "raw_response": str(result)}
        return result

    def _mock_response(self, user_message: str) -> str:
        step_match = re.search(r"STEP (\d+)", user_message)
        step = int(step_match.group(1)) if step_match else 0
        has_events = "event_id" in user_message or "HYPOTHESES" in user_message
        if step == 0 and not has_events:
            return json.dumps({
                "action": "plan",
                "step": 0,
                "rationale": "Initial probe: inspect metric and log for top candidate entities.",
                "investigations": [
                    {"tool": "find_time_anchors"},
                    {"tool": "inspect_metric", "component": "unknown", "anchor_index": 0},
                    {"tool": "inspect_log", "component": "unknown", "anchor_index": 0},
                ],
            })
        if step < 2 or not has_events:
            return json.dumps({
                "action": "plan",
                "step": step + 1,
                "rationale": "Gathering basic coverage across top entities.",
                "investigations": [
                    {"tool": "inspect_trace", "component": "unknown", "anchor_index": 0},
                    {"tool": "get_topology_neighbors", "component": "unknown"},
                ],
            })
        return json.dumps({
            "action": "stop",
            "rationale": "Investigation complete based on available evidence.",
            "root_cause_events": [],
        })

    def _log_message(self, label: str, prompt: str, response: str) -> None:
        if not self.config.debug_dir:
            return
        debug_dir = Path(self.config.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        log_path = debug_dir / f"llm_{label}.json"
        log_path.write_text(
            json.dumps({"prompt": prompt, "response": response}, indent=2),
            encoding="utf-8",
        )


AVAILABLE_TOOLS = [
    "find_time_anchors",
    "inspect_metric",
    "inspect_log",
    "inspect_trace",
    "get_topology_neighbors",
    "test_counterfactual",
    "explain_residual",
    "compare_events",
]
