"""Async LLM client for meta-learning reasoning (diagnosis, patch planning).

Uses any OpenAI-compatible chat completions API via aiohttp.
Supports local vLLM, Ollama, OpenAI, Anthropic-via-proxy, etc.
"""

from __future__ import annotations

import json
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MetaLLMConfig:
    """Configuration for the meta-reasoning LLM backend."""

    model: str = ""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_sec: float = 120.0

    def effective_api_key(self) -> str:
        return self.api_key or os.environ.get("META_LLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self.model)


@dataclass
class RecordedConversation:
    """A recorded LLM conversation for meta-RL training."""
    messages: list[dict[str, str]]
    response: str
    role: str = ""  # "diagnosis" or "planning"


class MetaLLMClient:
    """Lightweight async wrapper around an OpenAI-compatible chat API.

    Records all conversations so they can be used as meta-RL training data.
    """

    def __init__(self, config: MetaLLMConfig) -> None:
        self._cfg = config
        self._session: aiohttp.ClientSession | None = None
        self._recorded: list[RecordedConversation] = []
        self._current_role: str = ""

    def set_role(self, role: str) -> None:
        """Tag subsequent calls with a role (e.g. 'diagnosis', 'planning')."""
        self._current_role = role

    def pop_recorded(self) -> list[RecordedConversation]:
        """Pop and return all recorded conversations since last call."""
        convs = self._recorded
        self._recorded = []
        return convs

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._cfg.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant content string."""
        session = await self._ensure_session()

        url = f"{self._cfg.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._cfg.effective_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._cfg.temperature,
            "max_tokens": max_tokens or self._cfg.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        async with session.post(url, json=body, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"LLM API error {resp.status}: {text[:500]}")
            data = await resp.json()

        content = data["choices"][0]["message"]["content"]

        self._recorded.append(RecordedConversation(
            messages=list(messages),
            response=content,
            role=self._current_role,
        ))

        return content

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Chat and parse the response as JSON (single LLM call).

        Always uses json_mode=False to avoid double-call with thinking models.
        Strips <think> blocks before extracting JSON.
        """
        raw = await self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=False
        )
        return _extract_json(raw)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output."""
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return stripped if stripped else text


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from LLM output.

    Handles thinking models that wrap output in <think> tags, and
    attempts to repair truncated JSON from max_tokens cutoff.
    """
    cleaned = _strip_think_tags(text)

    for source in (cleaned, text):
        for pattern in [
            r"```json\s*\n(.*?)\n\s*```",
            r"```\s*\n(.*?)\n\s*```",
        ]:
            m = re.search(pattern, source, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    repaired = _try_repair_json(m.group(1))
                    if repaired is not None:
                        return repaired

        # Match the outermost { ... } block
        m = re.search(r"(\{.*\})", source, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                repaired = _try_repair_json(m.group(1))
                if repaired is not None:
                    return repaired

    # Last resort: find anything starting with { and try to close it
    m = re.search(r"(\{.*)", cleaned or text, re.DOTALL)
    if m:
        repaired = _try_repair_json(m.group(1))
        if repaired is not None:
            return repaired

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")


def _try_repair_json(text: str) -> dict[str, Any] | None:
    """Try to repair truncated JSON by closing open braces/brackets."""
    text = text.rstrip()
    # Remove trailing comma
    text = re.sub(r",\s*$", "", text)

    # Close any open strings
    if text.count('"') % 2 != 0:
        text += '"'

    # Count open braces/brackets and close them
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    suffix = "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    if suffix:
        candidate = text + suffix
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    return None
