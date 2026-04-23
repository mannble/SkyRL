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
        Strips <think> blocks before extracting JSON. If extraction fails,
        performs one additional "JSON reformat" call as a best-effort repair.
        """
        raw = await self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=False
        )
        try:
            return _extract_json(raw)
        except ValueError as first_error:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You convert model output into strict JSON.\n"
                        "Return ONLY one valid JSON object.\n"
                        "Do not include markdown fences, explanations, or <think> tags."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Convert the following text into a single valid JSON object, "
                        "preserving keys/values whenever possible:\n\n"
                        f"{raw}"
                    ),
                },
            ]
            repaired_raw = await self.chat(
                repair_messages,
                temperature=0.0,
                max_tokens=max_tokens,
                json_mode=False,
            )
            try:
                return _extract_json(repaired_raw)
            except ValueError as repair_error:
                raise ValueError(
                    "Could not extract JSON from LLM response after repair attempt. "
                    f"first_error={first_error}; repair_error={repair_error}"
                ) from repair_error

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

    seen_sources: set[str] = set()
    for source in (cleaned, text):
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)

        for fenced in _iter_fenced_blocks(source):
            parsed = _parse_json_candidate(fenced)
            if parsed is not None:
                return parsed
            for obj in _iter_balanced_json_objects(fenced):
                parsed = _parse_json_candidate(obj)
                if parsed is not None:
                    return parsed

        for obj in _iter_balanced_json_objects(source):
            parsed = _parse_json_candidate(obj)
            if parsed is not None:
                return parsed

    # Last resort: find anything starting with { and try to close it
    last_resort = cleaned or text
    brace_idx = last_resort.find("{")
    if brace_idx >= 0:
        repaired = _try_repair_json(last_resort[brace_idx:])
        if repaired is not None:
            return repaired

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")


def _iter_fenced_blocks(source: str) -> list[str]:
    blocks: list[str] = []
    for pattern in (
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
    ):
        for match in re.finditer(pattern, source, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            if block:
                blocks.append(block)
    return blocks


def _iter_balanced_json_objects(source: str):
    """Yield balanced {...} substrings while respecting quoted strings."""
    start: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(source):
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield source[start : i + 1]
                start = None


def _parse_json_candidate(candidate: str) -> dict[str, Any] | None:
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return _try_repair_json(candidate)


def _try_repair_json(text: str) -> dict[str, Any] | None:
    """Try to repair truncated JSON by closing open braces/brackets."""
    text = text.strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # If there is trailing garbage after a complete object, trim to last '}'.
    last_close = text.rfind("}")
    if last_close != -1:
        clipped = text[: last_close + 1].rstrip()
        try:
            parsed = json.loads(clipped)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            text = clipped

    # Remove trailing comma
    text = re.sub(r",\s*$", "", text)
    # Truncated escape sequence at EOF breaks JSON parsing.
    text = re.sub(r"\\+$", "", text)

    # Close any open strings
    if text.count('"') % 2 != 0:
        text += '"'

    # Count open braces/brackets while ignoring JSON strings.
    open_braces, open_brackets = _count_unclosed_json_delimiters(text)

    suffix = "]" * max(0, open_brackets) + "}" * max(0, open_braces)
    if suffix:
        candidate = text + suffix
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _count_unclosed_json_delimiters(text: str) -> tuple[int, int]:
    """Count unmatched '{' and '[' delimiters while respecting quoted strings."""
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in text:
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces = max(0, open_braces - 1)
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets = max(0, open_brackets - 1)

    return open_braces, open_brackets
