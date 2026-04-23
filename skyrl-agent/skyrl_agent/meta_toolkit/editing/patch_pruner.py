"""PatchPruner: enforces hard limits on strategy and hook counts via LLM-based
semantic deduplication.

Replaces PatchArbiter and the hook_edits removal mechanism with a single
consolidated quality gate that runs after the planner and before canary.

The pruner operates on the *entire* patch library (existing + newly proposed),
clustering semantically similar patches and keeping the best representative.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient

logger = logging.getLogger(__name__)

_ALL_HOOK_POINTS = (
    "before_llm_call",
    "before_execute",
    "after_execute",
    "on_timeout",
    "after_round",
)

_MAX_LLM_RETRIES = 3

_STRATEGY_PRUNE_SYSTEM = """\
You are a patch-library curator for a terminal-task AI agent.

# Task
You are given {total} strategy entries (pattern + steps), numbered [0] to [{last}].
Your budget is {budget}. You must select EXACTLY {budget} strategies to KEEP.

# Rules
1. Cluster strategies that are semantically redundant (same situation / same advice).
2. From each redundant cluster, keep only the BEST representative (most general, most actionable).
3. Drop strategies about JSON parsing errors (the agent handles these already).
4. Drop overly task-specific patterns (referencing a single task ID).
5. Prefer strategies about file creation, shell scripting, verification workflows, heredoc.

# Output
Return STRICT JSON, nothing else — no markdown, no explanation, no <think> tags:
{{"keep": [list of 0-based integer indices to keep, exactly {budget} items]}}
"""

_HOOK_PRUNE_SYSTEM = """\
You are a hook-library curator for a terminal-task AI agent.

# Task
You are given {total} Python hook functions for the "{hook_point}" hook point,
numbered [0] to [{last}].
Your budget is {budget}. You must select EXACTLY {budget} hooks to KEEP.

# Rules for keeping
- Uses context fields (context.kv, context.episode, etc.) effectively.
- Handles edge cases (try/except, setdefault for kv).
- Addresses a real agent failure mode (verification loops, heredoc issues, stuck states).

# Rules for dropping
- Trivial (< 5 lines, does nothing meaningful).
- Has known issues (missing imports, undefined names, syntax warnings).
- Duplicates the intent of a higher-quality hook.

# Output
Return STRICT JSON, nothing else — no markdown, no explanation, no <think> tags:
{{"keep": [list of 0-based integer indices to keep, exactly {budget} items]}}
"""

_RETRY_PROMPT = """\
Your previous response was invalid: {error}

You MUST return STRICT JSON with exactly one key "keep" containing a list of \
exactly {budget} integer indices (0-based) from 0 to {last}.

Example for budget=3: {{"keep": [0, 2, 5]}}

Return ONLY the JSON object, nothing else.
"""


class PatchPruner:
    """Enforces hard limits on patch library size via LLM semantic deduplication.

    The pruner runs before canary evaluation so that the evaluated candidate
    always operates within the library size limits.
    """

    def __init__(
        self,
        client: MetaLLMClient | None,
        override_base: str,
        max_strategies: int = 32,
        max_hooks_per_category: int = 8,
    ) -> None:
        self._client = client
        self._base = Path(override_base)
        self._max_strategies = max_strategies
        self._max_hooks = max_hooks_per_category
        self._last_report: dict[str, Any] = {}

    @property
    def last_report(self) -> dict[str, Any]:
        return self._last_report

    async def prune(self) -> dict[str, Any]:
        """Prune the patch library in-place, enforcing hard limits.

        Returns a report dict with before/after counts and lists of dropped items.
        """
        report: dict[str, Any] = {"strategies": {}, "hooks": {}}

        strat_report = await self._prune_strategies()
        report["strategies"] = strat_report

        hook_report = await self._prune_hooks()
        report["hooks"] = hook_report

        self._last_report = report
        total_dropped = strat_report.get("dropped", 0) + sum(
            cat.get("dropped", 0) for cat in hook_report.values()
        )
        logger.info(
            f"PatchPruner: strategies {strat_report.get('before', '?')}"
            f"->{strat_report.get('after', '?')}, "
            f"hooks total dropped={total_dropped}"
        )
        return report

    # ------------------------------------------------------------------
    # Strategy pruning
    # ------------------------------------------------------------------

    async def _prune_strategies(self) -> dict[str, Any]:
        sl_path = self._base / "strategy_library.yaml"
        if not sl_path.exists():
            return {"before": 0, "after": 0, "dropped": 0}

        content = yaml.safe_load(sl_path.read_text(encoding="utf-8")) or {}
        strategies = content.get("strategies", [])
        before = len(strategies)

        if before <= self._max_strategies:
            return {"before": before, "after": before, "dropped": 0, "dropped_indices": []}

        keep_indices = await self._llm_select_with_retry(
            self._build_strategy_messages(strategies),
            total=before,
            budget=self._max_strategies,
            label="strategies",
        )

        pruned = [strategies[i] for i in sorted(keep_indices)]
        content["strategies"] = pruned
        sl_path.write_text(
            yaml.dump(content, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

        dropped_indices = sorted(set(range(before)) - set(keep_indices))
        dropped_patterns = [str(strategies[i].get("pattern", ""))[:80] for i in dropped_indices]
        after = len(pruned)

        logger.info(f"Strategies pruned: {before} -> {after} (dropped {before - after})")
        return {
            "before": before,
            "after": after,
            "dropped": before - after,
            "dropped_indices": dropped_indices,
            "dropped_patterns": dropped_patterns[:20],
        }

    def _build_strategy_messages(self, strategies: list[dict]) -> list[dict[str, str]]:
        items = []
        for i, s in enumerate(strategies):
            pattern = str(s.get("pattern", ""))[:200]
            steps = [str(st)[:100] for st in s.get("steps", [])[:5]]
            items.append(f"[{i}] pattern: {pattern}\n    steps: {steps}")

        system = _STRATEGY_PRUNE_SYSTEM.format(
            total=len(strategies),
            last=len(strategies) - 1,
            budget=self._max_strategies,
        )
        user_msg = "\n".join(items)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Hook pruning
    # ------------------------------------------------------------------

    async def _prune_hooks(self) -> dict[str, dict[str, Any]]:
        hooks_dir = self._base / "hooks"
        if not hooks_dir.is_dir():
            return {}

        by_point: dict[str, list[tuple[str, str]]] = {}
        for hf in sorted(hooks_dir.glob("*.py")):
            hp = _hook_point_from_name(hf.stem)
            if hp:
                by_point.setdefault(hp, []).append((hf.stem, hf.read_text(encoding="utf-8")))

        report: dict[str, dict[str, Any]] = {}
        for hp in _ALL_HOOK_POINTS:
            entries = by_point.get(hp, [])
            before = len(entries)
            if before <= self._max_hooks:
                report[hp] = {"before": before, "after": before, "dropped": 0}
                continue

            keep_indices = await self._llm_select_with_retry(
                self._build_hook_messages(hp, entries),
                total=before,
                budget=self._max_hooks,
                label=f"hooks/{hp}",
            )

            keep_names = {entries[i][0] for i in keep_indices}
            deleted_names = []
            for name, _ in entries:
                if name not in keep_names:
                    hook_path = hooks_dir / f"{name}.py"
                    if hook_path.exists():
                        hook_path.unlink()
                        deleted_names.append(name)

            after = before - len(deleted_names)
            report[hp] = {
                "before": before,
                "after": after,
                "dropped": len(deleted_names),
                "dropped_names": deleted_names,
            }
            if deleted_names:
                logger.info(f"Hooks [{hp}] pruned: {before} -> {after} (dropped {deleted_names})")

        return report

    def _build_hook_messages(
        self, hook_point: str, entries: list[tuple[str, str]]
    ) -> list[dict[str, str]]:
        items = []
        for i, (name, source) in enumerate(entries):
            src_preview = source.strip()[:600]
            items.append(f"[{i}] filename: {name}\n```python\n{src_preview}\n```")

        system = _HOOK_PRUNE_SYSTEM.format(
            total=len(entries),
            last=len(entries) - 1,
            budget=self._max_hooks,
            hook_point=hook_point,
        )
        user_msg = "\n\n".join(items)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Robust multi-round LLM selection
    # ------------------------------------------------------------------

    async def _llm_select_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        total: int,
        budget: int,
        label: str,
    ) -> list[int]:
        """Call LLM to select indices, with multi-round retry on failure.

        Round 1: initial request.
        Round 2+: append the error + retry prompt to the conversation and
                  ask again, giving the LLM a chance to self-correct.

        If all retries fail, raises the last exception (no silent fallback).
        """
        if self._client is None:
            raise RuntimeError("LLM client is None")

        self._client.set_role("planning")
        conv = list(messages)
        last_error: Exception | None = None

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                raw = await self._client.chat(
                    conv, temperature=0.1, max_tokens=1024, json_mode=False,
                )
                keep = self._parse_keep_list(raw, total=total, budget=budget)
                logger.info(
                    f"Pruner [{label}]: LLM selected {len(keep)}/{total} "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                return keep

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Pruner [{label}] attempt {attempt}/{_MAX_LLM_RETRIES} "
                    f"failed: {exc}"
                )
                if attempt < _MAX_LLM_RETRIES:
                    conv.append({"role": "assistant", "content": raw if 'raw' in dir() else ""})
                    conv.append({
                        "role": "user",
                        "content": _RETRY_PROMPT.format(
                            error=str(exc)[:300],
                            budget=budget,
                            last=total - 1,
                        ),
                    })
                    await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Pruner [{label}]: all {_MAX_LLM_RETRIES} LLM attempts failed. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_keep_list(raw: str, *, total: int, budget: int) -> list[int]:
        """Extract and validate a keep-list from raw LLM output."""
        import re

        # Strip <think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Try to find JSON object
        json_match = re.search(r'\{[^{}]*"keep"\s*:\s*\[[\d\s,]*\][^{}]*\}', cleaned)
        if json_match:
            data = json.loads(json_match.group())
        else:
            # Try extracting just a list of numbers
            list_match = re.search(r'\[[\d\s,]+\]', cleaned)
            if list_match:
                data = {"keep": json.loads(list_match.group())}
            else:
                raise ValueError(f"No JSON or number list found in response: {cleaned[:200]}")

        keep = data.get("keep", [])
        if not isinstance(keep, list):
            raise ValueError(f"'keep' is not a list: {type(keep)}")

        valid = sorted(set(i for i in keep if isinstance(i, int) and 0 <= i < total))
        if not valid:
            raise ValueError("keep list is empty after validation")

        if len(valid) > budget:
            valid = valid[:budget]

        return valid


def _hook_point_from_name(stem: str) -> str | None:
    for hp in _ALL_HOOK_POINTS:
        if stem == hp or stem.startswith(f"{hp}_"):
            return hp
    return None
