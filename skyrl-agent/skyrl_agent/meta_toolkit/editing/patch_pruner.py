"""PatchPruner: curates strategy and hook patch libraries before canary.

Replaces PatchArbiter and the hook_edits removal mechanism with a single
consolidated quality gate that runs after the planner and before canary.

The pruner operates on the *entire* patch library (existing + newly proposed),
deduplicating semantically similar patches and enforcing hard count limits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient

logger = logging.getLogger(__name__)

_ALL_HOOK_POINTS = (
    "before_llm_call",
    "after_execute",
    "on_timeout",
    "after_round",
)
_DISABLED_HOOK_POINTS = {"before_execute"}

_MAX_LLM_RETRIES = 3

_STRATEGY_CURATE_SYSTEM = """\
You are a patch-library curator for a terminal-task AI agent.

# Task
You are given {total} strategy entries (pattern + steps + selection metadata),
numbered [0] to [{last}].
Return delete/merge decisions for this strategy library.

Hard budget: the final library must have AT MOST {budget} strategies.
If the library is already within budget and the strategies are useful and distinct,
return empty delete_indices and merge_groups.

# What to do
1. Find strategies that describe the same situation, give the same advice, or
   give conflicting advice for the same trigger.
2. For each duplicate or conflict cluster, choose one existing strategy index
   to replace with a merged strategy, and list the duplicate/conflicting
   indices to drop.
3. When merging conflicting strategies, keep the shared reusable intent and
   only the non-conflicting, evidence-safe steps. Remove duplicated or
   contradictory specifics.
4. List low-quality standalone strategies to delete.
5. Drop strategies about JSON parsing errors; those are handled outside the strategy library.
6. Drop overly task-specific global patterns: single task IDs, hard-coded answers,
   or concrete paths without a reusable trigger.
7. Prefer reusable guidance about file creation, shell scripting, verification,
   archives, permissions, Docker/database access, and heredoc/quoting pitfalls.
8. Prefer reusable strategies. Keep a narrow strategy only when it has a clear,
   reusable trigger and does not hard-code a task ID, answer, or one-off path.
9. If duplicate/conflict removal alone is not enough to fit the hard budget, add the
   weakest remaining strategy indices to delete_indices until the final count
   is within budget.
10. Do not delete or merge a strategy just to make a change.

# Important
Do NOT rewrite standalone strategy text outside merge_groups.
Only rewrite a representative strategy by providing merged_strategy inside a
merge_groups entry.
Do NOT invent unrelated strategies.
This curation step only deletes existing numbered strategies or replaces one
existing representative with a merged version of the same duplicate/conflict cluster.
Use ONLY the visible numeric strategy IDs from the input.
For merge_groups, "replace" is the existing strategy index that will be overwritten
by merged_strategy; "drop" lists duplicate strategy indices removed from that cluster.
merged_strategy must combine only the ideas already present in replace + drop.
merged_strategy may include optional selection metadata:
`scope` ("core", "general", or "task_specific"), `triggers`, and
`anti_triggers`. Preserve useful metadata from the merged cluster; do not invent
unrelated triggers.
For conflict clusters, merged_strategy must preserve common reusable guidance
and omit the conflicting specifics.

# Output
Return STRICT JSON, nothing else — no markdown, no explanation, no <think> tags:
{{
  "delete_indices": [list of 0-based integer indices to delete],
  "merge_groups": [
    {{
      "replace": 0,
      "drop": [2, 5],
      "merged_strategy": {{"pattern": "...", "steps": ["...", "..."], "scope": "general", "triggers": ["..."]}},
      "reason": "same failure mode; merged the reusable parts"
    }}
  ],
  "notes": ["brief note about important standalone deletions or duplicate clusters"]
}}
"""

_HOOK_GROUP_CURATE_SYSTEM = """\
You are a hook-library curator for a terminal-task AI agent.

# Task
You are given {total} hook groups, numbered [0] to [{last}].
Each group may contain one or more Python hook functions that were generated together.
Return which groups to DELETE.

Hard budget: keep at most {budget} hook groups total.
If the hook groups are already within budget and useful/distinct, return an empty
delete_groups list.

# Hook points and intended roles
- before_llm_call: initial or periodic short prompt guidance using append_prompt.
- after_execute: observe successful command output and record compact signals in context.kv.
- on_timeout: observe timeout output and record compact timeout signals in context.kv.
- after_round: append concise next_prompt guidance after the base observation.

# Keep groups that
- Address a real terminal-task failure mode, not a generic reminder.
- Use context.kv safely with get/setdefault and tolerate missing sibling signals.
- Add concise, evidence-based guidance only when the trigger is clear.
- Complement each other across hook points, for example after_execute records a
  signal that after_round later consumes.

# Drop groups that
- Duplicate the intent of a better group.
- Are trivial, noisy, or likely to fire every round without a clear trigger.
- Mainly repeat routine completion verification.
- Depend on fragile exact output text when a broader signal would work.
- Have suspicious code quality: undefined names, missing imports, unsafe kv reads,
  or broad behavior that could distract the base agent.

# Important
Do NOT rewrite hook code. This curation step only keeps or deletes whole groups.
If uncertain, keep the safer and shorter representative.
Use ONLY the visible numeric group IDs from the input.
Do not delete a group just to make a change.

# Output
Return STRICT JSON, nothing else — no markdown, no explanation, no <think> tags:
{{
  "delete_groups": [list of 0-based integer group indices to delete],
  "notes": ["brief reason for important deletions"]
}}
"""

_HOOK_GROUP_BUDGET_DELETE_SYSTEM = """\
You are a hook-library budget curator for a terminal-task AI agent.

# Task
You are given {total} already-curated hook groups, numbered [0] to [{last}].
The hard budget is {budget} hook groups, so you must delete {required_delete}
hook groups.

Return which hook groups to DELETE. Do not return keep indices.

# What to delete first
1. Near-duplicates: keep the group with clearer triggers and safer code.
2. Noisy groups: likely to fire every round, mostly generic reminders, or broad
   completion checks without concrete evidence.
3. Fragile groups: exact-output matching, unsafe context.kv reads, undefined
   names, or code that depends on missing sibling signals.
4. Narrower single-hook groups when a broader multi-hook group covers the same
   failure mode more safely.

# Important
Delete exactly {required_delete} groups when possible.
Do NOT delete more than {required_delete}; extra deletions may remove useful hooks.
Use ONLY the visible numeric group IDs from the input, shown as [0], [1], ...
Do not rewrite hook code in this stage.

# Output
Return STRICT JSON, nothing else — no markdown, no explanation, no <think> tags:
{{
  "delete_groups": [list of 0-based integer group indices to delete],
  "notes": ["brief reason for important deletions"]
}}
"""

_STRATEGY_BUDGET_DELETE_SYSTEM = """\
You are a patch-library budget curator for a terminal-task AI agent.

# Task
You are given {total} already-curated strategy entries, numbered [0] to [{last}].
Entries include pattern, steps, and optional selection metadata.
The hard budget is {budget} strategies, so you must delete {required_delete}
strategy entries.

Return which strategies to DELETE. Do not return keep indices.

# What to delete first
1. Near-duplicates: keep the stronger/more general strategy and delete the
   narrower duplicate.
2. Low-quality strategies: vague reminders, one-off paths, task IDs,
   hard-coded answers, or advice already covered by a broader strategy.
3. Overly specific strategies that are unlikely to recur in Terminal-Bench tasks.

# Important
Delete exactly {required_delete} indices when possible.
Do NOT delete more than {required_delete}; extra deletions may remove useful skills.
Use ONLY the visible numeric strategy IDs from the input, shown as [0], [1], ...
Do not rewrite strategies in this stage.

# Output
Return STRICT JSON, nothing else — no markdown, no explanation, no <think> tags:
{{
  "delete_indices": [list of 0-based integer indices to delete],
  "notes": ["brief reason for important deletions"]
}}
"""

_BUDGET_DELETE_RETRY_PROMPT = """\
Your previous response was invalid: {error}

Return STRICT JSON with:
- "delete_indices": a list of integer indices (0-based) from 0 to {last}.
- Optional "notes": a list of short strings.
Delete at most {required_delete} strategies.

Example: {{"delete_indices": [0, 2, 5], "notes": ["near-duplicates"]}}

Return ONLY the JSON object, nothing else.
"""

_HOOK_GROUP_RETRY_PROMPT = """\
Your previous response was invalid: {error}

Return STRICT JSON with:
- "delete_groups": a list of integer group indices (0-based) from 0 to {last}.
- Optional "notes": a list of short strings.
If no hook group should be deleted, return {{"delete_groups": []}}.

Return ONLY the JSON object, nothing else.
"""

_HOOK_GROUP_BUDGET_DELETE_RETRY_PROMPT = """\
Your previous response was invalid: {error}

Return STRICT JSON with:
- "delete_groups": a list of integer group indices (0-based) from 0 to {last}.
- Optional "notes": a list of short strings.
Delete at most {required_delete} hook groups.

Example: {{"delete_groups": [0, 2], "notes": ["near-duplicates"]}}

Return ONLY the JSON object, nothing else.
"""

_STRATEGY_RETRY_PROMPT = """\
Your previous response was invalid: {error}

Return STRICT JSON with:
- "delete_indices": a list of integer indices.
- "merge_groups": a list of objects with integer "replace", list[int] "drop",
  and "merged_strategy" containing "pattern", "steps", and optional
  "scope", "triggers", "anti_triggers".
- Optional "notes": a list of short strings.
The final kept strategy count must be at most {budget}.
If no strategy should change and the library is within budget, return empty
"delete_indices" and "merge_groups".

Return ONLY the JSON object, nothing else.
"""


class PatchPruner:
    """Curates and bounds patch library size via semantic deduplication.

    The pruner runs before canary evaluation so that the evaluated candidate
    always operates within the library size limits.
    """

    def __init__(
        self,
        client: MetaLLMClient | None,
        override_base: str,
        max_strategies: int = 32,
        max_hook_groups: int = 8,
        max_hooks_per_category: int | None = None,
    ) -> None:
        self._client = client
        self._base = Path(override_base)
        self._max_strategies = max_strategies
        # max_hooks_per_category is a legacy alias. Hook curation is now
        # group-budgeted, not hook-point-budgeted.
        self._max_hook_groups = (
            max_hook_groups
            if max_hooks_per_category is None
            else max_hooks_per_category
        )
        self._last_report: dict[str, Any] = {}

    @property
    def last_report(self) -> dict[str, Any]:
        return self._last_report

    async def prune(self) -> dict[str, Any]:
        """Curate the patch library in-place before canary.

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
        strategies = [
            s for s in content.get("strategies", [])
            if isinstance(s, dict) and s.get("pattern") and isinstance(s.get("steps"), list)
        ]
        before = len(strategies)

        if before == 0:
            return {"before": 0, "after": 0, "dropped": 0, "dropped_indices": []}

        curated = list(strategies)
        curator_delete_indices: list[int] = []
        curator_merge_groups: list[dict[str, Any]] = []
        notes: list[str] = []
        curator_error: str | None = None
        llm_used = False

        if self._client is not None and len(strategies) > 1:
            try:
                payload = await self._llm_curate_strategies(strategies)
                curated = self._apply_strategy_curator_decisions(strategies, payload)
                curator_delete_indices = payload.get("delete_indices", [])
                curator_merge_groups = payload.get("merge_groups", [])
                notes = payload.get("notes", [])
                llm_used = True
            except Exception as exc:
                curator_error = str(exc)[:500]
                logger.warning(f"Strategy curator failed; keeping strategies before budget pass: {exc}")

        if len(curated) > self._max_strategies:
            budget_delete_indices: list[int] = []
            budget_rule_delete_indices: list[int] = []
            if self._client is not None:
                try:
                    budget_delete_indices = await self._llm_delete_strategies_for_budget(
                        curated,
                        total=len(curated),
                        budget=self._max_strategies,
                    )
                except Exception as exc:
                    curator_error = (curator_error or "") + f" | budget delete fallback: {exc}"

            complete_delete_indices, budget_rule_delete_indices = (
                self._complete_strategy_budget_deletions(
                    curated,
                    budget_delete_indices,
                    self._max_strategies,
                )
            )
            delete_set = set(complete_delete_indices)
            curated = [
                strategy for idx, strategy in enumerate(curated)
                if idx not in delete_set
            ]
        else:
            budget_delete_indices = []
            budget_rule_delete_indices = []

        after = len(curated)
        changed = (
            before != after
            or self._strategy_payload_signature(strategies)
            != self._strategy_payload_signature(curated)
        )

        if changed:
            content["strategies"] = curated
            sl_path.write_text(
                yaml.dump(content, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )

        dropped_indices = self._estimate_dropped_strategy_indices(strategies, curated)
        dropped_patterns = [str(strategies[i].get("pattern", ""))[:80] for i in dropped_indices]

        logger.info(f"Strategies curated: {before} -> {after} (dropped {before - after})")
        return {
            "before": before,
            "after": after,
            "dropped": before - after,
            "dropped_indices": dropped_indices,
            "dropped_patterns": dropped_patterns[:20],
            "llm_used": llm_used,
            "curator_error": curator_error,
            "curator_delete_indices": curator_delete_indices[:50],
            "curator_merge_groups": curator_merge_groups[:20],
            "budget_delete_indices": budget_delete_indices[:50],
            "budget_rule_delete_indices": budget_rule_delete_indices[:50],
            "notes": notes[:20],
        }

    @staticmethod
    def _clip_curator_field(value: Any, limit: int = 1600) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 24].rstrip() + " ... [truncated]"

    @classmethod
    def _format_strategy_for_curator(cls, idx: int, strategy: dict) -> str:
        """Render one numbered strategy for semantic curation.

        Normal strategy entries are shown in full. The per-field cap only
        protects the curator prompt from pathological long text.
        """
        pattern = cls._clip_curator_field(strategy.get("pattern", ""))
        raw_steps = strategy.get("steps", [])
        if isinstance(raw_steps, list):
            step_lines = [
                f"   {step_idx}. {cls._clip_curator_field(step)}"
                for step_idx, step in enumerate(raw_steps)
            ]
        else:
            step_lines = [f"   0. {cls._clip_curator_field(raw_steps)}"]
        steps_text = "\n".join(step_lines) if step_lines else "   (no steps)"
        meta_lines: list[str] = []
        scope = str(strategy.get("scope", "")).strip()
        if scope:
            meta_lines.append(f"   scope: {cls._clip_curator_field(scope)}")
        for key in ("triggers", "anti_triggers"):
            raw = strategy.get(key)
            if isinstance(raw, list) and raw:
                value = ", ".join(cls._clip_curator_field(x, limit=80) for x in raw)
                meta_lines.append(f"   {key}: [{value}]")
            elif isinstance(raw, str) and raw.strip():
                meta_lines.append(f"   {key}: {cls._clip_curator_field(raw)}")
        metadata_text = ("\n" + "\n".join(meta_lines)) if meta_lines else ""
        return f"[{idx}] pattern: {pattern}{metadata_text}\n   steps:\n{steps_text}"

    def _build_strategy_curate_messages(self, strategies: list[dict]) -> list[dict[str, str]]:
        items = [
            self._format_strategy_for_curator(i, s)
            for i, s in enumerate(strategies)
        ]

        system = _STRATEGY_CURATE_SYSTEM.format(
            total=len(strategies),
            last=len(strategies) - 1,
            budget=self._max_strategies,
        )
        user_msg = "\n".join(items)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    def _build_strategy_budget_delete_messages(
        self,
        strategies: list[dict],
        *,
        required_delete: int,
    ) -> list[dict[str, str]]:
        items = [
            self._format_strategy_for_curator(i, s)
            for i, s in enumerate(strategies)
        ]

        system = _STRATEGY_BUDGET_DELETE_SYSTEM.format(
            total=len(strategies),
            last=len(strategies) - 1,
            budget=self._max_strategies,
            required_delete=required_delete,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(items)},
        ]

    async def _llm_curate_strategies(self, strategies: list[dict]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("LLM client is None")

        self._client.set_role("planning")
        conv = self._build_strategy_curate_messages(strategies)
        last_error: Exception | None = None
        raw = ""

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                raw = await self._client.chat(
                    conv, temperature=0.1, max_tokens=4096, json_mode=False,
                )
                payload = self._parse_curated_strategies(
                    raw,
                    total=len(strategies),
                    budget=self._max_strategies,
                )
                logger.info(
                    f"Strategy curator: LLM returned "
                    f"{len(payload.get('delete_indices', []))} deletions and "
                    f"{len(payload.get('merge_groups', []))} merge groups "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                return payload
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Strategy curator attempt {attempt}/{_MAX_LLM_RETRIES} failed: {exc}"
                )
                if attempt < _MAX_LLM_RETRIES:
                    conv.append({"role": "assistant", "content": raw})
                    conv.append({
                        "role": "user",
                        "content": _STRATEGY_RETRY_PROMPT.format(
                            error=str(exc)[:300],
                            budget=self._max_strategies,
                        ),
                    })
                    await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Strategy curator: all {_MAX_LLM_RETRIES} attempts failed. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_curated_strategies(
        raw: str,
        *,
        total: int,
        budget: int,
    ) -> dict[str, Any]:
        data = json.loads(_extract_json_object(raw))

        delete_raw = data.get("delete_indices", [])
        if not isinstance(delete_raw, list):
            raise ValueError("'delete_indices' must be a list")

        delete_set: set[int] = {
            idx for idx in delete_raw
            if isinstance(idx, int) and 0 <= idx < total
        }

        raw_merge_groups = data.get("merge_groups", [])
        if not isinstance(raw_merge_groups, list):
            raise ValueError("'merge_groups' must be a list")

        merge_groups: list[dict[str, Any]] = []
        protected_replacements: set[int] = set()
        for group in raw_merge_groups:
            if not isinstance(group, dict):
                continue
            replace = group.get("replace", group.get("keep"))
            drop_raw = group.get("drop", [])
            if not isinstance(replace, int) or not (0 <= replace < total):
                continue
            if not isinstance(drop_raw, list):
                continue
            drop = [
                idx for idx in drop_raw
                if isinstance(idx, int) and 0 <= idx < total and idx != replace
            ]
            if not drop:
                continue
            merged_strategy = _normalize_strategy(group.get("merged_strategy"))
            if merged_strategy is None:
                continue
            reason = str(group.get("reason", "")).strip()[:200]
            merge_groups.append({
                "replace": replace,
                "drop": drop,
                "merged_strategy": merged_strategy,
                "reason": reason,
            })
            protected_replacements.add(replace)
            delete_set.update(drop)

        delete_set.difference_update(protected_replacements)
        if total - len(delete_set) <= 0:
            raise ValueError("curator would delete every strategy")

        notes_raw = data.get("notes", data.get("merge_notes", []))
        notes = [
            str(note).strip()[:200]
            for note in notes_raw
            if str(note).strip()
        ] if isinstance(notes_raw, list) else []

        return {
            "delete_indices": sorted(delete_set),
            "merge_groups": merge_groups,
            "notes": notes,
        }

    @staticmethod
    def _apply_strategy_curator_decisions(
        strategies: list[dict[str, Any]],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        delete_indices = set(payload.get("delete_indices", []))
        merged_by_index: dict[int, dict[str, Any]] = {}
        for group in payload.get("merge_groups", []):
            if not isinstance(group, dict):
                continue
            replace = group.get("replace", group.get("keep"))
            merged_strategy = group.get("merged_strategy")
            if isinstance(replace, int) and isinstance(merged_strategy, dict):
                merged_by_index[replace] = merged_strategy

        merged: list[dict[str, Any]] = []
        for idx, strategy in enumerate(strategies):
            if idx in delete_indices:
                continue
            merged.append(merged_by_index.get(idx, strategy))
        return [
            strategy for strategy in merged
            if isinstance(strategy, dict) and strategy.get("pattern") and strategy.get("steps")
        ]

    @staticmethod
    def _strategy_payload_signature(strategies: list[dict[str, Any]]) -> list[str]:
        return [_strategy_signature(s) for s in strategies]

    def _complete_strategy_budget_deletions(
        self,
        strategies: list[dict[str, Any]],
        delete_indices: list[int],
        budget: int,
    ) -> tuple[list[int], list[int]]:
        """Cap LLM deletions and supplement them until the hard budget is met."""
        required_delete = max(0, len(strategies) - budget)
        if required_delete <= 0:
            return [], []

        complete: list[int] = []
        seen: set[int] = set()
        for idx in delete_indices:
            if isinstance(idx, int) and 0 <= idx < len(strategies) and idx not in seen:
                complete.append(idx)
                seen.add(idx)
                if len(complete) >= required_delete:
                    return complete, []

        supplemental = self._rule_strategy_budget_delete_indices(
            strategies,
            already_delete=seen,
            needed=required_delete - len(complete),
        )
        complete.extend(supplemental)
        return complete, supplemental

    @staticmethod
    def _rule_strategy_budget_delete_indices(
        strategies: list[dict[str, Any]],
        *,
        already_delete: set[int],
        needed: int,
    ) -> list[int]:
        """Delete the most redundant remaining strategies as a deterministic fallback."""
        supplemental: list[int] = []
        delete = set(already_delete)

        while len(supplemental) < needed:
            remaining = [
                idx for idx in range(len(strategies))
                if idx not in delete
            ]
            if not remaining:
                break
            drop_idx = _select_most_redundant_strategy_index(strategies, remaining)
            if drop_idx is None or drop_idx in delete:
                break
            delete.add(drop_idx)
            supplemental.append(drop_idx)

        return supplemental

    @staticmethod
    def _estimate_dropped_strategy_indices(
        original: list[dict[str, Any]],
        curated: list[dict[str, Any]],
    ) -> list[int]:
        remaining = Counter(_strategy_signature(s) for s in curated)
        dropped: list[int] = []
        for idx, strategy in enumerate(original):
            sig = _strategy_signature(strategy)
            if remaining[sig] > 0:
                remaining[sig] -= 1
            else:
                dropped.append(idx)
        return dropped

    # ------------------------------------------------------------------
    # Hook pruning
    # ------------------------------------------------------------------

    async def _prune_hooks(self) -> dict[str, dict[str, Any]]:
        hooks_dir = self._base / "hooks"
        if not hooks_dir.is_dir():
            return {}

        report: dict[str, dict[str, Any]] = {}
        entries: list[dict[str, Any]] = []
        for hf in sorted(hooks_dir.glob("*.py")):
            hp = _hook_point_from_name(hf.stem)
            if hp in _DISABLED_HOOK_POINTS:
                hf.unlink()
                report.setdefault(hp, {"before": 0, "after": 0, "dropped": 0, "dropped_names": []})
                report[hp]["before"] += 1
                report[hp]["dropped"] += 1
                report[hp]["dropped_names"].append(hf.stem)
                continue
            if hp:
                entries.append({
                    "name": hf.stem,
                    "path": hf,
                    "hook_point": hp,
                    "source": hf.read_text(encoding="utf-8"),
                    "group_key": _hook_group_key(hf.stem, hp),
                })

        for hp in _ALL_HOOK_POINTS:
            report.setdefault(hp, {
                "before": sum(1 for e in entries if e["hook_point"] == hp),
                "after": 0,
                "dropped": 0,
                "dropped_names": [],
            })

        entries = self._drop_exact_duplicate_hooks(entries, report)
        groups = _build_hook_groups(entries)
        groups_before = len(groups)

        if groups and self._client is not None:
            try:
                delete_indices = await self._llm_delete_hook_groups_with_retry(groups)
                delete_set = set(delete_indices)
                dropped_groups = [
                    group for idx, group in enumerate(groups)
                    if idx in delete_set
                ]
                self._delete_hook_groups(dropped_groups, report)
            except Exception as exc:
                logger.warning(f"Hook group curator failed; keeping hooks before budget pass: {exc}")
                for hp in _ALL_HOOK_POINTS:
                    report[hp]["curator_error"] = str(exc)[:500]

        entries = self._existing_hook_entries(entries)
        groups = _build_hook_groups(entries)
        hook_budget_delete_indices: list[int] = []
        hook_budget_rule_delete_indices: list[int] = []
        hook_budget_delete_keys: list[str] = []
        hook_budget_rule_delete_keys: list[str] = []

        if len(groups) > self._max_hook_groups:
            if self._client is not None:
                try:
                    hook_budget_delete_indices = await self._llm_delete_hook_groups_for_budget(
                        groups,
                        total=len(groups),
                        budget=self._max_hook_groups,
                    )
                except Exception as exc:
                    logger.warning(f"Hook group budget curator failed; using similarity fallback: {exc}")
                    for hp in _ALL_HOOK_POINTS:
                        report[hp]["budget_error"] = str(exc)[:500]

            complete_delete_indices, hook_budget_rule_delete_indices = (
                self._complete_hook_group_budget_deletions(
                    groups,
                    hook_budget_delete_indices,
                    self._max_hook_groups,
                )
            )
            hook_budget_delete_keys = _hook_group_keys(groups, hook_budget_delete_indices)
            hook_budget_rule_delete_keys = _hook_group_keys(groups, hook_budget_rule_delete_indices)
            delete_set = set(complete_delete_indices)
            dropped_groups = [
                group for idx, group in enumerate(groups)
                if idx in delete_set
            ]
            self._delete_hook_groups(dropped_groups, report)

        groups_after = len(_build_hook_groups(self._existing_hook_entries(entries)))
        report["_groups"] = {
            "before": groups_before,
            "after": groups_after,
            "dropped_groups": max(0, groups_before - groups_after),
            "budget": self._max_hook_groups,
            "budget_delete_groups": hook_budget_delete_indices[:50],
            "budget_rule_delete_groups": hook_budget_rule_delete_indices[:50],
            "budget_delete_group_keys": hook_budget_delete_keys[:50],
            "budget_rule_delete_group_keys": hook_budget_rule_delete_keys[:50],
        }

        for hp in _ALL_HOOK_POINTS:
            after = sum(
                1 for path in hooks_dir.glob(f"{hp}_*.py")
                if path.is_file()
            )
            report[hp]["after"] = after
            report[hp]["dropped"] = max(0, report[hp].get("before", 0) - after)
            if report[hp]["dropped_names"]:
                logger.info(
                    f"Hooks [{hp}] curated: {report[hp].get('before', '?')} -> {after} "
                    f"(dropped {report[hp]['dropped_names']})"
                )

        return report

    def _build_hook_group_messages(
        self, groups: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        items = []
        for i, group in enumerate(groups):
            hook_blocks = []
            for entry in group["entries"]:
                source = entry["source"].strip()
                hook_blocks.append(
                    f"- hook_point: {entry['hook_point']}\n"
                    f"  filename: {entry['name']}\n"
                    f"```python\n{source}\n```"
                )
            items.append(
                f"[{i}] group: {group['key']}\n"
                f"hook_count: {len(group['entries'])}\n"
                + "\n".join(hook_blocks)
            )

        system = _HOOK_GROUP_CURATE_SYSTEM.format(
            total=len(groups),
            last=len(groups) - 1,
            budget=self._max_hook_groups,
        )
        user_msg = "\n\n".join(items)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    def _build_hook_group_budget_delete_messages(
        self,
        groups: list[dict[str, Any]],
        *,
        required_delete: int,
    ) -> list[dict[str, str]]:
        items = []
        for i, group in enumerate(groups):
            hook_blocks = []
            for entry in group["entries"]:
                source = entry["source"].strip()
                hook_blocks.append(
                    f"- hook_point: {entry['hook_point']}\n"
                    f"  filename: {entry['name']}\n"
                    f"```python\n{source}\n```"
                )
            items.append(
                f"[{i}] group: {group['key']}\n"
                f"hook_count: {len(group['entries'])}\n"
                + "\n".join(hook_blocks)
            )

        system = _HOOK_GROUP_BUDGET_DELETE_SYSTEM.format(
            total=len(groups),
            last=len(groups) - 1,
            budget=self._max_hook_groups,
            required_delete=required_delete,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(items)},
        ]

    async def _llm_delete_hook_groups_with_retry(
        self,
        groups: list[dict[str, Any]],
    ) -> list[int]:
        if self._client is None:
            raise RuntimeError("LLM client is None")

        self._client.set_role("planning")
        conv = self._build_hook_group_messages(groups)
        last_error: Exception | None = None
        raw = ""

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                raw = await self._client.chat(
                    conv, temperature=0.1, max_tokens=1024, json_mode=False,
                )
                delete = self._parse_delete_list(
                    raw,
                    total=len(groups),
                    key="delete_groups",
                )
                logger.info(
                    f"Hook group curator: LLM selected {len(delete)}/{len(groups)} "
                    f"groups to delete "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                return delete
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Hook group curator attempt {attempt}/{_MAX_LLM_RETRIES} "
                    f"failed: {exc}"
                )
                if attempt < _MAX_LLM_RETRIES:
                    conv.append({"role": "assistant", "content": raw})
                    conv.append({
                        "role": "user",
                        "content": _HOOK_GROUP_RETRY_PROMPT.format(
                            error=str(exc)[:300],
                            last=len(groups) - 1,
                        ),
                    })
                    await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Hook group curator: all {_MAX_LLM_RETRIES} attempts failed. "
            f"Last error: {last_error}"
        )

    async def _llm_delete_hook_groups_for_budget(
        self,
        groups: list[dict[str, Any]],
        *,
        total: int,
        budget: int,
    ) -> list[int]:
        """Ask the LLM which hook groups to delete for the hard budget."""
        if self._client is None:
            raise RuntimeError("LLM client is None")

        required_delete = max(0, total - budget)
        if required_delete <= 0:
            return []

        self._client.set_role("planning")
        conv = self._build_hook_group_budget_delete_messages(
            groups,
            required_delete=required_delete,
        )
        last_error: Exception | None = None
        raw = ""

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                raw = await self._client.chat(
                    conv, temperature=0.1, max_tokens=1024, json_mode=False,
                )
                delete = self._parse_delete_list(
                    raw,
                    total=total,
                    key="delete_groups",
                )
                if len(delete) > required_delete:
                    delete = delete[:required_delete]
                logger.info(
                    f"Hook group budget curator: LLM selected {len(delete)}/"
                    f"{required_delete} deletions "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                return delete
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Hook group budget curator attempt {attempt}/{_MAX_LLM_RETRIES} "
                    f"failed: {exc}"
                )
                if attempt < _MAX_LLM_RETRIES:
                    conv.append({"role": "assistant", "content": raw})
                    conv.append({
                        "role": "user",
                        "content": _HOOK_GROUP_BUDGET_DELETE_RETRY_PROMPT.format(
                            error=str(exc)[:300],
                            last=len(groups) - 1,
                            required_delete=required_delete,
                        ),
                    })
                    await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Hook group budget curator: all {_MAX_LLM_RETRIES} attempts failed. "
            f"Last error: {last_error}"
        )

    def _drop_exact_duplicate_hooks(
        self,
        entries: list[dict[str, Any]],
        report: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        kept: list[dict[str, Any]] = []
        for entry in entries:
            sig = (entry["hook_point"], _normalize_hook_source(entry["source"]))
            if sig in seen:
                self._delete_hook_entry(entry, report)
                report[entry["hook_point"]].setdefault("exact_duplicate_names", []).append(
                    entry["name"]
                )
                continue
            seen[sig] = entry
            kept.append(entry)
        return kept

    @staticmethod
    def _existing_hook_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [entry for entry in entries if entry["path"].exists()]

    def _complete_hook_group_budget_deletions(
        self,
        groups: list[dict[str, Any]],
        delete_indices: list[int],
        budget: int,
    ) -> tuple[list[int], list[int]]:
        """Cap LLM deletions and supplement them until the hard budget is met."""
        required_delete = max(0, len(groups) - budget)
        if required_delete <= 0:
            return [], []

        complete: list[int] = []
        seen: set[int] = set()
        for idx in delete_indices:
            if isinstance(idx, int) and 0 <= idx < len(groups) and idx not in seen:
                complete.append(idx)
                seen.add(idx)
                if len(complete) >= required_delete:
                    return complete, []

        supplemental = self._rule_hook_group_budget_delete_indices(
            groups,
            already_delete=seen,
            needed=required_delete - len(complete),
        )
        complete.extend(supplemental)
        return complete, supplemental

    @staticmethod
    def _rule_hook_group_budget_delete_indices(
        groups: list[dict[str, Any]],
        *,
        already_delete: set[int],
        needed: int,
    ) -> list[int]:
        """Delete the most redundant hook groups as a deterministic fallback."""
        supplemental: list[int] = []
        delete = set(already_delete)

        while len(supplemental) < needed:
            remaining = [
                idx for idx in range(len(groups))
                if idx not in delete
            ]
            if not remaining:
                break
            drop_idx = _select_most_redundant_hook_group_index(groups, remaining)
            if drop_idx is None or drop_idx in delete:
                break
            delete.add(drop_idx)
            supplemental.append(drop_idx)

        return supplemental

    def _delete_hook_groups(
        self,
        groups: list[dict[str, Any]],
        report: dict[str, dict[str, Any]],
    ) -> None:
        for group in groups:
            for entry in group["entries"]:
                self._delete_hook_entry(entry, report)

    @staticmethod
    def _delete_hook_entry(
        entry: dict[str, Any],
        report: dict[str, dict[str, Any]],
    ) -> None:
        path = entry["path"]
        if path.exists():
            path.unlink()
        hp = entry["hook_point"]
        report.setdefault(hp, {"before": 0, "after": 0, "dropped": 0, "dropped_names": []})
        report[hp].setdefault("dropped_names", []).append(entry["name"])

    # ------------------------------------------------------------------
    # Robust multi-round LLM selection
    # ------------------------------------------------------------------

    async def _llm_delete_strategies_for_budget(
        self,
        strategies: list[dict[str, Any]],
        *,
        total: int,
        budget: int,
    ) -> list[int]:
        """Ask the LLM which strategy indices to delete for the hard budget."""
        if self._client is None:
            raise RuntimeError("LLM client is None")

        required_delete = max(0, total - budget)
        if required_delete <= 0:
            return []

        self._client.set_role("planning")
        conv = self._build_strategy_budget_delete_messages(
            strategies,
            required_delete=required_delete,
        )
        last_error: Exception | None = None
        raw = ""

        for attempt in range(1, _MAX_LLM_RETRIES + 1):
            try:
                raw = await self._client.chat(
                    conv, temperature=0.1, max_tokens=1024, json_mode=False,
                )
                delete = self._parse_delete_list(
                    raw,
                    total=total,
                    key="delete_indices",
                )
                if len(delete) > required_delete:
                    delete = delete[:required_delete]
                logger.info(
                    f"Strategy budget curator: LLM selected {len(delete)}/"
                    f"{required_delete} deletions "
                    f"(attempt {attempt}/{_MAX_LLM_RETRIES})"
                )
                return delete

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Strategy budget curator attempt {attempt}/{_MAX_LLM_RETRIES} "
                    f"failed: {exc}"
                )
                if attempt < _MAX_LLM_RETRIES:
                    conv.append({"role": "assistant", "content": raw})
                    conv.append({
                        "role": "user",
                        "content": _BUDGET_DELETE_RETRY_PROMPT.format(
                            error=str(exc)[:300],
                            last=total - 1,
                            required_delete=required_delete,
                        ),
                    })
                    await asyncio.sleep(1.0 * attempt)

        raise RuntimeError(
            f"Strategy budget curator: all {_MAX_LLM_RETRIES} attempts failed. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def _parse_delete_list(
        raw: str,
        *,
        total: int,
        key: str,
    ) -> list[int]:
        """Extract and validate a delete-list from raw LLM output.

        Unlike keep-lists, an empty delete-list is valid: it means the curator
        found no clear duplicate or harmful hook group.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        try:
            data = json.loads(_extract_json_object(cleaned))
        except Exception:
            list_match = re.search(r'\[[\d\s,]+\]', cleaned)
            if list_match:
                data = {key: json.loads(list_match.group())}
            else:
                raise ValueError(f"No JSON or number list found in response: {cleaned[:200]}")

        delete = data.get(key, data.get("delete", []))
        if not isinstance(delete, list):
            raise ValueError(f"'{key}' is not a list: {type(delete)}")

        valid: list[int] = []
        seen: set[int] = set()
        for item in delete:
            if isinstance(item, int) and 0 <= item < total and item not in seen:
                valid.append(item)
                seen.add(item)

        return valid


def _hook_point_from_name(stem: str) -> str | None:
    for hp in _DISABLED_HOOK_POINTS:
        if stem == hp or stem.startswith(f"{hp}_"):
            return hp
    for hp in _ALL_HOOK_POINTS:
        if stem == hp or stem.startswith(f"{hp}_"):
            return hp
    return None


def _extract_json_object(raw: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response: {cleaned[:200]}")

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start: idx + 1]

    raise ValueError(f"Unclosed JSON object in response: {cleaned[:200]}")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _strategy_signature(strategy: dict[str, Any]) -> str:
    pattern = _normalize_text(strategy.get("pattern", ""))
    steps = " | ".join(_normalize_text(step) for step in strategy.get("steps", []))
    return f"{pattern}\n{steps}"


def _strategy_tokens(strategy: dict[str, Any]) -> set[str]:
    text = _strategy_signature(strategy)
    return {
        token for token in re.findall(r"[a-z0-9_./-]+", text)
        if len(token) > 2
    }


def _strategy_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if _strategy_signature(left) == _strategy_signature(right):
        return 1.0
    left_tokens = _strategy_tokens(left)
    right_tokens = _strategy_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _strategy_quality_score(strategy: dict[str, Any]) -> float:
    pattern = str(strategy.get("pattern", "")).strip()
    steps_raw = strategy.get("steps", [])
    steps = [
        str(step).strip() for step in steps_raw
        if str(step).strip()
    ] if isinstance(steps_raw, list) else []
    total_chars = len(pattern) + sum(len(step) for step in steps)
    return min(len(steps), 6) * 10.0 + min(total_chars, 1000) / 100.0


def _select_most_redundant_strategy_index(
    strategies: list[dict[str, Any]],
    remaining_indices: list[int],
) -> int | None:
    if not remaining_indices:
        return None
    if len(remaining_indices) == 1:
        return remaining_indices[0]

    quality = {
        idx: _strategy_quality_score(strategies[idx])
        for idx in remaining_indices
    }
    best_rank: tuple[float, float, int] | None = None
    best_drop: int | None = None

    for pos, left_idx in enumerate(remaining_indices):
        for right_idx in remaining_indices[:pos]:
            sim = _strategy_similarity(strategies[left_idx], strategies[right_idx])
            if quality[left_idx] < quality[right_idx]:
                drop_idx = left_idx
            elif quality[right_idx] < quality[left_idx]:
                drop_idx = right_idx
            else:
                drop_idx = max(left_idx, right_idx)
            rank = (sim, -quality[drop_idx], drop_idx)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_drop = drop_idx

    if best_drop is not None:
        return best_drop

    return min(remaining_indices, key=lambda idx: (quality[idx], -idx))


def _normalize_strategy(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    pattern = str(value.get("pattern", "")).strip()
    steps_raw = value.get("steps", [])
    if not pattern or not isinstance(steps_raw, list):
        return None
    steps = [str(step).strip() for step in steps_raw if str(step).strip()]
    if not steps:
        return None
    normalized: dict[str, Any] = {
        "pattern": pattern[:500],
        "steps": steps[:6],
    }
    scope = str(value.get("scope", "")).lower().strip()
    if _strategy_has_hardcoded_literals(pattern, steps):
        normalized["scope"] = "task_specific"
    elif scope in {"core", "general", "task_specific"}:
        normalized["scope"] = scope
    triggers = _normalize_strategy_terms(value.get("triggers"))
    if triggers:
        normalized["triggers"] = triggers
    anti_triggers = _normalize_strategy_terms(value.get("anti_triggers"))
    if anti_triggers:
        normalized["anti_triggers"] = anti_triggers
    return normalized


def _strategy_has_hardcoded_literals(pattern: str, steps: list[str]) -> bool:
    """Detect concrete task literals that should not remain core/general."""
    text = "\n".join([pattern, *steps])
    lowered = text.lower()
    if re.search(r"/(?:home|tmp|app|workspace|var|etc|opt)/[^\s'\"`]+", text):
        return True
    if re.search(r"(?:^|[\s'\"`])~/[^\s'\"`]+", text):
        return True
    if re.search(r"\b(?:localhost|127\.0\.0\.1):\d{2,5}\b", lowered):
        return True
    if re.search(r"\bport\s+\d{2,5}\b|\b\d{2,5}\s+port\b", lowered):
        return True
    if re.search(
        r"(?:^|[/\s'\"`])(?:[\w.-]+\.)+"
        r"(?:tar\.gz|json|csv|log|conf|cfg|txt|sh|py|pub|yaml|yml|toml|env)\b",
        lowered,
    ):
        return True
    if re.search(r"(?:^|[/\s'\"`])\.[a-z0-9_-]+\b", lowered):
        return True
    if re.search(r"\b(?:tz|lang)=[a-z0-9_./-]+\b", lowered):
        return True
    return False


def _normalize_strategy_terms(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_terms = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw_terms = [str(part).strip() for part in value]
    else:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        term = term.lower().strip()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term[:80])
        if len(terms) >= 12:
            break
    return terms


def _hook_group_key(stem: str, hook_point: str) -> str:
    prefix = f"{hook_point}_"
    rest = stem[len(prefix):] if stem.startswith(prefix) else stem
    rest = re.sub(r"_[0-9a-f]{8}$", "", rest)
    return rest or stem


def _build_hook_groups(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        raw_groups.setdefault(entry["group_key"], []).append(entry)

    groups: list[dict[str, Any]] = []
    for key in sorted(raw_groups):
        group_entries = raw_groups[key]
        counts = Counter(entry["hook_point"] for entry in group_entries)
        if any(count > 1 for count in counts.values()):
            # Ambiguous legacy filenames can produce many hooks with the same
            # group tag and hook point. Split those into independent units so
            # duplicated after_round hooks can be removed safely.
            for entry in group_entries:
                groups.append({"key": f"{key}/{entry['name']}", "entries": [entry]})
        else:
            groups.append({"key": key, "entries": group_entries})
    return groups


def _hook_group_keys(groups: list[dict[str, Any]], indices: list[int]) -> list[str]:
    keys: list[str] = []
    seen: set[int] = set()
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(groups) and idx not in seen:
            seen.add(idx)
            keys.append(str(groups[idx].get("key", "")))
    return keys


def _hook_group_text(group: dict[str, Any]) -> str:
    parts = [str(group.get("key", ""))]
    for entry in group.get("entries", []):
        parts.append(str(entry.get("hook_point", "")))
        parts.append(str(entry.get("name", "")))
        parts.append(_normalize_hook_source(str(entry.get("source", ""))))
    return "\n".join(parts)


def _hook_group_tokens(group: dict[str, Any]) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9_./-]+", _hook_group_text(group).lower())
        if len(token) > 2
    }


def _hook_group_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_text = _hook_group_text(left)
    right_text = _hook_group_text(right)
    if left_text == right_text:
        return 1.0
    left_tokens = _hook_group_tokens(left)
    right_tokens = _hook_group_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _hook_group_quality_score(group: dict[str, Any]) -> float:
    entries = group.get("entries", [])
    hook_points = {
        str(entry.get("hook_point", ""))
        for entry in entries
    }
    source_text = "\n".join(str(entry.get("source", "")) for entry in entries)
    score = len(hook_points) * 12.0 + len(entries) * 6.0
    score += min(len(source_text), 2400) / 200.0
    if "context.kv" in source_text:
        score += 4.0
    if "setdefault" in source_text or ".get(" in source_text:
        score += 2.0
    return score


def _select_most_redundant_hook_group_index(
    groups: list[dict[str, Any]],
    remaining_indices: list[int],
) -> int | None:
    if not remaining_indices:
        return None
    if len(remaining_indices) == 1:
        return remaining_indices[0]

    quality = {
        idx: _hook_group_quality_score(groups[idx])
        for idx in remaining_indices
    }
    best_rank: tuple[float, float, int] | None = None
    best_drop: int | None = None

    for pos, left_idx in enumerate(remaining_indices):
        for right_idx in remaining_indices[:pos]:
            sim = _hook_group_similarity(groups[left_idx], groups[right_idx])
            if quality[left_idx] < quality[right_idx]:
                drop_idx = left_idx
            elif quality[right_idx] < quality[left_idx]:
                drop_idx = right_idx
            else:
                drop_idx = max(left_idx, right_idx)
            rank = (sim, -quality[drop_idx], drop_idx)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_drop = drop_idx

    if best_drop is not None:
        return best_drop

    return min(remaining_indices, key=lambda idx: (quality[idx], -idx))


def _normalize_hook_source(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return "\n".join(lines)
