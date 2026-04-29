"""Interactive multi-turn diagnoser with per-task parallel workers.

Each sampled task gets its own LLM worker that performs deep multi-turn
analysis. Tasks are sampled from three categories:
  - partial  (highest diagnostic value — contrastive analysis)
  - all_fail (root cause investigation)
  - all_pass (brief regression context)

Only partial-task workers are required to produce ``strategy_suggestions``.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import re
from typing import Any

from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import DiagnosisHistory
from skyrl_agent.meta_toolkit.diagnosis.diagnosis_tools import (
    TOOL_DESCRIPTIONS,
    TaskGroup,
    build_task_grouped_message,
    classify_task_groups,
    dispatch_tool,
    group_traces_by_task,
)
from skyrl_agent.meta_toolkit.diagnosis.rule_based_diagnoser import (
    DiagnosisResult,
    RuleBasedDiagnoser,
)
from skyrl_agent.meta_toolkit.llm_client import (
    MetaLLMClient,
    _extract_json as _extract_any_json,
)
from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 16
_DEFAULT_NUM_WORKERS = 16

_DEFAULT_PARTIAL_TASKS = 8
_DEFAULT_ALL_FAIL_TASKS = 8
_DEFAULT_ALL_PASS_TASKS = 0

# Keep diagnosis outputs compact and reserve budget for prompt/context.
_DIAGNOSIS_MAX_OUTPUT_TOKENS = 1024
_DIAGNOSIS_MODEL_CONTEXT_TOKENS = 32768
_DIAGNOSIS_CONTEXT_SAFETY_MARGIN = 512
_DIAGNOSIS_HISTORY_KEEP_RECENT_MESSAGES = 6
_DIAGNOSIS_COMPRESSED_HISTORY_MAX_CHARS = 4000
_DIAGNOSIS_MIN_TRUNCATE_CHARS = 600
_DIAGNOSIS_MAX_TRUNCATE_ROUNDS = 20

_SYSTEM_PROMPT = """\
You are an expert AI-agent performance analyst conducting an interactive
diagnosis session. You will examine execution traces from a terminal-based
AI agent and identify failure patterns.

## Workflow

1. You will receive one focused task-group assignment.
2. Use the diagnostic tools to investigate specific traces.
3. Compare successful and failed traces to find patterns when available.
4. Check history to avoid repeating past failed patches.
5. When ready, submit your diagnosis. The downstream planner will decide
   whether each diagnosis should become a strategy edit or a runtime hook.

{tool_descriptions}

{diagnosis_format}
"""

_DIAGNOSIS_FORMAT_WITH_STRATEGIES = """\
## Diagnosis format (for submit_diagnosis)

```json
{"tool": "submit_diagnosis", "diagnoses": [
  {
    "problem_type": "<string>",
    "root_cause_hypotheses": ["<specific evidence-based hypothesis>"],
    "confidence": <0.0-1.0>,
    "affected_trace_ids": ["<trajectory_id, e.g. 749-traj3>"],
    "strategy_suggestions": [
      {
        "pattern": "<when to apply this strategy — a recognizable situation>",
        "steps": ["<step 1>", "<step 2>", "..."],
        "source": "<evidence: e.g. 'task 749-traj3 succeeded by doing X'>"
      }
    ]
  }
]}
```

`strategy_suggestions` is optional but highly valuable. When you compare
successful vs failed traces on the SAME task, extract what the successful
trace did differently as a reusable strategy. Each strategy should have:
- `pattern`: a short description of the situation where this strategy applies
- `steps`: concrete step-by-step actions the agent should take
- `source`: which trace/task provided the evidence

Be specific — reference concrete trajectory IDs, turn counts, and tool patterns.
Do NOT just say "context_overload" without evidence of what causes it.
"""

_DIAGNOSIS_FORMAT_DIAGNOSIS_ONLY = """\
## Diagnosis format (for submit_diagnosis)

```json
{"tool": "submit_diagnosis", "diagnoses": [
  {
    "problem_type": "<string>",
    "root_cause_hypotheses": ["<specific evidence-based hypothesis>"],
    "confidence": <0.0-1.0>,
    "affected_trace_ids": ["<trajectory_id, e.g. 749-traj3>"]
  }
]}
```

For diagnosis-only workers such as all-fail or all-pass tasks, do NOT include
`strategy_suggestions`; there is no contrastive successful-vs-failed pair to
imitate. The downstream planner can still decide whether the diagnosis calls
for a hook or a strategy.

Be specific — reference concrete trajectory IDs, turn counts, commands, and
observed failure modes. Do NOT just say "context_overload" without evidence.
"""

_FORCE_SUBMIT_MSG = (
    "You have 1 turn remaining. You MUST call submit_diagnosis NOW with your "
    "findings so far. Summarize what you have observed and submit a diagnosis, "
    "even if your investigation is incomplete."
)


def _build_system_prompt(task_category: str) -> str:
    diagnosis_format = (
        _DIAGNOSIS_FORMAT_WITH_STRATEGIES
        if task_category == "partial"
        else _DIAGNOSIS_FORMAT_DIAGNOSIS_ONLY
    )
    return (
        _SYSTEM_PROMPT
        .replace("{tool_descriptions}", TOOL_DESCRIPTIONS)
        .replace("{diagnosis_format}", diagnosis_format)
    )


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a JSON tool call from LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    search_text = cleaned or text

    for source in (search_text, text):
        try:
            obj = _extract_any_json(source)
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except ValueError:
            pass

    for pattern in [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
        r"(\{[^{}]*\"tool\"[^{}]*\})",
    ]:
        m = re.search(pattern, search_text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and "tool" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

    m = re.search(r'(\{"tool"\s*:\s*"submit_diagnosis".*\})', search_text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _extract_observations_from_history(messages: list[dict[str, str]]) -> list[str]:
    """Extract useful observations from the conversation history for fallback diagnosis."""
    observations: list[str] = []
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if content.startswith("Tool result:"):
            result_text = content[len("Tool result:"):].strip()
            if "success:" in result_text and "reward:" in result_text:
                first_line = result_text.split("\n")[0]
                observations.append(first_line)
    return observations


def _build_analysis_summary(messages: list[dict[str, str]], max_chars: int = 800) -> str:
    """Extract a concise analysis summary from the worker's conversation history."""
    tool_findings: list[str] = []
    for msg in messages:
        if msg["role"] == "user" and msg["content"].startswith("Tool result:"):
            text = msg["content"][len("Tool result:"):].strip()
            first_lines = "\n".join(text.split("\n")[:3])
            tool_findings.append(first_lines)

    last_assistant = ""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            cleaned = re.sub(r"<think>.*?</think>", "", msg["content"], flags=re.DOTALL).strip()
            cleaned = re.sub(r"```json.*?```", "", cleaned, flags=re.DOTALL).strip()
            if cleaned:
                last_assistant = cleaned[:400]
                break

    parts: list[str] = []
    if tool_findings:
        parts.append("Inspected traces: " + "; ".join(tool_findings[:5]))
    if last_assistant:
        parts.append("LLM reasoning: " + last_assistant)

    summary = "\n".join(parts)
    return summary[:max_chars] if summary else ""


def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate for prompt-size guardrails."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        # Simple heuristic: 1 token ~= 4 chars + per-message framing overhead.
        total += max(1, len(content) // 4) + 8
    return total


def _summarize_for_compression(messages: list[dict[str, str]], max_chars: int) -> str:
    """Compress older turns into a compact textual summary."""
    if not messages:
        return ""

    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        if not content:
            continue

        if role == "assistant":
            tool_call = _extract_tool_call(content)
            if tool_call is not None:
                tool_name = str(tool_call.get("tool", "unknown_tool"))
                lines.append(f"assistant_tool: {tool_name}")
            else:
                compact = re.sub(r"\s+", " ", content)
                lines.append(f"assistant_note: {compact[:180]}")
        elif role == "user" and content.startswith("Tool result:"):
            result_text = content[len("Tool result:"):].strip()
            first_lines = " ".join(result_text.splitlines()[:3])
            compact = re.sub(r"\s+", " ", first_lines)
            lines.append(f"tool_result: {compact[:220]}")
        else:
            compact = re.sub(r"\s+", " ", content)
            lines.append(f"{role or 'user'}: {compact[:180]}")

        if len("\n".join(lines)) >= max_chars:
            break

    if not lines:
        return ""

    summary = "\n".join(f"- {line}" for line in lines)
    return summary[:max_chars]


def _truncate_middle(text: str, max_chars: int) -> str:
    """Truncate very long text while preserving both beginning and ending."""
    if len(text) <= max_chars:
        return text
    if max_chars <= _DIAGNOSIS_MIN_TRUNCATE_CHARS:
        return text[:max_chars]

    marker = "\n...[context trimmed for token budget]...\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return text[:max_chars]
    head = remaining // 2
    tail = remaining - head
    return text[:head] + marker + text[-tail:]


def _enforce_budget_with_truncation(
    messages: list[dict[str, str]],
    *,
    input_budget: int,
) -> list[dict[str, str]]:
    """Emergency guardrail: trim oversized message contents until budget fits."""
    adjusted = [dict(m) for m in messages]
    rounds = 0
    changed = 0

    while _estimate_prompt_tokens(adjusted) > input_budget and rounds < _DIAGNOSIS_MAX_TRUNCATE_ROUNDS:
        rounds += 1
        # Preserve the system prompt; trim the largest remaining message first.
        idx = -1
        max_len = 0
        for i in range(1, len(adjusted)):
            msg_len = len(adjusted[i].get("content", ""))
            if msg_len > max_len:
                max_len = msg_len
                idx = i

        if idx < 0 or max_len <= _DIAGNOSIS_MIN_TRUNCATE_CHARS:
            break

        target_chars = max(_DIAGNOSIS_MIN_TRUNCATE_CHARS, int(max_len * 0.7))
        old_content = adjusted[idx].get("content", "")
        new_content = _truncate_middle(old_content, target_chars)
        if new_content == old_content:
            break
        adjusted[idx]["content"] = new_content
        changed += 1

    if changed:
        logger.warning(
            "Diagnosis context emergency truncation applied: "
            f"messages_trimmed={changed}, "
            f"estimated_tokens_after={_estimate_prompt_tokens(adjusted)}, "
            f"budget={input_budget}"
        )
    return adjusted


def _compress_history_if_needed(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Compress older conversational turns when prompt budget gets tight.

    Preserves:
      - system prompt
      - initial assignment user message
      - recent turns
    Replaces older turns with one summary message.
    """
    if len(messages) <= 2:
        return messages

    input_budget = (
        _DIAGNOSIS_MODEL_CONTEXT_TOKENS
        - _DIAGNOSIS_MAX_OUTPUT_TOKENS
        - _DIAGNOSIS_CONTEXT_SAFETY_MARGIN
    )
    estimated = _estimate_prompt_tokens(messages)
    if estimated <= input_budget:
        return messages

    keep_recent = min(_DIAGNOSIS_HISTORY_KEEP_RECENT_MESSAGES, max(1, len(messages) - 2))
    if len(messages) <= 2 + keep_recent:
        return messages

    head = messages[:2]
    middle = messages[2:-keep_recent]
    tail = messages[-keep_recent:]
    summary = _summarize_for_compression(
        middle,
        max_chars=_DIAGNOSIS_COMPRESSED_HISTORY_MAX_CHARS,
    )
    if not summary:
        return messages

    compressed_messages = head + [
        {
            "role": "user",
            "content": (
                "Compressed prior diagnosis turns for context continuity.\n"
                "Use this summary as historical context and continue from recent turns:\n"
                f"{summary}"
            ),
        }
    ] + tail

    logger.info(
        "Diagnosis worker history compressed: "
        f"estimated_tokens_before={estimated}, "
        f"estimated_tokens_after={_estimate_prompt_tokens(compressed_messages)}"
    )
    if _estimate_prompt_tokens(compressed_messages) <= input_budget:
        return compressed_messages

    # Emergency fallback for pathological oversized tool returns.
    return _enforce_budget_with_truncation(
        compressed_messages,
        input_budget=input_budget,
    )


class InteractiveDiagnoser:
    """Per-task parallel multi-turn interactive diagnoser.

    Samples individual tasks from the partial / all-fail / all-pass
    buckets and launches a dedicated LLM worker for each task.
    Results are merged into a unified diagnosis list.
    """

    def __init__(
        self,
        client: MetaLLMClient,
        history: DiagnosisHistory | None = None,
        max_turns: int = _DEFAULT_MAX_TURNS,
        num_workers: int = _DEFAULT_NUM_WORKERS,
        max_partial_tasks: int = _DEFAULT_PARTIAL_TASKS,
        max_all_fail_tasks: int = _DEFAULT_ALL_FAIL_TASKS,
        max_all_pass_tasks: int = _DEFAULT_ALL_PASS_TASKS,
        override_base: str | None = None,
    ) -> None:
        self._client = client
        self._history = history or DiagnosisHistory()
        self._max_turns = max_turns
        self._num_workers = num_workers
        self._max_partial_tasks = max(0, int(max_partial_tasks))
        self._max_all_fail_tasks = max(0, int(max_all_fail_tasks))
        self._max_all_pass_tasks = max(0, int(max_all_pass_tasks))
        self._override_base = override_base
        self._fallback = RuleBasedDiagnoser()
        self._last_worker_details: list[dict[str, Any]] = []

    @property
    def last_worker_details(self) -> list[dict[str, Any]]:
        """Per-worker metadata from the most recent ``diagnose`` call (for logging)."""
        return self._last_worker_details

    async def diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        if not traces:
            return []

        try:
            return await self._parallel_diagnose(traces)
        except Exception as exc:
            logger.warning(f"Parallel diagnosis failed ({exc}); falling back to rules")
            return self._fallback.diagnose(traces)

    async def _parallel_diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        """Sample individual tasks and run one LLM worker per task."""
        groups = group_traces_by_task(traces)
        all_fail, partial, all_pass = classify_task_groups(groups)

        worker_configs: list[tuple[str, TaskGroup, str]] = []
        selected_keys: set[str] = set()
        selected_partial = 0
        selected_all_fail = 0
        selected_all_pass = 0

        def _add_worker(
            group: TaskGroup,
            category: str,
            *,
            respect_limit: bool,
        ) -> bool:
            nonlocal selected_partial, selected_all_fail, selected_all_pass
            if len(worker_configs) >= self._num_workers:
                return False
            if group.task_key in selected_keys:
                return False

            if respect_limit:
                if category == "partial" and selected_partial >= self._max_partial_tasks:
                    return False
                if category == "all_fail" and selected_all_fail >= self._max_all_fail_tasks:
                    return False
                if category == "all_pass" and selected_all_pass >= self._max_all_pass_tasks:
                    return False

            if category == "partial":
                name = f"partial_{group.task_key}"
                selected_partial += 1
            elif category == "all_fail":
                name = f"allfail_{group.task_key}"
                selected_all_fail += 1
            else:
                name = f"allpass_{group.task_key}"
                selected_all_pass += 1

            worker_configs.append((name, group, category))
            selected_keys.add(group.task_key)
            return True

        # First pass: respect configured caps for each category.
        for g in partial:
            if len(worker_configs) >= self._num_workers:
                break
            _add_worker(g, "partial", respect_limit=True)

        for g in all_fail:
            if len(worker_configs) >= self._num_workers:
                break
            _add_worker(g, "all_fail", respect_limit=True)

        for g in all_pass:
            if len(worker_configs) >= self._num_workers:
                break
            _add_worker(g, "all_pass", respect_limit=True)

        # Second pass: if one failure bucket is short, let the other failure bucket
        # fill the gap (can exceed its configured cap). This avoids forcing all-pass
        # tasks into diagnosis when failure tasks are still available.
        for pool, category in ((partial, "partial"), (all_fail, "all_fail")):
            if len(worker_configs) >= self._num_workers:
                break
            for g in pool:
                if len(worker_configs) >= self._num_workers:
                    break
                _add_worker(g, category, respect_limit=False)

        # Third pass (optional): allow all-pass backfill only within its explicit cap.
        if self._max_all_pass_tasks > selected_all_pass and len(worker_configs) < self._num_workers:
            for g in all_pass:
                if len(worker_configs) >= self._num_workers:
                    break
                _add_worker(g, "all_pass", respect_limit=True)

        if not worker_configs:
            if all_fail:
                worker_configs.append((f"allfail_{all_fail[0].task_key}", all_fail[0], "all_fail"))
            elif partial:
                worker_configs.append((f"partial_{partial[0].task_key}", partial[0], "partial"))
            elif all_pass:
                worker_configs.append((f"allpass_{all_pass[0].task_key}", all_pass[0], "all_pass"))
            else:
                worker_configs.append(("fallback_all", list(groups.values())[0], "all_fail"))

        worker_configs = worker_configs[:self._num_workers]

        logger.info(
            "Diagnosis sampling: "
            f"available(partial={len(partial)}, all_fail={len(all_fail)}, all_pass={len(all_pass)}), "
            f"selected(partial={sum(1 for _, _, c in worker_configs if c == 'partial')}, "
            f"all_fail={sum(1 for _, _, c in worker_configs if c == 'all_fail')}, "
            f"all_pass={sum(1 for _, _, c in worker_configs if c == 'all_pass')})"
        )
        logger.info(
            f"Launching {len(worker_configs)} per-task diagnosis workers: "
            f"{[(name, cat) for name, _, cat in worker_configs]}"
        )

        tasks = [
            self._run_single_worker(
                traces, target_task=group, task_category=category, worker_name=name,
            )
            for name, group, category in worker_configs
        ]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        all_diagnoses: list[DiagnosisResult] = []
        self._last_worker_details = []
        for i, result in enumerate(worker_results):
            name, group, category = worker_configs[i]
            detail: dict[str, Any] = {
                "worker_name": name,
                "task_key": group.task_key,
                "category": category,
            }
            if isinstance(result, Exception):
                logger.warning(f"Worker '{name}' failed: {result}")
                detail["status"] = "error"
                detail["error"] = str(result)
            else:
                diagnoses, turns_used = result
                logger.info(f"Worker '{name}' produced {len(diagnoses)} diagnoses in {turns_used} turns")
                all_diagnoses.extend(diagnoses)
                n_suggestions = sum(len(d.strategy_suggestions) for d in diagnoses)
                detail["status"] = "ok"
                detail["n_diagnoses"] = len(diagnoses)
                detail["n_strategy_suggestions"] = n_suggestions
                detail["turns_used"] = turns_used
            self._last_worker_details.append(detail)

        if not all_diagnoses:
            logger.warning("All workers produced empty results; using rule fallback")
            return self._fallback.diagnose(traces)

        merged = self._merge_diagnoses(all_diagnoses)
        logger.info(f"Merged diagnosis: {len(merged)} problem types")
        return merged

    async def _run_single_worker(
        self,
        traces: list[TraceRecord],
        *,
        target_task: TaskGroup,
        task_category: str,
        worker_name: str,
    ) -> tuple[list[DiagnosisResult], int]:
        """Run a single per-task diagnosis worker session.

        Returns (diagnoses, turns_used).
        """
        self._client.set_role("diagnosis")

        system_prompt = _build_system_prompt(task_category)
        user_msg = build_task_grouped_message(
            traces, self._history,
            target_task=target_task,
            task_category=task_category,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        turns_used = 0
        for turn in range(self._max_turns):
            turns_used = turn + 1
            is_penultimate = turn == self._max_turns - 2
            is_last = turn == self._max_turns - 1

            messages = _compress_history_if_needed(messages)
            response = await self._client.chat(
                messages,
                temperature=0.2,
                max_tokens=_DIAGNOSIS_MAX_OUTPUT_TOKENS,
            )
            messages.append({"role": "assistant", "content": response})

            tool_call = _extract_tool_call(response)

            if tool_call is not None and tool_call.get("tool") == "submit_diagnosis":
                diagnoses_raw = tool_call.get("diagnoses", [])
                result = self._parse_diagnoses(
                    diagnoses_raw, task_category=task_category,
                )
                summary = _build_analysis_summary(messages)
                for d in result:
                    d.metadata["analysis_summary"] = summary
                    d.metadata["worker_name"] = worker_name
                    d.metadata["task_category"] = task_category
                logger.info(f"Worker '{worker_name}' submitted diagnosis at turn {turn+1}")
                return result, turns_used

            if tool_call is not None:
                result = dispatch_tool(tool_call, traces, self._history, self._override_base)
                if result == "__SUBMIT__":
                    diagnoses_raw = tool_call.get("diagnoses", [])
                    parsed = self._parse_diagnoses(
                        diagnoses_raw, task_category=task_category,
                    )
                    summary = _build_analysis_summary(messages)
                    for d in parsed:
                        d.metadata["analysis_summary"] = summary
                        d.metadata["worker_name"] = worker_name
                        d.metadata["task_category"] = task_category
                    return parsed, turns_used

                next_msg = f"Tool result:\n{result}\n\n"
                if is_penultimate:
                    next_msg += _FORCE_SUBMIT_MSG
                else:
                    next_msg += "Continue investigation or submit_diagnosis."
                messages.append({"role": "user", "content": next_msg})
            else:
                if is_last:
                    break
                if is_penultimate:
                    messages.append({"role": "user", "content": _FORCE_SUBMIT_MSG})
                else:
                    messages.append({
                        "role": "user",
                        "content": "Please call a diagnostic tool or submit your diagnosis as JSON.",
                    })

        logger.warning(
            f"Worker '{worker_name}' hit turn limit; "
            "synthesizing diagnosis from conversation"
        )
        return self._synthesize_from_conversation(
            messages,
            traces,
            worker_name,
            task_category,
        ), turns_used

    def _synthesize_from_conversation(
        self,
        messages: list[dict[str, str]],
        traces: list[TraceRecord],
        worker_name: str,
        task_category: str,
    ) -> list[DiagnosisResult]:
        """Build a diagnosis from what the LLM observed even without submit_diagnosis."""
        observations = _extract_observations_from_history(messages)

        last_assistant = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        tool_call = _extract_tool_call(last_assistant)
        if tool_call and tool_call.get("tool") == "submit_diagnosis":
            diagnoses_raw = tool_call.get("diagnoses", [])
            result = self._parse_diagnoses(
                diagnoses_raw, task_category=task_category,
            )
            summary = _build_analysis_summary(messages)
            for d in result:
                d.metadata["analysis_summary"] = summary
                d.metadata["worker_name"] = worker_name
                d.metadata["task_category"] = task_category
            if result and result[0].problem_type != "general":
                return result

        fail_traces = [t for t in traces if not t.success]
        ctx_traces = [t for t in traces if t.finish_reason == "context_length"]

        hypotheses: list[str] = []
        if observations:
            hypotheses.append(
                f"Worker '{worker_name}' inspected {len(observations)} traces "
                f"but did not submit a formal diagnosis"
            )
        if ctx_traces:
            hypotheses.append(
                f"{len(ctx_traces)}/{len(traces)} traces hit context_length limit"
            )
        if fail_traces:
            hypotheses.append(
                f"{len(fail_traces)}/{len(traces)} traces failed (reward=0)"
            )
        if not hypotheses:
            hypotheses.append("No clear failure pattern identified")

        problem_type = "context_overload" if ctx_traces else "task_failure"
        affected = [t.task_id for t in fail_traces[:5]]

        return [DiagnosisResult(
            problem_type=problem_type,
            root_cause_hypotheses=hypotheses,
            candidate_modules=[],
            confidence=0.4,
            affected_task_ids=affected,
            metadata={
                "source": f"synthesized_from_{worker_name}",
                "worker_name": worker_name,
                "task_category": task_category,
            },
        )]

    @staticmethod
    def _merge_diagnoses(diagnoses: list[DiagnosisResult]) -> list[DiagnosisResult]:
        """Merge and deduplicate diagnoses from multiple workers.

        Every worker's evidence is preserved.  Results are grouped by
        ``problem_type`` to keep the downstream planner prompt organized, but
        the merged diagnosis carries all hypotheses, affected trace IDs,
        allowed strategy suggestions, and per-worker summaries from that group.

        The merged confidence is the maximum confidence among the workers in
        the group.  This is conservative: agreement adds evidence via merged
        hypotheses/metadata, but does not inflate the numeric confidence.
        """
        grouped: dict[str, list[DiagnosisResult]] = {}
        for d in diagnoses:
            # Hard guard: only partial-task workers can contribute strategy suggestions.
            suggestions = (
                d.strategy_suggestions
                if d.metadata.get("task_category") == "partial"
                else []
            )

            problem_type = str(d.problem_type or "general").strip() or "general"
            grouped.setdefault(problem_type, []).append(
                DiagnosisResult(
                    problem_type=problem_type,
                    root_cause_hypotheses=list(d.root_cause_hypotheses),
                    candidate_modules=list(d.candidate_modules),
                    confidence=float(d.confidence),
                    affected_task_ids=list(d.affected_task_ids),
                    metadata=dict(d.metadata),
                    strategy_suggestions=deepcopy(suggestions),
                )
            )

        merged: list[DiagnosisResult] = []
        for problem_type, items in grouped.items():
            hypotheses: list[str] = []
            affected: list[str] = []
            modules: list[str] = []
            strategy_suggestions: list[dict[str, Any]] = []
            seen_suggestion_keys: set[str] = set()
            worker_evidence: list[dict[str, Any]] = []
            confidence = max((d.confidence for d in items), default=0.0)

            for d in items:
                for h in d.root_cause_hypotheses:
                    if h not in hypotheses:
                        hypotheses.append(h)
                for tid in d.affected_task_ids:
                    if tid not in affected:
                        affected.append(tid)
                for module in d.candidate_modules:
                    if module not in modules:
                        modules.append(module)

                for suggestion in d.strategy_suggestions:
                    if not isinstance(suggestion, dict):
                        continue
                    suggestion_key = json.dumps(
                        suggestion,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    )
                    if suggestion_key in seen_suggestion_keys:
                        continue
                    seen_suggestion_keys.add(suggestion_key)
                    strategy_suggestions.append(deepcopy(suggestion))

                worker_evidence.append({
                    "worker_name": d.metadata.get("worker_name", ""),
                    "task_category": d.metadata.get("task_category", ""),
                    "confidence": d.confidence,
                    "affected_trace_ids": list(d.affected_task_ids),
                    "root_cause_hypotheses": list(d.root_cause_hypotheses),
                    "analysis_summary": d.metadata.get("analysis_summary", ""),
                    "strategy_suggestions": deepcopy(d.strategy_suggestions),
                })

            summary_parts: list[str] = []
            for evidence in worker_evidence:
                summary = str(evidence.get("analysis_summary") or "").strip()
                if not summary:
                    continue
                worker = evidence.get("worker_name") or "unknown_worker"
                category = evidence.get("task_category") or "unknown_category"
                conf = evidence.get("confidence", 0.0)
                summary_parts.append(f"{worker} [{category}, conf={conf:.2f}]: {summary}")

            merged.append(
                DiagnosisResult(
                    problem_type=problem_type,
                    root_cause_hypotheses=hypotheses,
                    candidate_modules=modules,
                    confidence=confidence,
                    affected_task_ids=affected,
                    metadata={
                        "source": "merged_interactive_llm",
                        "merged_count": len(items),
                        "confidence_policy": "max_worker_confidence",
                        "merged_confidences": [d.confidence for d in items],
                        "analysis_summary": "\n".join(summary_parts),
                        "worker_evidence": worker_evidence,
                    },
                    strategy_suggestions=strategy_suggestions,
                )
            )

        merged = sorted(merged, key=lambda d: d.confidence, reverse=True)
        return merged

    @staticmethod
    def _parse_diagnoses(
        raw: list[dict[str, Any]],
        *,
        task_category: str = "",
    ) -> list[DiagnosisResult]:
        results: list[DiagnosisResult] = []
        for obj in raw:
            strategy_suggestions = obj.get("strategy_suggestions", [])
            if not isinstance(strategy_suggestions, list):
                strategy_suggestions = []
            valid_suggestions = []
            if task_category == "partial":
                for s in strategy_suggestions:
                    if isinstance(s, dict) and "pattern" in s and "steps" in s:
                        valid_suggestions.append(s)
            results.append(
                DiagnosisResult(
                    problem_type=obj.get("problem_type", "general"),
                    root_cause_hypotheses=obj.get("root_cause_hypotheses", []),
                    candidate_modules=[],
                    confidence=float(obj.get("confidence", 0.5)),
                    affected_task_ids=(
                        obj.get("affected_trace_ids")
                        or obj.get("affected_task_ids", [])
                    ),
                    metadata={"source": "interactive_llm", "task_category": task_category},
                    strategy_suggestions=valid_suggestions,
                )
            )
        return results if results else [DiagnosisResult(
            problem_type="general",
            root_cause_hypotheses=["Interactive session did not produce clear diagnosis"],
            candidate_modules=[],
            confidence=0.3,
            affected_task_ids=[],
            metadata={"source": "interactive_llm_fallback"},
        )]
