"""Interactive multi-turn diagnoser with parallel workers.

Instead of a single LLM session scanning all traces, multiple workers
analyse different task subsets concurrently (all-fail, partial/contrastive,
context-length) and their findings are merged into a unified diagnosis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import DiagnosisHistory
from skyrl_agent.meta_toolkit.diagnosis.diagnosis_tools import (
    TOOL_DESCRIPTIONS,
    build_task_grouped_message,
    classify_task_groups,
    dispatch_tool,
    get_failure_distribution,
    group_traces_by_task,
)
from skyrl_agent.meta_toolkit.diagnosis.rule_based_diagnoser import (
    DiagnosisResult,
    RuleBasedDiagnoser,
)
from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient
from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TURNS = 16
_DEFAULT_NUM_WORKERS = 3

_SYSTEM_PROMPT = """\
You are an expert AI-agent performance analyst conducting an interactive
diagnosis session. You will examine execution traces from a terminal-based
AI agent and identify failure patterns.

## Patchable modules

| Module | Responsibility |
|--------|---------------|
| strategy_library | Pattern-matched step-by-step strategies (prompt injection) |
| hook:before_llm_call | Modify prompt before each LLM call |
| hook:before_execute | Filter or modify commands before execution |
| hook:after_execute | Process terminal output after command execution |
| hook:on_timeout | Handle command timeouts |
| hook:on_parse_error | Handle LLM response parse failures |
| hook:after_round | Post-round control: verify completion, detect loops, trigger extra LLM calls |

## Workflow

1. You will receive a statistical summary of the current batch.
2. Use the diagnostic tools to investigate specific traces.
3. Compare successful and failed traces to find patterns.
4. Check history to avoid repeating past failed patches.
5. When ready, submit your diagnosis.

{tool_descriptions}

## Diagnosis format (for submit_diagnosis)

```json
{{"tool": "submit_diagnosis", "diagnoses": [
  {{
    "problem_type": "<string>",
    "root_cause_hypotheses": ["<specific evidence-based hypothesis>"],
    "candidate_modules": ["<module_name>"],
    "confidence": <0.0-1.0>,
    "affected_task_ids": ["<task_id>"],
    "strategy_suggestions": [
      {{
        "pattern": "<when to apply this strategy — a recognizable situation>",
        "steps": ["<step 1>", "<step 2>", "..."],
        "source": "<evidence: e.g. 'task 749-traj3 succeeded by doing X'>"
      }}
    ]
  }}
]}}
```

`strategy_suggestions` is optional but highly valuable. When you compare
successful vs failed traces on the SAME task, extract what the successful
trace did differently as a reusable strategy. Each strategy should have:
- `pattern`: a short description of the situation where this strategy applies
- `steps`: concrete step-by-step actions the agent should take
- `source`: which trace/task provided the evidence

Be specific — reference concrete task IDs, turn counts, and tool patterns.
Do NOT just say "context_overload" without evidence of what causes it.
"""

_FORCE_SUBMIT_MSG = (
    "You have 1 turn remaining. You MUST call submit_diagnosis NOW with your "
    "findings so far. Summarize what you have observed and submit a diagnosis, "
    "even if your investigation is incomplete."
)


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a JSON tool call from LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    search_text = cleaned or text

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

    # Also try matching nested JSON for submit_diagnosis
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
    """Extract a concise analysis summary from the worker's conversation history.

    Collects tool results (trace inspections, comparisons) and the final
    assistant reasoning to give the planner concrete evidence.
    """
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


class InteractiveDiagnoser:
    """Parallel multi-turn interactive diagnoser.

    Launches multiple worker sessions that analyse different task subsets,
    then merges their diagnoses.
    """

    def __init__(
        self,
        client: MetaLLMClient,
        history: DiagnosisHistory | None = None,
        max_turns: int = _DEFAULT_MAX_TURNS,
        num_workers: int = _DEFAULT_NUM_WORKERS,
        override_base: str | None = None,
    ) -> None:
        self._client = client
        self._history = history or DiagnosisHistory()
        self._max_turns = max_turns
        self._num_workers = num_workers
        self._override_base = override_base
        self._fallback = RuleBasedDiagnoser()

    async def diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        if not traces:
            return []

        try:
            return await self._parallel_diagnose(traces)
        except Exception as exc:
            logger.warning(f"Parallel diagnosis failed ({exc}); falling back to rules")
            return self._fallback.diagnose(traces)

    async def _parallel_diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        """Run multiple diagnosis workers concurrently and merge results."""
        groups = group_traces_by_task(traces)
        all_fail, partial, all_pass = classify_task_groups(groups)

        # Decide which workers to launch based on available data
        worker_configs: list[tuple[str, str]] = []

        if partial:
            worker_configs.append(("partial", "partial"))
        if all_fail:
            worker_configs.append(("all_fail", "all_fail"))

        ctx_groups = [g for g in groups.values() if g.has_context_length]
        if ctx_groups:
            worker_configs.append(("context_length", "context_length"))

        if not worker_configs:
            worker_configs.append(("all", "all"))

        # Cap at num_workers
        worker_configs = worker_configs[:self._num_workers]

        logger.info(
            f"Launching {len(worker_configs)} diagnosis workers: "
            f"{[wc[0] for wc in worker_configs]}"
        )

        tasks = [
            self._run_single_worker(traces, focus=focus, worker_name=name)
            for name, focus in worker_configs
        ]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all diagnoses
        all_diagnoses: list[DiagnosisResult] = []
        for i, result in enumerate(worker_results):
            name = worker_configs[i][0]
            if isinstance(result, Exception):
                logger.warning(f"Worker '{name}' failed: {result}")
                continue
            logger.info(f"Worker '{name}' produced {len(result)} diagnoses")
            all_diagnoses.extend(result)

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
        focus: str,
        worker_name: str,
    ) -> list[DiagnosisResult]:
        """Run a single diagnosis worker session."""
        self._client.set_role("diagnosis")

        system_prompt = _SYSTEM_PROMPT.format(tool_descriptions=TOOL_DESCRIPTIONS)
        user_msg = build_task_grouped_message(traces, self._history, focus=focus)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        for turn in range(self._max_turns):
            # Inject force-submit prompt near the end
            is_penultimate = turn == self._max_turns - 2
            is_last = turn == self._max_turns - 1

            response = await self._client.chat(messages, temperature=0.2)
            messages.append({"role": "assistant", "content": response})

            tool_call = _extract_tool_call(response)

            if tool_call is not None and tool_call.get("tool") == "submit_diagnosis":
                diagnoses_raw = tool_call.get("diagnoses", [])
                result = self._parse_diagnoses(diagnoses_raw)
                summary = _build_analysis_summary(messages)
                for d in result:
                    d.metadata["analysis_summary"] = summary
                logger.info(f"Worker '{worker_name}' submitted diagnosis at turn {turn+1}")
                return result

            if tool_call is not None:
                result = dispatch_tool(tool_call, traces, self._history, self._override_base)
                if result == "__SUBMIT__":
                    diagnoses_raw = tool_call.get("diagnoses", [])
                    return self._parse_diagnoses(diagnoses_raw)

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

        # Reached turn limit — synthesize from conversation instead of falling back
        logger.warning(
            f"Worker '{worker_name}' hit turn limit; "
            "synthesizing diagnosis from conversation"
        )
        return self._synthesize_from_conversation(messages, traces, worker_name)

    def _synthesize_from_conversation(
        self,
        messages: list[dict[str, str]],
        traces: list[TraceRecord],
        worker_name: str,
    ) -> list[DiagnosisResult]:
        """Build a diagnosis from what the LLM observed even without submit_diagnosis."""
        observations = _extract_observations_from_history(messages)

        # Check the last assistant message for any partial diagnosis
        last_assistant = ""
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                last_assistant = msg["content"]
                break

        # Try to extract a submit_diagnosis from the last response (it might be
        # incomplete JSON but still parseable)
        tool_call = _extract_tool_call(last_assistant)
        if tool_call and tool_call.get("tool") == "submit_diagnosis":
            diagnoses_raw = tool_call.get("diagnoses", [])
            result = self._parse_diagnoses(diagnoses_raw)
            if result and result[0].problem_type != "general":
                return result

        # Build a synthetic diagnosis from observed patterns
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
        modules = ["hook:before_llm_call", "strategy_library"]
        if ctx_traces:
            modules.append("hook:after_round")

        affected = [t.task_id for t in fail_traces[:5]]

        return [DiagnosisResult(
            problem_type=problem_type,
            root_cause_hypotheses=hypotheses,
            candidate_modules=modules,
            confidence=0.4,
            affected_task_ids=affected,
            metadata={"source": f"synthesized_from_{worker_name}"},
        )]

    @staticmethod
    def _merge_diagnoses(diagnoses: list[DiagnosisResult]) -> list[DiagnosisResult]:
        """Merge and deduplicate diagnoses from multiple workers."""
        seen_types: dict[str, DiagnosisResult] = {}
        for d in diagnoses:
            key = d.problem_type
            if key not in seen_types or d.confidence > seen_types[key].confidence:
                seen_types[key] = d
            else:
                existing = seen_types[key]
                for h in d.root_cause_hypotheses:
                    if h not in existing.root_cause_hypotheses:
                        existing.root_cause_hypotheses.append(h)
                for m in d.candidate_modules:
                    if m not in existing.candidate_modules:
                        existing.candidate_modules.append(m)
                for tid in d.affected_task_ids:
                    if tid not in existing.affected_task_ids:
                        existing.affected_task_ids.append(tid)

        merged = sorted(seen_types.values(), key=lambda d: d.confidence, reverse=True)
        return merged

    @staticmethod
    def _parse_diagnoses(raw: list[dict[str, Any]]) -> list[DiagnosisResult]:
        results: list[DiagnosisResult] = []
        for obj in raw:
            strategy_suggestions = obj.get("strategy_suggestions", [])
            if not isinstance(strategy_suggestions, list):
                strategy_suggestions = []
            valid_suggestions = []
            for s in strategy_suggestions:
                if isinstance(s, dict) and "pattern" in s and "steps" in s:
                    valid_suggestions.append(s)
            results.append(
                DiagnosisResult(
                    problem_type=obj.get("problem_type", "general"),
                    root_cause_hypotheses=obj.get("root_cause_hypotheses", []),
                    candidate_modules=obj.get("candidate_modules", ["strategy_library"]),
                    confidence=float(obj.get("confidence", 0.5)),
                    affected_task_ids=obj.get("affected_task_ids", []),
                    metadata={"source": "interactive_llm"},
                    strategy_suggestions=valid_suggestions,
                )
            )
        return results if results else [DiagnosisResult(
            problem_type="general",
            root_cause_hypotheses=["Interactive session did not produce clear diagnosis"],
            candidate_modules=["strategy_library"],
            confidence=0.3,
            affected_task_ids=[],
            metadata={"source": "interactive_llm_fallback"},
        )]
