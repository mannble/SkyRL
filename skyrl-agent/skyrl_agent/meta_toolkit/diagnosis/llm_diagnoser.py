"""LLM-powered failure diagnoser for agent trajectories.

Replaces (or supplements) the rule-based diagnoser by asking an LLM to
analyse a serialised batch of TraceRecords and return structured diagnoses.
Falls back to the rule-based diagnoser on LLM failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from skyrl_agent.meta_toolkit.diagnosis.rule_based_diagnoser import (
    DiagnosisResult,
    RuleBasedDiagnoser,
)
from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient
from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert AI-agent performance analyst.  You receive execution traces
from a terminal-based AI agent and must diagnose failure patterns.

## Agent architecture

The agent operates in a loop: receive prompt → call LLM → parse response →
execute commands in terminal → observe output → repeat.

The following **patchable policy modules** control its behavior — each can be
tuned via YAML overrides or code hooks:

| Module | Responsibility |
|--------|---------------|
| planner_policy | Task decomposition, sub-goal generation, planning style |
| retry_policy | Error recovery, retry count/delay, fallback strategy |
| verification_policy | Post-step verification, acceptance threshold |
| finish_policy | Termination decision, early-exit threshold |
| system_prompt_overrides | Strategy hints injected into agent prompt |
| strategy_library | Pattern-matched step-by-step strategies |
| behavior_policy | Command filtering, output limits, loop detection |

## Known problem types

Choose one or more of the following. Be precise — the downstream planner
uses your diagnosis to select the right fix strategy.

| Problem type | Typical evidence |
|---|---|
| premature_completion | Low reward despite few turns; agent marks task_complete early |
| output_overflow | Very high turn counts; agent repeats commands; context_overload tags |
| wasteful_waiting | High wall_clock_ms relative to turn count; long durations on fast commands |
| parse_error_loop | High tool_failures; repeated parse errors; agent stuck retrying format |
| stuck_in_loop | Same commands repeated across turns; no progress in reward |
| planning_failure | Agent takes many turns on simple tasks; no clear strategy |
| timeout | Many agent_timeout finish reasons; commands exceeding duration |
| verification_gap | Tasks pass but with low reward; partial solutions accepted |
| recovery_failure | Agent fails to recover after errors; no retry or wrong retry |
| context_overload | Token limit hit; summarization triggered; information lost |
| tool_exhaustion | Agent runs out of episodes before completing |
| general | Does not fit above categories |

## Diagnostic guidelines

1. Look at the **distribution** of failures, not just individual traces.
2. Check reward vs turn count: low reward + few turns → premature_completion;
   low reward + many turns → stuck_in_loop or planning_failure.
3. Check wall_clock_ms vs turns: high ratio → wasteful_waiting.
4. Check tool_failures count: high → parse_error_loop or recovery_failure.
5. A single trace can contribute to multiple diagnoses.
6. Always include `candidate_modules` — these tell the planner which modules
   to modify. Include `system_prompt_overrides`, `strategy_library`, and
   `behavior_policy` when relevant, not just the four core modules.

## Output format

Return a JSON **array** of diagnosis objects (one per distinct problem found).
Each object has:

```json
{
  "problem_type": "<string>",
  "root_cause_hypotheses": ["<string>", ...],
  "candidate_modules": ["<module_name>", ...],
  "confidence": <float 0-1>,
  "affected_task_ids": ["<task_id>", ...]
}
```

If no issues are found, return an empty array `[]`.
Be specific in hypotheses — reference concrete trace evidence (task IDs,
turn counts, finish reasons, tool failure counts, etc.).
"""


def _serialise_traces(traces: list[TraceRecord], max_traces: int = 30) -> str:
    """Convert traces to a compact text representation for the LLM context."""
    selected = traces[:max_traces]
    rows: list[str] = []
    for t in selected:
        row = {
            "task_id": t.task_id,
            "success": t.success,
            "reward": t.final_reward,
            "turns": t.turn_count,
            "finish_reason": t.finish_reason,
            "tool_calls": t.tool_calls,
            "tool_failures": t.tool_failures,
            "sync_bottleneck": round(t.sync_bottleneck_score, 2),
            "failure_tags": t.failure_tags,
            "verification": t.verification_attempted,
            "retry": t.retry_attempted,
            "wall_clock_ms": t.wall_clock_latency_ms,
        }
        rows.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(rows)


def _parse_diagnoses(raw: list[dict[str, Any]]) -> list[DiagnosisResult]:
    results: list[DiagnosisResult] = []
    for obj in raw:
        results.append(
            DiagnosisResult(
                problem_type=obj.get("problem_type", "general"),
                root_cause_hypotheses=obj.get("root_cause_hypotheses", []),
                candidate_modules=obj.get("candidate_modules", ["planner_policy"]),
                confidence=float(obj.get("confidence", 0.5)),
                affected_task_ids=obj.get("affected_task_ids", []),
                metadata={"source": "llm"},
            )
        )
    return results


class LLMDiagnoser:
    """Batch-level failure diagnoser powered by an LLM.

    Falls back to ``RuleBasedDiagnoser`` when the LLM call fails.
    """

    def __init__(self, client: MetaLLMClient) -> None:
        self._client = client
        self._fallback = RuleBasedDiagnoser()

    async def diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        if not traces:
            return []

        try:
            return await self._llm_diagnose(traces)
        except Exception as exc:
            logger.warning(f"LLM diagnosis failed ({exc}); falling back to rules")
            return self._fallback.diagnose(traces)

    async def _llm_diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        self._client.set_role("diagnosis")
        summary = self._build_summary(traces)
        trace_text = _serialise_traces(traces)

        user_msg = (
            f"## Batch summary\n{summary}\n\n"
            f"## Individual traces ({len(traces)} total, showing up to 30)\n"
            f"{trace_text}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        data = await self._client.chat_json(messages, temperature=0.2)

        diagnoses_raw = data if isinstance(data, list) else data.get("diagnoses", [data])
        results = _parse_diagnoses(diagnoses_raw)

        logger.info(
            f"LLM diagnosed {len(results)} issues: "
            f"{[r.problem_type for r in results]}"
        )
        return results

    @staticmethod
    def _build_summary(traces: list[TraceRecord]) -> str:
        n = len(traces)
        successes = sum(1 for t in traces if t.success)
        avg_reward = sum((t.final_reward or 0) for t in traces) / n if n else 0
        avg_turns = sum(t.turn_count for t in traces) / n if n else 0
        timeouts = sum(1 for t in traces if t.finish_reason == "agent_timeout")
        tool_failures = sum(t.tool_failures for t in traces)
        tags: dict[str, int] = {}
        for t in traces:
            for tag in t.failure_tags:
                tags[tag] = tags.get(tag, 0) + 1

        lines = [
            f"- Total traces: {n}",
            f"- Success rate: {successes}/{n} ({successes/n*100:.0f}%)" if n else "- N/A",
            f"- Avg reward: {avg_reward:.3f}",
            f"- Avg turns: {avg_turns:.1f}",
            f"- Timeouts: {timeouts}",
            f"- Total tool failures: {tool_failures}",
            f"- Failure tags: {json.dumps(tags, ensure_ascii=False)}",
        ]
        return "\n".join(lines)
