"""Rule-based failure diagnoser for agent trajectories.

This module provides a rule-driven DiagnosisResult from a batch of TraceRecords,
without requiring an LLM call. Rules are based on finish_reason, failure_tags,
sync_bottleneck_score, turn_count, and tool_failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord


@dataclass(slots=True)
class DiagnosisResult:
    """Structured diagnosis output from a batch of TraceRecords."""

    problem_type: str
    root_cause_hypotheses: list[str] = field(default_factory=list)
    candidate_modules: list[str] = field(default_factory=list)
    confidence: float = 0.0
    affected_task_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    strategy_suggestions: list[dict[str, Any]] = field(default_factory=list)


# Module candidates keyed by problem type
_MODULE_MAP: dict[str, list[str]] = {
    "timeout": ["hook:on_timeout", "strategy_library"],
    "verification_failure": ["hook:after_round", "strategy_library"],
    "termination_failure": ["hook:after_round", "strategy_library"],
    "planning_failure": ["hook:before_llm_call", "strategy_library"],
    "context_overload": [],
    "sync_bottleneck": ["strategy_library"],
    "tool_exhaustion": ["hook:before_llm_call", "strategy_library"],
    "low_reward": ["hook:after_round", "hook:before_llm_call", "strategy_library"],
    "general": ["strategy_library"],
}

# Rules: each rule is a predicate that, when matched, contributes to diagnosis
_RULES: list[dict[str, Any]] = [
    # Timeout / sync bottleneck rules
    {
        "id": "sync_bottleneck",
        "check": lambda batch, stats: stats["max_sync_bottleneck"] >= 0.5,
        "problem_type": "sync_bottleneck",
        "hypotheses": [
            "Agent spends too many sequential turns on long-running tasks",
            "No parallel tool execution or fanout, causing timeouts",
            "Planner fails to decompose into independently executable sub-tasks",
        ],
        "min_prevalence": 0.1,
    },
    {
        "id": "timeout",
        "check": lambda batch, stats: stats["timeout_count"] > 0,
        "problem_type": "timeout",
        "hypotheses": [
            "Agent exceeds time budget, likely due to poor planning or blocking operations",
        ],
        "min_prevalence": 0.05,
    },
    {
        "id": "tool_exhaustion",
        "check": lambda batch, stats: stats["avg_turns"] >= 15 and stats["avg_tool_calls"] >= 10,
        "problem_type": "tool_exhaustion",
        "hypotheses": [
            "Agent makes too many tool calls without converging; planning horizon too short",
        ],
        "min_prevalence": 0.15,
    },
    {
        "id": "context_overload",
        "check": lambda batch, stats: stats["context_error_count"] > 0,
        "problem_type": "context_overload",
        "hypotheses": [
            "Context window fills up before task completion; summarization or planning needed",
        ],
        "min_prevalence": 0.03,
    },
    {
        "id": "verification_missing",
        "check": lambda batch, stats: stats["success_count"] > 0 and stats["avg_reward"] < 0.5,
        "problem_type": "verification_failure",
        "hypotheses": [
            "Tasks appear to complete but fail verification; missing or weak verification step",
        ],
        "min_prevalence": 0.1,
    },
    {
        "id": "termination_premature",
        "check": lambda batch, stats: stats["success_count"] == 0 and stats["avg_turns"] <= 3,
        "problem_type": "termination_failure",
        "hypotheses": [
            "Agent gives up too early; finish policy may be too aggressive or miscalibrated",
        ],
        "min_prevalence": 0.05,
    },
    {
        "id": "planning_weak",
        "check": lambda batch, stats: stats["avg_reward"] < 0.3 and stats["avg_turns"] > 5,
        "problem_type": "planning_failure",
        "hypotheses": [
            "Planner generates ineffective sub-goals; reward stays low despite many turns",
        ],
        "min_prevalence": 0.1,
    },
]


def _compute_stats(traces: list[TraceRecord]) -> dict[str, Any]:
    n = len(traces)
    if n == 0:
        return {}
    return {
        "n": n,
        "success_count": sum(1 for t in traces if t.success),
        "failure_count": sum(1 for t in traces if not t.success),
        "timeout_count": sum(1 for t in traces if t.finish_reason == "agent_timeout"),
        "error_count": sum(1 for t in traces if t.finish_reason == "error"),
        "context_error_count": sum(
            1 for t in traces if t.finish_reason == "context_length" or "context_overload" in t.failure_tags
        ),
        "max_sync_bottleneck": max((t.sync_bottleneck_score for t in traces), default=0.0),
        "avg_turns": sum(t.turn_count for t in traces) / n,
        "avg_tool_calls": sum(t.tool_calls for t in traces) / n,
        "tool_error_count": sum(t.tool_failures for t in traces),
        "failure_with_tool_errors": sum(
            1 for t in traces if not t.success and t.tool_failures > 0
        ),
        "avg_reward": sum((t.final_reward or 0) for t in traces) / n,
        "termination_failures": sum(
            1 for t in traces if "termination_failure" in t.failure_tags
        ),
        "verification_failures": sum(
            1 for t in traces if "verification_failure" in t.failure_tags
        ),
        "planning_failures": sum(1 for t in traces if "planning_failure" in t.failure_tags),
    }


class RuleBasedDiagnoser:
    """Batch-level failure diagnoser using hand-crafted rules.

    Rules are evaluated in order. The first matching rule with sufficient
    prevalence drives the primary DiagnosisResult; additional rules contribute
    secondary hypotheses.
    """

    def diagnose(self, traces: list[TraceRecord]) -> list[DiagnosisResult]:
        """Analyze a batch of TraceRecords and return zero or more DiagnosisResult.

        At least one result is returned if the batch is non-empty and any rule fires.
        """
        if not traces:
            return []

        stats = _compute_stats(traces)
        results: list[DiagnosisResult] = []
        seen_problem_types: set[str] = set()

        for rule in _RULES:
            try:
                matched = rule["check"](traces, stats)
            except Exception:
                continue

            if not matched:
                continue

            prevalence = self._prevalence(traces, rule)
            if prevalence < rule.get("min_prevalence", 0):
                continue

            problem_type = rule["problem_type"]
            if problem_type in seen_problem_types:
                continue
            seen_problem_types.add(problem_type)

            affected = [
                t.task_id for t in traces if self._trace_matches_problem(t, problem_type)
            ]

            result = DiagnosisResult(
                problem_type=problem_type,
                root_cause_hypotheses=list(rule["hypotheses"]),
                candidate_modules=_MODULE_MAP.get(problem_type, _MODULE_MAP["general"]),
                confidence=min(prevalence * 2, 1.0),
                affected_task_ids=affected,
                metadata={
                    "rule_id": rule["id"],
                    "prevalence": prevalence,
                    "stats": {k: v for k, v in stats.items() if k != "n"},
                },
            )
            results.append(result)

        # If nothing matched but there are failures, return a generic low-reward result
        if not results and stats.get("failure_count", 0) > 0:
            results.append(
                DiagnosisResult(
                    problem_type="general",
                    root_cause_hypotheses=["Failures detected but no specific rule matched"],
                    candidate_modules=_MODULE_MAP["general"],
                    confidence=0.3,
                    affected_task_ids=[t.task_id for t in traces if not t.success],
                    metadata={"stats": {k: v for k, v in stats.items() if k != "n"}},
                )
            )

        return results

    @staticmethod
    def _prevalence(traces: list[TraceRecord], rule: dict[str, Any]) -> float:
        """Compute what fraction of traces are affected by a rule."""
        rule_id = rule["id"]
        match_fn = _TRACE_MATCH_FNS.get(rule_id, lambda t: False)
        affected = sum(1 for t in traces if match_fn(t))
        return affected / len(traces) if traces else 0.0

    @staticmethod
    def _trace_matches_problem(trace: TraceRecord, problem_type: str) -> bool:
        """Return True if a trace matches a given problem type."""
        if problem_type == "sync_bottleneck":
            return trace.sync_bottleneck_score >= 0.5
        if problem_type == "timeout":
            return trace.finish_reason == "agent_timeout"
        if problem_type == "context_overload":
            return trace.finish_reason == "context_length" or "context_overload" in trace.failure_tags
        if problem_type == "verification_failure":
            return "verification_failure" in trace.failure_tags
        if problem_type == "termination_failure":
            return "termination_failure" in trace.failure_tags
        if problem_type == "planning_failure":
            return "planning_failure" in trace.failure_tags
        if problem_type == "tool_exhaustion":
            return trace.turn_count >= 15 and trace.tool_calls >= 10
        if problem_type == "low_reward":
            return not trace.success and (trace.final_reward or 0) < 0.5
        return False


# --- Per-trace matching helpers used by _prevalence ---
def _match_sync_bottleneck(t: TraceRecord) -> bool:
    return t.sync_bottleneck_score >= 0.5


def _match_timeout(t: TraceRecord) -> bool:
    return t.finish_reason == "agent_timeout"


def _match_context(t: TraceRecord) -> bool:
    return t.finish_reason == "context_length" or "context_overload" in t.failure_tags


def _match_verification(t: TraceRecord) -> bool:
    return "verification_failure" in t.failure_tags


def _match_termination(t: TraceRecord) -> bool:
    return "termination_failure" in t.failure_tags


def _match_planning(t: TraceRecord) -> bool:
    return "planning_failure" in t.failure_tags


def _match_tool_exhaustion(t: TraceRecord) -> bool:
    return t.turn_count >= 15 and t.tool_calls >= 10


def _match_low_reward(t: TraceRecord) -> bool:
    return not t.success and (t.final_reward or 0) < 0.5


_TRACE_MATCH_FNS: dict[str, callable] = {
    "sync_bottleneck": _match_sync_bottleneck,
    "timeout": _match_timeout,
    "context_overload": _match_context,
    "verification_missing": _match_verification,
    "termination_premature": _match_termination,
    "planning_weak": _match_planning,
    "tool_exhaustion": _match_tool_exhaustion,
    "low_reward": _match_low_reward,
}
