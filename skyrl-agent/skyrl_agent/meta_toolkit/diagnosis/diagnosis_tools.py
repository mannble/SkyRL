"""Diagnosis tools: callable functions that the interactive diagnoser can invoke.

Each tool takes structured arguments and returns a text result that gets
appended to the multi-turn conversation, simulating function-calling.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord
from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import DiagnosisHistory


# ---------------------------------------------------------------------------
# Task-level grouping helpers
# ---------------------------------------------------------------------------

def _task_key(task_id: str) -> str:
    """Extract the instance-level key from a trace task_id like '749-traj3'."""
    m = re.match(r"^(.+)-traj\d+$", task_id)
    return m.group(1) if m else task_id


@dataclass
class TaskGroup:
    """Aggregated statistics for all trajectories of a single task."""
    task_key: str
    traces: list[TraceRecord] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.traces)

    @property
    def n_success(self) -> int:
        return sum(1 for t in self.traces if t.success)

    @property
    def n_fail(self) -> int:
        return self.n_total - self.n_success

    @property
    def pass_rate(self) -> float:
        return self.n_success / self.n_total if self.n_total else 0.0

    @property
    def avg_reward(self) -> float:
        rewards = [t.final_reward for t in self.traces if t.final_reward is not None]
        return sum(rewards) / len(rewards) if rewards else 0.0

    @property
    def has_context_length(self) -> bool:
        return any(t.finish_reason == "context_length" for t in self.traces)

    def fail_ids(self) -> list[str]:
        return [t.task_id for t in self.traces if not t.success]

    def success_ids(self) -> list[str]:
        return [t.task_id for t in self.traces if t.success]


def group_traces_by_task(traces: list[TraceRecord]) -> dict[str, TaskGroup]:
    """Group traces by instance-level task key."""
    groups: dict[str, TaskGroup] = {}
    for t in traces:
        key = _task_key(t.task_id)
        if key not in groups:
            groups[key] = TaskGroup(task_key=key)
        groups[key].traces.append(t)
    return groups


def classify_task_groups(
    groups: dict[str, TaskGroup],
) -> tuple[list[TaskGroup], list[TaskGroup], list[TaskGroup]]:
    """Split task groups into (all_fail, partial, all_pass) buckets."""
    all_fail: list[TaskGroup] = []
    partial: list[TaskGroup] = []
    all_pass: list[TaskGroup] = []
    for g in groups.values():
        if g.n_success == 0:
            all_fail.append(g)
        elif g.n_fail == 0:
            all_pass.append(g)
        else:
            partial.append(g)
    all_fail.sort(key=lambda g: g.avg_reward)
    partial.sort(key=lambda g: g.pass_rate)
    all_pass.sort(key=lambda g: g.avg_reward, reverse=True)
    return all_fail, partial, all_pass


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def inspect_trace(traces: list[TraceRecord], task_id: str, traj_id: str | None = None) -> str:
    """Return detailed info for a specific trace (or all traces matching a task_id)."""
    matches = [t for t in traces if t.task_id == task_id]
    if traj_id:
        matches = [t for t in matches if traj_id in t.task_id]
    if not matches:
        all_ids = sorted(set(t.task_id for t in traces))[:20]
        return f"No trace found for task_id='{task_id}'. Available IDs (first 20): {all_ids}"

    parts: list[str] = []
    for t in matches[:3]:
        events_summary = []
        for ev in t.events[:30]:
            snippet = ev.content[:120].replace("\n", " ") if ev.content else ""
            events_summary.append(f"  turn {ev.turn_id} [{ev.role}/{ev.action_type}]: {snippet}")
        events_text = "\n".join(events_summary) if events_summary else "  (no events)"
        parts.append(
            f"task_id: {t.task_id}\n"
            f"success: {t.success}, reward: {t.final_reward}, turns: {t.turn_count}\n"
            f"finish_reason: {t.finish_reason}, tool_calls: {t.tool_calls}, "
            f"tool_failures: {t.tool_failures}\n"
            f"failure_tags: {t.failure_tags}\n"
            f"sync_bottleneck_score: {t.sync_bottleneck_score}\n"
            f"events ({len(t.events)} total, showing first 30):\n{events_text}"
        )
    return "\n---\n".join(parts)


def compare_traces(
    traces: list[TraceRecord],
    task_id_a: str,
    task_id_b: str,
) -> str:
    """Compare two traces side by side, highlighting key differences."""
    a = next((t for t in traces if t.task_id == task_id_a), None)
    b = next((t for t in traces if t.task_id == task_id_b), None)
    if not a or not b:
        missing = []
        if not a:
            missing.append(task_id_a)
        if not b:
            missing.append(task_id_b)
        return f"Trace(s) not found: {missing}"

    def _row(label: str, va: Any, vb: Any) -> str:
        marker = " <--DIFF" if va != vb else ""
        return f"  {label:25s}: {str(va):>20s} | {str(vb):>20s}{marker}"

    lines = [
        f"Comparing: {task_id_a} vs {task_id_b}",
        f"{'Field':27s}  {'A':>20s} | {'B':>20s}",
        "-" * 75,
        _row("success", a.success, b.success),
        _row("reward", a.final_reward, b.final_reward),
        _row("turns", a.turn_count, b.turn_count),
        _row("finish_reason", a.finish_reason, b.finish_reason),
        _row("tool_calls", a.tool_calls, b.tool_calls),
        _row("tool_failures", a.tool_failures, b.tool_failures),
        _row("sync_bottleneck", f"{a.sync_bottleneck_score:.2f}", f"{b.sync_bottleneck_score:.2f}"),
        _row("failure_tags", str(a.failure_tags), str(b.failure_tags)),
        _row("num_events", len(a.events), len(b.events)),
    ]
    return "\n".join(lines)


def get_task_overview(traces: list[TraceRecord], task_key: str) -> str:
    """Show an overview of all trajectories for a given task instance."""
    groups = group_traces_by_task(traces)
    group = groups.get(task_key)
    if not group:
        available = sorted(groups.keys())[:20]
        return f"No task found for key='{task_key}'. Available keys (first 20): {available}"

    lines = [
        f"Task: {group.task_key}  ({group.n_success}/{group.n_total} passed, "
        f"avg_reward={group.avg_reward:.3f})",
        "",
    ]
    for t in group.traces:
        tag_str = ",".join(t.failure_tags) if t.failure_tags else "-"
        lines.append(
            f"  {t.task_id}: reward={t.final_reward}, turns={t.turn_count}, "
            f"finish={t.finish_reason}, tags=[{tag_str}]"
        )
    return "\n".join(lines)


def get_failure_distribution(traces: list[TraceRecord]) -> str:
    """Aggregate failure statistics across all traces."""
    n = len(traces)
    if n == 0:
        return "No traces available."
    successes = sum(1 for t in traces if t.success)
    tag_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for t in traces:
        reason = t.finish_reason or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for tag in t.failure_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    avg_turns_ok = 0.0
    avg_turns_fail = 0.0
    ok_traces = [t for t in traces if t.success]
    fail_traces = [t for t in traces if not t.success]
    if ok_traces:
        avg_turns_ok = sum(t.turn_count for t in ok_traces) / len(ok_traces)
    if fail_traces:
        avg_turns_fail = sum(t.turn_count for t in fail_traces) / len(fail_traces)

    rewards = [t.final_reward for t in traces if t.final_reward is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    zero_reward = sum(1 for r in rewards if r == 0.0)
    full_reward = sum(1 for r in rewards if r >= 1.0)
    sorted_rewards = sorted(rewards)
    p25 = sorted_rewards[len(sorted_rewards) // 4] if sorted_rewards else 0.0
    p50 = sorted_rewards[len(sorted_rewards) // 2] if sorted_rewards else 0.0
    p75 = sorted_rewards[3 * len(sorted_rewards) // 4] if sorted_rewards else 0.0

    lines = [
        f"Total: {n}, Success (reward>0): {successes} ({successes/n*100:.0f}%), Fail (reward=0): {n-successes}",
        f"Reward distribution: avg={avg_reward:.3f}, p25={p25:.2f}, p50={p50:.2f}, p75={p75:.2f}",
        f"  zero_reward={zero_reward}/{n} ({zero_reward/n*100:.0f}%), full_reward={full_reward}/{n} ({full_reward/n*100:.0f}%)",
        f"Avg turns (success): {avg_turns_ok:.1f}, Avg turns (failure): {avg_turns_fail:.1f}",
        f"Finish reasons: {json.dumps(reason_counts)}",
        f"Failure tags: {json.dumps(tag_counts)}",
    ]

    groups = group_traces_by_task(traces)
    all_fail, partial, all_pass = classify_task_groups(groups)
    n_tasks = len(groups)
    lines.append("")
    lines.append(f"Per-task breakdown ({n_tasks} tasks):")
    lines.append(f"  All-fail: {len(all_fail)} ({len(all_fail)/n_tasks*100:.0f}%)")
    lines.append(f"  Partial:  {len(partial)} ({len(partial)/n_tasks*100:.0f}%)")
    lines.append(f"  All-pass: {len(all_pass)} ({len(all_pass)/n_tasks*100:.0f}%)")
    return "\n".join(lines)


def get_history(history: DiagnosisHistory, n_cycles: int = 5) -> str:
    """Return recent meta-learning cycle history."""
    return history.format_for_prompt(n_cycles)


def get_active_patches(override_base: str | None) -> str:
    """Return the content of all currently active patches (YAML overrides + hooks).

    This lets the diagnoser review whether existing patches are still appropriate
    and recommend modifications or removals.
    """
    from pathlib import Path
    import yaml as _yaml

    if not override_base:
        return "(No override_base configured — cannot inspect active patches.)"

    base = Path(override_base)
    if not base.is_dir():
        return "(Override directory does not exist — no active patches.)"

    parts: list[str] = []

    # YAML overrides
    for yf in sorted(base.glob("*.yaml")):
        try:
            content = _yaml.safe_load(yf.read_text(encoding="utf-8"))
            if content:
                parts.append(f"**{yf.name}**:\n```yaml\n{_yaml.dump(content, default_flow_style=False).strip()}\n```")
        except Exception:
            parts.append(f"**{yf.name}**: (failed to parse)")

    # Hook code
    hooks_dir = base / "hooks"
    if hooks_dir.is_dir():
        for hf in sorted(hooks_dir.glob("*.py")):
            try:
                source = hf.read_text(encoding="utf-8")
                parts.append(f"**hook: {hf.stem}**:\n```python\n{source.strip()}\n```")
            except Exception:
                parts.append(f"**hook: {hf.stem}**: (failed to read)")

    if not parts:
        return "(No active patches — all modules use default configuration.)"

    return "Currently active patches:\n\n" + "\n\n".join(parts)


def build_task_grouped_message(
    traces: list[TraceRecord],
    history: DiagnosisHistory,
    *,
    focus: str = "all",
    target_task: TaskGroup | None = None,
    task_category: str = "",
) -> str:
    """Build the initial user message with task-grouped sampling.

    Args:
        focus: "all" (default), "all_fail", "partial", or "context_length".
               Workers can use different focus values.
        target_task: When set, the worker analyses only this single task.
        task_category: "partial", "all_fail", or "all_pass" -- describes the
                       target task's category for focused instructions.
    """
    history_text = history.format_for_prompt(5)

    # --- Single-task mode ---
    if target_task is not None:
        return _build_single_task_message(
            history_text, target_task, task_category,
        )

    stats = get_failure_distribution(traces)
    groups = group_traces_by_task(traces)
    all_fail, partial, all_pass = classify_task_groups(groups)

    sections: list[str] = []

    if focus in ("all", "partial"):
        partial_lines: list[str] = []
        for g in partial[:6]:
            fail = g.fail_ids()[:2]
            succ = g.success_ids()[:2]
            partial_lines.append(
                f"  {g.task_key}: pass={g.n_success}/{g.n_total}, "
                f"fail_ids={fail}, success_ids={succ}"
            )
        if partial_lines:
            sections.append(
                "Partial tasks (compare success vs fail on same task — highest diagnostic value):\n"
                + "\n".join(partial_lines)
            )

    if focus in ("all", "all_fail"):
        fail_lines: list[str] = []
        for g in all_fail[:6]:
            fail_lines.append(
                f"  {g.task_key}: 0/{g.n_total}, avg_reward={g.avg_reward:.2f}, "
                f"sample_id='{g.traces[0].task_id}'"
            )
        if fail_lines:
            sections.append(
                "All-fail tasks (worst performance):\n" + "\n".join(fail_lines)
            )

    if focus in ("all", "context_length"):
        ctx_groups = [g for g in groups.values() if g.has_context_length]
        ctx_groups.sort(key=lambda g: g.pass_rate)
        ctx_lines: list[str] = []
        for g in ctx_groups[:5]:
            ctx_ids = [
                t.task_id for t in g.traces if t.finish_reason == "context_length"
            ][:2]
            ctx_lines.append(
                f"  {g.task_key}: pass={g.n_success}/{g.n_total}, "
                f"context_length_ids={ctx_ids}"
            )
        if ctx_lines:
            sections.append(
                "Tasks with context_length failures:\n" + "\n".join(ctx_lines)
            )

    if focus == "all":
        pass_lines: list[str] = []
        for g in all_pass[:3]:
            pass_lines.append(
                f"  {g.task_key}: {g.n_success}/{g.n_total}, "
                f"sample_id='{g.traces[0].task_id}'"
            )
        if pass_lines:
            sections.append(
                "All-pass tasks (for regression detection):\n" + "\n".join(pass_lines)
            )

    investigation = "\n\n".join(sections) if sections else "(no investigation targets)"

    focus_instruction = ""
    if focus == "all_fail":
        focus_instruction = (
            "Focus on ALL-FAIL tasks: investigate why the agent consistently fails. "
        )
    elif focus == "partial":
        focus_instruction = (
            "Focus on PARTIAL tasks: compare success and failure trajectories on "
            "the SAME task to find what differs. Use compare_traces.\n\n"
            "**MANDATORY**: For each partial task you analyse, your submit_diagnosis "
            "MUST include at least one `strategy_suggestions` entry. Extract what "
            "the successful trace did RIGHT as a reusable strategy. Format:\n"
            '  "strategy_suggestions": [{"pattern": "<situation>", '
            '"steps": ["<step1>", "<step2>"], '
            '"source": "<which trace provided evidence>"}]\n'
            "If you cannot determine a strategy, write a generic one based on "
            "the successful trace's approach (e.g. fewer turns, different commands). "
            "The strategy library is critical for agent improvement.\n"
        )
    elif focus == "context_length":
        focus_instruction = (
            "Focus on CONTEXT_LENGTH failures: investigate why the agent runs out "
            "of context and what could be done to prevent it. "
        )

    return (
        f"## Current batch statistics\n{stats}\n\n"
        f"## Recent meta-learning history\n{history_text}\n\n"
        f"## Investigation targets\n{investigation}\n\n"
        f"{focus_instruction}"
        "Investigate specific traces using inspect_trace and compare_traces. "
        "Use get_task_overview to see all trajectories for a task. "
        "When done, call submit_diagnosis with your findings."
    )


def _build_single_task_message(
    history_text: str,
    group: TaskGroup,
    category: str,
) -> str:
    """Build a focused user message for a single-task diagnosis worker."""
    lines: list[str] = [
        f"## Recent meta-learning history\n{history_text}\n",
        f"## Your assignment: analyse task **{group.task_key}** ({category})\n",
        f"Task statistics: {group.n_success}/{group.n_total} passed, "
        f"avg_reward={group.avg_reward:.3f}",
        "",
        "Trajectories:",
    ]
    for t in group.traces:
        tag_str = ",".join(t.failure_tags) if t.failure_tags else "-"
        lines.append(
            f"  {t.task_id}: reward={t.final_reward}, turns={t.turn_count}, "
            f"finish={t.finish_reason}, tags=[{tag_str}], success={t.success}"
        )

    lines.append("")

    if category == "partial":
        fail_ids = group.fail_ids()[:3]
        succ_ids = group.success_ids()[:3]
        lines.append(
            "This is a PARTIAL task — some trajectories succeed and some fail. "
            "You MUST:\n"
            f"  1. Use inspect_trace on at least one success ({succ_ids}) and one failure ({fail_ids}).\n"
            f"  2. Use compare_traces to contrast a success vs a failure.\n"
            "  3. Identify what the successful trajectory did differently.\n"
            "  4. Your submit_diagnosis MUST include `strategy_suggestions` — "
            "extract the winning approach as a reusable strategy with "
            '`pattern`, `steps`, and `source` fields.\n'
        )
    elif category == "all_fail":
        sample_ids = [t.task_id for t in group.traces[:3]]
        lines.append(
            "This task ALWAYS FAILS across all trajectories. Investigate:\n"
            f"  1. Use inspect_trace on a few traces ({sample_ids}) to understand the failure.\n"
            "  2. Identify the root cause: is it a planning issue, timeout, "
            "weak verification, repeated command failure, or wrong approach?\n"
            "  3. Submit a diagnosis with specific `root_cause_hypotheses` and "
            "affected trajectory IDs.\n"
            "  4. Do NOT submit strategy_suggestions for all-fail tasks "
            "(no successful trajectory to learn from).\n"
        )
    elif category == "all_pass":
        lines.append(
            "This task ALWAYS PASSES. Examine it briefly for regression "
            "context — what does a successful execution look like for this "
            "type of task? Keep your investigation short.\n"
        )

    lines.append(
        "Use inspect_trace, compare_traces, and get_task_overview as needed. "
        "When done, call submit_diagnosis with your findings."
    )
    return "\n".join(lines)


TOOL_DESCRIPTIONS = """\
Available diagnostic tools (call by outputting JSON):

1. {"tool": "inspect_trace", "task_id": "<id>"}
   -> View detailed events for a specific trace

2. {"tool": "compare_traces", "task_id_a": "<id>", "task_id_b": "<id>"}
   -> Side-by-side comparison of two traces

3. {"tool": "get_task_overview", "task_key": "<instance_id>"}
   -> Overview of all trajectories for a task (e.g. "749" shows 749-traj0..7)

4. {"tool": "submit_diagnosis", "diagnoses": [...]}
   -> Submit your final diagnosis (required to end the session)

Call ONE tool at a time. After reviewing results, call more tools or submit.
"""


def dispatch_tool(
    tool_call: dict[str, Any],
    traces: list[TraceRecord],
    history: DiagnosisHistory,
    override_base: str | None = None,
) -> str:
    """Dispatch a parsed tool call dict to the appropriate function."""
    name = tool_call.get("tool", "")

    if name == "inspect_trace":
        return inspect_trace(traces, tool_call.get("task_id", ""), tool_call.get("traj_id"))
    elif name == "compare_traces":
        return compare_traces(traces, tool_call.get("task_id_a", ""), tool_call.get("task_id_b", ""))
    elif name == "get_task_overview":
        return get_task_overview(traces, tool_call.get("task_key", ""))
    elif name == "get_failure_distribution":
        return get_failure_distribution(traces)
    elif name == "get_history":
        return get_history(history, tool_call.get("n_cycles", 5))
    elif name == "get_active_patches":
        return get_active_patches(override_base)
    elif name == "submit_diagnosis":
        return "__SUBMIT__"
    else:
        return (
            f"Unknown tool: '{name}'. Available: inspect_trace, compare_traces, "
            "get_task_overview, submit_diagnosis"
        )
