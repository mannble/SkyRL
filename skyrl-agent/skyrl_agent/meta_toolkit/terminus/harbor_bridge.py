from __future__ import annotations

from typing import Any

from skyrl_agent.meta_toolkit.observability.trace_schema import TraceEvent, TraceRecord


def _infer_failure_tags(stop_reason: str, successful: bool, results: Any) -> list[str]:
    tags: list[str] = []
    if successful:
        return tags

    exc_type = None
    if results is not None and getattr(results, "exception_info", None) is not None:
        exc_type = getattr(results.exception_info, "exception_type", None)

    if stop_reason == "agent_timeout":
        tags.extend(["termination_failure", "sync_bottleneck"])
    elif stop_reason == "context_length":
        tags.append("context_overload")
    else:
        tags.append("recovery_failure")

    if exc_type == "ContextLengthExceededError" and "context_overload" not in tags:
        tags.append("context_overload")
    if exc_type == "AgentTimeoutError" and "sync_bottleneck" not in tags:
        tags.append("sync_bottleneck")
    if exc_type and "Timeout" in exc_type and "sync_bottleneck" not in tags:
        tags.append("sync_bottleneck")
    return tags


def build_trace_from_harbor_trial(
    *,
    trajectory_id: Any,
    agent_version: str,
    prompt: Any,
    results: Any,
    reward: float,
    stop_reason: str,
    num_turns: int,
    summarization_count: int,
    successful: bool,
) -> TraceRecord:
    """Build a minimal TraceRecord from Harbor trial outputs."""

    task_id = f"{trajectory_id.instance_id}-traj{trajectory_id.repetition_id}"
    messages = []
    if results is not None and getattr(results, "agent_result", None) is not None:
        messages = results.agent_result.metadata.get("all_messages", []) or []

    events: list[TraceEvent] = []
    for turn_id, message in enumerate(messages):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        action_type = "thought"
        if role == "tool":
            action_type = "tool_result"
        events.append(
            TraceEvent(
                turn_id=turn_id,
                role=role,
                action_type=action_type,
                content=content if isinstance(content, str) else str(content),
            )
        )

    tool_calls = sum(1 for message in messages if message.get("role") == "assistant")
    tool_failures = 0 if successful else int(stop_reason in {"error", "agent_timeout"})
    sync_bottleneck_score = 1.0 if stop_reason == "agent_timeout" else 0.0
    if results is not None and getattr(results, "exception_info", None) is not None:
        exc_type = getattr(results.exception_info, "exception_type", None)
        if exc_type == "AgentTimeoutError":
            sync_bottleneck_score = 1.0

    return TraceRecord(
        task_id=task_id,
        agent_version=agent_version,
        policy_version="unknown",
        success=successful,
        final_reward=float(reward),
        turn_count=num_turns,
        finish_reason=stop_reason,
        sync_bottleneck_score=sync_bottleneck_score,
        failure_tags=_infer_failure_tags(stop_reason, successful, results),
        tool_calls=tool_calls,
        tool_failures=tool_failures,
        verification_attempted=False,
        retry_attempted=False,
        events=events,
        metadata={
            "prompt": prompt,
            "summarization_count": summarization_count,
        },
    )
