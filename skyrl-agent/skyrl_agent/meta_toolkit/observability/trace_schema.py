from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ActionType = Literal[
    "thought",
    "tool_call",
    "tool_result",
    "plan_update",
    "verification",
    "finish",
    "error",
]


@dataclass(slots=True)
class TraceEvent:
    """A single structured event emitted during an agent trajectory."""

    turn_id: int
    role: str
    action_type: ActionType
    content: str = ""
    tool_name: str | None = None
    tool_args: str | None = None
    observation: str | None = None
    exit_code: int | None = None
    latency_ms: int | None = None
    idle_wait_ms: int | None = None
    reward_delta: float | None = None
    read_only_opportunity: bool = False
    fanout_opportunity: bool = False
    timeout_source: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceRecord:
    """Structured trajectory record used by diagnosis and patch planning."""

    task_id: str
    agent_version: str
    policy_version: str
    success: bool
    final_reward: float | None = None
    turn_count: int = 0
    finish_reason: str | None = None
    sync_bottleneck_score: float = 0.0
    failure_tags: list[str] = field(default_factory=list)
    tool_calls: int = 0
    tool_failures: int = 0
    verification_attempted: bool = False
    retry_attempted: bool = False
    wall_clock_latency_ms: int | None = None
    idle_wait_ms: int | None = None
    events: list[TraceEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
