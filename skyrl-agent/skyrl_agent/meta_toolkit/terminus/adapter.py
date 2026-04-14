from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord


@dataclass(slots=True)
class TerminusTaskInput:
    """Normalized task input passed from SkyRL-side rollouts into a Terminus runtime."""

    task_id: str
    prompt: Any
    env_class: str | None = None
    env_extras: dict[str, Any] = field(default_factory=dict)
    sampling_params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TerminusRunResult:
    """Structured result returned by a Terminus runtime.

    `messages` is kept for compatibility with existing SkyRL-Agent post-processing.
    `transitions` is intentionally left generic so a future Terminus integration can
    reuse the existing training-data conversion path.
    """

    task_id: str
    trace: TraceRecord
    reward: float | bool
    success: bool
    finish_reason: str | None
    messages: list[dict[str, str]] = field(default_factory=list)
    transitions: list[Any] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    eval_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TerminusRuntime(Protocol):
    """Protocol for a Terminus-like runtime that can be driven by the adapter."""

    async def run_task(self, task_input: TerminusTaskInput) -> TerminusRunResult:
        """Run one task and return a normalized result bundle."""


class TerminusAdapter:
    """Thin bridge between SkyRL generator inputs and a Terminus-style runtime.

    This class deliberately does not know how Terminus internally plans, retries,
    verifies, or parallelizes work. It only normalizes task inputs, executes the
    runtime, and converts outputs into a shape that downstream SkyRL-style code can
    consume.
    """

    def __init__(self, runtime: TerminusRuntime, agent_version: str) -> None:
        self.runtime = runtime
        self.agent_version = agent_version

    def build_task_input(
        self,
        *,
        task_id: str,
        prompt: Any,
        env_class: str | None = None,
        env_extras: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TerminusTaskInput:
        return TerminusTaskInput(
            task_id=task_id,
            prompt=prompt,
            env_class=env_class,
            env_extras=env_extras or {},
            sampling_params=sampling_params or {},
            metadata=metadata or {},
        )

    def task_input_from_skyrl_generator(
        self,
        *,
        prompt: Any,
        trajectory_id: Any,
        env_class: str | None = None,
        env_extras: dict[str, Any] | None = None,
        sampling_params: dict[str, Any] | None = None,
        batch_metadata: Any | None = None,
    ) -> TerminusTaskInput:
        metadata = {
            "batch_metadata": batch_metadata,
            "agent_version": self.agent_version,
        }
        task_id = self._task_id_from_trajectory(trajectory_id)
        return self.build_task_input(
            task_id=task_id,
            prompt=prompt,
            env_class=env_class,
            env_extras=env_extras,
            sampling_params=sampling_params,
            metadata=metadata,
        )

    async def run_task_input(self, task_input: TerminusTaskInput) -> TerminusRunResult:
        result = await self.runtime.run_task(task_input)
        result.trace.agent_version = self.agent_version
        if not result.trace.task_id:
            result.trace.task_id = task_input.task_id
        result.metadata.setdefault("agent_version", self.agent_version)
        return result

    def to_skyrl_result(self, run_result: TerminusRunResult) -> dict[str, Any]:
        """Convert a Terminus result into the SkyRL-Agent-compatible result shape."""

        return {
            "instance_id": run_result.task_id,
            "trajectory_id": run_result.metadata.get("trajectory_id", 0),
            "messages": run_result.messages,
            "transitions": run_result.transitions,
            "result": run_result.result,
            "error": run_result.error,
            "finish": bool(run_result.success),
            "finish_reason": run_result.finish_reason,
            "reward": run_result.reward,
            "eval_error": run_result.eval_error,
            "trace_record": run_result.trace,
            "metadata": {
                **run_result.metadata,
                "agent_version": self.agent_version,
                "sync_bottleneck_score": run_result.trace.sync_bottleneck_score,
            },
        }

    @staticmethod
    def _task_id_from_trajectory(trajectory_id: Any) -> str:
        instance_id = getattr(trajectory_id, "instance_id", None)
        repetition_id = getattr(trajectory_id, "repetition_id", None)
        if instance_id is None:
            return str(trajectory_id)
        if repetition_id is None:
            return str(instance_id)
        return f"{instance_id}-traj{repetition_id}"
