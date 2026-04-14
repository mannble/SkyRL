from __future__ import annotations

from dataclasses import dataclass, field

from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry
from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord


@dataclass(slots=True)
class MetaCycleResult:
    decision: str
    selected_failure_tags: list[str] = field(default_factory=list)
    candidate_modules: list[str] = field(default_factory=list)
    notes: str = ""


class MetaController:
    """Minimal controller skeleton for future diagnose->patch->promote loops."""

    def __init__(self, registry: ModuleRegistry) -> None:
        self.registry = registry

    def summarize_failures(self, traces: list[TraceRecord]) -> MetaCycleResult:
        failure_tags: dict[str, int] = {}
        for trace in traces:
            for tag in trace.failure_tags:
                failure_tags[tag] = failure_tags.get(tag, 0) + 1

        ordered_tags = [tag for tag, _count in sorted(failure_tags.items(), key=lambda item: item[1], reverse=True)]
        candidate_modules: list[str] = []
        if "sync_bottleneck" in ordered_tags:
            candidate_modules.extend(["parallelization_policy", "merge_policy"])
        if "verification_failure" in ordered_tags:
            candidate_modules.append("verification_policy")
        if "recovery_failure" in ordered_tags:
            candidate_modules.append("retry_policy")
        if "termination_failure" in ordered_tags:
            candidate_modules.append("finish_policy")
        if "planning_failure" in ordered_tags:
            candidate_modules.append("planner_policy")

        deduped_modules = [name for i, name in enumerate(candidate_modules) if name not in candidate_modules[:i]]
        return MetaCycleResult(
            decision="inspect",
            selected_failure_tags=ordered_tags,
            candidate_modules=[m for m in deduped_modules if m in {mod.name for mod in self.registry.list()}],
            notes="Heuristic summary only. Replace with diagnoser model later.",
        )
