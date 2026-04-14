"""Patch planner: generates PatchCandidate objects from DiagnosisResult + ModuleRegistry.

Supports patching of:
  - planner_policy, retry_policy, verification_policy, finish_policy (YAML overrides)
  - system_prompt_overrides, strategy_library (prompt-level)
  - behavior_policy (code-level behavioral hooks)
"""

from __future__ import annotations

from dataclasses import dataclass

from skyrl_agent.meta_toolkit.diagnosis import DiagnosisResult
from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry
from skyrl_agent.meta_toolkit.editing.patch_schema import ChangeType, PatchCandidate, PatchFileEdit

_ALLOWED_PHASE1_MODULES: frozenset[str] = frozenset([
    "strategy_library",
    "hook:before_llm_call", "hook:before_execute", "hook:after_execute",
    "hook:on_timeout", "hook:on_parse_error", "hook:after_round",
])

_MODULE_OVERRIDE_FILE: dict[str, str] = {
    "strategy_library": "strategy_library.yaml",
}

_PROBLEM_MODULE_PRIORITY: dict[str, list[str]] = {
    "sync_bottleneck": ["strategy_library"],
    "timeout": ["strategy_library"],
    "verification_failure": ["strategy_library"],
    "recovery_failure": ["strategy_library"],
    "termination_failure": ["strategy_library"],
    "planning_failure": ["strategy_library"],
    "context_overload": ["strategy_library"],
    "tool_exhaustion": ["strategy_library"],
    "low_reward": ["strategy_library"],
    "general": ["strategy_library"],
}


@dataclass
class PlannerConfig:
    """Configuration for the patch planner."""

    # Base path where SkyRL stores its override YAML files
    override_base: str = "/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2"

    # Whether to allow patching modules beyond the Phase-1 whitelist
    allow_all_modules: bool = False


class PatchPlanner:
    """Generate PatchCandidate(s) from DiagnosisResult(s).

    Phase 1 behaviour:
      - Only proposes edits to the four core policy modules via YAML override files.
      - Each candidate is low-risk and can be tested via canary eval.
    """

    def __init__(self, registry: ModuleRegistry, config: PlannerConfig | None = None) -> None:
        self.registry = registry
        self.config = config or PlannerConfig()

    def plan_from_diagnosis(
        self, diagnosis: DiagnosisResult
    ) -> list[PatchCandidate]:
        """Generate PatchCandidates from a single DiagnosisResult.

        Returns 0–2 candidates, prioritising the module most likely to fix the problem.
        """
        candidates: list[PatchCandidate] = []

        # Pick module order based on problem type, filtered by Phase-1 whitelist
        priority = _PROBLEM_MODULE_PRIORITY.get(
            diagnosis.problem_type, ["planner_policy"]
        )
        allowed = self._allowed_modules()
        modules_to_try = [m for m in priority if m in allowed]

        for module_name in modules_to_try[:2]:  # at most 2 candidates
            try:
                module = self.registry.get(module_name)
            except KeyError:
                continue

            override_file = _MODULE_OVERRIDE_FILE.get(module_name)
            if not override_file:
                continue

            # Compute absolute path for Phase-1 override file
            override_path = f"{self.config.override_base}/{override_file}"

            candidate = self._build_candidate(module, override_path, diagnosis)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def plan_from_batch(
        self, diagnoses: list[DiagnosisResult]
    ) -> list[PatchCandidate]:
        """Generate PatchCandidates from multiple DiagnosisResults.

        De-duplicates module targets across diagnoses and returns at most
        one candidate per module.
        """
        seen_modules: set[str] = set()
        candidates: list[PatchCandidate] = []

        for diagnosis in diagnoses:
            for candidate in self.plan_from_diagnosis(diagnosis):
                for target in candidate.target_modules:
                    if target not in seen_modules:
                        seen_modules.add(target)
                        candidates.append(candidate)
                        break  # one candidate per module

        return candidates

    def _allowed_modules(self) -> frozenset[str]:
        if self.config.allow_all_modules:
            return frozenset(m.name for m in self.registry.list())
        return _ALLOWED_PHASE1_MODULES

    def _build_candidate(
        self,
        module,
        override_path: str,
        diagnosis: DiagnosisResult,
    ) -> PatchCandidate | None:
        """Build a PatchCandidate that writes a YAML override hint for the given module."""

        intent = (
            f"Fix {diagnosis.problem_type} (confidence={diagnosis.confidence:.2f}) "
            f"by patching {module.name}. "
            f"Hypotheses: {'; '.join(diagnosis.root_cause_hypotheses)}"
        )

        file_edit = PatchFileEdit(
            path=override_path,
            change_type="modify",
            intent=intent,
        )

        # Validate that the override path is in the module's owned files
        # Phase-1: we relax this check because override files live in SkyRL side,
        # not in the Terminus source tree.
        return PatchCandidate(
            target_modules=[module.name],
            files=[file_edit],
            risk="low",
            required_tests=["canary_eval"],
            rollback_if=["canary_regression"],
            notes=intent,
        )
