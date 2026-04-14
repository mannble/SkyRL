from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CapabilityRuntimeConfig:
    """Safety shell for self-evolving logic.

    This runtime is where we allow models to evolve *policy logic* while keeping
    execution, shared-state mutation, and evaluation behind stable interfaces.
    """

    allow_parallelization_policy: bool = True
    allow_merge_policy: bool = True
    allow_retry_policy: bool = True
    allow_verification_policy: bool = True
    allow_finish_policy: bool = True
    require_state_manager: bool = True
    require_conflict_guard: bool = True


class CapabilityRuntime:
    """Thin orchestrator boundary for policy-level self-evolution.

    The implementation is intentionally minimal for now. A future TerminusAdapter
    can depend on this class to wire registered policy modules into runtime calls.
    """

    def __init__(self, config: CapabilityRuntimeConfig | None = None) -> None:
        self.config = config or CapabilityRuntimeConfig()

    def build_runtime_contract(self) -> dict[str, Any]:
        return {
            "allow_parallelization_policy": self.config.allow_parallelization_policy,
            "allow_merge_policy": self.config.allow_merge_policy,
            "allow_retry_policy": self.config.allow_retry_policy,
            "allow_verification_policy": self.config.allow_verification_policy,
            "allow_finish_policy": self.config.allow_finish_policy,
            "require_state_manager": self.config.require_state_manager,
            "require_conflict_guard": self.config.require_conflict_guard,
        }
