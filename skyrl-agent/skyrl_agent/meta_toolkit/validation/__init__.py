"""Validation names, canary evaluation, and comparison for patch promotion."""

from skyrl_agent.meta_toolkit.validation.comparison import ComparisonConfig, ComparisonEngine
from skyrl_agent.meta_toolkit.validation.patch_eval_result import PatchEvalResult

from .canary_runner import CanaryConfig, CanaryRunResult, CanaryRunner

SMOKE_TERMINAL = "smoke_terminal"
PLAN_GENERATION_SMOKE = "plan_generation_smoke"
RETRY_RECOVERY_SMOKE = "retry_recovery_smoke"
VERIFICATION_SMOKE = "verification_smoke"
FINISH_DECISION_SMOKE = "finish_decision_smoke"
PARALLELIZATION_SMOKE = "parallelization_smoke"
MERGE_SMOKE = "merge_smoke"
TB2_CANARY = "tb2_canary"

__all__ = [
    "CanaryConfig",
    "CanaryRunResult",
    "CanaryRunner",
    "ComparisonConfig",
    "ComparisonEngine",
    "FINISH_DECISION_SMOKE",
    "MERGE_SMOKE",
    "PARALLELIZATION_SMOKE",
    "PLAN_GENERATION_SMOKE",
    "PatchEvalResult",
    "RETRY_RECOVERY_SMOKE",
    "SMOKE_TERMINAL",
    "TB2_CANARY",
    "VERIFICATION_SMOKE",
]
