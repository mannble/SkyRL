"""Compare before/after scores and detect regressions."""

from __future__ import annotations

from dataclasses import dataclass

from skyrl_agent.meta_toolkit.validation.patch_eval_result import PatchEvalResult


@dataclass
class ComparisonConfig:
    """Thresholds for regression detection."""

    # Minimum delta required to consider a patch beneficial
    min_delta: float = 0.0

    # If delta < regression_threshold, mark as regression
    regression_threshold: float = -0.05

    # Require at least this many tasks in the canary set
    min_canary_tasks: int = 1


class ComparisonEngine:
    """Compare before/after scores and produce a PatchEvalResult."""

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config = config or ComparisonConfig()

    def evaluate(
        self,
        before_scores: list[float],
        after_scores: list[float],
        task_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> PatchEvalResult:
        """Compute comparison metrics between before and after scores.

        Args:
            before_scores: List of per-task scores without the patch.
            after_scores: List of per-task scores with the patch.
            task_ids: Optional list of task IDs for the canary set.
            metadata: Extra fields to attach to the result.

        Returns:
            PatchEvalResult with delta, regression flag, and notes.
        """
        if not before_scores or not after_scores:
            return PatchEvalResult(
                before_score=0.0,
                after_score=0.0,
                delta_score=0.0,
                regression=True,
                notes="Empty score lists provided",
            )

        if len(before_scores) != len(after_scores):
            n = min(len(before_scores), len(after_scores))
            before_scores = before_scores[:n]
            after_scores = after_scores[:n]
            notes = f"Length mismatch truncated to {n} tasks"
        else:
            notes = ""

        before_avg = sum(before_scores) / len(before_scores)
        after_avg = sum(after_scores) / len(after_scores)
        delta = after_avg - before_avg

        before_successes = sum(1 for s in before_scores if s > 0)
        after_successes = sum(1 for s in after_scores if s > 0)

        regression = self._is_regression(delta, len(after_scores))

        if regression:
            notes = (notes + " " if notes else "") + "REGRESSION detected."
        elif delta > self.config.min_delta:
            notes = (notes + " " if notes else "") + f"Improvement: delta={delta:.4f}"
        else:
            notes = (notes + " " if notes else "") + f"No meaningful change: delta={delta:.4f}"

        return PatchEvalResult(
            before_score=before_avg,
            after_score=after_avg,
            delta_score=delta,
            regression=regression,
            notes=notes.strip(),
            num_tasks=len(after_scores),
            before_successes=before_successes,
            after_successes=after_successes,
            metadata=metadata or {},
        )

    def _is_regression(self, delta: float, num_tasks: int) -> bool:
        if num_tasks < self.config.min_canary_tasks:
            return True
        return delta < self.config.regression_threshold
