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

    # Per-task regression guard: reject if any single task's avg reward drops
    # by more than this amount vs baseline.  Set to 0.0 to disable.
    max_per_task_drop: float = 0.55


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

        # Per-task regression guard
        worst_task_drop = 0.0
        worst_task_idx = -1
        if self.config.max_per_task_drop > 0:
            for idx, (b, a) in enumerate(zip(before_scores, after_scores)):
                drop = b - a
                if drop > worst_task_drop:
                    worst_task_drop = drop
                    worst_task_idx = idx

        per_task_regressed = (
            self.config.max_per_task_drop > 0
            and worst_task_drop > self.config.max_per_task_drop
        )
        if per_task_regressed:
            regression = True

        if per_task_regressed:
            tid = task_ids[worst_task_idx] if task_ids and worst_task_idx < len(task_ids) else f"#{worst_task_idx}"
            notes = (
                (notes + " " if notes else "")
                + f"PER-TASK REGRESSION: task {tid} dropped {worst_task_drop:.4f} "
                f"(threshold {self.config.max_per_task_drop})."
            )
        elif regression:
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
