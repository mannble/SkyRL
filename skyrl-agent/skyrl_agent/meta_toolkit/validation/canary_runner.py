"""Canary runner: evaluate patch candidates on a small task set before promotion.

Baseline trials run ONCE and are shared across all candidates in a cycle.
Candidate trials run concurrently per candidate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

from skyrl_agent.meta_toolkit.validation.comparison import ComparisonConfig, ComparisonEngine
from skyrl_agent.meta_toolkit.validation.patch_eval_result import PatchEvalResult

TrialRunnerFn = Callable[[Any, dict[str, Any] | None], Awaitable[float]]


@dataclass
class CanaryConfig:
    num_tasks: int = 4
    n_samples_per_task: int = 3
    trial_timeout_sec: int = 600
    max_candidates: int = 4
    override_base: str = "/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2"
    canary_log_dir: str = "/tmp/skyrl-logs/canary"
    comparison: ComparisonConfig | None = None


@dataclass
class CanaryRunResult:
    candidate_id: str
    patch_eval_result: PatchEvalResult
    before_scores: list[float] = field(default_factory=list)
    after_scores: list[float] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    error: str | None = None


class CanaryRunner:
    """Canary evaluator with shared baseline.

    Usage (from controller):
        baseline = await canary.run_baseline(task_paths, baseline_overrides)
        for candidate in candidates:
            result = await canary.evaluate_against_baseline(
                baseline, candidate_id, patch_files, candidate_overrides
            )
    """

    def __init__(
        self,
        config: CanaryConfig | None = None,
        trial_fn: TrialRunnerFn | None = None,
    ) -> None:
        self.config = config or CanaryConfig()
        self._comparison = ComparisonEngine(self.config.comparison)
        self._trial_fn = trial_fn

    @property
    def has_trial_runner(self) -> bool:
        return self._trial_fn is not None

    @staticmethod
    def summarize_scores(scores: list[float]) -> dict[str, float]:
        """Return average reward and pass rate for per-task score lists."""
        if not scores:
            return {"avg_reward": 0.0, "pass_rate": 0.0}
        passed = sum(1 for s in scores if s > 0)
        return {
            "avg_reward": sum(scores) / len(scores),
            "pass_rate": passed / len(scores),
        }

    # ------------------------------------------------------------------
    # Public API: run baseline once, evaluate candidates against it
    # ------------------------------------------------------------------

    async def run_baseline(
        self,
        task_paths: list[Any],
        baseline_overrides: dict[str, Any] | None = None,
    ) -> list[float] | None:
        """Run baseline trials ONCE. Returns per-task average scores, or None on error."""
        if not task_paths or self._trial_fn is None:
            return None

        n = self.config.n_samples_per_task
        total = len(task_paths) * n
        logger.info(
            f"Canary baseline: {len(task_paths)} tasks × {n} samples = {total} trials"
        )
        try:
            return await self._run_trial_batch(task_paths, baseline_overrides, label="baseline")
        except Exception as e:
            logger.error(f"Canary baseline failed: {e}")
            return None

    async def evaluate_against_baseline(
        self,
        baseline_scores: list[float],
        task_paths: list[Any],
        candidate_id: str,
        patch_files: list[str],
        candidate_overrides: dict[str, Any] | None = None,
    ) -> CanaryRunResult:
        """Evaluate one candidate against pre-computed baseline scores."""
        if self._trial_fn is None:
            return _placeholder_result(candidate_id, "No trial runner configured")

        task_labels = [str(p) for p in task_paths]
        n = self.config.n_samples_per_task
        logger.info(
            f"Canary {candidate_id}: {len(task_paths)} tasks × {n} samples "
            f"= {len(task_paths) * n} candidate trials"
        )

        try:
            after_scores = await self._run_trial_batch(
                task_paths, candidate_overrides, label=f"candidate:{candidate_id}"
            )

            eval_result = self._comparison.evaluate(
                baseline_scores,
                after_scores,
                task_ids=task_labels,
                metadata={"candidate_id": candidate_id, "patch_files": patch_files},
            )

            logger.info(
                f"Canary {candidate_id}: "
                f"baseline={_fmt_scores(baseline_scores)} "
                f"candidate={_fmt_scores(after_scores)} "
                f"delta={eval_result.delta_score:.4f} "
                f"regression={eval_result.regression}"
            )

            return CanaryRunResult(
                candidate_id=candidate_id,
                patch_eval_result=eval_result,
                before_scores=baseline_scores,
                after_scores=after_scores,
                task_ids=task_labels,
            )

        except Exception as e:
            logger.error(f"CanaryRunner error for {candidate_id}: {e}")
            return CanaryRunResult(
                candidate_id=candidate_id,
                patch_eval_result=PatchEvalResult(
                    before_score=0.0, after_score=0.0, delta_score=0.0,
                    regression=True, notes=f"Canary evaluation failed: {e}",
                ),
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Legacy single-candidate API (backward compat)
    # ------------------------------------------------------------------

    async def evaluate_candidate(
        self,
        candidate_id: str,
        patch_files: list[str],
        task_paths: list[Any] | None = None,
        baseline_overrides: dict[str, Any] | None = None,
        candidate_overrides: dict[str, Any] | None = None,
    ) -> CanaryRunResult:
        if not task_paths:
            return _placeholder_result(candidate_id, "No canary tasks available")
        if self._trial_fn is None:
            return _placeholder_result(candidate_id, "No trial runner configured")

        baseline_scores = await self.run_baseline(task_paths, baseline_overrides)
        if baseline_scores is None:
            return _placeholder_result(candidate_id, "Baseline run failed")

        return await self.evaluate_against_baseline(
            baseline_scores, task_paths, candidate_id, patch_files, candidate_overrides,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_trial_batch(
        self,
        task_paths: list[Any],
        overrides: dict[str, Any] | None,
        label: str = "",
    ) -> list[float]:
        """Run n_samples_per_task trials per task concurrently, return per-task avg reward."""
        assert self._trial_fn is not None
        n_samples = self.config.n_samples_per_task
        total_trials = len(task_paths) * n_samples
        completed = 0
        succeeded = 0
        reward_sum = 0.0
        _progress_lock = asyncio.Lock()

        async def _single(ti: int, si: int, tp: Any) -> tuple[int, float]:
            nonlocal completed, succeeded, reward_sum
            try:
                reward = await self._trial_fn(tp, overrides or None)
                async with _progress_lock:
                    completed += 1
                    reward_sum += reward
                    if reward > 0:
                        succeeded += 1
                    if completed % max(1, total_trials // 10) == 0 or completed == total_trials:
                        logger.info(
                            f"Canary [{label}] progress: {completed}/{total_trials} "
                            f"({100*completed/total_trials:.0f}%) "
                            f"avg_reward={reward_sum/completed:.3f} "
                            f"pass_rate={succeeded}/{completed}"
                        )
                return ti, float(reward)
            except Exception as e:
                async with _progress_lock:
                    completed += 1
                    if completed % max(1, total_trials // 10) == 0 or completed == total_trials:
                        logger.info(
                            f"Canary [{label}] progress: {completed}/{total_trials} "
                            f"({100*completed/total_trials:.0f}%) "
                            f"avg_reward={reward_sum/max(1,completed-1):.3f} "
                            f"pass_rate={succeeded}/{completed}"
                        )
                logger.warning(
                    f"Canary [{label}] task {ti+1}/{len(task_paths)} "
                    f"sample {si+1}/{n_samples} failed: {e}"
                )
                return ti, 0.0

        results = await asyncio.gather(*[
            _single(ti, si, tp)
            for ti, tp in enumerate(task_paths)
            for si in range(n_samples)
        ])

        per_task: dict[int, list[float]] = {}
        for task_idx, reward in results:
            per_task.setdefault(task_idx, []).append(reward)

        per_task_avg: list[float] = []
        for i, task_path in enumerate(task_paths):
            values = per_task.get(i, [0.0])
            avg = sum(values) / max(len(values), 1)
            per_task_avg.append(avg)
            passed = sum(1 for v in values if v > 0)
            logger.info(
                f"Canary [{label}] task[{i}] avg_reward={avg:.3f} "
                f"pass_rate={passed}/{len(values)} path={task_path}"
            )

        return per_task_avg


def _placeholder_result(candidate_id: str, reason: str) -> CanaryRunResult:
    return CanaryRunResult(
        candidate_id=candidate_id,
        patch_eval_result=PatchEvalResult(
            before_score=0.0, after_score=0.0, delta_score=0.0,
            regression=True, notes=f"{reason}; skipping evaluation",
        ),
    )


def _fmt_scores(scores: list[float]) -> str:
    if not scores:
        return "[]"
    avg = sum(scores) / len(scores)
    return f"[avg={avg:.3f}, n={len(scores)}]"
