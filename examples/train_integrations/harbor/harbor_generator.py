import asyncio
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

from skyrl.train.generators.base import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.train.utils.rate_limiter import create_rate_limiter

from tqdm import tqdm

# Suppress LiteLLM verbose logging
import litellm
import logging

litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# We have N retries for each trial, if one of the rollout (out of n_samples_per_prompt) fails
# after N attempts, we skip this prompt altogether.
MAX_NUM_RETRIES_PER_TRIAL = 2
_IGNORED_PATCH_PROBLEM_TYPES = {"context_overload", "context_length"}


def _normalize_problem_type(problem_type: str) -> str:
    return str(problem_type or "").strip().lower().replace("-", "_").replace(" ", "_")


@dataclass
class MetaLoopConfig:
    """Configuration for the meta-learning closed loop."""

    enabled: bool = True
    interval_batches: int = 20  # Run meta cycle every N batches
    max_candidates: int = 2  # Max patch candidates per cycle
    canary_num_tasks: int = 4  # Tasks for canary eval
    canary_n_samples: int = 3  # Trials per task per side (baseline/candidate)
    canary_diagnosed_ratio: float = 0.75  # Fraction of canary failures from diagnosed tasks
    canary_min_undiagnosed: int = 2  # Keep a few unseen tasks for generalization check
    override_base: str = "/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2"
    log_dir: str = "/tmp/skyrl-logs"

    # Diagnosis worker sampling (per meta cycle)
    diagnosis_num_workers: int = 16
    diagnosis_partial_tasks: int = 8
    diagnosis_all_fail_tasks: int = 8
    diagnosis_all_pass_tasks: int = 0

    # Phase 3: LLM-based diagnosis & patch planning
    llm_model: str = ""  # e.g. "gpt-4o", "qwen3-32b" — empty = use rule-based
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

    # Patch library limits (enforced by PatchPruner)
    max_strategies: int = 32
    max_hook_groups: int = 8
    max_hooks_per_category: Optional[int] = None  # legacy alias for max_hook_groups

    @property
    def hook_group_budget(self) -> int:
        return (
            self.max_hooks_per_category
            if self.max_hooks_per_category is not None
            else self.max_hook_groups
        )

    @classmethod
    def from_cfg(cls, cfg) -> "MetaLoopConfig":
        if cfg is None:
            return cls()
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True) or {}
        elif hasattr(cfg, "__dataclass_fields__"):
            from dataclasses import asdict
            cfg = asdict(cfg)
        elif not isinstance(cfg, dict):
            cfg = vars(cfg)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class MetaTrainingSample:
    """A meta-reasoning sample to be mixed into the RL training batch.

    Planning samples from the same cycle share an ``instance_id`` with different
    ``repetition_id`` values, forming a GRPO group.  Each gets a per-candidate
    reward (canary delta_score), enabling the model to learn which planning
    responses produce better patches.
    """
    messages: List[dict]  # full chat: [system, user, assistant]
    reward: float
    role: str = ""  # "diagnosis" or "planning"
    instance_id: str = ""
    repetition_id: int = 0


# =============================================================================
# Closed-loop meta-learning controller
# =============================================================================


class _MetaLoopController:
    """Closed-loop meta-learning controller that orchestrates diagnose→plan→patch→canary→promote.

    This class is instantiated by HarborGenerator when meta.enabled=true.
    """

    def __init__(
        self,
        registry,
        config: MetaLoopConfig,
        trial_config_template: DictConfig | None = None,
        rate_limiter=None,
        adapter=None,
        agent_name: str = "",
    ) -> None:
        from skyrl_agent.meta_toolkit.editing import PatchExecutor
        from skyrl_agent.meta_toolkit.validation import CanaryRunner, CanaryConfig, ComparisonConfig
        from skyrl_agent.meta_toolkit.runtime import Promoter, PromoterConfig
        from skyrl_agent.meta_toolkit.runtime import PatchVersionControl, PatchVersionControlConfig

        # Create a timestamped subdirectory for this run's meta logs
        from datetime import datetime as _dt
        _run_ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        config.log_dir = str(Path(config.log_dir) / f"meta_run_{_run_ts}")
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"MetaLoop: log directory = {config.log_dir}")

        self.config = config
        self._registry = registry
        self._adapter = adapter
        self._override_base = Path(config.override_base)
        self._trial_config_template = trial_config_template
        self._rate_limiter = rate_limiter

        self._llm_client = None
        self._use_llm = bool(config.llm_model)
        self._pruner = None
        accepted_patches_dir = Path(config.log_dir) / "accepted_patches"

        if self._use_llm:
            from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient, MetaLLMConfig
            from skyrl_agent.meta_toolkit.diagnosis.interactive_diagnoser import InteractiveDiagnoser
            from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import DiagnosisHistory
            from skyrl_agent.meta_toolkit.editing.llm_patch_planner import LLMPatchPlanner
            from skyrl_agent.meta_toolkit.editing.patch_pruner import PatchPruner

            llm_cfg = MetaLLMConfig(
                model=config.llm_model,
                base_url=config.llm_base_url,
                api_key=config.llm_api_key,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            self._llm_client = MetaLLMClient(llm_cfg)
            history_path = f"{config.log_dir}/diagnosis_history.jsonl"
            self._diagnosis_history = DiagnosisHistory(persist_path=history_path)
            self._diagnoser = InteractiveDiagnoser(
                self._llm_client,
                history=self._diagnosis_history,
                num_workers=config.diagnosis_num_workers,
                max_partial_tasks=config.diagnosis_partial_tasks,
                max_all_fail_tasks=config.diagnosis_all_fail_tasks,
                max_all_pass_tasks=config.diagnosis_all_pass_tasks,
                override_base=config.override_base,
            )
            self._llm_planner = LLMPatchPlanner(
                self._llm_client, registry, config.override_base,
                agent_name=agent_name,
            )
            self._pruner = PatchPruner(
                self._llm_client,
                config.override_base,
                max_strategies=config.max_strategies,
                max_hook_groups=config.hook_group_budget,
            )
            logger.info(f"MetaLoop: interactive LLM mode (model={config.llm_model})")
        else:
            from skyrl_agent.meta_toolkit.diagnosis import RuleBasedDiagnoser
            from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import DiagnosisHistory
            from skyrl_agent.meta_toolkit.editing import PatchPlanner
            from skyrl_agent.meta_toolkit.editing.patch_planner import PlannerConfig

            self._diagnosis_history = DiagnosisHistory()
            self._diagnoser = RuleBasedDiagnoser()
            self._rule_planner = PatchPlanner(
                registry,
                config=PlannerConfig(override_base=config.override_base),
            )
            logger.info("MetaLoop: rule-based mode (no LLM model configured)")

        self._executor = PatchExecutor(
            override_base=config.override_base,
        )

        trial_fn = self._build_trial_fn() if trial_config_template is not None else None
        self._canary = CanaryRunner(
            config=CanaryConfig(
                num_tasks=config.canary_num_tasks,
                n_samples_per_task=config.canary_n_samples,
                override_base=config.override_base,
                comparison=ComparisonConfig(),
            ),
            trial_fn=trial_fn,
        )
        self._vc = PatchVersionControl(
            PatchVersionControlConfig(
                override_base=config.override_base,
            )
        )
        self._promoter = Promoter(
            config=PromoterConfig(
                accepted_patches_dir=str(accepted_patches_dir),
            ),
            version_control=self._vc,
        )

        self._batch_counter = 0
        self._candidate_counter = 0
        self._accumulated_traces: list = []
        self._trace_writer_initialized = False

        # Pre-load existing patches from override_base into _ACTIVE_META_OVERRIDES
        # so that baseline canary AND main RL sampling both use historical patches
        # even after a training process restart.
        self._preload_active_overrides()

        if trial_fn is not None:
            logger.info("MetaLoop: canary enabled (real Harbor Trial evaluation)")
        else:
            logger.info("MetaLoop: canary placeholder (no trial runner)")

    @staticmethod
    def _filter_patchable_diagnoses(diagnoses: list) -> list:
        """Drop diagnosis types that should not produce meta patches."""
        return [
            d for d in diagnoses
            if _normalize_problem_type(getattr(d, "problem_type", "")) not in _IGNORED_PATCH_PROBLEM_TYPES
        ]

    def _build_trial_fn(self):
        """Build an async callable that runs a single Harbor Trial and returns its reward.

        The callable captures trial_config_template and rate_limiter from the
        controller so CanaryRunner stays decoupled from Harbor internals.
        """
        template = self._trial_config_template
        rate_limiter = self._rate_limiter

        async def _run_single_trial(
            task_path: Any, meta_overrides: dict[str, Any] | None = None
        ) -> float:
            raw = deepcopy(template)
            if isinstance(raw, DictConfig):
                config = raw
            else:
                config = OmegaConf.create(raw)
            OmegaConf.set_struct(config, False)

            config["task"] = {"path": task_path}
            config["agent"]["kwargs"]["session_id"] = uuid4().hex

            # Keep meta overrides as plain Python dicts and inject them AFTER
            # OmegaConf resolution. This prevents strings like "${CSV_HASH}"
            # inside strategy text from being treated as OmegaConf interpolation.
            overrides_for_agent: dict[str, Any] | None = None
            hooks_dict: dict[str, Any] | None = None
            if meta_overrides:
                overrides_for_agent = deepcopy(meta_overrides)
                # Extract hooks and pass them separately
                hooks_raw = overrides_for_agent.pop("_meta_hooks", None)
                if isinstance(hooks_raw, dict) and hooks_raw:
                    hooks_dict = hooks_raw

            config_dict = OmegaConf.to_container(config, resolve=True)
            if not isinstance(config_dict, dict):
                config_dict = {}
            agent_kwargs = config_dict.setdefault("agent", {}).setdefault("kwargs", {})
            if overrides_for_agent:
                agent_kwargs["meta_overrides"] = overrides_for_agent
            if hooks_dict:
                agent_kwargs["meta_hooks"] = hooks_dict
            trial_config = TrialConfig.model_validate(config_dict)
            trial = Trial(trial_config)

            if rate_limiter is not None:
                async with rate_limiter:
                    results = await trial.run()
            else:
                results = await trial.run()

            exc_type = (
                results.exception_info.exception_type if results.exception_info else None
            )
            if exc_type in ("AgentTimeoutError", "ContextLengthExceededError"):
                return 0.0
            if results.verifier_result:
                return float(results.verifier_result.rewards.get("reward", 0.0))
            return 0.0

        return _run_single_trial

    def record_traces(self, traces: list) -> None:
        """Accumulate traces for later batch-level analysis."""
        self._accumulated_traces.extend(traces)

    async def maybe_run_cycle(
        self, run_name: str
    ) -> tuple[dict[str, Any], list[MetaTrainingSample]]:
        """Run the full meta cycle if interval_batches batches have elapsed.

        Returns:
            metrics: dict to be merged into rollout_metrics.
            meta_samples: list of MetaTrainingSample for RL training.
        """
        self._batch_counter += 1
        if not self.config.enabled:
            return {}, []
        if self._batch_counter % self.config.interval_batches != 0:
            return {}, []

        from skyrl_agent.meta_toolkit.observability import TraceWriter

        # ---- 1. Persist accumulated traces ----
        if not self._trace_writer_initialized:
            self._trace_writer = TraceWriter(log_root=self.config.log_dir)
            self._trace_writer_initialized = True

        self._trace_writer.write_batch(self._accumulated_traces, run_name=run_name)
        logger.info(
            f"Meta cycle #{self._batch_counter}: persisting {len(self._accumulated_traces)} traces to {run_name}"
        )

        traces = self._accumulated_traces
        self._accumulated_traces = []  # reset

        # Clear recorded LLM conversations before this cycle
        if self._use_llm and self._llm_client:
            self._llm_client.pop_recorded()

        # ---- 2. Diagnose ----
        if self._use_llm:
            diagnoses = await self._diagnoser.diagnose(traces)
        else:
            diagnoses = self._diagnoser.diagnose(traces)
        raw_diagnosis_count = len(diagnoses)
        diagnoses = self._filter_patchable_diagnoses(diagnoses)
        if len(diagnoses) != raw_diagnosis_count:
            logger.info(
                "Meta: ignored {} context-length/context-overload diagnoses before planning",
                raw_diagnosis_count - len(diagnoses),
            )
        logger.info(f"Meta: diagnosed {len(diagnoses)} problem types: {[d.problem_type for d in diagnoses]}")
        diagnosis_suggestions_count = sum(len(d.strategy_suggestions) for d in diagnoses)
        self._save_diagnosis_events(diagnoses)

        # ---- 3. Plan ----
        n_plan_samples = self.config.max_candidates
        if self._use_llm:
            self._llm_planner.meta_step = self._batch_counter
            candidates = await self._llm_planner.plan_from_batch(
                diagnoses, traces, n_samples=n_plan_samples
            )
        else:
            candidates = self._rule_planner.plan_from_batch(diagnoses)
            candidates = candidates[:n_plan_samples]
        planned_candidate_count = len(candidates)
        logger.info(f"Meta: generated {planned_candidate_count} patch candidates")

        metrics: dict[str, Any] = {
            "meta/cycle_batch": self._batch_counter,
            "meta/num_diagnoses": len(diagnoses),
            "meta/diagnosis_strategy_suggestions_count": diagnosis_suggestions_count,
            "meta/num_candidates_planned": planned_candidate_count,
            "meta/num_candidates": len(candidates),
        }

        self._save_planner_events(
            candidates,
            planned_count=planned_candidate_count,
        )

        if not candidates:
            return metrics, []

        # ---- 4. Prepare canary: baseline + candidates in PARALLEL ----
        canary_task_paths = self._extract_canary_task_paths(traces, diagnoses)
        baseline_overrides = deepcopy(_ACTIVE_META_OVERRIDES) or None

        candidate_deltas: list[float | None] = [None] * len(candidates)
        promoted = 0
        rejected = 0
        pruner_strategies_dropped = 0
        pruner_hooks_dropped = 0
        best_candidate_overrides = None
        best_delta = float("-inf")
        canary_report: dict[str, Any] = {
            "task_paths": [str(p) for p in canary_task_paths],
            "baseline": None,
            "candidates": [],
        }

        import shutil
        import tempfile

        # Pre-snapshot before any candidate modifications
        pre_all_sha = self._vc.snapshot(label="pre-all-candidates")
        active_patch_signature = self._patch_content_signature_from_dir(self._override_base)

        # Phase A: prepare each candidate in an isolated temp directory
        prepared: list[dict] = []
        noop_skipped = 0
        noop_delta = -0.05
        for i, candidate in enumerate(candidates):
            self._candidate_counter += 1
            cid = f"candidate_{self._candidate_counter}"
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"meta_canary_{cid}_")
            tmp_dir = tmp_ctx.name
            try:
                src_base = str(self._override_base)
                for item in Path(src_base).iterdir():
                    if item.name.startswith(".git"):
                        continue
                    dst = Path(tmp_dir) / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)

                from skyrl_agent.meta_toolkit.editing.patch_executor import PatchExecutor
                tmp_executor = PatchExecutor(
                    override_base=tmp_dir,
                )
                # Rewrite file paths: candidate was built with the real
                # override_base, but tmp_executor expects paths under tmp_dir.
                orig_base = str(self._override_base)
                remapped_candidate = deepcopy(candidate)
                for fe in remapped_candidate.files:
                    if fe.path.startswith(orig_base):
                        fe.path = fe.path.replace(orig_base, tmp_dir, 1)
                tmp_executor.apply(remapped_candidate)

                # Enforce strategy/hook limits on this candidate's merged patch-set
                # (historical active patches + candidate edits) BEFORE canary.
                prune_report = None
                if self._pruner is not None:
                    try:
                        candidate_pruner = type(self._pruner)(
                            self._llm_client,
                            tmp_dir,
                            max_strategies=self.config.max_strategies,
                            max_hook_groups=self.config.hook_group_budget,
                        )
                        prune_report = await candidate_pruner.prune()
                        self._append_jsonl("canary", {
                            "type": "pruner_result",
                            "stage": "candidate_pre_canary",
                            "cycle": self._batch_counter,
                            "candidate_id": cid,
                            "candidate_index": i,
                            **prune_report,
                        })
                        pruner_strategies_dropped += prune_report.get("strategies", {}).get(
                            "dropped", 0
                        )
                        pruner_hooks_dropped += sum(
                            cat.get("dropped", 0)
                            for cat in prune_report.get("hooks", {}).values()
                        )
                    except Exception as prune_exc:
                        self._append_jsonl("canary", {
                            "type": "pruner_error",
                            "stage": "candidate_pre_canary",
                            "cycle": self._batch_counter,
                            "candidate_id": cid,
                            "candidate_index": i,
                            "error": str(prune_exc),
                        })
                        raise RuntimeError(
                            f"Candidate pruner failed for {cid}: {prune_exc}"
                        ) from prune_exc

                patch_files = self._list_patch_files_in_dir(tmp_dir)
                overrides = self._load_override_content_from_dir(tmp_dir)
                candidate_patch_signature = self._patch_content_signature_from_dir(tmp_dir)
                if candidate_patch_signature == active_patch_signature:
                    noop_skipped += 1
                    rejected += 1
                    candidate_deltas[i] = noop_delta
                    metrics[f"meta/canary_candidate_{i}_avg_reward"] = 0.0
                    metrics[f"meta/canary_candidate_{i}_pass_rate"] = 0.0
                    self._append_jsonl("canary", {
                        "type": "candidate_noop_skip",
                        "cycle": self._batch_counter,
                        "candidate_id": cid,
                        "candidate_index": i,
                        "delta": noop_delta,
                        "reason": "candidate patch content unchanged after apply+prune",
                    })
                    canary_report["candidates"].append({
                        "candidate_id": cid,
                        "avg_reward": 0.0,
                        "pass_rate": 0.0,
                        "delta": noop_delta,
                        "decision": "noop_skip",
                        "tasks": [],
                    })
                    logger.info(
                        f"Meta: skipping {cid}; patch content unchanged after apply+prune"
                    )
                    tmp_ctx.cleanup()
                    continue

                prepared.append({
                    "candidate": candidate,
                    "candidate_id": cid,
                    "candidate_index": i,
                    "tmp_dir": tmp_dir,
                    "tmp_ctx": tmp_ctx,
                    "patch_files": patch_files,
                    "prune_report": prune_report,
                    "overrides": overrides,
                })
                logger.info(f"Meta: prepared {cid} in {tmp_dir} -> {len(patch_files)} files")
            except Exception as e:
                logger.error(f"Meta: error preparing {cid}: {e}")
                tmp_ctx.cleanup()
                metrics[f"meta/canary_candidate_{i}_avg_reward"] = 0.0
                metrics[f"meta/canary_candidate_{i}_pass_rate"] = 0.0
                candidate_deltas[i] = -1.0
                rejected += 1
                self._append_jsonl("canary", {
                    "type": "candidate_prepare_error",
                    "cycle": self._batch_counter,
                    "candidate_id": cid,
                    "error": str(e),
                })
                canary_report["candidates"].append({
                    "candidate_id": cid,
                    "avg_reward": 0.0,
                    "pass_rate": 0.0,
                    "delta": -1.0,
                    "decision": "prepare_error",
                    "tasks": [],
                })

        # Phase B: run baseline AND all candidates CONCURRENTLY
        if prepared:
            async def _run_baseline():
                return await self._canary.run_baseline(canary_task_paths, baseline_overrides)

            async def _run_candidate(info: dict):
                return await self._canary._run_trial_batch(
                    canary_task_paths, info["overrides"],
                    label=f"candidate:{info['candidate_id']}",
                )

            all_tasks = [_run_baseline()] + [_run_candidate(info) for info in prepared]
            logger.info(
                f"Meta: launching baseline + {len(prepared)} candidates concurrently "
                f"({len(canary_task_paths)} tasks × {self.config.canary_n_samples} samples each)"
            )
            all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

            baseline_result = all_results[0]
            candidate_raw_results = all_results[1:]

            if isinstance(baseline_result, Exception) or baseline_result is None:
                logger.error(f"Meta: baseline canary failed: {baseline_result}")
                metrics["meta/canary_baseline_avg_reward"] = 0.0
                metrics["meta/canary_baseline_pass_rate"] = 0.0
                candidate_deltas = [-1.0] * len(candidates)
                for info in prepared:
                    cand_idx = info["candidate_index"]
                    metrics[f"meta/canary_candidate_{cand_idx}_avg_reward"] = 0.0
                    metrics[f"meta/canary_candidate_{cand_idx}_pass_rate"] = 0.0
                    canary_report["candidates"].append({
                        "candidate_id": info["candidate_id"],
                        "avg_reward": 0.0,
                        "pass_rate": 0.0,
                        "delta": -1.0,
                        "decision": "baseline_error",
                        "tasks": [],
                    })
                rejected = len(candidates)
                self._append_jsonl("canary", {
                    "type": "baseline_error",
                    "cycle": self._batch_counter,
                    "error": str(baseline_result),
                })
            else:
                baseline_scores = baseline_result
                baseline_summary = self._canary.summarize_scores(baseline_scores)
                metrics["meta/canary_baseline_avg_reward"] = baseline_summary["avg_reward"]
                metrics["meta/canary_baseline_pass_rate"] = baseline_summary["pass_rate"]
                canary_report["baseline"] = baseline_summary
                self._append_jsonl("canary", {
                    "type": "baseline_summary",
                    "cycle": self._batch_counter,
                    "avg_reward": baseline_summary["avg_reward"],
                    "pass_rate": baseline_summary["pass_rate"],
                    "num_tasks": len(baseline_scores),
                    "scores": baseline_scores,
                })

                # Phase C: compare each candidate against baseline, shortlist
                # acceptable candidates, then promote only the single best one.
                best_prepared = None
                best_eval_result = None
                best_patch_files: list[str] = []
                for info, raw_result in zip(prepared, candidate_raw_results):
                    cand_idx = info["candidate_index"]
                    cid = info["candidate_id"]
                    if isinstance(raw_result, Exception):
                        logger.error(f"Meta: canary eval failed for {cid}: {raw_result}")
                        metrics[f"meta/canary_candidate_{cand_idx}_avg_reward"] = 0.0
                        metrics[f"meta/canary_candidate_{cand_idx}_pass_rate"] = 0.0
                        candidate_deltas[cand_idx] = -1.0
                        rejected += 1
                        self._append_jsonl("canary", {
                            "type": "candidate_error",
                            "cycle": self._batch_counter,
                            "candidate_id": cid,
                            "error": str(raw_result),
                        })
                        canary_report["candidates"].append({
                            "candidate_id": cid,
                            "avg_reward": 0.0,
                            "pass_rate": 0.0,
                            "delta": -1.0,
                            "decision": "error",
                            "tasks": [],
                        })
                        continue

                    after_scores = raw_result
                    candidate_summary = self._canary.summarize_scores(after_scores)
                    metrics[f"meta/canary_candidate_{cand_idx}_avg_reward"] = candidate_summary["avg_reward"]
                    metrics[f"meta/canary_candidate_{cand_idx}_pass_rate"] = candidate_summary["pass_rate"]
                    task_labels = [str(p) for p in canary_task_paths]
                    eval_result = self._canary._comparison.evaluate(
                        baseline_scores, after_scores,
                        task_ids=task_labels,
                        metadata={"candidate_id": cid, "patch_files": info["patch_files"]},
                    )
                    logger.info(
                        f"Canary {cid}: baseline_avg={sum(baseline_scores)/max(len(baseline_scores),1):.3f} "
                        f"candidate_avg={sum(after_scores)/max(len(after_scores),1):.3f} "
                        f"baseline_pass_rate={baseline_summary['pass_rate']:.3f} "
                        f"candidate_pass_rate={candidate_summary['pass_rate']:.3f} "
                        f"delta={eval_result.delta_score:.4f}"
                    )

                    task_rows: list[dict[str, Any]] = []
                    for task_path, before_score, after_score in zip(
                        task_labels, baseline_scores, after_scores
                    ):
                        task_delta = after_score - before_score
                        row = {
                            "task_path": task_path,
                            "baseline_reward": before_score,
                            "candidate_reward": after_score,
                            "delta": task_delta,
                            "baseline_pass": before_score > 0,
                            "candidate_pass": after_score > 0,
                        }
                        task_rows.append(row)
                        self._append_jsonl("canary", {
                            "type": "task_detail",
                            "cycle": self._batch_counter,
                            "candidate_id": cid,
                            **row,
                        })

                    self._append_jsonl("canary", {
                        "type": "candidate_summary",
                        "cycle": self._batch_counter,
                        "candidate_id": cid,
                        "avg_reward": candidate_summary["avg_reward"],
                        "pass_rate": candidate_summary["pass_rate"],
                        "delta_score": eval_result.delta_score,
                        "num_tasks": len(after_scores),
                    })

                    candidate_deltas[cand_idx] = eval_result.delta_score
                    decision = self._promoter.decide(
                        candidate_id=cid,
                        eval_result=eval_result,
                        write_record=False,
                    )
                    if decision.decision == "accept":
                        if eval_result.delta_score > best_delta:
                            best_delta = eval_result.delta_score
                            best_candidate_overrides = info["overrides"]
                            best_prepared = info
                            best_eval_result = eval_result
                            best_patch_files = info["patch_files"]
                    else:
                        pass
                    logger.info(
                        f"Meta: {cid} → {decision.decision} "
                        f"(delta={decision.delta_score:.4f}, notes={decision.notes})"
                    )
                    canary_report["candidates"].append({
                        "candidate_id": cid,
                        "avg_reward": candidate_summary["avg_reward"],
                        "pass_rate": candidate_summary["pass_rate"],
                        "delta": eval_result.delta_score,
                        "decision": "eligible" if decision.decision == "accept" else "reject",
                        "tasks": task_rows,
                    })

                # Phase D: apply the best candidate to the real override_base
                if best_prepared is not None and best_eval_result is not None:
                    self._vc.restore(pre_all_sha)
                    applied_files = self._sync_override_base_from_dir(best_prepared["tmp_dir"])
                    final_decision = self._promoter.promote(
                        candidate_id=best_prepared["candidate_id"],
                        patch_files=applied_files or best_patch_files,
                        eval_result=best_eval_result,
                        cycle=self._batch_counter,
                    )
                    promoted = 1 if final_decision.decision == "accept" else 0
                    rejected = max(0, len(candidates) - promoted)
                    canary_report["winner_candidate_id"] = best_prepared["candidate_id"]
                    logger.info(
                        f"Meta: applied pruned winner {best_prepared['candidate_id']} "
                        f"to override_base ({len(applied_files)} files)"
                    )

                else:
                    self._vc.restore(pre_all_sha)
                    promoted = 0
                    rejected = len(candidates)

            # Cleanup temp directories
            for info in prepared:
                tmp_ctx = info.get("tmp_ctx")
                if tmp_ctx is not None:
                    tmp_ctx.cleanup()
                else:
                    shutil.rmtree(info["tmp_dir"], ignore_errors=True)
        else:
            # No prepared candidates — all were no-op and/or failed during preparation.
            if noop_skipped:
                logger.info(
                    "Meta: no changed candidates prepared "
                    f"({noop_skipped} no-op, {len(candidates) - noop_skipped} prepare errors); "
                    "skipping canary evaluation"
                )
                self._append_jsonl("canary", {
                    "type": "all_candidates_skipped",
                    "cycle": self._batch_counter,
                    "noop_skipped": noop_skipped,
                    "reason": "no candidate changed patch content after apply+prune",
                })
            else:
                logger.warning("Meta: all candidates failed preparation, skipping canary evaluation")
                self._append_jsonl("canary", {
                    "type": "candidate_prepare_error",
                    "cycle": self._batch_counter,
                    "error": "all candidates failed preparation",
                })
            metrics["meta/canary_baseline_avg_reward"] = 0.0
            metrics["meta/canary_baseline_pass_rate"] = 0.0
            for cand_idx in range(len(candidates)):
                if candidate_deltas[cand_idx] is None:
                    candidate_deltas[cand_idx] = -1.0
                metrics[f"meta/canary_candidate_{cand_idx}_avg_reward"] = 0.0
                metrics[f"meta/canary_candidate_{cand_idx}_pass_rate"] = 0.0
            rejected = max(rejected, len(candidates))

        metrics["meta/pruner_strategies_dropped"] = pruner_strategies_dropped
        metrics["meta/pruner_hooks_dropped"] = pruner_hooks_dropped
        metrics["meta/candidates_noop_skipped"] = noop_skipped

        if best_candidate_overrides is not None:
            self._inject_overrides_to_agent_config(best_candidate_overrides)

        candidate_deltas = [d if d is not None else -1.0 for d in candidate_deltas]

        metrics.update({
            "meta/promoted": promoted,
            "meta/rejected": rejected,
        })

        # Write human-readable cycle summary
        self._save_cycle_summary(
            diagnoses,
            candidates,
            candidate_deltas,
            promoted,
            rejected,
            best_delta,
            canary_report=canary_report,
        )

        # Record cycle outcome for cross-cycle diagnosis context
        if hasattr(self, '_diagnosis_history'):
            from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import CycleSummary
            self._diagnosis_history.append(CycleSummary(
                cycle_id=self._batch_counter,
                diagnosed_types=[d.problem_type for d in diagnoses],
                candidates_generated=len(candidates),
                promoted=promoted,
                rejected=rejected,
                best_delta=best_delta if best_delta > float("-inf") else 0.0,
                active_overrides_snapshot={k: list(v.keys()) for k, v in _ACTIVE_META_OVERRIDES.items()},
                notes=f"batch_{self._batch_counter}",
            ))

        # ---- 5. Build GRPO-compatible meta-RL training samples ----
        # Planning samples share one instance_id (same prompt, different responses)
        # so GRPO can compute within-group relative advantages.
        meta_samples: list[MetaTrainingSample] = []
        if self._use_llm and self._llm_client:
            conversations = self._llm_client.pop_recorded()

            # Persist all LLM conversations for debugging
            self._save_conversations(conversations, candidate_deltas)

            planning_convs = [c for c in conversations if c.role == "planning"]

            if planning_convs and candidate_deltas:
                cycle_id = f"meta_plan_cycle_{self._batch_counter}"
                for rep_id, (conv, delta) in enumerate(
                    zip(planning_convs, candidate_deltas)
                ):
                    full_messages = conv.messages + [
                        {"role": "assistant", "content": conv.response}
                    ]
                    meta_samples.append(MetaTrainingSample(
                        messages=full_messages,
                        reward=delta,
                        role=conv.role,
                        instance_id=cycle_id,
                        repetition_id=rep_id,
                    ))

            if meta_samples:
                rewards = [s.reward for s in meta_samples]
                metrics["meta/num_meta_train_samples"] = len(meta_samples)
                metrics["meta/meta_reward_mean"] = sum(rewards) / len(rewards)
                metrics["meta/meta_reward_std"] = (
                    (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards)) ** 0.5
                )
                logger.info(
                    f"Meta-RL: {len(meta_samples)} GRPO samples "
                    f"(instance={meta_samples[0].instance_id}, "
                    f"rewards={[f'{r:.3f}' for r in rewards]})"
                )

        return metrics, meta_samples

    def _append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        """Append a machine-readable meta event to the unified event log.

        ``filename`` is kept as a logical stream name for call-site clarity,
        but all structured details are written to one file so each meta run has
        a small, predictable log surface.
        """
        path = Path(self.config.log_dir) / "events.jsonl"
        stream = Path(filename).stem
        payload = {"stream": stream, **record}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Meta: failed writing events.jsonl ({stream}): {e}")

    @staticmethod
    def _is_context_overflow_error(error_text: str) -> bool:
        lower = str(error_text or "").lower()
        return (
            "maximum context length" in lower
            or "parameter=input_tokens" in lower
            or "context length is" in lower
        )

    def _compute_diagnosis_stats(self, worker_details: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate per-cycle diagnosis worker stats for observability."""
        stats: dict[str, Any] = {
            "total_workers": len(worker_details),
            "ok_workers": 0,
            "error_workers": 0,
            "context_overflow_errors": 0,
            "strategy_suggestions_total": 0,
            "avg_turns_ok": None,
            "min_turns_ok": None,
            "max_turns_ok": None,
            "by_category": {},
        }
        turns_ok: list[int] = []

        for detail in worker_details:
            category = str(detail.get("category", "unknown"))
            bucket = stats["by_category"].setdefault(
                category,
                {
                    "total": 0,
                    "ok": 0,
                    "error": 0,
                    "context_overflow_errors": 0,
                    "strategy_suggestions": 0,
                },
            )
            bucket["total"] += 1

            if detail.get("status") == "ok":
                stats["ok_workers"] += 1
                bucket["ok"] += 1
                n_suggestions = int(detail.get("n_strategy_suggestions", 0) or 0)
                stats["strategy_suggestions_total"] += n_suggestions
                bucket["strategy_suggestions"] += n_suggestions
                turns = detail.get("turns_used")
                if isinstance(turns, int):
                    turns_ok.append(turns)
            else:
                stats["error_workers"] += 1
                bucket["error"] += 1
                if self._is_context_overflow_error(detail.get("error", "")):
                    stats["context_overflow_errors"] += 1
                    bucket["context_overflow_errors"] += 1

        if turns_ok:
            stats["avg_turns_ok"] = sum(turns_ok) / len(turns_ok)
            stats["min_turns_ok"] = min(turns_ok)
            stats["max_turns_ok"] = max(turns_ok)

        return stats

    def _save_diagnosis_events(self, diagnoses: list) -> None:
        """Persist diagnosis worker details and final diagnoses to events.jsonl."""
        worker_details = getattr(self._diagnoser, "last_worker_details", []) or []
        for detail in worker_details:
            self._append_jsonl("diagnosis", {
                "type": "worker_detail",
                "cycle": self._batch_counter,
                **detail,
            })

        for d in diagnoses:
            self._append_jsonl("diagnosis", {
                "type": "diagnosis",
                "cycle": self._batch_counter,
                "problem_type": d.problem_type,
                "confidence": d.confidence,
                "candidate_modules": d.candidate_modules,
                "affected_trace_ids": d.affected_task_ids,
                "strategy_suggestions_count": len(d.strategy_suggestions),
                "strategy_suggestions": d.strategy_suggestions,
                "metadata": d.metadata,
            })

        stats = self._compute_diagnosis_stats(worker_details)
        self._append_jsonl("diagnosis", {
            "type": "diagnosis_stats",
            "cycle": self._batch_counter,
            "num_diagnoses": len(diagnoses),
            "problem_types": [d.problem_type for d in diagnoses],
            **stats,
        })
        logger.info(
            "Diagnosis stats: "
            f"workers={stats['total_workers']}, "
            f"ok={stats['ok_workers']}, "
            f"errors={stats['error_workers']}, "
            f"context_overflow_errors={stats['context_overflow_errors']}, "
            f"strategy_suggestions={stats['strategy_suggestions_total']}"
        )

    def _save_planner_events(
        self,
        candidates: list,
        *,
        planned_count: int,
    ) -> None:
        """Persist planner sub-agent outputs and candidate summaries."""
        self._append_jsonl("planner", {
            "type": "planner_summary",
            "cycle": self._batch_counter,
            "planned_candidates": planned_count,
            "selected_candidates": len(candidates),
        })

        if self._use_llm and hasattr(self, "_llm_planner"):
            for detail in self._llm_planner.last_subagent_details:
                self._append_jsonl("planner", {
                    "type": "planner_subagent",
                    "cycle": self._batch_counter,
                    **detail,
                })

        for idx, candidate in enumerate(candidates):
            self._append_jsonl("planner", {
                "type": "candidate",
                "cycle": self._batch_counter,
                "index": idx,
                "target_modules": candidate.target_modules,
                "risk": candidate.risk,
                "required_tests": candidate.required_tests,
                "rollback_if": candidate.rollback_if,
                "notes": candidate.notes,
                "files": [
                    {
                        "path": fe.path,
                        "change_type": fe.change_type,
                        "intent_preview": fe.intent[:300],
                    }
                    for fe in candidate.files
                ],
            })

    def _save_conversations(self, conversations, candidate_deltas: list[float]) -> None:
        """Persist all LLM conversations to one role-tagged JSONL file."""
        import os
        os.makedirs(self.config.log_dir, exist_ok=True)

        conversations_path = os.path.join(self.config.log_dir, "llm_conversations.jsonl")

        try:
            with open(conversations_path, "a", encoding="utf-8") as f:
                for i, conv in enumerate(conversations):
                    reward = candidate_deltas[i] if i < len(candidate_deltas) else None
                    record = {
                        "cycle": self._batch_counter,
                        "role": conv.role,
                        "messages": conv.messages,
                        "response": conv.response[:3000],
                        "reward": reward,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                f"Meta: saved {len(conversations)} conversations "
                f"(llm_conversations.jsonl → {self.config.log_dir})"
            )
        except Exception as e:
            logger.warning(f"Meta: failed to save conversations: {e}")

    def _save_cycle_summary(
        self,
        diagnoses,
        candidates,
        candidate_deltas: list[float],
        promoted: int,
        rejected: int,
        best_delta: float,
        canary_report: dict[str, Any] | None = None,
    ) -> None:
        """Append a human-readable cycle summary to {log_dir}/cycle_summary.md."""
        import os
        from datetime import datetime

        summary_path = os.path.join(self.config.log_dir, "cycle_summary.md")
        try:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"Meta Cycle #{self._batch_counter}  ({ts})\n")
                f.write(f"{'='*60}\n\n")

                # Diagnosis summary
                f.write(f"[Diagnosis] {len(diagnoses)} problems identified:\n")
                for d in diagnoses:
                    hyp = d.root_cause_hypotheses[0] if d.root_cause_hypotheses else "N/A"
                    f.write(
                        f"  - {d.problem_type} (conf={d.confidence:.2f})"
                        f"  modules={d.candidate_modules}"
                        f"  traces={d.affected_task_ids[:3]}\n"
                    )
                    f.write(f"    hypothesis: {hyp[:120]}\n")
                worker_details = getattr(self._diagnoser, "last_worker_details", []) or []
                diagnosis_stats = self._compute_diagnosis_stats(worker_details)
                if diagnosis_stats["total_workers"] > 0:
                    f.write(
                        "  worker_stats: "
                        f"total={diagnosis_stats['total_workers']}, "
                        f"ok={diagnosis_stats['ok_workers']}, "
                        f"error={diagnosis_stats['error_workers']}, "
                        f"context_overflow_error={diagnosis_stats['context_overflow_errors']}, "
                        f"strategy_suggestions={diagnosis_stats['strategy_suggestions_total']}\n"
                    )
                    if diagnosis_stats["avg_turns_ok"] is not None:
                        f.write(
                            "  turns(ok_workers): "
                            f"avg={diagnosis_stats['avg_turns_ok']:.2f}, "
                            f"min={diagnosis_stats['min_turns_ok']}, "
                            f"max={diagnosis_stats['max_turns_ok']}\n"
                        )
                    by_category = diagnosis_stats.get("by_category", {})
                    for category in sorted(by_category.keys()):
                        cat_stats = by_category[category]
                        f.write(
                            f"  category[{category}]: "
                            f"total={cat_stats.get('total', 0)}, "
                            f"ok={cat_stats.get('ok', 0)}, "
                            f"error={cat_stats.get('error', 0)}, "
                            f"context_overflow_error={cat_stats.get('context_overflow_errors', 0)}, "
                            f"strategy_suggestions={cat_stats.get('strategy_suggestions', 0)}\n"
                        )
                f.write("\n")

                # Candidate summary
                f.write(f"[Planning] {len(candidates)} patch candidates generated:\n")
                for i, cand in enumerate(candidates):
                    delta = candidate_deltas[i] if i < len(candidate_deltas) else None
                    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
                    status = "ACCEPTED" if delta is not None and delta > 0 else "REJECTED"
                    f.write(
                        f"  candidate_{i+1}: {cand.target_modules}"
                        f"  risk={cand.risk}  delta={delta_str}  -> {status}\n"
                    )
                    if cand.notes:
                        f.write(f"    notes: {cand.notes[:150]}\n")
                f.write("\n")

                if canary_report:
                    baseline = canary_report.get("baseline")
                    if baseline:
                        f.write(
                            "[Canary] baseline: "
                            f"avg_reward={baseline.get('avg_reward', 0.0):.4f}, "
                            f"pass_rate={baseline.get('pass_rate', 0.0):.4f}\n"
                        )
                    else:
                        f.write("[Canary] baseline: unavailable\n")

                    for c in canary_report.get("candidates", []):
                        f.write(
                            "  - "
                            f"{c.get('candidate_id', 'unknown')}: "
                            f"avg_reward={c.get('avg_reward', 0.0):.4f}, "
                            f"pass_rate={c.get('pass_rate', 0.0):.4f}, "
                            f"delta={c.get('delta', 0.0):+.4f}, "
                            f"decision={c.get('decision', 'unknown')}\n"
                        )
                        for task in c.get("tasks", []):
                            f.write(
                                "      "
                                f"task={task.get('task_path')} "
                                f"baseline={task.get('baseline_reward', 0.0):.4f} "
                                f"candidate={task.get('candidate_reward', 0.0):.4f} "
                                f"delta={task.get('delta', 0.0):+.4f} "
                                f"pass={task.get('candidate_pass', False)}\n"
                            )
                    f.write("\n")

                # Overall result
                f.write(f"[Result] promoted={promoted}, rejected={rejected}")
                if best_delta > float("-inf"):
                    f.write(f", best_delta={best_delta:+.4f}")
                f.write("\n")

            logger.info(f"Meta: cycle summary written to {summary_path}")
        except Exception as e:
            logger.warning(f"Meta: failed to write cycle summary: {e}")

    def _load_override_content(self, patch_files: list[str]) -> dict[str, dict]:
        """Load strategy_library and code hooks from patch files into meta_overrides dict.

        Only strategy_library.yaml and hooks/*.py are loaded.
        """
        import os

        result: dict[str, dict] = {}
        hooks: dict[str, str] = {}

        for path_str in patch_files:
            p = str(path_str)
            filename = os.path.basename(p)

            if filename.endswith(".py") and "/hooks/" in p and os.path.exists(p):
                hook_name = filename.removesuffix(".py")
                try:
                    with open(p, encoding="utf-8") as f:
                        hooks[hook_name] = f.read()
                except Exception:
                    pass
                continue

            if filename == "strategy_library.yaml" and os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        flat = yaml.safe_load(f) or {}
                    result["strategy_library"] = {k: v for k, v in flat.items() if not k.startswith("_")}
                except Exception:
                    pass

        hooks_dir = self._override_base / "hooks"
        if hooks_dir.is_dir():
            for hook_file in hooks_dir.glob("*.py"):
                hook_name = hook_file.stem
                if hook_name not in hooks:
                    try:
                        hooks[hook_name] = hook_file.read_text(encoding="utf-8")
                    except Exception:
                        pass

        if hooks:
            result["_meta_hooks"] = hooks

        return result

    def _load_override_content_from_dir(self, directory: str) -> dict[str, dict]:
        """Load strategy_library and code hooks from a directory (for canary eval).

        Only strategy_library.yaml and hooks/*.py are loaded — all other YAML
        override files have been removed from the meta-learning pipeline.
        """
        result: dict[str, dict] = {}
        hooks: dict[str, str] = {}
        base = Path(directory)

        sl_path = base / "strategy_library.yaml"
        if sl_path.exists():
            try:
                flat = yaml.safe_load(sl_path.read_text(encoding="utf-8")) or {}
                result["strategy_library"] = {k: v for k, v in flat.items() if not k.startswith("_")}
            except Exception:
                pass

        hooks_dir = base / "hooks"
        if hooks_dir.is_dir():
            for hf in hooks_dir.glob("*.py"):
                try:
                    hooks[hf.stem] = hf.read_text(encoding="utf-8")
                except Exception:
                    pass

        if hooks:
            result["_meta_hooks"] = hooks

        return result

    def _patch_content_signature_from_dir(self, directory: str | Path) -> str:
        """Return a stable signature of strategy/hooks content in an override dir."""
        content = self._load_override_content_from_dir(str(directory))
        payload = json.dumps(
            content,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _list_patch_files_in_dir(self, directory: str) -> list[str]:
        """Return normalized patch file paths in an override directory."""
        base = Path(directory)
        files: list[str] = []

        sl_path = base / "strategy_library.yaml"
        if sl_path.exists():
            files.append(str(sl_path))

        hooks_dir = base / "hooks"
        if hooks_dir.is_dir():
            for hf in sorted(hooks_dir.glob("*.py")):
                files.append(str(hf))

        return files

    def _sync_override_base_from_dir(self, source_dir: str) -> list[str]:
        """Replace active patch files with the evaluated candidate directory state."""
        import shutil

        src = Path(source_dir)
        dst = self._override_base
        applied_files: list[str] = []

        # strategy_library.yaml
        src_sl = src / "strategy_library.yaml"
        dst_sl = dst / "strategy_library.yaml"
        if src_sl.exists():
            dst_sl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_sl, dst_sl)
            applied_files.append(str(dst_sl))
        elif dst_sl.exists():
            dst_sl.unlink()

        # hooks/*.py
        src_hooks = src / "hooks"
        dst_hooks = dst / "hooks"
        dst_hooks.mkdir(parents=True, exist_ok=True)

        for existing in dst_hooks.glob("*.py"):
            existing.unlink()

        if src_hooks.is_dir():
            for hf in sorted(src_hooks.glob("*.py")):
                target = dst_hooks / hf.name
                shutil.copy2(hf, target)
                applied_files.append(str(target))

        return applied_files

    def _extract_canary_task_paths(
        self, traces: list, diagnoses: list | None = None
    ) -> list:
        """Select task paths for canary evaluation, informed by diagnosis results.

        Strategy:
          1. Collect per-task stats (avg reward, failure count) from all traces.
          2. If diagnoses provide affected trace IDs, prioritise those tasks.
          3. Among remaining failures, sort by avg reward ascending (worst first).
          4. Reserve 1 slot for a successful task (regression detection).
          5. De-duplicate by task path string.
        """
        max_tasks = self.config.canary_num_tasks
        if max_tasks <= 0:
            return []

        # --- aggregate per-task stats ---
        task_stats: dict[str, dict] = {}
        for trace in traces:
            prompt = trace.metadata.get("prompt")
            if prompt is None:
                continue
            key = str(prompt)
            if key not in task_stats:
                task_stats[key] = {
                    "prompt": prompt,
                    "rewards": [],
                    "failures": 0,
                    "successes": 0,
                    "diagnosed": False,
                }
            entry = task_stats[key]
            entry["rewards"].append(trace.final_reward or 0.0)
            if trace.success:
                entry["successes"] += 1
            else:
                entry["failures"] += 1

        for entry in task_stats.values():
            rw = entry["rewards"]
            entry["avg_reward"] = sum(rw) / len(rw) if rw else 0.0

        # --- build short-id → prompt-path mapping for diagnosis matching ---
        # Diagnosis uses short IDs like "1386" or "1386-traj0", but task_stats
        # keys are full paths like "/home/ray/data/harbor/TG/influxdb_...".
        # We also collect the trace task_id (e.g. "1386-traj0") per prompt path.
        short_id_to_key: dict[str, str] = {}
        for trace in traces:
            prompt = trace.metadata.get("prompt")
            if prompt is None:
                continue
            key = str(prompt)
            tid = trace.task_id  # e.g. "1386-traj0"
            short_id_to_key[tid] = key
            # Also map the instance-level key (e.g. "1386")
            import re as _re
            m = _re.match(r"^(.+)-traj\d+$", tid)
            if m:
                short_id_to_key[m.group(1)] = key

        # --- mark tasks referenced by diagnoses ---
        if diagnoses:
            diagnosed_ids = set()
            for d in diagnoses:
                diagnosed_ids.update(d.affected_task_ids)
            # Resolve short IDs to full prompt-path keys
            diagnosed_keys = set()
            for did in diagnosed_ids:
                if did in short_id_to_key:
                    diagnosed_keys.add(short_id_to_key[did])
                else:
                    diagnosed_keys.add(did)
            n_matched = 0
            for key, entry in task_stats.items():
                if key in diagnosed_keys:
                    entry["diagnosed"] = True
                    n_matched += 1
            logger.info(
                f"Meta canary: {len(diagnosed_ids)} diagnosed IDs → "
                f"{len(diagnosed_keys)} resolved keys → {n_matched} tasks marked"
            )

        # --- split into buckets ---
        failed_tasks = [
            e for e in task_stats.values() if e["failures"] > 0
        ]
        success_tasks = [
            e for e in task_stats.values() if e["failures"] == 0 and e["successes"] > 0
        ]

        diagnosed_failed = [e for e in failed_tasks if e["diagnosed"]]
        undiagnosed_failed = [e for e in failed_tasks if not e["diagnosed"]]
        diagnosed_failed.sort(key=lambda e: e["avg_reward"])
        undiagnosed_failed.sort(key=lambda e: e["avg_reward"])

        # reserve 1 slot for regression detection (a successful task)
        regression_slots = min(1, max_tasks // 4, len(success_tasks))
        failure_slots = max_tasks - regression_slots

        # Majority should be diagnosed tasks, but keep some unseen tasks.
        diagnosed_ratio = min(max(float(self.config.canary_diagnosed_ratio), 0.0), 1.0)
        desired_diagnosed = int(round(failure_slots * diagnosed_ratio))
        desired_diagnosed = min(max(desired_diagnosed, 0), failure_slots)
        desired_undiagnosed = failure_slots - desired_diagnosed

        if failure_slots > 0:
            desired_undiagnosed = max(desired_undiagnosed, int(self.config.canary_min_undiagnosed))
            desired_undiagnosed = min(desired_undiagnosed, failure_slots)
            desired_diagnosed = failure_slots - desired_undiagnosed

        pick_diagnosed = min(desired_diagnosed, len(diagnosed_failed))
        pick_undiagnosed = min(desired_undiagnosed, len(undiagnosed_failed))
        remaining_failure_slots = failure_slots - pick_diagnosed - pick_undiagnosed
        if remaining_failure_slots > 0:
            extra_diag = min(
                remaining_failure_slots,
                len(diagnosed_failed) - pick_diagnosed,
            )
            pick_diagnosed += extra_diag
            remaining_failure_slots -= extra_diag
        if remaining_failure_slots > 0:
            extra_undiag = min(
                remaining_failure_slots,
                len(undiagnosed_failed) - pick_undiagnosed,
            )
            pick_undiagnosed += extra_undiag
            remaining_failure_slots -= extra_undiag

        paths: list = []
        seen: set[str] = set()

        for entry in diagnosed_failed[:pick_diagnosed]:
            key = str(entry["prompt"])
            if key not in seen:
                seen.add(key)
                paths.append(entry["prompt"])

        for entry in undiagnosed_failed[:pick_undiagnosed]:
            key = str(entry["prompt"])
            if key not in seen:
                seen.add(key)
                paths.append(entry["prompt"])

        # pick the success task with the highest reward (most likely to regress visibly)
        success_tasks.sort(key=lambda e: e["avg_reward"], reverse=True)
        for entry in success_tasks[:regression_slots]:
            if len(paths) >= max_tasks:
                break
            key = str(entry["prompt"])
            if key not in seen:
                seen.add(key)
                paths.append(entry["prompt"])

        # Backfill if slots remain.
        if len(paths) < max_tasks:
            backfill_pool = diagnosed_failed[pick_diagnosed:] + undiagnosed_failed[pick_undiagnosed:] + success_tasks[regression_slots:]
            for entry in backfill_pool:
                if len(paths) >= max_tasks:
                    break
                key = str(entry["prompt"])
                if key in seen:
                    continue
                seen.add(key)
                paths.append(entry["prompt"])

        logger.info(
            "Meta canary sampling: "
            f"max={max_tasks}, failure_slots={failure_slots}, "
            f"picked_diagnosed_fail={pick_diagnosed}, picked_undiagnosed_fail={pick_undiagnosed}, "
            f"regression_success={regression_slots}, final={len(paths)}"
        )

        return paths[:max_tasks]

    def _inject_overrides_to_agent_config(self, overrides: dict[str, dict]) -> None:
        """Store active overrides so HarborGenerator can inject them into trial config."""
        _ACTIVE_META_OVERRIDES.update(overrides)

    def _preload_active_overrides(self) -> None:
        """Load existing patches from override_base into _ACTIVE_META_OVERRIDES.

        Called once during __init__ so that after a process restart, both the
        main RL sampling loop and the canary baseline automatically use
        previously adopted patches (strategy_library + hooks).
        """
        overrides = self._load_override_content_from_dir(str(self._override_base))
        if overrides:
            _ACTIVE_META_OVERRIDES.update(overrides)
            strat_count = len(overrides.get("strategy_library", {}))
            hook_count = len(overrides.get("_meta_hooks", {}))
            logger.info(
                f"MetaLoop: preloaded active overrides from {self._override_base} "
                f"({strat_count} strategies, {hook_count} hooks)"
            )
        else:
            logger.info("MetaLoop: no existing patches to preload")

    def flush(self, run_name: str) -> None:
        """Flush any remaining accumulated traces and release file handles."""
        if self._accumulated_traces and self._trace_writer_initialized:
            self._trace_writer.write_batch(self._accumulated_traces, run_name=run_name)
            self._accumulated_traces = []
        if self._trace_writer_initialized:
            self._trace_writer.flush_all()
            self._trace_writer.close()

    async def close(self) -> None:
        """Release async resources (LLM client session)."""
        if self._llm_client is not None:
            await self._llm_client.close()


# Module-level registry for active meta overrides (injected into Harbor trials)
_ACTIVE_META_OVERRIDES: dict[str, dict] = {}


# =============================================================================
# Harbor Generator
# =============================================================================


@dataclass
class HarborAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None
    num_turns: Optional[int] = None
    trace_record: Optional[Any] = None


class HarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            harbor_cfg: DictConfig object containing the Harbor configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            max_seq_len: Maximum total sequence length (prompt + response). Used to truncate responses.
        """
        ie_cfg = generator_cfg.inference_engine
        self.base_url = f"http://{ie_cfg.http_endpoint_host}:{ie_cfg.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # ---- Meta-learning config ----
        meta_cfg = getattr(generator_cfg, "meta", None)
        self._meta_config = MetaLoopConfig.from_cfg(meta_cfg)

        # Harbor config template
        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        # Set model_name and api_base once (constant across all trials)
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert (
            "/" not in ie_cfg.served_model_name
        ), "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = (
            f"{self.base_url}/v1"
        )
        self._agent_name = self._harbor_trial_config_template.get("agent", {}).get("name", "")

        # Read custom chat template
        custom_chat_template_path = ie_cfg.engine_init_kwargs.get("chat_template", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(
                f"HarborGenerator initialized with custom chat template read from: {custom_chat_template_path}"
            )
        else:
            self.custom_chat_template_content = None

        # Rate limiter — must be created before MetaLoopController
        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

        # Meta-learning setup — use adapter pattern for agent generality
        self._meta_loop: Optional[_MetaLoopController] = None
        self._meta_controller = None
        self._meta_adapter = None

        if self._meta_config.enabled and isinstance(self._agent_name, str):
            from skyrl_agent.meta_toolkit.adapters import Terminus2Adapter, GenericInstructionAdapter
            from skyrl_agent.meta_toolkit.runtime import MetaController

            # Select the appropriate adapter for this agent (order matters: most specific first)
            adapters = [Terminus2Adapter(), GenericInstructionAdapter()]
            for adapter in adapters:
                if adapter.matches(self._agent_name):
                    self._meta_adapter = adapter
                    break

            registry = self._meta_adapter.build_registry()
            self._meta_controller = MetaController(registry)
            self._meta_loop = _MetaLoopController(
                registry=registry,
                config=self._meta_config,
                trial_config_template=self._harbor_trial_config_template,
                rate_limiter=self._rate_limiter,
                adapter=self._meta_adapter,
                agent_name=self._agent_name,
            )
            logger.info(
                f"MetaLoopController initialized: agent={self._agent_name}, "
                f"adapter={type(self._meta_adapter).__name__}, "
                f"interval={self._meta_config.interval_batches}, "
                f"override_base={self._meta_config.override_base}"
            )

        logger.info(
            f"HarborGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}"
        )

        # Run name derived from log path / run_name
        self._run_name = self._get_run_name_from_cfg(generator_cfg)

    def _get_run_name_from_cfg(self, generator_cfg: DictConfig) -> str:
        """Extract run_name for trace persistence."""
        if hasattr(generator_cfg, "run_name"):
            return str(generator_cfg.run_name)
        if hasattr(generator_cfg, "log_path"):
            lp = str(generator_cfg.log_path)
            return lp.split("/")[-1] if "/" in lp else lp
        return "harbor_default"

    def _tokenize_meta_samples(
        self,
        samples: list[MetaTrainingSample],
        existing_trajectory_ids: list[TrajectoryID],
    ) -> list["HarborAgentOutput"]:
        """Tokenize meta-learning reasoning samples into the same format as regular outputs.

        Each sample is a (system+user → assistant) conversation with a reward.
        """
        if not samples:
            return []

        outputs: list[HarborAgentOutput] = []
        for i, sample in enumerate(samples):
            try:
                prompt_msgs = [m for m in sample.messages if m["role"] != "assistant"]
                response_msgs = [m for m in sample.messages if m["role"] == "assistant"]

                if not response_msgs:
                    continue

                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt_msgs,
                    add_generation_prompt=True,
                    return_dict=False,
                    tokenize=True,
                    chat_template=self.custom_chat_template_content,
                )

                response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
                    response_msgs,
                    self.tokenizer,
                    None,
                    chat_template=self.custom_chat_template_content,
                )

                max_response_tokens = max(0, self.max_seq_len - len(prompt_ids))
                response_ids = response_ids[:max_response_tokens]
                loss_mask = loss_mask[:max_response_tokens]

                meta_tid = TrajectoryID(
                    instance_id=sample.instance_id or f"meta_{sample.role}_{i}_{uuid4().hex[:8]}",
                    repetition_id=sample.repetition_id,
                )

                outputs.append(HarborAgentOutput(
                    response_ids=response_ids,
                    reward=sample.reward,
                    stop_reason="complete",
                    loss_mask=loss_mask,
                    prompt_ids=prompt_ids,
                    trajectory_id=meta_tid,
                    num_turns=1,
                ))

            except Exception as e:
                logger.warning(f"Failed to tokenize meta sample {i}: {e}")
                continue

        if outputs:
            logger.info(
                f"Meta-RL: tokenized {len(outputs)} meta samples "
                f"(avg response len={sum(len(o.response_ids) for o in outputs)/len(outputs):.0f})"
            )

        return outputs

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match "
                f"trajectory_ids count ({len(trajectory_ids)})"
            )

        all_outputs: List[HarborAgentOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id):
            result = await self.harbor_agent_loop(prompt=prompt, trajectory_id=trajectory_id)
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        all_outputs, rollout_metrics = self._mask_failed_instances_and_compute_metrics(all_outputs)

        # ---- Collect trace records ----
        trace_records = [output.trace_record for output in all_outputs if output.trace_record is not None]

        # ---- Meta-learning: record traces ----
        meta_training_samples: list[MetaTrainingSample] = []
        if self._meta_loop and trace_records:
            self._meta_loop.record_traces(trace_records)

            # Run meta cycle (diagnose → plan → patch → canary → promote)
            try:
                meta_metrics, meta_training_samples = await self._meta_loop.maybe_run_cycle(
                    run_name=self._run_name
                )
                rollout_metrics.update(meta_metrics)
            except Exception as e:
                logger.error(f"MetaLoop error: {e}")
                rollout_metrics["meta/error"] = str(e)

            # Log summary from legacy MetaController
            meta_result = self._meta_controller.summarize_failures(trace_records)
            rollout_metrics["meta/num_trace_records"] = len(trace_records)
            rollout_metrics["meta/num_failure_tags"] = len(meta_result.selected_failure_tags)
            rollout_metrics["meta/num_candidate_modules"] = len(meta_result.candidate_modules)
            logger.info(
                "Meta summary: failure_tags={} candidate_modules={}",
                meta_result.selected_failure_tags,
                meta_result.candidate_modules,
            )

        # ---- Meta-RL: tokenize and stage for a separate mini training step ----
        # Meta samples are NOT appended to the main batch (would break
        # validate_generator_output's prompts==responses check).  Instead
        # we store them on self._pending_meta_batch so the trainer can
        # pick them up after the main RL step.
        self._pending_meta_batch = None
        if meta_training_samples:
            meta_outputs = self._tokenize_meta_samples(meta_training_samples, trajectory_ids)
            self._pending_meta_batch = {
                "prompt_token_ids": [o.prompt_ids for o in meta_outputs],
                "response_ids": [o.response_ids for o in meta_outputs],
                "rewards": [o.reward for o in meta_outputs],
                "loss_masks": [o.loss_mask for o in meta_outputs],
                "stop_reasons": [o.stop_reason for o in meta_outputs],
                "rollout_logprobs": None,
            }
            rollout_metrics["meta/staged_samples"] = len(meta_outputs)
            logger.info(
                f"Meta-RL: {len(meta_outputs)} samples staged for "
                f"separate mini training step"
            )

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [o.prompt_ids for o in all_outputs],
            "response_ids": [o.response_ids for o in all_outputs],
            "rewards": [o.reward for o in all_outputs],
            "loss_masks": [o.loss_mask for o in all_outputs],
            "stop_reasons": [o.stop_reason for o in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output

    @staticmethod
    def _mask_failed_instances_and_compute_metrics(
        all_outputs: List[HarborAgentOutput],
    ) -> tuple[List[HarborAgentOutput], dict]:
        """Mutates all_outputs in-place: zeros out every output belonging to a failed instance.

        For a group of trajectories (n_samples_per_prompt for the same prompt),
        if one trajectory fails we skip training the entire group.
        """
        num_timeout_trajectories = 0
        num_error_trajectories = 0
        timeout_instance_ids = set()
        error_instance_ids = set()
        all_instance_ids = set()

        for output in all_outputs:
            cur_instance_id = output.trajectory_id.instance_id
            all_instance_ids.add(cur_instance_id)
            if output.stop_reason == "agent_timeout":
                num_timeout_trajectories += 1
                timeout_instance_ids.add(cur_instance_id)
            elif output.stop_reason == "error":
                num_error_trajectories += 1
                error_instance_ids.add(cur_instance_id)

        masked_instance_ids = timeout_instance_ids | error_instance_ids

        successful_outputs: List[HarborAgentOutput] = []
        for output in all_outputs:
            if output.trajectory_id.instance_id in masked_instance_ids:
                output.response_ids = [0]
                output.stop_reason = "error"
                output.loss_mask = [0]
                output.prompt_ids = [0]
                output.reward = 0
            else:
                successful_outputs.append(output)

        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if output.summarization_count > 0
            )
            rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
                1 for output in successful_outputs if output.stop_reason == "context_length"
            )
            rollout_metrics["generate/avg_num_turns"] = sum(output.num_turns for output in successful_outputs) / len(
                successful_outputs
            )
        else:
            rollout_metrics = {}

        rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
        rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
        rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)

        logger.info(
            f"\n# of masked instances: {len(mixed_instance_ids := masked_instance_ids)} / {len(all_instance_ids)}\n"
            f"# of timeout trajectories: {num_timeout_trajectories}\n"
            f"# of error trajectories: {num_error_trajectories}"
        )

        return all_outputs, rollout_metrics

    def _inject_meta_overrides(self, config) -> None:
        """Inject active meta overrides into the trial config's agent kwargs."""
        if not _ACTIVE_META_OVERRIDES:
            return
        overrides_copy = deepcopy(_ACTIVE_META_OVERRIDES)
        hooks_dict = overrides_copy.pop("_meta_hooks", None)

        if isinstance(config, DictConfig):
            was_struct = OmegaConf.is_struct(config)
            OmegaConf.set_struct(config, False)
            try:
                agent_kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
                agent_kwargs["meta_overrides"] = overrides_copy
                if hooks_dict:
                    agent_kwargs["meta_hooks"] = hooks_dict
            finally:
                OmegaConf.set_struct(config, was_struct)
        else:
            agent_kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
            agent_kwargs["meta_overrides"] = overrides_copy
            if hooks_dict:
                agent_kwargs["meta_hooks"] = hooks_dict

    async def harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
    ) -> HarborAgentOutput:
        """Run a single harbor agent."""
        reward = None
        chat_history = None
        summarization_count = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False
        results = None

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = uuid4().hex

                # ---- Inject active meta overrides ----
                self._inject_meta_overrides(config)

                trial_config = TrialConfig.model_validate(config)
                trial = Trial(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                # Parse exception type
                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                # Determine reward
                if is_agent_timeout_error:
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break
                elif is_context_length_error:
                    logger.debug(
                        f"{prefix} hit ContextLengthExceededError, will train with reward=0. Results: {results}"
                    )
                    reward = 0
                elif not results.verifier_result:
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue
                else:
                    reward = results.verifier_result.rewards["reward"]

                # Extract chat history and check for success
                chat_history = results.agent_result.metadata["all_messages"]
                summarization_count = results.agent_result.metadata.get("summarization_count", 0)
                num_turns = results.agent_result.metadata["n_episodes"]
                if len(chat_history) > 1 and chat_history[0]["role"] == "user":
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}. Results: {results}")
                    break
                else:
                    logger.warning(
                        f"{prefix} failed: Did not return a chat history with a user message. "
                        f"chat_history: {chat_history}\nResults: {results}"
                    )
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = (
                f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), "
                "will set loss mask to [0]."
            )
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            return HarborAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason=stop_reason,
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
                trace_record=_build_trace(
                    trajectory_id=trajectory_id,
                    agent_version=self._agent_name or "unknown",
                    prompt=prompt,
                    results=results,
                    reward=0,
                    stop_reason=stop_reason,
                    num_turns=0,
                    summarization_count=0,
                    successful=False,
                ),
            )

        # Build response from chat history
        assert chat_history[0]["role"] == "user", "The first message should be a user message"
        prompt_msg = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt_msg,
            add_generation_prompt=False,
            return_dict=False,
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)

        response_messages = chat_history[1:]
        assistant_logprobs = getattr(results.agent_result, "output_logprobs", None)
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs,
            chat_template=self.custom_chat_template_content,
        )

        max_response_tokens = max(0, self.max_seq_len - initial_prompt_length)
        if is_context_length_error or len(response_ids) > max_response_tokens:
            stop_reason = "context_length"
        else:
            stop_reason = "complete"

        if self.generator_cfg.apply_overlong_filtering and stop_reason == "context_length":
            loss_mask = [0] * len(loss_mask)

        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        task_successful = reward is not None and reward > 0

        return HarborAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            num_turns=num_turns,
            trace_record=_build_trace(
                trajectory_id=trajectory_id,
                agent_version=self._agent_name or "unknown",
                prompt=prompt,
                results=results,
                reward=reward,
                stop_reason=stop_reason,
                num_turns=num_turns or 0,
                summarization_count=summarization_count or 0,
                successful=task_successful,
            ),
        )


# =============================================================================
# Trace builder (local import to avoid circular dependencies)
# =============================================================================


def _build_trace(
    trajectory_id: Any,
    agent_version: str,
    prompt: Any,
    results: Any,
    reward: float,
    stop_reason: str,
    num_turns: int,
    summarization_count: int,
    successful: bool,
):
    """Build a TraceRecord from Harbor trial results."""
    from skyrl_agent.meta_toolkit.terminus.harbor_bridge import build_trace_from_harbor_trial

    return build_trace_from_harbor_trial(
        trajectory_id=trajectory_id,
        agent_version=agent_version,
        prompt=prompt,
        results=results,
        reward=reward,
        stop_reason=stop_reason,
        num_turns=num_turns,
        summarization_count=summarization_count,
        successful=successful,
    )
