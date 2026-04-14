import asyncio
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


@dataclass
class MetaLoopConfig:
    """Configuration for the meta-learning closed loop."""

    enabled: bool = True
    interval_batches: int = 20  # Run meta cycle every N batches
    max_candidates: int = 2  # Max patch candidates per cycle
    canary_num_tasks: int = 4  # Tasks for canary eval
    canary_n_samples: int = 3  # Trials per task per side (baseline/candidate)
    override_base: str = "/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2"
    log_dir: str = "/tmp/skyrl-logs"

    # Phase 3: LLM-based diagnosis & patch planning
    llm_model: str = ""  # e.g. "gpt-4o", "qwen3-32b" — empty = use rule-based
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096

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

        if self._use_llm:
            from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient, MetaLLMConfig
            from skyrl_agent.meta_toolkit.diagnosis.interactive_diagnoser import InteractiveDiagnoser
            from skyrl_agent.meta_toolkit.diagnosis.diagnosis_history import DiagnosisHistory
            from skyrl_agent.meta_toolkit.editing.llm_patch_planner import LLMPatchPlanner

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
                override_base=config.override_base,
            )
            self._llm_planner = LLMPatchPlanner(
                self._llm_client, registry, config.override_base,
                agent_name=agent_name,
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
                accepted_patches_dir=str(Path(config.log_dir) / "accepted_patches"),
            ),
            version_control=self._vc,
        )

        self._batch_counter = 0
        self._candidate_counter = 0
        self._accumulated_traces: list = []
        self._trace_writer_initialized = False

        if trial_fn is not None:
            logger.info("MetaLoop: canary enabled (real Harbor Trial evaluation)")
        else:
            logger.info("MetaLoop: canary placeholder (no trial runner)")

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

            if meta_overrides:
                overrides_for_agent = deepcopy(meta_overrides)
                # Extract hooks and pass them separately
                hooks_dict = overrides_for_agent.pop("_meta_hooks", None)
                config["agent"]["kwargs"]["meta_overrides"] = overrides_for_agent
                if hooks_dict:
                    config["agent"]["kwargs"]["meta_hooks"] = hooks_dict

            config_dict = OmegaConf.to_container(config, resolve=True)
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
        logger.info(f"Meta: diagnosed {len(diagnoses)} problem types: {[d.problem_type for d in diagnoses]}")

        # ---- 3. Plan ----
        n_plan_samples = self.config.max_candidates
        if self._use_llm:
            candidates = await self._llm_planner.plan_from_batch(
                diagnoses, traces, n_samples=n_plan_samples
            )
        else:
            candidates = self._rule_planner.plan_from_batch(diagnoses)
            candidates = candidates[:n_plan_samples]
        logger.info(f"Meta: generated {len(candidates)} patch candidates")

        metrics: dict[str, Any] = {
            "meta/cycle_batch": self._batch_counter,
            "meta/num_diagnoses": len(diagnoses),
            "meta/num_candidates": len(candidates),
        }

        if not candidates:
            return metrics, []

        # ---- 4. Prepare canary: baseline + candidates in PARALLEL ----
        canary_task_paths = self._extract_canary_task_paths(traces, diagnoses)
        baseline_overrides = deepcopy(_ACTIVE_META_OVERRIDES) or None

        candidate_deltas: list[float] = []
        promoted = 0
        rejected = 0
        best_candidate_overrides = None
        best_delta = float("-inf")

        import shutil
        import tempfile

        # Pre-snapshot before any candidate modifications
        pre_all_sha = self._vc.snapshot(label="pre-all-candidates")

        # Phase A: prepare each candidate in an isolated temp directory
        prepared: list[dict] = []
        for i, candidate in enumerate(candidates):
            self._candidate_counter += 1
            cid = f"candidate_{self._candidate_counter}"
            tmp_dir = tempfile.mkdtemp(prefix=f"meta_canary_{cid}_")
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
                patch_files = tmp_executor.apply(remapped_candidate)
                overrides = self._load_override_content_from_dir(tmp_dir)
                prepared.append({
                    "candidate": candidate,
                    "candidate_id": cid,
                    "tmp_dir": tmp_dir,
                    "patch_files": patch_files,
                    "overrides": overrides,
                })
                logger.info(f"Meta: prepared {cid} in {tmp_dir} → {len(patch_files)} files")
            except Exception as e:
                logger.error(f"Meta: error preparing {cid}: {e}")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                candidate_deltas.append(-1.0)
                rejected += 1

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
                candidate_deltas = [-1.0] * len(candidates)
                rejected = len(candidates)
            else:
                baseline_scores = baseline_result

                # Phase C: compare each candidate against baseline, pick the best
                best_prepared = None
                for info, raw_result in zip(prepared, candidate_raw_results):
                    cid = info["candidate_id"]
                    if isinstance(raw_result, Exception):
                        logger.error(f"Meta: canary eval failed for {cid}: {raw_result}")
                        candidate_deltas.append(-1.0)
                        rejected += 1
                        continue

                    after_scores = raw_result
                    task_labels = [str(p) for p in canary_task_paths]
                    eval_result = self._canary._comparison.evaluate(
                        baseline_scores, after_scores,
                        task_ids=task_labels,
                        metadata={"candidate_id": cid, "patch_files": info["patch_files"]},
                    )
                    logger.info(
                        f"Canary {cid}: baseline_avg={sum(baseline_scores)/max(len(baseline_scores),1):.3f} "
                        f"candidate_avg={sum(after_scores)/max(len(after_scores),1):.3f} "
                        f"delta={eval_result.delta_score:.4f}"
                    )

                    candidate_deltas.append(eval_result.delta_score)
                    decision = self._promoter.promote(
                        candidate_id=cid,
                        patch_files=info["patch_files"],
                        eval_result=eval_result,
                        cycle=self._batch_counter,
                    )
                    if decision.decision == "accept":
                        promoted += 1
                        if eval_result.delta_score > best_delta:
                            best_delta = eval_result.delta_score
                            best_candidate_overrides = info["overrides"]
                            best_prepared = info
                    else:
                        rejected += 1
                    logger.info(
                        f"Meta: {cid} → {decision.decision} "
                        f"(delta={decision.delta_score:.4f}, notes={decision.notes})"
                    )

                # Phase D: apply the best candidate to the real override_base
                if best_prepared is not None:
                    self._vc.restore(pre_all_sha)
                    self._executor.apply(best_prepared["candidate"])
                else:
                    self._vc.restore(pre_all_sha)

            # Cleanup temp directories
            for info in prepared:
                shutil.rmtree(info["tmp_dir"], ignore_errors=True)
        else:
            # No prepared candidates (all failed during preparation) — skip canary entirely
            logger.warning("Meta: all candidates failed preparation, skipping canary evaluation")
            candidate_deltas = [-1.0] * len(candidates)
            rejected = len(candidates)

        if best_candidate_overrides is not None:
            self._inject_overrides_to_agent_config(best_candidate_overrides)

        metrics.update({
            "meta/promoted": promoted,
            "meta/rejected": rejected,
        })

        # Write human-readable cycle summary
        self._save_cycle_summary(
            diagnoses, candidates, candidate_deltas, promoted, rejected, best_delta,
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

    def _save_conversations(self, conversations, candidate_deltas: list[float]) -> None:
        """Persist LLM conversations to separate JSONL files by role.

        Output:
          {log_dir}/diagnosis_conversations.jsonl  — diagnosis worker turns
          {log_dir}/planning_conversations.jsonl   — planning LLM calls
          {log_dir}/meta_conversations.jsonl       — all (backward compat)
        """
        import os
        os.makedirs(self.config.log_dir, exist_ok=True)

        all_path = os.path.join(self.config.log_dir, "meta_conversations.jsonl")
        diag_path = os.path.join(self.config.log_dir, "diagnosis_conversations.jsonl")
        plan_path = os.path.join(self.config.log_dir, "planning_conversations.jsonl")

        try:
            with (
                open(all_path, "a", encoding="utf-8") as f_all,
                open(diag_path, "a", encoding="utf-8") as f_diag,
                open(plan_path, "a", encoding="utf-8") as f_plan,
            ):
                for i, conv in enumerate(conversations):
                    reward = candidate_deltas[i] if i < len(candidate_deltas) else None
                    record = {
                        "cycle": self._batch_counter,
                        "role": conv.role,
                        "messages": conv.messages,
                        "response": conv.response[:3000],
                        "reward": reward,
                    }
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                    f_all.write(line)
                    if conv.role == "diagnosis":
                        f_diag.write(line)
                    elif conv.role == "planning":
                        f_plan.write(line)
            logger.info(
                f"Meta: saved {len(conversations)} conversations "
                f"(diag/plan/all → {self.config.log_dir})"
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
    ) -> None:
        """Append a human-readable cycle summary to {log_dir}/cycle_summary.log."""
        import os
        from datetime import datetime

        summary_path = os.path.join(self.config.log_dir, "cycle_summary.log")
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
                        f"  tasks={d.affected_task_ids[:3]}\n"
                    )
                    f.write(f"    hypothesis: {hyp[:120]}\n")
                f.write("\n")

                # Candidate summary
                f.write(f"[Planning] {len(candidates)} patch candidates generated:\n")
                for i, cand in enumerate(candidates):
                    delta = candidate_deltas[i] if i < len(candidate_deltas) else None
                    delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
                    status = "ACCEPTED" if delta is not None and delta > 0 else "REJECTED"
                    f.write(
                        f"  candidate_{i+1}: {cand.target_modules}"
                        f"  risk={cand.risk}  delta={delta_str}  → {status}\n"
                    )
                    if cand.notes:
                        f.write(f"    notes: {cand.notes[:150]}\n")
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

    def _extract_canary_task_paths(
        self, traces: list, diagnoses: list | None = None
    ) -> list:
        """Select task paths for canary evaluation, informed by diagnosis results.

        Strategy:
          1. Collect per-task stats (avg reward, failure count) from all traces.
          2. If diagnoses provide affected_task_ids, prioritise those tasks.
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

        # sort failures: diagnosed first, then by avg_reward ascending (worst first)
        failed_tasks.sort(key=lambda e: (not e["diagnosed"], e["avg_reward"]))

        # reserve 1 slot for regression detection (a successful task)
        regression_slots = min(1, max_tasks // 4, len(success_tasks))
        failure_slots = max_tasks - regression_slots

        paths: list = []
        seen: set[str] = set()

        for entry in failed_tasks[:failure_slots]:
            key = str(entry["prompt"])
            if key not in seen:
                seen.add(key)
                paths.append(entry["prompt"])

        # pick the success task with the highest reward (most likely to regress visibly)
        success_tasks.sort(key=lambda e: e["avg_reward"], reverse=True)
        for entry in success_tasks:
            if len(paths) >= max_tasks:
                break
            key = str(entry["prompt"])
            if key not in seen:
                seen.add(key)
                paths.append(entry["prompt"])

        return paths[:max_tasks]

    def _inject_overrides_to_agent_config(self, overrides: dict[str, dict]) -> None:
        """Store active overrides so HarborGenerator can inject them into trial config."""
        _ACTIVE_META_OVERRIDES.update(overrides)

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
