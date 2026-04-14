"""Promoter: accept or reject patch candidates based on canary eval results.

Phase 1 rules:
  - ACCEPT if delta_score > 0 and not regression
  - REJECT otherwise
On accept: writes accepted_patch.json and updates the active override symlink/copy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from skyrl_agent.meta_toolkit.validation.patch_eval_result import PatchEvalResult

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from skyrl_agent.meta_toolkit.runtime.patch_version_control import PatchVersionControl

# Override base path (must match PatchExecutor / CanaryRunner)
_OVERRIDE_BASE = Path(
    "/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2"
)


@dataclass
class PromotionDecision:
    """Outcome of evaluating a patch candidate."""

    candidate_id: str
    decision: str  # "accept" | "reject"
    delta_score: float
    regression: bool
    notes: str = ""
    promoted_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromoterConfig:
    """Thresholds for the promotion gate."""

    # Minimum delta to accept.
    # Must be strictly positive — delta=0 means "same as baseline", not an improvement.
    min_delta: float = 0.01

    # If delta below this, always reject
    reject_threshold: float = -0.05

    # Path to write accepted patch records
    accepted_patches_dir: str = "/tmp/skyrl-logs/accepted_patches"

    # Active override base (the live overrides that Harbor reads)
    active_override_base: str = str(_OVERRIDE_BASE)


class Promoter:
    """Gate keeper for patch promotion.

    Accepts patches when:
      1. delta_score > min_delta (improvement)
      2. no regression (delta >= reject_threshold)
      3. canary eval completed without error

    When a ``PatchVersionControl`` instance is attached, every accepted patch
    is automatically committed to the isolated git repo for full traceability.
    """

    def __init__(
        self,
        config: PromoterConfig | None = None,
        version_control: "PatchVersionControl | None" = None,
    ) -> None:
        self.config = config or PromoterConfig()
        self._vc = version_control
        self._accepted_dir = Path(self.config.accepted_patches_dir)
        self._accepted_dir.mkdir(parents=True, exist_ok=True)

    def decide(self, candidate_id: str, eval_result: PatchEvalResult) -> PromotionDecision:
        """Decide whether to accept or reject a patch candidate.

        Returns a PromotionDecision with the reason.
        """
        delta = eval_result.delta_score
        regression = eval_result.regression

        if eval_result.notes and "error" in eval_result.notes.lower():
            return PromotionDecision(
                candidate_id=candidate_id,
                decision="reject",
                delta_score=delta,
                regression=True,
                notes=f"Rejected: canary eval error — {eval_result.notes}",
                metadata=asdict(eval_result),
            )

        if delta < self.config.reject_threshold or regression:
            return PromotionDecision(
                candidate_id=candidate_id,
                decision="reject",
                delta_score=delta,
                regression=True,
                notes=f"Rejected: regression detected (delta={delta:.4f})",
                metadata=asdict(eval_result),
            )

        if delta < self.config.min_delta:
            return PromotionDecision(
                candidate_id=candidate_id,
                decision="reject",
                delta_score=delta,
                regression=False,
                notes=f"Rejected: delta below threshold (delta={delta:.4f}, min={self.config.min_delta})",
                metadata=asdict(eval_result),
            )

        # ACCEPT
        decision = PromotionDecision(
            candidate_id=candidate_id,
            decision="accept",
            delta_score=delta,
            regression=False,
            notes=f"Accepted: delta={delta:.4f}, no regression",
            promoted_at=datetime.utcnow().isoformat(),
            metadata=asdict(eval_result),
        )

        # Write accepted patch record
        self._write_accepted_record(decision, eval_result)

        return decision

    def _write_accepted_record(
        self, decision: PromotionDecision, eval_result: PatchEvalResult
    ) -> Path:
        """Write an accepted_patch.json record for auditability."""
        patch_id = decision.candidate_id or f"patch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        record_path = self._accepted_dir / f"{patch_id}.json"

        record = {
            "patch_id": patch_id,
            "decision": decision.decision,
            "delta_score": decision.delta_score,
            "regression": decision.regression,
            "notes": decision.notes,
            "promoted_at": decision.promoted_at,
            "eval_result": asdict(eval_result),
        }

        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            f"Promoter: wrote accepted patch record to {record_path} "
            f"(delta={decision.delta_score:.4f})"
        )
        return record_path

    def promote(
        self,
        candidate_id: str,
        patch_files: list[str],
        eval_result: PatchEvalResult,
        *,
        cycle: int = 0,
    ) -> PromotionDecision:
        """Full promote flow: decide → activate → version-control commit.

        After accepting, the patch override files (already written by
        PatchExecutor) are confirmed active and then committed to the
        isolated git repo for full traceability.
        """
        decision = self.decide(candidate_id, eval_result)

        if decision.decision == "accept":
            self._activate_overrides(patch_files, candidate_id)
            self._vc_commit(candidate_id, eval_result.delta_score, cycle=cycle)
            logger.info(f"Promoter: candidate {candidate_id} PROMOTED (delta={decision.delta_score:.4f})")
        else:
            logger.info(f"Promoter: candidate {candidate_id} REJECTED — {decision.notes}")

        return decision

    def _vc_commit(
        self, candidate_id: str, delta_score: float, *, cycle: int = 0
    ) -> None:
        """Commit the current override state to the version-control repo."""
        if self._vc is None:
            return
        try:
            self._vc.commit_patch(
                candidate_id=candidate_id,
                delta_score=delta_score,
                cycle=cycle,
                notes=f"auto-commit by Promoter",
            )
        except Exception as exc:
            logger.warning(f"Promoter: version-control commit failed — {exc}")

    def _activate_overrides(self, patch_files: list[str], candidate_id: str) -> None:
        """Ensure patch override files are live for Harbor to pick up.

        Phase 1: patch files are already written by PatchExecutor to the active
        override directory. This method just confirms visibility.
        """
        for path_str in patch_files:
            p = Path(path_str)
            if not p.exists():
                logger.warning(
                    f"Promoter: override file {path_str} does not exist — "
                    f"cannot activate for candidate {candidate_id}"
                )
            else:
                logger.debug(f"Promoter: override {path_str} is active for candidate {candidate_id}")
