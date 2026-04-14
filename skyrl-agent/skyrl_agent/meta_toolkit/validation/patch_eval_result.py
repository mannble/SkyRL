"""Patch evaluation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PatchEvalResult:
    """Result of comparing before/after agent performance on a canary task set."""

    before_score: float
    after_score: float
    delta_score: float
    regression: bool
    notes: str = ""
    num_tasks: int = 0
    before_successes: int = 0
    after_successes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
