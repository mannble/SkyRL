"""Cross-cycle diagnosis history for context accumulation.

Maintains a rolling window of past cycle summaries so the diagnoser can
see what was tried before and whether it helped, avoiding repetitive patches.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTORY = 20


@dataclass
class CycleSummary:
    """Compact record of one meta-learning cycle's outcome."""

    cycle_id: int
    diagnosed_types: list[str]
    candidates_generated: int
    promoted: int
    rejected: int
    best_delta: float
    active_overrides_snapshot: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class DiagnosisHistory:
    """Thread-safe rolling history of meta-learning cycle outcomes.

    Persists to a JSONL file so history survives restarts.
    """

    def __init__(self, persist_path: str | Path | None = None, max_entries: int = _MAX_HISTORY) -> None:
        self._entries: list[CycleSummary] = []
        self._max = max_entries
        self._path = Path(persist_path) if persist_path else None
        if self._path and self._path.exists():
            self._load()

    def append(self, summary: CycleSummary) -> None:
        self._entries.append(summary)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]
        if self._path:
            self._persist_append(summary)

    def recent(self, n: int = 5) -> list[CycleSummary]:
        return self._entries[-n:]

    def format_for_prompt(self, n: int = 5) -> str:
        """Render recent history as a text block suitable for LLM context."""
        entries = self.recent(n)
        if not entries:
            return "(No prior cycles recorded.)"

        lines: list[str] = []
        for e in entries:
            status = f"promoted={e.promoted}, rejected={e.rejected}, best_delta={e.best_delta:.3f}"
            overrides_keys = list(e.active_overrides_snapshot.keys()) if e.active_overrides_snapshot else []
            lines.append(
                f"- Cycle {e.cycle_id}: diagnosed={e.diagnosed_types}, "
                f"{status}, active_modules={overrides_keys}"
            )
            if e.notes:
                lines.append(f"  Note: {e.notes}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def _persist_append(self, summary: CycleSummary) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to persist diagnosis history: {exc}")

    def _load(self) -> None:
        try:
            for line in self._path.read_text(encoding="utf-8").strip().splitlines():
                data = json.loads(line)
                self._entries.append(CycleSummary(**{
                    k: v for k, v in data.items()
                    if k in CycleSummary.__dataclass_fields__
                }))
            if len(self._entries) > self._max:
                self._entries = self._entries[-self._max:]
        except Exception as exc:
            logger.warning(f"Failed to load diagnosis history from {self._path}: {exc}")
