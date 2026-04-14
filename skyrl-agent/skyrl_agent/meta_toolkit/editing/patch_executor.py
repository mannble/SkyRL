"""Patch executor: applies PatchCandidate edits to override files and code hooks.

Write targets:
  - override_base: strategy_library.yaml + hooks/*.py (SkyRL-controlled)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from skyrl_agent.meta_toolkit.editing.patch_schema import ChangeType, PatchCandidate, PatchFileEdit

_OVERRIDE_SCHEMA = {
    "strategy_library": [
        "strategies",
    ],
}


def _timestamp() -> str:
    return datetime.utcnow().isoformat()


class PatchExecutor:
    """Execute PatchCandidate edits by writing strategy_library.yaml and code hooks."""

    def __init__(
        self,
        override_base: str | Path | None = None,
        template_base: str | Path | None = None,  # kept for backward compat, ignored
    ) -> None:
        self._override_base = (
            Path(override_base)
            if override_base
            else Path(
                "/home/ray/SkyRL/examples/train_integrations/harbor/meta_patches/terminus2"
            )
        )

    def apply(self, candidate: PatchCandidate) -> list[str]:
        """Apply a PatchCandidate by writing its override files.

        Returns list of paths that were written.
        """
        written: list[str] = []
        for file_edit in candidate.files:
            path = self._write_override_file(file_edit)
            if path:
                written.append(path)
        return written

    def _write_override_file(self, file_edit: PatchFileEdit) -> str | None:
        """Write or update a single override file or hook.

        Returns the written path, or None on failure.
        """
        path = Path(file_edit.path)

        if not str(path).startswith(str(self._override_base)):
            raise PermissionError(
                f"PatchExecutor refuses to write outside allowed zone: {file_edit.path}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        # Hook files: validate Python syntax then write
        if path.suffix == ".py" and "/hooks/" in str(path):
            return self._write_hook(path, file_edit.intent)

        hints = self._parse_intent(file_edit.intent)

        if path.suffix in {".yaml", ".yml"}:
            return self._write_yaml(path, hints, file_edit.change_type)
        elif path.suffix == ".json":
            return self._write_json(path, hints, file_edit.change_type)
        elif path.suffix in {".txt", ".md"}:
            return self._write_text(path, hints, file_edit.change_type, file_edit.intent)
        else:
            return self._write_yaml(path, hints, file_edit.change_type)

    def _write_yaml(
        self, path: Path, hints: dict[str, Any], change_type: ChangeType
    ) -> str:
        """Write a YAML override file with the given hints."""
        # Strategy library uses dedicated merge logic
        if path.name == "strategy_library.yaml" and "strategy_edits" in hints:
            return self._write_strategy_library(path, hints)

        # Load existing data if present
        data: dict[str, Any] = {}
        if path.exists() and change_type == "modify":
            try:
                data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
            except Exception:
                data = {}

        # Merge new hints
        data.update(hints)
        data["_patch_applied_at"] = _timestamp()
        data["_change_type"] = change_type

        # Write back as YAML
        conf = OmegaConf.create(data)
        path.write_text(OmegaConf.to_yaml(conf), encoding="utf-8")
        return str(path)

    def _write_strategy_library(self, path: Path, hints: dict[str, Any]) -> str:
        """Apply add/edit/remove operations to strategy_library.yaml."""
        import json as _json
        import logging as _logging

        _logger = _logging.getLogger(__name__)

        # Load existing strategies
        strategies: list[dict[str, Any]] = []
        if path.exists():
            try:
                data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
                strategies = list(data.get("strategies", []))
            except Exception:
                strategies = []

        edits_raw = hints.get("strategy_edits", [])
        if isinstance(edits_raw, str):
            try:
                edits_raw = _json.loads(edits_raw)
            except (ValueError, TypeError):
                edits_raw = []

        # Process edits in order: removes first (descending index), then edits, then adds
        removes = sorted(
            [e for e in edits_raw if isinstance(e, dict) and e.get("action") == "remove"],
            key=lambda e: e.get("index", -1),
            reverse=True,
        )
        edits = [e for e in edits_raw if isinstance(e, dict) and e.get("action") == "edit"]
        adds = [e for e in edits_raw if isinstance(e, dict) and e.get("action") == "add"]

        for r in removes:
            idx = r.get("index", -1)
            if 0 <= idx < len(strategies):
                removed = strategies.pop(idx)
                _logger.info(f"Strategy library: removed index {idx} ({removed.get('pattern', '?')})")

        for e in edits:
            idx = e.get("index", -1)
            new_strategy = e.get("strategy", {})
            if 0 <= idx < len(strategies) and isinstance(new_strategy, dict):
                old_pattern = strategies[idx].get("pattern", "?")
                strategies[idx] = new_strategy
                _logger.info(f"Strategy library: edited index {idx} ({old_pattern} -> {new_strategy.get('pattern', '?')})")

        for a in adds:
            new_strategy = a.get("strategy", {})
            if isinstance(new_strategy, dict) and "pattern" in new_strategy and "steps" in new_strategy:
                # Dedup: skip if a strategy with very similar pattern already exists
                pattern = new_strategy["pattern"].lower().strip()
                duplicate = False
                for existing in strategies:
                    existing_pattern = existing.get("pattern", "").lower().strip()
                    if existing_pattern == pattern:
                        duplicate = True
                        break
                if not duplicate:
                    strategies.append(new_strategy)
                    _logger.info(f"Strategy library: added '{new_strategy['pattern']}'")
                else:
                    _logger.info(f"Strategy library: skipped duplicate '{new_strategy['pattern']}'")

        out: dict[str, Any] = {
            "strategies": strategies,
            "_patch_applied_at": _timestamp(),
            "_change_type": "modify",
        }
        conf = OmegaConf.create(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(OmegaConf.to_yaml(conf), encoding="utf-8")
        return str(path)

    def _write_json(
        self, path: Path, hints: dict[str, Any], change_type: ChangeType
    ) -> str:
        """Write a JSON override file with the given hints."""
        data: dict[str, Any] = {}
        if path.exists() and change_type == "modify":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        data.update(hints)
        data["_patch_applied_at"] = _timestamp()
        data["_change_type"] = change_type
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def _write_text(
        self, path: Path, hints: dict[str, Any], change_type: ChangeType, intent: str
    ) -> str:
        """Write a plain-text override file (append-mode hint)."""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
        else:
            existing = ""
        entry = (
            f"\n--- patch @ {_timestamp()} ---\n"
            f"change_type: {change_type}\n"
            f"hints: {hints}\n"
            f"intent: {intent}\n"
        )
        path.write_text(existing + entry, encoding="utf-8")
        return str(path)

    def _write_hook(self, path: Path, source: str) -> str | None:
        """Write a validated Python hook file.

        The intent/source is the raw Python source code of the hook function.
        """
        import logging as _logging

        _logger = _logging.getLogger(__name__)

        try:
            from skyrl_agent.meta_toolkit.hooks.hook_executor import (
                _validate_hook_source, HookPoint,
            )
            hook_name = path.stem
            try:
                hp = HookPoint(hook_name)
            except ValueError:
                hp = None
            errors = _validate_hook_source(source, hp)
            if errors:
                _logger.error(f"Hook validation failed for {path.name}: {errors}")
                return None
        except ImportError:
            _logger.warning("Hook validator not available, writing without validation")

        if path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

        path.write_text(source, encoding="utf-8")
        _logger.info(f"Hook written: {path} ({len(source)} chars)")
        return str(path)

    @staticmethod
    def _parse_intent(intent: str) -> dict[str, Any]:
        """Extract structured hints from an intent string.

        Supports:
          1. Strategy edits JSON: {"strategy_edits": [...], "rationale": ...}
          2. LLM-generated JSON: {"module": ..., "fields": {...}, "rationale": ...}
          3. Raw text fallback.
        """
        import json as _json

        try:
            parsed = _json.loads(intent)
            if isinstance(parsed, dict) and "strategy_edits" in parsed:
                return {"strategy_edits": parsed["strategy_edits"],
                        "intent_raw": parsed.get("rationale", intent[:200])}
            if isinstance(parsed, dict) and "fields" in parsed:
                fields = parsed["fields"]
                if isinstance(fields, dict) and fields:
                    hints: dict[str, Any] = dict(fields)
                    hints["intent_raw"] = parsed.get("rationale", intent[:200])
                    return hints
        except (ValueError, TypeError):
            pass

        return {"intent_raw": intent}

    def apply_hint(self, module_name: str, hints: dict[str, Any]) -> str:
        """Convenience: apply a dict of hints directly to a module's override file.

        Looks up the override path from _MODULE_OVERRIDE_FILE.
        """
        from .patch_planner import _MODULE_OVERRIDE_FILE

        override_file = _MODULE_OVERRIDE_FILE.get(module_name)
        if not override_file:
            raise ValueError(f"No override file mapped for module: {module_name}")

        path = self._override_base / override_file
        file_edit = PatchFileEdit(
            path=str(path),
            change_type="modify",
            intent=str(hints),
        )
        hints["_module_name"] = module_name
        return self._write_yaml(path, hints, "modify")
