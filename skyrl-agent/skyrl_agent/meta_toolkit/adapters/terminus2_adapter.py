"""Terminus-2 adapter: injects strategy library via agent kwargs + prompt prefix.

Supports two levels of modification:
  1. strategy_library (prompt-level strategy injection)
  2. code_hooks (Python functions injected into agent loop at hook points)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from skyrl_agent.meta_toolkit.adapters.base import MetaOverrideAdapter
from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry

class Terminus2Adapter(MetaOverrideAdapter):
    """Adapter for Harbor Terminus-2 (and the bare 'terminus' alias).

    Terminus-2 supports ``meta_overrides`` (strategy_library only) and
    ``meta_hooks`` (code hooks) as constructor kwargs.
    """

    def agent_name_pattern(self) -> str:
        return r"terminus(-2)?$"

    def build_registry(self) -> ModuleRegistry:
        from skyrl_agent.meta_toolkit.terminus.patchable_modules import build_terminus_registry
        return build_terminus_registry()

    def build_override_section(self, overrides: dict[str, Any]) -> str:
        """Build [LEARNED STRATEGIES] text from strategy_library only."""
        if not overrides:
            return ""

        sections: list[str] = []
        strategies = overrides.get("strategy_library", {})

        if isinstance(strategies, dict):
            entries = strategies.get("strategies", [])
            if isinstance(entries, list) and entries:
                sections.append("[LEARNED STRATEGIES]")
                for entry in entries:
                    pattern = entry.get("pattern", "")
                    steps = entry.get("steps", [])
                    if pattern and steps:
                        sections.append(f"  When: {pattern}")
                        for j, step in enumerate(steps, 1):
                            sections.append(f"    {j}. {step}")

        if not sections:
            return ""
        return "\n".join(sections)

    def inject_into_trial_config(self, config: dict[str, Any], overrides: dict[str, Any]) -> None:
        agent_kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
        agent_kwargs["meta_overrides"] = deepcopy(overrides)
