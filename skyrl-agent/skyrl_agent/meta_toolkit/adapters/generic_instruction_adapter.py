"""Generic instruction-injection adapter for non-Terminus agents.

For agents that don't have a dedicated ``meta_overrides`` kwarg (e.g.
Claude Code, Aider, or any ``BaseInstalledAgent``), overrides are injected
by prepending a strategy block to the task instruction text.
"""

from __future__ import annotations

from typing import Any

from skyrl_agent.meta_toolkit.adapters.base import MetaOverrideAdapter
from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry, PatchableModule
from skyrl_agent.meta_toolkit.validation import (
    SMOKE_TERMINAL,
    TB2_CANARY,
)


class GenericInstructionAdapter(MetaOverrideAdapter):
    """Fallback adapter that works with any agent by modifying the task instruction.

    Override injection happens at the instruction level: a ``[STRATEGY GUIDANCE]``
    block is prepended to the task instruction markdown before the trial runs.
    """

    def agent_name_pattern(self) -> str:
        return r".*"

    def build_registry(self) -> ModuleRegistry:
        registry = ModuleRegistry()
        registry.register(PatchableModule(
            name="strategy_library",
            description="Pattern-matched step-by-step strategies injected via instruction.",
            owned_files=[],
            interface_contract="instruction-prepend only",
            allowed_change_types=["modify"],
            forbidden_change_types=["delete"],
            required_tests=[SMOKE_TERMINAL],
            risk_level="low",
        ))
        return registry

    def build_override_section(self, overrides: dict[str, Any]) -> str:
        if not overrides:
            return ""

        sections: list[str] = []

        strategies = overrides.get("strategy_library", {})
        entries = strategies.get("strategies", [])
        if isinstance(entries, list) and entries:
            for entry in entries:
                pattern = entry.get("pattern", "")
                steps = entry.get("steps", [])
                if pattern and steps:
                    steps_text = "; ".join(f"{i+1}) {s}" for i, s in enumerate(steps))
                    sections.append(f"- When {pattern}: {steps_text}")

        if not sections:
            return ""
        return "[STRATEGY GUIDANCE]\n" + "\n".join(sections)

    def inject_into_trial_config(self, config: dict[str, Any], overrides: dict[str, Any]) -> None:
        section = self.build_override_section(overrides)
        if section:
            config.setdefault("_meta_instruction_prefix", section)

    def supported_override_files(self) -> list[str]:
        return [
            "strategy_library.yaml",
        ]
