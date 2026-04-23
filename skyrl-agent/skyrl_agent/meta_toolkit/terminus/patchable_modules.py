from __future__ import annotations

from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry, PatchableModule
from skyrl_agent.meta_toolkit.validation import (
    SMOKE_TERMINAL,
    TB2_CANARY,
)


def build_terminus_registry(base_path: str = "terminus") -> ModuleRegistry:
    """Build a Terminus-style patchable module registry.

    Only two modification tiers remain:
      1. strategy_library  — prompt-level strategy injection
      2. hook:*            — model-generated Python code hooks
    """

    registry = ModuleRegistry()

    # ---- Strategy library (prompt-level) ----

    registry.register(
        PatchableModule(
            name="strategy_library",
            description="Learned strategy patterns injected into the agent prompt.",
            owned_files=[f"{base_path}/strategy_library.yaml"],
            interface_contract="YAML with 'strategies' list; each entry has 'pattern' and 'steps'",
            allowed_change_types=["create", "modify"],
            forbidden_change_types=["delete"],
            required_tests=[TB2_CANARY],
            risk_level="low",
        )
    )

    # ---- Code hooks: model-generated Python injected into agent loop ----

    for hook_name, hook_desc in [
        ("hook:before_execute", "Transform or reorder commands before execution"),
        ("hook:after_execute", "Transform terminal output before it becomes the next prompt"),
        ("hook:on_timeout", "Custom handling when a command times out"),
        ("hook:before_llm_call", "Transform the prompt just before sending to the LLM"),
        ("hook:after_round", "Post-round control: inject text to verify completion, detect loops, guide agent"),
    ]:
        registry.register(
            PatchableModule(
                name=hook_name,
                description=(
                    f"Model-generated Python hook: {hook_desc}. "
                    "Runs in sandboxed try/except with limited imports."
                ),
                owned_files=[f"{base_path}/hooks/{hook_name.split(':')[1]}.py"],
                interface_contract=(
                    "Must define a function named 'hook' with correct signature; "
                    "only stdlib imports allowed (re, json, math, collections, etc.)"
                ),
                allowed_change_types=["create", "modify"],
                forbidden_change_types=["delete"],
                required_tests=[SMOKE_TERMINAL, TB2_CANARY],
                risk_level="high",
                allow_new_helpers=True,
            )
        )

    return registry
