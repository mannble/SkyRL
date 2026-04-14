"""Base adapter interface for agent-specific meta-learning integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry


class MetaOverrideAdapter(ABC):
    """Abstract adapter that bridges the meta-learning loop with a specific agent.

    Each agent type (terminus-2, generic installed agents, etc.)
    implements this interface to define:
    - What modules can be patched
    - How overrides are serialised for prompt injection
    - How overrides are injected into the trial config
    """

    @abstractmethod
    def agent_name_pattern(self) -> str:
        """Regex or prefix pattern that matches supported agent names."""
        ...

    def matches(self, agent_name: str) -> bool:
        """Check if this adapter handles the given agent name."""
        import re
        return bool(re.match(self.agent_name_pattern(), agent_name))

    @abstractmethod
    def build_registry(self) -> ModuleRegistry:
        """Build the patchable module registry for this agent type."""
        ...

    @abstractmethod
    def build_override_section(self, overrides: dict[str, Any]) -> str:
        """Serialise overrides into a text block for prompt injection."""
        ...

    @abstractmethod
    def inject_into_trial_config(self, config: dict[str, Any], overrides: dict[str, Any]) -> None:
        """Mutate the trial config dict to carry overrides to the agent."""
        ...

    def supported_override_files(self) -> list[str]:
        """List of override file basenames this adapter recognises."""
        return [
            "strategy_library.yaml",
        ]
