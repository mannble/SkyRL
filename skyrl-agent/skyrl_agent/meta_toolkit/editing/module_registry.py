from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PatchableModule:
    """Registry entry describing one model-editable agent module."""

    name: str
    description: str
    owned_files: list[str]
    interface_contract: str
    allowed_change_types: list[str]
    forbidden_change_types: list[str]
    required_tests: list[str]
    dependent_modules: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    allow_new_helpers: bool = False
    allow_new_config: bool = True


class ModuleRegistry:
    """In-memory registry for patchable modules.

    The registry is intentionally simple so it can be reused by a future
    meta-controller, patch validator, or reviewer.
    """

    def __init__(self) -> None:
        self._modules: dict[str, PatchableModule] = {}

    def register(self, module: PatchableModule) -> None:
        if module.name in self._modules:
            raise ValueError(f"Duplicate patchable module: {module.name}")
        self._modules[module.name] = module

    def get(self, name: str) -> PatchableModule:
        try:
            return self._modules[name]
        except KeyError as exc:
            raise KeyError(f"Unknown patchable module: {name}") from exc

    def list(self) -> list[PatchableModule]:
        return list(self._modules.values())

    def owned_paths(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for module in self._modules.values():
            for path in module.owned_files:
                mapping[path] = module.name
        return mapping
