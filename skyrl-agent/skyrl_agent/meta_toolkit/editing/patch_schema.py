from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .module_registry import ModuleRegistry


ChangeType = Literal["modify", "add", "delete", "rename"]
RiskLevel = Literal["low", "medium", "high"]


class PatchValidationError(ValueError):
    """Raised when a candidate patch violates registry or schema constraints."""


@dataclass(slots=True)
class PatchFileEdit:
    path: str
    change_type: ChangeType
    intent: str


@dataclass(slots=True)
class PatchCandidate:
    target_modules: list[str]
    files: list[PatchFileEdit]
    risk: RiskLevel
    required_tests: list[str]
    rollback_if: list[str] = field(default_factory=list)
    notes: str = ""

    def validate_against_registry(self, registry: ModuleRegistry) -> None:
        owned_paths = registry.owned_paths()
        for module_name in self.target_modules:
            registry.get(module_name)

        for file_edit in self.files:
            owner = owned_paths.get(file_edit.path)
            if owner is None:
                raise PatchValidationError(f"Patch touches unregistered path: {file_edit.path}")
            if owner not in self.target_modules:
                raise PatchValidationError(
                    f"Patch file {file_edit.path} belongs to {owner}, "
                    f"but target_modules={self.target_modules}"
                )
