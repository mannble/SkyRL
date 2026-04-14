"""Meta-learning toolkit primitives for agent observation, editing, and promotion."""

from .editing.module_registry import PatchableModule, ModuleRegistry
from .editing.patch_schema import PatchCandidate, PatchFileEdit
from .observability.trace_schema import TraceRecord, TraceEvent
from .adapters.base import MetaOverrideAdapter
from .hooks import HookContext, HookExecutor, HookPoint

__all__ = [
    "HookContext",
    "HookExecutor",
    "HookPoint",
    "MetaOverrideAdapter",
    "ModuleRegistry",
    "PatchCandidate",
    "PatchFileEdit",
    "PatchableModule",
    "TraceEvent",
    "TraceRecord",
]
