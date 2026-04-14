"""Terminus-oriented registry examples and adapter primitives for the meta-learning toolkit."""

from .adapter import TerminusAdapter, TerminusRunResult, TerminusRuntime, TerminusTaskInput
from .patchable_modules import build_terminus_registry

__all__ = [
    "TerminusAdapter",
    "TerminusRunResult",
    "TerminusRuntime",
    "TerminusTaskInput",
    "build_terminus_registry",
]
