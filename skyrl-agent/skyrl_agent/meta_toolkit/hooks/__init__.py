"""Agent hook system: allows meta-learning to inject Python code into agent execution."""

from .hook_context import HookContext
from .hook_executor import HookExecutor, HookPoint

__all__ = ["HookContext", "HookExecutor", "HookPoint"]
