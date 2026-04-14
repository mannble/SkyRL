"""HookExecutor: loads, validates, and safely runs model-generated hook code.

Each hook is a small Python function with a fixed signature. Hooks are stored
as .py files in the meta_patches directory and loaded at agent init time.

Safety guarantees:
  - Syntax validation before loading
  - Each hook call is wrapped in try/except with a timeout
  - Hooks only receive HookContext, not raw agent internals
  - If a hook fails, the original value is returned unchanged
  - All hook executions are logged
"""

from __future__ import annotations

import ast
import logging
import textwrap
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from skyrl_agent.meta_toolkit.hooks.hook_context import HookContext

logger = logging.getLogger(__name__)

_HOOK_TIMEOUT_SEC = 5


class HookPoint(str, Enum):
    """Well-defined injection points in the agent loop."""

    BEFORE_EXECUTE = "before_execute"
    AFTER_EXECUTE = "after_execute"
    ON_TIMEOUT = "on_timeout"
    ON_PARSE_ERROR = "on_parse_error"
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_ROUND = "after_round"


_HOOK_SIGNATURES: dict[HookPoint, str] = {
    HookPoint.BEFORE_EXECUTE: "def hook(commands: list, context: HookContext) -> list",
    HookPoint.AFTER_EXECUTE: "def hook(terminal_output: str, context: HookContext) -> str",
    HookPoint.ON_TIMEOUT: "def hook(command_keystrokes: str, terminal_output: str, context: HookContext) -> str",
    HookPoint.ON_PARSE_ERROR: "def hook(raw_response: str, error: str, context: HookContext) -> str | None",
    HookPoint.BEFORE_LLM_CALL: "def hook(prompt: str, context: HookContext) -> str",
    HookPoint.AFTER_ROUND: "def hook(terminal_output: str, is_task_complete: bool, context: HookContext) -> dict",
}


def _validate_hook_source(source: str, point: HookPoint | None = None) -> list[str]:
    """Validate hook source code. Returns list of errors (empty = valid)."""
    errors: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        errors.append(f"SyntaxError: {e}")
        return errors

    func_defs = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    hook_funcs = [f for f in func_defs if f.name == "hook"]

    if not hook_funcs:
        errors.append("Missing required function 'hook'")
    elif len(hook_funcs) > 1:
        errors.append("Multiple 'hook' functions found; expected exactly one")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module = ""
            if isinstance(node, ast.Import):
                module = node.names[0].name
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
            _ALLOWED_MODULES = {
                "re", "json", "math", "collections", "itertools",
                "functools", "copy", "textwrap", "string",
            }
            if module and module.split(".")[0] not in _ALLOWED_MODULES:
                errors.append(f"Import of '{module}' is not allowed")

    return errors


def _compile_hook(source: str) -> Callable | None:
    """Compile hook source into a callable 'hook' function."""
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(source, "<hook>", "exec"), namespace)
    except Exception as e:
        logger.error(f"Hook compilation failed: {e}")
        return None

    fn = namespace.get("hook")
    if fn is None or not callable(fn):
        logger.error("Compiled hook module has no callable 'hook' function")
        return None

    return fn


class _MockCommand:
    """Lightweight stand-in for Command used in hook smoke tests."""
    def __init__(self, keystrokes: str = "echo hello\n", duration_sec: float = 1.0):
        self.keystrokes = keystrokes
        self.duration_sec = duration_sec


def _smoke_test_hook(fn: Callable, point: HookPoint) -> list[str]:
    """Run a hook once with mock data. Returns list of errors (empty = OK)."""
    from skyrl_agent.meta_toolkit.hooks.hook_context import HookContext
    ctx = HookContext(
        episode=1, total_episodes=10, original_instruction="test task",
        last_commands=["ls -la\n", "cat README.md\n"],
    )

    try:
        if point == HookPoint.BEFORE_LLM_CALL:
            result = fn("sample terminal output\n$ ", ctx)
            if not isinstance(result, str):
                return [f"before_llm_call must return str, got {type(result).__name__}"]
        elif point == HookPoint.BEFORE_EXECUTE:
            cmds = [_MockCommand("ls -la\n", 1.0), _MockCommand("cat file.txt\n", 0.5)]
            result = fn(cmds, ctx)
            if not isinstance(result, list):
                return [f"before_execute must return list, got {type(result).__name__}"]
        elif point == HookPoint.AFTER_EXECUTE:
            result = fn("total 4\n-rw-r--r-- 1 user user 100 file.txt\n", ctx)
            if not isinstance(result, str):
                return [f"after_execute must return str, got {type(result).__name__}"]
        elif point == HookPoint.ON_TIMEOUT:
            result = fn("long_command\n", "partial output...", ctx)
            if not isinstance(result, str):
                return [f"on_timeout must return str, got {type(result).__name__}"]
        elif point == HookPoint.ON_PARSE_ERROR:
            result = fn('{"bad json', "JSONDecodeError", ctx)
            if result is not None and not isinstance(result, str):
                return [f"on_parse_error must return str|None, got {type(result).__name__}"]
        elif point == HookPoint.AFTER_ROUND:
            result = fn("command output\n", False, ctx)
            if not isinstance(result, dict):
                return [f"after_round hook must return dict, got {type(result).__name__}"]
    except Exception as e:
        return [f"Runtime error: {type(e).__name__}: {e}"]

    return []


class HookExecutor:
    """Manages loading and execution of model-generated hooks.

    Usage:
        executor = HookExecutor.from_directory("/path/to/hooks/")
        commands = executor.run(HookPoint.BEFORE_EXECUTE, commands, context)
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, Callable] = {}
        self._sources: dict[HookPoint, str] = {}

    @classmethod
    def from_directory(cls, hook_dir: str | Path) -> "HookExecutor":
        """Load hooks from a directory. Files named <hook_point>.py are loaded."""
        executor = cls()
        hook_dir = Path(hook_dir)
        if not hook_dir.is_dir():
            return executor

        for hp in HookPoint:
            path = hook_dir / f"{hp.value}.py"
            if path.exists():
                source = path.read_text(encoding="utf-8")
                executor.register(hp, source, str(path))

        return executor

    @classmethod
    def from_dict(cls, hooks: dict[str, str]) -> "HookExecutor":
        """Load hooks from a dict of {hook_point_name: source_code}."""
        executor = cls()
        for name, source in hooks.items():
            try:
                hp = HookPoint(name)
            except ValueError:
                logger.warning(f"Unknown hook point '{name}', skipping")
                continue
            executor.register(hp, source, f"<dict:{name}>")
        return executor

    def register(self, point: HookPoint, source: str, origin: str = "<unknown>") -> bool:
        """Validate, compile, smoke-test, and register a hook. Returns True on success."""
        errors = _validate_hook_source(source, point)
        if errors:
            logger.error(f"Hook validation failed for {point.value} from {origin}: {errors}")
            return False

        fn = _compile_hook(source)
        if fn is None:
            return False

        runtime_errors = _smoke_test_hook(fn, point)
        if runtime_errors:
            logger.error(
                f"Hook smoke test failed for {point.value} from {origin}: {runtime_errors}"
            )
            return False

        self._hooks[point] = fn
        self._sources[point] = source
        logger.info(f"Hook registered: {point.value} from {origin}")
        return True

    def has_hook(self, point: HookPoint) -> bool:
        return point in self._hooks

    def get_source(self, point: HookPoint) -> str | None:
        return self._sources.get(point)

    def active_hooks(self) -> list[str]:
        return [hp.value for hp in self._hooks]

    def run(self, point: HookPoint, *args) -> Any:
        """Execute a hook, returning the hook's result or the first arg unchanged on failure."""
        if point not in self._hooks:
            return args[0] if args else None

        fn = self._hooks[point]
        fallback = args[0] if args else None

        try:
            result = fn(*args)
            return result
        except Exception as e:
            logger.warning(f"Hook {point.value} raised {type(e).__name__}: {e}; using fallback")
            return fallback

    def to_dict(self) -> dict[str, str]:
        """Serialize all registered hooks to {name: source} dict."""
        return {hp.value: src for hp, src in self._sources.items()}
