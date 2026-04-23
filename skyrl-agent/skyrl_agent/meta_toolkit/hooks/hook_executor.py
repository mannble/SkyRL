"""HookExecutor: loads, validates, and safely runs model-generated hook code.

Each hook is a small Python function with a fixed signature. Hooks are stored
as .py files in the meta_patches directory and loaded at agent init time.

Multiple hooks per hook-point are supported. They execute as a sequential
pipeline (output of one feeds into the next) for most hook types. The
``after_round`` hook uses independent-then-merge: each hook receives the
original inputs and outputs are merged (`inject`, `next_prompt`, continuation).
When multiple hooks are active at the same point, each hook receives an
isolated ``context.kv`` namespace to avoid key conflicts.

Safety guarantees:
  - Syntax validation before loading
  - Each hook call is wrapped in try/except with a timeout
  - Hooks only receive HookContext, not raw agent internals
  - If a hook fails, the original value is returned unchanged
  - All hook executions are logged
"""

from __future__ import annotations

import ast
import builtins as _py_builtins
import copy as _copy
import logging
import re as _re
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


_ALLOWED_MODULES = frozenset({
    "re", "json", "math", "collections", "itertools",
    "functools", "copy", "textwrap", "string",
})

# Python builtins that are always available without import
_BUILTINS = frozenset(dir(_py_builtins))


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

    # Collect imported names (top-level and inside functions)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name
                imported_names.add(bound.split(".")[0])
                module = alias.name
                if module.split(".")[0] not in _ALLOWED_MODULES:
                    errors.append(f"Import of '{module}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module and module.split(".")[0] not in _ALLOWED_MODULES:
                errors.append(f"Import of '{module}' is not allowed")
            for alias in node.names:
                bound = alias.asname if alias.asname else alias.name
                imported_names.add(bound)

    # Detect references to names that are not defined/imported in hook scope.
    # Walk only the hook function body so parameter names and local assignments
    # are properly scoped.
    if hook_funcs:
        hook_fn = hook_funcs[0]
        param_names = {arg.arg for arg in hook_fn.args.args}
        local_names: set[str] = set()
        for node in ast.walk(hook_fn):
            # Covers assignment targets in for/with/comprehensions/except/as, etc.
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                local_names.add(node.id)

            # Nested function names become local symbols too.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not hook_fn:
                local_names.add(node.name)

        known_names = imported_names | param_names | local_names | _BUILTINS | {"hook"}
        undefined_name_errors: set[str] = set()

        for node in ast.walk(hook_fn):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in _ALLOWED_MODULES
                and node.value.id not in known_names
            ):
                errors.append(
                    f"Module '{node.value.id}' is used ('{node.value.id}.{node.attr}') "
                    f"but never imported"
                )

            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
                if name in known_names:
                    continue
                # Allowed-module names are validated via attribute check above.
                if name in _ALLOWED_MODULES:
                    continue
                undefined_name_errors.add(name)

        for name in sorted(undefined_name_errors):
            errors.append(f"Name '{name}' is used but never defined/imported")

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


def _parse_hook_filename(filename: str) -> tuple[str, str] | None:
    """Parse a hook filename into (hook_point_name, variant_id) or None.

    Accepted formats:
      ``before_llm_call.py``       -> ("before_llm_call", "")
      ``before_llm_call_v1.py``    -> ("before_llm_call", "v1")
      ``after_round_cycle3.py``    -> ("after_round", "cycle3")
    """
    stem = Path(filename).stem
    for hp in HookPoint:
        name = hp.value
        if stem == name:
            return (name, "")
        if stem.startswith(name + "_"):
            variant = stem[len(name) + 1:]
            return (name, variant)
    return None


class HookExecutor:
    """Manages loading and execution of model-generated hooks.

    Supports multiple hooks per hook-point. Pipeline hooks (before_llm_call,
    before_execute, after_execute, on_timeout, on_parse_error) chain
    sequentially -- the output of one feeds as input to the next.
    ``after_round`` hooks run independently on the *same* original inputs and
    their ``inject`` texts and ``force_continue`` flags are merged.

    Usage::

        executor = HookExecutor.from_directory("/path/to/hooks/")
        commands = executor.run(HookPoint.BEFORE_EXECUTE, commands, context)
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[tuple[str, Callable]]] = {}
        self._sources: dict[HookPoint, list[tuple[str, str]]] = {}

    @classmethod
    def from_directory(cls, hook_dir: str | Path) -> "HookExecutor":
        """Load hooks from a directory.

        Accepts ``<hook_point>.py`` (legacy) and ``<hook_point>_<variant>.py``
        (multi-hook). Files are sorted by name so execution order is stable.
        """
        executor = cls()
        hook_dir = Path(hook_dir)
        if not hook_dir.is_dir():
            return executor

        for path in sorted(hook_dir.glob("*.py")):
            parsed = _parse_hook_filename(path.name)
            if parsed is None:
                continue
            hp_name, variant = parsed
            try:
                hp = HookPoint(hp_name)
            except ValueError:
                continue
            source = path.read_text(encoding="utf-8")
            label = variant or hp_name
            executor.register(hp, source, str(path), name=label)

        return executor

    @classmethod
    def from_dict(cls, hooks: dict[str, str]) -> "HookExecutor":
        """Load hooks from ``{hook_point_name: source}`` or ``{hook_point_name_variant: source}``."""
        executor = cls()
        for key, source in hooks.items():
            # Try exact match first
            try:
                hp = HookPoint(key)
                executor.register(hp, source, f"<dict:{key}>", name=key)
                continue
            except ValueError:
                pass
            # Try parsing as hook_point + variant
            parsed = _parse_hook_filename(key + ".py")
            if parsed is not None:
                hp_name, variant = parsed
                try:
                    hp = HookPoint(hp_name)
                except ValueError:
                    logger.warning(f"Unknown hook point in key '{key}', skipping")
                    continue
                label = variant or hp_name
                executor.register(hp, source, f"<dict:{key}>", name=label)
            else:
                logger.warning(f"Unknown hook point '{key}', skipping")
        return executor

    def register(
        self,
        point: HookPoint,
        source: str,
        origin: str = "<unknown>",
        *,
        name: str = "",
    ) -> bool:
        """Validate, compile, smoke-test, and register a hook. Returns True on success."""
        errors = _validate_hook_source(source, point)
        if errors:
            logger.error(f"Hook validation failed for {point.value} ({name}) from {origin}: {errors}")
            return False

        fn = _compile_hook(source)
        if fn is None:
            return False

        runtime_errors = _smoke_test_hook(fn, point)
        if runtime_errors:
            logger.error(
                f"Hook smoke test failed for {point.value} ({name}) from {origin}: {runtime_errors}"
            )
            return False

        hook_name = name or point.value
        self._hooks.setdefault(point, []).append((hook_name, fn))
        self._sources.setdefault(point, []).append((hook_name, source))
        logger.info(f"Hook registered: {point.value}/{hook_name} from {origin}")
        return True

    def deregister(self, point: HookPoint, name: str) -> bool:
        """Remove a specific named hook. Returns True if found and removed."""
        hooks = self._hooks.get(point, [])
        sources = self._sources.get(point, [])
        before = len(hooks)
        self._hooks[point] = [(n, fn) for n, fn in hooks if n != name]
        self._sources[point] = [(n, src) for n, src in sources if n != name]
        removed = before - len(self._hooks[point])
        if removed:
            logger.info(f"Hook deregistered: {point.value}/{name}")
        return removed > 0

    def has_hook(self, point: HookPoint) -> bool:
        return bool(self._hooks.get(point))

    def get_source(self, point: HookPoint) -> str | None:
        """Return the source of the first registered hook for backward compat."""
        entries = self._sources.get(point, [])
        return entries[0][1] if entries else None

    def get_all_sources(self, point: HookPoint) -> list[tuple[str, str]]:
        """Return [(name, source), ...] for all hooks at this point."""
        return list(self._sources.get(point, []))

    def active_hooks(self) -> list[str]:
        result: list[str] = []
        for hp, entries in self._hooks.items():
            for name, _ in entries:
                result.append(f"{hp.value}/{name}")
        return result

    def hook_count(self, point: HookPoint) -> int:
        return len(self._hooks.get(point, []))

    def run(self, point: HookPoint, *args) -> Any:
        """Execute hooks for a point.

        Pipeline hooks: sequential chaining (output of one is input to next).
        after_round: independent-then-merge (all receive same original args).
        """
        entries = self._hooks.get(point, [])
        if not entries:
            return args[0] if args else None

        isolate_context = len(entries) > 1
        if point == HookPoint.AFTER_ROUND:
            return self._run_after_round_merged(
                entries,
                *args,
                isolate_context=isolate_context,
            )

        return self._run_pipeline(
            point,
            entries,
            *args,
            isolate_context=isolate_context,
        )

    @staticmethod
    def _extract_hook_group(hook_name: str) -> str:
        """Extract the generation group from a hook filename.

        Hooks produced by the same planner sub-agent share a group key like
        ``candidate_0_hookgrp_post_action_controls``.  Hooks within the same
        group get a **shared** ``context.kv`` so that e.g. an after_execute
        hook can write data that a sibling after_round hook can read.

        Different groups (different candidates or different categories) are
        isolated from each other.

        Supports both old format (``..._candidate_N_hookgrp_cat_HASH``) and
        new format (``..._sN_candidate_N_hookgrp_cat_HASH``).

        Falls back to the full hook_name if no group pattern is found.
        """
        import re as _re
        m = _re.search(r'(candidate_\d+_hookgrp_\w+?)_[a-f0-9]{6,}$', hook_name)
        if m:
            return m.group(1)
        return hook_name

    def _build_hook_call_args(
        self,
        point: HookPoint,
        hook_name: str,
        current_args: list[Any],
        *,
        isolate_context: bool,
    ) -> tuple[Any, ...]:
        """Build hook-call args; isolate context.kv by generation group.

        Hooks from the same planner sub-agent (same candidate + category)
        share a single kv namespace so they can communicate across hook
        points.  Hooks from different groups are fully isolated.
        """
        if not isolate_context or not current_args:
            return tuple(current_args)

        ctx = current_args[-1]
        kv = getattr(ctx, "kv", None)
        if not isinstance(kv, dict):
            return tuple(current_args)

        try:
            ctx_view = _copy.copy(ctx)
        except Exception:
            return tuple(current_args)

        scopes = kv.setdefault("_multi_hook_scopes", {})
        if not isinstance(scopes, dict):
            scopes = {}
            kv["_multi_hook_scopes"] = scopes

        scope_key = self._extract_hook_group(hook_name)
        hook_kv = scopes.setdefault(scope_key, {})
        if not isinstance(hook_kv, dict):
            hook_kv = {}
            scopes[scope_key] = hook_kv

        ctx_view.kv = hook_kv
        call_args = list(current_args)
        call_args[-1] = ctx_view
        return tuple(call_args)

    def _run_pipeline(
        self,
        point: HookPoint,
        entries: list[tuple[str, Callable]],
        *args,
        isolate_context: bool = False,
    ) -> Any:
        """Sequential pipeline: first positional arg is replaced by each hook's return value."""
        current_args = list(args)
        fallback = args[0] if args else None

        for hook_name, fn in entries:
            try:
                hook_args = self._build_hook_call_args(
                    point, hook_name, current_args, isolate_context=isolate_context,
                )
                result = fn(*hook_args)
                current_args[0] = result
            except Exception as e:
                logger.warning(
                    f"Hook {point.value}/{hook_name} raised {type(e).__name__}: {e}; "
                    f"skipping this hook in the pipeline"
                )

        return current_args[0] if current_args else fallback

    def _run_after_round_merged(
        self,
        entries: list[tuple[str, Callable]],
        *args,
        isolate_context: bool = False,
    ) -> dict:
        """Independent execution for after_round: each hook gets original args.

        Merge strategy:
          - Concatenate all ``inject`` texts with newline separator.
          - ``request_new_turn``/``force_continue`` is True if any hook requests it.
          - ``next_prompt`` supports append/replace policies:
              * append: concatenate all append prompts
              * replace: last replace prompt wins
        """
        merged_inject_parts: list[str] = []
        append_prompt_parts: list[str] = []
        replace_prompt: str | None = None
        request_new_turn = False

        for hook_name, fn in entries:
            try:
                hook_args = self._build_hook_call_args(
                    HookPoint.AFTER_ROUND,
                    hook_name,
                    list(args),
                    isolate_context=isolate_context,
                )
                result = fn(*hook_args)
                if not isinstance(result, dict):
                    continue
                inject = result.get("inject")
                if inject and isinstance(inject, str):
                    merged_inject_parts.append(inject)

                next_prompt = result.get("next_prompt")
                if next_prompt and isinstance(next_prompt, str):
                    mode = str(result.get("prompt_mode", "append")).lower()
                    if mode == "replace":
                        replace_prompt = next_prompt
                    else:
                        append_prompt_parts.append(next_prompt)

                if result.get("request_new_turn") or result.get("force_continue"):
                    request_new_turn = True
            except Exception as e:
                logger.warning(
                    f"Hook after_round/{hook_name} raised {type(e).__name__}: {e}; skipping"
                )

        merged: dict[str, Any] = {}
        if merged_inject_parts:
            merged["inject"] = "\n\n".join(merged_inject_parts)

        if replace_prompt is not None:
            merged["next_prompt"] = replace_prompt
            merged["prompt_mode"] = "replace"
        elif append_prompt_parts:
            merged["next_prompt"] = "\n\n".join(append_prompt_parts)
            merged["prompt_mode"] = "append"

        if request_new_turn:
            merged["request_new_turn"] = True
            # Backward-compatible alias.
            merged["force_continue"] = True
        return merged

    def to_dict(self) -> dict[str, str]:
        """Serialize all registered hooks to ``{name: source}`` dict.

        For multi-hook points the keys use ``hook_point_variant`` format.
        """
        result: dict[str, str] = {}
        for hp, entries in self._sources.items():
            if len(entries) == 1:
                result[hp.value] = entries[0][1]
            else:
                for name, src in entries:
                    key = name if name != hp.value else hp.value
                    if key in result:
                        key = f"{hp.value}_{name}"
                    result[key] = src
        return result
