"""HookExecutor: loads, validates, and safely runs model-generated hook code.

Each hook is a small Python function with a fixed signature. Hooks are stored
as .py files in the meta_patches directory and loaded at agent init time.

Multiple hooks per hook-point are supported. ``before_llm_call`` hooks return
append-only prompt guidance that is merged. ``after_execute`` and
``on_timeout`` are observer hooks whose return values are ignored. The
``after_round`` hook uses independent-then-merge: each hook receives the
original inputs and outputs are merged (`next_prompt` guidance). Legacy prompt
fields are still accepted internally for backward compatibility.
Each generated hook group receives an isolated ``context.kv`` namespace, and
hooks from the same group share that namespace across hook points.

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
    HookPoint.AFTER_EXECUTE: "def hook(terminal_output: str, context: HookContext) -> None",
    HookPoint.ON_TIMEOUT: "def hook(command_keystrokes: str, terminal_output: str, context: HookContext) -> None",
    HookPoint.ON_PARSE_ERROR: "def hook(raw_response: str, error: str, context: HookContext) -> str | None",
    HookPoint.BEFORE_LLM_CALL: "def hook(prompt: str, context: HookContext) -> dict | None",
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

        errors.extend(_validate_context_kv_access(hook_fn))

    return errors


def _is_context_kv_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "kv"
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
    )


def _kv_key_from_subscript(node: ast.Subscript) -> str | None:
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def _is_kv_subscript(node: ast.AST, kv_aliases: set[str]) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    return _is_context_kv_attr(value) or (
        isinstance(value, ast.Name) and value.id in kv_aliases
    )


def _validate_context_kv_access(hook_fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Catch likely KeyError-prone context.kv reads before canary/runtime.

    This is intentionally conservative and key-based rather than path-sensitive:
    reads are allowed when the hook contains a `setdefault`, `get`, or direct
    assignment for the same literal key. Dynamic-key reads are rejected because
    the smoke tests cannot reliably cover them.
    """
    errors: list[str] = []
    kv_aliases: set[str] = set()
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(hook_fn):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    for node in ast.walk(hook_fn):
        if isinstance(node, ast.Assign) and _is_context_kv_attr(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    kv_aliases.add(target.id)

    initialized_keys: set[str] = set()
    for node in ast.walk(hook_fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            receiver_is_kv = _is_context_kv_attr(receiver) or (
                isinstance(receiver, ast.Name) and receiver.id in kv_aliases
            )
            if receiver_is_kv and node.func.attr in {"setdefault", "get", "pop"}:
                if node.args and isinstance(node.args[0], ast.Constant):
                    key = node.args[0].value
                    if isinstance(key, str):
                        initialized_keys.add(key)

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_kv_subscript(target, kv_aliases):
                    key = _kv_key_from_subscript(target)
                    if key is not None:
                        initialized_keys.add(key)

    reported: set[str] = set()
    for node in ast.walk(hook_fn):
        if not _is_kv_subscript(node, kv_aliases):
            continue
        parent_node = parent.get(node)

        is_plain_assignment_target = (
            isinstance(parent_node, ast.Assign) and node in parent_node.targets
        ) or (
            isinstance(parent_node, ast.AnnAssign) and node is parent_node.target
        )
        if is_plain_assignment_target:
            continue

        key = _kv_key_from_subscript(node)
        if key is None:
            errors.append(
                "context.kv dynamic-key reads are not allowed; use get/setdefault "
                "with a literal key"
            )
            continue
        if key not in initialized_keys and key not in reported:
            reported.add(key)
            errors.append(
                f"context.kv['{key}'] is read without get/setdefault/default assignment; "
                "use context.kv.get(...) or context.kv.setdefault(...)"
            )

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
    """Run a hook across representative mock paths. Returns errors (empty = OK)."""
    from skyrl_agent.meta_toolkit.hooks.hook_context import HookContext

    def make_ctx(
        *,
        episode: int = 1,
        last_commands: list[str] | None = None,
        terminal_output: str = "",
        is_task_complete: bool = False,
    ) -> HookContext:
        next_observation = terminal_output
        if is_task_complete:
            next_observation = (
                f"Current terminal state:\n{terminal_output}\n\n"
                "Are you sure you want to mark the task as complete?"
            )
        return HookContext(
            episode=episode,
            total_episodes=10,
            original_instruction="create the required file and verify it",
            last_commands=last_commands or [],
            last_terminal_output=terminal_output,
            next_observation=next_observation,
            is_task_complete=is_task_complete,
        )

    def check_after_round_result(result: Any, label: str) -> list[str]:
        if not isinstance(result, dict):
            return [f"after_round hook must return dict for {label}, got {type(result).__name__}"]
        if "next_prompt" in result and result["next_prompt"] is not None and not isinstance(result["next_prompt"], str):
            return [f"after_round next_prompt must be str|None for {label}"]
        return []

    try:
        if point == HookPoint.BEFORE_LLM_CALL:
            for label, prompt, ctx in [
                ("initial", "Initial task prompt", make_ctx(episode=0)),
                ("middle", "Previous observation", make_ctx(episode=1, last_commands=["ls -la\n"])),
                ("periodic", "Previous observation", make_ctx(episode=6, last_commands=["cat file.txt\n"])),
            ]:
                result = fn(prompt, ctx)
                if result is None:
                    continue
                if isinstance(result, dict):
                    append_prompt = result.get("append_prompt")
                    if append_prompt is not None and not isinstance(append_prompt, str):
                        return [f"before_llm_call append_prompt must be str|None for {label}"]
                    continue
                if isinstance(result, str):
                    # Legacy append-only hooks returned the full prompt string.
                    if not result.startswith(prompt):
                        return [
                            f"before_llm_call legacy str must keep original prompt "
                            f"as prefix for {label}"
                        ]
                    continue
                return [
                    f"before_llm_call must return dict|None "
                    f"(legacy str accepted) for {label}, got {type(result).__name__}"
                ]
        elif point == HookPoint.BEFORE_EXECUTE:
            for label, cmds in [
                ("simple", [_MockCommand("ls -la\n", 1.0), _MockCommand("cat file.txt\n", 0.5)]),
                ("empty", []),
                ("complex", [_MockCommand("cat <<'EOF' > file.txt\nhello\nEOF\n", 1.0)]),
            ]:
                result = fn(cmds, make_ctx(last_commands=[c.keystrokes for c in cmds]))
                if not isinstance(result, list):
                    return [f"before_execute must return list for {label}, got {type(result).__name__}"]
        elif point == HookPoint.AFTER_EXECUTE:
            cases = [
                ("empty", "", []),
                ("cat", "cat file.txt\nhello\n", ["cat file.txt\n"]),
                ("ls", "total 4\n-rw-r--r-- 1 user user 100 file.txt\n", ["ls -la\n"]),
                ("missing_path", "cat: missing.txt: No such file or directory\n", ["cat missing.txt\n"]),
                ("permission", "Permission denied\n", ["touch /root/file\n"]),
                ("large", "line\n" * 2500, ["cat big.log\n"]),
            ]
            shared_ctx = make_ctx()
            for idx, (label, sample_output, commands) in enumerate(cases):
                shared_ctx.episode = idx
                shared_ctx.last_commands = commands
                shared_ctx.last_terminal_output = sample_output
                result = fn(sample_output, shared_ctx)
                if result is not None and result != sample_output:
                    return [
                        f"after_execute observer must return None for {label} "
                        "(legacy unchanged str accepted)"
                    ]
        elif point == HookPoint.ON_TIMEOUT:
            cases = [
                ("simple", "sleep 999\n", "partial output..."),
                ("broad_search", "find / -name target\n", "Command timed out after 60 seconds"),
                ("install", "pip install package\n", "Collecting package...\n"),
                ("cat_loop", "cat large.log\n", "many lines\n"),
            ]
            shared_ctx = make_ctx()
            for idx, (label, command, sample_output) in enumerate(cases):
                shared_ctx.episode = idx
                shared_ctx.last_commands = [command]
                shared_ctx.last_terminal_output = sample_output
                result = fn(command, sample_output, shared_ctx)
                if result is not None and result != sample_output:
                    return [
                        f"on_timeout observer must return None for {label} "
                        "(legacy unchanged str accepted)"
                    ]
        elif point == HookPoint.ON_PARSE_ERROR:
            for label, raw_response, error in [
                ("json", '{"bad json', "JSONDecodeError"),
                ("empty", "", "EmptyResponse"),
            ]:
                result = fn(raw_response, error, make_ctx())
                if result is not None and not isinstance(result, str):
                    return [f"on_parse_error must return str|None for {label}, got {type(result).__name__}"]
        elif point == HookPoint.AFTER_ROUND:
            cases = [
                ("empty", "", False, []),
                ("empty_complete", "", True, []),
                ("cat", "cat file.txt\nhello\n", False, ["cat file.txt\n"]),
                ("cat_complete", "cat file.txt\nhello\n", True, ["cat file.txt\n"]),
                ("ls_complete", "total 4\n-rw-r--r-- 1 user user 100 file.txt\n", True, ["ls -la\n"]),
                ("missing_path", "cat: missing.txt: No such file or directory\n", False, ["cat missing.txt\n"]),
                ("timeout", "Command timed out after 60 seconds\nfind / -name target\n", False, ["find / -name target\n"]),
            ]
            for idx, (label, output, complete, commands) in enumerate(cases):
                ctx = make_ctx(
                    episode=idx,
                    last_commands=commands,
                    terminal_output=output,
                    is_task_complete=complete,
                )
                result = fn(output, complete, ctx)
                errors = check_after_round_result(result, label)
                if errors:
                    return errors

            shared_ctx = make_ctx()
            sequence = [
                ("cat_a", "cat a.txt\ncontent\n", False, ["cat a.txt\n"]),
                ("cat_b", "cat b.txt\ncontent\n", False, ["cat b.txt\n"]),
                ("ls", "total 4\n-rw-r--r-- 1 user user 10 a.txt\n", False, ["ls -la\n"]),
                ("complete_1", "", True, []),
                ("complete_2", "", True, []),
            ]
            for idx, (label, output, complete, commands) in enumerate(sequence):
                shared_ctx.episode = idx
                shared_ctx.last_commands = commands
                shared_ctx.last_terminal_output = output
                shared_ctx.next_observation = (
                    f"Current terminal state:\n{output}\n\n"
                    "Are you sure you want to mark the task as complete?"
                    if complete
                    else output
                )
                shared_ctx.is_task_complete = complete
                result = fn(output, complete, shared_ctx)
                errors = check_after_round_result(result, f"multi_round:{label}")
                if errors:
                    return errors
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
    before_execute, on_parse_error) chain sequentially -- the output of one
    feeds as input to the next. after_execute and on_timeout are observers:
    each receives the original terminal output and return values are ignored.
    ``after_round`` hooks run independently on the *same* original inputs and
    their prompt guidance is merged.

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

        # Always scope hook state by generated hook group. This lets sibling
        # hooks from one group communicate across hook points while preventing
        # unrelated groups from sharing scratch state.
        isolate_context = True
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
        ``candidate_0_hookgrp_runtime_recovery``.  Hooks within the same
        group get a **shared** ``context.kv`` so that e.g. an after_execute
        hook can write data that a sibling after_round hook can read.

        Different groups (different candidates or different categories) are
        isolated from each other.

        Supports both old format (``..._candidate_N_hookgrp_cat_HASH``) and
        new format (``..._sN_candidate_N_hookgrp_cat_HASH``).

        Falls back to the full hook_name if no group pattern is found.
        """
        import re as _re
        m = _re.search(
            r'((?:s\d+_)?candidate_\d+_hookgrp_\w+?)_[a-f0-9]{6,}$',
            hook_name,
        )
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
        """Run pipeline hooks, or observer hooks with ignored return values."""
        current_args = list(args)
        fallback = args[0] if args else None

        if point == HookPoint.BEFORE_LLM_CALL:
            base_prompt = args[0] if args and isinstance(args[0], str) else ""
            append_prompt_parts: list[str] = []
            for hook_name, fn in entries:
                try:
                    hook_args = self._build_hook_call_args(
                        point,
                        hook_name,
                        list(args),
                        isolate_context=isolate_context,
                    )
                    result = fn(*hook_args)
                    if isinstance(result, dict):
                        append_prompt = result.get("append_prompt")
                        if isinstance(append_prompt, str) and append_prompt:
                            append_prompt_parts.append(append_prompt)
                    elif isinstance(result, str) and result.startswith(base_prompt):
                        suffix = result[len(base_prompt):]
                        if suffix:
                            append_prompt_parts.append(suffix)
                    elif result is not None:
                        logger.warning(
                            f"Hook {point.value}/{hook_name} returned unsupported "
                            f"{type(result).__name__}; skipping this hook"
                        )
                except Exception as e:
                    logger.warning(
                        f"Hook {point.value}/{hook_name} raised {type(e).__name__}: {e}; "
                        f"skipping this hook"
                    )
            if append_prompt_parts:
                return {"append_prompt": "\n\n".join(append_prompt_parts)}
            return {}

        if point in (HookPoint.AFTER_EXECUTE, HookPoint.ON_TIMEOUT):
            for hook_name, fn in entries:
                try:
                    hook_args = self._build_hook_call_args(
                        point,
                        hook_name,
                        list(args),
                        isolate_context=isolate_context,
                    )
                    fn(*hook_args)
                except Exception as e:
                    logger.warning(
                        f"Hook {point.value}/{hook_name} raised {type(e).__name__}: {e}; "
                        f"skipping this observer hook"
                    )
            return fallback

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
          - Concatenate all ``next_prompt`` texts with newline separator.
          - Legacy ``inject`` and ``prompt_mode=replace`` are accepted for old
            hooks; the Terminus runtime applies guidance in append mode.
        """
        merged_inject_parts: list[str] = []
        append_prompt_parts: list[str] = []
        replace_prompt: str | None = None

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
