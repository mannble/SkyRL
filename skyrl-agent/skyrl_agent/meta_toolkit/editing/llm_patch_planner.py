"""LLM-powered patch planner: generates concrete override values for agent modules.

Replaces the keyword-based PatchExecutor._parse_intent with an LLM that
reasons about traces + diagnosis and produces specific YAML override values.
Falls back to the rule-based planner on LLM failure.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from skyrl_agent.meta_toolkit.diagnosis.rule_based_diagnoser import DiagnosisResult
from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry
from skyrl_agent.meta_toolkit.editing.patch_schema import (
    PatchCandidate,
    PatchFileEdit,
)
from skyrl_agent.meta_toolkit.llm_client import MetaLLMClient
from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord

logger = logging.getLogger(__name__)

_OVERRIDE_FIELDS_DOC = """\
## code_hooks reference

Each hook is a Python function named `hook`. Hooks are sandboxed — NO access
to agent classes or external code. If a hook raises an exception, the original
value is used unchanged.

**Allowed imports:** re, json, math, collections, itertools, functools, copy,
textwrap, string.

### context (HookContext) — a dataclass, NOT a dict

Access all fields as **attributes** (e.g. `context.episode`, NOT `context["episode"]`).

| Field | Type | Description |
|-------|------|-------------|
| context.episode | int | Current round number (0-based) |
| context.total_episodes | int | Max rounds allowed |
| context.n_commands_executed | int | Total commands run so far |
| context.n_parse_errors | int | Total parse failures so far |
| context.n_timeouts | int | Total command timeouts so far |
| context.last_commands | list[str] | Command strings executed THIS round |
| context.last_analysis | str | LLM's analysis text this round |
| context.last_plan | str | LLM's plan text this round |
| context.is_task_complete | bool | LLM marked task_complete this round |
| context.original_instruction | str | The original task description |
| context.last_terminal_output | str | Terminal output from the most recent command |
| context.execution_history | list[dict] | Past rounds: [{"commands":[..], "output":"...", "episode":N}] (last 20) |
| context.last_llm_response | str | Raw LLM response text before parsing |
| context.forced_continue_count | int | Times force_continue was triggered so far |
| context.kv | dict | **Writable** persistent store across rounds |

### context.kv usage (IMPORTANT)

`context.kv` is the ONLY writable field. Use it to track state across rounds.
When multiple hooks are active on the same hook point, each hook gets an
isolated `context.kv` namespace to avoid key collisions.
Always initialize with `setdefault` to avoid KeyError:

```python
# CORRECT usage:
context.kv.setdefault("counter", 0)
context.kv["counter"] += 1

# WRONG — will raise KeyError on first call:
context.kv["counter"] += 1

# WRONG — context is NOT a dict:
context.get("kv")
context["episode"]
```

### Hook signatures and examples

**before_llm_call**(prompt: str, context) -> str
```python
def hook(prompt, context):
    context.kv.setdefault("call_count", 0)
    context.kv["call_count"] += 1
    if context.kv["call_count"] > 5:
        prompt += "\\n[REMINDER] Stay focused on the original task."
    return prompt
```

**before_execute**(commands: list, context) -> list
```python
def hook(commands, context):
    for cmd in commands:
        if cmd.duration_sec > 30:
            cmd.duration_sec = 10
    return [c for c in commands if c.keystrokes.strip()]
```

**after_execute**(terminal_output: str, context) -> str
```python
def hook(terminal_output, context):
    if len(terminal_output) > 5000:
        terminal_output = terminal_output[:2000] + "\\n...truncated...\\n" + terminal_output[-2000:]
    return terminal_output
```

**on_timeout**(command_keystrokes: str, terminal_output: str, context) -> str
```python
def hook(command_keystrokes, terminal_output, context):
    return terminal_output + "\\n[TIMEOUT] Command timed out. Try a different approach."
```

**after_round**(terminal_output: str, is_task_complete: bool, context) -> dict
The most powerful hook. Runs after every round and may control the next LLM turn.
Return fields (all optional):
- `request_new_turn` (bool): ask for an extra turn, even when completion would end.
- `next_prompt` (str): prompt content for the next turn.
- `prompt_mode` ("append" | "replace"): how `next_prompt` is applied.
- `inject` (str): backward-compatible alias for appending guidance text.
- `force_continue` (bool): legacy alias for `request_new_turn`.

NOTE on completion flow: when the agent outputs task_complete=true, terminus-2
uses a two-step confirmation (agent must say task_complete twice in a row).
The after_round hook runs BEFORE the confirmation check, so it can inject text
into the confirmation prompt. If your hook injects a verification reminder on
the first task_complete, the agent will see it and can decide whether to
confirm or continue working.

Example — completion verification with explicit next-turn control:
```python
def hook(terminal_output, is_task_complete, context):
    count = context.kv.setdefault("complete_count", 0)
    if is_task_complete:
        context.kv["complete_count"] = count + 1
        if count < 1:
            return {{
                "request_new_turn": True,
                "next_prompt": (
                "[VERIFY] You said task_complete. Before confirming, please:\\n"
                "1. Run the tests.\\n"
                "2. Check the output matches the expected format.\\n"
                "If all correct, submit task_complete again."
                ),
                "prompt_mode": "append",
            }}
    else:
        context.kv["complete_count"] = 0
    return {{}}
```

Example — loop detection (warn when commands repeat):
```python
def hook(terminal_output, is_task_complete, context):
    history = context.kv.setdefault("cmd_history", [])
    for cmd in context.last_commands:
        history.append(cmd.strip())
    if len(history) > 20:
        history[:] = history[-20:]
    if len(history) >= 4 and history[-2:] == history[-4:-2]:
        return {{
            "next_prompt": (
            "[LOOP DETECTED] You are repeating commands. "
            "Try a different approach. Task: "
            + context.original_instruction[:200]
            ),
            "prompt_mode": "append",
        }}
    return {{}}
```
"""

# Agent-specific code context — shown to the planner so it understands the
# actual code it is modifying via hooks.

_TERMINUS2_CODE_CONTEXT = """\
## Agent source code context

This is the actual agent loop code. Study it carefully to understand the
agent's behavior, then write hooks that address the diagnosed problems.

```python
@dataclass
class Command:
    keystrokes: str      # e.g. "ls -la\\n"
    duration_sec: float  # max wait seconds (capped to 60)

async def _run_agent_loop(self, initial_prompt, chat, logging_dir, original_instruction):
    prompt = initial_prompt
    for episode in range(self._max_episodes):
        # 1. before_llm_call hook
        prompt = self._run_hook("before_llm_call", prompt, context)
        # 2. LLM interaction
        commands, is_task_complete, feedback, analysis, plan, llm_response = \\
            await self._handle_llm_interaction(chat, prompt, ...)
        # 3. before_execute hook
        commands = self._run_hook("before_execute", commands, context)
        # 4. Execute commands in tmux
        timeout_occurred, terminal_output = await self._execute_commands(commands, session)
        if timeout_occurred:
            self._run_hook("on_timeout", last_cmd, terminal_output, context)
        else:
            terminal_output = self._run_hook("after_execute", terminal_output, context)
        # 5. after_round hook — can request extra turns and customize next prompt
        control = await self._run_after_round_hook(terminal_output, is_task_complete)
        if control.get("inject"): terminal_output += "\\n\\n" + control["inject"]
        # 6. Completion check (two-step confirmation)
        if is_task_complete:
            if self._pending_completion: return
            else: self._pending_completion = True; prompt = confirmation_msg; continue
        prompt = terminal_output
```

Key facts:
- Commands execute serially in tmux; context.kv persists within a trial.
- context.last_commands contains the keystrokes from the PREVIOUS step.
- The after_round hook is the most powerful: it can request extra turns and
  customize the next prompt (append or replace mode).

**CRITICAL sandbox rules for hooks:**
- Hooks run in an isolated namespace. You CANNOT use `Command(...)` or any
  agent class. For `before_execute`, modify command objects in-place
  (e.g. `cmd.keystrokes = ...`, `cmd.duration_sec = ...`) or filter the
  list. Do NOT try to construct new objects.
- Only allowed imports: re, json, math, collections, itertools, functools,
  copy, textwrap, string.
- All hooks are tested with mock data before deployment. If a hook raises
  any exception during testing, it will be rejected.
"""

_IMPROVEMENT_STRATEGIES = """\
## Improvement strategies

When the diagnosis points to certain failure patterns, consider these
high-value strategies. These are NOT mandatory — only apply them when the
trace evidence supports the need.

### 1. Premature completion → Completion verification via after_round
If traces show the agent marking `task_complete` too early:
- Use an `after_round` hook: when `is_task_complete` is True, track a
  confirm counter in `context.kv`. If first time, inject a verification
  prompt. Only allow completion after consecutive task_complete outputs.

### 2. Output overflow → Smarter truncation
If traces show `context_overload` or very long terminal output:
- Use an `after_execute` hook to intelligently summarize or filter terminal
  output (e.g. keep only the last N lines of compiler errors, strip ANSI
  escape codes, collapse repeated blank lines).

### 3. Wasteful waiting → Duration optimisation
If traces show high wall-clock time relative to turn count:
- Use a `before_execute` hook to cap `duration_sec` for known-fast commands
  (e.g. `cd`, `ls`, `echo`, `cat` → 0.3s) and set reasonable defaults for
  build commands (e.g. `make`, `gcc` → 5-10s).

### 4. Stuck in loops → Loop detection via after_round
If traces show the agent repeating the same commands without progress:
- Use an `after_round` hook that tracks command history in `context.kv`.
  When repeated patterns are detected, inject a warning suggesting a new
  approach. Alternatively use a `before_llm_call` hook for earlier detection.

### 5. Poor planning → Strategic guidance
If traces show planning_failure or the agent taking many turns:
- Add strategy_edits with task-decomposition advice (e.g. "Read the task
  fully before acting", "Verify each sub-goal before moving on").

### 6. Timeout handling → Graceful recovery
If traces show many timeouts:
- Use an `on_timeout` hook to append recovery guidance to terminal output.

### 7. Verification gaps → Post-step verification via after_round
If traces show tasks passing but with low reward (partial solutions):
- Use an `after_round` hook to periodically inject reminders for the agent
  to verify its work (e.g. "Run the tests before proceeding").
"""

_SYSTEM_PROMPT_BASE = """\
You are an expert AI-agent architect. Given a diagnosis and sample traces,
generate targeted fixes to improve agent behaviour.

Two modification tiers:

### Tier 1: Strategy library edits (prompt-level)
Add, edit, or remove reusable strategies in the agent's strategy library.
Each strategy has a `pattern` (when to apply) and `steps` (what to do).
The agent sees these as [LEARNED STRATEGIES] in its prompt.

### Tier 2: Code hooks (code-level)
Python functions injected into the agent loop at specific hook points.
The `after_round` hook is the most powerful: it can control whether to run
another LLM turn and what prompt to use next.

{override_fields_doc}

{agent_code_context}

{improvement_strategies}

## Rules

1. Include all high-quality fixes that are strongly supported by diagnosis evidence.
   Do not drop good fixes only because there are multiple valid edits.
2. Only generate hooks for hook points listed in the diagnosis `candidate_modules`.
   `strategy_library` is always allowed — you can ALWAYS add strategy_edits.
3. When `strategy_suggestions` are provided by the diagnoser, you MUST add
   them to the strategy library via `strategy_edits` with action `add`.
4. Be conservative — change the minimum necessary.
5. Ground choices in trace evidence (mention task IDs, failure modes).
6. For `code_hooks`: write a complete function named `hook` with the correct
   signature. Only use allowed imports. Write defensive code with fallbacks.
   Always initialize `context.kv` keys with setdefault before using them.
7. For `after_round` hooks: return a dict with optional keys:
   `request_new_turn`, `next_prompt`, `prompt_mode` ("append"/"replace"),
   and optional `inject` (legacy append helper). Return {{}} for no-op.
   Use `context.kv` to track state across rounds (e.g. counters/history).
8. Match the strategy to the diagnosis. Do NOT blindly apply strategies.
9. For `strategy_edits`: prefer `add` for new patterns, `edit` to refine
   existing strategies (reference by index), `remove` to delete obsolete ones.
10. Never emit `${VAR}`-style placeholders in strategy text; they can break
   runtime config interpolation. Use concrete values or plain placeholder words.

## Output format

Return a JSON object:

```json
{{
  "strategy_edits": [
    {{"action": "add", "strategy": {{"pattern": "...", "steps": ["..."]}}}},
    {{"action": "edit", "index": 0, "strategy": {{"pattern": "...", "steps": ["..."]}}}},
    {{"action": "remove", "index": 2}}
  ],
  "code_hooks": {{
    "<hook_point>": "<Python source>"
  }},
  "rationale": "<brief explanation>"
}}
```

- `strategy_edits`: ordered list of add/edit/remove operations on the
  strategy library. Each `add` needs a `strategy` with `pattern` and `steps`.
  Each `edit` needs an `index` (0-based) and new `strategy`.
  Each `remove` needs an `index`.
- `code_hooks`: Python hook functions keyed by hook point name. Valid keys:
  before_llm_call, before_execute, after_execute, on_timeout, after_round.
"""

_MODULE_OVERRIDE_FILE: dict[str, str] = {
    "strategy_library": "strategy_library.yaml",
}

_HOOK_CATEGORIES: dict[str, tuple[str, ...]] = {
    # Pre-action controls: prompt shaping and command/time control.
    "pre_action_controls": ("before_llm_call", "before_execute", "on_timeout"),
    # Post-action controls: output shaping and round-level continuation control.
    "post_action_controls": ("after_execute", "after_round"),
}
_ALL_HOOK_POINTS: tuple[str, ...] = tuple(
    hp for group in _HOOK_CATEGORIES.values() for hp in group
)
_MAX_DIAGNOSES_FOR_STRATEGY_PROMPT = 10
_MAX_DIAGNOSES_FOR_HOOK_PROMPT = 6
_MAX_TRACES_FOR_PROMPT = 8
_STRATEGY_SUBAGENT_MAX_TOKENS = 3072
_HOOK_SUBAGENT_MAX_TOKENS = 2048
_JSON_RETRY_HINT_STRATEGY = (
    "Your previous response could not be parsed as strict JSON.\n"
    "Retry now and return ONLY one compact JSON object with keys:\n"
    "  strategy_edits, rationale\n"
    "Requirements:\n"
    "- No markdown fences, no explanations, no <think> tags.\n"
    "- Keep strategy_edits concise and high-impact.\n"
    "- If many possible edits exist, prioritize the most impactful ones first."
)
_JSON_RETRY_HINT_HOOK = (
    "Your previous response could not be parsed as strict JSON.\n"
    "Retry now and return ONLY one compact JSON object with keys:\n"
    "  code_hooks, rationale\n"
    "Requirements:\n"
    "- No markdown fences, no explanations, no <think> tags.\n"
    "- Keep hook code minimal and robust.\n"
    "- If uncertain, return empty code_hooks ({})."
)


def _serialise_traces_compact(traces: list[TraceRecord], max_traces: int = 15) -> str:
    """Compact trace summary for the planning prompt (shorter than diagnoser)."""
    rows: list[str] = []
    for t in traces[:max_traces]:
        row = {
            "task": t.task_id,
            "ok": t.success,
            "reward": t.final_reward,
            "turns": t.turn_count,
            "reason": t.finish_reason,
            "tools": t.tool_calls,
            "tool_err": t.tool_failures,
            "tags": t.failure_tags,
        }
        rows.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(rows)


class LLMPatchPlanner:
    """Generate concrete YAML overrides by asking an LLM.

    Falls back to the keyword-based ``PatchPlanner`` on LLM failure.
    """

    def __init__(
        self,
        client: MetaLLMClient,
        registry: ModuleRegistry,
        override_base: str,
        agent_name: str = "",
    ) -> None:
        self._client = client
        self._registry = registry
        self._override_base = override_base
        self._agent_name = agent_name
        self._last_subagent_details: list[dict[str, Any]] = []
        self.meta_step: int = 0

    @property
    def last_subagent_details(self) -> list[dict[str, Any]]:
        """Planner sub-agent diagnostics from the most recent planning call."""
        return list(self._last_subagent_details)

    async def plan_from_batch(
        self,
        diagnoses: list[DiagnosisResult],
        traces: list[TraceRecord],
        n_samples: int = 4,
    ) -> list[PatchCandidate]:
        """Generate N independent PatchCandidates from the SAME prompt.

        Each candidate comes from a separate LLM call (temperature > 0 gives
        diversity), so they can form a GRPO group with different rewards.
        """
        if not diagnoses:
            return []

        try:
            return await self._llm_plan_multi(diagnoses, traces, n_samples)
        except Exception as exc:
            logger.warning(f"LLM patch planning failed ({exc}); falling back to rules")
            from skyrl_agent.meta_toolkit.editing.patch_planner import (
                PatchPlanner,
                PlannerConfig,
            )
            fallback = PatchPlanner(
                self._registry,
                config=PlannerConfig(override_base=self._override_base),
            )
            return fallback.plan_from_batch(diagnoses)[:n_samples]

    def _build_planning_messages(
        self,
        diagnoses: list[DiagnosisResult],
        traces: list[TraceRecord],
    ) -> list[dict[str, str]]:
        """Build the (system, user) messages for planning — reusable across N calls."""
        diag_text = self._format_diagnoses(diagnoses)
        trace_text = _serialise_traces_compact(
            [t for t in traces if not t.success][:10]
        ) or _serialise_traces_compact(traces[:10])

        # Show currently active hooks and overrides so the model avoids duplicating
        active_context = self._get_active_context()

        user_msg = (
            f"## Diagnoses\n{diag_text}\n\n"
            f"## Sample traces (failures prioritised)\n{trace_text}"
        )
        if active_context:
            user_msg += f"\n\n## Currently active modifications\n{active_context}"

        system = _SYSTEM_PROMPT_BASE.format(
            override_fields_doc=_OVERRIDE_FIELDS_DOC,
            agent_code_context=_TERMINUS2_CODE_CONTEXT,
            improvement_strategies=_IMPROVEMENT_STRATEGIES,
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    def _get_active_context(
        self,
        *,
        max_strategies: int = 12,
        max_steps_per_strategy: int = 1,
        max_step_chars: int = 180,
        max_hooks: int = 6,
        include_hook_source: bool = False,
        hook_source_chars: int = 320,
    ) -> str:
        """Summarize currently active strategy library and hooks for planning.

        Keeps the prompt compact by default, because full active context can
        easily exceed model context limits and starve hook-generation calls.
        """
        parts: list[str] = []

        override_dir = Path(self._override_base)
        sl_path = override_dir / "strategy_library.yaml"
        if sl_path.exists():
            try:
                import yaml
                content = yaml.safe_load(sl_path.read_text(encoding="utf-8")) or {}
                strategies = content.get("strategies", [])
                if strategies:
                    lines = ["**strategy_library.yaml** (current strategies):"]
                    for idx, s in enumerate(strategies[:max_strategies]):
                        pattern = str(s.get("pattern", "?"))[:140]
                        steps = s.get("steps", [])
                        lines.append(f"  [{idx}] pattern: {pattern}")
                        for step in steps[:max_steps_per_strategy]:
                            lines.append(f"      - {str(step)[:max_step_chars]}")
                        if len(steps) > max_steps_per_strategy:
                            lines.append(f"      ... ({len(steps)} steps total)")
                    if len(strategies) > max_strategies:
                        lines.append(
                            f"  ... ({len(strategies) - max_strategies} more strategies omitted)"
                        )
                    parts.append("\n".join(lines))
                else:
                    parts.append("**strategy_library.yaml**: (empty — no strategies yet)")
            except Exception:
                pass

        hooks_dir = override_dir / "hooks"
        if hooks_dir.is_dir():
            hook_files = sorted(hooks_dir.glob("*.py"))
            for hook_file in hook_files[:max_hooks]:
                try:
                    source = hook_file.read_text(encoding="utf-8")
                    if include_hook_source:
                        preview = source[:hook_source_chars]
                        parts.append(
                            f"**Active hook {hook_file.stem}**:\n```python\n{preview}\n```"
                        )
                    else:
                        first_line = source.strip().split("\n")[0][:140]
                        parts.append(f"**Active hook {hook_file.stem}**: {first_line}")
                except Exception:
                    pass
            if len(hook_files) > max_hooks:
                parts.append(f"... ({len(hook_files) - max_hooks} more hooks omitted)")

        return "\n".join(parts) if parts else ""

    @staticmethod
    def _rank_diagnoses(
        diagnoses: list[DiagnosisResult],
    ) -> list[DiagnosisResult]:
        """Sort diagnoses by confidence descending."""
        return sorted(diagnoses, key=lambda d: d.confidence, reverse=True)

    @staticmethod
    def _select_hook_relevant_diagnoses(
        diagnoses: list[DiagnosisResult],
        hook_names: list[str],
        *,
        max_items: int,
    ) -> list[DiagnosisResult]:
        """Pick diagnoses relevant to requested hook points.

        Supports both generic module mentions (e.g. ``hook:before_execute``)
        and concrete historical hook-instance mentions
        (e.g. ``hook:before_execute_candidate_1_xxx``).
        """
        requested_hook_points = set(hook_names)
        filtered: list[DiagnosisResult] = []
        for diagnosis in diagnoses:
            for module_name in diagnosis.candidate_modules:
                if not isinstance(module_name, str) or not module_name.startswith("hook:"):
                    continue
                hook_token = module_name[len("hook:"):].strip()
                hook_point = LLMPatchPlanner._hook_point_from_hook_name(hook_token)
                if hook_point is not None and hook_point in requested_hook_points:
                    filtered.append(diagnosis)
                    break
        if not filtered:
            filtered = diagnoses
        return filtered[:max_items]

    @staticmethod
    def _hook_point_from_hook_name(name: str) -> str | None:
        """Infer hook point from a hook filename stem/id."""
        stem = str(name).strip()
        if stem.endswith(".py"):
            stem = stem[:-3]
        for hook_point in _ALL_HOOK_POINTS:
            if stem == hook_point or stem.startswith(f"{hook_point}_"):
                return hook_point
        return None

    def _build_user_content(
        self,
        diagnoses: list[DiagnosisResult],
        traces: list[TraceRecord],
        *,
        include_strategy_suggestions: bool,
        active_context: str = "",
        diagnosis_heading: str = "## Diagnoses",
        max_traces: int = _MAX_TRACES_FOR_PROMPT,
    ) -> str:
        """Build compact planning user content."""
        diag_text = self._format_diagnoses_limited(
            diagnoses,
            include_strategy_suggestions=include_strategy_suggestions,
        )
        failed_first = [t for t in traces if not t.success][:max_traces]
        trace_text = _serialise_traces_compact(
            failed_first, max_traces=max_traces,
        ) or _serialise_traces_compact(
            traces[:max_traces], max_traces=max_traces,
        )

        user_content = (
            f"{diagnosis_heading}\n{diag_text}\n\n"
            f"## Sample traces (failures prioritised)\n{trace_text}"
        )
        if active_context:
            user_content += f"\n\n## Currently active modifications\n{active_context}"
        return user_content

    async def _repair_hooks(
        self,
        code_hooks: dict[str, str],
        max_retries: int = 3,
    ) -> dict[str, str]:
        """Validate each hook; on failure, ask the LLM to fix it (up to max_retries)."""
        from skyrl_agent.meta_toolkit.hooks.hook_executor import (
            _validate_hook_source, _compile_hook, _smoke_test_hook, HookPoint,
        )

        _HOOK_SIG_HINTS = {
            "before_llm_call": "def hook(prompt, context):\n    # prompt: str, context: HookContext\n    # Must return str",
            "before_execute": "def hook(commands, context):\n    # commands: list of objects with .keystrokes (str) and .duration_sec (float)\n    # Must return list (modify in-place or filter, do NOT construct new objects)",
            "after_execute": "def hook(terminal_output, context):\n    # terminal_output: str, context: HookContext\n    # Must return str",
            "on_timeout": "def hook(command_keystrokes, terminal_output, context):\n    # Must return str",
            "after_round": (
                "def hook(terminal_output, is_task_complete, context):\n"
                "    # Return dict with optional keys:\n"
                "    #   request_new_turn: bool\n"
                "    #   next_prompt: str\n"
                "    #   prompt_mode: 'append' | 'replace'\n"
                "    #   inject: str (legacy append helper)\n"
                "    # Use context.kv.setdefault(\"key\", default) for persistent state."
            ),
        }

        repaired: dict[str, str] = {}
        for hook_name, source in code_hooks.items():
            if not isinstance(source, str) or not source.strip():
                continue
            try:
                hp = HookPoint(hook_name)
            except ValueError:
                continue

            current_source = source
            for attempt in range(max_retries + 1):
                errors = _validate_hook_source(current_source, hp)
                if not errors:
                    fn = _compile_hook(current_source)
                    if fn is not None:
                        runtime_errors = _smoke_test_hook(fn, hp)
                        if not runtime_errors:
                            repaired[hook_name] = current_source
                            break
                        errors = runtime_errors
                    else:
                        errors = ["Compilation failed"]

                if attempt >= max_retries:
                    logger.warning(
                        f"Hook {hook_name} still broken after {max_retries} repair "
                        f"attempts; discarding. Last errors: {errors}"
                    )
                    break

                logger.info(
                    f"Hook {hook_name} validation failed (attempt {attempt+1}), "
                    f"asking LLM to repair: {errors}"
                )
                try:
                    sig_hint = _HOOK_SIG_HINTS.get(hook_name, "def hook(...):")
                    repair_msgs = [
                        {"role": "system", "content": (
                            "You are a Python code fixer. Fix the hook function below.\n"
                            "Rules:\n"
                            "- The function MUST be named `hook`.\n"
                            f"- Required signature for {hook_name}:\n"
                            f"  {sig_hint}\n"
                            "- `context` is a DATACLASS, NOT a dict.\n"
                            "  CORRECT: context.episode, context.kv, context.last_commands\n"
                            "  WRONG: context['episode'], context.get('kv')\n"
                            "- context.kv is a dict for persistent state. Always use:\n"
                            "  context.kv.setdefault('key', default_value)\n"
                            "- Only allowed imports: re, json, math, collections, "
                            "itertools, functools, copy, textwrap, string.\n"
                            "- For before_execute: do NOT construct new command objects.\n"
                            "- For after_round: return a dict with optional keys "
                            "{request_new_turn, next_prompt, prompt_mode, inject}.\n"
                            "- Return ONLY the fixed Python code, no markdown fences."
                        )},
                        {"role": "user", "content": (
                            f"Hook point: {hook_name}\n"
                            f"Errors: {errors}\n\n"
                            f"Broken code:\n{current_source}"
                        )},
                    ]
                    fixed = await self._client.chat(repair_msgs, temperature=0.2)
                    import re as _re
                    fixed = _re.sub(r"```python\s*\n?", "", fixed)
                    fixed = _re.sub(r"```\s*$", "", fixed).strip()
                    fixed = _re.sub(r"<think>.*?</think>", "", fixed, flags=_re.DOTALL).strip()
                    current_source = fixed
                except Exception as e:
                    logger.warning(f"Hook repair LLM call failed: {e}")
                    break

        return repaired

    async def _llm_plan_multi(
        self,
        diagnoses: list[DiagnosisResult],
        traces: list[TraceRecord],
        n_samples: int,
    ) -> list[PatchCandidate]:
        """Launch 3 sub-agents per candidate (strategy + two hook groups)."""
        self._client.set_role("planning")
        self._last_subagent_details: list[dict[str, Any]] = []

        ranked_diagnoses = self._rank_diagnoses(diagnoses)
        strategy_diagnoses = ranked_diagnoses[:_MAX_DIAGNOSES_FOR_STRATEGY_PROMPT]
        strategy_active_context = self._get_active_context(
            max_strategies=12,
            max_steps_per_strategy=1,
            max_hooks=4,
            include_hook_source=False,
        )
        strategy_user_content = self._build_user_content(
            strategy_diagnoses,
            traces,
            include_strategy_suggestions=True,
            active_context=strategy_active_context,
            diagnosis_heading="## Diagnoses",
            max_traces=_MAX_TRACES_FOR_PROMPT,
        )
        strategy_msgs = self._build_strategy_only_messages(strategy_user_content)

        hook_payloads: dict[str, tuple[list[str], list[dict[str, str]], set[str]]] = {}
        for category, category_hooks in _HOOK_CATEGORIES.items():
            allowed_hooks = list(category_hooks)
            allowed_hook_point_set = set(allowed_hooks)
            hook_diagnoses = self._select_hook_relevant_diagnoses(
                ranked_diagnoses,
                allowed_hooks,
                max_items=_MAX_DIAGNOSES_FOR_HOOK_PROMPT,
            )
            hook_active_context = self._get_active_context(
                max_strategies=8,
                max_steps_per_strategy=1,
                max_hooks=4,
                include_hook_source=False,
            )
            hook_user_content = self._build_user_content(
                hook_diagnoses,
                traces,
                include_strategy_suggestions=False,
                active_context=hook_active_context,
                diagnosis_heading=f"## Diagnoses ({category})",
                max_traces=_MAX_TRACES_FOR_PROMPT,
            )
            hook_msgs = self._build_hook_category_messages(
                category, allowed_hooks, hook_user_content,
            )
            hook_payloads[category] = (
                allowed_hooks,
                hook_msgs,
            )

        async def _run_strategy_subagent(candidate_idx: int) -> tuple[PatchCandidate | None, dict[str, Any]]:
            label = f"candidate_{candidate_idx}/strategy"
            detail: dict[str, Any] = {"label": label, "candidate_index": candidate_idx}
            try:
                data = await self._chat_json_with_compact_retry(
                    strategy_msgs,
                    label=label,
                    retry_hint=_JSON_RETRY_HINT_STRATEGY,
                    temperature=0.7,
                    max_tokens=_STRATEGY_SUBAGENT_MAX_TOKENS,
                )
            except Exception as exc:
                logger.warning(f"Strategy sub-agent {label} failed: {exc}")
                detail["status"] = "error"
                detail["error"] = str(exc)
                return None, detail

            data["code_hooks"] = {}
            candidate = self._parse_response_to_candidate(
                data, tag=f"candidate_{candidate_idx}_strategy",
            )
            if candidate is None:
                detail["status"] = "empty"
            else:
                detail["status"] = "ok"
                detail["modules"] = candidate.target_modules
                detail["n_files"] = len(candidate.files)
            return candidate, detail

        async def _run_hook_subagent(
            candidate_idx: int,
            category: str,
            allowed_hooks: list[str],
            hook_msgs: list[dict[str, str]],
        ) -> tuple[PatchCandidate | None, dict[str, Any]]:
            label = f"candidate_{candidate_idx}/hookgrp_{category}"
            detail: dict[str, Any] = {
                "label": label,
                "candidate_index": candidate_idx,
                "category": category,
            }
            try:
                data = await self._chat_json_with_compact_retry(
                    hook_msgs,
                    label=label,
                    retry_hint=_JSON_RETRY_HINT_HOOK,
                    temperature=0.7,
                    max_tokens=_HOOK_SUBAGENT_MAX_TOKENS,
                )
            except Exception as exc:
                logger.warning(f"Hook sub-agent {label} failed: {exc}")
                detail["status"] = "error"
                detail["error"] = str(exc)
                return None, detail

            raw_hooks = data.get("code_hooks", {})
            if not isinstance(raw_hooks, dict):
                raw_hooks = {}
            allowed = set(allowed_hooks)
            raw_hooks = {k: v for k, v in raw_hooks.items() if k in allowed}
            if raw_hooks:
                repaired = await self._repair_hooks(raw_hooks)
                data["code_hooks"] = repaired
            else:
                data["code_hooks"] = {}
            data["strategy_edits"] = []

            candidate = self._parse_response_to_candidate(
                data,
                tag=f"candidate_{candidate_idx}_hookgrp_{category}",
            )
            if candidate is None:
                detail["status"] = "empty"
            else:
                detail["status"] = "ok"
                detail["modules"] = candidate.target_modules
                detail["n_files"] = len(candidate.files)
            return candidate, detail

        async def _run_candidate_group(candidate_idx: int) -> tuple[PatchCandidate | None, list[dict[str, Any]]]:
            strategy_task = asyncio.ensure_future(_run_strategy_subagent(candidate_idx))
            hook_tasks = [
                asyncio.ensure_future(
                    _run_hook_subagent(
                        candidate_idx,
                        category,
                        allowed_hooks,
                        hook_msgs,
                    ),
                )
                for category, (allowed_hooks, hook_msgs) in hook_payloads.items()
            ]
            sub_results = await asyncio.gather(strategy_task, *hook_tasks)

            partials: list[PatchCandidate] = []
            details: list[dict[str, Any]] = []
            for candidate, detail in sub_results:
                details.append(detail)
                if candidate is not None:
                    partials.append(candidate)

            merged = self._merge_candidate_partials(
                partials, candidate_idx=candidate_idx,
            )
            summary = {
                "label": f"candidate_{candidate_idx}",
                "candidate_index": candidate_idx,
                "status": "ok" if merged is not None else "empty",
                "n_partials": len(partials),
            }
            if merged is not None:
                summary["modules"] = merged.target_modules
                summary["n_files"] = len(merged.files)
            details.append(summary)
            return merged, details

        candidate_tasks: list[asyncio.Task] = [
            asyncio.ensure_future(_run_candidate_group(i))
            for i in range(n_samples)
        ]
        results = await asyncio.gather(*candidate_tasks, return_exceptions=True) if candidate_tasks else []

        candidates: list[PatchCandidate] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Planner candidate group {idx} raised: {result}")
                self._last_subagent_details.append({
                    "label": f"candidate_{idx}",
                    "candidate_index": idx,
                    "status": "error",
                    "error": str(result),
                })
                continue
            merged_candidate, details = result
            self._last_subagent_details.extend(details)
            if merged_candidate is not None:
                candidates.append(merged_candidate)

        logger.info(
            f"LLM planner produced {len(candidates)} candidates from "
            f"{len(candidate_tasks)} candidate groups "
            f"({len(candidate_tasks) * (1 + len(_HOOK_CATEGORIES))} sub-agent calls)"
        )
        return candidates

    @staticmethod
    def _merge_candidate_partials(
        partials: list[PatchCandidate],
        *,
        candidate_idx: int,
    ) -> PatchCandidate | None:
        """Merge strategy/hook partial patches into one candidate."""
        if not partials:
            return None

        modules: list[str] = []
        files: list[PatchFileEdit] = []
        required_tests: list[str] = []
        rollback_if: list[str] = []
        notes: list[str] = []
        risk = "low"

        for part in partials:
            for m in part.target_modules:
                if m not in modules:
                    modules.append(m)
            files.extend(part.files)
            for t in part.required_tests:
                if t not in required_tests:
                    required_tests.append(t)
            for r in part.rollback_if:
                if r not in rollback_if:
                    rollback_if.append(r)
            if part.notes:
                notes.append(part.notes)
            if part.risk == "high":
                risk = "high"
            elif part.risk == "medium" and risk == "low":
                risk = "medium"

        if not files:
            return None

        return PatchCandidate(
            target_modules=modules,
            files=files,
            risk=risk,
            required_tests=required_tests or ["canary_eval"],
            rollback_if=rollback_if or ["canary_regression"],
            notes=" | ".join(notes)[:500] or f"[LLM candidate_{candidate_idx}] merged partial edits",
        )

    async def _chat_json_with_compact_retry(
        self,
        messages: list[dict[str, str]],
        *,
        label: str,
        retry_hint: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Call chat_json with one compact-output retry on parse failure."""
        try:
            return await self._client.chat_json(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as first_error:
            logger.warning(
                f"{label}: first JSON call failed ({first_error}); retrying with compact hint"
            )
            retry_messages = list(messages) + [
                {"role": "user", "content": retry_hint},
            ]
            return await self._client.chat_json(
                retry_messages,
                temperature=min(temperature, 0.3),
                max_tokens=min(max_tokens, 2048),
            )

    def _build_strategy_only_messages(self, user_content: str) -> list[dict[str, str]]:
        """System prompt focused solely on strategy library edits."""
        system = (
            "You are an expert AI-agent architect. Based on the diagnosis and traces, "
            "generate ONLY strategy_edits (add/edit/remove) for the agent's strategy library.\n"
            "Do NOT generate code_hooks.\n\n"
            "Do NOT use unresolved config placeholders like ${VAR} in strategy text.\n"
            "Use concrete shell snippets or explicit placeholder words without ${...} syntax.\n\n"
            "Output must be STRICT JSON only (no markdown fences / no <think> blocks).\n"
            "Keep output compact and high-impact to avoid truncation.\n\n"
            f"{_IMPROVEMENT_STRATEGIES}\n\n"
            "## Output format\n\n"
            '```json\n{{\n  "strategy_edits": [\n'
            '    {{"action": "add", "strategy": {{"pattern": "...", "steps": ["..."]}}}}\n'
            "  ],\n"
            '  "rationale": "<brief explanation>"\n}}\n```\n'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _build_hook_only_messages(self, hook_name: str, user_content: str) -> list[dict[str, str]]:
        """System prompt focused on generating a single hook."""
        system = (
            f"You are an expert AI-agent architect. Generate ONLY a code hook for "
            f"the **{hook_name}** hook point. Do NOT generate strategy_edits.\n\n"
            f"{_OVERRIDE_FIELDS_DOC}\n\n"
            f"{_TERMINUS2_CODE_CONTEXT}\n\n"
            "## Output format\n\n"
            "```json\n{{\n"
            f'  "code_hooks": {{"{hook_name}": "<Python source>"}},\n'
            '  "rationale": "<brief explanation>"\n}}\n```\n'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _build_hook_category_messages(
        self,
        category: str,
        hook_names: list[str],
        user_content: str,
    ) -> list[dict[str, str]]:
        """System prompt focused on a hook category (multiple hook points)."""
        hooks_literal = ", ".join(hook_names)
        system = (
            "You are an expert AI-agent architect. Generate ONLY code hooks.\n"
            f"Hook category: {category}\n"
            f"Allowed hook points in this category: {hooks_literal}\n"
            "You may generate one or more hooks from this allowed set.\n"
            "Do NOT generate strategy_edits.\n\n"
            "Output must be STRICT JSON only (no markdown fences / no <think> blocks).\n"
            "Keep output compact and robust.\n\n"
            f"{_OVERRIDE_FIELDS_DOC}\n\n"
            f"{_TERMINUS2_CODE_CONTEXT}\n\n"
            "## Output format\n\n"
            "```json\n{{\n"
            '  "code_hooks": {\n'
            f'    "{hook_names[0]}": "<Python source>"\n'
            "  },\n"
            '  "rationale": "<brief explanation>"\n}}\n```\n'
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _merge_strategy_and_hook_candidates(
        candidates: list[PatchCandidate], n_target: int,
    ) -> list[PatchCandidate]:
        """Combine pure-strategy and pure-hook candidates into mixed ones.

        If we have more candidates than ``n_target``, pair up strategy and hook
        candidates so each resulting candidate has both edits (richer for GRPO).
        Leftover candidates are kept as-is.
        """
        strategy_only = [c for c in candidates if all(m == "strategy_library" for m in c.target_modules)]
        hook_only = [c for c in candidates if all(m.startswith("hook:") for m in c.target_modules)]
        mixed = [c for c in candidates if c not in strategy_only and c not in hook_only]

        merged: list[PatchCandidate] = list(mixed)

        si, hi = 0, 0
        while si < len(strategy_only) and hi < len(hook_only):
            sc = strategy_only[si]
            hc = hook_only[hi]
            merged.append(PatchCandidate(
                target_modules=sc.target_modules + hc.target_modules,
                files=sc.files + hc.files,
                risk=hc.risk,
                required_tests=["canary_eval"],
                rollback_if=["canary_regression"],
                notes=f"[merged] {sc.notes} + {hc.notes}",
            ))
            si += 1
            hi += 1

        for c in strategy_only[si:]:
            merged.append(c)
        for c in hook_only[hi:]:
            merged.append(c)

        return merged[:max(n_target, len(merged))]

    def _parse_response_to_candidate(
        self,
        data: dict[str, Any],
        tag: str = "",
    ) -> PatchCandidate | None:
        """Parse one LLM JSON response into a single PatchCandidate.

        Two-tier output: strategy_edits + code_hooks.
        """
        strategy_edits: list[dict[str, Any]] = data.get("strategy_edits", [])
        code_hooks: dict[str, str] = data.get("code_hooks", {})
        rationale: str = data.get("rationale", "")

        overrides_map = data.get("overrides", {})
        if isinstance(overrides_map, dict) and "strategy_library" in overrides_map and not strategy_edits:
            sl = overrides_map.pop("strategy_library")
            strategies = sl.get("strategies", []) if isinstance(sl, dict) else []
            for s in strategies:
                if isinstance(s, dict) and "pattern" in s:
                    strategy_edits.append({"action": "add", "strategy": s})

        if not code_hooks and not strategy_edits:
            return None

        target_modules: list[str] = []
        file_edits: list[PatchFileEdit] = []
        risk = "low"

        # Strategy library edits: add/edit/remove operations
        if strategy_edits:
            valid_edits = []
            for edit in strategy_edits:
                if not isinstance(edit, dict):
                    continue
                action = edit.get("action", "")
                if action == "add" and isinstance(edit.get("strategy"), dict):
                    s = edit["strategy"]
                    if "pattern" in s and "steps" in s:
                        valid_edits.append(edit)
                elif action == "edit" and isinstance(edit.get("index"), int) and isinstance(edit.get("strategy"), dict):
                    valid_edits.append(edit)
                elif action == "remove" and isinstance(edit.get("index"), int):
                    valid_edits.append(edit)

            if valid_edits:
                intent = json.dumps(
                    {"strategy_edits": valid_edits, "rationale": rationale},
                    ensure_ascii=False,
                )
                override_path = f"{self._override_base}/strategy_library.yaml"
                file_edits.append(PatchFileEdit(
                    path=override_path, change_type="modify", intent=intent,
                ))
                target_modules.append("strategy_library")

        # Code hooks: model-generated Python functions
        if code_hooks:
            for hook_name, hook_source in code_hooks.items():
                if not isinstance(hook_source, str) or not hook_source.strip():
                    continue
                try:
                    from skyrl_agent.meta_toolkit.hooks.hook_executor import (
                        _validate_hook_source, _compile_hook, _smoke_test_hook,
                        HookPoint,
                    )
                    hp = HookPoint(hook_name)
                    errors = _validate_hook_source(hook_source, hp)
                    if errors:
                        logger.warning(f"Hook {hook_name} static validation failed: {errors}")
                        continue
                    fn = _compile_hook(hook_source)
                    if fn is None:
                        logger.warning(f"Hook {hook_name} compilation failed")
                        continue
                    runtime_errors = _smoke_test_hook(fn, hp)
                    if runtime_errors:
                        logger.warning(f"Hook {hook_name} smoke test failed: {runtime_errors}")
                        continue
                except (ValueError, ImportError) as e:
                    logger.warning(f"Hook {hook_name} rejected: {e}")
                    continue

                tag_variant = "".join(ch if ch.isalnum() else "_" for ch in tag).strip("_")
                prefix = f"hook_{hook_name}_"
                if tag_variant.startswith(prefix):
                    tag_variant = tag_variant[len(prefix):]
                elif tag_variant == f"hook_{hook_name}":
                    tag_variant = ""
                unique_suffix = uuid4().hex[:8]
                step_tag = f"s{self.meta_step}" if self.meta_step > 0 else "s0"
                if tag_variant:
                    hook_filename = f"{hook_name}_{step_tag}_{tag_variant}_{unique_suffix}.py"
                else:
                    hook_filename = f"{hook_name}_{step_tag}_{unique_suffix}.py"

                hook_path = f"{self._override_base}/hooks/{hook_filename}"
                file_edits.append(PatchFileEdit(
                    path=hook_path, change_type="create", intent=hook_source,
                ))
                target_modules.append(f"hook:{hook_name}")
                risk = "high"

        if not file_edits:
            return None

        return PatchCandidate(
            target_modules=target_modules,
            files=file_edits,
            risk=risk,
            required_tests=["canary_eval"],
            rollback_if=["canary_regression"],
            notes=f"[LLM {tag}] {rationale[:200]}",
        )

    @staticmethod
    def _format_diagnoses(diagnoses: list[DiagnosisResult]) -> str:
        return LLMPatchPlanner._format_diagnoses_limited(
            diagnoses,
            include_strategy_suggestions=True,
        )

    @staticmethod
    def _format_diagnoses_limited(
        diagnoses: list[DiagnosisResult],
        *,
        include_strategy_suggestions: bool,
        max_hypotheses_per_diagnosis: int = 3,
        max_tasks_per_diagnosis: int = 5,
        max_strategy_suggestions: int = 12,
        max_steps_per_suggestion: int = 5,
    ) -> str:
        parts: list[str] = []
        all_suggestions: list[dict] = []
        for d in diagnoses:
            hypotheses = [str(h)[:260] for h in d.root_cause_hypotheses[:max_hypotheses_per_diagnosis]]
            block = (
                f"- **{d.problem_type}** (conf={d.confidence:.2f})\n"
                f"  Hypotheses: {'; '.join(hypotheses)}\n"
                f"  Candidate modules: {d.candidate_modules}\n"
                f"  Affected tasks: {d.affected_task_ids[:max_tasks_per_diagnosis]}"
            )
            summary = str(d.metadata.get("analysis_summary", ""))[:700]
            if summary:
                block += f"\n  Evidence: {summary}"
            if include_strategy_suggestions and d.strategy_suggestions:
                block += f"\n  Strategy suggestions: {len(d.strategy_suggestions)} extracted"
                all_suggestions.extend(d.strategy_suggestions)
            parts.append(block)

        result = "\n".join(parts)
        if include_strategy_suggestions and all_suggestions:
            result += "\n\n## Extracted strategy suggestions from successful traces\n"
            result += (
                "The diagnoser extracted these strategies from comparing successful "
                "vs failed traces. You MUST add them to the strategy library via "
                "`strategy_edits` with action `add`.\n\n"
            )
            shown = all_suggestions[:max_strategy_suggestions]
            for i, s in enumerate(shown):
                steps = s.get("steps", [])
                if isinstance(steps, list):
                    steps = steps[:max_steps_per_suggestion]
                result += (
                    f"{i+1}. **Pattern**: {s.get('pattern', '?')}\n"
                    f"   **Steps**: {steps}\n"
                    f"   **Source**: {s.get('source', 'unknown')}\n"
                )
            if len(all_suggestions) > max_strategy_suggestions:
                result += (
                    f"... ({len(all_suggestions) - max_strategy_suggestions} more "
                    "strategy suggestions omitted)\n"
                )
        return result
