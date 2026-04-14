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

from skyrl_agent.meta_toolkit.diagnosis.rule_based_diagnoser import DiagnosisResult
from skyrl_agent.meta_toolkit.editing.module_registry import ModuleRegistry
from skyrl_agent.meta_toolkit.editing.patch_schema import (
    ChangeType,
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
| context.kv | dict | **Writable** persistent store across rounds |

### context.kv usage (IMPORTANT)

`context.kv` is the ONLY writable field. Use it to track state across rounds.
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

**on_parse_error**(raw_response: str, error: str, context) -> str|None
```python
def hook(raw_response, error, context):
    return "Your response had a formatting error: " + error + ". Please use the correct JSON format."
```

**after_round**(terminal_output: str, is_task_complete: bool, context) -> dict
The most powerful hook. Runs after every round. Return {{"inject": "text"}}
to append text to the next observation, or {{}} for no-op.

NOTE on completion flow: when the agent outputs task_complete=true, terminus-2
uses a two-step confirmation (agent must say task_complete twice in a row).
The after_round hook runs BEFORE the confirmation check, so it can inject text
into the confirmation prompt. If your hook injects a verification reminder on
the first task_complete, the agent will see it and can decide whether to
confirm or continue working.

Example — completion verification (inject verification prompt on first complete):
```python
def hook(terminal_output, is_task_complete, context):
    count = context.kv.setdefault("complete_count", 0)
    if is_task_complete:
        context.kv["complete_count"] = count + 1
        if count < 1:
            return {{"inject": (
                "[VERIFY] You said task_complete. Before confirming, please:\\n"
                "1. Run the tests.\\n"
                "2. Check the output matches the expected format.\\n"
                "If all correct, submit task_complete again."
            )}}
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
        return {{"inject": (
            "[LOOP DETECTED] You are repeating commands. "
            "Try a different approach. Task: "
            + context.original_instruction[:200]
        )}}
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
        if feedback and "ERROR:" in feedback:
            self._run_hook("on_parse_error", llm_response.content, feedback, context)
            continue
        # 3. before_execute hook
        commands = self._run_hook("before_execute", commands, context)
        # 4. Execute commands in tmux
        timeout_occurred, terminal_output = await self._execute_commands(commands, session)
        if timeout_occurred:
            self._run_hook("on_timeout", last_cmd, terminal_output, context)
        else:
            terminal_output = self._run_hook("after_execute", terminal_output, context)
        # 5. after_round hook — returns dict; if {"inject": text}, appends text to observation
        inject_text = await self._run_after_round_hook(terminal_output, is_task_complete)
        if inject_text: terminal_output += "\\n\\n" + inject_text
        # 6. Completion check (two-step confirmation)
        if is_task_complete:
            if self._pending_completion: return
            else: self._pending_completion = True; prompt = confirmation_msg; continue
        prompt = terminal_output
```

Key facts:
- Commands execute serially in tmux; context.kv persists within a trial.
- context.last_commands contains the keystrokes from the PREVIOUS step.
- The after_round hook is the most powerful: if it returns {"inject": text},
  that text is appended to the observation, giving the model extra guidance.

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

### 4. Repeated errors → Error recovery
If traces show high `tool_failures` or repeated parse errors:
- Use an `on_parse_error` hook to construct a targeted recovery prompt that
  tells the agent exactly what went wrong and how to fix the format.
- Add strategies reminding the agent of the correct output schema.

### 5. Stuck in loops → Loop detection via after_round
If traces show the agent repeating the same commands without progress:
- Use an `after_round` hook that tracks command history in `context.kv`.
  When repeated patterns are detected, inject a warning suggesting a new
  approach. Alternatively use a `before_llm_call` hook for earlier detection.

### 6. Poor planning → Strategic guidance
If traces show planning_failure or the agent taking many turns:
- Add strategy_edits with task-decomposition advice (e.g. "Read the task
  fully before acting", "Verify each sub-goal before moving on").

### 7. Timeout handling → Graceful recovery
If traces show many timeouts:
- Use an `on_timeout` hook to append recovery guidance to terminal output.

### 8. Verification gaps → Post-step verification via after_round
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
The `after_round` hook is the most powerful: it can inject extra text into
the agent's observation, enabling completion verification and loop detection.

{override_fields_doc}

{agent_code_context}

{improvement_strategies}

## Rules

1. **Maximum 3 changes per patch.** Each strategy_edit counts as 1 change,
   each code_hook counts as 1 change. Pick the highest-impact changes only.
2. Only generate hooks for hook points listed in the diagnosis `candidate_modules`.
   `strategy_library` is always allowed — you can ALWAYS add strategy_edits.
3. When `strategy_suggestions` are provided by the diagnoser, you MUST add
   them to the strategy library via `strategy_edits` with action `add`.
4. Be conservative — change the minimum necessary.
5. Ground choices in trace evidence (mention task IDs, failure modes).
6. For `code_hooks`: write a complete function named `hook` with the correct
   signature. Only use allowed imports. Write defensive code with fallbacks.
   Always initialize `context.kv` keys with setdefault before using them.
7. For `after_round` hooks: the hook returns a dict. To inject text into the
   next observation, return {{"inject": "your text"}}. Return {{}} otherwise.
   Use `context.kv` to track state across rounds (e.g. confirm counts).
8. Match the strategy to the diagnosis. Do NOT blindly apply strategies.
9. For `strategy_edits`: prefer `add` for new patterns, `edit` to refine
   existing strategies (reference by index), `remove` to delete obsolete ones.

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
  before_llm_call, before_execute, after_execute, on_timeout, on_parse_error,
  after_round.
- Total changes (strategy_edits count + hooks count) must be ≤ 3.
"""

_MODULE_OVERRIDE_FILE: dict[str, str] = {
    "strategy_library": "strategy_library.yaml",
}


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

    def _get_active_context(self) -> str:
        """Summarize currently active strategy library and hooks for the planning prompt."""
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
                    for idx, s in enumerate(strategies):
                        pattern = s.get("pattern", "?")
                        steps = s.get("steps", [])
                        lines.append(f"  [{idx}] pattern: {pattern}")
                        for step in steps[:3]:
                            lines.append(f"      - {step}")
                        if len(steps) > 3:
                            lines.append(f"      ... ({len(steps)} steps total)")
                    parts.append("\n".join(lines))
                else:
                    parts.append("**strategy_library.yaml**: (empty — no strategies yet)")
            except Exception:
                pass

        hooks_dir = override_dir / "hooks"
        if hooks_dir.is_dir():
            for hook_file in sorted(hooks_dir.glob("*.py")):
                try:
                    source = hook_file.read_text(encoding="utf-8")
                    parts.append(f"**Active hook {hook_file.stem}**:\n```python\n{source}\n```")
                except Exception:
                    pass

        return "\n".join(parts) if parts else ""

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
            "on_parse_error": "def hook(raw_response, error, context):\n    # Must return str or None",
            "after_round": (
                "def hook(terminal_output, is_task_complete, context):\n"
                "    # Return dict: {\"inject\": \"text\"} to add text to observation, or {} for no-op.\n"
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
                            "- For after_round: return a dict (e.g. {'inject': 'text'} or {}).\n"
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
        """Call the LLM N times with the SAME prompt, producing N diverse candidates."""
        self._client.set_role("planning")
        messages = self._build_planning_messages(diagnoses, traces)

        async def _single_call(idx: int) -> PatchCandidate | None:
            data = None
            # First attempt with default max_tokens
            try:
                data = await self._client.chat_json(messages, temperature=0.7)
            except Exception as exc:
                logger.warning(
                    f"LLM planning sample {idx} first attempt failed: {exc}; "
                    "retrying with higher max_tokens"
                )

            # Retry with doubled max_tokens in case of truncation
            if data is None:
                try:
                    higher_tokens = (self._client._cfg.max_tokens or 4096) * 2
                    data = await self._client.chat_json(
                        messages, temperature=0.7, max_tokens=higher_tokens,
                    )
                except Exception as exc:
                    logger.warning(f"LLM planning sample {idx} retry also failed: {exc}")
                    return None

            # Attempt hook repair if any hooks fail validation
            code_hooks = data.get("code_hooks", {})
            if code_hooks:
                repaired = await self._repair_hooks(code_hooks)
                data["code_hooks"] = repaired

            return self._parse_response_to_candidate(data, tag=f"sample_{idx}")

        results = await asyncio.gather(*[_single_call(i) for i in range(n_samples)])
        candidates = [c for c in results if c is not None]

        logger.info(
            f"LLM planner produced {len(candidates)}/{n_samples} valid candidates"
        )
        return candidates

    def _parse_response_to_candidate(
        self, data: dict[str, Any], tag: str = ""
    ) -> PatchCandidate | None:
        """Parse one LLM JSON response into a single PatchCandidate.

        Two-tier output: strategy_edits + code_hooks.
        """
        strategy_edits: list[dict[str, Any]] = data.get("strategy_edits", [])
        code_hooks: dict[str, str] = data.get("code_hooks", {})
        rationale: str = data.get("rationale", "")

        # Backward compat: if old "overrides" contains strategy_library, convert
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
        change_count = 0
        max_changes = 3

        # Strategy library edits: add/edit/remove operations
        if strategy_edits and change_count < max_changes:
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
                change_count += 1

        # Code hooks: model-generated Python functions
        if code_hooks:
            for hook_name, hook_source in code_hooks.items():
                if change_count >= max_changes:
                    break
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

                hook_path = f"{self._override_base}/hooks/{hook_name}.py"
                file_edits.append(PatchFileEdit(
                    path=hook_path, change_type="create", intent=hook_source,
                ))
                target_modules.append(f"hook:{hook_name}")
                change_count += 1
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
        parts: list[str] = []
        all_suggestions: list[dict] = []
        for d in diagnoses:
            block = (
                f"- **{d.problem_type}** (conf={d.confidence:.2f})\n"
                f"  Hypotheses: {'; '.join(d.root_cause_hypotheses)}\n"
                f"  Candidate modules: {d.candidate_modules}\n"
                f"  Affected tasks: {d.affected_task_ids[:5]}"
            )
            summary = d.metadata.get("analysis_summary", "")
            if summary:
                block += f"\n  Evidence: {summary}"
            if d.strategy_suggestions:
                block += f"\n  Strategy suggestions: {len(d.strategy_suggestions)} extracted"
                all_suggestions.extend(d.strategy_suggestions)
            parts.append(block)

        result = "\n".join(parts)
        if all_suggestions:
            result += "\n\n## Extracted strategy suggestions from successful traces\n"
            result += (
                "The diagnoser extracted these strategies from comparing successful "
                "vs failed traces. You MUST add them to the strategy library via "
                "`strategy_edits` with action `add`.\n\n"
            )
            for i, s in enumerate(all_suggestions):
                result += (
                    f"{i+1}. **Pattern**: {s.get('pattern', '?')}\n"
                    f"   **Steps**: {s.get('steps', [])}\n"
                    f"   **Source**: {s.get('source', 'unknown')}\n"
                )
        return result
