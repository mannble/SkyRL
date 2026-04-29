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
| context.total_episodes | int | Configured round safety cap; may be very large |
| context.n_commands_executed | int | Total commands run so far |
| context.n_parse_errors | int | Total parse failures so far |
| context.n_timeouts | int | Total command timeouts so far |
| context.last_commands | list[str] | Most recent parsed command strings |
| context.last_analysis | str | Most recent parsed LLM analysis text |
| context.last_plan | str | Most recent parsed LLM plan text |
| context.is_task_complete | bool | Most recent parsed task_complete value |
| context.original_instruction | str | The original task description |
| context.last_terminal_output | str | Terminal output from the most recent command |
| context.next_observation | str | Base message that will be shown to the agent next, before hook guidance |
| context.execution_history | list[dict] | Past rounds: [{"commands":[..], "output":"...", "episode":N}] (last 20) |
| context.last_llm_response | str | Raw LLM response text before parsing |
| context.kv | dict | **Writable** persistent store across rounds |

Timing note:
- `before_llm_call` sees the previous completed round in `last_commands`,
  `last_analysis`, `last_plan`, `is_task_complete`, and
  `last_terminal_output`; these are empty/default values on the first call.
- `after_execute`, `on_timeout`, and `after_round` see the current round in
  `last_commands`, `last_analysis`, `last_plan`, and `is_task_complete`.
- `after_execute` and `on_timeout` receive the current terminal output as an
  argument; `context.last_terminal_output` is updated after they return.
- `after_round` also sees the current round terminal output through both the
  `terminal_output` argument and `context.last_terminal_output`.
- `after_round` runs after the base next observation is constructed. Its
  `context.next_observation` is the exact message the agent would receive next
  without hook guidance. For `task_complete=true`, this includes the normal
  terminus-2 completion confirmation message.

### context.kv usage (IMPORTANT)

`context.kv` is the ONLY writable field. Use it to track state across rounds.
Hooks produced by the same generated hook group share one `context.kv`
namespace across hook points. Different hook groups get isolated namespaces to
avoid key collisions.
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

Current hook groups:
- `prompt_guidance`: `before_llm_call` only. Use for initial or periodic short
  reminders (for example once at episode 0, then every 6-8 rounds when helpful).
  Do not add guidance every turn. Its `context.kv` is separate from runtime
  recovery hooks.
- `runtime_recovery`: `after_execute`, `on_timeout`, and `after_round`. Use
  observer hooks to write compact signals into `context.kv`; use `after_round`
  to consume those signals and append prompt guidance.

Communication pattern:
- Write small, typed values such as counters, booleans, labels, and short
  command snippets. Store compact summaries instead of full terminal output.
- Bound any history list to a small size.
- Use one-shot keys for transient events, and remove them with `pop` after
  `after_round` consumes them.
- Prefer keys that describe signals, e.g. `error_kind`, `timeout_kind`,
  `loop_count`, `last_failed_command`.

### Hook roles and examples

**before_llm_call**(prompt: str, context) -> dict
Runs immediately before each LLM call. Best for initial or periodic short
reminders; keep appended text brief. Return `{"append_prompt": "..."}` to add
guidance after the existing prompt, or `{}` when no guidance is needed.
```python
def hook(prompt, context):
    if context.episode == 0 and not context.kv.get("initial_reminder_sent"):
        context.kv["initial_reminder_sent"] = True
        return {
            "append_prompt": (
                "[REMINDER] Read the task carefully, identify required files, "
                "then act with small verifiable commands."
            )
        }
    if context.episode > 0 and context.episode % 6 == 0:
        return {
            "append_prompt": (
                "[REMINDER] Avoid repeating checks; move toward the deliverable."
            )
        }
    return {}
```

**after_execute**(terminal_output: str, context) -> None
Observer hook. Record compact signals for a sibling `after_round` hook.
```python
def hook(terminal_output, context):
    lower = terminal_output.lower()
    if "no such file" in lower or "not found" in lower:
        context.kv["error_kind"] = "missing_path"
        context.kv["last_failed_command"] = (context.last_commands[-1] if context.last_commands else "")[:200]
    elif "permission denied" in lower:
        context.kv["error_kind"] = "permission"
        context.kv["last_failed_command"] = (context.last_commands[-1] if context.last_commands else "")[:200]
    context.kv["last_output_chars"] = len(terminal_output)
```

**on_timeout**(command_keystrokes: str, terminal_output: str, context) -> None
Observer hook. Record timeout-specific signals for a sibling `after_round` hook.
```python
def hook(command_keystrokes, terminal_output, context):
    cmd = command_keystrokes.strip()
    context.kv["timeout_count"] = context.kv.setdefault("timeout_count", 0) + 1
    context.kv["last_timeout_command"] = cmd[:200]
    if "find /" in cmd or "grep -R /" in cmd:
        context.kv["timeout_kind"] = "broad_search"
    elif "apt " in cmd or "pip " in cmd:
        context.kv["timeout_kind"] = "install_or_network"
    else:
        context.kv["timeout_kind"] = "long_command"
```

**after_round**(terminal_output: str, is_task_complete: bool, context) -> dict
Runs after every round. In `runtime_recovery`, this is the controller that
reads signals written by sibling observer hooks.
Return fields (all optional):
- `next_prompt` (str): prompt guidance appended after `context.next_observation`.
- `prompt_mode` ("append"): runtime applies guidance in append mode.

Runtime effect model: hooks influence later behavior through `context.kv` and
through appended prompt guidance. `after_execute` and `on_timeout` are
observer hooks; they update `context.kv` and do not return terminal output.
`after_round` returns guidance in `next_prompt`.

NOTE on completion flow: when the agent outputs task_complete=true, terminus-2
uses a two-step confirmation (agent must say task_complete twice in a row).
The after_round hook sees the base confirmation message in
`context.next_observation` and can append short guidance after it. Do not repeat
the confirmation text; add only evidence-based guidance that helps the agent
decide whether to confirm completion or fix a concrete issue.

Example — consume observer signals from the same `runtime_recovery` group:
```python
def hook(terminal_output, is_task_complete, context):
    timeout_kind = context.kv.pop("timeout_kind", "")
    if timeout_kind == "broad_search":
        return {
            "next_prompt": (
                "[TIMEOUT] The previous command looked like an overly broad search. "
                "Narrow the path or pattern before retrying."
            ),
            "prompt_mode": "append",
        }

    error_kind = context.kv.pop("error_kind", "")
    cmd = context.kv.pop("last_failed_command", "") if error_kind else ""
    if error_kind == "missing_path":
        return {
            "next_prompt": (
                "[PATH ISSUE] A recent command referenced a missing path. "
                "List the parent directory or use find/ls before retrying. "
                + ("Command: " + cmd if cmd else "")
            ),
            "prompt_mode": "append",
        }
    if error_kind == "permission":
        return {
            "next_prompt": (
                "[PERMISSION] A recent command hit a permission error. "
                "Check ownership/mode or choose a writable target before retrying. "
                + ("Command: " + cmd if cmd else "")
            ),
            "prompt_mode": "append",
        }
    return {}
```

Example — completion-aware prompt guidance:
```python
def hook(terminal_output, is_task_complete, context):
    count = context.kv.setdefault("complete_count", 0)
    if is_task_complete:
        context.kv["complete_count"] = count + 1
        if count < 1:
            return {
                "next_prompt": (
                    "[COMPLETION CHECK] If the previous output already verifies "
                    "the required artifact, confirm completion now. Only run "
                    "another command if there is a concrete missing file, wrong "
                    "content, failed check, or unclear artifact location."
                ),
                "prompt_mode": "append",
            }
    else:
        context.kv["complete_count"] = 0
    return {}
```

Example — loop detection with state only:
```python
def hook(terminal_output, is_task_complete, context):
    history = context.kv.setdefault("cmd_history", [])
    for cmd in context.last_commands:
        history.append(cmd.strip())
    if len(history) > 20:
        history[:] = history[-20:]
    if len(history) >= 4 and history[-2:] == history[-4:-2]:
        context.kv["loop_detected"] = True
        return {
            "next_prompt": (
                "[LOOP DETECTED] You are repeating commands. Try a different approach."
            ),
            "prompt_mode": "append",
        }
    return {}
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
        # 1. before_llm_call hook — returns append-only prompt guidance
        prompt_control = self._run_hook("before_llm_call", prompt, context)
        # Expected return: {"append_prompt": "..."} or {}
        if prompt_control.get("append_prompt"):
            prompt = prompt + "\\n\\n" + prompt_control["append_prompt"]
        # 2. LLM interaction
        commands, is_task_complete, feedback, analysis, plan, llm_response = \\
            await self._handle_llm_interaction(chat, prompt, ...)
        # 3. Execute validated commands in tmux
        timeout_occurred, terminal_output = await self._execute_commands(commands, session)
        if timeout_occurred:
            self._run_hook("on_timeout", last_cmd, terminal_output, context)
        else:
            self._run_hook("after_execute", terminal_output, context)
        # 4. Build the base observation for the next LLM call.
        if is_task_complete and not self._pending_completion:
            observation = completion_confirmation_msg(terminal_output)
        else:
            observation = terminal_output
        context.next_observation = observation
        # 5. after_round hook — can append guidance after context.next_observation
        control = await self._run_after_round_hook(terminal_output, is_task_complete)
        # Hook guidance is appended after the base observation; terminal_output is immutable.
        next_prompt = observation
        if control.get("next_prompt"):
            next_prompt = observation + "\\n\\n" + control["next_prompt"]
        # 6. Completion check (two-step confirmation)
        if is_task_complete:
            if self._pending_completion: return
            else: self._pending_completion = True; prompt = next_prompt; continue
        prompt = next_prompt
```

Key facts:
- Commands execute serially in tmux; context.kv persists within a trial.
- In `before_llm_call`, last_* fields refer to the previous completed round.
- `before_llm_call` can return `{"append_prompt": "..."}` for initial or
  periodic guidance; the runtime appends that text after the existing prompt.
- In `after_execute`, `on_timeout`, and `after_round`, last_* fields refer
  to the current round.
- `after_round` sees `context.next_observation`, the base message that will be
  shown to the agent next before hook guidance is appended.

**CRITICAL sandbox rules for hooks:**
- Hooks run in an isolated namespace with primitive inputs plus `HookContext`.
- Use `context.kv` for cross-round state and `next_prompt` for prompt guidance.
- Only allowed imports: re, json, math, collections, itertools, functools,
  copy, textwrap, string.
- All hooks are tested with mock data before deployment. If a hook raises
  any exception during testing, it will be rejected.
"""

_STRATEGY_IMPROVEMENT_STRATEGIES = """\
## Strategy improvement guide

Generate long-lived Terminal-Bench skills, not runtime recovery logic.
Each strategy should have:
- `pattern`: a reusable task situation or failure-prone workflow.
- `steps`: concrete terminal actions, ordered as inspect -> modify -> verify.
- `scope`: one of `core`, `general`, or `task_specific`.
- `triggers`: 2-6 short lowercase keywords/artifacts that should select it.
- `anti_triggers`: optional keywords/artifacts that should suppress it.

Do not write strategies that depend on seeing a specific terminal output in the
current round; that belongs in hooks. Keep strategy text operational and avoid
generic reminders like "be careful".
Avoid hard-coded paths, filenames, ports, exact answers, or task IDs in all
strategies; use placeholders like `<target_file>`, `<log_dir>`, and `<port>`.

Use `core` rarely for broadly useful terminal discipline that should appear on
nearly every task. Use `general` for reusable workflows with clear triggers.
Use `task_specific` when the strategy mentions concrete files, paths, formats,
tools, or task families; it should only inject when the current task matches.

High-value strategy patterns:
1. Path discovery: before reading/writing files, run `pwd`, inspect likely
   parent directories, and use targeted `find`/`ls` instead of assuming paths.
2. File creation and shell writing: use reliable `cat <<'EOF'`, `printf`, or
   small scripts; quote intentionally; verify the written file content.
3. Bounded search: avoid `find /` or broad recursive grep; start from likely
   roots, exact names, and narrowed patterns.
4. Permissions/user context: check `id`, ownership, mode, and writable targets
   before changing permissions or switching users.
5. Large/noisy output: inspect with `grep`, `head`, `tail`, exact paths, and
   concise summaries instead of dumping full logs repeatedly.
6. Deliverable-first workflow: identify required output paths/formats, make the
   minimal change, then run one targeted verification.
7. Completion criteria: before `task_complete`, confirm required artifacts
   exist, content/format match the task, and relevant tests/checks pass.
"""

_HOOK_IMPROVEMENT_STRATEGIES = """\
## Hook improvement guide

Only apply a hook shape when trace evidence supports it. Prefer one small,
robust hook over broad generic advice.
Avoid hard-coding task-specific filenames, section names, exact expected
outputs, or commands into a global hook unless that detail is present in the
diagnosis/trace and the hook gates on `context.original_instruction`,
`terminal_output`, or a sibling `context.kv` signal. Prefer reusable failure
patterns over one-task checklists. Any appended prompt should be concise and
trigger only under a clear condition, not every round.

### 1. Initial or periodic task discipline → `prompt_guidance`
Use `before_llm_call` for broad recurring behavior before the LLM acts: the
agent starts before identifying deliverables, forgets required format, or drifts
into low-value checks. Return `{"append_prompt": "..."}` or `{}`.
- At `context.episode == 0`, add one short reminder to identify required files,
  expected output format, and a minimal verification plan.
- For long trials, add a short reminder every 6-8 rounds only when useful; use
  `context.kv` to avoid duplicates.
- Keep it terminal-specific and operational. Avoid generic text like "be
  careful". Do not use this for immediate recovery from a concrete terminal
  error.

### 2. Concrete terminal errors → `runtime_recovery`
Use sibling hooks when the trace shows real terminal failures. `after_execute`
or `on_timeout` records compact `context.kv` signals; `after_round` consumes the
signal once with a short `next_prompt`.
- Missing path/wrong directory: set `error_kind="missing_path"` and
  `last_failed_command`; prompt to inspect parent/current directory or use a
  targeted `find`.
- Permission/user-context: set `error_kind="permission"`; prompt to check
  owner/mode, choose a writable target, or use the required user only if the
  task evidence supports it.
- Shell/heredoc/quoting: set `error_kind="shell_syntax"`; prompt to simplify
  quoting, use a plain heredoc, or split the command.
- Timeout/broad search: in `on_timeout`, set `timeout_kind` to `broad_search`,
  `install_or_network`, or `long_command`; prompt to narrow path/pattern or
  split the command.

### 3. Loops, noisy output, and context pressure → `after_round`
If traces show repeated `cat`/`ls`/`grep`, huge output, or command patterns
without progress, track a bounded history in `context.kv`. When repetition is
clear, append one prompt to stop re-checking the same artifact, use targeted
`grep`/`tail`/`head`, narrow the path, or take the next constructive action.

### 4. Premature completion or partial artifacts → conservative `after_round`
If traces show `task_complete` before required files/content exist, inspect
`context.next_observation` so guidance fits after the normal completion
confirmation. Use a one-shot `context.kv` key and append concise guidance only
for a concrete missing/uncertain deliverable that is evidenced by the current
task or terminal output. Avoid routine or periodic verification reminders;
repeated verification often hurts terminal-task performance.
"""

_SYSTEM_PROMPT_BASE = """\
You are an expert AI-agent architect. Given a diagnosis and sample traces,
generate targeted fixes to improve agent behaviour.

Two modification tiers:

### Tier 1: Strategy library edits (prompt-level)
Add reusable strategies to the agent's strategy library.
Each strategy has a `pattern` (when to apply), `steps` (what to do), and
optional selection metadata (`scope`, `triggers`, `anti_triggers`) used to
choose which strategies are injected for a task.
The agent sees these as [LEARNED STRATEGIES] in its prompt.
Do not edit or remove existing strategies here; a later curator handles
deduplication, merging, deletion, and library-size control.

### Tier 2: Code hooks (code-level)
Python functions injected into the agent loop at specific hook points.

{override_fields_doc}

{agent_code_context}

{improvement_strategies}

## Rules

1. Include all high-quality fixes that are strongly supported by diagnosis evidence.
   Do not drop good fixes only because there are multiple valid edits.
2. Only generate hooks from the allowed hook points for the current prompt.
   When a hook category contains sibling hooks, you may generate multiple hooks
   from that category as one coherent fix when the trace evidence supports it.
   `strategy_library` is always allowed — you can ALWAYS add strategy_edits.
3. When `strategy_suggestions` are provided by the diagnoser, you MUST add
   them to the strategy library via `strategy_edits` with action `add`, adding
   selection metadata (`scope`, `triggers`, optional `anti_triggers`).
4. Be conservative — change the minimum necessary.
5. Ground choices in trace evidence (mention trajectory IDs, failure modes).
6. For `code_hooks`: write a complete function named `hook` with the correct
   signature. Only use allowed imports. Write defensive code with fallbacks.
   Always initialize `context.kv` keys with setdefault before using them.
7. For `after_round` hooks: return a dict with optional keys:
   `next_prompt`, `prompt_mode` ("append" only in runtime).
   Runtime treats `next_prompt` as prompt guidance after
   `context.next_observation`. Use `context.kv` to track state across rounds
   (e.g. counters/history).
8. Match the strategy to the diagnosis and trace evidence.
9. For `strategy_edits`: only use action `add` for new reusable patterns.
   Do not emit `edit` or `remove`; strategy curation runs later.
10. Use concrete values or plain placeholder words in strategy text, avoiding
   `${VAR}`-style placeholders that can break runtime config interpolation.

## Output format

Return a JSON object:

```json
{{
  "strategy_edits": [
    {{"action": "add", "strategy": {{"pattern": "...", "steps": ["..."], "scope": "general", "triggers": ["..."]}}}}
  ],
  "code_hooks": {{
    "<hook_point>": "<Python source>"
  }},
  "rationale": "<brief explanation>"
}}
```

- `strategy_edits`: ordered list of add operations on the strategy library.
  Each `add` needs a `strategy` with `pattern` and `steps`.
- `code_hooks`: Python hook functions keyed by hook point name. Valid keys:
  before_llm_call, after_execute, on_timeout, after_round.
"""

_MODULE_OVERRIDE_FILE: dict[str, str] = {
    "strategy_library": "strategy_library.yaml",
}

_HOOK_CATEGORIES: dict[str, tuple[str, ...]] = {
    # Standalone prompt guidance. Kept separate from runtime observers so it
    # does not need to share transient terminal-state signals.
    "prompt_guidance": ("before_llm_call",),
    # Observer hooks and the round controller share one generated hook group,
    # so after_execute/on_timeout can write context.kv signals that after_round
    # can consume in the same round.
    "runtime_recovery": ("after_execute", "on_timeout", "after_round"),
}
_ALL_HOOK_POINTS: tuple[str, ...] = tuple(
    hp for group in _HOOK_CATEGORIES.values() for hp in group
)
_MAX_TRACES_FOR_PROMPT = 8
_STRATEGY_SUBAGENT_MAX_TOKENS = 3072
_HOOK_SUBAGENT_MAX_TOKENS = 2048
_JSON_RETRY_HINT_STRATEGY = (
    "Your previous response could not be parsed as strict JSON.\n"
    "Retry now and return ONLY one compact JSON object with keys:\n"
    "  strategy_edits, rationale\n"
    "Requirements:\n"
    "- No markdown fences, no explanations, no <think> tags.\n"
    "- strategy_edits may only contain action `add`.\n"
    "- Each strategy must include pattern, steps, scope, and triggers.\n"
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


def _normalize_strategy_edit(strategy: dict[str, Any]) -> dict[str, Any] | None:
    pattern = str(strategy.get("pattern", "")).strip()
    steps_raw = strategy.get("steps", [])
    if not pattern or not isinstance(steps_raw, list):
        return None
    steps = [str(step).strip() for step in steps_raw if str(step).strip()]
    if not steps:
        return None

    normalized: dict[str, Any] = {
        "pattern": pattern,
        "steps": steps,
    }
    scope = str(strategy.get("scope", "")).lower().strip()
    if scope in {"core", "general", "task_specific"}:
        normalized["scope"] = scope

    for key in ("triggers", "anti_triggers"):
        terms = _normalize_strategy_terms(strategy.get(key))
        if terms:
            normalized[key] = terms
    return normalized


def _normalize_strategy_terms(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_terms = [part.strip() for part in value.split(",")]
    elif isinstance(value, list):
        raw_terms = [str(part).strip() for part in value]
    else:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        term = term.lower().strip()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term[:80])
        if len(terms) >= 12:
            break
    return terms


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

        user_msg = (
            f"## Diagnoses\n{diag_text}\n\n"
            f"## Sample traces (failures prioritised)\n{trace_text}"
        )

        system = _SYSTEM_PROMPT_BASE.format(
            override_fields_doc=_OVERRIDE_FIELDS_DOC,
            agent_code_context=_TERMINUS2_CODE_CONTEXT,
            improvement_strategies=(
                f"{_STRATEGY_IMPROVEMENT_STRATEGIES}\n\n"
                f"{_HOOK_IMPROVEMENT_STRATEGIES}"
            ),
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def _rank_diagnoses(
        diagnoses: list[DiagnosisResult],
    ) -> list[DiagnosisResult]:
        """Sort diagnoses by confidence descending."""
        return sorted(diagnoses, key=lambda d: d.confidence, reverse=True)

    def _build_user_content(
        self,
        diagnoses: list[DiagnosisResult],
        traces: list[TraceRecord],
        *,
        include_strategy_suggestions: bool,
        diagnosis_heading: str = "## Diagnoses",
        max_traces: int = _MAX_TRACES_FOR_PROMPT,
        include_traces: bool = True,
    ) -> str:
        """Build compact planning user content."""
        diag_text = self._format_diagnoses_limited(
            diagnoses,
            include_strategy_suggestions=include_strategy_suggestions,
        )
        user_content = f"{diagnosis_heading}\n{diag_text}"
        if include_traces:
            failed_first = [t for t in traces if not t.success][:max_traces]
            trace_text = _serialise_traces_compact(
                failed_first, max_traces=max_traces,
            ) or _serialise_traces_compact(
                traces[:max_traces], max_traces=max_traces,
            )
            user_content += f"\n\n## Sample traces (failures prioritised)\n{trace_text}"
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
            "before_llm_call": (
                "def hook(prompt, context):\n"
                "    # prompt: str, context: HookContext\n"
                "    # Return {'append_prompt': '<short guidance>'} or {}"
            ),
            "after_execute": "def hook(terminal_output, context):\n    # terminal_output: str, context: HookContext\n    # Observer hook; update context.kv and return None",
            "on_timeout": "def hook(command_keystrokes, terminal_output, context):\n    # Observer hook; update context.kv and return None",
            "after_round": (
                "def hook(terminal_output, is_task_complete, context):\n"
                "    # Return dict with optional keys:\n"
                "    #   next_prompt: str\n"
                "    #   prompt_mode: 'append'\n"
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
                            "  or context.kv.get('key', default_value) before reading keys.\n"
                            "- Only allowed imports: re, json, math, collections, "
                            "itertools, functools, copy, textwrap, string.\n"
                            "- before_llm_call returns a dict with optional key "
                            "{append_prompt}; return {} when no guidance is needed.\n"
                            "- after_execute/on_timeout are observer hooks: record signals in context.kv and return None.\n"
                            "- For after_round: return a dict with optional keys "
                            "{next_prompt, prompt_mode}; "
                            "next_prompt is prompt guidance appended after context.next_observation.\n"
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
        strategy_diagnoses = ranked_diagnoses
        strategy_user_content = self._build_user_content(
            strategy_diagnoses,
            traces,
            include_strategy_suggestions=True,
            diagnosis_heading="## Diagnoses",
            max_traces=_MAX_TRACES_FOR_PROMPT,
            include_traces=False,
        )
        strategy_msgs = self._build_strategy_only_messages(strategy_user_content)

        hook_payloads: dict[str, tuple[list[str], list[dict[str, str]], set[str]]] = {}
        for category, category_hooks in _HOOK_CATEGORIES.items():
            allowed_hooks = list(category_hooks)
            hook_diagnoses = ranked_diagnoses
            hook_user_content = self._build_user_content(
                hook_diagnoses,
                traces,
                include_strategy_suggestions=False,
                diagnosis_heading=f"## Diagnoses ({category})",
                max_traces=_MAX_TRACES_FOR_PROMPT,
                include_traces=False,
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
            "You are an expert AI-agent architect. Based on the diagnoses, "
            "generate ONLY new strategy_edits for the agent's strategy library.\n"
            "Return strategy_edits and rationale only.\n\n"
            "Only use action `add`. Do not edit or remove existing strategies; "
            "a later curator handles deduplication, merging, deletion, and budget control.\n"
            "Use concrete shell snippets or explicit placeholder words in strategy text.\n"
            "Each strategy should include `scope`, `triggers`, and optional "
            "`anti_triggers` so runtime can select relevant strategies per task.\n\n"
            "Output must be STRICT JSON only (no markdown fences / no <think> blocks).\n"
            "Keep output compact and high-impact to avoid truncation.\n\n"
            f"{_STRATEGY_IMPROVEMENT_STRATEGIES}\n\n"
            "## Output format\n\n"
            "Return ONLY this JSON object shape:\n"
            "{\n"
            '  "strategy_edits": [\n'
            '    {"action": "add", "strategy": {"pattern": "...", "steps": ["..."], '
            '"scope": "general", "triggers": ["..."], "anti_triggers": ["..."]}}\n'
            "  ],\n"
            '  "rationale": "<brief explanation>"\n'
            "}\n"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _build_hook_only_messages(self, hook_name: str, user_content: str) -> list[dict[str, str]]:
        """System prompt focused on generating a single hook."""
        system = (
            f"You are an expert AI-agent architect. Generate ONLY a code hook for "
            f"the **{hook_name}** hook point. Return code_hooks only.\n\n"
            "Available effects: before_llm_call returns append_prompt for initial or periodic short guidance; "
            "after_execute/on_timeout record signals in context.kv; "
            "after_round sees context.next_observation and appends prompt guidance "
            "after that base observation.\n\n"
            f"{_OVERRIDE_FIELDS_DOC}\n\n"
            f"{_TERMINUS2_CODE_CONTEXT}\n\n"
            f"{_HOOK_IMPROVEMENT_STRATEGIES}\n\n"
            "## Output format\n\n"
            "Return ONLY this JSON object shape:\n"
            "{\n"
            f'  "code_hooks": {{"{hook_name}": "<Python source>"}},\n'
            '  "rationale": "<brief explanation>"\n'
            "}\n"
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
            "Return code_hooks only.\n\n"
            "Hook interfaces:\n"
            "- before_llm_call(prompt: str, context) -> dict; runs immediately before each LLM call and may return append_prompt.\n"
            "- after_execute(terminal_output: str, context) -> None; runs after a command round completes successfully.\n"
            "- on_timeout(command_keystrokes: str, terminal_output: str, context) -> None; runs after a command timeout.\n"
            "- after_round(terminal_output: str, is_task_complete: bool, context) -> dict; runs after the base next observation is built.\n\n"
            "Output must be STRICT JSON only (no markdown fences / no <think> blocks).\n"
            "Keep output compact and robust.\n\n"
            f"{_OVERRIDE_FIELDS_DOC}\n\n"
            f"{_TERMINUS2_CODE_CONTEXT}\n\n"
            f"{_HOOK_IMPROVEMENT_STRATEGIES}\n\n"
            "## Output format\n\n"
            "Return ONLY this JSON object shape:\n"
            "{\n"
            '  "code_hooks": {\n'
            f'    "{hook_names[0]}": "<Python source>"\n'
            "  },\n"
            '  "rationale": "<brief explanation>"\n'
            "}\n"
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

        # Strategy library additions. Editing/removal is handled later by PatchPruner.
        if strategy_edits:
            valid_edits = []
            for edit in strategy_edits:
                if not isinstance(edit, dict):
                    continue
                action = edit.get("action", "")
                if action == "add" and isinstance(edit.get("strategy"), dict):
                    strategy = _normalize_strategy_edit(edit["strategy"])
                    if strategy is not None:
                        clean_edit = dict(edit)
                        clean_edit["strategy"] = strategy
                        valid_edits.append(clean_edit)

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
        max_hypotheses_per_diagnosis: int | None = None,
        max_tasks_per_diagnosis: int | None = None,
        max_strategy_suggestions: int | None = None,
        max_steps_per_suggestion: int | None = None,
    ) -> str:
        parts: list[str] = []
        all_suggestions: list[dict] = []
        for d in diagnoses:
            hypotheses_source = (
                d.root_cause_hypotheses
                if max_hypotheses_per_diagnosis is None
                else d.root_cause_hypotheses[:max_hypotheses_per_diagnosis]
            )
            affected_source = (
                d.affected_task_ids
                if max_tasks_per_diagnosis is None
                else d.affected_task_ids[:max_tasks_per_diagnosis]
            )
            hypotheses = [str(h) for h in hypotheses_source]
            block = (
                f"- **{d.problem_type}** (conf={d.confidence:.2f})\n"
                f"  Hypotheses: {'; '.join(hypotheses)}\n"
                f"  Affected traces: {affected_source}"
            )
            confidence_policy = str(d.metadata.get("confidence_policy", ""))
            merged_confidences = d.metadata.get("merged_confidences")
            if confidence_policy or merged_confidences:
                block += (
                    f"\n  Confidence merge: {confidence_policy or 'n/a'}"
                    f" from {merged_confidences or []}"
                )
            summary = str(d.metadata.get("analysis_summary", ""))
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
                "`strategy_edits` with action `add`, adding `scope`, `triggers`, "
                "and optional `anti_triggers` for runtime selection.\n\n"
            )
            shown = (
                all_suggestions
                if max_strategy_suggestions is None
                else all_suggestions[:max_strategy_suggestions]
            )
            for i, s in enumerate(shown):
                steps = s.get("steps", [])
                if isinstance(steps, list) and max_steps_per_suggestion is not None:
                    steps = steps[:max_steps_per_suggestion]
                result += (
                    f"{i+1}. **Pattern**: {s.get('pattern', '?')}\n"
                    f"   **Steps**: {steps}\n"
                    f"   **Source**: {s.get('source', 'unknown')}\n"
                )
            if (
                max_strategy_suggestions is not None
                and len(all_suggestions) > max_strategy_suggestions
            ):
                result += (
                    f"... ({len(all_suggestions) - max_strategy_suggestions} more "
                    "strategy suggestions omitted)\n"
                )
        return result
