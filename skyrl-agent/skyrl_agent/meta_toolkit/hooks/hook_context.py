"""HookContext: the limited API surface that hook functions can access.

Hook functions receive this object instead of raw agent internals.
This constrains what hooks can do and makes them safe to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookContext:
    """Context passed to every hook invocation.

    All fields are read-only EXCEPT ``kv`` which hooks can freely modify.

    Fields (updated by the agent loop before each hook call):
      episode              - current round number (0-based)
      total_episodes       - configured round safety cap
      n_commands_executed   - total commands run so far
      n_parse_errors       - total LLM parse failures so far
      n_timeouts           - total command timeouts so far
      last_analysis        - most recent parsed LLM analysis text
      last_plan            - most recent parsed LLM plan text
      last_commands        - most recent parsed command strings
      is_task_complete     - most recent parsed task_complete value
      original_instruction - the original task description (read-only)
      last_terminal_output - terminal output from the most recent command execution
      next_observation     - base message that will be shown to the agent next
      execution_history    - list of past round records (capped at 20)
      last_llm_response    - raw LLM response text before parsing
      kv                   - persistent dict for cross-round state (writable)
    """

    episode: int = 0
    total_episodes: int = 0

    n_commands_executed: int = 0
    n_parse_errors: int = 0
    n_timeouts: int = 0

    last_analysis: str = ""
    last_plan: str = ""
    last_commands: list[str] = field(default_factory=list)

    is_task_complete: bool = False
    original_instruction: str = ""

    last_terminal_output: str = ""
    next_observation: str = ""
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    last_llm_response: str = ""

    kv: dict = field(default_factory=dict)

    _MAX_HISTORY: int = field(default=20, repr=False)

    @property
    def round_number(self) -> int:
        """Alias for episode (0-based round counter)."""
        return self.episode

    def record_round(
        self,
        commands: list[str],
        output: str,
        episode: int,
    ) -> None:
        """Append a round to execution_history, keeping only the last N entries."""
        self.execution_history.append({
            "commands": list(commands),
            "output": output,
            "episode": episode,
        })
        if len(self.execution_history) > self._MAX_HISTORY:
            self.execution_history[:] = self.execution_history[-self._MAX_HISTORY:]
