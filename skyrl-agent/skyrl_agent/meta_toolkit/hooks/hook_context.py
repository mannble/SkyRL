"""HookContext: the limited API surface that hook functions can access.

Hook functions receive this object instead of raw agent internals.
This constrains what hooks can do and makes them safe to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HookContext:
    """Context passed to every hook invocation.

    All fields are read-only EXCEPT `kv` which hooks can freely modify.

    Fields (updated by the agent loop before each hook call):
      episode            - current round number (0-based)
      total_episodes     - max rounds allowed
      n_commands_executed - total commands run so far
      n_parse_errors     - total LLM parse failures so far
      n_timeouts         - total command timeouts so far
      last_analysis      - LLM's analysis text from this round
      last_plan          - LLM's plan text from this round
      last_commands      - list of command strings from this round
      is_task_complete   - whether the LLM marked task_complete this round
      original_instruction - the original task description (read-only)
      kv                 - persistent dict for cross-round state (writable)
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

    kv: dict = field(default_factory=dict)

    @property
    def round_number(self) -> int:
        """Alias for episode (0-based round counter)."""
        return self.episode
