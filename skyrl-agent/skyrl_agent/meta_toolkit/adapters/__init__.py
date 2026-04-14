"""Agent adapters for the meta-learning system.

Each adapter tells the meta-learning loop how to inject overrides into a
specific agent type (Terminus-2, generic installed agents, etc.).
"""

from .base import MetaOverrideAdapter
from .terminus2_adapter import Terminus2Adapter
from .generic_instruction_adapter import GenericInstructionAdapter

__all__ = [
    "MetaOverrideAdapter",
    "Terminus2Adapter",
    "GenericInstructionAdapter",
]
