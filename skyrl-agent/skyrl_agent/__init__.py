"""Top-level package exports for skyrl_agent.

Keep imports lightweight so utility subpackages such as meta_toolkit can be used
without eagerly importing the full agent runtime dependency chain.
"""

__all__ = ["AutoAgentRunner"]


def __getattr__(name: str):
    if name == "AutoAgentRunner":
        from .auto import AutoAgentRunner

        return AutoAgentRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
