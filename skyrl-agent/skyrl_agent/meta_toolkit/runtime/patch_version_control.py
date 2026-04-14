"""Git-based version control for meta-learning patch overrides.

Maintains a **separate** git repository inside the override_base directory
(e.g. meta_patches/terminus2/) so that every accepted patch is versioned
without polluting the parent SkyRL repo.

Capabilities:
  - Auto-init a git repo if one does not exist
  - Commit after each accepted patch with structured metadata
  - List history of all accepted patches
  - Rollback to any previous commit (by SHA or relative offset)
  - Tag known-good snapshots
  - Snapshot/restore for safe candidate evaluation with git rollback
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class PatchCommitInfo:
    """Metadata for a single version-controlled patch commit."""

    sha: str
    message: str
    timestamp: str
    author: str = ""


@dataclass
class PatchVersionControlConfig:
    """Configuration for PatchVersionControl."""

    override_base: str = ""
    author_name: str = "skyrl-meta"
    author_email: str = "meta@skyrl.local"


class PatchVersionControl:
    """Manage patch override files in an isolated git repository."""

    def __init__(self, config: PatchVersionControlConfig) -> None:
        self._root = Path(config.override_base)
        self._author_name = config.author_name
        self._author_email = config.author_email
        self._root.mkdir(parents=True, exist_ok=True)

        self._ensure_repo()

    # ------------------------------------------------------------------
    # Public API — snapshot / restore (for candidate evaluation)
    # ------------------------------------------------------------------

    def snapshot(self, label: str = "pre-candidate") -> str:
        """Save the current state as a git commit and return its SHA.

        Call this BEFORE applying any candidate patches.  If the candidate
        is later rejected, call ``restore(sha)`` to roll back all files
        (overrides + hooks) atomically.
        """
        self._run(["git", "add", "-A"])
        if not self._has_staged_changes():
            return self.current_sha()
        self._run(["git", "commit", "-m", f"snapshot: {label}"])
        sha = self.current_sha()
        logger.debug(f"PatchVC: snapshot {sha} ({label})")
        return sha

    def restore(self, sha: str) -> bool:
        """Restore ALL tracked files (overrides + hooks) to *sha*.

        Uses ``git checkout <sha> -- .`` so history is preserved.
        """
        try:
            self._run(["git", "checkout", sha, "--", "."])
            self._run(["git", "add", "-A"])
            if self._has_staged_changes():
                self._run(["git", "commit", "-m", f"restore to {sha}"])
            logger.info(f"PatchVC: restored to {sha}")
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(f"PatchVC: restore to {sha} failed — {exc}")
            return False

    # ------------------------------------------------------------------
    # Public API — commit / rollback / history
    # ------------------------------------------------------------------

    def commit_patch(
        self,
        candidate_id: str,
        delta_score: float,
        *,
        cycle: int = 0,
        notes: str = "",
    ) -> Optional[str]:
        """Stage all changes and commit with structured metadata.

        Returns the commit SHA, or None if there was nothing to commit.
        """
        self._run(["git", "add", "-A"])

        if not self._has_staged_changes():
            logger.debug("PatchVC: nothing to commit")
            return None

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        msg_lines = [
            f"meta-patch: {candidate_id}",
            "",
            f"cycle:       {cycle}",
            f"delta_score: {delta_score:+.4f}",
            f"timestamp:   {ts}",
        ]
        if notes:
            msg_lines.append(f"notes:       {notes}")

        msg = "\n".join(msg_lines)
        self._run(["git", "commit", "-m", msg])

        sha = self._run(["git", "rev-parse", "--short", "HEAD"]).strip()
        logger.info(f"PatchVC: committed {sha} for {candidate_id} (delta={delta_score:+.4f})")
        return sha

    def rollback(self, target: str = "HEAD~1") -> bool:
        """Rollback override files to a previous state.

        ``target`` can be a commit SHA, tag, or relative ref like ``HEAD~1``.
        Uses ``git checkout`` to restore files without rewriting history.
        Returns True on success.
        """
        try:
            self._run(["git", "checkout", target, "--", "."])
            self._run(["git", "add", "-A"])
            self._run(["git", "commit", "-m", f"rollback to {target}"])
            logger.info(f"PatchVC: rolled back to {target}")
            return True
        except subprocess.CalledProcessError as exc:
            logger.error(f"PatchVC: rollback failed — {exc}")
            return False

    def tag(self, tag_name: str, message: str = "") -> None:
        """Create an annotated tag at the current HEAD."""
        cmd = ["git", "tag", "-a", tag_name, "-m", message or tag_name]
        self._run(cmd)
        logger.info(f"PatchVC: tagged HEAD as {tag_name}")

    def history(self, n: int = 20) -> list[PatchCommitInfo]:
        """Return the last *n* commits as structured objects."""
        fmt = "%H||%s||%aI||%an"
        out = self._run(
            ["git", "log", f"--max-count={n}", f"--pretty=format:{fmt}"]
        )
        commits: list[PatchCommitInfo] = []
        for line in out.strip().splitlines():
            parts = line.split("||", 3)
            if len(parts) == 4:
                commits.append(PatchCommitInfo(
                    sha=parts[0][:8],
                    message=parts[1],
                    timestamp=parts[2],
                    author=parts[3],
                ))
        return commits

    def current_sha(self) -> str:
        """Return the short SHA of the current HEAD."""
        return self._run(["git", "rev-parse", "--short", "HEAD"]).strip()

    def diff_from(self, ref: str = "HEAD~1") -> str:
        """Return a unified diff from *ref* to HEAD."""
        try:
            return self._run(["git", "diff", ref, "HEAD"])
        except subprocess.CalledProcessError:
            return ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_repo(self) -> None:
        """Initialize a git repo if one does not exist yet."""
        git_dir = self._root / ".git"
        if git_dir.is_dir():
            return

        self._run(["git", "init"])
        self._run(["git", "config", "user.name", self._author_name])
        self._run(["git", "config", "user.email", self._author_email])

        gitignore = self._root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("__pycache__/\n*.pyc\n", encoding="utf-8")

        self._run(["git", "add", "-A"])
        self._run(["git", "commit", "--allow-empty", "-m", "init: meta-patches repo"])
        logger.info(f"PatchVC: initialized git repo at {self._root}")

    def _has_staged_changes(self) -> bool:
        """Return True if there are staged changes ready to commit."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=self._root,
            capture_output=True,
        )
        return result.returncode != 0

    def _run(self, cmd: list[str]) -> str:
        """Run a git command inside the override_base directory."""
        result = subprocess.run(
            cmd,
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=30,
            env={
                "GIT_AUTHOR_NAME": self._author_name,
                "GIT_AUTHOR_EMAIL": self._author_email,
                "GIT_COMMITTER_NAME": self._author_name,
                "GIT_COMMITTER_EMAIL": self._author_email,
                "HOME": str(Path.home()),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd, result.stdout, result.stderr,
            )
        return result.stdout

