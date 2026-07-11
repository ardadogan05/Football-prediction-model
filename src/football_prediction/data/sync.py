"""Synchronize the local immutable StatsBomb Open Data checkout."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from football_prediction.config import STATSBOMB_REPOSITORY_URL


class SynchronizationError(RuntimeError):
    """Raised when the StatsBomb Git checkout cannot be synchronized."""


@dataclass(frozen=True)
class SyncResult:
    repository: Path
    commit: str
    changed: bool
    action: str


def _run_git(arguments: list[str], cwd: Path | None = None) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise SynchronizationError("Git is required for update-data but was not found")
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            [executable, "-c", "credential.helper=", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            env=environment,
            timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        stderr = getattr(error, "stderr", "") or ""
        raise SynchronizationError(f"Git synchronization failed: {stderr.strip()}") from error
    return result.stdout.strip()


def _commit(repository: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd=repository)


def sync_repository(
    repository: Path,
    repository_url: str = STATSBOMB_REPOSITORY_URL,
) -> SyncResult:
    """Clone once, then fast-forward the existing checkout on later updates."""

    repository = repository.resolve()
    git_directory = repository / ".git"
    if git_directory.is_dir():
        before = _commit(repository)
        _run_git(["pull", "--ff-only"], cwd=repository)
        after = _commit(repository)
        return SyncResult(repository, after, before != after, "updated")

    if repository.exists() and any(repository.iterdir()):
        raise SynchronizationError(
            f"Raw data path exists but is not a Git checkout: {repository}"
        )

    repository.parent.mkdir(parents=True, exist_ok=True)
    if repository.exists():
        repository.rmdir()
    temporary = repository.with_name(f"{repository.name}.clone-tmp")
    if temporary.exists():
        raise SynchronizationError(
            f"Temporary clone path already exists; inspect or remove it: {temporary}"
        )
    try:
        _run_git(["clone", "--depth", "1", repository_url, str(temporary)])
        temporary.replace(repository)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return SyncResult(repository, _commit(repository), True, "cloned")
