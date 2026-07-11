"""Small, explicit project configuration for data processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STATSBOMB_REPOSITORY_URL = "https://github.com/statsbomb/open-data.git"

TOP_FIVE_MENS_LEAGUES: dict[str, list[str]] = {
    "England": ["Premier League"],
    "France": ["Ligue 1"],
    "Germany": ["1. Bundesliga", "Bundesliga"],
    "Italy": ["Serie A"],
    "Spain": ["La Liga"],
}


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the Phase 1 data pipeline."""

    root: Path
    raw_repository: Path
    processed_directory: Path
    matches_file: Path
    manifest_file: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        processed = root / "data" / "processed"
        return cls(
            root=root,
            raw_repository=root / "data" / "raw" / "statsbomb-open-data",
            processed_directory=processed,
            matches_file=processed / "matches.parquet",
            manifest_file=processed / "manifest.json",
        )
