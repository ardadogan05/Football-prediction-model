"""Small, explicit project configuration for data processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


STATSBOMB_COMMIT_URL = (
    "https://api.github.com/repos/statsbomb/open-data/commits/master"
)
STATSBOMB_RAW_URL = "https://raw.githubusercontent.com/statsbomb/open-data"

TOP_FIVE_MENS_LEAGUES: dict[str, list[str]] = {
    "England": ["Premier League"],
    "France": ["Ligue 1"],
    "Germany": ["1. Bundesliga", "Bundesliga"],
    "Italy": ["Serie A"],
    "Spain": ["La Liga"],
}


def is_top_five_mens_league(record: dict) -> bool:
    if str(record.get("competition_gender", "")).lower() != "male":
        return False

    country = record.get("country_name")
    if country not in TOP_FIVE_MENS_LEAGUES:
        return False

    competition_name = record.get("competition_name")
    return competition_name in TOP_FIVE_MENS_LEAGUES[country]


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem locations used by the two data pipelines."""

    root: Path
    raw_repository: Path
    processed_directory: Path
    matches_file: Path
    manifest_file: Path
    football_data_raw_directory: Path
    football_data_matches_file: Path
    football_data_manifest_file: Path
    model_file: Path
    backtest_predictions_file: Path
    metrics_file: Path

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
            football_data_raw_directory=root / "data" / "raw" / "football-data",
            football_data_matches_file=processed / "football_data_matches.parquet",
            football_data_manifest_file=processed / "football_data_manifest.json",
            model_file=root / "models" / "model.pkl",
            backtest_predictions_file=root / "reports" / "backtest_predictions.csv",
            metrics_file=root / "reports" / "metrics.json",
        )
