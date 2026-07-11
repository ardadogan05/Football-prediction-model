"""Canonical Phase 1 match schema and validation."""

from __future__ import annotations

import pandas as pd


MATCH_COLUMNS = (
    "match_id",
    "match_date",
    "kick_off",
    "competition_id",
    "competition_name",
    "season_id",
    "season_name",
    "home_team_id",
    "home_team_name",
    "away_team_id",
    "away_team_name",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "missing_shot_xg_count",
    "eligible_for_model",
    "source_commit",
)


class DataValidationError(ValueError):
    """Raised when source or processed football data violates its contract."""


def require_fields(record: dict, fields: list[str], context: str) -> None:
    missing = []
    for field in fields:
        if field not in record:
            missing.append(field)
    if missing:
        raise DataValidationError(f"{context} is missing required fields: {missing}")


def validate_matches(matches: pd.DataFrame) -> None:
    """Validate the compact match table before it is persisted or loaded."""

    missing_columns = sorted(set(MATCH_COLUMNS) - set(matches.columns))
    if missing_columns:
        raise DataValidationError(
            f"Processed matches are missing columns: {missing_columns}"
        )
    if matches.empty:
        raise DataValidationError("No eligible top-five men's league matches were processed")
    if matches["match_id"].isna().any() or matches["match_id"].duplicated().any():
        raise DataValidationError("match_id must be present and unique")
    if (matches["home_team_id"] == matches["away_team_id"]).any():
        raise DataValidationError("Home and away teams must differ")
    if matches[["home_goals", "away_goals"]].isna().any().any():
        raise DataValidationError("Every processed match must have a regulation-time score")
    if (matches[["home_goals", "away_goals"]] < 0).any().any():
        raise DataValidationError("Goals must be non-negative")
    if matches["match_date"].isna().any():
        raise DataValidationError("Every processed match must have a valid match_date")

    eligible = matches["eligible_for_model"]
    eligible_xg = matches.loc[eligible, ["home_xg", "away_xg"]]
    if eligible_xg.isna().any().any() or (eligible_xg < 0).any().any():
        raise DataValidationError("Model-eligible matches require finite, non-negative xG")
