"""Build pre-match xG features without using future matches."""

from __future__ import annotations

import math

import pandas as pd


FEATURE_COLUMNS = [
    "home_rolling_xg_for",
    "home_rolling_xg_against",
    "away_rolling_xg_for",
    "away_rolling_xg_against",
    "home_season_xg_for",
    "home_season_xg_against",
    "away_season_xg_for",
    "away_season_xg_against",
    "home_history_matches",
    "away_history_matches",
]

OUTPUT_COLUMNS = [
    "match_id",
    "match_date",
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
    *FEATURE_COLUMNS,
]


def _mean(values: list[float], fallback: float) -> float:
    if not values:
        return fallback
    return sum(values) / len(values)


def _season_average(totals: dict, key: tuple, fallback: float) -> float:
    values = totals.get(key)
    if values is None or values["matches"] == 0:
        return fallback
    return values["total"] / values["matches"]


def _competition_average(totals: dict, competition_id: int) -> float:
    values = totals.get(competition_id)
    if values is None or values["teams"] == 0:
        return math.nan
    return values["xg"] / values["teams"]


def build_features(matches: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Create one feature row per eligible match using earlier dates only."""

    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")

    required = {
        "match_id",
        "match_date",
        "competition_id",
        "season_id",
        "home_team_id",
        "away_team_id",
        "home_xg",
        "away_xg",
        "home_goals",
        "away_goals",
        "eligible_for_model",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Matches are missing feature columns: {missing}")

    data = matches.loc[matches["eligible_for_model"]].copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    histories = {}
    season_for = {}
    season_against = {}
    competition_totals = {}
    feature_rows = []

    #all matches on the same date are calculated before that date updates the history.
    for _, date_matches in data.groupby("match_date", sort=True):
        pending_updates = []

        for match in date_matches.itertuples(index=False):
            competition_id = int(match.competition_id)
            season_id = int(match.season_id)
            home_id = int(match.home_team_id)
            away_id = int(match.away_team_id)
            competition_fallback = _competition_average(
                competition_totals, competition_id
            )

            home_key = (competition_id, home_id)
            away_key = (competition_id, away_id)
            home_history = histories.get(home_key, {"for": [], "against": []})
            away_history = histories.get(away_key, {"for": [], "against": []})
            home_season_key = (competition_id, season_id, home_id)
            away_season_key = (competition_id, season_id, away_id)

            row = {
                "match_id": match.match_id,
                "match_date": match.match_date,
                "competition_id": competition_id,
                "competition_name": match.competition_name,
                "season_id": season_id,
                "season_name": match.season_name,
                "home_team_id": home_id,
                "home_team_name": match.home_team_name,
                "away_team_id": away_id,
                "away_team_name": match.away_team_name,
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
                "home_rolling_xg_for": _mean(
                    home_history["for"][-rolling_window:], competition_fallback
                ),
                "home_rolling_xg_against": _mean(
                    home_history["against"][-rolling_window:], competition_fallback
                ),
                "away_rolling_xg_for": _mean(
                    away_history["for"][-rolling_window:], competition_fallback
                ),
                "away_rolling_xg_against": _mean(
                    away_history["against"][-rolling_window:], competition_fallback
                ),
                "home_season_xg_for": _season_average(
                    season_for, home_season_key, competition_fallback
                ),
                "home_season_xg_against": _season_average(
                    season_against, home_season_key, competition_fallback
                ),
                "away_season_xg_for": _season_average(
                    season_for, away_season_key, competition_fallback
                ),
                "away_season_xg_against": _season_average(
                    season_against, away_season_key, competition_fallback
                ),
                "home_history_matches": len(home_history["for"]),
                "away_history_matches": len(away_history["for"]),
            }
            feature_rows.append(row)
            pending_updates.append(match)

        for match in pending_updates:
            competition_id = int(match.competition_id)
            season_id = int(match.season_id)
            home_id = int(match.home_team_id)
            away_id = int(match.away_team_id)
            home_xg = float(match.home_xg)
            away_xg = float(match.away_xg)

            home_history = histories.setdefault(
                (competition_id, home_id), {"for": [], "against": []}
            )
            away_history = histories.setdefault(
                (competition_id, away_id), {"for": [], "against": []}
            )
            home_history["for"].append(home_xg)
            home_history["against"].append(away_xg)
            away_history["for"].append(away_xg)
            away_history["against"].append(home_xg)

            home_season_key = (competition_id, season_id, home_id)
            away_season_key = (competition_id, season_id, away_id)
            for totals, key, value in [
                (season_for, home_season_key, home_xg),
                (season_against, home_season_key, away_xg),
                (season_for, away_season_key, away_xg),
                (season_against, away_season_key, home_xg),
            ]:
                current = totals.setdefault(key, {"total": 0.0, "matches": 0})
                current["total"] += value
                current["matches"] += 1

            competition = competition_totals.setdefault(
                competition_id, {"xg": 0.0, "teams": 0}
            )
            competition["xg"] += home_xg + away_xg
            competition["teams"] += 2

    return pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS)
