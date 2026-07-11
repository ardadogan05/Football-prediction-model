"""Build pre-match goal features without using future matches."""

from __future__ import annotations

import math

import pandas as pd


FEATURE_COLUMNS = [
    "home_rolling_goals_for",
    "home_rolling_goals_against",
    "away_rolling_goals_for",
    "away_rolling_goals_against",
    "home_season_goals_for",
    "home_season_goals_against",
    "away_season_goals_for",
    "away_season_goals_against",
    "home_history_matches",
    "away_history_matches",
    "home_season_matches",
    "away_season_matches",
]

OUTPUT_COLUMNS = [
    "source",
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
    "feature_supported",
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


def _season_matches(totals: dict, key: tuple) -> int:
    values = totals.get(key)
    if values is None:
        return 0
    return int(values["matches"])


def _add_season_goals(totals: dict, key: tuple, goals: float) -> None:
    """Add one earlier match to a team's season totals."""

    if key not in totals:
        totals[key] = {"total": 0.0, "matches": 0}
    totals[key]["total"] += goals
    totals[key]["matches"] += 1


def _competition_average(totals: dict, competition_key: tuple) -> float:
    values = totals.get(competition_key)
    if values is None or values["teams"] == 0:
        return math.nan
    return values["goals"] / values["teams"]


def build_features(matches: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """Create one feature row per eligible match using earlier dates only."""

    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")

    required = {
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
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Matches are missing feature columns: {missing}")

    data = matches.copy()
    if "source" not in data.columns:
        data["source"] = "statsbomb"
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # These dictionaries only contain information from earlier match dates.
    team_histories = {}
    season_goals_for_totals = {}
    season_goals_against_totals = {}
    competition_goal_totals = {}
    feature_rows = []

    # Calculate every match on a date before adding that date to the history.
    # This stops one same-day match from affecting another same-day match.
    for _, date_matches in data.groupby("match_date", sort=True):
        pending_updates = []

        for match in date_matches.itertuples(index=False):
            source = str(match.source)
            competition_id = int(match.competition_id)
            season_id = int(match.season_id)
            home_id = int(match.home_team_id)
            away_id = int(match.away_team_id)
            competition_key = (source, competition_id)
            # New teams use the competition's earlier goal average as a fallback.
            competition_fallback = _competition_average(
                competition_goal_totals, competition_key
            )

            # Include the source so equal IDs from two providers never mix teams.
            home_key = (source, competition_id, home_id)
            away_key = (source, competition_id, away_id)
            home_history = team_histories.get(home_key, {"for": [], "against": []})
            away_history = team_histories.get(away_key, {"for": [], "against": []})
            home_season_key = (source, competition_id, season_id, home_id)
            away_season_key = (source, competition_id, season_id, away_id)

            row = {
                "source": source,
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
                "home_rolling_goals_for": _mean(
                    home_history["for"][-rolling_window:], competition_fallback
                ),
                "home_rolling_goals_against": _mean(
                    home_history["against"][-rolling_window:], competition_fallback
                ),
                "away_rolling_goals_for": _mean(
                    away_history["for"][-rolling_window:], competition_fallback
                ),
                "away_rolling_goals_against": _mean(
                    away_history["against"][-rolling_window:], competition_fallback
                ),
                "home_season_goals_for": _season_average(
                    season_goals_for_totals, home_season_key, competition_fallback
                ),
                "home_season_goals_against": _season_average(
                    season_goals_against_totals, home_season_key, competition_fallback
                ),
                "away_season_goals_for": _season_average(
                    season_goals_for_totals, away_season_key, competition_fallback
                ),
                "away_season_goals_against": _season_average(
                    season_goals_against_totals, away_season_key, competition_fallback
                ),
                "home_history_matches": len(home_history["for"]),
                "away_history_matches": len(away_history["for"]),
                "home_season_matches": _season_matches(
                    season_goals_for_totals, home_season_key
                ),
                "away_season_matches": _season_matches(
                    season_goals_for_totals, away_season_key
                ),
            }

            # The model can only use a row when every feature is a real number.
            row["feature_supported"] = True
            for column in FEATURE_COLUMNS:
                if not math.isfinite(float(row[column])):
                    row["feature_supported"] = False
                    break
            feature_rows.append(row)
            pending_updates.append(match)

        # Only now is this date added to the histories used by later dates.
        for match in pending_updates:
            source = str(match.source)
            competition_id = int(match.competition_id)
            season_id = int(match.season_id)
            home_id = int(match.home_team_id)
            away_id = int(match.away_team_id)
            home_goals = float(match.home_goals)
            away_goals = float(match.away_goals)
            competition_key = (source, competition_id)

            home_key = (source, competition_id, home_id)
            away_key = (source, competition_id, away_id)
            if home_key not in team_histories:
                team_histories[home_key] = {"for": [], "against": []}
            if away_key not in team_histories:
                team_histories[away_key] = {"for": [], "against": []}

            home_history = team_histories[home_key]
            away_history = team_histories[away_key]
            home_history["for"].append(home_goals)
            home_history["against"].append(away_goals)
            away_history["for"].append(away_goals)
            away_history["against"].append(home_goals)

            home_season_key = (source, competition_id, season_id, home_id)
            away_season_key = (source, competition_id, season_id, away_id)
            _add_season_goals(season_goals_for_totals, home_season_key, home_goals)
            _add_season_goals(season_goals_against_totals, home_season_key, away_goals)
            _add_season_goals(season_goals_for_totals, away_season_key, away_goals)
            _add_season_goals(season_goals_against_totals, away_season_key, home_goals)

            if competition_key not in competition_goal_totals:
                competition_goal_totals[competition_key] = {"goals": 0.0, "teams": 0}
            competition = competition_goal_totals[competition_key]
            competition["goals"] += home_goals + away_goals
            competition["teams"] += 2

    return pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS)
