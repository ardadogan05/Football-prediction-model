# Build pre-match features without using the current or future match results.

import math

import pandas as pd


MINIMUM_HISTORY = 3

FEATURE_COLUMNS = [
    "home_rolling_goals_for",
    "home_rolling_goals_against",
    "away_rolling_goals_for",
    "away_rolling_goals_against",
    "home_season_goals_for",
    "home_season_goals_against",
    "away_season_goals_for",
    "away_season_goals_against",
]

HISTORY_COLUMNS = [
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
]
OUTPUT_COLUMNS.extend(FEATURE_COLUMNS)
OUTPUT_COLUMNS.extend(HISTORY_COLUMNS)
OUTPUT_COLUMNS.append("feature_supported")


def average(values):
    #There is no meaningful average before a team has played a match.
    if not values:
        return math.nan
    return sum(values) / len(values)


def season_average(totals, key):
    #The dictionary stores both the total goals and number of matches.
    values = totals.get(key)
    if values is None or values["matches"] == 0:
        return math.nan
    return values["total"] / values["matches"]


def season_matches(totals, key):
    values = totals.get(key)
    if values is None:
        return 0
    return values["matches"]


def add_season_goals(totals, key, goals):
    #Create a team's season record the first time the team appears.
    if key not in totals:
        totals[key] = {"total": 0.0, "matches": 0}
    totals[key]["total"] += goals
    totals[key]["matches"] += 1


def build_features(matches, rolling_window=5, minimum_history=MINIMUM_HISTORY):
    # The rolling window controls how many recent matches describe current form.
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")
    if minimum_history < 1:
        raise ValueError("minimum_history must be at least 1")

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
    #Sorting makes sure the history is built from oldest match to newest match.
    data = data.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # Each dictionary contains results from earlier dates only.
    team_histories = {}
    season_goals_for = {}
    season_goals_against = {}
    feature_rows = []

    # Every match on one date is calculated before that date updates history.
    # This is why two matches played on the same date cannot leak into each other.
    for _, date_matches in data.groupby("match_date", sort=True):
        pending_updates = []

        for match in date_matches.itertuples(index=False):
            source = str(match.source)
            competition_id = int(match.competition_id)
            season_id = int(match.season_id)
            home_id = int(match.home_team_id)
            away_id = int(match.away_team_id)

            # Provider is part of the key because providers use different team IDs.
            home_key = (source, competition_id, home_id)
            away_key = (source, competition_id, away_id)
            home_history = team_histories.get(home_key, {"for": [], "against": []})
            away_history = team_histories.get(away_key, {"for": [], "against": []})

            #Season averages reset each season, while rolling form can cross seasons.
            home_season_key = (source, competition_id, season_id, home_id)
            away_season_key = (source, competition_id, season_id, away_id)
            home_season_matches = season_matches(season_goals_for, home_season_key)
            away_season_matches = season_matches(season_goals_for, away_season_key)

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
                #Only the chosen number of most recent matches represents current form.
                "home_rolling_goals_for": average(
                    home_history["for"][-rolling_window:]
                ),
                "home_rolling_goals_against": average(
                    home_history["against"][-rolling_window:]
                ),
                "away_rolling_goals_for": average(
                    away_history["for"][-rolling_window:]
                ),
                "away_rolling_goals_against": average(
                    away_history["against"][-rolling_window:]
                ),
                #Season features use every earlier match from the current season.
                "home_season_goals_for": season_average(
                    season_goals_for, home_season_key
                ),
                "home_season_goals_against": season_average(
                    season_goals_against, home_season_key
                ),
                "away_season_goals_for": season_average(
                    season_goals_for, away_season_key
                ),
                "away_season_goals_against": season_average(
                    season_goals_against, away_season_key
                ),
                "home_history_matches": len(home_history["for"]),
                "away_history_matches": len(away_history["for"]),
                "home_season_matches": home_season_matches,
                "away_season_matches": away_season_matches,
            }

            # Early-season rows are excluded instead of being filled by fallbacks.
            row["feature_supported"] = (
                home_season_matches >= minimum_history
                and away_season_matches >= minimum_history
            )
            feature_rows.append(row)
            pending_updates.append(match)

        # Shift the goals by updating histories only after features are recorded.
        for match in pending_updates:
            source = str(match.source)
            competition_id = int(match.competition_id)
            season_id = int(match.season_id)
            home_id = int(match.home_team_id)
            away_id = int(match.away_team_id)
            home_goals = float(match.home_goals)
            away_goals = float(match.away_goals)

            home_key = (source, competition_id, home_id)
            away_key = (source, competition_id, away_id)
            if home_key not in team_histories:
                team_histories[home_key] = {"for": [], "against": []}
            if away_key not in team_histories:
                team_histories[away_key] = {"for": [], "against": []}

            #For means goals scored and against means goals conceded.
            team_histories[home_key]["for"].append(home_goals)
            team_histories[home_key]["against"].append(away_goals)
            team_histories[away_key]["for"].append(away_goals)
            team_histories[away_key]["against"].append(home_goals)

            home_season_key = (source, competition_id, season_id, home_id)
            away_season_key = (source, competition_id, season_id, away_id)
            add_season_goals(season_goals_for, home_season_key, home_goals)
            add_season_goals(season_goals_against, home_season_key, away_goals)
            add_season_goals(season_goals_for, away_season_key, away_goals)
            add_season_goals(season_goals_against, away_season_key, home_goals)

    return pd.DataFrame(feature_rows, columns=OUTPUT_COLUMNS)
