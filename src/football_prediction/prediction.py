# Save the fitted model and use it for one future fixture.

from pathlib import Path

import joblib
import pandas as pd

from football_prediction.features import FEATURE_COLUMNS, MINIMUM_HISTORY
from football_prediction.model import predict_goals
from football_prediction.probabilities import calculate_probabilities


def save_model(model_bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def load_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return joblib.load(path)


def team_goal_history(matches, team_name):
    goals_for = []
    goals_against = []
    for match in matches.itertuples(index=False):
        if match.home_team_name == team_name:
            goals_for.append(float(match.home_goals))
            goals_against.append(float(match.away_goals))
        elif match.away_team_name == team_name:
            goals_for.append(float(match.away_goals))
            goals_against.append(float(match.home_goals))
    return goals_for, goals_against


def predict_match(home_team, away_team, competition, matches, model_bundle):
    if home_team == away_team:
        raise ValueError("Home and away teams must be different")

    data = matches.copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    competition_matches = data.loc[data["competition_name"] == competition].copy()
    competition_matches = competition_matches.sort_values(["match_date", "match_id"])
    if competition_matches.empty:
        raise ValueError(f"Unknown competition: {competition}")

    competition_ids = competition_matches["competition_id"].unique()
    if len(competition_ids) != 1:
        raise ValueError("Competition name must identify one competition")
    competition_id = int(competition_ids[0])
    if competition_id not in model_bundle["competitions"]:
        raise ValueError("The saved model does not support this competition")

    known_teams = set(competition_matches["home_team_name"])
    known_teams.update(competition_matches["away_team_name"])
    if home_team not in known_teams or away_team not in known_teams:
        raise ValueError("Both teams need earlier matches in this competition")

    latest_season = competition_matches.iloc[-1]["season_name"]
    season_matches = competition_matches.loc[
        competition_matches["season_name"] == latest_season
    ]
    home_all_for, home_all_against = team_goal_history(
        competition_matches, home_team
    )
    away_all_for, away_all_against = team_goal_history(
        competition_matches, away_team
    )
    home_season_for, home_season_against = team_goal_history(
        season_matches, home_team
    )
    away_season_for, away_season_against = team_goal_history(
        season_matches, away_team
    )

    if (
        len(home_season_for) < MINIMUM_HISTORY
        or len(away_season_for) < MINIMUM_HISTORY
    ):
        raise ValueError("Both teams need at least 3 previous matches this season")

    window = model_bundle["rolling_window"]
    feature_values = {
        "home_rolling_goals_for": sum(home_all_for[-window:]) / len(home_all_for[-window:]),
        "home_rolling_goals_against": sum(home_all_against[-window:])
        / len(home_all_against[-window:]),
        "away_rolling_goals_for": sum(away_all_for[-window:]) / len(away_all_for[-window:]),
        "away_rolling_goals_against": sum(away_all_against[-window:])
        / len(away_all_against[-window:]),
        "home_season_goals_for": sum(home_season_for) / len(home_season_for),
        "home_season_goals_against": sum(home_season_against)
        / len(home_season_against),
        "away_season_goals_for": sum(away_season_for) / len(away_season_for),
        "away_season_goals_against": sum(away_season_against)
        / len(away_season_against),
        "competition_id": competition_id,
        "feature_supported": True,
    }

    feature_row = pd.DataFrame([feature_values])
    home_lambdas, away_lambdas = predict_goals(model_bundle, feature_row)
    probabilities = calculate_probabilities(home_lambdas[0], away_lambdas[0])
    return {
        "home_team": home_team,
        "away_team": away_team,
        "competition": competition,
        "lambda_home": probabilities["lambda_home"],
        "lambda_away": probabilities["lambda_away"],
        "home_probability": probabilities["home_win"],
        "draw_probability": probabilities["draw"],
        "away_probability": probabilities["away_win"],
        "most_likely_score": probabilities["most_likely_score"],
        "feature_columns": FEATURE_COLUMNS.copy(),
    }
