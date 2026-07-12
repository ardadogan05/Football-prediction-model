# Evaluate the frozen fitted model and a simple causal baseline.

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from football_prediction.features import build_features
from football_prediction.model import predict_goals, supported_rows
from football_prediction.probabilities import calculate_probabilities


PREDICTION_COLUMNS = [
    "match_id",
    "match_date",
    "competition_name",
    "home_team_name",
    "away_team_name",
    "home_goals",
    "away_goals",
    "lambda_home",
    "lambda_away",
    "home_probability",
    "draw_probability",
    "away_probability",
    "most_likely_score",
    "predicted_result",
    "actual_result",
    "model_name",
]


def match_result(home_value, away_value):
    #This works for both actual goals and predicted goal values.
    if home_value > away_value:
        return "home"
    if home_value == away_value:
        return "draw"
    return "away"


def competition_average_baseline(matches):
    # This baseline knows only the earlier home and away goal averages in a league.
    data = matches.copy()
    data["match_date"] = pd.to_datetime(data["match_date"])
    data = data.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    totals = {}
    rows = []

    # Same-date matches are predicted before any result from that date is added.
    for _, date_matches in data.groupby("match_date", sort=True):
        for match in date_matches.itertuples(index=False):
            key = int(match.competition_id)
            previous = totals.get(key)
            if previous is None or previous["matches"] == 0:
                home_lambda = math.nan
                away_lambda = math.nan
            else:
                home_lambda = previous["home_goals"] / previous["matches"]
                away_lambda = previous["away_goals"] / previous["matches"]

            rows.append(
                {
                    "match_id": match.match_id,
                    "baseline_lambda_home": home_lambda,
                    "baseline_lambda_away": away_lambda,
                }
            )

        for match in date_matches.itertuples(index=False):
            key = int(match.competition_id)
            if key not in totals:
                totals[key] = {"home_goals": 0.0, "away_goals": 0.0, "matches": 0}
            totals[key]["home_goals"] += float(match.home_goals)
            totals[key]["away_goals"] += float(match.away_goals)
            totals[key]["matches"] += 1

    return pd.DataFrame(rows)


def make_prediction_rows(features, home_lambdas, away_lambdas, model_name):
    rows = []
    for i, match in enumerate(features.itertuples(index=False)):
        probabilities = calculate_probabilities(home_lambdas[i], away_lambdas[i])
        score = probabilities["most_likely_score"]
        result_probabilities = [
            probabilities["home_win"],
            probabilities["draw"],
            probabilities["away_win"],
        ]
        #The final 1X2 prediction is whichever total probability is largest.
        predicted = ["home", "draw", "away"][int(np.argmax(result_probabilities))]

        rows.append(
            {
                "match_id": match.match_id,
                "match_date": match.match_date,
                "competition_name": match.competition_name,
                "home_team_name": match.home_team_name,
                "away_team_name": match.away_team_name,
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
                "lambda_home": float(home_lambdas[i]),
                "lambda_away": float(away_lambdas[i]),
                "home_probability": probabilities["home_win"],
                "draw_probability": probabilities["draw"],
                "away_probability": probabilities["away_win"],
                "most_likely_score": f"{score[0]}-{score[1]}",
                "predicted_result": predicted,
                "actual_result": match_result(match.home_goals, match.away_goals),
                "model_name": model_name,
            }
        )

    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def calculate_metrics(predictions):
    if predictions.empty:
        raise ValueError("Cannot calculate metrics without predictions")

    #Lower log loss and Brier score both mean better probability estimates.
    log_losses = []
    brier_scores = []
    for row in predictions.itertuples(index=False):
        probabilities = {
            "home": row.home_probability,
            "draw": row.draw_probability,
            "away": row.away_probability,
        }
        #Log loss only reads the probability given to the result that happened.
        actual_probability = probabilities[row.actual_result]
        log_losses.append(-math.log(max(actual_probability, 1e-15)))

        #Brier score compares all three probabilities with the actual outcome.
        squared_errors = 0.0
        for result in ["home", "draw", "away"]:
            actual_value = 1.0 if row.actual_result == result else 0.0
            squared_errors += (probabilities[result] - actual_value) ** 2
        brier_scores.append(squared_errors)

    home_error = np.abs(predictions["home_goals"] - predictions["lambda_home"])
    away_error = np.abs(predictions["away_goals"] - predictions["lambda_away"])
    #Accuracy ignores probability size and checks only the final 1X2 choice.
    accuracy = np.mean(predictions["predicted_result"] == predictions["actual_result"])

    return {
        "matches": int(len(predictions)),
        "multiclass_log_loss": float(np.mean(log_losses)),
        "multiclass_brier_score": float(np.mean(brier_scores)),
        "accuracy": float(accuracy),
        "home_goal_mae": float(np.mean(home_error)),
        "away_goal_mae": float(np.mean(away_error)),
        "combined_goal_mae": float(np.mean([np.mean(home_error), np.mean(away_error)])),
    }


def run_backtest(model_bundle, recent_matches, test_season="2025/2026"):
    # The model is already frozen. This function never tunes or refits it.
    features = build_features(
        recent_matches,
        rolling_window=model_bundle["rolling_window"],
    )
    test_features = supported_rows(
        features.loc[features["season_name"] == test_season]
    ).reset_index(drop=True)
    if test_features.empty:
        raise ValueError(f"No supported matches found for {test_season}")

    home_lambdas, away_lambdas = predict_goals(model_bundle, test_features)
    model_predictions = make_prediction_rows(
        test_features, home_lambdas, away_lambdas, "fitted_model"
    )

    #The simple baseline shows whether fitting the model was actually worthwhile.
    baseline = competition_average_baseline(recent_matches)
    test_with_baseline = test_features.merge(baseline, on="match_id", how="left")
    baseline_home = test_with_baseline["baseline_lambda_home"].to_numpy(dtype=float)
    baseline_away = test_with_baseline["baseline_lambda_away"].to_numpy(dtype=float)
    if not np.isfinite(baseline_home).all() or not np.isfinite(baseline_away).all():
        raise ValueError("The baseline needs earlier competition matches")
    baseline_predictions = make_prediction_rows(
        test_with_baseline, baseline_home, baseline_away, "competition_average"
    )

    if set(model_predictions["match_id"]) != set(baseline_predictions["match_id"]):
        raise ValueError("The fitted model and baseline must use identical matches")

    #Both prediction sets are saved together so they can be compared match by match.
    predictions = pd.concat(
        [model_predictions, baseline_predictions], ignore_index=True
    )
    model_metrics = calculate_metrics(model_predictions)
    baseline_metrics = calculate_metrics(baseline_predictions)
    metrics = {
        "test_season": test_season,
        "fitted_model": model_metrics,
        "competition_average_baseline": baseline_metrics,
        "model_beats_baseline": (
            model_metrics["multiclass_log_loss"]
            < baseline_metrics["multiclass_log_loss"]
        ),
    }
    return {"predictions": predictions, "metrics": metrics}


def save_backtest(result, predictions_path, metrics_path):
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    if predictions_path.exists() or metrics_path.exists():
        raise FileExistsError(
            "Backtest artifacts already exist; do not repeatedly inspect the test set"
        )

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(predictions_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(result["metrics"], file, indent=2, sort_keys=True)
