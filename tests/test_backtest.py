import json
import math

import pandas as pd
import pytest

from football_prediction.backtest import (
    PREDICTION_COLUMNS,
    calculate_metrics,
    competition_average_baseline,
    run_backtest,
    save_backtest,
)
from football_prediction.model import tune_poisson_models
from helpers import training_and_recent_matches


def fitted_model_and_recent_matches():
    #A single model setting is enough because these tests focus on backtesting.
    training, recent = training_and_recent_matches()
    tuning = tune_poisson_models(
        training, recent, rolling_windows=(3,), alphas=(0.1,)
    )
    return tuning["model"], recent


def test_fitted_model_and_baseline_use_identical_matches_and_columns():
    model, recent = fitted_model_and_recent_matches()
    result = run_backtest(model, recent)
    predictions = result["predictions"]

    #A fair comparison requires both methods to predict exactly the same fixtures.
    model_rows = predictions.loc[predictions["model_name"] == "fitted_model"]
    baseline_rows = predictions.loc[
        predictions["model_name"] == "competition_average"
    ]
    assert set(model_rows["match_id"]) == set(baseline_rows["match_id"])
    assert list(predictions.columns) == PREDICTION_COLUMNS


def test_baseline_does_not_use_the_current_match_result():
    _, recent = training_and_recent_matches()
    original = competition_average_baseline(recent)
    changed = recent.copy()
    target_index = changed.index[-1]
    target_match_id = changed.loc[target_index, "match_id"]
    #Changing the target score should not change its already-made prediction.
    changed.loc[target_index, ["home_goals", "away_goals"]] = 99
    mutated = competition_average_baseline(changed)

    original_row = original.loc[original["match_id"] == target_match_id].iloc[0]
    mutated_row = mutated.loc[mutated["match_id"] == target_match_id].iloc[0]
    assert original_row["baseline_lambda_home"] == mutated_row["baseline_lambda_home"]
    assert original_row["baseline_lambda_away"] == mutated_row["baseline_lambda_away"]


def test_metric_calculations_on_hand_written_predictions():
    #Two simple rows make every expected metric possible to calculate by hand.
    predictions = pd.DataFrame(
        [
            {
                "home_goals": 2,
                "away_goals": 1,
                "lambda_home": 1.5,
                "lambda_away": 1.0,
                "home_probability": 0.6,
                "draw_probability": 0.2,
                "away_probability": 0.2,
                "predicted_result": "home",
                "actual_result": "home",
            },
            {
                "home_goals": 0,
                "away_goals": 0,
                "lambda_home": 1.0,
                "lambda_away": 0.5,
                "home_probability": 0.4,
                "draw_probability": 0.4,
                "away_probability": 0.2,
                "predicted_result": "home",
                "actual_result": "draw",
            },
        ]
    )
    metrics = calculate_metrics(predictions)

    assert metrics["multiclass_log_loss"] == pytest.approx(
        (-math.log(0.6) - math.log(0.4)) / 2
    )
    assert metrics["multiclass_brier_score"] == pytest.approx(0.4)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["home_goal_mae"] == pytest.approx(0.75)
    assert metrics["away_goal_mae"] == pytest.approx(0.25)
    assert metrics["combined_goal_mae"] == pytest.approx(0.5)


def test_final_test_season_is_after_validation():
    _, recent = training_and_recent_matches()
    validation = recent.loc[recent["season_name"] == "2024/2025"]
    test = recent.loc[recent["season_name"] == "2025/2026"]
    assert validation["match_date"].max() < test["match_date"].min()


def test_backtest_artifacts_are_saved_only_once(tmp_path):
    model, recent = fitted_model_and_recent_matches()
    result = run_backtest(model, recent)
    predictions_path = tmp_path / "reports" / "backtest_predictions.csv"
    metrics_path = tmp_path / "reports" / "metrics.json"
    save_backtest(result, predictions_path, metrics_path)

    assert predictions_path.is_file()
    assert metrics_path.is_file()
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "fitted_model" in saved_metrics
    assert "competition_average_baseline" in saved_metrics

    #Refusing to overwrite keeps the original test result reproducible.
    with pytest.raises(FileExistsError, match="already exist"):
        save_backtest(result, predictions_path, metrics_path)
