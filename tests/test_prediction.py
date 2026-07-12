import pytest

from football_prediction.model import tune_poisson_models
from football_prediction.prediction import load_model, predict_match, save_model
from helpers import training_and_recent_matches


def fitted_model_and_recent_matches():
    #One setting keeps these tests quick; tuning itself is tested elsewhere.
    training, recent = training_and_recent_matches()
    tuning = tune_poisson_models(
        training, recent, rolling_windows=(3,), alphas=(0.1,)
    )
    return tuning["model"], recent


def test_model_can_be_saved_loaded_and_used_for_a_future_match(tmp_path):
    model, recent = fitted_model_and_recent_matches()
    model_path = tmp_path / "models" / "model.pkl"
    save_model(model, model_path)
    loaded = load_model(model_path)

    result = predict_match(
        "Premier League Team 1",
        "Premier League Team 2",
        "Premier League",
        recent,
        loaded,
    )
    #A valid 1X2 prediction must distribute all probability across three results.
    total = (
        result["home_probability"]
        + result["draw_probability"]
        + result["away_probability"]
    )
    assert result["lambda_home"] > 0
    assert result["lambda_away"] > 0
    assert total == pytest.approx(1.0)


def test_prediction_rejects_the_same_team_twice():
    model, recent = fitted_model_and_recent_matches()
    with pytest.raises(ValueError, match="different"):
        predict_match(
            "Premier League Team 1",
            "Premier League Team 1",
            "Premier League",
            recent,
            model,
        )


def test_prediction_requires_three_current_season_matches():
    model, recent = fitted_model_and_recent_matches()
    #Keep only two matches per league, which is below the required history.
    short_history = recent.loc[recent["season_name"] == "2025/2026"].groupby(
        "competition_id", group_keys=False
    ).head(2)

    with pytest.raises(ValueError, match="at least 3"):
        predict_match(
            "Premier League Team 1",
            "Premier League Team 2",
            "Premier League",
            short_history,
            model,
        )
