import numpy as np
import pandas as pd
import pytest

from football_prediction.features import build_features
from football_prediction.model import (
    fit_poisson_models,
    predict_goals,
    supported_rows,
    tune_poisson_models,
)
from helpers import COMPETITIONS, training_and_recent_matches


def test_both_models_fit_and_predictions_are_positive_and_deterministic():
    training, _ = training_and_recent_matches()
    features = supported_rows(build_features(training, rolling_window=3))

    first = fit_poisson_models(features, alpha=0.1)
    second = fit_poisson_models(features, alpha=0.1)
    first_home, first_away = predict_goals(first, features)
    second_home, second_away = predict_goals(second, features)

    assert np.isfinite(first_home).all()
    assert np.isfinite(first_away).all()
    assert (first_home > 0).all()
    assert (first_away > 0).all()
    np.testing.assert_allclose(first_home, second_home)
    np.testing.assert_allclose(first_away, second_away)


def test_competition_encoder_contains_all_five_leagues():
    training, _ = training_and_recent_matches()
    features = supported_rows(build_features(training, rolling_window=3))
    model = fit_poisson_models(features, alpha=0.1)

    preparation = model["home_model"].named_steps["prepare"]
    encoder = preparation.named_transformers_["competition"]
    encoded_competitions = set(encoder.categories_[0])
    expected_competitions = set(competition_id for competition_id, _ in COMPETITIONS)
    assert encoded_competitions == expected_competitions


def test_tuning_records_every_window_and_alpha_configuration():
    training, recent = training_and_recent_matches()
    result = tune_poisson_models(
        training,
        recent,
        rolling_windows=(3, 5),
        alphas=(0.01, 0.1),
    )

    configurations = set()
    for row in result["results"].itertuples(index=False):
        configurations.add((row.rolling_window, row.alpha))
    assert configurations == {(3, 0.01), (3, 0.1), (5, 0.01), (5, 0.1)}


def test_final_test_goals_cannot_change_tuning_selection():
    training, recent = training_and_recent_matches()
    changed = recent.copy()
    changed.loc[
        changed["season_name"] == "2025/2026", ["home_goals", "away_goals"]
    ] = 99

    original = tune_poisson_models(
        training, recent, rolling_windows=(3,), alphas=(0.1,)
    )
    mutated = tune_poisson_models(
        training, changed, rolling_windows=(3,), alphas=(0.1,)
    )

    pd.testing.assert_frame_equal(original["results"], mutated["results"])
    assert original["best_window"] == mutated["best_window"]
    assert original["best_alpha"] == mutated["best_alpha"]


def test_training_validation_and_test_periods_are_chronological():
    training, recent = training_and_recent_matches()
    result = tune_poisson_models(
        training, recent, rolling_windows=(3,), alphas=(0.1,)
    )
    validation = recent.loc[recent["season_name"] == "2024/2025"]
    test = result["test_features"]

    assert training["match_date"].max() < validation["match_date"].min()
    assert validation["match_date"].max() < test["match_date"].min()


def test_unsupported_rows_are_not_predicted():
    training, _ = training_and_recent_matches()
    features = build_features(training, rolling_window=3)
    supported = supported_rows(features)
    model = fit_poisson_models(supported, alpha=0.1)
    unsupported = features.loc[~features["feature_supported"]].iloc[[0]]

    with pytest.raises(ValueError, match="enough previous matches"):
        predict_goals(model, unsupported)
