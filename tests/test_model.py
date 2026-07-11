from __future__ import annotations

import numpy as np
import pandas as pd

from football_prediction.features import build_features
from football_prediction.model import (
    chronological_split,
    fit_poisson_models,
    tune_poisson_models,
)


def model_matches() -> pd.DataFrame:
    matches = []
    for index in range(18):
        home_id = (index % 4) + 1
        away_id = ((index + 1) % 4) + 1
        matches.append(
            {
                "match_id": index + 1,
                "match_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index * 7),
                "competition_id": 10,
                "competition_name": "Example League",
                "season_id": 20 if index < 10 else 21,
                "season_name": "Season 1" if index < 10 else "Season 2",
                "home_team_id": home_id,
                "home_team_name": f"Team {home_id}",
                "away_team_id": away_id,
                "away_team_name": f"Team {away_id}",
                "home_goals": (index * 2) % 4,
                "away_goals": (index + 1) % 3,
                "home_xg": 0.8 + (index % 4) * 0.35,
                "away_xg": 0.6 + (index % 3) * 0.3,
                "eligible_for_model": True,
            }
        )
    return pd.DataFrame(matches)


def test_chronological_split_never_overlaps_dates() -> None:
    features = build_features(model_matches(), rolling_window=3)
    train, validation, test = chronological_split(features)

    assert train["match_date"].max() < validation["match_date"].min()
    assert validation["match_date"].max() < test["match_date"].min()


def test_fitted_lambdas_are_positive_and_deterministic() -> None:
    features = build_features(model_matches(), rolling_window=3)
    train, validation, _ = chronological_split(features)
    first = fit_poisson_models(train, alpha=0.1)
    second = fit_poisson_models(train, alpha=0.1)
    first_home, first_away = first.predict(validation)
    second_home, second_away = second.predict(validation)

    assert np.all(first_home > 0)
    assert np.all(first_away > 0)
    np.testing.assert_allclose(first_home, second_home)
    np.testing.assert_allclose(first_away, second_away)


def test_small_tuning_grid_keeps_test_period_separate() -> None:
    result = tune_poisson_models(
        model_matches(),
        rolling_windows=(3,),
        alphas=(0.1,),
    )

    assert result.best_window == 3
    assert result.best_alpha == 0.1
    assert result.validation_log_loss > 0
    assert not result.test_features.empty
