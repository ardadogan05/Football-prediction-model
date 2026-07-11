from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_prediction import model as model_module
from football_prediction.features import build_features
from football_prediction.model import (
    _validation_log_loss,
    chronological_split,
    fit_poisson_models,
    tune_external_poisson_models,
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
                "match_date": pd.Timestamp("2020-01-01")
                + pd.Timedelta(index * 7, unit="D"),
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
    assert len(train) == pytest.approx(len(features) * 0.6, abs=1)
    assert len(validation) == pytest.approx(len(features) * 0.2, abs=1)
    assert len(test) == pytest.approx(len(features) * 0.2, abs=1)


@pytest.mark.parametrize("invalid_fraction", [float("nan"), float("inf")])
def test_chronological_split_rejects_non_finite_fractions(
    invalid_fraction: float,
) -> None:
    features = build_features(model_matches(), rolling_window=3)

    with pytest.raises(ValueError, match="finite and positive"):
        chronological_split(features, train_fraction=invalid_fraction)


def test_chronological_split_keeps_duplicate_dates_together() -> None:
    features = build_features(model_matches(), rolling_window=3)
    features.loc[features["match_id"].isin([10, 11, 12]), "match_date"] = pd.Timestamp(
        "2020-03-04"
    )

    train, validation, test = chronological_split(features)

    train_dates = set(train["match_date"])
    validation_dates = set(validation["match_date"])
    test_dates = set(test["match_date"])
    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(test_dates)


def test_fitted_lambdas_are_positive_and_deterministic() -> None:
    features = build_features(model_matches(), rolling_window=3)
    train, validation, _ = chronological_split(features)
    first = fit_poisson_models(train, alpha=0.1)
    second = fit_poisson_models(train, alpha=0.1)
    first_home, first_away = first.predict(validation)
    second_home, second_away = second.predict(validation)

    assert np.all(first_home > 0)
    assert np.all(first_away > 0)
    assert np.isfinite(first_home).all()
    assert np.isfinite(first_away).all()
    np.testing.assert_allclose(first_home, second_home)
    np.testing.assert_allclose(first_away, second_away)


def test_small_tuning_grid_keeps_test_period_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted_match_ids = []
    original_fit = model_module.fit_poisson_models

    def recording_fit(features: pd.DataFrame, alpha: float = 0.1):
        fitted_match_ids.append(set(features["match_id"]))
        return original_fit(features, alpha=alpha)

    monkeypatch.setattr(model_module, "fit_poisson_models", recording_fit)
    result = tune_poisson_models(
        model_matches(),
        rolling_windows=(3,),
        alphas=(0.1,),
    )

    assert result.best_window == 3
    assert result.best_alpha == 0.1
    assert result.validation_log_loss > 0
    assert not result.test_features.empty
    assert {
        "home_poisson_deviance",
        "away_poisson_deviance",
        "mean_poisson_deviance",
    }.issubset(result.results.columns)
    assert np.isfinite(result.results["mean_poisson_deviance"]).all()

    test_ids = set(result.test_features["match_id"])
    assert len(fitted_match_ids) == 2  # grid training, then train+validation refit
    assert all(test_ids.isdisjoint(match_ids) for match_ids in fitted_match_ids)


def test_unsupported_rows_are_not_imputed_or_predicted() -> None:
    features = build_features(model_matches(), rolling_window=3)
    unsupported = features.loc[~features["feature_supported"]]

    with pytest.raises(ValueError, match="No supported"):
        fit_poisson_models(unsupported, alpha=0.1)

    train, validation, _ = chronological_split(features)
    fitted = fit_poisson_models(train, alpha=0.1)
    with pytest.raises(ValueError, match="unsupported"):
        fitted.predict(unsupported)

    invalid = validation.copy()
    invalid.loc[invalid.index[0], "home_rolling_goals_for"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fitted.predict(invalid)


def test_validation_log_loss_adapts_to_high_finite_lambdas() -> None:
    validation = pd.DataFrame([{"home_goals": 4, "away_goals": 2}])

    loss = _validation_log_loss(
        validation,
        np.array([15.0]),
        np.array([12.0]),
    )

    assert np.isfinite(loss)
    assert loss > 0


def test_test_period_targets_cannot_change_tuning_selection() -> None:
    matches = model_matches()
    features = build_features(matches, rolling_window=3)
    _, _, test = chronological_split(features)
    test_ids = set(test["match_id"])
    changed = matches.copy()
    changed.loc[
        changed["match_id"].isin(test_ids), ["home_goals", "away_goals"]
    ] = 99

    original = tune_poisson_models(matches, rolling_windows=(3,), alphas=(0.1,))
    mutated = tune_poisson_models(changed, rolling_windows=(3,), alphas=(0.1,))

    pd.testing.assert_frame_equal(original.results, mutated.results)
    assert original.best_window == mutated.best_window
    assert original.best_alpha == mutated.best_alpha

    # The first test date cannot use goals from any earlier test match.
    first_test_date = original.test_features["match_date"].min()
    original_first_date = original.test_features.loc[
        original.test_features["match_date"] == first_test_date
    ]
    mutated_first_date = mutated.test_features.loc[
        mutated.test_features["match_date"] == first_test_date
    ]
    original_home, original_away = original.model.predict(original_first_date)
    mutated_home, mutated_away = mutated.model.predict(mutated_first_date)
    np.testing.assert_array_equal(original_home, mutated_home)
    np.testing.assert_array_equal(original_away, mutated_away)


def recent_source_matches() -> pd.DataFrame:
    matches = model_matches().copy()
    matches["source"] = "football_data"
    matches["match_id"] += 100
    matches["match_date"] += pd.Timedelta(500, unit="D")
    matches.loc[matches.index < 9, "season_name"] = "2024/2025"
    matches.loc[matches.index < 9, "season_id"] = 24
    matches.loc[matches.index >= 9, "season_name"] = "2025/2026"
    matches.loc[matches.index >= 9, "season_id"] = 25
    return matches


def test_external_tuning_uses_both_sources_but_never_fits_test_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted_rows = []
    original_fit = model_module.fit_poisson_models

    def recording_fit(features: pd.DataFrame, alpha: float = 0.1):
        fitted_rows.append(set(zip(features["source"], features["match_id"])))
        return original_fit(features, alpha=alpha)

    monkeypatch.setattr(model_module, "fit_poisson_models", recording_fit)
    recent = recent_source_matches()
    result = tune_external_poisson_models(
        model_matches(),
        recent,
        rolling_windows=(3,),
        alphas=(0.1,),
    )

    test_rows = set(
        zip(result.test_features["source"], result.test_features["match_id"])
    )
    assert len(fitted_rows) == 2
    assert all(test_rows.isdisjoint(rows) for rows in fitted_rows)
    assert all(source == "statsbomb" for source, _ in fitted_rows[0])
    assert {source for source, _ in fitted_rows[1]} == {
        "statsbomb",
        "football_data",
    }
    assert set(result.test_features["season_name"]) == {"2025/2026"}


def test_external_test_goals_cannot_change_model_selection() -> None:
    recent = recent_source_matches()
    changed = recent.copy()
    changed.loc[
        changed["season_name"] == "2025/2026", ["home_goals", "away_goals"]
    ] = 99

    original = tune_external_poisson_models(
        model_matches(), recent, rolling_windows=(3,), alphas=(0.1,)
    )
    mutated = tune_external_poisson_models(
        model_matches(), changed, rolling_windows=(3,), alphas=(0.1,)
    )

    pd.testing.assert_frame_equal(original.results, mutated.results)
    assert original.best_window == mutated.best_window
    assert original.best_alpha == mutated.best_alpha

    first_test_date = original.test_features["match_date"].min()
    original_first_date = original.test_features.loc[
        original.test_features["match_date"] == first_test_date
    ]
    mutated_first_date = mutated.test_features.loc[
        mutated.test_features["match_date"] == first_test_date
    ]
    original_home, original_away = original.model.predict(original_first_date)
    mutated_home, mutated_away = mutated.model.predict(mutated_first_date)
    np.testing.assert_array_equal(original_home, mutated_home)
    np.testing.assert_array_equal(original_away, mutated_away)


def test_external_tuning_requires_chronological_source_periods() -> None:
    training = model_matches().copy()
    training["match_date"] += pd.Timedelta(1_000, unit="D")

    with pytest.raises(ValueError, match="Training matches must end before"):
        tune_external_poisson_models(
            training,
            recent_source_matches(),
            rolling_windows=(3,),
            alphas=(0.1,),
        )
