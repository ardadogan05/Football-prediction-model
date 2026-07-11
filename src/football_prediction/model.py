"""Fit home-goal and away-goal Poisson regression models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from football_prediction.features import FEATURE_COLUMNS, build_features
from football_prediction.probabilities import calculate_probabilities


NUMERIC_FEATURES = FEATURE_COLUMNS
CATEGORICAL_FEATURES = ["competition_id"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class PoissonModelBundle:
    home_model: Pipeline
    away_model: Pipeline
    rolling_window: int
    alpha: float

    def predict(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home = self.home_model.predict(features[MODEL_FEATURES])
        away = self.away_model.predict(features[MODEL_FEATURES])
        return home, away


@dataclass(frozen=True)
class TuningResult:
    model: PoissonModelBundle
    best_window: int
    best_alpha: float
    validation_log_loss: float
    results: pd.DataFrame
    test_features: pd.DataFrame


def _model_pipeline(alpha: float) -> Pipeline:
    numeric = Pipeline(
        [
            ("missing_values", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categories = OneHotEncoder(handle_unknown="ignore")
    preparation = ColumnTransformer(
        [
            ("numbers", numeric, NUMERIC_FEATURES),
            ("competition", categories, CATEGORICAL_FEATURES),
        ]
    )

    #alpha controls regularization. Larger values keep coefficients closer to zero.
    return Pipeline(
        [
            ("prepare", preparation),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )


def fit_poisson_models(features: pd.DataFrame, alpha: float = 0.1) -> PoissonModelBundle:
    """Fit separate models for home goals and away goals."""

    if alpha < 0:
        raise ValueError("alpha cannot be negative")
    home_model = _model_pipeline(alpha)
    away_model = _model_pipeline(alpha)
    home_model.fit(features[MODEL_FEATURES], features["home_goals"])
    away_model.fit(features[MODEL_FEATURES], features["away_goals"])
    return PoissonModelBundle(home_model, away_model, rolling_window=0, alpha=alpha)


def chronological_split(
    features: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete match dates into training, validation, and testing periods."""

    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("The final test period must be larger than zero")

    ordered = features.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    dates = list(ordered["match_date"].drop_duplicates().sort_values())
    if len(dates) < 5:
        raise ValueError("At least five separate match dates are required")

    train_end = max(1, int(len(dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])

    train = ordered.loc[ordered["match_date"].isin(train_dates)].copy()
    validation = ordered.loc[ordered["match_date"].isin(validation_dates)].copy()
    test = ordered.loc[ordered["match_date"].isin(test_dates)].copy()
    return train, validation, test


def _actual_result(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def _validation_log_loss(
    features: pd.DataFrame, home_lambdas: np.ndarray, away_lambdas: np.ndarray
) -> float:
    losses = []
    for position, match in enumerate(features.itertuples(index=False)):
        probabilities = calculate_probabilities(
            home_lambdas[position], away_lambdas[position]
        )
        predicted = [
            probabilities.home_win,
            probabilities.draw,
            probabilities.away_win,
        ]
        actual = _actual_result(match.home_goals, match.away_goals)
        losses.append(-np.log(max(predicted[actual], 1e-15)))
    return float(np.mean(losses))


def tune_poisson_models(
    matches: pd.DataFrame,
    rolling_windows: tuple[int, ...] = (3, 5, 8),
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0),
) -> TuningResult:
    """Choose a small feature/model grid using chronological validation log loss."""

    tuning_rows = []
    best_loss = math.inf
    best_window = None
    best_alpha = None
    best_train = None
    best_validation = None
    best_test = None

    for window in rolling_windows:
        features = build_features(matches, rolling_window=window)
        train, validation, test = chronological_split(features)
        for alpha in alphas:
            fitted = fit_poisson_models(train, alpha=alpha)
            home_lambdas, away_lambdas = fitted.predict(validation)
            loss = _validation_log_loss(validation, home_lambdas, away_lambdas)
            tuning_rows.append(
                {
                    "rolling_window": window,
                    "alpha": alpha,
                    "validation_log_loss": loss,
                }
            )
            if loss < best_loss:
                best_loss = loss
                best_window = window
                best_alpha = alpha
                best_train = train
                best_validation = validation
                best_test = test

    if best_window is None or best_alpha is None:
        raise ValueError("No tuning configurations were supplied")

    final_training = pd.concat([best_train, best_validation], ignore_index=True)
    model = fit_poisson_models(final_training, alpha=best_alpha)
    model = PoissonModelBundle(
        model.home_model,
        model.away_model,
        rolling_window=best_window,
        alpha=best_alpha,
    )
    results = pd.DataFrame(tuning_rows).sort_values("validation_log_loss")
    return TuningResult(
        model=model,
        best_window=best_window,
        best_alpha=best_alpha,
        validation_log_loss=best_loss,
        results=results.reset_index(drop=True),
        test_features=best_test,
    )
