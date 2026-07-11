"""Fit home-goal and away-goal Poisson regression models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from football_prediction.features import FEATURE_COLUMNS, build_features
from football_prediction.probabilities import (
    calculate_probabilities,
    required_score_cutoff,
)


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
        # A row is unsupported when there was no earlier league information.
        if "feature_supported" in features.columns:
            supported = features["feature_supported"].fillna(False)
            if not supported.all():
                raise ValueError("Cannot predict unsupported feature rows")
        _validate_model_rows(features, require_targets=False)

        home = self.home_model.predict(features[MODEL_FEATURES])
        away = self.away_model.predict(features[MODEL_FEATURES])

        lambdas_are_finite = np.isfinite(home).all() and np.isfinite(away).all()
        lambdas_are_positive = (home > 0).all() and (away > 0).all()
        if not lambdas_are_finite or not lambdas_are_positive:
            raise ValueError("Predicted Poisson lambdas must be finite and positive")
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
    """Prepare the feature columns and then fit one Poisson regression."""

    categories = OneHotEncoder(handle_unknown="ignore")

    # Scaling stops large-number features from dominating regularization.
    # Encoding turns each competition ID into a column the model can use.
    preparation = ColumnTransformer(
        [
            ("numbers", StandardScaler(), NUMERIC_FEATURES),
            ("competition", categories, CATEGORICAL_FEATURES),
        ]
    )

    # Alpha controls regularization. Larger values keep coefficients closer to zero.
    return Pipeline(
        [
            ("prepare", preparation),
            ("poisson", PoissonRegressor(alpha=alpha, max_iter=1000)),
        ]
    )


def _validate_model_rows(
    features: pd.DataFrame,
    require_targets: bool,
) -> None:
    required = set(MODEL_FEATURES)
    if require_targets:
        required.update(["home_goals", "away_goals"])
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Model data is missing columns: {missing}")
    if features.empty:
        raise ValueError("Model data cannot be empty")

    numeric = features[NUMERIC_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Model features must be finite")
    if features[CATEGORICAL_FEATURES].isna().any().any():
        raise ValueError("Categorical model features cannot be missing")

    if require_targets:
        targets = features[["home_goals", "away_goals"]].to_numpy(dtype=float)
        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("Goal targets must be finite and non-negative")


def fit_poisson_models(features: pd.DataFrame, alpha: float = 0.1) -> PoissonModelBundle:
    """Fit separate models after removing rows with no earlier league data."""

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    training = features
    if "feature_supported" in training.columns:
        supported = training["feature_supported"].fillna(False)
        training = training.loc[supported].copy()
    if training.empty:
        raise ValueError("No supported feature rows are available for fitting")
    _validate_model_rows(training, require_targets=True)

    home_model = _model_pipeline(alpha)
    away_model = _model_pipeline(alpha)
    home_model.fit(training[MODEL_FEATURES], training["home_goals"])
    away_model.fit(training[MODEL_FEATURES], training["away_goals"])
    # The generic fit function does not choose a window. Tuning records it later.
    return PoissonModelBundle(home_model, away_model, rolling_window=0, alpha=alpha)


def chronological_split(
    features: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete match dates into training, validation, and testing periods."""

    if (
        not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or train_fraction <= 0
        or validation_fraction <= 0
    ):
        raise ValueError("Split fractions must be finite and positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("The final test period must be larger than zero")

    # Always split oldest matches first.
    ordered = features.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    dates = list(ordered["match_date"].drop_duplicates().sort_values())
    if len(dates) < 5:
        raise ValueError("At least five separate match dates are required")

    # Count matches on each date so one date is never split across two periods.
    date_counts = ordered.groupby("match_date", sort=True).size()
    cumulative_rows = date_counts.cumsum().to_numpy()
    train_target = len(ordered) * train_fraction
    validation_target = len(ordered) * (train_fraction + validation_fraction)

    # Find the complete-date boundaries closest to the requested row counts.
    train_differences = np.abs(cumulative_rows - train_target)
    validation_differences = np.abs(cumulative_rows - validation_target)
    train_end = int(np.argmin(train_differences)) + 1
    validation_end = int(np.argmin(validation_differences)) + 1
    train_end = max(train_end, 1)
    train_end = min(train_end, len(dates) - 2)
    validation_end = max(validation_end, train_end + 1)
    validation_end = min(validation_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end:validation_end])
    test_dates = set(dates[validation_end:])

    train = ordered.loc[ordered["match_date"].isin(train_dates)].copy()
    validation = ordered.loc[ordered["match_date"].isin(validation_dates)].copy()
    test = ordered.loc[ordered["match_date"].isin(test_dates)].copy()
    return train, validation, test


def _validation_log_loss(
    features: pd.DataFrame, home_lambdas: np.ndarray, away_lambdas: np.ndarray
) -> float:
    """Measure how much probability the model gave to the result that happened."""

    if len(features) != len(home_lambdas) or len(features) != len(away_lambdas):
        raise ValueError("Each validation row requires one home and away lambda")
    losses = []
    for position, match in enumerate(features.itertuples(index=False)):
        max_score = required_score_cutoff(
            float(home_lambdas[position]),
            float(away_lambdas[position]),
        )
        probabilities = calculate_probabilities(
            home_lambdas[position],
            away_lambdas[position],
            max_score=max_score,
        )
        if match.home_goals > match.away_goals:
            actual_probability = probabilities.home_win
        elif match.home_goals == match.away_goals:
            actual_probability = probabilities.draw
        else:
            actual_probability = probabilities.away_win

        # Avoid log(0), which would produce an infinite loss.
        safe_probability = max(actual_probability, 1e-15)
        losses.append(-np.log(safe_probability))
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
        # Each window creates slightly different rolling goal features.
        features = build_features(matches, rolling_window=window)
        train, validation, test = chronological_split(features)
        for alpha in alphas:
            fitted = fit_poisson_models(train, alpha=alpha)
            home_lambdas, away_lambdas = fitted.predict(validation)

            # Log loss chooses the best settings. Lower values are better.
            loss = _validation_log_loss(validation, home_lambdas, away_lambdas)

            # Deviance is an extra check of how well the goal predictions fit.
            home_deviance = float(
                mean_poisson_deviance(validation["home_goals"], home_lambdas)
            )
            away_deviance = float(
                mean_poisson_deviance(validation["away_goals"], away_lambdas)
            )
            tuning_rows.append(
                {
                    "rolling_window": window,
                    "alpha": alpha,
                    "validation_log_loss": loss,
                    "home_poisson_deviance": home_deviance,
                    "away_poisson_deviance": away_deviance,
                    "mean_poisson_deviance": (home_deviance + away_deviance) / 2,
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

    # After choosing settings, refit on train + validation. No test row is fitted.
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


def tune_external_poisson_models(
    training_matches: pd.DataFrame,
    recent_matches: pd.DataFrame,
    validation_season: str = "2024/2025",
    test_season: str = "2025/2026",
    rolling_windows: tuple[int, ...] = (3, 5, 8),
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0),
) -> TuningResult:
    """Train on StatsBomb, tune on one recent season, and return the next."""

    if validation_season == test_season:
        raise ValueError("Validation and test seasons must be different")

    tuning_rows = []
    best_loss = math.inf
    best_window = None
    best_alpha = None
    best_train = None
    best_validation = None
    best_test = None

    for window in rolling_windows:
        # Provider team IDs are different, so each source builds its own team
        # histories. The fitted regressions still learn the same feature columns.
        training_features = build_features(training_matches, rolling_window=window)
        recent_features = build_features(recent_matches, rolling_window=window)

        validation = recent_features.loc[
            (recent_features["season_name"] == validation_season)
            & recent_features["feature_supported"]
        ].copy()
        test = recent_features.loc[
            (recent_features["season_name"] == test_season)
            & recent_features["feature_supported"]
        ].copy()

        if validation.empty:
            raise ValueError(f"No supported matches found for {validation_season}")
        if test.empty:
            raise ValueError(f"No supported matches found for {test_season}")

        # Keep the three roles in their real time order. This also protects the
        # split if either data source adds new seasons later.
        if training_features["match_date"].max() >= validation["match_date"].min():
            raise ValueError("Training matches must end before validation starts")
        if validation["match_date"].max() >= test["match_date"].min():
            raise ValueError("Validation matches must end before testing starts")

        for alpha in alphas:
            # Candidate settings learn only from the older StatsBomb rows.
            fitted = fit_poisson_models(training_features, alpha=alpha)
            home_lambdas, away_lambdas = fitted.predict(validation)
            loss = _validation_log_loss(validation, home_lambdas, away_lambdas)
            home_deviance = float(
                mean_poisson_deviance(validation["home_goals"], home_lambdas)
            )
            away_deviance = float(
                mean_poisson_deviance(validation["away_goals"], away_lambdas)
            )
            tuning_rows.append(
                {
                    "rolling_window": window,
                    "alpha": alpha,
                    "validation_log_loss": loss,
                    "home_poisson_deviance": home_deviance,
                    "away_poisson_deviance": away_deviance,
                    "mean_poisson_deviance": (home_deviance + away_deviance) / 2,
                }
            )

            if loss < best_loss:
                best_loss = loss
                best_window = window
                best_alpha = alpha
                best_train = training_features
                best_validation = validation
                best_test = test

    if best_window is None or best_alpha is None:
        raise ValueError("No tuning configurations were supplied")

    # The final model learns from both sources, but never fits the test season.
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
