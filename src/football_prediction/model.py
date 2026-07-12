# Fit one Poisson regression for home goals and one for away goals.

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from football_prediction.features import FEATURE_COLUMNS, build_features
from football_prediction.probabilities import calculate_probabilities


MODEL_COLUMNS = FEATURE_COLUMNS + ["competition_id"]
SUPPORTED_COMPETITIONS = [2, 7, 9, 11, 12]
DEFAULT_WINDOWS = (3, 5, 8)
DEFAULT_ALPHAS = (0.01, 0.1, 1.0)


def check_rows(features, need_goals):
    required = set(MODEL_COLUMNS)
    if need_goals:
        required.update(["home_goals", "away_goals"])

    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Model data is missing columns: {missing}")
    if features.empty:
        raise ValueError("Model data cannot be empty")

    numbers = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Model features must be finite")
    if features["competition_id"].isna().any():
        raise ValueError("Competition IDs cannot be missing")
    unknown_competitions = set(features["competition_id"]) - set(
        SUPPORTED_COMPETITIONS
    )
    if unknown_competitions:
        raise ValueError(f"Unsupported competition IDs: {unknown_competitions}")

    if need_goals:
        goals = features[["home_goals", "away_goals"]].to_numpy(dtype=float)
        if not np.isfinite(goals).all() or (goals < 0).any():
            raise ValueError("Goal targets must be finite and non-negative")


def make_poisson_pipeline(alpha):
    # Scaling puts numerical features on comparable ranges.
    # One-hot encoding gives every competition its own average scoring effect.
    preparation = ColumnTransformer(
        [
            ("numbers", StandardScaler(), FEATURE_COLUMNS),
            (
                "competition",
                OneHotEncoder(
                    categories=[SUPPORTED_COMPETITIONS],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["competition_id"],
            ),
        ]
    )

    # Alpha controls regularization: larger values pull coefficients closer to zero.
    regression = PoissonRegressor(alpha=alpha, max_iter=1000, tol=1e-9)
    return Pipeline([("prepare", preparation), ("poisson", regression)])


def supported_rows(features):
    if "feature_supported" not in features.columns:
        return features.copy()
    return features.loc[features["feature_supported"].fillna(False)].copy()


def fit_poisson_models(features, alpha=0.1):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    training = supported_rows(features)
    if training.empty:
        raise ValueError("No supported feature rows are available for fitting")
    check_rows(training, need_goals=True)

    home_model = make_poisson_pipeline(alpha)
    away_model = make_poisson_pipeline(alpha)
    home_model.fit(training[MODEL_COLUMNS], training["home_goals"])
    away_model.fit(training[MODEL_COLUMNS], training["away_goals"])

    return {
        "home_model": home_model,
        "away_model": away_model,
        "alpha": alpha,
        "feature_columns": MODEL_COLUMNS.copy(),
    }


def predict_goals(model_bundle, features):
    prediction_rows = supported_rows(features)
    if len(prediction_rows) != len(features):
        raise ValueError("Cannot predict rows without enough previous matches")
    check_rows(prediction_rows, need_goals=False)

    home = model_bundle["home_model"].predict(prediction_rows[MODEL_COLUMNS])
    away = model_bundle["away_model"].predict(prediction_rows[MODEL_COLUMNS])
    if not np.isfinite(home).all() or not np.isfinite(away).all():
        raise ValueError("Predicted Poisson lambdas must be finite")
    if (home <= 0).any() or (away <= 0).any():
        raise ValueError("Predicted Poisson lambdas must be positive")
    return home, away


def validation_log_loss(features, home_lambdas, away_lambdas):
    # Chronological validation measures settings on later, unseen matches.
    if len(features) != len(home_lambdas) or len(features) != len(away_lambdas):
        raise ValueError("Each validation row requires one home and away lambda")

    losses = []
    for i, match in enumerate(features.itertuples(index=False)):
        probabilities = calculate_probabilities(home_lambdas[i], away_lambdas[i])
        if match.home_goals > match.away_goals:
            actual_probability = probabilities["home_win"]
        elif match.home_goals == match.away_goals:
            actual_probability = probabilities["draw"]
        else:
            actual_probability = probabilities["away_win"]
        losses.append(-math.log(max(actual_probability, 1e-15)))

    return float(np.mean(losses))


def tune_poisson_models(
    training_matches,
    recent_matches,
    validation_season="2024/2025",
    test_season="2025/2026",
    rolling_windows=DEFAULT_WINDOWS,
    alphas=DEFAULT_ALPHAS,
):
    # The test season is identified here but its results never select settings.
    if validation_season == test_season:
        raise ValueError("Validation and test seasons must be different")

    tuning_rows = []
    best = None

    for window in rolling_windows:
        training_features = supported_rows(
            build_features(training_matches, rolling_window=window)
        )
        recent_features = build_features(recent_matches, rolling_window=window)
        validation = supported_rows(
            recent_features.loc[recent_features["season_name"] == validation_season]
        )
        test = supported_rows(
            recent_features.loc[recent_features["season_name"] == test_season]
        )

        if training_features.empty:
            raise ValueError("No supported older training matches were found")
        if validation.empty:
            raise ValueError(f"No supported matches found for {validation_season}")
        if test.empty:
            raise ValueError(f"No supported matches found for {test_season}")
        if training_features["match_date"].max() >= validation["match_date"].min():
            raise ValueError("Training matches must end before validation starts")
        if validation["match_date"].max() >= test["match_date"].min():
            raise ValueError("Validation matches must end before testing starts")

        for alpha in alphas:
            model_bundle = fit_poisson_models(training_features, alpha=alpha)
            home_lambdas, away_lambdas = predict_goals(model_bundle, validation)
            loss = validation_log_loss(validation, home_lambdas, away_lambdas)
            tuning_rows.append(
                {
                    "rolling_window": window,
                    "alpha": alpha,
                    "validation_log_loss": loss,
                }
            )

            if best is None or loss < best["loss"]:
                best = {
                    "loss": loss,
                    "window": window,
                    "alpha": alpha,
                    "training": training_features,
                    "validation": validation,
                    "test": test,
                }

    if best is None:
        raise ValueError("No tuning configurations were supplied")

    # After selection, validation can join training. The test season stays untouched.
    final_training = pd.concat(
        [best["training"], best["validation"]], ignore_index=True
    )
    final_model = fit_poisson_models(final_training, alpha=best["alpha"])
    final_model.update(
        {
            "rolling_window": best["window"],
            "training_end": final_training["match_date"].max().date().isoformat(),
            "competitions": SUPPORTED_COMPETITIONS.copy(),
        }
    )

    results = pd.DataFrame(tuning_rows).sort_values("validation_log_loss")
    return {
        "model": final_model,
        "best_window": best["window"],
        "best_alpha": best["alpha"],
        "validation_log_loss": best["loss"],
        "results": results.reset_index(drop=True),
        "test_features": best["test"].reset_index(drop=True),
    }
