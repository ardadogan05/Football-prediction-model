# Fit simple Poisson regressions for home and away goals.
# NumPy builds the matrices and SciPy finds the weights that give the smallest loss.

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football_prediction.features import FEATURE_COLUMNS, build_features
from football_prediction.probabilities import calculate_probabilities
from football_prediction.probabilities import required_score_cutoff


MODEL_COLUMNS = FEATURE_COLUMNS + ["competition_id"]


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

    if need_goals:
        goals = features[["home_goals", "away_goals"]].to_numpy(dtype=float)
        if not np.isfinite(goals).all() or (goals < 0).any():
            raise ValueError("Goal targets must be finite and non-negative")


def make_input_matrix(features, means, scales, competitions):
    # The first column is all ones for the intercept. The numeric columns are
    # scaled, then one 0/1 column is added for every known competition.
    numbers = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    scaled_numbers = (numbers - means) / scales

    competition_ids = features["competition_id"].to_numpy()
    competition_columns = []
    for competition in competitions:
        competition_columns.append((competition_ids == competition).astype(float))

    if competition_columns:
        encoded_competitions = np.column_stack(competition_columns)
    else:
        encoded_competitions = np.empty((len(features), 0))

    intercept = np.ones((len(features), 1))
    return np.column_stack([intercept, scaled_numbers, encoded_competitions])


def fit_one_poisson_model(features, goals, alpha):
    numbers = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    means = np.mean(numbers, axis=0)
    scales = np.std(numbers, axis=0)

    # A constant column has a standard deviation of zero.  Dividing by 1 keeps
    # that column at zero after centering and avoids a division-by-zero error.
    scales[scales == 0] = 1.0

    competitions = sorted(features["competition_id"].unique())
    x = make_input_matrix(features, means, scales, competitions)
    y = np.asarray(goals, dtype=float)

    def poisson_loss(weights):
        log_expected_goals = x @ weights
        expected_goals = np.exp(np.clip(log_expected_goals, -30, 30))

        # Constants such as log(y!) do not affect which weights are best, so the
        # Poisson negative log likelihood can be written in this shorter form.
        loss = np.mean(expected_goals - y * log_expected_goals)

        # Do not regularize the intercept (weights[0]).
        penalty = 0.5 * alpha * np.sum(weights[1:] ** 2)
        return loss + penalty

    def poisson_gradient(weights):
        log_expected_goals = x @ weights
        expected_goals = np.exp(np.clip(log_expected_goals, -30, 30))
        gradient = x.T @ (expected_goals - y) / len(y)
        gradient[1:] += alpha * weights[1:]
        return gradient

    start = np.zeros(x.shape[1])
    start[0] = math.log(max(float(np.mean(y)), 0.01))
    result = minimize(
        poisson_loss,
        start,
        jac=poisson_gradient,
        method="L-BFGS-B",
    )

    if not np.isfinite(result.x).all():
        raise ValueError("Poisson model fitting did not produce finite weights")

    return {
        "weights": result.x,
        "means": means,
        "scales": scales,
        "competitions": competitions,
    }


def fit_poisson_models(features, alpha=0.1):
    # Fit one model for home goals and another for away goals.
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")

    training = features
    if "feature_supported" in training.columns:
        training = training.loc[training["feature_supported"].fillna(False)].copy()
    if training.empty:
        raise ValueError("No supported feature rows are available for fitting")

    check_rows(training, need_goals=True)
    return {
        "home": fit_one_poisson_model(training, training["home_goals"], alpha),
        "away": fit_one_poisson_model(training, training["away_goals"], alpha),
        "rolling_window": 0,
        "alpha": alpha,
    }


def predict_one_poisson_model(model, features):
    x = make_input_matrix(
        features,
        model["means"],
        model["scales"],
        model["competitions"],
    )
    return np.exp(np.clip(x @ model["weights"], -30, 30))


def predict_goals(model, features):
    # Predict expected home and away goals for each feature row.
    if "feature_supported" in features.columns:
        supported = features["feature_supported"].fillna(False)
        if not supported.all():
            raise ValueError("Cannot predict unsupported feature rows")

    check_rows(features, need_goals=False)
    home = predict_one_poisson_model(model["home"], features)
    away = predict_one_poisson_model(model["away"], features)

    if not np.isfinite(home).all() or not np.isfinite(away).all():
        raise ValueError("Predicted Poisson lambdas must be finite")
    if (home <= 0).any() or (away <= 0).any():
        raise ValueError("Predicted Poisson lambdas must be positive")

    return home, away


def chronological_split(features, train_fraction=0.6, validation_fraction=0.2):
    # Split complete dates into old training, validation, and test periods.
    fractions_are_valid = (
        math.isfinite(train_fraction)
        and math.isfinite(validation_fraction)
        and train_fraction > 0
        and validation_fraction > 0
    )
    if not fractions_are_valid:
        raise ValueError("Split fractions must be finite and positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("The final test period must be larger than zero")

    ordered = features.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    dates = list(ordered["match_date"].drop_duplicates().sort_values())
    if len(dates) < 5:
        raise ValueError("At least five separate match dates are required")

    matches_per_date = ordered.groupby("match_date", sort=True).size()
    cumulative_matches = matches_per_date.cumsum().to_numpy()

    train_target = len(ordered) * train_fraction
    validation_target = len(ordered) * (train_fraction + validation_fraction)
    train_end = int(np.argmin(np.abs(cumulative_matches - train_target))) + 1
    validation_end = int(np.argmin(np.abs(cumulative_matches - validation_target))) + 1

    train_end = min(max(train_end, 1), len(dates) - 2)
    validation_end = min(max(validation_end, train_end + 1), len(dates) - 1)

    train_dates = dates[:train_end]
    validation_dates = dates[train_end:validation_end]
    test_dates = dates[validation_end:]

    train = ordered.loc[ordered["match_date"].isin(train_dates)].copy()
    validation = ordered.loc[ordered["match_date"].isin(validation_dates)].copy()
    test = ordered.loc[ordered["match_date"].isin(test_dates)].copy()
    return train, validation, test


def validation_log_loss(features, home_lambdas, away_lambdas):
    # Measure how much probability was given to the result that happened.
    if len(features) != len(home_lambdas) or len(features) != len(away_lambdas):
        raise ValueError("Each validation row requires one home and away lambda")

    losses = []
    for i, match in enumerate(features.itertuples(index=False)):
        max_score = required_score_cutoff(home_lambdas[i], away_lambdas[i])
        probabilities = calculate_probabilities(
            home_lambdas[i],
            away_lambdas[i],
            max_score=max_score,
        )

        if match.home_goals > match.away_goals:
            actual_probability = probabilities["home_win"]
        elif match.home_goals == match.away_goals:
            actual_probability = probabilities["draw"]
        else:
            actual_probability = probabilities["away_win"]

        losses.append(-math.log(max(actual_probability, 1e-15)))

    return float(np.mean(losses))


def tune_poisson_models(matches, rolling_windows=(3, 5, 8), alphas=(0.01, 0.1, 1.0)):
    # Try a small settings grid and keep the lowest validation log loss.
    tuning_rows = []
    best = None

    for window in rolling_windows:
        features = build_features(matches, rolling_window=window)
        train, validation, test = chronological_split(features)

        for alpha in alphas:
            model = fit_poisson_models(train, alpha=alpha)
            home, away = predict_goals(model, validation)
            loss = validation_log_loss(validation, home, away)
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
                    "train": train,
                    "validation": validation,
                    "test": test,
                }

    if best is None:
        raise ValueError("No tuning configurations were supplied")

    final_training = pd.concat([best["train"], best["validation"]], ignore_index=True)
    final_model = fit_poisson_models(final_training, alpha=best["alpha"])
    final_model["rolling_window"] = best["window"]

    results = pd.DataFrame(tuning_rows).sort_values("validation_log_loss")
    return {
        "model": final_model,
        "best_window": best["window"],
        "best_alpha": best["alpha"],
        "validation_log_loss": best["loss"],
        "results": results.reset_index(drop=True),
        "test_features": best["test"],
    }


def tune_external_poisson_models(
    training_matches,
    recent_matches,
    validation_season="2024/2025",
    test_season="2025/2026",
    rolling_windows=(3, 5, 8),
    alphas=(0.01, 0.1, 1.0),
):
    # Tune on one recent season and leave the following season untouched.
    if validation_season == test_season:
        raise ValueError("Validation and test seasons must be different")

    tuning_rows = []
    best = None

    for window in rolling_windows:
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
        if training_features["match_date"].max() >= validation["match_date"].min():
            raise ValueError("Training matches must end before validation starts")
        if validation["match_date"].max() >= test["match_date"].min():
            raise ValueError("Validation matches must end before testing starts")

        for alpha in alphas:
            model = fit_poisson_models(training_features, alpha=alpha)
            home, away = predict_goals(model, validation)
            loss = validation_log_loss(validation, home, away)
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
                    "train": training_features,
                    "validation": validation,
                    "test": test,
                }

    if best is None:
        raise ValueError("No tuning configurations were supplied")

    final_training = pd.concat([best["train"], best["validation"]], ignore_index=True)
    final_model = fit_poisson_models(final_training, alpha=best["alpha"])
    final_model["rolling_window"] = best["window"]

    results = pd.DataFrame(tuning_rows).sort_values("validation_log_loss")
    return {
        "model": final_model,
        "best_window": best["window"],
        "best_alpha": best["alpha"],
        "validation_log_loss": best["loss"],
        "results": results.reset_index(drop=True),
        "test_features": best["test"],
    }
