"""Turn fitted Poisson goal rates into score and result probabilities."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import poisson


@dataclass(frozen=True)
class MatchProbabilities:
    lambda_home: float
    lambda_away: float
    home_win: float
    draw: float
    away_win: float
    most_likely_score: tuple[int, int]
    score_matrix: np.ndarray
    captured_mass: float


def _validate_probability_inputs(
    lambda_home: float,
    lambda_away: float,
    minimum_captured_mass: float,
) -> None:
    lambdas_are_finite = math.isfinite(lambda_home) and math.isfinite(lambda_away)
    lambdas_are_positive = lambda_home > 0 and lambda_away > 0
    if not lambdas_are_finite or not lambdas_are_positive:
        raise ValueError("Poisson lambdas must be finite and positive")

    captured_mass_is_valid = (
        math.isfinite(minimum_captured_mass)
        and minimum_captured_mass > 0
        and minimum_captured_mass < 1
    )
    if not captured_mass_is_valid:
        raise ValueError(
            "minimum_captured_mass must be finite and strictly between 0 and 1"
        )


def required_score_cutoff(
    lambda_home: float,
    lambda_away: float,
    minimum_captured_mass: float = 0.99,
    minimum_max_score: int = 10,
) -> int:
    """Increase the score limit when 0-10 leaves out too much probability."""

    _validate_probability_inputs(
        lambda_home,
        lambda_away,
        minimum_captured_mass,
    )
    if (
        isinstance(minimum_max_score, bool)
        or not isinstance(minimum_max_score, (int, np.integer))
        or minimum_max_score < 1
    ):
        raise ValueError("minimum_max_score must be a positive integer")

    # Start at the normal 0-10 score range. Add one goal at a time until the
    # matrix contains enough of the two Poisson distributions.
    cutoff = minimum_max_score
    captured_mass = float(
        poisson.cdf(cutoff, lambda_home) * poisson.cdf(cutoff, lambda_away)
    )
    while captured_mass < minimum_captured_mass:
        cutoff += 1
        # More than 100 goals is not a meaningful football score matrix.
        if cutoff > 100:
            raise ValueError("Poisson lambdas are too large for a football score matrix")
        captured_mass = float(
            poisson.cdf(cutoff, lambda_home) * poisson.cdf(cutoff, lambda_away)
        )
    return cutoff


def calculate_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_score: int = 10,
    minimum_captured_mass: float = 0.99,
) -> MatchProbabilities:
    """Calculate deterministic match probabilities from two Poisson means."""

    _validate_probability_inputs(
        lambda_home,
        lambda_away,
        minimum_captured_mass,
    )
    if (
        isinstance(max_score, bool)
        or not isinstance(max_score, (int, np.integer))
        or max_score < 1
    ):
        raise ValueError("max_score must be a positive integer")

    goals = np.arange(max_score + 1)
    home_probabilities = poisson.pmf(goals, lambda_home)
    away_probabilities = poisson.pmf(goals, lambda_away)

    # Each cell is: P(home scores i) multiplied by P(away scores j).
    matrix = np.outer(home_probabilities, away_probabilities)
    captured_mass = float(matrix.sum())

    # A score limit that cuts off too much probability would be misleading.
    if captured_mass < minimum_captured_mass:
        raise ValueError(
            f"max_score={max_score} captures only {captured_mass:.3%} probability"
        )

    # The cutoff removes a tiny tail. Normalize so displayed probabilities sum to 1.
    matrix = matrix / captured_mass
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    highest_probability = 0.0
    most_likely_score = (0, 0)

    for home_goals in range(max_score + 1):
        for away_goals in range(max_score + 1):
            score_probability = float(matrix[home_goals, away_goals])
            if home_goals > away_goals:
                home_win += score_probability
            elif home_goals == away_goals:
                draw += score_probability
            else:
                away_win += score_probability

            if score_probability > highest_probability:
                highest_probability = score_probability
                most_likely_score = (home_goals, away_goals)

    return MatchProbabilities(
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        most_likely_score=most_likely_score,
        score_matrix=matrix,
        captured_mass=captured_mass,
    )
