# Turn expected goals into score and match-result probabilities.

import math

import numpy as np
from scipy.stats import poisson


def check_probability_inputs(lambda_home, lambda_away, minimum_captured_mass):
    lambdas_are_valid = (
        math.isfinite(lambda_home)
        and math.isfinite(lambda_away)
        and lambda_home > 0
        and lambda_away > 0
    )
    if not lambdas_are_valid:
        raise ValueError("Poisson lambdas must be finite and positive")

    mass_is_valid = (
        math.isfinite(minimum_captured_mass)
        and 0 < minimum_captured_mass < 1
    )
    if not mass_is_valid:
        raise ValueError(
            "minimum_captured_mass must be finite and strictly between 0 and 1"
        )


def check_score_limit(max_score, name):
    is_integer = isinstance(max_score, (int, np.integer))
    if isinstance(max_score, bool) or not is_integer or max_score < 1:
        raise ValueError(f"{name} must be a positive integer")


def required_score_cutoff(
    lambda_home,
    lambda_away,
    minimum_captured_mass=0.99,
    minimum_max_score=10,
):
    # Find a score limit that contains enough of both Poisson distributions.
    check_probability_inputs(lambda_home, lambda_away, minimum_captured_mass)
    check_score_limit(minimum_max_score, "minimum_max_score")

    max_score = minimum_max_score
    captured = poisson.cdf(max_score, lambda_home) * poisson.cdf(
        max_score, lambda_away
    )

    while captured < minimum_captured_mass:
        max_score += 1
        if max_score > 100:
            raise ValueError("Poisson lambdas are too large for a football score matrix")
        captured = poisson.cdf(max_score, lambda_home) * poisson.cdf(
            max_score, lambda_away
        )

    return max_score


def calculate_probabilities(
    lambda_home,
    lambda_away,
    max_score=10,
    minimum_captured_mass=0.99,
):
    # Calculate score, home-win, draw, and away-win probabilities.
    check_probability_inputs(lambda_home, lambda_away, minimum_captured_mass)
    check_score_limit(max_score, "max_score")

    goals = np.arange(max_score + 1)
    home_goal_probabilities = poisson.pmf(goals, lambda_home)
    away_goal_probabilities = poisson.pmf(goals, lambda_away)

    # matrix[i, j] is P(home scores i) * P(away scores j).
    matrix = np.outer(home_goal_probabilities, away_goal_probabilities)
    captured_mass = float(np.sum(matrix))

    if captured_mass < minimum_captured_mass:
        raise ValueError(
            f"max_score={max_score} captures only {captured_mass:.3%} probability"
        )

    # A tiny tail lies outside the matrix.  Normalize the cells so the shown
    # home/draw/away probabilities add up to exactly one.
    matrix = matrix / captured_mass
    home_win = float(np.sum(np.tril(matrix, k=-1)))
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, k=1)))

    highest_cell = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely_score = (int(highest_cell[0]), int(highest_cell[1]))

    return {
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "most_likely_score": most_likely_score,
        "score_matrix": matrix,
        "captured_mass": captured_mass,
    }
