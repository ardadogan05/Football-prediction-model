# Turn expected goals into score and match-result probabilities.

import math

import numpy as np
from scipy.stats import poisson


MAX_SCORE = 10


def calculate_probabilities(lambda_home, lambda_away):
    # A Poisson lambda is the expected number of goals scored by one team.
    lambdas_are_valid = (
        math.isfinite(lambda_home)
        and math.isfinite(lambda_away)
        and lambda_home > 0
        and lambda_away > 0
    )
    if not lambdas_are_valid:
        raise ValueError("Poisson lambdas must be finite and positive")

    goals = np.arange(MAX_SCORE + 1)
    home_goal_probabilities = poisson.pmf(goals, lambda_home)
    away_goal_probabilities = poisson.pmf(goals, lambda_away)

    # Cell [i, j] is the probability of home i - j away.
    score_matrix = np.outer(home_goal_probabilities, away_goal_probabilities)
    captured_mass = float(np.sum(score_matrix))

    # Normalizing the tiny omitted tail makes the three result probabilities sum to 1.
    score_matrix = score_matrix / captured_mass
    home_win = float(np.sum(np.tril(score_matrix, k=-1)))
    draw = float(np.sum(np.diag(score_matrix)))
    away_win = float(np.sum(np.triu(score_matrix, k=1)))

    highest_cell = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    most_likely_score = (int(highest_cell[0]), int(highest_cell[1]))

    return {
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "most_likely_score": most_likely_score,
        "score_matrix": score_matrix,
        "captured_mass": captured_mass,
    }
