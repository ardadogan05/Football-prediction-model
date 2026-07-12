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

    #Calculate the chance of scoring 0, 1, 2 and so on up to 10 goals.
    goals = np.arange(MAX_SCORE + 1)
    home_goal_probabilities = poisson.pmf(goals, lambda_home)
    away_goal_probabilities = poisson.pmf(goals, lambda_away)

    # Cell [i, j] is the probability of home i - j away.
    score_matrix = np.outer(home_goal_probabilities, away_goal_probabilities)
    captured_mass = float(np.sum(score_matrix))

    # Normalizing the tiny omitted tail makes the three result probabilities sum to 1.
    score_matrix = score_matrix / captured_mass
    #Below the diagonal the home score is larger than the away score.
    home_win = float(np.sum(np.tril(score_matrix, k=-1)))
    #The diagonal contains scores such as 0-0, 1-1 and 2-2.
    draw = float(np.sum(np.diag(score_matrix)))
    #Above the diagonal the away score is larger than the home score.
    away_win = float(np.sum(np.triu(score_matrix, k=1)))

    #The most likely exact score is separate from the most likely 1X2 result.
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
