"""Turn fitted Poisson goal rates into score and result probabilities."""

from __future__ import annotations

from dataclasses import dataclass

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


def calculate_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_score: int = 10,
    minimum_captured_mass: float = 0.99,
) -> MatchProbabilities:
    """Calculate deterministic match probabilities from two Poisson means."""

    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Poisson lambdas must be positive")
    if max_score < 1:
        raise ValueError("max_score must be at least 1")

    goals = np.arange(max_score + 1)
    home_probabilities = poisson.pmf(goals, lambda_home)
    away_probabilities = poisson.pmf(goals, lambda_away)
    matrix = np.outer(home_probabilities, away_probabilities)
    captured_mass = float(matrix.sum())

    if captured_mass < minimum_captured_mass:
        raise ValueError(
            f"max_score={max_score} captures only {captured_mass:.3%} probability"
        )

    #normalizing adds the very small missing tail back into the displayed matrix.
    matrix = matrix / captured_mass
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    most_likely_index = np.unravel_index(np.argmax(matrix), matrix.shape)

    return MatchProbabilities(
        lambda_home=float(lambda_home),
        lambda_away=float(lambda_away),
        home_win=home_win,
        draw=draw,
        away_win=away_win,
        most_likely_score=(int(most_likely_index[0]), int(most_likely_index[1])),
        score_matrix=matrix,
        captured_mass=captured_mass,
    )
