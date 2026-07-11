from __future__ import annotations

import pytest

from football_prediction.probabilities import calculate_probabilities


def test_poisson_probabilities_sum_to_one() -> None:
    result = calculate_probabilities(1.5, 1.1)
    total = result.home_win + result.draw + result.away_win
    assert total == pytest.approx(1.0)
    assert result.score_matrix.sum() == pytest.approx(1.0)


def test_equal_lambdas_give_equal_home_and_away_probabilities() -> None:
    result = calculate_probabilities(1.4, 1.4)
    assert result.home_win == pytest.approx(result.away_win)


def test_larger_home_lambda_increases_home_win_probability() -> None:
    lower = calculate_probabilities(1.0, 1.2)
    higher = calculate_probabilities(2.0, 1.2)
    assert higher.home_win > lower.home_win


def test_small_score_limit_rejects_a_large_missing_tail() -> None:
    with pytest.raises(ValueError, match="captures only"):
        calculate_probabilities(5.0, 4.0, max_score=3)
