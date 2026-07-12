import numpy as np
import pytest

from football_prediction.probabilities import calculate_probabilities


def test_score_matrix_is_nonnegative_and_probabilities_sum_to_one():
    result = calculate_probabilities(1.5, 1.1)
    total = result["home_win"] + result["draw"] + result["away_win"]

    assert np.all(result["score_matrix"] >= 0)
    assert result["score_matrix"].shape == (11, 11)
    assert result["score_matrix"].sum() == pytest.approx(1.0)
    assert total == pytest.approx(1.0)


def test_symmetric_lambdas_give_symmetric_result_probabilities():
    result = calculate_probabilities(1.4, 1.4)
    assert result["home_win"] == pytest.approx(result["away_win"])


def test_larger_home_lambda_increases_home_win_probability():
    lower = calculate_probabilities(1.0, 1.2)
    higher = calculate_probabilities(2.0, 1.2)
    assert higher["home_win"] > lower["home_win"]


def test_most_likely_score_is_deterministic():
    first = calculate_probabilities(1.5, 1.1)
    second = calculate_probabilities(1.5, 1.1)
    assert first["most_likely_score"] == second["most_likely_score"]
    np.testing.assert_array_equal(first["score_matrix"], second["score_matrix"])


def test_fixed_matrix_captures_realistic_football_lambdas():
    for home_lambda, away_lambda in [(0.5, 0.5), (1.5, 1.1), (3.0, 2.5)]:
        result = calculate_probabilities(home_lambda, away_lambda)
        assert result["captured_mass"] >= 0.99


def test_invalid_lambdas_are_rejected():
    for invalid_lambda in [float("nan"), float("inf"), 0.0, -1.0]:
        with pytest.raises(ValueError, match="finite and positive"):
            calculate_probabilities(invalid_lambda, 1.0)
