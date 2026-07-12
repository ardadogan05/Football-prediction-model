from __future__ import annotations

import numpy as np
import pytest

from football_prediction.probabilities import (
    calculate_probabilities,
    required_score_cutoff,
)


def test_poisson_probabilities_sum_to_one() -> None:
    result = calculate_probabilities(1.5, 1.1)
    total = result["home_win"] + result["draw"] + result["away_win"]
    assert total == pytest.approx(1.0)
    assert result["score_matrix"].sum() == pytest.approx(1.0)


def test_equal_lambdas_give_equal_home_and_away_probabilities() -> None:
    result = calculate_probabilities(1.4, 1.4)
    assert result["home_win"] == pytest.approx(result["away_win"])


def test_larger_home_lambda_increases_home_win_probability() -> None:
    lower = calculate_probabilities(1.0, 1.2)
    higher = calculate_probabilities(2.0, 1.2)
    assert higher["home_win"] > lower["home_win"]


def test_small_score_limit_rejects_a_large_missing_tail() -> None:
    with pytest.raises(ValueError, match="captures only"):
        calculate_probabilities(5.0, 4.0, max_score=3)


@pytest.mark.parametrize("invalid_max_score", [True, 0, 3.5])
def test_invalid_score_limits_are_rejected(invalid_max_score: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calculate_probabilities(1.5, 1.1, max_score=invalid_max_score)


@pytest.mark.parametrize(
    "invalid_lambda",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_lambdas_are_rejected(invalid_lambda: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        calculate_probabilities(invalid_lambda, 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        calculate_probabilities(1.0, invalid_lambda)


@pytest.mark.parametrize(
    "invalid_mass",
    [float("nan"), float("inf"), -float("inf"), -0.1, 0.0, 1.0, 1.1],
)
def test_invalid_minimum_captured_mass_is_rejected(invalid_mass: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calculate_probabilities(1.5, 1.1, minimum_captured_mass=invalid_mass)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        required_score_cutoff(1.5, 1.1, minimum_captured_mass=invalid_mass)


def test_score_matrix_is_nonnegative_and_deterministic() -> None:
    first = calculate_probabilities(1.5, 1.1)
    second = calculate_probabilities(1.5, 1.1)

    assert np.all(first["score_matrix"] >= 0)
    np.testing.assert_array_equal(first["score_matrix"], second["score_matrix"])
    assert first["most_likely_score"] == second["most_likely_score"]


def test_required_score_cutoff_handles_high_lambdas() -> None:
    with pytest.raises(ValueError, match="captures only"):
        calculate_probabilities(15.0, 12.0)

    cutoff = required_score_cutoff(15.0, 12.0)

    assert cutoff > 10
    result = calculate_probabilities(15.0, 12.0, max_score=cutoff)
    assert result["captured_mass"] >= 0.99


def test_unrealistic_lambdas_do_not_create_an_enormous_matrix() -> None:
    with pytest.raises(ValueError, match="too large"):
        required_score_cutoff(1000.0, 1000.0)
