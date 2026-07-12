import pytest

from simulation import base_lambdas, calculateSimulation


def example_stats():
    return {
        "team 1 league coef": 1.0,
        "team 2 league coef": 1.0,
        "team 1 league": "League A",
        "team 2 league": "League A",
        "homeAdvantage": 1.2,
        "awayDisadvantage": 0.9,
        "xG form 1": 1.4,
        "xG szn avg 1": 1.5,
        "xGa form 1": 1.1,
        "xGa szn avg 1": 1.2,
        "xG form 2": 1.0,
        "xG szn avg 2": 1.1,
        "xGa form 2": 1.3,
        "xGa szn avg 2": 1.4,
        "xG h2h team 1": 0.0,
        "xG h2h team 2": 0.0,
    }


def test_base_lambdas_use_the_no_head_to_head_weights():
    home, away = base_lambdas(example_stats())

    assert home == pytest.approx(1.40)
    assert away == pytest.approx(1.10)


def test_simulation_rates_cover_every_result():
    stats = example_stats()
    home, draw, away, home_league, away_league = calculateSimulation(stats)

    assert home + draw + away == pytest.approx(1.0)
    assert stats["xG 1"] > 0
    assert stats["xG 2"] > 0
    assert home_league == "League A"
    assert away_league == "League A"
