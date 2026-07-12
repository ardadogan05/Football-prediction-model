# Legacy simulation used by the old desktop interface.
# The fitted model under src/football_prediction is the main model.

import numpy as np


NUMBER_OF_SIMULATIONS = 10000


def base_lambdas(stats):
    has_head_to_head = stats["xG h2h team 1"] != 0 or stats["xG h2h team 2"] != 0

    if has_head_to_head:
        lambda_home = (
            0.1875 * stats["xG form 1"]
            + 0.2626 * stats["xG szn avg 1"]
            + 0.1875 * stats["xGa form 2"]
            + 0.2625 * stats["xGa szn avg 2"]
            + 0.10 * stats["xG h2h team 1"]
        )
        lambda_away = (
            0.1875 * stats["xG form 2"]
            + 0.2625 * stats["xG szn avg 2"]
            + 0.1875 * stats["xGa form 1"]
            + 0.2625 * stats["xGa szn avg 1"]
            + 0.10 * stats["xG h2h team 2"]
        )
    else:
        lambda_home = (
            0.20 * stats["xG form 1"]
            + 0.30 * stats["xG szn avg 1"]
            + 0.30 * stats["xGa form 2"]
            + 0.20 * stats["xGa szn avg 2"]
        )
        lambda_away = (
            0.20 * stats["xG form 2"]
            + 0.30 * stats["xG szn avg 2"]
            + 0.30 * stats["xGa form 1"]
            + 0.20 * stats["xGa szn avg 1"]
        )

    return lambda_home, lambda_away


def increase_lambda_gap(lambda_home, lambda_away):
    difference = lambda_home - lambda_away
    absolute_difference = abs(difference)

    if absolute_difference <= 0.15:
        beta = 1.8
    elif absolute_difference <= 0.45:
        beta = 1.7
    else:
        beta = 0.75

    boost = min(absolute_difference * beta, 0.8)
    multiplier = 1 + boost / 2

    if difference >= 0:
        return lambda_home * multiplier, lambda_away / multiplier
    return lambda_home / multiplier, lambda_away * multiplier


def calculateSimulation(stats):
    # Return simulated home, draw, and away rates for the legacy GUI.
    lambda_home, lambda_away = base_lambdas(stats)
    lambda_home *= stats["homeAdvantage"]
    lambda_away *= stats["awayDisadvantage"]

    home_league_coefficient = stats["team 1 league coef"]
    away_league_coefficient = stats["team 2 league coef"]
    if home_league_coefficient != away_league_coefficient:
        lambda_home *= home_league_coefficient
        lambda_away *= away_league_coefficient
    lambda_home, lambda_away = increase_lambda_gap(lambda_home, lambda_away)

    stats["xG 1"] = lambda_home
    stats["xG 2"] = lambda_away

    home_goals = np.random.poisson(lambda_home, NUMBER_OF_SIMULATIONS)
    away_goals = np.random.poisson(lambda_away, NUMBER_OF_SIMULATIONS)

    home_win_rate = float(np.mean(home_goals > away_goals))
    draw_rate = float(np.mean(home_goals == away_goals))
    away_win_rate = float(np.mean(home_goals < away_goals))

    return (
        home_win_rate,
        draw_rate,
        away_win_rate,
        stats["team 1 league"],
        stats["team 2 league"],
    )
