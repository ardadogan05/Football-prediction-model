import pandas as pd
import pytest

from football_prediction.features import FEATURE_COLUMNS, build_features


def example_matches():
    #A small repeating league is enough to build several matches of history.
    matches = []
    for index in range(16):
        home_id = (index % 4) + 1
        away_id = ((index + 1) % 4) + 1
        matches.append(
            {
                "source": "example",
                "match_id": index + 1,
                "match_date": pd.Timestamp("2024-01-01")
                + pd.Timedelta(index * 7, unit="D"),
                "competition_id": 2,
                "competition_name": "Premier League",
                "season_id": 24,
                "season_name": "2024/2025",
                "home_team_id": home_id,
                "home_team_name": f"Team {home_id}",
                "away_team_id": away_id,
                "away_team_name": f"Team {away_id}",
                "home_goals": index % 4,
                "away_goals": (index + 1) % 3,
            }
        )
    return pd.DataFrame(matches)


def test_current_and_future_goals_do_not_change_current_features():
    matches = example_matches()
    original = build_features(matches, rolling_window=3)

    #Changing match 9 and later must not change features made for match 9 or earlier.
    changed_matches = matches.copy()
    changed_matches.loc[changed_matches["match_id"] >= 9, ["home_goals", "away_goals"]] = 99
    changed = build_features(changed_matches, rolling_window=3)

    compared_columns = FEATURE_COLUMNS.copy()
    compared_columns.extend(["home_season_matches", "away_season_matches"])
    pd.testing.assert_frame_equal(
        original.loc[original["match_id"] <= 9, compared_columns],
        changed.loc[changed["match_id"] <= 9, compared_columns],
    )


def test_same_date_matches_do_not_leak_into_each_other():
    matches = example_matches()
    matches.loc[matches["match_id"] == 10, "match_date"] = matches.loc[
        matches["match_id"] == 9, "match_date"
    ].iloc[0]
    original = build_features(matches, rolling_window=3)

    #Match 10 is on the same date, so it must not know match 9's changed score.
    changed_matches = matches.copy()
    changed_matches.loc[changed_matches["match_id"] == 9, ["home_goals", "away_goals"]] = 99
    changed = build_features(changed_matches, rolling_window=3)

    original_row = original.loc[original["match_id"] == 10, FEATURE_COLUMNS].iloc[0]
    changed_row = changed.loc[changed["match_id"] == 10, FEATURE_COLUMNS].iloc[0]
    pd.testing.assert_series_equal(original_row, changed_row)


def test_rolling_features_use_only_the_requested_window():
    matches = example_matches()
    features = build_features(matches, rolling_window=2, minimum_history=1)
    target = features.loc[features["match_id"] == 13].iloc[0]

    earlier = matches.loc[matches["match_date"] < target["match_date"]]
    team_id = int(target["home_team_id"])
    goals_for = []
    for match in earlier.itertuples(index=False):
        if match.home_team_id == team_id:
            goals_for.append(match.home_goals)
        elif match.away_team_id == team_id:
            goals_for.append(match.away_goals)

    #Only the last two earlier results should be part of this average.
    expected = sum(goals_for[-2:]) / 2
    assert target["home_rolling_goals_for"] == pytest.approx(expected)


def test_season_statistics_reset_for_a_new_season():
    matches = example_matches()
    matches.loc[matches["match_id"] >= 13, "season_id"] = 25
    matches.loc[matches["match_id"] >= 13, "season_name"] = "2025/2026"
    features = build_features(matches, rolling_window=3)
    season_opener = features.loc[features["match_id"] == 13].iloc[0]

    #Overall history remains, but current-season history starts from zero.
    assert season_opener["home_history_matches"] > 0
    assert season_opener["away_history_matches"] > 0
    assert season_opener["home_season_matches"] == 0
    assert season_opener["away_season_matches"] == 0
    assert not bool(season_opener["feature_supported"])


def test_rows_need_three_previous_matches_for_both_teams():
    features = build_features(example_matches(), rolling_window=3)
    early = features.loc[
        (features["home_season_matches"] < 3)
        | (features["away_season_matches"] < 3)
    ]
    supported = features.loc[features["feature_supported"]]

    #A match is usable only when both teams have enough earlier information.
    assert not early["feature_supported"].any()
    assert not supported.empty
    assert (supported["home_season_matches"] >= 3).all()
    assert (supported["away_season_matches"] >= 3).all()


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="rolling_window"):
        build_features(example_matches(), rolling_window=0)
    with pytest.raises(ValueError, match="minimum_history"):
        build_features(example_matches(), minimum_history=0)
