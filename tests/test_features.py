from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_prediction.features import FEATURE_COLUMNS, build_features


def example_matches() -> pd.DataFrame:
    rows = [
        (1, "2020-01-01", 1, 2, 2, 1),
        (2, "2020-01-01", 3, 4, 1, 0),
        (3, "2020-01-08", 2, 1, 1, 1),
        (4, "2020-01-08", 4, 3, 0, 1),
        (5, "2020-01-15", 1, 3, 2, 1),
        (6, "2020-01-22", 2, 4, 1, 2),
    ]
    matches = []
    for match_id, date, home_id, away_id, home_goals, away_goals in rows:
        matches.append(
            {
                "match_id": match_id,
                "match_date": date,
                "competition_id": 10,
                "competition_name": "Example League",
                "season_id": 20,
                "season_name": "2019/2020",
                "home_team_id": home_id,
                "home_team_name": f"Team {home_id}",
                "away_team_id": away_id,
                "away_team_name": f"Team {away_id}",
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    return pd.DataFrame(matches)


def test_features_use_only_matches_from_earlier_dates() -> None:
    features = build_features(example_matches(), rolling_window=3)

    first_date = features.loc[features["match_date"] == pd.Timestamp("2020-01-01")]
    assert first_date["home_rolling_goals_for"].isna().all()
    assert first_date["away_rolling_goals_for"].isna().all()
    assert not first_date["feature_supported"].any()

    second_date_match = features.loc[features["match_id"] == 3].iloc[0]
    assert second_date_match["home_rolling_goals_for"] == pytest.approx(1.0)
    assert second_date_match["home_rolling_goals_against"] == pytest.approx(2.0)
    assert second_date_match["away_rolling_goals_for"] == pytest.approx(2.0)
    assert second_date_match["away_rolling_goals_against"] == pytest.approx(1.0)
    assert second_date_match["home_season_matches"] == 1
    assert second_date_match["away_season_matches"] == 1
    assert bool(second_date_match["feature_supported"])


def test_current_and_future_goals_cannot_change_earlier_features() -> None:
    original_matches = example_matches()
    original = build_features(original_matches, rolling_window=3)

    changed_matches = original_matches.copy()
    changed_matches.loc[
        changed_matches["match_id"] >= 5, ["home_goals", "away_goals"]
    ] = 9
    changed = build_features(changed_matches, rolling_window=3)

    columns = [
        column
        for column in original.columns
        if column not in ["home_goals", "away_goals"]
    ]
    pd.testing.assert_frame_equal(
        original.loc[original["match_id"] <= 5, columns].reset_index(drop=True),
        changed.loc[changed["match_id"] <= 5, columns].reset_index(drop=True),
    )


def test_same_date_goals_cannot_change_another_matchs_fallback_features() -> None:
    matches = example_matches()
    matches.loc[matches["match_id"] == 6, "match_date"] = "2020-01-15"
    matches.loc[matches["match_id"] == 6, "home_team_id"] = 5
    matches.loc[matches["match_id"] == 6, "home_team_name"] = "Team 5"
    matches.loc[matches["match_id"] == 6, "away_team_id"] = 6
    matches.loc[matches["match_id"] == 6, "away_team_name"] = "Team 6"

    original = build_features(matches, rolling_window=3)
    changed_matches = matches.copy()
    changed_matches.loc[
        changed_matches["match_id"] == 5, ["home_goals", "away_goals"]
    ] = 99
    changed = build_features(changed_matches, rolling_window=3)

    compared_columns = [*FEATURE_COLUMNS, "feature_supported"]
    pd.testing.assert_series_equal(
        original.loc[original["match_id"] == 6, compared_columns].iloc[0],
        changed.loc[changed["match_id"] == 6, compared_columns].iloc[0],
    )
    assert bool(original.loc[original["match_id"] == 6, "feature_supported"].iloc[0])


def test_season_counts_reset_without_resetting_competition_history() -> None:
    matches = example_matches()
    matches.loc[matches["match_id"] >= 5, "season_id"] = 21
    matches.loc[matches["match_id"] >= 5, "season_name"] = "2020/2021"
    matches.loc[matches["match_id"] == 6, "home_team_id"] = 3
    matches.loc[matches["match_id"] == 6, "home_team_name"] = "Team 3"
    matches.loc[matches["match_id"] == 6, "away_team_id"] = 1
    matches.loc[matches["match_id"] == 6, "away_team_name"] = "Team 1"

    features = build_features(matches, rolling_window=3)
    season_opener = features.loc[features["match_id"] == 5].iloc[0]
    later_match = features.loc[features["match_id"] == 6].iloc[0]

    assert season_opener["home_history_matches"] == 2
    assert season_opener["away_history_matches"] == 2
    assert season_opener["home_season_matches"] == 0
    assert season_opener["away_season_matches"] == 0
    assert later_match["home_season_matches"] == 1
    assert later_match["away_season_matches"] == 1


def test_feature_supported_matches_finiteness_of_model_features() -> None:
    features = build_features(example_matches(), rolling_window=3)
    finite_features = np.isfinite(
        features[FEATURE_COLUMNS].to_numpy(dtype=float)
    ).all(axis=1)

    np.testing.assert_array_equal(features["feature_supported"], finite_features)


def test_team_ids_from_different_sources_do_not_share_history() -> None:
    matches = example_matches()
    matches["source"] = "statsbomb"
    new_source_match = matches.iloc[[0]].copy()
    new_source_match["source"] = "football_data"
    new_source_match["match_id"] = 100
    new_source_match["match_date"] = "2020-02-01"
    matches = pd.concat([matches, new_source_match], ignore_index=True)

    features = build_features(matches, rolling_window=3)
    new_source_features = features.loc[features["match_id"] == 100].iloc[0]

    assert new_source_features["home_history_matches"] == 0
    assert new_source_features["away_history_matches"] == 0
    assert not bool(new_source_features["feature_supported"])


def test_invalid_rolling_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_features(example_matches(), rolling_window=0)
