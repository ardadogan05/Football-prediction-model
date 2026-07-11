from __future__ import annotations

import pandas as pd
import pytest

from football_prediction.features import build_features


def example_matches() -> pd.DataFrame:
    rows = [
        (1, "2020-01-01", 1, 2, 2.0, 1.0, 2, 1),
        (2, "2020-01-01", 3, 4, 1.5, 0.5, 1, 0),
        (3, "2020-01-08", 2, 1, 0.8, 1.2, 1, 1),
        (4, "2020-01-08", 4, 3, 0.7, 1.0, 0, 1),
        (5, "2020-01-15", 1, 3, 1.7, 0.9, 2, 1),
        (6, "2020-01-22", 2, 4, 1.1, 1.3, 1, 2),
    ]
    matches = []
    for match_id, date, home_id, away_id, home_xg, away_xg, home_goals, away_goals in rows:
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
                "home_xg": home_xg,
                "away_xg": away_xg,
                "eligible_for_model": True,
            }
        )
    return pd.DataFrame(matches)


def test_features_use_only_matches_from_earlier_dates() -> None:
    features = build_features(example_matches(), rolling_window=3)

    first_date = features.loc[features["match_date"] == pd.Timestamp("2020-01-01")]
    assert first_date["home_rolling_xg_for"].isna().all()
    assert first_date["away_rolling_xg_for"].isna().all()

    second_date_match = features.loc[features["match_id"] == 3].iloc[0]
    assert second_date_match["home_rolling_xg_for"] == pytest.approx(1.0)
    assert second_date_match["home_rolling_xg_against"] == pytest.approx(2.0)
    assert second_date_match["away_rolling_xg_for"] == pytest.approx(2.0)
    assert second_date_match["away_rolling_xg_against"] == pytest.approx(1.0)


def test_current_and_future_xg_cannot_change_earlier_features() -> None:
    original_matches = example_matches()
    original = build_features(original_matches, rolling_window=3)

    changed_matches = original_matches.copy()
    changed_matches.loc[changed_matches["match_id"] >= 5, ["home_xg", "away_xg"]] = 9.0
    changed = build_features(changed_matches, rolling_window=3)

    columns = [column for column in original.columns if column not in ["home_goals", "away_goals"]]
    pd.testing.assert_frame_equal(
        original.loc[original["match_id"] <= 5, columns].reset_index(drop=True),
        changed.loc[changed["match_id"] <= 5, columns].reset_index(drop=True),
    )


def test_invalid_rolling_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        build_features(example_matches(), rolling_window=0)
