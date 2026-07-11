from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from football_prediction.data import sync as sync_module
from football_prediction.data.loader import load_manifest, load_matches
from football_prediction.data.process import (
    is_selected_competition,
    process_repository,
)
from football_prediction.data.schema import DataValidationError, validate_matches
from football_prediction.data.sync import SynchronizationError, sync_repository


FIXTURE_REPOSITORY = Path(__file__).parent / "fixtures" / "statsbomb"


def test_top_five_selection_is_explicit_and_mens_only() -> None:
    assert is_selected_competition(
        {
            "country_name": "England",
            "competition_name": "Premier League",
            "competition_gender": "male",
        }
    )
    assert not is_selected_competition(
        {
            "country_name": "England",
            "competition_name": "FA Women's Super League",
            "competition_gender": "female",
        }
    )
    assert not is_selected_competition(
        {
            "country_name": "Europe",
            "competition_name": "Champions League",
            "competition_gender": "male",
        }
    )


def test_repository_processing_aggregates_xg_and_preserves_official_score(
    tmp_path: Path,
) -> None:
    matches_file = tmp_path / "processed" / "matches.parquet"
    manifest_file = tmp_path / "processed" / "manifest.json"

    summary = process_repository(
        FIXTURE_REPOSITORY,
        matches_file,
        manifest_file,
        source_commit="fixture-sha",
    )
    matches = load_matches(matches_file)

    assert summary.selected_competition_seasons == 1
    assert summary.source_matches == 3
    assert summary.processed_matches == 2
    assert summary.model_eligible_matches == 1
    assert summary.rejected_matches == 1

    first = matches.loc[matches["match_id"] == 1001].iloc[0]
    assert first["home_goals"] == 2  # penalty plus own goal in source score
    assert first["away_goals"] == 0
    assert first["home_xg"] == pytest.approx(0.86)
    assert first["away_xg"] == pytest.approx(0.2)
    assert first["home_shots"] == 2
    assert first["away_shots"] == 1
    assert bool(first["eligible_for_model"])


def test_missing_shot_xg_is_not_silently_treated_as_zero(tmp_path: Path) -> None:
    matches_file = tmp_path / "matches.parquet"
    manifest_file = tmp_path / "manifest.json"
    process_repository(
        FIXTURE_REPOSITORY,
        matches_file,
        manifest_file,
        source_commit="fixture-sha",
    )

    second = load_matches(matches_file).loc[lambda frame: frame["match_id"] == 1002].iloc[0]
    assert second["missing_shot_xg_count"] == 1
    assert not bool(second["eligible_for_model"])
    assert pd.isna(second["home_xg"])
    assert pd.isna(second["away_xg"])


def test_extra_time_match_is_rejected_and_reported(tmp_path: Path) -> None:
    matches_file = tmp_path / "matches.parquet"
    manifest_file = tmp_path / "manifest.json"
    process_repository(
        FIXTURE_REPOSITORY,
        matches_file,
        manifest_file,
        source_commit="fixture-sha",
    )

    manifest = load_manifest(manifest_file)
    assert manifest["rejected_matches"] == 1
    assert manifest["rejection_reasons"] == {"contains non-regulation periods: [3]": 1}


def test_validation_rejects_duplicate_match_ids(tmp_path: Path) -> None:
    matches_file = tmp_path / "matches.parquet"
    manifest_file = tmp_path / "manifest.json"
    process_repository(
        FIXTURE_REPOSITORY,
        matches_file,
        manifest_file,
        source_commit="fixture-sha",
    )
    matches = load_matches(matches_file)
    duplicated = pd.concat([matches, matches.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="unique"):
        validate_matches(duplicated)


def test_manifest_is_valid_json_and_records_provenance(tmp_path: Path) -> None:
    matches_file = tmp_path / "matches.parquet"
    manifest_file = tmp_path / "manifest.json"
    process_repository(
        FIXTURE_REPOSITORY,
        matches_file,
        manifest_file,
        source_commit="fixture-sha",
    )

    with manifest_file.open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["source_commit"] == "fixture-sha"
    assert manifest["processing_schema_version"] == "1"
    assert manifest["latest_match_date"] == "2015-08-15"
    assert len(manifest["output_sha256"]) == 64
    assert manifest["available_competition_seasons"] == [
        {
            "competition_id": 2,
            "competition_name": "Premier League",
            "country_name": "England",
            "season_id": 27,
            "season_name": "2015/2016",
        }
    ]

    repeated = process_repository(
        FIXTURE_REPOSITORY,
        matches_file,
        manifest_file,
        source_commit="fixture-sha",
    )
    assert repeated.output_sha256 == manifest["output_sha256"]


def test_sync_refuses_non_repository_data_directory(tmp_path: Path) -> None:
    destination = tmp_path / "statsbomb-open-data"
    destination.mkdir()
    (destination / "unexpected.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(SynchronizationError, match="not a Git checkout"):
        sync_repository(destination)

    assert (destination / "unexpected.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_first_sync_clones_to_temporary_path_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "statsbomb-open-data"

    def fake_git(arguments: list[str], cwd: Path | None = None) -> str:
        if arguments[0] == "clone":
            clone_path = Path(arguments[-1])
            assert clone_path.name.endswith(".clone-tmp")
            (clone_path / ".git").mkdir(parents=True)
            (clone_path / "data").mkdir()
            return ""
        assert arguments == ["rev-parse", "HEAD"]
        assert cwd == destination
        return "fixture-sha"

    monkeypatch.setattr(sync_module, "_run_git", fake_git)
    result = sync_repository(destination)

    assert result.commit == "fixture-sha"
    assert result.action == "cloned"
    assert (destination / "data").is_dir()
    assert not destination.with_name("statsbomb-open-data.clone-tmp").exists()
