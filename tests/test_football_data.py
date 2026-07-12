from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pandas as pd
import pytest

from football_prediction.config import ProjectPaths
from football_prediction.data import football_data as football_data_module
from football_prediction.data.football_data import (
    DEFAULT_COMPETITION_CODES,
    DEFAULT_SEASONS,
    FOOTBALL_DATA_COLUMNS,
    FOOTBALL_DATA_COMPETITIONS,
    FootballDataError,
    load_api_key,
    load_football_matches,
    sync_football_data,
)


def example_payload() -> dict:
    return {
        "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
        "matches": [
            {
                "id": 5001,
                "utcDate": "2025-01-11T15:00:00Z",
                "status": "FINISHED",
                "season": {"id": 2292, "startDate": "2024-08-16"},
                "homeTeam": {"id": 61, "name": "Chelsea FC"},
                "awayTeam": {"id": 64, "name": "Liverpool FC"},
                "score": {
                    "duration": "REGULAR",
                    "fullTime": {"home": 2, "away": 1},
                },
            },
            {
                "id": 5002,
                "utcDate": "2025-01-18T15:00:00Z",
                "status": "FINISHED",
                "score": {
                    "duration": "EXTRA_TIME",
                    "fullTime": {"home": 2, "away": 1},
                },
            },
        ],
    }


class FakeResponse:
    def __init__(self, payload: dict, headers: dict[str, str] | None = None) -> None:
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_api_key_prefers_environment_and_falls_back_to_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        'FOOTBALL_DATA_API_KEY="dotenv-secret"\n', encoding="utf-8"
    )

    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "environment-secret")
    assert load_api_key(tmp_path) == "environment-secret"

    monkeypatch.delenv("FOOTBALL_DATA_API_KEY")
    assert load_api_key(tmp_path) == "dotenv-secret"


def test_rate_limit_headers_sleep_only_when_nearly_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = []
    monkeypatch.setattr(football_data_module.time, "sleep", sleeps.append)

    football_data_module._respect_rate_limit(
        {"x-requests-available-minute": "2", "X-RequestCounter-Reset": "30"}
    )
    football_data_module._respect_rate_limit(
        {"x-requests-available-minute": "1", "X-RequestCounter-Reset": "2.5"}
    )
    football_data_module._respect_rate_limit(
        {"X-RequestsAvailable": "0", "X-RequestCounter-Reset": "3"}
    )
    football_data_module._respect_rate_limit(
        {"X-RequestCounter-Reset": "99"}
    )

    assert sleeps == [2.75, 3.25]


def test_download_normalizes_finished_matches_and_reuses_raw_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    requests: list[Request] = []
    secret = "not-for-output"

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        assert timeout == 60
        requests.append(request)
        return FakeResponse(example_payload(), {"X-RequestsAvailable": "2"})

    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", secret)
    monkeypatch.setattr(football_data_module, "urlopen", fake_urlopen)
    summary = sync_football_data(
        paths,
        competition_codes=("PL",),
        seasons=(2024,),
    )

    assert summary.requested_files == 1
    assert summary.downloaded_files == 1
    assert summary.cached_files == 0
    assert summary.source_matches == 2
    assert summary.finished_matches == 1
    assert summary.latest_match_date == "2025-01-11"
    assert len(requests) == 1
    assert requests[0].full_url.endswith("/competitions/PL/matches?season=2024")
    request_headers = {key.lower(): value for key, value in requests[0].header_items()}
    assert request_headers["x-auth-token"] == secret

    matches = load_football_matches(paths.football_data_matches_file)
    assert tuple(matches.columns) == FOOTBALL_DATA_COLUMNS
    match = matches.iloc[0]
    assert match["source"] == "football_data"
    assert match["match_id"] == 5001
    assert match["source_match_id"] == 5001
    assert match["competition_id"] == 2
    assert match["competition_name"] == "Premier League"
    assert match["source_competition_id"] == 2021
    assert match["season_id"] == 2292
    assert match["source_season_id"] == 2292
    assert match["season_start_year"] == 2024
    assert match["season_name"] == "2024/2025"
    assert match["source_home_team_id"] == 61
    assert match["source_home_team_name"] == "Chelsea FC"
    assert match["home_team_id"] == 61
    assert match["home_team_name"] == "Chelsea FC"
    assert match["source_away_team_id"] == 64
    assert match["source_away_team_name"] == "Liverpool FC"
    assert match["away_team_id"] == 64
    assert match["away_team_name"] == "Liverpool FC"
    assert match["home_goals"] == 2
    assert match["away_goals"] == 1

    raw_file = paths.football_data_raw_directory / "PL" / "2024.json"
    raw_before = raw_file.read_bytes()
    modified_before = raw_file.stat().st_mtime_ns
    manifest = json.loads(paths.football_data_manifest_file.read_text(encoding="utf-8"))
    assert manifest["source"] == "football_data"
    assert manifest["competition_codes"] == ["PL"]
    assert manifest["seasons"] == [2024]
    assert secret not in raw_file.read_text(encoding="utf-8")
    assert secret not in paths.football_data_manifest_file.read_text(encoding="utf-8")
    assert secret not in capsys.readouterr().out
    assert not list(tmp_path.rglob("*.tmp"))

    monkeypatch.delenv("FOOTBALL_DATA_API_KEY")
    monkeypatch.setattr(
        football_data_module,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("cached data must not use the network"),
    )
    repeated = sync_football_data(
        paths,
        competition_codes=("PL",),
        seasons=(2024,),
    )

    assert repeated.downloaded_files == 0
    assert repeated.cached_files == 1
    assert raw_file.read_bytes() == raw_before
    assert raw_file.stat().st_mtime_ns == modified_before


def test_http_429_waits_for_reset_then_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    secret = "retry-secret"
    attempts = []
    sleeps = []

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        attempts.append(request)
        if len(attempts) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"X-RequestsAvailable": "0", "X-RequestCounter-Reset": "2"},
                None,
            )
        return FakeResponse(example_payload(), {"X-RequestsAvailable": "2"})

    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", secret)
    monkeypatch.setattr(football_data_module, "urlopen", fake_urlopen)
    monkeypatch.setattr(football_data_module.time, "sleep", sleeps.append)

    summary = sync_football_data(
        paths,
        competition_codes=("PL",),
        seasons=(2024,),
    )

    assert summary.downloaded_files == 1
    assert len(attempts) == 2
    assert sleeps == [pytest.approx(2.25)]
    assert secret not in capsys.readouterr().out
    assert secret not in (
        paths.football_data_raw_directory / "PL" / "2024.json"
    ).read_text(encoding="utf-8")


def test_invalid_refresh_preserves_existing_cache_and_processed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-secret")
    monkeypatch.setattr(
        football_data_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(example_payload()),
    )
    sync_football_data(paths, competition_codes=("PL",), seasons=(2024,))

    raw_file = paths.football_data_raw_directory / "PL" / "2024.json"
    raw_before = raw_file.read_bytes()
    output_before = paths.football_data_matches_file.read_bytes()
    manifest_before = paths.football_data_manifest_file.read_bytes()
    invalid_payload = {
        "competition": {"id": 2021, "name": "Premier League", "code": "PL"}
    }
    monkeypatch.setattr(
        football_data_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(invalid_payload),
    )

    with pytest.raises(FootballDataError, match="no match list"):
        sync_football_data(
            paths,
            competition_codes=("PL",),
            seasons=(2024,),
            refresh=True,
        )

    assert raw_file.read_bytes() == raw_before
    assert paths.football_data_matches_file.read_bytes() == output_before
    assert paths.football_data_manifest_file.read_bytes() == manifest_before


@pytest.mark.parametrize(
    ("identity", "message"),
    [("competition", "competition code"), ("season", "season start year")],
)
def test_payload_identity_must_match_requested_code_and_season(
    identity: str, message: str
) -> None:
    payload = example_payload()
    if identity == "competition":
        payload["competition"]["code"] = "PD"
    else:
        payload["matches"][0]["season"]["startDate"] = "2023-08-16"

    with pytest.raises(FootballDataError, match=message):
        football_data_module._normalize_payload(payload, "PL", 2024)


def test_empty_refresh_does_not_replace_processed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-secret")
    monkeypatch.setattr(
        football_data_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(example_payload()),
    )
    sync_football_data(paths, competition_codes=("PL",), seasons=(2024,))
    output_before = paths.football_data_matches_file.read_bytes()
    manifest_before = paths.football_data_manifest_file.read_bytes()

    scheduled_only = {
        "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
        "matches": [{"id": 9001, "status": "SCHEDULED"}],
    }
    monkeypatch.setattr(
        football_data_module,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(scheduled_only),
    )

    with pytest.raises(FootballDataError, match="No finished regular-time"):
        sync_football_data(
            paths,
            competition_codes=("PL",),
            seasons=(2024,),
            refresh=True,
        )

    assert paths.football_data_matches_file.read_bytes() == output_before
    assert paths.football_data_manifest_file.read_bytes() == manifest_before


def test_loader_rejects_invalid_processed_rows(tmp_path: Path) -> None:
    records, _ = football_data_module._normalize_payload(example_payload(), "PL", 2024)
    valid = pd.DataFrame.from_records(records, columns=FOOTBALL_DATA_COLUMNS)
    invalid_frames = []

    invalid_frames.append(pd.concat([valid, valid], ignore_index=True))
    negative_goals = valid.copy()
    negative_goals.loc[0, "home_goals"] = -1
    invalid_frames.append(negative_goals)
    same_team = valid.copy()
    same_team.loc[0, "away_team_id"] = same_team.loc[0, "home_team_id"]
    invalid_frames.append(same_team)
    invalid_date = valid.copy()
    invalid_date.loc[0, "match_date"] = None
    invalid_frames.append(invalid_date)
    invalid_source = valid.copy()
    invalid_source.loc[0, "source"] = "statsbomb"
    invalid_frames.append(invalid_source)

    output = tmp_path / "football_data_matches.parquet"
    for invalid in invalid_frames:
        invalid.to_parquet(output, index=False)
        with pytest.raises(FootballDataError):
            load_football_matches(output)


def test_load_football_matches_rejects_an_incomplete_schema(tmp_path: Path) -> None:
    matches_file = tmp_path / "football_data_matches.parquet"
    pd.DataFrame([{"source": "football_data"}]).to_parquet(
        matches_file, index=False
    )

    with pytest.raises(FootballDataError, match="missing columns"):
        load_football_matches(matches_file)


def test_defaults_and_competition_mapping_match_project_contract(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)

    assert DEFAULT_COMPETITION_CODES == ("PL", "BL1", "FL1", "SA", "PD")
    assert DEFAULT_SEASONS == (2024, 2025)
    assert {
        code: details["competition_id"]
        for code, details in FOOTBALL_DATA_COMPETITIONS.items()
    } == {"PL": 2, "BL1": 9, "FL1": 7, "SA": 12, "PD": 11}
    assert paths.football_data_raw_directory == tmp_path / "data/raw/football-data"
    assert paths.football_data_matches_file == (
        tmp_path / "data/processed/football_data_matches.parquet"
    )
    assert paths.football_data_manifest_file == (
        tmp_path / "data/processed/football_data_manifest.json"
    )
