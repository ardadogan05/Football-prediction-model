"""Download and normalize finished football-data.org league matches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from football_prediction.config import ProjectPaths


FOOTBALL_DATA_API_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_API_KEY_ENV = "FOOTBALL_DATA_API_KEY"
DEFAULT_COMPETITION_CODES = ("PL", "BL1", "FL1", "SA", "PD")
DEFAULT_SEASONS = (2024, 2025)
MAX_DOWNLOAD_ATTEMPTS = 3
RATE_LIMIT_RETRY_MARGIN_SECONDS = 0.25

# The canonical competition fields intentionally reuse the StatsBomb IDs and names
# used by the rest of the project. Team IDs remain explicitly source-specific.
FOOTBALL_DATA_COMPETITIONS: dict[str, dict[str, int | str]] = {
    "PL": {"competition_id": 2, "competition_name": "Premier League"},
    "BL1": {"competition_id": 9, "competition_name": "1. Bundesliga"},
    "FL1": {"competition_id": 7, "competition_name": "Ligue 1"},
    "SA": {"competition_id": 12, "competition_name": "Serie A"},
    "PD": {"competition_id": 11, "competition_name": "La Liga"},
}

FOOTBALL_DATA_COLUMNS = (
    "source",
    "match_id",
    "source_match_id",
    "match_date",
    "kick_off",
    "competition_id",
    "competition_name",
    "competition_code",
    "source_competition_id",
    "source_competition_name",
    "season_id",
    "season_start_year",
    "season_name",
    "source_season_id",
    "home_team_id",
    "home_team_name",
    "source_home_team_id",
    "source_home_team_name",
    "away_team_id",
    "away_team_name",
    "source_away_team_id",
    "source_away_team_name",
    "home_goals",
    "away_goals",
)


class FootballDataError(RuntimeError):
    """Raised when football-data.org data cannot be obtained or normalized."""


@dataclass(frozen=True)
class FootballDataSummary:
    requested_files: int
    downloaded_files: int
    cached_files: int
    source_matches: int
    finished_matches: int
    latest_match_date: str | None
    output_file: str
    output_sha256: str
    manifest_file: str


def _dotenv_api_key(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise FootballDataError(f"Could not read credential file: {path}") from error

    for line in lines:
        candidate = line.strip()
        if not candidate or candidate.startswith("#"):
            continue
        if candidate.startswith("export "):
            candidate = candidate.removeprefix("export ").lstrip()
        key, separator, value = candidate.partition("=")
        if separator and key.strip() == FOOTBALL_DATA_API_KEY_ENV:
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value or None
    return None


def load_api_key(project_root: Path) -> str:
    """Load the API token from the environment, then the project's ignored .env."""

    token = os.environ.get(FOOTBALL_DATA_API_KEY_ENV, "").strip()
    if not token:
        token = _dotenv_api_key(project_root / ".env") or ""
    if not token:
        raise FootballDataError(
            f"Set {FOOTBALL_DATA_API_KEY_ENV} or add it to the project's .env file"
        )
    return token


def _atomic_json(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise FootballDataError(f"Could not read cached JSON: {path}") from error
    if not isinstance(value, dict):
        raise FootballDataError(f"Cached response is not a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_headers(headers: Any) -> dict[str, str]:
    try:
        items = headers.items()
    except AttributeError:
        return {}
    return {str(key).lower(): str(value).strip() for key, value in items}


def _respect_rate_limit(headers: Any) -> None:
    normalized = _normalized_headers(headers)
    remaining_values = []
    for name in ("x-requests-available-minute", "x-requestsavailable"):
        value = normalized.get(name)
        if value is None:
            continue
        try:
            remaining_values.append(float(value))
        except ValueError:
            continue

    if not remaining_values or min(remaining_values) > 1:
        return

    reset_value = normalized.get("x-requestcounter-reset")
    try:
        wait_seconds = max(0.0, float(reset_value)) if reset_value else 60.0
    except ValueError:
        wait_seconds = 60.0
    if wait_seconds > 0:
        # A small margin avoids sending the next request exactly on the reset edge.
        time.sleep(wait_seconds + RATE_LIMIT_RETRY_MARGIN_SECONDS)


def _wait_for_rate_limit_retry(headers: Any) -> None:
    normalized = _normalized_headers(headers)
    reset_value = normalized.get("x-requestcounter-reset")
    try:
        reset_seconds = max(0.0, float(reset_value)) if reset_value else 60.0
    except ValueError:
        reset_seconds = 60.0
    time.sleep(reset_seconds + RATE_LIMIT_RETRY_MARGIN_SECONDS)


def _download_payload(code: str, season: int, api_key: str) -> dict[str, Any]:
    query = urlencode({"season": season})
    url = f"{FOOTBALL_DATA_API_URL}/competitions/{code}/matches?{query}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "football-prediction-portfolio-project",
            "X-Auth-Token": api_key,
        },
    )
    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            with urlopen(request, timeout=60) as response:
                content = response.read()
                headers = response.headers
            break
        except HTTPError as error:
            if error.code == 429 and attempt < MAX_DOWNLOAD_ATTEMPTS - 1:
                _wait_for_rate_limit_retry(error.headers)
                continue
            raise FootballDataError(
                f"Could not download football-data.org matches for {code} {season} "
                f"(HTTP {error.code})"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise FootballDataError(
                f"Could not download football-data.org matches for {code} {season}"
            ) from error

    _respect_rate_limit(headers)
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FootballDataError(
            f"football-data.org returned invalid JSON for {code} {season}"
        ) from error
    if not isinstance(payload, dict):
        raise FootballDataError(
            f"football-data.org response is not an object for {code} {season}"
        )
    return payload


def _required_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FootballDataError(f"{context} must be a JSON object")
    return value


def _required_value(record: Mapping[str, Any], field: str, context: str) -> Any:
    value = record.get(field)
    if value is None or value == "":
        raise FootballDataError(f"{context} is missing {field}")
    return value


def _normalize_payload(
    payload: Mapping[str, Any], code: str, season_start_year: int
) -> tuple[list[dict[str, Any]], int]:
    matches = payload.get("matches")
    if not isinstance(matches, list):
        raise FootballDataError(f"Response for {code} {season_start_year} has no match list")

    canonical = FOOTBALL_DATA_COMPETITIONS[code]
    payload_competition = payload.get("competition")
    if payload_competition is None:
        payload_competition = {}
    payload_competition = _required_mapping(
        payload_competition, f"{code} {season_start_year} competition"
    )
    payload_code = str(
        _required_value(
            payload_competition,
            "code",
            f"{code} {season_start_year} competition",
        )
    ).upper()
    if payload_code != code:
        raise FootballDataError(
            f"Response for {code} {season_start_year} has competition code "
            f"{payload_code}"
        )
    records = []
    for match in matches:
        match = _required_mapping(match, f"{code} {season_start_year} match")
        if match.get("status") != "FINISHED":
            continue

        context = f"finished {code} {season_start_year} match"
        competition = _required_mapping(
            match.get("competition", payload_competition), f"{context} competition"
        )
        match_code = str(competition.get("code", payload_code)).upper()
        if match_code != code:
            raise FootballDataError(f"{context} has competition code {match_code}")
        score = _required_mapping(match.get("score"), f"{context} score")
        if score.get("duration") != "REGULAR":
            continue
        season = _required_mapping(match.get("season"), f"{context} season")
        try:
            source_season_start = pd.Timestamp(
                _required_value(season, "startDate", f"{context} season")
            )
        except (TypeError, ValueError) as error:
            raise FootballDataError(f"{context} has an invalid season startDate") from error
        if source_season_start.year != season_start_year:
            raise FootballDataError(
                f"{context} has season start year {source_season_start.year}"
            )
        home_team = _required_mapping(match.get("homeTeam"), f"{context} homeTeam")
        away_team = _required_mapping(match.get("awayTeam"), f"{context} awayTeam")
        full_time = _required_mapping(score.get("fullTime"), f"{context} fullTime score")

        home_goals = int(_required_value(full_time, "home", context))
        away_goals = int(_required_value(full_time, "away", context))
        if home_goals < 0 or away_goals < 0:
            raise FootballDataError(f"{context} has negative goals")

        try:
            kick_off = pd.to_datetime(
                _required_value(match, "utcDate", context), utc=True, errors="raise"
            )
        except (TypeError, ValueError) as error:
            raise FootballDataError(f"{context} has an invalid utcDate") from error

        source_competition_id = competition.get(
            "id", payload_competition.get("id")
        )
        source_competition_name = competition.get(
            "name", payload_competition.get("name")
        )
        if source_competition_id is None or not source_competition_name:
            raise FootballDataError(f"{context} lacks source competition identity")

        source_match_id = int(_required_value(match, "id", context))
        source_season_id = int(_required_value(season, "id", context))
        source_home_team_id = int(
            _required_value(home_team, "id", f"{context} homeTeam")
        )
        source_home_team_name = str(
            _required_value(home_team, "name", f"{context} homeTeam")
        )
        source_away_team_id = int(
            _required_value(away_team, "id", f"{context} awayTeam")
        )
        source_away_team_name = str(
            _required_value(away_team, "name", f"{context} awayTeam")
        )

        records.append(
            {
                "source": "football_data",
                "match_id": source_match_id,
                "source_match_id": source_match_id,
                "match_date": kick_off.tz_convert(None).normalize(),
                "kick_off": kick_off,
                "competition_id": int(canonical["competition_id"]),
                "competition_name": str(canonical["competition_name"]),
                "competition_code": code,
                "source_competition_id": int(source_competition_id),
                "source_competition_name": str(source_competition_name),
                "season_id": source_season_id,
                "season_start_year": season_start_year,
                "season_name": f"{season_start_year}/{season_start_year + 1}",
                "source_season_id": source_season_id,
                "home_team_id": source_home_team_id,
                "home_team_name": source_home_team_name,
                "source_home_team_id": source_home_team_id,
                "source_home_team_name": source_home_team_name,
                "away_team_id": source_away_team_id,
                "away_team_name": source_away_team_name,
                "source_away_team_id": source_away_team_id,
                "source_away_team_name": source_away_team_name,
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
    return records, len(matches)


def _validated_codes(codes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(code).upper() for code in codes))
    if not normalized:
        raise ValueError("At least one football-data.org competition code is required")
    unknown = sorted(set(normalized) - set(FOOTBALL_DATA_COMPETITIONS))
    if unknown:
        raise ValueError(f"Unsupported football-data.org competition codes: {unknown}")
    return normalized


def _validated_seasons(seasons: Sequence[int]) -> tuple[int, ...]:
    try:
        normalized = tuple(dict.fromkeys(int(season) for season in seasons))
    except (TypeError, ValueError) as error:
        raise ValueError("Football-data.org seasons must be integer start years") from error
    if not normalized or any(season < 1900 for season in normalized):
        raise ValueError("Football-data.org seasons must be valid start years")
    return normalized


def load_football_matches(path: Path) -> pd.DataFrame:
    """Load the normalized football-data.org match table."""

    if not path.is_file():
        raise FileNotFoundError(f"Football-data.org match data does not exist: {path}")
    matches = pd.read_parquet(path)
    missing = sorted(set(FOOTBALL_DATA_COLUMNS) - set(matches.columns))
    if missing:
        raise FootballDataError(f"Football-data.org data is missing columns: {missing}")
    if matches.empty:
        raise FootballDataError("Football-data.org match data is empty")

    if matches["source"].isna().any() or not matches["source"].eq(
        "football_data"
    ).all():
        raise FootballDataError("Football-data.org data has an invalid source")
    if (
        matches[["match_id", "source_match_id"]].isna().any().any()
        or matches["source_match_id"].duplicated().any()
        or not matches["match_id"].eq(matches["source_match_id"]).all()
    ):
        raise FootballDataError("Football-data.org source match IDs must be unique")

    match_dates = pd.to_datetime(matches["match_date"], errors="coerce")
    kick_offs = pd.to_datetime(matches["kick_off"], utc=True, errors="coerce")
    if match_dates.isna().any() or kick_offs.isna().any():
        raise FootballDataError("Football-data.org match dates must be valid")

    goals = matches[["home_goals", "away_goals"]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_goals = goals.apply(lambda column: column.map(math.isfinite))
    if (
        goals.isna().any().any()
        or not finite_goals.all().all()
        or (goals < 0).any().any()
        or ((goals % 1) != 0).any().any()
    ):
        raise FootballDataError(
            "Football-data.org goals must be finite, non-negative integers"
        )

    teams = matches[["home_team_id", "away_team_id"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if teams.isna().any().any() or teams["home_team_id"].eq(
        teams["away_team_id"]
    ).any():
        raise FootballDataError("Football-data.org home and away teams must differ")

    required_text = (
        "competition_name",
        "competition_code",
        "season_name",
        "home_team_name",
        "away_team_name",
    )
    for column in required_text:
        if matches[column].isna().any() or matches[column].astype(str).str.strip().eq(
            ""
        ).any():
            raise FootballDataError(f"Football-data.org data has invalid {column}")

    unknown_codes = sorted(
        set(matches["competition_code"].astype(str)) - set(FOOTBALL_DATA_COMPETITIONS)
    )
    if unknown_codes:
        raise FootballDataError(
            f"Football-data.org data has unsupported competition codes: {unknown_codes}"
        )
    for code, details in FOOTBALL_DATA_COMPETITIONS.items():
        selected = matches["competition_code"].eq(code)
        if selected.any() and not matches.loc[selected, "competition_id"].eq(
            int(details["competition_id"])
        ).all():
            raise FootballDataError(
                f"Football-data.org data has an invalid competition ID for {code}"
            )
    return matches


def sync_football_data(
    paths: ProjectPaths,
    *,
    competition_codes: Sequence[str] = DEFAULT_COMPETITION_CODES,
    seasons: Sequence[int] = DEFAULT_SEASONS,
    refresh: bool = False,
) -> FootballDataSummary:
    """Cache requested API responses and write one normalized finished-match table."""

    codes = _validated_codes(competition_codes)
    selected_seasons = _validated_seasons(seasons)
    raw_files = []
    downloaded_files = 0
    cached_files = 0
    api_key = None
    records = []
    source_matches = 0

    for code in codes:
        for season in selected_seasons:
            raw_file = paths.football_data_raw_directory / code / f"{season}.json"
            if raw_file.is_file() and not refresh:
                payload = _read_json(raw_file)
                normalized, payload_match_count = _normalize_payload(
                    payload, code, season
                )
                cached_files += 1
            else:
                if api_key is None:
                    api_key = load_api_key(paths.root)
                payload = _download_payload(code, season, api_key)
                normalized, payload_match_count = _normalize_payload(
                    payload, code, season
                )
                _atomic_json(payload, raw_file)
                downloaded_files += 1
            records.extend(normalized)
            source_matches += payload_match_count
            raw_files.append(
                {
                    "competition_code": code,
                    "season": season,
                    "path": str(raw_file),
                    "sha256": _sha256(raw_file),
                }
            )

    matches = pd.DataFrame.from_records(records, columns=FOOTBALL_DATA_COLUMNS)
    if matches.empty:
        raise FootballDataError(
            "No finished regular-time football-data.org matches were found"
        )
    if matches["source_match_id"].duplicated().any():
        raise FootballDataError("football-data.org returned duplicate source match IDs")
    matches = matches.sort_values(["match_date", "source_match_id"]).reset_index(drop=True)
    _atomic_parquet(matches, paths.football_data_matches_file)

    latest = matches["match_date"].max() if not matches.empty else None
    latest_date = latest.date().isoformat() if latest is not None else None
    summary = FootballDataSummary(
        requested_files=len(codes) * len(selected_seasons),
        downloaded_files=downloaded_files,
        cached_files=cached_files,
        source_matches=source_matches,
        finished_matches=len(matches),
        latest_match_date=latest_date,
        output_file=str(paths.football_data_matches_file),
        output_sha256=_sha256(paths.football_data_matches_file),
        manifest_file=str(paths.football_data_manifest_file),
    )
    manifest = {
        "source": "football_data",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "competition_codes": list(codes),
        "seasons": list(selected_seasons),
        "raw_files": raw_files,
    }
    manifest.update(asdict(summary))
    _atomic_json(manifest, paths.football_data_manifest_file)
    return summary
