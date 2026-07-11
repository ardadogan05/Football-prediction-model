"""Convert StatsBomb match and event JSON into a compact match table."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from football_prediction.config import is_top_five_mens_league
from football_prediction.data.schema import (
    MATCH_COLUMNS,
    DataValidationError,
    require_fields,
    validate_matches,
)


PROCESSING_SCHEMA_VERSION = "1"
REGULATION_PERIODS = {1, 2}


@dataclass(frozen=True)
class ProcessingSummary:
    source_commit: str
    selected_competition_seasons: int
    source_matches: int
    processed_matches: int
    model_eligible_matches: int
    rejected_matches: int
    latest_match_date: str | None
    output_file: str
    output_sha256: str
    rejection_reasons: dict[str, int]


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"Could not read valid JSON from {path}: {error}") from error


def is_selected_competition(record: dict[str, Any]) -> bool:
    """Return whether a competition-season belongs to the top five men's leagues."""
    return is_top_five_mens_league(record)


def selected_competitions(repository: Path) -> list[dict[str, Any]]:
    records = _read_json(repository / "data" / "competitions.json")
    if not isinstance(records, list):
        raise DataValidationError("competitions.json must contain a JSON list")

    selected = []
    for record in records:
        if is_selected_competition(record):
            selected.append(record)
    return selected


def _team_details(match: dict[str, Any], role: str) -> tuple[int, str]:
    field = f"{role}_team"
    team = match[field]
    require_fields(team, [f"{role}_team_id", f"{role}_team_name"], field)
    return int(team[f"{role}_team_id"]), str(team[f"{role}_team_name"])


def process_match(
    match: dict[str, Any],
    events: list[dict[str, Any]],
    competition: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Create one canonical regulation-time match record."""

    require_fields(
        match,
        [
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
        "match",
    )
    home_id, home_name = _team_details(match, "home")
    away_id, away_name = _team_details(match, "away")
    if home_id == away_id:
        raise DataValidationError("home and away team IDs are identical")
    if not isinstance(events, list):
        raise DataValidationError("event file must contain a JSON list")

    periods = set()
    for event in events:
        if "period" in event:
            periods.add(int(event["period"]))
    non_regulation_periods = periods - REGULATION_PERIODS
    if non_regulation_periods:
        raise DataValidationError(
            f"contains non-regulation periods: {sorted(non_regulation_periods)}"
        )

    xg = {home_id: 0.0, away_id: 0.0}
    shots = {home_id: 0, away_id: 0}
    missing_xg = 0
    for event in events:
        event_type = event.get("type", {}).get("name")
        if event_type != "Shot" or event.get("period") not in REGULATION_PERIODS:
            continue
        team_id = event.get("team", {}).get("id")
        if team_id not in xg:
            raise DataValidationError(f"shot references unknown team ID: {team_id}")
        shots[team_id] += 1
        shot_xg = event.get("shot", {}).get("statsbomb_xg")
        if shot_xg is None:
            missing_xg += 1
            continue
        try:
            value = float(shot_xg)
        except (TypeError, ValueError) as error:
            raise DataValidationError(f"invalid shot xG value: {shot_xg!r}") from error
        if value < 0:
            raise DataValidationError(f"negative shot xG value: {value}")
        xg[team_id] += value

    eligible = missing_xg == 0
    require_fields(competition, ["competition_id", "season_id"], "competition")
    return {
        "match_id": int(match["match_id"]),
        "match_date": pd.to_datetime(match["match_date"], errors="raise"),
        "kick_off": match.get("kick_off"),
        "competition_id": int(competition["competition_id"]),
        "competition_name": str(competition["competition_name"]),
        "season_id": int(competition["season_id"]),
        "season_name": str(competition.get("season_name", competition["season_id"])),
        "home_team_id": home_id,
        "home_team_name": home_name,
        "away_team_id": away_id,
        "away_team_name": away_name,
        "home_goals": int(match["home_score"]),
        "away_goals": int(match["away_score"]),
        "home_xg": xg[home_id] if eligible else None,
        "away_xg": xg[away_id] if eligible else None,
        "home_shots": shots[home_id],
        "away_shots": shots[away_id],
        "missing_shot_xg_count": missing_xg,
        "eligible_for_model": eligible,
        "source_commit": source_commit,
    }


def _increment_reason(reasons: dict[str, int], error: Exception) -> None:
    reason = str(error)
    reasons[reason] = reasons.get(reason, 0) + 1


def _atomic_parquet(matches: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    matches.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        chunk = file.read(1024 * 1024)
        while chunk:
            digest.update(chunk)
            chunk = file.read(1024 * 1024)
    return digest.hexdigest()


def _atomic_json(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, destination)


def process_repository(
    repository: Path,
    matches_file: Path,
    manifest_file: Path,
    source_commit: str,
) -> ProcessingSummary:
    """Process all available top-five men's league competition-seasons."""

    repository = repository.resolve()
    competitions = selected_competitions(repository)
    processed_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    source_matches = 0
    handled_matches = 0
    rejection_reasons: dict[str, int] = {}

    for competition in competitions:
        competition_id = competition["competition_id"]
        season_id = competition["season_id"]
        match_path = (
            repository / "data" / "matches" / str(competition_id) / f"{season_id}.json"
        )
        matches = _read_json(match_path)
        if not isinstance(matches, list):
            raise DataValidationError(f"Match file must contain a JSON list: {match_path}")
        source_matches += len(matches)
        for match in matches:
            event_path = repository / "data" / "events" / f"{match.get('match_id')}.json"
            try:
                events = _read_json(event_path)
                records.append(
                    process_match(match, events, competition, source_commit)
                )
            except DataValidationError as error:
                _increment_reason(rejection_reasons, error)
            handled_matches += 1
            if handled_matches % 100 == 0 or handled_matches == source_matches:
                print(f"Processed matches: {handled_matches}", flush=True)

    matches = pd.DataFrame.from_records(records, columns=MATCH_COLUMNS)
    if not matches.empty:
        matches = matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    validate_matches(matches)
    _atomic_parquet(matches, matches_file)

    latest = matches["match_date"].max()
    summary = ProcessingSummary(
        source_commit=source_commit,
        selected_competition_seasons=len(competitions),
        source_matches=source_matches,
        processed_matches=len(matches),
        model_eligible_matches=int(matches["eligible_for_model"].sum()),
        rejected_matches=source_matches - len(matches),
        latest_match_date=latest.date().isoformat() if pd.notna(latest) else None,
        output_file=str(matches_file),
        output_sha256=_sha256(matches_file),
        rejection_reasons=rejection_reasons,
    )
    available_seasons = []
    for competition in competitions:
        available_seasons.append(
            {
                "competition_id": competition["competition_id"],
                "competition_name": competition["competition_name"],
                "country_name": competition["country_name"],
                "season_id": competition["season_id"],
                "season_name": competition.get("season_name"),
            }
        )

    manifest = {
        "processing_schema_version": PROCESSING_SCHEMA_VERSION,
        "processed_at_utc": processed_at,
        "available_competition_seasons": available_seasons,
    }
    manifest.update(asdict(summary))
    _atomic_json(manifest, manifest_file)
    return summary
