"""Synchronize selected files from StatsBomb Open Data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from football_prediction.config import (
    STATSBOMB_COMMIT_URL,
    STATSBOMB_RAW_URL,
    is_top_five_mens_league,
)


class SynchronizationError(RuntimeError):
    """Raised when StatsBomb files cannot be synchronized."""


@dataclass(frozen=True)
class SyncResult:
    repository: Path
    commit: str
    changed: bool
    action: str
    downloaded_events: int


def _request(url: str) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "football-prediction-portfolio-project",
        },
    )
    last_error = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise SynchronizationError(f"Could not download {url}: {last_error}")


def _parse_json(content: bytes, source: str) -> Any:
    try:
        return json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SynchronizationError(f"Invalid JSON downloaded from {source}") from error


def _write_file(content: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, destination)


def _download_json(url: str, destination: Path) -> Any:
    content = _request(url)
    data = _parse_json(content, url)
    _write_file(content, destination)
    return data


def _latest_commit() -> str:
    content = _request(STATSBOMB_COMMIT_URL)
    response = _parse_json(content, STATSBOMB_COMMIT_URL)
    commit = response.get("sha") if isinstance(response, dict) else None
    if not commit:
        raise SynchronizationError("GitHub did not return a StatsBomb commit SHA")
    return str(commit)


def _load_snapshot(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            snapshot = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return snapshot if isinstance(snapshot, dict) else {}


def _event_url(commit: str, match_id: int) -> str:
    return f"{STATSBOMB_RAW_URL}/{commit}/data/events/{match_id}.json"


def sync_repository(repository: Path, workers: int = 8) -> SyncResult:
    """Download only top-five men's league files and reuse an unchanged cache."""

    repository = repository.resolve()
    snapshot_file = repository / "snapshot.json"
    commit = _latest_commit()
    snapshot = _load_snapshot(snapshot_file)
    competitions_file = repository / "data" / "competitions.json"

    if (
        snapshot.get("commit") == commit
        and snapshot.get("complete") is True
        and competitions_file.is_file()
    ):
        return SyncResult(repository, commit, False, "unchanged", 0)

    raw_base = f"{STATSBOMB_RAW_URL}/{commit}/data"
    competitions = _download_json(
        f"{raw_base}/competitions.json",
        competitions_file,
    )
    if not isinstance(competitions, list):
        raise SynchronizationError("StatsBomb competitions.json is not a list")

    selected = []
    for competition in competitions:
        if is_top_five_mens_league(competition):
            selected.append(competition)

    match_ids = set()
    for number, competition in enumerate(selected, start=1):
        competition_id = competition["competition_id"]
        season_id = competition["season_id"]
        destination = (
            repository
            / "data"
            / "matches"
            / str(competition_id)
            / f"{season_id}.json"
        )
        matches = _download_json(
            f"{raw_base}/matches/{competition_id}/{season_id}.json",
            destination,
        )
        if not isinstance(matches, list):
            raise SynchronizationError(f"Match file is not a list: {destination}")
        for match in matches:
            if "match_id" in match:
                match_ids.add(int(match["match_id"]))
        print(
            f"Found season {number}/{len(selected)}: "
            f"{competition['competition_name']} {competition['season_name']}",
            flush=True,
        )

    event_directory = repository / "data" / "events"
    event_directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    print(f"Downloading {len(match_ids)} match event files...", flush=True)

    def download_event(match_id: int) -> None:
        _download_json(
            _event_url(commit, match_id),
            event_directory / f"{match_id}.json",
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = []
        for match_id in sorted(match_ids):
            futures.append(executor.submit(download_event, match_id))
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(match_ids):
                print(f"Downloaded events: {completed}/{len(match_ids)}", flush=True)

    snapshot_data = {
        "commit": commit,
        "competition_seasons": len(selected),
        "matches": len(match_ids),
        "complete": True,
    }
    _write_file(
        json.dumps(snapshot_data, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        snapshot_file,
    )
    action = "updated" if snapshot else "downloaded"
    return SyncResult(repository, commit, True, action, completed)
