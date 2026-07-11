"""Command-line entry points for data updates and model tuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from football_prediction.config import ProjectPaths
from football_prediction.data.football_data import (
    load_football_matches,
    sync_football_data,
)
from football_prediction.data.loader import load_manifest, load_matches
from football_prediction.data.process import process_repository
from football_prediction.data.sync import sync_repository
from football_prediction.model import tune_external_poisson_models


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="football-prediction")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the data directory",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("update-data", help="Synchronize and process StatsBomb data")
    process = subcommands.add_parser(
        "process-data", help="Reprocess an existing local StatsBomb checkout"
    )
    process.add_argument("--source-commit", default="local-fixture")
    subcommands.add_parser("data-status", help="Print the current processed-data manifest")
    football_data = subcommands.add_parser(
        "update-football-data",
        help="Download and process recent football-data.org league matches",
    )
    football_data.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2024, 2025],
        help="Season start years to download",
    )
    football_data.add_argument(
        "--refresh",
        action="store_true",
        help="Download files again instead of using the raw cache",
    )
    subcommands.add_parser(
        "football-data-status",
        help="Print the football-data.org processed-data manifest",
    )
    subcommands.add_parser(
        "tune-model",
        help="Tune on StatsBomb training data and recent football-data.org data",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    paths = ProjectPaths.from_root(arguments.project_root)

    if arguments.command == "update-data":
        sync = sync_repository(paths.raw_repository)
        summary = process_repository(
            sync.repository,
            paths.matches_file,
            paths.manifest_file,
            sync.commit,
        )
        output = {
            "sync_action": sync.action,
            "source_changed": sync.changed,
            "downloaded_events": sync.downloaded_events,
        }
        output.update(asdict(summary))
    elif arguments.command == "process-data":
        summary = process_repository(
            paths.raw_repository,
            paths.matches_file,
            paths.manifest_file,
            arguments.source_commit,
        )
        output = asdict(summary)
    elif arguments.command == "data-status":
        output = load_manifest(paths.manifest_file)
    elif arguments.command == "update-football-data":
        summary = sync_football_data(
            paths,
            seasons=arguments.seasons,
            refresh=arguments.refresh,
        )
        output = asdict(summary)
    elif arguments.command == "football-data-status":
        output = load_manifest(paths.football_data_manifest_file)
    else:
        training_matches = load_matches(paths.matches_file)
        recent_matches = load_football_matches(paths.football_data_matches_file)
        result = tune_external_poisson_models(training_matches, recent_matches)
        output = {
            "best_window": result.best_window,
            "best_alpha": result.best_alpha,
            "validation_log_loss": result.validation_log_loss,
            "test_matches": len(result.test_features),
            "test_start": result.test_features["match_date"].min().date().isoformat(),
            "test_end": result.test_features["match_date"].max().date().isoformat(),
            "grid": result.results.to_dict(orient="records"),
        }

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
