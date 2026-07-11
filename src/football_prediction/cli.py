"""Command-line entry points for Phase 1 data operations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from football_prediction.config import ProjectPaths
from football_prediction.data.loader import load_manifest
from football_prediction.data.process import process_repository
from football_prediction.data.sync import sync_repository


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
    else:
        output = load_manifest(paths.manifest_file)

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
