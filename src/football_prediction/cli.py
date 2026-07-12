"""Small command-line interface for data, training, backtesting, and prediction."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from football_prediction.backtest import run_backtest, save_backtest
from football_prediction.config import ProjectPaths
from football_prediction.data.football_data import (
    load_football_matches,
    sync_football_data,
)
from football_prediction.data.loader import load_manifest, load_matches
from football_prediction.data.process import process_repository
from football_prediction.data.sync import sync_repository
from football_prediction.model import tune_poisson_models
from football_prediction.prediction import load_model, predict_match, save_model


def parser():
    command_parser = argparse.ArgumentParser(prog="football-prediction")
    command_parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing data, models, and reports",
    )
    commands = command_parser.add_subparsers(dest="command", required=True)

    commands.add_parser("update-data", help="Synchronize and process StatsBomb data")
    process = commands.add_parser(
        "process-data", help="Reprocess an existing local StatsBomb checkout"
    )
    process.add_argument("--source-commit", default="local-fixture")
    commands.add_parser("data-status", help="Print the StatsBomb data manifest")

    football_data = commands.add_parser(
        "update-football-data",
        help="Download and process recent football-data.org matches",
    )
    football_data.add_argument(
        "--seasons", nargs="+", type=int, default=[2024, 2025]
    )
    football_data.add_argument("--refresh", action="store_true")
    commands.add_parser(
        "football-data-status", help="Print the football-data.org manifest"
    )

    commands.add_parser(
        "tune-model",
        help="Tune, refit on training plus validation, and save models/model.pkl",
    )
    commands.add_parser(
        "train",
        help="Alias for tune-model; use one of these commands, not both",
    )
    commands.add_parser(
        "backtest",
        help="Evaluate the saved model once on the untouched test season",
    )

    predict = commands.add_parser("predict", help="Predict one future fixture")
    predict.add_argument("--home-team", required=True)
    predict.add_argument("--away-team", required=True)
    predict.add_argument("--competition", required=True)
    return command_parser


def tune_and_save(paths):
    training_matches = load_matches(paths.matches_file)
    recent_matches = load_football_matches(paths.football_data_matches_file)
    result = tune_poisson_models(training_matches, recent_matches)
    save_model(result["model"], paths.model_file)
    return {
        "best_window": result["best_window"],
        "best_alpha": result["best_alpha"],
        "validation_log_loss": result["validation_log_loss"],
        "model_file": str(paths.model_file),
        "grid": result["results"].to_dict(orient="records"),
    }


def main(argv=None):
    arguments = parser().parse_args(argv)
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
    elif arguments.command in ["tune-model", "train"]:
        output = tune_and_save(paths)
    elif arguments.command == "backtest":
        model_bundle = load_model(paths.model_file)
        recent_matches = load_football_matches(paths.football_data_matches_file)
        result = run_backtest(model_bundle, recent_matches)
        save_backtest(
            result,
            paths.backtest_predictions_file,
            paths.metrics_file,
        )
        output = result["metrics"]
        output["predictions_file"] = str(paths.backtest_predictions_file)
        output["metrics_file"] = str(paths.metrics_file)
    else:
        model_bundle = load_model(paths.model_file)
        recent_matches = load_football_matches(paths.football_data_matches_file)
        output = predict_match(
            arguments.home_team,
            arguments.away_team,
            arguments.competition,
            recent_matches,
            model_bundle,
        )

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
