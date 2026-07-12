from pathlib import Path
import shutil

import pandas as pd

from football_prediction import cli as cli_module
from football_prediction.cli import main
from football_prediction.data.football_data import FootballDataSummary


FIXTURE_REPOSITORY = Path(__file__).parent / "fixtures" / "statsbomb"


def test_process_data_and_status_commands_are_offline(
    tmp_path, capsys
):
    raw = tmp_path / "data" / "raw" / "statsbomb-open-data"
    shutil.copytree(FIXTURE_REPOSITORY, raw)

    assert main(
        [
            "--project-root",
            str(tmp_path),
            "process-data",
            "--source-commit",
            "fixture-sha",
        ]
    ) == 0
    process_output = capsys.readouterr().out
    assert '"processed_matches": 2' in process_output

    assert main(["--project-root", str(tmp_path), "data-status"]) == 0
    status_output = capsys.readouterr().out
    assert '"source_commit": "fixture-sha"' in status_output


def test_football_data_commands_report_summary_and_status(
    tmp_path, capsys, monkeypatch
):
    manifest = tmp_path / "data" / "processed" / "football_data_manifest.json"

    def fake_sync(paths, seasons, refresh):
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            '{"source": "football_data", "finished_matches": 760}',
            encoding="utf-8",
        )
        return FootballDataSummary(
            requested_files=2,
            downloaded_files=2,
            cached_files=0,
            source_matches=760,
            finished_matches=760,
            latest_match_date="2026-05-24",
            output_file=str(paths.football_data_matches_file),
            output_sha256="example",
            manifest_file=str(paths.football_data_manifest_file),
        )

    monkeypatch.setattr(cli_module, "sync_football_data", fake_sync)
    assert main(
        [
            "--project-root",
            str(tmp_path),
            "update-football-data",
            "--seasons",
            "2024",
            "2025",
        ]
    ) == 0
    update_output = capsys.readouterr().out
    assert '"finished_matches": 760' in update_output

    assert main(["--project-root", str(tmp_path), "football-data-status"]) == 0
    status_output = capsys.readouterr().out
    assert '"source": "football_data"' in status_output


def test_tune_model_command_loads_both_sources_and_reports_result(
    tmp_path, capsys, monkeypatch
):
    training_matches = object()
    recent_matches = object()
    loaded_paths = []

    def fake_load_matches(path):
        loaded_paths.append(path)
        return training_matches

    def fake_load_football_matches(path):
        loaded_paths.append(path)
        return recent_matches

    def fake_tune(training, recent):
        assert training is training_matches
        assert recent is recent_matches
        return {
            "best_window": 5,
            "best_alpha": 0.1,
            "validation_log_loss": 0.91,
            "test_features": pd.DataFrame(
                {
                    "match_date": [
                        pd.Timestamp("2025-08-15"),
                        pd.Timestamp("2026-05-24"),
                    ]
                }
            ),
            "results": pd.DataFrame(
                [
                    {
                        "rolling_window": 5,
                        "alpha": 0.1,
                        "validation_log_loss": 0.91,
                    }
                ]
            ),
        }

    monkeypatch.setattr(cli_module, "load_matches", fake_load_matches)
    monkeypatch.setattr(
        cli_module, "load_football_matches", fake_load_football_matches
    )
    monkeypatch.setattr(cli_module, "tune_external_poisson_models", fake_tune)

    assert main(["--project-root", str(tmp_path), "tune-model"]) == 0

    paths = cli_module.ProjectPaths.from_root(tmp_path)
    assert loaded_paths == [paths.matches_file, paths.football_data_matches_file]
    output = capsys.readouterr().out
    assert '"best_window": 5' in output
    assert '"best_alpha": 0.1' in output
    assert '"test_matches": 2' in output
    assert '"test_start": "2025-08-15"' in output
    assert '"test_end": "2026-05-24"' in output
