from __future__ import annotations

from pathlib import Path
import shutil

from football_prediction.cli import main


FIXTURE_REPOSITORY = Path(__file__).parent / "fixtures" / "statsbomb"


def test_process_data_and_status_commands_are_offline(
    tmp_path: Path, capsys
) -> None:
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
