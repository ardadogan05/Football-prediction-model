"""Load validated locally processed match data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from football_prediction.data.schema import DataValidationError, validate_matches


def load_matches(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Processed match data does not exist: {path}")
    matches = pd.read_parquet(path)
    validate_matches(matches)
    return matches


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Processing manifest does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except json.JSONDecodeError as error:
        raise DataValidationError(f"Invalid processing manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise DataValidationError("Processing manifest must be a JSON object")
    return manifest
