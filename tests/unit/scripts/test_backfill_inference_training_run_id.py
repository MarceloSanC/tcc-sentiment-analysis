from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backfill_inference_training_run_id import (
    backfill_inference_training_run_id,
)


def _write_table(silver_dir: Path, table_name: str, rows: list[dict]) -> Path:
    path = silver_dir / table_name / "asset=AAPL" / f"{table_name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _artifact_rows(*, duplicate: bool = False) -> list[dict]:
    rows = [
        {
            "run_id": "train_run_abc",
            "training_run_id": "train_run_abc",
            "asset": "AAPL",
            "config_path": "data/models/AAPL/runs/20260303_120000_B/config.json",
            "checkpoint_path_final": "data/models/AAPL/runs/20260303_120000_B/model_state.pt",
        }
    ]
    if duplicate:
        rows.append(
            {
                "run_id": "train_run_other",
                "training_run_id": "train_run_other",
                "asset": "AAPL",
                "config_path": "data/models/AAPL/runs/20260303_120000_B/config.json",
                "checkpoint_path_final": "data/models/AAPL/runs/20260303_120000_B/model_state.pt",
            }
        )
    return rows


def _legacy_prediction_row() -> dict:
    return {
        "inference_run_id": "inf_1",
        "run_id": None,
        "training_run_id": None,
        "asset": "AAPL",
        "model_path": "data/models/AAPL/runs/20260303_120000_B",
    }


def test_backfill_dry_run_does_not_modify_silver(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver"
    _write_table(silver_dir, "fact_model_artifacts", _artifact_rows())
    predictions_path = _write_table(
        silver_dir,
        "fact_inference_predictions",
        [_legacy_prediction_row()],
    )
    before = pd.read_parquet(predictions_path)

    report = backfill_inference_training_run_id(silver_dir=silver_dir, apply=False)

    after = pd.read_parquet(predictions_path)
    assert report.dry_run is True
    assert report.total_backfilled == 1
    pd.testing.assert_frame_equal(before, after)


def test_backfill_resolves_unique_match(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver"
    _write_table(silver_dir, "fact_model_artifacts", _artifact_rows())
    predictions_path = _write_table(
        silver_dir,
        "fact_inference_predictions",
        [_legacy_prediction_row()],
    )

    report = backfill_inference_training_run_id(silver_dir=silver_dir, apply=True)

    after = pd.read_parquet(predictions_path)
    assert report.total_processed == 1
    assert report.total_backfilled == 1
    assert after.loc[0, "run_id"] == "train_run_abc"
    assert after.loc[0, "training_run_id"] == "train_run_abc"


def test_backfill_marks_ambiguous_match_as_pending(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver"
    _write_table(silver_dir, "fact_model_artifacts", _artifact_rows(duplicate=True))
    predictions_path = _write_table(
        silver_dir,
        "fact_inference_predictions",
        [_legacy_prediction_row()],
    )

    report = backfill_inference_training_run_id(silver_dir=silver_dir, apply=True)

    after = pd.read_parquet(predictions_path)
    assert report.total_ambiguous == 1
    assert pd.isna(after.loc[0, "run_id"])
    assert pd.isna(after.loc[0, "training_run_id"])
    assert after.loc[0, "_backfill_status"] == "ambiguous"
