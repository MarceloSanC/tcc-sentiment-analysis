# tests/unit/test_main_dataset_tft.py

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd

from src import main_dataset_tft


def test_main_dataset_tft_skips_when_exists_without_overwrite(monkeypatch, tmp_path: Path) -> None:
    asset_id = "AAPL"

    raw_candles = tmp_path / "raw" / "market" / "candles" / asset_id
    indicators = tmp_path / "processed" / "technical_indicators" / asset_id
    sentiment_daily = tmp_path / "processed" / "sentiment_daily"
    fundamentals = tmp_path / "processed" / "fundamentals"
    dataset_tft = tmp_path / "processed" / "dataset_tft"

    raw_candles.mkdir(parents=True, exist_ok=True)
    indicators.mkdir(parents=True, exist_ok=True)
    sentiment_daily.mkdir(parents=True, exist_ok=True)
    fundamentals.mkdir(parents=True, exist_ok=True)
    (dataset_tft / asset_id).mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_tft / asset_id / f"dataset_tft_{asset_id}.parquet"
    pd.DataFrame({"timestamp": [pd.Timestamp("2024-01-01", tz="UTC")]}).to_parquet(
        dataset_path, index=False
    )

    def _fake_paths() -> dict:
        return {
            "raw_candles": tmp_path / "raw" / "market" / "candles",
            "processed_technical_indicators": tmp_path / "processed" / "technical_indicators",
            "processed_sentiment_daily": sentiment_daily,
            "processed_fundamentals": fundamentals,
            "dataset_tft": dataset_tft,
        }

    monkeypatch.setattr(main_dataset_tft, "load_data_paths", _fake_paths)
    monkeypatch.setattr(
        main_dataset_tft,
        "load_config",
        lambda: {
            "assets": [
                {"symbol": asset_id, "start_date": "2024-01-01", "end_date": "2024-12-31"}
            ]
        },
    )
    monkeypatch.setattr(
        main_dataset_tft, "parse_args", lambda: Namespace(asset=asset_id, overwrite=False)
    )

    calls: list[Path] = []
    monkeypatch.setattr(main_dataset_tft.DataQualityReporter, "report_exists", lambda *_: False)
    monkeypatch.setattr(
        main_dataset_tft.DataQualityReporter,
        "report_from_parquet",
        lambda path, **_: calls.append(path),
    )

    main_dataset_tft.main()

    assert calls == [dataset_path]
