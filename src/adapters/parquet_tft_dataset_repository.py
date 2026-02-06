from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.infrastructure.schemas.tft_dataset_parquet_schema import (
    TFT_DATASET_BASE_COLUMNS,
    TFT_DATASET_DTYPES,
)
from src.interfaces.tft_dataset_repository import TFTDatasetRepository

logger = logging.getLogger(__name__)


class ParquetTFTDatasetRepository(TFTDatasetRepository):
    """
    Parquet repository for TFT dataset.

    Storage layout:
      data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"TFT dataset output_dir is not a directory: {self.output_dir.resolve()}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ParquetTFTDatasetRepository initialized",
            extra={"output_dir": str(self.output_dir.resolve())},
        )

    @staticmethod
    def _normalize_symbol(asset_id: str) -> str:
        return asset_id.split(".")[0].upper()

    def _asset_dir(self, asset_id: str) -> Path:
        symbol = self._normalize_symbol(asset_id)
        return self.output_dir / symbol

    def _filepath(self, asset_id: str) -> Path:
        symbol = self._normalize_symbol(asset_id)
        return self._asset_dir(symbol) / f"dataset_tft_{symbol}.parquet"

    def save(self, asset_id: str, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("TFT dataset is empty")

        missing = set(TFT_DATASET_BASE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing base columns for TFT dataset: {sorted(missing)}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        for col, dtype in TFT_DATASET_DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        filepath = self._filepath(asset_id)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False)

        logger.info(
            "TFT dataset persisted",
            extra={
                "asset_id": self._normalize_symbol(asset_id),
                "rows": len(df),
                "cols": list(df.columns),
                "path": str(filepath.resolve()),
            },
        )

    def load(self, asset_id: str) -> pd.DataFrame:
        filepath = self._filepath(asset_id)
        if not filepath.exists():
            raise FileNotFoundError(f"TFT dataset not found for {asset_id}")
        return pd.read_parquet(filepath)
