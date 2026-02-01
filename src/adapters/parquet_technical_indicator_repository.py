# src/adapters/parquet_technical_indicator_repository.py
from pathlib import Path

import pandas as pd

from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository
from src.infrastructure.schemas.technical_indicator_parquet_schema import (
    TECHNICAL_INDICATOR_BASE_COLUMNS,
    TECHNICAL_INDICATOR_DTYPES,
    TECHNICAL_INDICATOR_INDEX,
)


class ParquetTechnicalIndicatorRepository(TechnicalIndicatorRepository):
    """
    Adapter de persistência de TechnicalIndicatorSet em Parquet (wide format).
    """

    def __init__(
        self,
        output_dir: str | Path,
        overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

    def save(self, asset_id: str, indicators: list[TechnicalIndicatorSet]) -> None:
        if not indicators:
            raise ValueError("No TechnicalIndicatorSet to persist")

        filepath = self.output_dir / f"technical_indicators_{asset_id}.parquet"

        if filepath.exists() and not self.overwrite:
            raise FileExistsError(
                f"Feature file already exists: {filepath.resolve()}\n"
                "Use --overwrite to replace it."
            )

        rows: list[dict] = []

        for item in indicators:
            row = {"asset_id": item.asset_id, "timestamp": item.timestamp}
            row.update(item.indicators)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Schema validation
        missing = TECHNICAL_INDICATOR_BASE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing TechnicalIndicator columns: {missing}")

        # Dtypes
        df = df.astype(TECHNICAL_INDICATOR_DTYPES)
        for col in df.columns:
            if col in TECHNICAL_INDICATOR_BASE_COLUMNS:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        # Temporal guarantee
        df = df.sort_values(TECHNICAL_INDICATOR_INDEX).reset_index(drop=True)

        df.to_parquet(filepath, index=False)

    def load(self, asset_id: str) -> list[TechnicalIndicatorSet]:
        filepath = self.output_dir / f"technical_indicators_{asset_id}.parquet"

        if not filepath.exists():
            raise FileNotFoundError(f"No features found for {asset_id}")

        df = pd.read_parquet(filepath)

        # Schema validation
        missing = TECHNICAL_INDICATOR_BASE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Invalid TechnicalIndicator schema in {filepath.name}. "
                f"Missing columns: {missing}"
            )

        df = df.sort_values(TECHNICAL_INDICATOR_INDEX)

        indicator_cols = [
            c for c in df.columns if c not in TECHNICAL_INDICATOR_BASE_COLUMNS
        ]
        result: list[TechnicalIndicatorSet] = []

        for _, row in df.iterrows():
            indicators = {
                name: float(row[name])
                for name in indicator_cols
                if pd.notna(row[name])
            }
            if not indicators:
                raise ValueError("TechnicalIndicatorSet cannot be empty")

            timestamp = row["timestamp"]
            result.append(
                TechnicalIndicatorSet(
                    asset_id=row["asset_id"],
                    timestamp=(
                        timestamp.to_pydatetime()
                        if hasattr(timestamp, "to_pydatetime")
                        else timestamp
                    ),
                    indicators=indicators,
                )
            )

        # Garantia temporal final no domínio
        return sorted(result, key=lambda item: item.timestamp)
