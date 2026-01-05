# src/adapters/parquet_feature_set_repository.py
from pathlib import Path

import pandas as pd

from src.entities.feature_set import FeatureSet
from src.interfaces.feature_set_repository import FeatureSetRepository
from src.infrastructure.schemas.feature_set_parquet_schema import (
    FEATURE_SET_BASE_COLUMNS,
    FEATURE_SET_DTYPES,
    FEATURE_SET_INDEX,
)


class ParquetFeatureSetRepository(FeatureSetRepository):
    """
    Adapter de persistência de FeatureSets em Parquet (long format).
    """

    def __init__(
        self,
        output_dir: str | Path,
        overwrite: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

    def save(self, asset_id: str, features: list[FeatureSet]) -> None:
        if not features:
            raise ValueError("No FeatureSet to persist")

        filepath = self.output_dir / f"features_{asset_id}.parquet"

        if filepath.exists() and not self.overwrite:
            raise FileExistsError(
                f"Feature file already exists: {filepath.resolve()}\n"
                "Use --overwrite to replace it."
            )

        rows: list[dict] = []

        for fs in features:
            for name, value in fs.features.items():
                rows.append(
                    {
                        "asset_id": fs.asset_id,
                        "timestamp": fs.timestamp,
                        "feature_name": name,
                        "feature_value": float(value) if value is not None else None,
                    }
                )

        df = pd.DataFrame(rows)

        # Schema validation
        missing = FEATURE_SET_BASE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing FeatureSet columns: {missing}")

        # Dtypes
        df = df.astype(FEATURE_SET_DTYPES)

        # Temporal guarantee
        df = df.sort_values(FEATURE_SET_INDEX).reset_index(drop=True)

        df.to_parquet(filepath, index=False)

    def load(self, asset_id: str) -> list[FeatureSet]:
        filepath = self.output_dir / f"features_{asset_id}.parquet"

        if not filepath.exists():
            raise FileNotFoundError(f"No features found for {asset_id}")

        df = pd.read_parquet(filepath)

        # Schema validation
        missing = FEATURE_SET_BASE_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Invalid FeatureSet schema in {filepath.name}. "
                f"Missing columns: {missing}"
            )

        df = df.sort_values(FEATURE_SET_INDEX)

        grouped: dict[
            tuple[str, pd.Timestamp], dict[str, float]
        ] = {}

        for _, row in df.iterrows():
            key = (row["asset_id"], row["timestamp"])
            grouped.setdefault(key, {})
            grouped[key][row["feature_name"]] = row["feature_value"]

        result: list[FeatureSet] = []

        for (asset_id, timestamp), features in grouped.items():
            result.append(
                FeatureSet(
                    asset_id=asset_id,
                    timestamp=(
                        timestamp.to_pydatetime()
                        if hasattr(timestamp, "to_pydatetime")
                        else timestamp
                    ),
                    features=features,
                )
            )

        # Garantia temporal final no domínio
        return sorted(result, key=lambda fs: fs.timestamp)
