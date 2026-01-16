# src/adapters/parquet_candle_repository.py

from pathlib import Path
import logging

import pandas as pd

from src.entities.candle import Candle
from src.entities.daily_sentiment import DailySentiment
from src.interfaces.candle_repository import CandleRepository

from src.infrastructure.schemas.candle_parquet_schema import (
    CANDLE_PARQUET_COLUMNS,
    CANDLE_PARQUET_DTYPES,
)

logger = logging.getLogger(__name__)


class ParquetCandleRepository(CandleRepository):
    """
    Repository adapter for Candle persistence using Parquet files.

    Current behavior:
    - Overwrites existing candle files
    - No incremental append
    - No deduplication by timestamp

    This is intentional for early-stage development
    and deterministic pipelines.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

        if not self.output_dir.exists():
            raise FileNotFoundError(
                f"Candle directory does not exist: {self.output_dir.resolve()}\n"
                "Check data_paths.yaml or environment configuration."
            )

        if not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"Candle path is not a directory: {self.output_dir.resolve()}"
            )

        logger.info(
            "ParquetCandleRepository initialized",
            extra={"output_dir": str(self.output_dir.resolve())},
        )

    def _normalize_symbol(self, asset_id: str) -> str:
        return asset_id.split(".")[0].upper()

    def _filepath(self, asset_id: str) -> Path:
        clean_symbol = self._normalize_symbol(asset_id)
        return self.output_dir / clean_symbol / f"candles_{clean_symbol}_1d.parquet"

    def load_candles(self, asset_id: str) -> list[Candle]:
        filepath = self._filepath(asset_id)

        if not filepath.exists():
            raise FileNotFoundError(
                f"No candle file found for {asset_id}\n"
                f"Expected path: {filepath.resolve()}"
            )

        logger.info(
            "Loading candles from parquet",
            extra={"asset_id": asset_id, "path": str(filepath)},
        )

        df = pd.read_parquet(filepath)

        missing = CANDLE_PARQUET_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Invalid candle parquet schema for {asset_id}. "
                f"Missing columns: {missing}. "
                f"File: {filepath.resolve()}"
            )

        df = df.astype(CANDLE_PARQUET_DTYPES, errors="ignore")

        df = df.sort_values("timestamp").reset_index(drop=True)

        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError(
                f"Timestamps are not monotonic after sorting: {filepath.name}"
            )

        candles: list[Candle] = []
        for _, row in df.iterrows():
            candles.append(
                Candle(
                    timestamp=row["timestamp"].to_pydatetime()
                    if hasattr(row["timestamp"], "to_pydatetime")
                    else row["timestamp"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
            )

        logger.info(
            "Candles loaded successfully",
            extra={"asset_id": asset_id, "count": len(candles)},
        )

        return candles

    def save_candles(self, asset_id: str, candles: list[Candle]) -> None:
        if not candles:
            raise ValueError("No candles to save")

        filepath = self._filepath(asset_id)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        )

        df = df[list(CANDLE_PARQUET_COLUMNS)]
        df = df.astype(CANDLE_PARQUET_DTYPES)
        df = df.sort_values("timestamp").reset_index(drop=True)

        df.to_parquet(filepath, index=False)

        logger.info(
            "Candles saved successfully",
            extra={
                "asset_id": asset_id,
                "count": len(candles),
                "path": str(filepath),
            },
        )

    def update_sentiment(
        self,
        asset_id: str,
        daily_sentiments: list[DailySentiment],
    ) -> None:
        """
        Atualiza candles existentes com sentimento diário agregado.

        Cenário A (incremental):
        - Atualiza apenas as datas presentes em daily_sentiments
        - Preserva valores já persistidos fora desse range
        - Evita colunas *_x / *_y no parquet final
        """

        if not daily_sentiments:
            logger.info(
                "No daily sentiments to persist",
                extra={"asset_id": asset_id},
            )
            return

        filepath = self._filepath(asset_id)

        logger.info("Updating sentiment (read)", extra={"path": str(filepath.resolve())})

        if not filepath.exists():
            raise FileNotFoundError(
                f"Cannot update sentiment: candle file not found for "
                f"{asset_id} ({filepath.resolve()})"
            )

        logger.info(
            "Updating candle sentiment (incremental)",
            extra={
                "asset_id": asset_id,
                "days": len(daily_sentiments),
                "path": str(filepath),
            },
        )

        # Load existing candles
        df_candles = pd.read_parquet(filepath)

        # Ensure timestamp is UTC-aware datetime
        # - if parquet stores epoch ms (int), unit="ms" is correct
        # - if parquet stores datetime already, pd.to_datetime will keep it
        ts = pd.to_datetime(df_candles["timestamp"], utc=True, errors="coerce")
        if ts.isna().any():
            raise ValueError(
                f"Invalid timestamp values found in parquet for {asset_id}: {filepath}"
            )

        # Canonical join key: date (UTC trading day)
        df_candles["date"] = ts.dt.date

        # Prepare sentiment df (domain -> infra)
        df_sentiment = pd.DataFrame(
            [
                {
                    "date": ds.day,  # ds.day is a date
                    "sentiment_score": float(ds.sentiment_score),
                    "sentiment_std": float(ds.sentiment_std),
                    "n_articles": int(ds.n_articles),
                }
                for ds in daily_sentiments
            ]
        )

        # Optional: drop duplicate days (keep last) to avoid merge explosion
        df_sentiment = df_sentiment.drop_duplicates(subset=["date"], keep="last")

        # Merge with suffixes to preserve existing values
        df_merged = df_candles.merge(
            df_sentiment,
            on="date",
            how="left",
            suffixes=("_old", ""),
        )

        # Coalesce: prefer new values when available, otherwise keep old ones
        for col in ["sentiment_score", "sentiment_std", "n_articles"]:
            old_col = f"{col}_old"
            if old_col in df_merged.columns:
                df_merged[col] = df_merged[col].combine_first(df_merged[old_col])
                df_merged = df_merged.drop(columns=[old_col])

        # Cleanup helper column
        df_merged = df_merged.drop(columns=["date"])

        # Basic validation: did we actually write anything new?
        # (counts rows where new sentiment exists for the provided days)
        matched_days = set(df_sentiment["date"].tolist())
        df_check = pd.to_datetime(df_merged["timestamp"], utc=True).dt.date
        mask_days = df_check.isin(matched_days)

        updated_non_null = 0
        if mask_days.any():
            updated_non_null = int(df_merged.loc[mask_days, "sentiment_score"].notna().sum())

        logger.info(
            "Sentiment merge completed",
            extra={
                "asset_id": asset_id,
                "days_input": len(df_sentiment),
                "rows_in_window": int(mask_days.sum()),
                "rows_with_sentiment_after": updated_non_null,
            },
        )

        # Persist updated candles
        df_merged.to_parquet(filepath, index=False)

        logger.info("Updating sentiment (write)", extra={"path": str(filepath.resolve()), "rows": len(df_merged), "cols": list(df_merged.columns)})

        logger.info(
            "Sentiment successfully persisted into candles",
            extra={
                "asset_id": asset_id,
                "path": str(filepath),
            },
        )


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(data-pipeline):
# Suportar persistência incremental de candles
# (append ou upsert por timestamp)

# TODO(data-pipeline):
# Implementar deduplicação temporal
# (manter último candle por timestamp)

# TODO(architecture):
# Expor política explícita de persistência:
# overwrite | append | upsert

# TODO(stat-validation):
# Validar gaps temporais excessivos
# (ex: dias úteis faltantes)

# TODO(reproducibility):
# Versionar datasets de candles persistidos
# (ex: hash do arquivo + metadata JSON)