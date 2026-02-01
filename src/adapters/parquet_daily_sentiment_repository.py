from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.daily_sentiment import DailySentiment
from src.infrastructure.schemas.daily_sentiment_parquet_schema import (
    DAILY_SENTIMENT_PARQUET_COLUMNS,
    DAILY_SENTIMENT_PARQUET_DTYPES,
)
from src.interfaces.daily_sentiment_repository import DailySentimentRepository

logger = logging.getLogger(__name__)


class ParquetDailySentimentRepository(DailySentimentRepository):
    """
    Parquet-based repository for DailySentiment.

    Storage layout:
      data/processed/sentiment_daily/AAPL/daily_sentiment_AAPL.parquet

    Temporal contract:
    - day stored as datetime64[ns, UTC] in parquet.
    - inputs use timezone-aware datetimes for range queries.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"Daily sentiment output_dir is not a directory: {self.output_dir.resolve()}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ParquetDailySentimentRepository initialized",
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
        return self._asset_dir(symbol) / f"daily_sentiment_{symbol}.parquet"

    @staticmethod
    def _is_valid_parquet(path: Path) -> bool:
        try:
            with path.open("rb") as f:
                head = f.read(4)
                if head != b"PAR1":
                    return False
                f.seek(-4, 2)
                tail = f.read(4)
                return tail == b"PAR1"
        except OSError:
            return False

    def upsert_daily_sentiment_batch(
        self,
        daily_sentiments: list[DailySentiment],
    ) -> None:
        if not daily_sentiments:
            raise ValueError("No daily sentiments to upsert")

        asset = daily_sentiments[0].asset_id
        if any(s.asset_id != asset for s in daily_sentiments):
            raise ValueError("All daily sentiments in a batch must share the same asset_id")

        rows: list[dict] = []
        for s in daily_sentiments:
            rows.append(
                {
                    "asset_id": self._normalize_symbol(s.asset_id),
                    "day": pd.Timestamp(s.day, tz="UTC"),
                    "sentiment_score": float(s.sentiment_score),
                    "n_articles": int(s.n_articles),
                    "sentiment_std": float(s.sentiment_std) if s.sentiment_std is not None else None,
                }
            )

        df_new = pd.DataFrame(rows)
        df_new = df_new[DAILY_SENTIMENT_PARQUET_COLUMNS]

        df_new["day"] = pd.to_datetime(df_new["day"], utc=True, errors="raise")
        for col, dtype in DAILY_SENTIMENT_PARQUET_DTYPES.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(dtype)

        filepath = self._filepath(asset)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.exists():
            if not self._is_valid_parquet(filepath):
                logger.warning(
                    "Invalid parquet detected for daily sentiment, overwriting",
                    extra={"path": str(filepath.resolve())},
                )
                df = df_new
            else:
                df_old = pd.read_parquet(filepath)
                if not df_old.empty:
                    df_old["day"] = pd.to_datetime(df_old["day"], utc=True, errors="coerce")
                    df = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df = df_new
        else:
            df = df_new

        df = df.drop_duplicates(subset=["day"], keep="last")
        df = df.sort_values("day").reset_index(drop=True)

        missing = set(DAILY_SENTIMENT_PARQUET_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for daily sentiment parquet: {sorted(missing)}")

        df.to_parquet(filepath, index=False)

        logger.info(
            "Daily sentiment upserted",
            extra={
                "asset_id": self._normalize_symbol(asset),
                "saved_rows": len(df_new),
                "total_rows": len(df),
                "path": str(filepath.resolve()),
            },
        )

    def list_daily_sentiment(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DailySentiment]:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        filepath = self._filepath(asset_id)
        if not filepath.exists():
            return []

        df = pd.read_parquet(filepath)
        if df.empty:
            return []

        df["day"] = pd.to_datetime(df["day"], utc=True, errors="coerce")
        if df["day"].isna().any():
            raise ValueError(f"Invalid day values found in {filepath}")

        start_day = pd.Timestamp(start_utc.date(), tz="UTC")
        end_day = pd.Timestamp(end_utc.date(), tz="UTC")

        mask = (df["day"] >= start_day) & (df["day"] <= end_day)
        df = df.loc[mask].sort_values("day")

        out: list[DailySentiment] = []
        for _, r in df.iterrows():
            day_value = r.get("day")
            if isinstance(day_value, pd.Timestamp):
                day_value = day_value.date()
            if not isinstance(day_value, date):
                raise ValueError(f"Invalid day value in {filepath}: {day_value}")

            sentiment_std = r.get("sentiment_std")
            if pd.isna(sentiment_std):
                sentiment_std = None

            out.append(
                DailySentiment(
                    asset_id=str(r.get("asset_id") or self._normalize_symbol(asset_id)),
                    day=day_value,
                    sentiment_score=float(r.get("sentiment_score")),
                    n_articles=int(r.get("n_articles")),
                    sentiment_std=float(sentiment_std) if sentiment_std is not None else None,
                )
            )

        return out
