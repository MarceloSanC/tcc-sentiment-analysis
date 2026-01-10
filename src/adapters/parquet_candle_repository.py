# src/adapters/parquet_candle_repository.py
from pathlib import Path

import pandas as pd

from src.entities.candle import Candle
from src.entities.daily_sentiment import DailySentiment
from src.interfaces.candle_repository import CandleRepository

from src.infrastructure.schemas.candle_parquet_schema import (
    CANDLE_PARQUET_COLUMNS,
    CANDLE_PARQUET_DTYPES,
)

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
    def _normalize_symbol(self, asset_id: str) -> str:
        return asset_id.split(".")[0].upper()
    
    def _filepath(self, asset_id: str) -> Path:
        clean_symbol = self._normalize_symbol(asset_id)
        return self.output_dir / f"candles_{clean_symbol}_1d.parquet"

    def load_candles(self, asset_id: str) -> list[Candle]:
        filepath = self._filepath(asset_id)

        if not filepath.exists():
            raise FileNotFoundError(
                f"No candle file found for {asset_id}\n"
                f"Expected path: {filepath.resolve()}"
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

        # Garantia explícita de ordenação temporal
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

        return candles

    def save_candles(self, asset_id: str, candles: list[Candle]) -> None:
        if not candles:
            raise ValueError("No candles to save")

        # Normalizar símbolo para nome de arquivo (ex: PETR4.SA → PETR4)
        filepath = self._filepath(asset_id)

        # Converter para DataFrame com schema explícito
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

        # Garantir contrato de colunas
        df = df[list(CANDLE_PARQUET_COLUMNS)]

        # Forçar dtypes para eficiência e reprodutibilidade
        df = df.astype(CANDLE_PARQUET_DTYPES)

        # Ordenar por timestamp (garantir série temporal)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Salvar
        df.to_parquet(filepath, index=False)
        print(f"✅ Salvo {len(candles)} candles em {filepath}")

    def update_sentiment(
        self,
        asset_id: str,
        daily_sentiments: list[DailySentiment],
    ) -> None:
        """
        Atualiza candles existentes com sentimento diário agregado.

        Este método preserva a série temporal de candles e realiza
        um enriquecimento por junção temporal (date-based join),
        evitando recriação completa do dataset.
        """

        if not daily_sentiments:
            return

        clean_symbol = self._normalize_symbol(asset_id)
        filepath = self.output_dir / f"candles_{clean_symbol}_1d.parquet"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Cannot update sentiment: candle file not found for {asset_id}"
            )

        # Load existing candles
        df_candles = pd.read_parquet(filepath)

        # Prepare sentiment DataFrame (domain → infra)
        df_sentiment = pd.DataFrame(
            [
                {
                    "date": ds.day,
                    "sentiment_score": ds.sentiment_score,
                    "sentiment_std": ds.sentiment_std,
                    "n_articles": ds.n_articles,
                }
                for ds in daily_sentiments
            ]
        )

        # Normalize candle date
        df_candles["date"] = pd.to_datetime(
            df_candles["timestamp"]
        ).dt.date

        # Left join: candles × sentiment
        df_updated = df_candles.merge(
            df_sentiment,
            on="date",
            how="left",
        )

        # Cleanup helper column
        df_updated = df_updated.drop(columns=["date"])

        # Persist updated candles
        df_updated.to_parquet(filepath, index=False)

        print(
            f"✅ Updated sentiment for {len(daily_sentiments)} days "
            f"in {filepath.name}"
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