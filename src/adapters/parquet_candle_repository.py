# src/adapters/parquet_candle_repository.py
from pathlib import Path

import pandas as pd

from src.entities.candle import Candle
from src.interfaces.candle_repository import CandleRepository

from src.infrastructure.schemas.candle_parquet_schema import (
    CANDLE_PARQUET_COLUMNS,
    CANDLE_PARQUET_DTYPES,
)

class ParquetCandleRepository(CandleRepository):
    """
    Repository adapter for Candle persistence using Parquet files.
    NOTE:
    This repository currently OVERWRITES existing candle files.

    - No incremental append
    - No deduplication by timestamp
    - Full dataset is persisted on each execution

    This behavior is intentional for early-stage development and
    deterministic pipelines.

    TODO(data-pipeline):
    - Support incremental candle updates
    - Deduplicate by timestamp (keep last)
    - Optionally expose persistence mode (overwrite | append | upsert)
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
        
    def _normalize_symbol(self, symbol: str) -> str:
        return symbol.split(".")[0].upper()

    def load_candles(self, symbol: str) -> list[Candle]:
        clean_symbol = self._normalize_symbol(symbol)
        filepath = self.output_dir / f"candles_{clean_symbol}_1d.parquet"

        if not filepath.exists():
            raise FileNotFoundError(
                f"No candle file found for {symbol}\n"
                f"Expected path: {filepath.resolve()}"
            )

        df = pd.read_parquet(filepath)

        missing = CANDLE_PARQUET_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Invalid candle parquet schema for {symbol}. "
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

    def save_candles(self, symbol: str, candles: list[Candle]) -> None:
        if not candles:
            raise ValueError("No candles to save")

        # Normalizar símbolo para nome de arquivo (ex: PETR4.SA → PETR4)
        clean_symbol = self._normalize_symbol(symbol)
        filepath = self.output_dir / f"candles_{clean_symbol}_1d.parquet"

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
