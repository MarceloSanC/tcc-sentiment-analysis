# src/adapters/parquet_data_repository.py
from pathlib import Path

import pandas as pd

from src.entities.candle import Candle
from interfaces.candle_repository import CandleRepository


class ParquetCandleRepository(CandleRepository):
    """
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
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_candles(self, symbol: str, candles: list[Candle]) -> None:
        if not candles:
            raise ValueError("No candles to save")

        # Normalizar símbolo para nome de arquivo (ex: PETR4.SA → PETR4)
        clean_symbol = symbol.split(".")[0].upper()
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

        # Forçar dtypes para eficiência e reprodutibilidade
        df = df.astype(
            {
                "open": "float32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "volume": "int64",
            }
        )

        # Ordenar por timestamp (garantir série temporal)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Salvar
        df.to_parquet(filepath, index=False)
        print(f"✅ Salvo {len(candles)} candles em {filepath}")
