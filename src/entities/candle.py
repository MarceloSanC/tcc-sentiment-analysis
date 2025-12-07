# src/entities/candle.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    sentiment_score: Optional[float] = None  # -1.0 (neg) → 0 (neutro) → +1.0 (pos)