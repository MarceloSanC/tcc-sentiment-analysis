# src/entities/candle.py
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self):
        if self.close <= 0:
            raise ValueError("Close price must be positive")

        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
