from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    metrics: dict[str, float]
    history: list[dict[str, float]]


class ModelTrainer(ABC):
    @abstractmethod
    def train(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        target_col: str,
        time_idx_col: str,
        group_col: str,
        known_real_cols: list[str],
        config: dict,
    ) -> TrainingResult:
        raise NotImplementedError
