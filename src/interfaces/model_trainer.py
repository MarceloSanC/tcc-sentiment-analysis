from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    metrics: dict[str, float]
    history: list[dict[str, float]]
    split_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_importance: list[dict[str, float | str]] = field(default_factory=list)
    checkpoint_path: str | None = None
    dataset_parameters: dict[str, Any] = field(default_factory=dict)


class ModelTrainer(ABC):
    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        feature_cols: list[str],
        target_col: str,
        time_idx_col: str,
        group_col: str,
        known_real_cols: list[str],
        config: dict,
    ) -> TrainingResult:
        raise NotImplementedError
