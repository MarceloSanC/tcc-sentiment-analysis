from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelRepository(ABC):
    @abstractmethod
    def save_training_artifacts(
        self,
        asset_id: str,
        version: str,
        model: Any,
        *,
        metrics: dict[str, float],
        history: list[dict[str, float]],
        split_metrics: dict[str, dict[str, float]],
        features_used: list[str],
        training_window: dict[str, str],
        split_window: dict[str, str],
        config: dict,
        feature_importance: list[dict[str, float | str]] | None = None,
        ablation_results: list[dict[str, float | str]] | None = None,
        checkpoint_path: str | None = None,
        dataset_parameters: dict[str, Any] | None = None,
        plots: dict[str, str] | None = None,
    ) -> str:
        raise NotImplementedError
