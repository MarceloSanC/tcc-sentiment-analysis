from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DuplicateKeyError(ValueError):
    """Raised when an append collides with an existing logical primary key."""


class AnalyticsRunRepository(ABC):
    @abstractmethod
    def upsert_dim_run(self, row: dict[str, Any]) -> None:
        """Upsert a dim_run row by run_id."""
        ...

    @abstractmethod
    def append_fact_run_snapshot(
        self,
        row: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """Append one fact_run_snapshot row."""
        ...

    @abstractmethod
    def append_fact_split_timestamps_ref(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for fact_split_timestamps_ref."""
        ...

    @abstractmethod
    def append_fact_config(
        self,
        row: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """Append one fact_config row."""
        ...

    @abstractmethod
    def append_fact_split_metrics(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for fact_split_metrics."""
        ...

    @abstractmethod
    def append_fact_epoch_metrics(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for fact_epoch_metrics."""
        ...

    @abstractmethod
    def append_fact_oos_predictions(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for fact_oos_predictions."""
        ...

    @abstractmethod
    def append_fact_model_artifacts(
        self,
        row: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """Append one fact_model_artifacts row."""
        ...

    @abstractmethod
    def append_bridge_run_features(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for bridge_run_features."""
        ...

    @abstractmethod
    def append_fact_inference_runs(
        self,
        row: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """Append one fact_inference_runs row."""
        ...

    @abstractmethod
    def append_fact_inference_predictions(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for fact_inference_predictions."""
        ...

    @abstractmethod
    def append_fact_feature_contrib_local(
        self,
        rows: list[dict[str, Any]],
        overwrite: bool = False,
    ) -> None:
        """Append rows for fact_feature_contrib_local."""
        ...

    @abstractmethod
    def append_fact_failures(
        self,
        row: dict[str, Any],
        overwrite: bool = False,
    ) -> None:
        """Append one fact_failures row."""
        ...
