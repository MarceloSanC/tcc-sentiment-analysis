from __future__ import annotations

import logging

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.infrastructure.schemas.analytics_store_schema import (
    ANALYTICS_TABLE_SCHEMAS,
    validate_table_payload,
)
from src.interfaces.analytics_run_repository import (
    AnalyticsRunRepository,
    DuplicateKeyError,
)

logger = logging.getLogger(__name__)


class ParquetAnalyticsRunRepository(AnalyticsRunRepository):
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_partition(value: Any) -> str:
        if value is None:
            return "__none__"
        text = str(value).strip()
        return text if text else "__none__"

    def _dim_run_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        sweep = self._safe_partition(row.get("parent_sweep_id"))
        return self.output_dir / "dim_run" / f"asset={asset}" / f"sweep_id={sweep}" / "dim_run.parquet"

    def _fact_run_snapshot_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        sweep = self._safe_partition(row.get("parent_sweep_id"))
        return self.output_dir / "fact_run_snapshot" / f"asset={asset}" / f"sweep_id={sweep}" / "fact_run_snapshot.parquet"

    def _fact_config_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        sweep = self._safe_partition(row.get("parent_sweep_id"))
        return self.output_dir / "fact_config" / f"asset={asset}" / f"sweep_id={sweep}" / "fact_config.parquet"

    def _fact_split_timestamps_ref_path(self, row: dict[str, Any]) -> Path:
        run_id = self._safe_partition(row.get("run_id"))
        return self.output_dir / "fact_split_timestamps_ref" / f"run_id={run_id}" / "fact_split_timestamps_ref.parquet"

    def _fact_failures_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        return self.output_dir / "fact_failures" / f"asset={asset}" / "fact_failures.parquet"

    def _fact_epoch_metrics_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        sweep = self._safe_partition(row.get("parent_sweep_id"))
        fold = self._safe_partition(row.get("fold"))
        return self.output_dir / "fact_epoch_metrics" / f"asset={asset}" / f"sweep_id={sweep}" / f"fold={fold}" / "fact_epoch_metrics.parquet"

    def _fact_split_metrics_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        sweep = self._safe_partition(row.get("parent_sweep_id"))
        return self.output_dir / "fact_split_metrics" / f"asset={asset}" / f"sweep_id={sweep}" / "fact_split_metrics.parquet"

    def _fact_oos_predictions_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        feature_set_name = self._safe_partition(row.get("feature_set_name"))
        year = self._safe_partition(row.get("year"))
        return self.output_dir / "fact_oos_predictions" / f"asset={asset}" / f"feature_set_name={feature_set_name}" / f"year={year}" / "fact_oos_predictions.parquet"

    def _fact_model_artifacts_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        return self.output_dir / "fact_model_artifacts" / f"asset={asset}" / "fact_model_artifacts.parquet"

    def _bridge_run_features_path(self, row: dict[str, Any]) -> Path:
        run_id = self._safe_partition(row.get("run_id"))
        return self.output_dir / "bridge_run_features" / f"run_id={run_id}" / "bridge_run_features.parquet"

    def _fact_inference_runs_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        return self.output_dir / "fact_inference_runs" / f"asset={asset}" / "fact_inference_runs.parquet"

    def _fact_inference_predictions_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        model_version = self._safe_partition(row.get("model_version"))
        year = self._safe_partition(row.get("year"))
        return self.output_dir / "fact_inference_predictions" / f"asset={asset}" / f"model_version={model_version}" / f"year={year}" / "fact_inference_predictions.parquet"

    def _fact_feature_contrib_local_path(self, row: dict[str, Any]) -> Path:
        asset = self._safe_partition(row.get("asset"))
        model_version = self._safe_partition(row.get("model_version"))
        year = self._safe_partition(row.get("year"))
        return self.output_dir / "fact_feature_contrib_local" / f"asset={asset}" / f"model_version={model_version}" / f"year={year}" / "fact_feature_contrib_local.parquet"

    @staticmethod
    def _pk_tuples(df: pd.DataFrame, pk_cols: tuple[str, ...]) -> set[tuple[Any, ...]]:
        return set(df.loc[:, list(pk_cols)].itertuples(index=False, name=None))

    def _write_with_overwrite_policy(
        self,
        *,
        table_name: str,
        path: Path,
        incoming: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        schema = ANALYTICS_TABLE_SCHEMAS[table_name]
        pk_cols = schema.logical_pk

        if not path.exists():
            incoming.to_parquet(path, index=False)
            return

        current = pd.read_parquet(path)
        current_keys = self._pk_tuples(current, pk_cols)
        incoming_keys = self._pk_tuples(incoming, pk_cols)
        collisions = current_keys.intersection(incoming_keys)

        if collisions and not overwrite:
            sample = sorted(collisions, key=str)[:5]
            raise DuplicateKeyError(
                f"Duplicate logical PK collision in {table_name}: "
                f"pk_columns={pk_cols} collisions={sample} path={path}"
            )

        if collisions:
            current_pk = pd.MultiIndex.from_frame(current.loc[:, list(pk_cols)])
            collision_index = pd.MultiIndex.from_tuples(
                list(collisions),
                names=list(pk_cols),
            )
            current = current.loc[~current_pk.isin(collision_index)]

        merged = pd.concat([current, incoming], ignore_index=True)
        merged.to_parquet(path, index=False)

    def _append_rows_partitioned(
        self,
        rows: list[dict[str, Any]],
        path_resolver: Any,
    ) -> None:
        """
        Append rows in batch per partition file path to avoid row-by-row
        parquet rewrites.
        """
        if not rows:
            return

        buckets: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            path = path_resolver(row)
            buckets[path].append(row)

        for path, bucket_rows in buckets.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            incoming = pd.DataFrame(bucket_rows)
            self._append_to_parquet(path, incoming)

    def upsert_dim_run(self, row: dict[str, Any]) -> None:
        validate_table_payload("dim_run", [row])

        path = self._dim_run_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame([row])

        if path.exists():
            current = pd.read_parquet(path)
            current = current[current["run_id"] != row["run_id"]]
            merged = pd.concat([current, incoming], ignore_index=True)
        else:
            merged = incoming

        merged.to_parquet(path, index=False)
        logger.info(
            "analytics dim_run upserted",
            extra={"path": str(path.resolve()), "run_id": row.get("run_id")},
        )

    def append_fact_run_snapshot(self, row: dict[str, Any]) -> None:
        validate_table_payload("fact_run_snapshot", [row])
        path = self._fact_run_snapshot_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame([row])
        self._append_to_parquet(path, incoming)
        logger.info(
            "analytics fact_run_snapshot appended",
            extra={"path": str(path.resolve()), "run_id": row.get("run_id")},
        )

    def append_fact_config(self, row: dict[str, Any]) -> None:
        validate_table_payload("fact_config", [row])
        path = self._fact_config_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame([row])
        self._append_to_parquet(path, incoming)
        logger.info(
            "analytics fact_config appended",
            extra={"path": str(path.resolve()), "run_id": row.get("run_id")},
        )

    def append_fact_split_metrics(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("fact_split_metrics", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._fact_split_metrics_path)
        logger.info(
            "analytics fact_split_metrics appended",
            extra={"run_id": rows[0].get("run_id"), "rows": len(rows)},
        )

    def append_fact_split_timestamps_ref(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("fact_split_timestamps_ref", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._fact_split_timestamps_ref_path)
        logger.info(
            "analytics fact_split_timestamps_ref appended",
            extra={"run_id": rows[0].get("run_id"), "rows": len(rows)},
        )

    def append_fact_epoch_metrics(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("fact_epoch_metrics", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._fact_epoch_metrics_path)
        logger.info(
            "analytics fact_epoch_metrics appended",
            extra={"run_id": rows[0].get("run_id"), "rows": len(rows)},
        )

    def append_fact_oos_predictions(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("fact_oos_predictions", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._fact_oos_predictions_path)
        logger.info(
            "analytics fact_oos_predictions appended",
            extra={"run_id": rows[0].get("run_id"), "rows": len(rows)},
        )

    def append_fact_model_artifacts(self, row: dict[str, Any]) -> None:
        validate_table_payload("fact_model_artifacts", [row])
        path = self._fact_model_artifacts_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame([row])
        self._append_to_parquet(path, incoming)
        logger.info(
            "analytics fact_model_artifacts appended",
            extra={"path": str(path.resolve()), "run_id": row.get("run_id")},
        )

    def append_bridge_run_features(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("bridge_run_features", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._bridge_run_features_path)
        logger.info(
            "analytics bridge_run_features appended",
            extra={"run_id": rows[0].get("run_id"), "rows": len(rows)},
        )

    def append_fact_inference_runs(self, row: dict[str, Any]) -> None:
        validate_table_payload("fact_inference_runs", [row])
        path = self._fact_inference_runs_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame([row])
        self._append_to_parquet(path, incoming)
        logger.info(
            "analytics fact_inference_runs appended",
            extra={"path": str(path.resolve()), "inference_run_id": row.get("inference_run_id")},
        )

    def append_fact_inference_predictions(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("fact_inference_predictions", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._fact_inference_predictions_path)
        logger.info(
            "analytics fact_inference_predictions appended",
            extra={"inference_run_id": rows[0].get("inference_run_id"), "rows": len(rows)},
        )

    def append_fact_feature_contrib_local(self, rows: list[dict[str, Any]]) -> None:
        validate_table_payload("fact_feature_contrib_local", rows)
        if not rows:
            return
        self._append_rows_partitioned(rows, self._fact_feature_contrib_local_path)
        logger.info(
            "analytics fact_feature_contrib_local appended",
            extra={"inference_run_id": rows[0].get("inference_run_id"), "rows": len(rows)},
        )

    def append_fact_failures(self, row: dict[str, Any]) -> None:
        validate_table_payload("fact_failures", [row])
        path = self._fact_failures_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = pd.DataFrame([row])
        self._append_to_parquet(path, incoming)
        logger.info(
            "analytics fact_failures appended",
            extra={"path": str(path.resolve()), "run_id": row.get("run_id")},
        )
