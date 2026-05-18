from __future__ import annotations

import json
import warnings

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.services.quantile_contract_analyzer import (
    QuantileBlockAThresholds,
    QuantileContractAnalyzer,
    QuantileDegeneracyThresholds,
)
from src.domain.services.scope_spec import ScopeSpec, validate_scope_spec


@dataclass(frozen=True)
class AnalyticsQualityResult:
    passed: bool
    checks: list[dict[str, object]]


class ValidateAnalyticsQualityUseCase:
    def __init__(
        self,
        *,
        analytics_silver_dir: str | Path,
        analytics_gold_dir: str | Path | None = None,
        scope_spec: ScopeSpec | None = None,
        min_samples_train: int = 1,
        min_samples_val: int = 1,
        min_samples_test: int = 1,
        block_a_parent_sweep_prefixes: list[str] | None = None,
        block_a_splits: list[str] | None = None,
        block_a_horizons: list[int] | None = None,
        block_a_max_crossing_bruto_rate: float = 0.001,
        block_a_max_negative_interval_width_count: int = 0,
        block_a_max_crossing_post_guardrail_rate: float = 0.0,
        block_a_require_post_guardrail: bool = False,
        degeneracy_min_rows_for_gate: int = 1000,
        degeneracy_max_p10_eq_p90_rate: float = 0.05,
    ) -> None:
        self.analytics_silver_dir = Path(analytics_silver_dir)
        self.analytics_gold_dir = Path(analytics_gold_dir) if analytics_gold_dir is not None else None
        self.scope_spec = scope_spec
        self.min_samples_train = int(min_samples_train)
        self.min_samples_val = int(min_samples_val)
        self.min_samples_test = int(min_samples_test)

        self.block_a_parent_sweep_prefixes = [
            str(v).strip() for v in (block_a_parent_sweep_prefixes or []) if str(v).strip()
        ]
        self.block_a_splits = [str(v).strip() for v in (block_a_splits or []) if str(v).strip()]
        self.block_a_horizons = [int(v) for v in (block_a_horizons or [])]
        self.block_a_thresholds = QuantileBlockAThresholds(
            max_crossing_bruto_rate=float(block_a_max_crossing_bruto_rate),
            max_negative_interval_width_count=int(block_a_max_negative_interval_width_count),
            max_crossing_post_guardrail_rate=float(block_a_max_crossing_post_guardrail_rate),
            require_post_guardrail=bool(block_a_require_post_guardrail),
        )
        self.degeneracy_thresholds = QuantileDegeneracyThresholds(
            min_rows_for_gate=int(degeneracy_min_rows_for_gate),
            max_p10_eq_p90_rate=float(degeneracy_max_p10_eq_p90_rate),
        )

    @staticmethod
    def _load_partitioned_table(base_dir: Path, table_name: str) -> pd.DataFrame:
        # Supports both partitioned tables (<base>/<table>/**/*.parquet)
        # and flat parquet files (<base>/<table>.parquet), used by gold outputs.
        flat_file = base_dir / f"{table_name}.parquet"
        if flat_file.exists():
            return pd.read_parquet(flat_file)

        table_dir = base_dir / table_name
        if not table_dir.exists():
            return pd.DataFrame()
        files = sorted(table_dir.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(fp) for fp in files]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _record(checks: list[dict[str, object]], name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    def _resolve_scope_spec(self) -> ScopeSpec:
        legacy_scope_used = bool(
            self.block_a_parent_sweep_prefixes or self.block_a_splits or self.block_a_horizons
        )
        if legacy_scope_used:
            warnings.warn(
                "block_a_parent_sweep_prefixes/block_a_splits/block_a_horizons are deprecated; "
                "use scope_spec=ScopeSpec(...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.scope_spec is not None:
            return validate_scope_spec(self.scope_spec)

        if legacy_scope_used:
            return validate_scope_spec(
                ScopeSpec.create(
                    scope_mode="cohort_decision",
                    parent_sweep_prefixes=self.block_a_parent_sweep_prefixes,
                    splits=self.block_a_splits,
                    horizons=self.block_a_horizons,
                )
            )

        return validate_scope_spec(ScopeSpec.create(scope_mode="global_health"))

    @staticmethod
    def _scope_detail(scope_spec: ScopeSpec) -> str:
        return (
            "scope_mode="
            + str(scope_spec.scope_mode)
            + ", scope_parent_sweep_prefixes="
            + str(list(scope_spec.parent_sweep_prefixes))
            + ", scope_splits="
            + str(list(scope_spec.splits))
            + ", scope_horizons="
            + str(list(scope_spec.horizons))
        )

    @staticmethod
    def _scope_table(df: pd.DataFrame, *, scope_spec: ScopeSpec, run_ids: set[str] | None) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        out = df.copy()

        if run_ids is not None and "run_id" in out.columns:
            out = out[out["run_id"].astype(str).isin(run_ids)].copy()

        if scope_spec.parent_sweep_prefixes and "parent_sweep_id" in out.columns:
            out = out[
                out["parent_sweep_id"].astype(str).apply(
                    lambda value: any(value.startswith(prefix) for prefix in scope_spec.parent_sweep_prefixes)
                )
            ].copy()

        if scope_spec.splits and "split" in out.columns:
            out = out[out["split"].astype(str).isin(set(scope_spec.splits))].copy()

        if scope_spec.horizons and "horizon" in out.columns:
            horizon = pd.to_numeric(out["horizon"], errors="coerce")
            out = out[horizon.isin(set(scope_spec.horizons))].copy()

        return out

    @staticmethod
    def _build_scoped_run_ids(
        *,
        dim_run: pd.DataFrame,
        fact_oos_predictions: pd.DataFrame,
        scope_spec: ScopeSpec,
    ) -> set[str]:
        if dim_run.empty or "run_id" not in dim_run.columns:
            return set()

        dim_scoped = ValidateAnalyticsQualityUseCase._scope_table(
            dim_run,
            scope_spec=ScopeSpec.create(
                scope_mode=scope_spec.scope_mode,
                parent_sweep_prefixes=scope_spec.parent_sweep_prefixes,
            ),
            run_ids=None,
        )
        run_ids = set(dim_scoped["run_id"].dropna().astype(str).tolist())
        if not run_ids:
            return set()

        if not (scope_spec.splits or scope_spec.horizons):
            return run_ids

        if fact_oos_predictions.empty or "run_id" not in fact_oos_predictions.columns:
            return set()

        oos_scoped = ValidateAnalyticsQualityUseCase._scope_table(
            fact_oos_predictions,
            scope_spec=ScopeSpec.create(
                scope_mode=scope_spec.scope_mode,
                splits=scope_spec.splits,
                horizons=scope_spec.horizons,
            ),
            run_ids=run_ids,
        )
        if oos_scoped.empty:
            return set()
        return set(oos_scoped["run_id"].dropna().astype(str).tolist())

    def execute(self) -> AnalyticsQualityResult:
        checks: list[dict[str, object]] = []

        dim_run = self._load_partitioned_table(self.analytics_silver_dir, "dim_run")
        fact_run_snapshot = self._load_partitioned_table(self.analytics_silver_dir, "fact_run_snapshot")
        fact_config = self._load_partitioned_table(self.analytics_silver_dir, "fact_config")
        fact_split_metrics = self._load_partitioned_table(self.analytics_silver_dir, "fact_split_metrics")
        fact_epoch_metrics = self._load_partitioned_table(self.analytics_silver_dir, "fact_epoch_metrics")
        fact_oos_predictions = self._load_partitioned_table(self.analytics_silver_dir, "fact_oos_predictions")
        fact_model_artifacts = self._load_partitioned_table(self.analytics_silver_dir, "fact_model_artifacts")
        fact_inference_runs = self._load_partitioned_table(self.analytics_silver_dir, "fact_inference_runs")
        fact_inference_predictions = self._load_partitioned_table(self.analytics_silver_dir, "fact_inference_predictions")
        fact_feature_contrib_local = self._load_partitioned_table(self.analytics_silver_dir, "fact_feature_contrib_local")
        fact_failures = self._load_partitioned_table(self.analytics_silver_dir, "fact_failures")
        bridge_run_features = self._load_partitioned_table(self.analytics_silver_dir, "bridge_run_features")
        fact_split_timestamps_ref = self._load_partitioned_table(
            self.analytics_silver_dir, "fact_split_timestamps_ref"
        )

        scope_spec = self._resolve_scope_spec()
        scope_detail = self._scope_detail(scope_spec)
        scoped_run_ids = self._build_scoped_run_ids(
            dim_run=dim_run,
            fact_oos_predictions=fact_oos_predictions,
            scope_spec=scope_spec,
        )
        run_id_scope_filter: set[str] | None = scoped_run_ids if scope_spec.has_cohort_filters() else None

        dim_run = self._scope_table(dim_run, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_run_snapshot = self._scope_table(fact_run_snapshot, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_config = self._scope_table(fact_config, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_split_metrics = self._scope_table(fact_split_metrics, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_epoch_metrics = self._scope_table(fact_epoch_metrics, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_oos_predictions = self._scope_table(fact_oos_predictions, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_model_artifacts = self._scope_table(fact_model_artifacts, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_inference_runs = self._scope_table(fact_inference_runs, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_inference_predictions = self._scope_table(
            fact_inference_predictions, scope_spec=scope_spec, run_ids=run_id_scope_filter
        )
        fact_feature_contrib_local = self._scope_table(
            fact_feature_contrib_local, scope_spec=scope_spec, run_ids=run_id_scope_filter
        )
        fact_failures = self._scope_table(fact_failures, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        bridge_run_features = self._scope_table(bridge_run_features, scope_spec=scope_spec, run_ids=run_id_scope_filter)
        fact_split_timestamps_ref = self._scope_table(
            fact_split_timestamps_ref, scope_spec=scope_spec, run_ids=run_id_scope_filter
        )

        required_tables = {
            "dim_run": dim_run,
            "fact_run_snapshot": fact_run_snapshot,
            "fact_config": fact_config,
            "fact_split_metrics": fact_split_metrics,
            "fact_epoch_metrics": fact_epoch_metrics,
            "fact_oos_predictions": fact_oos_predictions,
            "fact_model_artifacts": fact_model_artifacts,
            "bridge_run_features": bridge_run_features,
        }
        missing_required_tables = [name for name, df in required_tables.items() if df.empty]
        self._record(
            checks,
            "required_tables_presence",
            len(missing_required_tables) == 0,
            "ok" if not missing_required_tables else ", ".join(missing_required_tables),
        )

        # P0: inference table continuity (run summary -> row-level predictions).
        inf_continuity_ok = True
        inf_continuity_detail = "ok"
        if not fact_inference_runs.empty:
            if fact_inference_predictions.empty:
                inf_continuity_ok = False
                inf_continuity_detail = "fact_inference_predictions empty while fact_inference_runs has rows"
            elif "inference_run_id" not in fact_inference_predictions.columns:
                inf_continuity_ok = False
                inf_continuity_detail = "fact_inference_predictions missing inference_run_id"
            else:
                run_ids = set(fact_inference_runs.get("inference_run_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
                pred_ids = set(fact_inference_predictions["inference_run_id"].dropna().astype(str).tolist())
                missing_inf = sorted(run_ids - pred_ids)
                if missing_inf:
                    inf_continuity_ok = False
                    inf_continuity_detail = f"inference_runs_without_predictions={len(missing_inf)}"
        self._record(checks, "inference_predictions_continuity", inf_continuity_ok, inf_continuity_detail)

        local_continuity_ok = True
        local_continuity_detail = "ok"
        if not fact_inference_predictions.empty:
            if fact_feature_contrib_local.empty:
                local_continuity_ok = False
                local_continuity_detail = "fact_feature_contrib_local empty while fact_inference_predictions has rows"
            elif "inference_run_id" not in fact_feature_contrib_local.columns:
                local_continuity_ok = False
                local_continuity_detail = "fact_feature_contrib_local missing inference_run_id"
            else:
                pred_ids = set(fact_inference_predictions.get("inference_run_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
                local_ids = set(fact_feature_contrib_local["inference_run_id"].dropna().astype(str).tolist())
                missing_local = sorted(pred_ids - local_ids)
                if missing_local:
                    local_continuity_ok = False
                    local_continuity_detail = f"inference_runs_without_local_contrib={len(missing_local)}"
        self._record(checks, "feature_contrib_local_continuity", local_continuity_ok, local_continuity_detail)

        # P0: run_id/execution_id consistency.
        if dim_run.empty:
            self._record(checks, "run_id_execution_consistency", False, "dim_run is empty")
        else:
            run_id_null = int(dim_run["run_id"].isna().sum()) if "run_id" in dim_run else len(dim_run)
            dup_run_id = int(dim_run["run_id"].duplicated().sum()) if "run_id" in dim_run else len(dim_run)
            non_null_exec = (
                dim_run["execution_id"].dropna().astype(str).str.strip()
                if "execution_id" in dim_run.columns
                else pd.Series(dtype=str)
            )
            dup_exec_id = int(non_null_exec.duplicated().sum()) if not non_null_exec.empty else 0
            ok = run_id_null == 0 and dup_run_id == 0 and dup_exec_id == 0
            self._record(
                checks,
                "run_id_execution_consistency",
                ok,
                f"run_id_null={run_id_null}, duplicate_run_id={dup_run_id}, duplicate_execution_id={dup_exec_id}",
            )

        # P0: referential integrity facts -> dim_run
        fact_tables = {
            "fact_run_snapshot": fact_run_snapshot,
            "fact_config": fact_config,
            "fact_split_metrics": fact_split_metrics,
            "fact_epoch_metrics": fact_epoch_metrics,
            "fact_oos_predictions": fact_oos_predictions,
            "fact_model_artifacts": fact_model_artifacts,
            "fact_inference_runs": fact_inference_runs,
            "fact_inference_predictions": fact_inference_predictions,
            "fact_feature_contrib_local": fact_feature_contrib_local,
            "fact_failures": fact_failures,
            "bridge_run_features": bridge_run_features,
            "fact_split_timestamps_ref": fact_split_timestamps_ref,
        }
        dim_ids = set(dim_run["run_id"].dropna().astype(str).tolist()) if "run_id" in dim_run.columns else set()
        missing_refs: list[str] = []
        for table_name, df in fact_tables.items():
            if df.empty or "run_id" not in df.columns:
                continue
            ids = set(df["run_id"].dropna().astype(str).tolist())
            missing = ids - dim_ids
            if missing:
                missing_refs.append(f"{table_name}:{len(missing)}")
        self._record(
            checks,
            "referential_integrity",
            len(missing_refs) == 0,
            "ok" if not missing_refs else ", ".join(missing_refs),
        )

        # P0: no NaN in required metrics
        nan_issues: list[str] = []
        for df, cols, label in [
            (fact_split_metrics, ["rmse", "mae", "directional_accuracy", "n_samples"], "fact_split_metrics"),
            (fact_epoch_metrics, ["train_loss", "val_loss"], "fact_epoch_metrics"),
        ]:
            if df.empty:
                continue
            for col in cols:
                if col not in df.columns:
                    nan_issues.append(f"{label}.{col}:missing")
                    continue
                values = pd.to_numeric(df[col], errors="coerce")
                n_bad = int(values.isna().sum())
                if n_bad > 0:
                    nan_issues.append(f"{label}.{col}:nan={n_bad}")
        self._record(checks, "required_metrics_nan", len(nan_issues) == 0, "ok" if not nan_issues else ", ".join(nan_issues))

        # P0: temporal consistency + OOS key/type/coverage checks
        temporal_issues: list[str] = []
        oos_unique_issues: list[str] = []
        oos_type_issues: list[str] = []
        oos_coverage_issues: list[str] = []
        oos_supervised_null_issues: list[str] = []
        oos_interval_issues: list[str] = []
        oos_quantile_order_issues: list[str] = []
        oos_alignment_issues: list[str] = []
        expected_by_run: dict[str, list[int]] = {}

        if not fact_oos_predictions.empty:
            required_ts_cols = ["timestamp_utc", "target_timestamp_utc"]
            missing_ts_cols = [c for c in required_ts_cols if c not in fact_oos_predictions.columns]
            if missing_ts_cols:
                temporal_issues.append(f"missing_timestamp_columns={missing_ts_cols}")
            else:
                ts = pd.to_datetime(fact_oos_predictions["timestamp_utc"], utc=True, errors="coerce")
                tgt = pd.to_datetime(
                    fact_oos_predictions["target_timestamp_utc"], utc=True, errors="coerce"
                )
                bad_parse = int(ts.isna().sum() + tgt.isna().sum())
                if bad_parse > 0:
                    temporal_issues.append(f"timestamp_parse_errors={bad_parse}")
                bad_order = int((tgt < ts).sum())
                if bad_order > 0:
                    temporal_issues.append(f"target_before_timestamp={bad_order}")

                group_cols = [
                    c for c in ["run_id", "split", "horizon"] if c in fact_oos_predictions.columns
                ]
                if group_cols:
                    tmp = fact_oos_predictions.copy()
                    tmp["_ts"] = ts
                    tmp["_tgt"] = tgt
                    for _, g in tmp.groupby(group_cols, dropna=False):
                        ordered = g.sort_values(["_ts", "_tgt"], kind="mergesort")
                        tgt_series = ordered["_tgt"].dropna()
                        if not tgt_series.is_monotonic_increasing:
                            temporal_issues.append("non_monotonic_target_timestamp_in_group")
                            break

            # Key uniqueness: run_id + split + horizon + timestamp + target_timestamp
            key_cols = [
                "run_id",
                "split",
                "horizon",
                "timestamp_utc",
                "target_timestamp_utc",
            ]
            missing_key_cols = [c for c in key_cols if c not in fact_oos_predictions.columns]
            if missing_key_cols:
                oos_unique_issues.append(f"missing_key_columns={missing_key_cols}")
            else:
                dup_count = int(fact_oos_predictions.duplicated(subset=key_cols).sum())
                if dup_count > 0:
                    oos_unique_issues.append(f"duplicate_oos_rows={dup_count}")

            # Numeric consistency in prediction/error columns.
            numeric_cols = [
                "y_true",
                "y_pred",
                "error",
                "abs_error",
                "sq_error",
                "quantile_p10",
                "quantile_p50",
                "quantile_p90",
            ]
            for col in numeric_cols:
                if col not in fact_oos_predictions.columns:
                    oos_type_issues.append(f"missing_numeric_column={col}")
                    continue
                src = fact_oos_predictions[col]
                coerced = pd.to_numeric(src, errors="coerce")
                bad_non_numeric = int((src.notna() & coerced.isna()).sum())
                if bad_non_numeric > 0:
                    oos_type_issues.append(f"non_numeric_{col}={bad_non_numeric}")

            # Supervised splits should not have null y_true/y_pred.
            if {"split", "y_true", "y_pred"}.issubset(set(fact_oos_predictions.columns)):
                supervised = fact_oos_predictions[
                    fact_oos_predictions["split"].astype(str).isin(["train", "val", "test"])
                ]
                null_true = int(supervised["y_true"].isna().sum())
                null_pred = int(supervised["y_pred"].isna().sum())
                if null_true > 0:
                    oos_supervised_null_issues.append(f"null_y_true={null_true}")
                if null_pred > 0:
                    oos_supervised_null_issues.append(f"null_y_pred={null_pred}")

            # Prediction interval checks use post-guardrail quantiles when available.
            q10_col = "quantile_p10_post_guardrail" if "quantile_p10_post_guardrail" in fact_oos_predictions.columns else "quantile_p10"
            q50_col = "quantile_p50_post_guardrail" if "quantile_p50_post_guardrail" in fact_oos_predictions.columns else "quantile_p50"
            q90_col = "quantile_p90_post_guardrail" if "quantile_p90_post_guardrail" in fact_oos_predictions.columns else "quantile_p90"

            if {q10_col, q90_col}.issubset(set(fact_oos_predictions.columns)):
                q10 = pd.to_numeric(fact_oos_predictions[q10_col], errors="coerce")
                q90 = pd.to_numeric(fact_oos_predictions[q90_col], errors="coerce")
                negative_width = int(((~q10.isna()) & (~q90.isna()) & ((q90 - q10) < 0.0)).sum())
                if negative_width > 0:
                    oos_interval_issues.append(f"negative_interval_width={negative_width}|source={q10_col},{q90_col}")

            if {q10_col, q50_col, q90_col}.issubset(set(fact_oos_predictions.columns)):
                q10 = pd.to_numeric(fact_oos_predictions[q10_col], errors="coerce")
                q50 = pd.to_numeric(fact_oos_predictions[q50_col], errors="coerce")
                q90 = pd.to_numeric(fact_oos_predictions[q90_col], errors="coerce")
                bad_order = int(((~q10.isna()) & (~q50.isna()) & (~q90.isna()) & ((q10 > q50) | (q50 > q90))).sum())
                if bad_order > 0:
                    oos_quantile_order_issues.append(f"invalid_quantile_order={bad_order}|source={q10_col},{q50_col},{q90_col}")

            # Horizon coverage per run_id/split using fact_config.evaluation_horizons_json.
            if not fact_config.empty and "run_id" in fact_config.columns:
                cfg_cols = [c for c in ["run_id", "evaluation_horizons_json", "max_prediction_length"] if c in fact_config.columns]
                cfg_df = fact_config[cfg_cols].dropna(subset=["run_id"]) if cfg_cols else pd.DataFrame()
                for _, cfg_row in cfg_df.iterrows():
                    run_key = str(cfg_row["run_id"])
                    parsed: list[int] = []
                    raw_h = cfg_row.get("evaluation_horizons_json")
                    if isinstance(raw_h, str) and raw_h.strip():
                        try:
                            obj = json.loads(raw_h)
                            if isinstance(obj, list):
                                parsed = [max(1, int(v)) for v in obj if isinstance(v, (int, float))]
                        except json.JSONDecodeError:
                            parsed = []
                    if not parsed:
                        mpl = pd.to_numeric(cfg_row.get("max_prediction_length"), errors="coerce")
                        parsed = [1] if pd.isna(mpl) or int(mpl) < 1 else [1]
                    expected_by_run[run_key] = sorted(set(parsed))

            if {"run_id", "split", "horizon"}.issubset(set(fact_oos_predictions.columns)):
                tmp = fact_oos_predictions.copy()
                tmp["horizon"] = pd.to_numeric(tmp["horizon"], errors="coerce")
                for (run_id, split), g in tmp.groupby(["run_id", "split"], dropna=False):
                    run_key = str(run_id)
                    expected = expected_by_run.get(run_key, [1])
                    actual = sorted(
                        {
                            int(h)
                            for h in g["horizon"].dropna().tolist()
                            if pd.notna(h)
                        }
                    )
                    missing_horizons = sorted(set(expected) - set(actual))
                    if missing_horizons:
                        oos_coverage_issues.append(
                            f"run_id={run_key}|split={split}|missing_horizons={missing_horizons}|expected={expected}|actual={actual}"
                        )

            # Pairwise alignment readiness by target_timestamp across models.
            needed = {"run_id", "split", "horizon", "target_timestamp_utc"}
            if needed.issubset(set(fact_oos_predictions.columns)) and not dim_run.empty and "run_id" in dim_run.columns:
                meta_cols = [c for c in ["run_id", "asset", "config_signature", "parent_sweep_id"] if c in dim_run.columns]
                aligned = fact_oos_predictions.merge(dim_run[meta_cols].drop_duplicates("run_id"), on="run_id", how="left")
                if "config_signature" not in aligned.columns:
                    if "config_signature_x" in aligned.columns:
                        aligned["config_signature"] = aligned["config_signature_x"]
                    elif "config_signature_y" in aligned.columns:
                        aligned["config_signature"] = aligned["config_signature_y"]
                if "asset" not in aligned.columns:
                    if "asset_x" in aligned.columns:
                        aligned["asset"] = aligned["asset_x"]
                    elif "asset_y" in aligned.columns:
                        aligned["asset"] = aligned["asset_y"]
                if {"asset", "config_signature"}.issubset(set(aligned.columns)):
                    aligned = aligned[aligned["split"].astype(str).isin(["val", "test"])].copy()
                    aligned["target_timestamp_utc"] = pd.to_datetime(aligned["target_timestamp_utc"], utc=True, errors="coerce")
                    gcols = [c for c in ["asset", "parent_sweep_id", "split_signature", "split", "horizon"] if c in aligned.columns]
                    if gcols:
                        for keys, grp in aligned.groupby(gcols, dropna=False):
                            kv = dict(zip(gcols, keys if isinstance(keys, tuple) else (keys,)))
                            sweep_value = kv.get("parent_sweep_id")
                            if pd.isna(sweep_value) or str(sweep_value).strip().lower() in {"", "none", "nan", "null", "<na>"}:
                                continue
                            per_cfg: dict[str, set[str]] = {}
                            for cfg, gc in grp.groupby("config_signature", dropna=False):
                                per_cfg[str(cfg)] = set(gc["target_timestamp_utc"].dropna().astype(str).tolist())
                            if len(per_cfg) < 2:
                                continue
                            sets = list(per_cfg.values())
                            base = sets[0]
                            if any(s != base for s in sets[1:]):
                                min_len = min(len(s) for s in sets)
                                max_len = max(len(s) for s in sets)
                                oos_alignment_issues.append(
                                    f"asset={kv.get('asset')}|sweep={kv.get('parent_sweep_id')}|split_sig={kv.get('split_signature')}|split={kv.get('split')}|h={kv.get('horizon')}|configs={len(per_cfg)}|min_ts={min_len}|max_ts={max_len}"
                                )

        self._record(
            checks,
            "temporal_consistency",
            len(temporal_issues) == 0,
            "ok" if not temporal_issues else ", ".join(temporal_issues),
        )
        self._record(
            checks,
            "oos_unique_key",
            len(oos_unique_issues) == 0,
            "ok" if not oos_unique_issues else ", ".join(oos_unique_issues),
        )
        self._record(
            checks,
            "oos_numeric_types",
            len(oos_type_issues) == 0,
            "ok" if not oos_type_issues else ", ".join(oos_type_issues),
        )
        self._record(
            checks,
            "oos_horizon_coverage",
            len(oos_coverage_issues) == 0,
            "ok" if not oos_coverage_issues else ", ".join(oos_coverage_issues),
        )
        self._record(
            checks,
            "oos_supervised_nulls",
            len(oos_supervised_null_issues) == 0,
            "ok" if not oos_supervised_null_issues else ", ".join(oos_supervised_null_issues),
        )
        self._record(
            checks,
            "oos_interval_width_non_negative",
            len(oos_interval_issues) == 0,
            "ok" if not oos_interval_issues else ", ".join(oos_interval_issues),
        )
        self._record(
            checks,
            "oos_quantile_order",
            len(oos_quantile_order_issues) == 0,
            "ok" if not oos_quantile_order_issues else ", ".join(oos_quantile_order_issues),
        )

        block_a_df = QuantileContractAnalyzer.filter_scope(
            fact_oos_predictions=fact_oos_predictions,
            dim_run=dim_run,
            parent_sweep_prefixes=list(scope_spec.parent_sweep_prefixes) or None,
            splits=list(scope_spec.splits) or None,
            horizons=list(scope_spec.horizons) or None,
        )
        block_a_metrics = QuantileContractAnalyzer.analyze(block_a_df)
        block_a_eval = QuantileContractAnalyzer.evaluate_block_a(
            block_a_metrics,
            thresholds=self.block_a_thresholds,
        )
        self._record(
            checks,
            "oos_quantile_block_a_acceptance",
            block_a_eval.passed,
            block_a_eval.detail,
        )
        degeneracy_metrics = QuantileContractAnalyzer.analyze_degeneracy(
            fact_oos_predictions,
            fact_config,
        )
        degeneracy_eval = QuantileContractAnalyzer.evaluate_degeneracy(
            degeneracy_metrics,
            thresholds=self.degeneracy_thresholds,
        )
        self._record(
            checks,
            "block_quantile_degeneracy_gate",
            degeneracy_eval.passed,
            degeneracy_eval.detail,
        )
        self._record(
            checks,
            "oos_pairwise_target_alignment",
            len(oos_alignment_issues) == 0,
            "ok" if not oos_alignment_issues else ", ".join(oos_alignment_issues),
        )

        # P0: inference predictions key/type/quantile checks
        inf_unique_issues: list[str] = []
        inf_type_issues: list[str] = []
        inf_quantile_order_issues: list[str] = []
        if not fact_inference_predictions.empty:
            key_cols = ["inference_run_id", "horizon", "timestamp_utc", "target_timestamp_utc"]
            missing_key_cols = [c for c in key_cols if c not in fact_inference_predictions.columns]
            if missing_key_cols:
                inf_unique_issues.append(f"missing_key_columns={missing_key_cols}")
            else:
                dup = int(fact_inference_predictions.duplicated(subset=key_cols).sum())
                if dup > 0:
                    inf_unique_issues.append(f"duplicate_inference_prediction_rows={dup}")

            numeric_cols = ["y_pred", "quantile_p10", "quantile_p50", "quantile_p90"]
            for col in numeric_cols:
                if col not in fact_inference_predictions.columns:
                    continue
                src = fact_inference_predictions[col]
                coerced = pd.to_numeric(src, errors="coerce")
                bad_non_numeric = int((src.notna() & coerced.isna()).sum())
                if bad_non_numeric > 0:
                    inf_type_issues.append(f"non_numeric_{col}={bad_non_numeric}")

            iq10_col = "quantile_p10_post_guardrail" if "quantile_p10_post_guardrail" in fact_inference_predictions.columns else "quantile_p10"
            iq50_col = "quantile_p50_post_guardrail" if "quantile_p50_post_guardrail" in fact_inference_predictions.columns else "quantile_p50"
            iq90_col = "quantile_p90_post_guardrail" if "quantile_p90_post_guardrail" in fact_inference_predictions.columns else "quantile_p90"
            if {iq10_col, iq50_col, iq90_col}.issubset(set(fact_inference_predictions.columns)):
                q10 = pd.to_numeric(fact_inference_predictions[iq10_col], errors="coerce")
                q50 = pd.to_numeric(fact_inference_predictions[iq50_col], errors="coerce")
                q90 = pd.to_numeric(fact_inference_predictions[iq90_col], errors="coerce")
                bad_order = int(((~q10.isna()) & (~q50.isna()) & (~q90.isna()) & ((q10 > q50) | (q50 > q90))).sum())
                if bad_order > 0:
                    inf_quantile_order_issues.append(f"invalid_quantile_order={bad_order}|source={iq10_col},{iq50_col},{iq90_col}")

        self._record(
            checks,
            "inference_predictions_unique_key",
            len(inf_unique_issues) == 0,
            "ok" if not inf_unique_issues else ", ".join(inf_unique_issues),
        )
        self._record(
            checks,
            "inference_predictions_numeric_types",
            len(inf_type_issues) == 0,
            "ok" if not inf_type_issues else ", ".join(inf_type_issues),
        )
        self._record(
            checks,
            "inference_predictions_quantile_order",
            len(inf_quantile_order_issues) == 0,
            "ok" if not inf_quantile_order_issues else ", ".join(inf_quantile_order_issues),
        )

        # P0: expected cardinality config x fold x seed
        cardinality_ok = True
        cardinality_detail = "ok"
        if not dim_run.empty:
            key_cols = [c for c in ["asset", "feature_set_name", "config_signature", "fold", "seed"] if c in dim_run.columns]
            if len(key_cols) >= 3:
                dup = int(dim_run.duplicated(subset=key_cols).sum())
                cardinality_ok = dup == 0
                cardinality_detail = f"duplicate_groups={dup}"
        self._record(checks, "cardinality_config_fold_seed", cardinality_ok, cardinality_detail)

        # P0: minimum split samples
        split_min_ok = True
        split_min_detail = "ok"
        if not fact_run_snapshot.empty:
            checks_df = fact_run_snapshot.copy()
            missing_cols = [
                c
                for c in ["n_samples_train", "n_samples_val", "n_samples_test"]
                if c not in checks_df.columns
            ]
            if missing_cols:
                split_min_ok = False
                split_min_detail = f"missing_columns={missing_cols}"
            else:
                bad_train = int((pd.to_numeric(checks_df["n_samples_train"], errors="coerce") < self.min_samples_train).sum())
                bad_val = int((pd.to_numeric(checks_df["n_samples_val"], errors="coerce") < self.min_samples_val).sum())
                bad_test = int((pd.to_numeric(checks_df["n_samples_test"], errors="coerce") < self.min_samples_test).sum())
                split_min_ok = (bad_train + bad_val + bad_test) == 0
                split_min_detail = f"below_min_train={bad_train}, below_min_val={bad_val}, below_min_test={bad_test}"
        self._record(checks, "min_samples_by_split", split_min_ok, split_min_detail)


        # P0: DM/MCS executability using persisted data only.
        executable_ok = True
        executable_detail = "skipped(no_gold_dir)"
        if self.analytics_gold_dir is not None:
            dm_gold = self._load_partitioned_table(self.analytics_gold_dir, "gold_dm_pairwise_results")
            mcs_gold = self._load_partitioned_table(self.analytics_gold_dir, "gold_mcs_results")
            report_gold = self._load_partitioned_table(self.analytics_gold_dir, "gold_quality_statistics_report")
            dm_gold = self._scope_table(dm_gold, scope_spec=scope_spec, run_ids=None)
            mcs_gold = self._scope_table(mcs_gold, scope_spec=scope_spec, run_ids=None)
            report_gold = self._scope_table(report_gold, scope_spec=scope_spec, run_ids=None)

            feasible_keys_dm: set[tuple[object, object, object, object]] = set()
            feasible_keys_mcs: set[tuple[object, object, object, object]] = set()

            if not fact_oos_predictions.empty and not dim_run.empty:
                keep = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "status"] if c in dim_run.columns]
                if "run_id" in keep:
                    tmp = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
                    if "config_signature" not in tmp.columns:
                        if "config_signature_x" in tmp.columns:
                            tmp["config_signature"] = tmp["config_signature_x"]
                        elif "config_signature_y" in tmp.columns:
                            tmp["config_signature"] = tmp["config_signature_y"]
                    if "feature_set_name" not in tmp.columns:
                        if "feature_set_name_x" in tmp.columns:
                            tmp["feature_set_name"] = tmp["feature_set_name_x"]
                        elif "feature_set_name_y" in tmp.columns:
                            tmp["feature_set_name"] = tmp["feature_set_name_y"]
                    if "asset" not in tmp.columns:
                        if "asset_x" in tmp.columns:
                            tmp["asset"] = tmp["asset_x"]
                        elif "asset_y" in tmp.columns:
                            tmp["asset"] = tmp["asset_y"]
                    if "status" in tmp.columns:
                        tmp = tmp[tmp["status"].astype(str).str.lower() == "ok"].copy()
                    tmp = tmp[tmp["split"].astype(str) == "test"].copy() if "split" in tmp.columns else pd.DataFrame()
                    if not tmp.empty and {"horizon", "target_timestamp_utc", "y_true", "y_pred", "config_signature", "feature_set_name", "asset"}.issubset(set(tmp.columns)):
                        for c in ["horizon", "y_true", "y_pred"]:
                            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                        tmp["target_timestamp_utc"] = pd.to_datetime(tmp["target_timestamp_utc"], utc=True, errors="coerce")
                        tmp = tmp.dropna(subset=["horizon", "target_timestamp_utc", "y_true", "y_pred", "config_signature", "feature_set_name", "asset"]).copy()
                        if not tmp.empty:
                            tmp["horizon"] = tmp["horizon"].astype(int)
                            tmp["config_label"] = tmp["feature_set_name"].astype(str) + "|" + tmp["config_signature"].astype(str)
                            tmp["squared_error"] = (tmp["y_pred"] - tmp["y_true"]) ** 2
                            group_cols = [c for c in ["asset", "parent_sweep_id", "split", "horizon"] if c in tmp.columns]
                            if group_cols:
                                for keys, g in tmp.groupby(group_cols, dropna=False):
                                    by_ts = (
                                        g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                                        .mean()
                                        .reset_index()
                                    )
                                    loss_matrix = by_ts.pivot(index="target_timestamp_utc", columns="config_label", values="squared_error")
                                    loss_matrix = loss_matrix.dropna(axis=0, how="any")
                                    kv = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
                                    key4 = (
                                        kv.get("asset"),
                                        kv.get("parent_sweep_id"),
                                        kv.get("split"),
                                        kv.get("horizon"),
                                    )
                                    if loss_matrix.shape[1] >= 2 and loss_matrix.shape[0] >= 1:
                                        feasible_keys_mcs.add(key4)
                                    if loss_matrix.shape[1] >= 2 and loss_matrix.shape[0] >= 5:
                                        feasible_keys_dm.add(key4)

            dm_keys_gold: set[tuple[object, object, object, object]] = set()
            if not dm_gold.empty and {"asset", "parent_sweep_id", "split", "horizon"}.issubset(set(dm_gold.columns)):
                for keys, _ in dm_gold.groupby(["asset", "parent_sweep_id", "split", "horizon"], dropna=False):
                    dm_keys_gold.add((keys[0], keys[1], keys[2], keys[3]))

            mcs_keys_gold: set[tuple[object, object, object, object]] = set()
            if not mcs_gold.empty and {"asset", "parent_sweep_id", "split", "horizon"}.issubset(set(mcs_gold.columns)):
                for keys, _ in mcs_gold.groupby(["asset", "parent_sweep_id", "split", "horizon"], dropna=False):
                    mcs_keys_gold.add((keys[0], keys[1], keys[2], keys[3]))

            missing_dm = sorted(feasible_keys_dm - dm_keys_gold)
            missing_mcs = sorted(feasible_keys_mcs - mcs_keys_gold)

            report_has_stats_ready = (
                not report_gold.empty
                and "statistics_ready" in report_gold.columns
                and bool(report_gold["statistics_ready"].fillna(False).astype(bool).any())
            )
            report_expected = len(feasible_keys_dm) > 0 or len(feasible_keys_mcs) > 0

            executable_ok = len(missing_dm) == 0 and len(missing_mcs) == 0 and ((not report_expected) or report_has_stats_ready)
            executable_detail = (
                f"feasible_dm={len(feasible_keys_dm)}, feasible_mcs={len(feasible_keys_mcs)}, "
                f"missing_dm={len(missing_dm)}, missing_mcs={len(missing_mcs)}, report_stats_ready={report_has_stats_ready}"
            )

        self._record(checks, "dm_mcs_persisted_executable", executable_ok, executable_detail)

        # P0: calibrated confidence must exist by horizon in gold metrics.
        confidence_ok = True
        confidence_detail = "skipped(no_gold_dir)"
        if self.analytics_gold_dir is not None:
            gold_conf = self._load_partitioned_table(
                self.analytics_gold_dir,
                "gold_prediction_metrics_by_run_split_horizon",
            )
            gold_conf = self._scope_table(gold_conf, scope_spec=scope_spec, run_ids=run_id_scope_filter)
            if gold_conf.empty:
                confidence_ok = False
                confidence_detail = "missing_gold_prediction_metrics_by_run_split_horizon"
            else:
                needed = {"run_id", "split", "horizon", "confidence_calibrated"}
                missing_cols = sorted(needed - set(gold_conf.columns))
                if missing_cols:
                    confidence_ok = False
                    confidence_detail = f"missing_columns={missing_cols}"
                else:
                    conf = gold_conf[gold_conf["split"].astype(str).isin(["val", "test"])].copy()
                    conf["horizon"] = pd.to_numeric(conf["horizon"], errors="coerce")
                    conf["confidence_calibrated"] = pd.to_numeric(conf["confidence_calibrated"], errors="coerce")
                    # Stage F.0.6: `confidence_calibrated` is a probabilistic
                    # metric (depends on PICP). Point baselines (zero_return,
                    # historical_mean_rolling) and any run where Stage 9 marks
                    # `is_quantile_genuine=False` legitimately produce NaN
                    # confidence — filtering them out prevents false-positive
                    # `bad_confidence` rows that triggered the F.1 smoke
                    # 2026-05-18 failure (8 rows, 4 per point baseline).
                    if "is_quantile_genuine" in conf.columns:
                        is_genuine = conf["is_quantile_genuine"]
                        # is_quantile_genuine arrives as bool, object("True"/"False"),
                        # or NaN depending on writer. Coerce to bool with strict
                        # interpretation: only True counts as quantile-genuine.
                        is_genuine_bool = is_genuine.astype(str).str.strip().str.lower().eq("true")
                        conf = conf[is_genuine_bool].copy()
                    bad_conf = int(conf["confidence_calibrated"].isna().sum())
                    non_finite = int(conf["confidence_calibrated"].isin([float("inf"), float("-inf")]).sum())
                    horizon_misses: list[str] = []
                    if expected_by_run and {"run_id", "horizon"}.issubset(set(conf.columns)):
                        for run_id, grp in conf.groupby("run_id", dropna=False):
                            run_key = str(run_id)
                            expected_horizons = expected_by_run.get(run_key)
                            if not expected_horizons:
                                continue
                            actual = sorted({int(h) for h in grp["horizon"].dropna().tolist()})
                            missing_h = sorted(set(expected_horizons) - set(actual))
                            if missing_h:
                                horizon_misses.append(f"run_id={run_key}:missing={missing_h}")
                    confidence_ok = bad_conf == 0 and non_finite == 0 and len(horizon_misses) == 0
                    confidence_detail = (
                        f"bad_confidence={bad_conf}, non_finite={non_finite}, missing_expected_horizons={len(horizon_misses)}"
                    )

        self._record(checks, "gold_confidence_calibrated_by_horizon", confidence_ok, confidence_detail)

        # P0: gold_prediction_metrics_by_config must expose n_oos and be consistent with run-level n_samples.
        n_oos_ok = True
        n_oos_detail = "skipped(no_gold_dir)"
        if self.analytics_gold_dir is not None:
            gold_by_cfg = self._load_partitioned_table(
                self.analytics_gold_dir,
                "gold_prediction_metrics_by_config",
            )
            gold_run_h = self._load_partitioned_table(
                self.analytics_gold_dir,
                "gold_prediction_metrics_by_run_split_horizon",
            )
            gold_by_cfg = self._scope_table(gold_by_cfg, scope_spec=scope_spec, run_ids=None)
            gold_run_h = self._scope_table(gold_run_h, scope_spec=scope_spec, run_ids=run_id_scope_filter)
            if gold_by_cfg.empty:
                n_oos_ok = False
                n_oos_detail = "missing_gold_prediction_metrics_by_config"
            elif "n_oos" not in gold_by_cfg.columns:
                n_oos_ok = False
                n_oos_detail = "missing_n_oos_column"
            else:
                by = gold_by_cfg.copy()
                by["n_oos"] = pd.to_numeric(by["n_oos"], errors="coerce")
                non_positive = int((by["n_oos"].fillna(0) <= 0).sum())
                if gold_run_h.empty:
                    n_oos_ok = non_positive == 0
                    n_oos_detail = f"non_positive_n_oos={non_positive}, consistency=skipped(no_run_level_gold)"
                else:
                    needed = {"asset", "feature_set_name", "parent_sweep_id", "config_signature", "split", "horizon", "n_samples"}
                    needed_by_cfg = needed - {"n_samples"}
                    missing_by_cfg_cols = sorted(needed_by_cfg - set(by.columns))
                    if missing_by_cfg_cols:
                        n_oos_ok = False
                        n_oos_detail = f"missing_by_config_columns={missing_by_cfg_cols}"
                    else:
                        missing_run_level_cols = sorted(needed - set(gold_run_h.columns))
                        if missing_run_level_cols:
                            n_oos_ok = False
                            n_oos_detail = f"missing_run_level_columns={missing_run_level_cols}"
                        else:
                            run = gold_run_h.copy()
                            run["n_samples"] = pd.to_numeric(run["n_samples"], errors="coerce").fillna(0)
                            key_cols = ["asset", "feature_set_name", "parent_sweep_id", "config_signature", "split", "horizon"]
                            expected = (
                                run.groupby(key_cols, dropna=False)["n_samples"]
                                .sum()
                                .reset_index()
                                .rename(columns={"n_samples": "expected_n_oos"})
                            )
                            cmp = by.merge(expected, on=key_cols, how="left")
                            cmp["expected_n_oos"] = pd.to_numeric(cmp["expected_n_oos"], errors="coerce")
                            cmp["expected_n_oos"] = cmp["expected_n_oos"].fillna(0)
                            mismatch = int((cmp["n_oos"].fillna(0).astype(int) != cmp["expected_n_oos"].astype(int)).sum())
                            n_oos_ok = non_positive == 0 and mismatch == 0
                            n_oos_detail = f"non_positive_n_oos={non_positive}, mismatch_with_run_level={mismatch}"

        self._record(checks, "gold_metrics_by_config_n_oos_contract", n_oos_ok, n_oos_detail)

        # Stage 12 gate: candidate sweeps must include at least one baseline run
        # sharing the same parent_sweep_id. Active only in cohort_decision scope
        # to avoid false positives on legacy/global_health silvers (pre-Stage 12).
        baseline_parity_ok = True
        baseline_parity_detail = "skipped(not_cohort_decision)"
        if scope_spec.scope_mode == "cohort_decision" and not dim_run.empty:
            needed_cols = {"parent_sweep_id", "feature_set_name", "model_version"}
            missing_cols = sorted(needed_cols - set(dim_run.columns))
            if missing_cols:
                baseline_parity_ok = False
                baseline_parity_detail = f"missing_columns={missing_cols}"
            else:
                dim_view = dim_run.copy()
                feature_set_lower = dim_view["feature_set_name"].astype(str).str.strip().str.lower()
                model_version_lower = dim_view["model_version"].astype(str).str.strip().str.lower()
                dim_view["_is_baseline"] = feature_set_lower.eq("baseline") | model_version_lower.str.startswith(
                    "baseline_"
                )
                issues: list[str] = []
                for sweep_id, grp in dim_view.groupby("parent_sweep_id", dropna=False):
                    sweep_value = str(sweep_id) if pd.notna(sweep_id) else "<null>"
                    if not sweep_value or sweep_value.lower() in {"none", "nan", "null", "<na>"}:
                        continue
                    n_candidates = int((~grp["_is_baseline"]).sum())
                    n_baselines = int(grp["_is_baseline"].sum())
                    if n_candidates > 0 and n_baselines == 0:
                        issues.append(
                            f"sweep={sweep_value},candidates={n_candidates},baselines=0"
                        )
                if issues:
                    baseline_parity_ok = False
                    baseline_parity_detail = ", ".join(issues)
                else:
                    baseline_parity_detail = "ok"
        self._record(
            checks,
            "baselines_share_parent_sweep_id_with_candidates",
            baseline_parity_ok,
            baseline_parity_detail,
        )

        # P0: official runs quantile/attention contract
        # Baselines (feature_set_name='baseline' OR model_version startswith
        # 'baseline_') by design do not own torch model artifacts (no
        # checkpoint, no feature_importance, no attention). They participate
        # in quantile contract (p10/p50/p90 must be present in oos predictions)
        # but are excluded from the artifact contract — Stage F.0.7 (registered
        # in docs/07_reports/smoke_confirmatory_2026-05-18.md and
        # docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md Stage F.0).
        contract_ok = True
        contract_issues: list[str] = []
        if not dim_run.empty:
            official_runs = set(
                dim_run.loc[dim_run["status"].astype(str).str.lower() == "ok", "run_id"].astype(str).tolist()
            ) if "status" in dim_run.columns else set()
            if official_runs:
                if {"feature_set_name", "model_version"}.issubset(dim_run.columns):
                    baseline_dim = dim_run[
                        dim_run["feature_set_name"].astype(str).str.strip().str.lower().eq("baseline")
                        | dim_run["model_version"].astype(str).str.strip().str.lower().str.startswith("baseline_")
                    ]
                    baseline_run_ids = set(baseline_dim["run_id"].astype(str).tolist())
                else:
                    baseline_run_ids = set()
                official_candidates = official_runs - baseline_run_ids
                if fact_oos_predictions.empty:
                    contract_ok = False
                    contract_issues.append("missing_fact_oos_predictions")
                else:
                    oos_off = fact_oos_predictions[fact_oos_predictions["run_id"].astype(str).isin(official_runs)]
                    for q in ["quantile_p10", "quantile_p50", "quantile_p90"]:
                        if q not in oos_off.columns:
                            contract_ok = False
                            contract_issues.append(f"missing_{q}")
                        else:
                            nulls = int(pd.to_numeric(oos_off[q], errors="coerce").isna().sum())
                            if nulls > 0:
                                contract_ok = False
                                contract_issues.append(f"{q}_nan={nulls}")

                if not official_candidates:
                    # Sweep with only baselines — artifact contract is vacuously
                    # satisfied; surface as informational note in detail.
                    contract_issues.append("artifact_check_skipped_baselines_only")
                elif fact_model_artifacts.empty:
                    contract_ok = False
                    contract_issues.append("missing_fact_model_artifacts")
                else:
                    mar = fact_model_artifacts[
                        fact_model_artifacts["run_id"].astype(str).isin(official_candidates)
                    ]
                    missing_mar = len(official_candidates - set(mar["run_id"].astype(str).tolist()))
                    if missing_mar > 0:
                        contract_ok = False
                        contract_issues.append(f"missing_model_artifacts={missing_mar}")
                    for c in ["feature_importance_json", "attention_summary_json"]:
                        if c not in mar.columns:
                            contract_ok = False
                            contract_issues.append(f"missing_{c}")
                        else:
                            empty = int(mar[c].fillna("").astype(str).str.strip().eq("").sum())
                            if empty > 0:
                                contract_ok = False
                                contract_issues.append(f"empty_{c}={empty}")

        self._record(
            checks,
            "official_contract_quantile_attention",
            contract_ok,
            "ok" if not contract_issues else ", ".join(contract_issues),
        )

        for item in checks:
            current = str(item.get("detail", ""))
            item["detail"] = f"{current} | {scope_detail}"

        passed = all(bool(item["passed"]) for item in checks)
        return AnalyticsQualityResult(passed=passed, checks=checks)
