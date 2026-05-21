"""Cardinality / presence quality checks.

Migrated from `src/use_cases/validate_analytics_quality_use_case.py` per
ADR-0004 and Stage R-21 of the remediation plan. Each class preserves the
exact (passed, detail) output the monolith emitted so the orchestrator
remains bit-identical across both `scope_mode` values.
"""
from __future__ import annotations

import pandas as pd

from src.domain.services.quality_checks.base import (
    AnalyticsSnapshot,
    CheckResult,
    QualityCheck,
)

_REQUIRED_TABLES: tuple[str, ...] = (
    "dim_run",
    "fact_run_snapshot",
    "fact_config",
    "fact_split_metrics",
    "fact_epoch_metrics",
    "fact_oos_predictions",
    "fact_model_artifacts",
    "bridge_run_features",
)


class RequiredTablesPresenceCheck(QualityCheck):
    """Every silver table required by the analytics contract must be non-empty."""

    name = "required_tables_presence"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        missing = [name for name in _REQUIRED_TABLES if not snapshot.has(name)]
        return CheckResult(
            name=self.name,
            passed=len(missing) == 0,
            detail="ok" if not missing else ", ".join(missing),
        )


class InferencePredictionsContinuityCheck(QualityCheck):
    """If `fact_inference_runs` has rows, every `inference_run_id` must
    also appear in `fact_inference_predictions`.
    """

    name = "inference_predictions_continuity"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        inf_runs = snapshot.get("fact_inference_runs")
        inf_preds = snapshot.get("fact_inference_predictions")
        if inf_runs.empty:
            return CheckResult(self.name, True, "ok")
        if inf_preds.empty:
            return CheckResult(
                self.name,
                False,
                "fact_inference_predictions empty while fact_inference_runs has rows",
            )
        if "inference_run_id" not in inf_preds.columns:
            return CheckResult(
                self.name,
                False,
                "fact_inference_predictions missing inference_run_id",
            )
        run_ids = set(
            inf_runs.get("inference_run_id", pd.Series(dtype=str))
            .dropna().astype(str).tolist()
        )
        pred_ids = set(inf_preds["inference_run_id"].dropna().astype(str).tolist())
        missing_inf = sorted(run_ids - pred_ids)
        if missing_inf:
            return CheckResult(
                self.name,
                False,
                f"inference_runs_without_predictions={len(missing_inf)}",
            )
        return CheckResult(self.name, True, "ok")


class FeatureContribLocalContinuityCheck(QualityCheck):
    """If `fact_inference_predictions` has rows, every `inference_run_id`
    must also appear in `fact_feature_contrib_local`.
    """

    name = "feature_contrib_local_continuity"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        inf_preds = snapshot.get("fact_inference_predictions")
        contribs = snapshot.get("fact_feature_contrib_local")
        if inf_preds.empty:
            return CheckResult(self.name, True, "ok")
        if contribs.empty:
            return CheckResult(
                self.name,
                False,
                "fact_feature_contrib_local empty while fact_inference_predictions has rows",
            )
        if "inference_run_id" not in contribs.columns:
            return CheckResult(
                self.name,
                False,
                "fact_feature_contrib_local missing inference_run_id",
            )
        pred_ids = set(
            inf_preds.get("inference_run_id", pd.Series(dtype=str))
            .dropna().astype(str).tolist()
        )
        local_ids = set(contribs["inference_run_id"].dropna().astype(str).tolist())
        missing_local = sorted(pred_ids - local_ids)
        if missing_local:
            return CheckResult(
                self.name,
                False,
                f"inference_runs_without_local_contrib={len(missing_local)}",
            )
        return CheckResult(self.name, True, "ok")


class RunIdExecutionConsistencyCheck(QualityCheck):
    """dim_run must have no null run_id, no duplicate run_id, and no
    duplicate non-null execution_id.
    """

    name = "run_id_execution_consistency"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        dim_run = snapshot.get("dim_run")
        if dim_run.empty:
            return CheckResult(self.name, False, "dim_run is empty")
        run_id_null = int(dim_run["run_id"].isna().sum()) if "run_id" in dim_run else len(dim_run)
        dup_run_id = (
            int(dim_run["run_id"].duplicated().sum()) if "run_id" in dim_run else len(dim_run)
        )
        non_null_exec = (
            dim_run["execution_id"].dropna().astype(str).str.strip()
            if "execution_id" in dim_run.columns
            else pd.Series(dtype=str)
        )
        dup_exec_id = int(non_null_exec.duplicated().sum()) if not non_null_exec.empty else 0
        ok = run_id_null == 0 and dup_run_id == 0 and dup_exec_id == 0
        return CheckResult(
            self.name,
            ok,
            f"run_id_null={run_id_null}, duplicate_run_id={dup_run_id}, duplicate_execution_id={dup_exec_id}",
        )


class RequiredMetricsNanCheck(QualityCheck):
    """No NaN allowed in `fact_split_metrics.{rmse,mae,directional_accuracy,n_samples}`
    or `fact_epoch_metrics.{train_loss,val_loss}`.
    """

    name = "required_metrics_nan"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_split_metrics = snapshot.get("fact_split_metrics")
        fact_epoch_metrics = snapshot.get("fact_epoch_metrics")
        nan_issues: list[str] = []
        for df, cols, label in [
            (
                fact_split_metrics,
                ["rmse", "mae", "directional_accuracy", "n_samples"],
                "fact_split_metrics",
            ),
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
        return CheckResult(
            self.name,
            len(nan_issues) == 0,
            "ok" if not nan_issues else ", ".join(nan_issues),
        )


class CardinalityConfigFoldSeedCheck(QualityCheck):
    """dim_run rows must be unique on (asset, feature_set_name,
    config_signature, fold, seed) when at least 3 of those columns exist.
    """

    name = "cardinality_config_fold_seed"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        dim_run = snapshot.get("dim_run")
        if dim_run.empty:
            return CheckResult(self.name, True, "ok")
        key_cols = [
            c
            for c in ["asset", "feature_set_name", "config_signature", "fold", "seed"]
            if c in dim_run.columns
        ]
        if len(key_cols) < 3:
            return CheckResult(self.name, True, "ok")
        dup = int(dim_run.duplicated(subset=key_cols).sum())
        return CheckResult(self.name, dup == 0, f"duplicate_groups={dup}")


class MinSamplesBySplitCheck(QualityCheck):
    """Each row in `fact_run_snapshot` must have `n_samples_train >= min_train`,
    `n_samples_val >= min_val`, and `n_samples_test >= min_test`.
    """

    name = "min_samples_by_split"

    def __init__(
        self,
        *,
        min_train: int = 1,
        min_val: int = 1,
        min_test: int = 1,
    ) -> None:
        self.min_train = int(min_train)
        self.min_val = int(min_val)
        self.min_test = int(min_test)

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_run_snapshot = snapshot.get("fact_run_snapshot")
        if fact_run_snapshot.empty:
            return CheckResult(self.name, True, "ok")
        checks_df = fact_run_snapshot.copy()
        missing_cols = [
            c
            for c in ["n_samples_train", "n_samples_val", "n_samples_test"]
            if c not in checks_df.columns
        ]
        if missing_cols:
            return CheckResult(self.name, False, f"missing_columns={missing_cols}")
        bad_train = int(
            (pd.to_numeric(checks_df["n_samples_train"], errors="coerce") < self.min_train).sum()
        )
        bad_val = int(
            (pd.to_numeric(checks_df["n_samples_val"], errors="coerce") < self.min_val).sum()
        )
        bad_test = int(
            (pd.to_numeric(checks_df["n_samples_test"], errors="coerce") < self.min_test).sum()
        )
        return CheckResult(
            self.name,
            (bad_train + bad_val + bad_test) == 0,
            f"below_min_train={bad_train}, below_min_val={bad_val}, below_min_test={bad_test}",
        )
