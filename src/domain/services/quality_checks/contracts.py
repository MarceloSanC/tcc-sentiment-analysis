"""Schema / contract / referential-integrity quality checks.

Migrated from `src/use_cases/validate_analytics_quality_use_case.py` per
ADR-0004 and Stage R-21. Each check preserves bit-identical (passed,
detail) output relative to the monolithic implementation.
"""
from __future__ import annotations

import pandas as pd

from src.domain.services.quality_checks.base import (
    AnalyticsSnapshot,
    CheckResult,
    QualityCheck,
)

_FACT_TABLES_FOR_REFERENTIAL = (
    "fact_run_snapshot",
    "fact_config",
    "fact_split_metrics",
    "fact_epoch_metrics",
    "fact_oos_predictions",
    "fact_model_artifacts",
    "fact_inference_runs",
    "fact_inference_predictions",
    "fact_feature_contrib_local",
    "fact_failures",
    "bridge_run_features",
    "fact_split_timestamps_ref",
)


class ReferentialIntegrityCheck(QualityCheck):
    """Every `run_id` appearing in a fact table must exist in `dim_run`."""

    name = "referential_integrity"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        dim_run = snapshot.get("dim_run")
        dim_ids = (
            set(dim_run["run_id"].dropna().astype(str).tolist())
            if "run_id" in dim_run.columns
            else set()
        )
        missing_refs: list[str] = []
        for table_name in _FACT_TABLES_FOR_REFERENTIAL:
            df = snapshot.get(table_name)
            if df.empty or "run_id" not in df.columns:
                continue
            ids = set(df["run_id"].dropna().astype(str).tolist())
            missing = ids - dim_ids
            if missing:
                missing_refs.append(f"{table_name}:{len(missing)}")
        return CheckResult(
            self.name,
            len(missing_refs) == 0,
            "ok" if not missing_refs else ", ".join(missing_refs),
        )


class TemporalConsistencyCheck(QualityCheck):
    """`fact_oos_predictions` must have parseable timestamps, target >=
    timestamp, and `target_timestamp_utc` monotonic per (run_id, split,
    horizon).
    """

    name = "temporal_consistency"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        issues: list[str] = []
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        required_ts_cols = ["timestamp_utc", "target_timestamp_utc"]
        missing_ts_cols = [c for c in required_ts_cols if c not in fact_oos.columns]
        if missing_ts_cols:
            issues.append(f"missing_timestamp_columns={missing_ts_cols}")
            return CheckResult(self.name, False, ", ".join(issues))

        ts = pd.to_datetime(fact_oos["timestamp_utc"], utc=True, errors="coerce")
        tgt = pd.to_datetime(fact_oos["target_timestamp_utc"], utc=True, errors="coerce")
        bad_parse = int(ts.isna().sum() + tgt.isna().sum())
        if bad_parse > 0:
            issues.append(f"timestamp_parse_errors={bad_parse}")
        bad_order = int((tgt < ts).sum())
        if bad_order > 0:
            issues.append(f"target_before_timestamp={bad_order}")

        group_cols = [c for c in ["run_id", "split", "horizon"] if c in fact_oos.columns]
        if group_cols:
            tmp = fact_oos.copy()
            tmp["_ts"] = ts
            tmp["_tgt"] = tgt
            for _, g in tmp.groupby(group_cols, dropna=False):
                ordered = g.sort_values(["_ts", "_tgt"], kind="mergesort")
                tgt_series = ordered["_tgt"].dropna()
                if not tgt_series.is_monotonic_increasing:
                    issues.append("non_monotonic_target_timestamp_in_group")
                    break

        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class OosUniqueKeyCheck(QualityCheck):
    """`fact_oos_predictions` must be unique on
    (run_id, split, horizon, timestamp_utc, target_timestamp_utc).
    """

    name = "oos_unique_key"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        issues: list[str] = []
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        key_cols = [
            "run_id", "split", "horizon", "timestamp_utc", "target_timestamp_utc",
        ]
        missing_key_cols = [c for c in key_cols if c not in fact_oos.columns]
        if missing_key_cols:
            issues.append(f"missing_key_columns={missing_key_cols}")
        else:
            dup_count = int(fact_oos.duplicated(subset=key_cols).sum())
            if dup_count > 0:
                issues.append(f"duplicate_oos_rows={dup_count}")
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


_OOS_NUMERIC_COLUMNS = (
    "y_true",
    "y_pred",
    "error",
    "abs_error",
    "sq_error",
    "quantile_p10",
    "quantile_p50",
    "quantile_p90",
)


class OosNumericTypesCheck(QualityCheck):
    """All numeric columns in `fact_oos_predictions` must be coercible to
    float (no rogue non-numeric strings).
    """

    name = "oos_numeric_types"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        for col in _OOS_NUMERIC_COLUMNS:
            if col not in fact_oos.columns:
                issues.append(f"missing_numeric_column={col}")
                continue
            src = fact_oos[col]
            coerced = pd.to_numeric(src, errors="coerce")
            bad_non_numeric = int((src.notna() & coerced.isna()).sum())
            if bad_non_numeric > 0:
                issues.append(f"non_numeric_{col}={bad_non_numeric}")
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class OosHorizonCoverageCheck(QualityCheck):
    """Every `run_id`/`split` group in `fact_oos_predictions` must cover
    the horizons declared in `fact_config.evaluation_horizons_json`.
    """

    name = "oos_horizon_coverage"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        expected_by_run = snapshot.expected_horizons_by_run
        if {"run_id", "split", "horizon"}.issubset(set(fact_oos.columns)):
            tmp = fact_oos.copy()
            tmp["horizon"] = pd.to_numeric(tmp["horizon"], errors="coerce")
            for (run_id, split), g in tmp.groupby(["run_id", "split"], dropna=False):
                run_key = str(run_id)
                expected = expected_by_run.get(run_key, [1])
                actual = sorted(
                    {int(h) for h in g["horizon"].dropna().tolist() if pd.notna(h)}
                )
                missing_horizons = sorted(set(expected) - set(actual))
                if missing_horizons:
                    issues.append(
                        f"run_id={run_key}|split={split}|missing_horizons={missing_horizons}"
                        f"|expected={expected}|actual={actual}"
                    )
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class OosSupervisedNullsCheck(QualityCheck):
    """Supervised splits (train/val/test) must have non-null y_true/y_pred."""

    name = "oos_supervised_nulls"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        if {"split", "y_true", "y_pred"}.issubset(set(fact_oos.columns)):
            supervised = fact_oos[
                fact_oos["split"].astype(str).isin(["train", "val", "test"])
            ]
            null_true = int(supervised["y_true"].isna().sum())
            null_pred = int(supervised["y_pred"].isna().sum())
            if null_true > 0:
                issues.append(f"null_y_true={null_true}")
            if null_pred > 0:
                issues.append(f"null_y_pred={null_pred}")
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class OosIntervalWidthNonNegativeCheck(QualityCheck):
    """`q90 - q10 >= 0` everywhere. Uses post_guardrail columns when
    available; falls back to raw quantiles.
    """

    name = "oos_interval_width_non_negative"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        q10_col = (
            "quantile_p10_post_guardrail"
            if "quantile_p10_post_guardrail" in fact_oos.columns
            else "quantile_p10"
        )
        q90_col = (
            "quantile_p90_post_guardrail"
            if "quantile_p90_post_guardrail" in fact_oos.columns
            else "quantile_p90"
        )
        if {q10_col, q90_col}.issubset(set(fact_oos.columns)):
            q10 = pd.to_numeric(fact_oos[q10_col], errors="coerce")
            q90 = pd.to_numeric(fact_oos[q90_col], errors="coerce")
            negative_width = int(
                ((~q10.isna()) & (~q90.isna()) & ((q90 - q10) < 0.0)).sum()
            )
            if negative_width > 0:
                issues.append(f"negative_interval_width={negative_width}|source={q10_col},{q90_col}")
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class OosQuantileOrderCheck(QualityCheck):
    """`q10 <= q50 <= q90` everywhere. Uses post_guardrail columns when
    available; falls back to raw quantiles.
    """

    name = "oos_quantile_order"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        q10_col = (
            "quantile_p10_post_guardrail"
            if "quantile_p10_post_guardrail" in fact_oos.columns
            else "quantile_p10"
        )
        q50_col = (
            "quantile_p50_post_guardrail"
            if "quantile_p50_post_guardrail" in fact_oos.columns
            else "quantile_p50"
        )
        q90_col = (
            "quantile_p90_post_guardrail"
            if "quantile_p90_post_guardrail" in fact_oos.columns
            else "quantile_p90"
        )
        if {q10_col, q50_col, q90_col}.issubset(set(fact_oos.columns)):
            q10 = pd.to_numeric(fact_oos[q10_col], errors="coerce")
            q50 = pd.to_numeric(fact_oos[q50_col], errors="coerce")
            q90 = pd.to_numeric(fact_oos[q90_col], errors="coerce")
            bad_order = int(
                ((~q10.isna()) & (~q50.isna()) & (~q90.isna()) & ((q10 > q50) | (q50 > q90))).sum()
            )
            if bad_order > 0:
                issues.append(
                    f"invalid_quantile_order={bad_order}|source={q10_col},{q50_col},{q90_col}"
                )
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class InferencePredictionsUniqueKeyCheck(QualityCheck):
    """`fact_inference_predictions` must be unique on
    (inference_run_id, horizon, timestamp_utc, target_timestamp_utc).
    """

    name = "inference_predictions_unique_key"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        inf_preds = snapshot.get("fact_inference_predictions")
        if inf_preds.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        key_cols = ["inference_run_id", "horizon", "timestamp_utc", "target_timestamp_utc"]
        missing_key_cols = [c for c in key_cols if c not in inf_preds.columns]
        if missing_key_cols:
            issues.append(f"missing_key_columns={missing_key_cols}")
        else:
            dup = int(inf_preds.duplicated(subset=key_cols).sum())
            if dup > 0:
                issues.append(f"duplicate_inference_prediction_rows={dup}")
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


_INF_NUMERIC_COLUMNS = (
    "y_pred",
    "quantile_p10",
    "quantile_p50",
    "quantile_p90",
)


class InferencePredictionsNumericTypesCheck(QualityCheck):
    """Numeric columns in `fact_inference_predictions` must be coercible
    (no rogue non-numeric strings).
    """

    name = "inference_predictions_numeric_types"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        inf_preds = snapshot.get("fact_inference_predictions")
        if inf_preds.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        for col in _INF_NUMERIC_COLUMNS:
            if col not in inf_preds.columns:
                continue
            src = inf_preds[col]
            coerced = pd.to_numeric(src, errors="coerce")
            bad_non_numeric = int((src.notna() & coerced.isna()).sum())
            if bad_non_numeric > 0:
                issues.append(f"non_numeric_{col}={bad_non_numeric}")
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class InferencePredictionsQuantileOrderCheck(QualityCheck):
    """`q10 <= q50 <= q90` in `fact_inference_predictions`; uses
    post_guardrail when present.
    """

    name = "inference_predictions_quantile_order"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        inf_preds = snapshot.get("fact_inference_predictions")
        if inf_preds.empty:
            return CheckResult(self.name, True, "ok")
        issues: list[str] = []
        iq10_col = (
            "quantile_p10_post_guardrail"
            if "quantile_p10_post_guardrail" in inf_preds.columns
            else "quantile_p10"
        )
        iq50_col = (
            "quantile_p50_post_guardrail"
            if "quantile_p50_post_guardrail" in inf_preds.columns
            else "quantile_p50"
        )
        iq90_col = (
            "quantile_p90_post_guardrail"
            if "quantile_p90_post_guardrail" in inf_preds.columns
            else "quantile_p90"
        )
        if {iq10_col, iq50_col, iq90_col}.issubset(set(inf_preds.columns)):
            q10 = pd.to_numeric(inf_preds[iq10_col], errors="coerce")
            q50 = pd.to_numeric(inf_preds[iq50_col], errors="coerce")
            q90 = pd.to_numeric(inf_preds[iq90_col], errors="coerce")
            bad_order = int(
                ((~q10.isna()) & (~q50.isna()) & (~q90.isna()) & ((q10 > q50) | (q50 > q90))).sum()
            )
            if bad_order > 0:
                issues.append(
                    f"invalid_quantile_order={bad_order}|source={iq10_col},{iq50_col},{iq90_col}"
                )
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class OfficialContractQuantileAttentionCheck(QualityCheck):
    """Official-status runs (`status='ok'`, non-baseline) must have all
    quantile columns present in `fact_oos_predictions` (no NaN) and have
    non-empty `feature_importance_json` + `attention_summary_json` in
    `fact_model_artifacts`. Baselines are exempt by design (Stage F.0.7).
    """

    name = "official_contract_quantile_attention"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        dim_run = snapshot.get("dim_run")
        fact_oos = snapshot.get("fact_oos_predictions")
        fact_model_artifacts = snapshot.get("fact_model_artifacts")
        issues: list[str] = []
        if dim_run.empty:
            return CheckResult(self.name, True, "ok")

        official_runs = (
            set(
                dim_run.loc[
                    dim_run["status"].astype(str).str.lower() == "ok", "run_id"
                ]
                .astype(str).tolist()
            )
            if "status" in dim_run.columns
            else set()
        )
        if not official_runs:
            return CheckResult(self.name, True, "ok")

        if {"feature_set_name", "model_version"}.issubset(dim_run.columns):
            baseline_dim = dim_run[
                dim_run["feature_set_name"].astype(str).str.strip().str.lower().eq("baseline")
                | dim_run["model_version"].astype(str).str.strip().str.lower().str.startswith(
                    "baseline_"
                )
            ]
            baseline_run_ids = set(baseline_dim["run_id"].astype(str).tolist())
        else:
            baseline_run_ids = set()
        official_candidates = official_runs - baseline_run_ids

        contract_ok = True
        if fact_oos.empty:
            contract_ok = False
            issues.append("missing_fact_oos_predictions")
        else:
            oos_off = fact_oos[fact_oos["run_id"].astype(str).isin(official_runs)]
            for q in ["quantile_p10", "quantile_p50", "quantile_p90"]:
                if q not in oos_off.columns:
                    contract_ok = False
                    issues.append(f"missing_{q}")
                else:
                    nulls = int(pd.to_numeric(oos_off[q], errors="coerce").isna().sum())
                    if nulls > 0:
                        contract_ok = False
                        issues.append(f"{q}_nan={nulls}")

        if not official_candidates:
            issues.append("artifact_check_skipped_baselines_only")
        elif fact_model_artifacts.empty:
            contract_ok = False
            issues.append("missing_fact_model_artifacts")
        else:
            mar = fact_model_artifacts[
                fact_model_artifacts["run_id"].astype(str).isin(official_candidates)
            ]
            missing_mar = len(official_candidates - set(mar["run_id"].astype(str).tolist()))
            if missing_mar > 0:
                contract_ok = False
                issues.append(f"missing_model_artifacts={missing_mar}")
            for c in ["feature_importance_json", "attention_summary_json"]:
                if c not in mar.columns:
                    contract_ok = False
                    issues.append(f"missing_{c}")
                else:
                    empty = int(mar[c].fillna("").astype(str).str.strip().eq("").sum())
                    if empty > 0:
                        contract_ok = False
                        issues.append(f"empty_{c}={empty}")

        return CheckResult(
            self.name,
            contract_ok,
            "ok" if not issues else ", ".join(issues),
        )
