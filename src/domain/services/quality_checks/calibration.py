"""Calibration / quantile-contract / DM-MCS readiness quality checks.

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
from src.domain.services.quantile_contract_analyzer import (
    QuantileBlockAThresholds,
    QuantileContractAnalyzer,
    QuantileDegeneracyThresholds,
)


class OosQuantileBlockAAcceptanceCheck(QualityCheck):
    """`QuantileContractAnalyzer.evaluate_block_a` over the scope-filtered
    `fact_oos_predictions`, against configured thresholds.
    """

    name = "oos_quantile_block_a_acceptance"

    def __init__(self, *, thresholds: QuantileBlockAThresholds) -> None:
        self.thresholds = thresholds

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        dim_run = snapshot.get("dim_run")
        scope = snapshot.scope_spec
        # Match the monolith call signature: empty list collapses to None
        # so QuantileContractAnalyzer.filter_scope short-circuits the filter
        # rather than filtering against an empty allow-list.
        block_a_df = QuantileContractAnalyzer.filter_scope(
            fact_oos_predictions=fact_oos,
            dim_run=dim_run,
            parent_sweep_prefixes=(list(scope.parent_sweep_prefixes) or None) if scope else None,
            splits=(list(scope.splits) or None) if scope else None,
            horizons=(list(scope.horizons) or None) if scope else None,
        )
        metrics = QuantileContractAnalyzer.analyze(block_a_df)
        evaluation = QuantileContractAnalyzer.evaluate_block_a(
            metrics, thresholds=self.thresholds
        )
        return CheckResult(self.name, bool(evaluation.passed), evaluation.detail)


class BlockQuantileDegeneracyGateCheck(QualityCheck):
    """`QuantileContractAnalyzer.evaluate_degeneracy` over the
    scope-filtered `fact_oos_predictions` against configured thresholds.
    """

    name = "block_quantile_degeneracy_gate"

    def __init__(self, *, thresholds: QuantileDegeneracyThresholds) -> None:
        self.thresholds = thresholds

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        fact_config = snapshot.get("fact_config")
        metrics = QuantileContractAnalyzer.analyze_degeneracy(fact_oos, fact_config)
        evaluation = QuantileContractAnalyzer.evaluate_degeneracy(
            metrics, thresholds=self.thresholds
        )
        return CheckResult(self.name, bool(evaluation.passed), evaluation.detail)


class DmMcsPersistedExecutableCheck(QualityCheck):
    """DM/MCS feasibility check: for every (asset, parent_sweep_id, split,
    horizon) group where the silver data could feed a pairwise loss
    matrix, the gold DM/MCS parquets must contain the group. Also requires
    `gold_quality_statistics_report.statistics_ready` to be True somewhere
    if DM or MCS is feasible.

    Emits `skipped(no_gold_dir)` when the use case has no gold dir.
    """

    name = "dm_mcs_persisted_executable"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        if not snapshot.has_gold_dir():
            return CheckResult(self.name, True, "skipped(no_gold_dir)")

        fact_oos = snapshot.get("fact_oos_predictions")
        dim_run = snapshot.get("dim_run")
        dm_gold = snapshot.scope_gold("gold_dm_pairwise_results", with_run_id_filter=False)
        mcs_gold = snapshot.scope_gold("gold_mcs_results", with_run_id_filter=False)
        report_gold = snapshot.scope_gold(
            "gold_quality_statistics_report", with_run_id_filter=False
        )

        feasible_keys_dm: set[tuple[object, object, object, object]] = set()
        feasible_keys_mcs: set[tuple[object, object, object, object]] = set()

        if not fact_oos.empty and not dim_run.empty:
            keep = [
                c
                for c in [
                    "run_id", "asset", "feature_set_name", "config_signature",
                    "parent_sweep_id", "status",
                ]
                if c in dim_run.columns
            ]
            if "run_id" in keep:
                tmp = fact_oos.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
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
                tmp = (
                    tmp[tmp["split"].astype(str) == "test"].copy()
                    if "split" in tmp.columns
                    else pd.DataFrame()
                )
                needed = {
                    "horizon", "target_timestamp_utc", "y_true", "y_pred",
                    "config_signature", "feature_set_name", "asset",
                }
                if not tmp.empty and needed.issubset(set(tmp.columns)):
                    for c in ["horizon", "y_true", "y_pred"]:
                        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                    tmp["target_timestamp_utc"] = pd.to_datetime(
                        tmp["target_timestamp_utc"], utc=True, errors="coerce"
                    )
                    tmp = tmp.dropna(
                        subset=[
                            "horizon", "target_timestamp_utc", "y_true", "y_pred",
                            "config_signature", "feature_set_name", "asset",
                        ]
                    ).copy()
                    if not tmp.empty:
                        tmp["horizon"] = tmp["horizon"].astype(int)
                        tmp["config_label"] = (
                            tmp["feature_set_name"].astype(str) + "|" + tmp["config_signature"].astype(str)
                        )
                        tmp["squared_error"] = (tmp["y_pred"] - tmp["y_true"]) ** 2
                        group_cols = [
                            c for c in ["asset", "parent_sweep_id", "split", "horizon"]
                            if c in tmp.columns
                        ]
                        if group_cols:
                            for keys, g in tmp.groupby(group_cols, dropna=False):
                                by_ts = (
                                    g.groupby(
                                        ["target_timestamp_utc", "config_label"], dropna=False
                                    )["squared_error"]
                                    .mean()
                                    .reset_index()
                                )
                                loss_matrix = by_ts.pivot(
                                    index="target_timestamp_utc",
                                    columns="config_label",
                                    values="squared_error",
                                )
                                loss_matrix = loss_matrix.dropna(axis=0, how="any")
                                kv = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
                                key4 = (
                                    kv.get("asset"), kv.get("parent_sweep_id"),
                                    kv.get("split"), kv.get("horizon"),
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

        executable_ok = (
            len(missing_dm) == 0
            and len(missing_mcs) == 0
            and ((not report_expected) or report_has_stats_ready)
        )
        detail = (
            f"feasible_dm={len(feasible_keys_dm)}, feasible_mcs={len(feasible_keys_mcs)}, "
            f"missing_dm={len(missing_dm)}, missing_mcs={len(missing_mcs)}, "
            f"report_stats_ready={report_has_stats_ready}"
        )
        return CheckResult(self.name, executable_ok, detail)


class GoldConfidenceCalibratedByHorizonCheck(QualityCheck):
    """`gold_prediction_metrics_by_run_split_horizon.confidence_calibrated`
    must not be NaN/inf for quantile-genuine runs, and must cover all
    expected horizons per run (val/test only).

    Emits `skipped(no_gold_dir)` when the use case has no gold dir.
    """

    name = "gold_confidence_calibrated_by_horizon"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        if not snapshot.has_gold_dir():
            return CheckResult(self.name, True, "skipped(no_gold_dir)")
        gold_conf = snapshot.scope_gold(
            "gold_prediction_metrics_by_run_split_horizon",
            with_run_id_filter=True,
        )
        if gold_conf.empty:
            return CheckResult(
                self.name, False, "missing_gold_prediction_metrics_by_run_split_horizon"
            )
        needed = {"run_id", "split", "horizon", "confidence_calibrated"}
        missing_cols = sorted(needed - set(gold_conf.columns))
        if missing_cols:
            return CheckResult(self.name, False, f"missing_columns={missing_cols}")

        conf = gold_conf[gold_conf["split"].astype(str).isin(["val", "test"])].copy()
        conf["horizon"] = pd.to_numeric(conf["horizon"], errors="coerce")
        conf["confidence_calibrated"] = pd.to_numeric(conf["confidence_calibrated"], errors="coerce")
        if "is_quantile_genuine" in conf.columns:
            is_genuine_bool = (
                conf["is_quantile_genuine"].astype(str).str.strip().str.lower().eq("true")
            )
            conf = conf[is_genuine_bool].copy()
        bad_conf = int(conf["confidence_calibrated"].isna().sum())
        non_finite = int(conf["confidence_calibrated"].isin([float("inf"), float("-inf")]).sum())
        horizon_misses: list[str] = []
        expected_by_run = snapshot.expected_horizons_by_run
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
        detail = (
            f"bad_confidence={bad_conf}, non_finite={non_finite}, "
            f"missing_expected_horizons={len(horizon_misses)}"
        )
        return CheckResult(self.name, confidence_ok, detail)


class GoldMetricsByConfigNOosContractCheck(QualityCheck):
    """`gold_prediction_metrics_by_config.n_oos` must exist, be positive,
    and equal the sum of `n_samples` from
    `gold_prediction_metrics_by_run_split_horizon` per group.

    Emits `skipped(no_gold_dir)` when the use case has no gold dir.
    """

    name = "gold_metrics_by_config_n_oos_contract"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        if not snapshot.has_gold_dir():
            return CheckResult(self.name, True, "skipped(no_gold_dir)")
        gold_by_cfg = snapshot.scope_gold(
            "gold_prediction_metrics_by_config", with_run_id_filter=False
        )
        gold_run_h = snapshot.scope_gold(
            "gold_prediction_metrics_by_run_split_horizon", with_run_id_filter=True
        )
        if gold_by_cfg.empty:
            return CheckResult(self.name, False, "missing_gold_prediction_metrics_by_config")
        if "n_oos" not in gold_by_cfg.columns:
            return CheckResult(self.name, False, "missing_n_oos_column")

        by = gold_by_cfg.copy()
        by["n_oos"] = pd.to_numeric(by["n_oos"], errors="coerce")
        non_positive = int((by["n_oos"].fillna(0) <= 0).sum())
        if gold_run_h.empty:
            return CheckResult(
                self.name,
                non_positive == 0,
                f"non_positive_n_oos={non_positive}, consistency=skipped(no_run_level_gold)",
            )
        needed = {
            "asset", "feature_set_name", "parent_sweep_id", "config_signature",
            "split", "horizon", "n_samples",
        }
        needed_by_cfg = needed - {"n_samples"}
        missing_by_cfg_cols = sorted(needed_by_cfg - set(by.columns))
        if missing_by_cfg_cols:
            return CheckResult(
                self.name, False, f"missing_by_config_columns={missing_by_cfg_cols}"
            )
        missing_run_level_cols = sorted(needed - set(gold_run_h.columns))
        if missing_run_level_cols:
            return CheckResult(
                self.name, False, f"missing_run_level_columns={missing_run_level_cols}"
            )

        run = gold_run_h.copy()
        run["n_samples"] = pd.to_numeric(run["n_samples"], errors="coerce").fillna(0)
        key_cols = [
            "asset", "feature_set_name", "parent_sweep_id",
            "config_signature", "split", "horizon",
        ]
        expected = (
            run.groupby(key_cols, dropna=False)["n_samples"]
            .sum()
            .reset_index()
            .rename(columns={"n_samples": "expected_n_oos"})
        )
        cmp = by.merge(expected, on=key_cols, how="left")
        cmp["expected_n_oos"] = pd.to_numeric(cmp["expected_n_oos"], errors="coerce")
        cmp["expected_n_oos"] = cmp["expected_n_oos"].fillna(0)
        mismatch = int(
            (cmp["n_oos"].fillna(0).astype(int) != cmp["expected_n_oos"].astype(int)).sum()
        )
        n_oos_ok = non_positive == 0 and mismatch == 0
        detail = f"non_positive_n_oos={non_positive}, mismatch_with_run_level={mismatch}"
        return CheckResult(self.name, n_oos_ok, detail)
