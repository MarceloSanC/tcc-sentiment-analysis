"""Cross-pipeline alignment quality checks (Stage 12 / F.0 cohort contracts).

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


class OosPairwiseTargetAlignmentCheck(QualityCheck):
    """Per (asset, parent_sweep_id, split_fingerprint, split, horizon) and per
    `config_signature` within that group, the set of `target_timestamp_utc`
    emitted on val/test splits must be identical across configs (otherwise
    pairwise DM/MCS builders can't intersect them).

    The group keys include `split_fingerprint` (from `dim_run`) so that
    walk-forward folds sharing a `parent_sweep_id` are still compared
    independently — without this, configs from different folds collapse
    into a single group and the timestamp-equality assertion is broken
    by design (folds cover disjoint OOS periods).
    """

    name = "oos_pairwise_target_alignment"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        fact_oos = snapshot.get("fact_oos_predictions")
        dim_run = snapshot.get("dim_run")
        if fact_oos.empty:
            return CheckResult(self.name, True, "ok")

        needed = {"run_id", "split", "horizon", "target_timestamp_utc"}
        if not needed.issubset(set(fact_oos.columns)):
            return CheckResult(self.name, True, "ok")
        if dim_run.empty or "run_id" not in dim_run.columns:
            return CheckResult(self.name, True, "ok")

        meta_cols = [
            c
            for c in [
                "run_id",
                "asset",
                "config_signature",
                "parent_sweep_id",
                "split_fingerprint",
            ]
            if c in dim_run.columns
        ]
        aligned = fact_oos.merge(
            dim_run[meta_cols].drop_duplicates("run_id"),
            on="run_id",
            how="left",
        )
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
        if not {"asset", "config_signature"}.issubset(set(aligned.columns)):
            return CheckResult(self.name, True, "ok")

        aligned = aligned[aligned["split"].astype(str).isin(["val", "test"])].copy()
        aligned["target_timestamp_utc"] = pd.to_datetime(
            aligned["target_timestamp_utc"], utc=True, errors="coerce"
        )
        gcols = [
            c
            for c in [
                "asset",
                "parent_sweep_id",
                "split_fingerprint",
                "split",
                "horizon",
            ]
            if c in aligned.columns
        ]
        issues: list[str] = []
        if gcols:
            for keys, grp in aligned.groupby(gcols, dropna=False):
                kv = dict(zip(gcols, keys if isinstance(keys, tuple) else (keys,)))
                sweep_value = kv.get("parent_sweep_id")
                if pd.isna(sweep_value) or str(sweep_value).strip().lower() in {
                    "", "none", "nan", "null", "<na>",
                }:
                    continue
                per_cfg: dict[str, set[str]] = {}
                for cfg, gc in grp.groupby("config_signature", dropna=False):
                    per_cfg[str(cfg)] = set(
                        gc["target_timestamp_utc"].dropna().astype(str).tolist()
                    )
                if len(per_cfg) < 2:
                    continue
                sets = list(per_cfg.values())
                base = sets[0]
                if any(s != base for s in sets[1:]):
                    min_len = min(len(s) for s in sets)
                    max_len = max(len(s) for s in sets)
                    issues.append(
                        f"asset={kv.get('asset')}|sweep={kv.get('parent_sweep_id')}"
                        f"|split_fp={kv.get('split_fingerprint')}|split={kv.get('split')}"
                        f"|h={kv.get('horizon')}|configs={len(per_cfg)}"
                        f"|min_ts={min_len}|max_ts={max_len}"
                    )
        return CheckResult(
            self.name,
            len(issues) == 0,
            "ok" if not issues else ", ".join(issues),
        )


class BaselinesSharedParentSweepCheck(QualityCheck):
    """Stage 12 gate: candidate sweeps must include at least one baseline
    run sharing the same `parent_sweep_id`. Active only when
    `scope_mode == 'cohort_decision'` (legacy/global_health silvers may
    legitimately mix without the Stage 12 contract).
    """

    name = "baselines_share_parent_sweep_id_with_candidates"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        scope = snapshot.scope_spec
        if scope is None or scope.scope_mode != "cohort_decision":
            return CheckResult(self.name, True, "skipped(not_cohort_decision)")
        dim_run = snapshot.get("dim_run")
        if dim_run.empty:
            return CheckResult(self.name, True, "skipped(not_cohort_decision)")
        needed_cols = {"parent_sweep_id", "feature_set_name", "model_version"}
        missing_cols = sorted(needed_cols - set(dim_run.columns))
        if missing_cols:
            return CheckResult(self.name, False, f"missing_columns={missing_cols}")

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
                issues.append(f"sweep={sweep_value},candidates={n_candidates},baselines=0")
        if issues:
            return CheckResult(self.name, False, ", ".join(issues))
        return CheckResult(self.name, True, "ok")


class TftBaselinesTimestampSubsetAlignmentCheck(QualityCheck):
    """Stage F.0.3 gate: per `(parent_sweep_id, split, horizon)`, TFT
    candidate runs and baseline runs must emit identical
    `target_timestamp_utc` sets. Active only under
    `scope_mode == 'cohort_decision'`.
    """

    name = "tft_baselines_timestamp_subset_alignment"

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        scope = snapshot.scope_spec
        scope_mode = scope.scope_mode if scope is not None else None
        if scope_mode != "cohort_decision":
            return CheckResult(self.name, True, "skipped(not_cohort_decision)")
        dim_run = snapshot.get("dim_run")
        fact_oos = snapshot.get("fact_oos_predictions")
        if dim_run.empty or fact_oos.empty:
            return CheckResult(self.name, True, "skipped(empty_silver)")
        if not {"feature_set_name", "model_version"}.issubset(dim_run.columns):
            return CheckResult(self.name, False, "missing_required_dim_run_columns")
        if "parent_sweep_id" not in dim_run.columns:
            return CheckResult(self.name, False, "missing_parent_sweep_id_in_dim_run")

        dim_view = dim_run.copy()
        fset_lower = dim_view["feature_set_name"].astype(str).str.strip().str.lower()
        mver_lower = dim_view["model_version"].astype(str).str.strip().str.lower()
        dim_view["_is_baseline"] = fset_lower.eq("baseline") | mver_lower.str.startswith(
            "baseline_"
        )
        join_cols = ["run_id", "_is_baseline"]
        if "parent_sweep_id" not in fact_oos.columns:
            join_cols.append("parent_sweep_id")
        join_view = dim_view[join_cols].drop_duplicates("run_id")
        oos_view = fact_oos.merge(join_view, on="run_id", how="left")
        oos_view["_is_baseline"] = oos_view["_is_baseline"].fillna(False).astype(bool)
        grp_cols = [
            c for c in ["asset", "parent_sweep_id", "split", "horizon"]
            if c in oos_view.columns
        ]
        if "parent_sweep_id" not in grp_cols:
            return CheckResult(self.name, True, "skipped(parent_sweep_id_join_failed)")

        issues: list[str] = []
        for keys, grp in oos_view.groupby(grp_cols, dropna=False):
            kv = dict(zip(grp_cols, keys if isinstance(keys, tuple) else (keys,)))
            sweep_val = str(kv.get("parent_sweep_id"))
            if not sweep_val or sweep_val.lower() in {"none", "nan", "null", "<na>"}:
                continue
            tft_ts = set(
                grp.loc[~grp["_is_baseline"], "target_timestamp_utc"]
                .dropna().astype(str).tolist()
            )
            bas_ts = set(
                grp.loc[grp["_is_baseline"], "target_timestamp_utc"]
                .dropna().astype(str).tolist()
            )
            if not tft_ts or not bas_ts:
                continue
            if tft_ts != bas_ts:
                issues.append(
                    f"sweep={sweep_val}|split={kv.get('split')}"
                    f"|h={kv.get('horizon')}|tft_n={len(tft_ts)}"
                    f"|baseline_n={len(bas_ts)}"
                    f"|symdiff={len(tft_ts.symmetric_difference(bas_ts))}"
                )
        if issues:
            return CheckResult(self.name, False, ", ".join(issues))
        return CheckResult(self.name, True, "ok")
