from __future__ import annotations

import json
import math

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.services.dm_tft_vs_baseline import (
    compute_dm_family,
    mean_pinball_post_guardrail,
)
from src.domain.services.fold_dedup_resolver import FoldPeriod
from src.domain.services.holm_family_6 import apply_holm_one_sided
from src.domain.services.marginal_coverage_calculator import compute_marginal_coverage
from src.domain.services.phase_b_tier_policy import (
    PhaseBTierPolicy,
    default_phase_b_policy,
)
from src.domain.services.tier_classifier_per_hypothesis import (
    TierVerdict,
    classify_h1,
    classify_h2a,
    classify_h2b,
)
from src.interfaces.phase_b_tier_sidecar_writer import PhaseBTierSidecarWriter


@dataclass(frozen=True)
class PhaseBTierMetricsResult:
    cohort_id: str
    sidecar_paths: dict[str, Path]
    verdicts: list[TierVerdict]


def _read_parquet_tree(root: Path) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    if root.is_file():
        return pd.read_parquet(root)
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[numeric.notna()]
    return float(numeric.median()) if len(numeric) else math.nan


def _json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


_MISSING_FOLD_LABELS = {"", "none", "nan", "None", "NaT", "<NA>"}


def _is_missing_fold(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().isin(_MISSING_FOLD_LABELS)


def _utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


class ComputePhaseBTierMetricsUseCase:
    def __init__(
        self,
        *,
        silver_dir: Path,
        gold_dir: Path,
        sidecar_writer: PhaseBTierSidecarWriter,
        policy: PhaseBTierPolicy | None = None,
    ) -> None:
        self.silver_dir = Path(silver_dir)
        self.gold_dir = Path(gold_dir)
        self.sidecar_writer = sidecar_writer
        self.policy = policy or default_phase_b_policy()

    def _load_dim_run(self, asset: str, parent_sweep_id: str) -> pd.DataFrame:
        dim_run = _read_parquet_tree(self.silver_dir / "dim_run")
        if dim_run.empty:
            raise ValueError(f"dim_run not found under {self.silver_dir}")
        required = {"run_id", "asset", "parent_sweep_id", "feature_set_name", "model_version"}
        missing = sorted(required - set(dim_run.columns))
        if missing:
            raise ValueError(f"dim_run missing required columns: {missing}")
        dim_run = dim_run.copy()
        dim_run["run_id"] = dim_run["run_id"].astype(str)
        scoped = dim_run[
            dim_run["asset"].astype(str).eq(str(asset))
            & dim_run["parent_sweep_id"].astype(str).eq(str(parent_sweep_id))
        ].copy()
        if "status" in scoped.columns:
            scoped = scoped[scoped["status"].astype(str).str.lower().eq("ok")].copy()
        if scoped.empty:
            raise ValueError(f"no dim_run rows for asset={asset} parent_sweep_id={parent_sweep_id}")
        return scoped.reset_index(drop=True)

    def _load_fact_oos(self, dim_run: pd.DataFrame, asset: str) -> pd.DataFrame:
        fact_oos = _read_parquet_tree(self.silver_dir / "fact_oos_predictions")
        if fact_oos.empty:
            raise ValueError(f"fact_oos_predictions not found under {self.silver_dir}")
        fact_oos = fact_oos.copy()
        fact_oos["run_id"] = fact_oos["run_id"].astype(str)
        run_ids = set(dim_run["run_id"].astype(str))
        fact_oos = fact_oos[fact_oos["run_id"].isin(run_ids)].copy()
        if "asset" in fact_oos.columns:
            fact_oos = fact_oos[fact_oos["asset"].astype(str).eq(str(asset))].copy()
        if fact_oos.empty:
            raise ValueError(f"no fact_oos_predictions rows for asset={asset}")
        return fact_oos.reset_index(drop=True)

    def _load_fact_run_snapshot(self, dim_run: pd.DataFrame, asset: str) -> pd.DataFrame:
        snapshot = _read_parquet_tree(self.silver_dir / "fact_run_snapshot")
        if snapshot.empty:
            return pd.DataFrame()
        snapshot = snapshot.copy()
        snapshot["run_id"] = snapshot["run_id"].astype(str)
        snapshot = snapshot[snapshot["run_id"].isin(set(dim_run["run_id"].astype(str)))].copy()
        if "asset" in snapshot.columns:
            snapshot = snapshot[snapshot["asset"].astype(str).eq(str(asset))].copy()
        return snapshot.reset_index(drop=True)

    def _load_gold_calibration(self, dim_run: pd.DataFrame, asset: str, parent_sweep_id: str) -> pd.DataFrame:
        path = self.gold_dir / "gold_prediction_calibration.parquet"
        calibration = _read_parquet_tree(path)
        if calibration.empty:
            raise ValueError(f"gold_prediction_calibration not found at {path}")
        calibration = calibration.copy()
        calibration["run_id"] = calibration["run_id"].astype(str)
        calibration = calibration[
            calibration["asset"].astype(str).eq(str(asset))
            & calibration["parent_sweep_id"].astype(str).eq(str(parent_sweep_id))
            & calibration["run_id"].isin(set(dim_run["run_id"].astype(str)))
        ].copy()
        if calibration.empty:
            raise ValueError(
                f"no gold_prediction_calibration rows for asset={asset} parent_sweep_id={parent_sweep_id}"
            )
        meta = dim_run[["run_id", "model_version", "feature_set_name", "fold", "seed"]].drop_duplicates("run_id")
        calibration = calibration.merge(meta, on="run_id", how="left", suffixes=("", "_dim"))
        return calibration.reset_index(drop=True)

    @staticmethod
    def _resolve_fold_periods(
        dim_run: pd.DataFrame,
        fact_run_snapshot: pd.DataFrame,
    ) -> dict[str, FoldPeriod]:
        if "fold" not in dim_run.columns:
            raise ValueError("dim_run must contain fold for Phase B tier metrics")
        source = dim_run[["run_id", "fold"]].copy()
        if not fact_run_snapshot.empty and "train_end_utc" in fact_run_snapshot.columns:
            snap = fact_run_snapshot[["run_id", "train_end_utc"]].copy()
            source = source.merge(snap.drop_duplicates("run_id"), on="run_id", how="left")
        elif "train_end_utc" in dim_run.columns:
            source["train_end_utc"] = dim_run["train_end_utc"]
        else:
            raise ValueError("train_end_utc not available in fact_run_snapshot or dim_run")

        source["fold"] = source["fold"].astype(str)
        source["train_end_utc"] = pd.to_datetime(
            source["train_end_utc"],
            utc=True,
            errors="coerce",
        )
        source = source.dropna(subset=["fold", "train_end_utc"])
        source = source[~source["fold"].isin(["", "none", "nan", "None"])].copy()
        periods: dict[str, FoldPeriod] = {}
        for fold_name, group in source.groupby("fold"):
            periods[str(fold_name)] = FoldPeriod(
                name=str(fold_name),
                train_end_utc=group["train_end_utc"].max(),
            )
        if not periods:
            raise ValueError("no fold periods could be resolved")
        return periods

    @staticmethod
    def _enrich_missing_tft_folds(
        dim_run: pd.DataFrame,
        fact_run_snapshot: pd.DataFrame,
        fold_periods: dict[str, FoldPeriod],
    ) -> pd.DataFrame:
        out = dim_run.copy()
        if "fold" not in out.columns:
            raise ValueError("dim_run must contain fold for Phase B tier metrics")

        tft_mask = out["feature_set_name"].astype(str).ne("baseline")
        missing_tft_fold = tft_mask & _is_missing_fold(out["fold"])
        if not missing_tft_fold.any():
            return out.reset_index(drop=True)

        if fact_run_snapshot.empty or "train_end_utc" not in fact_run_snapshot.columns:
            raise ValueError("cannot resolve missing TFT folds without fact_run_snapshot.train_end_utc")

        train_end_to_fold = {
            _utc_timestamp(period.train_end_utc): str(fold_name)
            for fold_name, period in fold_periods.items()
        }
        snapshot = fact_run_snapshot[["run_id", "train_end_utc"]].copy()
        snapshot["run_id"] = snapshot["run_id"].astype(str)
        snapshot["_snapshot_train_end_utc"] = pd.to_datetime(
            snapshot["train_end_utc"],
            utc=True,
            errors="coerce",
        )
        snapshot = snapshot.drop(columns=["train_end_utc"]).drop_duplicates("run_id")

        out["run_id"] = out["run_id"].astype(str)
        out = out.merge(snapshot, on="run_id", how="left")
        tft_mask = out["feature_set_name"].astype(str).ne("baseline")
        missing_tft_fold = tft_mask & _is_missing_fold(out["fold"])
        out.loc[missing_tft_fold, "fold"] = out.loc[
            missing_tft_fold,
            "_snapshot_train_end_utc",
        ].map(train_end_to_fold)
        unresolved = tft_mask & _is_missing_fold(out["fold"])
        if unresolved.any():
            sample = ", ".join(out.loc[unresolved, "run_id"].astype(str).head(5))
            raise ValueError(f"unresolved TFT fold mapping for Phase B DM: {sample}")

        return out.drop(columns=["_snapshot_train_end_utc"]).reset_index(drop=True)

    def _run_groups(self, dim_run: pd.DataFrame) -> tuple[set[str], dict[str, set[str]]]:
        tft = dim_run[dim_run["feature_set_name"].astype(str).ne("baseline")].copy()
        baselines = dim_run[dim_run["feature_set_name"].astype(str).eq("baseline")].copy()
        tft_run_ids = set(tft["run_id"].astype(str))
        if not tft_run_ids:
            raise ValueError("no TFT candidate runs found")
        baselines_by_model_version: dict[str, set[str]] = {}
        for model_version in self.policy.baseline_model_versions:
            rows = baselines[baselines["model_version"].astype(str).eq(model_version)]
            run_ids = set(rows["run_id"].astype(str))
            if not run_ids:
                raise ValueError(f"missing baseline runs for model_version={model_version}")
            baselines_by_model_version[str(model_version)] = run_ids
        return tft_run_ids, baselines_by_model_version

    def _dm_family_6(
        self,
        fact_oos: pd.DataFrame,
        dim_run: pd.DataFrame,
        tft_run_ids: set[str],
        baselines_by_model_version: dict[str, set[str]],
        fold_periods: dict[str, FoldPeriod],
    ) -> pd.DataFrame:
        dm = compute_dm_family(
            fact_oos,
            dim_run,
            tft_run_ids,
            baselines_by_model_version,
            fold_periods,
            horizons=list(self.policy.confirmatory_horizons),
            split="test",
        )
        holm = apply_holm_one_sided(dm["pvalue_one_sided_less"])
        out = pd.concat([dm.reset_index(drop=True), holm.reset_index(drop=True)], axis=1)
        out["pvalue_one_sided"] = out["pvalue_one_sided_less"]
        out["analysis_role"] = "primary_family_6"
        return out

    def _dm_family_18_sensitivity(
        self,
        fact_oos: pd.DataFrame,
        dim_run: pd.DataFrame,
        baselines_by_model_version: dict[str, set[str]],
        fold_periods: dict[str, FoldPeriod],
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for fold_name in sorted(fold_periods):
            fold_dim = dim_run[dim_run["fold"].astype(str).eq(fold_name)]
            tft_ids = set(
                fold_dim[fold_dim["feature_set_name"].astype(str).ne("baseline")]["run_id"].astype(str)
            )
            fold_baselines = {
                baseline: set(
                    fold_dim[
                        fold_dim["model_version"].astype(str).eq(baseline)
                        & fold_dim["feature_set_name"].astype(str).eq("baseline")
                    ]["run_id"].astype(str)
                )
                for baseline in baselines_by_model_version
            }
            dm = compute_dm_family(
                fact_oos,
                fold_dim,
                tft_ids,
                fold_baselines,
                {fold_name: fold_periods[fold_name]},
                horizons=list(self.policy.confirmatory_horizons),
                split="test",
            )
            dm["fold_name"] = fold_name
            rows.append(dm)
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if out.empty:
            return out
        holm = apply_holm_one_sided(out["pvalue_one_sided_less"])
        out = pd.concat([out.reset_index(drop=True), holm.reset_index(drop=True)], axis=1)
        out["pvalue_one_sided"] = out["pvalue_one_sided_less"]
        out["analysis_role"] = "sensitivity_conservative"
        return out

    def _delta_pinball(
        self,
        fact_oos: pd.DataFrame,
        dim_run: pd.DataFrame,
        tft_run_ids: set[str],
    ) -> pd.DataFrame:
        required = {
            "run_id",
            "split",
            "horizon",
            "y_true",
            "quantile_p10_post_guardrail",
            "quantile_p50_post_guardrail",
            "quantile_p90_post_guardrail",
        }
        missing = sorted(required - set(fact_oos.columns))
        if missing:
            raise ValueError(f"fact_oos missing required columns for delta pinball: {missing}")

        oos = fact_oos[fact_oos["split"].astype(str).eq("test")].copy()
        oos["run_id"] = oos["run_id"].astype(str)
        oos["horizon"] = pd.to_numeric(oos["horizon"], errors="coerce")
        oos["mean_pinball_post_guardrail"] = mean_pinball_post_guardrail(oos)
        oos = oos.dropna(subset=["run_id", "horizon", "mean_pinball_post_guardrail"])
        per_run = (
            oos.groupby(["run_id", "horizon"], dropna=False)["mean_pinball_post_guardrail"]
            .mean()
            .reset_index()
        )
        meta = dim_run[["run_id", "model_version", "feature_set_name"]].copy()
        meta["run_id"] = meta["run_id"].astype(str)
        per_run = per_run.merge(meta.drop_duplicates("run_id"), on="run_id", how="left")

        rows: list[dict[str, object]] = []
        for horizon in self.policy.confirmatory_horizons:
            horizon_rows = per_run[per_run["horizon"].eq(int(horizon))]
            tft_pinball = _median(
                horizon_rows[horizon_rows["run_id"].astype(str).isin(tft_run_ids)][
                    "mean_pinball_post_guardrail"
                ]
            )
            for baseline in self.policy.baseline_model_versions:
                baseline_pinball = _median(
                    horizon_rows[horizon_rows["model_version"].astype(str).eq(baseline)][
                        "mean_pinball_post_guardrail"
                    ]
                )
                delta = (
                    (baseline_pinball - tft_pinball) / baseline_pinball
                    if baseline_pinball and math.isfinite(baseline_pinball)
                    else math.nan
                )
                rows.append(
                    {
                        "horizon": int(horizon),
                        "baseline_model_version": baseline,
                        "mean_pinball_tft_post_guardrail": tft_pinball,
                        "mean_pinball_baseline_post_guardrail": baseline_pinball,
                        "delta_mean_pinball_rel": float(delta),
                    }
                )
        return pd.DataFrame(rows)

    def _classify(
        self,
        marginal_coverage: pd.DataFrame,
        dm_family_6: pd.DataFrame,
        delta_pinball: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[TierVerdict]]:
        rows: list[dict[str, object]] = []
        verdicts: list[TierVerdict] = []
        coverage_test = marginal_coverage[marginal_coverage["split"].astype(str).eq("test")]
        for horizon in self.policy.confirmatory_horizons:
            h_cov = coverage_test[pd.to_numeric(coverage_test["horizon"], errors="coerce").eq(int(horizon))]
            coverage_q10 = _median(h_cov["coverage_q10"])
            coverage_q50 = _median(h_cov["coverage_q50"])
            coverage_q90 = _median(h_cov["coverage_q90"])
            picp_error = _median(h_cov["coverage_error_picp"])
            mpiw = _median(h_cov["mpiw"])
            h1 = classify_h1(
                coverage_q10,
                coverage_q50,
                coverage_q90,
                picp_error,
                mpiw,
                self.policy,
                horizon=int(horizon),
            )

            dm_h = dm_family_6[pd.to_numeric(dm_family_6["horizon"], errors="coerce").eq(int(horizon))]
            delta_h = delta_pinball[pd.to_numeric(delta_pinball["horizon"], errors="coerce").eq(int(horizon))]
            p_by_baseline = {
                str(row["baseline_model_version"]): float(row["pvalue_adj_holm"])
                for _, row in dm_h.iterrows()
            }
            delta_by_baseline = {
                str(row["baseline_model_version"]): float(row["delta_mean_pinball_rel"])
                for _, row in delta_h.iterrows()
            }
            h2a = classify_h2a(
                p_by_baseline.get(self.policy.h2a_primary_baseline, math.nan),
                delta_by_baseline.get(self.policy.h2a_primary_baseline, math.nan),
                mpiw,
                self.policy,
                horizon=int(horizon),
            )
            h2b = classify_h2b(
                p_by_baseline,
                delta_by_baseline,
                mpiw,
                self.policy,
                horizon=int(horizon),
            )
            for verdict in [h1, h2a, h2b]:
                verdicts.append(verdict)
                rows.append(
                    {
                        "hypothesis": verdict.hypothesis,
                        "horizon": verdict.horizon,
                        "tier": verdict.tier,
                        "criteria_passed_dict": _json(verdict.criteria_passed),
                        "numerical_inputs": _json(verdict.numerical_inputs),
                        "justification": verdict.justification,
                    }
                )
        return pd.DataFrame(rows), verdicts

    def execute(self, *, asset: str, parent_sweep_id: str) -> PhaseBTierMetricsResult:
        dim_run = self._load_dim_run(asset, parent_sweep_id)
        fact_oos = self._load_fact_oos(dim_run, asset)
        fact_run_snapshot = self._load_fact_run_snapshot(dim_run, asset)
        fold_periods = self._resolve_fold_periods(dim_run, fact_run_snapshot)
        dim_run = self._enrich_missing_tft_folds(dim_run, fact_run_snapshot, fold_periods)
        tft_run_ids, baselines_by_model_version = self._run_groups(dim_run)

        marginal_coverage = compute_marginal_coverage(
            fact_oos,
            tft_run_ids,
            [0.1, 0.5, 0.9],
        )
        dm_family_6 = self._dm_family_6(
            fact_oos,
            dim_run,
            tft_run_ids,
            baselines_by_model_version,
            fold_periods,
        )
        dm_family_18 = self._dm_family_18_sensitivity(
            fact_oos,
            dim_run,
            baselines_by_model_version,
            fold_periods,
        )
        delta_pinball = self._delta_pinball(fact_oos, dim_run, tft_run_ids)
        tier_verdict, verdicts = self._classify(
            marginal_coverage,
            dm_family_6,
            delta_pinball,
        )

        for df in [marginal_coverage, dm_family_6, dm_family_18, delta_pinball, tier_verdict]:
            df.insert(0, "parent_sweep_id", str(parent_sweep_id))
            df.insert(1, "asset", str(asset))

        sidecar_paths = {
            "marginal_coverage": self.sidecar_writer.write_marginal_coverage(
                marginal_coverage,
                parent_sweep_id,
            ),
            "dm_family_6": self.sidecar_writer.write_dm_family_6(
                dm_family_6,
                parent_sweep_id,
            ),
            "dm_family_18_sensitivity": self.sidecar_writer.write_dm_family_18_sensitivity(
                dm_family_18,
                parent_sweep_id,
            ),
            "delta_pinball": self.sidecar_writer.write_delta_pinball(
                delta_pinball,
                parent_sweep_id,
            ),
            "tier_verdict": self.sidecar_writer.write_tier_verdict(
                tier_verdict,
                parent_sweep_id,
            ),
        }
        return PhaseBTierMetricsResult(
            cohort_id=str(parent_sweep_id),
            sidecar_paths=sidecar_paths,
            verdicts=verdicts,
        )
