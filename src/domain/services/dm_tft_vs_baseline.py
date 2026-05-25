from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.domain.services.fold_dedup_resolver import (
    FoldPeriod,
    RunInFoldCandidate,
    select_operationally_latest_fold,
)


@dataclass(frozen=True)
class DMResult:
    dm_stat: float
    pvalue_two_sided: float
    pvalue_one_sided_less: float
    hac_lag_used: int
    hln_applied: bool
    n_obs: int


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _to_utc(value: Any) -> pd.Timestamp | None:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _pinball(y_true: pd.Series, y_hat: pd.Series, q: float) -> pd.Series:
    diff = y_true - y_hat
    return np.maximum(q * diff, (q - 1.0) * diff)


def _mean_pinball_post_guardrail(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(df["y_true"], errors="coerce")
    q10 = pd.to_numeric(df["quantile_p10_post_guardrail"], errors="coerce")
    q50 = pd.to_numeric(df["quantile_p50_post_guardrail"], errors="coerce")
    q90 = pd.to_numeric(df["quantile_p90_post_guardrail"], errors="coerce")
    return (_pinball(y, q10, 0.1) + _pinball(y, q50, 0.5) + _pinball(y, q90, 0.9)) / 3.0


def _candidate_for_run(
    row: pd.Series,
    fold_periods: dict[str, FoldPeriod],
) -> RunInFoldCandidate | None:
    fold_name = str(row.get("fold") or "")
    period = fold_periods.get(fold_name)
    if period is None:
        return None
    seed_raw = row.get("seed")
    seed = None if pd.isna(seed_raw) else int(seed_raw)
    return RunInFoldCandidate(
        run_id=str(row["run_id"]),
        fold_name=fold_name,
        train_end_utc=period.train_end_utc,
        model_version=str(row.get("model_version") or ""),
        seed=seed,
    )


def _prepare_loss_frame(
    fact_oos: pd.DataFrame,
    dim_run: pd.DataFrame,
    run_ids: set[str],
    *,
    horizon: int,
    split: str,
) -> pd.DataFrame:
    required_oos = {
        "run_id",
        "split",
        "horizon",
        "target_timestamp_utc",
        "y_true",
        "quantile_p10_post_guardrail",
        "quantile_p50_post_guardrail",
        "quantile_p90_post_guardrail",
    }
    missing_oos = sorted(required_oos - set(fact_oos.columns))
    if missing_oos:
        raise ValueError(f"fact_oos missing required columns: {missing_oos}")
    required_dim = {"run_id", "fold", "model_version", "seed"}
    missing_dim = sorted(required_dim - set(dim_run.columns))
    if missing_dim:
        raise ValueError(f"dim_run missing required columns: {missing_dim}")

    df = fact_oos.copy()
    df["run_id"] = df["run_id"].astype(str)
    df = df[df["run_id"].isin({str(r) for r in run_ids})].copy()
    df = df[df["split"].astype(str).eq(str(split))].copy()
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    df = df[df["horizon"].eq(int(horizon))].copy()
    if df.empty:
        return pd.DataFrame()

    keep = ["run_id", "fold", "model_version", "seed"]
    if "feature_set_name" in dim_run.columns:
        keep.append("feature_set_name")
    meta = dim_run[keep].copy()
    meta["run_id"] = meta["run_id"].astype(str)
    df = df.merge(meta.drop_duplicates("run_id"), on="run_id", how="left")
    df["target_timestamp_utc"] = pd.to_datetime(
        df["target_timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    df["loss"] = _mean_pinball_post_guardrail(df)
    df = df.dropna(subset=["target_timestamp_utc", "loss", "fold"])
    return df


def build_loss_diff_series(
    fact_oos: pd.DataFrame,
    dim_run: pd.DataFrame,
    tft_run_ids: set[str],
    baseline_run_ids: set[str],
    fold_periods: dict[str, FoldPeriod],
    horizon: int,
    split: str,
) -> pd.Series:
    tft = _prepare_loss_frame(
        fact_oos,
        dim_run,
        tft_run_ids,
        horizon=int(horizon),
        split=str(split),
    )
    baseline = _prepare_loss_frame(
        fact_oos,
        dim_run,
        baseline_run_ids,
        horizon=int(horizon),
        split=str(split),
    )
    if tft.empty or baseline.empty:
        return pd.Series(dtype="float64", name="loss_diff")

    tft_meta = tft[["run_id", "fold", "model_version", "seed"]].drop_duplicates()
    baseline_meta = baseline[["run_id", "fold", "model_version", "seed"]].drop_duplicates()
    tft_by_ts = {ts: g for ts, g in tft.groupby("target_timestamp_utc")}
    baseline_by_ts = {ts: g for ts, g in baseline.groupby("target_timestamp_utc")}
    common_ts = sorted(set(tft_by_ts).intersection(baseline_by_ts))

    values: dict[pd.Timestamp, float] = {}
    for target_ts in common_ts:
        tft_candidates = [
            candidate
            for _, row in tft_meta[tft_meta["run_id"].isin(tft_by_ts[target_ts]["run_id"])].iterrows()
            if (candidate := _candidate_for_run(row, fold_periods)) is not None
        ]
        baseline_candidates = [
            candidate
            for _, row in baseline_meta[
                baseline_meta["run_id"].isin(baseline_by_ts[target_ts]["run_id"])
            ].iterrows()
            if (candidate := _candidate_for_run(row, fold_periods)) is not None
        ]
        tft_selected = select_operationally_latest_fold(
            tft_candidates,
            target_ts,
            int(horizon),
        )
        baseline_selected = select_operationally_latest_fold(
            baseline_candidates,
            target_ts,
            int(horizon),
        )
        if tft_selected is None or baseline_selected is None:
            continue

        tft_losses = tft_by_ts[target_ts][
            tft_by_ts[target_ts]["fold"].astype(str).eq(tft_selected.fold_name)
        ]["loss"]
        baseline_losses = baseline_by_ts[target_ts][
            baseline_by_ts[target_ts]["fold"].astype(str).eq(baseline_selected.fold_name)
        ]["loss"]
        if tft_losses.empty or baseline_losses.empty:
            continue
        values[target_ts] = float(tft_losses.mean() - baseline_losses.mean())

    return pd.Series(values, name="loss_diff", dtype="float64").sort_index()


def compute_dm_hln_hac(
    d_series: pd.Series,
    horizon: int,
    alpha: float = 0.05,
) -> DMResult:
    del alpha  # alpha is part of the public contract; p-values remain continuous.
    d = pd.to_numeric(d_series, errors="coerce").dropna().to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    n = int(len(d))
    lag = max(int(horizon) - 1, 1)
    if n < 2:
        return DMResult(math.nan, math.nan, math.nan, lag, True, n)

    mean_d = float(np.mean(d))
    centered = d - mean_d
    gamma0 = float(np.dot(centered, centered) / n)
    hac = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        cov = float(np.dot(centered[k:], centered[:-k]) / n)
        weight = 1.0 - (k / (lag + 1.0))
        hac += 2.0 * weight * cov
    var_mean = hac / n
    if var_mean <= 0.0 or not math.isfinite(var_mean):
        if abs(mean_d) <= 1e-15:
            dm_stat = 0.0
        else:
            dm_stat = math.copysign(math.inf, mean_d)
    else:
        dm_stat = mean_d / math.sqrt(var_mean)

    h = int(horizon)
    hln_factor = (n + 1 - 2 * h + (h * (h - 1) / n)) / n
    if hln_factor <= 0.0 or not math.isfinite(hln_factor):
        dm_hln = math.nan
    else:
        dm_hln = float(dm_stat * math.sqrt(hln_factor))

    if not math.isfinite(dm_hln):
        p_two = 0.0 if math.isinf(dm_hln) else math.nan
        p_less = 0.0 if dm_hln < 0 else (1.0 if dm_hln > 0 else math.nan)
    else:
        p_two = float(2.0 * (1.0 - _norm_cdf(abs(dm_hln))))
        p_less = float(_norm_cdf(dm_hln))
    return DMResult(
        dm_stat=float(dm_hln),
        pvalue_two_sided=p_two,
        pvalue_one_sided_less=p_less,
        hac_lag_used=lag,
        hln_applied=True,
        n_obs=n,
    )


def compute_dm_family(
    fact_oos: pd.DataFrame,
    dim_run: pd.DataFrame,
    tft_run_ids: set[str],
    baselines_by_model_version: dict[str, set[str]],
    fold_periods: dict[str, FoldPeriod],
    horizons: list[int],
    split: str = "test",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in sorted({int(h) for h in horizons}):
        for baseline_model_version, baseline_run_ids in sorted(
            baselines_by_model_version.items()
        ):
            d_series = build_loss_diff_series(
                fact_oos,
                dim_run,
                tft_run_ids,
                baseline_run_ids,
                fold_periods,
                horizon=horizon,
                split=split,
            )
            result = compute_dm_hln_hac(d_series, horizon=horizon)
            mean_loss_diff = (
                float(pd.to_numeric(d_series, errors="coerce").mean())
                if len(d_series)
                else math.nan
            )
            rows.append(
                {
                    "split": str(split),
                    "horizon": int(horizon),
                    "baseline_model_version": str(baseline_model_version),
                    "mean_loss_diff_tft_minus_baseline": mean_loss_diff,
                    "dm_stat": result.dm_stat,
                    "pvalue_two_sided": result.pvalue_two_sided,
                    "pvalue_one_sided_less": result.pvalue_one_sided_less,
                    "hac_lag_used": result.hac_lag_used,
                    "hln_applied": result.hln_applied,
                    "n_obs_effective": result.n_obs,
                    "direction": "tft_better" if mean_loss_diff < 0 else "baseline_better_or_equal",
                    "dedup_rule": "operationally_latest_fold",
                    "seed_aggregation": "mean_loss_diff_by_timestamp",
                    "loss_function": "pinball_loss_post_guardrail",
                }
            )
    return pd.DataFrame(rows)

