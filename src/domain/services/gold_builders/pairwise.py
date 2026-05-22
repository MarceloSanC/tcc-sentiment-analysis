"""Pairwise-cluster gold builders.

4 Tier 1 builders: Diebold-Mariano (with Holm correction), Model
Confidence Set, win-rate, and paired-intersection-by-horizon. All
share `_pairwise_preprocess()` which materializes the same
`config_label`/`squared_error` shape the monolith produced. Statistical
helpers (`_compute_dm_pairwise_from_loss_matrix`,
`_compute_mcs_from_loss_matrix`, `_apply_holm_adjustment_for_dm`) are
module-level so the unit tests can target them directly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.domain.services.gold_builders.base import (
    BuildContext,
    GoldBuilder,
    GoldBuilderSnapshot,
)


def _pairwise_group_cols(df: pd.DataFrame) -> list[str]:
    cols = ["asset", "parent_sweep_id", "split", "horizon"]
    if "split_signature" in df.columns:
        cols.insert(2, "split_signature")
    return [c for c in cols if c in df.columns]


def _ensure_split_signature_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "split_signature" not in out.columns and "split_fingerprint" in out.columns:
        out["split_signature"] = out["split_fingerprint"]
    return out


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _select_top_configs_for_pairwise(
    grouped_oos: pd.DataFrame,
    *,
    max_configs: int = 50,
) -> pd.DataFrame:
    if (
        grouped_oos.empty
        or "config_label" not in grouped_oos.columns
        or "squared_error" not in grouped_oos.columns
    ):
        return grouped_oos
    cfg_count = int(grouped_oos["config_label"].nunique())
    if cfg_count <= max_configs:
        return grouped_oos
    rank = (
        grouped_oos.groupby("config_label", dropna=False)["squared_error"]
        .mean()
        .sort_values(ascending=True)
        .head(max_configs)
    )
    keep = set(rank.index.tolist())
    return grouped_oos[grouped_oos["config_label"].isin(keep)].copy()


def _compute_dm_pairwise_from_loss_matrix(loss_matrix: pd.DataFrame) -> pd.DataFrame:
    if loss_matrix.empty or loss_matrix.shape[1] < 2:
        return pd.DataFrame()

    dm_rows: list[dict[str, object]] = []
    configs = loss_matrix.columns.tolist()
    for i, left in enumerate(configs):
        for right in configs[i + 1 :]:
            d = (loss_matrix[left] - loss_matrix[right]).to_numpy(dtype=float)
            d = d[np.isfinite(d)]
            n = len(d)
            if n < 5:
                continue
            mean_d = float(np.mean(d))
            d_centered = d - mean_d
            lag = int(min(max(1, n ** (1 / 3)), 10))
            gamma0 = float(np.dot(d_centered, d_centered) / n)
            hac = gamma0
            for k in range(1, lag + 1):
                cov = float(np.dot(d_centered[k:], d_centered[:-k]) / n)
                weight = 1.0 - (k / (lag + 1))
                hac += 2.0 * weight * cov
            var_mean = hac / n
            if var_mean <= 0 or not math.isfinite(var_mean):
                continue
            stat = mean_d / math.sqrt(var_mean)
            pvalue = 2.0 * (1.0 - _norm_cdf(abs(stat)))
            dm_rows.append(
                {
                    "left_config": str(left),
                    "right_config": str(right),
                    "n": int(n),
                    "mean_loss_diff_left_minus_right": float(mean_d),
                    "dm_stat": float(stat),
                    "pvalue_two_sided": float(pvalue),
                }
            )
    return pd.DataFrame(dm_rows).sort_values("pvalue_two_sided") if dm_rows else pd.DataFrame()


def _compute_mcs_from_loss_matrix(
    loss_matrix: pd.DataFrame,
    *,
    alpha: float = 0.05,
    bootstrap_samples: int = 300,
    block_len: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    if loss_matrix.empty or loss_matrix.shape[1] < 2:
        return pd.DataFrame()

    mean_loss = loss_matrix.mean(axis=0).sort_values()
    configs_order = mean_loss.index.tolist()
    mat = loss_matrix[configs_order].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    active = list(range(mat.shape[1]))

    def _block_bootstrap_indices(n_obs: int) -> np.ndarray:
        idx: list[int] = []
        while len(idx) < n_obs:
            start = int(rng.integers(0, n_obs))
            block = [(start + o) % n_obs for o in range(block_len)]
            idx.extend(block)
        return np.asarray(idx[:n_obs], dtype=int)

    while len(active) > 1:
        sub = mat[:, active]
        n_obs, n_models = sub.shape
        dbar = np.zeros((n_models, n_models), dtype=float)
        for i in range(n_models):
            for j in range(n_models):
                dbar[i, j] = float(np.mean(sub[:, i] - sub[:, j]))

        boot = np.zeros((bootstrap_samples, n_models, n_models), dtype=float)
        for b in range(bootstrap_samples):
            idx = _block_bootstrap_indices(n_obs)
            sample = sub[idx, :]
            for i in range(n_models):
                for j in range(n_models):
                    boot[b, i, j] = float(np.mean(sample[:, i] - sample[:, j]))
        var = np.var(boot, axis=0, ddof=1)
        var[var <= 1e-12] = np.nan

        tmat = np.abs(dbar / np.sqrt(var))
        if np.isnan(tmat).all():
            break
        tr_stat = float(np.nanmax(tmat))
        boot_centered = boot - dbar[None, :, :]
        tboot = np.abs(boot_centered / np.sqrt(var)[None, :, :])
        tr_boot = np.array(
            [
                (float(np.nanmax(tb)) if not np.isnan(tb).all() else float("nan"))
                for tb in tboot
            ],
            dtype=float,
        )
        tr_boot = tr_boot[np.isfinite(tr_boot)]
        if len(tr_boot) == 0:
            break
        pvalue = float(np.mean(tr_boot >= tr_stat))
        if pvalue >= alpha or not np.isfinite(pvalue):
            break
        losses_mean = np.mean(sub, axis=0)
        worst_local = int(np.argmax(losses_mean))
        active.pop(worst_local)

    selected = [configs_order[i] for i in active]
    return pd.DataFrame(
        {
            "config_label": configs_order,
            "selected_in_mcs_alpha_0_05": [c in selected for c in configs_order],
            "mean_loss": [float(mean_loss[c]) for c in configs_order],
        }
    )


def _apply_holm_adjustment_for_dm(dm_results: pd.DataFrame) -> pd.DataFrame:
    if dm_results.empty or "pvalue_two_sided" not in dm_results.columns:
        return dm_results

    out = dm_results.copy()
    out["pvalue_two_sided"] = pd.to_numeric(out["pvalue_two_sided"], errors="coerce")
    out["pvalue_adj_holm"] = np.nan

    group_cols = [c for c in ["asset", "parent_sweep_id", "split", "horizon"] if c in out.columns]
    if group_cols:
        groups = out.groupby(group_cols, dropna=False)
    else:
        groups = [((), out)]

    for _, g in groups:
        pv = pd.to_numeric(g["pvalue_two_sided"], errors="coerce")
        valid = pv.dropna()
        m = int(len(valid))
        if m == 0:
            continue
        ordered = valid.sort_values()
        adj_vals = []
        for j, (_, pval) in enumerate(ordered.items(), start=1):
            adj_vals.append((m - j + 1) * float(pval))
        adj_vals = np.minimum(1.0, np.maximum.accumulate(adj_vals))
        for (row_idx, _), adj in zip(ordered.items(), adj_vals):
            out.at[row_idx, "pvalue_adj_holm"] = float(adj)

    out["significant_adj_0_05"] = (
        pd.to_numeric(out["pvalue_adj_holm"], errors="coerce") < 0.05
    )
    return out


def _pairwise_preprocess(
    dim_run: pd.DataFrame,
    fact_oos_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Shared preprocessing for DM/MCS/WinRate builders.

    Returns the test-split, status='ok', dropna-cleaned DataFrame with
    `config_label` and `squared_error` columns. Empty DataFrame if any
    required column is missing or the data is otherwise unusable.
    """
    if fact_oos_predictions.empty or dim_run.empty:
        return pd.DataFrame()
    required = ["run_id", "split", "horizon", "target_timestamp_utc", "y_true", "y_pred"]
    missing = [c for c in required if c not in fact_oos_predictions.columns]
    if missing:
        return pd.DataFrame()

    keep = [
        c
        for c in [
            "run_id",
            "asset",
            "feature_set_name",
            "config_signature",
            "parent_sweep_id",
            "split_signature",
            "split_fingerprint",
            "status",
        ]
        if c in dim_run.columns
    ]
    if "run_id" not in keep:
        return pd.DataFrame()
    df = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
    df = _ensure_split_signature_column(df)
    if "config_signature" not in df.columns:
        if "config_signature_x" in df.columns:
            df["config_signature"] = df["config_signature_x"]
        elif "config_signature_y" in df.columns:
            df["config_signature"] = df["config_signature_y"]
    if "feature_set_name" not in df.columns:
        if "feature_set_name_x" in df.columns:
            df["feature_set_name"] = df["feature_set_name_x"]
        elif "feature_set_name_y" in df.columns:
            df["feature_set_name"] = df["feature_set_name_y"]
    if "asset" not in df.columns:
        if "asset_x" in df.columns:
            df["asset"] = df["asset_x"]
        elif "asset_y" in df.columns:
            df["asset"] = df["asset_y"]
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    df = df[df["split"].astype(str) == "test"].copy()
    if df.empty:
        return pd.DataFrame()
    for c in ["horizon", "y_true", "y_pred"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["target_timestamp_utc"] = pd.to_datetime(df["target_timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["horizon", "target_timestamp_utc", "y_true", "y_pred", "config_signature"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["horizon"] = df["horizon"].astype(int)
    df["config_label"] = df["feature_set_name"].astype(str) + "|" + df["config_signature"].astype(str)
    df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2
    return df


class DmPairwiseResultsGoldBuilder(GoldBuilder):
    output_table = "gold_dm_pairwise_results"
    requires = ("dim_run", "fact_oos_predictions")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        df = _pairwise_preprocess(
            snapshot.get("dim_run"),
            snapshot.get("fact_oos_predictions"),
        )
        if df.empty:
            return pd.DataFrame()

        rows: list[pd.DataFrame] = []
        group_cols = _pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            g = _select_top_configs_for_pairwise(g, max_configs=50)
            by_ts = (
                g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                .mean()
                .reset_index()
            )
            loss_matrix = by_ts.pivot(
                index="target_timestamp_utc", columns="config_label", values="squared_error"
            )
            loss_matrix = loss_matrix.dropna(axis=0, how="any")
            if loss_matrix.empty or loss_matrix.shape[1] < 2:
                continue
            dm_df = _compute_dm_pairwise_from_loss_matrix(loss_matrix)
            if dm_df.empty:
                continue
            dm_df["asset"] = keys[0]
            dm_df["parent_sweep_id"] = keys[1]
            if "split_signature" in group_cols:
                dm_df["split_signature"] = keys[2]
                dm_df["split"] = keys[3]
                dm_df["horizon"] = int(keys[4]) if pd.notna(keys[4]) else None
            else:
                dm_df["split"] = keys[2]
                dm_df["horizon"] = int(keys[3]) if pd.notna(keys[3]) else None
            dm_df["aligned_timestamps"] = int(loss_matrix.shape[0])
            dm_df["n_configs"] = int(loss_matrix.shape[1])
            rows.append(dm_df)

        if not rows:
            return pd.DataFrame()
        # Holm correction applied here (monolith applied it in execute()
        # right after building DM; we keep the same per-group adjustment
        # so the persisted gold parquet already includes pvalue_adj_holm
        # and significant_adj_0_05).
        return _apply_holm_adjustment_for_dm(pd.concat(rows, ignore_index=True))


class McsResultsGoldBuilder(GoldBuilder):
    output_table = "gold_mcs_results"
    requires = ("dim_run", "fact_oos_predictions")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        df = _pairwise_preprocess(
            snapshot.get("dim_run"),
            snapshot.get("fact_oos_predictions"),
        )
        if df.empty:
            return pd.DataFrame()

        rows: list[pd.DataFrame] = []
        group_cols = _pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            g = _select_top_configs_for_pairwise(g, max_configs=50)
            by_ts = (
                g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                .mean()
                .reset_index()
            )
            loss_matrix = by_ts.pivot(
                index="target_timestamp_utc", columns="config_label", values="squared_error"
            )
            loss_matrix = loss_matrix.dropna(axis=0, how="any")
            if loss_matrix.empty or loss_matrix.shape[1] < 2:
                continue
            mcs_df = _compute_mcs_from_loss_matrix(loss_matrix)
            if mcs_df.empty:
                continue
            mcs_df["asset"] = keys[0]
            mcs_df["parent_sweep_id"] = keys[1]
            if "split_signature" in group_cols:
                mcs_df["split_signature"] = keys[2]
                mcs_df["split"] = keys[3]
                mcs_df["horizon"] = int(keys[4]) if pd.notna(keys[4]) else None
            else:
                mcs_df["split"] = keys[2]
                mcs_df["horizon"] = int(keys[3]) if pd.notna(keys[3]) else None
            mcs_df["aligned_timestamps"] = int(loss_matrix.shape[0])
            mcs_df["n_configs"] = int(loss_matrix.shape[1])
            rows.append(mcs_df)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


class WinRatePairwiseResultsGoldBuilder(GoldBuilder):
    output_table = "gold_win_rate_pairwise_results"
    requires = ("dim_run", "fact_oos_predictions")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        df = _pairwise_preprocess(
            snapshot.get("dim_run"),
            snapshot.get("fact_oos_predictions"),
        )
        if df.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        group_cols = _pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            g = _select_top_configs_for_pairwise(g, max_configs=50)
            by_ts = (
                g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                .mean()
                .reset_index()
            )
            loss_matrix = by_ts.pivot(
                index="target_timestamp_utc", columns="config_label", values="squared_error"
            )
            loss_matrix = loss_matrix.dropna(axis=0, how="any")
            if loss_matrix.empty or loss_matrix.shape[1] < 2:
                continue
            configs = loss_matrix.columns.tolist()
            for i, left in enumerate(configs):
                for right in configs[i + 1 :]:
                    comp = pd.DataFrame({"l": loss_matrix[left], "r": loss_matrix[right]}).dropna()
                    n = int(len(comp))
                    if n == 0:
                        continue
                    left_wins = int((comp["l"] < comp["r"]).sum())
                    right_wins = int((comp["r"] < comp["l"]).sum())
                    ties = int(n - left_wins - right_wins)
                    non_ties = max(1, left_wins + right_wins)
                    rows.append(
                        {
                            "asset": keys[0],
                            "parent_sweep_id": keys[1],
                            "split_signature": keys[2] if "split_signature" in group_cols else None,
                            "split": keys[3] if "split_signature" in group_cols else keys[2],
                            "horizon": int(keys[4])
                            if "split_signature" in group_cols and pd.notna(keys[4])
                            else (int(keys[3]) if pd.notna(keys[3]) else None),
                            "left_config": str(left),
                            "right_config": str(right),
                            "aligned_timestamps": n,
                            "left_wins": left_wins,
                            "right_wins": right_wins,
                            "ties": ties,
                            "left_win_rate": float(left_wins / n),
                            "right_win_rate": float(right_wins / n),
                            "left_win_rate_ex_ties": float(left_wins / non_ties),
                            "right_win_rate_ex_ties": float(right_wins / non_ties),
                            "left_mean_loss": float(comp["l"].mean()),
                            "right_mean_loss": float(comp["r"].mean()),
                            "left_minus_right_mean_loss": float(
                                comp["l"].mean() - comp["r"].mean()
                            ),
                        }
                    )
        return pd.DataFrame(rows)


class PairedOosIntersectionByHorizonGoldBuilder(GoldBuilder):
    output_table = "gold_paired_oos_intersection_by_horizon"
    requires = ("dim_run", "fact_oos_predictions")

    def __init__(self, *, target_horizons: tuple[int, ...] = (1, 7, 30)) -> None:
        self._target_horizons = tuple(int(h) for h in target_horizons)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        fact_oos_predictions = snapshot.get("fact_oos_predictions")
        if fact_oos_predictions.empty or dim_run.empty:
            return pd.DataFrame()

        required = ["run_id", "split", "horizon", "target_timestamp_utc"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        keep = [
            c
            for c in [
                "run_id",
                "asset",
                "feature_set_name",
                "config_signature",
                "parent_sweep_id",
                "split_signature",
                "split_fingerprint",
                "status",
            ]
            if c in dim_run.columns
        ]
        if "run_id" not in keep:
            return pd.DataFrame()

        df = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
        df = _ensure_split_signature_column(df)
        if "config_signature" not in df.columns:
            if "config_signature_x" in df.columns:
                df["config_signature"] = df["config_signature_x"]
            elif "config_signature_y" in df.columns:
                df["config_signature"] = df["config_signature_y"]
        if "asset" not in df.columns:
            if "asset_x" in df.columns:
                df["asset"] = df["asset_x"]
            elif "asset_y" in df.columns:
                df["asset"] = df["asset_y"]
        if "feature_set_name" not in df.columns:
            if "feature_set_name_x" in df.columns:
                df["feature_set_name"] = df["feature_set_name_x"]
            elif "feature_set_name_y" in df.columns:
                df["feature_set_name"] = df["feature_set_name_y"]

        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower() == "ok"].copy()
        if df.empty:
            return pd.DataFrame()

        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["target_timestamp_utc"] = pd.to_datetime(df["target_timestamp_utc"], utc=True, errors="coerce")
        df = df.dropna(subset=["horizon", "target_timestamp_utc", "config_signature"]).copy()
        if df.empty:
            return pd.DataFrame()

        df["horizon"] = df["horizon"].astype(int)
        df = df[df["horizon"].isin(list(self._target_horizons))].copy()
        df = df[df["split"].astype(str) == "test"].copy()
        if df.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        group_cols = _pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            per_cfg: dict[str, set[str]] = {}
            for cfg, gc in g.groupby("config_signature", dropna=False):
                per_cfg[str(cfg)] = set(gc["target_timestamp_utc"].dropna().astype(str).tolist())
            if not per_cfg:
                continue
            sets = list(per_cfg.values())
            union = set.union(*sets) if sets else set()
            intersection = set.intersection(*sets) if sets else set()
            min_count = min((len(s) for s in sets), default=0)
            max_count = max((len(s) for s in sets), default=0)
            exact = all(s == sets[0] for s in sets[1:]) if len(sets) > 1 else True
            rows.append(
                {
                    "asset": keys[0],
                    "parent_sweep_id": keys[1],
                    "split_signature": keys[2] if "split_signature" in group_cols else None,
                    "split": keys[3] if "split_signature" in group_cols else keys[2],
                    "horizon": int(keys[4])
                    if "split_signature" in group_cols and pd.notna(keys[4])
                    else (int(keys[3]) if pd.notna(keys[3]) else None),
                    "n_configs": int(len(per_cfg)),
                    "n_union": int(len(union)),
                    "n_common": int(len(intersection)),
                    "coverage_ratio": float(len(intersection) / len(union)) if len(union) > 0 else 1.0,
                    "aligned_exact": bool(exact),
                    "target_union_count": int(len(union)),
                    "target_intersection_count": int(len(intersection)),
                    "target_min_count_per_config": int(min_count),
                    "target_max_count_per_config": int(max_count),
                    "target_exact_alignment": bool(exact),
                    "target_jaccard_alignment": float(len(intersection) / len(union))
                    if len(union) > 0
                    else 1.0,
                    "pairwise_ready_dm": bool(len(per_cfg) >= 2 and len(intersection) >= 5),
                    "pairwise_ready_mcs": bool(len(per_cfg) >= 2 and len(intersection) >= 1),
                }
            )

        return pd.DataFrame(rows)
