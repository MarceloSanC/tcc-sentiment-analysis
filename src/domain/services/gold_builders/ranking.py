"""Ranking-cluster gold builders.

3 Tier 1 builders sharing the same `_base_join_runs_split_metrics`
input shape: `dim_run` left-joined onto `fact_split_metrics`. Migrated
verbatim from `RefreshAnalyticsStoreUseCase._build_gold_*` (R-22.1)
preserving byte-identical column ordering, aggregation semantics, and
sort keys per ADR-0005.

The transformation per builder lives in a module-level helper
(`_build_<name>_from_base(base)`); the `GoldBuilder` subclasses
materialize `base` from the snapshot and delegate. The helpers are
internal to this module — they previously enabled a test-only bridge
eliminated 2026-05-22 in PR #56 but are kept module-level for
readability of the join+aggregate pattern shared across the 3 ranking
builders.
"""
from __future__ import annotations

import pandas as pd

from src.domain.services.gold_builders.base import (
    BuildContext,
    GoldBuilder,
    GoldBuilderSnapshot,
    normalize_parent_sweep_id_for_merge,
)

_REQUIRES = ("dim_run", "fact_split_metrics")


def _base_join_runs_split_metrics(
    dim_run: pd.DataFrame, fact_split_metrics: pd.DataFrame
) -> pd.DataFrame:
    """`dim_run` + `fact_split_metrics` join shared by ranking-cluster
    builders. Kept identical to the monolith helper so the modular
    builders see the same `base` shape (columns, ordering) that the
    god-object pre-R-22 produced.
    """
    if dim_run.empty or fact_split_metrics.empty:
        return pd.DataFrame()
    cols = [
        "run_id",
        "asset",
        "feature_set_name",
        "feature_set_hash",
        "config_signature",
        "model_version",
        "parent_sweep_id",
        "trial_number",
        "fold",
        "seed",
        "status",
        "created_at_utc",
    ]
    keep = [c for c in cols if c in dim_run.columns]
    dim_trim = dim_run[keep].copy()
    out = fact_split_metrics.merge(dim_trim, on="run_id", how="left")
    if "asset_x" in out.columns and "asset" not in out.columns:
        out["asset"] = out["asset_x"]
    if "asset_y" in out.columns and "asset" not in out.columns:
        out["asset"] = out["asset_y"]
    if "parent_sweep_id_x" in out.columns and "parent_sweep_id" not in out.columns:
        out["parent_sweep_id"] = out["parent_sweep_id_x"]
    if "parent_sweep_id_y" in out.columns:
        if "parent_sweep_id" not in out.columns:
            out["parent_sweep_id"] = out["parent_sweep_id_y"]
        else:
            out["parent_sweep_id"] = out["parent_sweep_id"].where(
                out["parent_sweep_id"].notna(),
                out["parent_sweep_id_y"],
            )
    out = normalize_parent_sweep_id_for_merge(out)
    return out


def _load_base(snapshot: GoldBuilderSnapshot) -> pd.DataFrame:
    return _base_join_runs_split_metrics(
        snapshot.get("dim_run"),
        snapshot.get("fact_split_metrics"),
    )


def _build_runs_long_from_base(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    cols = [
        "run_id",
        "asset",
        "feature_set_name",
        "feature_set_hash",
        "config_signature",
        "model_version",
        "parent_sweep_id",
        "trial_number",
        "fold",
        "seed",
        "split",
        "rmse",
        "mae",
        "mape",
        "smape",
        "directional_accuracy",
        "n_samples",
        "status",
        "created_at_utc",
    ]
    keep = [c for c in cols if c in base.columns]
    return base[keep].copy()


def _build_ranking_by_config_from_base(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    df = base[base["split"] == "test"].copy()
    if df.empty:
        return pd.DataFrame()
    if "parent_sweep_id" not in df.columns:
        if "parent_sweep_id_y" in df.columns:
            df["parent_sweep_id"] = df["parent_sweep_id_y"]
            if "parent_sweep_id_x" in df.columns:
                df["parent_sweep_id"] = df["parent_sweep_id"].combine_first(df["parent_sweep_id_x"])
        elif "parent_sweep_id_x" in df.columns:
            df["parent_sweep_id"] = df["parent_sweep_id_x"]
        else:
            df["parent_sweep_id"] = None
    df = normalize_parent_sweep_id_for_merge(df)
    group_cols = ["asset", "feature_set_name", "parent_sweep_id", "config_signature"]
    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            mean_test_rmse=("rmse", "mean"),
            std_test_rmse=("rmse", "std"),
            mean_test_mae=("mae", "mean"),
            std_test_mae=("mae", "std"),
            mean_test_da=("directional_accuracy", "mean"),
            std_test_da=("directional_accuracy", "std"),
        )
        .reset_index()
    )
    ranking_group_cols = ["asset", "feature_set_name", "parent_sweep_id"]
    agg["rank_test_rmse"] = agg.groupby(ranking_group_cols, dropna=False)["mean_test_rmse"].rank(
        method="min", ascending=True
    )
    agg["rank_test_mae"] = agg.groupby(ranking_group_cols, dropna=False)["mean_test_mae"].rank(
        method="min", ascending=True
    )
    agg["rank_test_da"] = agg.groupby(ranking_group_cols, dropna=False)["mean_test_da"].rank(
        method="min", ascending=False
    )
    return agg.sort_values(
        ["asset", "feature_set_name", "parent_sweep_id", "rank_test_rmse"]
    ).reset_index(drop=True)


def _build_consistency_topk_from_base(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    df = base[base["split"] == "test"].copy()
    if df.empty or "fold" not in df.columns or "seed" not in df.columns:
        return pd.DataFrame()

    keys = ["asset", "feature_set_name", "parent_sweep_id", "fold", "seed"]
    agg_keys = ["asset", "feature_set_name", "parent_sweep_id", "config_signature"]
    ranked = df.sort_values(keys + ["rmse"]).copy()
    ranked["position"] = ranked.groupby(keys).cumcount() + 1

    all_counts = (
        ranked.groupby(agg_keys, dropna=False)
        .size()
        .rename("total_groups")
        .reset_index()
    )
    top1 = (
        ranked[ranked["position"] <= 1]
        .groupby(agg_keys, dropna=False)
        .size()
        .rename("top1_hits")
        .reset_index()
    )
    top3 = (
        ranked[ranked["position"] <= 3]
        .groupby(agg_keys, dropna=False)
        .size()
        .rename("top3_hits")
        .reset_index()
    )
    top5 = (
        ranked[ranked["position"] <= 5]
        .groupby(agg_keys, dropna=False)
        .size()
        .rename("top5_hits")
        .reset_index()
    )
    out = all_counts.merge(top1, on=agg_keys, how="left")
    out = out.merge(top3, on=agg_keys, how="left")
    out = out.merge(top5, on=agg_keys, how="left")
    for c in ["top1_hits", "top3_hits", "top5_hits"]:
        out[c] = out[c].fillna(0).astype(int)
    out["top1_pct"] = out["top1_hits"] / out["total_groups"]
    out["top3_pct"] = out["top3_hits"] / out["total_groups"]
    out["top5_pct"] = out["top5_hits"] / out["total_groups"]
    return out.sort_values(
        ["asset", "feature_set_name", "parent_sweep_id", "top1_pct"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


class RunsLongGoldBuilder(GoldBuilder):
    output_table = "gold_runs_long"
    requires = _REQUIRES

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return _build_runs_long_from_base(_load_base(snapshot))


class RankingByConfigGoldBuilder(GoldBuilder):
    output_table = "gold_ranking_by_config"
    requires = _REQUIRES

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return _build_ranking_by_config_from_base(_load_base(snapshot))


class ConsistencyTopkGoldBuilder(GoldBuilder):
    output_table = "gold_consistency_topk"
    requires = _REQUIRES

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return _build_consistency_topk_from_base(_load_base(snapshot))
