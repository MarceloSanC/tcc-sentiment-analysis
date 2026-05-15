from __future__ import annotations

import math

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.path_policy import to_project_relative


@dataclass(frozen=True)
class RefreshAnalyticsStoreResult:
    gold_dir: str
    outputs: dict[str, str]


class RefreshAnalyticsStoreUseCase:
    def __init__(self, *, analytics_silver_dir: str | Path, analytics_gold_dir: str | Path) -> None:
        self.analytics_silver_dir = Path(analytics_silver_dir)
        self.analytics_gold_dir = Path(analytics_gold_dir)
        self.analytics_gold_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_partitioned_table(base_dir: Path, table_name: str) -> pd.DataFrame:
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
    def _safe_write(df: pd.DataFrame, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return str(to_project_relative(path))

    @staticmethod
    def _normalize_parent_sweep_id_for_merge(df: pd.DataFrame, column: str = "parent_sweep_id") -> pd.DataFrame:
        if df.empty or column not in df.columns:
            return df

        out = df.copy()

        def _norm(value: object) -> object:
            if pd.isna(value):
                return None
            text = str(value).strip()
            if not text or text.lower() in {"nan", "none", "null", "<na>"}:
                return None
            if text.endswith(".0") and text[:-2].isdigit():
                return text[:-2]
            return text

        out[column] = out[column].map(_norm).astype("object")
        return out

    @staticmethod
    def _pairwise_group_cols(df: pd.DataFrame) -> list[str]:
        cols = ["asset", "parent_sweep_id", "split", "horizon"]
        if "split_signature" in df.columns:
            cols.insert(2, "split_signature")
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _ensure_split_signature_column(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "split_signature" not in out.columns and "split_fingerprint" in out.columns:
            out["split_signature"] = out["split_fingerprint"]
        return out

    @staticmethod
    def _base_join_runs_split_metrics(dim_run: pd.DataFrame, fact_split_metrics: pd.DataFrame) -> pd.DataFrame:
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
        return out

    @staticmethod
    def _build_gold_runs_long(base: pd.DataFrame) -> pd.DataFrame:
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

    @staticmethod
    def _build_gold_ranking_by_config(base: pd.DataFrame) -> pd.DataFrame:
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
        df = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(df)
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
        return agg.sort_values(["asset", "feature_set_name", "parent_sweep_id", "rank_test_rmse"]).reset_index(drop=True)

    @staticmethod
    def _build_gold_consistency_topk(base: pd.DataFrame) -> pd.DataFrame:
        if base.empty:
            return pd.DataFrame()
        df = base[base["split"] == "test"].copy()
        if df.empty or "fold" not in df.columns or "seed" not in df.columns:
            return pd.DataFrame()

        keys = ["asset", "feature_set_name", "fold", "seed"]
        ranked = df.sort_values(keys + ["rmse"]).copy()
        ranked["position"] = ranked.groupby(keys).cumcount() + 1

        all_counts = (
            ranked.groupby(["asset", "feature_set_name", "config_signature"], dropna=False)
            .size()
            .rename("total_groups")
            .reset_index()
        )
        top1 = (
            ranked[ranked["position"] <= 1]
            .groupby(["asset", "feature_set_name", "config_signature"], dropna=False)
            .size()
            .rename("top1_hits")
            .reset_index()
        )
        top3 = (
            ranked[ranked["position"] <= 3]
            .groupby(["asset", "feature_set_name", "config_signature"], dropna=False)
            .size()
            .rename("top3_hits")
            .reset_index()
        )
        top5 = (
            ranked[ranked["position"] <= 5]
            .groupby(["asset", "feature_set_name", "config_signature"], dropna=False)
            .size()
            .rename("top5_hits")
            .reset_index()
        )
        out = all_counts.merge(top1, on=["asset", "feature_set_name", "config_signature"], how="left")
        out = out.merge(top3, on=["asset", "feature_set_name", "config_signature"], how="left")
        out = out.merge(top5, on=["asset", "feature_set_name", "config_signature"], how="left")
        for c in ["top1_hits", "top3_hits", "top5_hits"]:
            out[c] = out[c].fillna(0).astype(int)
        out["top1_pct"] = out["top1_hits"] / out["total_groups"]
        out["top3_pct"] = out["top3_hits"] / out["total_groups"]
        out["top5_pct"] = out["top5_hits"] / out["total_groups"]
        return out.sort_values(["asset", "feature_set_name", "top1_pct"], ascending=[True, True, False]).reset_index(drop=True)

    @staticmethod
    def _build_gold_ic95(base: pd.DataFrame) -> pd.DataFrame:
        if base.empty:
            return pd.DataFrame()
        df = base[base["split"] == "test"].copy()
        if df.empty:
            return pd.DataFrame()
        metrics = [("rmse", "test_rmse"), ("mae", "test_mae"), ("directional_accuracy", "test_da")]
        rows: list[dict[str, object]] = []
        for metric_col, metric_name in metrics:
            grouped = (
                df.groupby(["asset", "feature_set_name", "config_signature"], dropna=False)[metric_col]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            grouped["std"] = grouped["std"].fillna(0.0)
            grouped["se"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
            grouped["ci95_low"] = grouped["mean"] - (1.96 * grouped["se"])
            grouped["ci95_high"] = grouped["mean"] + (1.96 * grouped["se"])
            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "asset": row["asset"],
                        "feature_set_name": row["feature_set_name"],
                        "config_signature": row["config_signature"],
                        "metric": metric_name,
                        "n": int(row["count"]),
                        "mean": float(row["mean"]),
                        "std": float(row["std"]),
                        "se": float(row["se"]),
                        "ci95_low": float(row["ci95_low"]),
                        "ci95_high": float(row["ci95_high"]),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_gold_feature_set_impact(base: pd.DataFrame) -> pd.DataFrame:
        if base.empty:
            return pd.DataFrame()
        df = base.copy()
        metric_cols = ["rmse", "mae", "directional_accuracy"]
        out_rows: list[dict[str, object]] = []
        group_cols = ["asset", "feature_set_name", "parent_sweep_id", "split"]
        for metric_col in metric_cols:
            grouped = (
                df.groupby(group_cols, dropna=False)[metric_col]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            for _, row in grouped.iterrows():
                out_rows.append(
                    {
                        "asset": row["asset"],
                        "feature_set_name": row["feature_set_name"],
                        "parent_sweep_id": row["parent_sweep_id"],
                        "split": row["split"],
                        "metric": metric_col,
                        "n_runs": int(row["count"]),
                        "mean_value": float(row["mean"]),
                        "std_value": float(row["std"]) if pd.notna(row["std"]) else 0.0,
                    }
                )
        return pd.DataFrame(out_rows)



    @staticmethod
    def _pinball_loss(y_true: pd.Series, y_pred_q: pd.Series, quantile: float) -> pd.Series:
        q = float(quantile)
        diff = y_true - y_pred_q
        return np.maximum(q * diff, (q - 1.0) * diff)

    @staticmethod
    def _prob_up_from_quantiles(q10: pd.Series, q50: pd.Series, q90: pd.Series) -> pd.Series:
        # Piecewise-linear approximation of CDF using q10/q90 anchors:
        # CDF(q10)=0.1 and CDF(q90)=0.9.
        width = (q90 - q10).astype(float)
        safe_width = width.where(width.abs() > 1e-12, np.nan)

        cdf0 = 0.1 + 0.8 * ((0.0 - q10) / safe_width)
        cdf0 = cdf0.clip(lower=0.1, upper=0.9)

        # Hard bounds when interval fully above/below zero.
        cdf0 = np.where(q10 > 0.0, 0.0, cdf0)
        cdf0 = np.where(q90 < 0.0, 1.0, cdf0)

        # Fallback for zero-width intervals.
        fallback = np.where(q50 > 0.0, 1.0, np.where(q50 < 0.0, 0.0, 0.5))
        cdf0 = np.where(np.isnan(cdf0), fallback, cdf0)

        return pd.Series(1.0 - cdf0, index=q10.index, dtype='float64')

    @staticmethod
    def _build_gold_prediction_metrics_by_run_split_horizon(
        dim_run: pd.DataFrame,
        fact_oos_predictions: pd.DataFrame,
        *,
        quantile_columns: tuple[str, str, str] = ("quantile_p10", "quantile_p50", "quantile_p90"),
    ) -> pd.DataFrame:
        if fact_oos_predictions.empty:
            return pd.DataFrame()

        q10_col, q50_col, q90_col = quantile_columns
        df = fact_oos_predictions.copy()

        required = [
            'run_id',
            'split',
            'horizon',
            'y_true',
            'y_pred',
            q10_col,
            q50_col,
            q90_col,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return pd.DataFrame()

        for c in ['horizon', 'y_true', 'y_pred', q10_col, q50_col, q90_col]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        valid = df.dropna(subset=['horizon', 'y_true', 'y_pred', q10_col, q50_col, q90_col]).copy()
        if valid.empty:
            return pd.DataFrame()

        valid['horizon'] = valid['horizon'].astype(int)
        valid['error'] = valid['y_pred'] - valid['y_true']
        valid['abs_error'] = valid['error'].abs()
        valid['sq_error'] = valid['error'] ** 2
        valid['pred_interval_width'] = valid[q90_col] - valid[q10_col]
        valid['covered_80'] = (
            (valid['y_true'] >= valid[q10_col]) &
            (valid['y_true'] <= valid[q90_col])
        ).astype(float)

        denom = valid['y_true'].replace(0.0, np.nan)
        valid['ape'] = (valid['abs_error'] / denom.abs())
        smape_denom = valid['y_true'].abs() + valid['y_pred'].abs()
        valid['smape_row'] = (2.0 * valid['abs_error'] / smape_denom.replace(0.0, np.nan))

        valid['da_row'] = (np.sign(valid['y_true']) == np.sign(valid['y_pred'])).astype(float)
        valid['pinball_q10_row'] = RefreshAnalyticsStoreUseCase._pinball_loss(valid['y_true'], valid[q10_col], 0.1)
        valid['pinball_q50_row'] = RefreshAnalyticsStoreUseCase._pinball_loss(valid['y_true'], valid[q50_col], 0.5)
        valid['pinball_q90_row'] = RefreshAnalyticsStoreUseCase._pinball_loss(valid['y_true'], valid[q90_col], 0.9)
        valid['pinball_mean_row'] = (
            valid['pinball_q10_row'] + valid['pinball_q50_row'] + valid['pinball_q90_row']
        ) / 3.0
        valid['prob_up_row'] = RefreshAnalyticsStoreUseCase._prob_up_from_quantiles(
            valid[q10_col], valid[q50_col], valid[q90_col]
        )

        group_cols = [
            c for c in [
                'run_id', 'asset', 'feature_set_name', 'config_signature',
                'split', 'fold', 'seed', 'horizon'
            ] if c in valid.columns
        ]

        agg = (
            valid.groupby(group_cols, dropna=False)
            .agg(
                n_samples=('y_true', 'count'),
                rmse=('sq_error', lambda x: float(np.sqrt(np.mean(x)))),
                mae=('abs_error', 'mean'),
                mape=('ape', 'mean'),
                smape=('smape_row', 'mean'),
                directional_accuracy=('da_row', 'mean'),
                bias=('error', 'mean'),
                pinball_q10=('pinball_q10_row', 'mean'),
                pinball_q50=('pinball_q50_row', 'mean'),
                pinball_q90=('pinball_q90_row', 'mean'),
                mean_pinball=('pinball_mean_row', 'mean'),
                picp=('covered_80', 'mean'),
                mpiw=('pred_interval_width', 'mean'),
                pred_interval_width=('pred_interval_width', 'mean'),
                prob_up=('prob_up_row', 'mean'),
            )
            .reset_index()
        )

        agg['coverage_nominal'] = 0.80
        agg['coverage_error'] = agg['picp'] - agg['coverage_nominal']
        agg['prob_down'] = 1.0 - agg['prob_up']
        calibration_term = (1.0 - (agg['coverage_error'].abs() / agg['coverage_nominal'])).clip(lower=0.0, upper=1.0)
        width_term = 1.0 / (1.0 + agg['pred_interval_width'].clip(lower=0.0))
        agg['confidence_calibrated'] = calibration_term * width_term

        if not dim_run.empty and 'run_id' in dim_run.columns:
            keep = [c for c in ['run_id', 'model_version', 'feature_set_hash', 'parent_sweep_id', 'trial_number', 'status'] if c in dim_run.columns]
            if keep:
                agg = agg.merge(dim_run[keep].drop_duplicates('run_id'), on='run_id', how='left')

        return agg

    @staticmethod
    def _build_gold_quantile_guardrail_audit(
        dim_run: pd.DataFrame,
        fact_oos_predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        required = {
            'run_id', 'split', 'horizon',
            'quantile_p10', 'quantile_p50', 'quantile_p90',
            'quantile_p10_post_guardrail', 'quantile_p50_post_guardrail', 'quantile_p90_post_guardrail',
        }
        if fact_oos_predictions.empty or not required.issubset(set(fact_oos_predictions.columns)):
            return pd.DataFrame()

        base_metrics = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
            dim_run,
            fact_oos_predictions,
            quantile_columns=('quantile_p10', 'quantile_p50', 'quantile_p90'),
        )
        post_metrics = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
            dim_run,
            fact_oos_predictions,
            quantile_columns=(
                'quantile_p10_post_guardrail',
                'quantile_p50_post_guardrail',
                'quantile_p90_post_guardrail',
            ),
        )
        if base_metrics.empty or post_metrics.empty:
            return pd.DataFrame()

        key_cols = [
            c for c in [
                'run_id', 'asset', 'feature_set_name', 'config_signature',
                'split', 'fold', 'seed', 'horizon',
                'model_version', 'feature_set_hash', 'parent_sweep_id', 'trial_number', 'status'
            ] if c in base_metrics.columns and c in post_metrics.columns
        ]
        metric_cols = [
            c for c in ['mean_pinball', 'picp', 'mpiw', 'pred_interval_width', 'coverage_error', 'confidence_calibrated']
            if c in base_metrics.columns and c in post_metrics.columns
        ]

        before = base_metrics[key_cols + metric_cols].copy().rename(
            columns={m: f"{m}_before" for m in metric_cols}
        )
        after = post_metrics[key_cols + metric_cols].copy().rename(
            columns={m: f"{m}_after" for m in metric_cols}
        )
        out = before.merge(after, on=key_cols, how='inner')
        if out.empty:
            return pd.DataFrame()

        for m in metric_cols:
            out[f"delta_{m}_after_minus_before"] = out[f"{m}_after"] - out[f"{m}_before"]

        qdf = fact_oos_predictions.copy()
        for c in [
            'horizon',
            'quantile_p10', 'quantile_p50', 'quantile_p90',
            'quantile_p10_post_guardrail', 'quantile_p50_post_guardrail', 'quantile_p90_post_guardrail',
            'quantile_guardrail_applied',
        ]:
            if c in qdf.columns:
                qdf[c] = pd.to_numeric(qdf[c], errors='coerce')

        group_cols = [c for c in ['run_id', 'split', 'horizon'] if c in qdf.columns]
        crossing = (
            qdf.assign(
                crossing_before=((qdf['quantile_p10'] > qdf['quantile_p50']) | (qdf['quantile_p50'] > qdf['quantile_p90'])).astype(float),
                crossing_after=((qdf['quantile_p10_post_guardrail'] > qdf['quantile_p50_post_guardrail']) | (qdf['quantile_p50_post_guardrail'] > qdf['quantile_p90_post_guardrail'])).astype(float),
                negative_width_before=((qdf['quantile_p90'] - qdf['quantile_p10']) < 0.0).astype(float),
                negative_width_after=((qdf['quantile_p90_post_guardrail'] - qdf['quantile_p10_post_guardrail']) < 0.0).astype(float),
                guardrail_applied=qdf.get('quantile_guardrail_applied', pd.Series(0.0, index=qdf.index)).fillna(0.0),
            )
            .groupby(group_cols, dropna=False)
            .agg(
                n_rows=('run_id', 'count'),
                crossing_before_count=('crossing_before', 'sum'),
                crossing_after_count=('crossing_after', 'sum'),
                negative_width_before_count=('negative_width_before', 'sum'),
                negative_width_after_count=('negative_width_after', 'sum'),
                guardrail_applied_count=('guardrail_applied', 'sum'),
            )
            .reset_index()
        )

        for c in [
            'crossing_before_count', 'crossing_after_count',
            'negative_width_before_count', 'negative_width_after_count',
            'guardrail_applied_count',
        ]:
            crossing[c] = pd.to_numeric(crossing[c], errors='coerce').fillna(0.0).astype(int)

        n_rows = pd.to_numeric(crossing['n_rows'], errors='coerce').replace(0, np.nan)
        crossing['crossing_before_rate'] = crossing['crossing_before_count'] / n_rows
        crossing['crossing_after_rate'] = crossing['crossing_after_count'] / n_rows
        crossing['negative_width_before_rate'] = crossing['negative_width_before_count'] / n_rows
        crossing['negative_width_after_rate'] = crossing['negative_width_after_count'] / n_rows
        crossing['guardrail_applied_rate'] = crossing['guardrail_applied_count'] / n_rows

        out = out.merge(crossing, on=['run_id', 'split', 'horizon'], how='left')
        return out

    @staticmethod
    def _build_gold_prediction_metrics_by_config(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        cols = ['asset', 'feature_set_name', 'config_signature', 'split', 'horizon']
        if not set(cols).issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        metric_cols = [
            'rmse', 'mae', 'mape', 'smape', 'directional_accuracy', 'bias',
            'pinball_q10', 'pinball_q50', 'pinball_q90', 'mean_pinball',
            'picp', 'mpiw', 'pred_interval_width', 'prob_up', 'prob_down',
            'coverage_error', 'confidence_calibrated'
        ]
        available_metric_cols = [c for c in metric_cols if c in metrics_run_split_h.columns]

        frame = metrics_run_split_h.copy()
        if 'n_samples' in frame.columns:
            frame['n_samples'] = pd.to_numeric(frame['n_samples'], errors='coerce').fillna(0)

        grouped = frame.groupby(cols, dropna=False)
        agg_spec: dict[str, tuple[str, str]] = {'n_runs': ('run_id', 'count')}
        if 'n_samples' in frame.columns:
            agg_spec['n_oos'] = ('n_samples', 'sum')
        out = grouped.agg(**agg_spec).reset_index()
        if 'n_oos' in out.columns:
            out['n_oos'] = pd.to_numeric(out['n_oos'], errors='coerce').fillna(0).astype(int)

        for m in available_metric_cols:
            g = grouped[m].agg(['mean', 'std']).reset_index().rename(columns={'mean': f'mean_{m}', 'std': f'std_{m}'})
            out = out.merge(g, on=cols, how='left')
            iqr = grouped[m].agg(lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))).reset_index().rename(columns={m: f'iqr_{m}'})
            out = out.merge(iqr, on=cols, how='left')

        return out

    @staticmethod
    def _build_gold_prediction_metrics_by_horizon(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        cols = ['asset', 'feature_set_name', 'parent_sweep_id', 'split', 'horizon']
        if not set(cols).issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        metric_cols = [
            'rmse', 'mae', 'mape', 'smape', 'directional_accuracy', 'bias',
            'pinball_q10', 'pinball_q50', 'pinball_q90', 'mean_pinball',
            'picp', 'mpiw', 'pred_interval_width', 'prob_up', 'prob_down',
            'coverage_error', 'confidence_calibrated'
        ]
        available_metric_cols = [c for c in metric_cols if c in metrics_run_split_h.columns]

        frame = metrics_run_split_h.copy()
        if 'n_samples' in frame.columns:
            frame['n_samples'] = pd.to_numeric(frame['n_samples'], errors='coerce').fillna(0)

        grouped = frame.groupby(cols, dropna=False)
        agg_spec: dict[str, tuple[str, str]] = {'n_runs': ('run_id', 'count')}
        if 'n_samples' in frame.columns:
            agg_spec['n_oos'] = ('n_samples', 'sum')
        out = grouped.agg(**agg_spec).reset_index()
        if 'n_oos' in out.columns:
            out['n_oos'] = pd.to_numeric(out['n_oos'], errors='coerce').fillna(0).astype(int)

        for m in available_metric_cols:
            g = grouped[m].agg(['mean', 'std']).reset_index().rename(columns={'mean': f'mean_{m}', 'std': f'std_{m}'})
            out = out.merge(g, on=cols, how='left')
            iqr = grouped[m].agg(lambda s: float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))).reset_index().rename(columns={m: f'iqr_{m}'})
            out = out.merge(iqr, on=cols, how='left')

        return out

    @staticmethod
    def _build_gold_prediction_calibration(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        cols = ['asset', 'feature_set_name', 'config_signature', 'split', 'horizon']
        if not set(cols).issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        keep = [
            c for c in [
                'run_id', *cols, 'n_samples', 'pinball_q10', 'pinball_q50', 'pinball_q90',
                'mean_pinball', 'picp', 'mpiw', 'pred_interval_width', 'coverage_nominal',
                'coverage_error', 'prob_up', 'prob_down', 'confidence_calibrated'
            ] if c in metrics_run_split_h.columns
        ]
        return metrics_run_split_h[keep].copy()



    @staticmethod
    def _build_gold_prediction_generalization_gap(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        req = {'asset', 'feature_set_name', 'config_signature', 'horizon', 'split'}
        if not req.issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        metric_cols = [
            c for c in [
                'rmse', 'mae', 'mape', 'smape', 'directional_accuracy', 'bias',
                'mean_pinball', 'picp', 'mpiw', 'pred_interval_width',
                'prob_up', 'prob_down', 'coverage_error', 'confidence_calibrated'
            ] if c in metrics_run_split_h.columns
        ]
        if not metric_cols:
            return pd.DataFrame()

        grouped = metrics_run_split_h.groupby(
            ['asset', 'feature_set_name', 'config_signature', 'horizon', 'split'],
            dropna=False,
        )

        means = grouped[metric_cols].mean().reset_index()
        test_df = means[means['split'].astype(str) == 'test'].drop(columns=['split'])
        val_df = means[means['split'].astype(str) == 'val'].drop(columns=['split'])
        if test_df.empty or val_df.empty:
            return pd.DataFrame()

        key_cols = ['asset', 'feature_set_name', 'config_signature', 'horizon']
        out = test_df.merge(val_df, on=key_cols, how='inner', suffixes=('_test', '_val'))
        if out.empty:
            return pd.DataFrame()

        for m in metric_cols:
            out[f'gap_{m}_test_minus_val'] = out[f'{m}_test'] - out[f'{m}_val']

        keep = key_cols + [c for c in out.columns if c.startswith('gap_')]
        return out[keep].copy()

    @staticmethod
    def _build_gold_prediction_robustness_by_horizon(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        req = {'asset', 'feature_set_name', 'config_signature', 'horizon', 'split'}
        if not req.issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        df = metrics_run_split_h[metrics_run_split_h['split'].astype(str) == 'test'].copy()
        if df.empty:
            return pd.DataFrame()

        metric_cols = [
            c for c in [
                'rmse', 'mae', 'mape', 'smape', 'directional_accuracy', 'bias',
                'mean_pinball', 'picp', 'mpiw', 'pred_interval_width',
                'prob_up', 'prob_down', 'coverage_error', 'confidence_calibrated'
            ] if c in df.columns
        ]
        if not metric_cols:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        gcols = ['asset', 'feature_set_name', 'config_signature', 'horizon']
        for keys, g in df.groupby(gcols, dropna=False):
            for m in metric_cols:
                series = pd.to_numeric(g[m], errors='coerce').dropna()
                if series.empty:
                    continue
                n = int(len(series))
                mean = float(series.mean())
                std = float(series.std(ddof=1)) if n > 1 else 0.0
                se = std / np.sqrt(max(1, n))
                rows.append(
                    {
                        'asset': keys[0],
                        'feature_set_name': keys[1],
                        'config_signature': keys[2],
                        'horizon': int(keys[3]) if pd.notna(keys[3]) else None,
                        'metric': m,
                        'n_runs': n,
                        'mean': mean,
                        'std': std,
                        'median': float(np.nanmedian(series)),
                        'iqr': float(np.nanpercentile(series, 75) - np.nanpercentile(series, 25)),
                        'ci95_low': float(mean - 1.96 * se),
                        'ci95_high': float(mean + 1.96 * se),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_gold_feature_impact_by_horizon(
        fact_model_artifacts: pd.DataFrame,
        metrics_run_split_h: pd.DataFrame,
    ) -> pd.DataFrame:
        if fact_model_artifacts.empty or metrics_run_split_h.empty:
            return pd.DataFrame()
        if not {'run_id', 'feature_importance_json'}.issubset(set(fact_model_artifacts.columns)):
            return pd.DataFrame()
        req = {'run_id', 'asset', 'feature_set_name', 'parent_sweep_id', 'split', 'horizon'}
        if not req.issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        import json

        split_h = metrics_run_split_h[
            ['run_id', 'asset', 'feature_set_name', 'parent_sweep_id', 'split', 'horizon']
        ].drop_duplicates()
        split_h = split_h[split_h['split'].astype(str).isin(['val', 'test'])].copy()
        if split_h.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for _, art in fact_model_artifacts.iterrows():
            run_id = art.get('run_id')
            raw = art.get('feature_importance_json')
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue
            horizons = split_h[split_h['run_id'] == run_id]
            if horizons.empty:
                continue
            for _, hz in horizons.iterrows():
                for item in parsed:
                    if not isinstance(item, dict) or 'feature' not in item:
                        continue
                    rows.append(
                        {
                            'run_id': run_id,
                            'asset': hz.get('asset'),
                            'feature_set_name': hz.get('feature_set_name'),
                            'parent_sweep_id': hz.get('parent_sweep_id'),
                            'split': hz.get('split'),
                            'horizon': int(hz.get('horizon')) if pd.notna(hz.get('horizon')) else None,
                            'feature_name': str(item.get('feature')),
                            'delta_rmse': pd.to_numeric(item.get('delta_rmse'), errors='coerce'),
                            'delta_mae': pd.to_numeric(item.get('delta_mae'), errors='coerce'),
                            'baseline_rmse': pd.to_numeric(item.get('baseline_rmse'), errors='coerce'),
                            'baseline_mae': pd.to_numeric(item.get('baseline_mae'), errors='coerce'),
                            'method': 'global_importance_reused_by_horizon',
                        }
                    )

        if not rows:
            return pd.DataFrame()
        detail = pd.DataFrame(rows)
        agg = (
            detail.groupby(
                ['asset', 'feature_set_name', 'parent_sweep_id', 'split', 'horizon', 'feature_name', 'method'],
                dropna=False,
            )
            .agg(
                n_runs=('run_id', 'count'),
                mean_delta_rmse=('delta_rmse', 'mean'),
                std_delta_rmse=('delta_rmse', 'std'),
                mean_delta_mae=('delta_mae', 'mean'),
                std_delta_mae=('delta_mae', 'std'),
            )
            .reset_index()
        )
        return agg

    @staticmethod
    def _build_gold_quality_run_sweep_summary(
        quality_report: pd.DataFrame,
        quality_statistics: pd.DataFrame,
        dim_run: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        if not quality_report.empty and {'scope', 'run_id', 'passed'}.issubset(set(quality_report.columns)):
            qr = quality_report[quality_report['scope'].astype(str) == 'run_split_horizon'].copy()
            if not qr.empty:
                run_ag = qr.groupby('run_id', dropna=False).agg(
                    quality_rows=('passed', 'count'),
                    quality_passed_rows=('passed', lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
                ).reset_index()
                run_ag['quality_passed_all'] = run_ag['quality_rows'] == run_ag['quality_passed_rows']
                if not dim_run.empty and 'run_id' in dim_run.columns:
                    keep = [c for c in ['run_id', 'asset', 'parent_sweep_id', 'feature_set_name', 'config_signature', 'model_version'] if c in dim_run.columns]
                    run_ag = run_ag.merge(dim_run[keep].drop_duplicates('run_id'), on='run_id', how='left')
                run_ag['scope'] = 'run'
                rows.extend(run_ag.to_dict(orient='records'))

        if not quality_statistics.empty and {'asset', 'parent_sweep_id'}.issubset(set(quality_statistics.columns)):
            qs = quality_statistics.copy()
            sweep_ag = qs.groupby(['asset', 'parent_sweep_id'], dropna=False).agg(
                group_rows=('statistics_ready', 'count'),
                statistics_ready_rows=('statistics_ready', lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
                dm_available_rows=('dm_available', lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
                mcs_available_rows=('mcs_available', lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
                win_rate_available_rows=('win_rate_available', lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            ).reset_index()
            sweep_ag['scope'] = 'sweep'
            rows.extend(sweep_ag.to_dict(orient='records'))

        return pd.DataFrame(rows)

    @staticmethod
    def _build_gold_prediction_risk(dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame) -> pd.DataFrame:
        if fact_oos_predictions.empty:
            return pd.DataFrame()

        required = ["run_id", "split", "horizon", "y_pred", "quantile_p10", "quantile_p50"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        df = fact_oos_predictions.copy()
        for c in ["horizon", "y_pred", "quantile_p10", "quantile_p50"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["horizon", "y_pred", "quantile_p10", "quantile_p50"]).copy()
        if df.empty:
            return pd.DataFrame()

        df["horizon"] = df["horizon"].astype(int)
        df["expected_move_row"] = df["y_pred"].abs()
        df["downside_risk_row"] = np.maximum(-df["y_pred"], 0.0)
        df["var_10_row"] = df["quantile_p10"]
        # ES_10 approximado via extrapolacao linear da funcao quantil entre p10 e p50:
        # ES_10 ~= 1.125*q10 - 0.125*q50
        df["es_10_approx_row"] = 1.125 * df["quantile_p10"] - 0.125 * df["quantile_p50"]
        df["es_10_approx_row"] = np.minimum(df["es_10_approx_row"], df["var_10_row"])

        group_cols = [
            c for c in [
                "run_id", "asset", "feature_set_name", "config_signature",
                "split", "fold", "seed", "horizon"
            ] if c in df.columns
        ]

        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_samples=("y_pred", "count"),
                expected_move=("expected_move_row", "mean"),
                downside_risk=("downside_risk_row", "mean"),
                var_10=("var_10_row", "mean"),
                es_10_approx=("es_10_approx_row", "mean"),
            )
            .reset_index()
        )

        if not dim_run.empty and "run_id" in dim_run.columns:
            keep = [
                c for c in [
                    "run_id", "model_version", "feature_set_hash", "parent_sweep_id", "trial_number", "status"
                ] if c in dim_run.columns
            ]
            if keep:
                out = out.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")

        return out


    @staticmethod
    def _build_gold_oos_consolidated(dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame) -> pd.DataFrame:
        if fact_oos_predictions.empty:
            return pd.DataFrame()
        if dim_run.empty:
            return fact_oos_predictions.copy()
        cols = [
            "run_id",
            "model_version",
            "feature_set_hash",
            "parent_sweep_id",
            "trial_number",
            "status",
        ]
        keep = [c for c in cols if c in dim_run.columns]
        dim_trim = dim_run[keep].copy()
        return fact_oos_predictions.merge(dim_trim, on="run_id", how="left")

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
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
                pvalue = 2.0 * (1.0 - RefreshAnalyticsStoreUseCase._norm_cdf(abs(stat)))
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

    @staticmethod
    def _select_top_configs_for_pairwise(
        grouped_oos: pd.DataFrame,
        *,
        max_configs: int = 50,
    ) -> pd.DataFrame:
        if grouped_oos.empty or "config_label" not in grouped_oos.columns or "squared_error" not in grouped_oos.columns:
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

    @staticmethod
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

    @staticmethod
    def _build_gold_oos_quality_report(dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame) -> pd.DataFrame:
        if fact_oos_predictions.empty:
            return pd.DataFrame()
        required = ["run_id", "split", "horizon", "timestamp_utc", "target_timestamp_utc", "y_true", "y_pred"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame([{"scope": "dataset", "status": "invalid", "detail": f"missing_columns={missing}"}])

        df = fact_oos_predictions.copy()
        for c in ["horizon", "y_true", "y_pred", "quantile_p10", "quantile_p90"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df["target_timestamp_utc"] = pd.to_datetime(df["target_timestamp_utc"], utc=True, errors="coerce")

        if not dim_run.empty and "run_id" in dim_run.columns:
            keep = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "split_signature", "split_fingerprint"] if c in dim_run.columns]
            df = df.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")

        key_cols = ["run_id", "split", "horizon", "timestamp_utc", "target_timestamp_utc"]
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        df["_dup"] = dup_mask.astype(int)
        df["_target_before_ts"] = (df["target_timestamp_utc"] < df["timestamp_utc"]).fillna(False).astype(int)
        df["_y_true_null"] = df["y_true"].isna().astype(int)
        df["_y_pred_null"] = df["y_pred"].isna().astype(int)
        if {"quantile_p10", "quantile_p90"}.issubset(set(df.columns)):
            width = df["quantile_p90"] - df["quantile_p10"]
            df["_pred_interval_negative"] = ((~width.isna()) & (width < 0.0)).astype(int)
        else:
            df["_pred_interval_negative"] = 0

        rows: list[dict[str, object]] = []
        group_cols = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "split", "horizon"] if c in df.columns]
        for _, g in df.groupby(group_cols, dropna=False):
            first = g.iloc[0]
            tgt = g["target_timestamp_utc"].dropna().sort_values()
            monotonic = bool(tgt.is_monotonic_increasing)
            rows.append(
                {
                    "scope": "run_split_horizon",
                    "run_id": first.get("run_id"),
                    "asset": first.get("asset"),
                    "feature_set_name": first.get("feature_set_name"),
                    "config_signature": first.get("config_signature"),
                    "parent_sweep_id": first.get("parent_sweep_id"),
                    "split": first.get("split"),
                    "horizon": int(first.get("horizon")) if pd.notna(first.get("horizon")) else None,
                    "n_rows": int(len(g)),
                    "n_duplicate_key_rows": int(g["_dup"].sum()),
                    "n_target_before_timestamp": int(g["_target_before_ts"].sum()),
                    "n_y_true_null": int(g["_y_true_null"].sum()),
                    "n_y_pred_null": int(g["_y_pred_null"].sum()),
                    "n_negative_interval_width": int(g["_pred_interval_negative"].sum()),
                    "target_timestamp_monotonic": monotonic,
                    "passed": bool(
                        int(g["_dup"].sum()) == 0
                        and int(g["_target_before_ts"].sum()) == 0
                        and int(g["_y_true_null"].sum()) == 0
                        and int(g["_y_pred_null"].sum()) == 0
                        and int(g["_pred_interval_negative"].sum()) == 0
                        and monotonic
                    ),
                }
            )

        align_rows: list[dict[str, object]] = []
        align_group_cols = RefreshAnalyticsStoreUseCase._pairwise_group_cols(df)
        if {"config_signature", "target_timestamp_utc"}.issubset(set(df.columns)) and align_group_cols:
            for keys, g in df.groupby(align_group_cols, dropna=False):
                per_cfg = {}
                for cfg, gc in g.groupby("config_signature", dropna=False):
                    per_cfg[str(cfg)] = set(gc["target_timestamp_utc"].dropna().astype(str).tolist())
                if len(per_cfg) < 2:
                    continue
                sets = list(per_cfg.values())
                intersection = set.intersection(*sets) if sets else set()
                union = set.union(*sets) if sets else set()
                min_count = min((len(s) for s in sets), default=0)
                max_count = max((len(s) for s in sets), default=0)
                exact = all(s == sets[0] for s in sets[1:])
                has_sig = "split_signature" in align_group_cols
                row = {
                    "scope": "alignment_group",
                    "asset": keys[0] if len(keys) > 0 else None,
                    "parent_sweep_id": keys[1] if len(keys) > 1 else None,
                    "split_signature": keys[2] if has_sig and len(keys) > 2 else None,
                    "split": keys[3] if has_sig and len(keys) > 3 else (keys[2] if len(keys) > 2 else None),
                    "horizon": int(keys[4]) if has_sig and len(keys) > 4 and pd.notna(keys[4]) else (int(keys[3]) if len(keys) > 3 and pd.notna(keys[3]) else None),
                    "n_configs": int(len(per_cfg)),
                    "target_intersection_count": int(len(intersection)),
                    "target_union_count": int(len(union)),
                    "target_min_count_per_config": int(min_count),
                    "target_max_count_per_config": int(max_count),
                    "target_exact_alignment": bool(exact),
                    "target_jaccard_alignment": float(len(intersection) / len(union)) if len(union) > 0 else 1.0,
                    "passed": bool(exact),
                }
                align_rows.append(row)

        return pd.DataFrame(rows + align_rows)

    @staticmethod
    def _build_gold_dm_pairwise_results(dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame) -> pd.DataFrame:
        if fact_oos_predictions.empty or dim_run.empty:
            return pd.DataFrame()
        required = ["run_id", "split", "horizon", "target_timestamp_utc", "y_true", "y_pred"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        keep = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "split_signature", "split_fingerprint", "status"] if c in dim_run.columns]
        if "run_id" not in keep:
            return pd.DataFrame()
        df = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
        df = RefreshAnalyticsStoreUseCase._ensure_split_signature_column(df)
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

        rows: list[pd.DataFrame] = []
        group_cols = RefreshAnalyticsStoreUseCase._pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            g = RefreshAnalyticsStoreUseCase._select_top_configs_for_pairwise(g, max_configs=50)
            by_ts = (
                g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                .mean()
                .reset_index()
            )
            loss_matrix = by_ts.pivot(index="target_timestamp_utc", columns="config_label", values="squared_error")
            loss_matrix = loss_matrix.dropna(axis=0, how="any")
            if loss_matrix.empty or loss_matrix.shape[1] < 2:
                continue
            dm_df = RefreshAnalyticsStoreUseCase._compute_dm_pairwise_from_loss_matrix(loss_matrix)
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
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    @staticmethod
    def _build_gold_mcs_results(dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame) -> pd.DataFrame:
        if fact_oos_predictions.empty or dim_run.empty:
            return pd.DataFrame()
        required = ["run_id", "split", "horizon", "target_timestamp_utc", "y_true", "y_pred"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        keep = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "split_signature", "split_fingerprint", "status"] if c in dim_run.columns]
        if "run_id" not in keep:
            return pd.DataFrame()
        df = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
        df = RefreshAnalyticsStoreUseCase._ensure_split_signature_column(df)
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

        rows: list[pd.DataFrame] = []
        group_cols = RefreshAnalyticsStoreUseCase._pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            g = RefreshAnalyticsStoreUseCase._select_top_configs_for_pairwise(g, max_configs=50)
            by_ts = (
                g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                .mean()
                .reset_index()
            )
            loss_matrix = by_ts.pivot(index="target_timestamp_utc", columns="config_label", values="squared_error")
            loss_matrix = loss_matrix.dropna(axis=0, how="any")
            if loss_matrix.empty or loss_matrix.shape[1] < 2:
                continue
            mcs_df = RefreshAnalyticsStoreUseCase._compute_mcs_from_loss_matrix(loss_matrix)
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



    @staticmethod
    def _build_gold_paired_oos_intersection_by_horizon(
        dim_run: pd.DataFrame,
        fact_oos_predictions: pd.DataFrame,
        *,
        target_horizons: tuple[int, ...] = (1, 7, 30),
    ) -> pd.DataFrame:
        if fact_oos_predictions.empty or dim_run.empty:
            return pd.DataFrame()

        required = ["run_id", "split", "horizon", "target_timestamp_utc"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        keep = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "split_signature", "split_fingerprint", "status"] if c in dim_run.columns]
        if "run_id" not in keep:
            return pd.DataFrame()

        df = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
        df = RefreshAnalyticsStoreUseCase._ensure_split_signature_column(df)
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
        df = df[df["horizon"].isin([int(h) for h in target_horizons])].copy()
        df = df[df["split"].astype(str) == "test"].copy()
        if df.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        group_cols = RefreshAnalyticsStoreUseCase._pairwise_group_cols(df)
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
                    "horizon": int(keys[4]) if "split_signature" in group_cols and pd.notna(keys[4]) else (int(keys[3]) if pd.notna(keys[3]) else None),
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
                    "target_jaccard_alignment": float(len(intersection) / len(union)) if len(union) > 0 else 1.0,
                    "pairwise_ready_dm": bool(len(per_cfg) >= 2 and len(intersection) >= 5),
                    "pairwise_ready_mcs": bool(len(per_cfg) >= 2 and len(intersection) >= 1),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _build_gold_model_decision_final(
        metrics_by_config: pd.DataFrame,
        robustness_by_horizon: pd.DataFrame,
        generalization_gap: pd.DataFrame,
        dm_results: pd.DataFrame,
        mcs_results: pd.DataFrame,
        win_rate_results: pd.DataFrame,
        paired_intersection: pd.DataFrame,
    ) -> pd.DataFrame:
        if metrics_by_config.empty:
            return pd.DataFrame()

        req = {"asset", "feature_set_name", "config_signature", "split", "horizon"}
        if not req.issubset(set(metrics_by_config.columns)):
            return pd.DataFrame()

        base = metrics_by_config[metrics_by_config["split"].astype(str) == "test"].copy()
        if base.empty:
            return pd.DataFrame()

        # keep core metrics used in academic comparison
        keep = [
            c for c in [
                "asset", "feature_set_name", "config_signature", "split", "horizon", "n_runs",
                "mean_rmse", "std_rmse", "mean_mae", "std_mae", "mean_directional_accuracy", "std_directional_accuracy",
                "mean_pinball_q10", "mean_pinball_q50", "mean_pinball_q90", "mean_mean_pinball",
                "mean_picp", "mean_mpiw",
            ] if c in base.columns
        ]
        out = base[keep].copy()
        out["config_label"] = out["feature_set_name"].astype(str) + "|" + out["config_signature"].astype(str)
        out = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(out)

        # merge CI95 from robustness (metric granularity -> wide)
        if not robustness_by_horizon.empty and {"asset", "feature_set_name", "config_signature", "horizon", "metric", "ci95_low", "ci95_high"}.issubset(set(robustness_by_horizon.columns)):
            ci = robustness_by_horizon.copy()
            ci = ci[ci["metric"].astype(str).isin(["rmse", "mae", "directional_accuracy"])].copy()
            ci_wide = (
                ci.pivot_table(
                    index=["asset", "feature_set_name", "config_signature", "horizon"],
                    columns="metric",
                    values=["ci95_low", "ci95_high"],
                    aggfunc="first",
                )
                .reset_index()
            )
            ci_wide.columns = [
                "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
                for col in ci_wide.columns
            ]
            out = out.merge(
                ci_wide,
                on=["asset", "feature_set_name", "config_signature", "horizon"],
                how="left",
            )

        # merge generalization gap
        if not generalization_gap.empty and {"asset", "feature_set_name", "config_signature", "horizon"}.issubset(set(generalization_gap.columns)):
            gap_keep = [
                c for c in [
                    "asset", "feature_set_name", "config_signature", "horizon",
                    "gap_rmse_test_minus_val", "gap_mae_test_minus_val", "gap_directional_accuracy_test_minus_val",
                    "gap_mean_pinball_test_minus_val", "gap_picp_test_minus_val", "gap_mpiw_test_minus_val",
                ] if c in generalization_gap.columns
            ]
            out = out.merge(generalization_gap[gap_keep], on=["asset", "feature_set_name", "config_signature", "horizon"], how="left")

        # DM summary per config
        dm_rows: list[dict[str, object]] = []
        if not dm_results.empty and {"asset", "parent_sweep_id", "split", "horizon", "left_config", "right_config", "pvalue_two_sided", "mean_loss_diff_left_minus_right"}.issubset(set(dm_results.columns)):
            for keys, g in dm_results.groupby(["asset", "parent_sweep_id", "split", "horizon"], dropna=False):
                winners: dict[str, int] = {}
                losers: dict[str, int] = {}
                for _, r in g.iterrows():
                    p = pd.to_numeric(r.get("pvalue_two_sided"), errors="coerce")
                    d = pd.to_numeric(r.get("mean_loss_diff_left_minus_right"), errors="coerce")
                    lcfg = str(r.get("left_config"))
                    rcfg = str(r.get("right_config"))
                    if pd.isna(p) or pd.isna(d) or p >= 0.05:
                        continue
                    if d < 0:
                        winners[lcfg] = winners.get(lcfg, 0) + 1
                        losers[rcfg] = losers.get(rcfg, 0) + 1
                    elif d > 0:
                        winners[rcfg] = winners.get(rcfg, 0) + 1
                        losers[lcfg] = losers.get(lcfg, 0) + 1
                configs = set(list(winners.keys()) + list(losers.keys()))
                for cfg in configs:
                    dm_rows.append(
                        {
                            "asset": keys[0],
                            "parent_sweep_id": keys[1],
                            "split": keys[2],
                            "horizon": int(keys[3]) if pd.notna(keys[3]) else None,
                            "config_label": cfg,
                            "dm_significant_wins": int(winners.get(cfg, 0)),
                            "dm_significant_losses": int(losers.get(cfg, 0)),
                            "dm_net_wins": int(winners.get(cfg, 0) - losers.get(cfg, 0)),
                        }
                    )
        dm_summary = pd.DataFrame(dm_rows)

        # MCS summary per config
        mcs_summary = pd.DataFrame()
        if not mcs_results.empty and {"asset", "parent_sweep_id", "split", "horizon", "config_label", "selected_in_mcs_alpha_0_05"}.issubset(set(mcs_results.columns)):
            mcs_summary = mcs_results[["asset", "parent_sweep_id", "split", "horizon", "config_label", "selected_in_mcs_alpha_0_05"]].copy()
            mcs_summary = mcs_summary.rename(columns={"selected_in_mcs_alpha_0_05": "mcs_selected_alpha_0_05"})

        # win-rate summary per config
        wr_rows: list[dict[str, object]] = []
        if not win_rate_results.empty and {"asset", "parent_sweep_id", "split", "horizon", "left_config", "right_config", "left_win_rate_ex_ties", "right_win_rate_ex_ties"}.issubset(set(win_rate_results.columns)):
            recs: dict[tuple[object, object, object, object, str], list[float]] = {}
            for _, r in win_rate_results.iterrows():
                key_left = (r.get("asset"), r.get("parent_sweep_id"), r.get("split"), r.get("horizon"), str(r.get("left_config")))
                key_right = (r.get("asset"), r.get("parent_sweep_id"), r.get("split"), r.get("horizon"), str(r.get("right_config")))
                lw = pd.to_numeric(r.get("left_win_rate_ex_ties"), errors="coerce")
                rw = pd.to_numeric(r.get("right_win_rate_ex_ties"), errors="coerce")
                if pd.notna(lw):
                    recs.setdefault(key_left, []).append(float(lw))
                if pd.notna(rw):
                    recs.setdefault(key_right, []).append(float(rw))
            for (asset, sweep, split, horizon, cfg), vals in recs.items():
                horizon_value = pd.to_numeric(horizon, errors="coerce")
                wr_rows.append(
                    {
                        "asset": asset,
                        "parent_sweep_id": sweep,
                        "split": split,
                        "horizon": int(horizon_value) if pd.notna(horizon_value) else None,
                        "config_label": cfg,
                        "win_rate_ex_ties_mean": float(np.mean(vals)),
                    }
                )
        wr_summary = pd.DataFrame(wr_rows)

        dm_summary = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(dm_summary)
        mcs_summary = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(mcs_summary)
        wr_summary = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(wr_summary)

        # infer parent_sweep from config-specific pairwise tables
        sweep_map = pd.DataFrame()
        for df in [mcs_summary, dm_summary, wr_summary]:
            if not df.empty and {"asset", "split", "horizon", "config_label", "parent_sweep_id"}.issubset(set(df.columns)):
                part = df[["asset", "split", "horizon", "config_label", "parent_sweep_id"]].drop_duplicates()
                sweep_map = pd.concat([sweep_map, part], ignore_index=True) if not sweep_map.empty else part
        if not sweep_map.empty:
            sweep_map = sweep_map.drop_duplicates(["asset", "split", "horizon", "config_label"])
            out = out.merge(sweep_map, on=["asset", "split", "horizon", "config_label"], how="left")
        else:
            out["parent_sweep_id"] = None

        # merge pairwise summaries
        for df in [dm_summary, mcs_summary, wr_summary]:
            if df.empty:
                continue
            merge_cols = [c for c in ["asset", "parent_sweep_id", "split", "horizon", "config_label"] if c in df.columns and c in out.columns]
            if not merge_cols:
                continue
            out = out.merge(df.drop_duplicates(merge_cols), on=merge_cols, how="left")

        # merge explicit intersection readiness
        if not paired_intersection.empty and {"asset", "parent_sweep_id", "split", "horizon"}.issubset(set(paired_intersection.columns)):
            inter_keep = [
                c for c in [
                    "asset", "parent_sweep_id", "split", "horizon",
                    "n_common", "n_union", "coverage_ratio", "aligned_exact",
                    "target_intersection_count", "target_exact_alignment", "target_jaccard_alignment",
                    "pairwise_ready_dm", "pairwise_ready_mcs"
                ] if c in paired_intersection.columns
            ]
            pi = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(paired_intersection[inter_keep])
            key_cols = ["asset", "parent_sweep_id", "split", "horizon"]
            if not pi.empty and pi.duplicated(subset=key_cols).any():
                agg: dict[str, str] = {}
                for col in ["n_common", "target_intersection_count"]:
                    if col in pi.columns:
                        agg[col] = "min"
                for col in ["n_union"]:
                    if col in pi.columns:
                        agg[col] = "max"
                for col in ["coverage_ratio", "target_jaccard_alignment"]:
                    if col in pi.columns:
                        agg[col] = "min"
                for col in ["aligned_exact", "target_exact_alignment", "pairwise_ready_dm", "pairwise_ready_mcs"]:
                    if col in pi.columns:
                        agg[col] = "all"
                if agg:
                    pi = pi.groupby(key_cols, dropna=False).agg(agg).reset_index()
                else:
                    pi = pi.drop_duplicates(key_cols)
            out = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(out)
            pi = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(pi)
            out = out.merge(pi, on=key_cols, how="left")

        # ranking columns for decision
        out = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(out)
        decision_rank_cols = ["asset", "parent_sweep_id", "horizon"]
        out["rank_rmse"] = out.groupby(decision_rank_cols, dropna=False)['mean_rmse'].rank(method='min', ascending=True) if 'mean_rmse' in out.columns else np.nan
        out["rank_mae"] = out.groupby(decision_rank_cols, dropna=False)['mean_mae'].rank(method='min', ascending=True) if 'mean_mae' in out.columns else np.nan
        out["rank_da"] = out.groupby(decision_rank_cols, dropna=False)['mean_directional_accuracy'].rank(method='min', ascending=False) if 'mean_directional_accuracy' in out.columns else np.nan

        for col in ["pairwise_ready_dm", "pairwise_ready_mcs", "target_exact_alignment"]:
            if col not in out.columns:
                out[col] = False
        out["academic_decision_ready"] = (
            out["pairwise_ready_dm"].fillna(False).astype(bool)
            & out["pairwise_ready_mcs"].fillna(False).astype(bool)
            & out["target_exact_alignment"].fillna(False).astype(bool)
        )

        return out.sort_values(["asset", "parent_sweep_id", "horizon", "rank_rmse", "rank_mae"], ascending=[True, True, True, True, True]).reset_index(drop=True)

    @staticmethod
    def _build_gold_win_rate_pairwise_results(dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame) -> pd.DataFrame:
        if fact_oos_predictions.empty or dim_run.empty:
            return pd.DataFrame()
        required = ["run_id", "split", "horizon", "target_timestamp_utc", "y_true", "y_pred"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        keep = [c for c in ["run_id", "asset", "feature_set_name", "config_signature", "parent_sweep_id", "split_signature", "split_fingerprint", "status"] if c in dim_run.columns]
        if "run_id" not in keep:
            return pd.DataFrame()
        df = fact_oos_predictions.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")
        df = RefreshAnalyticsStoreUseCase._ensure_split_signature_column(df)
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

        rows: list[dict[str, object]] = []
        group_cols = RefreshAnalyticsStoreUseCase._pairwise_group_cols(df)
        for keys, g in df.groupby(group_cols, dropna=False):
            g = RefreshAnalyticsStoreUseCase._select_top_configs_for_pairwise(g, max_configs=50)
            by_ts = (
                g.groupby(["target_timestamp_utc", "config_label"], dropna=False)["squared_error"]
                .mean()
                .reset_index()
            )
            loss_matrix = by_ts.pivot(index="target_timestamp_utc", columns="config_label", values="squared_error")
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
                            "horizon": int(keys[4]) if "split_signature" in group_cols and pd.notna(keys[4]) else (int(keys[3]) if pd.notna(keys[3]) else None),
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
                            "left_minus_right_mean_loss": float(comp["l"].mean() - comp["r"].mean()),
                        }
                    )
        return pd.DataFrame(rows)

    @staticmethod
    def _build_gold_quality_statistics_report(
        quality_report: pd.DataFrame,
        dm_results: pd.DataFrame,
        mcs_results: pd.DataFrame,
        win_rate_results: pd.DataFrame,
    ) -> pd.DataFrame:
        if quality_report.empty and dm_results.empty and mcs_results.empty and win_rate_results.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []

        q = quality_report.copy() if not quality_report.empty else pd.DataFrame()
        if not q.empty and "scope" in q.columns:
            q = q[q["scope"].astype(str).isin(["run_split_horizon", "alignment_group"])].copy()

        keys = ["asset", "parent_sweep_id", "split", "horizon"]

        q_by_group = {}
        if not q.empty and set(keys).issubset(set(q.columns)):
            for k, g in q.groupby(keys, dropna=False):
                q_by_group[k] = {
                    "quality_rows": int(len(g)),
                    "quality_passed_all": bool(g.get("passed", pd.Series(dtype=bool)).fillna(False).all()),
                }

        dm_by_group = {}
        if not dm_results.empty and set(keys).issubset(set(dm_results.columns)):
            for k, g in dm_results.groupby(keys, dropna=False):
                dm_by_group[k] = {
                    "dm_pairs": int(len(g)),
                    "dm_min_pvalue": float(pd.to_numeric(g.get("pvalue_two_sided"), errors="coerce").min()),
                    "aligned_timestamps_dm": int(pd.to_numeric(g.get("aligned_timestamps"), errors="coerce").max()),
                    "n_configs_dm": int(pd.to_numeric(g.get("n_configs"), errors="coerce").max()),
                }

        mcs_by_group = {}
        if not mcs_results.empty and set(keys).issubset(set(mcs_results.columns)):
            for k, g in mcs_results.groupby(keys, dropna=False):
                selected = pd.to_numeric(g.get("selected_in_mcs_alpha_0_05"), errors="coerce")
                mcs_by_group[k] = {
                    "mcs_models": int(len(g)),
                    "mcs_selected_models": int(selected.fillna(0).astype(int).sum()) if not selected.empty else 0,
                    "aligned_timestamps_mcs": int(pd.to_numeric(g.get("aligned_timestamps"), errors="coerce").max()),
                    "n_configs_mcs": int(pd.to_numeric(g.get("n_configs"), errors="coerce").max()),
                }

        wr_by_group = {}
        if not win_rate_results.empty and set(keys).issubset(set(win_rate_results.columns)):
            for k, g in win_rate_results.groupby(keys, dropna=False):
                wr_by_group[k] = {
                    "win_rate_pairs": int(len(g)),
                    "aligned_timestamps_win_rate": int(pd.to_numeric(g.get("aligned_timestamps"), errors="coerce").max()),
                }

        all_keys = set(q_by_group.keys()) | set(dm_by_group.keys()) | set(mcs_by_group.keys()) | set(wr_by_group.keys())
        for k in sorted(all_keys, key=lambda x: tuple(str(v) for v in x)):
            base = {
                "asset": k[0],
                "parent_sweep_id": k[1],
                "split": k[2],
                "horizon": k[3],
                "quality_rows": 0,
                "quality_passed_all": False,
                "dm_pairs": 0,
                "mcs_models": 0,
                "mcs_selected_models": 0,
                "win_rate_pairs": 0,
                "aligned_timestamps_dm": 0,
                "aligned_timestamps_mcs": 0,
                "aligned_timestamps_win_rate": 0,
                "n_configs_dm": 0,
                "n_configs_mcs": 0,
            }
            base.update(q_by_group.get(k, {}))
            base.update(dm_by_group.get(k, {}))
            base.update(mcs_by_group.get(k, {}))
            base.update(wr_by_group.get(k, {}))
            base["dm_available"] = bool(base["dm_pairs"] > 0)
            base["mcs_available"] = bool(base["mcs_models"] > 0)
            base["win_rate_available"] = bool(base["win_rate_pairs"] > 0)
            base["statistics_ready"] = bool(base["quality_passed_all"] and base["dm_available"] and base["mcs_available"])
            rows.append(base)

        return pd.DataFrame(rows)


    @staticmethod
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

    @staticmethod
    def _build_gold_feature_contrib_local_summary(
        fact_feature_contrib_local: pd.DataFrame,
        dim_run: pd.DataFrame,
    ) -> pd.DataFrame:
        if fact_feature_contrib_local.empty:
            return pd.DataFrame()

        req = {"asset", "feature_set_name", "horizon", "feature_name", "contribution", "abs_contribution"}
        if not req.issubset(set(fact_feature_contrib_local.columns)):
            return pd.DataFrame()

        df = fact_feature_contrib_local.copy()
        if "run_id" in df.columns and {"run_id", "parent_sweep_id"}.issubset(set(dim_run.columns)):
            dim_trim = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(
                dim_run[["run_id", "parent_sweep_id"]].drop_duplicates("run_id")
            )
            df = df.merge(dim_trim, on="run_id", how="left")
            df = RefreshAnalyticsStoreUseCase._normalize_parent_sweep_id_for_merge(df)
        else:
            df["parent_sweep_id"] = None
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce")
        df["abs_contribution"] = pd.to_numeric(df["abs_contribution"], errors="coerce")
        if "feature_rank" in df.columns:
            df["feature_rank"] = pd.to_numeric(df["feature_rank"], errors="coerce")
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        df = df.dropna(subset=["horizon", "contribution", "abs_contribution", "feature_name"]).copy()
        if df.empty:
            return pd.DataFrame()
        df["horizon"] = df["horizon"].astype(int)
        if "method" not in df.columns:
            df["method"] = "unknown"

        group_cols = ["asset", "feature_set_name", "parent_sweep_id", "horizon", "feature_name", "method"]
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_rows=("contribution", "count"),
                n_inference_runs=("inference_run_id", "nunique") if "inference_run_id" in df.columns else ("contribution", "count"),
                mean_contribution=("contribution", "mean"),
                mean_abs_contribution=("abs_contribution", "mean"),
                median_abs_contribution=("abs_contribution", "median"),
                std_abs_contribution=("abs_contribution", "std"),
                top3_frequency=("feature_rank", lambda s: float((pd.to_numeric(pd.Series(s), errors="coerce") <= 3).fillna(False).mean()))
                if "feature_rank" in df.columns
                else ("contribution", lambda s: np.nan),
                positive_share=("contribution", lambda s: float((pd.to_numeric(pd.Series(s), errors="coerce") > 0).fillna(False).mean())),
            )
            .reset_index()
        )

        stability_rows: list[dict[str, object]] = []
        if "inference_run_id" in df.columns:
            stab_group_cols = ["asset", "feature_set_name", "parent_sweep_id", "horizon", "method"]
            for keys, g in df.groupby(stab_group_cols, dropna=False):
                run_top: dict[str, set[str]] = {}
                for rid, rg in g.groupby("inference_run_id", dropna=False):
                    top = (
                        rg.groupby("feature_name", dropna=False)["abs_contribution"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(3)
                    )
                    run_top[str(rid)] = set(top.index.astype(str).tolist())

                vals: list[float] = []
                run_ids = sorted(run_top.keys())
                if len(run_ids) >= 2:
                    for a, b in combinations(run_ids, 2):
                        sa = run_top.get(a, set())
                        sb = run_top.get(b, set())
                        if not sa and not sb:
                            vals.append(1.0)
                        else:
                            union = sa | sb
                            inter = sa & sb
                            vals.append(float(len(inter) / len(union)) if union else 1.0)
                stability_rows.append(
                    {
                        "asset": keys[0],
                        "feature_set_name": keys[1],
                        "parent_sweep_id": keys[2],
                        "horizon": int(keys[3]) if pd.notna(keys[3]) else None,
                        "method": keys[4],
                        "local_top3_jaccard_mean": float(np.mean(vals)) if vals else np.nan,
                        "local_top3_jaccard_pairs": int(len(vals)),
                    }
                )

        if stability_rows:
            stab = pd.DataFrame(stability_rows)
            agg = agg.merge(stab, on=["asset", "feature_set_name", "parent_sweep_id", "horizon", "method"], how="left")

        return agg

    def execute(self) -> RefreshAnalyticsStoreResult:
        dim_run = self._load_partitioned_table(self.analytics_silver_dir, "dim_run")
        fact_split_metrics = self._load_partitioned_table(self.analytics_silver_dir, "fact_split_metrics")
        fact_oos_predictions = self._load_partitioned_table(self.analytics_silver_dir, "fact_oos_predictions")
        fact_model_artifacts = self._load_partitioned_table(self.analytics_silver_dir, "fact_model_artifacts")
        fact_feature_contrib_local = self._load_partitioned_table(self.analytics_silver_dir, "fact_feature_contrib_local")

        base = self._base_join_runs_split_metrics(dim_run, fact_split_metrics)

        outputs: dict[str, str] = {}
        outputs["gold_runs_long"] = self._safe_write(
            self._build_gold_runs_long(base),
            self.analytics_gold_dir / "gold_runs_long.parquet",
        )
        outputs["gold_ranking_by_config"] = self._safe_write(
            self._build_gold_ranking_by_config(base),
            self.analytics_gold_dir / "gold_ranking_by_config.parquet",
        )
        outputs["gold_consistency_topk"] = self._safe_write(
            self._build_gold_consistency_topk(base),
            self.analytics_gold_dir / "gold_consistency_topk.parquet",
        )
        outputs["gold_ic95_by_config_metric"] = self._safe_write(
            self._build_gold_ic95(base),
            self.analytics_gold_dir / "gold_ic95_by_config_metric.parquet",
        )
        outputs["gold_feature_set_impact"] = self._safe_write(
            self._build_gold_feature_set_impact(base),
            self.analytics_gold_dir / "gold_feature_set_impact.parquet",
        )
        gold_oos_consolidated = self._build_gold_oos_consolidated(dim_run, fact_oos_predictions)
        outputs["gold_oos_consolidated"] = self._safe_write(
            gold_oos_consolidated,
            self.analytics_gold_dir / "gold_oos_consolidated.parquet",
        )

        gold_prediction_metrics_by_run_split_horizon = self._build_gold_prediction_metrics_by_run_split_horizon(
            dim_run,
            fact_oos_predictions,
        )
        outputs["gold_prediction_metrics_by_run_split_horizon"] = self._safe_write(
            gold_prediction_metrics_by_run_split_horizon,
            self.analytics_gold_dir / "gold_prediction_metrics_by_run_split_horizon.parquet",
        )
        outputs["gold_quantile_guardrail_audit"] = self._safe_write(
            self._build_gold_quantile_guardrail_audit(dim_run, fact_oos_predictions),
            self.analytics_gold_dir / "gold_quantile_guardrail_audit.parquet",
        )
        outputs["gold_prediction_metrics_by_config"] = self._safe_write(
            self._build_gold_prediction_metrics_by_config(gold_prediction_metrics_by_run_split_horizon),
            self.analytics_gold_dir / "gold_prediction_metrics_by_config.parquet",
        )
        outputs["gold_prediction_metrics_by_horizon"] = self._safe_write(
            self._build_gold_prediction_metrics_by_horizon(gold_prediction_metrics_by_run_split_horizon),
            self.analytics_gold_dir / "gold_prediction_metrics_by_horizon.parquet",
        )
        outputs["gold_prediction_calibration"] = self._safe_write(
            self._build_gold_prediction_calibration(gold_prediction_metrics_by_run_split_horizon),
            self.analytics_gold_dir / "gold_prediction_calibration.parquet",
        )
        outputs["gold_prediction_risk"] = self._safe_write(
            self._build_gold_prediction_risk(dim_run, fact_oos_predictions),
            self.analytics_gold_dir / "gold_prediction_risk.parquet",
        )
        outputs["gold_prediction_generalization_gap"] = self._safe_write(
            self._build_gold_prediction_generalization_gap(gold_prediction_metrics_by_run_split_horizon),
            self.analytics_gold_dir / "gold_prediction_generalization_gap.parquet",
        )
        outputs["gold_prediction_robustness_by_horizon"] = self._safe_write(
            self._build_gold_prediction_robustness_by_horizon(gold_prediction_metrics_by_run_split_horizon),
            self.analytics_gold_dir / "gold_prediction_robustness_by_horizon.parquet",
        )
        outputs["gold_feature_impact_by_horizon"] = self._safe_write(
            self._build_gold_feature_impact_by_horizon(
                fact_model_artifacts,
                gold_prediction_metrics_by_run_split_horizon,
            ),
            self.analytics_gold_dir / "gold_feature_impact_by_horizon.parquet",
        )
        outputs["gold_feature_contrib_local_summary"] = self._safe_write(
            self._build_gold_feature_contrib_local_summary(fact_feature_contrib_local, dim_run),
            self.analytics_gold_dir / "gold_feature_contrib_local_summary.parquet",
        )
        gold_quality = self._build_gold_oos_quality_report(dim_run, fact_oos_predictions)
        outputs["gold_oos_quality_report"] = self._safe_write(
            gold_quality,
            self.analytics_gold_dir / "gold_oos_quality_report.parquet",
        )
        gold_dm = self._apply_holm_adjustment_for_dm(
            self._build_gold_dm_pairwise_results(dim_run, fact_oos_predictions)
        )
        outputs["gold_dm_pairwise_results"] = self._safe_write(
            gold_dm,
            self.analytics_gold_dir / "gold_dm_pairwise_results.parquet",
        )
        gold_mcs = self._build_gold_mcs_results(dim_run, fact_oos_predictions)
        outputs["gold_mcs_results"] = self._safe_write(
            gold_mcs,
            self.analytics_gold_dir / "gold_mcs_results.parquet",
        )
        gold_win_rate = self._build_gold_win_rate_pairwise_results(dim_run, fact_oos_predictions)
        outputs["gold_win_rate_pairwise_results"] = self._safe_write(
            gold_win_rate,
            self.analytics_gold_dir / "gold_win_rate_pairwise_results.parquet",
        )
        gold_paired_intersection = self._build_gold_paired_oos_intersection_by_horizon(
            dim_run,
            fact_oos_predictions,
            target_horizons=(1, 7, 30),
        )
        outputs["gold_paired_oos_intersection_by_horizon"] = self._safe_write(
            gold_paired_intersection,
            self.analytics_gold_dir / "gold_paired_oos_intersection_by_horizon.parquet",
        )
        outputs["gold_model_decision_final"] = self._safe_write(
            self._build_gold_model_decision_final(
                metrics_by_config=self._build_gold_prediction_metrics_by_config(gold_prediction_metrics_by_run_split_horizon),
                robustness_by_horizon=self._build_gold_prediction_robustness_by_horizon(gold_prediction_metrics_by_run_split_horizon),
                generalization_gap=self._build_gold_prediction_generalization_gap(gold_prediction_metrics_by_run_split_horizon),
                dm_results=gold_dm,
                mcs_results=gold_mcs,
                win_rate_results=gold_win_rate,
                paired_intersection=gold_paired_intersection,
            ),
            self.analytics_gold_dir / "gold_model_decision_final.parquet",
        )
        gold_quality_statistics = self._build_gold_quality_statistics_report(
            quality_report=gold_quality,
            dm_results=gold_dm,
            mcs_results=gold_mcs,
            win_rate_results=gold_win_rate,
        )
        outputs["gold_quality_statistics_report"] = self._safe_write(
            gold_quality_statistics,
            self.analytics_gold_dir / "gold_quality_statistics_report.parquet",
        )
        outputs["gold_quality_run_sweep_summary"] = self._safe_write(
            self._build_gold_quality_run_sweep_summary(
                quality_report=gold_quality,
                quality_statistics=gold_quality_statistics,
                dim_run=dim_run,
            ),
            self.analytics_gold_dir / "gold_quality_run_sweep_summary.parquet",
        )

        return RefreshAnalyticsStoreResult(
            gold_dir=str(to_project_relative(self.analytics_gold_dir)),
            outputs=outputs,
        )
