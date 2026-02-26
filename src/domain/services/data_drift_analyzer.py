from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftFeatureStats:
    feature: str
    n_train: int
    n_val: int
    n_test: int
    train_mean: float | None
    val_mean: float | None
    test_mean: float | None
    train_std: float | None
    val_std: float | None
    test_std: float | None
    train_missing_rate: float
    val_missing_rate: float
    test_missing_rate: float
    ks_train_vs_val: float | None
    ks_train_vs_test: float | None
    psi_train_vs_val: float | None
    psi_train_vs_test: float | None


class DataDriftAnalyzer:
    @staticmethod
    def _to_numeric_finite(series: pd.Series) -> np.ndarray:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
        return arr[np.isfinite(arr)]

    @staticmethod
    def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float | None:
        if reference.size == 0 or current.size == 0:
            return None
        x = np.sort(reference)
        y = np.sort(current)
        all_vals = np.sort(np.unique(np.concatenate([x, y])))
        cdf_x = np.searchsorted(x, all_vals, side="right") / x.size
        cdf_y = np.searchsorted(y, all_vals, side="right") / y.size
        return float(np.max(np.abs(cdf_x - cdf_y)))

    @staticmethod
    def psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float | None:
        if reference.size == 0 or current.size == 0:
            return None
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.quantile(reference, quantiles)
        edges = np.unique(edges)
        if edges.size < 2:
            return None
        edges[0] = -np.inf
        edges[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)
        ref_prop = ref_counts / max(ref_counts.sum(), 1)
        cur_prop = cur_counts / max(cur_counts.sum(), 1)
        eps = 1e-6
        ref_prop = np.clip(ref_prop, eps, None)
        cur_prop = np.clip(cur_prop, eps, None)
        return float(np.sum((ref_prop - cur_prop) * np.log(ref_prop / cur_prop)))

    @staticmethod
    def analyze_features(
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        psi_bins: int = 10,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for feature in feature_cols:
            if feature not in train_df.columns or feature not in val_df.columns or feature not in test_df.columns:
                continue
            train_all = pd.to_numeric(train_df[feature], errors="coerce")
            val_all = pd.to_numeric(val_df[feature], errors="coerce")
            test_all = pd.to_numeric(test_df[feature], errors="coerce")
            train = DataDriftAnalyzer._to_numeric_finite(train_df[feature])
            val = DataDriftAnalyzer._to_numeric_finite(val_df[feature])
            test = DataDriftAnalyzer._to_numeric_finite(test_df[feature])

            rows.append(
                DriftFeatureStats(
                    feature=feature,
                    n_train=int(train.size),
                    n_val=int(val.size),
                    n_test=int(test.size),
                    train_mean=float(np.mean(train)) if train.size else None,
                    val_mean=float(np.mean(val)) if val.size else None,
                    test_mean=float(np.mean(test)) if test.size else None,
                    train_std=float(np.std(train)) if train.size else None,
                    val_std=float(np.std(val)) if val.size else None,
                    test_std=float(np.std(test)) if test.size else None,
                    train_missing_rate=float(train_all.isna().mean()),
                    val_missing_rate=float(val_all.isna().mean()),
                    test_missing_rate=float(test_all.isna().mean()),
                    ks_train_vs_val=DataDriftAnalyzer.ks_statistic(train, val),
                    ks_train_vs_test=DataDriftAnalyzer.ks_statistic(train, test),
                    psi_train_vs_val=DataDriftAnalyzer.psi(train, val, bins=psi_bins),
                    psi_train_vs_test=DataDriftAnalyzer.psi(train, test, bins=psi_bins),
                ).__dict__
            )

        detail_df = pd.DataFrame(rows)
        if detail_df.empty:
            return detail_df, {
                "n_features": 0,
                "avg_ks_train_vs_val": None,
                "avg_ks_train_vs_test": None,
                "avg_psi_train_vs_val": None,
                "avg_psi_train_vs_test": None,
            }

        for col in [
            "ks_train_vs_val",
            "ks_train_vs_test",
            "psi_train_vs_val",
            "psi_train_vs_test",
        ]:
            detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce")

        summary = {
            "n_features": int(len(detail_df)),
            "avg_ks_train_vs_val": float(detail_df["ks_train_vs_val"].mean(skipna=True)),
            "avg_ks_train_vs_test": float(detail_df["ks_train_vs_test"].mean(skipna=True)),
            "avg_psi_train_vs_val": float(detail_df["psi_train_vs_val"].mean(skipna=True)),
            "avg_psi_train_vs_test": float(detail_df["psi_train_vs_test"].mean(skipna=True)),
            "n_ks_train_vs_val_gt_0_2": int((detail_df["ks_train_vs_val"] > 0.2).sum()),
            "n_ks_train_vs_test_gt_0_2": int((detail_df["ks_train_vs_test"] > 0.2).sum()),
            "n_psi_train_vs_val_gt_0_25": int((detail_df["psi_train_vs_val"] > 0.25).sum()),
            "n_psi_train_vs_test_gt_0_25": int((detail_df["psi_train_vs_test"] > 0.25).sum()),
        }
        return detail_df, summary
