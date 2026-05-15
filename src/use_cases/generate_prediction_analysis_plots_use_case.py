from __future__ import annotations

import json
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.matplotlib_backend import ensure_non_interactive_matplotlib_backend
from src.utils.path_policy import to_project_relative

ensure_non_interactive_matplotlib_backend()


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate prediction analysis plots. "
            "Install optional plotting dependencies to run this feature."
        ) from exc
    return plt


@dataclass(frozen=True)
class GeneratePredictionAnalysisPlotsResult:
    output_dir: str
    outputs: dict[str, str]


class GeneratePredictionAnalysisPlotsUseCase:
    def __init__(
        self,
        *,
        analytics_gold_dir: str | Path,
        analytics_silver_dir: str | Path,
        output_dir: str | Path,
    ) -> None:
        self.analytics_gold_dir = Path(analytics_gold_dir)
        self.analytics_silver_dir = Path(analytics_silver_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_gold_table(gold_dir: Path, name: str) -> pd.DataFrame:
        p = gold_dir / f"{name}.parquet"
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    @staticmethod
    def _load_partitioned_table(base_dir: Path, table_name: str) -> pd.DataFrame:
        table_dir = base_dir / table_name
        if not table_dir.exists():
            return pd.DataFrame()
        files = sorted(table_dir.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)

    @staticmethod
    def _config_label(df: pd.DataFrame) -> pd.Series:
        if "config_label" in df.columns:
            return df["config_label"].astype(str)
        fs = df["feature_set_name"].astype(str) if "feature_set_name" in df.columns else pd.Series(["?"] * len(df))
        cs = df["config_signature"].astype(str) if "config_signature" in df.columns else pd.Series(["?"] * len(df))
        return fs + "|" + cs

    @staticmethod
    def _short_config_label(raw_label: str) -> str:
        label = str(raw_label)
        if "|" not in label:
            return label
        left, right = label.split("|", 1)
        right = right.strip()
        if not right:
            return left.strip()
        if len(right) <= 8:
            return f"{left.strip()}|{right}"
        return f"{left.strip()}|{right[:5]}..."

    @staticmethod
    def _dedupe_labels(labels: list[str]) -> list[str]:
        seen: dict[str, int] = {}
        out: list[str] = []
        for lb in labels:
            seen[lb] = seen.get(lb, 0) + 1
            n = seen[lb]
            out.append(lb if n == 1 else f"{lb}#{n}")
        return out

    @staticmethod
    def _save_no_data(path: Path, title: str, reason: str) -> None:
        plt = _require_pyplot()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(0.5, 0.4, reason, ha="center", va="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _pick_horizons(df: pd.DataFrame, preferred: tuple[int, ...] = (1, 7, 30)) -> list[int]:
        if "horizon" not in df.columns or df.empty:
            return list(preferred)
        h = pd.to_numeric(df["horizon"], errors="coerce").dropna().astype(int)
        if h.empty:
            return list(preferred)
        available = sorted(set(h.tolist()))
        chosen = [x for x in preferred if x in available]
        return chosen if chosen else available[:3]

    @staticmethod
    def _filter_asset(df: pd.DataFrame, asset: str | None) -> pd.DataFrame:
        if asset is None or df.empty or "asset" not in df.columns:
            return df
        return df[df["asset"].astype(str).str.upper() == asset.upper()].copy()

    @staticmethod
    def _normalize_sweep_id(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "<na>"}:
            return None
        if text.endswith('.0') and text[:-2].isdigit():
            return text[:-2]
        return text

    def _build_scope_selection(
        self,
        *,
        scope_csv_path: str | Path | None,
        scope_sweep_prefixes: list[str] | None,
    ) -> dict[str, set[str] | set[tuple[str, str]]] | None:
        has_csv = bool(scope_csv_path)
        has_prefixes = bool(scope_sweep_prefixes)
        if not has_csv and not has_prefixes:
            return None

        dim_run = self._load_partitioned_table(self.analytics_silver_dir, "dim_run")
        if dim_run.empty:
            return None

        selected = dim_run.copy()
        if has_prefixes:
            prefixes = [str(x).strip() for x in (scope_sweep_prefixes or []) if str(x).strip()]
            if prefixes:
                masks: list[pd.Series] = []
                if "parent_sweep_id" in selected.columns:
                    sid = selected["parent_sweep_id"].map(self._normalize_sweep_id).fillna("").astype(str)
                    masks.append(sid.str.startswith(tuple(prefixes)))

                # Fallback for runs where parent_sweep_id is null but artifact paths include the sweep folder.
                path_cols = [
                    c for c in ["checkpoint_path_best", "checkpoint_path_final", "model_path"]
                    if c in selected.columns
                ]
                for c in path_cols:
                    vals = selected[c].astype(str)
                    masks.append(vals.str.contains("|".join(prefixes), regex=True, na=False))

                if masks:
                    mask = masks[0].copy()
                    for m in masks[1:]:
                        mask = mask | m
                    selected = selected[mask].copy()

        if has_csv:
            csv_df = pd.read_csv(Path(scope_csv_path))
            if csv_df.empty:
                return None
            selected_parts: list[pd.DataFrame] = []
            if "run_id" in csv_df.columns and "run_id" in selected.columns:
                run_ids = set(csv_df["run_id"].dropna().astype(str).tolist())
                if run_ids:
                    selected_parts.append(selected[selected["run_id"].astype(str).isin(run_ids)].copy())
            if {"feature_set_name", "config_signature"}.issubset(csv_df.columns) and {"feature_set_name", "config_signature"}.issubset(selected.columns):
                keys = set(
                    (str(a), str(b))
                    for a, b in csv_df[["feature_set_name", "config_signature"]].dropna().itertuples(index=False, name=None)
                )
                if keys:
                    mask = selected.apply(lambda r: (str(r.get("feature_set_name")), str(r.get("config_signature"))) in keys, axis=1)
                    selected_parts.append(selected[mask].copy())
            if "config_label" in csv_df.columns and {"feature_set_name", "config_signature"}.issubset(selected.columns):
                labels = set(csv_df["config_label"].dropna().astype(str).tolist())
                if labels:
                    mask = selected.apply(
                        lambda r: f"{str(r.get('feature_set_name'))}|{str(r.get('config_signature'))}" in labels,
                        axis=1,
                    )
                    selected_parts.append(selected[mask].copy())

            if selected_parts:
                selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=[c for c in ["run_id", "feature_set_name", "config_signature", "model_version"] if c in selected.columns])
            else:
                return None

        if selected.empty:
            return None

        pairs: set[tuple[str, str]] = set()
        labels: set[str] = set()
        run_ids: set[str] = set()
        model_versions: set[str] = set()
        parent_sweep_ids: set[str] = set()
        if {"feature_set_name", "config_signature"}.issubset(selected.columns):
            for fs, cs in selected[["feature_set_name", "config_signature"]].dropna().itertuples(index=False, name=None):
                fs_s, cs_s = str(fs), str(cs)
                pairs.add((fs_s, cs_s))
                labels.add(f"{fs_s}|{cs_s}")
        if "run_id" in selected.columns:
            run_ids = set(selected["run_id"].dropna().astype(str).tolist())
        if "model_version" in selected.columns:
            model_versions = set(selected["model_version"].dropna().astype(str).tolist())
        if "parent_sweep_id" in selected.columns:
            parent_sweep_ids = {
                str(v)
                for v in selected["parent_sweep_id"].map(self._normalize_sweep_id).dropna().tolist()
            }

        return {
            "pairs": pairs,
            "labels": labels,
            "run_ids": run_ids,
            "model_versions": model_versions,
            "parent_sweep_ids": parent_sweep_ids,
        }

    @staticmethod
    def _filter_df_by_scope(
        df: pd.DataFrame,
        scope: dict[str, set[str] | set[tuple[str, str]]] | None,
    ) -> pd.DataFrame:
        if df.empty or not scope:
            return df

        out = df.copy()
        pairs = scope.get("pairs", set())
        labels = scope.get("labels", set())
        run_ids = scope.get("run_ids", set())
        model_versions = scope.get("model_versions", set())
        parent_sweep_ids = scope.get("parent_sweep_ids", set())

        if "parent_sweep_id" in out.columns and parent_sweep_ids:
            sweep_ids = out["parent_sweep_id"].map(GeneratePredictionAnalysisPlotsUseCase._normalize_sweep_id)
            out = out[sweep_ids.isin(parent_sweep_ids)].copy()
            if out.empty:
                return out

        if {"left_config", "right_config"}.issubset(out.columns) and labels:
            left = out["left_config"].astype(str)
            right = out["right_config"].astype(str)
            return out[left.isin(labels) & right.isin(labels)].copy()

        if {"feature_set_name", "config_signature"}.issubset(out.columns) and pairs:
            mask = out.apply(
                lambda r: (str(r.get("feature_set_name")), str(r.get("config_signature"))) in pairs,
                axis=1,
            )
            return out[mask].copy()

        if "config_label" in out.columns and labels:
            return out[out["config_label"].astype(str).isin(labels)].copy()

        if "run_id" in out.columns and run_ids:
            return out[out["run_id"].astype(str).isin(run_ids)].copy()

        if "model_version" in out.columns and model_versions:
            return out[out["model_version"].astype(str).isin(model_versions)].copy()

        return out

    def _build_fig_heatmap_metrics_by_horizon(
        self,
        *,
        path: Path,
        metrics_by_config: pd.DataFrame,
        horizons: list[int],
        top_n_configs: int,
    ) -> None:
        plt = _require_pyplot()
        df = metrics_by_config.copy()
        if df.empty or not {"split", "horizon", "mean_rmse", "mean_mae", "mean_directional_accuracy", "mean_mean_pinball"}.issubset(df.columns):
            self._save_no_data(path, "Heatmap Metrics by Horizon", "insufficient data in gold_prediction_metrics_by_config")
            return

        df = df[df["split"].astype(str) == "test"].copy()
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df = df[df["horizon"].isin(horizons)].copy()
        if df.empty:
            self._save_no_data(path, "Heatmap Metrics by Horizon", "no test rows for selected horizons")
            return

        df["config_label"] = self._config_label(df)

        rank = (
            df.groupby("config_label", dropna=False)["mean_rmse"].mean().sort_values(ascending=True)
        )
        keep = set(rank.head(max(1, top_n_configs)).index.tolist())
        df = df[df["config_label"].isin(keep)].copy()

        metrics = [
            ("mean_rmse", "RMSE"),
            ("mean_mae", "MAE"),
            ("mean_directional_accuracy", "DA"),
            ("mean_mean_pinball", "Pinball"),
        ]

        labels = sorted(df["config_label"].unique().tolist())
        display_labels = self._dedupe_labels([self._short_config_label(lb) for lb in labels])
        hs = sorted(set(int(x) for x in df["horizon"].dropna().tolist()))
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        axes = axes.flatten()

        for ax, (metric_col, title) in zip(axes, metrics):
            piv = (
                df.pivot_table(index="config_label", columns="horizon", values=metric_col, aggfunc="mean")
                .reindex(index=labels, columns=hs)
            )
            vals = piv.to_numpy(dtype=float)
            im = ax.imshow(vals, aspect="auto", interpolation="nearest")
            ax.set_title(title)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(display_labels, fontsize=7)
            ax.set_xticks(range(len(hs)))
            ax.set_xticklabels([f"h+{h}" for h in hs])
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if math.isfinite(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6, color="white")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

        fig.suptitle("fig_heatmap_metrics_by_horizon", fontsize=14)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_boxplot_error_by_fold_seed(
        self,
        *,
        path: Path,
        metrics_by_run: pd.DataFrame,
        metrics_by_config: pd.DataFrame,
        horizons: list[int],
        top_n_configs: int,
    ) -> None:
        plt = _require_pyplot()
        df = metrics_by_run.copy()
        if df.empty or not {"split", "horizon", "rmse", "feature_set_name", "config_signature"}.issubset(df.columns):
            self._save_no_data(path, "Boxplot Error by Fold/Seed", "insufficient data in gold_prediction_metrics_by_run_split_horizon")
            return
        df = df[df["split"].astype(str) == "test"].copy()
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df = df[df["horizon"].isin(horizons)].copy()
        df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
        df = df.dropna(subset=["horizon", "rmse"]).copy()
        if df.empty:
            self._save_no_data(path, "Boxplot Error by Fold/Seed", "no test rmse rows for selected horizons")
            return

        cfg = metrics_by_config.copy()
        cfg = cfg[cfg["split"].astype(str) == "test"].copy() if (not cfg.empty and "split" in cfg.columns) else pd.DataFrame()
        if not cfg.empty and {"mean_rmse", "feature_set_name", "config_signature"}.issubset(cfg.columns):
            cfg["config_label"] = self._config_label(cfg)
            keep_labels = set(cfg.groupby("config_label")["mean_rmse"].mean().sort_values().head(max(1, top_n_configs)).index.tolist())
        else:
            tmp = df.copy()
            tmp["config_label"] = self._config_label(tmp)
            keep_labels = set(tmp.groupby("config_label")["rmse"].mean().sort_values().head(max(1, top_n_configs)).index.tolist())

        df["config_label"] = self._config_label(df)
        df = df[df["config_label"].isin(keep_labels)].copy()
        if df.empty:
            self._save_no_data(path, "Boxplot Error by Fold/Seed", "no rows after top config filter")
            return

        hs = sorted(set(int(x) for x in df["horizon"].tolist()))
        fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 5), constrained_layout=True)
        if len(hs) == 1:
            axes = [axes]
        for ax, h in zip(axes, hs):
            d = df[df["horizon"] == h].copy()
            labels = sorted(d["config_label"].unique().tolist())
            series = [pd.to_numeric(d[d["config_label"] == lb]["rmse"], errors="coerce").dropna().to_numpy() for lb in labels]
            series = [s for s in series if len(s) > 0]
            if not series:
                ax.set_title(f"h+{h} (no data)")
                ax.axis("off")
                continue
            display_labels = [self._short_config_label(lb) for lb in labels]
            ax.boxplot(series, tick_labels=display_labels, showfliers=False)
            ax.tick_params(axis="x", rotation=35, labelsize=7)
            ax.set_title(f"h+{h}")
            ax.set_ylabel("RMSE")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.suptitle("fig_boxplot_error_by_fold_seed", fontsize=14)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_dm_pvalue_matrix(
        self,
        *,
        path: Path,
        dm_df: pd.DataFrame,
        horizons: list[int],
    ) -> None:
        plt = _require_pyplot()
        df = dm_df.copy()
        if df.empty or not {"horizon", "left_config", "right_config"}.issubset(df.columns):
            self._save_no_data(path, "DM P-Value Matrix", "gold_dm_pairwise_results empty or missing columns")
            return
        pcol = "pvalue_adj_holm" if "pvalue_adj_holm" in df.columns else ("pvalue_two_sided" if "pvalue_two_sided" in df.columns else None)
        if pcol is None:
            self._save_no_data(path, "DM P-Value Matrix", "no p-value column found")
            return

        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df[pcol] = pd.to_numeric(df[pcol], errors="coerce")
        df = df.dropna(subset=["horizon", pcol]).copy()
        df = df[df["horizon"].isin(horizons)].copy()
        if df.empty:
            self._save_no_data(path, "DM P-Value Matrix", "no DM rows for selected horizons")
            return

        hs = sorted(set(int(x) for x in df["horizon"].tolist()))
        fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 5), constrained_layout=True)
        if len(hs) == 1:
            axes = [axes]

        for ax, h in zip(axes, hs):
            d = df[df["horizon"] == h].copy()
            cfgs = sorted(set(d["left_config"].astype(str)).union(set(d["right_config"].astype(str))))
            n = len(cfgs)
            mat = np.full((n, n), np.nan)
            idx = {c: i for i, c in enumerate(cfgs)}
            for _, r in d.iterrows():
                i = idx.get(str(r["left_config"]))
                j = idx.get(str(r["right_config"]))
                if i is None or j is None:
                    continue
                p = float(r[pcol])
                mat[i, j] = p
                mat[j, i] = p
            for i in range(n):
                mat[i, i] = 0.0
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
            ax.set_title(f"h+{h}")
            disp = self._dedupe_labels([self._short_config_label(c) for c in cfgs])
            if n > 30:
                step = max(1, n // 20)
                ticks = list(range(0, n, step))
            else:
                ticks = list(range(n))
            ax.set_xticks(ticks)
            ax.set_xticklabels([disp[i] for i in ticks], rotation=35, ha="right", fontsize=6)
            ax.set_yticks(ticks)
            ax.set_yticklabels([disp[i] for i in ticks], fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        fig.suptitle("fig_dm_pvalue_matrix", fontsize=14)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_calibration_curve(self, *, path: Path, calibration_df: pd.DataFrame, horizons: list[int]) -> None:
        plt = _require_pyplot()
        df = calibration_df.copy()
        if df.empty or not {"split", "horizon", "coverage_nominal", "picp"}.issubset(df.columns):
            self._save_no_data(path, "Calibration Curve", "insufficient data in gold_prediction_calibration")
            return
        df = df[df["split"].astype(str) == "test"].copy()
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["coverage_nominal"] = pd.to_numeric(df["coverage_nominal"], errors="coerce")
        df["picp"] = pd.to_numeric(df["picp"], errors="coerce")
        df = df.dropna(subset=["horizon", "coverage_nominal", "picp"]).copy()
        df = df[df["horizon"].isin(horizons)].copy()
        if df.empty:
            self._save_no_data(path, "Calibration Curve", "no calibration rows for selected horizons")
            return

        agg = df.groupby("horizon", dropna=False).agg(
            nominal=("coverage_nominal", "mean"),
            observed=("picp", "mean"),
        ).reset_index()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
        for _, r in agg.iterrows():
            h = int(r["horizon"])
            x = float(r["nominal"])
            y = float(r["observed"])
            ax.scatter([x], [y], s=60, label=f"h+{h}")
            ax.text(x, y, f" h+{h}", fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Observed coverage (PICP)")
        ax.set_title("fig_calibration_curve")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_interval_width_vs_coverage(
        self,
        *,
        path: Path,
        metrics_by_config: pd.DataFrame,
        horizons: list[int],
    ) -> None:
        plt = _require_pyplot()
        df = metrics_by_config.copy()
        req = {"split", "horizon", "mean_mpiw", "mean_picp", "feature_set_name", "config_signature"}
        if df.empty or not req.issubset(df.columns):
            self._save_no_data(path, "Interval Width vs Coverage", "insufficient data in gold_prediction_metrics_by_config")
            return

        df = df[df["split"].astype(str) == "test"].copy()
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["mean_mpiw"] = pd.to_numeric(df["mean_mpiw"], errors="coerce")
        df["mean_picp"] = pd.to_numeric(df["mean_picp"], errors="coerce")
        df = df.dropna(subset=["horizon", "mean_mpiw", "mean_picp"]).copy()
        df = df[df["horizon"].isin(horizons)].copy()
        if df.empty:
            self._save_no_data(path, "Interval Width vs Coverage", "no rows for selected horizons")
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        for h in sorted(set(int(x) for x in df["horizon"].tolist())):
            d = df[df["horizon"] == h]
            ax.scatter(d["mean_mpiw"], d["mean_picp"], s=20, alpha=0.6, label=f"h+{h}")
        ax.set_xlabel("Mean MPIW (interval width)")
        ax.set_ylabel("Mean PICP (coverage)")
        ax.set_title("fig_interval_width_vs_coverage")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_oos_timeseries_examples(
        self,
        *,
        path: Path,
        oos_df: pd.DataFrame,
        decision_df: pd.DataFrame,
        metrics_by_config: pd.DataFrame,
        horizons: list[int],
        max_points: int,
    ) -> None:
        plt = _require_pyplot()
        if oos_df.empty or "split" not in oos_df.columns:
            self._save_no_data(path, "OOS Timeseries Examples", "gold_oos_consolidated empty")
            return

        oos = oos_df[oos_df["split"].astype(str) == "test"].copy()
        if oos.empty:
            self._save_no_data(path, "OOS Timeseries Examples", "no test rows in gold_oos_consolidated")
            return

        oos["horizon"] = pd.to_numeric(oos.get("horizon"), errors="coerce")
        oos["target_timestamp_utc"] = pd.to_datetime(oos.get("target_timestamp_utc"), utc=True, errors="coerce")
        for c in ["y_true", "y_pred", "quantile_p10", "quantile_p90"]:
            if c in oos.columns:
                oos[c] = pd.to_numeric(oos[c], errors="coerce")

        def pick_config(h: int) -> str | None:
            if not decision_df.empty and {"horizon", "feature_set_name", "config_signature", "rank_rmse"}.issubset(decision_df.columns):
                d = decision_df.copy()
                d["horizon"] = pd.to_numeric(d["horizon"], errors="coerce")
                d["rank_rmse"] = pd.to_numeric(d["rank_rmse"], errors="coerce")
                d = d[d["horizon"] == h].sort_values("rank_rmse", ascending=True)
                if not d.empty:
                    r = d.iloc[0]
                    return f"{r['feature_set_name']}|{r['config_signature']}"
            if not metrics_by_config.empty and {"split", "horizon", "feature_set_name", "config_signature", "mean_rmse"}.issubset(metrics_by_config.columns):
                m = metrics_by_config.copy()
                m = m[m["split"].astype(str) == "test"].copy()
                m["horizon"] = pd.to_numeric(m["horizon"], errors="coerce")
                m["mean_rmse"] = pd.to_numeric(m["mean_rmse"], errors="coerce")
                m = m[m["horizon"] == h].sort_values("mean_rmse", ascending=True)
                if not m.empty:
                    r = m.iloc[0]
                    return f"{r['feature_set_name']}|{r['config_signature']}"
            return None

        fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4 * len(horizons)), constrained_layout=True)
        if len(horizons) == 1:
            axes = [axes]

        for ax, h in zip(axes, horizons):
            cfg = pick_config(int(h))
            if cfg is None:
                ax.axis("off")
                ax.text(0.5, 0.5, f"h+{h}: no config available", ha="center", va="center")
                continue
            fs, cs = cfg.split("|", 1)
            d = oos[(oos["horizon"] == int(h)) & (oos["feature_set_name"].astype(str) == fs) & (oos["config_signature"].astype(str) == cs)].copy()
            d = d.dropna(subset=["target_timestamp_utc", "y_true", "y_pred"]).sort_values("target_timestamp_utc")
            if d.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"h+{h}: no OOS rows for {cfg}", ha="center", va="center")
                continue
            if len(d) > max_points:
                d = d.tail(max_points)
            ax.plot(d["target_timestamp_utc"], d["y_true"], label="y_true", linewidth=1.4)
            ax.plot(d["target_timestamp_utc"], d["y_pred"], label="y_pred/p50", linewidth=1.4)
            if {"quantile_p10", "quantile_p90"}.issubset(d.columns):
                q = d.dropna(subset=["quantile_p10", "quantile_p90"])
                if not q.empty:
                    ax.fill_between(q["target_timestamp_utc"], q["quantile_p10"], q["quantile_p90"], alpha=0.2, label="p10-p90")
            ax.set_title(f"h+{h} | {cfg}")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        fig.suptitle("fig_oos_timeseries_examples", fontsize=14)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_feature_importance_global(
        self,
        *,
        path: Path,
        impact_df: pd.DataFrame,
        horizons: list[int],
        top_k_features: int | None,
        figure_title: str = "fig_feature_importance_global",
    ) -> None:
        plt = _require_pyplot()
        df = impact_df.copy()
        req = {"split", "horizon", "feature_name", "mean_delta_rmse"}
        if df.empty or not req.issubset(df.columns):
            self._save_no_data(path, "Feature Importance Global", "insufficient data in gold_feature_impact_by_horizon")
            return

        df = df[df["split"].astype(str) == "test"].copy()
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["mean_delta_rmse"] = pd.to_numeric(df["mean_delta_rmse"], errors="coerce")
        df = df.dropna(subset=["horizon", "feature_name", "mean_delta_rmse"]).copy()
        df = df[df["horizon"].isin(horizons)].copy()
        if df.empty:
            self._save_no_data(path, "Feature Importance Global", "no feature impact rows for selected horizons")
            return

        hs = sorted(set(int(x) for x in df["horizon"].tolist()))
        fig, axes = plt.subplots(1, len(hs), figsize=(6 * len(hs), 5), constrained_layout=True)
        if len(hs) == 1:
            axes = [axes]

        limit = None
        if top_k_features is not None:
            k = int(top_k_features)
            if k > 0:
                limit = max(1, k)

        for ax, h in zip(axes, hs):
            d = (
                df[df["horizon"] == h]
                .groupby("feature_name", dropna=False)["mean_delta_rmse"]
                .mean()
                .abs()
                .sort_values(ascending=False)
            )
            if limit is not None:
                d = d.head(limit)
            d = d.sort_values(ascending=True)
            if d.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"h+{h}: no data", ha="center", va="center")
                continue
            ax.barh(d.index.astype(str), d.values)
            ax.set_title(f"h+{h}")
            ax.set_xlabel("|mean_delta_rmse|")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        fig.suptitle(str(figure_title), fontsize=14)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def _build_fig_feature_contrib_local_cases(
        self,
        *,
        path: Path,
        local_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        top_k_features: int,
    ) -> None:
        plt = _require_pyplot()
        if local_df.empty:
            self._save_no_data(path, "Feature Contribution Local Cases", "fact_feature_contrib_local empty")
            return

        local = local_df.copy()
        req = {"horizon", "feature_name", "contribution", "abs_contribution", "timestamp_utc", "target_timestamp_utc"}
        if not req.issubset(local.columns):
            self._save_no_data(path, "Feature Contribution Local Cases", "missing columns in fact_feature_contrib_local")
            return

        local["run_id"] = local["run_id"].astype(str)
        local["horizon"] = pd.to_numeric(local["horizon"], errors="coerce")
        local["contribution"] = pd.to_numeric(local["contribution"], errors="coerce")
        local["abs_contribution"] = pd.to_numeric(local["abs_contribution"], errors="coerce")
        local["timestamp_utc"] = pd.to_datetime(local["timestamp_utc"], utc=True, errors="coerce")
        local["target_timestamp_utc"] = pd.to_datetime(local["target_timestamp_utc"], utc=True, errors="coerce")
        local = local.dropna(subset=["horizon", "feature_name", "contribution", "abs_contribution", "timestamp_utc", "target_timestamp_utc"]).copy()
        if local.empty:
            self._save_no_data(path, "Feature Contribution Local Cases", "no valid local contribution rows")
            return

        key_cols = ["run_id", "horizon", "timestamp_utc", "target_timestamp_utc"]
        case_scores = (
            local.groupby(key_cols, dropna=False)["abs_contribution"].sum().reset_index(name="proxy_abs_strength")
        )

        # If possible, enrich with true OOS abs_error for "maior erro/acerto" selection.
        if not oos_df.empty and {"run_id", "horizon", "timestamp_utc", "target_timestamp_utc", "abs_error"}.issubset(oos_df.columns):
            o = oos_df.copy()
            o["run_id"] = o["run_id"].astype(str)
            o["horizon"] = pd.to_numeric(o["horizon"], errors="coerce")
            o["timestamp_utc"] = pd.to_datetime(o["timestamp_utc"], utc=True, errors="coerce")
            o["target_timestamp_utc"] = pd.to_datetime(o["target_timestamp_utc"], utc=True, errors="coerce")
            o["abs_error"] = pd.to_numeric(o["abs_error"], errors="coerce")
            o = o.dropna(subset=["horizon", "timestamp_utc", "target_timestamp_utc", "abs_error"]).copy()
            o = o[["run_id", "horizon", "timestamp_utc", "target_timestamp_utc", "abs_error"]].drop_duplicates()
            case_scores = case_scores.merge(o, on=key_cols, how="left")
        else:
            case_scores["abs_error"] = np.nan

        use_error = case_scores["abs_error"].notna().any()
        score_col = "abs_error" if use_error else "proxy_abs_strength"

        worst = case_scores.sort_values(score_col, ascending=False).head(1)
        best = case_scores.sort_values(score_col, ascending=True).head(1)
        picks = [("worst_case", worst), ("best_case", best)]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, (label, pick_df) in zip(axes, picks):
            if pick_df.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{label}: no data", ha="center", va="center")
                continue
            pk = pick_df.iloc[0]
            cond = (
                (local["run_id"].astype(str) == str(pk.get("run_id")))
                & (local["horizon"] == float(pk.get("horizon")))
                & (local["timestamp_utc"] == pk.get("timestamp_utc"))
                & (local["target_timestamp_utc"] == pk.get("target_timestamp_utc"))
            )
            d = local[cond].copy()
            d = d.sort_values("abs_contribution", ascending=False).head(max(1, top_k_features)).sort_values("contribution")
            if d.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{label}: case empty", ha="center", va="center")
                continue
            colors = ["#2ca02c" if v >= 0 else "#d62728" for v in d["contribution"]]
            ax.barh(d["feature_name"].astype(str), d["contribution"], color=colors)
            s = float(pk.get(score_col)) if pd.notna(pk.get(score_col)) else float("nan")
            h = int(pk.get("horizon")) if pd.notna(pk.get("horizon")) else None
            metric_name = "abs_error" if use_error else "proxy_abs_strength"
            ax.set_title(f"{label} | h+{h} | {metric_name}={s:.4f}")
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        fig.suptitle("fig_feature_contrib_local_cases", fontsize=14)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    def execute(
        self,
        *,
        asset: str | None = None,
        top_n_configs: int = 12,
        top_k_features: int = 10,
        max_timeseries_points: int = 120,
        scope_csv_path: str | Path | None = None,
        scope_sweep_prefixes: list[str] | None = None,
    ) -> GeneratePredictionAnalysisPlotsResult:
        gold_cfg = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_prediction_metrics_by_config"), asset)
        gold_run = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_prediction_metrics_by_run_split_horizon"), asset)
        gold_dm = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_dm_pairwise_results"), asset)
        gold_cal = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_prediction_calibration"), asset)
        gold_oos = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_oos_consolidated"), asset)
        gold_dec = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_model_decision_final"), asset)
        gold_imp = self._filter_asset(self._load_gold_table(self.analytics_gold_dir, "gold_feature_impact_by_horizon"), asset)
        local_contrib = self._filter_asset(self._load_partitioned_table(self.analytics_silver_dir, "fact_feature_contrib_local"), asset)

        scope = self._build_scope_selection(
            scope_csv_path=scope_csv_path,
            scope_sweep_prefixes=scope_sweep_prefixes,
        )
        scope_requested = bool(scope_csv_path) or bool(scope_sweep_prefixes)
        if scope_requested and scope is None:
            raise ValueError(
                "Plot scope resolved no candidates. Check --plots-scope-csv / --plots-scope-sweep-prefixes."
            )

        gold_cfg = self._filter_df_by_scope(gold_cfg, scope)
        gold_run = self._filter_df_by_scope(gold_run, scope)
        gold_dm = self._filter_df_by_scope(gold_dm, scope)
        gold_cal = self._filter_df_by_scope(gold_cal, scope)
        gold_oos = self._filter_df_by_scope(gold_oos, scope)
        gold_dec = self._filter_df_by_scope(gold_dec, scope)
        gold_imp = self._filter_df_by_scope(gold_imp, scope)
        local_contrib = self._filter_df_by_scope(local_contrib, scope)

        horizons = self._pick_horizons(gold_cfg if not gold_cfg.empty else gold_run)

        outputs: dict[str, str] = {}
        tasks = [
            ("fig_heatmap_metrics_by_horizon", self._build_fig_heatmap_metrics_by_horizon, dict(metrics_by_config=gold_cfg, horizons=horizons, top_n_configs=top_n_configs)),
            ("fig_boxplot_error_by_fold_seed", self._build_fig_boxplot_error_by_fold_seed, dict(metrics_by_run=gold_run, metrics_by_config=gold_cfg, horizons=horizons, top_n_configs=top_n_configs)),
            ("fig_dm_pvalue_matrix", self._build_fig_dm_pvalue_matrix, dict(dm_df=gold_dm, horizons=horizons)),
            ("fig_calibration_curve", self._build_fig_calibration_curve, dict(calibration_df=gold_cal, horizons=horizons)),
            ("fig_interval_width_vs_coverage", self._build_fig_interval_width_vs_coverage, dict(metrics_by_config=gold_cfg, horizons=horizons)),
            ("fig_oos_timeseries_examples", self._build_fig_oos_timeseries_examples, dict(oos_df=gold_oos, decision_df=gold_dec, metrics_by_config=gold_cfg, horizons=horizons, max_points=max_timeseries_points)),
            ("fig_feature_importance_global", self._build_fig_feature_importance_global, dict(impact_df=gold_imp, horizons=horizons, top_k_features=top_k_features, figure_title="fig_feature_importance_global")),
            ("fig_feature_importance_global_all_features", self._build_fig_feature_importance_global, dict(impact_df=gold_imp, horizons=horizons, top_k_features=None, figure_title="fig_feature_importance_global_all_features")),
            ("fig_feature_contrib_local_cases", self._build_fig_feature_contrib_local_cases, dict(local_df=local_contrib, oos_df=gold_oos, top_k_features=top_k_features)),
        ]

        for name, fn, kwargs in tasks:
            out = self.output_dir / f"{name}.png"
            fn(path=out, **kwargs)
            outputs[name] = str(to_project_relative(out))

        manifest = {
            "asset": asset,
            "horizons": horizons,
            "top_n_configs": int(top_n_configs),
            "top_k_features": int(top_k_features),
            "max_timeseries_points": int(max_timeseries_points),
            "scope_csv_path": str(scope_csv_path) if scope_csv_path else None,
            "scope_sweep_prefixes": list(scope_sweep_prefixes or []),
            "scope_selected_labels": sorted(list(scope.get("labels", set()))) if scope else [],
            "outputs": outputs,
        }
        manifest_path = self.output_dir / "prediction_analysis_plots_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        outputs["manifest"] = str(to_project_relative(manifest_path))

        return GeneratePredictionAnalysisPlotsResult(
            output_dir=str(to_project_relative(self.output_dir)),
            outputs=outputs,
        )
