from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.domain.services.data_drift_analyzer import DataDriftAnalyzer
from src.domain.services.tft_sweep_experiment_builder import (
    SweepExperiment,
    build_one_at_a_time_experiments,
)
from src.infrastructure.schemas.tft_dataset_parquet_schema import (
    BASELINE_FEATURES,
    DEFAULT_TFT_FEATURES,
    FUNDAMENTAL_FEATURES,
    SENTIMENT_FEATURES,
    TECHNICAL_FEATURES,
)
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


DEFAULT_SWEEP_PARAM_RANGES: dict[str, list[Any]] = {
    "max_encoder_length": [120],
    "batch_size": [32, 64, 128],
    "max_epochs": [20, 30, 50],
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "hidden_size": [16, 32, 64],
    "attention_head_size": [1, 2, 4],
    "dropout": [0.05, 0.1, 0.2, 0.3],
    "hidden_continuous_size": [4, 8, 16],
    "early_stopping_patience": [3, 5, 10],
    "early_stopping_min_delta": [0.0, 1e-5, 1e-4],
}


@dataclass
class AnalysisRunRecord:
    fold_name: str
    run_label: str
    varied_param: str | None
    varied_value: Any | None
    config_signature: str
    version: str | None
    status: str
    error: str | None = None
    val_rmse: float | None = None
    val_mae: float | None = None
    test_rmse: float | None = None
    test_mae: float | None = None


@dataclass(frozen=True)
class RunTFTModelAnalysisResult:
    asset: str
    sweep_name: str
    sweep_dir: str
    runs_ok: int
    runs_failed: int
    top_5_runs: list[dict[str, Any]]


class RunTFTModelAnalysisUseCase:
    def __init__(
        self,
        *,
        train_runner: Any,
        base_training_config: dict[str, Any],
        param_ranges: dict[str, list[Any]] | None = None,
        replica_seeds: list[int] | None = None,
        split_config: dict[str, str] | None = None,
        compute_confidence_interval: bool = False,
        walk_forward_config: dict[str, Any] | None = None,
        generate_comparison_plots: bool = True,
    ) -> None:
        self.train_runner = train_runner
        self.base_training_config = dict(base_training_config)
        self.param_ranges = param_ranges or DEFAULT_SWEEP_PARAM_RANGES
        self.replica_seeds = replica_seeds or [7, 42, 123]
        self.split_config = dict(split_config or {})
        self.compute_confidence_interval = bool(compute_confidence_interval)
        self.walk_forward_config = dict(walk_forward_config or {})
        self.generate_comparison_plots = bool(generate_comparison_plots)

    @staticmethod
    def _parse_feature_tokens(features: str | None) -> list[str] | None:
        if features is None:
            return None
        tokens = [t.strip() for t in str(features).split(",") if t.strip()]
        return tokens or None

    @staticmethod
    def _resolve_features_for_drift(
        df: pd.DataFrame,
        features: list[str] | None,
    ) -> list[str]:
        base_exclude = {"timestamp", "asset_id", "target_return"}
        token_groups: dict[str, list[str]] = {
            "BASELINE_FEATURES": BASELINE_FEATURES,
            "TECHNICAL_FEATURES": TECHNICAL_FEATURES,
            "SENTIMENT_FEATURES": SENTIMENT_FEATURES,
            "FUNDAMENTAL_FEATURES": FUNDAMENTAL_FEATURES,
            "B": BASELINE_FEATURES,
            "T": TECHNICAL_FEATURES,
            "S": SENTIMENT_FEATURES,
            "F": FUNDAMENTAL_FEATURES,
        }
        if features is None:
            available = [c for c in DEFAULT_TFT_FEATURES if c in df.columns and c not in base_exclude]
            if not available:
                raise ValueError(
                    "Default feature set is empty or not found in dataset. "
                    f"Expected any of: {DEFAULT_TFT_FEATURES}"
                )
            return available

        selected: list[str] = []
        selected_set: set[str] = set()
        missing: list[str] = []
        for raw_token in features:
            token = raw_token.strip()
            upper = token.upper()
            if upper in token_groups:
                for col in token_groups[upper]:
                    if col in df.columns and col not in selected_set:
                        selected.append(col)
                        selected_set.add(col)
                continue

            col = token if token in df.columns else token.lower()
            if col not in df.columns:
                missing.append(token)
                continue
            if col not in selected_set:
                selected.append(col)
                selected_set.add(col)

        if missing:
            raise ValueError(f"Requested features not found in dataset: {missing}")
        selected = [c for c in selected if c not in base_exclude]
        if not selected:
            raise ValueError("No valid features resolved from provided --features")
        return selected

    @staticmethod
    def _parse_yyyymmdd(value: str) -> datetime:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)

    @staticmethod
    def _apply_time_split_for_drift(
        df: pd.DataFrame,
        *,
        split_config: dict[str, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        t_start = RunTFTModelAnalysisUseCase._parse_yyyymmdd(split_config["train_start"])
        t_end = RunTFTModelAnalysisUseCase._parse_yyyymmdd(split_config["train_end"])
        v_start = RunTFTModelAnalysisUseCase._parse_yyyymmdd(split_config["val_start"])
        v_end = RunTFTModelAnalysisUseCase._parse_yyyymmdd(split_config["val_end"])
        te_start = RunTFTModelAnalysisUseCase._parse_yyyymmdd(split_config["test_start"])
        te_end = RunTFTModelAnalysisUseCase._parse_yyyymmdd(split_config["test_end"])
        if not (t_start <= t_end < v_start <= v_end < te_start <= te_end):
            raise ValueError("Temporal split ranges must be sequential and non-overlapping")

        train_df = df[(ts >= t_start) & (ts <= t_end)].copy()
        val_df = df[(ts >= v_start) & (ts <= v_end)].copy()
        test_df = df[(ts >= te_start) & (ts <= te_end)].copy()
        return train_df, val_df, test_df

    def _resolve_walk_forward_folds(self) -> list[tuple[str, dict[str, str] | None]]:
        enabled = bool(self.walk_forward_config.get("enabled", False))
        raw_folds = self.walk_forward_config.get("folds")
        if not enabled or not isinstance(raw_folds, list) or not raw_folds:
            return [("default", self.split_config or None)]

        required_keys = {
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
        }
        folds: list[tuple[str, dict[str, str] | None]] = []
        for idx, item in enumerate(raw_folds, start=1):
            if not isinstance(item, dict):
                continue
            fold_name = str(item.get("name") or f"fold_{idx}").strip() or f"fold_{idx}"
            split = {k: str(item[k]) for k in required_keys if item.get(k) is not None}
            if required_keys.issubset(set(split.keys())):
                folds.append((fold_name, split))
        return folds or [("default", self.split_config or None)]

    @staticmethod
    def _load_existing_fold_split_map(summary_path: Path) -> dict[str, dict[str, str]]:
        if not summary_path.exists() or not summary_path.is_file():
            return {}
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        fold_summaries = payload.get("fold_summaries")
        if not isinstance(fold_summaries, list):
            return {}
        split_map: dict[str, dict[str, str]] = {}
        for item in fold_summaries:
            if not isinstance(item, dict):
                continue
            fold_name = str(item.get("fold_name") or "").strip()
            split_cfg = item.get("split_config")
            if not fold_name or not isinstance(split_cfg, dict):
                continue
            split_map[fold_name] = {str(k): str(v) for k, v in split_cfg.items() if v is not None}
        return split_map

    @staticmethod
    def _merge_run_records(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        if existing_df.empty:
            merged = new_df.copy()
        elif new_df.empty:
            merged = existing_df.copy()
        else:
            merged = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        dedup_cols = [
            c
            for c in ["fold_name", "run_label", "config_signature"]
            if c in merged.columns
        ]
        if dedup_cols:
            merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
        return merged.reset_index(drop=True)

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left == right:
            return True
        try:
            return float(left) == float(right)
        except Exception:
            return str(left) == str(right)

    @classmethod
    def _contains_value(cls, values: list[Any], target: Any) -> bool:
        for candidate in values:
            if cls._values_equal(candidate, target):
                return True
        return False

    @staticmethod
    def _strip_merge_allowed_fields(config: dict[str, Any]) -> dict[str, Any]:
        reduced = deepcopy(config)
        reduced.pop("merge_tests", None)
        reduced.pop("param_ranges", None)
        reduced.pop("replica_seeds", None)
        walk_forward = reduced.get("walk_forward")
        if isinstance(walk_forward, dict):
            walk_forward = dict(walk_forward)
            walk_forward.pop("folds", None)
            reduced["walk_forward"] = walk_forward
        return reduced

    @staticmethod
    def _diff_config_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
        diffs: list[str] = []
        if isinstance(left, dict) and isinstance(right, dict):
            keys = sorted(set(left.keys()) | set(right.keys()))
            for key in keys:
                key_str = str(key)
                path = f"{prefix}.{key_str}" if prefix else key_str
                if key not in left:
                    diffs.append(f"{path} (missing in existing)")
                    continue
                if key not in right:
                    diffs.append(f"{path} (missing in new)")
                    continue
                diffs.extend(RunTFTModelAnalysisUseCase._diff_config_paths(left[key], right[key], path))
            return diffs
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                diffs.append(
                    f"{prefix} (list length differs: existing={len(left)} new={len(right)})"
                )
                return diffs
            for idx, (l_item, r_item) in enumerate(zip(left, right)):
                path = f"{prefix}[{idx}]"
                diffs.extend(RunTFTModelAnalysisUseCase._diff_config_paths(l_item, r_item, path))
            return diffs
        if left != right:
            diffs.append(f"{prefix} (existing={left!r} new={right!r})")
        return diffs

    @classmethod
    def _validate_merge_analysis_config(
        cls,
        *,
        existing_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> None:
        existing_reduced = cls._strip_merge_allowed_fields(existing_config)
        new_reduced = cls._strip_merge_allowed_fields(new_config)
        if existing_reduced != new_reduced:
            diff_paths = cls._diff_config_paths(existing_reduced, new_reduced)
            diff_msg = "; ".join(diff_paths[:8]) if diff_paths else "unknown difference"
            raise ValueError(
                "merge_tests requires the same sweep configuration. "
                "Only param_ranges, walk_forward.folds, and replica_seeds may change. "
                f"Differences: {diff_msg}"
            )

        existing_ranges = existing_config.get("param_ranges") or {}
        new_ranges = new_config.get("param_ranges") or {}
        if isinstance(existing_ranges, dict):
            if not isinstance(new_ranges, dict):
                raise ValueError("merge_tests requires param_ranges to be an object")
            for param_name, old_values_raw in existing_ranges.items():
                old_values = list(old_values_raw) if isinstance(old_values_raw, list) else []
                if param_name not in new_ranges:
                    raise ValueError(
                        f"merge_tests cannot remove param_ranges entry: {param_name}"
                    )
                new_values_raw = new_ranges.get(param_name)
                new_values = list(new_values_raw) if isinstance(new_values_raw, list) else []
                for old_value in old_values:
                    if not cls._contains_value(new_values, old_value):
                        raise ValueError(
                            "merge_tests cannot remove old param_ranges values. "
                            f"Missing value for {param_name}: {old_value}"
                        )

        existing_seeds_raw = existing_config.get("replica_seeds") or []
        new_seeds_raw = new_config.get("replica_seeds") or []
        existing_seeds = list(existing_seeds_raw) if isinstance(existing_seeds_raw, list) else []
        new_seeds = list(new_seeds_raw) if isinstance(new_seeds_raw, list) else []
        for old_seed in existing_seeds:
            if not cls._contains_value(new_seeds, old_seed):
                raise ValueError(
                    "merge_tests cannot remove old replica_seeds values. "
                    f"Missing seed: {old_seed}"
                )

        existing_walk = existing_config.get("walk_forward") or {}
        new_walk = new_config.get("walk_forward") or {}
        existing_folds_raw = existing_walk.get("folds") if isinstance(existing_walk, dict) else []
        new_folds_raw = new_walk.get("folds") if isinstance(new_walk, dict) else []
        existing_folds = list(existing_folds_raw) if isinstance(existing_folds_raw, list) else []
        new_folds = list(new_folds_raw) if isinstance(new_folds_raw, list) else []

        old_by_name: dict[str, dict[str, Any]] = {}
        for fold in existing_folds:
            if not isinstance(fold, dict):
                continue
            name = str(fold.get("name") or "").strip()
            if name:
                old_by_name[name] = fold
        new_by_name: dict[str, dict[str, Any]] = {}
        for fold in new_folds:
            if not isinstance(fold, dict):
                continue
            name = str(fold.get("name") or "").strip()
            if name:
                new_by_name[name] = fold

        for fold_name, old_fold in old_by_name.items():
            if fold_name not in new_by_name:
                raise ValueError(f"merge_tests cannot remove fold: {fold_name}")
            if old_fold != new_by_name[fold_name]:
                raise ValueError(
                    "merge_tests cannot alter existing fold definition. "
                    f"Fold changed: {fold_name}"
                )

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        try:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _existing_run_key(*, fold_name: str, run_label: str, config_signature: str) -> tuple[str, str, str]:
        return (str(fold_name), str(run_label), str(config_signature))

    @classmethod
    def _build_existing_ok_index(cls, run_df: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
        if run_df.empty:
            return {}
        required = {"fold_name", "run_label", "config_signature", "status"}
        if not required.issubset(set(run_df.columns)):
            return {}
        ok_df = run_df[run_df["status"].astype(str) == "ok"].copy()
        if ok_df.empty:
            return {}
        idx: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in ok_df.to_dict(orient="records"):
            key = cls._existing_run_key(
                fold_name=str(row.get("fold_name", "")),
                run_label=str(row.get("run_label", "")),
                config_signature=str(row.get("config_signature", "")),
            )
            idx[key] = row
        return idx

    @staticmethod
    def _is_valid_saved_model_artifacts(
        *,
        version_dir: Path,
    ) -> tuple[bool, str | None, dict[str, Any] | None]:
        required_files = [
            "model_state.pt",
            "metrics.json",
            "split_metrics.json",
            "history.csv",
            "features.json",
            "config.json",
            "metadata.json",
        ]
        for name in required_files:
            path = version_dir / name
            if not path.exists() or not path.is_file():
                return False, f"missing required artifact: {name}", None
            try:
                if path.stat().st_size <= 0:
                    return False, f"empty artifact file: {name}", None
            except OSError:
                return False, f"unreadable artifact file: {name}", None

        metadata_path = version_dir / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return False, f"invalid metadata.json: {exc}", None
        if not isinstance(metadata, dict):
            return False, "metadata.json root must be an object", None
        split_metrics = metadata.get("split_metrics")
        if not isinstance(split_metrics, dict):
            return False, "metadata missing split_metrics", None
        for split_name in ["val", "test"]:
            metrics = split_metrics.get(split_name)
            if not isinstance(metrics, dict):
                return False, f"metadata missing split_metrics.{split_name}", None
            for metric_name in ["rmse", "mae"]:
                value = metrics.get(metric_name)
                try:
                    float(value)
                except Exception:
                    return False, f"invalid split_metrics.{split_name}.{metric_name}", None

        # Validate model_state.pt is loadable to avoid silently skipping a corrupt model.
        model_state_path = version_dir / "model_state.pt"
        try:
            import torch

            model_state = torch.load(model_state_path, map_location="cpu")
        except Exception as exc:
            return False, f"invalid model_state.pt: {exc}", None
        if not isinstance(model_state, dict):
            return False, "model_state.pt is not a state_dict dict", None
        if not model_state:
            return False, "model_state.pt state_dict is empty", None

        return True, None, metadata

    @staticmethod
    def _save_comparison_plots(
        *,
        sweep_dir: Path,
        run_df: pd.DataFrame,
        config_ranking: pd.DataFrame,
        impact_detail: pd.DataFrame,
        impact_summary: pd.DataFrame,
        param_ranges: dict[str, list[Any]] | None = None,
        baseline_config: dict[str, Any] | None = None,
    ) -> list[str]:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator
        except ImportError as exc:
            raise RuntimeError(
                "matplotlib is required to generate comparison charts. "
                "Install it with: pip install matplotlib"
            ) from exc

        plots_dir = sweep_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generated_paths: list[str] = []

        def _value_order_index(param_name: str, value: Any) -> int:
            if not param_ranges or param_name not in param_ranges:
                return 9999
            values = param_ranges.get(param_name, [])
            for idx, candidate in enumerate(values):
                if str(candidate) == str(value):
                    return idx
                try:
                    if float(candidate) == float(value):
                        return idx
                except Exception:
                    pass
            return 9999

        def _apply_config_order(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            ordered = df.copy()
            ordered["__is_baseline"] = ordered["varied_param"].isna().astype(int)
            ordered["__param_name"] = ordered["varied_param"].fillna("")
            ordered["__param_order"] = ordered["__param_name"].apply(
                lambda k: 0 if (param_ranges and k in param_ranges) else 9999
            )
            ordered["__value_order"] = ordered.apply(
                lambda r: -1
                if int(r["__is_baseline"]) == 1
                else _value_order_index(str(r["__param_name"]), r.get("varied_value")),
                axis=1,
            )
            ordered = ordered.sort_values(
                ["__is_baseline", "__param_order", "__value_order", "run_label"],
                ascending=[False, True, True, True],
            )
            return ordered.drop(columns=["__is_baseline", "__param_name", "__param_order", "__value_order"])

        def _save(fig: Any, filename: str) -> None:
            path = plots_dir / filename
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            generated_paths.append(str(path))

        def _empty_plot(title: str, filename: str, message: str) -> None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, message, ha="center", va="center")
            ax.set_axis_off()
            ax.set_title(title)
            _save(fig, filename)

        ok_runs = run_df[run_df["status"] == "ok"].copy() if not run_df.empty else pd.DataFrame()
        for col in ["val_rmse", "test_rmse", "val_mae", "test_mae"]:
            if col in ok_runs.columns:
                ok_runs[col] = pd.to_numeric(ok_runs[col], errors="coerce")

        # 1) Ranking with separate val/test panels to preserve scale readability.
        chart = _apply_config_order(config_ranking.copy())
        if chart.empty:
            _empty_plot(
                "Config Ranking (Val/Test)",
                "01_config_ranking_overlay.png",
                "No ranking data available.",
            )
        else:
            chart = chart.copy()
            labels = chart["run_label"].fillna("config").astype(str).tolist()
            x = np.arange(len(chart))
            fig_width = max(12, min(30, 0.55 * len(chart)))
            fig, (ax_val, ax_test) = plt.subplots(2, 1, figsize=(fig_width, 8), sharex=True)
            ax_val.errorbar(
                x,
                chart["mean_val_rmse"],
                yerr=chart["std_val_rmse"].fillna(0.0),
                fmt="-o",
                capsize=3,
                label="mean_val_rmse",
            )
            ax_test.errorbar(
                x,
                chart["mean_test_rmse"],
                yerr=chart["std_test_rmse"].fillna(0.0),
                fmt="-s",
                capsize=3,
                label="mean_test_rmse",
            )
            ax_test.set_xticks(x)
            ax_test.set_xticklabels(labels, rotation=45, ha="right")
            ax_val.set_ylabel("Val RMSE")
            ax_test.set_ylabel("Test RMSE")
            ax_test.set_xlabel("configuration")
            ax_val.set_title("Configuration Ranking: Val/Test in Separate Panels (with Std)")
            ax_val.legend()
            ax_test.legend()
            _save(fig, "01_config_ranking_overlay.png")

        # 2) Scatter val vs test per model configuration.
        if chart.empty and config_ranking.empty:
            _empty_plot(
                "Val vs Test RMSE Scatter",
                "02_val_vs_test_scatter.png",
                "No ranking data available.",
            )
        else:
            scatter_df = _apply_config_order(config_ranking.copy()).reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            if not scatter_df.empty:
                x = pd.to_numeric(scatter_df["mean_val_rmse"], errors="coerce")
                y = pd.to_numeric(scatter_df["mean_test_rmse"], errors="coerce")
                cmap = plt.cm.get_cmap("tab20", max(len(scatter_df), 1))
                for i, row in scatter_df.iterrows():
                    label = str(row.get("run_label", f"cfg_{i+1}"))
                    ax.scatter(
                        x.iloc[i],
                        y.iloc[i],
                        s=55,
                        alpha=0.9,
                        color=cmap(i % max(len(scatter_df), 1)),
                        label=label,
                    )
                ax.legend(
                    loc="best",
                    framealpha=0.35,
                    fontsize=7,
                    ncol=1 if len(scatter_df) <= 12 else 2,
                )
            ax.set_xlabel("mean_val_rmse")
            ax.set_ylabel("mean_test_rmse")
            ax.set_title("Val vs Test RMSE by Configuration")
            _save(fig, "02_val_vs_test_scatter.png")

        # 3) Parameter impact summary (RMSE/MAE deltas).
        if impact_summary.empty:
            _empty_plot(
                "Parameter Impact Summary",
                "03_param_impact_summary.png",
                "No impact summary available.",
            )
        else:
            summary = impact_summary.copy()
            for c in ["avg_delta_rmse_vs_baseline", "avg_delta_mae_vs_baseline"]:
                summary[c] = pd.to_numeric(summary[c], errors="coerce")
            n_params = len(summary)
            x = np.arange(n_params)
            fig, ax = plt.subplots(figsize=(10, 5))
            # For sweeps with few varied parameters (e.g., only max_encoder_length),
            # use thinner bars and extra x padding to avoid visually oversized blocks.
            if n_params <= 2:
                w = 0.08
            else:
                w = 0.22
            ax.bar(x - w / 2, summary["avg_delta_rmse_vs_baseline"], width=w, label="avg delta test RMSE")
            ax.bar(x + w / 2, summary["avg_delta_mae_vs_baseline"], width=w, label="avg delta test MAE")
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(summary["varied_param"].astype(str), rotation=35, ha="right")
            if n_params == 1:
                ax.set_xlim(-1.0, 1.0)
            elif n_params == 2:
                ax.set_xlim(-0.8, 1.8)
            ax.set_ylabel("Delta vs baseline")
            # Deltas can be positive and negative, so symlog is safer than pure log scale.
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.set_title("Parameter Impact Summary")
            ax.legend()
            _save(fig, "03_param_impact_summary.png")

        # 4) Parameter impact detail by value (boxplot deltas).
        if impact_detail.empty:
            _empty_plot(
                "Parameter Impact Detail",
                "04_param_impact_detail_boxplot.png",
                "No impact detail available.",
            )
        else:
            detail = impact_detail.copy()
            detail["delta_test_rmse_vs_baseline"] = pd.to_numeric(
                detail["delta_test_rmse_vs_baseline"], errors="coerce"
            )
            detail["label"] = (
                detail["varied_param"].astype(str) + "=" + detail["varied_value"].astype(str)
            )
            grouped = detail.groupby("label", dropna=False)["delta_test_rmse_vs_baseline"].apply(list)
            labels = grouped.index.tolist()
            if param_ranges and "varied_param" in detail.columns and "varied_value" in detail.columns:
                def _label_key(label: str) -> tuple[int, int, str]:
                    if "=" not in label:
                        return (9999, 9999, label)
                    param_name, value = label.split("=", 1)
                    if param_name in param_ranges:
                        return (0, _value_order_index(param_name, value), label)
                    return (9999, 9999, label)
                labels = sorted(labels, key=_label_key)
            values = [grouped[label] for label in labels]
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.boxplot(values, labels=labels, vert=True, patch_artist=True)
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("delta_test_rmse_vs_baseline")
            ax.set_title("Impact Detail Distribution by Parameter Value")
            _save(fig, "04_param_impact_detail_boxplot.png")

        # 5) Parameter trend by value (one subplot per parameter, same image).
        # Uses ranking metrics (mean_val_rmse / mean_test_rmse) per parameter value.
        if config_ranking.empty:
            _empty_plot(
                "Parameter Trend by Value",
                "05_param_trend_by_value.png",
                "No ranking data available.",
            )
        else:
            trend = config_ranking.copy()
            for c in ["mean_val_rmse", "mean_test_rmse"]:
                if c in trend.columns:
                    trend[c] = pd.to_numeric(trend[c], errors="coerce")
            baseline_rows = trend[trend["varied_param"].isna()].copy()
            baseline_val_mean: float | None = None
            baseline_test_mean: float | None = None
            if not baseline_rows.empty:
                baseline_val_mean = float(pd.to_numeric(baseline_rows["mean_val_rmse"], errors="coerce").mean())
                baseline_test_mean = float(pd.to_numeric(baseline_rows["mean_test_rmse"], errors="coerce").mean())

            trend = trend.dropna(subset=["varied_param", "varied_value", "mean_val_rmse", "mean_test_rmse"])
            if trend.empty:
                _empty_plot(
                    "Parameter Trend by Value",
                    "05_param_trend_by_value.png",
                    "No parameter variation data available.",
                )
            else:
                def _value_key(v: Any) -> Any:
                    try:
                        return float(v)
                    except Exception:
                        return str(v).strip()

                def _param_key(name: str) -> tuple[int, str]:
                    if param_ranges and name in param_ranges:
                        return (0, str(name))
                    return (1, name)

                if param_ranges:
                    params = [str(k) for k in param_ranges.keys()]
                    present_params = trend["varied_param"].astype(str).unique().tolist()
                    for p in sorted(present_params):
                        if p not in params:
                            params.append(p)
                else:
                    params = sorted(trend["varied_param"].astype(str).unique().tolist(), key=_param_key)

                n_params = len(params)
                n_cols = 3 if n_params >= 3 else n_params
                n_rows = int(np.ceil(n_params / max(n_cols, 1)))
                fig, axes = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(5 * max(n_cols, 1), 3.6 * max(n_rows, 1)),
                    squeeze=False,
                )
                axes_flat = axes.flatten()

                for idx, param_name in enumerate(params):
                    ax = axes_flat[idx]
                    part = trend[trend["varied_param"].astype(str) == param_name].copy()
                    if part.empty:
                        ax.set_title(param_name)
                        ax.text(0.5, 0.5, "No data", ha="center", va="center")
                        ax.set_axis_off()
                        continue

                    # Build ordered x values with available tested points plus baseline.
                    available_values = part["varied_value"].dropna().unique().tolist()
                    baseline_param_value = None
                    if isinstance(baseline_config, dict) and param_name in baseline_config:
                        baseline_param_value = baseline_config.get(param_name)
                    if baseline_param_value is not None:
                        keys_now = {_value_key(v) for v in available_values}
                        if _value_key(baseline_param_value) not in keys_now:
                            available_values.append(baseline_param_value)

                    if param_ranges and param_name in param_ranges:
                        ordered_values_raw = []
                        available_keys = {_value_key(v) for v in available_values}
                        for v in param_ranges.get(param_name, []):
                            if _value_key(v) in available_keys:
                                ordered_values_raw.append(v)
                    else:
                        ordered_values_raw = available_values
                        try:
                            ordered_values_raw = sorted(
                                ordered_values_raw,
                                key=lambda x: float(x),
                            )
                        except Exception:
                            ordered_values_raw = sorted([str(v) for v in ordered_values_raw])

                    grouped = (
                        part.groupby(part["varied_value"].apply(_value_key), dropna=False)
                        .agg(
                            mean_val_rmse=("mean_val_rmse", "mean"),
                            mean_test_rmse=("mean_test_rmse", "mean"),
                        )
                        .reset_index()
                        .rename(columns={"varied_value": "value_key"})
                    )
                    val_map = {
                        _value_key(r["value_key"]): float(r["mean_val_rmse"])
                        for _, r in grouped.iterrows()
                    }
                    test_map = {
                        _value_key(r["value_key"]): float(r["mean_test_rmse"])
                        for _, r in grouped.iterrows()
                    }

                    labels = [str(v) for v in ordered_values_raw]
                    keys = [_value_key(v) for v in ordered_values_raw]
                    y_val = [
                        baseline_val_mean if (baseline_param_value is not None and k == _value_key(baseline_param_value))
                        else val_map.get(k, np.nan)
                        for k in keys
                    ]
                    y_test = [
                        baseline_test_mean if (baseline_param_value is not None and k == _value_key(baseline_param_value))
                        else test_map.get(k, np.nan)
                        for k in keys
                    ]
                    x = np.arange(len(labels))
                    # Draw connected lines for all points.
                    ax.plot(x, y_val, linestyle="-", linewidth=1.5, label="mean_val_rmse")
                    ax.plot(x, y_test, linestyle="-", linewidth=1.5, label="mean_test_rmse")

                    # Markers: default for non-baseline points, special marker for baseline point.
                    baseline_x = None
                    if baseline_param_value is not None:
                        try:
                            baseline_x = keys.index(_value_key(baseline_param_value))
                        except ValueError:
                            baseline_x = None
                    normal_x = [xi for xi in x if baseline_x is None or xi != baseline_x]
                    if normal_x:
                        ax.scatter(
                            normal_x,
                            [y_val[xi] for xi in normal_x],
                            marker="o",
                            s=35,
                            zorder=4,
                        )
                        ax.scatter(
                            normal_x,
                            [y_test[xi] for xi in normal_x],
                            marker="s",
                            s=35,
                            zorder=4,
                        )
                    if baseline_x is not None:
                        ax.scatter(
                            [baseline_x],
                            [y_val[baseline_x]],
                            marker="*",
                            s=180,
                            color="black",
                            edgecolor="white",
                            linewidth=0.8,
                            zorder=6,
                        )
                        ax.scatter(
                            [baseline_x],
                            [y_test[baseline_x]],
                            marker="*",
                            s=180,
                            color="black",
                            edgecolor="white",
                            linewidth=0.8,
                            zorder=6,
                        )
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=35, ha="right")
                    ax.set_title(param_name)
                    ax.set_ylabel("RMSE")
                    ax.legend(loc="best", framealpha=0.35, fontsize=8)

                for idx in range(n_params, len(axes_flat)):
                    axes_flat[idx].set_axis_off()

                fig.suptitle("Parameter Trend by Value (Mean Val/Test RMSE)", y=1.02)
                _save(fig, "05_param_trend_by_value.png")

        # 6) Run-by-run progression (val/test overlay).
        if ok_runs.empty:
            _empty_plot(
                "Run Order Metrics",
                "06_run_order_metrics.png",
                "No successful runs available.",
            )
        else:
            seq = ok_runs.reset_index(drop=True).copy()
            seq["run_idx"] = np.arange(1, len(seq) + 1)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(seq["run_idx"], seq["val_rmse"], marker="o", label="val_rmse")
            ax.plot(seq["run_idx"], seq["test_rmse"], marker="s", label="test_rmse")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("run order")
            ax.set_ylabel("RMSE")
            ax.set_title("Run-by-Run Metrics Progression")
            ax.legend()
            _save(fig, "06_run_order_metrics.png")

        # 7) Top-5 leaderboard (val/test/robust).
        if config_ranking.empty:
            _empty_plot(
                "Top-5 Leaderboard",
                "07_top5_leaderboard.png",
                "No ranking data available.",
            )
        else:
            top = RunTFTModelAnalysisUseCase._sort_config_ranking_for_performance(config_ranking).head(5).copy()
            labels = top["run_label"].fillna("config").astype(str).tolist()
            y = np.arange(len(top))
            fig, ax = plt.subplots(figsize=(11, 5))
            h = 0.25
            ax.barh(y - h, top["mean_val_rmse"], height=h, label="mean_val_rmse")
            ax.barh(y, top["mean_test_rmse"], height=h, label="mean_test_rmse")
            ax.barh(y + h, top["robust_score"], height=h, label="robust_score")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel("score")
            ax.set_title("Top-5 Configurations Comparison")
            ax.legend()
            _save(fig, "07_top5_leaderboard.png")

        # 8) Walk-forward comparison by fold (generated only for walk-forward runs).
        if "fold_name" in ok_runs.columns:
            fold_values = ok_runs["fold_name"].dropna().astype(str).unique().tolist()
            valid_folds = [f for f in fold_values if f and f != "default"]
            if len(valid_folds) >= 2:
                wf = ok_runs.copy()
                wf["fold_name"] = wf["fold_name"].astype(str)
                grouped = (
                    wf.groupby(["fold_name", "config_signature"], dropna=False)
                    .agg(
                        mean_val_rmse=("val_rmse", "mean"),
                        mean_test_rmse=("test_rmse", "mean"),
                        run_label=("run_label", "first"),
                    )
                    .reset_index()
                )
                # Limit to top configurations by performance ranking for readability.
                if not config_ranking.empty:
                    ranked = RunTFTModelAnalysisUseCase._sort_config_ranking_for_performance(config_ranking)
                    top_signatures = ranked["config_signature"].head(5).astype(str).tolist()
                else:
                    top_signatures = grouped["config_signature"].astype(str).drop_duplicates().head(5).tolist()
                grouped = grouped[grouped["config_signature"].astype(str).isin(top_signatures)].copy()
                fold_order = sorted(valid_folds)
                x = np.arange(len(fold_order))

                fig, (ax_val, ax_test) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                for sig in top_signatures:
                    part = grouped[grouped["config_signature"].astype(str) == str(sig)].copy()
                    if part.empty:
                        continue
                    label = str(part["run_label"].iloc[0]).split("|seed=")[0]
                    val_map = {
                        str(r["fold_name"]): float(r["mean_val_rmse"])
                        for _, r in part.iterrows()
                    }
                    test_map = {
                        str(r["fold_name"]): float(r["mean_test_rmse"])
                        for _, r in part.iterrows()
                    }
                    y_val = [val_map.get(f, np.nan) for f in fold_order]
                    y_test = [test_map.get(f, np.nan) for f in fold_order]
                    ax_val.plot(x, y_val, marker="o", label=label)
                    ax_test.plot(x, y_test, marker="s", label=label)

                ax_test.set_xticks(x)
                ax_test.set_xticklabels(fold_order, rotation=35, ha="right")
                ax_val.set_ylabel("mean_val_rmse")
                ax_test.set_ylabel("mean_test_rmse")
                ax_test.set_xlabel("walk-forward fold")
                ax_val.set_title("Walk-forward Fold Comparison (Top Configurations)")
                ax_val.legend(loc="best", framealpha=0.35, fontsize=8)
                ax_test.legend(loc="best", framealpha=0.35, fontsize=8)
                _save(fig, "08_walk_forward_by_fold.png")

        # 9) Drift by fold summary (generated when aggregated drift report exists).
        drift_by_fold_path = sweep_dir / "drift_ks_psi_by_fold.csv"
        if drift_by_fold_path.exists():
            try:
                drift_folds = pd.read_csv(drift_by_fold_path)
            except Exception:
                drift_folds = pd.DataFrame()
            if drift_folds.empty:
                _empty_plot(
                    "Drift Summary by Fold",
                    "09_drift_by_fold.png",
                    "No drift-by-fold data available.",
                )
            else:
                for col in [
                    "avg_ks_train_vs_val",
                    "avg_ks_train_vs_test",
                    "avg_psi_train_vs_val",
                    "avg_psi_train_vs_test",
                ]:
                    if col in drift_folds.columns:
                        drift_folds[col] = pd.to_numeric(drift_folds[col], errors="coerce")
                x_labels = drift_folds["fold_name"].astype(str).tolist() if "fold_name" in drift_folds.columns else [str(i + 1) for i in range(len(drift_folds))]
                x = np.arange(len(drift_folds))
                fig, (ax_ks, ax_psi) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                ax_ks.plot(x, drift_folds.get("avg_ks_train_vs_val", pd.Series(np.nan, index=drift_folds.index)), marker="o", label="KS train vs val")
                ax_ks.plot(x, drift_folds.get("avg_ks_train_vs_test", pd.Series(np.nan, index=drift_folds.index)), marker="s", label="KS train vs test")
                ax_ks.axhline(0.2, color="red", linewidth=1, linestyle="--", label="KS reference 0.2")
                ax_ks.set_ylabel("KS")
                ax_ks.set_title("Drift by Fold: KS and PSI")
                ax_ks.legend(loc="best", framealpha=0.35, fontsize=8)

                ax_psi.plot(x, drift_folds.get("avg_psi_train_vs_val", pd.Series(np.nan, index=drift_folds.index)), marker="o", label="PSI train vs val")
                ax_psi.plot(x, drift_folds.get("avg_psi_train_vs_test", pd.Series(np.nan, index=drift_folds.index)), marker="s", label="PSI train vs test")
                ax_psi.axhline(0.25, color="red", linewidth=1, linestyle="--", label="PSI reference 0.25")
                ax_psi.set_ylabel("PSI")
                ax_psi.set_xlabel("walk-forward fold")
                ax_psi.legend(loc="best", framealpha=0.35, fontsize=8)
                ax_psi.set_xticks(x)
                ax_psi.set_xticklabels(x_labels, rotation=35, ha="right")
                _save(fig, "09_drift_by_fold.png")

        # 10) Drift feature detail for a fold (generated when fold-level drift detail exists).
        drift_detail_path = sweep_dir / "drift_ks_psi_detail.csv"
        if drift_detail_path.exists():
            try:
                drift_detail_df = pd.read_csv(drift_detail_path)
            except Exception:
                drift_detail_df = pd.DataFrame()
            if drift_detail_df.empty:
                _empty_plot(
                    "Drift Detail by Feature",
                    "10_drift_feature_detail.png",
                    "No drift detail data available.",
                )
            else:
                for col in [
                    "ks_train_vs_val",
                    "ks_train_vs_test",
                    "psi_train_vs_val",
                    "psi_train_vs_test",
                ]:
                    if col in drift_detail_df.columns:
                        drift_detail_df[col] = pd.to_numeric(drift_detail_df[col], errors="coerce")
                labels = drift_detail_df["feature"].astype(str).tolist() if "feature" in drift_detail_df.columns else [str(i + 1) for i in range(len(drift_detail_df))]
                y = np.arange(len(drift_detail_df))
                fig, (ax_ks, ax_psi) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
                h = 0.35
                ax_ks.barh(y - h / 2, drift_detail_df.get("ks_train_vs_val", pd.Series(np.nan, index=drift_detail_df.index)), height=h, label="KS train vs val")
                ax_ks.barh(y + h / 2, drift_detail_df.get("ks_train_vs_test", pd.Series(np.nan, index=drift_detail_df.index)), height=h, label="KS train vs test")
                ax_ks.axvline(0.2, color="red", linewidth=1, linestyle="--", label="KS reference 0.2")
                ax_ks.set_ylabel("feature")
                ax_ks.set_title("Drift Detail by Feature: KS and PSI")
                ax_ks.legend(loc="best", framealpha=0.35, fontsize=8)
                ax_ks.set_yticks(y)
                ax_ks.set_yticklabels(labels)

                ax_psi.barh(y - h / 2, drift_detail_df.get("psi_train_vs_val", pd.Series(np.nan, index=drift_detail_df.index)), height=h, label="PSI train vs val")
                ax_psi.barh(y + h / 2, drift_detail_df.get("psi_train_vs_test", pd.Series(np.nan, index=drift_detail_df.index)), height=h, label="PSI train vs test")
                ax_psi.axvline(0.25, color="red", linewidth=1, linestyle="--", label="PSI reference 0.25")
                ax_psi.set_xlabel("value")
                ax_psi.set_yticks(y)
                ax_psi.set_yticklabels(labels)
                ax_psi.legend(loc="best", framealpha=0.35, fontsize=8)
                _save(fig, "10_drift_feature_detail.png")

        return generated_paths

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _config_signature(config: dict[str, Any]) -> str:
        normalized = {k: v for k, v in config.items() if k != "seed"}
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _collect_all_models(models_asset_dir: Path) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run_dir in sorted([p.parent for p in models_asset_dir.rglob("metadata.json")]):
            meta_path = run_dir / "metadata.json"
            try:
                meta = RunTFTModelAnalysisUseCase._load_metadata(meta_path)
            except Exception:
                continue
            split_metrics = meta.get("split_metrics", {})
            test_metrics = split_metrics.get("test", {})
            val_metrics = split_metrics.get("val", {})
            train_metrics = split_metrics.get("train", {})
            training_cfg = meta.get("training_config", {})
            rows.append(
                {
                    "version": meta.get("version", run_dir.name),
                    "created_at": meta.get("created_at"),
                    "config_signature": RunTFTModelAnalysisUseCase._config_signature(training_cfg),
                    "feature_set_tag": training_cfg.get("feature_set_tag"),
                    "features_count": len(meta.get("features_used", [])),
                    "test_rmse": test_metrics.get("rmse"),
                    "test_mae": test_metrics.get("mae"),
                    "val_rmse": val_metrics.get("rmse"),
                    "val_mae": val_metrics.get("mae"),
                    "train_rmse": train_metrics.get("rmse"),
                    "train_mae": train_metrics.get("mae"),
                    "learning_rate": training_cfg.get("learning_rate"),
                    "hidden_size": training_cfg.get("hidden_size"),
                    "max_encoder_length": training_cfg.get("max_encoder_length"),
                    "batch_size": training_cfg.get("batch_size"),
                    "dropout": training_cfg.get("dropout"),
                    "seed": training_cfg.get("seed"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["test_rmse", "test_mae"], ascending=[True, True]).reset_index(
                drop=True
            )
        return df

    @staticmethod
    def _ranking_sort_columns(df: pd.DataFrame) -> list[str]:
        return [
            c
            for c in ["robust_score", "mean_val_rmse", "mean_test_rmse", "mean_val_mae", "mean_test_mae"]
            if c in df.columns
        ]

    @classmethod
    def _sort_config_ranking_for_performance(cls, config_ranking: pd.DataFrame) -> pd.DataFrame:
        if config_ranking.empty:
            return config_ranking
        ordered = config_ranking.copy()
        for c in ["robust_score", "mean_val_rmse", "mean_test_rmse", "mean_val_mae", "mean_test_mae"]:
            if c in ordered.columns:
                ordered[c] = pd.to_numeric(ordered[c], errors="coerce")
        sort_cols = cls._ranking_sort_columns(ordered)
        if sort_cols:
            ordered = ordered.sort_values(sort_cols, ascending=[True] * len(sort_cols))
        return ordered.reset_index(drop=True)

    @classmethod
    def _sort_all_models_with_config_ranking(
        cls,
        all_models_df: pd.DataFrame,
        config_ranking: pd.DataFrame,
    ) -> pd.DataFrame:
        if all_models_df.empty:
            return all_models_df
        ordered = all_models_df.copy()
        if not config_ranking.empty and "config_signature" in ordered.columns and "config_signature" in config_ranking.columns:
            ranked = cls._sort_config_ranking_for_performance(config_ranking)
            if "robust_score" in ranked.columns:
                robust_map = (
                    ranked[["config_signature", "robust_score"]]
                    .drop_duplicates(subset=["config_signature"], keep="first")
                    .set_index("config_signature")["robust_score"]
                    .to_dict()
                )
                ordered["robust_score"] = ordered["config_signature"].map(robust_map)
        for col in ["robust_score", "val_rmse", "test_rmse", "val_mae", "test_mae"]:
            if col in ordered.columns:
                ordered[col] = pd.to_numeric(ordered[col], errors="coerce")
        sort_cols = [c for c in ["robust_score", "val_rmse", "test_rmse", "val_mae", "test_mae"] if c in ordered.columns]
        if sort_cols:
            ordered = ordered.sort_values(sort_cols, ascending=[True] * len(sort_cols))
        return ordered.reset_index(drop=True)

    @staticmethod
    def _build_param_impact(
        run_df: pd.DataFrame,
        baseline_test_rmse: float,
        baseline_test_mae: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if run_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        varied = run_df[run_df["varied_param"].notna()].copy()
        if varied.empty:
            return pd.DataFrame(), pd.DataFrame()
        varied["delta_test_rmse_vs_baseline"] = varied["test_rmse"] - baseline_test_rmse
        varied["delta_test_mae_vs_baseline"] = varied["test_mae"] - baseline_test_mae
        summary = (
            varied.groupby("varied_param", dropna=True)
            .agg(
                runs=("run_label", "count"),
                best_test_rmse=("test_rmse", "min"),
                avg_test_rmse=("test_rmse", "mean"),
                median_test_rmse=("test_rmse", "median"),
                best_delta_rmse_vs_baseline=("delta_test_rmse_vs_baseline", "min"),
                worst_delta_rmse_vs_baseline=("delta_test_rmse_vs_baseline", "max"),
                avg_delta_rmse_vs_baseline=("delta_test_rmse_vs_baseline", "mean"),
                avg_delta_mae_vs_baseline=("delta_test_mae_vs_baseline", "mean"),
            )
            .reset_index()
            .sort_values("best_delta_rmse_vs_baseline", ascending=True)
        )
        return varied.sort_values("test_rmse", ascending=True), summary

    @classmethod
    def _build_config_ranking(
        cls,
        run_df: pd.DataFrame,
        *,
        compute_confidence_interval: bool,
    ) -> pd.DataFrame:
        if run_df.empty:
            return pd.DataFrame()
        ok = run_df[run_df["status"] == "ok"].copy()
        if ok.empty:
            return pd.DataFrame()
        ok["val_rmse"] = pd.to_numeric(ok["val_rmse"], errors="coerce")
        ok["val_mae"] = pd.to_numeric(ok["val_mae"], errors="coerce")
        ok["test_rmse"] = pd.to_numeric(ok["test_rmse"], errors="coerce")
        ok["test_mae"] = pd.to_numeric(ok["test_mae"], errors="coerce")

        candidates = ok[ok["val_rmse"].notna()].copy()
        if candidates.empty:
            candidates = ok

        ranking = (
            candidates.groupby("config_signature", dropna=False)
            .agg(
                n_runs=("run_label", "count"),
                run_label=("run_label", "first"),
                varied_param=("varied_param", "first"),
                varied_value=("varied_value", "first"),
                mean_val_rmse=("val_rmse", "mean"),
                std_val_rmse=("val_rmse", "std"),
                mean_val_mae=("val_mae", "mean"),
                std_val_mae=("val_mae", "std"),
                mean_test_rmse=("test_rmse", "mean"),
                std_test_rmse=("test_rmse", "std"),
                mean_test_mae=("test_mae", "mean"),
                std_test_mae=("test_mae", "std"),
            )
            .reset_index()
        )

        # Keep run labels representative of all folds in the aggregated ranking.
        if "fold_name" in candidates.columns:
            all_folds = sorted(
                {
                    str(v).strip()
                    for v in candidates["fold_name"].dropna().astype(str).tolist()
                    if str(v).strip()
                }
            )
            fold_map = (
                candidates.groupby("config_signature", dropna=False)["fold_name"]
                .apply(
                    lambda s: ",".join(
                        sorted(
                            {
                                str(v).strip()
                                for v in s.dropna().astype(str).tolist()
                                if str(v).strip()
                            }
                        )
                    )
                )
                .to_dict()
            )
            ranking["run_label"] = ranking.apply(
                lambda r: (
                    f"{str(r['run_label']).split('|seed=')[0]}|folds="
                    f"{'all_folds' if sorted(fold_map.get(r['config_signature'], '').split(',')) == all_folds else fold_map.get(r['config_signature'], '')}"
                    if fold_map.get(r["config_signature"], "")
                    else str(r["run_label"]).split("|seed=")[0]
                ),
                axis=1,
            )
        else:
            ranking["run_label"] = ranking["run_label"].astype(str).map(
                lambda v: v.split("|seed=")[0]
            )
        for col in ["std_val_rmse", "std_val_mae", "std_test_rmse", "std_test_mae"]:
            ranking[col] = ranking[col].fillna(0.0)
        ranking["robust_score"] = ranking["mean_val_rmse"] + ranking["std_val_rmse"]
        if compute_confidence_interval:
            z = 1.96
            n = ranking["n_runs"].astype(float).replace(0.0, np.nan)
            ranking["ci95_val_rmse_low"] = ranking["mean_val_rmse"] - z * (
                ranking["std_val_rmse"] / np.sqrt(n)
            )
            ranking["ci95_val_rmse_high"] = ranking["mean_val_rmse"] + z * (
                ranking["std_val_rmse"] / np.sqrt(n)
            )
            ranking["ci95_test_rmse_low"] = ranking["mean_test_rmse"] - z * (
                ranking["std_test_rmse"] / np.sqrt(n)
            )
            ranking["ci95_test_rmse_high"] = ranking["mean_test_rmse"] + z * (
                ranking["std_test_rmse"] / np.sqrt(n)
            )
        return cls._sort_config_ranking_for_performance(ranking)

    @staticmethod
    def _param_value_order_index(
        *,
        param_ranges: dict[str, list[Any]] | None,
        param_name: str,
        value: Any,
    ) -> int:
        if not param_ranges or param_name not in param_ranges:
            return 9999
        for idx, candidate in enumerate(param_ranges.get(param_name, [])):
            if str(candidate) == str(value):
                return idx
            try:
                if float(candidate) == float(value):
                    return idx
            except Exception:
                continue
        return 9999

    @staticmethod
    def _order_config_ranking_for_report(
        config_ranking: pd.DataFrame,
        *,
        param_ranges: dict[str, list[Any]] | None,
    ) -> pd.DataFrame:
        if config_ranking.empty:
            return config_ranking
        ordered = config_ranking.copy()
        ordered["__is_baseline"] = ordered["varied_param"].isna().astype(int)
        ordered["__param_name"] = ordered["varied_param"].fillna("").astype(str)
        ordered["__param_order"] = ordered["__param_name"].apply(
            lambda k: 0 if (param_ranges and k in param_ranges) else 9999
        )
        ordered["__value_order"] = ordered.apply(
            lambda r: -1
            if int(r["__is_baseline"]) == 1
            else RunTFTModelAnalysisUseCase._param_value_order_index(
                param_ranges=param_ranges,
                param_name=str(r["__param_name"]),
                value=r.get("varied_value"),
            ),
            axis=1,
        )
        ordered = ordered.sort_values(
            ["__is_baseline", "__param_order", "__value_order", "run_label"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        return ordered.drop(columns=["__is_baseline", "__param_name", "__param_order", "__value_order"])

    def execute(
        self,
        *,
        asset: str,
        models_asset_dir: Path,
        features: str | None = None,
        continue_on_error: bool = False,
        merge_tests: bool = False,
        max_runs: int | None = None,
        output_subdir: str | None = None,
        analysis_config: dict[str, Any] | None = None,
    ) -> RunTFTModelAnalysisResult:
        asset = asset.strip().upper()
        models_asset_dir.mkdir(parents=True, exist_ok=True)
        sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sweep_name = output_subdir or f"sweep_{sweep_ts}"
        sweep_dir = models_asset_dir / "sweeps" / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_folds_dir = sweep_dir / "folds"
        sweep_folds_dir.mkdir(parents=True, exist_ok=True)
        existing_runs_path = sweep_dir / "sweep_runs.csv"
        existing_df = pd.DataFrame()
        existing_ok_index: dict[tuple[str, str, str], dict[str, Any]] = {}
        if merge_tests and existing_runs_path.exists() and existing_runs_path.is_file():
            try:
                existing_df = pd.read_csv(existing_runs_path)
                existing_ok_index = self._build_existing_ok_index(existing_df)
            except Exception:
                logger.exception("Failed to load existing sweep_runs.csv for merge/skip")
                existing_df = pd.DataFrame()
                existing_ok_index = {}

        analysis_config_path = sweep_dir / "analysis_config.json"
        if merge_tests and analysis_config_path.exists():
            if analysis_config is None:
                raise ValueError(
                    "merge_tests requires analysis_config to validate compatibility."
                )
            try:
                existing_cfg = json.loads(analysis_config_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid existing analysis_config.json in {sweep_dir}"
                ) from exc
            if not isinstance(existing_cfg, dict):
                raise ValueError("Existing analysis_config.json root must be an object")
            self._validate_merge_analysis_config(
                existing_config=existing_cfg,
                new_config=analysis_config,
            )
        if analysis_config is not None:
            analysis_config_path.write_text(
                json.dumps(analysis_config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        experiments = build_one_at_a_time_experiments(
            base_config=self.base_training_config,
            param_ranges=self.param_ranges,
        )
        if max_runs is not None:
            experiments = experiments[: max(1, max_runs)]

        logger.info(
            "Starting TFT model analysis",
            extra={
                "asset": asset,
                "runs": len(experiments),
                "features": features or "(default from main_train_tft)",
                "sweep_dir": str(sweep_dir),
            },
        )

        run_records: list[AnalysisRunRecord] = []
        folds = self._resolve_walk_forward_folds()
        fold_split_map: dict[str, dict[str, str]] = {
            name: dict(split or self.split_config) for name, split in folds
        }
        if merge_tests:
            existing_fold_splits = self._load_existing_fold_split_map(sweep_dir / "summary.json")
            for fold_name, split_cfg in existing_fold_splits.items():
                fold_split_map.setdefault(fold_name, split_cfg)

        dataset_df = pd.DataFrame()
        drift_feature_cols: list[str] = []
        try:
            paths = load_data_paths()
            dataset_repo = ParquetTFTDatasetRepository(output_dir=Path(paths["dataset_tft"]))
            dataset_df = dataset_repo.load(asset)
            feature_tokens = self._parse_feature_tokens(features)
            drift_feature_cols = self._resolve_features_for_drift(dataset_df, feature_tokens)
        except Exception:
            logger.exception(
                "Failed to initialize drift analysis dataset context",
                extra={"asset": asset},
            )

        for fold_name, fold_split_config in folds:
            fold_dir = sweep_folds_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_models_dir = fold_dir / "models"
            fold_models_dir.mkdir(parents=True, exist_ok=True)
            fold_staging_dir = fold_models_dir / asset
            fold_staging_dir.mkdir(parents=True, exist_ok=True)

            for idx, exp in enumerate(experiments, start=1):
                for seed in self.replica_seeds:
                    logger.info(
                        "Analysis run started",
                        extra={
                            "fold_name": fold_name,
                            "run_index": idx,
                            "total_runs": len(experiments),
                            "run_label": exp.run_label,
                            "seed": seed,
                        },
                    )
                    cfg_with_seed = dict(exp.config)
                    cfg_with_seed["seed"] = seed
                    run_label = f"{exp.run_label}|seed={seed}|fold={fold_name}"
                    config_signature = self._config_signature(cfg_with_seed)
                    existing_key = self._existing_run_key(
                        fold_name=fold_name,
                        run_label=run_label,
                        config_signature=config_signature,
                    )
                    if merge_tests and existing_key in existing_ok_index:
                        existing_row = existing_ok_index[existing_key]
                        existing_version = str(existing_row.get("version") or "").strip()
                        version_dir = fold_models_dir / existing_version if existing_version else None
                        if version_dir and version_dir.exists() and version_dir.is_dir():
                            valid, reason, metadata = self._is_valid_saved_model_artifacts(
                                version_dir=version_dir
                            )
                            if valid:
                                val_rmse = self._to_float_or_none(existing_row.get("val_rmse"))
                                val_mae = self._to_float_or_none(existing_row.get("val_mae"))
                                test_rmse = self._to_float_or_none(existing_row.get("test_rmse"))
                                test_mae = self._to_float_or_none(existing_row.get("test_mae"))
                                if metadata and (val_rmse is None or val_mae is None or test_rmse is None or test_mae is None):
                                    split_metrics = (
                                        metadata.get("split_metrics", {})
                                        if isinstance(metadata, dict)
                                        else {}
                                    )
                                    val_metrics = split_metrics.get("val", {}) if isinstance(split_metrics, dict) else {}
                                    test_metrics = split_metrics.get("test", {}) if isinstance(split_metrics, dict) else {}
                                    val_rmse = val_rmse if val_rmse is not None else self._to_float_or_none(val_metrics.get("rmse"))
                                    val_mae = val_mae if val_mae is not None else self._to_float_or_none(val_metrics.get("mae"))
                                    test_rmse = test_rmse if test_rmse is not None else self._to_float_or_none(test_metrics.get("rmse"))
                                    test_mae = test_mae if test_mae is not None else self._to_float_or_none(test_metrics.get("mae"))

                                run_records.append(
                                    AnalysisRunRecord(
                                        fold_name=fold_name,
                                        run_label=run_label,
                                        varied_param=exp.varied_param,
                                        varied_value=exp.varied_value,
                                        config_signature=config_signature,
                                        version=existing_version,
                                        status="ok",
                                        val_rmse=val_rmse,
                                        val_mae=val_mae,
                                        test_rmse=test_rmse,
                                        test_mae=test_mae,
                                    )
                                )
                                logger.info(
                                    "Skipping training for existing valid run",
                                    extra={
                                        "fold_name": fold_name,
                                        "run_label": run_label,
                                        "version_dir": str(version_dir),
                                    },
                                )
                                continue
                            logger.warning(
                                "Existing run artifacts invalid, retraining",
                                extra={
                                    "fold_name": fold_name,
                                    "run_label": run_label,
                                    "version_dir": str(version_dir),
                                    "reason": reason,
                                },
                            )
                        else:
                            logger.warning(
                                "Existing run version directory missing, retraining",
                                extra={
                                    "fold_name": fold_name,
                                    "run_label": run_label,
                                    "version": existing_version,
                                },
                            )
                    try:
                        version, metadata = self.train_runner.run(
                            asset=asset,
                            features=features,
                            config=cfg_with_seed,
                            split_config=fold_split_config,
                            models_asset_dir=fold_staging_dir,
                        )
                        if version is None or metadata is None:
                            run_records.append(
                                AnalysisRunRecord(
                                    fold_name=fold_name,
                                    run_label=run_label,
                                    varied_param=exp.varied_param,
                                    varied_value=exp.varied_value,
                                    config_signature=config_signature,
                                    version=version,
                                    status="failed",
                                    error="Missing generated model version or metadata.json",
                                )
                            )
                            if not continue_on_error:
                                break
                            continue

                        val_metrics = metadata.get("split_metrics", {}).get("val", {})
                        test_metrics = metadata.get("split_metrics", {}).get("test", {})
                        run_records.append(
                            AnalysisRunRecord(
                                fold_name=fold_name,
                                run_label=run_label,
                                varied_param=exp.varied_param,
                                varied_value=exp.varied_value,
                                config_signature=config_signature,
                                version=version,
                                status="ok",
                                val_rmse=float(val_metrics.get("rmse")),
                                val_mae=float(val_metrics.get("mae")),
                                test_rmse=float(test_metrics.get("rmse")),
                                test_mae=float(test_metrics.get("mae")),
                            )
                        )
                        staged_version_dir = fold_staging_dir / version
                        flat_version_dir = fold_models_dir / version
                        if staged_version_dir.exists():
                            if flat_version_dir.exists():
                                shutil.rmtree(flat_version_dir)
                            shutil.move(str(staged_version_dir), str(flat_version_dir))
                    except Exception as exc:
                        run_records.append(
                            AnalysisRunRecord(
                                fold_name=fold_name,
                                run_label=run_label,
                                varied_param=exp.varied_param,
                                varied_value=exp.varied_value,
                                config_signature=config_signature,
                                version=None,
                                status="failed",
                                error=str(exc),
                            )
                        )
                        logger.exception(
                            "Analysis run failed",
                            extra={"fold_name": fold_name, "run_label": exp.run_label, "seed": seed},
                        )
                        if not continue_on_error:
                            break
                if run_records and run_records[-1].status == "failed" and not continue_on_error:
                    break
            if fold_staging_dir.exists() and not any(fold_staging_dir.iterdir()):
                fold_staging_dir.rmdir()
            if run_records and run_records[-1].status == "failed" and not continue_on_error:
                break

        run_df = pd.DataFrame([asdict(r) for r in run_records])
        if merge_tests:
            run_df = self._merge_run_records(existing_df, run_df)

        run_df.to_csv(sweep_dir / "sweep_runs.csv", index=False)
        (sweep_dir / "sweep_runs.json").write_text(
            json.dumps(run_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        fold_summaries: list[dict[str, Any]] = []
        top_5_runs: list[dict[str, Any]] = []
        baseline_rmse: float | None = None
        baseline_mae: float | None = None
        drift_by_fold: list[dict[str, Any]] = []

        report_fold_names = sorted(
            set(
                run_df["fold_name"].dropna().astype(str).tolist()
                if not run_df.empty and "fold_name" in run_df.columns
                else []
            )
        )
        if not report_fold_names:
            report_fold_names = [name for name, _ in folds]

        for fold_name in report_fold_names:
            fold_dir = sweep_folds_dir / fold_name
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_models_dir = fold_dir / "models"
            fold_df = run_df[run_df["fold_name"] == fold_name].copy()

            fold_df.to_csv(fold_dir / "sweep_runs.csv", index=False)
            (fold_dir / "sweep_runs.json").write_text(
                json.dumps(fold_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            fold_all_models_df = self._collect_all_models(fold_models_dir)

            fold_baseline_rmse: float | None = None
            fold_baseline_mae: float | None = None
            if not fold_df.empty:
                fold_baseline_rows = fold_df[
                    (fold_df["status"] == "ok") & (fold_df["varied_param"].isna())
                ].copy()
                if not fold_baseline_rows.empty:
                    fold_baseline_rmse = float(
                        pd.to_numeric(fold_baseline_rows["test_rmse"], errors="coerce").mean()
                    )
                    fold_baseline_mae = float(
                        pd.to_numeric(fold_baseline_rows["test_mae"], errors="coerce").mean()
                    )

            fold_impact_detail = pd.DataFrame()
            fold_impact_summary = pd.DataFrame()
            if fold_baseline_rmse is not None and fold_baseline_mae is not None:
                fold_impact_detail, fold_impact_summary = self._build_param_impact(
                    fold_df[fold_df["status"] == "ok"].copy(),
                    fold_baseline_rmse,
                    fold_baseline_mae,
                )
                fold_impact_detail.to_csv(fold_dir / "param_impact_detail.csv", index=False)
                fold_impact_summary.to_csv(fold_dir / "param_impact_summary.csv", index=False)

            fold_config_ranking_raw = self._build_config_ranking(
                fold_df,
                compute_confidence_interval=self.compute_confidence_interval,
            )
            fold_config_ranking_report = self._order_config_ranking_for_report(
                fold_config_ranking_raw,
                param_ranges=self.param_ranges,
            )
            fold_config_ranking_raw.to_csv(fold_dir / "config_ranking.csv", index=False)
            fold_config_ranking_report.to_csv(
                fold_dir / "config_ranking_report_order.csv",
                index=False,
            )

            fold_all_models_df = self._sort_all_models_with_config_ranking(
                fold_all_models_df,
                fold_config_ranking_raw,
            )
            fold_all_models_df.to_csv(fold_dir / "all_models_ranked.csv", index=False)

            fold_top_5 = fold_config_ranking_raw.head(5).to_dict(orient="records")
            if len(report_fold_names) == 1:
                top_5_runs = fold_top_5
                baseline_rmse = fold_baseline_rmse
                baseline_mae = fold_baseline_mae

            drift_detail = pd.DataFrame()
            drift_summary: dict[str, Any] = {}
            fold_split_cfg = fold_split_map.get(fold_name, self.split_config)
            if not dataset_df.empty and drift_feature_cols and fold_split_cfg:
                try:
                    train_df, val_df, test_df = self._apply_time_split_for_drift(
                        dataset_df,
                        split_config=fold_split_cfg,
                    )
                    drift_detail, drift_summary = DataDriftAnalyzer.analyze_features(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        feature_cols=drift_feature_cols,
                    )
                except Exception:
                    logger.exception(
                        "Failed to compute drift metrics for fold",
                        extra={"fold_name": fold_name},
                    )
                    drift_detail = pd.DataFrame()
                    drift_summary = {}

            drift_detail_path = fold_dir / "drift_ks_psi_detail.csv"
            drift_summary_path = fold_dir / "drift_ks_psi_summary.json"
            drift_detail.to_csv(drift_detail_path, index=False)
            drift_summary_payload = {
                "fold_name": fold_name,
                "split_config": fold_split_cfg,
                **drift_summary,
            }
            drift_summary_path.write_text(
                json.dumps(drift_summary_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            drift_by_fold.append(drift_summary_payload)

            fold_plot_artifacts: list[str] = []
            if self.generate_comparison_plots:
                fold_plot_artifacts = self._save_comparison_plots(
                    sweep_dir=fold_dir,
                    run_df=fold_df,
                    config_ranking=fold_config_ranking_raw,
                    impact_detail=fold_impact_detail,
                    impact_summary=fold_impact_summary,
                    param_ranges=self.param_ranges,
                    baseline_config=self.base_training_config,
                )

            fold_summaries.append(
                {
                    "fold_name": fold_name,
                    "split_config": fold_split_cfg,
                    "runs_total": int(len(fold_df)),
                    "runs_ok": int((fold_df["status"] == "ok").sum()) if not fold_df.empty else 0,
                    "runs_failed": int((fold_df["status"] == "failed").sum()) if not fold_df.empty else 0,
                    "baseline_test_rmse": fold_baseline_rmse,
                    "baseline_test_mae": fold_baseline_mae,
                    "top_5_runs": fold_top_5,
                    "drift": drift_summary_payload,
                    "artifacts": {
                        "fold_dir": str(fold_dir),
                        "models_dir": str(fold_models_dir),
                        "sweep_runs_csv": str(fold_dir / "sweep_runs.csv"),
                        "all_models_ranked_csv": str(fold_dir / "all_models_ranked.csv"),
                        "config_ranking_csv": str(fold_dir / "config_ranking.csv"),
                        "config_ranking_report_order_csv": str(
                            fold_dir / "config_ranking_report_order.csv"
                        ),
                        "param_impact_detail_csv": str(fold_dir / "param_impact_detail.csv"),
                        "param_impact_summary_csv": str(fold_dir / "param_impact_summary.csv"),
                        "drift_ks_psi_detail_csv": str(drift_detail_path),
                        "drift_ks_psi_summary_json": str(drift_summary_path),
                        "comparison_plots": fold_plot_artifacts,
                    },
                }
            )

        drift_folds_df = pd.DataFrame(drift_by_fold)
        drift_folds_csv = sweep_dir / "drift_ks_psi_by_fold.csv"
        drift_folds_df.to_csv(drift_folds_csv, index=False)
        drift_overall = {
            "n_folds": int(len(drift_by_fold)),
            "avg_ks_train_vs_val": None,
            "avg_ks_train_vs_test": None,
            "avg_psi_train_vs_val": None,
            "avg_psi_train_vs_test": None,
        }
        if not drift_folds_df.empty:
            for col in [
                "avg_ks_train_vs_val",
                "avg_ks_train_vs_test",
                "avg_psi_train_vs_val",
                "avg_psi_train_vs_test",
            ]:
                if col in drift_folds_df.columns:
                    drift_folds_df[col] = pd.to_numeric(drift_folds_df[col], errors="coerce")
                    drift_overall[col] = float(drift_folds_df[col].mean(skipna=True))
        drift_overall_json = sweep_dir / "drift_ks_psi_overall_summary.json"
        drift_overall_json.write_text(
            json.dumps(drift_overall, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary = {
            "asset": asset,
            "sweep_name": sweep_name,
            "features": features or "(default from main_train_tft)",
            "runs_total": int(len(run_df)),
            "runs_ok": int((run_df["status"] == "ok").sum()) if not run_df.empty else 0,
            "runs_failed": int((run_df["status"] == "failed").sum()) if not run_df.empty else 0,
            "baseline_test_rmse": baseline_rmse,
            "baseline_test_mae": baseline_mae,
            "replica_seeds": self.replica_seeds,
            "split_config": self.split_config or None,
            "walk_forward": {
                "enabled": len(report_fold_names) > 1,
                "folds": report_fold_names,
            },
            "compute_confidence_interval": self.compute_confidence_interval,
            "top_5_runs": top_5_runs,
            "fold_summaries": fold_summaries,
            "drift_overall_summary": drift_overall,
            "artifacts": {
                "sweep_dir": str(sweep_dir),
                "folds_dir": str(sweep_folds_dir),
                "sweep_runs_csv": str(sweep_dir / "sweep_runs.csv"),
                "sweep_runs_json": str(sweep_dir / "sweep_runs.json"),
                "drift_ks_psi_by_fold_csv": str(drift_folds_csv),
                "drift_ks_psi_overall_summary_json": str(drift_overall_json),
                "analysis_config_json": str(sweep_dir / "analysis_config.json")
                if analysis_config is not None
                else None,
            },
        }
        (sweep_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "TFT model analysis completed",
            extra={
                "asset": asset,
                "sweep_dir": str(sweep_dir),
                "runs_ok": summary["runs_ok"],
                "runs_failed": summary["runs_failed"],
                "top_5_runs": summary["top_5_runs"],
            },
        )
        return RunTFTModelAnalysisResult(
            asset=asset,
            sweep_name=sweep_name,
            sweep_dir=str(sweep_dir),
            runs_ok=summary["runs_ok"],
            runs_failed=summary["runs_failed"],
            top_5_runs=top_5_runs,
        )
