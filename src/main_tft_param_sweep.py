from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.infrastructure.schemas.model_artifact_schema import TFT_TRAINING_DEFAULTS
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


# Realistic one-at-a-time sweep ranges for TFT training.
PARAM_RANGES: dict[str, list[Any]] = {
    "max_encoder_length": [30, 60, 90, 120],
    "batch_size": [32, 64, 128],
    "max_epochs": [20, 30, 50],
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "hidden_size": [16, 32, 64],
    "attention_head_size": [1, 2, 4],
    "dropout": [0.05, 0.1, 0.2, 0.3],
    "hidden_continuous_size": [4, 8, 16],
    "seed": [7, 42, 123],
    "early_stopping_patience": [3, 5, 10],
    "early_stopping_min_delta": [0.0, 1e-5, 1e-4],
}


@dataclass
class SweepRun:
    run_label: str
    varied_param: str | None
    varied_value: Any | None
    version: str | None
    status: str
    error: str | None = None
    test_rmse: float | None = None
    test_mae: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-at-a-time hyperparameter sweep for TFT and build comparison tables "
            "from metadata.json artifacts."
        )
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "Feature tokens/columns forwarded to main_train_tft "
            "(comma-separated). If omitted, main_train_tft uses its default feature set."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue sweep when one run fails.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for number of runs (useful for quick testing).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Optional custom subdirectory name under data/models/{ASSET}/sweeps.",
    )
    return parser.parse_args()


def _key_to_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _build_experiments() -> list[tuple[str, dict[str, Any], str | None, Any | None]]:
    """
    Returns list of:
    - run_label
    - config dict
    - varied_param (None for baseline)
    - varied_value (None for baseline)
    """
    base = dict(TFT_TRAINING_DEFAULTS)
    experiments: list[tuple[str, dict[str, Any], str | None, Any | None]] = [
        ("baseline", base, None, None)
    ]
    for param, values in PARAM_RANGES.items():
        default_value = base.get(param)
        for value in values:
            if value == default_value:
                continue
            cfg = dict(base)
            cfg[param] = value
            # Keep valid relation.
            if cfg["max_prediction_length"] > cfg["max_encoder_length"]:
                continue
            label = f"{param}={value}"
            experiments.append((label, cfg, param, value))
    return experiments


def _load_metadata(metrics_path: Path) -> dict[str, Any]:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _pick_new_version(models_asset_dir: Path, before_versions: set[str]) -> str | None:
    after_versions = {p.name for p in models_asset_dir.iterdir() if p.is_dir()}
    created = sorted(after_versions - before_versions)
    if not created:
        return None
    # Pick newest by mtime to handle edge cases.
    newest = max(created, key=lambda v: (models_asset_dir / v).stat().st_mtime)
    return newest


def _run_train(
    asset: str,
    features: str | None,
    cfg: dict[str, Any],
    models_asset_dir: Path,
) -> tuple[str | None, dict[str, Any] | None]:
    before_versions = {p.name for p in models_asset_dir.iterdir() if p.is_dir()}
    cmd = [sys.executable, "-m", "src.main_train_tft", "--asset", asset]
    if features:
        cmd.extend(["--features", features])
    for key, value in cfg.items():
        cmd.extend([_key_to_flag(key), str(value)])

    logger.info("Starting sweep run", extra={"cmd": cmd})
    subprocess.run(cmd, check=True)

    version = _pick_new_version(models_asset_dir, before_versions)
    if version is None:
        return None, None
    metadata_path = models_asset_dir / version / "metadata.json"
    if not metadata_path.exists():
        return version, None
    return version, _load_metadata(metadata_path)


def _collect_all_models(models_asset_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted([p for p in models_asset_dir.iterdir() if p.is_dir()]):
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = _load_metadata(meta_path)
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


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    asset = args.asset.strip().upper()
    features = args.features.strip() if args.features else None

    paths = load_data_paths()
    models_asset_dir = Path(paths["models"]) / asset
    models_asset_dir.mkdir(parents=True, exist_ok=True)

    sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_name = args.output_subdir or f"sweep_{sweep_ts}"
    sweep_dir = models_asset_dir / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    experiments = _build_experiments()
    if args.max_runs is not None:
        experiments = experiments[: max(1, args.max_runs)]

    logger.info(
        "Starting TFT parameter sweep",
        extra={
            "asset": asset,
            "runs": len(experiments),
            "features": features or "(default from main_train_tft)",
            "sweep_dir": str(sweep_dir),
        },
    )

    run_records: list[SweepRun] = []
    baseline_rmse: float | None = None
    baseline_mae: float | None = None

    for idx, (run_label, cfg, varied_param, varied_value) in enumerate(experiments, start=1):
        logger.info(
            "Sweep run started",
            extra={
                "run_index": idx,
                "total_runs": len(experiments),
                "run_label": run_label,
            },
        )
        try:
            version, metadata = _run_train(asset, features, cfg, models_asset_dir)
            if version is None or metadata is None:
                run_records.append(
                    SweepRun(
                        run_label=run_label,
                        varied_param=varied_param,
                        varied_value=varied_value,
                        version=version,
                        status="failed",
                        error="Missing generated model version or metadata.json",
                    )
                )
                if not args.continue_on_error:
                    break
                continue

            test_metrics = metadata.get("split_metrics", {}).get("test", {})
            test_rmse = float(test_metrics.get("rmse"))
            test_mae = float(test_metrics.get("mae"))
            run_records.append(
                SweepRun(
                    run_label=run_label,
                    varied_param=varied_param,
                    varied_value=varied_value,
                    version=version,
                    status="ok",
                    test_rmse=test_rmse,
                    test_mae=test_mae,
                )
            )
            if run_label == "baseline":
                baseline_rmse = test_rmse
                baseline_mae = test_mae
        except Exception as exc:
            run_records.append(
                SweepRun(
                    run_label=run_label,
                    varied_param=varied_param,
                    varied_value=varied_value,
                    version=None,
                    status="failed",
                    error=str(exc),
                )
            )
            logger.exception("Sweep run failed", extra={"run_label": run_label})
            if not args.continue_on_error:
                break

    run_df = pd.DataFrame([r.__dict__ for r in run_records])
    run_df.to_csv(sweep_dir / "sweep_runs.csv", index=False)
    (sweep_dir / "sweep_runs.json").write_text(
        json.dumps(run_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    all_models_df = _collect_all_models(models_asset_dir)
    all_models_df.to_csv(sweep_dir / "all_models_ranked.csv", index=False)

    if baseline_rmse is not None and baseline_mae is not None:
        impact_detail, impact_summary = _build_param_impact(
            run_df[run_df["status"] == "ok"].copy(),
            baseline_rmse,
            baseline_mae,
        )
        impact_detail.to_csv(sweep_dir / "param_impact_detail.csv", index=False)
        impact_summary.to_csv(sweep_dir / "param_impact_summary.csv", index=False)
    else:
        impact_detail = pd.DataFrame()
        impact_summary = pd.DataFrame()

    summary = {
        "asset": asset,
        "sweep_name": sweep_name,
        "features": features or "(default from main_train_tft)",
        "runs_total": int(len(run_df)),
        "runs_ok": int((run_df["status"] == "ok").sum()) if not run_df.empty else 0,
        "runs_failed": int((run_df["status"] == "failed").sum()) if not run_df.empty else 0,
        "baseline_test_rmse": baseline_rmse,
        "baseline_test_mae": baseline_mae,
        "best_run": (
            run_df[run_df["status"] == "ok"]
            .sort_values("test_rmse", ascending=True)
            .head(1)
            .to_dict(orient="records")
        )
        if not run_df.empty and (run_df["status"] == "ok").any()
        else [],
        "artifacts": {
            "sweep_runs_csv": str(sweep_dir / "sweep_runs.csv"),
            "all_models_ranked_csv": str(sweep_dir / "all_models_ranked.csv"),
            "param_impact_detail_csv": str(sweep_dir / "param_impact_detail.csv"),
            "param_impact_summary_csv": str(sweep_dir / "param_impact_summary.csv"),
        },
    }
    (sweep_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "TFT sweep completed",
        extra={
            "asset": asset,
            "sweep_dir": str(sweep_dir),
            "runs_ok": summary["runs_ok"],
            "runs_failed": summary["runs_failed"],
            "best_run": summary["best_run"],
        },
    )


if __name__ == "__main__":
    main()
