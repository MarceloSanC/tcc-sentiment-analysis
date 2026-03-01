from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.domain.services.data_drift_analyzer import DataDriftAnalyzer
from src.use_cases.run_tft_model_analysis_use_case import RunTFTModelAnalysisUseCase
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sweep artifacts (reports + comparison plots) for existing TFT sweep directories."
    )
    parser.add_argument(
        "--sweep-dir",
        action="append",
        required=True,
        help=(
            "Sweep directory path. Can be provided multiple times. "
            "Example: --sweep-dir data/models/AAPL/sweeps/sweep_x"
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _update_summary_plot_artifacts(sweep_dir: Path, plot_paths: list[str]) -> None:
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        summary["artifacts"] = artifacts
    artifacts["comparison_plots"] = plot_paths
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_summary_global_ranking_fields(sweep_dir: Path) -> None:
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    run_df = _read_csv(sweep_dir / "sweep_runs.csv")
    config_ranking = _read_csv(sweep_dir / "config_ranking.csv")

    baseline_rmse = None
    baseline_mae = None
    baseline_da = None
    if not run_df.empty and {"status", "varied_param", "test_rmse", "test_mae"}.issubset(set(run_df.columns)):
        baseline_rows = run_df[(run_df["status"] == "ok") & (run_df["varied_param"].isna())].copy()
        if not baseline_rows.empty:
            baseline_rmse = float(pd.to_numeric(baseline_rows["test_rmse"], errors="coerce").mean())
            baseline_mae = float(pd.to_numeric(baseline_rows["test_mae"], errors="coerce").mean())
            if "test_da" in baseline_rows.columns:
                baseline_da = float(pd.to_numeric(baseline_rows["test_da"], errors="coerce").mean())

    top_5_runs: list[dict[str, Any]] = []
    if not config_ranking.empty:
        ranked = RunTFTModelAnalysisUseCase._sort_config_ranking_for_performance(config_ranking)
        top_5_runs = ranked.head(5).to_dict(orient="records")

    summary["baseline_test_rmse"] = baseline_rmse
    summary["baseline_test_mae"] = baseline_mae
    summary["baseline_test_da"] = baseline_da
    summary["top_5_runs"] = top_5_runs
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_summary_fold_plot_artifacts(
    sweep_dir: Path,
    fold_plot_paths: dict[str, list[str]],
) -> None:
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fold_summaries = summary.get("fold_summaries")
    if isinstance(fold_summaries, list):
        for fold_entry in fold_summaries:
            if not isinstance(fold_entry, dict):
                continue
            fold_name = str(fold_entry.get("fold_name", ""))
            if not fold_name:
                continue
            artifacts = fold_entry.get("artifacts")
            if not isinstance(artifacts, dict):
                artifacts = {}
                fold_entry["artifacts"] = artifacts
            artifacts["comparison_plots"] = fold_plot_paths.get(fold_name, [])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_asset_from_sweep_dir(sweep_dir: Path) -> str:
    summary_path = sweep_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            asset = str(summary.get("asset", "")).strip().upper()
            if asset:
                return asset
        except Exception:
            pass
    # .../data/models/<ASSET>/sweeps/<SWEEP_NAME>
    return sweep_dir.parent.parent.name.upper()


def _resolve_fold_split_configs(
    *,
    analysis_config: dict[str, Any],
    sweep_dir: Path,
) -> list[tuple[str, dict[str, str]]]:
    walk_forward = analysis_config.get("walk_forward", {})
    if isinstance(walk_forward, dict) and isinstance(walk_forward.get("folds"), list):
        resolved: list[tuple[str, dict[str, str]]] = []
        required = {"train_start", "train_end", "val_start", "val_end", "test_start", "test_end"}
        for idx, fold in enumerate(walk_forward.get("folds", []), start=1):
            if not isinstance(fold, dict):
                continue
            name = str(fold.get("name") or f"fold_{idx}").strip() or f"fold_{idx}"
            split = {k: str(fold[k]) for k in required if fold.get(k) is not None}
            if required.issubset(set(split.keys())):
                resolved.append((name, split))
        if resolved:
            return resolved

    split_cfg = analysis_config.get("split_config", {})
    required = ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]
    if isinstance(split_cfg, dict) and all(split_cfg.get(k) is not None for k in required):
        return [("default", {k: str(split_cfg[k]) for k in required})]

    # Fallback using run file fold names + root split if available
    run_df = _read_csv(sweep_dir / "sweep_runs.csv")
    fold_names = (
        run_df.get("fold_name", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
        if not run_df.empty
        else []
    )
    if fold_names and isinstance(split_cfg, dict) and all(split_cfg.get(k) is not None for k in required):
        split = {k: str(split_cfg[k]) for k in required}
        return [(str(name), split) for name in fold_names]
    return []


def _update_summary_drift_artifacts(
    *,
    sweep_dir: Path,
    drift_by_fold: list[dict[str, Any]],
    drift_overall: dict[str, Any],
) -> None:
    summary_path = sweep_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["drift_overall_summary"] = drift_overall
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        summary["artifacts"] = artifacts
    artifacts["drift_ks_psi_by_fold_csv"] = str(sweep_dir / "drift_ks_psi_by_fold.csv")
    artifacts["drift_ks_psi_overall_summary_json"] = str(sweep_dir / "drift_ks_psi_overall_summary.json")

    fold_summaries = summary.get("fold_summaries")
    drift_map = {str(x.get("fold_name", "")): x for x in drift_by_fold}
    if isinstance(fold_summaries, list):
        for fold_entry in fold_summaries:
            if not isinstance(fold_entry, dict):
                continue
            fold_name = str(fold_entry.get("fold_name", ""))
            if not fold_name or fold_name not in drift_map:
                continue
            fold_entry["drift"] = drift_map[fold_name]
            fold_artifacts = fold_entry.get("artifacts")
            if not isinstance(fold_artifacts, dict):
                fold_artifacts = {}
                fold_entry["artifacts"] = fold_artifacts
            fold_artifacts["drift_ks_psi_detail_csv"] = str(
                sweep_dir / "folds" / fold_name / "drift_ks_psi_detail.csv"
            )
            fold_artifacts["drift_ks_psi_summary_json"] = str(
                sweep_dir / "folds" / fold_name / "drift_ks_psi_summary.json"
            )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_drift_reports_for_sweep(
    *,
    sweep_dir: Path,
    analysis_config: dict[str, Any],
) -> None:
    asset = _resolve_asset_from_sweep_dir(sweep_dir)
    if not asset:
        return
    paths = load_data_paths()
    dataset_repo = ParquetTFTDatasetRepository(output_dir=Path(paths["dataset_tft"]))
    dataset_df = dataset_repo.load(asset)
    feature_tokens = RunTFTModelAnalysisUseCase._parse_feature_tokens(analysis_config.get("features"))
    feature_cols = RunTFTModelAnalysisUseCase._resolve_features_for_drift(dataset_df, feature_tokens)
    fold_splits = _resolve_fold_split_configs(analysis_config=analysis_config, sweep_dir=sweep_dir)
    if not fold_splits:
        return

    drift_by_fold: list[dict[str, Any]] = []
    for fold_name, split_cfg in fold_splits:
        fold_dir = sweep_dir / "folds" / fold_name
        if not fold_dir.exists():
            fold_dir.mkdir(parents=True, exist_ok=True)
        train_df, val_df, test_df = RunTFTModelAnalysisUseCase._apply_time_split_for_drift(
            dataset_df,
            split_config=split_cfg,
        )
        detail_df, summary = DataDriftAnalyzer.analyze_features(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_cols=feature_cols,
        )
        detail_path = fold_dir / "drift_ks_psi_detail.csv"
        summary_path = fold_dir / "drift_ks_psi_summary.json"
        detail_df.to_csv(detail_path, index=False)
        payload = {"fold_name": fold_name, "split_config": split_cfg, **summary}
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        drift_by_fold.append(payload)

    drift_df = pd.DataFrame(drift_by_fold)
    drift_df.to_csv(sweep_dir / "drift_ks_psi_by_fold.csv", index=False)
    drift_overall = {
        "n_folds": int(len(drift_by_fold)),
        "avg_ks_train_vs_val": None,
        "avg_ks_train_vs_test": None,
        "avg_psi_train_vs_val": None,
        "avg_psi_train_vs_test": None,
    }
    if not drift_df.empty:
        for col in [
            "avg_ks_train_vs_val",
            "avg_ks_train_vs_test",
            "avg_psi_train_vs_val",
            "avg_psi_train_vs_test",
        ]:
            if col in drift_df.columns:
                drift_df[col] = pd.to_numeric(drift_df[col], errors="coerce")
                drift_overall[col] = float(drift_df[col].mean(skipna=True))
    (sweep_dir / "drift_ks_psi_overall_summary.json").write_text(
        json.dumps(drift_overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _update_summary_drift_artifacts(
        sweep_dir=sweep_dir,
        drift_by_fold=drift_by_fold,
        drift_overall=drift_overall,
    )


def _generate_for_analysis_dir(
    analysis_dir: Path,
    *,
    param_ranges: dict[str, list] | None = None,
    baseline_config: dict[str, Any] | None = None,
) -> list[str]:
    run_df = _read_csv(analysis_dir / "sweep_runs.csv")
    config_ranking = _read_csv(analysis_dir / "config_ranking.csv")
    impact_detail = _read_csv(analysis_dir / "param_impact_detail.csv")
    impact_summary = _read_csv(analysis_dir / "param_impact_summary.csv")
    return RunTFTModelAnalysisUseCase._save_comparison_plots(
        sweep_dir=analysis_dir,
        run_df=run_df,
        config_ranking=config_ranking,
        impact_detail=impact_detail,
        impact_summary=impact_summary,
        param_ranges=param_ranges,
        baseline_config=baseline_config,
    )


def _build_reports_for_analysis_dir(
    analysis_dir: Path,
    *,
    run_df: pd.DataFrame,
    param_ranges: dict[str, list] | None = None,
    compute_confidence_interval: bool = True,
) -> None:
    run_df.to_csv(analysis_dir / "sweep_runs.csv", index=False)
    (analysis_dir / "sweep_runs.json").write_text(
        json.dumps(run_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline_rmse = None
    baseline_mae = None
    baseline_da = None
    if not run_df.empty:
        baseline_rows = run_df[(run_df["status"] == "ok") & (run_df["varied_param"].isna())].copy()
        if not baseline_rows.empty:
            baseline_rmse = float(pd.to_numeric(baseline_rows["test_rmse"], errors="coerce").mean())
            baseline_mae = float(pd.to_numeric(baseline_rows["test_mae"], errors="coerce").mean())
            if "test_da" in baseline_rows.columns:
                baseline_da = float(pd.to_numeric(baseline_rows["test_da"], errors="coerce").mean())

    impact_detail = pd.DataFrame()
    impact_summary = pd.DataFrame()
    if baseline_rmse is not None and baseline_mae is not None:
        impact_detail, impact_summary = RunTFTModelAnalysisUseCase._build_param_impact(
            run_df[run_df["status"] == "ok"].copy(),
            baseline_rmse,
            baseline_mae,
        )
    impact_detail.to_csv(analysis_dir / "param_impact_detail.csv", index=False)
    impact_summary.to_csv(analysis_dir / "param_impact_summary.csv", index=False)

    config_ranking = RunTFTModelAnalysisUseCase._build_config_ranking(
        run_df,
        compute_confidence_interval=compute_confidence_interval,
    )
    config_ranking_report = RunTFTModelAnalysisUseCase._order_config_ranking_for_report(
        config_ranking,
        param_ranges=param_ranges,
    )
    config_ranking.to_csv(analysis_dir / "config_ranking.csv", index=False)
    config_ranking_report.to_csv(analysis_dir / "config_ranking_report_order.csv", index=False)


def _build_root_all_models_ranked(sweep_dir: Path) -> None:
    folds_dir = sweep_dir / "folds"
    if not folds_dir.exists():
        return
    parts: list[pd.DataFrame] = []
    for fold_dir in sorted([p for p in folds_dir.iterdir() if p.is_dir()]):
        models_dir = fold_dir / "models"
        if not models_dir.exists():
            continue
        df = RunTFTModelAnalysisUseCase._collect_all_models(models_dir)
        if not df.empty:
            parts.append(df)
    if not parts:
        pd.DataFrame().to_csv(sweep_dir / "all_models_ranked.csv", index=False)
        return
    merged = pd.concat(parts, ignore_index=True)
    if "version" in merged.columns:
        merged = merged.drop_duplicates(subset=["version"], keep="first")
    config_ranking_path = sweep_dir / "config_ranking.csv"
    config_ranking = _read_csv(config_ranking_path)
    merged = RunTFTModelAnalysisUseCase._sort_all_models_with_config_ranking(merged, config_ranking)
    merged.to_csv(sweep_dir / "all_models_ranked.csv", index=False)


def _bootstrap_fold_reports_if_missing(
    sweep_dir: Path,
    *,
    param_ranges: dict[str, list] | None = None,
) -> list[str]:
    root_runs_path = sweep_dir / "sweep_runs.csv"
    if not root_runs_path.exists():
        return []

    root_df = pd.read_csv(root_runs_path)
    if root_df.empty or "fold_name" not in root_df.columns:
        return []

    fold_names = (
        root_df["fold_name"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    if not fold_names:
        return []

    folds_dir = sweep_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    generated_folds: list[str] = []
    for fold_name in sorted(fold_names):
        fold_dir = folds_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Skip if fold-level analysis files already exist.
        if (fold_dir / "sweep_runs.csv").exists() and (fold_dir / "config_ranking.csv").exists():
            generated_folds.append(fold_name)
            continue

        fold_df = root_df[root_df["fold_name"].astype(str) == fold_name].copy()
        fold_df.to_csv(fold_dir / "sweep_runs.csv", index=False)
        (fold_dir / "sweep_runs.json").write_text(
            json.dumps(fold_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        models_dir = fold_dir / "models"
        if models_dir.exists():
            all_models_df = RunTFTModelAnalysisUseCase._collect_all_models(models_dir)
        else:
            all_models_df = pd.DataFrame()

        baseline_rmse = None
        baseline_mae = None
        baseline_rows = fold_df[(fold_df["status"] == "ok") & (fold_df["varied_param"].isna())].copy()
        if not baseline_rows.empty:
            baseline_rmse = float(pd.to_numeric(baseline_rows["test_rmse"], errors="coerce").mean())
            baseline_mae = float(pd.to_numeric(baseline_rows["test_mae"], errors="coerce").mean())
            if "test_da" in baseline_rows.columns:
                baseline_da = float(pd.to_numeric(baseline_rows["test_da"], errors="coerce").mean())

        impact_detail = pd.DataFrame()
        impact_summary = pd.DataFrame()
        if baseline_rmse is not None and baseline_mae is not None:
            impact_detail, impact_summary = RunTFTModelAnalysisUseCase._build_param_impact(
                fold_df[fold_df["status"] == "ok"].copy(),
                baseline_rmse,
                baseline_mae,
            )
        impact_detail.to_csv(fold_dir / "param_impact_detail.csv", index=False)
        impact_summary.to_csv(fold_dir / "param_impact_summary.csv", index=False)

        config_ranking = RunTFTModelAnalysisUseCase._build_config_ranking(
            fold_df,
            compute_confidence_interval=True,
        )
        config_ranking_report = RunTFTModelAnalysisUseCase._order_config_ranking_for_report(
            config_ranking,
            param_ranges=param_ranges,
        )
        config_ranking.to_csv(fold_dir / "config_ranking.csv", index=False)
        config_ranking_report.to_csv(fold_dir / "config_ranking_report_order.csv", index=False)

        all_models_df = RunTFTModelAnalysisUseCase._sort_all_models_with_config_ranking(
            all_models_df,
            config_ranking,
        )
        all_models_df.to_csv(fold_dir / "all_models_ranked.csv", index=False)
        generated_folds.append(fold_name)

    return generated_folds


def generate_for_sweep(sweep_dir: Path) -> list[str]:
    if not sweep_dir.exists() or not sweep_dir.is_dir():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    param_ranges: dict[str, list] | None = None
    baseline_config: dict[str, Any] | None = None
    compute_confidence_interval = True
    analysis_config_path = sweep_dir / "analysis_config.json"
    analysis_config_content: dict[str, Any] = {}
    if analysis_config_path.exists():
        try:
            content = json.loads(analysis_config_path.read_text(encoding="utf-8"))
            if isinstance(content, dict):
                analysis_config_content = content
            maybe_ranges = content.get("param_ranges")
            if isinstance(maybe_ranges, dict):
                param_ranges = {
                    str(k): list(v) for k, v in maybe_ranges.items() if isinstance(v, list)
                }
            maybe_training_cfg = content.get("training_config")
            if isinstance(maybe_training_cfg, dict):
                baseline_config = dict(maybe_training_cfg)
            compute_confidence_interval = bool(content.get("compute_confidence_interval", True))
        except Exception:
            param_ranges = None
            baseline_config = None
            compute_confidence_interval = True
            analysis_config_content = {}

    # Always rebuild root-level reports when sweep_runs exists.
    root_runs_path = sweep_dir / "sweep_runs.csv"
    if root_runs_path.exists():
        root_df = pd.read_csv(root_runs_path)
        _build_reports_for_analysis_dir(
            sweep_dir,
            run_df=root_df,
            param_ranges=param_ranges,
            compute_confidence_interval=compute_confidence_interval,
        )
        _build_root_all_models_ranked(sweep_dir)

    # Build fold-level reports from root sweep runs when running on legacy outputs.
    _bootstrap_fold_reports_if_missing(sweep_dir, param_ranges=param_ranges)

    # Rebuild drift reports without retraining (from dataset + saved split config).
    if analysis_config_content:
        _build_drift_reports_for_sweep(
            sweep_dir=sweep_dir,
            analysis_config=analysis_config_content,
        )

    folds_dir = sweep_dir / "folds"
    root_plot_paths = _generate_for_analysis_dir(
        sweep_dir,
        param_ranges=param_ranges,
        baseline_config=baseline_config,
    )
    _update_summary_global_ranking_fields(sweep_dir)
    _update_summary_plot_artifacts(sweep_dir, root_plot_paths)

    if folds_dir.exists() and folds_dir.is_dir():
        fold_plot_paths: dict[str, list[str]] = {}
        for fold_dir in sorted([p for p in folds_dir.iterdir() if p.is_dir()]):
            if (fold_dir / "sweep_runs.csv").exists() and (fold_dir / "config_ranking.csv").exists():
                fold_plot_paths[fold_dir.name] = _generate_for_analysis_dir(
                    fold_dir,
                    param_ranges=param_ranges,
                    baseline_config=baseline_config,
                )
        _update_summary_fold_plot_artifacts(sweep_dir, fold_plot_paths)
        if fold_plot_paths:
            summary_path = sweep_dir / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                artifacts = summary.get("artifacts")
                if not isinstance(artifacts, dict):
                    artifacts = {}
                    summary["artifacts"] = artifacts
                merged: list[str] = []
                for fold_name in sorted(fold_plot_paths.keys()):
                    merged.extend(fold_plot_paths[fold_name])
                artifacts["comparison_plots"] = merged
                summary_path.write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        merged: list[str] = []
        merged.extend(root_plot_paths)
        for fold_name in sorted(fold_plot_paths.keys()):
            merged.extend(fold_plot_paths[fold_name])
        return merged

    return root_plot_paths


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()
    failures: list[str] = []
    for sweep_dir_raw in args.sweep_dir:
        sweep_dir = Path(sweep_dir_raw)
        try:
            plot_paths = generate_for_sweep(sweep_dir)
            logger.info(
                "Sweep artifacts generated",
                extra={"sweep_dir": str(sweep_dir), "artifacts_generated": len(plot_paths)},
            )
        except Exception as exc:
            failures.append(f"{sweep_dir}: {exc}")
            logger.exception("Failed to generate sweep artifacts", extra={"sweep_dir": str(sweep_dir)})

    if failures:
        raise RuntimeError("Failed sweeps:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
