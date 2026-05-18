"""Sibling test pipeline use case for baselines (Stage F.0.2).

Mirrors the architecture of `RunTFTTestPipelineUseCase`: consumes the same
sweep JSON config used by `main_tft_test_pipeline`, derives `parent_sweep_id`
from `output_subdir`, and delegates per-fold/per-seed/per-baseline execution
to `RunBaselinesUseCase` (Stage 12).

Critical decision (F.0.2): the warmup offset of every baseline is **derived
from `training_config.max_encoder_length`** (start) and `max_prediction_length - 1`
(end) in the shared JSON, so the emitted `target_timestamp_utc` set matches
the TFT candidate's exactly. Without that alignment,
`gold_paired_oos_intersection_by_horizon` falls back to empty intersection
and the TFT is excluded from DM/MCS (Cenario Falha D observed in F.1 smoke
2026-05-18).

Offsets are passed to `RunBaselinesUseCase.execute()` as
`evaluation_start_offset_days` / `evaluation_end_offset_days` (additive
parameters, default 0) so trading-day arithmetic happens at idx level
inside the use case rather than via calendar/trading conversion in the
pipeline.

The pipeline tolerates configs without a `baselines` section by emitting
a warning and exiting no-op (non-breaking compat with Stages 1-14 JSONs).
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.use_cases.run_baselines_use_case import (
    BASELINE_SPECS,
    RunBaselinesResult,
    RunBaselinesUseCase,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunBaselinesTestPipelineResult:
    parent_sweep_id: str | None
    folds_processed: int
    seeds_processed: int
    baselines_persisted_total: int
    per_invocation: list[RunBaselinesResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Fold:
    name: str
    train: tuple[str, str]
    val: tuple[str, str]
    test: tuple[str, str]


def _extract_folds(config: dict[str, Any]) -> list[_Fold]:
    wf = config.get("walk_forward") or {}
    if isinstance(wf, dict) and bool(wf.get("enabled")) and wf.get("folds"):
        out: list[_Fold] = []
        for entry in wf["folds"]:
            out.append(
                _Fold(
                    name=str(entry.get("name") or f"wf_{len(out) + 1}"),
                    train=(str(entry["train_start"]), str(entry["train_end"])),
                    val=(str(entry["val_start"]), str(entry["val_end"])),
                    test=(str(entry["test_start"]), str(entry["test_end"])),
                )
            )
        return out
    split_cfg = config.get("split_config") or {}
    return [
        _Fold(
            name="single",
            train=(str(split_cfg["train_start"]), str(split_cfg["train_end"])),
            val=(str(split_cfg["val_start"]), str(split_cfg["val_end"])),
            test=(str(split_cfg["test_start"]), str(split_cfg["test_end"])),
        )
    ]


def _extract_seeds(config: dict[str, Any]) -> list[int]:
    raw = config.get("replica_seeds") or []
    seeds = [int(s) for s in raw]
    if not seeds:
        # dim_run hashes seed; default deterministic fallback so the run
        # row is reproducible.
        fallback = int((config.get("training_config") or {}).get("seed", 42))
        seeds = [fallback]
    return seeds


def _extract_baseline_specs(
    config: dict[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    declared = config.get("baselines")
    if not declared:
        return [], {}
    if not isinstance(declared, list):
        raise ValueError("config['baselines'] must be a list")
    names: list[str] = []
    per_config: dict[str, dict[str, Any]] = {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise ValueError("each baselines entry must be an object")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("baselines[].name is required")
        if name not in BASELINE_SPECS:
            raise ValueError(
                f"unsupported baseline: {name}. "
                f"Stage 12 MVP supports: {sorted(BASELINE_SPECS.keys())}"
            )
        names.append(name)
        per_config[name] = dict(entry.get("config") or {})
    return names, per_config


def _resolve_window_overrides(
    per_baseline_config: dict[str, dict[str, Any]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, cfg in per_baseline_config.items():
        spec = BASELINE_SPECS.get(name)
        if spec is None or spec.window is None:
            continue
        win = cfg.get("window_days")
        if win is None:
            continue
        out[name] = int(win)
    return out


def _resolve_offset_days(
    config: dict[str, Any], per_baseline_config: dict[str, dict[str, Any]]
) -> int:
    training = config.get("training_config") or {}
    default_offset = int(training.get("max_encoder_length") or 0)
    overrides = {
        cfg.get("evaluation_start_offset_days_override")
        for cfg in per_baseline_config.values()
        if "evaluation_start_offset_days_override" in cfg
    }
    overrides.discard(None)
    if not overrides:
        return default_offset
    if len(overrides) > 1:
        raise ValueError(
            "per-baseline evaluation_start_offset_days_override must agree across "
            f"all declared baselines or be left at default; found {overrides}"
        )
    return int(next(iter(overrides)))


class RunBaselinesTestPipelineUseCase:
    """Dispatch baseline runs across folds × seeds × declared baselines.

    Reads the sweep JSON used by TFT pipeline; emits a warning and returns
    a no-op result when no `baselines` section is present (non-breaking).
    """

    def __init__(self, *, baselines_runner: RunBaselinesUseCase) -> None:
        self.baselines_runner = baselines_runner

    def execute(
        self,
        *,
        asset: str,
        config: dict[str, Any],
        dataset_path: Path | str,
    ) -> RunBaselinesTestPipelineResult:
        warnings_emitted: list[str] = []

        baselines, per_config = _extract_baseline_specs(config)
        if not baselines:
            msg = "config has no 'baselines' section; pipeline is no-op"
            warnings_emitted.append(msg)
            logger.warning(msg)
            return RunBaselinesTestPipelineResult(
                parent_sweep_id=None,
                folds_processed=0,
                seeds_processed=0,
                baselines_persisted_total=0,
                warnings=warnings_emitted,
            )

        parent_sweep_id_root = str(config.get("output_subdir") or "").strip()
        if not parent_sweep_id_root:
            raise ValueError(
                "config['output_subdir'] is required to derive parent_sweep_id"
            )

        folds = _extract_folds(config)
        seeds = _extract_seeds(config)
        training = config.get("training_config") or {}
        max_pred = int(training.get("max_prediction_length") or 1)
        # Empirical alignment (validated against F.1 smoke 2026-05-18 re-run):
        # the TFT trainer's first emitted decision per split lands at
        # `max_encoder_length + max_prediction_length - 1` trading rows
        # past split_start, not just `max_encoder_length`. The extra
        # `max_prediction_length - 1` rows are skipped because the TFT
        # TimeSeriesDataSet requires a full decoder window to be available
        # from the first decision.
        base_offset = _resolve_offset_days(config, per_config)
        offset_start = base_offset + max(max_pred - 1, 0)
        # TFT emits up to the last row of each split (the use case's own
        # `y_true_idx >= len(target_returns)` guard already drops impossible
        # rows per horizon for the baseline). No explicit end-side offset.
        offset_end = 0

        horizons_raw = training.get("evaluation_horizons")
        if horizons_raw is None:
            horizons = [1]
            warnings_emitted.append(
                "training_config.evaluation_horizons missing; defaulting to [1]"
            )
        else:
            horizons = [int(h) for h in horizons_raw]

        window_overrides = _resolve_window_overrides(per_config)

        per_invocation: list[RunBaselinesResult] = []
        for fold in folds:
            for seed in seeds:
                if len(folds) == 1 and fold.name == "single":
                    sweep_id = parent_sweep_id_root
                else:
                    sweep_id = f"{parent_sweep_id_root}__{fold.name}"
                result = self.baselines_runner.execute(
                    asset=str(asset),
                    dataset_path=dataset_path,
                    parent_sweep_id=sweep_id,
                    split_definitions={
                        "train": fold.train,
                        "val": fold.val,
                        "test": fold.test,
                    },
                    horizons=list(horizons),
                    baselines=list(baselines),
                    seed=int(seed),
                    overwrite_on_collision=bool(
                        config.get("overwrite_on_collision", False)
                    ),
                    baseline_windows=window_overrides or None,
                    evaluation_start_offset_days=offset_start,
                    evaluation_end_offset_days=offset_end,
                )
                per_invocation.append(result)
                logger.info(
                    "Baselines fold/seed completed",
                    extra={
                        "fold": fold.name,
                        "seed": seed,
                        "parent_sweep_id": sweep_id,
                        "baselines": result.baselines_persisted,
                        "n_rows_per_baseline": result.n_rows_per_baseline,
                        "evaluation_start_offset_days": offset_start,
                        "evaluation_end_offset_days": offset_end,
                    },
                )

        total_persisted = sum(
            len(res.baselines_persisted) for res in per_invocation
        )

        return RunBaselinesTestPipelineResult(
            parent_sweep_id=parent_sweep_id_root,
            folds_processed=len(folds),
            seeds_processed=len(seeds),
            baselines_persisted_total=total_persisted,
            per_invocation=per_invocation,
            warnings=warnings_emitted,
        )
