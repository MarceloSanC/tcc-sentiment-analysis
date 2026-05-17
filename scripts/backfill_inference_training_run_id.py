from __future__ import annotations

# Operacional: este script e one-shot manual. Nao rodar concorrentemente
# com `refresh_analytics_store_use_case` - nao ha lock no Parquet e a
# sobrescrita pode perder writes intermediarios.

import argparse
import json

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

INFERENCE_TABLES = (
    "fact_inference_runs",
    "fact_inference_predictions",
    "fact_feature_contrib_local",
)


@dataclass(frozen=True)
class BackfillReport:
    total_processed: int = 0
    total_backfilled: int = 0
    total_ambiguous: int = 0
    total_no_match: int = 0
    dry_run: bool = True


@dataclass(frozen=True)
class _ArtifactCandidate:
    training_run_id: str
    asset: str | None
    model_path: str | None
    config_path: str | None


@dataclass
class _TableFrame:
    table_name: str
    path: Path
    frame: pd.DataFrame
    modified: bool = False


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def _normalize_path(value: Any) -> str | None:
    if _is_nullish(value):
        return None
    text = str(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/") or None


def _parent_path(value: Any) -> str | None:
    normalized = _normalize_path(value)
    if normalized is None:
        return None
    parent = Path(normalized).parent.as_posix()
    return parent.rstrip("/") or None


def _table_files(silver_dir: Path, table_name: str) -> list[Path]:
    table_dir = silver_dir / table_name
    if not table_dir.exists():
        return []
    return sorted(table_dir.rglob("*.parquet"))


def _load_table_frames(silver_dir: Path, table_name: str) -> list[_TableFrame]:
    frames: list[_TableFrame] = []
    for path in _table_files(silver_dir, table_name):
        frames.append(_TableFrame(table_name=table_name, path=path, frame=pd.read_parquet(path)))
    return frames


def _load_fact_model_artifacts(silver_dir: Path) -> list[_ArtifactCandidate]:
    candidates: list[_ArtifactCandidate] = []
    for path in _table_files(silver_dir, "fact_model_artifacts"):
        frame = pd.read_parquet(path)
        for row in frame.to_dict(orient="records"):
            training_run_id = row.get("training_run_id")
            if _is_nullish(training_run_id):
                training_run_id = row.get("run_id")
            if _is_nullish(training_run_id):
                continue

            config_path = _normalize_path(row.get("config_path"))
            model_path = _parent_path(config_path) or _parent_path(row.get("checkpoint_path_final"))
            candidates.append(
                _ArtifactCandidate(
                    training_run_id=str(training_run_id),
                    asset=None if _is_nullish(row.get("asset")) else str(row.get("asset")),
                    model_path=model_path,
                    config_path=config_path,
                )
            )
    return candidates


def _prediction_context(table_frames: list[_TableFrame]) -> dict[str, set[str]]:
    context: dict[str, set[str]] = {}
    for table in table_frames:
        if table.table_name != "fact_inference_predictions":
            continue
        if "inference_run_id" not in table.frame.columns or "model_path" not in table.frame.columns:
            continue
        for row in table.frame.to_dict(orient="records"):
            inference_run_id = row.get("inference_run_id")
            model_path = _normalize_path(row.get("model_path"))
            if _is_nullish(inference_run_id) or model_path is None:
                continue
            context.setdefault(str(inference_run_id), set()).add(model_path)
    return context


def _resolve_model_paths(row: dict[str, Any], context: dict[str, set[str]]) -> set[str]:
    paths: set[str] = set()
    model_path = _normalize_path(row.get("model_path"))
    if model_path is not None:
        paths.add(model_path)
    config_parent = _parent_path(row.get("config_path"))
    if config_parent is not None:
        paths.add(config_parent)
    inference_run_id = row.get("inference_run_id")
    if not _is_nullish(inference_run_id):
        paths.update(context.get(str(inference_run_id), set()))
    return paths


def _matching_candidates(
    row: dict[str, Any],
    *,
    candidates: list[_ArtifactCandidate],
    context: dict[str, set[str]],
) -> list[_ArtifactCandidate]:
    row_asset = None if _is_nullish(row.get("asset")) else str(row.get("asset"))
    row_model_paths = _resolve_model_paths(row, context)
    if not row_model_paths:
        return []

    matches: list[_ArtifactCandidate] = []
    for candidate in candidates:
        if row_asset is not None and candidate.asset is not None and row_asset != candidate.asset:
            continue
        candidate_paths = {p for p in (candidate.model_path, _parent_path(candidate.config_path)) if p}
        if row_model_paths.intersection(candidate_paths):
            matches.append(candidate)
    return matches


def _legacy_mask(frame: pd.DataFrame) -> pd.Series:
    if "training_run_id" not in frame.columns:
        frame["training_run_id"] = None
    if "run_id" not in frame.columns:
        frame["run_id"] = None
    return frame["training_run_id"].map(_is_nullish) | frame["run_id"].map(_is_nullish)


def backfill_inference_training_run_id(
    *,
    silver_dir: str | Path = "data/analytics/silver",
    apply: bool = False,
) -> BackfillReport:
    silver_path = Path(silver_dir)
    table_frames = [
        table
        for table_name in INFERENCE_TABLES
        for table in _load_table_frames(silver_path, table_name)
    ]
    candidates = _load_fact_model_artifacts(silver_path)
    context = _prediction_context(table_frames)

    processed = backfilled = ambiguous = no_match = 0
    for table in table_frames:
        frame = table.frame
        if frame.empty:
            continue
        mask = _legacy_mask(frame)
        for index in frame.index[mask]:
            processed += 1
            row = frame.loc[index].to_dict()
            matches = _matching_candidates(row, candidates=candidates, context=context)
            unique_ids = sorted({m.training_run_id for m in matches})
            if len(unique_ids) == 1:
                frame.at[index, "run_id"] = unique_ids[0]
                frame.at[index, "training_run_id"] = unique_ids[0]
                table.modified = True
                backfilled += 1
            elif len(unique_ids) > 1:
                frame.at[index, "_backfill_status"] = "ambiguous"
                table.modified = True
                ambiguous += 1
            else:
                frame.at[index, "_backfill_status"] = "no_match"
                table.modified = True
                no_match += 1

    if apply:
        for table in table_frames:
            if table.modified:
                table.frame.to_parquet(table.path, index=False)

    return BackfillReport(
        total_processed=processed,
        total_backfilled=backfilled,
        total_ambiguous=ambiguous,
        total_no_match=no_match,
        dry_run=not apply,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill training_run_id in current analytics silver inference tables."
    )
    parser.add_argument(
        "--silver-dir",
        default="data/analytics/silver",
        help="Current silver analytics directory. Do not point this at the pre-Phase B archive.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write resolved updates to Parquet. Without this flag the command is dry-run only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit no-write mode. This is the default.",
    )
    args = parser.parse_args()
    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive")
    return args


def main() -> int:
    args = _parse_args()
    report = backfill_inference_training_run_id(
        silver_dir=args.silver_dir,
        apply=bool(args.apply),
    )
    print(json.dumps(asdict(report), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
