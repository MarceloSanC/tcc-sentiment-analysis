from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pandas as pd


class IncompletePredictionWindowError(ValueError):
    """Raised when decision_idx + h exceeds dataset bounds (skip row)."""


@dataclass(frozen=True)
class RunContext:
    schema_version: int
    run_id: str
    model_version: str
    asset: str
    feature_set_name: str
    config_signature: str
    split: str
    fold: str
    seed: int


@dataclass(frozen=True)
class PredictionRecord:
    schema_version: int
    run_id: str
    model_version: str
    asset: str
    feature_set_name: str
    config_signature: str
    split: str
    fold: str
    seed: int
    horizon: int
    decision_idx: int
    timestamp_utc: str
    target_timestamp_utc: str
    y_true: float
    y_pred: float
    error: float
    abs_error: float
    sq_error: float
    quantile_p10: float | None
    quantile_p50: float | None
    quantile_p90: float | None
    quantile_p10_post_guardrail: float | None
    quantile_p50_post_guardrail: float | None
    quantile_p90_post_guardrail: float | None
    quantile_guardrail_applied: int
    year: int

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHorizonPredictionPersister:
    """Canonical convention for fact_oos_predictions rows.

    Per ADR-0003 (Stage R-20, Opcao d revision):
      - timestamp_utc = decision_day = dataset_timestamps[decision_idx]
      - target_timestamp_utc = dataset_timestamps[decision_idx + h]
      - y_true is supplied by callers indexing target_return[decision_idx + h].
        target_return is now backward-indexed (target_return[t] =
        log(close[t]/close[t-1])), so this position holds
        log(close[decision_day + h] / close[decision_day + h - 1]) — the
        h-period return ending at target_timestamp. For h=1 that yields
        "next-day return after decision" (literature semantics preserved).
      - decision_idx is a column in fact_oos_predictions, materializing the
        invariant in the schema (not only in code).

    Raises IncompletePredictionWindowError if decision_idx + h exceeds
    dataset bounds, so callers can `continue` (AGENT_CORE: skip rows
    with missing supervision).
    """

    @staticmethod
    def build_record(
        *,
        decision_idx: int,
        h: int,
        y_true: float,
        y_pred: float,
        q10: float | None,
        q50: float | None,
        q90: float | None,
        q10_post: float | None,
        q50_post: float | None,
        q90_post: float | None,
        guardrail_applied: bool,
        dataset_timestamps: Sequence[pd.Timestamp],
        run_context: RunContext,
    ) -> PredictionRecord:
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")
        if decision_idx < 0:
            raise ValueError(f"decision_idx must be >= 0, got {decision_idx}")
        target_pos = decision_idx + h
        if target_pos >= len(dataset_timestamps):
            raise IncompletePredictionWindowError(
                f"decision_idx={decision_idx} + h={h} = {target_pos} "
                f">= len(timestamps)={len(dataset_timestamps)}"
            )
        if decision_idx >= len(dataset_timestamps):
            raise IncompletePredictionWindowError(
                f"decision_idx={decision_idx} >= len(timestamps)={len(dataset_timestamps)}"
            )
        ts = pd.Timestamp(dataset_timestamps[decision_idx])
        target_ts = pd.Timestamp(dataset_timestamps[target_pos])
        err = float(y_pred - y_true)
        return PredictionRecord(
            schema_version=int(run_context.schema_version),
            run_id=str(run_context.run_id),
            model_version=str(run_context.model_version),
            asset=str(run_context.asset),
            feature_set_name=str(run_context.feature_set_name),
            config_signature=str(run_context.config_signature),
            split=str(run_context.split),
            fold=str(run_context.fold),
            seed=int(run_context.seed),
            horizon=int(h),
            decision_idx=int(decision_idx),
            timestamp_utc=str(ts.isoformat()),
            target_timestamp_utc=str(target_ts.isoformat()),
            y_true=float(y_true),
            y_pred=float(y_pred),
            error=err,
            abs_error=float(abs(err)),
            sq_error=float(err * err),
            quantile_p10=None if q10 is None else float(q10),
            quantile_p50=None if q50 is None else float(q50),
            quantile_p90=None if q90 is None else float(q90),
            quantile_p10_post_guardrail=None if q10_post is None else float(q10_post),
            quantile_p50_post_guardrail=None if q50_post is None else float(q50_post),
            quantile_p90_post_guardrail=None if q90_post is None else float(q90_post),
            quantile_guardrail_applied=1 if guardrail_applied else 0,
            year=int(ts.year),
        )
