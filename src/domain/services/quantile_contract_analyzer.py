from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class QuantileBlockAThresholds:
    max_crossing_bruto_rate: float = 0.001
    max_negative_interval_width_count: int = 0
    max_crossing_post_guardrail_rate: float = 0.0
    require_post_guardrail: bool = False


@dataclass(frozen=True)
class QuantileContractMetrics:
    total_rows: int
    order_rows: int
    crossing_bruto_count: int
    crossing_bruto_rate: float
    negative_interval_width_count: int
    negative_interval_width_rate: float
    post_guardrail_rows: int
    crossing_post_guardrail_count: int | None
    crossing_post_guardrail_rate: float | None


@dataclass(frozen=True)
class QuantileBlockAEvaluation:
    passed: bool
    detail: str
    metrics: QuantileContractMetrics


@dataclass(frozen=True)
class QuantileDegeneracyThresholds:
    min_rows_for_gate: int = 1000
    max_p10_eq_p90_rate: float = 0.05


@dataclass(frozen=True)
class QuantileDegeneracyMetrics:
    parent_sweep_id: str | None
    split: str | None
    horizon: int | None
    prediction_mode: str | None
    n_rows: int
    p10_eq_p90_count: int
    p10_eq_p90_rate: float
    p10_eq_p50_eq_p90_count: int
    p10_eq_p50_eq_p90_rate: float


@dataclass(frozen=True)
class QuantileDegeneracyEvaluation:
    passed: bool
    detail: str
    metrics_per_group: list[QuantileDegeneracyMetrics]


class QuantileContractAnalyzer:
    """Compute reusable quantile contract metrics and Block A acceptance checks."""

    POST_GUARDRAIL_COLS = (
        "quantile_p10_post_guardrail",
        "quantile_p50_post_guardrail",
        "quantile_p90_post_guardrail",
    )

    @staticmethod
    def _normalize_list(values: list[str] | list[int] | None) -> list[str] | None:
        if not values:
            return None
        out = [str(v).strip() for v in values if str(v).strip()]
        return out or None

    @staticmethod
    def _normalize_parent_sweep_value(value: object) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "<na>"}:
            return None
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        return text

    @staticmethod
    def filter_scope(
        *,
        fact_oos_predictions: pd.DataFrame,
        dim_run: pd.DataFrame,
        parent_sweep_prefixes: list[str] | None = None,
        splits: list[str] | None = None,
        horizons: list[int] | None = None,
    ) -> pd.DataFrame:
        df = fact_oos_predictions.copy()
        if df.empty:
            return df

        split_filter = QuantileContractAnalyzer._normalize_list(splits)
        if split_filter is not None and "split" in df.columns:
            df = df[df["split"].astype(str).isin(split_filter)].copy()

        if horizons and "horizon" in df.columns:
            hset = {int(h) for h in horizons}
            hcol = pd.to_numeric(df["horizon"], errors="coerce")
            df = df[hcol.isin(hset)].copy()

        prefixes = QuantileContractAnalyzer._normalize_list(parent_sweep_prefixes)
        if prefixes is not None and "run_id" in df.columns and not dim_run.empty and "run_id" in dim_run.columns:
            meta = dim_run[[c for c in ["run_id", "parent_sweep_id"] if c in dim_run.columns]].drop_duplicates("run_id")
            scoped = df.merge(meta, on="run_id", how="left")
            ps = scoped.get("parent_sweep_id", pd.Series(index=scoped.index, dtype=object)).astype(str)
            mask = ps.apply(lambda v: isinstance(v, str) and any(v.startswith(p) for p in prefixes))
            df = scoped[mask].drop(columns=[c for c in ["parent_sweep_id"] if c in scoped.columns]).copy()

        return df

    @staticmethod
    def analyze(
        fact_oos_predictions: pd.DataFrame,
        *,
        post_guardrail_cols: tuple[str, str, str] | None = None,
    ) -> QuantileContractMetrics:
        if fact_oos_predictions.empty:
            return QuantileContractMetrics(
                total_rows=0,
                order_rows=0,
                crossing_bruto_count=0,
                crossing_bruto_rate=0.0,
                negative_interval_width_count=0,
                negative_interval_width_rate=0.0,
                post_guardrail_rows=0,
                crossing_post_guardrail_count=None,
                crossing_post_guardrail_rate=None,
            )

        df = fact_oos_predictions.copy()
        total_rows = int(len(df))

        q10 = pd.to_numeric(df.get("quantile_p10"), errors="coerce")
        q50 = pd.to_numeric(df.get("quantile_p50"), errors="coerce")
        q90 = pd.to_numeric(df.get("quantile_p90"), errors="coerce")

        order_mask = (~q10.isna()) & (~q50.isna()) & (~q90.isna())
        order_rows = int(order_mask.sum())
        bad_order = int((order_mask & ((q10 > q50) | (q50 > q90))).sum())
        crossing_bruto_rate = float(bad_order / order_rows) if order_rows > 0 else 0.0

        width_mask = (~q10.isna()) & (~q90.isna())
        bad_width = int((width_mask & ((q90 - q10) < 0.0)).sum())
        negative_width_rate = float(bad_width / int(width_mask.sum())) if int(width_mask.sum()) > 0 else 0.0

        pcols = post_guardrail_cols or QuantileContractAnalyzer.POST_GUARDRAIL_COLS
        pg_count: int | None = None
        pg_rate: float | None = None
        pg_rows = 0
        if set(pcols).issubset(set(df.columns)):
            p10 = pd.to_numeric(df[pcols[0]], errors="coerce")
            p50 = pd.to_numeric(df[pcols[1]], errors="coerce")
            p90 = pd.to_numeric(df[pcols[2]], errors="coerce")
            pg_mask = (~p10.isna()) & (~p50.isna()) & (~p90.isna())
            pg_rows = int(pg_mask.sum())
            pg_count = int((pg_mask & ((p10 > p50) | (p50 > p90))).sum())
            pg_rate = float(pg_count / pg_rows) if pg_rows > 0 else 0.0

        return QuantileContractMetrics(
            total_rows=total_rows,
            order_rows=order_rows,
            crossing_bruto_count=bad_order,
            crossing_bruto_rate=crossing_bruto_rate,
            negative_interval_width_count=bad_width,
            negative_interval_width_rate=negative_width_rate,
            post_guardrail_rows=pg_rows,
            crossing_post_guardrail_count=pg_count,
            crossing_post_guardrail_rate=pg_rate,
        )

    @staticmethod
    def evaluate_block_a(
        metrics: QuantileContractMetrics,
        *,
        thresholds: QuantileBlockAThresholds,
    ) -> QuantileBlockAEvaluation:
        issues: list[str] = []

        if metrics.crossing_bruto_rate > float(thresholds.max_crossing_bruto_rate):
            issues.append(
                f"crossing_bruto_rate={metrics.crossing_bruto_rate:.8f}>max={float(thresholds.max_crossing_bruto_rate):.8f}"
            )

        if metrics.negative_interval_width_count > int(thresholds.max_negative_interval_width_count):
            issues.append(
                f"negative_interval_width_count={metrics.negative_interval_width_count}>max={int(thresholds.max_negative_interval_width_count)}"
            )

        if metrics.crossing_post_guardrail_rate is None:
            if thresholds.require_post_guardrail:
                issues.append("missing_post_guardrail_quantiles")
        else:
            if metrics.crossing_post_guardrail_rate > float(thresholds.max_crossing_post_guardrail_rate):
                issues.append(
                    f"crossing_post_guardrail_rate={metrics.crossing_post_guardrail_rate:.8f}>max={float(thresholds.max_crossing_post_guardrail_rate):.8f}"
                )

        detail_parts = [
            f"total_rows={metrics.total_rows}",
            f"order_rows={metrics.order_rows}",
            f"crossing_bruto_count={metrics.crossing_bruto_count}",
            f"crossing_bruto_rate={metrics.crossing_bruto_rate:.8f}",
            f"negative_interval_width_count={metrics.negative_interval_width_count}",
            f"post_guardrail_rows={metrics.post_guardrail_rows}",
            (
                "crossing_post_guardrail_rate=NA"
                if metrics.crossing_post_guardrail_rate is None
                else f"crossing_post_guardrail_rate={metrics.crossing_post_guardrail_rate:.8f}"
            ),
        ]
        if issues:
            detail_parts.append("issues=" + "|".join(issues))

        return QuantileBlockAEvaluation(
            passed=len(issues) == 0,
            detail=", ".join(detail_parts),
            metrics=metrics,
        )

    @staticmethod
    def analyze_degeneracy(
        fact_oos_predictions: pd.DataFrame,
        fact_config: pd.DataFrame,
        *,
        group_cols: tuple[str, ...] = ("parent_sweep_id", "split", "horizon"),
    ) -> list[QuantileDegeneracyMetrics]:
        required_prediction_cols = {
            "run_id",
            "quantile_p10",
            "quantile_p50",
            "quantile_p90",
        }
        if fact_oos_predictions.empty or not required_prediction_cols.issubset(set(fact_oos_predictions.columns)):
            return []

        df = fact_oos_predictions.copy()
        if "horizon" in df.columns:
            df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")

        if not fact_config.empty and "run_id" in fact_config.columns:
            cfg_cols = [c for c in ["run_id", "prediction_mode", "parent_sweep_id"] if c in fact_config.columns]
            cfg = fact_config[cfg_cols].drop_duplicates("run_id").copy()
            merged = df.merge(cfg, on="run_id", how="left", suffixes=("", "_config"))
            if "parent_sweep_id_config" in merged.columns:
                config_parent = merged["parent_sweep_id_config"].map(
                    QuantileContractAnalyzer._normalize_parent_sweep_value
                )
                if "parent_sweep_id" in merged.columns:
                    source_parent = merged["parent_sweep_id"].map(
                        QuantileContractAnalyzer._normalize_parent_sweep_value
                    )
                    merged["parent_sweep_id"] = config_parent.combine_first(source_parent)
                else:
                    merged["parent_sweep_id"] = config_parent
                merged = merged.drop(columns=["parent_sweep_id_config"])
            elif "parent_sweep_id" in merged.columns:
                merged["parent_sweep_id"] = merged["parent_sweep_id"].map(
                    QuantileContractAnalyzer._normalize_parent_sweep_value
                )
            df = merged
        else:
            if "prediction_mode" not in df.columns:
                df["prediction_mode"] = None
            if "parent_sweep_id" in df.columns:
                df["parent_sweep_id"] = df["parent_sweep_id"].map(
                    QuantileContractAnalyzer._normalize_parent_sweep_value
                )

        for col in group_cols:
            if col not in df.columns:
                df[col] = None

        q10 = pd.to_numeric(df["quantile_p10"], errors="coerce")
        q50 = pd.to_numeric(df["quantile_p50"], errors="coerce")
        q90 = pd.to_numeric(df["quantile_p90"], errors="coerce")
        valid_width = q10.notna() & q90.notna()
        valid_triplet = valid_width & q50.notna()

        df = df.loc[valid_width].copy()
        if df.empty:
            return []
        df["_p10_eq_p90"] = (q10.loc[df.index] == q90.loc[df.index]).astype(int)
        df["_p10_eq_p50_eq_p90"] = (
            valid_triplet.loc[df.index]
            & (q10.loc[df.index] == q50.loc[df.index])
            & (q50.loc[df.index] == q90.loc[df.index])
        ).astype(int)

        df["prediction_mode"] = (
            df.get("prediction_mode", pd.Series(index=df.index, dtype=object))
            .astype("object")
            .map(lambda v: str(v).strip().lower() if pd.notna(v) and str(v).strip() else None)
        )

        group_keys = list(group_cols) + ["prediction_mode"]
        out: list[QuantileDegeneracyMetrics] = []
        for keys, group in df.groupby(group_keys, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_map = dict(zip(group_keys, keys, strict=True))
            prediction_mode_value = key_map.get("prediction_mode")
            prediction_mode = None if pd.isna(prediction_mode_value) else str(prediction_mode_value)
            n_rows = int(len(group))
            p10_eq_p90_count = int(group["_p10_eq_p90"].sum())
            p10_eq_p50_eq_p90_count = int(group["_p10_eq_p50_eq_p90"].sum())
            horizon_value = key_map.get("horizon")
            horizon = None if pd.isna(horizon_value) else int(horizon_value)
            parent_value = key_map.get("parent_sweep_id")
            parent_sweep_id = None if pd.isna(parent_value) else str(parent_value)
            split_value = key_map.get("split")
            split = None if pd.isna(split_value) else str(split_value)
            out.append(
                QuantileDegeneracyMetrics(
                    parent_sweep_id=parent_sweep_id,
                    split=split,
                    horizon=horizon,
                    prediction_mode=prediction_mode,
                    n_rows=n_rows,
                    p10_eq_p90_count=p10_eq_p90_count,
                    p10_eq_p90_rate=float(p10_eq_p90_count / n_rows) if n_rows > 0 else 0.0,
                    p10_eq_p50_eq_p90_count=p10_eq_p50_eq_p90_count,
                    p10_eq_p50_eq_p90_rate=float(p10_eq_p50_eq_p90_count / n_rows) if n_rows > 0 else 0.0,
                )
            )

        return out

    @staticmethod
    def evaluate_degeneracy(
        metrics_per_group: list[QuantileDegeneracyMetrics],
        *,
        thresholds: QuantileDegeneracyThresholds,
    ) -> QuantileDegeneracyEvaluation:
        issues: list[str] = []
        diagnostic_only = 0
        quantile_groups = 0
        point_groups = 0
        unknown_mode_groups = 0

        for metrics in metrics_per_group:
            mode = (metrics.prediction_mode or "").strip().lower()
            if mode == "point":
                point_groups += 1
                continue
            if mode != "quantile":
                unknown_mode_groups += 1
                continue

            quantile_groups += 1
            group_ref = (
                f"parent_sweep_id={metrics.parent_sweep_id},split={metrics.split},"
                f"horizon={metrics.horizon},prediction_mode={metrics.prediction_mode}"
            )
            if metrics.n_rows < int(thresholds.min_rows_for_gate):
                diagnostic_only += 1
                continue
            if metrics.p10_eq_p90_rate >= float(thresholds.max_p10_eq_p90_rate):
                issues.append(
                    f"{group_ref}:n_rows={metrics.n_rows},p10_eq_p90_rate={metrics.p10_eq_p90_rate:.8f}"
                )

        detail_parts = [
            f"groups={len(metrics_per_group)}",
            f"quantile_groups={quantile_groups}",
            f"point_groups_ignored={point_groups}",
            f"unknown_mode_groups_diagnostic_only={unknown_mode_groups}",
            f"small_quantile_groups_diagnostic_only={diagnostic_only}",
            f"min_rows_for_gate={int(thresholds.min_rows_for_gate)}",
            f"max_p10_eq_p90_rate={float(thresholds.max_p10_eq_p90_rate):.8f}",
        ]
        if metrics_per_group:
            sample = sorted(
                metrics_per_group,
                key=lambda m: (m.p10_eq_p90_rate, m.n_rows),
                reverse=True,
            )[:5]
            detail_parts.append(
                "top_groups="
                + "|".join(
                    (
                        f"parent_sweep_id={m.parent_sweep_id},split={m.split},horizon={m.horizon},"
                        f"prediction_mode={m.prediction_mode},n_rows={m.n_rows},"
                        f"p10_eq_p90_rate={m.p10_eq_p90_rate:.8f},"
                        f"p10_eq_p50_eq_p90_rate={m.p10_eq_p50_eq_p90_rate:.8f}"
                    )
                    for m in sample
                )
            )
        if issues:
            detail_parts.append("issues=" + "|".join(issues))

        return QuantileDegeneracyEvaluation(
            passed=len(issues) == 0,
            detail=", ".join(detail_parts),
            metrics_per_group=metrics_per_group,
        )
