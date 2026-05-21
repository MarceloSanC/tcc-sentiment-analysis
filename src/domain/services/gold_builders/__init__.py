"""Gold builders modularization (ADR-0005).

25 gold builders previously living inside
`RefreshAnalyticsStoreUseCase` (~2.58k LOC) extracted into per-category
classes implementing `GoldBuilder.build(snapshot, ctx) -> pd.DataFrame`.

`build_default_registry()` returns the production
`GoldBuildersRegistry` in the order the legacy
`RefreshAnalyticsStoreUseCase.execute()` emitted parquets — this order
is the bit-identical contract per ADR-0005 §Consequences and is also
the topological order honoring each builder's `requires_gold`. The
topology test at
`tests/unit/domain/services/gold_builders/test_topology.py` enforces
this; do not reorder without updating the smoke regression baseline.
"""
from __future__ import annotations

from src.domain.services.gold_builders.base import (
    AnalyticsSnapshot,
    BuildContext,
    GoldBuilder,
    GoldBuilderRequirementError,
    GoldBuilderSnapshot,
    GoldBuildersRegistry,
    PrimaryQuantileContract,
    normalize_parent_sweep_id_for_merge,
)
from src.domain.services.gold_builders.confidence import (
    FeatureContribLocalSummaryGoldBuilder,
    FeatureImpactByHorizonGoldBuilder,
    ModelDecisionFinalGoldBuilder,
    PredictionGeneralizationGapGoldBuilder,
    PredictionRiskGoldBuilder,
    PredictionRobustnessByHorizonGoldBuilder,
    QualityRunSweepSummaryGoldBuilder,
    QualityStatisticsReportGoldBuilder,
)
from src.domain.services.gold_builders.descriptive import (
    FeatureSetImpactGoldBuilder,
    Ic95GoldBuilder,
    OosConsolidatedGoldBuilder,
    OosQualityReportGoldBuilder,
    PredictionCalibrationGoldBuilder,
    PredictionMetricsByHorizonGoldBuilder,
)
from src.domain.services.gold_builders.pairwise import (
    DmPairwiseResultsGoldBuilder,
    McsResultsGoldBuilder,
    PairedOosIntersectionByHorizonGoldBuilder,
    WinRatePairwiseResultsGoldBuilder,
)
from src.domain.services.gold_builders.quantile import (
    PredictionMetricsByConfigGoldBuilder,
    PredictionMetricsByRunSplitHorizonGoldBuilder,
    QuantileDegeneracyReportGoldBuilder,
    QuantileGuardrailAuditGoldBuilder,
)
from src.domain.services.gold_builders.ranking import (
    ConsistencyTopkGoldBuilder,
    RankingByConfigGoldBuilder,
    RunsLongGoldBuilder,
)
from src.domain.services.quantile_contract_analyzer import (
    QuantileDegeneracyThresholds,
)


def build_default_registry(
    *,
    degeneracy_thresholds: QuantileDegeneracyThresholds | None = None,
    paired_intersection_horizons: tuple[int, ...] = (1, 7, 30),
) -> GoldBuildersRegistry:
    """Build the production GoldBuildersRegistry with the canonical 25
    gold builders in the exact order the legacy
    `RefreshAnalyticsStoreUseCase.execute()` emitted parquets.

    The registration order is the bit-identical contract per ADR-0005
    §Consequences AND the topological order honoring `requires_gold`;
    do not reorder without also changing the smoke regression baseline
    and the topology test.
    """
    degeneracy = degeneracy_thresholds or QuantileDegeneracyThresholds()

    reg = GoldBuildersRegistry()

    # --- Tier 1: silver/dim only (ordering matches monolith execute()) ---
    reg.register(RunsLongGoldBuilder())
    reg.register(RankingByConfigGoldBuilder())
    reg.register(ConsistencyTopkGoldBuilder())
    reg.register(Ic95GoldBuilder())
    reg.register(FeatureSetImpactGoldBuilder())
    reg.register(OosConsolidatedGoldBuilder())
    reg.register(PredictionMetricsByRunSplitHorizonGoldBuilder())
    reg.register(QuantileGuardrailAuditGoldBuilder())
    reg.register(QuantileDegeneracyReportGoldBuilder(thresholds=degeneracy))

    # --- Tier 2: consume gold_prediction_metrics_by_run_split_horizon ---
    reg.register(PredictionMetricsByConfigGoldBuilder())
    reg.register(PredictionMetricsByHorizonGoldBuilder())
    reg.register(PredictionCalibrationGoldBuilder())
    reg.register(PredictionRiskGoldBuilder())
    reg.register(PredictionGeneralizationGapGoldBuilder())
    reg.register(PredictionRobustnessByHorizonGoldBuilder())
    reg.register(FeatureImpactByHorizonGoldBuilder())

    # --- Tier 1 (resumes): feature contrib + remaining pairwise inputs ---
    reg.register(FeatureContribLocalSummaryGoldBuilder())
    reg.register(OosQualityReportGoldBuilder())
    reg.register(DmPairwiseResultsGoldBuilder())
    reg.register(McsResultsGoldBuilder())
    reg.register(WinRatePairwiseResultsGoldBuilder())
    reg.register(PairedOosIntersectionByHorizonGoldBuilder(target_horizons=paired_intersection_horizons))

    # --- Tier 3: consume multiple upstream gold tables ---
    reg.register(ModelDecisionFinalGoldBuilder())
    reg.register(QualityStatisticsReportGoldBuilder())
    reg.register(QualityRunSweepSummaryGoldBuilder())

    return reg


__all__ = [
    "AnalyticsSnapshot",
    "BuildContext",
    "ConsistencyTopkGoldBuilder",
    "DmPairwiseResultsGoldBuilder",
    "FeatureContribLocalSummaryGoldBuilder",
    "FeatureImpactByHorizonGoldBuilder",
    "FeatureSetImpactGoldBuilder",
    "GoldBuilder",
    "GoldBuilderRequirementError",
    "GoldBuilderSnapshot",
    "GoldBuildersRegistry",
    "Ic95GoldBuilder",
    "McsResultsGoldBuilder",
    "ModelDecisionFinalGoldBuilder",
    "OosConsolidatedGoldBuilder",
    "OosQualityReportGoldBuilder",
    "PairedOosIntersectionByHorizonGoldBuilder",
    "PredictionCalibrationGoldBuilder",
    "PredictionGeneralizationGapGoldBuilder",
    "PredictionMetricsByConfigGoldBuilder",
    "PredictionMetricsByHorizonGoldBuilder",
    "PredictionMetricsByRunSplitHorizonGoldBuilder",
    "PredictionRiskGoldBuilder",
    "PredictionRobustnessByHorizonGoldBuilder",
    "PrimaryQuantileContract",
    "QualityRunSweepSummaryGoldBuilder",
    "QualityStatisticsReportGoldBuilder",
    "QuantileDegeneracyReportGoldBuilder",
    "QuantileGuardrailAuditGoldBuilder",
    "RankingByConfigGoldBuilder",
    "RunsLongGoldBuilder",
    "WinRatePairwiseResultsGoldBuilder",
    "build_default_registry",
    "normalize_parent_sweep_id_for_merge",
]
