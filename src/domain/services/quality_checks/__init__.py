"""Quality check registry for analytics validation.

Per ADR-0004, quality checks live as small `QualityCheck` classes
registered in a `QualityCheckRegistry`. The orchestrator (use case)
loads the data snapshot once, then iterates the registry per scope.

`build_default_registry()` returns the production registry: 27 checks
in the exact order the monolithic
`ValidateAnalyticsQualityUseCase.execute()` emitted them, preserving
bit-identical contract per ADR-0004 §Consequences.
"""
from __future__ import annotations

from src.domain.services.quality_checks.alignment import (
    BaselinesSharedParentSweepCheck,
    OosPairwiseTargetAlignmentCheck,
    TftBaselinesTimestampSubsetAlignmentCheck,
)
from src.domain.services.quality_checks.base import (
    AnalyticsSnapshot,
    CheckResult,
    QualityCheck,
    QualityCheckRegistry,
)
from src.domain.services.quality_checks.calibration import (
    BlockQuantileDegeneracyGateCheck,
    DmMcsPersistedExecutableCheck,
    GoldConfidenceCalibratedByHorizonCheck,
    GoldMetricsByConfigNOosContractCheck,
    OosQuantileBlockAAcceptanceCheck,
)
from src.domain.services.quality_checks.cardinality import (
    CardinalityConfigFoldSeedCheck,
    FeatureContribLocalContinuityCheck,
    InferencePredictionsContinuityCheck,
    MinSamplesBySplitCheck,
    RequiredMetricsNanCheck,
    RequiredTablesPresenceCheck,
    RunIdExecutionConsistencyCheck,
)
from src.domain.services.quality_checks.contracts import (
    InferencePredictionsNumericTypesCheck,
    InferencePredictionsQuantileOrderCheck,
    InferencePredictionsUniqueKeyCheck,
    OfficialContractQuantileAttentionCheck,
    OosHorizonCoverageCheck,
    OosIntervalWidthNonNegativeCheck,
    OosNumericTypesCheck,
    OosQuantileOrderCheck,
    OosSupervisedNullsCheck,
    OosUniqueKeyCheck,
    ReferentialIntegrityCheck,
    TemporalConsistencyCheck,
)
from src.domain.services.quantile_contract_analyzer import (
    QuantileBlockAThresholds,
    QuantileDegeneracyThresholds,
)


def build_default_registry(
    *,
    min_samples_train: int = 1,
    min_samples_val: int = 1,
    min_samples_test: int = 1,
    block_a_thresholds: QuantileBlockAThresholds | None = None,
    degeneracy_thresholds: QuantileDegeneracyThresholds | None = None,
) -> QualityCheckRegistry:
    """Build the production QualityCheckRegistry with the canonical 27
    checks in the order the legacy
    `ValidateAnalyticsQualityUseCase.execute()` emitted them.

    The registration order is the bit-identical contract per
    ADR-0004 §Consequences; do not reorder without also changing the
    smoke regression baseline.
    """
    block_a = block_a_thresholds or QuantileBlockAThresholds()
    degeneracy = degeneracy_thresholds or QuantileDegeneracyThresholds()

    reg = QualityCheckRegistry()

    # --- Block 1: presence + continuity + run identity (lines 337-448) ---
    reg.register(RequiredTablesPresenceCheck())
    reg.register(InferencePredictionsContinuityCheck())
    reg.register(FeatureContribLocalContinuityCheck())
    reg.register(RunIdExecutionConsistencyCheck())
    reg.register(ReferentialIntegrityCheck())
    reg.register(RequiredMetricsNanCheck())

    # --- Block 2: oos temporal/key/type/coverage/null/interval/quantile ---
    reg.register(TemporalConsistencyCheck())
    reg.register(OosUniqueKeyCheck())
    reg.register(OosNumericTypesCheck())
    reg.register(OosHorizonCoverageCheck())
    reg.register(OosSupervisedNullsCheck())
    reg.register(OosIntervalWidthNonNegativeCheck())
    reg.register(OosQuantileOrderCheck())

    # --- Block 3: quantile contract + degeneracy + pairwise alignment ---
    reg.register(OosQuantileBlockAAcceptanceCheck(thresholds=block_a))
    reg.register(BlockQuantileDegeneracyGateCheck(thresholds=degeneracy))
    reg.register(OosPairwiseTargetAlignmentCheck())

    # --- Block 4: inference predictions ---
    reg.register(InferencePredictionsUniqueKeyCheck())
    reg.register(InferencePredictionsNumericTypesCheck())
    reg.register(InferencePredictionsQuantileOrderCheck())

    # --- Block 5: cardinality + min samples ---
    reg.register(CardinalityConfigFoldSeedCheck())
    reg.register(
        MinSamplesBySplitCheck(
            min_train=min_samples_train,
            min_val=min_samples_val,
            min_test=min_samples_test,
        )
    )

    # --- Block 6: gold-backed checks (DM/MCS, confidence, n_oos contract) ---
    reg.register(DmMcsPersistedExecutableCheck())
    reg.register(GoldConfidenceCalibratedByHorizonCheck())
    reg.register(GoldMetricsByConfigNOosContractCheck())

    # --- Block 7: Stage 12 + F.0.3 alignment + official contract ---
    reg.register(BaselinesSharedParentSweepCheck())
    reg.register(TftBaselinesTimestampSubsetAlignmentCheck())
    reg.register(OfficialContractQuantileAttentionCheck())

    return reg


__all__ = [
    "AnalyticsSnapshot",
    "CheckResult",
    "QualityCheck",
    "QualityCheckRegistry",
    "build_default_registry",
]
