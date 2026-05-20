---
title: ADR-0005 GoldBuilders modulares por categoria
scope: Decisao arquitetural de extrair os 26 builders de tabelas gold hoje
  inline em refresh_analytics_store_use_case.py para modulos por categoria em
  src/domain/services/gold_builders/. Status Accepted.
update_when:
  - decisao for revisada (status mudar de Accepted para Superseded)
  - nova categoria de gold table for introduzida
  - contrato de GoldBuilder mudar (signature de build)
canonical_for: [adr_0005, gold_builders_modularization, refresh_analytics_store_architecture]
---

# ADR-0005: GoldBuilders modulares por categoria

## Status
Accepted (2026-05-19)

## Context

[`refresh_analytics_store_use_case.py`](../../../src/use_cases/refresh_analytics_store_use_case.py)
tem **2.568 LOC** com **26 builders** privados:

`_build_gold_runs_long`, `_build_gold_ranking_by_config`,
`_build_gold_consistency_topk`, `_build_gold_ic95`,
`_build_gold_feature_set_impact`, `_build_gold_prediction_metrics_*` (4
variantes), `_build_gold_quantile_guardrail_audit`,
`_build_gold_quantile_degeneracy_report`, `_build_gold_dm_pairwise_results`,
`_build_gold_mcs_results`, `_build_gold_paired_oos_intersection_by_horizon`,
`_build_gold_model_decision_final`, `_build_gold_confidence_calibrated_by_horizon`,
... (26 total).

Cada Stage do PHASE_B tocou um subset:
- Stage 5/6 — 5 builders (cohort-aware groupby + parent_sweep_id)
- Stage 8 — builders de raw + post-guardrail
- Stage 9 — builders de quantile filtrados por `is_quantile_genuine`
- Stage 10 — builders com FK `training_run_id`
- Stage 11 — `_build_gold_quantile_degeneracy_report`
- Stage 12 — builders de baselines

Heatmap dos ultimos 60 commits mostra `refresh_analytics_store_use_case.py`
com 3 modificacoes; soma com `validate_analytics_quality_use_case.py` (5)
e `run_baselines_use_case.py` (4) totaliza 12 modificacoes (20% dos
commits) em 3 arquivos.

Cada builder e logica de negocio com semantica propria (ranking statistic,
DM test, MCS, intersection cardinality) mas vive como metodo `@classmethod`
privado numa classe `RefreshAnalyticsStoreUseCase` — ja **god object**:
47 metodos, 26 builders, 5 categorias de responsabilidade misturadas
(I/O, scope resolution, normalizacao, builders, orquestracao).

## Decision

Extrair builders para pacote `gold_builders/` organizado por categoria
semantica:

```
src/domain/services/gold_builders/

base.py            — GoldBuilder ABC, BuildContext, build(snapshot) -> pd.DataFrame
ranking.py         — runs_long, ranking_by_config, consistency_topk
pairwise.py        — dm_pairwise_results, mcs_results, paired_oos_intersection_by_horizon
quantile.py        — quantile_guardrail_audit, quantile_degeneracy_report,
                     prediction_metrics_by_run_split_horizon (+ single_contract variant),
                     prediction_metrics_by_config
descriptive.py     — ic95, feature_set_impact, model_decision_final
confidence.py      — confidence_calibrated_by_horizon, metrics_by_config_n_oos_contract
__init__.py        — exporta builders registry
```

Contrato comum:

```python
class GoldBuilder(ABC):
    output_table: str
    requires: list[str]   # silver/dim tables required
    @abstractmethod
    def build(self, snapshot: AnalyticsSnapshot, ctx: BuildContext) -> pd.DataFrame: ...
```

`RefreshAnalyticsStoreUseCase` reduz a orquestrador:

```python
for builder in self._gold_builders.applicable(self._scope_spec):
    df = builder.build(snapshot, ctx)
    self._safe_write(df, self._path_for(builder.output_table))
```

Cada categoria vive em **1 arquivo de ~300-500 LOC** com testes proprios
em `tests/unit/domain/services/gold_builders/test_<categoria>.py`.

## Consequences

**Positivas:**

- `refresh_analytics_store` cai de 2.568 LOC para ~400 LOC
  (orquestracao + I/O + scope filtering).
- Stage 5/6 pattern (mesmos 5 builders reabertos uma semana depois)
  vira impossivel: alguem mexendo em `ranking.py` nao toca `pairwise.py`.
- Testes granulares: refator de DM nao roda fixtures de quantile.
- `requires` torna explicito o schema dependency hoje implicito.
- Habilita future Stage 16 (Caminho C "hook sweep fechado"): cada
  builder e invocavel independente para write-time em vez de bulk refresh.

**Negativas:**

- Maior reorganizacao das 3: ~2.600 LOC de logica reorganizada (sem
  mudanca comportamental). Tests proporcionais (~2.500 LOC).
- Diff de Stage 22 sera grande (~5.000 LOC moved). Mitigacao em
  PHASE_B §Stage 22: tasks incrementais por categoria (22.2-22.6),
  cada categoria validada em smoke regression byte-identical antes de
  passar para a proxima.
- Adiciona indireccao no grep: "onde esta DM?" precisa de indice
  exportado em `gold_builders/__init__.py`.

## Cross-link

- Diagnostico: [`docs/07_reports/phase-gates/B_architectural_debt_2026-05-19.md`](../../07_reports/phase-gates/B_architectural_debt_2026-05-19.md) §"M5-refresh"
- Implementacao: PHASE_B_IMPLEMENTATION_CHECKLIST.md §Stage 22
- Relacionado: ADR-0001 (Analytics Store como source of truth — este ADR refina internals do builder side)
- Habilita future: PHASE_B §Stages 15-18 (Caminho C write-time per-run) se forem ativados.
