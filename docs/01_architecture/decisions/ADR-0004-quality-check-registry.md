---
title: ADR-0004 QualityCheckRegistry para validate_analytics_quality
scope: Decisao arquitetural de extrair os ~20 checks heterogeneos hoje empilhados
  no metodo execute() de validate_analytics_quality_use_case.py para um registry
  de classes QualityCheck independentes, com selecao por scope e teste isolado.
  Status Accepted.
update_when:
  - decisao for revisada (status mudar de Accepted para Superseded)
  - contrato base QualityCheck mudar (signature de run/applies_when)
canonical_for: [adr_0004, quality_check_registry, validate_analytics_quality_architecture]
---

# ADR-0004: QualityCheckRegistry

## Status
Accepted (2026-05-19); **Implemented** em Stage R-21 (PR ainda aberta em
2026-05-21). Migracao completa dos 27 checks (count corrigido vs ~20
estimados no Context) para 4 cluster files; orchestrator reduzido de
1.152 LOC para 307 LOC. Bit-identical verificado empiricamente contra
`data/analytics_archive_pre_phase_b/silver` em ambos `scope_mode`
(`cohort_decision` + `global_health`), com fixtures gravadas em
`tests/integration/fixtures/quality_checks/`.

## Context

[`validate_analytics_quality_use_case.py`](../../../src/use_cases/validate_analytics_quality_use_case.py)
tem 1.152 LOC com **~20 checks** registrados num unico `execute()` via
`self._record(checks, "<name>", passed, detail)`. Lista parcial:

- `cardinality_config_fold_seed`, `min_samples_by_split`,
  `run_id_execution_consistency`, `inference_predictions_continuity`,
  `feature_contrib_local_continuity`, `required_metrics_nan`,
  `dm_mcs_persisted_executable`, `gold_confidence_calibrated_by_horizon`,
  `gold_metrics_by_config_n_oos_contract`, `baseline_present_per_candidate_sweep`,
  `tft_baselines_timestamp_subset_alignment`, `official_contract_quantile_attention`,
  `oos_pairwise_target_alignment`, `referential_integrity`, ...

Cada Stage do PHASE_B adicionou ou modificou checks; Stage F.0 adicionou
2 filtros (F.0.6 `is_quantile_genuine`, F.0.7 `baseline` exclusion) e 1
gate novo (F.0.3 `tft_baselines_timestamp_subset_alignment`).

Caso clinico de "ajusta aqui, esquece la":
[`584373e`](../../../) introduziu F.0.3 e
[`ab79996`](../../../) corrigiu-o uma hora depois ("F.0.3 enrich oos_view
com parent_sweep_id via dim_run join") — a versao inicial silenciosamente
skipava com motivo `missing_group_columns`. Causa: autor do check novo
nao tinha visibilidade do schema implicito navegado pelos outros 19 checks.

Cada novo check precisa entender 20 vizinhos para nao regredir; cada
modificacao toca uma classe de 1.152 LOC; cobertura de teste mistura
fixtures de checks nao-relacionados.

## Decision

Extrair os checks para classes pequenas com contrato comum:

```
src/domain/services/quality_checks/

class QualityCheck(ABC):
    name: str
    @abstractmethod
    def applies_when(self, scope: ScopeSpec) -> bool: ...
    @abstractmethod
    def run(self, data: AnalyticsSnapshot) -> CheckResult: ...

class QualityCheckRegistry:
    def register(self, check: QualityCheck) -> None: ...
    def applicable(self, scope: ScopeSpec) -> list[QualityCheck]: ...
```

Estrutura:

- `quality_checks/base.py` — `QualityCheck`, `CheckResult`,
  `AnalyticsSnapshot`, `QualityCheckRegistry`.
- `quality_checks/cardinality.py` — checks de cardinalidade,
  min_samples, continuity.
- `quality_checks/alignment.py` — `tft_baselines_timestamp_subset_alignment`,
  `oos_pairwise_target_alignment`.
- `quality_checks/calibration.py` — `confidence_calibrated`,
  `dm_mcs_executable`, `n_oos_contract`.
- `quality_checks/contracts.py` — `official_contract_quantile_attention`,
  `required_metrics_nan`, `baseline_present`.
- `quality_checks/__init__.py` — exporta registry default povoada.

`ValidateAnalyticsQualityUseCase.execute()` reduz a:

```python
checks_to_run = self._registry.applicable(self._resolve_scope_spec())
results = [check.run(self._snapshot) for check in checks_to_run]
```

Cada check vive em **um arquivo proprio com testes proprios**. Adicionar
novo check = criar 1 classe + 1 arquivo de teste, sem tocar nos outros 19.

## Consequences

**Positivas:**

- Blast-radius de checks novos cai para 1 arquivo.
- Cobertura de teste granular: `tests/unit/domain/services/quality_checks/test_alignment.py`
  testa so alignment checks.
- `applies_when(scope)` torna explicita a aplicabilidade por
  `scope_mode` — hoje implicito em conditionals dentro de `execute()`.
- Caso F.0.3/ab79996 ("missing_group_columns" mascarado) vira impossivel:
  a fixture do check obriga o autor a declarar dependencias de schema.
- `AnalyticsSnapshot` (struct dos parquets carregados) materializa o
  schema implicito que hoje cada check navega ad-hoc.

**Negativas:**

- Migracao de ~20 checks heterogeneos sem regressao exige cobertura de
  smoke regression (mesmo input → mesmo output) — Task 21.7 de Stage 21.
- Adiciona ~5 arquivos novos onde antes havia 1; navegar requer indice.
- `CheckResult` precisa preservar shape exato do `dict` atual (`name`,
  `passed`, `detail`) para nao quebrar consumers downstream (gates de CI,
  parsers de relatorios).

## Cross-link

- Diagnostico: [`docs/07_reports/phase-gates/B_architectural_debt_2026-05-19.md`](../../07_reports/phase-gates/B_architectural_debt_2026-05-19.md) §"M-validate_quality"
- Implementacao: PHASE_B_IMPLEMENTATION_CHECKLIST.md §Stage 21 + Stage R-21 (remediation plan)
- Caso clinico: commits `584373e` (F.0.3 inicial) + `ab79996` (re-fix do mesmo F.0.3)
- Stage R-21 implementacao final: [`docs/ai/STAGE_20_23_REMEDIATION_PLAN.md`](../../ai/STAGE_20_23_REMEDIATION_PLAN.md) §"Stage R-21"
- Extension point (como adicionar check novo): [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../ANALYTICS_STORE_ARCHITECTURE.md) §"Quality check extension point"
