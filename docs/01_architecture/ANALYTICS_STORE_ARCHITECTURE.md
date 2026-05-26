---
title: Analytics Store Architecture
scope: Camadas, contratos e principios da analytics store (Parquet silver source-of-truth + Parquet/DuckDB gold reconstruivel). Inclui tabela de contratos minimos por tabela e referencia ADRs.
update_when:
  - nova camada (alem de silver/gold) for introduzida
  - politica de append-only/upsert mudar
  - particionamento canonico mudar
  - contrato minimo de tabela silver/gold for revisado
canonical_for: [analytics_store, silver_layer, gold_layer, parquet_duckdb, table_contracts]
---

# Analytics Store Architecture

## Proposito
Documentar as camadas, os contratos e os principios da analytics store
(Parquet silver + DuckDB/Parquet gold), que e o substrato de toda decisao
estatistica do projeto sem necessidade de retrain. Nao cobre o catalogo de
metricas (ver `04_evaluation/`) nem o fluxo end-to-end (ver `DATA_FLOW.md`).

## Conceitos-chave
- **Silver:** fatos atomicos (uma linha por evento) e dimensoes (`dim_run`,
  `fact_config`, `fact_oos_predictions`, `fact_split_metrics`,
  `fact_epoch_metrics`, etc.). Source of truth.
- **Gold:** tabelas derivadas e agregadas para decisao
  (`gold_prediction_metrics_by_config`, `gold_dm_pairwise_results`,
  `gold_mcs_results`, `gold_model_decision_final`, etc.).
- **`run_id`:** identidade logica deterministica de um experimento — hash
  estavel de `(asset, feature_set_hash, trial_number, fold, seed,
  model_version, config_signature, split_signature, pipeline_version)`.
- **`schema_version`:** coluna obrigatoria em toda `dim_*`/`fact_*` para
  evolucao de schema com retrocompatibilidade controlada.
- **`schema_version=2` (2026-05-17):** adiciona `training_run_id` em
  `fact_model_artifacts` e nas tabelas silver de inferencia para materializar
  a FK inferencia -> treino exigida pela Lei 3. Em rows legadas pre-Stage 10,
  `training_run_id` pode permanecer `None` ate backfill resolvivel ou pendencia
  manual documentada.
  Para consumidores gold, `inference_run_id` identifica o batch operacional de
  inferencia, enquanto `training_run_id` identifica o `dim_run.run_id` do treino
  que gerou o artefato carregado; nas tabelas `fact_inference_*`, `run_id`
  replica `training_run_id` para preservar joins historicos.
- **Fingerprints:** `dataset_fingerprint`, `split_fingerprint`,
  `feature_set_hash`, `config_signature` — hashes que garantem que dois runs
  com o mesmo `run_id` realmente usaram exatamente os mesmos dados/configs.
- **Scope:** filtro por `parent_sweep_id` (ou `config_signature`) para
  garantir que comparacoes estatisticas usem coortes homogeneas.

## Decisoes e escolhas
- **Silver Parquet como source of truth, DuckDB reconstruivel.** *Why:*
  Parquet e portavel e auditavel; DuckDB acelera consultas analiticas mas pode
  ser regenerado a qualquer momento sem perda de informacao.
- **Append-only por padrao em fatos; upsert apenas em reprocessamento explicito.**
  *Why:* preserva historico completo para auditoria; reprocessamento e operacao
  consciente, nao silenciosa.
- **`run_id` deterministico (hash de campos canonicos).** *Why:* mesmo experimento
  sempre gera mesmo `run_id`, permitindo deduplicacao e idempotencia.
- **Particionamento Parquet por `asset` + chaves de alta cardinalidade.** *Why:*
  consultas analiticas filtram quase sempre por ativo; reduz I/O em uma ordem
  de grandeza.
- **`parent_sweep_id` propagado ate as tabelas gold de decisao.** *Why:* sem
  isso, qualquer agregacao mistura coortes silenciosamente — invalida DM/MCS.

## Archive pre-Phase B

`data/analytics_archive_pre_phase_b/` preserva o snapshot historico de
`data/analytics/silver/` e `data/analytics/gold/` anterior a `2026-05-10`.
Esse material e diagnostico historico/exploratorio, nao evidencia
confirmatoria da Phase B. A razao esta registrada no gate M6 de
`docs/07_reports/phase-gates/A_code_audit.md`: o store historico mistura runs
pre-ScopeSpec, `parent_sweep_id=NULL` e tabelas gold que ainda nao preservam
escopo de coorte em todos os agregados.

O archive deve permanecer read-only e nao deve ser modificado ate o fechamento
do pre-registro da Phase B. Se Stages 4+ precisarem de rollback de dados,
restaurar a partir de `data/analytics_archive_pre_phase_b/` e o caminho oficial;
nao recomputar nem editar manualmente o snapshot arquivado.

### Archive Phase B confirmatorio (2026-05-25)

`data/analytics_archive_phase_b_20260524/` (read-only, `chmod a-w` aplicado em
Stage E6.3 do
[`PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md))
preserva o subset de evidencia confirmatoria da Phase B filtrado por
`parent_sweep_id=phase_b_confirmatorio_20260524`. Tamanho total: 9.4M.
Conteudo:

- `silver/dim_run/sweep_id=phase_b_confirmatorio_20260524/` — 60 dim_run
  rows (15 TFT + 45 baseline) com `fold` e `seed` populados per Emenda E1.7.
- `silver/fact_oos_predictions/` — 330.050 rows filtradas por `run_id` nos
  60 da cohort (todas as 21 particoes hive que continham linhas da cohort).
- `gold/` — 24 das 25 tabelas gold filtradas por `parent_sweep_id` (uma
  tabela nao contem coluna `parent_sweep_id` por design ou ficou vazia
  para a cohort; nao bloqueador).
- `sidecars/` — 5 sidecars Phase B (PR-A4, Emenda E1.8):
  `phase_b_marginal_coverage.parquet`, `phase_b_dm_family_6.parquet`,
  `phase_b_dm_family_18_sensitivity.parquet`, `phase_b_delta_pinball.parquet`,
  `phase_b_tier_verdict.parquet`.

Dados referenciados pelo relatorio
[`docs/07_reports/phase-gates/B_confirmatory_2026-05-25.md`](../07_reports/phase-gates/B_confirmatory_2026-05-25.md)
e pelo pre-registro §15. Nao modificar. Em caso de necessidade de re-derivar
metricas, reproduzir via
`config/sweeps/explicit/phase_b_confirmatorio_20260524.json` (sealed sha256
`fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4`) +
`main_compute_phase_b_tier_metrics` (runbook
[`docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md`](../06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md)).
Archive nao versionado em git (parquets cobertos por `.gitignore:data/**`);
apenas esta nota documental.

## Contratos minimos por linha (exemplos)

| Tabela | Grao logico | Chaves obrigatorias |
|---|---|---|
| `dim_run` | 1 linha por run | `run_id`, `parent_sweep_id`, `asset`, `model_version`, `created_at_utc`, `status` |
| `fact_config` | 1 linha por run | `run_id`, `prediction_mode`, `loss_name`, `quantile_levels_json` |
| `fact_oos_predictions` | 1 linha por (`run_id`, `split`, `horizon`, `target_timestamp`) | + `y_true`, `y_pred`, `quantile_p10/50/90`, `quantile_p*_post_guardrail` |
| `gold_prediction_metrics_by_config` | 1 linha por (`config_signature`, `split`, `horizon`) | + agregados (mean/std/iqr) por metrica |
| `gold_dm_pairwise_results` | 1 linha por par de candidatos | `parent_sweep_id`, `aligned_timestamps`, `dm_stat`, `pvalue_adj_holm` |

## Quality check extension point (ADR-0004)

`ValidateAnalyticsQualityUseCase` e um orchestrator fino: carrega
silver/gold + escopo, monta um `AnalyticsSnapshot`, e itera um
`QualityCheckRegistry` (`src/domain/services/quality_checks/`). Cada
check vive como uma `QualityCheck` subclass com `name`, `applies_when`
e `run(snapshot)`.

Para **adicionar um novo check**:

```python
# src/domain/services/quality_checks/<cluster>.py
from src.domain.services.quality_checks.base import (
    AnalyticsSnapshot,
    CheckResult,
    QualityCheck,
)


class MyNewCheck(QualityCheck):
    name = "my_new_check"

    def applies_when(self, scope):
        # Default herda True; override para scope-specific gating, e.g.:
        # return scope is not None and scope.scope_mode == "cohort_decision"
        return True

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        df = snapshot.get("fact_oos_predictions")
        if df.empty:
            return CheckResult(self.name, True, "ok")
        # ... compute issues
        return CheckResult(self.name, passed=..., detail=...)
```

E **registrar na ordem topologica** no factory
`build_default_registry()` em
[`src/domain/services/quality_checks/__init__.py`](../../src/domain/services/quality_checks/__init__.py).
A ordem de registro e o contrato bit-identical do ADR-0004 §Consequences;
reordenar requer atualizar tambem o teste
`tests/unit/domain/services/quality_checks/test_registry_order.py` +
regenerar as fixtures em
`tests/integration/fixtures/quality_checks/archive_silver_*.json` rodando
contra `data/analytics_archive_pre_phase_b/silver` no estado pre-mudanca.

DI: o use case aceita `registry=` no construtor para testes. Default e
o `build_default_registry()` com os 27 checks canonicos.

## Gold builder extension point (ADR-0005)

`RefreshAnalyticsStoreUseCase` segue o mesmo padrao do validate: e um
orchestrator fino (300 LOC pos-R-22, vs 2.579 LOC pre-R-22) que carrega
silver/dim + escopo, monta um `GoldBuilderSnapshot`, instancia
`BuildContext` (com `primary_quantile_contract` + `gold_outputs` dict),
e itera um `GoldBuildersRegistry` (`src/domain/services/gold_builders/`).
Cada builder vive como uma `GoldBuilder` subclass com `output_table`,
`requires` (silver/dim consumidas), `requires_gold` (gold tables
upstream — Tier 2/3), e `build(snapshot, ctx) -> pd.DataFrame`.

Para **adicionar um novo builder**:

```python
# src/domain/services/gold_builders/<cluster>.py
from src.domain.services.gold_builders.base import (
    BuildContext,
    GoldBuilder,
    GoldBuilderSnapshot,
)


class MyNewGoldBuilder(GoldBuilder):
    output_table = "gold_my_new_metric"
    requires = ("dim_run", "fact_oos_predictions")  # silver/dim
    requires_gold = ()  # Tier 1; for Tier 2 set to upstream gold tables

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        # Tier 2/3 read upstream gold from ctx.gold_outputs[...]
        # ctx.primary_quantile_contract is available for raw vs post-guardrail
        return df  # the orchestrator persists to gold/<output_table>.parquet
```

E **registrar na ordem topologica** no factory `build_default_registry()`
em [`src/domain/services/gold_builders/__init__.py`](../../src/domain/services/gold_builders/__init__.py).
A ordem de registro e simultaneamente: (a) o contrato bit-identical do
ADR-0005 §Consequences (sentinel test em
`tests/integration/test_gold_builders_byte_identical_archive.py`); (b) a
ordem topologica honrando `requires_gold` — Tier 2/3 builders precisam
aparecer estritamente depois de seus upstreams.

Topology test em
[`tests/unit/domain/services/gold_builders/test_topology.py`](../../tests/unit/domain/services/gold_builders/test_topology.py)
falha rapido se a ordem for violada; orchestrator tambem enforca em
runtime via `GoldBuilderRequirementError` antes de chamar `build()`.

Reordenar a registry requer:
1. Atualizar `test_topology.py::test_registry_output_tables_match_monolith_emit_sequence`
2. Regenerar fixtures em `tests/integration/fixtures/gold_builders/byte_identical_archive/*.json`
   rodando `RefreshAnalyticsStoreUseCase` contra
   `data/analytics_archive_pre_phase_b/silver/` no estado pre-mudanca

DI: o use case aceita `registry=` no construtor para testes. Default e
o `build_default_registry()` com os 25 builders canonicos.

## Fonte da verdade (codigo)
- `src/infrastructure/schemas/analytics_store_schema.py` — definicao de schemas
- `src/use_cases/refresh_analytics_store_use_case.py` — orchestrator fino sobre gold_builders (ADR-0005)
- `src/use_cases/validate_analytics_quality_use_case.py` — orchestrator fino sobre quality_checks
- `src/domain/services/gold_builders/` — 25 builders em 5 clusters (ranking/descriptive/quantile/pairwise/confidence), `GoldBuildersRegistry`, `GoldBuilderSnapshot`, `BuildContext`, `build_default_registry()` factory (ADR-0005)
- `src/domain/services/quality_checks/` — 27 checks em 4 clusters (cardinality/alignment/calibration/contracts), `QualityCheckRegistry`, `AnalyticsSnapshot`, `build_default_registry()` factory (ADR-0004)
- `src/adapters/parquet_analytics_run_repository.py` — leitura/escrita silver
- `src/domain/services/scope_spec.py` — governanca de escopo

## Documentos relacionados
- `01_architecture/DATA_FLOW.md` — fluxo end-to-end raw -> processed -> silver -> gold
- `02_data/DATA_CONTRACTS.md` — contratos detalhados de chaves e fingerprints
- `02_data/QUALITY_GATES.md` — checks aplicados durante refresh
- `05_checklists/ANALYTICS_STORE_CHECKLIST.md` — historico de implementacao
- `01_architecture/decisions/ADR-0001-analytics-store.md` — decisao arquitetural inicial
