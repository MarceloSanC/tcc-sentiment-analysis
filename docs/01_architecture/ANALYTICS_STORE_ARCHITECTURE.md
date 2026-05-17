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

## Contratos minimos por linha (exemplos)

| Tabela | Grao logico | Chaves obrigatorias |
|---|---|---|
| `dim_run` | 1 linha por run | `run_id`, `parent_sweep_id`, `asset`, `model_version`, `created_at_utc`, `status` |
| `fact_config` | 1 linha por run | `run_id`, `prediction_mode`, `loss_name`, `quantile_levels_json` |
| `fact_oos_predictions` | 1 linha por (`run_id`, `split`, `horizon`, `target_timestamp`) | + `y_true`, `y_pred`, `quantile_p10/50/90`, `quantile_p*_post_guardrail` |
| `gold_prediction_metrics_by_config` | 1 linha por (`config_signature`, `split`, `horizon`) | + agregados (mean/std/iqr) por metrica |
| `gold_dm_pairwise_results` | 1 linha por par de candidatos | `parent_sweep_id`, `aligned_timestamps`, `dm_stat`, `pvalue_adj_holm` |

## Fonte da verdade (codigo)
- `src/infrastructure/schemas/analytics_store_schema.py` — definicao de schemas
- `src/use_cases/refresh_analytics_store_use_case.py` — materializacao gold
- `src/use_cases/validate_analytics_quality_use_case.py` — quality gates
- `src/adapters/parquet_analytics_run_repository.py` — leitura/escrita silver
- `src/domain/services/scope_spec.py` — governanca de escopo

## Documentos relacionados
- `01_architecture/DATA_FLOW.md` — fluxo end-to-end raw -> processed -> silver -> gold
- `02_data/DATA_CONTRACTS.md` — contratos detalhados de chaves e fingerprints
- `02_data/QUALITY_GATES.md` — checks aplicados durante refresh
- `05_checklists/ANALYTICS_STORE_CHECKLIST.md` — historico de implementacao
- `01_architecture/decisions/ADR-0001-analytics-store.md` — decisao arquitetural inicial
