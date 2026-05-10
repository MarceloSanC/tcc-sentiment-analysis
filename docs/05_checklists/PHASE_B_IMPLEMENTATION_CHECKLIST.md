---
title: Phase B Implementation Checklist (Pre-Experiment Engineering)
scope: Passo a passo tecnico de implementacao das alteracoes de codigo exigidas pelo gate de Phase A (docs/07_reports/phase-gates/A_code_audit.md). Cada Stage = 1 PR; cada task dentro do Stage = 1 commit. Caminho B obrigatorio; Caminho C opcional. Para definicoes canonicas, ver 01_architecture/ANALYTICS_STORE_ARCHITECTURE.md e o proprio A_code_audit.md.
update_when:
  - status (checkbox) de uma task for atualizado
  - novo achado da auditoria for promovido para implementacao
  - decisao Caminho B vs C mudar
  - Stage for fechado (PR mergeado) ou reaberto
canonical_for: [phase_b_implementation_checklist, phase_b_pre_experiment_engineering, audit_to_code_traceability]
---

# Phase B — Implementation Checklist

**Status:** pendente
**Criado:** 2026-05-10
**Pre-requisito:** [A_code_audit.md](../07_reports/phase-gates/A_code_audit.md) com todos os modulos P0 em GREEN ou YELLOW com acao concluida.
**Bloqueante para:** abertura formal da Fase B (treino confirmatorio).

Este checklist traduz as Acoes do `A_code_audit.md` em PRs e commits. Cada Stage
abaixo eh um PR atomico; cada task numerada eh um commit que pode ser revisado
isoladamente.

## Convencao

- `[ ]` = pendente
- `[x]` = mergeado em main
- `[~]` = em revisao (PR aberto)
- `(opt)` = opcional, so se Caminho C foi escolhido
- `(B+C)` = obrigatorio em ambos os caminhos
- Cada task termina com **Aceite:** criterio que comprova que o commit esta pronto.

## Decisao Caminho B vs C

- [ ] Decisao Caminho B vs Caminho C registrada antes de iniciar Stage 4.
  - Caminho B (refresh scoped + cohort-aware gold): ~600-1.000 LOC, 4-6 d.p.
    Stages 1-13 obrigatorios; Stages 14-18 nao se aplicam.
  - Caminho C (write-time + refresh deprecado): ~2.000-3.000 LOC, 8-15 d.p.
    Stages 1-13 obrigatorios + Stages 14-18.
  - Decisao registrada em: ___________________________________________

---

## Stage 1 — Idempotencia das escritas em silver (Lei 2) (B+C)

**Objetivo:** substituir `_append_to_parquet` (concat cego) por mecanismo
overwrite-on-flag em todos os `append_*`. Pre-condicao para reset confiavel
de Phase B; sem isso, re-rodar o mesmo `run_id` duplica linhas em silver.

**Cross-link:** A_code_audit.md §M5-Q8.

### Tasks

- [ ] **1.1** Definir interface `AnalyticsRunRepository` com novo contrato
      `overwrite: bool = False` em todos os `append_*`/`upsert_*`.
      Atualizar `src/interfaces/analytics_run_repository.py`.
      **Aceite:** type checker passa; nenhuma chamada existente quebrada
      (default `overwrite=False` preserva semantica atual exceto pelo bug).

- [ ] **1.2** Implementar `_write_with_overwrite_policy` em
      [parquet_analytics_run_repository.py:94-100](../../src/adapters/parquet_analytics_run_repository.py#L94-L100).
      Comportamento: se path nao existe, escreve. Se existe e nao ha
      colisao de chave logica, append. Se existe e ha colisao e
      `overwrite=False`, levanta `DuplicateKeyError`. Se `overwrite=True`,
      remove linhas colidentes e regrava.
      **Aceite:** novo metodo cobre os 13 casos de chave logica documentados
      em `ANALYTICS_STORE_ARCHITECTURE.md`.

- [ ] **1.3** Migrar os 12 `append_*` para usar
      `_write_with_overwrite_policy`. Manter `upsert_dim_run` como esta
      (ja honra a lei).
      **Aceite:** todos os metodos em
      [parquet_analytics_run_repository.py:144-267](../../src/adapters/parquet_analytics_run_repository.py#L144-L267)
      delegam ao novo metodo.

- [ ] **1.4** Testes unitarios cobrindo: (a) primeira escrita; (b) append
      sem colisao; (c) colisao sem flag = erro; (d) colisao com flag =
      sobrescrita correta; (e) particoes diferentes nao se afetam.
      **Aceite:** `pytest tests/unit/adapters/test_parquet_analytics_run_repository.py` passa.

- [ ] **1.5** Atualizar call sites em
      `src/use_cases/train_tft_model_use_case.py` e
      `src/use_cases/run_tft_inference_use_case.py` para passar
      `overwrite=True` apenas quando o usuario explicitamente solicitar
      (nova flag `--overwrite-on-collision` em `main_train_tft.py`).
      **Aceite:** smoke test re-executavel produz erro claro quando
      `run_id` colide; com flag, sobrescreve corretamente.

---

## Stage 2 — Teste de regressao da invariante 9f0ccec (B+C)

**Objetivo:** congelar a invariante que faz `config_signature` segregar
coortes automaticamente. Sem esse teste, qualquer refactor futuro em
`_build_trainer_config` ou `compute_config_signature` quebra a hipotese
de coexistencia por coluna sem aviso.

**Cross-link:** A_code_audit.md §M5-Q6.

### Tasks

- [ ] **2.1** Adicionar teste
      `test_config_signature_changes_when_parent_sweep_id_changes` em
      [tests/unit/infrastructure/schemas/test_analytics_store_schema.py](../../tests/unit/infrastructure/schemas/test_analytics_store_schema.py).
      Cobrir: dois `training_config` iguais com `parent_sweep_id` diferente
      = hashes diferentes. Idem para `fold`, `trial_number`, `seed`.
      **Aceite:** 4 asserts independentes, todos passando.

- [ ] **2.2** Adicionar teste
      `test_config_signature_stable_across_volatile_keys` que garante que
      `created_at`/`started_at`/`ended_at`/`timestamp` continuam sendo
      ignorados (regressao do comportamento atual).
      **Aceite:** dois configs iguais exceto pelas chaves volateis = mesmo hash.

---

## Stage 3 — Reset arquivamento do silver/gold pre-Phase B (B+C)

**Objetivo:** arquivar dados historicos como evidencia exploratoria
(nao confirmatoria). So executar apos Stages 1 e 2 mergeados.

### Tasks

- [ ] **3.1** Mover `data/analytics/silver/` para
      `data/analytics_archive_pre_phase_b/silver/` (script ou comando manual,
      registrar comando exato no PR).
      **Aceite:** `data/analytics/silver/` vazio; archive populado;
      `git status` confirma tracking files inalterados (apenas dados).

- [ ] **3.2** Mover `data/analytics/gold/` para
      `data/analytics_archive_pre_phase_b/gold/`.
      **Aceite:** mesmo criterio.

- [ ] **3.3** Atualizar `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`
      com nota sobre o archive: dados pre-`2026-05-10` sao diagnostico
      historico, nao evidencia confirmatoria.
      **Aceite:** doc canonico atualizado e linkado em
      `A_code_audit.md` §M6.

---

## Stage 4 — Cohort-aware ranking em gold de decisao (B+C)

**Objetivo:** corrigir ranking cross-cohort em
`gold_model_decision_final` (Q7) e `gold_ranking_by_config`. Severidade RED
porque `gold_model_decision_final` eh a tabela final de decisao do paper.

**Cross-link:** A_code_audit.md §M5-Q7.

### Tasks

- [ ] **4.1** Adicionar `parent_sweep_id` ao groupby em
      [refresh_analytics_store_use_case.py:1564-1566](../../src/use_cases/refresh_analytics_store_use_case.py#L1564-L1566)
      (`gold_model_decision_final`: `rank_rmse`, `rank_mae`, `rank_da`).
      **Aceite:** rank eh calculado dentro de
      `(asset, parent_sweep_id, horizon)`, nao mais cross-coorte.

- [ ] **4.2** Adicionar `parent_sweep_id` ao groupby em
      [refresh_analytics_store_use_case.py:156-163](../../src/use_cases/refresh_analytics_store_use_case.py#L156-L163)
      (`gold_ranking_by_config`: `rank_test_rmse`, `rank_test_mae`,
      `rank_test_da`).
      **Aceite:** ranks sao por
      `(asset, parent_sweep_id, feature_set_name)`.

- [ ] **4.3** Testes unitarios: criar dois sweeps sinteticos com mesmas
      configs, verificar que ranks sao independentes por sweep.
      **Aceite:** `pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py`
      passa com novos casos.

---

## Stage 5 — Cohort-aware groupby em 5 tabelas que misturam coortes (B+C)

**Objetivo:** adicionar `parent_sweep_id` ao **groupby** das 5 tabelas que
omitem `config_signature` e hoje misturam coortes silenciosamente.

**Cross-link:** A_code_audit.md §M5-Q2 subgrupo "5 sem config_signature".

### Tasks

- [ ] **5.1** `gold_feature_set_impact`: adicionar `parent_sweep_id` ao
      groupby em [refresh_analytics_store_use_case.py:259](../../src/use_cases/refresh_analytics_store_use_case.py#L259).
      **Aceite:** output tem coluna `parent_sweep_id`; agregados nao
      somam runs de sweeps diferentes.

- [ ] **5.2** `gold_prediction_metrics_by_horizon`: idem em
      [:556](../../src/use_cases/refresh_analytics_store_use_case.py#L556).
      **Aceite:** mesmo criterio.

- [ ] **5.3** `gold_feature_impact_by_horizon`: idem em
      [:760](../../src/use_cases/refresh_analytics_store_use_case.py#L760).
      **Aceite:** mesmo criterio.

- [ ] **5.4** `gold_feature_contrib_local_summary`: idem em
      [:1820](../../src/use_cases/refresh_analytics_store_use_case.py#L1820).
      **Aceite:** mesmo criterio.

- [ ] **5.5** `gold_consistency_topk`: adicionar `parent_sweep_id` ao
      `keys` em [:175](../../src/use_cases/refresh_analytics_store_use_case.py#L175).
      **Aceite:** rankings de top-k sao por sweep.

- [ ] **5.6** Atualizar testes existentes em
      `test_refresh_analytics_store_use_case.py` que assumem o shape
      antigo dessas 5 tabelas.
      **Aceite:** suite passa com novos shapes.

---

## Stage 6 — Adicionar parent_sweep_id ao output das 5 tabelas protegidas (B+C)

**Objetivo:** as 5 tabelas com `config_signature` no groupby ja segregam
coortes via invariante 9f0ccec; falta apenas explicitar `parent_sweep_id`
no output para rastreabilidade direta (Lei 3).

**Cross-link:** A_code_audit.md §M5-Q2 subgrupo "5 com config_signature".

### Tasks

- [ ] **6.1** `gold_ic95_by_config_metric`
      ([refresh_analytics_store_use_case.py:227](../../src/use_cases/refresh_analytics_store_use_case.py#L227)):
      adicionar `parent_sweep_id` derivado via merge com `dim_run`.
      **Aceite:** output tem coluna nao-nula para runs pos-9f0ccec.

- [ ] **6.2** `gold_prediction_metrics_by_config`
      ([:519](../../src/use_cases/refresh_analytics_store_use_case.py#L519)):
      idem.

- [ ] **6.3** `gold_prediction_calibration`
      ([:597-604](../../src/use_cases/refresh_analytics_store_use_case.py#L597-L604)):
      idem.

- [ ] **6.4** `gold_prediction_generalization_gap`
      ([:627-630](../../src/use_cases/refresh_analytics_store_use_case.py#L627-L630)):
      idem.

- [ ] **6.5** `gold_prediction_robustness_by_horizon`
      ([:673](../../src/use_cases/refresh_analytics_store_use_case.py#L673)):
      idem.

- [ ] **6.6** Atualizar `gold_metrics_by_config_n_oos_contract` em
      `validate_analytics_quality_use_case.py` para filtrar pelos dois
      lados via `parent_sweep_id` direto (resolve mismatch_with_run_level=13800
      historico documentado em M5-Q4).
      **Aceite:** check passa em `cohort_decision` para sweep limpo.

---

## Stage 7 — Refresh scoped via ScopeSpec (B+C)

**Objetivo:** permitir que `refresh_analytics_store` opere sobre uma
coorte declarada em vez de ler o silver inteiro.

**Cross-link:** A_code_audit.md §M5-Q3.

### Tasks

- [ ] **7.1** Adicionar `scope_spec: ScopeSpec | None = None` em
      `RefreshAnalyticsStoreUseCase.__init__` e em `execute`.
      **Aceite:** assinatura atualizada; default mantem comportamento global.

- [ ] **7.2** Modificar `_load_partitioned_table`
      ([refresh_analytics_store_use_case.py:28-38](../../src/use_cases/refresh_analytics_store_use_case.py#L28-L38))
      para aplicar `filter_dataframe_by_scope` quando `scope_spec` nao for None.
      **Aceite:** com scope, leitura de silver retorna apenas linhas da coorte.

- [ ] **7.3** Adicionar flags de scope em
      [main_refresh_analytics_store.py](../../src/main_refresh_analytics_store.py):
      `--scope-mode`, `--scope-sweep-prefixes`, `--scope-splits`,
      `--scope-horizons`. Reusar parsing existente do block-A.
      **Aceite:** `python -m src.main_refresh_analytics_store --scope-mode cohort_decision --scope-sweep-prefixes round_1_` produz gold scoped.

- [ ] **7.4** Testes unitarios: refresh com scope produz gold reduzido
      consistente; refresh sem scope produz comportamento atual.
      **Aceite:** suite passa.

---

## Stage 8 — Raw + post-guardrail em paralelo (B+C)

**Objetivo:** persistir metricas probabilisticas em duas variantes
paralelas; pre-registro fixa qual eh primaria para o claim.

**Cross-link:** A_code_audit.md §M5-Q1.

### Tasks

- [ ] **8.1** Refatorar
      `_build_gold_prediction_metrics_by_run_split_horizon`
      ([:309-409](../../src/use_cases/refresh_analytics_store_use_case.py#L309-L409))
      para emitir colunas duplas: `picp_raw`/`picp_post_guardrail`,
      `mpiw_raw`/`mpiw_post_guardrail`,
      `pinball_q*_raw`/`pinball_q*_post_guardrail`,
      `mean_pinball_raw`/`mean_pinball_post_guardrail`,
      `coverage_error_raw`/`coverage_error_post_guardrail`.
      **Aceite:** output contem ambos os conjuntos; `gold_quantile_guardrail_audit`
      vira redundante (manter por compatibilidade ate Phase B fechar).

- [ ] **8.2** Propagar dualidade para tabelas derivadas:
      `gold_prediction_metrics_by_config`, `gold_prediction_calibration`,
      `gold_prediction_robustness_by_horizon`, `gold_prediction_generalization_gap`,
      `gold_prediction_metrics_by_horizon`.
      **Aceite:** todas as agregacoes preservam ambas as variantes.

- [ ] **8.3** Atualizar `gold_model_decision_final` para selecionar a
      variante primaria via flag de configuracao (default: `post_guardrail`,
      sobrepuijavel via `--primary-quantile-contract`).
      **Aceite:** decisao explicita; flag persistida em log do refresh.

- [ ] **8.4** Atualizar plot generators
      ([generate_prediction_analysis_plots_use_case.py](../../src/use_cases/generate_prediction_analysis_plots_use_case.py))
      para ler a variante primaria selecionada.
      **Aceite:** plots refletem o contrato escolhido.

---

## Stage 9 — Filtro de metricas probabilisticas por modo + degeneracao (B+C)

**Objetivo:** PICP/MPIW/pinball/calibration so calculados em runs
genuinamente quantilicos e nao-degenerados.

**Cross-link:** A_code_audit.md §M5-Q5.

### Tasks

- [ ] **9.1** Em `_build_gold_prediction_metrics_by_run_split_horizon`,
      filtrar `fact_oos_predictions` por `prediction_mode='quantile'`
      (via merge com `fact_config`) AND `quantile_p10 != quantile_p90`
      antes do calculo das metricas probabilisticas. Manter calculo
      pontual (RMSE/MAE/DA) para todos os runs.
      **Aceite:** runs `point` ou degenerados nao contribuem para
      PICP/MPIW/pinball; aparecem em RMSE/MAE/DA normalmente.

- [ ] **9.2** Adicionar coluna `is_quantile_genuine` no output de
      `gold_prediction_metrics_by_run_split_horizon` (calculada
      dinamicamente, nao persistida em silver).
      **Aceite:** consumidores podem filtrar diretamente.

- [ ] **9.3** Testes unitarios cobrindo: (a) run quantile genuino entra
      em PICP; (b) run point nao entra; (c) run quantile degenerado
      (`p10==p90`) nao entra.
      **Aceite:** suite passa.

---

## Stage 10 — Rastreabilidade da inferencia (Lei 3) (B+C)

**Objetivo:** substituir `run_id=None` literal nas 3 tabelas de silver
de inferencia por FK explicita para o run de treino.

**Cross-link:** A_code_audit.md §M4 achado adicional (Lei 3).

### Tasks

- [ ] **10.1** Adicionar parametro `training_run_id` ao schema de
      `fact_inference_runs`, `fact_inference_predictions` e
      `fact_feature_contrib_local`. Atualizar
      `src/infrastructure/schemas/analytics_store_schema.py`.
      **Aceite:** schema versao bumped; documentado em
      `ANALYTICS_STORE_ARCHITECTURE.md`.

- [ ] **10.2** Em
      [run_tft_inference_use_case.py:76](../../src/use_cases/run_tft_inference_use_case.py#L76),
      [:120](../../src/use_cases/run_tft_inference_use_case.py#L120) e
      [:206](../../src/use_cases/run_tft_inference_use_case.py#L206):
      derivar `training_run_id` de `dim_run` ou `fact_model_artifacts`
      via `model_version` antes da escrita.
      **Aceite:** `run_id` deixa de ser `None`; aponta para o run que
      treinou o modelo.

- [ ] **10.3** Migrar dados existentes (se nao foi feito reset no Stage 3,
      ou se inferencia rodou em sweep limpo): script de backfill que
      preenche `training_run_id` retroativamente via lookup por
      `model_version`.
      **Aceite:** zero linhas com `training_run_id` nulo em
      `fact_inference_*`.

- [ ] **10.4** Testes unitarios e de integracao do fluxo de inferencia.
      **Aceite:** `pytest tests/unit/use_cases/test_run_tft_inference_use_case.py` passa.

---

## Stage 11 — Gate de degeneracao quantilica (M2) (B+C)

**Objetivo:** bloquear runs com quantis colapsados conforme contrato
declarado em `prediction_mode`.

**Cross-link:** A_code_audit.md §M2-Q4.

### Tasks

- [ ] **11.1** Adicionar `block_quantile_degeneracy_gate` em
      `validate_analytics_quality_use_case.py` ou
      `quantile_contract_analyzer.py`. Regra:
      - `prediction_mode='quantile'` + `% p10==p90 >= 5%` (com
        `n_rows >= 1000`) = falha.
      - `prediction_mode='point'` + degeneracao = OK (esperado).
      **Aceite:** gate detecta runs do historico com 91,25% de degeneracao.

- [ ] **11.2** Reportar `n_rows`, `% p10==p90`, `% p10==p50==p90`
      por `(parent_sweep_id, split, horizon)` no output do gate.
      **Aceite:** quality report eh suficiente para debug pos-treino.

- [ ] **11.3** Adicionar teste unitario do fallback
      `_manual_forward_quantiles_and_actuals` com `H>1`, validando
      valores especificos `(n, h, q)` em layouts
      `[batch, horizon, quantile]` e `[batch, quantile, horizon]`.
      **Aceite:** teste em `tests/unit/adapters/test_pytorch_forecasting_tft_trainer.py`
      passa.

---

## Stage 12 — Bloquear test set como objective_metric do HPO (M1) (B+C)

**Objetivo:** remover leakage metodologico em Optuna/HPO.

**Cross-link:** A_code_audit.md §M1-Q4.

### Tasks

- [ ] **12.1** Em
      [run_tft_optuna_search_use_case.py:49,176-184](../../src/use_cases/run_tft_optuna_search_use_case.py#L49):
      remover `mean_test_rmse` e `joint_val_test_rmse` da lista de
      `objective_metric` aceitos.
      **Aceite:** instanciar com essas opcoes levanta `ValueError`.

- [ ] **12.2** Atualizar CLI [main_tft_optuna_sweep.py](../../src/main_tft_optuna_sweep.py)
      para nao expor essas opcoes em `--objective-metric`.
      **Aceite:** help da CLI nao lista as opcoes invalidas.

- [ ] **12.3** Atualizar testes que cobrem essas opcoes.
      **Aceite:** suite passa.

---

## Stage 13 — Preservar effective_date no dataset (M3) (B+C)

**Objetivo:** rastreabilidade direta de fundamentals no dataset final
sem cruzar com fontes processadas.

**Cross-link:** A_code_audit.md §M3-Q4.

### Tasks

- [ ] **13.1** Renomear `effective_date` para
      `fundamentals_effective_date` durante o merge em
      `build_tft_dataset_use_case.py` e remover o `df.drop(...)` em
      [linha 519](../../src/use_cases/build_tft_dataset_use_case.py#L519).
      **Aceite:** dataset_tft_AAPL.parquet contem coluna nao-nula
      apos primeiro report disponivel.

- [ ] **13.2** Documentar a justificativa do fallback "+45 dias" para
      `reported_date` ausente em `02_data/DATA_CONTRACTS.md` ou
      `02_data/DATA_SOURCES.md` (motivacao SEC 10-Q/10-K, cobertura,
      sensibilidade).
      **Aceite:** doc canonico atualizado.

- [ ] **13.3** Rebuild do dataset AAPL bundled com o PR (~segundos).
      **Aceite:** `data/processed/dataset_tft_AAPL.parquet` regenerado;
      smoke teste basico passa.

---

## Stages opcionais (Caminho C — write-time + refresh deprecado)

Os Stages abaixo so se aplicam se a decisao em "Decisao Caminho B vs C" for C.
Em B, refresh continua sendo o caminho padrao para gerar gold; estes Stages
sao future work.

---

## Stage 14 — Extrair builders per-run para write-time (opt)

**Objetivo:** mover calculo de gold per-run para o fim de cada
`TrainTFTModelUseCase.execute(...)`, eliminando dependencia de refresh
para essas tabelas.

**Cross-link:** A_code_audit.md §"Leis do Analytics Store" — Lei 1.

### Tasks

- [ ] **14.1 (opt)** Criar `RunLevelGoldBuilder` em
      `src/use_cases/run_level_gold_builder.py` encapsulando os 8 builders
      per-run: `_build_gold_runs_long`, `_build_gold_oos_consolidated`,
      `_build_gold_prediction_metrics_by_run_split_horizon`,
      `_build_gold_quantile_guardrail_audit`, `_build_gold_prediction_calibration`,
      `_build_gold_prediction_risk`, `_build_gold_oos_quality_report` (escopo run),
      e gold de inferencia per-run aplicaveis.
      **Aceite:** builder eh testavel isoladamente (recebe DataFrames,
      retorna outputs).

- [ ] **14.2 (opt)** Invocar `RunLevelGoldBuilder` em
      [train_tft_model_use_case.py:876-1029](../../src/use_cases/train_tft_model_use_case.py#L876)
      apos as escritas de silver, com flag `--skip-write-time-gold` para
      desligar (debug).
      **Aceite:** smoke test gera gold per-run sem chamar refresh.

- [ ] **14.3 (opt)** Manter as funcoes em
      `refresh_analytics_store_use_case.py` por enquanto (rebuild scoped
      ainda usa). So remove no Stage 18.
      **Aceite:** sem regressao em refresh existente.

---

## Stage 15 — Hook de "sweep fechado" (opt)

**Objetivo:** ponto de invocacao para gold pareado/agregado (DM, MCS,
win-rate) que precisa de multiplos runs do mesmo sweep.

### Tasks

- [ ] **15.1 (opt)** Definir contrato de "sweep fechado": barrier-counter
      persistido em silver (ex.: nova tabela `dim_sweep_lifecycle` com
      `parent_sweep_id`, `expected_runs`, `completed_runs`,
      `closed_at_utc`).
      **Aceite:** schema documentado em `ANALYTICS_STORE_ARCHITECTURE.md`.

- [ ] **15.2 (opt)** Implementar `SweepLifecycleService` que: (a) registra
      sweep ao iniciar; (b) incrementa `completed_runs` apos cada run;
      (c) dispara `SweepLevelGoldBuilder` quando `completed_runs == expected_runs`.
      **Aceite:** testes unitarios cobrem race conditions basicas.

- [ ] **15.3 (opt)** Criar `SweepLevelGoldBuilder` encapsulando os ~12
      builders pareados/agregados: `_build_gold_dm_pairwise_results`,
      `_build_gold_mcs_results`, `_build_gold_win_rate_pairwise_results`,
      `_build_gold_paired_oos_intersection_by_horizon`,
      `_build_gold_model_decision_final`, `_build_gold_quality_statistics_report`,
      `_build_gold_quality_run_sweep_summary`,
      `_build_gold_ranking_by_config`, `_build_gold_consistency_topk`,
      `_build_gold_ic95`, `_build_gold_feature_set_impact`,
      `_build_gold_prediction_metrics_by_config` e variantes by_horizon.
      **Aceite:** builder testavel isoladamente.

- [ ] **15.4 (opt)** Wire-up em `main_tft_param_sweep.py` e
      `RunTFTOptunaSearchUseCase` para registrar o lifecycle.
      **Aceite:** sweep completo dispara `SweepLevelGoldBuilder`
      automaticamente.

---

## Stage 16 — Atomicidade multi-tabela (opt)

**Objetivo:** garantir que falha no meio da escrita de gold per-run
ou per-sweep nao deixe estado parcial.

### Tasks

- [ ] **16.1 (opt)** Implementar padrao "write to temp + atomic rename":
      cada builder escreve em diretorio temporario; ao concluir todas as
      tabelas, faz rename atomico de todos os paths.
      **Aceite:** falha simulada no meio da escrita deixa estado
      consistente (ou tudo antigo, ou tudo novo).

- [ ] **16.2 (opt)** Adicionar tabela `dim_gold_manifest` que registra
      `(parent_sweep_id, gold_set_version, written_at, status)` para
      rastrear escrita parcial vs completa.
      **Aceite:** consumidor pode filtrar por `status='complete'`.

- [ ] **16.3 (opt)** Testes de integracao com falha injetada.
      **Aceite:** zero estado corrompido apos falha.

---

## Stage 17 — Migrar consumidores (opt)

**Objetivo:** atualizar plot generators, validate quality e qualquer
downstream para esperar gold cohort-aware nativo (sem refresh global).

### Tasks

- [ ] **17.1 (opt)** Atualizar
      [generate_prediction_analysis_plots_use_case.py](../../src/use_cases/generate_prediction_analysis_plots_use_case.py)
      para ler gold filtrado por `parent_sweep_id` direto, sem assumir
      conjunto global.
      **Aceite:** plots geram para uma coorte sem necessidade de refresh.

- [ ] **17.2 (opt)** Atualizar
      [validate_analytics_quality_use_case.py](../../src/use_cases/validate_analytics_quality_use_case.py)
      para considerar a presenca de `dim_gold_manifest` ao avaliar
      completude.
      **Aceite:** quality gate distingue "gold em construcao" de
      "gold completo".

- [ ] **17.3 (opt)** Atualizar `main_refresh_analytics_store.py` para
      modo "rebuild scoped" apenas (ja nao eh fluxo padrao).
      **Aceite:** CLI exige `--scope-sweep-prefixes` obrigatorio.

---

## Stage 18 — Deprecar refresh global (opt)

**Objetivo:** marcar `RefreshAnalyticsStoreUseCase` como deprecated e
documentar caminho oficial.

### Tasks

- [ ] **18.1 (opt)** Adicionar `DeprecationWarning` em
      `RefreshAnalyticsStoreUseCase.__init__` quando `scope_spec is None`.
      **Aceite:** chamadas globais geram warning claro com link para
      `PHASE_B_IMPLEMENTATION_CHECKLIST.md`.

- [ ] **18.2 (opt)** Atualizar `ANALYTICS_STORE_ARCHITECTURE.md` declarando
      write-time como caminho oficial e refresh global como deprecated.
      **Aceite:** doc canonico reflete novo paradigma.

- [ ] **18.3 (opt)** Eventualmente (post-Phase B): remover
      `RefreshAnalyticsStoreUseCase` se rebuild scoped via
      `SweepLevelGoldBuilder.rebuild(parent_sweep_id)` cobrir o use case
      de emergencia.
      **Aceite:** sem callsite de refresh global no main; testes
      atualizados.

---

## Stage final — Smoke confirmatorio + pre-registro (B+C)

### Tasks

- [ ] **F.1** Smoke confirmatorio com `max_epochs >= 5`, `n_rows >= 1000`.
      **Aceite:** % p10==p90 < 5% (gate do Stage 11), zero violacoes
      probabilisticas, gold cohort-aware gerado corretamente.

- [ ] **F.2** Pre-registro com contrato quantilico primario fundamentado
      (raw vs post-guardrail) e politica de baselines (mesmo
      `parent_sweep_id` dos candidatos ou `cohort_id` logico).
      **Aceite:** documento de pre-registro mergeado em
      `docs/06_pre_registration/` (ou local equivalente).

- [ ] **F.3** Marcar `A_code_audit.md` "Gate de saida da Fase A" como
      satisfeito.
      **Aceite:** todos os checkboxes do gate marcados; data de abertura
      da Fase B preenchida.

---

## Cross-link com A_code_audit.md

| Stage | Acao no A_code_audit.md | Veredicto original |
|---|---|---|
| 1 | M5-Q8 (overwrite explicito) | RED |
| 2 | M5-Q6 (regressao 9f0ccec) | YELLOW promovido a gate |
| 3 | M6 (estado do store) + Implementacao passo 10 | YELLOW |
| 4 | M5-Q7 (cross-cohort ranking) | RED |
| 5 | M5-Q2 subgrupo "5 sem config_signature" | RED |
| 6 | M5-Q2 subgrupo "5 com config_signature" + M5-Q4 | RED |
| 7 | M5-Q3 (refresh scoped) | YELLOW |
| 8 | M5-Q1 (raw vs post-guardrail) | RED |
| 9 | M5-Q5 (filtro probabilistico) | RED |
| 10 | M4 achado adicional (Lei 3 inferencia) | YELLOW |
| 11 | M2-Q4 (gate degeneracao) + M2-Q3 (teste H>1) | RED + YELLOW |
| 12 | M1-Q4 (HPO objective) | RED |
| 13 | M3-Q4 (effective_date) | YELLOW |
| 14-18 | Lei 1 (write-time, refresh deprecado) | future direction |
| Final | Gate de saida da Fase A | — |
