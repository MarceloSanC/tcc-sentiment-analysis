---
title: Phase B Implementation Checklist (Pre-Experiment Engineering)
scope: Passo a passo tecnico de implementacao das alteracoes de codigo exigidas pelo gate de Phase A (docs/07_reports/phase-gates/A_code_audit.md). Cada Stage = 1 PR; cada task dentro do Stage = 1 commit. Caminho C eh estado-alvo; Caminho B eh intermediario aceitavel. Para definicoes canonicas, ver 01_architecture/ANALYTICS_STORE_ARCHITECTURE.md e o proprio A_code_audit.md. Fixes descobertos APOS o closure (via smoke F.1) sao rastreados em POST_CLOSURE_FIXES_CHECKLIST.md.
update_when:
  - status (checkbox) de uma task for atualizado
  - novo achado da auditoria for promovido para implementacao
  - decisao Caminho B vs C mudar
  - Stage for fechado (PR mergeado) ou reaberto
  - convencao de branch/commit/PR do governance doc mudar
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

**Escopo:** pre-experiment engineering planejado upfront a partir de
`A_code_audit.md`. Stages 1-19 + Stage final F.1/F.2/F.3. Fixes
descobertos **apos** o closure parcial da Fase A (via smoke F.1 ou
suas re-rodadas) sao rastreados em
[`POST_CLOSURE_FIXES_CHECKLIST.md`](POST_CLOSURE_FIXES_CHECKLIST.md)
para preservar este arquivo como snapshot do escopo original.

## Convencao

- `[ ]` = pendente
- `[x]` = mergeado em main
- `[~]` = em revisao (PR aberto)
- `(opt)` = opcional, so se Caminho C foi escolhido
- `(B+C)` = obrigatorio em ambos os caminhos
- Cada task termina com **Aceite:** criterio que comprova que o commit esta pronto.

## Decisao Caminho B vs C

Caminho C eh o **estado-alvo arquitetural** (alinhamento total com as tres
Leis do Analytics Store: write-time, overwrite explicito, rastreabilidade
PK/FK). Caminho B eh um **passo intermediario aceitavel** quando o custo
de C nao se justifica no curto prazo. A decisao operacional abaixo registra
qual caminho sera entregue ate a abertura da Phase B.

- [x] Decisao operacional Caminho B vs Caminho C registrada antes de
      iniciar Stage 15. Stages 1-14 sao `(B+C)` (obrigatorios em ambos os
      caminhos), entao podem comecar sem a decisao fechada.
  - Caminho B (refresh scoped + cohort-aware gold): ~600-1.000 LOC, 4-6 d.p.
    Stages 1-14 obrigatorios; Stages 15-19 nao se aplicam.
  - Caminho C (write-time + refresh deprecado): ~2.000-3.000 LOC, 8-15 d.p.
    Stages 1-14 obrigatorios + Stages 15-19.
  - **Decisao registrada em 2026-05-17: Caminho B.**
    Justificativa:
    1. Stages 1-14 (todos mergeados em main, PRs #14-34) cobrem todos os
       itens P0 transversais do `A_code_audit.md` §"Gate de saida da Fase A"
       (Q1 Stage 8, Q6 Stage 2, Q7 Stage 4, Q8 Stage 1, Lei 3 Stage 10) +
       cohort-aware gold (Stages 4-6) + refresh scoped (Stage 7) +
       baselines persistidos (Stage 12) + gate de degeneracao (Stage 11) +
       anti-leakage HPO (Stage 13) + rastreabilidade fundamentals
       (Stage 14). Suficiente para Phase B confirmatoria.
    2. Caminho C entregaria +2000-3000 LOC adicionais (Stages 15-19:
       write-time per-run gold, hook sweep fechado, atomicidade
       multi-tabela, scoped-only CLI, deprecation refresh global) sem
       ganho cientifico para escopo TCC — sao otimizacoes operacionais
       para volumes (milhares de sweeps simultaneos) que o escopo do TCC
       (AAPL piloto, 5 seeds x 3 folds em Phase B per
       STRATEGIC_DIRECTION §B.1) nao atinge.
    3. Refresh global continua sendo caminho default em Caminho B e e
       operacionalmente aceitavel no volume previsto (refresh leva
       minutos, nao horas).
    4. Stages 15-19 ficam como future work registrado; se durante Phase B
       o refresh global virar gargalo empirico medido (nao teorizado),
       reavaliar como otimizacao targetada.
    5. Custo de oportunidade decisivo: 8-15 d.p. de engenharia adicional
       em Caminho C = 8-15 d.p. nao gastos em escrita de capitulos do TCC
       (POST_AUDIT_EXECUTION_PLAN §Etapa 8 explicita que texto LaTeX e
       etapa de consolidacao, nao fonte primaria de decisoes — escrever
       requer Phase B + Fase C executadas primeiro).

## Convencao de branch, commit e PR

Esta checklist segue o fluxo canonico definido em
[`docs/08_governance/GOVERNANCE_AND_VERSIONING.md`](../08_governance/GOVERNANCE_AND_VERSIONING.md):
- branch naming `<tipo>/<area>-<objetivo>`
- commit format `<tipo>(<escopo>): <resumo no imperativo>`
- PR template com Summary/Changes/Validation/Notes
- merge readiness checklist (CI verde, testes locais, sem conflitos)

Cada Stage abaixo eh um PR no padrao do governance doc; cada task numerada
eh um commit no padrao do governance doc.

---

## Stage 1 — Idempotencia das escritas em silver (Lei 2) (B+C)

**Objetivo:** substituir `_append_to_parquet` (concat cego) por mecanismo
overwrite-on-flag em todos os `append_*`. Pre-condicao para reset confiavel
de Phase B; sem isso, re-rodar o mesmo `run_id` duplica linhas em silver.

**Cross-link:** A_code_audit.md §M5-Q8.

### Tasks


- [x] **1.1** Definir interface `AnalyticsRunRepository` com novo contrato
      `overwrite: bool = False` em todos os `append_*`/`upsert_*`.
      Atualizar `src/interfaces/analytics_run_repository.py`.
      **Aceite:** type checker passa; nenhuma chamada existente quebrada
      (default `overwrite=False` preserva semantica atual exceto pelo bug).

- [x] **1.2** Implementar `_write_with_overwrite_policy` em
      [parquet_analytics_run_repository.py:94-100](../../src/adapters/parquet_analytics_run_repository.py#L94-L100).
      Comportamento: se path nao existe, escreve. Se existe e nao ha
      colisao de chave logica, append. Se existe e ha colisao e
      `overwrite=False`, levanta `DuplicateKeyError`. Se `overwrite=True`,
      remove linhas colidentes e regrava.
      **Aceite:** novo metodo cobre os contratos de PK logica definidos em
      `src/infrastructure/schemas/analytics_store_schema.py` e validados em
      `tests/unit/infrastructure/schemas/test_analytics_store_schema.py`.

- [x] **1.3** Migrar os 12 `append_*` para usar
      `_write_with_overwrite_policy`. Manter `upsert_dim_run` como esta
      (ja honra a lei).
      **Aceite:** todos os metodos em
      [parquet_analytics_run_repository.py:144-267](../../src/adapters/parquet_analytics_run_repository.py#L144-L267)
      delegam ao novo metodo.

- [x] **1.4** Testes unitarios cobrindo: (a) primeira escrita; (b) append
      sem colisao; (c) colisao sem flag = erro; (d) colisao com flag =
      sobrescrita correta; (e) particoes diferentes nao se afetam.
      **Aceite:** `pytest tests/unit/adapters/test_parquet_analytics_run_repository.py` passa.

- [x] **1.5** Atualizar call sites em
      `src/use_cases/train_tft_model_use_case.py` e
      `src/use_cases/run_tft_inference_use_case.py` para passar
      `overwrite=True` apenas quando o usuario explicitamente solicitar
      (nova flag `--overwrite-on-collision` em `main_train_tft.py`).
      **Aceite:** smoke test re-executavel produz erro claro quando
      `run_id` colide; com flag, sobrescreve corretamente.

### Notas de revisao:

-  `--overwrite` e `--overwrite-on-collision` devem
permanecer contratos separados. `--overwrite` controla apenas sobrescrita
operacional da inferencia; overwrite no silver analytics exige
`--overwrite-on-collision` explicito.


---

## Stage 2 — Teste de regressao da invariante 9f0ccec (B+C)

**Objetivo:** congelar a invariante que faz `config_signature` segregar
coortes automaticamente. Sem esse teste, qualquer refactor futuro em
`_build_trainer_config` ou `compute_config_signature` quebra a hipotese
de coexistencia por coluna sem aviso.

**Cross-link:** A_code_audit.md §M5-Q6.

### Notas de revisao:

- 2026-05-14: adicionados testes de regressao para garantir que
  `compute_config_signature` muda com `parent_sweep_id`, `fold`,
  `trial_number` e `seed`, e permanece estavel quando variam apenas chaves
  temporais volateis (`created_at`, `started_at`, `ended_at`, `timestamp`).

### Tasks

- [x] **2.1** Adicionar teste
      `test_config_signature_changes_when_parent_sweep_id_changes` em
      [tests/unit/infrastructure/schemas/test_analytics_store_schema.py](../../tests/unit/infrastructure/schemas/test_analytics_store_schema.py).
      Cobrir: dois `training_config` iguais com `parent_sweep_id` diferente
      = hashes diferentes. Idem para `fold`, `trial_number`, `seed`.
      **Aceite:** 4 asserts independentes, todos passando.

- [x] **2.2** Adicionar teste
      `test_config_signature_stable_across_volatile_keys` que garante que
      `created_at`/`started_at`/`ended_at`/`timestamp` continuam sendo
      ignorados (regressao do comportamento atual).
      **Aceite:** dois configs iguais exceto pelas chaves volateis = mesmo hash.

---

## Stage 3 — Reset arquivamento do silver/gold pre-Phase B (B+C)

**Objetivo:** arquivar dados historicos como evidencia exploratoria
(nao confirmatoria). So executar apos Stages 1 e 2 mergeados.

### Notas de revisao:

- 2026-05-14: Stage 3 pronto para revisao. Comandos executados:
  `mkdir -p data/analytics_archive_pre_phase_b`;
  `mv data/analytics/silver data/analytics_archive_pre_phase_b/silver`;
  `mv data/analytics/gold data/analytics_archive_pre_phase_b/gold`;
  `mkdir -p data/analytics/silver data/analytics/gold`;
  `chmod -R a-w data/analytics_archive_pre_phase_b`.
- 2026-05-14: archive `data/analytics_archive_pre_phase_b/` protegido como
  read-only por permissao de filesystem. Os dados arquivados sao diagnostico
  historico/exploratorio pre-`2026-05-10`, nao evidencia confirmatoria da
  Phase B.

### Tasks

- [x] **3.1** Mover `data/analytics/silver/` para
      `data/analytics_archive_pre_phase_b/silver/` (script ou comando manual,
      registrar comando exato no PR).
      **Aceite:** `data/analytics/silver/` vazio; archive populado;
      `git status` confirma tracking files inalterados (apenas dados).

- [x] **3.2** Mover `data/analytics/gold/` para
      `data/analytics_archive_pre_phase_b/gold/`.
      **Aceite:** mesmo criterio.

- [x] **3.3** Atualizar `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`
      com nota sobre o archive: dados pre-`2026-05-10` sao diagnostico
      historico, nao evidencia confirmatoria.
      **Aceite:** doc canonico atualizado e linkado em
      `A_code_audit.md` §M6.

- [x] **3.4** Proteger archive como rollback: aplicar permissoes
      read-only em `data/analytics_archive_pre_phase_b/` (`chmod -R a-w`)
      e documentar em `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`
      que o archive nao deve ser modificado ate o fechamento do
      pre-registro da Phase B. Em caso de necessidade de rollback de
      Stages 4+, restaurar do archive eh o caminho oficial.
      **Aceite:** tentativa de escrita em `data/analytics_archive_pre_phase_b/`
      falha com permission denied.

---

## Stage 4 — Cohort-aware ranking em gold de decisao (B+C)

**Objetivo:** corrigir ranking cross-cohort em
`gold_model_decision_final` (Q7) e `gold_ranking_by_config`. Severidade RED
porque `gold_model_decision_final` eh a tabela final de decisao do paper.

**Cross-link:** A_code_audit.md §M5-Q7.

### Notas de revisao:

- 2026-05-15: Review Stage 4 aprovado com ressalvas YELLOW, sem RED.
  Validacao local: `.venv/bin/pytest -q tests/unit/use_cases/test_refresh_analytics_store_use_case.py -k "cohort_aware"`
  (`2 passed, 2 deselected`) e `.venv/bin/pytest -q tests/unit/use_cases/test_refresh_analytics_store_use_case.py`
  (`4 passed, 3 warnings`).
- Follow-up incorporado no PR: normalizacao explicita de `parent_sweep_id`
  antes dos rankings em `gold_ranking_by_config` e
  `gold_model_decision_final`, inclusive quando `paired_intersection` esta
  vazio.
- Desvio aceito do enunciado de 4.3: o teste de
  `gold_model_decision_final` usa `config_signature` distintos por sweep porque
  o `sweep_map` atual deduplica por `(asset, split, horizon, config_label)`;
  os valores de metrica ainda provam que o rank nao compete cross-coorte.
- Gap registrado para etapa posterior: adicionar cobertura explicita para
  `parent_sweep_id=None` legado pre-`9f0ccec`.

### Tasks

- [x] **4.1** Adicionar `parent_sweep_id` ao groupby em
      [refresh_analytics_store_use_case.py:1575-1590](../../src/use_cases/refresh_analytics_store_use_case.py#L1575-L1590)
      (`gold_model_decision_final`: `rank_rmse`, `rank_mae`, `rank_da`).
      **Aceite:** rank eh calculado dentro de
      `(asset, parent_sweep_id, horizon)`, nao mais cross-coorte.

- [x] **4.2** Adicionar `parent_sweep_id` ao groupby em
      [refresh_analytics_store_use_case.py:151-176](../../src/use_cases/refresh_analytics_store_use_case.py#L151-L176)
      (`gold_ranking_by_config`: `rank_test_rmse`, `rank_test_mae`,
      `rank_test_da`).
      **Aceite:** ranks sao por
      `(asset, parent_sweep_id, feature_set_name)`.

- [x] **4.3** Testes unitarios: criar dois sweeps sinteticos com mesmas
      configs, verificar que ranks sao independentes por sweep.
      **Aceite:** `pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py`
      passa com novos casos.

---

## Stage 5 — Cohort-aware groupby em 5 tabelas que misturam coortes (B+C)

**Objetivo:** adicionar `parent_sweep_id` ao **groupby** das 5 tabelas que
omitem `config_signature` e hoje misturam coortes silenciosamente.

**Cross-link:** A_code_audit.md §M5-Q2 subgrupo "5 sem config_signature".

### Notas de revisao:

- 2026-05-15: Stage 5 implementado em branch
  `fix/analytics-store-stage5-cohort-aware-groupby` para corrigir o subgrupo
  RED "5 sem config_signature" de A_code_audit.md §M5-Q2.
  Validacao local: `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
  (`9 passed, 3 warnings`),
  `.venv/bin/pytest tests/unit/use_cases/test_generate_prediction_analysis_plots_use_case.py -v`
  (`7 passed`) e `.venv/bin/pytest tests/unit/use_cases/ -v`
  (`134 passed, 5 warnings`).
- Busca de consumidores executada com
  `grep -rnE --exclude='*.pyc' "gold_feature_set_impact|gold_prediction_metrics_by_horizon|gold_feature_impact_by_horizon|gold_feature_contrib_local_summary|gold_consistency_topk" src/ tests/`
  e `grep -rn --exclude='*.pyc' "load_gold_table" src/ tests/`.
- Ressalva de 5.4: `gold_feature_contrib_local_summary` deriva
  `parent_sweep_id` via `dim_run.run_id`; rows legadas de inferencia sem
  `run_id` permanecem com `parent_sweep_id=None` ate Stage 10.
- Stage 6 permanece fora deste PR; as 5 tabelas com `config_signature` no
  groupby tem tratamento proprio no Stage 6.
- 2026-05-15: Follow-ups YELLOW da revisao Stage 5 aplicados:
  `752f6b7` cobre o branch legado de
  `gold_feature_contrib_local_summary` sem `parent_sweep_id`, e `33fdebe`
  pondera a agregacao global de feature importance por `n_runs`.

### Tasks

- [x] **5.1** `gold_feature_set_impact`: adicionar `parent_sweep_id` ao
      groupby em [refresh_analytics_store_use_case.py:259](../../src/use_cases/refresh_analytics_store_use_case.py#L259).
      **Aceite:** output tem coluna `parent_sweep_id`; agregados nao
      somam runs de sweeps diferentes.

- [x] **5.2** `gold_prediction_metrics_by_horizon`: idem em
      [:556](../../src/use_cases/refresh_analytics_store_use_case.py#L556).
      **Aceite:** mesmo criterio.

- [x] **5.3** `gold_feature_impact_by_horizon`: idem em
      [:760](../../src/use_cases/refresh_analytics_store_use_case.py#L760).
      **Aceite:** mesmo criterio.

- [x] **5.4** `gold_feature_contrib_local_summary`: idem em
      [:1820](../../src/use_cases/refresh_analytics_store_use_case.py#L1820).
      **Aceite:** mesmo criterio.

- [x] **5.5** `gold_consistency_topk`: adicionar `parent_sweep_id` ao
      `keys` em [:175](../../src/use_cases/refresh_analytics_store_use_case.py#L175).
      **Aceite:** rankings de top-k sao por sweep.

- [x] **5.6** Atualizar testes existentes em
      `test_refresh_analytics_store_use_case.py` que assumem o shape
      antigo dessas 5 tabelas.
      **Aceite:** suite passa com novos shapes.

- [x] **5.7** Atualizar consumidores que leem as 5 tabelas alteradas
      para acomodar o novo shape (coluna `parent_sweep_id` adicional e
      grao mais fino):
      - [generate_prediction_analysis_plots_use_case.py](../../src/use_cases/generate_prediction_analysis_plots_use_case.py)
      - [validate_analytics_quality_use_case.py](../../src/use_cases/validate_analytics_quality_use_case.py)
      Buscar tambem callsites em `tests/` e demais use cases.
      **Aceite:** smoke run end-to-end (refresh + quality + plots) nao
      regride por shape mismatch.

---

## Stage 6 — Adicionar parent_sweep_id ao output das 5 tabelas protegidas (B+C)

**Objetivo:** as 5 tabelas com `config_signature` no groupby ja segregam
coortes via invariante 9f0ccec; falta apenas explicitar `parent_sweep_id`
no output para rastreabilidade direta (Lei 3).

**Cross-link:** A_code_audit.md §M5-Q2 subgrupo "5 com config_signature".

### Notas de revisao:

- 2026-05-16: Stage 6 implementado em branch
  `fix/analytics-store-stage6-parent-sweep-id-output` para corrigir o
  subgrupo RED "5 com config_signature" de A_code_audit.md §M5-Q2 e fechar o
  gap M5-Q4 de `gold_metrics_by_config_n_oos_contract`.
  Validacao local: `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
  (`11 passed, 3 warnings`),
  `.venv/bin/pytest tests/unit/use_cases/test_validate_analytics_quality_use_case.py -v`
  (`19 passed, 2 warnings`) e `.venv/bin/pytest tests/unit/use_cases/ -v`
  (`138 passed, 5 warnings`).
- Busca de consumidores executada com
  `rg -n "gold_ic95_by_config_metric|gold_prediction_metrics_by_config|gold_prediction_calibration|gold_prediction_generalization_gap|gold_prediction_robustness_by_horizon" src/ tests/`.
- Ressalva: para runs pos-`9f0ccec`, `config_signature` ja segrega coortes e
  o grao das 5 tabelas nao muda; legado pre-`9f0ccec` com
  `parent_sweep_id=None` permanece visivel via `dropna=False` e pode aparecer
  como bucket separado.
- 2026-05-16 (follow-up): 4 YELLOW da revisao enderecados antes do merge —
  refactor da indentacao do n_oos_contract, teste de cenario M5-Q4 com
  config_signature cross-sweep, teste de bucket legado parent_sweep_id=None,
  e Finding 4 dispensado por falta de evidencia downstream concreta.
  Validacao: `.venv/bin/pytest tests/unit/use_cases/` (`140 passed, 5 warnings`);
  `.venv/bin/ruff check src/use_cases/validate_analytics_quality_use_case.py`
  (`All checks passed!`).

### Tasks

- [x] **6.1** `gold_ic95_by_config_metric`
      ([refresh_analytics_store_use_case.py:227](../../src/use_cases/refresh_analytics_store_use_case.py#L227)):
      adicionar `parent_sweep_id` derivado via merge com `dim_run`.
      **Aceite:** output tem coluna nao-nula para runs pos-9f0ccec.

- [x] **6.2** `gold_prediction_metrics_by_config`
      ([:519](../../src/use_cases/refresh_analytics_store_use_case.py#L519)):
      idem.

- [x] **6.3** `gold_prediction_calibration`
      ([:597-604](../../src/use_cases/refresh_analytics_store_use_case.py#L597-L604)):
      idem.

- [x] **6.4** `gold_prediction_generalization_gap`
      ([:627-630](../../src/use_cases/refresh_analytics_store_use_case.py#L627-L630)):
      idem.

- [x] **6.5** `gold_prediction_robustness_by_horizon`
      ([:673](../../src/use_cases/refresh_analytics_store_use_case.py#L673)):
      idem.

- [x] **6.6** Atualizar `gold_metrics_by_config_n_oos_contract` em
      `validate_analytics_quality_use_case.py` para filtrar pelos dois
      lados via `parent_sweep_id` direto (resolve mismatch_with_run_level=13800
      historico documentado em M5-Q4).
      **Aceite:** check passa em `cohort_decision` para sweep limpo.

- [x] **6.7** Atualizar consumidores que leem as 5 tabelas com nova
      coluna `parent_sweep_id` no output:
      - [generate_prediction_analysis_plots_use_case.py](../../src/use_cases/generate_prediction_analysis_plots_use_case.py)
      - [validate_analytics_quality_use_case.py](../../src/use_cases/validate_analytics_quality_use_case.py)
      Como a coluna eh apenas adicionada (groupby nao mudou), o impacto
      eh menor que no Stage 5, mas testes/fixtures que assumem schema
      exato precisam aceitar a nova coluna.
      **Aceite:** smoke run end-to-end nao regride.

---

## Stage 7 — Refresh scoped via ScopeSpec (B+C)

**Objetivo:** permitir que `refresh_analytics_store` opere sobre uma
coorte declarada em vez de ler o silver inteiro.

**Cross-link:** A_code_audit.md §M5-Q3.

### Notas de revisao:

- 2026-05-16: Stage 7 implementado em branch
  `feat/analytics-store-stage7-refresh-scoped` para enderecar
  A_code_audit.md §M5-Q3 (YELLOW) com refresh scoped via `ScopeSpec`.
  Validacao local: `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
  (`16 passed, 12 warnings`) e `.venv/bin/pytest tests/unit/use_cases/ -v`
  (`144 passed, 14 warnings`).
- Smoke CLI executado com
  `.venv/bin/python -m src.main_refresh_analytics_store --scope-mode cohort_decision --scope-sweep-prefixes round_1_`
  (`exit 0`). O silver local estava vazio para esse prefixo, entao os golds
  scoped foram materializados vazios; a garantia de gold reduzido nao vazio
  fica coberta pelo teste unitario `test_refresh_with_scope_spec_cohort_decision_filters_silver`.
- Decisao de assinatura: `scope_spec` foi aceito no construtor e tambem em
  `execute(...)` como override por chamada, preservando `None` como
  comportamento global default.
- Tratamento semantico por tabela: `dim_run` define `scoped_run_ids`, filtros
  diretos de `parent_sweep_id`/`split`/`horizon` so sao aplicados quando a
  coluna existe, e tabelas sem colunas de escopo nao sao esvaziadas
  arbitrariamente. `fact_feature_contrib_local` nao tem `parent_sweep_id`; sob
  scope cohort, filtra por `run_id` quando disponivel e rows legadas com
  `run_id=None` caem fora.
- Reuso de contrato: a leitura scoped usa `filter_dataframe_by_scope` de
  `src/domain/services/scope_spec.py`; Stage 8, Stage 9, Stage 10 e etapas
  posteriores permanecem fora deste PR.
- 2026-05-16: Follow-up da revisao Stage 7 fechado sem alterar codigo de
  producao. Findings cobertos:
  `test_main_refresh_passes_scope_flags_to_plots_use_case` e
  `test_main_refresh_passes_block_a_flags_to_quality_use_case` atualizam os
  fixtures `Namespace`; `test_refresh_without_scope_spec_is_bitwise_equivalent_to_legacy`
  compara golds principais com `assert_frame_equal(check_exact=True)`;
  `test_refresh_execute_scope_spec_overrides_instance_default` cobre override
  por chamada; `test_refresh_global_health_ignores_cohort_filters` cobre o
  descarte de filtros em `global_health`; e
  `test_refresh_with_scope_spec_filters_by_split_and_horizon` cobre o ramo
  `split`/`horizon` de `_build_scoped_run_ids`.
  Validacao local: `.venv/bin/pytest tests/unit/test_main_refresh_analytics_store.py -v`
  (`2 passed`), `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
  (`20 passed, 30 warnings`), `.venv/bin/pytest tests/unit/use_cases/ -q`
  (`148 passed, 32 warnings`) e `.venv/bin/pytest tests/unit/ -q`
  (`419 passed, 32 warnings`).

### Tasks

- [x] **7.1** Adicionar `scope_spec: ScopeSpec | None = None` em
      `RefreshAnalyticsStoreUseCase.__init__` e em `execute`.
      **Aceite:** assinatura atualizada; default mantem comportamento global.

- [x] **7.2** Modificar `_load_partitioned_table`
      ([refresh_analytics_store_use_case.py:28-38](../../src/use_cases/refresh_analytics_store_use_case.py#L28-L38))
      para suportar scoping semantico por tabela (nao filtro cego global):
      - construir `scoped_run_ids` a partir de `dim_run` filtrado por scope;
      - aplicar filtro por `run_id` nas tabelas run-level;
      - aplicar filtro por `parent_sweep_id`/`split`/`horizon` apenas onde
        essas colunas existem e sao semanticamente parte do grao.
      **Aceite:** com scope, leitura retorna coorte correta sem esvaziar
      tabelas que nao possuem todas as colunas de escopo.

- [x] **7.3** Adicionar flags de scope em
      [main_refresh_analytics_store.py](../../src/main_refresh_analytics_store.py):
      `--scope-mode`, `--scope-sweep-prefixes`, `--scope-splits`,
      `--scope-horizons`. Reusar parsing existente do block-A.
      **Aceite:** `python -m src.main_refresh_analytics_store --scope-mode cohort_decision --scope-sweep-prefixes round_1_` produz gold scoped.

- [x] **7.4** Testes unitarios: refresh com scope produz gold reduzido
      consistente; refresh sem scope produz comportamento atual.
      **Aceite:** suite passa.

---

## Stage 8 — Raw + post-guardrail em paralelo (B+C)

**Objetivo:** persistir metricas probabilisticas em duas variantes paralelas
**conforme categorizacao A/B/C** por tabela gold; pre-registro fixa
variante primaria para H1/H2a/H2b (default `post_guardrail`). A
categorizacao canonica vive em
[`docs/04_evaluation/METRICS_DEFINITIONS.md`](../04_evaluation/METRICS_DEFINITIONS.md)
§"Variante quantilica" e justificativa academica em
[`docs/07_reports/living-paper/20_method.md`](../07_reports/living-paper/20_method.md)
§"Politica de variante quantilica".

**Cross-link:** A_code_audit.md §M5-Q1; external review 2026-04-09
§Entregavel 2 / Gate 3; `METRICS_DEFINITIONS.md` §Variante quantilica
(categorizacao A/B/C); `20_method.md` §Politica de variante quantilica
(justificativa metodologica com Chernozhukov et al. 2010, Gneiting &
Raftery 2007, Jorion 2007, Acerbi & Tasche 2002).

### Categorias canonicas aplicadas a este Stage

Toda tabela gold cai em uma das tres categorias (criterio canonico em
`METRICS_DEFINITIONS.md`):

- **Categoria A — dual obrigatorio** (raw + post-guardrail): metricas
  onde as duas perguntas sao cientificamente legitimas e dao respostas
  diferentes. Tasks 8.1, 8.2, 8.7.
- **Categoria B — post-guardrail apenas**: metricas onde raw e
  matematicamente computavel mas semanticamente quebrado (risk metrics,
  selecao final, artefatos operacionais). Tasks 8.3, 8.4, 8.5.
- **Categoria C — raw apenas**: checks de patologia onde post-guardrail
  seria tautologicamente satisfeito e perderia funcao diagnostica. Task
  8.8 (anotacao protetiva; sem mudanca de comportamento).

### Notas de revisao:

- 2026-05-16: Tasks 8.1-8.4 implementadas em branch
  `feat/analytics-store-stage8-raw-post-guardrail-dual` para enderecar
  A_code_audit.md §M5-Q1 (RED). Validacao local:
  `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
  (`25 passed, 30 warnings`);
  `.venv/bin/pytest tests/unit/use_cases/test_generate_prediction_analysis_plots_use_case.py -v`
  (`9 passed`);
  `.venv/bin/pytest tests/unit/use_cases/ -v`
  (`154 passed, 32 warnings`);
  `.venv/bin/pytest tests/unit/ -v` (`425 passed, 32 warnings`).
- Decisao de implementacao 8.1: estrategia B (single-pass public-contract
  com helper interno single-contract). Builder oficial calcula metricas
  pontuais uma vez e emite colunas pareadas para familias probabilisticas;
  helper single-contract preserva comportamento legado de
  `gold_quantile_guardrail_audit`.
- Decisao sobre `confidence_calibrated`: duplicada como
  `confidence_calibrated_raw`/`confidence_calibrated_post_guardrail`
  (deriva de `coverage_error` e `pred_interval_width`, ambos Cat A);
  coluna `confidence_calibrated` permanece como alias de compatibilidade
  para o contrato primario disponivel.
- Compatibilidade: `gold_quantile_guardrail_audit` permanece materializada
  como diagnostico secundario ate o fechamento da Phase B. Em silver
  legado sem colunas `quantile_p*_post_guardrail`, as colunas
  `*_post_guardrail` do gold ficam `NaN` e o refresh emite warning unico.
- 2026-05-16: Stage 8 **expandido** para 8.5-8.8 apos merge da policy
  canonica em `METRICS_DEFINITIONS.md` §Variante quantilica e
  `20_method.md` §Politica de variante quantilica. O checklist literal
  original (8.1-8.4) cobria 5 familias de metricas mas omitia:
  (a) `gold_prediction_risk` como Categoria B — `var_10`/`es_10_approx`
  continuam silenciosamente raw; (b) coluna `delta_*_post_minus_raw` no
  run-level para substituir funcionalmente `gold_quantile_guardrail_audit`;
  (c) `prob_up`/`prob_down` como Categoria A (dependem de p10/p50/p90 via
  `_prob_up_from_quantiles`); (d) anotacao protetiva para Categoria C
  (`_pred_interval_negative`, gate Stage 11). Tasks 8.5-8.8 fecham esses
  gaps sem invalidar 8.1-8.4.
- 2026-05-16: Tasks 8.5-8.8 implementadas (4 commits) na mesma branch.
  Validacao local:
  `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
  (`32 passed, 30 warnings`);
  `.venv/bin/pytest tests/unit/use_cases/ -v` (`160 passed, 32 warnings`);
  `.venv/bin/pytest tests/unit/ -v` (`432 passed, 32 warnings` — zero regressao).
- Decisao sobre alias `prob_up`/`prob_down` (8.7): **mantido alias=
  post-guardrail** (com fallback para raw em silver legado). Mesmo
  criterio aplicado a `confidence_calibrated` em 8.1. Helpers locais
  `_set_alias_post_primary`/`_set_alias_raw_only` encapsulam a logica
  nos 3 caminhos de retorno do dual builder.
- Compatibilidade: `gold_quantile_guardrail_audit` segue materializada
  como diagnostico secundario ate o fechamento da Phase B (mesmo apos
  8.6 expor as deltas no contrato primario).
- `_pred_interval_negative` (Stage 8.8) e gate de degeneracao quantilica
  do Stage 11 permanecem **raw apenas** por design (Categoria C). Teste
  de regressao protege a primeira invariante; o gate do Stage 11 sera
  anotado quando implementado.
- 2026-05-16: **Follow-ups YELLOW do review do Stage 8 — estado pos-rebase
  com PR #21**:
  - (1) `METRICS_DEFINITIONS.md` §"Variante quantilica" — **fechado via
    PR #21** (`8ab1d39`, ja em main): categorizacao A/B/C, mapeamento por
    tabela gold (Cat A: 6 tabelas; Cat B: gold_prediction_risk,
    gold_model_decision_final, plots; Cat C: _pred_interval_negative +
    gate Stage 11), flag `primary_quantile_contract`, alias unsuffixed
    (confidence_calibrated, prob_up, prob_down) com fallback raw.
  - (2) `20_method.md` §"Politica de variante quantilica" + entrada
    `variante quantilica` em GLOSSARY.md (secao V) + INDEX.md anotado —
    **fechado via PR #21** (`8ab1d39`), com justificativa academica por
    hipotese e referencias canonicas (Chernozhukov 2010, Gneiting &
    Raftery 2007, Jorion 2007, Acerbi & Tasche 2002).
  - (3) `CALIBRATION_AND_RISK.md` §Risk Method e §Calibration Acceptance
    Thresholds sincronizados com quantis post-guardrail — **fechado via
    PR #21** (`8ab1d39`).
  - (4) `_POST_GUARDRAIL_MISSING_WARNING_EMITTED` resetado no inicio de
    cada `execute()` — **fechado nesta branch** (opcao 2 do prompt):
    preserva "warning unico por refresh" e elimina silencio permanente
    por processo apos o primeiro disparo. Teste
    `test_post_guardrail_missing_warning_emits_once_per_refresh_not_per_process`
    cobre o comportamento via caplog. Commit:
    `fix(analytics-store): resetar flag de warning post-guardrail por refresh`.
  - Rebase: branch originalmente continha 3 commits de doc duplicando
    PR #21 (METRICS_DEFINITIONS, 20_method/GLOSSARY/INDEX,
    CALIBRATION_AND_RISK). Foram dropados em rebase nao-interativo apos
    PR #21 ser mergeada; o conteudo permanece em main via PR #21.
  - Validacao apos rebase: `.venv/bin/pytest tests/unit/ -q` ->
    `433 passed` (baseline 432 + 1 teste novo do follow-up 4); zero
    regressao em `tests/unit/use_cases/`.

### Tasks

- [x] **8.1** Refatorar
      `_build_gold_prediction_metrics_by_run_split_horizon`
      ([:309-409](../../src/use_cases/refresh_analytics_store_use_case.py#L309-L409))
      para emitir colunas duplas (Categoria A): `picp_raw`/`picp_post_guardrail`,
      `mpiw_raw`/`mpiw_post_guardrail`,
      `pinball_q*_raw`/`pinball_q*_post_guardrail`,
      `mean_pinball_raw`/`mean_pinball_post_guardrail`,
      `coverage_error_raw`/`coverage_error_post_guardrail`,
      `pred_interval_width_raw`/`pred_interval_width_post_guardrail`,
      `confidence_calibrated_raw`/`confidence_calibrated_post_guardrail`.
      **Aceite:** output contem ambos os conjuntos; `gold_quantile_guardrail_audit`
      vira redundante (manter por compatibilidade ate Phase B fechar).

- [x] **8.2** Propagar dualidade para tabelas derivadas (Categoria A):
      `gold_prediction_metrics_by_config`, `gold_prediction_calibration`,
      `gold_prediction_robustness_by_horizon`, `gold_prediction_generalization_gap`,
      `gold_prediction_metrics_by_horizon`.
      **Aceite:** todas as agregacoes preservam ambas as variantes.

- [x] **8.3** Atualizar `gold_model_decision_final` (Categoria B — single
      por definicao de "selecao") para selecionar a variante primaria via
      flag de configuracao (default: `post_guardrail`, sobrepujavel via
      `--primary-quantile-contract`).
      **Aceite:** decisao explicita; flag persistida em log do refresh
      e em coluna `primary_quantile_contract` no output.

- [x] **8.4** Atualizar plot generators (Categoria B — single por design
      operacional)
      ([generate_prediction_analysis_plots_use_case.py](../../src/use_cases/generate_prediction_analysis_plots_use_case.py))
      para ler a variante primaria selecionada via flag.
      **Aceite:** plots refletem o contrato escolhido; titulos anotam
      qual variante esta sendo plotada.

- [x] **8.5** **Mover `gold_prediction_risk` para post-guardrail apenas
      (Categoria B).** Substituir `quantile_p10`/`quantile_p50` por
      `quantile_p10_post_guardrail`/`quantile_p50_post_guardrail` em
      [`refresh_analytics_store_use_case.py:1079,1082`](../../src/use_cases/refresh_analytics_store_use_case.py#L1079-L1082)
      (`var_10`, `es_10_approx`). Fallback: se colunas post-guardrail nao
      existem no silver legado, retornar `NaN` com warning unico (mesmo
      padrao do 8.1).
      **Justificativa:** VaR exige monotonicidade da funcao quantil por
      definicao (Jorion 2007; Acerbi & Tasche 2002); raw produz numero
      matematico sem interpretacao como risco. `expected_move` e
      `downside_risk` (independentes de quantis) permanecem inalterados.
      **Aceite:** `var_10` e `es_10_approx` calculados sobre post-guardrail
      por construcao; teste unitario com fixture de crossing demonstra que
      `var_10 != mean(quantile_p10)` raw quando ha crossing; documentado
      em `CALIBRATION_AND_RISK.md` (sincronizado no PR doc).

- [x] **8.6** **Adicionar colunas `delta_<metrica>_post_minus_raw`** no
      `_build_gold_prediction_metrics_by_run_split_horizon` para todas as
      familias Categoria A duplicadas em 8.1. Substitui funcionalmente
      `gold_quantile_guardrail_audit` (que continua materializado por
      compatibilidade) tornando o efeito do guardrail diretamente queryavel
      no contrato primario.
      **Aceite:** colunas `delta_picp_post_minus_raw`,
      `delta_mpiw_post_minus_raw`, `delta_mean_pinball_post_minus_raw`,
      `delta_coverage_error_post_minus_raw`,
      `delta_pred_interval_width_post_minus_raw`,
      `delta_confidence_calibrated_post_minus_raw`,
      `delta_pinball_q10/q50/q90_post_minus_raw` presentes; teste verifica
      que `delta_picp_post_minus_raw == picp_post_guardrail - picp_raw`
      por linha.

- [x] **8.7** **Duplicar `prob_up`/`prob_down` como Categoria A.** Hoje
      `prob_up_row` em
      [`refresh_analytics_store_use_case.py:482-484`](../../src/use_cases/refresh_analytics_store_use_case.py#L482-L484)
      eh calculado via `_prob_up_from_quantiles(p10, p50, p90)` somente
      sobre quantis raw. Computar `prob_up_raw`/`prob_up_post_guardrail` e
      `prob_down_raw`/`prob_down_post_guardrail` (=`1 - prob_up_*`) no
      mesmo padrao das demais metricas Cat A do 8.1.
      **Aceite:** colunas pareadas presentes em
      `gold_prediction_metrics_by_run_split_horizon` e propagadas nas
      5 tabelas derivadas (Stage 8.2); teste unitario com fixture de
      crossing demonstra divergencia entre `prob_up_raw` e
      `prob_up_post_guardrail`.

- [x] **8.8** **Anotacao protetiva Categoria C.** Adicionar comentarios
      explicitos marcando que os seguintes checks **devem permanecer raw**
      por design (sob post-guardrail seriam tautologicamente satisfeitos
      e perderiam funcao diagnostica):
      - `_pred_interval_negative` em `_build_gold_oos_quality_report`
        ([`refresh_analytics_store_use_case.py:1300-1302`](../../src/use_cases/refresh_analytics_store_use_case.py#L1300-L1302))
        — calcula `(quantile_p90 - quantile_p10) < 0`; pertence ao quality
        gate, nao a calibration acceptance (ver
        `CALIBRATION_AND_RISK.md` §Checks complementares).
      - Gate de degeneracao quantilica (a ser implementado no Stage 11
        sobre `quantile_p10 == quantile_p90`) — detecta colapso do quantil
        emitido pelo modelo antes do mascaramento por sort.
      **Aceite:** comentario inline com referencia a
      `METRICS_DEFINITIONS.md` §Variante quantilica - Categoria C nos
      pontos citados; teste de regressao que falharia se alguem trocasse
      por `*_post_guardrail` em qualquer dos dois checks.

---

## Stage 9 — Filtro de metricas probabilisticas por modo + degeneracao (B+C)

**Objetivo:** PICP/MPIW/pinball/calibration so calculados em runs
genuinamente quantilicos e nao-degenerados.

**Cross-link:** A_code_audit.md §M5-Q5.

### Notas de revisao:

- **Data:** 2026-05-17 (branch `feat/analytics-store-stage9-probabilistic-mode-filter`).
- **Pytest executado (pos correcoes RED/YELLOW da revisao):**
  - `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
    -> **39 passed, 36 warnings** (33 baseline + 6 novos Stage 9).
  - `.venv/bin/pytest tests/unit/ -q` -> **439 passed, 38 warnings**
    (zero regressoes). Warnings residuais sao FutureWarning de pandas em
    codigo nao relacionado ao Stage 9 (idem baseline pre-Stage 9).
  - `.venv/bin/ruff check src/use_cases/refresh_analytics_store_use_case.py` -> 1 erro
    `I001` em import order **pre-existente** ao Stage 9 (confirmado via `git stash`);
    nao introduzido por este stage. Fica como follow-up housekeeping.
- **Findings da revisao (4) tratados neste mesmo PR:**
  - **[RED #1] `gold_quantile_guardrail_audit` semanticamente neutralizado**:
    audit chamava `_build_gold_prediction_metrics_by_run_split_horizon_single_contract`
    sem `fact_config` -> todas as probabilisticas before/after viravam NaN
    em producao. Corrigido propagando `fact_config` (Opcao A): audit recebe
    o filtro Cat C; runs elegiveis voltam a expor mean_pinball/picp/mpiw
    before/after/delta numericos; runs point/degenerate continuam NaN
    (consistente com upstream). Commit `718d825`.
  - **[YELLOW #2] integration test mascarava regressao silenciosa**:
    `test_refresh_analytics_store_builds_gold_tables` so checava colunas
    sem assertar conteudo. Fortalecido: r1/r2 (ambos `prediction_mode='quantile'`
    com p10!=p90) devem expor `mean_pinball_before/after` e
    `delta_mean_pinball_after_minus_before` numericos. Commit `718d825`.
  - **[YELLOW #3] 762 RuntimeWarning all-NaN slice**: emergiram com
    Stage 9 (mascaramento criou slices all-NaN em grupos point/degenerate
    consumidos por `np.nanpercentile` nas lambdas IQR em
    `_build_gold_prediction_metrics_by_config:885` e `_by_horizon:917`).
    Solucao: helper `_safe_iqr` que faz `dropna()` upstream + `np.percentile`
    direto (resultado identico, sem warning). Sem `warnings.filterwarnings`
    global. Contagem cai de 762 -> 36 (igual baseline pre-Stage 9).
    Commit `afdbb70`.
  - **[YELLOW #4] regression guard fraco**:
    `test_pred_interval_negative_unchanged_by_stage9_filter` so chamava
    `_build_gold_oos_quality_report` (builder nao tocado por Stage 9).
    Fortalecido para chamar tambem `_build_gold_prediction_metrics_by_run_split_horizon`
    e provar que crossing (p10>p90, mas p10!=p90) coexiste com
    `is_quantile_genuine=True` -- Stage 9 distingue crossing de degeneracao.
    Fixture renomeada: `r_crossing_quantile` (mais preciso semanticamente).
    Commit `33347b1`.
- **Decisao sobre `is_quantile_genuine`:** definicao **permissiva** adotada
  (`is_quantile_genuine = n_probabilistic_samples > 0`, i.e. >=1 row elegivel).
  Stage 11 implementara o gate **estrito** (% rows com p10==p90 >= 5% bloqueia)
  separadamente, com semantica de bloqueio de run/promocao. Stage 9 e upstream:
  limpa contaminacao nos agregados sem decidir promocao.
- **Decisao sobre propagacao downstream:** `is_quantile_genuine` e
  `n_probabilistic_samples` ficam **apenas** em
  `gold_prediction_metrics_by_run_split_horizon` (run-level). Nao propagados
  para as 5 tabelas derivadas (`metrics_by_config`, `_by_horizon`,
  `_calibration`, `_generalization_gap`, `_robustness_by_horizon`) nesta entrega.
  Filtro upstream ja garante que probabilisticas dessas tabelas sao calculadas
  apenas sobre rows elegiveis via NaN-propagation nos agregados pandas.
  Propagar explicitamente fica como **follow-up YELLOW** caso consumidores
  precisem do predicado per-config/per-horizon.
- **Decisao sobre `gold_quantile_guardrail_audit` (pos-RED #1):** audit recebe
  o mesmo filtro Cat C via `fact_config` (Opcao A do plano de revisao). Justificativa:
  o comentario do dual builder declara audit como "materializada por compatibilidade
  ate Phase B fechar"; manter as colunas `mean_pinball_*/picp_*/mpiw_*/delta_*`
  semanticamente vivas para runs elegiveis preserva valor diagnostico do delta
  raw-vs-post para os consumidores legados, enquanto runs point/degenerate ficam
  NaN nessas colunas (consistente com a filosofia "essas metricas nao fazem
  sentido fora de runs quantilicos genuinos"). `crossing_*_count`,
  `negative_width_*_count`, `guardrail_applied_count` continuam computados sobre
  todas as rows (sao raw-only counters, nao dependem de eligibilidade).
- **`gold_prediction_risk` NAO foi tocado** (confirmado): risk metrics ja sao
  Categoria B post-guardrail por construcao (Stage 8.5). O filtro Cat C raw
  do Stage 9 nao se aplica.
- **`gold_oos_quality_report._pred_interval_negative` permanece raw-only e
  inalterado** (Stage 8.8 anotou; Cat C). Regressao guard adicionado em
  `test_pred_interval_negative_unchanged_by_stage9_filter`.
- **Filtro sobre quantis RAW** (Cat C de `METRICS_DEFINITIONS.md` §"Variante
  quantilica"): detectar colapso emitido pelo modelo antes do guardrail
  mascarar. Aplicado tambem quando o contrato avaliado e post-guardrail
  (mesma eligibilidade per-row, independente do contrato).
- **`pred_interval_width` row mascarado**: confirmado que so alimenta MPIW +
  `confidence_calibrated` (probabilisticos). Pontuais nao referenciam.
- **Casos limite tratados:** `fact_config` vazio/None ou sem `prediction_mode`
  -> nenhum run elegivel (probabilisticas NaN no agregado, conservador).
  `prediction_mode` != `'quantile'`/`'point'` tratado como nao-quantile.
- **Cross-link com Stage 11 (gate de degeneracao)**: ainda pendente. Stage 9
  e upstream (limpa agregados); Stage 11 sera downstream (bloqueia promocao
  se share de degeneracao >= threshold).

### Tasks

- [x] **9.1** Em `_build_gold_prediction_metrics_by_run_split_horizon`,
      filtrar `fact_oos_predictions` por `prediction_mode='quantile'`
      (via merge com `fact_config`) AND `quantile_p10 != quantile_p90`
      antes do calculo das metricas probabilisticas. Manter calculo
      pontual (RMSE/MAE/DA) para todos os runs.
      **Aceite:** runs `point` ou degenerados nao contribuem para
      PICP/MPIW/pinball; aparecem em RMSE/MAE/DA normalmente.

- [x] **9.2** Adicionar coluna `is_quantile_genuine` no output de
      `gold_prediction_metrics_by_run_split_horizon` (calculada
      dinamicamente, nao persistida em silver).
      **Aceite:** consumidores podem filtrar diretamente.

- [x] **9.3** Testes unitarios cobrindo: (a) run quantile genuino entra
      em PICP; (b) run point nao entra; (c) run quantile degenerado
      (`p10==p90`) nao entra.
      **Aceite:** suite passa.

---

## Stage 10 — Rastreabilidade da inferencia (Lei 3) (B+C)

**Objetivo:** substituir `run_id=None` literal nas 3 tabelas de silver
de inferencia por FK explicita para o run de treino.

**Cross-link:** A_code_audit.md §M4 achado adicional (Lei 3).

### Notas de revisao:

- 2026-05-17: Stage 10 implementado na branch
  `feat/analytics-store-stage10-inference-training-run-id-fk` e marcado como
  em revisao. Validacao local:
  `.venv/bin/pytest tests/unit/use_cases/test_run_tft_inference_use_case.py -v`
  (`21 passed`),
  `.venv/bin/pytest tests/unit/use_cases/test_train_tft_model_use_case.py -v`
  (`28 passed`),
  `.venv/bin/pytest tests/unit/adapters/test_local_tft_inference_model_loader.py -v`
  (`6 passed`),
  `.venv/bin/pytest tests/unit/infrastructure/schemas/test_analytics_store_schema.py -v`
  (`23 passed`),
  `.venv/bin/pytest tests/unit/scripts/test_backfill_inference_training_run_id.py -v`
  (`3 passed`),
  `.venv/bin/pytest tests/unit/use_cases/ -v`
  (`173 passed, 38 warnings`),
  `.venv/bin/pytest tests/unit/ -q`
  (`451 passed, 38 warnings`) e
  `.venv/bin/ruff check src/use_cases/run_tft_inference_use_case.py src/use_cases/train_tft_model_use_case.py src/adapters/local_tft_inference_model_loader.py src/infrastructure/schemas/ tests/unit/use_cases/test_run_tft_inference_use_case.py tests/unit/use_cases/test_train_tft_model_use_case.py tests/unit/adapters/test_local_tft_inference_model_loader.py tests/unit/adapters/repositories/test_parquet_analytics_run_repository.py tests/unit/scripts/test_backfill_inference_training_run_id.py scripts/backfill_inference_training_run_id.py`
  (`All checks passed!`).
- Decisoes registradas: `training_run_id` e nullable no schema de inferencia
  por compatibilidade com rows legadas pre-Stage 10, mas passa a ser required
  no fluxo pos-Stage 10; `run_id` em `fact_inference_*` recebe o valor de
  `training_run_id` para preservar joins historicos com `dim_run`, enquanto a
  coluna explicita `training_run_id` remove ambiguidade semantica da FK pela
  Lei 3; metadata legado sem `training_run_id` gera warning unico no loader e
  persiste `None`; backfill opera em dry-run por padrao, exige `--apply` para
  escrever, e match ambiguo vira pendencia manual via `_backfill_status` sem
  preenchimento silencioso.
- Confirmado que `data/analytics_archive_pre_phase_b/` nao foi tocado. O
  comando dry-run do backfill no silver atual retornou no-op:
  `total_processed=0`, `total_backfilled=0`, `total_ambiguous=0`,
  `total_no_match=0`.
- Cross-link Stage 5.4: o gap "rows legadas de inferencia sem `run_id`
  permanecem com `parent_sweep_id=None` ate Stage 10" fecha para runs novos
  pos-Stage 10; legado pre-Stage 10 permanece documentado como pendencia
  manual quando o backfill nao encontrar match unico.
- 2026-05-17: Cenario nao coberto: se `data/analytics/silver/` for
  reconstruido sem preservar `dim_run` historico, `fact_inference_*.run_id`
  (= `training_run_id`) pode apontar para `dim_run.run_id` inexistente.
  Resultado: join em `gold_feature_contrib_local_summary` retorna `NaN` ->
  `parent_sweep_id=None`, mesmo comportamento que pre-Stage 10. Nao e
  regressao; e limitacao conhecida do modelo append-only de `dim_run`.
  Mitigacao futura: validacao no quality gate detectando `fact_inference_*.run_id`
  sem match em `dim_run` (fora de escopo Stage 10).
- 2026-05-17: Backfill e one-shot manual e nao deve rodar concorrentemente com
  `refresh_analytics_store_use_case`; nao ha lock no Parquet e uma sobrescrita
  pode perder writes intermediarios.

### Tasks

- [x] **10.1** Adicionar parametro `training_run_id` ao schema de
      `fact_inference_runs`, `fact_inference_predictions` e
      `fact_feature_contrib_local`. Atualizar
      `src/infrastructure/schemas/analytics_store_schema.py`.
      **Aceite:** schema versao bumped; documentado em
      `ANALYTICS_STORE_ARCHITECTURE.md`.

- [x] **10.2** Eliminar dependencia ambigua de `model_version` isolado.
      Tres call sites coordenados:
      - **Treino — persistir `training_run_id` no metadata do artefato:**
        em [train_tft_model_use_case.py:1031-1091](../../src/use_cases/train_tft_model_use_case.py#L1031-L1091)
        (`_persist_fact_model_artifacts`), incluir `training_run_id`
        (=`run_id` do run corrente) no `metadata.json` escrito em
        [linha 1052](../../src/use_cases/train_tft_model_use_case.py#L1052)
        e no row de `fact_model_artifacts` ([:1091](../../src/use_cases/train_tft_model_use_case.py#L1091)).
        Atualizar `src/infrastructure/schemas/model_artifact_schema.py`
        para incluir o campo.
      - **Inferencia — carregar e propagar:** no inference loader
        (`local_tft_inference_model_loader` ou equivalente), ler
        `training_run_id` do metadata do artefato. Propagar para os 3
        callsites em
        [run_tft_inference_use_case.py:76](../../src/use_cases/run_tft_inference_use_case.py#L76),
        [:120](../../src/use_cases/run_tft_inference_use_case.py#L120) e
        [:206](../../src/use_cases/run_tft_inference_use_case.py#L206),
        substituindo `"run_id": None` por `"run_id": training_run_id`.
      **Aceite:** `run_id` deixa de ser `None` em `fact_inference_runs`,
      `fact_inference_predictions` e `fact_feature_contrib_local`; a FK
      eh deterministica, sem lookup heuristico por `model_version`.

- [x] **10.3** Migrar dados existentes (se nao foi feito reset no Stage 3,
      ou se inferencia rodou em sweep limpo): script de backfill que
      preenche `training_run_id` retroativamente usando chaves compostas
      (ex.: `model_path` + `asset` + hash de config) e valida unicidade;
      se ambiguo, marcar como pendencia manual e nao preencher silenciosamente.
      **Aceite:** zero linhas com `training_run_id` nulo em
      `fact_inference_*`.

- [x] **10.4** Testes unitarios e de integracao do fluxo de inferencia.
      **Aceite:** `pytest tests/unit/use_cases/test_run_tft_inference_use_case.py` passa.

---

## Stage 11 — Gate de degeneracao quantilica (M2) (B+C)

**Objetivo:** bloquear runs com quantis colapsados conforme contrato
declarado em `prediction_mode`.

**Cross-link:** A_code_audit.md §M2-Q4.

### Notas de revisao:

- 2026-05-17, branch
  `feat/analytics-store-stage11-quantile-degeneracy-gate`.
- Commits locais:
  - `7de6dbe feat(analytics-quality): adicionar gate de degeneracao quantilica`
  - `b5c2932 feat(analytics-store): materializar gold_quantile_degeneracy_report`
  - `08431c1 test(trainer): cobrir fallback quantilico com H maior que 1`
- Validacao executada:
  - `.venv/bin/pytest tests/unit/domain/services/test_quantile_contract_analyzer.py -v`
    -> `7 passed`.
  - `.venv/bin/pytest tests/unit/use_cases/test_validate_analytics_quality_use_case.py -v`
    -> `22 passed, 2 warnings` (warnings de deprecacao legacy Block A).
  - `.venv/bin/pytest tests/unit/adapters/test_pytorch_forecasting_tft_trainer.py -v`
    -> `15 passed`.
  - `.venv/bin/pytest tests/unit/use_cases/ -v`
    -> `176 passed, 38 warnings` (warnings pre-existentes/nao bloqueantes em
    refresh pairwise e legacy Block A).
  - `.venv/bin/pytest tests/unit/ -v`
    -> `462 passed, 38 warnings`.
  - `.venv/bin/ruff check src/domain/services/quantile_contract_analyzer.py src/use_cases/validate_analytics_quality_use_case.py src/main_refresh_analytics_store.py`
    -> `All checks passed!`.
- Decisoes registradas:
  - Threshold default do gate: `n_rows >= 1000` AND `% p10==p90 >= 5%`,
    conforme audit §M7-Q6.
  - Comparacao exata (`p10 == p90`, sem `abs() < eps`) para detectar
    degenerescencia literal emitida pelo modelo.
  - Gate opera sobre quantis raw (`quantile_p10`, `quantile_p90`) por
    Categoria C de `METRICS_DEFINITIONS.md`; post-guardrail pode mascarar a
    patologia upstream.
  - Grupos `prediction_mode='point'` sao ignorados pelo bloqueio; quantis
    colapsados sao esperados como placeholder de schema.
  - Grupos `prediction_mode='quantile'` com `n_rows < 1000` entram como
    diagnostic-only, sem issue bloqueante.
  - Foi escolhida tabela propria `gold_quantile_degeneracy_report`, em vez de
    incluir o diagnostico em `gold_oos_quality_report`, porque segue o padrao
    de auditoria separada de `gold_quantile_guardrail_audit` e fica queryavel
    por `(parent_sweep_id, split, horizon, prediction_mode)`.
  - Edge case `H == Q` no `_manual_forward_quantiles_and_actuals`: comportamento
    atual documentado por teste; a funcao prefere o caminho
    `[batch, horizon, quantile]` quando as duas dimensoes coincidem. Nao houve
    alteracao de design nesse ponto.
- Cross-link: Stage 9 continua sendo filtro permissivo per-row/upstream
  (`is_quantile_genuine`); Stage 11 adiciona gate estrito per-grupo/de
  promocao (`% p10==p90 >= threshold`).
- Confirmacao de escopo: `_pred_interval_negative` (Stage 8.8) e
  `is_quantile_genuine` (Stage 9) nao foram alterados. Stage 11 e gate
  adicional, nao substituto.
- 2026-05-17 — pos-review (correcoes iter 2):
  - Commits locais:
    - `10b35d1 test(analytics-quality): guard contra uniformizacao do gate cat C para post-guardrail`
    - `5d33608 feat(analytics-store): propagar thresholds CLI ao gold_quantile_degeneracy_report`
    - `63a786b feat(analytics-quality): separar grupos por prediction_mode no gate de degeneracao`
    - commit de notas/testes diagnosticos desta iteracao:
      `docs(checklist): registrar correcoes pos-review do Stage 11`
  - Validacao executada:
    - `.venv/bin/pytest tests/unit/domain/services/test_quantile_contract_analyzer.py -v`
      -> `11 passed`.
    - `.venv/bin/pytest tests/unit/use_cases/test_validate_analytics_quality_use_case.py -v`
      -> `22 passed, 2 warnings` (warnings de deprecacao legacy Block A).
    - `.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v`
      -> `41 passed, 36 warnings` (warnings pre-existentes/nao bloqueantes em
      refresh pairwise).
    - `.venv/bin/pytest tests/unit/ -v`
      -> `467 passed, 38 warnings`.
    - `.venv/bin/ruff check src/domain/services/quantile_contract_analyzer.py src/use_cases/refresh_analytics_store_use_case.py src/main_refresh_analytics_store.py`
      -> `All checks passed!`.
  - Regression-guard contra `*_post_guardrail` adicionado, analogo ao Stage
    8.8: `analyze_degeneracy` permanece raw-only para Categoria C.
  - Thresholds CLI agora propagam ao `gold_quantile_degeneracy_report` via
    `RefreshAnalyticsStoreUseCase.__init__(degeneracy_thresholds=...)`;
    `gate_passed` na tabela honra override do operador.
  - Grupo misto `prediction_mode` agora e split por modo internamente em
    `analyze_degeneracy`. Warning removido; nao ha mais cenario "misto"; cada
    modo e avaliado independentemente.
  - Cobertura diagnostica adicionada para `fact_config` vazio e
    `prediction_mode` desconhecido. Ambos permanecem conservadores e nao
    bloqueiam o gate.

### Tasks

- [x] **11.1** Adicionar `block_quantile_degeneracy_gate` em
      `validate_analytics_quality_use_case.py` ou
      `quantile_contract_analyzer.py`. Regra:
      - `prediction_mode='quantile'` + `% p10==p90 >= 5%` (com
        `n_rows >= 1000`) = falha.
      - `prediction_mode='point'` + degeneracao = OK (esperado).
      **Aceite:** gate detecta runs do historico com 91,25% de degeneracao.

- [x] **11.2** Reportar `n_rows`, `% p10==p90`, `% p10==p50==p90`
      por `(parent_sweep_id, split, horizon)` no output do gate.
      **Aceite:** quality report eh suficiente para debug pos-treino.

- [x] **11.3** Adicionar teste unitario do fallback
      `_manual_forward_quantiles_and_actuals` com `H>1`, validando
      valores especificos `(n, h, q)` em layouts
      `[batch, horizon, quantile]` e `[batch, quantile, horizon]`.
      **Aceite:** teste em `tests/unit/adapters/test_pytorch_forecasting_tft_trainer.py`
      passa.

---

## Stage 12 — Baselines estatisticos no mesmo grao/scope (M7-Q2) (B+C)

**Objetivo:** fechar RED de readiness operacional: persistir/avaliar baselines
no mesmo contrato de grao do TFT para comparacao pareada valida.

**Cross-link:** A_code_audit.md §M7-Q2 e caveat de `_pairwise_group_cols`.

### Notas de revisao:

- 2026-05-17, branch `feat/analytics-store-stage12-statistical-baselines`.
  Pytest: `tests/unit/use_cases/test_run_baselines_use_case.py` 9 testes
  passando; `test_validate_analytics_quality_use_case.py` 26 testes passando
  (22 originais + 4 do gate Stage 12); `test_refresh_analytics_store_use_case.py`
  43 testes passando (41 originais + 2 do par TFT+baseline). Ruff limpo nas
  areas tocadas.
- **Subset MVP entregue** (3 baselines): `zero_return` (point),
  `historical_mean_rolling` (window=30, point),
  `historical_quantiles_rolling` (window=252, quantile).
  **Follow-up YELLOW** (Stage 12-bis ou emenda pre-registro): `random_walk`,
  `AR(1)`, `EWMA-vol` — nao bloqueiam H2a/H2b com subset atual mas devem ser
  registrados como pendencia.
- Decisoes registradas:
  - `run_id` deterministico via sha256(canonical_json({baseline_name, asset,
    parent_sweep_id, seed, window})). Lei 2 do Stage 1 aplica: re-execucao com
    mesmos inputs colide ate `overwrite_on_collision=True`.
  - `parent_sweep_id` e parametro OBRIGATORIO no `RunBaselinesUseCase.execute`
    (sem default). `ValueError` imediato se vazio.
  - `feature_set_name='baseline'`, `model_version=f'baseline_{name}_v1'` —
    permite ao gate 12.2 e a downstream pairwise distinguir candidato vs
    baseline sem heuristica fragil.
  - `prediction_mode='point'` para `zero_return` e `historical_mean_rolling`;
    `'quantile'` para `historical_quantiles_rolling`. Stage 9 filter exclui
    baselines pontuais de PICP/MPIW automaticamente. Stage 11 degeneracy gate
    nao dispara em modo `point` (ja contemplado).
  - `training_run_id`: nao aplicavel ao Stage 12 (baseline nao gera artifacts
    de modelo treinado; `fact_model_artifacts` nao e populada para baselines).
    Stage 10 invariante preservada por contrato — `fact_inference_predictions`
    nao recebe linhas de baseline.
  - Guardrail monotonico (`QuantileGuardrailService.enforce_monotonic_triplet`)
    aplicado a TODOS os baselines (paridade metodologica com TFT — principio
    H2a/H2b do `20_method.md`). Para baselines pontuais com quantis triviais
    iguais, post_guardrail == raw (nao-op, mas o passo do pipeline e
    identico).
  - Janela default justificada: `historical_mean_rolling=30` (suaviza ruido
    diario sem capturar regime de medio prazo); `historical_quantiles_rolling=252`
    (~1 ano de pregoes, anualidade financeira). Janelas sao parametrizaveis
    via `BASELINE_SPECS` — basta declarar nova `BaselineSpec` para alterar.
  - Warmup incompleto -> skip da linha; sem zero/NaN silencioso (validado em
    `test_historical_mean_skips_rows_without_warmup_window`).
  - Sem look-ahead: `history = target_returns[:i]` estritamente passada
    relativa ao decision_timestamp (validado em `test_baseline_no_lookahead_invariant`).
- Gate 12.2 (`baselines_share_parent_sweep_id_with_candidates`) ativo APENAS
  em `scope_spec.scope_mode == 'cohort_decision'`. Em `global_health` retorna
  `skipped(not_cohort_decision)` — garante que silvers historicos pre-Stage
  12 nao disparam falso positivo. Sweep com `parent_sweep_id` nulo/none e
  ignorado.
- Cross-links: Stage 1 (overwrite_on_collision/Lei 2); Stage 5
  (`_pairwise_group_cols` cohort-aware via `parent_sweep_id`); Stage 8
  (variante quantilica raw/post-guardrail); Stage 9 (prediction_mode filter
  para metricas probabilisticas); Stage 10 (`training_run_id` invariante
  preservada por nao tocar `fact_model_artifacts`); Stage 11 (gate de
  degeneracao convive com baselines pontuais via `prediction_mode='point'`).

- 2026-05-17 (follow-up review), branch idem. Findings YELLOW do review
  inicial resolvidos:
  - **F1 (y_true multi-horizon): case (b)**. Investigacao em
    [`src/adapters/pytorch_forecasting_tft_trainer.py:480-544`](../../src/adapters/pytorch_forecasting_tft_trainer.py#L480-L544)
    confirmou que `actuals_matrix[i, h-1]` vem do dataloader
    pytorch-forecasting como o target_return no passo futuro (h-1 ahead);
    [`src/use_cases/train_tft_model_use_case.py:790`](../../src/use_cases/train_tft_model_use_case.py#L790)
    lê `y_true = y_true_m[i][h_idx]`. Baseline ajustado em
    [`src/use_cases/run_baselines_use_case.py`](../../src/use_cases/run_baselines_use_case.py)
    `_emit_oos_rows` para usar `y_true = target_returns[i + h - 1]` com skip
    quando `i + h - 1 >= len(df)` (sem ground truth futuro disponivel) ou
    quando o valor nao for finito.
  - **F2 (no-lookahead literal)**: teste deterministico
    `test_historical_mean_rolling_uses_only_strictly_past_history` (serie
    1..100, window=10, assert literal `y_pred == 25.5` em i=30 + assert
    negativo contra off-by-one).
  - **F3 (win_rate pairwise)**: cobertura adicionada em
    `test_refresh_win_rate_pairwise_includes_candidate_vs_baseline_when_shared_parent_sweep_id`.
  - **F4 (janelas parametrizaveis)**: parametro `baseline_windows: dict[str, int] | None`
    em `RunBaselinesUseCase.execute`. Janela efetiva entra em `run_id`,
    `feature_set_hash`, `config_signature`, `fact_config.max_encoder_length`
    e `training_config_json` — determinismo preservado (mesma override ->
    mesmo run_id; override diferente -> run_id distinto, sem colidir).
  - **F5 (asserts fracos quantis)**: cobertura literal
    `test_historical_quantiles_rolling_uses_only_strictly_past_history`
    bate p10/p50/p90 contra `np.percentile(target_returns[i-w:i], q)`.
  - **G3 (janela no pre-registro)**: janelas sao parametros de runtime via
    `baseline_windows`; defaults registrados nas Notas servem para reprodu-
    cao do MVP, override permitido para sweeps futuros sem code change.
  - **G10 (DataFrame vazio)**: `test_baseline_empty_list_returns_noop_result`
    confirma `baselines=[]` retorna noop sem escrever silver.
  - Renomeacao para clareza: `test_baseline_no_lookahead_invariant` ->
    `test_baseline_target_timestamp_ordering` (o teste original so cobria
    ordenacao temporal; literal no-lookahead passa aos novos testes
    deterministicos).
  Pytest: 492 passed (vs 482 baseline + 10 testes novos -- 9 em
  `test_run_baselines_use_case.py` (5 F2/F5/G10/F1-pairwise + 4 F4) e
  1 em `test_refresh_analytics_store_use_case.py` (F3 win_rate)).
  Ruff limpo nas areas tocadas.

### Tasks

- [x] **12.1** Implementar runner de baselines persistindo em
      `fact_oos_predictions` no mesmo grao:
      `(run_id, parent_sweep_id, split, horizon, target_timestamp_utc, y_true, y_pred)`
      e quantis quando aplicavel. Lista canonica conforme `A_code_audit.md`
      §M7-Q2:
      - `zero_return` (ponto + quantis triviais) — **entregue (MVP)**
      - random walk — *follow-up YELLOW*
      - media historica — **entregue (MVP)** como `historical_mean_rolling`
      - AR(1) — *follow-up YELLOW*
      - EWMA-vol (quantis historicos derivados de volatilidade EWMA) —
        *follow-up YELLOW*
      - quantis historicos (empirical p10/p50/p90) — **entregue (MVP)** como
        `historical_quantiles_rolling`
      Subset minimo a entregar deve ser fixado no pre-registro.
      **Aceite:** baselines aparecem em silver com `status=ok` e entram
      nas tabelas pareadas DM/MCS/win-rate.

- [x] **12.2** Garantir que baselines da rodada confirmatoria compartilham
      o mesmo `parent_sweep_id` dos candidatos TFT.
      **Aceite:** check automatizado falha quando baseline/candidato caem
      em sweep ids diferentes.

- [x] **12.3** Cobrir em testes/fixture de refresh e quality gate:
      candidato + baseline no mesmo sweep geram comparacoes pareadas
      nao vazias em `gold_dm_pairwise_results` e `gold_mcs_results`.
      **Aceite:** suite alvo passa.

---

## Stage 13 — Bloquear test set como objective_metric do HPO (M1) (B+C)

**Objetivo:** remover leakage metodologico em Optuna/HPO.

**Cross-link:** A_code_audit.md §M1-Q4.

### Notas de revisao:

- 2026-05-17, branch `fix/hpo-stage13-block-test-set-as-objective`.
  Suite: `.venv/bin/pytest tests/unit/` -> 499 passed, 38 warnings, 31.50s
  (baseline pos-Stage 12 + 7 novos em
  `tests/unit/use_cases/test_run_tft_optuna_search_use_case.py`). Ruff
  limpo em `src/use_cases/run_tft_optuna_search_use_case.py`,
  `src/main_tft_optuna_sweep.py` e nos testes novos.
- Decisoes registradas:
  - Validacao **eager** no `__init__` via classvar `VALID_OBJECTIVE_METRICS`
    (`("robust_score", "mean_val_rmse")`) + type hint
    `Literal["robust_score", "mean_val_rmse"]` no parametro.
    Falha imediata na instanciacao, antes de qualquer trial Optuna ser
    submetido — evita iniciar sessoes longas sob metrica leakage-prone.
  - Mensagem do `ValueError` cita literalmente "leakage" e "A_code_audit.md §M1-Q4"
    para auto-documentar a restricao a quem encontrar o erro.
  - `_objective_from_summary` ficou com apenas duas branches (`robust_score`,
    `mean_val_rmse`); fallback `ValueError` mantido como defesa em
    profundidade (unreachable se a validacao do `__init__` estiver intacta —
    coberto por `test_objective_from_summary_has_no_test_set_branches`).
  - Default `objective_metric=None`/string vazia continua normalizando para
    `"robust_score"` antes da validacao — preservou comportamento de configs
    antigos que omitem o campo.
  - CLI `--objective-metric` aceita apenas `{robust_score, mean_val_rmse}`;
    argparse mostra `choices` automaticamente no `--help`, sem mudanca de
    help text. Configs YAML/JSON com `objective_metric: mean_test_rmse`
    bypassam o argparse mas falham no `__init__` do use case — sem
    validacao duplicada na CLI (single source of truth no use case).
  - **NAO** tocado: `RunTFTModelAnalysisUseCase` (audit §M1-Q4 linhas
    217-219 explicitamente preserva ranking diagnostico por
    `mean_test_rmse`) e
    `tests/unit/domain/services/test_explicit_config_sweep_analysis_service.py:62`
    (coluna `mean_test_rmse` em `ranking_oos` e diagnostica, nao objective
    HPO). Plot `_save_val_test_metrics_plot` continua mostrando test rmse
    para visualizacao diagnostica de top-k trials — metrica e calculada e
    persistida normalmente; Stage 13 remove apenas como **opcao de
    selecao HPO**.
  - Smoke manual: `python -m src.main_tft_optuna_sweep --objective-metric mean_test_rmse`
    -> `argparse error: invalid choice`; instanciacao direta com
    `objective_metric='mean_test_rmse'` -> `ValueError` citando leakage / M1-Q4.

### Tasks

- [x] **13.1** Em
      [run_tft_optuna_search_use_case.py:49,176-184](../../src/use_cases/run_tft_optuna_search_use_case.py#L49):
      remover `mean_test_rmse` e `joint_val_test_rmse` da lista de
      `objective_metric` aceitos.
      **Aceite:** instanciar com essas opcoes levanta `ValueError`.

- [x] **13.2** Atualizar CLI [main_tft_optuna_sweep.py](../../src/main_tft_optuna_sweep.py)
      para nao expor essas opcoes em `--objective-metric`.
      **Aceite:** help da CLI nao lista as opcoes invalidas.

- [x] **13.3** Atualizar testes que cobrem essas opcoes.
      **Aceite:** suite passa.

---

## Stage 14 — Preservar effective_date no dataset (M3) (B+C)

**Objetivo:** rastreabilidade direta de fundamentals no dataset final
sem cruzar com fontes processadas.

**Cross-link:** A_code_audit.md §M3-Q4.

### Notas de revisao:

- 2026-05-17 — branch `feat/dataset-stage14-preserve-fundamentals-effective-date`,
  pytest `tests/unit/use_cases/test_build_tft_dataset_use_case.py -v` ->
  15 passed (3 novos, 12 baseline); suite completa
  `pytest tests/unit/` -> 502 passed, zero regressoes; ruff limpo nos
  arquivos tocados.
- **Estrategia de rename**: Opcao B — rename apenas no merge_asof
  (`fundamentals_df.rename(columns={"effective_date":
  "fundamentals_effective_date"})` antes do merge). Mudanca cirurgica que
  preserva `effective_date` interno na construcao em
  [_fundamentals_to_df:115-142](../../src/use_cases/build_tft_dataset_use_case.py#L115-L142)
  e mantem intacta a checagem anti-leakage em
  [_validate_feature_anti_leakage:409-416](../../src/use_cases/build_tft_dataset_use_case.py#L409-L416).
  Justificativa: minimal change, menor superficie de regressao.
- **Localizacao da doc do fallback**: `02_data/DATA_SOURCES.md` (e nao
  `DATA_CONTRACTS.md`). Justificativa: o fallback `fiscal_date_end +
  timedelta(days=45)` e *comportamento sobre dado ausente da fonte
  externa* (Alpha Vantage), nao contrato de schema — a coluna existe
  e mantem o mesmo tipo independente do fallback. `update_when` no
  frontmatter ganhou entrada explicita "fallback de `reported_date`
  ausente para fundamentals mudar".
- **Defense-in-depth**: assert pos-merge garante
  `fundamentals_effective_date <= date` (linha 525-535 do
  `build_tft_dataset_use_case.py` apos a edicao). `ValueError` cita
  literalmente "Stage 14 / A_code_audit.md §M3-Q4" para auto-documentar
  regressao futura. Coberto por
  `test_fundamentals_effective_date_never_after_sample_date`.
- **`feature_registry.py`** atualizado em 5 entries de fundamentals
  (revenue, net_income, operating_cash_flow, total_shareholder_equity,
  total_liabilities): `formula_desc` agora cita
  `fundamentals_effective_date`. Mudanca cosmetica/documental — sem
  alteracao em `name`, `group`, `source_cols`, `anti_leakage_tag`,
  `dtype` ou `warmup_count`.
- **Rebuild parquet**: `data/processed/dataset_tft_AAPL.parquet` esta
  no `.gitignore` (regra `data/**`). Rebuild executado localmente
  (`python -m src.main_dataset_tft --asset AAPL --overwrite`); parquet
  pos-Stage 14 tem 62 cols (era 61, +1 = `fundamentals_effective_date`),
  4023 rows (igual a baseline), `fundamentals_effective_date` populada
  em 3950/4023 linhas (98.2%, NaN apenas no warmup 2010-01-04 ->
  2010-04-19). Commit do rebuild e simbolico (sem mudanca em git).
- **Checagem anti-leakage em :409-416 NAO foi tocada**. Conforme nota
  do prompt, a checagem opera sobre `effective_date` (nome antigo) pos
  merge — apos Stage 14 a coluna existe com nome novo
  (`fundamentals_effective_date`) entao o `if "effective_date" in
  df.columns` daquela checagem fica falso e o bloco vira no-op. Isso e
  intencional: a checagem original era defesa contra o caso (que nunca
  existia em producao) onde o codigo deixasse `effective_date` no df
  apos merge sem dropar; pos-Stage 14, a guarda equivalente esta no
  novo assert (defense-in-depth acima) operando sobre o nome novo.
  Mantida sem mudanca para nao expandir o blast radius do PR.
- 2026-05-17 (follow-up post-review) — endereçados 3 YELLOWs da revisão:
  (1) `canonical_for` em DATA_SOURCES.md ganhou tag
  `fundamentals_fallback_policy`; (2) INDEX.md linha 76-77 expandida para
  refletir a politica de fallback; (3) novo teste
  `test_fundamentals_effective_date_applies_45d_fallback_when_reported_date_missing`
  cobre regressão em :117-118 (`fiscal_date_end + timedelta(days=45)`).
  Suite continua em 503 passed (502 + 1 novo); ruff limpo.

### Tasks

- [x] **14.1** Renomear `effective_date` para
      `fundamentals_effective_date` durante o merge em
      `build_tft_dataset_use_case.py` e remover o `df.drop(...)` em
      [linha 519](../../src/use_cases/build_tft_dataset_use_case.py#L519).
      **Aceite:** dataset_tft_AAPL.parquet contem coluna nao-nula
      apos primeiro report disponivel.

- [x] **14.2** Documentar a justificativa do fallback "+45 dias" para
      `reported_date` ausente em `02_data/DATA_CONTRACTS.md` ou
      `02_data/DATA_SOURCES.md` (motivacao SEC 10-Q/10-K, cobertura,
      sensibilidade).
      **Aceite:** doc canonico atualizado.

- [x] **14.3** Rebuild do dataset AAPL bundled com o PR (~segundos).
      **Aceite:** `data/processed/dataset_tft_AAPL.parquet` regenerado;
      smoke teste basico passa.

---

## Stages opcionais (Caminho C — write-time + refresh deprecado)

Os Stages abaixo so se aplicam se a decisao em "Decisao Caminho B vs C" for C.
Em B, refresh continua sendo o caminho padrao para gerar gold; estes Stages
sao future work.

---

## Stage 15 — Extrair builders per-run para write-time (opt)

**Objetivo:** mover calculo de gold per-run para o fim de cada
`TrainTFTModelUseCase.execute(...)`, eliminando dependencia de refresh
para essas tabelas.

**Cross-link:** A_code_audit.md §"Leis do Analytics Store" — Lei 1.

### Notas de revisao:

### Tasks

- [ ] **15.1 (opt)** Criar `RunLevelGoldBuilder` em
      `src/use_cases/run_level_gold_builder.py` encapsulando os 8 builders
      per-run: `_build_gold_runs_long`, `_build_gold_oos_consolidated`,
      `_build_gold_prediction_metrics_by_run_split_horizon`,
      `_build_gold_quantile_guardrail_audit`, `_build_gold_prediction_calibration`,
      `_build_gold_prediction_risk`, `_build_gold_oos_quality_report` (escopo run),
      e gold de inferencia per-run aplicaveis.
      **Aceite:** builder eh testavel isoladamente (recebe DataFrames,
      retorna outputs).

- [ ] **15.2 (opt)** Invocar `RunLevelGoldBuilder` em
      [train_tft_model_use_case.py:876-1029](../../src/use_cases/train_tft_model_use_case.py#L876)
      apos as escritas de silver, com flag `--skip-write-time-gold` para
      desligar (debug).
      **Aceite:** smoke test gera gold per-run sem chamar refresh.

- [ ] **15.3 (opt)** Manter as funcoes em
      `refresh_analytics_store_use_case.py` por enquanto (rebuild scoped
      ainda usa). So remove no Stage 19.
      **Aceite:** sem regressao em refresh existente.

---

## Stage 16 — Hook de "sweep fechado" (opt)

**Objetivo:** ponto de invocacao para gold pareado/agregado (DM, MCS,
win-rate) que precisa de multiplos runs do mesmo sweep.

### Notas de revisao:

### Tasks

- [ ] **16.1 (opt)** Definir contrato de "sweep fechado": barrier-counter
      persistido em silver (ex.: nova tabela `dim_sweep_lifecycle` com
      `parent_sweep_id`, `expected_runs`, `completed_runs`,
      `closed_at_utc`).
      **Aceite:** schema documentado em `ANALYTICS_STORE_ARCHITECTURE.md`.

- [ ] **16.2 (opt)** Implementar `SweepLifecycleService` que: (a) registra
      sweep ao iniciar; (b) incrementa `completed_runs` apos cada run;
      (c) dispara `SweepLevelGoldBuilder` quando `completed_runs == expected_runs`.
      **Aceite:** testes unitarios cobrem race conditions basicas.

- [ ] **16.3 (opt)** Criar `SweepLevelGoldBuilder` encapsulando os ~12
      builders pareados/agregados: `_build_gold_dm_pairwise_results`,
      `_build_gold_mcs_results`, `_build_gold_win_rate_pairwise_results`,
      `_build_gold_paired_oos_intersection_by_horizon`,
      `_build_gold_model_decision_final`, `_build_gold_quality_statistics_report`,
      `_build_gold_quality_run_sweep_summary`,
      `_build_gold_ranking_by_config`, `_build_gold_consistency_topk`,
      `_build_gold_ic95`, `_build_gold_feature_set_impact`,
      `_build_gold_prediction_metrics_by_config` e variantes by_horizon.
      **Aceite:** builder testavel isoladamente.

- [ ] **16.4 (opt)** Wire-up em `main_tft_param_sweep.py` e
      `RunTFTOptunaSearchUseCase` para registrar o lifecycle.
      **Aceite:** sweep completo dispara `SweepLevelGoldBuilder`
      automaticamente.

---

## Stage 17 — Atomicidade multi-tabela (opt)

**Objetivo:** garantir que falha no meio da escrita de gold per-run
ou per-sweep nao deixe estado parcial.

### Notas de revisao:

### Tasks

- [ ] **17.1 (opt)** Implementar padrao "write to temp + atomic rename":
      cada builder escreve em diretorio temporario; ao concluir todas as
      tabelas, faz rename atomico de todos os paths.
      **Aceite:** falha simulada no meio da escrita deixa estado
      consistente (ou tudo antigo, ou tudo novo).

- [ ] **17.2 (opt)** Adicionar tabela `dim_gold_manifest` que registra
      `(parent_sweep_id, gold_set_version, written_at, status)` para
      rastrear escrita parcial vs completa.
      **Aceite:** consumidor pode filtrar por `status='complete'`.

- [ ] **17.3 (opt)** Testes de integracao com falha injetada.
      **Aceite:** zero estado corrompido apos falha.

---

## Stage 18 — Migrar consumidores (opt)

**Objetivo:** atualizar plot generators, validate quality e qualquer
downstream para esperar gold cohort-aware nativo (sem refresh global).

### Notas de revisao:

### Tasks

- [ ] **18.1 (opt)** Atualizar
      [generate_prediction_analysis_plots_use_case.py](../../src/use_cases/generate_prediction_analysis_plots_use_case.py)
      para ler gold filtrado por `parent_sweep_id` direto, sem assumir
      conjunto global.
      **Aceite:** plots geram para uma coorte sem necessidade de refresh.

- [ ] **18.2 (opt)** Atualizar
      [validate_analytics_quality_use_case.py](../../src/use_cases/validate_analytics_quality_use_case.py)
      para considerar a presenca de `dim_gold_manifest` ao avaliar
      completude.
      **Aceite:** quality gate distingue "gold em construcao" de
      "gold completo".

- [ ] **18.3 (opt)** Atualizar `main_refresh_analytics_store.py` para
      modo "rebuild scoped" apenas (ja nao eh fluxo padrao).
      **Aceite:** CLI exige `--scope-sweep-prefixes` obrigatorio.

---

## Stage 19 — Deprecar refresh global (opt)

**Objetivo:** marcar `RefreshAnalyticsStoreUseCase` como deprecated e
documentar caminho oficial.

### Notas de revisao:

### Tasks

- [ ] **19.1 (opt)** Adicionar `DeprecationWarning` em
      `RefreshAnalyticsStoreUseCase.__init__` quando `scope_spec is None`.
      **Aceite:** chamadas globais geram warning claro com link para
      `PHASE_B_IMPLEMENTATION_CHECKLIST.md`.

- [ ] **19.2 (opt)** Atualizar `ANALYTICS_STORE_ARCHITECTURE.md` declarando
      write-time como caminho oficial e refresh global como deprecated.
      **Aceite:** doc canonico reflete novo paradigma.

- [ ] **19.3 (opt)** Eventualmente (post-Phase B): remover
      `RefreshAnalyticsStoreUseCase` se rebuild scoped via
      `SweepLevelGoldBuilder.rebuild(parent_sweep_id)` cobrir o use case
      de emergencia.
      **Aceite:** sem callsite de refresh global no main; testes
      atualizados.

---

## Stages de refator (debito arquitetural revelado pelo F.0 E2E)

Stages 20-22 abordam debito estrutural produzido pela execucao dos
Stages 1-14 + F.0, identificado em
[`B_architectural_debt_2026-05-19.md`](../07_reports/phase-gates/B_architectural_debt_2026-05-19.md).
Registro de descoberta vive em
[`POST_CLOSURE_FIXES_CHECKLIST.md`](POST_CLOSURE_FIXES_CHECKLIST.md) §"Stage F.A"
(origem: smoke F.1 iteration); execucao vive aqui (porte: Stage 15+ → promotion
rule do POST_CLOSURE). Os 3 Stages sao **pre-requisito do fix do Gap 6**
(TFT y_true convention) — extrair os god objects antes torna o fix uma
mudanca cirurgica em 1 arquivo, em vez de blast-radius em 5.

---

## Stage 20 — MultiHorizonPredictionPersister (refactor de habilitacao)

**Objetivo:** extrair a logica de persistencia multi-horizonte hoje
duplicada e divergente em
[`train_tft_model_use_case.py:778-810`](../../src/use_cases/train_tft_model_use_case.py#L778-L810)
e [`run_baselines_use_case.py:225-280`](../../src/use_cases/run_baselines_use_case.py#L225-L280)
para um domain service que materializa **explicitamente a convencao
target_timestamp / y_true / h-ahead**. Pre-requisito do fix do Gap 6.

**Cross-link:**
[`ADR-0003-multi-horizon-prediction-persister.md`](../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md);
[`B_architectural_debt_2026-05-19.md`](../07_reports/phase-gates/B_architectural_debt_2026-05-19.md) §"M-train_tft persistencia";
[`tft_y_true_investigation_2026-05-18.md`](../07_reports/tft_y_true_investigation_2026-05-18.md).

### Notas de revisao:

- 2026-05-19: Stage criado a partir do debito identificado em
  `B_architectural_debt_2026-05-19.md`. Bloqueia fix do Gap 6
  (registrado como candidate Stage 23 ou Stage F.0.10 promovido).

### Tasks

- [ ] **20.0** ADR-0003 mergeado (doc-only PR, precede codigo).
      **Aceite:** ADR define anchor canonico de `target_timestamp`
      (decision_day), trading-day arithmetic, signature
      `build_record(decision_idx, h, y_true, y_pred, q10, q50, q90,
      dataset_timestamps, run_context)`, e politica de borda
      (`IncompletePredictionWindowError`).

- [ ] **20.1** Criar `src/domain/services/multi_horizon_prediction_persister.py`
      com `MultiHorizonPredictionPersister`, `PredictionRecord`,
      `RunContext`, `IncompletePredictionWindowError`.
      **Aceite:** testes unitarios em
      `tests/unit/domain/services/test_multi_horizon_prediction_persister.py`
      cobrindo: (a) anchor `decision_day` em todos h ∈ {1, 7, 30};
      (b) trading-day arithmetic (`target_ts =
      dataset_timestamps[decision_idx + h]`); (c) erro explicito
      `IncompletePredictionWindowError` quando
      `decision_idx + h > len(dataset_timestamps)`.

- [ ] **20.2** Migrar `train_tft_model_use_case` para usar o persister.
      **Aceite:** linhas 770-810 viram chamada unica ao persister;
      `pytest tests/` green; teste de regressao de Stage 11.3
      (`_manual_forward_quantiles_and_actuals` H>1) continua passando.

- [ ] **20.3** Migrar `run_baselines_use_case` para usar o persister.
      **Aceite:** linhas 225-280 viram chamada unica; comentario
      enganoso de linhas 244-248 removido; testes em
      `tests/unit/use_cases/test_run_baselines_use_case.py` adaptados.

- [ ] **20.4** Teste de regressao cross-pipeline pareando
      `(decision_idx, h)` entre TFT e baseline.
      **Aceite:** novo teste em
      `tests/integration/test_tft_baselines_y_true_alignment.py` confirma
      `y_true_TFT(decision_idx, h) == y_true_baseline(decision_idx, h)`
      em >= 99.9% das linhas em dataset sintetico. **Fecha a porta para
      a categoria de divergencia que produziu o Gap 6.**

- [ ] **20.5** Atualizar `docs/03_modeling/MULTI_HORIZON.md` declarando
      a convencao canonica (referenciando ADR-0003) e remover
      ambiguidade da definicao atual de `target_timestamp`.
      **Aceite:** doc canonico fixa anchor, h-ahead semantics e
      trading-day arithmetic.

- [ ] **20.6** Smoke regression: F.1 (re-rodada) ainda PASS no gate
      `tft_baselines_timestamp_subset_alignment` (F.0.3) e nos demais
      gates nao-Gap-6.
      **Aceite:** relatorio anexo em
      `docs/07_reports/smoke_confirmatory_2026-05-18.md` secao
      "Post-Stage 20".

---

## Stage 21 — QualityCheckRegistry (refactor de habilitacao)

**Objetivo:** extrair os ~20 quality checks heterogeneos hoje
empilhados em [`validate_analytics_quality_use_case.py`](../../src/use_cases/validate_analytics_quality_use_case.py)
para um registry de classes `QualityCheck` independentes, com
`applies_when(scope)` explicito e teste isolado por check.

**Cross-link:**
[`ADR-0004-quality-check-registry.md`](../01_architecture/decisions/ADR-0004-quality-check-registry.md);
[`B_architectural_debt_2026-05-19.md`](../07_reports/phase-gates/B_architectural_debt_2026-05-19.md) §"M-validate_quality".

### Notas de revisao:

- 2026-05-19: Stage criado. Caso clinico E2 (commits `584373e` + `ab79996`)
  documentado no diagnostico.

### Tasks

- [ ] **21.0** ADR-0004 mergeado (doc-only PR, precede codigo).
      **Aceite:** ADR define contrato `QualityCheck` (name,
      applies_when, run), `CheckResult`, `AnalyticsSnapshot`,
      `QualityCheckRegistry`.

- [ ] **21.1** Criar `src/domain/services/quality_checks/base.py` com
      ABC `QualityCheck`, `CheckResult`, `AnalyticsSnapshot`,
      `QualityCheckRegistry`.
      **Aceite:** registry default povoavel; testes em
      `tests/unit/domain/services/quality_checks/test_base.py`.

- [ ] **21.2** Migrar cluster `cardinality.py` (cardinality_config_fold_seed,
      min_samples_by_split, run_id_execution_consistency,
      inference_predictions_continuity, feature_contrib_local_continuity).
      **Aceite:** 5 checks migrados; testes movidos para
      `tests/unit/domain/services/quality_checks/test_cardinality.py`;
      `execute()` chama via registry em vez de inline.

- [ ] **21.3** Migrar cluster `alignment.py`
      (tft_baselines_timestamp_subset_alignment,
      oos_pairwise_target_alignment).
      **Aceite:** 2 checks migrados; F.0.3 gate continua falhando
      corretamente para Gap 6 ate Stage 20 fechar.

- [ ] **21.4** Migrar cluster `calibration.py`
      (gold_confidence_calibrated_by_horizon, dm_mcs_persisted_executable,
      gold_metrics_by_config_n_oos_contract).
      **Aceite:** 3 checks migrados; filtro `is_quantile_genuine` (F.0.6)
      preservado.

- [ ] **21.5** Migrar cluster `contracts.py`
      (official_contract_quantile_attention, required_metrics_nan,
      baseline_present_per_candidate_sweep, referential_integrity).
      **Aceite:** 4 checks migrados; exclusao de baselines do artifact
      contract (F.0.7) preservada.

- [ ] **21.6** `ValidateAnalyticsQualityUseCase.execute()` vira
      orquestrador: itera `registry.applicable(scope)`.
      **Aceite:** classe principal reduz para ~200 LOC; comportamento
      bit-for-bit identico — smoke regression em `dim_run`
      `parent_sweep_id=phase_a_smoke_20260518_v2` produz mesmos 27
      checks com mesmos verdicts.

- [ ] **21.7** Atualizar
      [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md)
      documentando o registry como ponto de extensao oficial.
      **Aceite:** doc canonico explica como adicionar novo check sem
      tocar `execute()`.

---

## Stage 22 — GoldBuilders modulares (refactor de habilitacao)

**Objetivo:** extrair os 26 builders gold hoje inline em
[`refresh_analytics_store_use_case.py`](../../src/use_cases/refresh_analytics_store_use_case.py)
para pacote `src/domain/services/gold_builders/` organizado por
categoria semantica.

**Cross-link:**
[`ADR-0005-gold-builders-modularization.md`](../01_architecture/decisions/ADR-0005-gold-builders-modularization.md);
[`B_architectural_debt_2026-05-19.md`](../07_reports/phase-gates/B_architectural_debt_2026-05-19.md) §"M5-refresh".

### Notas de revisao:

- 2026-05-19: Stage criado. Maior dos 3 refators (~2.600 LOC
  reorganizados); diff comportamentalmente neutro garantido por smoke
  regression byte-identical entre cada task.
- Se a revisao indicar que diff e grande demais para 1 PR, dividir em
  Stage 22a (22.0 + 22.1 + 22.2 + 22.3) + Stage 22b (22.4 + 22.5 + 22.6
  + 22.7 + 22.8). Comecar como 1 Stage; dividir so se necessario.

### Tasks

- [ ] **22.0** ADR-0005 mergeado (doc-only PR, precede codigo).
      **Aceite:** ADR define contrato `GoldBuilder`, estrutura do
      pacote `gold_builders/` (5 categorias), e politica de
      `requires: list[str]` explicito.

- [ ] **22.1** Criar `src/domain/services/gold_builders/base.py` com
      ABC `GoldBuilder`, `BuildContext`, `AnalyticsSnapshot` (reusar do
      Stage 21 se mergeado).
      **Aceite:** skeleton do pacote; `__init__.py` com registry default;
      testes em `tests/unit/domain/services/gold_builders/test_base.py`.

- [ ] **22.2** Migrar categoria `ranking.py` (runs_long,
      ranking_by_config, consistency_topk).
      **Aceite:** 3 builders migrados; refresh continua produzindo
      output byte-identical para `gold_runs_long`,
      `gold_ranking_by_config`, `gold_consistency_topk`.

- [ ] **22.3** Migrar categoria `pairwise.py` (dm_pairwise_results,
      mcs_results, paired_oos_intersection_by_horizon).
      **Aceite:** 3 builders migrados; refresh byte-identical.

- [ ] **22.4** Migrar categoria `quantile.py` (quantile_guardrail_audit,
      quantile_degeneracy_report, prediction_metrics_by_run_split_horizon
      + single_contract variant, prediction_metrics_by_config).
      **Aceite:** 4 builders migrados; gate de degeneracao (Stage 11)
      continua aplicando.

- [ ] **22.5** Migrar categoria `descriptive.py` (ic95,
      feature_set_impact, model_decision_final).
      **Aceite:** 3 builders migrados.

- [ ] **22.6** Migrar categoria `confidence.py`
      (confidence_calibrated_by_horizon,
      metrics_by_config_n_oos_contract).
      **Aceite:** 2 builders migrados; filtro `is_quantile_genuine`
      (F.0.6) preservado se nao migrado em outra categoria.

- [ ] **22.7** `RefreshAnalyticsStoreUseCase` vira orquestrador:
      itera `gold_builders_registry.applicable(scope_spec)`.
      **Aceite:** classe principal reduz para ~400 LOC (de 2.568); pinball,
      IQR, prob_up viram helpers em `gold_builders/_metrics.py`.

- [ ] **22.8** Smoke regression byte-identical: F.1 v2 (mesmo input)
      produz todos os 10 gold tables com o mesmo conteudo bit-for-bit
      antes vs depois.
      **Aceite:** script de diff anexo em
      `docs/07_reports/smoke_confirmatory_2026-05-18.md` secao
      "Post-Stage 22"; diff vazio em todas as 10 tabelas.

---

## Stage F.0 — movido para POST_CLOSURE_FIXES_CHECKLIST.md

Stage F.0 (fixes pre-smoke descobertos via F.1 2026-05-18) foi migrado
em 2026-05-19 para
[`POST_CLOSURE_FIXES_CHECKLIST.md`](POST_CLOSURE_FIXES_CHECKLIST.md) §"Stage F.0",
preservando este arquivo como snapshot do pre-experiment engineering
planejado upfront. Conteudo identico (F.0.0 ... F.0.9); rastreabilidade
git via branch `feat/stage-f0-pre-smoke-fixes`.

---

## Stage final — Smoke confirmatorio + pre-registro (B+C)

### Notas de revisao:

- 2026-05-17: fechamento parcial da Fase A registrado em
  [`A_audit_closure_2026-05-17.md`](../07_reports/phase-gates/A_audit_closure_2026-05-17.md)
  e marcacao parcial do gate em
  [`A_code_audit.md`](../07_reports/phase-gates/A_code_audit.md)
  §"Gate de saida da Fase A":
  - P0 transversais (6 itens) → `[x]` todos fechados.
  - Modulos M4, M5 → `[x]` (todas as acoes cobertas por Stages
    mergeados ou explicitamente Fase C/future work).
  - Modulos M1, M2, M3, M6, M7 → `[~]` (codigo Stages 1-14 mergeados,
    mas acoes dependentes de F.1 e/ou F.2 permanecem abertas).
  - "Nenhum RED em aberto" → `[x]`.
  - `Data de abertura da Fase B` → em branco; preenchida apos F.1 + F.2.
  Closure mapping validado por sessao independente de revisao
  (veredicto APPROVE_WITH_CAVEATS; 5/5 spot-checks de mapping
  passaram; 4 caveats incorporados no closure final).
- 2026-05-18: F.1 smoke veredicto FAIL; gaps documentados em
  [`smoke_confirmatory_2026-05-18.md`](../07_reports/smoke_confirmatory_2026-05-18.md);
  Stage F.0 criado para desbloquear; F.1 re-rodara apos F.0 merged.
- 2026-05-19: Stage F.0 (e futuras ondas post-closure) migrado para
  [`POST_CLOSURE_FIXES_CHECKLIST.md`](POST_CLOSURE_FIXES_CHECKLIST.md).

### Tasks

- [ ] **F.1** Smoke confirmatorio com `max_epochs >= 5`, `n_rows >= 1000`.
      **Aceite:**
      - `% p10==p90 < 5%` (gate do Stage 11);
      - zero violacoes probabilisticas;
      - gold cohort-aware gerado corretamente;
      - **suite completa de testes do projeto passa** (`pytest tests/`);
      - **CI verde** em todos os PRs dos Stages anteriores ja mergeados;
      - refresh + quality gate executados sem erro no smoke (output do
        `main_refresh_analytics_store` sob scope da coorte do smoke).

- [ ] **F.2** Pre-registro com contrato quantilico primario fundamentado
      (raw vs post-guardrail) e politica de baselines (mesmo
      `parent_sweep_id` dos candidatos ou `cohort_id` logico).
      **Aceite:** documento de pre-registro mergeado em
      `docs/06_pre_registration/` (ou local equivalente).

- [~] **F.3** Marcar `A_code_audit.md` "Gate de saida da Fase A" como
      satisfeito.
      **Aceite:** todos os checkboxes do gate marcados; data de abertura
      da Fase B preenchida.
      **Estado parcial (2026-05-17):** P0 transversais e RED zerados
      marcados `[x]`; M1/M2/M3/M6/M7 marcados `[~]` aguardando F.1+F.2;
      `Data de abertura da Fase B` em branco. Satisfacao plena requer
      F.1 e F.2 mergeados antes do `[x]` final.

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
| 12 | M7-Q2 (baselines estatisticos) | RED |
| 13 | M1-Q4 (HPO objective) | RED |
| 14 | M3-Q4 (effective_date) | YELLOW |
| 15-19 | Lei 1 (write-time, refresh deprecado) | future direction |
| 20 | B_architectural_debt §M-train_tft persistencia (ADR-0003) | YELLOW (post-F.0) |
| 21 | B_architectural_debt §M-validate_quality (ADR-0004) | YELLOW (post-F.0) |
| 22 | B_architectural_debt §M5-refresh (ADR-0005) | YELLOW (post-F.0) |
| Final | Gate de saida da Fase A | — |
