---
title: A_code_audit Closure Mapping (2026-05-17)
scope: Mapeamento por modulo (M1-M7) do `A_code_audit.md` confirmando que cada acao YELLOW/RED da Fase A foi fechada por Stage 1-14 mergeado em main (PRs #14-34), F.1 smoke pendente, F.2 pre-registro pendente, ou explicitamente registrada como out-of-scope Phase B (deferida a Fase C ou future work). Documento de evidencia para marcar Gate de saida da Fase A em `A_code_audit.md`.
update_when:
  - Stage 1-14 receber follow-up corrigindo acao fechada
  - novo achado pos-Stage aparecer e ser mapeado para acao do audit
  - Phase B revelar gap nao previsto que exija reabrir modulo M_X
canonical_for: [phase_a_closure_mapping, audit_to_stage_traceability]
---

# A_code_audit Closure Mapping — 2026-05-17

Mapeia cada acao "Acao (se YELLOW/RED)" dos modulos M1-M7 do
[`A_code_audit.md`](A_code_audit.md) para o(s) Stage(s) do
[`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../../05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md)
que a fechou (ou para F.1/F.2/Fase C quando aplicavel). Base para marcar
**Gate de saida da Fase A** com confianca.

## Resumo

| Modulo | Veredicto original | Acoes mapeadas | Fechamento |
|---|---|---|---|
| M1 — train_tft + Optuna | RED | 3 | **fechado** (2 via Stage 13; 1 via F.2) |
| M2 — TFT trainer | RED | 5 | **fechado** (4 via Stages 9/11; 1 via F.1) |
| M3 — build dataset | YELLOW | 5 | **fechado** (2 via Stage 14; 2 via F.2; 1 multi-asset future) |
| M4 — inference + engine | RED | 4 | **fechado** (1 via Stage 10; 2 Fase C; 1 future work explicito) |
| M5 — refresh analytics | RED | 9 | **fechado** (todos via Stages 1, 2, 4-9) |
| M6 — estado do store | YELLOW | 7 | **fechado** (5 via Stages 3-6; 2 via F.1/F.2; 1 Fase C) |
| M7 — capacidade Fase B | RED | 10 | **fechado** (3 via Stages 8/12, Stage 6.6; 5 via F.1/F.2; 2 future) |

Todos os P0 transversais do `A_code_audit.md` §"Gate de saida da Fase A"
(linhas 1235-1268) **fechados** via Stages 1-14 mergeados (PRs #14-34) +
decisao Caminho B registrada (commit `b74035b` mergeado em main).

**Nenhum RED em aberto.** Itens nao-fechados sao:
- YELLOW residuais com destino explicito declarado (F.1 smoke confirmatorio,
  F.2 pre-registro, ou Fase C/future work).
- 2 itens P1 do `A_code_audit.md:1275-1283` nao-bloqueantes para Phase B
  (`feature_registry.py` audit por feature; `validate_analytics_quality`
  Block A MPIW=0 check literal); ambos devem ser revisitados antes da
  analise final.
- 1 pendencia condicional (baselines `random_walk`/`AR(1)`/`EWMA-vol`):
  vira bloqueante apenas se F.2 pre-registro fixar qualquer um como
  baseline primario.
- 4 follow-ups YELLOW registrados em Notas de revisao dos Stages 4, 9, 10,
  14 (cobertura defensiva, propagacao downstream, dividas tecnicas
  reconhecidas).

## Mapeamento detalhado

### M1 — `train_tft_model_use_case.py` + Optuna (RED → fechado)

Acoes em [`A_code_audit.md:235-247`](A_code_audit.md#L235):

1. **Bloquear `mean_test_rmse`/`joint_val_test_rmse` em Optuna/HPO** →
   Stage 13.1 (validacao eager no `__init__` levanta `ValueError`).
2. **CLI `main_tft_optuna_sweep` rejeita opcoes** →
   Stage 13.2 (`choices=["robust_score", "mean_val_rmse"]`).
3. **Garantir no pre-registro que hiperparams herdados de Round 0 foram
   escolhidos por criterio de validacao** → **F.2 pre-registro** (pendente).

### M2 — `pytorch_forecasting_tft_trainer.py` (RED → fechado)

Acoes em [`A_code_audit.md:315-349`](A_code_audit.md#L315):

1. **Gate bloqueante de degeneracao condicional a `prediction_mode`** →
   Stage 11.1 (`block_quantile_degeneracy_gate` no
   `QuantileContractAnalyzer` + `ValidateAnalyticsQualityUseCase`;
   `prediction_mode='quantile'` AND `% p10==p90 >= 5%` AND
   `n_rows >= 1000` falha; `point` ignorado).
2. **Onde o gate vive** → Stage 11.1 (escolhido `QuantileContractAnalyzer`
   + integracao no validator; permite scope cohort_decision).
3. **Filtrar metricas probabilisticas por modo + nao-degeneracao** →
   Stage 9.1/9.2 (`is_quantile_genuine` per-row + per-grupo; modo
   `point` exclui de PICP/MPIW/pinball; mantem RMSE/MAE/DA).
4. **Teste unitario do `_manual_forward_quantiles_and_actuals` com H>1
   em dois layouts** → Stage 11.3 (valores especificos por `(n, h, q)`
   validados em layouts `[batch, horizon, quantile]` e
   `[batch, quantile, horizon]`).
5. **Smoke real com `max_prediction_length>=7` e
   `evaluation_horizons=[1,7]`** → **F.1 smoke** (pendente).

### M3 — `build_tft_dataset_use_case.py` (YELLOW → fechado)

Acoes em [`A_code_audit.md:424-456`](A_code_audit.md#L424):

1. **Declarar no pre-registro politica de warmup
   (`strict_fail`/`drop_leading`), `effective_train_start`,
   `feature_list_ordered`** → **F.2 pre-registro** (pendente).
2. **Artefato de auditoria source-level AAPL** (reported_date/fallback,
   effective_day, ausencia de uso antes de disponibilidade efetiva,
   consistencia sentiment vs scored_news) → **F.2 pre-registro** anexo
   (pendente; alternativamente smoke F.1 valida operacionalmente).
3. **Multi-asset: repetir auditoria por ativo** → **Out-of-scope Phase B**
   (AAPL piloto unico per `STRATEGIC_DIRECTION.md` §B.1; future work se
   Fase B incluir outros ativos).
4. **Preservar `effective_date` → `fundamentals_effective_date` no
   dataset final, remover `df.drop(...)` linha 519** → Stage 14.1
   (rename Opcao B no merge_asof; assert anti-leakage pos-merge defesa
   em profundidade; `feature_registry.py` formula_desc atualizado).
5. **Documentar fallback "+45 dias" para `reported_date` ausente** →
   Stage 14.2 (`DATA_SOURCES.md` ganhou subsecao com motivacao SEC
   10-Q/10-K, cobertura empirica 17/81 AAPL, analise de sensibilidade,
   referencia auditavel via `fundamentals_effective_date`).
   Pre-registro F.2 referenciara essa documentacao.

### M4 — `run_tft_inference_use_case.py` + engine (RED → fechado)

Acoes em [`A_code_audit.md:553-576`](A_code_audit.md#L553):

1. **Persistencia multi-horizonte real no engine de inferencia** →
   **Out-of-scope Phase B** (audit linha 559 explicita: "Caso
   contrario, registrar como future work e manter a Fase B
   confirmatoria em `fact_oos_predictions`"). Fase B usa
   `fact_oos_predictions` via TFT trainer multi-horizonte (Stage 11.3
   teste H>1 cobre via training path), nao via engine rolling.
2. **Testes multi-horizonte do engine** → N/A (decorrente do item 1).
3. **Documentar `model_version + model_path + config.json` ou
   materializar `training_run_id`/`training_seed` no Analytics Store** →
   Stage 10 (escolheu materializar `training_run_id` como FK
   deterministica; schema bump v1→v2; backfill heuristico para legado;
   `"run_id": None` substituido em 3 callsites silver de inferencia).
4. **Marcar `local_magnitude_signed_v1` como diagnostico operacional,
   nao evidencia academica primaria; considerar `evidence_tier`** →
   **Fase C** (audit linha 570 explicita: "Antes da Fase C"). Fora de
   escopo Phase B; entra em pre-registro Fase C se houver claim de
   contribuicao local.

### M5 — `refresh_analytics_store_use_case.py` (RED → fechado)

Acoes em [`A_code_audit.md:756-845`](A_code_audit.md#L756):

P0 (independentes do Caminho B vs C):

1. **Q1 raw vs post-guardrail em paralelo** → Stage 8.1/8.2/8.3/8.4
   (run-level dual + 5 derivadas propagam + selecao via
   `--primary-quantile-contract` em decision_final + plots).
2. **Q7 cross-cohort ranking em `gold_model_decision_final`** →
   Stage 4.1 (groupby ganha `parent_sweep_id`; rank dentro de
   `(asset, parent_sweep_id, horizon)`).
3. **Q8 `_append_to_parquet` violando Lei 2** → Stage 1
   (`_write_with_overwrite_policy` com flag explicita; 12 `append_*`
   migrados; `--overwrite-on-collision` CLI).
4. **Q6 teste de regressao da invariante `9f0ccec`** → Stage 2.1/2.2
   (2 testes: muda hash com `parent_sweep_id`/`fold`/`trial_number`/
   `seed`; estavel com chaves volateis temporais).
5. **Q5 filtro de metricas probabilisticas por modo + nao-degeneracao** →
   Stage 9 (mesmo item que M2; cross-link).

Caminho B (escolhido — decisao registrada 2026-05-17):

6. **Q2 subgrupo "5 com `config_signature`"** → Stage 6
   (parent_sweep_id no output das 5 tabelas).
7. **Q2 subgrupo "5 sem `config_signature`"** → Stage 5
   (parent_sweep_id no groupby das 5 tabelas).
8. **`scope_spec: ScopeSpec | None` no `RefreshAnalyticsStoreUseCase`** →
   Stage 7.1/7.2.
9. **CLI flags de scope** → Stage 7.3 (`--scope-mode`,
   `--scope-sweep-prefixes`, `--scope-splits`, `--scope-horizons`).

Caminho C (Stages 15-19) → **dispensado** pela decisao Caminho B
registrada. Reabertura exige justificativa empirica (gargalo medido).

### M6 — Estado do Analytics Store (YELLOW → fechado)

Acoes em [`A_code_audit.md:993-1017`](A_code_audit.md#L993):

1. **Nao usar tabelas gold agregadas sem cohort-aware para claims
   confirmatorios** → Stages 4/5/6 fizeram cohort-aware nas 10 tabelas
   gold agregadas; risco mitigado por construcao do contrato.
2. **Tratar `parent_sweep_id=NULL` como historico
   global-health/pre-ScopeSpec** → Stage 3.1/3.2/3.3/3.4 (reset silver/
   gold pre-Phase B; archive read-only via `chmod -R a-w`;
   `ANALYTICS_STORE_ARCHITECTURE.md` documenta archive como diagnostico
   historico, nao evidencia confirmatoria).
3. **Snapshot pre-2026-05-10 arquivado** → Stage 3 (mesma acao acima).
4. **Declarar no pre-registro filtro de coorte reproduzivel** →
   **F.2 pre-registro** (pendente).
5. **Tabelas gold agregadas cohort-aware** → Stages 4/5/6 (mesmo item 1).
6. **Smoke H=7 multi-horizonte** → **F.1 smoke** (pendente).
7. **Auditar `fact_inference_*` antes da Fase C** → **Fase C**
   (out-of-scope Phase B; relacionado a M4 Fase C).

### M7 — Capacidade operacional Fase B (RED → fechado)

Acoes em [`A_code_audit.md:1177-1226`](A_code_audit.md#L1177):

1. **Runner de baselines persistindo em `fact_oos_predictions` no mesmo
   grao** → Stage 12.1 (MVP entregue: `zero_return`,
   `historical_mean_rolling`, `historical_quantiles_rolling`;
   `parent_sweep_id` shared obrigatorio; `training_run_id` auto-ref).
   Follow-up YELLOW registrado: `random_walk`, `AR(1)`, `EWMA-vol`
   pendentes para emenda ao pre-registro se necessario.
2. **Avaliar explicit-config como caminho preferencial para Round 1** →
   **F.2 pre-registro** (decisao metodologica).
3. **Validar `main_tft_param_sweep --max-runs 1` N×K** → **F.1 smoke**
   (validacao operacional) ou dispensado se pre-registro usar
   explicit-config exclusivamente.
4. **Calcular all-features podada AAPL conforme STRATEGIC §4.2** →
   **F.2 pre-registro** (criterio congelado antes do experimento
   confirmatorio).
5. **Smoke multi-horizonte com `n_rows>=1000`** → **F.1 smoke**
   (pendente).
6. **Smoke com `max_epochs>=5`** → **F.1 smoke** (pendente).
7. **P0 do pre-registro: contrato quantilico primario raw vs
   post-guardrail** → Stage 8 implementou flag
   `--primary-quantile-contract` com default `post_guardrail`;
   policy canonical em
   [`METRICS_DEFINITIONS.md`](../../04_evaluation/METRICS_DEFINITIONS.md)
   §"Variante quantilica" e
   [`20_method.md`](../living-paper/20_method.md) §"Politica de variante
   quantilica" (Chernozhukov 2010, Gneiting & Raftery 2007, Jorion 2007,
   Acerbi & Tasche 2002). **F.2 pre-registro** fixa declaracao final.
8. **Cross-link M5: falha `gold_metrics_by_config_n_oos_contract` 13800
   mismatches** → Stage 6.6 (atualizou check para filtrar pelos dois
   lados via `parent_sweep_id` direto).
9. **Deixar h=30 fora do smoke inicial** → **F.1 smoke decision** /
   **F.2 pre-registro**.
10. **Inferencia rolling multi-horizonte como dependencia de M4;
    refresh sob escopo Fase B como dependencia de M5** → M4
    (out-of-scope; item 1 do M4 acima) + M5 (Stage 7 scope_spec). Esta
    cobertura completa pela arquitetura pos-Stages 1-14.

## P0 transversais do Gate de Saida da Fase A

`A_code_audit.md` §"Gate de saida da Fase A" linhas 1249-1264:

| Item P0 | Status | Fechamento |
|---|---|---|
| Caminho B vs C decidido e registrado | **fechado** | commit `b74035b` mergeado em main (Caminho B) |
| Q8 `_append_to_parquet` substituido | **fechado** | Stage 1 (PR #14) |
| Q7 `gold_model_decision_final` ranking cohort | **fechado** | Stage 4 (PR #17) |
| Q6 teste de regressao 9f0ccec | **fechado** | Stage 2 (PR #15) |
| Q1 raw + post-guardrail paralelos | **fechado** | Stage 8 (PR #23) |
| Lei 3 `run_id=None` removido / FK explicita | **fechado** | Stage 10 (PR #26) |

## Modulos P1 (nao-bloqueantes para Fase B)

`A_code_audit.md:1275-1283` lista 4 itens P1 fora da estrutura M1-M7. Por
declaracao do proprio audit (linha 1275), nao bloqueiam abertura da Fase B,
mas devem ser revisados antes da analise final:

| Modulo P1 | Status | Fechamento |
|---|---|---|
| `run_tft_optuna_search_use_case.py` (desabilitar `mean_test_rmse`) | **fechado** | promovido a M1-Q4 P0 → Stage 13 (PR #32) |
| `feature_registry.py` (anti_leakage_tag/warmup_count audit por feature) | **YELLOW residual P1** | nao tratado por Stage 1-14; revisar antes da analise final |
| `validate_analytics_quality_use_case.py` (Block A MPIW=0 > 5% bloqueante) | **YELLOW residual P1** | semanticamente coberto pelo gate Stage 11 (`p10==p90`), mas literal "MPIW=0" nao foi explicitamente verificada — registrar follow-up |
| `sklearn_indicator_normalizer.py` (instanciacao fora do treino?) | **moot** | audit M3-Q1 ja confirmou "nao instanciado no caminho atual" (linhas 371-380) |

## Itens nao-fechados (com destino explicito)

Nada classificado como **bloqueante**. Distribuicao:

### Pendencias para F.1 (smoke confirmatorio)

- M2 acao 5: smoke `max_prediction_length>=7`, `evaluation_horizons=[1,7]`.
- M6 acao 6: smoke H=7 com `target_timestamp_utc` coerente.
- M7 acao 3: validar `main_tft_param_sweep --max-runs 1` N×K (se OFAT usado).
- M7 acao 5: smoke `n_rows>=1000`.
- M7 acao 6: smoke `max_epochs>=5` para verificar crossing.

### Pendencias para F.2 (pre-registro)

- M1 acao 3: justificar herdanca de hiperparametros Round 0.
- M3 acoes 1, 2: politica de warmup + artefato source-level AAPL.
- M6 acao 4: declarar filtro de coorte reproduzivel.
- M7 acao 2: decisao explicit-config vs OFAT como caminho preferencial.
- M7 acao 4: lista all-features podada AAPL congelada.
- M7 acao 7: declaracao formal contrato quantilico primario (Stage 8 ja
  implementou tecnologia; pre-registro fixa a politica).
- M7 acao 9: decisao sobre h=30 (incluir ou suplementar).

### Out-of-scope Phase B (Fase C ou future work)

- M3 acao 3: auditoria multi-asset (AAPL only em Phase B).
- M4 acao 1: persistencia multi-horizonte engine de inferencia
  (explicito como future work no audit, linha 559).
- M4 acoes 2, 4: testes multi-horizonte engine (decorrente) e
  `evidence_tier` para explicabilidade (Fase C, linha 570).
- M6 acao 7: auditar `fact_inference_*` antes da Fase C (linha 1015).

### Pendencia condicional a F.2 pre-registro

- **M7 follow-up — baselines restantes (`random_walk`, `AR(1)`,
  `EWMA-vol`)**: classificacao depende da escolha do baseline primario
  no F.2 pre-registro. Se F.2 fixar qualquer um destes tres como
  baseline primario para H2a/H2b, vira **bloqueante** para Phase B e
  exige Stage 12-bis ou emenda ao pre-registro. Alinha com a postura
  cautelosa registrada em Stage 12 Notas ("follow-up YELLOW para
  Stage 12-bis ou emenda pre-registro"). Audit M7-Q2 (linhas 1179-1183)
  lista os baselines com "e/ou", permitindo subset — mas o subset
  permitido depende da decisao formal em F.2.

### YELLOW residuais abertos (de Notas de revisao dos Stages)

Itens reconhecidos durante reviews dos Stages 1-14 como follow-ups
nao-bloqueantes para Phase B, mas reais. Nao cobrem nenhum item do gate
de saida, mas merecem registro para auditoria futura:

- **Stage 4**: cobertura explicita para `parent_sweep_id=None` legado
  pre-9f0ccec ausente em teste (PHASE_B_IMPLEMENTATION_CHECKLIST.md
  Stage 4 Notas). Filtro upstream ja exclui esse caso na pratica; gap
  apenas de cobertura de teste defensiva.
- **Stage 9**: propagacao de `is_quantile_genuine` para as 5 tabelas
  derivadas (`metrics_by_config`, `_by_horizon`, `_calibration`,
  `_generalization_gap`, `_robustness_by_horizon`) nao implementada.
  Filtro upstream ja garante limpeza dos agregados via
  NaN-propagation, mas predicado per-config/per-horizon ausente para
  consumidores que precisarem.
- **Stage 10**: backfill heuristico de `training_run_id` em silver
  legado e one-shot manual sem lock. Para silver reconstruido apos
  reset Stage 3, esses runs ficam com `training_run_id` nulo ou via
  match heuristico ambiguo. Documentado como pendencia manual.
- **Stage 14**: anti-leakage check em `build_tft_dataset_use_case.py:409-416`
  virou tautologico apos rename (`effective_date` → `fundamentals_effective_date`).
  Defense-in-depth assert pos-merge cobre semantica nova; check antigo
  e divida tecnica para remocao/refactor futuro.

## Conclusao

**Todos os RED do audit estao fechados.** Todos os P0 transversais do
Gate de saida da Fase A tem evidencia rastreavel (PRs #14-34 mergeados +
commit `b74035b` Caminho B).

**YELLOW residuais** estao distribuidos entre:
- F.1 smoke (5 itens) e F.2 pre-registro (7 itens) — bloqueam abertura
  formal da Phase B confirmatoria, mas nao a marcacao parcial do gate.
- Out-of-scope Phase B / Fase C / future work (5 itens) — declaracoes
  defensiveis com base no texto literal do audit.
- 1 pendencia condicional a F.2 (baselines restantes).
- 4 follow-ups YELLOW de Notas de revisao dos Stages (cobertura
  defensiva, propagacao downstream, dividas tecnicas).
- 2 itens P1 nao-bloqueantes (`feature_registry.py`,
  `validate_analytics_quality` Block A literal).

Criterio formal "Nenhum modulo com veredicto RED em aberto" +
"M_X GREEN ou YELLOW com acao concluida" do `A_code_audit.md:1247-1248`
**satisfeito para o subset de acoes de codigo** (Stages 1-14 mergeados).

**Estrategia de marcacao parcial do gate** (proxima acao):

- **P0 transversais** (`A_code_audit.md:1253-1264`): marcar `[x]` — todos
  os 6 itens sao puramente codigo, todos fechados via Stages mergeados.
- **Modulos M1-M7**: marcacao bipartida conforme dependencia:
  - M4, M5: `[x]` — todas as acoes cobertas por Stages mergeados
    (M4 itens out-of-scope estao explicitos no audit como future work;
    M5 todos os P0 + Caminho B fechados).
  - M1, M2, M3, M6, M7: `[~]` — depende de F.1 e/ou F.2 para fechamento
    pleno; codigo entregue mas marco documental/operacional pendente.
- **"Nenhum modulo com veredicto RED em aberto"** (`:1247`): `[x]`
  marcavel — verificado, nenhum RED restante.
- **`Data de abertura da Fase B`** (`:1270`): manter em branco com nota
  "preenchida apos F.1 smoke confirmatorio + F.2 pre-registro mergeados".

Esta estrategia preserva honestidade do gate: P0 fechados e RED zerados
sao fato; itens condicionados a F.1/F.2 mantem visibilidade ate
satisfacao plena.

**Sem gaps materiais descobertos** durante esta verificacao e revisao
independente (relatorio APPROVE_WITH_CAVEATS de sessao de auditoria
independente; 5/5 spot-checks de mapping passaram). Cada acao YELLOW/RED
do audit tem ou Stage que a fechou, ou destino declarado (F.1, F.2,
Fase C, future work, P1 nao-bloqueante).

## Findings post-closure (descobertos via F.1 smoke 2026-05-18)

Verificacao independente (closure original) deu APPROVE_WITH_CAVEATS em
2026-05-17. F.1 smoke executado em 2026-05-18 descobriu 5 gaps que
nenhuma das revisoes anteriores pegou (Stage 12 review, closure mapping,
revisao independente do closure). Registro honesto:

- **Gap 1 — Stage 12 sem CLI canonico**: `RunBaselinesUseCase` foi
  mergeada em PR #30 sem `main_run_baselines.py` ou equivalente.
  Stage 12 review aceitou entrypoint opcional; smoke evidenciou que e
  obrigatorio para reprodutibilidade publica.

- **Gap 2 — TFT vs baselines `target_timestamp_utc` misaligned por
  warmup**: TFT encoder warmup 60d reduz amostra (685/437 timestamps em
  test/val); baselines sem warmup (751/503). Gate
  `oos_pairwise_target_alignment` falhou; `gold_dm_pairwise_results` e
  `gold_mcs_results` ficaram vazios para o par TFT-vs-baseline. Stage 12
  review (Gap 4 §7 do prompt) antecipou o risco mas nao incluiu teste
  cross-stage exigindo intersecao exata.

- **Gap 3 — Bug pre-existente em `main_refresh_analytics_store --help`**:
  `%` nao escapado em help strings (linhas 145, 174) levanta
  `ValueError: unsupported format character ')'`. Bug fora de qualquer
  Stage; introduzido em refresh CLI pre-Stage 7. So afetava `--help`,
  nao execucao normal — escapou de todas as revisoes anteriores.

- **Gap 4 — `gold_confidence_calibrated_by_horizon` falsa-positivo em
  point baselines**: Stage 12 introduziu baselines com `prediction_mode
  ='point'` (zero_return, historical_mean_rolling) que produzem
  `confidence_calibrated=NaN` por design (sem metricas probabilisticas).
  O gate contava esses NaN como `bad_confidence`. F.1 smoke registrou
  `bad_confidence=8` correspondendo exatamente a 2 point baselines × 2
  splits × 2 horizons. Causa fora do alinhamento warmup; gap distinto
  dos Gap 1-2.

- **Gap 5 — `official_contract_quantile_attention` exigia artefato torch
  de baselines**: Stage 10 introduziu `fact_model_artifacts` como FK
  obrigatoria para runs official. Stage 12 baselines nao tem checkpoint,
  feature_importance ou attention — por design. Stage 12 review nao
  endereçou contrato cross-Stage 10. F.1 smoke registrou
  `missing_model_artifacts=3`, exatamente os 3 baselines.

- **Gap 6 — TFT y_true convention bug em decisoes boundary (descoberto
  via re-run do F.1 smoke pos-Stage F.0)**: TFT trainer emite
  `y_true_m[i][h_idx]` que empiricamente retorna `target_return[i]`
  (return no decision_ts) ao inves de `target_return[i + h - 1]`
  (return h-1 dias no futuro). Isso fica visivel nas decisoes
  boundary de h>=2 onde `i + h - 1 >= dataset_len` — TFT emite a
  predicao mesmo assim com y_true do dia atual, contrariando o
  comentario do proprio Stage 12 que afirma "actuals_matrix[i, h-1]
  is the target_return at the future step (h-1 ahead)". Re-run do
  smoke 2026-05-18 v2 com Stage F.0.2 alinhamento mostrou TFT
  emitindo 685 timestamps para test/h=7 (incluindo decisoes onde
  target_ts > dataset_end), enquanto baselines emitiram 679
  (corretamente filtrando rows sem y_true real). Symdiff=6 nas
  ultimas 6 decisoes do test split. Resultado: pre-existing bug em
  Stage 11 trainer ou pytorch_forecasting TimeSeriesDataSet —
  necessita Stage proprio para investigar e corrigir antes de F.1
  poder PASS pleno.

Esses **6 gaps** (5 originais + Gap 6 descoberto durante re-run F.0.5)
**nao invalidam o closure mapping** dos Stages 1-14 em si — todos os
P0 transversais permanecem fechados, todos RED zerados. Sao gaps **na
cobertura dos reviews**, nao nos Stages mergeados. Gap 6 e o unico
que aponta para bug latente em Stage 11 trainer; demais sao de
cobertura cross-Stage (Stage 12 ↔ Stages 9/10) e bug pre-existente em
CLI. Plano de fix em
[`POST_CLOSURE_FIXES_CHECKLIST.md` Stage F.0](../../05_checklists/POST_CLOSURE_FIXES_CHECKLIST.md)
(migrado de PHASE_B_IMPLEMENTATION_CHECKLIST.md em 2026-05-19);
Gap 6 promovido a follow-up bloqueante de F.1 PASS pleno (proximo
Stage de codigo, possivelmente Stage 15 ou Stage F.0.10).

Evidencia completa do smoke em
[`docs/07_reports/smoke_confirmatory_2026-05-18.md`](../smoke_confirmatory_2026-05-18.md).
Re-rodada do smoke pos-Stage F.0 (data: 2026-05-18 re-run) anexa
secao final nesse mesmo relatorio.

## Status final (2026-05-22, Stage R-23)

Stage F.0 mergeado (PR #40). Stage F.A registro mergeado (PRs #41-#43).
O ciclo de remediacao pos-auditoria tambem fechou os REDs remanescentes:

- Stage R-20 (#50, mergeado 2026-05-21): Gap 6 fechado pela Opcao d
  (`target_return` backward-shift + baseline `y_true_idx` ajustado).
- Stage R-E (#51, mergeado 2026-05-21): regressao do suffix fix isolada.
- Stage R-21 (#52, mergeado 2026-05-21): 27 quality checks migrados para
  registry real.
- Stage R-22 (#53, mergeado 2026-05-22): 25 gold builders migrados; refresh
  reduzido para 300 LOC.
- Stage R-23 (este PR): F.1 v4 revalidado com `max_epochs=5` e 6/6 criterios
  PASS. Apos merge manual por Marcelo, a substancia F.1 fica fechada.
  Reporte: [`smoke_confirmatory_2026-05-18.md`](../smoke_confirmatory_2026-05-18.md)
  §"F.1 v4 PASS pleno (Stage R-23, 2026-05-22)".

**F.2 pre-registro:** ainda pendente; documento separado em
`docs/06_pre_registration/`. **F.3 Gate de saida Fase A:** awaiting F.2.
Fase B tecnicamente pronta para abrir formalmente apos merge do Stage R-23
e conclusao F.2.
