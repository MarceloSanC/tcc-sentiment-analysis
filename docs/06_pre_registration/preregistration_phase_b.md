---
title: Pre-registration Phase B (TFT all-features confirmatorio, AAPL)
scope: Pre-registro operacional versionado da Fase B confirmatoria. Fixa, antes da execucao, hipoteses (H1/H2a/H2b), candidato declarado (hiperparametros, features, semente), contrato quantilico primario, baselines, coorte, protocolo estatistico, bandas de tolerancia, gates de elegibilidade tier 1/tier 2, horizontes oficiais e itens out-of-scope. F.2 do PHASE_B_IMPLEMENTATION_CHECKLIST. Nao duplica conteudo dos canonicos; sintetiza referenciando.
update_when:
  - emenda datada for adicionada (qualquer alteracao pos-merge requer entrada na secao "Emendas")
  - rodada confirmatoria for executada (apenas para anexar commit hash da execucao na secao "Execucao")
canonical_for: [phase_b_preregistration]
---

# Pre-registration Phase B (TFT all-features confirmatorio, AAPL)

**Status:** PROVISORIO (vigora apos merge desta PR; selado para a Fase B
confirmatoria a partir desse commit hash).
**Criado:** 2026-05-23.
**Bloqueante para:** abertura formal da Fase B (treino confirmatorio) e
fechamento de F.3 (gate de saida da Fase A).
**Supersede:** rascunho `preregistration_round1_<date>.md` mencionado em
[`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §A.3
(local de destino atualizado para `docs/06_pre_registration/`).

Este documento operacionaliza a Fase B per
[`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §5
(Fase B) e fecha as 7 pendencias F.2 listadas em
[`A_audit_closure_2026-05-17.md`](../07_reports/phase-gates/A_audit_closure_2026-05-17.md)
§"Pendencias para F.2". Decisoes nao redeclaradas aqui herdam dos
documentos canonicos referenciados.

---

## 1. Pre-condicoes verificadas

- F.1 smoke confirmatorio **PASS pleno** (Stage R-23, 2026-05-22; 6/6
  criterios). Evidencia em
  [`smoke_confirmatory_2026-05-18.md`](../07_reports/smoke_confirmatory_2026-05-18.md)
  §"F.1 v4 PASS pleno (Stage R-23, 2026-05-22)".
- Audit closure: **todos os RED zerados**, P0 transversais marcados `[x]`
  (per
  [`A_audit_closure_2026-05-17.md`](../07_reports/phase-gates/A_audit_closure_2026-05-17.md)
  §"Status final (2026-05-22, Stage R-23)").
- Stages 1-14 + 20-23 + R-20..R-23 mergeados em main (PRs #14-#54 + #56
  cleanup); decisao Caminho B registrada em
  [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md)
  §"Decisao Caminho B vs C".
- Archive pre-Phase B preservado em
  `data/analytics_archive_pre_phase_b/` (851 MB read-only) conforme
  [`ANALYTICS_STORE_ARCHITECTURE.md`](../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md).
  Dados anteriores a 2026-05-10 sao **diagnostico historico**, nao
  evidencia confirmatoria.
- Convencao `target_timestamp_utc` / `y_true` materializada por
  `MultiHorizonPredictionPersister` (Opcao d, Stage R-20) per
  [ADR-0003](../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md).

---

## 2. Escopo e ativo

- **Ativo piloto unico:** AAPL. Per
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §2 e
  §6.5: claim limitado ao ativo avaliado sob este protocolo. Multi-asset
  e *future work*, **out-of-scope Phase B**.
- **Dataset:** `data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet`
  pos-Stage 14 (`fundamentals_effective_date` presente; `effective_date`
  removido) e pos-Stage R-20 (formula de `target_return` backward shift,
  per ADR-0003 §"Historico do item 3").
- **Periodo OOS:**
  - train: `2010-10-18 -> 2020-12-31`
  - val:   `2021-01-01 -> 2022-12-31`
  - test:  `2023-01-01 -> 2025-12-31`
- **Walk-forward:** desativado para a Fase B confirmatoria. Folds de
  walk-forward foram avaliados como suplementar em B.4 e ficam como
  evidencia secundaria.

---

## 3. Hipoteses operacionais

Reproduzem [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
§7 sem reformulacao; aqui apenas fixadas para a rodada confirmatoria.

- **H1 (calibracao):** o candidato TFT all-features produz, para pelo
  menos um `h in {1, 7}`, calibracao marginal por quantil dentro das
  bandas declaradas (§9) **e** PICP intervalar p10-p90 dentro da banda
  declarada (§9).
- **H2a (primaria, TFT vs baseline primario):** para pelo menos um
  `h in {1, 7}`, o candidato apresenta `mean_pinball_post_guardrail`
  menor que o do baseline primario declarado (§6), com significancia
  estatistica DM Holm-corrigida (§7) e magnitude minima Δpinball
  relativa (§9).
- **H2b (evidencia forte adicional, TFT vs todos baselines):** para
  pelo menos um `h in {1, 7}`, o candidato apresenta
  `mean_pinball_post_guardrail` menor que o de **todos** os baselines
  declarados (§5), com significancia DM Holm-corrigida (§7). H2b usa
  como referencia probabilistica `historical_quantiles_rolling`
  (Stage 12 MVP).
- **H3 (contribuicao de familias):** **out-of-scope Phase B**. Fica
  declarada para a **Fase C** explicativa per
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §5
  e [`EXPLAINABILITY.md`](../04_evaluation/EXPLAINABILITY.md). H3 nao
  e pre-registrada aqui.

**Resultados de refutacao sao validos** e nao invalidam o trabalho per
[`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
§4.4 "Resultados validos mesmo com hipoteses refutadas".

---

## 4. Candidato declarado (M1.3)

**Selecao do candidato unico** segue o pivo metodologico em
[`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §4 e
[`SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md):
**1 candidato all-features com poda minima**, sem comparacao entre
feature sets.

### 4.1 Features (poda minima por principio)

Per [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
§4.2:

- **Familias incluidas:** PRICE (OHLCV), TECHNICAL, SENTIMENT,
  FUNDAMENTAL.
- **Poda a priori** (regras congeladas; aplicadas antes do experimento
  e sem inspecao de RMSE/pinball OOS):
  1. Correlacao par-a-par `> 0,95`: remover uma (criterio
     `keep_first_in_alphabetic_order`).
  2. Missing em treino `> 30%`: remover.
  3. Derivadas redundantes da mesma quantidade (ex.: `MA_5` vs
     `EMA_5`): manter a primeira em ordem alfabetica, documentar par
     descartado.
  4. Sem disponibilidade temporal comprovada no momento da inferencia:
     remover (vetado por `time_varying_known_reals` audit M3-Q3).
- **Lista final `feature_list_ordered`** (M3.1):
  congelada **no momento da execucao confirmatoria** (preserva
  ordenacao deterministica para reproducibilidade); persistida em
  `data/analytics/silver/dim_run.feature_list` do run confirmatorio.

### 4.2 Hiperparametros (M1.3 — heranca)

**Procedimento de fixacao** (operacional, sem leakage de test):

1. Ponto de partida: top-3 frozen candidates da Round 0 ordenados por
   `robust_score = mean_val_rmse + lambda * std_val_rmse` (objetivo
   nativo Optuna), conforme
   `data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv`
   (top-1 atual: `hidden_size=32, attention_head_size=4, dropout=0.15,
   lr=4.85e-4, batch_size=64, hidden_continuous_size=4`).
2. **Rodada Optuna pequena overnight (~20-30 trials)** sobre vizinhanca
   desses 3 com `max_encoder_length=60, max_prediction_length=7,
   evaluation_horizons=[1,7], prediction_mode=quantile,
   quantile_levels=[0.1, 0.5, 0.9]` (regime multi-horizonte da Phase B,
   diferente do regime `max_prediction_length=1` da Round 0). Esta
   rodada e exploratoria — nao alimenta claims do paper.
3. **Objetivo Optuna:** `robust_score` apenas (Stage 13 bloqueia
   `mean_test_rmse` e `joint_val_test_rmse` por construcao em
   `RunTFTOptunaSearchUseCase.__init__`).
4. Top-1 da rodada vira **config confirmatoria congelada**, persistida
   em `config/sweeps/explicit/phase_b_confirmatorio.json` com
   `parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>`. Hash do JSON
   referenciado nesta secao via **emenda datada** apos a fixacao.
5. Justificativa academica per audit M1.3: hiperparametros foram
   escolhidos por criterio de **validacao** (`robust_score`), nunca
   por `mean_test_rmse` ou `joint_val_test_rmse` (auditavel por codigo:
   Stage 13 bloqueia esses objetivos).

**Hiperparametros provisorios (smoke F.1 v4 R-23, referencia operacional
sem validade confirmatoria):**

| parametro                     | smoke F.1 v4         |
| ----------------------------- | -------------------- |
| `max_encoder_length`          | 60                   |
| `max_prediction_length`       | 7                    |
| `evaluation_horizons`         | `[1, 7]`             |
| `prediction_mode`             | `quantile`           |
| `quantile_levels`             | `[0.1, 0.5, 0.9]`    |
| `batch_size`                  | 64                   |
| `learning_rate`               | 5e-4                 |
| `hidden_size`                 | 32                   |
| `attention_head_size`         | 2                    |
| `dropout`                     | 0.1                  |
| `hidden_continuous_size`      | 8                    |
| `max_epochs` (smoke)          | 5 (smoke only)       |
| `early_stopping_patience`     | 5                    |
| `early_stopping_min_delta`    | 0.0                  |
| `warmup_policy`               | `drop_leading`       |
| `seed`                        | `20260517`           |

`max_epochs` confirmatorio sera fixado pela rodada Optuna overnight
(esperado: `30-50` com early stopping em val_loss).

### 4.3 Semente, folds, seeds replica

- **`seed_base`:** `20260517` (mesma do F.1 v4, ja registrada em
  metadados de dim_run; auditavel).
- **Numero de seeds replica:** **5** per
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
  §B.1. Seeds derivadas:
  `[20260517, 20260518, 20260519, 20260520, 20260521]`.
- **Folds walk-forward:** **3** per
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
  §B.1. Walk-forward folds especificos serao listados na config
  congelada (4.2 passo 4) e auditados via
  `data/analytics/silver/dim_run.fold`.
- **Total runs do candidato:** `5 seeds × 3 folds = 15` runs do
  candidato unico.

### 4.4 Politica de warmup (M3.1)

Per audit M3 Q2-Q3:

- **`warmup_policy = drop_leading`** (mesmo modo do F.1 v4 PASS pleno).
- `effective_train_start`, `required_warmup_count`, `warmup_features`
  e `warmup_applied` sao registrados em `dim_run.metadata` por run.
- Justificativa: `strict_fail` bloquearia o treino sem evidencia
  empirica de prejuizo; `drop_leading` ajusta `train_start` para a
  primeira data valida sem alterar split de val/test (auditavel via
  `train_tft_model_use_case._apply_warmup_policy_to_split`).

### 4.5 Auditoria source-level AAPL (M3.2)

Cobertura da auditoria de disponibilidade temporal real **para a
rodada confirmatoria**:

- **Fundamentals:** confirmar `reported_date` ou fallback `+45 dias`
  apos `fiscal_date_end` per
  [`DATA_SOURCES.md`](../02_data/DATA_SOURCES.md) §"Fallback de
  reported_date" (cobertura 17/81 reports AAPL com fallback aplicado,
  21%). Coluna `fundamentals_effective_date` materializada no dataset
  (Stage 14) preserva rastreabilidade direta.
- **Sentimento:** confirmar `daily_sentiment.n_articles` consistente
  com `scored_news.published_at` sob politica
  `open_hour=09:30, close_hour=16:00, weekends=false` (audit M3-Q4
  confirmou consistencia para dias uteis; 269 divergencias em fim de
  semana nao entram no dataset diario).
- **Anti-leakage:** `known_real_cols` restritos a
  `["time_idx", "day_of_week", "month"]` (audit M1-Q1, M3-Q3); preco,
  retorno, tecnicos, sentiment, fundamentals **vetados** em
  `time_varying_known_reals` por construcao do trainer.
- **Artefato consolidado:** anexar via emenda datada apos a execucao,
  citando `notebooks/aapl_source_level_audit_phase_b.{ipynb,html}` ou
  equivalente.

---

## 5. Baselines pre-declarados (M7-Q2)

Per [`BASELINES.md`](../04_evaluation/BASELINES.md) §"Baselines
Recomendados". Lista minima admitida para Phase B confirmatoria
(Stage 12 MVP):

| ID                              | Tipo            | Configuracao         | Modo            |
| ------------------------------- | --------------- | -------------------- | --------------- |
| `zero_return`                   | random walk     | —                    | `point`         |
| `historical_mean_rolling`       | media rolling   | `window_days=30`     | `point`         |
| `historical_quantiles_rolling`  | quantis empiric | `window_days=252`    | `quantile`      |

- **Baseline primario para H2a:** `zero_return`. Justificativa: random
  walk de retorno e o null model canonico em forecasting financeiro
  (default sugerido em
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
  §A.2); ja persistido com mesma convencao temporal do TFT via
  `MultiHorizonPredictionPersister` (Stage R-20).
- **Baseline probabilistico para H2b:** `historical_quantiles_rolling`
  (unico do MVP que emite quantis genuinos; modo `quantile`).
  Comparacao pinball post-guardrail vs TFT post-guardrail.
- **Baselines adicionais (`random_walk` distinto de `zero_return`,
  `AR(1)`, `EWMA-vol`) listados em
  [`BASELINES.md`](../04_evaluation/BASELINES.md) §"Baselines
  Recomendados" sao OUT-OF-SCOPE Phase B.** Sua adicao em rodadas
  futuras exige Stage 12-bis ou emenda datada.

**Requisitos de comparabilidade** (per
[`BASELINES.md`](../04_evaluation/BASELINES.md) §"Requisitos De
Comparabilidade"):

- Mesmo `parent_sweep_id` do candidato (config sweep JSON unico
  agrupa TFT + baselines per
  [`SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md)
  §"Schema da secao baselines").
- Mesmas seeds e folds.
- Mesmo grao em `fact_oos_predictions` (per
  [`MULTI_HORIZON.md`](../03_modeling/MULTI_HORIZON.md) §"Persistencia").
- Alinhamento OOS estrito por
  `(asset, split, horizon, target_timestamp)`.

---

## 6. Contrato quantilico primario (M5/M6/M7.5)

Per [`METRICS_DEFINITIONS.md`](../04_evaluation/METRICS_DEFINITIONS.md)
§"Variante quantilica" (categorizacao A/B/C) e justificativa academica
em [`20_method.md`](../07_reports/living-paper/20_method.md)
§"Politica de variante quantilica":

- **Primario para H1, H2a, H2b:** `post_guardrail`. Default tecnico
  ja persistido via flag `--primary-quantile-contract post_guardrail`
  em `RefreshAnalyticsStoreUseCase` (Stage 8.3); registrado como coluna
  `primary_quantile_contract` no output de `gold_model_decision_final`.
- **Raw persistido em paralelo** para auditoria e calculo do delta
  (`delta_<metrica>_post_minus_raw`); nao alimenta claims primarios.
- **Tabelas Categoria C** (`_pred_interval_negative`, gate de
  degeneracao Stage 11) permanecem raw apenas por construcao.
- **Justificativa academica:** rearranjo monotono e provadamente nao
  prejudicial sob perda convexa simetrica (Chernozhukov, Fernandez-Val
  & Galichon 2010, *Econometrica*); pinball pos-guardrail e a perda
  do estimador pre-registrado, em conformidade com Gneiting & Raftery
  (2007, *JASA*).

---

## 7. Protocolo estatistico

Per [`STATISTICAL_TESTS.md`](../04_evaluation/STATISTICAL_TESTS.md).

- **Alinhamento OOS estrito:** por
  `(asset, split, horizon, target_timestamp)`. Intersecao exata via
  `gold_paired_oos_intersection_by_horizon`. Sem intersecao exata, o
  par TFT-vs-baseline e excluido (Stage 12 lesson learned do smoke
  F.1 v1 →
  [`smoke_confirmatory_2026-05-18.md`](../07_reports/smoke_confirmatory_2026-05-18.md)
  §6.a; F.1 v4 PASS pleno confirmou alinhamento exato em todas as
  combinacoes).
- **Diebold-Mariano (DM)** pareado com correcao HAC e ajuste
  Harvey-Leybourne-Newbold (HLN) para amostras pequenas (1995, 1997);
  computado em `gold_dm_pairwise_results` per coorte.
- **Model Confidence Set (MCS)** ao nivel `alpha = 0.05`
  (Hansen-Lunde-Nason 2011), computado em `gold_mcs_results`.
- **Win-rate pareado** computado em `gold_win_rate_pairwise_results`
  (suplementar).
- **Coorte:** **scope unico** `parent_sweep_id =
  phase_b_confirmatorio_<YYYYMMDD>` (fixado na emenda da config
  congelada, §4.2 passo 4). Refresh executado com
  `scope_mode=cohort_decision`, `scope_sweep_prefixes=[phase_b_confirmatorio_]`,
  `primary_quantile_contract=post_guardrail`,
  `--fail-on-quality` (mesma sequencia do F.1 v4 R-23).
- **Alpha primario:** `0.05`.
- **Multiple testing:** **familia unica** Holm-Bonferroni (Holm 1979)
  sobre o conjunto `{TFT vs baseline_i | i in {zero_return,
  historical_mean_rolling, historical_quantiles_rolling}, h in {1, 7}}`
  = `2 horizontes × 3 baselines = 6 testes`. Resultado significativo
  exige `pvalue_adj_holm < 0.05`. Justificativa: familia unica e
  mais conservadora; alinha com
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
  §B.2 "Holm-Bonferroni sobre o conjunto de DMs reportados".

---

## 8. Horizontes oficiais

Per [`MULTI_HORIZON.md`](../03_modeling/MULTI_HORIZON.md) §"Decisoes e
escolhas":

- **Horizontes confirmatorios:** `h = 1` (curto prazo) e `h = 7`
  (medio prazo). Ambos primarios para H1/H2a/H2b.
- **`h = 30` OUT-OF-SCOPE Phase B.** Justificativa: audit M7 acao 9
  ja recomendou exclusao do smoke inicial; N efetivo nao-sobreposto
  ~25 para 3 anos OOS, poder estatistico DM/MCS muito baixo per
  [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
  §3. Permanece como *future work* documentado.
- **Reportagem por horizonte separada**, nunca agregada
  (per [`MULTI_HORIZON.md`](../03_modeling/MULTI_HORIZON.md)).
- Inferencia rolling multi-horizonte (engine, nao trainer) e
  out-of-scope per audit M4 acao 1: a Fase B usa
  `fact_oos_predictions` gerado pelo trainer, nao pelo engine de
  inferencia. ADR-0003 cobre a convencao formalmente.

---

## 9. Bandas de tolerancia e gates tier 1 / tier 2

Per [`CALIBRATION_AND_RISK.md`](../04_evaluation/CALIBRATION_AND_RISK.md)
§"Calibration Acceptance Thresholds". Politica em **dois tiers**
declarados ex-ante para preservar honestidade do trabalho mesmo sob
poder estatistico baixo.

### Tier 1 — Promocao a evidencia confirmatoria primaria

Candidato eleito para suportar H1/H2a/H2b em texto principal **se e
somente se** todos os criterios abaixo forem satisfeitos para o
horizonte avaliado, sobre a variante `post_guardrail`:

- **Calibracao marginal por quantil:**
  - `coverage_q10 in [0.07, 0.13]`
  - `coverage_q50 in [0.45, 0.55]`
  - `coverage_q90 in [0.87, 0.93]`
- **PICP intervalar:** `abs(coverage_error_post_guardrail) <= 0.02`
  (target = `0.80`; banda `[0.78, 0.82]`).
- **DM significancia (H2a/H2b):** `pvalue_adj_holm < 0.05` (familia
  unica, §7).
- **Relevancia pratica:** `delta_mean_pinball_rel >= 3%`
  ((`mean_pinball_baseline - mean_pinball_TFT`) /
  `mean_pinball_baseline` por horizonte).
- **MPIW:** `mpiw_post_guardrail > 0` e finito (garantido por
  construcao pos-Stage 11 gate).

### Tier 2 — Evidencia secundaria

Aplicada **somente se nenhum candidato Tier 1 emergir** apos a rodada
confirmatoria. Reporta o resultado com label explicito **"evidencia
secundaria / nao confirmatoria"** no texto, mantendo o trabalho
publicavel sob limite empirico declarado per
[`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §4.4
("Resultados validos mesmo com hipoteses refutadas"). Criterios
permissivos:

- **Calibracao marginal:**
  - `coverage_q10 in [0.05, 0.15]`
  - `coverage_q50 in [0.40, 0.60]`
  - `coverage_q90 in [0.85, 0.95]`
- **PICP:** `abs(coverage_error_post_guardrail) <= 0.05`
  (banda `[0.75, 0.85]`).
- **DM:** `pvalue_adj_holm < 0.10` (suplementar).
- **Relevancia pratica:** `delta_mean_pinball_rel >= 0%` (qualquer
  vantagem direcional, mesmo nao significativa).

**Politica de promocao Tier 1 → Tier 2:** declarada **ex-ante** neste
pre-registro para evitar cherry-picking pos-observacao. A escolha do
tier reportado nao pode ser feita apos inspecao dos resultados. Se
nenhum tier for satisfeito, reportar resultado como **"hipotese
refutada / TFT nao supera baselines neste protocolo"** e direcionar
trabalho para diagnostico de calibracao, conforme
[`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §3
e §4.4.

### Gate de degeneracao quantilica (M2 Stage 11)

Independente do tier, candidato com `% rows com p10 == p90 >= 5%` no
OOS post-pos guardrail (raw, ja que e Categoria C) e
**automaticamente excluido** de qualquer claim probabilistico per
Stage 11 gate (`block_quantile_degeneracy_gate` em
`ValidateAnalyticsQualityUseCase`). F.1 v4 R-23 PASS pleno verificou
`p10_eq_p90_rate=0.0` em 6/6 grupos quantile do smoke; espera-se
mesma propriedade na Phase B confirmatoria (5 seeds × 3 folds × 2
horizontes = 30 grupos).

---

## 10. Politica de coorte e ScopeSpec (M6)

Per [`SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md):

- **Identificador unico de coorte:**
  `parent_sweep_id = phase_b_confirmatorio_<YYYYMMDD>`
  (`<YYYYMMDD>` = data de execucao; congelado na emenda da config).
- **ScopeSpec da decisao:** `scope_mode=cohort_decision`,
  `scope_sweep_prefixes=["phase_b_confirmatorio_"]`,
  `scope_splits=["val", "test"]`, `scope_horizons=[1, 7]`.
- **Cohort-aware gold:** Stages 4-6 garantem `parent_sweep_id` em
  todas as 10 tabelas gold relevantes; rankings em
  `gold_model_decision_final` e `gold_ranking_by_config` ja sao
  segregados por coorte (Stage 4).
- **Filtro reproduzivel:** todo claim do paper deve citar a coorte
  exata e o commit hash desta pre-registracao na execucao. Refresh
  reproduzivel por:

  ```bash
  python -m src.main_refresh_analytics_store \
      --scope-mode cohort_decision \
      --scope-sweep-prefixes phase_b_confirmatorio_ \
      --scope-splits val test \
      --scope-horizons 1 7 \
      --primary-quantile-contract post_guardrail \
      --fail-on-quality
  ```

---

## 11. Tipo de sweep e operacao

Per [`SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md)
§"Schema da secao baselines":

- **`test_type=explicit_configs`** (sweep declarativo via JSON
  versionado em `config/sweeps/explicit/phase_b_confirmatorio.json`).
- Pipeline TFT (`main_tft_test_pipeline`) + baseline pipeline
  (`main_baselines_test_pipeline`, Stage F.0.2) consomem o **mesmo
  JSON** garantindo `parent_sweep_id`, folds, seeds e
  `evaluation_start_offset_days` alinhados (per Stage F.0.2 decisao
  warmup auto-derivado de `max_encoder_length`).
- **OFAT (`main_tft_param_sweep`)** **nao usado** para confirmatorio
  per decisao deste pre-registro (D13 do bloco de decisoes).
- **Nota operacional (R-23 workaround):** observado durante F.1 v4 que
  baselines podem derivar `parent_sweep_id` distinto se o JSON nao
  for o **mesmo arquivo fisico** consumido por ambos pipelines.
  Workaround atual: rodar ambos pipelines apontando para o mesmo
  arquivo `phase_b_confirmatorio.json` (sem copia /tmp). Esta
  fragilidade fica registrada como **divida tecnica YELLOW
  pos-Phase B** (follow-up: derivar `parent_sweep_id`
  automaticamente do `output_subdir` em ambos pipelines em PR
  separado).

---

## 12. Out-of-scope Phase B (lista oficial)

Itens declarados out-of-scope per audit closure §"Out-of-scope Phase B
/ Fase C / future work" e este pre-registro:

| Item                                                       | Destino           |
| ---------------------------------------------------------- | ----------------- |
| Multi-asset (alem de AAPL)                                 | Future work       |
| Persistencia multi-horizonte em engine de inferencia       | Fase C ou future  |
| Inferencia rolling como evidencia primaria                 | Fase C            |
| Explicabilidade local (`local_magnitude_signed_v1`)        | Fase C            |
| `evidence_tier` para diferenciacao primary vs illustrative | Fase C            |
| Auditoria `fact_inference_*` cohort-aware                  | Fase C            |
| `random_walk` distinto de `zero_return`, `AR(1)`, `EWMA-vol` | Future / Stage 12-bis |
| `h = 30` como horizonte oficial                            | Future work       |
| H3 (contribuicao de familias VSN/permutation/ablation)     | Fase C            |
| Walk-forward folds adicionais alem dos 3 declarados        | Sensibilidade B.4 |

Walk-forward sensibilidade (B.4 do `STRATEGIC_DIRECTION` §B): 2
janelas alternativas (curta, longa) **com o mesmo candidato unico
congelado**, sem reotimizacao por janela. Resultado suplementar.

---

## 13. Artefatos esperados da execucao

Toda execucao confirmatoria deve produzir, **no minimo**:

- `data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/`
  com 15 runs TFT + 3 baselines × 15 = 45 runs baseline.
- `data/analytics/silver/fact_oos_predictions/asset=AAPL/...` com
  cobertura completa de `(run_id, split, horizon)` per
  [`MULTI_HORIZON.md`](../03_modeling/MULTI_HORIZON.md)
  §"Validacoes obrigatorias".
- `data/analytics/gold/gold_prediction_metrics_by_run_split_horizon.parquet`
  com colunas `*_raw` e `*_post_guardrail` per
  [`METRICS_DEFINITIONS.md`](../04_evaluation/METRICS_DEFINITIONS.md)
  §"Variante quantilica".
- `data/analytics/gold/gold_dm_pairwise_results.parquet`,
  `gold_mcs_results.parquet`,
  `gold_win_rate_pairwise_results.parquet`,
  `gold_paired_oos_intersection_by_horizon.parquet`,
  `gold_quality_statistics_report.parquet`,
  `gold_model_decision_final.parquet`.
- Quality gate: `failed_checks=[]` (mesmo padrao R-23).
- Reporte de tier (1 ou 2) por horizonte declarado em §"Execucao"
  apos a rodada.

---

## 14. Politica de emendas

Mudancas a este pre-registro **apos merge** desta PR exigem:

- Entrada datada na secao "Emendas" abaixo, com `<YYYY-MM-DD>` e
  justificativa.
- Identificacao explicita de qual decisao §1-§13 e alterada.
- Cross-link ao commit/PR que motivou a emenda.
- Aprovacao via PR review (mesmo padrao governance do projeto).

**Mudancas vetadas apos execucao** (sem emenda admitida):

- Trocar metrica primaria (§9, Tier 1) por outra para reportar
  resultado mais favoravel.
- Trocar baseline primario (§5) apos observar pinball.
- Mover criterios de Tier 2 para Tier 1 ou vice-versa apos observar
  resultados.
- Adicionar/remover baselines apos execucao (exige Stage 12-bis +
  novo pre-registro).
- Alterar `parent_sweep_id` apos persistencia silver.

### Emendas

_(vazio na criacao; preenchido conforme necessario)_

---

## 15. Execucao

_(preenchido apenas apos rodada confirmatoria)_

- **Commit hash deste pre-registro (selo):** `<a preencher apos merge>`
- **Data de execucao confirmatoria:** `<YYYY-MM-DD>`
- **`parent_sweep_id` efetivo:** `phase_b_confirmatorio_<YYYYMMDD>`
- **Hash da config JSON:** `<sha256 de
  config/sweeps/explicit/phase_b_confirmatorio.json>`
- **Resultado tier (H1, H2a, H2b por horizonte):** `<reportar tier 1
  / tier 2 / refutado>`
- **Relatorio:** `docs/07_reports/phase-gates/B_confirmatory_<YYYY-MM-DD>.md`

---

## 16. Cross-links canonicos (nao duplicar — referenciar)

| Tema                            | Documento canonico                                                                                     |
| ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Direcao estrategica / hipoteses | [`docs/00_overview/STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §2, §4, §5, §6, §7 |
| Convencao multi-horizonte       | [`docs/03_modeling/MULTI_HORIZON.md`](../03_modeling/MULTI_HORIZON.md)                                |
| Politica de sweep / coorte      | [`docs/03_modeling/SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md)                  |
| Formulas / categoria A/B/C      | [`docs/04_evaluation/METRICS_DEFINITIONS.md`](../04_evaluation/METRICS_DEFINITIONS.md)                |
| DM / MCS / Holm                 | [`docs/04_evaluation/STATISTICAL_TESTS.md`](../04_evaluation/STATISTICAL_TESTS.md)                    |
| Contrato de baselines           | [`docs/04_evaluation/BASELINES.md`](../04_evaluation/BASELINES.md)                                    |
| Bandas de calibracao / risco    | [`docs/04_evaluation/CALIBRATION_AND_RISK.md`](../04_evaluation/CALIBRATION_AND_RISK.md)              |
| Convencao target_timestamp      | [`docs/01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md`](../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md) |
| Politica variante quantilica    | [`docs/07_reports/living-paper/20_method.md`](../07_reports/living-paper/20_method.md) §"Politica de variante quantilica" |
| Audit gate da Fase A            | [`docs/07_reports/phase-gates/A_code_audit.md`](../07_reports/phase-gates/A_code_audit.md)            |
| Audit closure mapping           | [`docs/07_reports/phase-gates/A_audit_closure_2026-05-17.md`](../07_reports/phase-gates/A_audit_closure_2026-05-17.md) |
| Smoke F.1 v4 evidencia          | [`docs/07_reports/smoke_confirmatory_2026-05-18.md`](../07_reports/smoke_confirmatory_2026-05-18.md) §"F.1 v4 PASS pleno" |
| Checklist Phase B / F.2 / F.3   | [`docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md) §"Stage final" |
| Analytics Store + archive       | [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md) |

---

## 17. Referencias academicas

- Chernozhukov, V.; Fernandez-Val, I.; Galichon, A. (2010). *Quantile
  and Probability Curves Without Crossing*. Econometrica.
- Diebold, F. X.; Mariano, R. S. (1995). *Comparing predictive
  accuracy*. Journal of Business & Economic Statistics.
- Gneiting, T.; Raftery, A. E. (2007). *Strictly proper scoring
  rules, prediction, and estimation*. JASA.
- Gu, S.; Kelly, B.; Xiu, D. (2020). *Empirical Asset Pricing via
  Machine Learning*. Review of Financial Studies.
- Hansen, P. R.; Lunde, A.; Nason, J. M. (2011). *The Model
  Confidence Set*. Econometrica.
- Harvey, D.; Leybourne, S.; Newbold, P. (1997). *Testing the
  equality of prediction mean squared errors*. International Journal
  of Forecasting.
- Holm, S. (1979). *A simple sequentially rejective multiple test
  procedure*. Scandinavian Journal of Statistics.
- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing
  Financial Risk* (3rd ed.). McGraw-Hill.
- Acerbi, C.; Tasche, D. (2002). *On the coherence of expected
  shortfall*. Journal of Banking & Finance.
- Koenker, R.; Bassett, G. (1978). *Regression quantiles*.
  Econometrica.
- Lim, B.; Arik, S. O.; Loeff, N.; Pfister, T. (2021). *Temporal
  Fusion Transformers for interpretable multi-horizon time series
  forecasting*. International Journal of Forecasting.
- Pineau, J. et al. (2021). *Improving reproducibility in machine
  learning research*. JMLR.
