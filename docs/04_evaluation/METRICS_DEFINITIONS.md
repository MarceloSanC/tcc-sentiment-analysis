---
title: Metrics Definitions
scope: Formulas, agregacoes e implementacao canonica das metricas pontuais (RMSE, MAE, DA, bias) e probabilisticas (pinball, PICP, MPIW, cobertura por quantil), incluindo a politica de variante quantilica (raw vs post-guardrail) por tabela gold. Para semantica de termos, ver GLOSSARY. Para criterios de aceitacao, ver CALIBRATION_AND_RISK.
update_when:
  - nova metrica for adicionada ao pipeline
  - formula de metrica existente mudar (agregacao, ponderacao, escala)
  - mapeamento metrica -> coluna nas tabelas gold mudar
  - politica de variante raw vs post-guardrail (Secao "Variante quantilica") mudar
canonical_for: [metrics, metric_formulas, rmse, mae, pinball, picp, mpiw, coverage, quantile_variant_policy, raw_vs_post_guardrail]
---

# Metrics Definitions

Objetivo: definir formulas, agregacoes e implementacao canonica das metricas.
Este arquivo complementa o `GLOSSARY` (sem repetir semantica basica).

## Canonical Implementation
- `src/use_cases/refresh_analytics_store_use_case.py`
- `src/adapters/pytorch_forecasting_tft_trainer.py` (split metrics de treino)

## Row-Level Derived Columns
Para cada linha OOS (run, split, horizon, timestamp, target_timestamp):
- `error = y_pred - y_true`
- `abs_error = abs(error)`
- `sq_error = error^2`
- `pred_interval_width = quantile_p90 - quantile_p10`
- `covered_80 = 1{quantile_p10 <= y_true <= quantile_p90}`
- `da_row = 1{sign(y_pred) == sign(y_true)}`

## Core Metrics (aggregated)
Agregacao por (`run_id`, `split`, `horizon`) e derivados superiores.

- `rmse = sqrt(mean(sq_error))`
- `mae = mean(abs_error)`
- `mape = mean(abs(error / y_true))` (com protecao para `y_true=0`)
- `smape = mean(2*abs_error / (abs(y_true)+abs(y_pred)))`
- `directional_accuracy = mean(da_row)`
- `bias = mean(error)`

## Probabilistic Metrics
- `pinball_q = mean(max(q*(y_true-y_pred_q), (q-1)*(y_true-y_pred_q)))`
- `mean_pinball = mean(pinball_q10, pinball_q50, pinball_q90)`
- `coverage_q10 = mean(y_true <= quantile_p10)` (alvo nominal 0.10)
- `coverage_q50 = mean(y_true <= quantile_p50)` (alvo nominal 0.50)
- `coverage_q90 = mean(y_true <= quantile_p90)` (alvo nominal 0.90)
- `picp = mean(covered_80)`
- `mpiw = mean(pred_interval_width)`
- `coverage_nominal = 0.8` (intervalo p10-p90)
- `coverage_error = picp - coverage_nominal`

## Confidence Proxy
Usada no gold para leitura operacional de calibracao + sharpness:
- `calibration_term = clip(1 - abs(coverage_error)/coverage_nominal, 0, 1)`
- `width_term = 1 / (1 + clip(pred_interval_width, lower=0))`
- `confidence_calibrated = calibration_term * width_term`

## Risk Metrics (from predictions)
- `expected_move = mean(abs(y_pred))`
- `downside_risk = mean(max(-y_pred, 0))`
- `var_10 = mean(quantile_p10)`
- `es_10_approx = mean(min(1.125*quantile_p10 - 0.125*quantile_p50, quantile_p10))`

## Gold Tables (primary)
- `gold_prediction_metrics_by_run_split_horizon.parquet`
- `gold_prediction_metrics_by_config.parquet`
- `gold_prediction_metrics_by_horizon.parquet`
- `gold_prediction_calibration.parquet`
- `gold_prediction_risk.parquet`

## Validation Rules
- All metric columns must be numeric finite values where required.
- `pred_interval_width >= 0` (apos `QuantileGuardrailService.enforce_monotonic_triplet`; em raw pode ser negativo)
- `0 <= picp <= 1`
- `n_samples > 0` for valid aggregated rows
- no duplicated OOS keys in source predictions

## Variante quantilica: raw vs post-guardrail

Metricas probabilisticas podem ser calculadas sobre dois conjuntos de quantis
emitidos por `fact_oos_predictions`:

- **raw**: saida direta do modelo (`quantile_p10`, `quantile_p50`, `quantile_p90`),
  podendo violar monotonicidade (`p10 > p90`, MPIW negativo).
- **post-guardrail**: rearranjo monotono via
  `QuantileGuardrailService.enforce_monotonic_triplet` (`sorted([a,b,c])`),
  garantindo `p10 <= p50 <= p90` por construcao.

A escolha entre as duas **nao e cosmetica**: o smoke M7 do projeto evidenciou
diferencas materiais em `picp`, `mpiw`, `mean_pinball` e `coverage_error` em
4/4 linhas auditadas; `crossing_before_count=402`, `crossing_after_count=0`;
`gold_prediction_metrics_by_run_split_horizon` contem 4 linhas com `mpiw<0`
historicamente. Ver `docs/07_reports/phase-gates/phase-a/A_code_audit.md` §M5-Q1 para
o achado original.

### Categorias por tabela

Cada tabela gold cai em **uma** das tres categorias abaixo, conforme criterio:
a metrica precisa de ambas variantes se e somente se (a) existe pergunta
cientifica defensavel cuja resposta muda entre raw e post-guardrail E (b)
ambas respostas sao semanticamente validas (nao geram artefato matematico sem
interpretacao).

#### Categoria A — dual obrigatorio (raw + post-guardrail)

Tabelas onde ambas perguntas sao legitimas; raw mede o que o modelo aprendeu,
post-guardrail mede o que o sistema entrega. A delta (`*_post_guardrail -
*_raw`) e diagnostico do efeito do rearranjo.

| Tabela | Metricas duplicadas |
|---|---|
| `gold_prediction_metrics_by_run_split_horizon` (source) | `pinball_q10/q50/q90`, `mean_pinball`, `picp`, `mpiw`, `pred_interval_width`, `coverage_error`, `coverage_q10/q50/q90` (quando implementado), `prob_up/prob_down`, `confidence_calibrated` |
| `gold_prediction_metrics_by_config` | agregados `mean_*`/`std_*`/`iqr_*` das mesmas metricas |
| `gold_prediction_metrics_by_horizon` | idem |
| `gold_prediction_calibration` | projecao das mesmas metricas |
| `gold_prediction_generalization_gap` | `gap_*_test_minus_val` das mesmas metricas |
| `gold_prediction_robustness_by_horizon` | CIs sobre as mesmas metricas |

Metricas pontuais (`rmse`, `mae`, `mape`, `smape`, `directional_accuracy`,
`bias`, `expected_move`, `downside_risk`) **nao dependem de quantis** e nao
sao duplicadas em nenhuma tabela.

#### Categoria B — post-guardrail apenas (raw e matematicamente sem semantica)

| Tabela / metrica | Razao |
|---|---|
| `gold_prediction_risk` (`var_10`, `es_10_approx`) | VaR exige monotonicidade do funcao quantil por definicao (Jorion 2007; Acerbi & Tasche 2002). Sob crossing, `var_10_raw = mean(p10)` nao e quantil 10%, e o ES aproximado `1.125*p10 - 0.125*p50` assume `p10 <= p50` para interpolacao linear. Raw produz numero matematico sem interpretacao como risco. |
| `gold_model_decision_final` | Tabela de selecao/ranking. Por construcao, ha uma resposta unica por `(asset, horizon)`; a variante e declarada via flag `--primary-quantile-contract` (default `post_guardrail`) e persistida em coluna `primary_quantile_contract`. |
| Plots oficiais (`generate_prediction_analysis_plots_use_case.py`) | Artefatos operacionais que vao para o paper/dashboard. Variante unica seguindo o contrato primario do pre-registro. Plots de auditoria raw-vs-post sao apendice metodologico separado. |

#### Categoria C — raw apenas (post-guardrail e tautologicamente inutil)

| Tabela / check | Razao |
|---|---|
| `gold_oos_quality_report._pred_interval_negative` (em `refresh_analytics_store_use_case.py`) | Detecta `(p90 - p10) < 0`. Sob post-guardrail, por construcao do `enforce_monotonic_triplet`, sempre `>= 0`; a contagem seria tautologicamente zero e o check perderia funcao. |
| Gate de degeneracao quantilica (Stage 11, `% p10==p90` por run/horizon) | Detecta colapso do quantil emitido pelo modelo (modelo "desligou" os quantis externos). O sort do guardrail pode mascarar a degenerescencia herdada (ex.: modelo emite `(0.7, 0.5, 0.3)`, post-guardrail emite `(0.3, 0.5, 0.7)` — patologia some). O gate quer ver o que o modelo aprendeu, antes do mascaramento. |

#### Estrutura especial

| Tabela | Tratamento |
|---|---|
| `gold_quantile_guardrail_audit` | Por design, materializa `*_before` (raw), `*_after` (post-guardrail) e `delta_*` lado-a-lado. Nao duplica colunas — e a tabela de comparacao. |

### Contrato primario do paper

Default: **`post_guardrail`** para as hipoteses H1, H2a e H2b (calibracao,
pinball vs baselines). Justificativa metodologica completa em
`docs/07_reports/living-paper/20_method.md`. O pre-registro fixa o contrato
formalmente e pode ressalvar H3 (contribuicao de features) se a analise de
ablation usar raw para evidenciar efeito de features sobre o aprendizado
quantilico antes do mascaramento por rearranjo.

### Implementacao

- Sufixos canonicos: `<metrica>_raw` e `<metrica>_post_guardrail` em todas as
  tabelas Categoria A.
- Builders downstream propagam ambos os sufixos sem agregar entre variantes
  (nunca media de raw e post; sempre separados).
- Quando `fact_oos_predictions` nao tiver as colunas `quantile_*_post_guardrail`
  (silver legado pre-Stage 8), `*_post_guardrail` e NaN e um warning unico e
  logado por run.
- Tabela `gold_prediction_metrics_by_run_split_horizon` carrega tambem colunas
  `delta_<metrica>_post_minus_raw` para tornar o efeito do guardrail
  diretamente queryavel sem reconstrucao via `gold_quantile_guardrail_audit`.
