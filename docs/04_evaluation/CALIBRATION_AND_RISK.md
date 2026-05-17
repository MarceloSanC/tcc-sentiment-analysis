---
title: Calibration and Risk
scope: Metodologia de calibracao probabilistica (calibracao marginal por quantil, PICP intervalar, MPIW), bandas de aceitacao para gate de elegibilidade e metricas de risco (VaR, ES). Distincao explicita entre cobertura intervalar (PICP do p10-p90) e calibracao marginal por quantil individual. Politica de variante quantilica (raw vs post-guardrail) por metrica eh canonica em `METRICS_DEFINITIONS.md` §Variante quantilica.
update_when:
  - bandas de tolerancia de calibracao marginal mudarem
  - thresholds de coverage_error/MPIW serem revisados
  - nova metrica de risco for adicionada (VaR, ES, downside)
  - politica de gate de elegibilidade por calibracao mudar
  - variante quantilica primaria de calibracao/risco mudar (sincronizar com METRICS_DEFINITIONS.md)
canonical_for: [calibration, risk_metrics, marginal_quantile_calibration, interval_picp, var_es]
---

# Calibration and Risk

Objetivo: definir metodologia de calibracao e criterios de aceite para risco
predito por horizonte, com base em artefatos persistidos.

## Canonical Implementation
- `src/use_cases/refresh_analytics_store_use_case.py`
- `src/use_cases/validate_analytics_quality_use_case.py`

## Calibration Method

### Inputs
- Quantis por linha em duas variantes (ver `METRICS_DEFINITIONS.md` §Variante
  quantilica):
  - **raw**: `quantile_p10`, `quantile_p50`, `quantile_p90` — saida direta
    do modelo, podendo violar monotonicidade.
  - **post-guardrail**: `quantile_p10_post_guardrail`, `quantile_p50_post_guardrail`,
    `quantile_p90_post_guardrail` — rearranjo monotono via
    `QuantileGuardrailService.enforce_monotonic_triplet`.
- Verdade-terreno `y_true` (splits supervisionados).

**Variante primaria para claims de calibracao (H1):** `post_guardrail`.
Calibracao exige objeto avaliado ser intervalo valido (`p10 <= p90`); a
versao raw e reportada em paralelo como auditoria do efeito do guardrail
(coluna `delta_*_post_minus_raw`).

### Computation: Interval Calibration (primaria: post-guardrail)
- Cobertura nominal: `0.80` (intervalo p10-p90)
- `PICP_post_guardrail = mean(quantile_p10_post_guardrail <= y_true <= quantile_p90_post_guardrail)`
- `MPIW_post_guardrail = mean(quantile_p90_post_guardrail - quantile_p10_post_guardrail)`
- `coverage_error_post_guardrail = PICP_post_guardrail - 0.80`
- `confidence_calibrated` conforme formula em `METRICS_DEFINITIONS.md`
- Variante `*_raw` calculada com as mesmas formulas sobre colunas raw, persistida
  em paralelo para auditoria; nao e contrato primario.

### Computation: Marginal Quantile Calibration (primaria: post-guardrail)
Para quantis individuais, nao usar o termo PICP. O criterio correto e a taxa
empirica de cobertura marginal, calculada sobre quantis post-guardrail por
default:

- `coverage_q10 = mean(y_true <= quantile_p10_post_guardrail)`; alvo nominal `0.10`
- `coverage_q50 = mean(y_true <= quantile_p50_post_guardrail)`; alvo nominal `0.50`
- `coverage_q90 = mean(y_true <= quantile_p90_post_guardrail)`; alvo nominal `0.90`
- `quantile_coverage_error_qtau = coverage_qtau - tau`

Variantes raw persistidas em paralelo (sufixo `_raw`); nao primarias.

### Calibration Interpretation
- `coverage_error ~ 0`: cobertura intervalar adequada para p10-p90
- `coverage_error < 0`: under-coverage intervalar (otimista demais)
- `coverage_error > 0`: over-coverage intervalar (intervalos conservadores)
- `coverage_qtau ~ tau`: calibracao marginal adequada do quantil `tau`
- `coverage_qtau < tau`: quantil previsto esta alto demais para o nivel nominal
- `coverage_qtau > tau`: quantil previsto esta baixo demais para o nivel nominal

## Calibration Acceptance Thresholds
Para decisao oficial, aplicar por horizonte e coorte sobre a **variante
primaria post-guardrail**:
- `abs(coverage_error_post_guardrail) <= 0.02` (target)
- bandas de calibracao marginal por quantil devem ser declaradas no
  pre-registro (default sugerido para debate: p10 in [0.07, 0.13],
  p50 in [0.45, 0.55], p90 in [0.87, 0.93])
- `MPIW_post_guardrail > 0` e finito (post-guardrail garante por construcao;
  violacao indica entrada degenerada — ver gate de degeneracao)
- Sem colapso de cobertura (`PICP_post_guardrail` nao proximo de 0 em todos
  modelos)

Observacao: limites podem ser endurecidos por pre-registro de rodada.

**Checks complementares sobre variante raw** (nao sao parte de "calibration
acceptance"; pertencem ao quality gate / Stage 11 do
`PHASE_B_IMPLEMENTATION_CHECKLIST.md`):
- `(quantile_p90 - quantile_p10) >= 0` por linha (Categoria C — `_pred_interval_negative`
  em `gold_oos_quality_report`).
- Contrato quantilico valido raw (`p10 <= p50 <= p90`) por linha — usado pelo
  gate de degeneracao para detectar colapso (`p10==p90`) e patologia upstream.

Os checks raw existem para detectar a patologia do modelo; sob post-guardrail
sao tautologicamente satisfeitos por construcao e perderiam funcao
diagnostica. Por isso permanecem **raw apenas**.

## Quantile Contract Guardrail (Block A)
- Crossing bruto aceito no maximo: `<= 0.10%` (provisorio)
- Crossing pos-guardrail: `0`
- Interval width negativo: `0`
- Impacto do guardrail:
  - `delta_pinball_rel <= +1.0%`
  - `abs(delta_picp) <= 0.02`
  - `delta_mpiw_rel <= +5.0%`

## Risk Method
Agregado por (`run_id`, `split`, `horizon`). Metricas de risco financeiro sao
**post-guardrail apenas** (Categoria B em `METRICS_DEFINITIONS.md`): Value-at-Risk
exige monotonicidade da funcao quantil por definicao (Jorion 2007; Acerbi &
Tasche 2002). Sob crossing raw, `var_10` perde semantica de quantil e ES
perde a interpolacao linear pressuposta (`q10 <= q50`).

- `expected_move = mean(abs(y_pred))` — independente de quantis.
- `downside_risk = mean(max(-y_pred, 0))` — independente de quantis.
- `var_10 = mean(quantile_p10_post_guardrail)`
- `es_10_approx = mean(min(1.125*q10_post_guardrail - 0.125*q50_post_guardrail, q10_post_guardrail))`

## Risk Acceptance (minimum)
- `var_10` e `es_10_approx` coerentes (`es_10_approx <= var_10`) — garantido
  por construcao sob post-guardrail.
- Sem valores nao finitos
- Comparacao sempre por mesmo horizonte e mesma coorte

## Canonical Outputs
- `gold_prediction_calibration.parquet`
- `gold_prediction_risk.parquet`
- `gold_quantile_guardrail_audit.parquet`

## Operational Command
```bash
python -m src.main_refresh_analytics_store --fail-on-quality
```
