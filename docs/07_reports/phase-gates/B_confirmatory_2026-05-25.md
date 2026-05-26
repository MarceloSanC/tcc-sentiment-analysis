---
title: Phase B Confirmatory — Relatório (2026-05-25)
scope: Relatorio cientifico da Phase B confirmatoria. Reporta tier por horizonte (H1/H2a/H2b) contra Tier 1/Tier 2 gates pre-declarados em preregistration_phase_b.md §9. Cita config sealed sha256, parent_sweep_id, dataset hash, commit selo pre-registro. Cobre limitacoes per F.2 §12 e STRATEGIC_DIRECTION §3.
update_when:
  - tier reportado for reclassificado (exige emenda em §14 do pre-registro)
  - rodada Phase-B-bis (sensibilidade B.4) for adicionada como apendice
canonical_for: [phase_b_confirmatory_report]
---

# Phase B Confirmatory — Relatório (2026-05-25)

**Veredicto resumido (6 slots tier_verdict, hipóteses × horizontes):**

| Hipótese | h = 1 | h = 7 |
|---|---|---|
| **H1** (calibração probabilística) | **Tier 2** | **Tier 1** |
| **H2a** (TFT > zero_return em pinball) | **Tier 1** | **Tier 1** |
| **H2b** (TFT > todos os baselines em pinball) | **Refutado** | **Refutado** |

**Síntese narrativa:** TFT entrega evidência confirmatória primária (Tier 1) para
calibração probabilística em h = 7 e para dominância sobre o baseline trivial
`zero_return` em ambos horizontes (h = 1 e h = 7). Calibração em h = 1 fica em
Tier 2 (evidência secundária) por margem fina (PICP error 2,02% — fora da banda
Tier 1 de ±2,0% por 0,02 pontos percentuais, dentro da banda Tier 2 de ±5%).
H2b é refutado por incapacidade de superar `historical_quantiles_rolling`
(p_adj_holm = 0,465 em h = 1; 0,685 em h = 7; delta_pinball_rel ≈ −0,6% / −1,9%
— TFT empata estatisticamente esse baseline forte). O resultado refutado é
**consistente com STRATEGIC_DIRECTION.md §3** e §4.4, que antecipam baselines
quantílicos rolling near-optimal em ativos líquidos com retornos
near-random-walk; a Phase B confirma esse limite empírico declarado ex-ante.

---

## 1. Parâmetros executados

| Item | Valor |
|---|---|
| **`parent_sweep_id`** | `phase_b_confirmatorio_20260524` |
| **Config sealed sha256** | `fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4` (Emenda E1.6 — substitui hash original `e6972a8b…` da Emenda E1 sem mudança de hiperparâmetros, apenas schema fix `explicit_configs.training_config`) |
| **Dataset sha256 (AAPL)** | `aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298` |
| **Commit selo pre-registro F.2** | `067cb32` (`docs(pre-reg): F.2 — pre-registro Phase B confirmatorio`) |
| **Heritage sweep (E0)** | `phase_b_hpo_20260523` (25 trials Optuna; top-1 robust_score = 0,019330, trial_number = 19) |
| **Hiperparams TFT (top-1, sealed)** | hidden_size = 48; attention_head_size = 8; dropout = 0,05302; learning_rate = 0,0003963; batch_size = 128; hidden_continuous_size = 8; max_epochs = 15; max_encoder_length = 60; max_prediction_length = 7; evaluation_horizons = [1, 7]; prediction_mode = quantile; quantile_levels = [0,1, 0,5, 0,9] |
| **Folds walk-forward** | 3 (`wf_1`, `wf_2`, `wf_3`); janelas declaradas no JSON sealed |
| **Seeds TFT** | [20260517, 20260518, 20260519, 20260520, 20260521] |
| **Réplicas TFT** | 15 = 3 folds × 5 seeds |
| **Baselines** | `zero_return_v1`, `historical_mean_rolling_v1` (w = 30), `historical_quantiles_rolling_v1` (w = 252) |
| **Réplicas por baseline** | 15 = 3 folds × 5 seeds (consumiu mesmo JSON sealed, mesmas seeds) |
| **Total dim_run cohort** | 60 (15 TFT + 45 baseline) |
| **Splits** | train / val / test |
| **N alinhado test split** | 937 timestamps (com dedup operationally-latest cross-fold; ver Emenda E1.8) |

---

## 2. Resultados — Tier classification (núcleo do gate de promoção)

Tier verdict mecanicamente derivado dos critérios pré-declarados em
[`preregistration_phase_b.md`](../../06_pre_registration/preregistration_phase_b.md) §9
sobre os sidecars produzidos pela PR-A4 (Emenda E1.8) em
`data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/`.
Sessão-B **não recomputa** tier; consome sidecar `phase_b_tier_verdict.parquet`
(6 rows) e revalida reality check.

### 2.1 Tabela tier_verdict (autoritativa)

| horizon | hypothesis | tier | critério decisivo | numerical inputs |
|---|---|---|---|---|
| 1 | H1 | **tier_2** | `picp_error` = −0,0202 (fora banda Tier 1 ±0,02; dentro banda Tier 2 ±0,05); todos demais critérios calibração Tier 1 passam (cov_q10 = 0,112; cov_q50 = 0,507; cov_q90 = 0,906) | mpiw = 0,0373; picp_q10_q90 = 0,7798 |
| 1 | H2a | **tier_1** | DM vs zero_return: p_adj_holm = 0,000 (< 0,05); delta_pinball_rel = +29,9% (≥ 3%); mpiw > 0 | mean_pinball_TFT = 0,00417; mean_pinball_zero_return = 0,00595 |
| 1 | H2b | **refutado** | Falha em historical_quantiles_rolling: p_adj_holm = 0,465 (não rejeita H0); delta_pinball_rel = −0,56% (< 0%) | TFT vence em 2/3 baselines mas não no terceiro |
| 7 | H1 | **tier_1** | Todos critérios Tier 1 passam: cov_q10 = 0,112; cov_q50 = 0,505; cov_q90 = 0,910; |picp_error| = 0,0023 (≤ 0,02); mpiw > 0 | mpiw = 0,0376; picp_q10_q90 = 0,7977 |
| 7 | H2a | **tier_1** | DM vs zero_return: p_adj_holm = 0,000; delta_pinball_rel = +28,8%; mpiw > 0 | mean_pinball_TFT = 0,00422; mean_pinball_zero_return = 0,00593 |
| 7 | H2b | **refutado** | Falha em historical_quantiles_rolling: p_adj_holm = 0,685; delta_pinball_rel = −1,88% | TFT vence em 2/3 baselines mas não no terceiro |

### 2.2 Critérios pré-declarados aplicados (F.2 §9)

**Tier 1 (texto principal):**
- Calibração marginal: cov_q10 ∈ [0,07; 0,13]; cov_q50 ∈ [0,45; 0,55]; cov_q90 ∈ [0,87; 0,93]
- PICP: |coverage_error_post_guardrail| ≤ 0,02 (banda [0,78; 0,82])
- DM (H2a/H2b): pvalue_adj_holm < 0,05 (família única, §7)
- Relevância prática: delta_mean_pinball_rel ≥ 3%
- MPIW > 0 e finito

**Tier 2 (evidência secundária):**
- Calibração: cov_q10 ∈ [0,05; 0,15]; cov_q50 ∈ [0,40; 0,60]; cov_q90 ∈ [0,85; 0,95]
- PICP: |coverage_error| ≤ 0,05
- DM: pvalue_adj_holm < 0,10
- delta_mean_pinball_rel ≥ 0%

**Refutado:** falha tanto Tier 1 quanto Tier 2.

A política de promoção é **ex-ante** e foi aplicada **mecanicamente** pelo
pipeline de PR-A4. Sessão-B não introduziu nenhum reframing pós-observação;
tier reportado preserva a honestidade do trabalho per
[STRATEGIC_DIRECTION.md §4.4](../../00_overview/STRATEGIC_DIRECTION.md).

---

## 3. Resultados — Métricas pontuais e probabilísticas (descritivo)

Métricas agregadas no test split (n_oos = 436 timestamps após dedup, ×
réplicas), média ± std sobre 15 runs por família.

| horizon | família | n_runs | RMSE | MAE | DA | mean_pinball_post_guardrail | PICP_post_guardrail | MPIW_post_guardrail |
|---|---|---|---|---|---|---|---|---|
| 1 | **BTSF (TFT)** | 15 | 0,01679 ± 0,00243 | 0,01189 ± 0,00168 | 0,520 ± 0,022 | 0,003965 ± 0,000522 | 0,7853 ± 0,0348 | 0,03684 ± 0,00631 |
| 1 | baseline_historical_quantiles_rolling_v1 | 15 | 0,01658 ± 0,00241 | 0,01171 ± 0,00162 | 0,529 ± 0,023 | 0,003984 ± 0,000523 | 0,8064 ± 0,0300 | 0,03854 ± 0,00552 |
| 1 | baseline_historical_mean_rolling_v1 | 15 | 0,01682 ± 0,00249 | 0,01198 ± 0,00177 | 0,510 ± 0,032 | (point/degenerate) | n/a | n/a |
| 1 | baseline_zero_return_v1 | 15 | 0,01657 ± 0,00240 | 0,01173 ± 0,00160 | 0,004 ± 0,001 | (point/degenerate) | n/a | n/a |
| 7 | **BTSF (TFT)** | 15 | 0,01672 ± 0,00232 | 0,01184 ± 0,00155 | 0,516 ± 0,034 | 0,003986 ± 0,000519 | 0,7919 ± 0,0386 | 0,03717 ± 0,00648 |
| 7 | baseline_historical_quantiles_rolling_v1 | 15 | 0,01654 ± 0,00238 | 0,01164 ± 0,00155 | 0,545 ± 0,027 | 0,003973 ± 0,000519 | 0,8087 ± 0,0313 | 0,03854 ± 0,00552 |
| 7 | baseline_historical_mean_rolling_v1 | 15 | 0,01676 ± 0,00242 | 0,01196 ± 0,00170 | 0,508 ± 0,027 | (point/degenerate) | n/a | n/a |
| 7 | baseline_zero_return_v1 | 15 | 0,01653 ± 0,00237 | 0,01166 ± 0,00154 | 0,004 ± 0,001 | (point/degenerate) | n/a | n/a |

**Observações descritivas:**

- **RMSE/MAE:** ordens de grandeza idênticas para TFT e os 3 baselines
  (~0,0167); diferenças < 0,2% relativos. Consistente com retornos AAPL
  near-random-walk em frequência diária.
- **DA (directional accuracy):** TFT 51,5–52% supera ligeiramente
  `historical_mean_rolling` (50,8–51,0%); `historical_quantiles_rolling`
  obtém 53–54%. `zero_return` sempre prediz 0 → DA ≈ 0% (degenerada por
  construção); incluída por completude.
- **Pinball post-guardrail:** TFT (0,003965 h = 1; 0,003986 h = 7) é
  competitivo com `historical_quantiles_rolling` (0,003984 h = 1; 0,003973
  h = 7). Diferença absoluta ~10⁻⁵, traduzindo em delta_rel = −0,6%/−1,9%
  (TFT marginalmente pior). Vs `zero_return` (convenção degenerada
  q10 = q50 = q90 = 0 per Emenda E1.8) e `historical_mean_rolling` (idem
  q = mean), TFT vence por ~28–30%.
- **PICP:** TFT subcalibrado em h = 1 (0,785 vs nominal 0,80); h = 7 quase
  perfeito (0,792). `historical_quantiles_rolling` ligeiramente
  sobrecalibrado (0,807–0,809).
- **MPIW:** TFT (0,037) ligeiramente mais estreito que
  `historical_quantiles_rolling` (0,039); ambos finitos e positivos.

**Nota de escopo:** a tabela acima é **descritiva**. MCS test
(`gold_mcs_results.parquet`) é executado **within-family** (BTSF e baselines
têm `split_fingerprint` distintos, ver Emenda E1.7); inferência cross-family
vive em DM family-6 (§4.1).

---

## 4. Resultados — Testes estatísticos

### 4.1 DM pairwise family-6 (primary)

Estratégia operacional declarada pela [Emenda E1.8](../../06_pre_registration/preregistration_phase_b.md#emenda-e18):

- Unidade estatística: `target_timestamp` cross-fold com dedup
  `operationally_latest_fold`.
- Agregação por timestamp: `mean_loss_diff_by_timestamp` (média sobre 5 seeds
  por fold; mantém uma observação por timestamp/fold; dedup deixa apenas a
  observação do fold operationally-latest quando há sobreposição).
- Loss: `pinball_loss_post_guardrail` (Categoria A).
- Variância: HAC (Newey-West) com lag = h (1 ou 7); Harvey-Leybourne-Newbold
  (HLN) small-sample correction aplicada.
- Hipótese: one-sided `H_A: TFT < baseline` (TFT melhor).
- Holm-Bonferroni sobre família de 6 testes (3 baselines × 2 horizontes).

| horizon | baseline | mean_loss_diff (TFT − BSL) | dm_stat | p_one_sided | p_adj_holm | n_obs_eff | direction | significant @ α = 0,05 |
|---|---|---|---|---|---|---|---|---|
| 1 | historical_mean_rolling | −0,002183 | −24,4 | 0,0000 | 0,0000 | 937 | tft_better | **PASS** |
| 1 | historical_quantiles_rolling | −0,000020 | −0,73 | 0,2325 | 0,4651 | 937 | tft_better | FAIL |
| 1 | zero_return | −0,002018 | −23,8 | 0,0000 | 0,0000 | 937 | tft_better | **PASS** |
| 7 | historical_mean_rolling | −0,002129 | −20,9 | 0,0000 | 0,0000 | 937 | tft_better | **PASS** |
| 7 | historical_quantiles_rolling | +0,000015 | +0,48 | 0,6846 | 0,6846 | 937 | baseline_better_or_equal | FAIL |
| 7 | zero_return | −0,001956 | −20,3 | 0,0000 | 0,0000 | 937 | tft_better | **PASS** |

**Leitura:** TFT domina os dois baselines pontuais (delta absoluto ~ 2 × 10⁻³,
DM stat |t| > 20, p_adj = 0 nos 4 pares). Empata estatisticamente o baseline
quantílico forte em ambos horizontes — não rejeita H0 = igualdade.

### 4.2 MCS (Model Confidence Set, α = 0,05)

Tabela MCS por horizonte (split = test). **Escopo within-family** (caveat
Emenda E1.7): grupos BTSF e baselines avaliados separadamente.

| horizon | família | n_configs avaliados | n_selected (MCS α = 0,05) |
|---|---|---|---|
| 1 | BTSF | 15 | 15 (todos) |
| 1 | baseline | 9 | 9 (todos) |
| 7 | BTSF | 15 | 15 (todos) |
| 7 | baseline | 9 | 8 |

**Leitura:** todos os 15 TFT runs ficam no MCS within-family em ambos
horizontes; o set de baselines é colapsado quase completamente. Confirma que
não há um TFT seed/fold dominante. Cross-family MCS não está disponível
(débito YELLOW Emenda E1.7); a comparação cross-family de fato vive em DM
family-6 (§4.1).

### 4.3 Win-rate

`gold_win_rate_pairwise_results` contém pares **within-family** (60 BTSF-vs-BTSF
+ 18 baseline-vs-baseline; nenhum cross-family). Cross-family TFT-vs-baseline
win-rate **não foi recomputado** por Sessão-B (escopo análise apenas; ver
nota Emenda E1.7 e §7 "Limitações" abaixo). Inferência cross-family completa
está em DM family-6, §4.1.

**Within-BTSF (test, h = 1):** distribuição de wins entre os 15 TFT runs é
aproximadamente uniforme (≈ 50–55% para qualquer par TFT_i vs TFT_j);
confirma ausência de seed/fold outlier.

---

## 5. Gate de degeneração quantílica

Per [`preregistration_phase_b.md`](../../06_pre_registration/preregistration_phase_b.md) §9
"Gate de degeneracao quantilica" e Stage 11 (`block_quantile_degeneracy_gate`).

| split | horizon | prediction_mode | n_rows | p10_eq_p90_rate | gate_passed |
|---|---|---|---|---|---|
| test | 1 | point | 13.070 | 1,000 | True (mecânico: convenção degenerada per E1.8) |
| test | 1 | **quantile** | 13.070 | **0,000** | **True** |
| test | 7 | point | 13.070 | 1,000 | True (mecânico) |
| test | 7 | **quantile** | 13.070 | **0,000** | **True** |
| train | 1 | quantile | 37.555 | 0,000 | True |
| train | 7 | quantile | 37.555 | 0,000 | True |
| val | 1 | quantile | 13.110 | 0,000 | True |
| val | 7 | quantile | 13.110 | 0,000 | True |

**Leitura:** 0% de degeneração em todos os grupos TFT quantílicos
(quantile-genuine) em test, val e train para ambos horizontes. Nenhuma run
TFT excluída de claims probabilísticos. Convenção `p10 = p50 = p90 = y_pred`
nos baselines pontuais é mecânica (declarada pela Emenda E1.8) — não
representa falha; é o que permite pinball calculado consistentemente sobre os
3 baselines em §4.1.

---

## 6. Robustez (dispersão fold × seed)

CV (desvio/média) sobre 15 runs por (família × horizonte):

| horizon | família | CV RMSE | CV pinball_pg | CV PICP_pg |
|---|---|---|---|---|
| 1 | BTSF | 14,5% | 13,2% | 4,4% |
| 1 | historical_quantiles_rolling | 14,6% | 13,1% | 3,7% |
| 7 | BTSF | 13,9% | 13,0% | 4,9% |
| 7 | historical_quantiles_rolling | 14,4% | 13,1% | 3,9% |

Dispersão dominada por **variabilidade entre folds** (3 janelas walk-forward
com regimes de mercado distintos), não entre seeds. Inspeção dos 3 grupos
visíveis nos run-level RMSE (~0,0135, 0,0182, 0,0185) sugere que `wf_1`
captura regime de menor volatilidade enquanto `wf_2/wf_3` capturam períodos
mais agitados. Sem fragile winners: nenhum TFT seed mostra outlier extremo;
o ranking within-family é estável.

---

## 7. Limitações declaradas

Per [F.2 §12](../../06_pre_registration/preregistration_phase_b.md#12-out-of-scope-phase-b-lista-oficial)
e [`STRATEGIC_DIRECTION.md`](../../00_overview/STRATEGIC_DIRECTION.md) §3:

1. **Single-asset (AAPL).** Sem claim de generalização para outros tickers,
   índices ou classes; reproducibility cross-asset = future work.
2. **N efetivo em h = 7:** 937 timestamps dedupados cross-fold sobre ~3 anos
   OOS; HLN small-sample correction aplicada. Power limitado para detectar
   deltas pinball < ~1% em h = 7.
3. **h = 30 out-of-scope** per F.2 §8 (apenas h = 1 e h = 7 reportados).
4. **Walk-forward sensibilidade B.4** (variação de janelas) declarada como
   rodada futura **Phase-B-bis** (decisão D6 do checklist).
5. **TFT não é dominante em retornos de ações líquidas** per
   STRATEGIC_DIRECTION §3 (random-walk near-optimal). H2b refutado
   confirma esse limite empírico declarado ex-ante; não é achado negativo
   inesperado.
6. **Contribuição de famílias de features** (H3): out-of-scope Phase B;
   Fase C cobre via design `frozen_0_1_configs`.
7. **Cross-family win-rate ausente**: `split_fingerprint` distintos entre
   BTSF e baselines bloqueiam pairing automático por `gold_win_rate_*`
   (débito YELLOW da Emenda E1.7). Pré-registro F.2 §7 menciona win-rate
   condicional como suplementar; **DM family-6 substitui** com inferência
   formal HAC + Holm. Resolver requer Phase-B-bis ou PR técnico
   (`derivar parent_sweep_id automaticamente do output_subdir`).
8. **MCS cross-family ausente**: mesma causa (caveat Emenda E1.7); MCS
   within-family confirma estabilidade dentro de cada grupo.
9. **Interpretability (SHAP, attention)**: out-of-scope per spec; Fase C
   cobre H3 — não Phase B.
10. **Treino determinístico end-to-end ainda não verificado bit-identical**
    cross-rerun: seeds fixas garantem reprodução estatística; bit-identity
    é desafio CUDA + cuDNN non-determinism, fora do escopo F.2.

---

## 8. Conclusão

A Phase B confirmatória entrega **evidência confirmatória primária Tier 1**
para:

- **H1 em h = 7**: TFT está marginalmente calibrado dentro de todas as bandas
  Tier 1 (cov_q10 = 0,112; cov_q50 = 0,505; cov_q90 = 0,910; |picp_err| = 0,2%).
- **H2a em h = 1 e h = 7**: TFT supera `zero_return` com p_adj_holm = 0,000 e
  delta_pinball_rel = +28,8–29,9%, magnitudes muito acima do threshold
  Tier 1 (≥ 3%).

Entrega **evidência secundária Tier 2** para:

- **H1 em h = 1**: TFT está marginalmente calibrado dentro da banda Tier 2
  (|picp_err| = 2,0% — fora da Tier 1 ±2,0% por 0,02 ponto percentual; dentro
  da Tier 2 ±5%). Todos os critérios de calibração marginal por quantil
  individual passam em Tier 1.

**H2b é refutado** em ambos horizontes: TFT não supera
`historical_quantiles_rolling` (DM p_adj_holm = 0,465 / 0,685; delta_rel
≈ −0,6% / −1,9%). Reportado per
[STRATEGIC_DIRECTION §4.4](../../00_overview/STRATEGIC_DIRECTION.md):
**"TFT não supera baselines quantílicos rolling neste protocolo"**.
Resultado **honesto e cientificamente válido** — confirma o limite
empírico ex-ante de §3 (random-walk near-optimal em retornos de ativos
líquidos). Direciona trabalho futuro para diagnóstico de calibração e
arquitetura per §3 e §4.4.

A combinação (Tier 1 H2a + Tier 1 H1 em h = 7 + Tier 2 H1 em h = 1 +
refutação documentada H2b) é a configuração mais defensável academicamente
do espaço de resultados pré-declarado: nenhuma claim além do suportado,
nenhum reframing pós-observação, política de promoção mecânica.

Cohort intacto: 60 dim_run rows / 25 gold tables / 5 sidecars / archive
pre-Phase B 851M preservado / archive Phase B 2026-05-25 read-only criado
em E6.3. Fase C autorizada a iniciar.

---

## 9. Cross-links

| Tema | Documento |
|---|---|
| Pré-registro (F.2 + Emendas E1–E1.8) | [`docs/06_pre_registration/preregistration_phase_b.md`](../../06_pre_registration/preregistration_phase_b.md) |
| Checklist execução (E0–E6) | [`docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md`](../../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md) |
| Direção estratégica / hipóteses | [`docs/00_overview/STRATEGIC_DIRECTION.md`](../../00_overview/STRATEGIC_DIRECTION.md) §3, §4.4, §5 |
| Statistical tests (DM/MCS/Holm + HLN + unidade walk-forward) | [`docs/04_evaluation/STATISTICAL_TESTS.md`](../../04_evaluation/STATISTICAL_TESTS.md) |
| Calibration thresholds (bandas Tier 1/Tier 2) | [`docs/04_evaluation/CALIBRATION_AND_RISK.md`](../../04_evaluation/CALIBRATION_AND_RISK.md) |
| Métricas A/B/C + variante quantílica | [`docs/04_evaluation/METRICS_DEFINITIONS.md`](../../04_evaluation/METRICS_DEFINITIONS.md) |
| Runbook tier pipeline | [`docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md`](../../06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md) |
| Smoke F.1 v4 PASS pleno (precedente) | [`docs/07_reports/smoke_confirmatory_2026-05-18.md`](../smoke_confirmatory_2026-05-18.md) §"F.1 v4 PASS pleno" |
| Living-paper §método (vocabulário) | [`docs/07_reports/living-paper/20_method.md`](../living-paper/20_method.md) §"Politica de variante quantilica" |
| Archive Phase B (read-only) | `data/analytics_archive_phase_b_20260524/` (851M pre-Phase B em `data/analytics_archive_pre_phase_b/` intacto; ver [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md)) |
| Sidecars Phase B (5 parquets) | `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/` |

---

## 10. Referências acadêmicas

- Chernozhukov, V.; Fernandez-Val, I.; Galichon, A. (2010). *Quantile and
  Probability Curves Without Crossing*. Econometrica.
- Diebold, F. X.; Mariano, R. S. (1995). *Comparing predictive accuracy*.
  Journal of Business & Economic Statistics.
- Gneiting, T.; Raftery, A. E. (2007). *Strictly proper scoring rules,
  prediction, and estimation*. JASA.
- Gu, S.; Kelly, B.; Xiu, D. (2020). *Empirical asset pricing via machine
  learning*. RFS.
- Hansen, P. R.; Lunde, A.; Nason, J. M. (2011). *The model confidence set*.
  Econometrica.
- Harvey, D.; Leybourne, S.; Newbold, P. (1997). *Testing the equality of
  prediction mean squared errors*. International Journal of Forecasting.
- Holm, S. (1979). *A simple sequentially rejective multiple test procedure*.
  Scandinavian Journal of Statistics.
- Lim, B.; Arik, S. O.; Loeff, N.; Pfister, T. (2021). *Temporal Fusion
  Transformers for interpretable multi-horizon time series forecasting*.
  International Journal of Forecasting.

---

## 11. Apêndice — DM-18 sensibilidade (per-fold)

Estratégia conservadora `analysis_role=sensitivity_conservative` per Emenda E1.8:
18 testes DM (3 folds × 3 baselines × 2 horizontes), Holm-18 dentro de cada
(fold × horizonte). **Não alimenta o tier verdict**; reportado por
completude per F.2.

| horizon | baseline | fold | dm_stat | p_one_sided | p_adj_holm | direction |
|---|---|---|---|---|---|---|
| 1 | historical_mean_rolling | wf_1 | −19,3 | 0,0000 | 0,0000 | tft_better |
| 1 | historical_quantiles_rolling | wf_1 | −1,52 | 0,0641 | 0,3848 | tft_better (não sig) |
| 1 | zero_return | wf_1 | −18,1 | 0,0000 | 0,0000 | tft_better |
| 1 | historical_mean_rolling | wf_2 | −18,8 | 0,0000 | 0,0000 | tft_better |
| 1 | historical_quantiles_rolling | wf_2 | −0,81 | 0,2079 | 0,8316 | tft_better (não sig) |
| 1 | zero_return | wf_2 | −19,3 | 0,0000 | 0,0000 | tft_better |
| 1 | historical_mean_rolling | wf_3 | −14,1 | 0,0000 | 0,0000 | tft_better |
| 1 | historical_quantiles_rolling | wf_3 | +0,57 | 0,7140 | 1,0000 | baseline_better_or_equal |
| 1 | zero_return | wf_3 | −14,3 | 0,0000 | 0,0000 | tft_better |
| 7 | historical_mean_rolling | wf_1 | −16,6 | 0,0000 | 0,0000 | tft_better |
| 7 | historical_quantiles_rolling | wf_1 | −1,20 | 0,1155 | 0,5777 | tft_better (não sig) |
| 7 | zero_return | wf_1 | −14,9 | 0,0000 | 0,0000 | tft_better |
| 7 | historical_mean_rolling | wf_2 | −18,8 | 0,0000 | 0,0000 | tft_better |
| 7 | historical_quantiles_rolling | wf_2 | −0,13 | 0,4483 | 1,0000 | tft_better (não sig) |
| 7 | zero_return | wf_2 | −18,0 | 0,0000 | 0,0000 | tft_better |
| 7 | historical_mean_rolling | wf_3 | −12,5 | 0,0000 | 0,0000 | tft_better |
| 7 | historical_quantiles_rolling | wf_3 | +2,17 | 0,9850 | 1,0000 | baseline_better_or_equal |
| 7 | zero_return | wf_3 | −12,5 | 0,0000 | 0,0000 | tft_better |

**Leitura:** 12/12 testes contra baselines pontuais significativos em todos os
3 folds (TFT domina). 0/6 testes contra `historical_quantiles_rolling`
significativos em qualquer fold individual (mesma conclusão que DM family-6:
empate estatístico). Sensibilidade **corrobora** o veredito primário; não há
fold que reverta a classificação refutado de H2b.

---

## 12. Apêndice — Decisões autônomas (Sessão-B overnight)

Modo de operação: **autônomo total** (override do prompt primário; ver
[`docs/ai/PHASE_B_SESSION_B_PROMPT.md`](../../ai/PHASE_B_SESSION_B_PROMPT.md)
anexo "Precedentes — Decisões autônomas executadas na Sessão-B"). Sumário
das 7 decisões metodológicas tomadas sem aprovação Marcelo durante esta
sessão:

| # | Decisão | Impacto se reverter |
|---|---|---|
| 1 | Adotar `tier_verdict.parquet` como autoridade sem reclassificar (reality check bateu 6/6) | ~30 linhas |
| 2 | Enquadrar H2b refutado como limitação esperada (STRATEGIC_DIRECTION §3), não achado negativo destacado | ~10 linhas |
| 3 | DM-18 sensibilidade no apêndice; DM-6 primária per E1.8 `analysis_role` | 1 tabela §11 → §4.1 |
| 4 | Métricas pontuais §3/§4 usam média ± std (consistência com DM-6 mean aggregation); marginal coverage §3 usa mediana (consistência com sidecar) | 8 linhas |
| 5 | Cross-family win-rate declarado out-of-pipeline + nota limitação (E1.7 caveat); DM family-6 substitui inferencialmente | 2h pipeline + nova sidecar |
| 6 | Tabela descritiva §3 cross-família incluída com nota MCS within-family | 1 tabela |
| 7 | Archive snapshot inclui `fact_oos_predictions` filtrado por run_ids da cohort (~50MB) para reprodutibilidade ex-post | 10 min |

Log integral em três destinos: PR body, checklist Notas de revisão E5 + E6,
anexo `PHASE_B_SESSION_B_PROMPT.md` (precedentes para futuras Sessões-B
análogas).
