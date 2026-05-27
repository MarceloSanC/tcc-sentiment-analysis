---
title: ADR-0003 MultiHorizonPredictionPersister como domain service compartilhado
scope: Decisao arquitetural de extrair a logica de persistencia de fact_oos_predictions
  multi-horizonte (hoje duplicada e divergente em train_tft_model_use_case e
  run_baselines_use_case) para um domain service unico que materializa
  explicitamente a convencao target_timestamp / y_true / h-ahead. Status Accepted,
  com revisao de formula (Stage R-20 / Opcao d) que preserva semantica financeira
  "next-day return after decision" mas indexa `target_return` backward.
update_when:
  - decisao for revisada (status mudar de Accepted para Superseded)
  - novo ADR derivado tornar este obsoleto
  - convencao target_timestamp evoluir (ex: novo anchor decidido)
canonical_for: [adr_0003, multi_horizon_prediction_persister, target_timestamp_convention]
---

# ADR-0003: MultiHorizonPredictionPersister

## Status
Accepted (2026-05-19); amendado em 2026-05-20 (Stage R-20, Opcao d) para fechar
Gap 6 sem alterar a semantica de "h=1 = next-day return after decision".
Validado end-to-end via R-23 smoke v4 (2026-05-22).

## Context

`fact_oos_predictions` e persistido em dois lugares com convencoes divergentes:

- [`train_tft_model_use_case.py:778-810`](../../../src/use_cases/train_tft_model_use_case.py#L778-L810) — TFT trainer; `split_tail = split_df.tail(n)` recupera o **decoder_end_day** como `timestamp_utc`, nao o **decision_day**.
- [`run_baselines_use_case.py:225-280`](../../../src/use_cases/run_baselines_use_case.py#L225-L280) — baseline runner; usa `timestamps[i]` (decision_day) e `y_true_idx = i + h - 1`.

Ambos materializam `target_ts = decision_ts + pd.Timedelta(days=h-1)` (calendar days num dataset trading-day — bug secundario E4 do investigation report).

Consequencia documentada em
[`docs/07_reports/phase-gates/phase-a/tft_y_true_investigation_2026-05-18.md`](../../07_reports/phase-gates/phase-a/tft_y_true_investigation_2026-05-18.md):
TFT e baseline reportam `y_true` referentes a dias diferentes do dataset
para o mesmo `target_timestamp_utc`. **0 de 685** linhas com `target_ts`
coincidente em h=1 tem `y_true` igual. Bug raiz do Gap 6.

A convencao em
[`docs/03_modeling/MULTI_HORIZON.md:28-29`](../../03_modeling/MULTI_HORIZON.md#L28-L29)
define `target_timestamp` como "timestamp do periodo alvo previsto,
calculado por horizonte" mas **nao fixa o anchor** (decision_day,
decoder_start, decoder_end). Ambiguidade permitiu as duas implementacoes
coexistirem ate o smoke F.1 expor a divergencia.

## Decision

Extrair a logica para um domain service:

```
src/domain/services/multi_horizon_prediction_persister.py

class MultiHorizonPredictionPersister:
    def build_record(
        self,
        *,
        decision_idx: int,
        h: int,
        y_true: float,
        y_pred: float,
        q10: float, q50: float, q90: float,
        dataset_timestamps: list[pd.Timestamp],
        run_context: RunContext,
    ) -> PredictionRecord: ...
```

Convencao canonica fixada (material e auditavel):

1. **Anchor de `timestamp_utc`**: `decision_day` = ultimo dia do encoder
   = `dataset_timestamps[decision_idx]`. Nao decoder_end.
2. **`target_timestamp_utc`**: `dataset_timestamps[decision_idx + h]`,
   indexado em **trading days** (consulta no dataset, nao
   `pd.Timedelta(days=h)`).
3. **`y_true`**: `dataset.target_return[decision_idx + h]`.
   `target_return[t] = log(close[t]/close[t-1])` por
   `src/use_cases/build_tft_dataset_use_case.py:563` (Stage R-20 Opcao d).
   Para `h=1` isso fornece `log(close[D+1]/close[D])`, i.e. "next-day
   return after decision" — semantica financeira preservada. A formula
   indexa `target_return` BACKWARD (em vez do forward shift original)
   justamente para que o decoder do `pytorch_forecasting`
   (`y[0][i, h-1]`, que naturalmente fetcha `target_return[D+h]`)
   coincida bit-a-bit com o que o baseline persiste, fechando Gap 6.
4. **Borda**: se `decision_idx + h >= len(dataset_timestamps)`, levantar
   `IncompletePredictionWindowError` em vez de emitir registro com
   `y_true` invalido. AGENT_CORE: skip rows with missing supervision.

Ambos os call-sites (TFT trainer e baseline runner) passam a chamar o
mesmo `build_record`. Convencao deixa de ser implicita e distribuida
e passa a viver em **um arquivo** com **testes proprios**.

### Historico do item 3 (Opcao a → Opcao d, 2026-05-20)

A versao original deste ADR (Opcao a) prescrevia
`y_true = target_return[decision_idx + h - 1]` com `target_return[t] =
log(close[t+1]/close[t])` (forward shift). Auditoria de 2026-05-20
confirmou empiricamente que o decoder do `pytorch_forecasting` fetcha
`target_return[decision_idx + h]` (uma posicao alem do que Opcao (a)
prescrevia para o baseline), causando off-by-one cross-pipeline (Gap 6,
663/685 linhas test/h=1 com TFT.y_true == baseline.y_true do dia
seguinte). Opcao (d) corrige a formula em `target_return` e ajusta o
indice em `run_baselines_use_case.py:259` (`y_true_idx = i + h_int`)
para casar com o decoder. Semantica "next-day return after decision"
e preservada porque a formula passa a indexar backward.

## Consequences

**Positivas:**

- Convencao `target_timestamp` materializada como entity do dominio,
  auditavel via unit test.
- Fix do Gap 6 vira mudanca de implementacao do persister (1 lugar),
  nao 5 lugares.
- `tft_baselines_timestamp_subset_alignment` (F.0.3) pode ser
  fortalecido para comparar **valores** de `y_true` por
  `(decision_idx, h)`, nao so set de timestamps.
- Reduz blast-radius do `train_tft_model_use_case.py` (hoje 1482 LOC, 30
  metodos) — persistencia sai do god object.

**Negativas:**

- Migracao requer reescrita de `fact_oos_predictions` historicos OU
  bump de `schema_version` invalidando historico (decisao em F.2
  pre-registro).
- Smoke F.1 v3 nao sera byte-identical com F.1 v2 — diff esperado em
  `timestamp_utc` (decoder_end → decision_day) e em `target_timestamp_utc`
  (calendar → trading day arithmetic).
- Adiciona um novo conceito ao dominio (`RunContext`, `PredictionRecord`)
  que precisa estar documentado em `MULTI_HORIZON.md`.
- Stage R-20 (Opcao d) muda a formula de `target_return` no dataset
  (backward em vez de forward). Dataset_tft precisa ser reconstruido;
  `data/analytics_archive_pre_phase_b/` mantem semantica antiga como
  registro historico documentado, nao debito ativo (Phase B ainda nao
  rodou; F.2 pre-registro nem foi feito; modelos antigos sao throwaway
  de smoke max_epochs=1, hidden_size=16).
- Validado end-to-end via R-23 smoke v4: TFT + baselines no mesmo cohort
  `phase_a_smoke_20260522_r23`, `max_epochs=5`, `failed_checks=[]`, F.1 6/6 PASS.

## Cross-link

- Diagnostico: [`docs/07_reports/phase-gates/phase-b/B_architectural_debt_2026-05-19.md`](../../07_reports/phase-gates/phase-b/B_architectural_debt_2026-05-19.md) §"M-train_tft persistencia"
- Investigacao raiz: [`docs/07_reports/phase-gates/phase-a/tft_y_true_investigation_2026-05-18.md`](../../07_reports/phase-gates/phase-a/tft_y_true_investigation_2026-05-18.md)
- Implementacao: PHASE_B_IMPLEMENTATION_CHECKLIST.md §Stage 20
- Doc canonico atualizado pos-merge: [`docs/03_modeling/MULTI_HORIZON.md`](../../03_modeling/MULTI_HORIZON.md) §"Convencao target_timestamp" (nova secao)
