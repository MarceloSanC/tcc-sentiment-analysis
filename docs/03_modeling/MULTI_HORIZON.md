---
title: Multi-Horizon Strategy
scope: Politica de previsao multi-horizonte (h+1, h+7, h+30): horizontes oficiais, contrato de persistencia por linha de predicao, validacoes obrigatorias e tratamento de N efetivo. Inclui justificativa de h+30 como suplementar.
update_when:
  - lista de horizontes oficiais mudar
  - contrato de persistencia de horizonte mudar
  - politica de h+30 (suplementar vs principal) mudar
  - regras de N_effective serem revisadas
canonical_for: [multi_horizon, horizon_policy, h_plus_n, n_effective, target_timestamp]
---

# Multi-Horizon Strategy

## Proposito
Documentar a politica de previsao multi-horizonte do projeto: quais horizontes
sao previstos, como sao representados nos dados persistidos, e como sao
avaliados separadamente. Nao cobre as metricas em si (ver
`04_evaluation/METRICS_DEFINITIONS.md`) nem comandos CLI (ver
`06_runbooks/RUN_TRAINING.md`).

## Conceitos-chave
- **Horizonte (`h+N`):** numero de passos a frente previsto. `h+1` = proximo
  dia util; `h+7` = 7 dias uteis; `h+30` = 30 dias uteis.
- **`max_prediction_length`:** parametro do TFT que define quantos passos a
  frente o modelo emite em uma unica forward pass.
- **`evaluation_horizons`:** lista de horizontes efetivamente avaliados (subset
  de `[1, ..., max_prediction_length]`).
- **`target_timestamp`:** `dataset_timestamps[decision_idx + h]`. Indexado
  em trading days (consulta no dataset, nao `pd.Timedelta`). Ver §"Convencao
  canonica de target_timestamp e y_true" abaixo.
- **N efetivo (nao-sobreposto):** numero de observacoes verdadeiramente
  independentes por horizonte. Para h+30 com janelas diarias, `N_overlapped /
  30` aproxima o N estatisticamente util.

## Convencao canonica de target_timestamp e y_true

Fixada em [ADR-0003](../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md)
(amendada em Stage R-20 / Opcao d) e materializada em
[`src/domain/services/multi_horizon_prediction_persister.py`](../../src/domain/services/multi_horizon_prediction_persister.py):

- **Anchor:** `timestamp_utc = decision_day = dataset_timestamps[decision_idx]`.
  E o ultimo dia do encoder (nao decoder_end, nao calendar arithmetic).
- **target_timestamp:** `target_timestamp_utc = dataset_timestamps[decision_idx + h]`,
  indexado em **trading days** (consulta no dataset). Pula weekends/holidays
  naturalmente porque o dataset ja e trading-day indexed.
- **y_true:** `dataset.target_return[decision_idx + h]`. Convencao
  financeira: `h=1` significa "return realizado no dia seguinte a decisao",
  consistente com literatura. `target_return[t] = log(close[t]/close[t-1])`
  (backward shift) por
  [`src/use_cases/build_tft_dataset_use_case.py:563`](../../src/use_cases/build_tft_dataset_use_case.py#L563).
  A indexacao backward foi escolhida (Stage R-20 / Opcao d) para casar com
  o decoder do `pytorch_forecasting` (`y[0][i, h-1]` fetcha
  `target_return[D+h]`), fechando o off-by-one cross-pipeline (Gap 6).
- **decision_idx:** coluna `int64` em `fact_oos_predictions` que mapeia
  diretamente para `time_idx` do dataset. Materializa o invariante no schema
  (nao apenas no codigo). Required em
  [`analytics_store_schema.py`](../../src/infrastructure/schemas/analytics_store_schema.py)
  `FACT_OOS_PREDICTIONS_SCHEMA.required_columns`.
- **Borda:** se `decision_idx + h >= len(dataset_timestamps)`, levantar
  `IncompletePredictionWindowError` e pular a linha. AGENT_CORE: skip rows
  with missing supervision.

Persistencia: ambos os pipelines (TFT trainer e baseline runner) usam
`MultiHorizonPredictionPersister.build_record()` para emitir registros.
Convencao deixa de ser implicita e distribuida; vira objeto explicito
do dominio com testes proprios em
[`tests/unit/domain/services/test_multi_horizon_prediction_persister.py`](../../tests/unit/domain/services/test_multi_horizon_prediction_persister.py)
e guard cross-pipeline em
[`tests/integration/test_tft_baselines_y_true_alignment.py`](../../tests/integration/test_tft_baselines_y_true_alignment.py)
+ adapter integration test em
[`tests/integration/test_tft_decoder_target_indexing.py`](../../tests/integration/test_tft_decoder_target_indexing.py).

## Decisoes e escolhas
- **Horizontes principais: h+1, h+7, h+30.** *Why:* cobrem curto, medio e
  longo prazo dentro do escopo single-asset; alinhados a horizontes praticos
  de decisao em forecasting financeiro.
- **Geracao em uma unica passagem do TFT (`max_prediction_length=30`).** *Why:*
  evita treinar 3 modelos separados; o TFT compartilha representacao temporal
  entre horizontes.
- **Metricas reportadas separadamente por horizonte, nunca agregadas no total.**
  *Why:* agregar mascara degradacao em horizontes longos, que e exatamente
  onde modelos complexos costumam falhar versus baselines simples. Reportar
  separado e exigencia metodologica.
- **Alinhamento OOS pareado feito por (`asset`, `split`, `horizon`,
  `target_timestamp`), separado por horizonte.** *Why:* DM/MCS exige amostras
  identicas; comparar h+1 de um modelo com h+30 de outro e invalido.
- **h+30 tratado com cautela na escrita do paper.** *Why:* N efetivo nao-sobreposto
  e baixo (tipicamente ~25 para 3 anos OOS); poder estatistico limitado para
  DM/MCS. Reportado sempre com declaracao de limitacao.

## Persistencia (contrato por linha de predicao)

| Coluna | Tipo | Descricao |
|---|---|---|
| `horizon` | `int64` | numero de passos a frente (1, 7, 30) |
| `decision_idx` | `int64` | indice no dataset original (time_idx) do dia de decisao. Materializa a convencao canonica (ADR-0003). |
| `timestamp_utc` | `timestamp[utc]` | decision_day = `dataset_timestamps[decision_idx]` |
| `target_timestamp_utc` | `timestamp[utc]` | `dataset_timestamps[decision_idx + h]` (trading-day indexed) |
| `y_true`, `y_pred` | `float64` | valor real e previsao pontual no horizonte |
| `quantile_p10/p50/p90` | `float64` | quantis raw |
| `quantile_p10/p50/p90_post_guardrail` | `float64` | quantis monotonicos |

Grao logico de `fact_oos_predictions`:
`run_id + split + horizon + timestamp + target_timestamp`.

## Validacoes obrigatorias
- Cobertura completa de horizontes esperados por `run_id`/`split`.
- `n_common` por horizonte na intersecao temporal (ver
  `gold_paired_oos_intersection_by_horizon`).
- `target_timestamp` monotonico por (`run_id`, `split`, `horizon`).

## Fonte da verdade (codigo)
- `src/domain/services/multi_horizon_prediction_persister.py` — domain
  service unico que materializa a convencao canonica (ADR-0003 amendado em
  Stage R-20 / Opcao d) para `fact_oos_predictions`. Ambos pipelines
  (TFT + baselines) emitem rows via
  `MultiHorizonPredictionPersister.build_record(...)`.
- `src/use_cases/train_tft_model_use_case.py` — define
  `evaluation_horizons` e propaga para o trainer
- `src/use_cases/run_baselines_use_case.py` — runner statistical baselines
  (zero_return, historical_mean_rolling, historical_quantiles_rolling)
  com mesma convencao via Persister
- `src/adapters/pytorch_forecasting_tft_trainer.py` — emissao de predicoes
  multi-horizonte (`selected_horizons`, `selected_idx`)
- `src/use_cases/refresh_analytics_store_use_case.py` — calculo de metricas
  por horizonte e intersecao pareada
- `src/infrastructure/schemas/analytics_store_schema.py` — `decision_idx`
  required em `FACT_OOS_PREDICTIONS_SCHEMA`

## Documentos relacionados
- `04_evaluation/METRICS_DEFINITIONS.md` — formulas das metricas por horizonte
- `04_evaluation/STATISTICAL_TESTS.md` — DM/MCS com alinhamento por horizonte
- `00_overview/STRATEGIC_DIRECTION.md` §5 (Fase B) — politica de horizontes
  na rodada confirmatoria
- `05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md` §1.0 — historico de
  implementacao multi-horizonte
