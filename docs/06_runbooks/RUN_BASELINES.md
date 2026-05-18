---
title: Run Baselines
scope: Comandos canonicos para executar baselines estatisticos (Stage 12) — tanto via sweep JSON (caminho canonico, F.0.2) quanto via CLI standalone (debug, F.0.1). Inclui contratos de `parent_sweep_id` compartilhado e alinhamento warmup com TFT.
update_when:
  - novo baseline for adicionado a `BASELINE_SPECS`
  - politica de window default mudar
  - schema da secao `baselines` do JSON evoluir
canonical_for: [run_baselines, baseline_test_pipeline, baseline_debug_cli]
---

# Rodando Baselines Estatisticos

Stage 12 entregou `RunBaselinesUseCase` com 3 baselines MVP:
`zero_return`, `historical_mean_rolling`, `historical_quantiles_rolling`.
Dois entrypoints expoem o use case:

1. **Canonical (sweep JSON)**: `main_baselines_test_pipeline` (F.0.2) —
   le o mesmo JSON do `main_tft_test_pipeline`. Compartilha
   `output_subdir → parent_sweep_id`, `walk_forward.folds`,
   `replica_seeds` e `training_config.max_encoder_length` (este ultimo
   deriva `evaluation_start_offset_days` para alinhar warmup do baseline
   ao do TFT). **Use sempre que possivel** — garante reprodutibilidade
   e alinhamento timestamp.

2. **Debug (standalone CLI)**: `main_run_baselines` (F.0.1) — flags
   diretas, sem JSON. Use para reproducoes ad-hoc, comparacoes pontuais
   contra candidato ja persistido, ou quando o sweep JSON nao existir
   ainda.

## 1) Canonical: via sweep JSON config

```bash
# 1. Treinar TFT
.venv/bin/python -m src.main_tft_test_pipeline \
  --asset AAPL \
  --config-json config/sweeps/explicit/phase_a_smoke_20260518_v2.json

# 2. Rodar baselines do mesmo sweep (alinhamento via shared JSON)
.venv/bin/python -m src.main_baselines_test_pipeline \
  --asset AAPL \
  --config-json config/sweeps/explicit/phase_a_smoke_20260518_v2.json
```

Schema da secao `baselines` no JSON (top-level): ver
[`docs/03_modeling/SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md)
secao "Schema da secao `baselines`".

### Validacoes esperadas pos-execucao

- `data/analytics/silver/dim_run/asset=AAPL/sweep_id=<output_subdir>/` deve
  conter 1 + N rows (1 TFT candidato + N baselines declarados no JSON).
- `data/analytics/silver/fact_oos_predictions/asset=AAPL/feature_set_name=baseline/year=*/`
  recebe rows dos baselines.
- `target_timestamp_utc` set TFT ⊆ baselines (interseção exata) por
  alinhamento de `evaluation_start_offset_days`.
- Refresh + quality gate em scope `cohort_decision`:
  `gold_paired_oos_intersection_by_horizon` reporta `aligned_exact=True`;
  `gold_dm_pairwise_results` e `gold_mcs_results` incluem TFT + baselines.

## 2) Debug: standalone CLI

```bash
.venv/bin/python -m src.main_run_baselines \
  --asset AAPL \
  --parent-sweep-id phase_a_smoke_20260518 \
  --baselines zero_return,historical_mean_rolling,historical_quantiles_rolling \
  --evaluation-horizons "[1, 7]" \
  --train-start 20101018 --train-end 20201231 \
  --val-start 20210101 --val-end 20221231 \
  --test-start 20230101 --test-end 20251231 \
  --seed 20260517
```

Override de window default por baseline:

```bash
.venv/bin/python -m src.main_run_baselines \
  --asset AAPL \
  --parent-sweep-id sw_debug \
  --baselines historical_mean_rolling \
  --baseline-windows '{"historical_mean_rolling": 60}' \
  --evaluation-horizons "[1]" \
  --train-start 20100101 --train-end 20201231 \
  --val-start 20210101 --val-end 20221231 \
  --test-start 20230101 --test-end 20251231
```

`--overwrite-on-collision` necessario se o `parent_sweep_id` ja tem
runs persistidos com mesmo `run_id` deterministico (Lei 2 explicit flag).

### Alinhamento warmup no modo debug

Diferente do canonical, a CLI standalone **nao** deriva warmup
automaticamente do `max_encoder_length` do TFT. O alinhamento e
responsabilidade do operador: ajustar `--train-start` para coincidir
com `effective_train_start_utc` do TFT candidato, ou aceitar mismatch
(em cujo caso `oos_pairwise_target_alignment` falhara no refresh).

Use o canonical sempre que possivel.

## Codigos de saida

Ambos entrypoints:

- `0`: baselines persistidos com sucesso.
- `2`: dataset_tft nao encontrado (`--dataset-path` invalido).
- ≥1: erro de validacao (baseline desconhecido, split invalido, etc.).

## Documentos relacionados

- [`BASELINES.md`](../04_evaluation/BASELINES.md) — contrato canonico Stage 12.
- [`SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md) — schema da secao `baselines` no JSON.
- [`RUN_SWEEPS.md`](RUN_SWEEPS.md) — execucao de sweeps TFT (Optuna, OFAT, explicit).
- [`RUN_REFRESH_ANALYTICS.md`](RUN_REFRESH_ANALYTICS.md) — refresh + quality gate.
- [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md) Stage 12, Stage F.0 — historico.
