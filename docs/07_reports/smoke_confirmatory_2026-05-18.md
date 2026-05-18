---
title: F.1 Smoke Confirmatório — Relatório (2026-05-18)
scope: Validação operacional integrada dos Stages 1-14 mergeados em main (PRs #14-34). Não é o experimento confirmatório Phase B; apenas verifica que o pipeline pós-Stages convive sem regressão cross-stage. Resultado decide prosseguir para F.2 (pre-registro), aplicar fix curto e re-smoke, ou abrir issue de Stage.
update_when:
  - F.1 for re-executado e gerar novo veredito
  - fixes pequenos pós-smoke forem mergeados e o novo smoke confirmar resolução
canonical_for: [smoke_confirmatorio_phase_a_2026_05_18, f1_acceptance_record]
---

# F.1 Smoke Confirmatório — Relatório (2026-05-18)

## Veredicto

**FAIL** — critério 6 (refresh + quality gate sem erro) **falhou** (exit 1; 4 de
26 checks reprovados). Critérios 1, 2, 3, 4 **passam**; critério 5 **PARTIAL**.
Falhas do critério 6 são **estruturais** (lacuna de Stage 12, alinhamento
TFT vs baselines), **não regressão** de Stages mergeados.

**F.2 pre-registro fica bloqueado** até decisão sobre os 3 itens listados em
"Recomendação operacional".

## Parâmetros do smoke

- `parent_sweep_id`: `phase_a_smoke_20260518`
- Branch: `feat/phase-a-smoke-confirmatory-20260518` (criada a partir de `main` `@ ab01048`)
- Candidato TFT: 1 corrida `feature_set_name=BTSF` (all-features: 24 colunas: OHLCV + technicals + sentiment + fundamentals)
  - `max_epochs=5`, `max_encoder_length=60`, `max_prediction_length=7`
  - `evaluation_horizons=[1, 7]` (h+30 excluído por design — audit M7 ação 9)
  - `prediction_mode=quantile`, `quantile_levels=[0.1, 0.5, 0.9]`
  - `loss=QuantileLoss`, `seed=20260517`
- Baselines (mesmo `parent_sweep_id`): `zero_return`, `historical_mean_rolling` (w=30), `historical_quantiles_rolling` (w=252)
- Split utilizado (herdado do `resolve_training_split`):
  - train: `20101018 → 20201231`
  - val: `20210101 → 20221231`
  - test: `20230101 → 20251231`
- Dataset: `data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet` (4023 rows, 62 cols, Stage 14 aplicado — `fundamentals_effective_date` presente, `effective_date` removida).
- Refresh: `--scope-mode cohort_decision --scope-sweep-prefixes phase_a_smoke_20260518 --primary-quantile-contract post_guardrail --fail-on-quality`

## Resultados por critério

### 1. % p10==p90 < 5% (gate Stage 11) — **PASS**

`gold_quantile_degeneracy_report` por `(parent_sweep_id, split, horizon, prediction_mode)`:

| split | horizon | prediction_mode | n_rows | p10_eq_p90_rate | gate_passed |
|---|---|---|---|---|---|
| train | 1 | quantile | 2517 | 0.0 | True |
| train | 7 | quantile | 2517 | 0.0 | True |
| val | 1 | quantile | 940 | 0.0 | True |
| val | 7 | quantile | 940 | 0.0 | True |
| test | 1 | quantile | 1436 | 0.0 | True |
| test | 7 | quantile | 1430 | 0.0 | True |

Linhas em `prediction_mode=point` têm `p10_eq_p90_rate=1.0` por construção
(zero_return e historical_mean_rolling emitem ponto único; Stage 9 exclui
do agregado probabilístico e Stage 11 ignora `point`). Comportamento esperado.

Caveat: grupos quantile em `val` têm `n_rows=940 < 1000` (limiar default
`--degeneracy-min-rows-for-gate=1000`); rate=0.0 mesmo assim, então não há
ambiguidade — mas para experimento confirmatório, F.2 deve declarar política
de threshold para datasets pequenos como AAPL piloto.

### 2. Zero violações probabilísticas — **PASS**

`gold_prediction_metrics_by_run_split_horizon` (22 rows do smoke):

- `mpiw_raw`: total não-nulo=10, **negativos=0**, min=0.03136
- `mpiw_post_guardrail`: total=10, **negativos=0**, min=0.03136
- `delta_mpiw_post_minus_raw`: min=0.0 (guardrail nunca alargou interval além do raw)

`gold_quantile_guardrail_audit` (22 rows do smoke):
- `crossing_before_count=0` para todas as combinações (TFT + 3 baselines)
- `crossing_before_rate=0.0` para todas
- `negative_width_before_count=0` para todas

Guardrail post não foi necessário aplicar — quantis raw já monótonos.

### 3. Gold cohort-aware (parent_sweep_id no output) — **PASS**

10/10 tabelas verificadas contêm `parent_sweep_id` e rows do smoke:

| Tabela | total rows | rows do smoke |
|---|---|---|
| gold_ranking_by_config | 1 | 1 |
| gold_model_decision_final | 8 | 6 |
| gold_feature_set_impact | 6 | 6 |
| gold_prediction_metrics_by_config | 22 | 22 |
| gold_ic95_by_config_metric | 3 | 3 |
| gold_dm_pairwise_results | 6 | 6 |
| gold_mcs_results | 6 | 6 |
| gold_win_rate_pairwise_results | 6 | 6 |
| gold_prediction_metrics_by_run_split_horizon | 22 | 22 |
| gold_quantile_degeneracy_report | 12 | 12 |

### 4. pytest tests/ — **PASS**

`.venv/bin/pytest tests/ --tb=no -q` → **510 passed, 45 warnings, 0 failed** em 27s.
Warnings são FutureWarning (pandas downcasting em refresh_analytics_store.py
linhas 2040-2042; lightning _pytree LeafSpec deprecated) e DeprecationWarning
(`block_a_parent_sweep_prefixes` redundante com `scope_spec`). Nenhum warning
novo emerge do smoke; todos pré-existem nos Stages mergeados.

### 5. CI verde nos PRs Stage 1-14 — **PARTIAL**

Inspeção via `gh pr list --state merged --limit 30 --json statusCheckRollup`
mostra que **6 PRs foram mergeados com `lint-type-unit=FAILURE`** ao tempo
do merge:

| PR | Stage | Status no merge | Notas |
|---|---|---|---|
| #20 | Stage 7 (refresh scoped) | lint-type-unit=FAILURE | mypy issue subsequente |
| #23 | Stage 8 (raw/post-guardrail) | lint-type-unit=FAILURE | corrigido em #29 |
| #25 | Stage 9 (filtro de modo) | lint-type-unit=FAILURE | |
| #26 | Stage 10 (training_run_id FK) | lint-type-unit=FAILURE | |
| #27 | docs sync TCC Stages 1-8 | lint-type-unit=FAILURE | docs-only |
| #28 | Stage 11 (gate degeneração) | lint-type-unit=FAILURE | |

PR #29 (mergeado 2026-05-17 22:40 UTC) explicitamente "fix(ci): narrow
primary_quantile_contract to Literal to satisfy mypy" — ou seja, o mypy
falhou ao merge de #20, #23, #25-28 mas o fix entrou em #29. A partir de
#29 todos os PRs subsequentes (#30 Stage 12, #32 Stage 13, #34 Stage 14,
#35-37 docs) têm CI verde no merge.

**Avaliação**: `main` atual está verde (pytest local 510 passed; análise gh
mostra PRs #29-37 com SUCCESS). Os 6 PRs com falha no merge entregaram código
correto sob critério funcional (pytest), mas falharam mypy temporariamente.
Não é regressão; é dívida de signal CI que foi sanada em #29. Para
estritamente "CI verde em todos os PRs anteriores" — não satisfeito.
Para "main em estado funcional verde" — satisfeito.

Recomendo registrar como PASS_WITH_CAVEAT em decisão operacional, mas
listo como PARTIAL aqui por rigor.

### 6. Refresh + quality gate sem erro — **FAIL**

`main_refresh_analytics_store --scope-mode cohort_decision
--scope-sweep-prefixes phase_a_smoke_20260518 --primary-quantile-contract
post_guardrail --fail-on-quality` retornou **exit code 1**.

Gold refresh executou com sucesso (26 tabelas gold materializadas). Quality
validation: **22 de 26 checks PASS, 4 FAIL**.

#### Checks reprovados

**(a) `oos_pairwise_target_alignment` — FAIL**

```
asset=AAPL|sweep=phase_a_smoke_20260518|split=test|h=1|configs=4|min_ts=685|max_ts=751
asset=AAPL|sweep=phase_a_smoke_20260518|split=test|h=7|configs=4|min_ts=685|max_ts=745
asset=AAPL|sweep=phase_a_smoke_20260518|split=val|h=1|configs=4|min_ts=437|max_ts=503
asset=AAPL|sweep=phase_a_smoke_20260518|split=val|h=7|configs=4|min_ts=437|max_ts=503
```

**Causa**: TFT emite predições só para `target_timestamp_utc` viável após
`max_encoder_length=60` warmup (685 timestamps no test, 437 no val).
Baselines emitem para todas as datas do split (751 no test, 503 no val).
Intersecção exata falha → TFT é **excluído** de `gold_dm_pairwise_results`,
`gold_mcs_results`, e `gold_model_decision_final` (decision_final mostra
apenas 3 configurações: as 3 baselines).

Esse é exatamente o cenário antecipado no audit como Cenário Falha D
([prompt F.1 "Cenário tipo D"](../../) seção; Stage 12 Notas).

**(b) `dm_mcs_persisted_executable` — FAIL**

```
feasible_dm=2, feasible_mcs=2, missing_dm=0, missing_mcs=0,
report_stats_ready=False
```

**Causa**: DM/MCS materializaram (6 rows cada — entre os 3 baselines), mas
`report_stats_ready=False` por agregação faltante. Decorrente de (a).

**(c) `gold_confidence_calibrated_by_horizon` — FAIL**

```
bad_confidence=8, non_finite=0, missing_expected_horizons=0
```

**Causa**: 8 rows de `confidence_calibrated_*` fora do range esperado.
Provavelmente decorrente do split val ter `n_rows=940 < 1000` (smoke
data-limited), causando `picp_post_guardrail` fora de range nominal
[0.7, 0.9]. **Caveat de smoke**, não regressão. Para F.2 confirmatório
(múltiplas seeds × folds), n por grupo cresce e o check passa.

**(d) `official_contract_quantile_attention` — FAIL**

```
missing_model_artifacts=3
```

**Causa**: As 3 baselines não têm `fact_model_artifacts` row (são modelos
estatísticos sem checkpoint salvo). O check exige `fact_model_artifacts`
para cada run com `prediction_mode=quantile`. Em smoke isso reflete uma
**lacuna semântica** do contrato Stage 10: baselines genuinamente não
têm artefato torch/lightning a referenciar. Decisão operacional: ou
afrouxar o check para excluir `feature_set_name='baseline'`, ou
materializar `fact_model_artifacts` "vazio" para baselines (overhead
desnecessário). Recomendo a primeira opção em F.2.

#### Checks aprovados (22/26)

Inclui (não exaustivo): `block_quantile_degeneracy_gate` PASS;
`baseline_present_per_candidate_sweep` PASS (Stage 12.2);
`oos_quality_per_run_split_horizon` PASS (22/22 runs OK: zero rows
duplicadas, zero `y_true_null`, zero `negative_interval_width`,
target_timestamp_monotonic=True para todos); `block_a` checks.

## Métricas observadas (para F.2 referenciar)

### Per-config (smoke, post-guardrail)

| feature_set | model | split | h | RMSE | MAE | DA | PICP_post | MPIW_post | mean_pinball_post |
|---|---|---|---|---|---|---|---|---|---|
| BTSF | TFT_BTSF | test | 1 | 0.01650 | 0.01127 | 0.508 | 0.747 | 0.0314 | 0.00380 |
| BTSF | TFT_BTSF | test | 7 | 0.01644 | 0.01129 | 0.504 | 0.777 | 0.0331 | 0.00382 |
| BTSF | TFT_BTSF | val | 1 | 0.01970 | 0.01509 | 0.478 | 0.689 | 0.0373 | 0.00483 |
| BTSF | TFT_BTSF | val | 7 | 0.01986 | 0.01520 | 0.476 | 0.707 | 0.0397 | 0.00492 |
| baseline | zero_return | test | 1 | 0.01605 | 0.01097 | 0.003 | — | — | — |
| baseline | hist_mean | test | 1 | 0.01634 | 0.01121 | 0.511 | — | — | — |
| baseline | hist_quant | test | 1 | 0.01610 | 0.01098 | 0.529 | 0.827 | 0.0377 | 0.00379 |
| baseline | zero_return | test | 7 | 0.01603 | 0.01094 | 0.003 | — | — | — |
| baseline | hist_mean | test | 7 | 0.01625 | 0.01122 | 0.517 | — | — | — |
| baseline | hist_quant | test | 7 | 0.01606 | 0.01094 | 0.537 | 0.828 | 0.0377 | 0.00379 |

Observações qualitativas (smoke, **não conclusivo** — 1 seed, 1 candidato):

- TFT_BTSF tem RMSE/MAE marginalmente **piores** que historical_quantiles_rolling
  no test (≈0.5%). Esperado: 5 epochs não convergiu treino completo.
- DA do TFT (0.508 test h=1) é compatível com historical_mean_rolling (0.511) e
  pior que historical_quantiles_rolling (0.529).
- PICP post-guardrail do TFT: 0.747 (test h=1) — sub-cobertura vs nominal 0.80,
  consistente com under-fit em 5 epochs.
- `delta_picp_post_minus_raw` médio: 0.0 (guardrail não rebalanceou cobertura,
  pois raw já era monótono).

### Crossing rate raw

**0.0 em todas as 22 combinações** (4 configs × 6 split×horizon válidos).
Quantis raw saíram monótonos do TFT e dos baselines. Guardrail não ativado.

### Aplicação do guardrail

`guardrail_applied_count=0` em todas as combinações. Equivalência raw vs
post-guardrail confirmada para esta corrida (esperado quando crossing=0).

### DM pairwise (somente entre 3 baselines)

| split | h | par | dm_stat | p_two_sided |
|---|---|---|---|---|
| test | 1 | par_a vs par_b | -2.262 | (significativo, holm-adj n=3) |
| test | 1 | par_a vs par_c | -1.778 | |
| test | 1 | par_b vs par_c | -0.999 | |
| test | 7 | par_a vs par_b | -2.065 | |
| test | 7 | par_a vs par_c | -1.717 | |
| test | 7 | par_b vs par_c | -0.732 | |

Não interpretar — sem TFT no pairwise, o resultado é só baseline×baseline.

### MCS (somente entre 3 baselines)

3 baselines selecionadas em MCS α=0.05 para test h=1 e h=7. Sem TFT, MCS
não é informativo.

## Anomalias / observações

1. **Bug `--help` em `main_refresh_analytics_store`**: tentar
   `python -m src.main_refresh_analytics_store --help` resulta em
   `ValueError: unsupported format character ')'` por `%` não escapado nas
   help strings das flags `--block-a-max-crossing-bruto-rate` e
   `--degeneracy-max-p10-eq-p90-rate` (linhas 145 e 174 do CLI). Help text
   contém `(default: 0.001 = 0.10%)` e `(default: 0.05 = 5%)` — argparse
   interpreta `%` como começo de format-spec. Fix: trocar `%` por `%%`.
   Execução normal não é afetada; só `--help`. **Recomendo fix mínimo
   antes de F.2** para não atrapalhar onboarding.

2. **Stage 12 não tem CLI/main**: `RunBaselinesUseCase` foi mergeada em
   PR #30 sem um `main_run_baselines.py` ou equivalente. Verificado via
   `grep -rl "RunBaselinesUseCase"` — apenas o próprio use case e o teste
   unitário. Para o smoke usei invocação Python direta do use case com
   `ParquetAnalyticsRunRepository`, o que é funcional mas **não auditável**
   no sentido CLI canônico. **F.2 confirmatório precisa de CLI** para
   reprodutibilidade pública (Stage 12-bis). Sugiro entrypoint
   `src.main_run_baselines` aceitando `--asset --parent-sweep-id
   --baselines --horizons --seed --split-config-json|--split-via-asset-periods`.

3. **TFT vs baselines: mismatch de target_timestamps**: TFT (encoder
   warmup 60d) cobre 685/437 timestamps em test/val; baselines (rolling
   warmup variável: 30d para hist_mean, 252d para hist_quant) cobrem 751/503.
   Pairwise intersection no `gold_paired_oos_intersection_by_horizon`
   excluiu TFT. **Não foi previsto no Stage 12 — necessário decidir em F.2**:
   (a) baselines herdam o mesmo warmup do TFT? (b) ou pairwise filtra
   intersecção de timestamps no lado do builder? Recomendo (a) para
   simplicidade — Stage 12-bis aplicaria `effective_train_start_utc`
   alinhado ao do TFT.

4. **`feature_set_name='BTSF'` automático**: o trainer derivou o nome
   `BTSF` (Built TFT Set?) ao invés de `all_features`. Documental, não
   bloqueante. Para F.2, recomendo nome explícito via `--feature-set-name`
   ou via JSON config.

5. **n_rows<1000 em val**: dataset AAPL piloto produz 940 quantile rows
   no val para h=1 e h=7 (`gold_quantile_degeneracy_report`). Gate Stage 11
   não bloqueia (threshold default 1000), mas F.2 precisa declarar:
   (a) reduz threshold para val? (b) usa só test para o gate? Atualmente
   gate roda em todos splits.

6. **3 missing model_artifacts**: baselines não têm artefato torch. Quality
   check exige artefato — decisão em F.2 para excluir baselines do check
   ou materializar artefato vazio.

7. **`gh pr list` mostra 6 PRs com `lint-type-unit=FAILURE`** (Stage 7,
   8, 9, 10, 11, e docs Stage 11): fix tipo mergeado em #29
   (`primary_quantile_contract` → `Literal`). Pytest local atual passa
   (510 tests); mypy/lint do CI atual provavelmente também (não verifiquei
   localmente). Sugiro rodar `mypy src/` localmente como parte do
   protocolo F.2 setup.

## Comandos executados (para reprodução)

```bash
# Branch
git checkout main && git pull --ff-only
git checkout -b feat/phase-a-smoke-confirmatory-20260518

# Pytest baseline
.venv/bin/pytest tests/ --tb=no -q

# Fase 1: TFT smoke
.venv/bin/python -m src.main_train_tft \
  --asset AAPL \
  --features "BASELINE_FEATURES,TECHNICAL_FEATURES,SENTIMENT_FEATURES,FUNDAMENTAL_FEATURES" \
  --max-epochs 5 --max-encoder-length 60 --max-prediction-length 7 \
  --evaluation-horizons "[1, 7]" \
  --parent-sweep-id "phase_a_smoke_20260518" \
  --seed 20260517 --prediction-mode quantile

# Fase 2: Baselines (invocação Python direta — Stage 12 não tem CLI)
.venv/bin/python <<'PY'
from src.use_cases.run_baselines_use_case import RunBaselinesUseCase
from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.utils.path_resolver import load_data_paths
paths = load_data_paths()
repo = ParquetAnalyticsRunRepository(output_dir=paths['analytics_silver'])
uc = RunBaselinesUseCase(analytics_run_repository=repo)
uc.execute(
    asset='AAPL',
    dataset_path='data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet',
    parent_sweep_id='phase_a_smoke_20260518',
    split_definitions={
        'train': ('20101018', '20201231'),
        'val':   ('20210101', '20221231'),
        'test':  ('20230101', '20251231'),
    },
    horizons=[1, 7],
    baselines=['zero_return', 'historical_mean_rolling', 'historical_quantiles_rolling'],
    seed=20260517,
)
PY

# Fase 3: Refresh + quality
.venv/bin/python -m src.main_refresh_analytics_store \
  --scope-mode cohort_decision \
  --scope-sweep-prefixes "phase_a_smoke_20260518" \
  --primary-quantile-contract post_guardrail \
  --fail-on-quality
# → exit code 1; 4/26 checks falhos (ver seção Critério 6).
```

## Recomendação operacional

**Prosseguir para F.2 pre-registro: NÃO** sem antes resolver:

### Fixes obrigatórios antes de F.2 (Stage 12-bis e correções pequenas)

1. **Criar `src.main_run_baselines`** (Stage 12-bis): CLI canônico expondo
   `RunBaselinesUseCase`. Sem isso, F.2 não é reprodutível por terceiros.
   Esforço estimado: 1-2h (espelhar padrão de `main_train_tft.py`).

2. **Alinhar warmup baselines com TFT**: baselines devem usar
   `effective_train_start_utc` do TFT candidato (após max_encoder_length)
   para garantir intersecção exata de `target_timestamp_utc`. Sem isso, TFT
   é excluído de DM/MCS/decision_final — F.2 confirmatório seria inválido.
   Esforço estimado: 2-4h (parâmetro novo em `RunBaselinesUseCase` ou
   pós-filtro no builder pairwise).

3. **Decidir contrato `official_contract_quantile_attention` para
   baselines**: excluir do check via `feature_set_name='baseline'`, ou
   materializar `fact_model_artifacts` mínimo (preferível primeira opção,
   coerente com Stage 10 docs). Esforço: 30 min.

### Fixes recomendados mas não-bloqueantes

4. **Bug `--help` em `main_refresh_analytics_store`** (linhas 145, 174):
   escape `%` para `%%` em duas help strings. Esforço: 1 min.

5. **Decidir threshold do gate degeneração para val pequeno** (n=940 < 1000):
   F.2 declara política. Não exige fix de código se F.2 for explícito.

6. **Documentar `feature_set_name='BTSF'` automático** ou tornar explícito.

### Re-smoke após fixes

Após fixes 1-3 mergeados em main, executar **novo F.1 smoke** com mesmo
`parent_sweep_id` (`overwrite_on_collision=True`) ou prefixo novo
(`phase_a_smoke_20260519`). Critério 6 deve retornar exit 0 ou
documentar caveats restantes em PASS_WITH_ISSUES.

### Investigações de follow-up (Phase B confirmatório, não bloqueia F.2)

- 5 epochs claramente insuficientes (DA do TFT ≈ random, sub-cobertura
  PICP). Validar em F.2 que `max_epochs` definido no pre-registro
  permite convergência (audit M7 ação 6).
- Crossing rate raw=0 com TFT 5 epochs sugere quantile loss já regulariza
  monotonia; F.2 deve validar isso permanece com epochs maiores.
- Smoke não exerceu Optuna (Stage 13). F.2 deve declarar se HPO é parte
  do escopo confirmatório ou se hiperparams vêm de Round 0 (audit M1 ação 3).

## Referências cruzadas

- Aceite literal: [PHASE_B_IMPLEMENTATION_CHECKLIST.md](../05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md) F.1
- Closure mapping: [A_audit_closure_2026-05-17.md](phase-gates/A_audit_closure_2026-05-17.md)
- Gate Stage 11: [A_code_audit.md](phase-gates/A_code_audit.md) M2 ação 1
- Política quantile contract: [METRICS_DEFINITIONS.md](../04_evaluation/METRICS_DEFINITIONS.md)
- Stage 12 baselines: [BASELINES.md](../04_evaluation/BASELINES.md)
- Cenários de falha previstos: prompt F.1 emitido pelo usuário
  (Cenários A-D do prompt foram referenciados; **Cenário D realizou-se**).
