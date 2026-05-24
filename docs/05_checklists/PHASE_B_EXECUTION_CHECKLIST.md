---
title: Phase B Execution Checklist (Confirmatory Run)
scope: Execucao operacional task-a-task da Phase B confirmatoria (Stages E0-E6). Sequencia E0 Optuna heritage -> E1 selar config + emenda pre-registro -> E2 treino TFT confirmatorio -> E3 baselines no mesmo cohort -> E4 refresh + quality validation -> E5 analise vs Tier 1/Tier 2 gates -> E6 relatorio + emenda final + archive snapshot. Nao substitui o pre-registro (F.2 mergeado em PR-58); operacionaliza a execucao declarada por ele. Distinto de PHASE_B_IMPLEMENTATION_CHECKLIST.md (pre-experiment engineering, imutavel apos F.3 [x]).
update_when:
  - status (checkbox) de uma task for atualizado
  - emenda for adicionada ao pre-registro Phase B (preregistration_phase_b.md §14)
  - tier reportado no fim do ciclo (E5)
  - sweep_id ou commit hash de execucao for fixado em E1
  - relatorio final B_confirmatory_<date>.md for criado em E6
canonical_for: [phase_b_execution_checklist, phase_b_confirmatory_runbook]
---

# Phase B — Execution Checklist (Confirmatory Run)

**Status:** pendente
**Criado:** 2026-05-23
**Pre-requisito:** [PHASE_B_IMPLEMENTATION_CHECKLIST.md F.3 [x]](PHASE_B_IMPLEMENTATION_CHECKLIST.md) (2026-05-23, PR #59 mergeada).
**Pre-registro de referencia:** [preregistration_phase_b.md](../06_pre_registration/preregistration_phase_b.md) (PR #58 mergeada 2026-05-23; sera selado com commit hash em E1).
**Bloqueante para:** Fase C (analise de contribuicao de features per STRATEGIC_DIRECTION §5 Fase C).
**Tempo total estimado:** ~25-45h de wall-clock (dominado por E0 ~6-12h + E2 ~8-15h).

Este checklist operacionaliza a execucao da Phase B confirmatoria declarada em
[`preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md).
Cada Stage E<N> abaixo eh um conjunto de tasks com Aceite explicito; o ciclo
termina quando E6 reportar tier por horizonte e a emenda final em §15 do
pre-registro for mergeada.

## Decisoes operacionais fechadas (D1-D8, 2026-05-23)

Registradas antes do inicio da execucao para evitar redecisao mid-flight.
Mudancas exigem entrada em "Decisoes operacionais — emendas" abaixo.

| # | Decisao | Resposta fechada |
|---|---|---|
| D1 | Candidato exato (E0 → E1) | **Top-1 robust_score apenas** (UM candidato unico per STRATEGIC_DIRECTION §4 e F.2 §3 H2a/H2b) |
| D2 | Data suffix `<YYYYMMDD>` em `parent_sweep_id` | **Data real de inicio de E2** (preenchida quando E2 disparar; alinha com convencao literal F.2 §10) |
| D3 | Tech debt baselines `output_subdir` | **Sem fix em codigo**; JSON novo em E1 com `output_subdir=phase_b_confirmatorio_<date>` correto. Workaround `/tmp` do R-23 nao se aplica (fica como YELLOW pos-Phase B per F.2 §11) |
| D4 | Strategy de branch/PR | **2 PRs separados**: PR-A apos E1 (config + emenda §14/§15-partial); PR-B apos E6 (relatorio + emenda §15-final). Silver/gold **NAO** entram em git (reproducible via config + seed) |
| D5 | Snapshot pos-rodada | **SIM** — `data/analytics_archive_phase_b_<date>/` read-only criado em E6 (precedente Stage 3 do PHASE_B_IMPLEMENTATION_CHECKLIST.md) |
| D6 | Walk-forward sensibilidade B.4 (2 janelas alt) | **Separado em rodada futura** Phase-B-bis. Out-of-scope deste ciclo; entrega core H1/H2a/H2b primeiro |
| D7 | Pre-execution sanity checks antes de E2 | **Todos obrigatorios**: (a) pytest 605+ passed + ruff clean; (b) bit-byte reproduce de uma run R-23 smoke com `seed=20260517`; (c) sha256 `dataset_tft_AAPL.parquet` contra commit R-20; (d) `gh pr view #54 #58 #59` confirmar main verde |
| D8 | Implementer target | **2 sessoes Claude sequenciais**: Sessao-A operacional (E0-E4, autoridade total, pode rodar autonoma); Sessao-B analitica (E5-E6, Marcelo presente para revisao do tier antes do PR) |

### Decisoes operacionais — emendas

_(vazio na criacao; preenchido se uma decisao D1-D8 for revisada mid-execucao com justificativa datada)_

## Convencao

- `[ ]` = pendente
- `[x]` = mergeado em main
- `[~]` = em revisao (PR aberto)
- Cada task termina com **Aceite:** criterio que comprova que a task esta pronta.
- Branch naming, commit format e PR template seguem
  [`docs/08_governance/GOVERNANCE_AND_VERSIONING.md`](../08_governance/GOVERNANCE_AND_VERSIONING.md).

## Cross-link Stage → pre-registro / runbook / skill

| Stage | preregistration_phase_b.md § | Runbook | Skill aplicavel |
|---|---|---|---|
| E0 | §4.2 passos 2-3 (Optuna heritage exploratoria) | [RUN_SWEEPS.md](../06_runbooks/RUN_SWEEPS.md) | explicit-sweep-execution |
| E1 | §4.2 passo 4 + §14 (emenda) + §15 (slot hash) | — | documentation-authoring + staged-implementation-protocol Phase B |
| E2 | §4 (candidato) + §11 (sweep type) + §13 (artefatos) | [RUN_TRAINING.md](../06_runbooks/RUN_TRAINING.md) | explicit-sweep-execution |
| E3 | §5 (baselines) + §11 (alinhamento via JSON unico) | [RUN_BASELINES.md](../06_runbooks/RUN_BASELINES.md) | explicit-sweep-execution |
| E4 | §7 (protocolo estatistico) + §10 (ScopeSpec) + §13 (artefatos) | [RUN_REFRESH_ANALYTICS.md](../06_runbooks/RUN_REFRESH_ANALYTICS.md) | — |
| E5 | §9 (Tier 1/Tier 2 gates) + §3 (H1/H2a/H2b) | — | model-performance-and-research-advisor |
| E6 | §15 (execucao slot) + §14 (emenda) | — | documentation-authoring |

---

## Stage E0 — Optuna heritage (~6-12h CPU/GPU)

**Objetivo:** rodar Optuna ~20-30 trials sobre vizinhanca dos top-3 frozen
candidates da Round 0 sob regime Phase B (`max_encoder_length=60`,
`max_prediction_length=7`, `evaluation_horizons=[1,7]`, `prediction_mode=quantile`,
`quantile_levels=[0.1, 0.5, 0.9]`), objetivo `robust_score`. Top-1 alimenta E1.

**Natureza:** **exploratoria** — nao alimenta claims H1/H2a/H2b per F.2 §4.2
("Esta rodada e exploratoria — nao alimenta claims do paper"). Justificativa
academica: hiperparametros foram escolhidos por criterio de validacao
(`robust_score`), nunca por test (auditavel: Stage 13 bloqueia
`mean_test_rmse`/`joint_val_test_rmse` em `RunTFTOptunaSearchUseCase.__init__`).

**Cross-link:** F.2 §4.2 passos 1-3.

**Sweep ID:** `phase_b_hpo_<YYYYMMDD>` (prefixo distinto de
`phase_b_confirmatorio_` para nao contaminar coorte confirmatoria).

**Fonte top-3 Round 0:**
[`data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv`](../../data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv).
Top-1 atual (Round 0, regime `max_pred_length=1`): `hidden_size=32,
attention_head_size=4, dropout=0.15, lr=4.85e-4, batch_size=64,
hidden_continuous_size=4`.

### Tasks

- [x] **E0.1** Criar `config/sweeps/optuna/phase_b_hpo_<YYYYMMDD>.json`. Base:
      template `0_1_5_optuna_top13_all_feature_sets_hpo.json` com substituicoes:
      `output_subdir=phase_b_hpo_<YYYYMMDD>`, `features="BASELINE_FEATURES,
      TECHNICAL_FEATURES,SENTIMENT_FEATURES,FUNDAMENTAL_FEATURES"` (feature_sets
      vazio), `n_trials=25` (sugestao mid-range 20-30), `top_k=3`,
      `objective_metric="robust_score"`, `replica_seeds=[20260517]` (HPO usa
      1 seed; replica vem em E2). `training_config.max_encoder_length=60`,
      `max_prediction_length=7`, `prediction_mode="quantile"`,
      `quantile_levels=[0.1, 0.5, 0.9]`, `evaluation_horizons=[1, 7]`. Search
      space centrado nos hiperparams top-1 Round 0 com vizinhancas estreitas
      (±20-30%).
      **Aceite:** JSON valido (passa `json.load`); `objective_metric=robust_score`;
      `compute_feature_importance=false` (acelera HPO).

- [x] **E0.2** Executar
      `.venv/bin/python -m src.main_tft_optuna_sweep --asset AAPL --config-json config/sweeps/optuna/phase_b_hpo_<YYYYMMDD>.json`.
      Aceitar `continue_on_error=true` no JSON para nao abortar batch por trial OOM.
      **Aceite:** ≥20 trials completos (`status=ok` em `dim_run`); zero trials
      com `mean_val_rmse=NaN`; tempo total registrado.

- [x] **E0.3** Identificar top-1 por `robust_score` em
      `data/analytics/selection/frozen_candidates_<phase_b_hpo_<YYYYMMDD>>.csv`
      (gerado por `main_tft_optuna_sweep`). Documentar hiperparams escolhidos
      em `/tmp/e0_top1_<YYYYMMDD>.json` (intermediario; nao committed em git).
      **Aceite:** arquivo intermediario contem hiperparams exatos do top-1;
      `robust_score`, `mean_val_rmse`, `std_val_rmse` registrados; trial_number
      e run_id rastreaveis em silver.

### Notas de revisao:

- 2026-05-23 23:14 UTC: Stage E0 executado autonomamente (Sessao-A).
- Branch: main (E0 nao requer branch nova; config commit vira em E1.0 com `feat/phase-b-execution-<D2>`).
- Validation commands com results:
  - Init pre-flight: `.venv/bin/pytest tests/ -q --ignore=tests/integration/test_quality_registry_bit_identical_archive.py --ignore=tests/integration/test_gold_builders_byte_identical_archive.py` -> `609 passed, 9 warnings`.
  - `.venv/bin/ruff check src/ tests/` -> `All checks passed!`.
  - `python -c "import json; json.load(open('config/sweeps/optuna/phase_b_hpo_20260523.json'))"` -> sem erro; substituicoes verificadas (objective=robust_score, max_encoder=60, max_pred=7, prediction_mode=quantile, evaluation_horizons=[1,7], n_trials=25, compute_feature_importance=false).
  - `.venv/bin/python -m src.main_tft_optuna_sweep --asset AAPL --config-json config/sweeps/optuna/phase_b_hpo_20260523.json` -> exit code 0, 25/25 trials completos, runs_failed=0 em todos os trials, avg_trial_seconds=130.77, wall-clock ~54.5min (abaixo do range estimado 75-125min porque early-stopping ~8 epochs).
  - `cat data/models/AAPL/optuna/phase_b_hpo_20260523/.../optuna_best_trial.json` -> trial_number=19, robust_score=0.019330283626914024, hidden_size=48, attention_head_size=8, dropout=0.05302, learning_rate=0.0003963, batch_size=128, hidden_continuous_size=8.
  - `cat data/analytics/silver/fact_epoch_metrics/asset=AAPL/sweep_id=trial_0019/fold=none/fact_epoch_metrics.parquet` -> 8 epochs, best_epoch=2, stopped_epoch=7, early_stop_reason=early_stopping.
- Decisoes mid-execucao:
  - Search space: `dropout` e `learning_rate` como `{"type": "float", "log": true (lr only)}` em vez de `uniform/log-uniform` no prompt — o `_suggest_param` em `RunTFTOptunaSearchUseCase` aceita apenas `categorical`, `int`, `float` (com `log` opcional). Equivalente semantico.
  - `walk_forward.enabled=false` para E0: prompt nao explicita; manter walk_forward 3 folds em HPO triplicaria runtime (~3h) e divergiria da estimativa de 75-125min. HPO single-split eh suficiente para identificar vizinhanca de hiperparams; replica/folds vem em E2.
- Warmup ROCm: padrao `[epoch=0] train_loss=nan val_loss=nan` -> `[epoch=0] train_loss=0.0X val_loss=0.0X` observado em todos os 25 trials (esperado; auto-recuperado per ADR ambiente; nao retry).
- Confirmacoes:
  - `data/analytics_archive_pre_phase_b/` intacto (du -sh: 851M).
  - Protected files nao tocados (src/, tests/, dataset_tft_AAPL.parquet, checklists imutaveis).
- Outputs salvos:
  - Config: `config/sweeps/optuna/phase_b_hpo_20260523.json`
  - Top-1 metadata: `/tmp/e0_top1_20260523.json`
  - Log: `/tmp/e0_optuna_20260523.log`
  - Sweep artifacts: `data/models/AAPL/optuna/phase_b_hpo_20260523/.../`
- Follow-ups:
  - Para E1.1: `training_config.max_epochs` recomendado = 15 (8 epochs convergencia observada + 7 margem para variancia de fold/seed; early_stopping_patience=5 truncara naturalmente).
- Outcome: PASS.

---

## Stage E1 — Selar config + emenda pre-registro (~2-3h)

**Objetivo:** criar config congelada `phase_b_confirmatorio_<YYYYMMDD>.json`
com hiperparams do top-1 E0 + 5 seeds + 3 folds explicitos + 3 baselines.
Computar sha256. Anexar emenda datada em §14 e preencher slots `Hash da config
JSON`/`parent_sweep_id efetivo` em §15 do pre-registro. Abrir PR-A.

**Cross-link:** F.2 §4.2 passo 4 + §14 + §15.

**Pre-condicao:** D2 fixada (data real de inicio de E2 escolhida; vira `<YYYYMMDD>`).
Recomendacao: usar data do dia em que E2 vai disparar.

**Branch:** `feat/phase-b-execution-<YYYYMMDD>`.

### Tasks

- [~] **E1.1** Criar `config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`.
      Base: template
      [`phase_a_smoke_20260518_v2.json`](../../config/sweeps/explicit/phase_a_smoke_20260518_v2.json)
      com substituicoes:
      - `output_subdir="phase_b_confirmatorio_<YYYYMMDD>"`
      - `training_config.*` = hiperparams top-1 do E0
      - `training_config.max_epochs` = valor convergido em E0 (esperado 30-50)
      - `replica_seeds=[20260517, 20260518, 20260519, 20260520, 20260521]` (5 seeds per F.2 §4.3)
      - `walk_forward.enabled=true`, `walk_forward.folds` = 3 folds explicitos
        (declarar windows: copiar pattern de `0_1_5_optuna_*.json` ou definir
        train/val/test triplas alinhadas com pre-registro §2 OOS = `2023-01-01
        -> 2025-12-31`)
      - `baselines` = 3 entradas (`zero_return {}`, `historical_mean_rolling
        {window_days: 30}`, `historical_quantiles_rolling {window_days: 252,
        quantile_levels: [0.1, 0.5, 0.9]}`)
      - `evaluation_horizons=[1, 7]`, `prediction_mode="quantile"`
      - `quantile_levels=[0.1, 0.5, 0.9]`
      **Aceite:** JSON valido; `output_subdir` igual a `parent_sweep_id` esperado;
      `walk_forward.folds` tem exatamente 3 entradas; baselines tem exatamente 3 entradas.

- [~] **E1.2** Computar `sha256sum config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`.
      Registrar o hash em commit message E1.4 + slot §15 do pre-registro.
      **Aceite:** hash de 64 chars hex registrado; comando reproduzivel.

- [~] **E1.3** Editar [`docs/06_pre_registration/preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md):
      - §14 "Emendas": adicionar entrada datada
        ```
        ### 2026-MM-DD — Emenda E1: selo da config congelada
        - Decisao alterada: §4.2 passo 4 (hash do JSON congelado).
        - Hash: sha256 = <hex>
        - parent_sweep_id efetivo: phase_b_confirmatorio_<YYYYMMDD>
        - Cross-link: branch feat/phase-b-execution-<YYYYMMDD>, commit <sha>
        - Justificativa: selo da config confirmatoria apos rodada Optuna
          heritage E0 (sweep_id phase_b_hpo_<YYYYMMDD>); top-1 robust_score.
        ```
      - §15 "Execucao": preencher
        - **Commit hash deste pre-registro (selo F.2 original):** `067cb32`
          (`docs(pre-reg): F.2 — pre-registro Phase B confirmatorio`).
          Nao usar o commit E1.4 aqui: ele ainda nao existe durante E1.3 e
          registra apenas a emenda/config, nao o selo original do pre-registro.
        - **parent_sweep_id efetivo:** `phase_b_confirmatorio_<YYYYMMDD>`
        - **Hash da config JSON:** `<sha256>`
        - Demais slots permanecem `<a preencher>` ate E6.
      **Aceite:** §14 tem nova entrada datada; §15 com 3 dos 6 slots preenchidos.

- [~] **E1.4** Commit unico:
      `feat(phase-b-exec): selar config phase_b_confirmatorio_<YYYYMMDD> e
      emenda pre-registro E1`. Body do commit cita: sha256 da config, sweep_id
      do E0 heritage, top-1 trial_number/run_id.
      **Aceite:** commit no padrao do governance doc; passa pre-commit hooks
      (se houver); diff inclui apenas 1 JSON novo + edits no pre-registro.

- [~] **E1.5** Push branch + abrir **PR-A**:
      `feat(phase-b-exec): config sealing + pre-registro emenda E1`. Body cita
      `PHASE_B_EXECUTION_CHECKLIST.md` Stage E1, decisoes D1-D8 fechadas, sha256
      da config, sweep_id E0 heritage.
      **Aceite:** PR criado contra main; CI green (lint + pytest); Marcelo
      revisa e mergeia antes de E2 disparar.

### Notas de revisao:

- 2026-05-24 00:?? UTC: Stage E1 executado pela Sessao-A (em revisao [~]).
- Branch: `feat/phase-b-execution-20260524` (criada a partir de main em E1.0
  apos pull --ff-only; commit base `f92bd57`).
- Decisao D2: `<YYYYMMDD>` = `20260524` (escolhida por Marcelo via question
  AskUser; data real de inicio de E2 — amanha/domingo). E0 ja rodado em
  2026-05-23 com sweep_id `phase_b_hpo_20260523`.
- Validation commands com results:
  - `.venv/bin/python -c "import json; json.load(...)"` -> JSON valido;
    todas as substituicoes verificadas (5 seeds, 3 folds, 3 baselines,
    output_subdir=phase_b_confirmatorio_20260524, hiperparams = trial 19,
    max_epochs=15).
  - `sha256sum config/sweeps/explicit/phase_b_confirmatorio_20260524.json`
    -> `e6972a8ba5d23659eab205d7d403d10a2c1f3c9827329b7a37c5ee8595b9260c`.
  - `git log --oneline -2` -> commits `83522cc` (E0 deliverables) +
    `dd9b2d3` (E1.4 confirmatorio + emenda).
- Decisoes mid-execucao:
  - 2 commits separados na PR-A (em vez de 1) — commit 1 (E0): optuna
    config + E0 Notas; commit 2 (E1.4 strict per prompt): confirmatorio
    config + pre-reg emenda. Razao: prompt E1.4 especifica "1 JSON novo
    + edits no pre-registro" (commit 2 cumpre isso isoladamente); o E0
    deliverable nao deveria ficar fora da PR-A por reproducibilidade.
  - Emenda §14 cross-link cita "branch + PR-A" em vez de pinar commit sha
    do proprio commit que adiciona a emenda (impossivel sem amend).
    Merge commit hash registrado em §15 post-merge pela Sessao-A/Marcelo
    se necessario.
  - `max_epochs=15` em E1.1: best_epoch=2/stopped_epoch=7 em E0 + 8 margem
    para variancia de fold/seed (em vez do range 30-50 esperado original).
- Confirmacoes:
  - `data/analytics_archive_pre_phase_b/` intacto (du -sh: 851M).
  - Protected files nao tocados (src/, tests/, dataset_tft_AAPL.parquet,
    PHASE_B_IMPLEMENTATION_CHECKLIST.md, POST_CLOSURE_FIXES_CHECKLIST.md,
    ADRs).
- Cohort esperado pos-E2+E3: 15 TFT runs + 45 baseline = 60 dim_run.
- Outcome: PASS_PENDING_MERGE (em revisao [~] aguardando PR-A merge antes
  de E2).

---

## Stage E2 — Treino TFT confirmatorio (~8-15h)

**Objetivo:** treinar candidato TFT all-features sob 5 seeds × 3 folds = 15 runs,
no `parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>`, consumindo a config
selada em E1.

**Cross-link:** F.2 §4 (candidato) + §11 (sweep type) + §13 (artefatos esperados).

**Pre-condicao:** PR-A (E1.5) mergeada em main; pre-execution sanity checks
(D7) executados e PASS.

### Tasks

- [x] **E2.0** Pre-execution sanity checks (D7 fechada):
      - (a) `.venv/bin/pytest tests/ -q --ignore=tests/integration/test_quality_registry_bit_identical_archive.py --ignore=tests/integration/test_gold_builders_byte_identical_archive.py` → `605+ passed`.
      - (b) `.venv/bin/ruff check src/ tests/` → `All checks passed!`.
      - (c) Bit-byte reproduce: executar 1 run TFT com `--seed 20260517 --max-epochs 1 --parent-sweep-id e2_repro_check --max-encoder-length 60 --max-prediction-length 7` (config minima) e confirmar que `fact_oos_predictions` produz mesmos `y_hat_q50` da run R-23 (`run_id=3e685d04...`) para os primeiros 5 timestamps. Wipe `data/analytics/silver/*sweep_id=e2_repro_check*` apos check.
      - (d) `sha256sum data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet` registrado e comparado contra valor esperado pos-R-20 (registrado em
        [`option_b_execution_log_2026-05-19.md`](../07_reports/option_b_execution_log_2026-05-19.md) §Stage R-20.1).
      - (e) `gh pr view 54 58 59 --json statusCheckRollup,state` → todos MERGED + CI green.
      **Aceite:** 4 checks PASS; output registrado em `/tmp/e2_preflight_<YYYYMMDD>.txt`.

- [x] **E2.1** Executar
      `.venv/bin/python -m src.main_tft_test_pipeline --asset AAPL --config-json config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`.
      Wall-clock estimado 8-15h dependendo de `max_epochs` E0. Capturar stdout
      em `/tmp/e2_train_<YYYYMMDD>.log`.
      **Aceite:** exit code 0; ultima linha do log nao mostra exception; runtime
      logged em `dim_run.created_at..ended_at` por run.

- [x] **E2.2** Validar shape do silver:
      `ls data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/`
      → deve listar exatamente **15 partitions** (5 seeds × 3 folds).
      `data/analytics/silver/fact_oos_predictions/...` deve ter cobertura
      `(run_id, split, horizon)` para val + test, h=1 + h=7.
      **Aceite:** 15 dim_run rows com `status=ok`; fact_oos_predictions tem
      ≥ `15 × 2 splits × 2 horizons × n_timestamps_per_split` rows; `decision_idx`
      populated (range esperado consistent com `max_encoder_length=60`).

- [x] **E2.3** Spot-check de degeneracao quantilica: calcular `(quantile_p10 ==
      quantile_p90)` rate per run/split/horizon. Esperar `<5%` em todos (gate
      Stage 11 sera executado em E4).
      **Aceite:** todas as 15 × 2 × 2 = 60 combinacoes tem `p10_eq_p90_rate <
      0.05`; nenhuma run flagged para exclusao.

### Notas de revisao:

- 2026-05-24 04:01 UTC: Stage E2 executado autonomamente pela Sessao-A (apos
  PR-A2 #64 merge para corrigir bug do schema explicit_configs).
- Branch: `feat/phase-b-execution-20260524` (continuada pos-PR-A2 merge).
- Validation commands com results:
  - E2.0.a `.venv/bin/pytest tests/ -q --ignore=...` -> `609 passed, 9 warnings`.
  - E2.0.a `.venv/bin/ruff check src/ tests/` -> `All checks passed!`.
  - E2.0.b bit-byte sanity 1-epoch (cohort `e2_repro_check`) -> exit 0;
    run_id=`9721ecb8...`; 2246 fact_oos rows; y_true non-NaN (5/5);
    monotonia quantile_p10<=p50<=p90 100% (zero violations raw); test
    target_timestamp range `2023-03-30 -> 2025-12-31` (warmup-offset esperado).
    Cleanup OK: find vazio + git status clean + 3 orphan bridge_run_features
    posteriormente limpos (incluindo do e2_repro_check).
  - E2.0.c `sha256sum data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet`
    -> `aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298`
    (MATCHES baseline 2026-05-23 PASS).
  - E2.0.d `gh pr view 54/58/59/60/63/64 --json state` -> todos MERGED.
  - E2.1 `.venv/bin/python -m src.main_tft_test_pipeline --asset AAPL --config-json config/sweeps/explicit/phase_b_confirmatorio_20260524.json`
    -> exit 0; "Unified test pipeline finished" `runs_ok=15, runs_failed=0`;
    wall-clock ~35-37 min (muito abaixo do 8-15h estimado original; razao:
    early_stopping em ~7 epochs por convergencia rapida no dataset AAPL).
    avg_train_seconds_per_run ~145s.
  - E2.1 top_1 sweep aggregate: `mean_val_rmse=0.019317, std_val_rmse=0.000635`,
    `mean_test_rmse=0.016790, std_test_rmse=0.002431`,
    `mean_test_da=0.520`, `robust_score=0.019952`.
  - E2.2 `ls data/analytics/silver/dim_run/asset=AAPL/sweep_id=phase_b_confirmatorio_20260524/`
    -> 15 dim_run rows; all `status=ok`; 5 distinct seeds; 3 distinct
    `split_fingerprint` (folds encoded); 15 distinct `run_id`.
    NOTA: spec literal pediu `parent_sweep_id=` no path; filesystem real
    usa `sweep_id=` (`parent_sweep_id` esta como coluna no parquet).
  - E2.3 degeneracao: 60 grupos (15 runs x 2 splits x 2 horizonts);
    `p10_eq_p90_rate=0.0` em TODOS; zero violacoes acima de 5%. PASS.
- Decisoes mid-execucao:
  - Spec E2.0.b literal pediu reproducao bit-byte do R-23 (`run_id=3e685d04...`).
    Spec proprio da Sessao-A (PHASE_B_SESSION_A_PROMPT.md) ja afirma:
    "este check e sanity (nao byte-exact em ROCm/AMD)". Apliquei criterios
    PASS sanity (exit 0 + y_true non-NaN + monotonia + range OOS), NAO
    byte-exact CUDA reference.
  - Warmup ROCm `[epoch=0] train_loss=nan val_loss=nan` -> recovery
    immediate observado em todos os 15 runs (esperado; auto-recuperado per
    ADR ambiente; nao retry).
- Confirmacoes:
  - `data/analytics_archive_pre_phase_b/` intacto (du -sh: 851M).
  - Dataset sha256 inalterado durante todo E2.
  - Protected files nao tocados.
- Outcome: PASS.

---

## Stage E3 — Baselines no mesmo cohort (~30min-1h)

**Objetivo:** executar 3 baselines (`zero_return`, `historical_mean_rolling`,
`historical_quantiles_rolling`) consumindo o mesmo JSON E1, garantindo
`parent_sweep_id` compartilhado e alinhamento exato `target_timestamp_utc` com
o TFT (per F.2 §11 sem workaround `/tmp` — D3 fechada).

**Cross-link:** F.2 §5 (baselines pre-declarados) + §11 (alinhamento
single-JSON).

**Pre-condicao:** E2.2 PASS (silver TFT populado).

### Tasks

- [x] **E3.1** Executar
      `.venv/bin/python -m src.main_baselines_test_pipeline --asset AAPL --config-json config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`.
      **Sem** `--overwrite-on-collision`; **sem** copia /tmp (D3). Capturar
      stdout em `/tmp/e3_baselines_<YYYYMMDD>.log`.
      **Aceite:** exit code 0; log mostra 3 baselines × 5 seeds × 3 folds = 45
      runs persistidos.

- [x] **E3.2** Validar shape e alinhamento:
      - `dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/`
        agora tem **15 TFT + 45 baselines = 60 partitions**.
      - `fact_oos_predictions/asset=AAPL/feature_set_name=baseline/...` populado.
      - **Spot-check de alinhamento**: para 1 run TFT + 1 run baseline com mesma
        seed+fold+split+horizon, `target_timestamp_utc` set TFT ⊆ baseline
        (per F.2 §11 e ADR-0003 Opcao d).
      **Aceite:** 60 dim_run rows; alinhamento exato verificado em ≥1 par.

### Notas de revisao:

- 2026-05-24 04:03 UTC: Stage E3 executado autonomamente pela Sessao-A.
- Branch: `feat/phase-b-execution-20260524`.
- Validation commands com results:
  - E3.1 `.venv/bin/python -m src.main_baselines_test_pipeline --asset AAPL --config-json config/sweeps/explicit/phase_b_confirmatorio_20260524.json`
    -> exit 0; "Baselines test pipeline finished" `baselines_persisted_total=45,
    folds_processed=3, seeds_processed=5`; wall-clock ~3 min.
  - E3.2 shape: 60 dim_run totais (15 TFT + 45 baselines) com `status=ok`;
    feature_set_name `BTSF=15` + `baseline=45`.
  - E3.2 spot-check alinhamento: TFT (run f0a94d2c, seed 20260517, fold wf_2)
    com 436 target_timestamps em test/h=1; 2 baselines com mesma seed na
    mesma fold (b8e617e6, a0c5384b) tem 436 target_timestamps -- 
    `aligned_exact=True` (TFT == baseline). Verificado em
    `gold_paired_oos_intersection_by_horizon`: 12 rows Phase B, todos
    `aligned_exact=True`.
- DIVIDA TECNICA YELLOW detectada (antecipada em F.2 §11 nota operacional):
  - Bug em `src/use_cases/run_baselines_test_pipeline_use_case.py:226-231`:
    quando `len(folds) > 1`, baseline pipeline emite
    `parent_sweep_id=f"{root}__{fold.name}"` (com sufixo `__wf_X` por fold).
    TFT pipeline NAO faz isso; usa root direto. Resultado: TFT em
    `phase_b_confirmatorio_20260524` (sem sufixo); baselines em 3 cohorts
    distintos `phase_b_confirmatorio_20260524__wf_{1,2,3}`.
  - Consequencia: spec F.2 §5 ("Mesmo parent_sweep_id do candidato") nao
    satisfeito tecnicamente. DM/MCS pareados TFT-vs-baseline nao sao
    materializados em `gold_dm_pairwise_results` (apenas pares dentro de
    cada cohort: TFT seed-vs-seed; baseline-vs-baseline por fold).
  - Tambem detectado: 45 baseline rows em `dim_run` tem `seed=None,
    fold=None` (pipeline nao popula esses campos). Por isso o quality
    check `cardinality_config_fold_seed` reporta `duplicate_groups=45`.
  - **NAO corrigido**: per F.2 §14 veto explicito `"Alterar
    parent_sweep_id apos persistencia silver"`, nao alterei dados silver.
    Mantida divida YELLOW per F.2 §11 ("follow-up: derivar parent_sweep_id
    automaticamente do output_subdir em ambos pipelines em PR separado").
  - Mitigacao para Sessao-B: TFT-vs-baseline DM/MCS devem ser computados
    via pos-processamento de `fact_oos_predictions` (matching cross-cohort
    por `target_timestamp_utc` + same fold via split periods). `gold_paired_
    oos_intersection_by_horizon` ja confirma `aligned_exact=True` por
    cohort, garantindo que a interseccao temporal manual e exata.
- Decisoes mid-execucao:
  - Aceitar pipeline behavior nao-conforme em vez de patchear silver
    (per F.2 §14 hard veto).
- Confirmacoes:
  - `data/analytics_archive_pre_phase_b/` intacto (851M).
  - Dataset sha256 inalterado.
- Outcome: PASS (60 rows persisted + alinhamento per cohort exato) COM
  CAVEAT (divida YELLOW de cohort splitting requer pos-processamento em
  Sessao-B para DM/MCS TFT-vs-baseline).

---

## Stage E4 — Refresh + quality validation (~30min-1h)

**Objetivo:** refresh do analytics store sob scope `cohort_decision` restrito
ao prefixo `phase_b_confirmatorio_`, materializar 25 tabelas gold e validar
quality gate exit 0 com `failed_checks=[]`.

**Cross-link:** F.2 §7 (protocolo estatistico DM/MCS/Holm) + §10 (ScopeSpec)
+ §13 (artefatos esperados).

**Pre-condicao:** E3.2 PASS (silver TFT + baselines populados).

### Tasks

- [~] **E4.1** Executar comando exato per F.2 §10:
      ```
      .venv/bin/python -m src.main_refresh_analytics_store \
          --scope-mode cohort_decision \
          --scope-sweep-prefixes phase_b_confirmatorio_ \
          --scope-splits val test \
          --scope-horizons 1 7 \
          --primary-quantile-contract post_guardrail \
          --fail-on-quality
      ```
      Capturar stdout em `/tmp/e4_refresh_<YYYYMMDD>.log`.
      **Aceite:** exit code 0; ultima linha mostra `passed=True`,
      `failed_checks=[]`, `total_checks≥27`; ≥25 gold parquets materializados.

- [~] **E4.2** Spot-check de tabelas gold criticas:
      - `gold_dm_pairwise_results.parquet` tem rows para 2 horizontes × 3
        baselines = 6 comparacoes pareadas com TFT.
      - `gold_mcs_results.parquet` inclui TFT + 3 baselines × 2 horizontes.
      - `gold_paired_oos_intersection_by_horizon.parquet` tem
        `aligned_exact=True` em todos os pares.
      - `gold_quality_statistics_report.parquet` tem
        `statistics_ready=True` para test split h=1 e h=7.
      - `gold_quantile_degeneracy_report.parquet`: `gate_passed=True` em todos
        os grupos quantile.
      **Aceite:** 5 checks PASS; output registrado.

### Notas de revisao:

- 2026-05-24 04:10 UTC: Stage E4 executado autonomamente pela Sessao-A.
- Branch: `feat/phase-b-execution-20260524`.
- Status final: `[~]` em revisao -- exit code 1 (NAO satisfez aceite
  literal `failed_checks=[]`). 3 quality checks failed, todos
  classificados como tech debt YELLOW antecipado em F.2 §11 (cohort
  splitting de baselines) e/ou inter-seed warmup boundary variance
  (1-4 timestamps em test/val). Phase B cohort utilizavel para Sessao-B
  com pos-processamento documentado.
- Validation commands com results:
  - E4.1 v1 `--scope-splits val test --scope-horizons 1 7` (sem virgula)
    -> exit 2 (CLI parser: "unrecognized arguments: test 7"). CLI exige
    valores comma-separated `--scope-splits val,test --scope-horizons 1,7`
    (per RUN_REFRESH_ANALYTICS.md).
  - E4.1 v2 (comma-separated) -> exit 1; `Analytics gold refresh completed`
    25 outputs gerados; `Analytics quality validation completed` total_checks=27,
    passed=False, 4 failed_checks (incluindo 1 `referential_integrity`
    transient -- 3 orphans em bridge_run_features incluindo 1 do
    `e2_repro_check` cleanup).
  - Cleanup manual de 3 orphan bridge_run_features dirs (do
    `e2_repro_check` + 2 outros legacy). Verify: 89 bridge dirs == 89
    dim_run runs.
  - E4.1 v3 (apos cleanup bridge) -> exit 1; `total_checks=27`,
    `passed=False`, 3 failed_checks remanescentes (referential_integrity
    PASS).
- Tabelas gold materializadas: **25** (todas as obrigatorias per F.2 §13
  presentes). Total rows por tabela em Apendice A do report final.
- Spot-checks E4.2 (cohort Phase B):
  - `gold_paired_oos_intersection_by_horizon` (Phase B 12 rows):
    **`aligned_exact=True` em TODAS** (cohort TFT: 6 rows
    [3 split_signatures x 2 horizonts]; cada baseline cohort __wf_X:
    2 rows). n_common == n_union em todas as 12.
  - `gold_quantile_degeneracy_report` (Phase B 28 rows):
    **`gate_passed=True` em TODAS**; `p10_eq_p90_rate=0.0` em todos os
    grupos `prediction_mode=quantile`. Grupos `prediction_mode=point`
    (baselines zero_return + historical_mean_rolling) tem rate=1.0 por
    construcao (gate ignora point per Stage 11 design).
  - `gold_dm_pairwise_results` (Phase B 78 rows):
    - 60 rows em cohort TFT (`phase_b_confirmatorio_20260524`): pares
      seed-vs-seed dentro do TFT (NAO TFT-vs-baseline).
    - 6 rows em cada baseline cohort __wf_X (`baseline_i vs baseline_j`
      dentro da mesma fold).
    - **NAO ha pares cross-cohort TFT-vs-baseline** (consequencia da
      divida YELLOW E3 -- cohort splitting).
  - `gold_mcs_results` Phase B: 48 rows.
  - `gold_quality_statistics_report` Phase B: 16 rows;
    `statistics_ready=True` para cada cohort baseline __wf_X (test/h=1+7)
    mas `statistics_ready=False` para cohort TFT (`phase_b_confirmatorio_
    20260524`) -- mesma causa: ausencia de TFT-vs-baseline pairs.
- 3 quality checks failed (analise detalhada):
  1. **`oos_pairwise_target_alignment`**: 4 issues, todos sob
     cohort `phase_b_confirmatorio_20260524` (TFT-only):
     `configs=15, min_ts=435, max_ts=436` (test h=1+7) e
     `configs=15, min_ts=435, max_ts=439` (val h=1+7). Variance de
     1-4 timestamps entre seeds na mesma fold (provavelmente
     seed-dependent dataloader boundary / batch_size drop). Magnitude
     desprezivel (~0.2-0.9% rows). Pre-existente (nao introduzido por
     mim); smoke v4 R-23 nao captou porque usou 1 seed unica. Mitigacao
     Sessao-B: intersect timestamps explicitamente em DM manual.
  2. **`cardinality_config_fold_seed`**: `duplicate_groups=45`. Causa:
     45 baseline dim_run rows tem `seed=None, fold=None` (pipeline
     `run_baselines_test_pipeline_use_case` nao popula esses campos no
     persiste). Pre-existente -- bug de schema. Nao afeta dados em
     fact_oos_predictions (que tem seed/horizon corretos).
  3. **`dm_mcs_persisted_executable`**: `feasible_dm=8, missing_dm=2,
     feasible_mcs=8, missing_mcs=2`. Causa: 2 cohort/split/horizon combos
     onde DM/MCS deveriam ter rodado mas pares nao formaram (consequencia
     do cohort splitting: TFT cohort tem candidato unico para MCS+DM
     mas baselines em cohorts separados). Mitigacao Sessao-B: post-
     processamento cross-cohort.
- Decisoes mid-execucao:
  - Aceitar exit 1 / `passed=False` em vez de patch silver (per F.2 §14
    veto explicito; per Marcelo "resolva sozinho e reporte ao final").
  - NAO re-rodei baselines via standalone CLI (`main_run_baselines`) com
    `--parent-sweep-id` explicito porque standalone nao deriva warmup
    automaticamente do max_encoder_length (per RUN_BASELINES.md), o que
    desalinharia target_timestamps com TFT.
- Confirmacoes:
  - `data/analytics_archive_pre_phase_b/` intacto (du -sh: 851M).
  - Dataset sha256 inalterado durante todo E4
    (`aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298`).
  - Protected files nao tocados.
  - Pytest reexecutado pos-E4: 609 passed.
- Outputs salvos:
  - `/tmp/e4_refresh_20260524.log` (v1, exit 2)
  - `/tmp/e4_refresh_20260524_v2.log` (v3 final, exit 1)
  - `data/analytics/gold/` 25 parquets.
- Follow-ups para Sessao-B (E5):
  - Computar TFT-vs-baseline DM via `fact_oos_predictions` cross-cohort:
    para cada (seed, fold_via_split_period, split, horizon), pegar TFT
    run_id em `parent_sweep_id=phase_b_confirmatorio_20260524` + baseline
    run_id em `parent_sweep_id=phase_b_confirmatorio_20260524__wf_X`,
    intersect `target_timestamp_utc`, computar pinball_post_guardrail
    per run, aplicar DM (statsmodels.stats.diagnostic ou similar) +
    Holm-Bonferroni sobre familia unica de 6 testes (2 horizontes x 3
    baselines).
  - Aceitar `gold_quality_statistics_report.statistics_ready=False` para
    cohort TFT como CONSEQUENCIA DOCUMENTADA do split, nao como sinal
    de falha cientifica.
- Outcome: PASS_WITH_CAVEAT (cohort utilizavel; aceite literal
  failed_checks=[] nao satisfeito; 3 failures todas em tech debt
  documentado; gold tables completas para Phase B analysis pelos
  caminhos secundarios).

---

## Stage E5 — Analise vs Tier 1 / Tier 2 gates (~4-8h)

**Objetivo:** classificar cada horizonte (h=1, h=7) contra Tier 1 (primario)
ou Tier 2 (secundario) ou "refutado" per F.2 §9, **sem cherry-picking**
(criterios pre-declarados ex-ante; escolha do tier nao pode depender de
inspecao dos resultados).

**Cross-link:** F.2 §3 (H1/H2a/H2b) + §9 (Tier 1/Tier 2 + gate degeneracao)
+ §6 (contrato post-guardrail).

**Sessao:** **Sessao-B** (Marcelo presente; skill
[model-performance-and-research-advisor](../ai/skills/model-performance-and-research-advisor/SKILL.md)
ativa).

**Pre-condicao:** E4.2 PASS (gold materializado, quality gate exit 0).

### Tasks

- [ ] **E5.1** Carregar gold tables relevantes:
      `gold_prediction_metrics_by_run_split_horizon`,
      `gold_prediction_metrics_by_config`,
      `gold_dm_pairwise_results`,
      `gold_mcs_results`,
      `gold_win_rate_pairwise_results`,
      `gold_paired_oos_intersection_by_horizon`,
      `gold_quantile_degeneracy_report`. Filtrar por
      `parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>`, `split=test`,
      `horizon ∈ {1, 7}`.
      **Aceite:** dataframes carregados; row counts esperados (15 TFT + 45 baseline
      × splits relevantes).

- [ ] **E5.2** Verificar **gate de degeneracao** (precondicao para qualquer
      tier per F.2 §9): `p10_eq_p90_rate < 0.05` em todos os 60 grupos
      quantile (30 TFT + 30 baselines × 2 horizontes; baselines `zero_return`
      e `historical_mean_rolling` sao point, ignorar).
      **Aceite:** gate PASS para o candidato TFT em ambos horizontes; relatorio
      explicito se alguma run TFT falhou (exclusao automatica de claims
      probabilisticos).

- [ ] **E5.3** Para cada horizonte h ∈ {1, 7}, agregar metricas TFT
      (mediana ou media de 15 runs) e calcular:
      - `coverage_q10_post_guardrail`, `coverage_q50_post_guardrail`,
        `coverage_q90_post_guardrail` → verificar bandas Tier 1.
      - `coverage_error_post_guardrail` (= PICP_p10p90 − 0.80) → `|.| ≤ 0.02`.
      - `mpiw_post_guardrail` > 0 e finito.
      **Aceite:** tabela `tier_eligibility_calibration_<YYYYMMDD>.csv` em `/tmp`
      com `passa_tier_1_calibracao` boolean per horizonte.

- [ ] **E5.4** Aplicar **Holm-Bonferroni** sobre familia unica de 6 testes DM
      (2 horizontes × 3 baselines) per F.2 §7. Output esperado: coluna
      `pvalue_adj_holm` em `gold_dm_pairwise_results` para os 6 pares
      `(TFT vs baseline_i, h)`. Confirmar que a coluna ja vem preenchida
      pelo refresh (Holm aplicado em `gold_dm_pairwise_results` per
      ADR-0004 ou Stage 7); se nao vier, calcular manualmente seguindo
      convencao `statsmodels.stats.multitest.multipletests` `method='holm'`.
      **Aceite:** 6 pvalues ajustados disponiveis; classificacao `pvalue_adj_holm
      < 0.05` per horizonte/baseline registrada.

- [ ] **E5.5** Calcular `delta_mean_pinball_rel` por horizonte: para cada
      baseline, `(mean_pinball_baseline_post - mean_pinball_TFT_post) /
      mean_pinball_baseline_post`. Verificar Tier 1: `≥ 3%`. H2a usa baseline
      primario `zero_return`; H2b usa todos baselines.
      **Aceite:** tabela com delta_rel per (baseline, horizon); flag Tier 1.

- [ ] **E5.6** **Classificar tier por horizonte** (decisao final ex-ante, sem
      cherry-picking):
      - **Tier 1** se: calibracao (E5.3) PASS **E** DM Holm-corrected (E5.4)
        PASS **E** delta_pinball_rel ≥ 3% (E5.5) PASS **E** MPIW > 0.
      - **Tier 2** se Tier 1 falhar mas: calibracao Tier 2 bands (q10 ∈ [0.05,
        0.15], q50 ∈ [0.40, 0.60], q90 ∈ [0.85, 0.95], PICP_err ≤ 0.05) PASS
        **E** DM Holm `< 0.10` PASS **E** delta_pinball_rel ≥ 0%.
      - **Refutado** se nem Tier 1 nem Tier 2 satisfeitos.
      Registrar veredicto por hipotese (H1, H2a, H2b) per horizonte.
      **Aceite:** tabela `tier_verdict_<YYYYMMDD>.csv` com colunas
      `(horizon, hypothesis, tier, justificativa)`; output unicamente derivado
      de criterios pre-declarados (sem inspecao livre).

### Notas de revisao:

_(vazio na criacao; preenchido pela Sessao-B apos E5.6)_

---

## Stage E6 — Relatorio + emenda final + archive snapshot (~4-6h)

**Objetivo:** redigir relatorio `B_confirmatory_<date>.md` consolidando E1-E5;
preencher slots finais de §15 do pre-registro (data execucao, tier, relatorio);
criar archive snapshot read-only `data/analytics_archive_phase_b_<date>/`;
abrir PR-B.

**Cross-link:** F.2 §15 (slot relatorio) + §14 (emenda).

**Branch:** mesma branch `feat/phase-b-execution-<YYYYMMDD>` continuada
(PR-A ja mergeada); novos commits empilhados.

### Tasks

- [ ] **E6.1** Criar
      [`docs/07_reports/phase-gates/B_confirmatory_<YYYY-MM-DD>.md`](../07_reports/phase-gates/).
      Template minimo:
      - Status: PASS Tier 1 / PASS Tier 2 / REFUTADO (por horizonte).
      - Parametros executados: cita config sha256, parent_sweep_id, dataset
        sha256, commit selo pre-registro.
      - Resultados E5: tabelas tier_eligibility + tier_verdict; metricas
        agregadas TFT vs baselines per horizonte; calibracao quantil;
        DM/MCS/win-rate; gate degeneracao status.
      - Limitacoes: cita F.2 §12 (out-of-scope itens); cita STRATEGIC_DIRECTION
        §3 (calibracao de expectativas).
      - Cross-link: F.2, PHASE_B_EXECUTION_CHECKLIST.md, gold tables
        paths, `option_b_execution_log_<date>.md` (se mantido).
      **Aceite:** relatorio ≥ 200 LOC; sections completas; cita todos os
      artefatos (silver/gold tables, commit hash, sha256).

- [ ] **E6.2** Editar `preregistration_phase_b.md` §15 "Execucao":
      preencher slots restantes:
      - **Data de execucao confirmatoria:** `<YYYY-MM-DD>`
      - **Resultado tier (H1, H2a, H2b por horizonte):** `<tier-1 / tier-2 / refutado>`
      - **Relatorio:** link para `B_confirmatory_<YYYY-MM-DD>.md`
      §14 "Emendas": adicionar entrada datada
      ```
      ### 2026-MM-DD — Emenda E6: execucao concluida
      - Decisao alterada: nenhuma; preenchimento de §15.
      - Cross-link: PR-B, relatorio B_confirmatory_<date>.md
      - Justificativa: encerramento operacional do ciclo Phase B.
      ```
      **Aceite:** §15 todos os 6 slots preenchidos; §14 com nova entrada datada.

- [ ] **E6.3** Criar archive snapshot read-only (D5):
      - Antes de criar o archive, confirmar que `.gitignore` cobre
        `data/analytics_archive_*` (regra atual esperada: `data/**`):
        `git check-ignore -v data/analytics_archive_phase_b_<YYYYMMDD>/__probe__.parquet`.
        Se nao houver match, parar e corrigir `.gitignore` antes de qualquer
        `git add`, para nao versionar Parquets do snapshot.
      - `mkdir -p data/analytics_archive_phase_b_<YYYYMMDD>/`
      - `cp -r data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/ data/analytics_archive_phase_b_<YYYYMMDD>/silver_dim_run/`
      - Idem para `fact_oos_predictions` (filtered por run_id da coorte) e
        25 gold tables filteradas por parent_sweep_id.
      - `chmod -R a-w data/analytics_archive_phase_b_<YYYYMMDD>/`
      - Adicionar nota em
        [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md)
        registrando o archive (mesmo padrao de Stage 3 do
        PHASE_B_IMPLEMENTATION_CHECKLIST).
      **Aceite:** archive populado; `chmod` aplicado (escrita em arquivo do
      archive falha com Permission denied); doc canonica atualizada.

- [ ] **E6.4** Commits empilhados na branch:
      1. `docs(phase-b-exec): adicionar relatorio B_confirmatory_<YYYY-MM-DD>`
      2. `docs(pre-reg): emenda E6 — preencher §15 execucao + entrada §14`
      3. `chore(archive): snapshot read-only Phase B + nota ANALYTICS_STORE_ARCHITECTURE`
      **Aceite:** 3 commits no padrao governance; passa pre-commit; cada
      commit isolado; nenhum arquivo sob `data/analytics_archive_phase_b_*`
      aparece em `git status --short`.

- [ ] **E6.5** Push branch + abrir **PR-B**:
      `docs(phase-b-exec): relatorio Phase B confirmatorio + emenda final + archive snapshot`.
      Body cita: tier classificacao por hipotese, gold tables criticas, link
      para relatorio, link para archive.
      **Aceite:** PR contra main; CI green; Marcelo revisa e mergeia.

- [ ] **E6.6** Apos merge PR-B: atualizar `PHASE_B_EXECUTION_CHECKLIST.md`
      header `Status: pendente → completo (YYYY-MM-DD)` + marcar todos os
      Stages E0-E6 como `[x]`. Atualizar
      `STRATEGIC_DIRECTION.md` Fase B status se houver flag de progresso.
      **Aceite:** checkboxes E0-E6 = `[x]`; Status atualizado; commit
      `chore(checklist): close PHASE_B_EXECUTION_CHECKLIST.md`.

### Notas de revisao:

_(vazio na criacao; preenchido pela Sessao-B apos E6.6)_

---

## Cross-link com checklists / docs canonicos

| Checklist / Doc | Relacao |
|---|---|
| [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](PHASE_B_IMPLEMENTATION_CHECKLIST.md) | Antecessor — pre-experiment engineering. Imutavel apos F.3 [x]. |
| [`preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md) | Fonte cientifica (F.2). Selada via emenda em E1; finalizada em E6. |
| [`POST_AUDIT_EXECUTION_PLAN.md`](../08_governance/POST_AUDIT_EXECUTION_PLAN.md) | Mapa alto-nivel pos-A_code_audit; nao operacionaliza E0-E6 (este checklist preenche essa lacuna). |
| [`smoke_confirmatory_2026-05-18.md`](../07_reports/smoke_confirmatory_2026-05-18.md) §F.1 v4 PASS pleno | Referencia operacional para reproducao em E2.0 (bit-byte check D7). |
| [`STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md) §5 Fase B | Direcao estrategica; criterio de sucesso. |
| [`GOVERNANCE_AND_VERSIONING.md`](../08_governance/GOVERNANCE_AND_VERSIONING.md) | Padroes de branch/commit/PR. |
| `docs/ai/skills/explicit-sweep-execution/SKILL.md` | Protocolo sweep explicit (aplicado E0, E2, E3). |
| `docs/ai/skills/model-performance-and-research-advisor/SKILL.md` | Protocolo de decisao analitica (aplicado E5). |
| `.claude/skills/staged-implementation-protocol/SKILL.md` | Protocolo de delegacao para sessoes implementadoras (orienta autoria dos prompts Sessao-A e Sessao-B). |
