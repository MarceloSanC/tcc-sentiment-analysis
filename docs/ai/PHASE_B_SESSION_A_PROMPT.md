# Phase B — Sessão-A (Operacional E0–E4)

Você é uma sessão Claude entrando em modo **executor operacional autônomo**
para os Stages E0–E4 do
[`PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md).

Sua missão é **executar** E0 (Optuna heritage), E1 (selar config + emenda
pré-registro + abrir PR-A), E2 (treino TFT confirmatório), E3 (baselines) e
E4 (refresh + quality validation). **Você NÃO conduz análise estatística
nem reporta tier** — isso é responsabilidade da Sessão-B (prompt separado).

Modo: **AUTÔNOMO com 1 pause point obrigatório após E1.5** (aguardar Marcelo
mergear PR-A em main antes de iniciar E2).

## Contexto verificado (confirme via comandos antes de iniciar)

Estado em main esperado:
- `PHASE_B_EXECUTION_CHECKLIST.md` mergeado em main (PR-60, commit `e55ab21`).
- F.1 ✓ MERGED (R-23 PR #54).
- F.2 ✓ MERGED (PR #58); pré-registro em
  [`docs/06_pre_registration/preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md).
- F.3 ✓ MERGED (PR #59); Gate de saída da Fase A fechado; data de abertura
  Phase B = 2026-05-23.
- Phase B formalmente autorizada a executar.
- `data/analytics_archive_pre_phase_b/` = 851M intocável (read-only).
- pytest 605+ passed; ruff clean.

Decisões D1–D8 fechadas (não redecidir):
- **D1:** UM candidato top-1 robust_score do E0.
- **D2:** `<YYYYMMDD>` suffix = data real de início de E2 (decidir em E1.1
  perguntando a Marcelo).
- **D3:** Sem fix de código em baselines `output_subdir`; JSON novo correto
  em E1.1.
- **D4:** 2 PRs (PR-A após E1; PR-B após E6 pela Sessão-B). Silver/gold
  **NÃO** entram em git.
- **D5:** Snapshot read-only criado em E6 (Sessão-B).
- **D6:** Walk-forward sensibilidade B.4 é rodada futura — fora do seu escopo.
- **D7:** Todos os pre-execution checks (a+b+c+d) obrigatórios antes de E2.
- **D8:** Esta é a Sessão-A; Sessão-B é separada.

## Required reading (na ordem; NÃO PULE)

1. [`PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md)
   — **fonte primária**. Stages E0–E4 com tasks numeradas e Aceite.
2. [`preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md)
   §2, §4, §5, §10, §11, §13 — protocolo científico que E0–E4 operacionalizam.
3. [`docs/06_runbooks/RUN_TRAINING.md`](../06_runbooks/RUN_TRAINING.md),
   [`RUN_BASELINES.md`](../06_runbooks/RUN_BASELINES.md),
   [`RUN_REFRESH_ANALYTICS.md`](../06_runbooks/RUN_REFRESH_ANALYTICS.md) —
   comandos canônicos.
4. [`docs/03_modeling/SWEEPS_AND_SELECTION.md`](../03_modeling/SWEEPS_AND_SELECTION.md)
   §"Schema da secao baselines" — formato do JSON unificado.
5. [`config/sweeps/explicit/phase_a_smoke_20260518_v2.json`](../../config/sweeps/explicit/phase_a_smoke_20260518_v2.json)
   — template para E1.1 (substituir conforme E1.1 task).
6. [`config/sweeps/optuna/0_1_5_optuna_top13_all_feature_sets_hpo.json`](../../config/sweeps/optuna/0_1_5_optuna_top13_all_feature_sets_hpo.json)
   — template para E0.1.
7. [`data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv`](../../data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv)
   — top-3 frozen Round 0 referenciados em F.2 §4.2 passo 1.
8. [`smoke_confirmatory_2026-05-18.md`](../07_reports/smoke_confirmatory_2026-05-18.md)
   §"F.1 v4 PASS pleno" — parâmetros R-23 referência para bit-byte check em E2.0.b.
9. [`docs/ai/skills/explicit-sweep-execution/SKILL.md`](skills/explicit-sweep-execution/SKILL.md)
   — protocolo canônico de sweep explicit aplicado em E0, E2, E3.

## Autoridade e pause points

| Stage | Autoridade | Pause? |
|---|---|---|
| E0 | **Total** — execute autônomo (~6–12h overnight). | Não. |
| E1 | **Total** — crie config, anexe emenda, commit, abra PR-A. | **PAUSE após E1.5**: aguardar Marcelo mergear PR-A em main. |
| E2 | **Total** após PR-A mergeada. Execute D7 pre-checks; aborte se falharem. | Não (apenas valide E2.3 antes de E3). |
| E3 | **Total**. | Não. |
| E4 | **Total**. Aborte se quality gate retornar `failed_checks != []`. | **PAUSE final** após E4.2: report a Marcelo + transição para Sessão-B. |

**Pause após E1.5 é obrigatório.** PR-A precisa estar mergeado antes de E2
porque (a) Marcelo precisa revisar a config selada + emenda; (b) sha256 da
config commitada vira parte da rastreabilidade científica. Se você executar
E2 sem PR-A mergeada, runs ficarão referenciando config potencialmente
diferente da final em main.

## Comandos proibidos (sempre)

- `git rebase` (qualquer forma)
- `git commit --amend`
- `git push --force` / `git push -f`
- Qualquer escrita em `data/analytics_archive_pre_phase_b/` (read-only por
  filesystem; `chmod` para escrita = **violação grave**)
- `--no-verify` em commit hooks
- Modificar `data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet`
- Tocar checklists imutáveis: `PHASE_B_IMPLEMENTATION_CHECKLIST.md` (F.3
  fechado), `POST_CLOSURE_FIXES_CHECKLIST.md`

## Protected files (NÃO modificar)

- Todo `src/` (Phase B é execução, não modificação de código de produção;
  qualquer bug encontrado pause e reporte a Marcelo — não tente fix).
- Todo `tests/` (idem).
- `data/analytics_archive_pre_phase_b/` (read-only).
- `data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet` (Phase B usa
  como está pós-R-20).
- `docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (imutável).
- `docs/05_checklists/POST_CLOSURE_FIXES_CHECKLIST.md` (imutável).
- Todos os ADRs em `docs/01_architecture/decisions/` (canônicos pós-merge).

**Permitido modificar:**
- `docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md` — apenas
  marcar tasks E0–E4 como `[~]` ou `[x]` e preencher "Notas de revisão" da
  Stage correspondente.
- `docs/06_pre_registration/preregistration_phase_b.md` — apenas §14
  (anexar emenda E1) e §15 (preencher 3 slots em E1.3; restantes só em E6
  pela Sessão-B).
- Criar `config/sweeps/optuna/phase_b_hpo_<YYYYMMDD>.json` (novo).
- Criar `config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json` (novo).

## Skills a invocar (proativo)

- [`explicit-sweep-execution`](skills/explicit-sweep-execution/SKILL.md):
  protocolo canônico para E0, E2, E3 (cohort isolation por prefixo;
  validation pós-run; resume policy se interromper).
- [`staged-implementation-protocol`](../../.claude/skills/staged-implementation-protocol/SKILL.md)
  Phase B step 8: estrutura de Notas de revisão por Stage (registrar data,
  branch, validation commands com results, decisões mid-execução, follow-ups).
- [`documentation-authoring`](skills/documentation-authoring/SKILL.md): para
  E1.3 (emenda pré-registro) e E6 não se aplica nesta sessão.

## Sucesso definido

Sessão-A entrega:
1. **E0.3 PASS:** top-1 hiperparams documentados em `/tmp/e0_top1_<date>.json`;
   sweep `phase_b_hpo_<date>` registrado em silver.
2. **E1.5 PR-A aberta:** branch `feat/phase-b-execution-<YYYYMMDD>` no GitHub;
   PR contra main com `config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`
   + emenda em `preregistration_phase_b.md` §14/§15 partial; CI green.
3. **E2.3 PASS:** 15 dim_run rows TFT (5 seeds × 3 folds) com `status=ok`;
   `p10_eq_p90_rate < 0.05` em 60 grupos.
4. **E3.2 PASS:** 60 dim_run rows totais (15 TFT + 45 baselines); alinhamento
   `target_timestamp_utc` verificado.
5. **E4.2 PASS:** quality gate exit 0, `failed_checks=[]`, ≥25 gold tables;
   `gold_dm_pairwise_results` com 6 pares TFT-vs-baseline; `gold_paired_oos_intersection_by_horizon.aligned_exact=True`.
6. **Handoff para Sessão-B:** Notas de revisão E0–E4 preenchidas no checklist;
   data efetiva `<YYYYMMDD>` registrada; `parent_sweep_id` efetivo declarado;
   archive `data/analytics_archive_pre_phase_b/` intacto (851M).

## Procedimento detalhado

### Phase A.1 — E0 + E1 (executável imediatamente após init)

#### E0 — Optuna heritage (~6–12h)

Pre-flight:
- Leia [PHASE_B_EXECUTION_CHECKLIST.md Stage E0](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md).
- Verifique top-3 Round 0:
  `head -1 data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv && grep ",1," data/analytics/selection/frozen_candidates_optuna_0_1_5__optuna_0_1_6.csv | head -3`.

**E0.1** — Criar `config/sweeps/optuna/phase_b_hpo_<YYYYMMDD>.json`.
Use `<YYYYMMDD>` = data de hoje (data de execução do E0, NÃO a data D2 do E2).
Base no template `0_1_5_optuna_top13_all_feature_sets_hpo.json`. Substituições
obrigatórias:
- `output_subdir`: `"phase_b_hpo_<YYYYMMDD>"`.
- `features`: `"BASELINE_FEATURES,TECHNICAL_FEATURES,SENTIMENT_FEATURES,FUNDAMENTAL_FEATURES"`.
- `feature_sets`: `[]` (vazio, pois `features` é direto).
- `n_trials`: `25`.
- `top_k`: `3`.
- `objective_metric`: `"robust_score"`.
- `objective_lambda`: `1.0`.
- `replica_seeds`: `[20260517]` (HPO usa 1 seed; replica vem em E2).
- `training_config.max_encoder_length`: `60`.
- `training_config.max_prediction_length`: `7`.
- `training_config.prediction_mode`: `"quantile"`.
- `training_config.quantile_levels`: `[0.1, 0.5, 0.9]`.
- `training_config.evaluation_horizons`: `[1, 7]`.
- `training_config.compute_feature_importance`: `false`.
- `search_space`: centrar nos hiperparams top-1 Round 0 com vizinhanças
  estreitas (±20–30%):
  - `hidden_size`: discrete `[16, 24, 32, 48, 64]`.
  - `attention_head_size`: discrete `[2, 4, 8]`.
  - `dropout`: uniform `[0.05, 0.25]`.
  - `learning_rate`: log-uniform `[1e-4, 1e-3]`.
  - `batch_size`: discrete `[32, 64, 128]`.
  - `hidden_continuous_size`: discrete `[4, 8, 16]`.
  - `max_epochs`: fixed `30` para HPO (Phase B `max_epochs` final vem do
    melhor trial).

Valide: `python -c "import json; json.load(open('config/sweeps/optuna/phase_b_hpo_<YYYYMMDD>.json'))"`.

**E0.2** — Execute (autônomo overnight):
```bash
.venv/bin/python -m src.main_tft_optuna_sweep \
  --asset AAPL \
  --config-json config/sweeps/optuna/phase_b_hpo_<YYYYMMDD>.json \
  > /tmp/e0_optuna_<YYYYMMDD>.log 2>&1 &
```
Monitor com tail periódico. Se algum trial falhar com OOM, `continue_on_error=true`
no JSON garante que o batch prossegue.

**Warmup ROCm — comportamento esperado, NÃO erro** (validado em calibração
2-trial Optuna em 2026-05-23):

Cada trial mostrará no início:
```
[epoch=0] train_loss=nan val_loss=nan       <-- warmup NaN do primeiro batch (ROCm lazy load)
[epoch=0] train_loss=0.008061 val_loss=0.009792  <-- recovery imediato; treino real começa
[epoch=1] train_loss=...
```

Esse padrão é **determinístico** em ROCm (driver lazy load no primeiro forward),
**auto-recuperado** pelo Lightning (substitui NaN→1e9, segue para próximo batch),
e o trial **conclui normalmente** com `runs_failed=0` e métricas válidas em
`top_5_runs`. NÃO interrompa, NÃO retry, NÃO trate como erro. Apenas registre
em Notas de revisão E0: "Warmup NaN observado em todos os N trials (esperado;
auto-recuperado per ADR ambiente)".

PAUSE apenas se `runs_failed > 0` ao final do sweep ou se trial completar com
`train_loss=nan` em **TODAS as epochs** (não só epoch=0 primeira linha).

**Estimativa wall-clock observada (calibração 2026-05-23):** ~2m18s/trial com
search_space colapsado a 1 ponto + early stopping em ~10 epochs. Para Phase B
com search_space realista (várias dims combinatorias + max_epochs=30 sem
early conv.), espere **3-5min/trial**. 25 trials ≈ **75-125min** (não 6-12h
como estimativa original). Se trial real ultrapassar 10min, investigar.

**E0.3** — Após término, identifique top-1:
```bash
ls data/analytics/selection/frozen_candidates_phase_b_hpo_<YYYYMMDD>*.csv
# Esse CSV (gerado por main_tft_optuna_sweep) tem rank 1 com robust_score top.
# Extraia hiperparams + run_id + trial_number e salve em /tmp/e0_top1_<YYYYMMDD>.json.
```

Marque E0.1, E0.2, E0.3 como `[x]` no checklist + preencha Notas de revisão:
data, branch (`docs/phase-b-session-prompts` — ou nova `feat/phase-b-execution-<YYYYMMDD>`),
n_trials completados, top-1 robust_score, observações.

#### E1 — Selar config + emenda pré-registro (~2–3h)

**Antes de E1.1, pergunte a Marcelo:** "Qual data usar como `<YYYYMMDD>` em
`parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>`?" (D2: data real de início
do E2 — provavelmente data do próximo dia útil após E0 terminar). Aguarde
resposta antes de prosseguir.

**E1.0** — Criar branch nova a partir de main:
```bash
git checkout main && git pull --ff-only
git checkout -b feat/phase-b-execution-<YYYYMMDD>
```

**E1.1** — Criar `config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`.
Use template `phase_a_smoke_20260518_v2.json` com substituições. **CRÍTICAS:**
- `output_subdir`: `"phase_b_confirmatorio_<YYYYMMDD>"` (D3: derivará
  `parent_sweep_id` correto para ambos pipelines — sem workaround /tmp).
- `training_config.*`: hiperparams top-1 do E0 (extraídos de
  `/tmp/e0_top1_<YYYYMMDD>.json`).
- `training_config.max_epochs`: valor convergido em E0 (esperado 30–50; se
  early-stopping disparou em epoch < 30, use esse valor + 5 de margem).
- `replica_seeds`: `[20260517, 20260518, 20260519, 20260520, 20260521]` (5
  seeds per F.2 §4.3).
- `walk_forward.enabled`: `true`.
- `walk_forward.folds`: 3 folds explícitos. Recomendação (alinhar com F.2 §2
  OOS = test `2023-01-01..2025-12-31`):
  ```json
  [
    {"name": "wf_1", "train_start": "20101018", "train_end": "20191231",
     "val_start": "20200101", "val_end": "20211231",
     "test_start": "20220101", "test_end": "20231231"},
    {"name": "wf_2", "train_start": "20101018", "train_end": "20201231",
     "val_start": "20210101", "val_end": "20221231",
     "test_start": "20230101", "test_end": "20241231"},
    {"name": "wf_3", "train_start": "20101018", "train_end": "20211231",
     "val_start": "20220101", "val_end": "20231231",
     "test_start": "20240101", "test_end": "20251231"}
  ]
  ```
  Se Marcelo preferir folds diferentes, pergunte antes de E1.1.
- `baselines`: as 3 entradas exatas conforme F.2 §5 + template smoke.
- `evaluation_horizons`: `[1, 7]`.
- `prediction_mode`: `"quantile"`, `quantile_levels`: `[0.1, 0.5, 0.9]`.
- `explicit_configs`: `[{"run_label": "phase_b_tft_all_features", "training_config": {}}]`.

Valide `json.load`.

**E1.2** — Compute hash:
```bash
sha256sum config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json
```
Registre o hash (64 chars hex).

**E1.3** — Edite `preregistration_phase_b.md`:
- §14 "Emendas": adicionar entrada datada conforme template do checklist E1.3.
- §15 "Execucao": preencher (apenas) **commit hash do pré-registro selado**
  com `067cb32` (`docs(pre-reg): F.2 — pre-registro Phase B confirmatorio`).
  Este campo referencia o selo F.2 original do pre-registro; nao use o commit
  E1.4, porque ele ainda nao existe durante E1.3 e registra apenas a emenda/config,
  **parent_sweep_id efetivo**, **Hash da config JSON**. Demais slots
  permanecem `<a preencher>` (Sessão-B preenche em E6).

**E1.4** — Commit único:
```bash
git add config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json \
        docs/06_pre_registration/preregistration_phase_b.md
git commit -m "$(cat <<'EOF'
feat(phase-b-exec): selar config phase_b_confirmatorio_<YYYYMMDD> e emenda pre-registro E1

- Config sealed: sha256=<hash>
- parent_sweep_id efetivo: phase_b_confirmatorio_<YYYYMMDD>
- Heritage sweep_id: phase_b_hpo_<E0-date>
- Top-1 robust_score trial_number: <N>, run_id: <run_id>

Closes E1.1-E1.4 do PHASE_B_EXECUTION_CHECKLIST.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

**E1.5** — Push + abrir PR-A:
```bash
git push -u origin feat/phase-b-execution-<YYYYMMDD>
gh pr create --base main --head feat/phase-b-execution-<YYYYMMDD> \
  --title "feat(phase-b-exec): config sealing + pre-registro emenda E1" \
  --body "<body com: link checklist E1, sha256 config, sweep_id E0, top-1 hiperparams, próximos passos E2-E4 após merge>"
```

Marque E1.0–E1.5 como `[~]` (em revisão).

### ⏸️ PAUSE OBRIGATÓRIO — Aguardar merge PR-A

**NÃO INICIE E2** até que Marcelo mergee PR-A em main. Reporte a Marcelo:
> "PR-A aberta: <URL>. Aguardando merge para iniciar E2 (treino TFT
> confirmatório, ~8–15h)."

Após Marcelo informar merge:
```bash
git checkout main && git pull --ff-only
# Confirme commit de merge em git log -3.
git checkout feat/phase-b-execution-<YYYYMMDD>
git merge --ff-only main  # branch local up-to-date com main pós-merge
```

### Phase A.2 — E2 + E3 + E4 (após PR-A mergeada)

#### E2 — Treino TFT confirmatório (~8–15h)

**E2.0** — Pre-execution sanity checks (D7, **todos obrigatórios**):

(a) pytest + ruff:
```bash
.venv/bin/pytest tests/ -q \
  --ignore=tests/integration/test_quality_registry_bit_identical_archive.py \
  --ignore=tests/integration/test_gold_builders_byte_identical_archive.py
.venv/bin/ruff check src/ tests/
```
Esperado: `605+ passed`; `All checks passed!`. Se falhar, ABORTE e reporte.

(b) Bit-byte sanity reproduce (não é byte-exact; sanity ponto-final):

Para o ambiente ROCm/AMD do projeto, R-23 reference (CUDA) não reproduz
bit-byte. Este check é **sanity**: pipeline executa 1 epoch sem regressão,
outputs shape-corretos, quantis válidos. (Bit-byte CUDA reference é
out-of-scope deste pre-check.)

```bash
.venv/bin/python -m src.main_train_tft \
  --asset AAPL \
  --features "BASELINE_FEATURES,TECHNICAL_FEATURES,SENTIMENT_FEATURES,FUNDAMENTAL_FEATURES" \
  --max-epochs 1 \
  --max-encoder-length 60 --max-prediction-length 7 \
  --evaluation-horizons "[1, 7]" \
  --parent-sweep-id "e2_repro_check" \
  --seed 20260517 --prediction-mode quantile \
  2>&1 | tee /tmp/e2_repro_check.log
```

PASS se: (i) exit 0; (ii) primeiros 5 rows de `(test, h=1)` têm `y_true`
não-NaN; (iii) quantis monotônicos (`p10 ≤ p50 ≤ p90` em cada row);
(iv) `target_timestamp_utc` ∈ `[2023-01-01, 2025-12-31]`.

**Warmup ROCm — esperado (validado em calibração 2026-05-23):** o log mostrará
no início:
```
[epoch=0] train_loss=nan val_loss=nan      <-- warmup NaN (esperado)
[epoch=0] train_loss=0.008061 val_loss=...  <-- recovery; treino real
```
Auto-recuperado pelo Lightning. Trial conclui normalmente. NÃO retry, NÃO
abort. Apenas registre em Notas E2.0. PAUSE apenas se TODAS epochs forem NaN
(treino realmente não converge).

Cleanup correto (paths reais — `fact_oos_predictions` não particiona por
`parent_sweep_id`; remove por `run_id` extraído de `dim_run`):

```bash
.venv/bin/python <<'PY'
import pandas as pd
from pathlib import Path
sweep = 'e2_repro_check'
silver = Path('data/analytics/silver')

dim_dir = silver / 'dim_run' / 'asset=AAPL'
run_ids = set()
for p in dim_dir.rglob(f'*parent_sweep_id={sweep}*/*.parquet'):
    df = pd.read_parquet(p)
    run_ids.update(df['run_id'].tolist())
print(f'Cohort run_ids: {len(run_ids)}')

fact_dir = silver / 'fact_oos_predictions' / 'asset=AAPL'
removed_total = 0
for fact_p in fact_dir.rglob('*.parquet'):
    df = pd.read_parquet(fact_p)
    if 'run_id' in df.columns:
        mask = df['run_id'].isin(run_ids)
        if mask.any():
            df_clean = df[~mask]
            df_clean.to_parquet(fact_p, index=False)
            removed_total += int(mask.sum())
print(f'Total fact_oos rows removed: {removed_total}')
PY

# dim_run + modelo (estes SIM particionados por parent_sweep_id no path)
rm -rf data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=e2_repro_check
rm -rf data/models/AAPL/e2_repro_check 2>/dev/null

# Confirme limpo
git status --short  # esperado vazio
find data -path "*e2_repro_check*" 2>/dev/null  # esperado vazio
```

(c) Hash dataset:
```bash
sha256sum data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet
```
**Esperado** (baseline Phase B registrado em pre-flight 2026-05-23):
`aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298`.
Se divergir, dataset foi modificado pós-pre-flight — PAUSE e reporte.
Esse hash entra no commit body E1.4 e no relatório E6.1 como baseline.

(d) Main verde:
```bash
gh pr view 54 58 59 60 --json statusCheckRollup,state,mergedAt
```
Todos com `state=MERGED`; `statusCheckRollup` com SUCCESS.

Registre output dos 4 checks em `/tmp/e2_preflight_<YYYYMMDD>.txt`.

**E2.1** — Execute treino confirmatório:
```bash
.venv/bin/python -m src.main_tft_test_pipeline \
  --asset AAPL \
  --config-json config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json \
  > /tmp/e2_train_<YYYYMMDD>.log 2>&1
```
8–15h. Monitor periódico com `tail -50 /tmp/e2_train_<YYYYMMDD>.log`.

**E2.2** — Validar shape:
```bash
ls data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/ | wc -l
# Esperado: exatamente 15 partitions (5 seeds × 3 folds).
```

**E2.3** — Spot-check degeneracao:
```python
.venv/bin/python <<'PY'
import pandas as pd
from pathlib import Path
base = Path('data/analytics/silver/fact_oos_predictions/asset=AAPL')
sweep = 'phase_b_confirmatorio_<YYYYMMDD>'
dfs = []
for p in base.rglob(f'parent_sweep_id={sweep}/**/*.parquet'):
    df = pd.read_parquet(p)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
group_cols = ['run_id', 'split', 'horizon']
rates = df.groupby(group_cols).apply(
    lambda g: (g['quantile_p10'] == g['quantile_p90']).mean()
).reset_index(name='p10_eq_p90_rate')
print(rates.describe())
print('Worst:', rates.nlargest(5, 'p10_eq_p90_rate'))
assert (rates['p10_eq_p90_rate'] < 0.05).all(), 'Degeneracao > 5% em ao menos 1 grupo'
print('PASS: todos < 5%')
PY
```

Marque E2.0–E2.3 como `[x]` + preencha Notas de revisão.

#### E3 — Baselines (~30min–1h)

```bash
.venv/bin/python -m src.main_baselines_test_pipeline \
  --asset AAPL \
  --config-json config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json \
  > /tmp/e3_baselines_<YYYYMMDD>.log 2>&1
```

Validate (E3.2):
```bash
ls data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/ | wc -l
# Esperado: 60 (15 TFT + 45 baselines = 3 × 5 × 3).
```

Spot-check alinhamento: pegue 1 run TFT + 1 baseline com mesma `(seed, fold, split,
horizon)` e confirme `target_timestamp_utc` TFT ⊆ baseline. Marque E3.1–E3.2
como `[x]`.

#### E4 — Refresh + quality (~30min–1h)

```bash
.venv/bin/python -m src.main_refresh_analytics_store \
  --scope-mode cohort_decision \
  --scope-sweep-prefixes phase_b_confirmatorio_ \
  --scope-splits val test \
  --scope-horizons 1 7 \
  --primary-quantile-contract post_guardrail \
  --fail-on-quality \
  > /tmp/e4_refresh_<YYYYMMDD>.log 2>&1
echo "Exit code: $?"
```
Exit 0 obrigatório. Tail log: `passed=True`, `failed_checks=[]`, `total_checks≥27`.

E4.2 spot-checks (lista no checklist):
- `gold_dm_pairwise_results.parquet`: 6 pares (2 horizontes × 3 baselines).
- `gold_paired_oos_intersection_by_horizon.parquet`: `aligned_exact=True`.
- `gold_quality_statistics_report.parquet`: `statistics_ready=True` para
  test h=1 e h=7.
- `gold_quantile_degeneracy_report.parquet`: `gate_passed=True` em todos.

Marque E4.1–E4.2 como `[x]`.

### Handoff para Sessão-B

Reporte a Marcelo:
> "Sessão-A concluída. Stages E0–E4 fechados. Outputs:
> - `parent_sweep_id` efetivo: `phase_b_confirmatorio_<YYYYMMDD>`
> - Config sha256: `<hash>`
> - 60 silver dim_run rows; 25 gold tables materializadas; quality gate exit 0.
> - Branch `feat/phase-b-execution-<YYYYMMDD>` (sem novos commits desde PR-A
>   merge; outputs em silver/gold não vão para git per D4).
> - Notas de revisão E0–E4 preenchidas em `PHASE_B_EXECUTION_CHECKLIST.md`.
>
> Pronto para Sessão-B (analítica E5–E6). Prompt em
> `docs/ai/PHASE_B_SESSION_B_PROMPT.md`."

## Init commands

Execute na ordem:
```bash
cd /home/marcelo/Code/financial-time-series-forecasting
git checkout main && git pull --ff-only
git log --oneline -5

# Confirmar pré-requisitos
gh pr view 60 --json state,mergedAt  # PR-60 checklist deve ser MERGED
test -f docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md && echo "Checklist OK"
test -f docs/06_pre_registration/preregistration_phase_b.md && echo "Pre-reg OK"

# Verificar estado
.venv/bin/pytest tests/ -q \
  --ignore=tests/integration/test_quality_registry_bit_identical_archive.py \
  --ignore=tests/integration/test_gold_builders_byte_identical_archive.py 2>&1 | tail -3
.venv/bin/ruff check src/ tests/
du -sh data/analytics_archive_pre_phase_b/  # esperado: 851M
ls data/analytics/silver/ data/analytics/gold/ 2>&1 | head -20  # estado inicial (pode estar não-vazio de smokes anteriores)
ls config/sweeps/explicit/ | grep phase_b 2>&1 || echo "Nenhuma config phase_b ainda (esperado)"
ls data/analytics/selection/frozen_candidates_optuna_0_1_5*.csv && echo "Round 0 frozen OK"
```

Após init, leia o Required reading na ordem listada acima. Depois inicie
**E0.1**.

## Notas de revisão template (preencher no checklist por Stage)

```markdown
### Notas de revisao:

- 2026-MM-DD HH:MM UTC: Stage E<N> executado.
- Branch: feat/phase-b-execution-<YYYYMMDD> (E1.0+) ou docs/phase-b-session-prompts (E0).
- Validation commands com results:
  - `.venv/bin/pytest tests/ -q ...` -> `X passed, Y warnings`.
  - `.venv/bin/ruff check src/ tests/` -> `All checks passed!`.
  - `<comando específico do Stage>` -> `<result>`.
- Decisões mid-execucao (se houver):
  - `<Question>`: `<Opcao escolhida + justificativa baseada em F.2 §X ou skill Y>`.
- Confirmacoes:
  - `data/analytics_archive_pre_phase_b/` intacto (du -sh: 851M).
  - Protected files nao tocados.
- Follow-ups (YELLOW se houver):
  - `<item deferido para Sessao-B ou rodada futura>`.
- Outcome: PASS / PASS_WITH_CAVEAT / BLOCKED.
```

## Constraints inegociáveis

- NUNCA toque `data/analytics_archive_pre_phase_b/`.
- NUNCA `git rebase`, `git commit --amend`, `git push --force`.
- NUNCA pule `--fail-on-quality` em E4.
- NUNCA inicie E2 sem PR-A mergeada.
- NUNCA conduza E5–E6 (responsabilidade da Sessão-B).
- Se encontrar bug em `src/`, pause e reporte a Marcelo (não fixe).
- Toda decisão mid-execução: declare no commit body + Notas de revisão; não
  decida silenciosamente.
- Se Marcelo não estiver disponível para D2 (data suffix) ou para confirmar
  merge PR-A, PARE e aguarde — não invente defaults.

## Sucesso

Sua sessão termina com:
- E0–E4 todos `[x]` no checklist + Notas de revisão preenchidas.
- PR-A mergeada em main (commit em `git log`).
- Silver populado com 60 dim_run rows do cohort.
- 25+ gold tables materializadas; quality gate exit 0.
- Reportagem clara para Marcelo + handoff para Sessão-B.
- Archive 851M intacto.
- Pytest + ruff inalterados.
