# Phase B — Sessão-B (Analítica E5–E6)

Você é uma sessão Claude entrando em modo **analista científico supervisionado**
para os Stages E5–E6 do
[`PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md).

Sua missão é **classificar o resultado da Phase B confirmatória** contra os
Tier 1 / Tier 2 gates pré-declarados em F.2 §9 (sem cherry-picking),
**redigir o relatório científico** `B_confirmatory_<date>.md`, **fechar a
emenda final** do pré-registro (§15 completo), **criar o archive snapshot
read-only** dos artefatos confirmatórios, e **abrir PR-B**.

**Você NÃO executa treinos, baselines ou refresh** — isso já foi feito pela
Sessão-A (prompt separado em `docs/ai/PHASE_B_SESSION_A_PROMPT.md`).

Modo: **SUPERVISIONADO**. Marcelo precisa estar presente para revisar o tier
classificado **antes** de você commitar o relatório. Toda decisão metodológica
que ainda não está pré-declarada em F.2 vira pergunta a Marcelo.

## Contexto verificado (confirme via comandos antes de iniciar)

Estado em main esperado:
- PR-A (Sessão-A E1) **+ PR-A2** (fix técnico schema explicit_configs, Emenda
  E1.6) **+ PR-A3** (cohort isolation + dim_run.fold + alignment check fix,
  Emenda E1.7) **+ PR-A4** (tier classification pipeline, Emenda E1.8) **todas
  mergeadas**. Branch `feat/phase-b-execution-<YYYYMMDD>` existe (mesma branch
  que você empilhará commits de E6).
- `parent_sweep_id` efetivo: `phase_b_confirmatorio_<YYYYMMDD>` (extrair de
  Notas de revisão E1 no checklist).
- Silver: 60 dim_run rows em cohort unificada (15 TFT + 45 baseline com
  `dim_run.fold` e `dim_run.seed` populados); gold: 25+ tables materializadas
  via refresh global; quality gate exit 0.
- **Sidecars Phase B prontos** em
  `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_<YYYYMMDD>/`
  (5 parquets gerados pela PR-A4: marginal_coverage, dm_family_6,
  dm_family_18_sensitivity, delta_pinball, tier_verdict). **Estes são seus
  inputs primários para E5.** Não recompute DM/Holm/calibração — leia direto.
- pytest 609+ passed; ruff clean; archive `data/analytics_archive_pre_phase_b/`
  intacto (851M).

Decisões D1–D8 fechadas (não redecidir; consulte tabela em
[`PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md)):
- **D5:** snapshot read-only obrigatório em E6.3.
- **D4:** PR-B é o segundo (e último) PR deste ciclo; empilhado na mesma
  branch `feat/phase-b-execution-<YYYYMMDD>` da Sessão-A.

## Required reading (na ordem; NÃO PULE)

1. [`PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md)
   Stages E5–E6 — **fonte primária**. Tasks E5.1–E5.6 + E6.1–E6.6 com Aceite.
   Leia também Notas de revisão E0–E4 deixadas pela Sessão-A (contexto) e
   "PR-A4 (pré-E5)" em E5 Notas (pipeline de pré-processamento que gera
   seus inputs).
2. [`preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md)
   §3 (H1/H2a/H2b), §6 (contrato post-guardrail), §7 (DM/MCS/Holm), §9
   (**Tier 1 / Tier 2 gates** — o coração do E5), §14 (Emendas E1.6, E1.7,
   E1.8 que formalizam decisões operacionais), §15 (slots a preencher em
   E6.2). **A Emenda E1.8 é especialmente importante** — declara protocolo
   operacional DM (unidade timestamp + dedup operationally-latest + seed
   mean + HAC + HLN + one-sided + Holm-6) que a PR-A4 implementa nos
   sidecars que você consome.
3. [`docs/04_evaluation/STATISTICAL_TESTS.md`](../04_evaluation/STATISTICAL_TESTS.md)
   — DM/MCS/Holm + regras de alinhamento OOS.
4. [`docs/04_evaluation/CALIBRATION_AND_RISK.md`](../04_evaluation/CALIBRATION_AND_RISK.md)
   §"Calibration Acceptance Thresholds" — bandas Tier 1/2 canônicas.
5. [`docs/04_evaluation/METRICS_DEFINITIONS.md`](../04_evaluation/METRICS_DEFINITIONS.md)
   §"Variante quantilica" (categorização A/B/C) — confirmar leitura
   `*_post_guardrail` em todas as métricas Cat A.
6. [`docs/00_overview/STRATEGIC_DIRECTION.md`](../00_overview/STRATEGIC_DIRECTION.md)
   §3 (calibração de expectativas) + §4.4 (resultados válidos com hipóteses
   refutadas) — informa tom do relatório se Tier não passar.
7. [`docs/07_reports/living-paper/20_method.md`](../07_reports/living-paper/20_method.md)
   §"Política de variante quantilica" — vocabulário e referências
   acadêmicas para o relatório.
8. [`docs/ai/skills/model-performance-and-research-advisor/SKILL.md`](skills/model-performance-and-research-advisor/SKILL.md)
   — **skill primária desta sessão**. Aplica o protocolo de decisão em E5.
9. [`docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md`](../06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md)
   — runbook canônico do CLI `main_compute_phase_b_tier_metrics` que gera
   seus sidecars (caso precise re-rodar; o output canônico já está em
   `data/analytics/reports/phase_b/`).
10. [`docs/04_evaluation/STATISTICAL_TESTS.md`](../04_evaluation/STATISTICAL_TESTS.md)
    §"Unidade estatística DM em walk-forward com folds sobrepostos"
    (adicionada pela PR-A4) — justifica a estratégia A refinada e por que
    sensibilidade D fica como apêndice.
11. Notas de revisão das Stages E0–E4 + "PR-A4 (pré-E5)" no checklist
    (contexto do que foi executado).

## Inputs primários E5 — sidecars Phase B

A PR-A4 (Emenda E1.8) entregou pipeline pré-processador que computa
mecanicamente os 6 testes DM declarados em F.2 §7, a calibração marginal
(coverage_q10/q50/q90), o delta_pinball_rel, e a classificação tier por
hipótese — TUDO ANTES da Sessão-B começar.

Você **NÃO recomputa** DM, Holm, calibração ou tier. Você **lê os 5
sidecars**, valida sanidade, interpreta cientificamente e escreve o
relatório E6.1.

Sidecars disponíveis em
`data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_<YYYYMMDD>/`:

| Sidecar | Shape | Conteúdo |
|---|---|---|
| `phase_b_marginal_coverage.parquet` | 60 rows | coverage_q10/q50/q90 por (run_id, split, horizon) |
| `phase_b_dm_family_6.parquet` | 6 rows | DM TFT-vs-baseline cross-fold por (horizon, baseline): dm_stat, pvalue_one_sided, pvalue_two_sided, pvalue_adj_holm (Holm sobre 6), n_obs_effective, hac_lag, hln_applied, direction |
| `phase_b_dm_family_18_sensitivity.parquet` | 18 rows | Mesma estrutura mas 3 folds × 6 (sensibilidade conservadora; **não** alimenta veredito) |
| `phase_b_delta_pinball.parquet` | 6 rows | delta_mean_pinball_rel por (horizon, baseline) |
| `phase_b_tier_verdict.parquet` | 6 rows | tier ∈ {tier_1, tier_2, refutado} por (horizon, hipótese ∈ {H1, H2a, H2b}) + criteria_passed dict + numerical_inputs + justification |

Se algum sidecar não existir ou shape divergir, **PAUSE e reporte** —
re-rodar via `python -m src.main_compute_phase_b_tier_metrics --asset AAPL
--parent-sweep-id phase_b_confirmatorio_<YYYYMMDD>` (ver runbook).

## Autoridade

| Task | Autoridade | Supervisão |
|---|---|---|
| E5.1 (carregar gold) | Autônoma. | — |
| E5.2 (gate degeneração) | Autônoma. | Reporte se algum grupo falhar. |
| E5.3 (calibração tier eligibility) | Autônoma. | — |
| E5.4 (DM Holm) | Autônoma. | Se Holm não vier pré-calculado em gold, declare como decidiu. |
| E5.5 (delta_pinball_rel) | Autônoma. | — |
| **E5.6 (classificação tier)** | **Supervisionada — confirme com Marcelo antes de commitar.** | Marcelo aprova o verdict final. |
| E6.1 (relatório) | Autônoma após E5.6 aprovado. | Marcelo revisa via PR-B. |
| E6.2 (emenda final) | Autônoma. | Conteúdo é mecânico (preencher slots). |
| E6.3 (archive snapshot) | Autônoma. | Confirme `chmod` aplicado. |
| E6.4–E6.5 (commits + PR-B) | Autônoma. | Marcelo revisa PR-B. |
| E6.6 (close checklist) | Após merge PR-B. | — |

## Comandos proibidos (sempre)

- `git rebase` / `git commit --amend` / `git push --force`
- Escrita em `data/analytics_archive_pre_phase_b/` (read-only)
- Escrita em `data/analytics_archive_phase_b_<date>/` após `chmod` aplicado
  em E6.3 (auto-violação)
- `--no-verify`
- **Re-executar** treino/baseline/refresh — outputs da Sessão-A são canônicos;
  re-rodar muda silver e invalida hash da config selada
- Modificar `src/` ou `tests/` (Phase B é análise, não modificação de código)
- Modificar checklists imutáveis (`PHASE_B_IMPLEMENTATION_CHECKLIST.md`,
  `POST_CLOSURE_FIXES_CHECKLIST.md`)
- **Decidir tier sob inspeção dos dados** (cherry-picking) — a classificação
  E5.6 deve seguir mecanicamente os critérios pré-declarados em F.2 §9

## Protected files (NÃO modificar)

- Todo `src/`, `tests/` (sem alterações de código).
- `data/analytics/silver/` e `data/analytics/gold/` (apenas leitura; outputs
  da Sessão-A são canônicos para o paper).
- `data/analytics_archive_pre_phase_b/`.
- `config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json` (selado em
  E1; mudar invalida hash em §15).
- `docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (imutável).
- ADRs em `docs/01_architecture/decisions/`.

**Permitido modificar:**
- `docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md` — apenas marcar
  E5–E6 como `[~]`/`[x]` e preencher Notas de revisão.
- `docs/06_pre_registration/preregistration_phase_b.md` — apenas §14
  (emenda E6) e §15 (slots restantes).
- `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` — adicionar nota
  sobre archive `analytics_archive_phase_b_<date>/` (E6.3).
- Criar `docs/07_reports/phase-gates/B_confirmatory_<YYYY-MM-DD>.md` (E6.1).
- Criar `data/analytics_archive_phase_b_<YYYYMMDD>/` (E6.3).

## Skills a invocar (proativo)

- [`model-performance-and-research-advisor`](skills/model-performance-and-research-advisor/SKILL.md):
  **skill primária E5**. Aplique protocolo:
  - Step 1 Lock cohort: cohort = `parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>`
    apenas; rejeite mistura.
  - Step 2 Quality gates: confirme `failed_checks=[]` (já validado pela
    Sessão-A; re-verifique).
  - Step 3 Point layer: RMSE/MAE/DA por horizonte/seed/fold.
  - Step 4 Probabilistic layer: pinball/PICP/MPIW, com penalização para
    degeneração.
  - Step 5 Statistical dominance: DM + MCS + win-rate **juntos**.
  - Step 6 Robustness: dispersão fold/seed; bandwidth CI95.
  - Step 7 Interpretability: out-of-scope (Fase C); declare explicitamente
    no relatório.
  - Step 8 Decision: tier classificado **mecanicamente** dos critérios
    pré-declarados F.2 §9. Sem reframing.
- [`documentation-authoring`](skills/documentation-authoring/SKILL.md): E6.1
  estilo de redação científica do relatório; cross-link sem duplicação.
- [`staged-implementation-protocol`](../../.claude/skills/staged-implementation-protocol/SKILL.md)
  Phase B step 8: Notas de revisão por Stage. Phase D (review prompt) não
  se aplica nesta sessão (PR-B é mergeado por Marcelo após revisão dele,
  não por sessão Claude separada).

## Sucesso definido

Sessão-B entrega:
1. **E5.6 tier classificado** (Tier 1 / Tier 2 / Refutado por horizonte e
   por hipótese), aprovado por Marcelo, registrado em
   `/tmp/tier_verdict_<YYYYMMDD>.csv`.
2. **E6.1 relatório** `docs/07_reports/phase-gates/B_confirmatory_<YYYY-MM-DD>.md`
   ≥200 LOC, contendo: parâmetros executados (config sha256, parent_sweep_id,
   dataset sha256, commit selo pre-registro), resultados E5 (tabelas tier
   eligibility + tier verdict; agregados TFT vs baselines; calibração; DM/MCS/
   win-rate; gate degeneração), limitações (cita F.2 §12 + STRATEGIC_DIRECTION
   §3), cross-links.
3. **E6.2 emenda final** §15 do pré-registro com 6/6 slots preenchidos +
   nova entrada datada em §14.
4. **E6.3 archive snapshot** `data/analytics_archive_phase_b_<YYYYMMDD>/`
   read-only (`chmod a-w`) + nota em `ANALYTICS_STORE_ARCHITECTURE.md`.
5. **E6.4–E6.5 PR-B aberta** com 3 commits empilhados:
   - `docs(phase-b-exec): adicionar relatorio B_confirmatory_<YYYY-MM-DD>`
   - `docs(pre-reg): emenda E6 — preencher §15 execucao + entrada §14`
   - `chore(archive): snapshot read-only Phase B + nota ANALYTICS_STORE_ARCHITECTURE`
6. **E6.6 (após merge PR-B):** checklist atualizado para `Status: completo
   (YYYY-MM-DD)` + E0–E6 todos `[x]`.

## Procedimento detalhado

### E5 — Análise vs Tier 1/Tier 2 gates (~4–8h)

#### E5.0 — Pre-flight

```bash
cd /home/marcelo/Code/financial-time-series-forecasting
git checkout main && git pull --ff-only
# Identificar parent_sweep_id efetivo da Sessao-A:
grep "parent_sweep_id efetivo" docs/06_pre_registration/preregistration_phase_b.md
# Anote o <YYYYMMDD>. Checkout da branch da Sessao-A:
git checkout feat/phase-b-execution-<YYYYMMDD>
git merge --ff-only main  # sincroniza com main
```

Confirme:
- `du -sh data/analytics_archive_pre_phase_b/` → `851M` (intacto).
- `ls data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>/ | wc -l` → `60`.
- `ls data/analytics/gold/*.parquet | wc -l` → `≥25`.
- pytest + ruff: PASS.

#### E5.1 — Carregar gold tables

```python
import pandas as pd
from pathlib import Path
gold = Path('data/analytics/gold')
sweep = 'phase_b_confirmatorio_<YYYYMMDD>'

tables = {}
for name in [
    'gold_prediction_metrics_by_run_split_horizon',
    'gold_prediction_metrics_by_config',
    'gold_dm_pairwise_results',
    'gold_mcs_results',
    'gold_win_rate_pairwise_results',
    'gold_paired_oos_intersection_by_horizon',
    'gold_quantile_degeneracy_report',
    'gold_quality_statistics_report',
]:
    df = pd.read_parquet(gold / f'{name}.parquet')
    df = df[df['parent_sweep_id'] == sweep]
    tables[name] = df
    print(f'{name}: {len(df)} rows')
```

Confirme row counts esperados:
- `metrics_by_run_split_horizon`: 60 runs × 2 splits × 2 horizons = 240 rows
  (ou subset se some baselines são point-only em h=7).
- `dm_pairwise_results`: TFT vs cada baseline × 2 horizontes = 6+ rows
  (split=test).
- `mcs_results`: 4 modelos × 2 horizontes (split=test).
- `quantile_degeneracy_report`: 60 runs × 2 splits × 2 horizons (subset
  para `prediction_mode='quantile'`).

#### E5.2 — Gate degeneração (precondição para tier)

```python
deg = tables['gold_quantile_degeneracy_report']
# Filtre quantile-genuine (exclui baselines point: zero_return, historical_mean_rolling)
deg_q = deg[deg['prediction_mode'] == 'quantile']
print(deg_q[['split', 'horizon', 'feature_set_name', 'model', 'p10_eq_p90_rate', 'gate_passed']].to_string())
all_pass = deg_q['gate_passed'].all()
print(f'Gate degeneracao PASS: {all_pass}')
```

Se algum grupo TFT tem `gate_passed=False`, a run correspondente é **excluída
automaticamente de claims probabilísticos** per F.2 §9. Reporte a Marcelo
imediatamente; pause E5.

#### E5.3 — Calibration eligibility (Tier 1/2)

Para cada (run, split=test, horizon ∈ {1,7}), extraia de
`metrics_by_run_split_horizon` colunas `*_post_guardrail`:
- `coverage_q10_post_guardrail`, `coverage_q50_post_guardrail`,
  `coverage_q90_post_guardrail` (= `picp_qN_post_guardrail` se existir;
  caso contrário derive de `quantile_pN` vs `y_true`).
- `coverage_error_post_guardrail` (= `picp_post_guardrail - 0.80`).
- `mpiw_post_guardrail`.

Filtre só rows do TFT (`feature_set_name != 'baseline'`). Para cada
horizonte, agregue (mediana ou média) sobre os 15 runs (5 seeds × 3 folds).

Aplique gates **Tier 1** (F.2 §9):
- `cov_q10 ∈ [0.07, 0.13]` AND `cov_q50 ∈ [0.45, 0.55]` AND `cov_q90 ∈ [0.87, 0.93]`
- `|coverage_error_post_guardrail| ≤ 0.02`
- `mpiw_post_guardrail > 0` e finito

E **Tier 2** (mais permissivo):
- `cov_q10 ∈ [0.05, 0.15]` AND `cov_q50 ∈ [0.40, 0.60]` AND `cov_q90 ∈ [0.85, 0.95]`
- `|coverage_error_post_guardrail| ≤ 0.05`

Salve `/tmp/tier_eligibility_calibration_<YYYYMMDD>.csv` com colunas
`(horizon, agg_method, cov_q10, cov_q50, cov_q90, picp_err, mpiw, passa_tier1_calib, passa_tier2_calib)`.

#### E5.4 — DM Holm-Bonferroni

Carregue `dm_pairwise_results` para split=test, horizon ∈ {1,7}. Confirme
existência de coluna `pvalue_adj_holm`.

- Se existe: filtre TFT vs cada baseline (6 pares); registre
  `pvalue_adj_holm < 0.05` (Tier 1) e `< 0.10` (Tier 2).
- Se NÃO existe: calcule manualmente:
  ```python
  from statsmodels.stats.multitest import multipletests
  dm = tables['gold_dm_pairwise_results']
  dm = dm[(dm['split'] == 'test') & (dm['horizon'].isin([1, 7]))]
  # Filtre pares TFT vs baseline (não baseline vs baseline)
  is_tft_vs_baseline = (
      (dm['model_a'].str.contains('TFT', case=False)) ^
      (dm['model_b'].str.contains('TFT', case=False))
  )
  family = dm[is_tft_vs_baseline].copy()
  assert len(family) == 6, f'Esperado 6 pares; obtido {len(family)}'
  family['pvalue_adj_holm_manual'] = multipletests(family['pvalue'], method='holm')[1]
  ```
  Declare no commit body que calculou manualmente.

#### E5.5 — Delta pinball relativo

```python
metrics = tables['gold_prediction_metrics_by_run_split_horizon']
metrics = metrics[metrics['split'] == 'test']

# Agregue mean_pinball_post_guardrail por (model_label, feature_set_name, horizon)
agg = metrics.groupby(['feature_set_name', 'horizon'])['mean_pinball_post_guardrail'].median()

# Para cada horizonte:
for h in [1, 7]:
    tft = agg.xs((<TFT_feature_set_name>, h))  # ex: 'BASELINE+TECHNICAL+SENTIMENT+FUNDAMENTAL'
    for baseline_name in ['zero_return', 'historical_mean_rolling', 'historical_quantiles_rolling']:
        try:
            base = agg.xs(('baseline', h))  # ou conforme schema
            # Filtre por model_label = baseline_name
            delta_rel = (base - tft) / base
            print(f'h={h}, baseline={baseline_name}: delta_rel={delta_rel:.4f}, passa_tier1={delta_rel >= 0.03}')
        except KeyError:
            pass
```

(Ajuste schema baseado nos colunas reais; pode ser que TFT esteja como
`feature_set_name='BTSF'` ou similar — inspecione `metrics.feature_set_name.unique()`.)

#### E5.6 — Classificação tier (PAUSE para Marcelo aprovar)

Compile `/tmp/tier_verdict_<YYYYMMDD>.csv`:
```
horizon,hypothesis,tier,calib_pass,dm_holm_pass,delta_rel_pass,justificativa
1,H1,<tier>,...
1,H2a,<tier>,...
1,H2b,<tier>,...
7,H1,<tier>,...
7,H2a,<tier>,...
7,H2b,<tier>,...
```

Regras:
- **Tier 1** se calibração Tier 1 PASS **E** DM Holm < 0.05 PASS **E**
  delta_rel ≥ 3% PASS **E** MPIW > 0.
- **Tier 2** se Tier 1 falhar mas calibração Tier 2 PASS **E** DM Holm
  < 0.10 PASS **E** delta_rel ≥ 0%.
- **Refutado** se nem Tier 1 nem Tier 2.

H1 usa apenas calibração (não envolve DM). H2a usa baseline=zero_return.
H2b usa todos baselines (precisa de evidência forte em todos os 3).

**Reporte a Marcelo:**
> "E5.6 verdict draft:
> - h=1, H1: <tier> — calib: cov_q10=<>, cov_q50=<>, cov_q90=<>, picp_err=<>; MPIW=<>.
> - h=1, H2a: <tier> — DM Holm vs zero_return: <pvalue_adj_holm>; delta_rel: <%>.
> - h=1, H2b: <tier> — DM Holm vs zero_return: <p>, hist_mean: <p>, hist_quant: <p>; delta_rel: <%>, <%>, <%>.
> - h=7, ... (idem)
> Por favor confirme antes de eu commitar o relatório E6.1."

Aguarde aprovação. Se Marcelo questionar uma classificação, **NÃO mude o
verdict para agradar** — defenda baseado em F.2 §9 critérios ex-ante. Se
houver erro de cálculo, corrija o cálculo, não o critério.

### E6 — Relatório + emenda final + archive (~4–6h)

#### E6.1 — Relatório B_confirmatory_<YYYY-MM-DD>.md

Template mínimo (≥200 LOC):

```markdown
---
title: Phase B Confirmatory — Relatório (<YYYY-MM-DD>)
scope: Relatorio cientifico da Phase B confirmatoria. Reporta tier por horizonte (H1/H2a/H2b) contra Tier 1/Tier 2 gates pre-declarados em preregistration_phase_b.md §9. Cita config sealed sha256, parent_sweep_id, dataset hash, commit selo pre-registro. Cobre limitacoes per F.2 §12 e STRATEGIC_DIRECTION §3.
update_when:
  - tier reportado for reclassificado (exige emenda em §14 do pre-registro)
  - rodada Phase-B-bis (sensibilidade B.4) for adicionada como apendice
canonical_for: [phase_b_confirmatory_report]
---

# Phase B Confirmatory — Relatório (<YYYY-MM-DD>)

**Veredicto:** Tier 1 PASS (...) / Tier 2 PASS (...) / REFUTADO (...) por
horizonte e hipótese.

## 1. Parâmetros executados

- **`parent_sweep_id`:** `phase_b_confirmatorio_<YYYYMMDD>`
- **Config sealed sha256:** `<hash>`
- **Dataset sha256:** `<hash AAPL parquet>`
- **Commit selo pre-registro:** `<sha do commit em main>`
- **Heritage sweep (E0):** `phase_b_hpo_<date>`
- **Hiperparams TFT:** (tabela mínima)
- **Folds walk-forward:** 3 folds, janelas declaradas no JSON sealed
- **Seeds:** [20260517, 20260518, 20260519, 20260520, 20260521]
- **Baselines:** zero_return, historical_mean_rolling (w=30),
  historical_quantiles_rolling (w=252)

## 2. Resultados — Tier classification

(Tabela tier_verdict por horizonte/hipótese; cite F.2 §9 critérios)

## 3. Resultados — Calibração quantil

(Tabela cov_q10/q50/q90 + PICP_err + MPIW per horizonte; mediana + IC sobre 15 runs)

## 4. Resultados — Statistical tests

### 4.1 DM pairwise + Holm
(Tabela 6 pares TFT vs baseline × 2 horizontes; pvalue + pvalue_adj_holm)

### 4.2 MCS
(Lista de modelos no MCS α=0.05 per horizonte)

### 4.3 Win-rate
(Tabela win-rate TFT vs cada baseline per horizonte)

## 5. Gate de degeneração quantilica

(Tabela: 60 grupos × p10_eq_p90_rate; gate_passed)

## 6. Robustness

(Dispersão fold/seed: RMSE/pinball CV; flag fragile winners)

## 7. Limitações declaradas

- Single-asset (AAPL) per F.2 §2; sem claim de generalização.
- N efetivo para h=7 em test (3 anos OOS): cite per F.2 §B.3.
- h=30 out-of-scope per F.2 §8.
- Walk-forward sensibilidade B.4: rodada futura Phase-B-bis (decisao D6).
- TFT não é dominante em retornos de ações líquidas per STRATEGIC_DIRECTION §3.
- Contribuição de famílias: out-of-scope Phase B; Fase C cobre H3.

## 8. Conclusão

(Tom: honesto. Se Tier 1 PASS, claim probabilístico defensável. Se Tier 2,
"evidência secundária / não confirmatória". Se refutado, "TFT não supera
baselines neste protocolo" + direcionar para diagnóstico calibração per
STRATEGIC_DIRECTION §3 e §4.4.)

## 9. Cross-links

| Tema | Documento |
|---|---|
| Pre-registro | docs/06_pre_registration/preregistration_phase_b.md |
| Checklist execução | docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md |
| Direção estratégica | docs/00_overview/STRATEGIC_DIRECTION.md §5 Fase B |
| Statistical tests | docs/04_evaluation/STATISTICAL_TESTS.md |
| Calibration thresholds | docs/04_evaluation/CALIBRATION_AND_RISK.md |
| Smoke F.1 PASS pleno | docs/07_reports/smoke_confirmatory_2026-05-18.md §"F.1 v4 PASS pleno" |
| Archive Phase B | data/analytics_archive_phase_b_<YYYYMMDD>/ (read-only) |

## 10. Referências acadêmicas

(Cite Diebold-Mariano 1995, Harvey-Leybourne-Newbold 1997, Hansen-Lunde-Nason
2011, Holm 1979, Gneiting-Raftery 2007, Chernozhukov-Fernandez-Val-Galichon
2010, Gu-Kelly-Xiu 2020, Lim et al. 2021. Reaproveite §17 do pré-registro
para sintaxe.)
```

#### E6.2 — Emenda final §15 + entrada §14

Edite `preregistration_phase_b.md`:

§15 "Execucao": preencha slots restantes:
- **Data de execucao confirmatoria:** `<YYYY-MM-DD>` (data real de E2.1 start).
- **Resultado tier (H1, H2a, H2b por horizonte):** `H1@h=1: <tier>; H1@h=7:
  <tier>; H2a@h=1: <tier>; H2a@h=7: <tier>; H2b@h=1: <tier>; H2b@h=7: <tier>`.
- **Relatorio:** `docs/07_reports/phase-gates/B_confirmatory_<YYYY-MM-DD>.md`.

§14 "Emendas": anexe entrada datada:
```
### <YYYY-MM-DD> — Emenda E6: execucao concluida
- Decisao alterada: nenhuma; preenchimento de §15.
- Cross-link: PR-B (a abrir), relatorio B_confirmatory_<YYYY-MM-DD>.md.
- Justificativa: encerramento operacional do ciclo Phase B confirmatorio.
```

#### E6.3 — Archive snapshot read-only

```bash
ARCHIVE="data/analytics_archive_phase_b_<YYYYMMDD>"
SWEEP="phase_b_confirmatorio_<YYYYMMDD>"

# Confirmar que o archive sera ignorado pelo git antes de criar/copiar Parquets.
# A regra esperada hoje eh `.gitignore:data/**`.
git check-ignore -v "$ARCHIVE/__probe__.parquet" || {
  echo "ERRO: $ARCHIVE nao esta coberto por .gitignore; corrija antes de criar snapshot."
  exit 1
}

mkdir -p "$ARCHIVE/silver" "$ARCHIVE/gold"

# Silver: snapshot apenas da cohort
cp -r "data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=$SWEEP" \
      "$ARCHIVE/silver/dim_run_$SWEEP"

# fact_oos_predictions filtrado por run_id da cohort (precisa de script Python)
.venv/bin/python <<'PY'
import pandas as pd, shutil
from pathlib import Path
sweep = 'phase_b_confirmatorio_<YYYYMMDD>'
silver = Path('data/analytics/silver')
archive = Path(f'data/analytics_archive_phase_b_<YYYYMMDD>/silver')

# Identificar run_ids da cohort via dim_run
dim_partitions = list((silver / 'dim_run' / 'asset=AAPL' / f'parent_sweep_id={sweep}').rglob('*.parquet'))
dims = pd.concat([pd.read_parquet(p) for p in dim_partitions])
cohort_run_ids = set(dims['run_id'])
print(f'Cohort runs: {len(cohort_run_ids)}')

# Copiar fact_oos_predictions filtrado
for fact_p in (silver / 'fact_oos_predictions').rglob('*.parquet'):
    df = pd.read_parquet(fact_p)
    if 'run_id' in df.columns:
        df_cohort = df[df['run_id'].isin(cohort_run_ids)]
        if len(df_cohort) > 0:
            rel = fact_p.relative_to(silver / 'fact_oos_predictions')
            dest = archive / 'fact_oos_predictions' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            df_cohort.to_parquet(dest, index=False)
            print(f'  Copied {rel}: {len(df_cohort)} rows')
PY

# Gold: filtrar 25 tabelas por parent_sweep_id
.venv/bin/python <<'PY'
import pandas as pd
from pathlib import Path
sweep = 'phase_b_confirmatorio_<YYYYMMDD>'
gold = Path('data/analytics/gold')
archive = Path(f'data/analytics_archive_phase_b_<YYYYMMDD>/gold')
archive.mkdir(exist_ok=True, parents=True)

for parquet in gold.glob('*.parquet'):
    df = pd.read_parquet(parquet)
    if 'parent_sweep_id' in df.columns:
        df_cohort = df[df['parent_sweep_id'] == sweep]
        if len(df_cohort) > 0:
            df_cohort.to_parquet(archive / parquet.name, index=False)
            print(f'  {parquet.name}: {len(df_cohort)} rows')
PY

chmod -R a-w "$ARCHIVE"
du -sh "$ARCHIVE"

# Verifique escrita bloqueada
touch "$ARCHIVE/test_write" 2>&1 | head -1
```

Esperado: erro `Permission denied`.
Se `git check-ignore` nao mostrar a regra aplicavel antes do `mkdir`, pare e
corrija `.gitignore` antes de qualquer `git add`. Nunca versionar Parquets do
snapshot `data/analytics_archive_phase_b_<YYYYMMDD>/`.

Adicione nota em `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`:
```markdown
### Archive Phase B confirmatorio (<YYYY-MM-DD>)

`data/analytics_archive_phase_b_<YYYYMMDD>/` (read-only) — snapshot dos
artefatos confirmatórios da Phase B (60 silver dim_run rows + 25 gold tables
filtradas por `parent_sweep_id=phase_b_confirmatorio_<YYYYMMDD>`). Dados
referenciados pelo relatório `B_confirmatory_<YYYY-MM-DD>.md` e pelo
pré-registro §15. Não modificar; em caso de necessidade de re-derivar
métricas, reproducir via `config/sweeps/explicit/phase_b_confirmatorio_<YYYYMMDD>.json`
(sealed sha256: `<hash>`).
```

#### E6.4 — Commits empilhados (3)

```bash
# Commit 1 — relatorio
git add docs/07_reports/phase-gates/B_confirmatory_<YYYY-MM-DD>.md
git commit -m "$(cat <<'EOF'
docs(phase-b-exec): adicionar relatorio B_confirmatory_<YYYY-MM-DD>

Veredicto: <tier sumarizado>.
Tier classification per F.2 §9 (Tier 1/Tier 2/Refutado) por horizonte e hipotese.
Cross-link com pre-registro selado em E1 (sha256 <hash>; parent_sweep_id
phase_b_confirmatorio_<YYYYMMDD>).

Closes E6.1 do PHASE_B_EXECUTION_CHECKLIST.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

# Commit 2 — emenda
git add docs/06_pre_registration/preregistration_phase_b.md
git commit -m "$(cat <<'EOF'
docs(pre-reg): emenda E6 — preencher §15 execucao + entrada §14

§15 Execucao: 6/6 slots preenchidos (data execucao, parent_sweep_id, sha256
config, commit selo, tier reportado por hipotese/horizonte, relatorio).
§14 Emendas: entrada <YYYY-MM-DD> registra encerramento operacional.

Closes E6.2 do PHASE_B_EXECUTION_CHECKLIST.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"

# Commit 3 — archive
git add docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md
# Archive em si nao vai para git (silver/gold parquets); apenas a doc note.
git commit -m "$(cat <<'EOF'
chore(archive): snapshot read-only Phase B + nota ANALYTICS_STORE_ARCHITECTURE

Snapshot em data/analytics_archive_phase_b_<YYYYMMDD>/ (read-only via chmod
a-w) contem subset do silver/gold filtrado por parent_sweep_id=
phase_b_confirmatorio_<YYYYMMDD>. Padrao espelha Stage 3 do
PHASE_B_IMPLEMENTATION_CHECKLIST (archive pre-Phase B).

Archive nao versionado em git (parquets); apenas a nota documental.

Closes E6.3 do PHASE_B_EXECUTION_CHECKLIST.md.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

#### E6.5 — Push + PR-B

```bash
git push origin feat/phase-b-execution-<YYYYMMDD>
gh pr create --base main --head feat/phase-b-execution-<YYYYMMDD> \
  --title "docs(phase-b-exec): relatorio Phase B + emenda final + archive snapshot" \
  --body "<body com: tier classificacao por hipotese, link relatorio, link
   archive, link checklist E5-E6, decisoes D1-D8 referenciadas>"
```

#### E6.6 — Após merge PR-B

```bash
git checkout main && git pull --ff-only
git checkout -b chore/close-phase-b-execution-checklist
```

Edite `PHASE_B_EXECUTION_CHECKLIST.md`:
- Header `Status: pendente → completo (YYYY-MM-DD)`.
- Marque E0–E6 todas como `[x]` (já marcadas pela Sessão-A E0–E4; agora
  E5–E6).

Commit + push + PR (chore-only). Após merge, ciclo Phase B confirmatório
fechado.

Atualize `STRATEGIC_DIRECTION.md` Fase B status se houver flag de progresso
(grep `Fase B` e revisar).

## Init commands

```bash
cd /home/marcelo/Code/financial-time-series-forecasting
git checkout main && git pull --ff-only
git log --oneline -10  # confirmar merge PR-A da Sessao-A

# Identificar parent_sweep_id e checkout da branch da Sessao-A
SWEEP=$(grep "parent_sweep_id efetivo" docs/06_pre_registration/preregistration_phase_b.md | head -1 | awk -F'`' '{print $2}')
echo "Cohort: $SWEEP"
DATE=$(echo "$SWEEP" | sed 's/phase_b_confirmatorio_//')
echo "Date: $DATE"

git checkout "feat/phase-b-execution-$DATE"
git merge --ff-only main

# Pre-flight
du -sh data/analytics_archive_pre_phase_b/   # 851M
ls "data/analytics/silver/dim_run/asset=AAPL/parent_sweep_id=$SWEEP/" | wc -l   # 60
ls data/analytics/gold/*.parquet | wc -l   # >=25
.venv/bin/pytest tests/ -q \
  --ignore=tests/integration/test_quality_registry_bit_identical_archive.py \
  --ignore=tests/integration/test_gold_builders_byte_identical_archive.py 2>&1 | tail -3
.venv/bin/ruff check src/ tests/

# Confirmar quality gate (re-validação leve sem refresh)
.venv/bin/python <<'PY'
import pandas as pd
q = pd.read_parquet('data/analytics/gold/gold_quality_statistics_report.parquet')
sweep = '<from above>'
q_cohort = q[q['parent_sweep_id'] == sweep]
print(q_cohort[['split', 'horizon', 'statistics_ready', 'quality_passed_all']])
PY
```

Após init, leia o Required reading e inicie **E5.1**.

## Constraints inegociáveis

- NUNCA re-execute treino/baseline/refresh — outputs Sessão-A são canônicos.
- NUNCA decida tier sob inspeção dos dados (cherry-picking) — siga F.2 §9.
- NUNCA toque archive `data/analytics_archive_pre_phase_b/` (read-only).
- NUNCA `git rebase`, `git commit --amend`, `git push --force`.
- NUNCA mude config sealed `phase_b_confirmatorio_<YYYYMMDD>.json` (invalida
  hash em §15).
- Marcelo aprova tier verdict (E5.6) antes de commitar E6.1.
- Se Marcelo questionar uma classificação, defenda baseado em F.2 §9 — não
  mude para agradar; corrija apenas erros de cálculo.
- Se algum dado contradiz pré-registro (ex: alinhamento OOS quebrou), pause
  e reporte; não tente fix de código.

## Sucesso

Sua sessão termina com:
- E5 todas tasks `[x]` + Notas de revisão preenchidas; tier verdict aprovado.
- E6 todas tasks `[x]` (E6.1–E6.5 antes de merge; E6.6 após merge).
- Relatório `B_confirmatory_<YYYY-MM-DD>.md` mergeado em main.
- §15 do pré-registro 6/6 slots preenchidos; §14 com entrada E6.
- Archive `data/analytics_archive_phase_b_<YYYYMMDD>/` read-only existente.
- PR-B mergeada em main.
- Checklist marcado `Status: completo (YYYY-MM-DD)`.
- Archive 851M pre-Phase B intacto.
- Ciclo Phase B confirmatório fechado; Fase C autorizada a iniciar
  (próximo passo: contribuição de features per STRATEGIC_DIRECTION §5 Fase C).
