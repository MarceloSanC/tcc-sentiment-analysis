---
title: Post-Closure Fixes Checklist (Smoke F.1 e iteracoes subsequentes)
scope: Rastreamento de fixes descobertos APOS o closure parcial da Fase A (A_audit_closure_2026-05-17.md), via execucao do smoke F.1 (smoke_confirmatory_2026-05-18.md) e suas re-rodadas. Cada onda de descobertas = 1 Stage; cada task = 1 commit (mesma convencao de PHASE_B_IMPLEMENTATION_CHECKLIST.md). Escopo separado para preservar PHASE_B como snapshot imutavel do pre-experiment engineering planejado a partir de A_code_audit.md.
update_when:
  - smoke F.1 (ou iteracao subsequente) descobrir novo gap nao coberto pelo escopo original de PHASE_B_IMPLEMENTATION_CHECKLIST.md
  - task de fix post-closure for aberta, em revisao ou mergeada
  - nova "onda" de descoberta abrir um novo Stage
  - decisao for tomada de promover fix para Stage proprio em PHASE_B (alteracao arquitetural profunda)
canonical_for: [post_closure_fixes, smoke_iteration_findings, phase_b_blockers]
---

# Post-Closure Fixes — Checklist

**Status:** Stage F.0 mergeado (PR #40, 2026-05-19). Stage F.A registro mergeado
(PRs #41-#43, 2026-05-20). Stages 20-23 originalmente mergeados em
PRs #44-#47 (2026-05-20) com REQUEST_CHANGES posterior; remediados em
Stages R-20/R-E/R-21/R-22/R-23 (PRs #50-#54, 2026-05-21 a 2026-05-22)
e cleanup do bridge transitorio em PR #56 (2026-05-22). Gap 6 fechado via
**Opcao (d)** do ADR-0003 em R-20 (target_return backward-shifted em
`build_tft_dataset_use_case`; baseline `y_true_idx = i + h_int`), nao
mais Opcao (a). F.1 PASS pleno 6/6 em R-23 v4 (`max_epochs=5`,
sweep `phase_a_smoke_20260522_r23`).
**Criado:** 2026-05-19 (apos separacao de [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md)).
**Pre-requisito:** [`A_audit_closure_2026-05-17.md`](../../07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md) com amendment "Findings post-closure".
**Bloqueante para:** F.1 smoke PASS pleno → abertura da Fase B (ver `Stage final` em PHASE_B).

Este checklist rastreia fixes descobertos **apos** o closure parcial da
Fase A — quando o smoke F.1 ou suas re-rodadas revelam gaps nao
cobertos pelos Stages 1-14 do `PHASE_B_IMPLEMENTATION_CHECKLIST.md`.
Foi separado de PHASE_B para:

1. Preservar PHASE_B como snapshot imutavel do pre-experiment
   engineering planejado a partir de `A_code_audit.md`. Checklists
   com escopo concluido sao tratados como imutaveis no projeto.
2. Restaurar a convencao "1 Stage = 1 PR" sem espremer sub-IDs
   cada vez maiores (F.0.0 … F.0.10 …) num Stage unico.
3. Acomodar futuras "ondas" de fixes descobertos em re-runs do smoke
   sem inflar PHASE_B.

## Convencao

- Mesma de [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md) §Convencao:
  `[ ]` pendente, `[x]` mergeado, `[~]` em revisao; Stage = 1 PR;
  task numerada = 1 commit; **Aceite:** ao fim de cada task.
- **Stage = onda de descoberta.** A primeira onda (5 gaps originais do
  smoke 2026-05-18) eh `Stage F.0`. Re-rodadas subsequentes que
  descobrirem novos gaps abrem novo Stage.
- **Sub-IDs de tasks preservados.** F.0.0 … F.0.9 sao mantidos
  literalmente porque ja estao referenciados em git commits
  (`feat/stage-f0-pre-smoke-fixes`), em
  [`A_audit_closure_2026-05-17.md`](../../07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md),
  em [`smoke_confirmatory_2026-05-18.md`](../../07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md)
  e em [`SWEEPS_AND_SELECTION.md`](../../03_modeling/SWEEPS_AND_SELECTION.md).
  Renomear quebraria rastreabilidade.
- **Promocao para PHASE_B.** Se um fix evoluir para alteracao
  arquitetural profunda (ex: Gap 6 que pode demandar trabalho do porte
  de um Stage 15+), ele eh promovido para Stage proprio em PHASE_B em
  vez de virar mais um sub-ID aqui.

## Cross-link com PHASE_B e gates

- [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md) — escopo original (Stages 1-19 + Stage final F.1/F.2/F.3 pre-experiment engineering).
- [`A_audit_closure_2026-05-17.md`](../../07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md) §"Findings post-closure" — relatorio de closure que disparou Stage F.0 (amendment 2026-05-18).
- [`smoke_confirmatory_2026-05-18.md`](../../07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md) — relatorio do smoke F.1 (run inicial FAIL + re-run pos-Stage F.0).
- [`tft_y_true_investigation_2026-05-18.md`](../../07_reports/phase-gates/phase-a/tft_y_true_investigation_2026-05-18.md) — investigacao do Gap 6 (TFT y_true convention) descoberto durante re-run de F.0.5; candidato a nova onda (Stage proprio neste arquivo ou Stage promovido em PHASE_B).

---

## Stage F.0 — Fixes pre-smoke (descobertos via F.1 2026-05-18)

**Objetivo:** desbloquear F.1 smoke confirmatorio com PASS em todos os
criterios. F.1 executado em 2026-05-18 reportou veredicto FAIL com 5 gaps
estruturais nao cobertos pelos reviews dos Stages 1-14:

1. Stage 12 entregou `RunBaselinesUseCase` sem CLI canonico.
2. TFT vs baselines `target_timestamp_utc` misaligned por warmup
   (encoder 60d).
3. Bug pre-existente `--help` em `main_refresh_analytics_store`
   (`%` nao escapado).
4. `gold_confidence_calibrated_by_horizon` falsa-positivo em point
   baselines (cross-Stage 9 nao testado).
5. `official_contract_quantile_attention` exigia artefato torch de
   baselines (cross-Stage 10 nao testado).

Decisao arquitetural: TFT e baselines tem orquestradores independentes
consumindo o mesmo JSON config schema; alinhamento por configuracao
compartilhada, nao por acoplamento de processo. Sibling architecture:
`main_baselines_test_pipeline.py` espelha `main_tft_test_pipeline.py`.

### Notas de revisao:

- 2026-05-18: Stage F.0 implementado pos-F.1 smoke FAIL. Evidencia:
  - [`docs/07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md`](../../07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md) (relatorio inicial + re-rodada).
  - [`docs/07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md`](../../07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md) §"Findings post-closure" (amendment 2026-05-18).
  - 5 gaps documentados; F.0.0-F.0.9 abordam cada um.
- 2026-05-19: Stage F.0 migrado de `PHASE_B_IMPLEMENTATION_CHECKLIST.md`
  para este arquivo para separar fixes post-closure do escopo pre-experiment
  engineering original. Conteudo preservado verbatim; apenas anchor
  interno `#stage-final-...` reapontado para `PHASE_B_IMPLEMENTATION_CHECKLIST.md`.

### Tasks

- [x] **F.0.0** Schema da secao `baselines` em sweep JSON config.
      **Aceite:** schema documentado em
      [`SWEEPS_AND_SELECTION.md`](../../03_modeling/SWEEPS_AND_SELECTION.md)
      §"Schema da secao baselines"; ao menos 1 config real em
      `config/sweeps/explicit/*.json` com seção `baselines` como exemplo;
      novo config `phase_a_smoke_20260518_v2.json` para re-rodada do smoke.

- [x] **F.0.1** Standalone CLI `src/main_run_baselines.py` (debug).
      **Aceite:** `python -m src.main_run_baselines --help` retorna exit 0;
      `RunBaselinesUseCase` reinvocavel sem JSON; runbook em
      [`RUN_BASELINES.md`](../../06_runbooks/RUN_BASELINES.md). NAO substitui
      caminho canonico (F.0.2).

- [x] **F.0.2** Sibling test pipeline canonico.
      **Aceite:** `src/use_cases/run_baselines_test_pipeline_use_case.py`
      + `src/main_baselines_test_pipeline.py`; mesmo JSON do
      `main_tft_test_pipeline` alimenta ambos; `parent_sweep_id` derivado
      de `output_subdir`; alinhamento warmup via
      `evaluation_start_offset_days = max_encoder_length + max_prediction_length - 1`
      (TFT TimeSeriesDataSet skip empirico, validado pos-F.1 v2 smoke);
      `evaluation_end_offset_days = 0`. Aceita parametros aditivos em
      `RunBaselinesUseCase.execute()` sem quebrar Stage 12.

- [x] **F.0.3** Defense-in-depth gate em quality validator.
      **Aceite:** check `tft_baselines_timestamp_subset_alignment` em
      `validate_analytics_quality_use_case.py`; ativo em
      `scope_mode=cohort_decision`; falha se
      `set(target_timestamp_utc | TFT) != set(target_timestamp_utc | baselines)`.
      Regression guard para Cenario Falha D do F.1.

- [x] **F.0.4** Fix `--help` em `main_refresh_analytics_store`.
      **Aceite:** `python -m src.main_refresh_analytics_store --help`
      retorna exit 0 (escape `%%` em help strings).

- [x] **F.0.5** Re-rodar F.1 smoke com fixes F.0.0-F.0.4 + F.0.6-F.0.7.
      **Aceite:** todos 6 criterios PASS conforme aceite literal de
      [F.1](../phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md#stage-final--smoke-confirmatorio--pre-registro-bc);
      relatorio atualizado em `docs/07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md`
      secao "Re-run pos-Stage F.0".

- [x] **F.0.6** Quality check `gold_confidence_calibrated_by_horizon`:
      filtrar `is_quantile_genuine=False`.
      **Aceite:** point baselines (NaN confidence por design) nao mais
      contadas como `bad_confidence`. Regression guard via teste: NaN em
      quantile-genuine ainda falha (defeito real preservado).

- [x] **F.0.7** Quality check `official_contract_quantile_attention`:
      filtrar baselines do artifact check.
      **Aceite:** runs com `feature_set_name='baseline'` OR `model_version`
      startswith `'baseline_'` excluidos do `fact_model_artifacts` check.
      Quantile contract continua aplicando a TODOS os runs.

- [x] **F.0.8** Amendment ao closure doc.
      **Aceite:** seção "Findings post-closure" em
      [`A_audit_closure_2026-05-17.md`](../../07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md)
      lista os 5 gaps; cross-link bidirecional com Stage F.0.

- [x] **F.0.9** Adicionar Stage F.0 ao checklist post-closure.
      **Aceite:** este Stage existe em
      [`POST_CLOSURE_FIXES_CHECKLIST.md`](POST_CLOSURE_FIXES_CHECKLIST.md)
      com Objetivo, Notas, Cross-link e Tasks. Tasks `[~]` durante
      implementacao; `[x]` apos merge do PR de Stage F.0.

---

## Stage F.A — Debito arquitetural revelado durante F.0 (registro)

**Objetivo:** documentar que a execucao do Stage F.0 (5 fixes + 1 re-fix
em `ab79996`) expos pattern de god-object em 3 modulos. Pattern "ajusta
aqui, esquece la" tornou-se observavel apenas quando todas as 14 Stages
+ F.0 rodaram juntas no E2E (smoke F.1 + re-runs). Tres extracoes
propostas — todas com porte de Stage 15+ — foram **promovidas para
PHASE_B §Stages 20-22** per §Convencao "Promocao para PHASE_B" deste
checklist.

**Cross-link:**
[`B_architectural_debt_2026-05-19.md`](../../07_reports/phase-gates/phase-b/B_architectural_debt_2026-05-19.md) (diagnostico canonico, M5-refresh + M-train_tft + M-validate_quality YELLOW);
[`ADR-0003-multi-horizon-prediction-persister.md`](../../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md);
[`ADR-0004-quality-check-registry.md`](../../01_architecture/decisions/ADR-0004-quality-check-registry.md);
[`ADR-0005-gold-builders-modularization.md`](../../01_architecture/decisions/ADR-0005-gold-builders-modularization.md);
[`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md) §Stages 20, 21, 22.

### Notas de revisao:

- 2026-05-19: Stage criado para registro da descoberta. Execucao em
  PHASE_B; sem tasks executivas aqui (apenas marcacao do origem da
  descoberta como smoke iteration).

### Tasks

Sem tasks executivas. Stage F.A e **registro de promocao**, nao
implementacao. As 3 extracoes sao implementadas em:

- PHASE_B §Stage 20 — `MultiHorizonPredictionPersister` (ADR-0003).
- PHASE_B §Stage 21 — `QualityCheckRegistry` (ADR-0004).
- PHASE_B §Stage 22 — `GoldBuilders` modulares (ADR-0005).

Status deste Stage F.A: `[x]` permanentemente apos registro mergeado.
Status das extracoes: rastreado em PHASE_B.

---

## Ondas futuras (placeholder)

Quando uma re-rodada subsequente do smoke F.1 descobrir novo gap,
abrir um Stage novo aqui (nao espremer mais sub-IDs em F.0.X).
Candidato corrente:

- **Gap 6 — TFT y_true convention bug.** **Gap 6 RESOLVIDO** (Opção d aplicada em Stage R-20 mergeado 2026-05-21;
  target_return backward-shift + baseline y_true_idx ajustado; off-by-one
  corrigido empiricamente; revalidado em Stage R-23 smoke v4).
  Cross-link: [`tft_y_true_investigation_2026-05-18.md`](../../07_reports/phase-gates/phase-a/tft_y_true_investigation_2026-05-18.md);
  [ADR-0003](../../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md);
  [`smoke_confirmatory_2026-05-18.md`](../../07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md) §"Post-Stage 20".
