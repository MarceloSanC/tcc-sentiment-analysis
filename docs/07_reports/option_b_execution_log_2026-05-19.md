---
title: Option B Execution Log (Stages 20-23 autonomous)
scope: Append-only log de decisoes, erros, completions e aborts durante execucao autonoma de Stages 20-23. Cada entry com timestamp UTC, contexto, e citacao do principio aplicado. Marcelo audita no retorno.
update_when:
  - decisao nao-obvia tomada
  - erro encontrado e resolvido
  - task ou Stage completado
  - abort condition gatilhada
canonical_for: [option_b_execution_log, autonomous_decisions_record]
---

# Option B Execution Log (Stages 20-23)

Append-only. Mais recente no topo.

---

## [2026-05-20 01:36 UTC] Stage 20.5 — completion

**Context:** Atualizar docs/03_modeling/MULTI_HORIZON.md com convencao
canonica Opcao (a).
**Changes:**
  - Nova §"Convencao canonica de target_timestamp e y_true" apos
    §"Conceitos-chave"
  - Conceito target_timestamp reescrito (de "Chave de alinhamento" para
    referencia explicita ao Persister)
  - Persistencia: tabela adicionou decision_idx + timestamp_utc rows
  - Fonte da verdade: aponta para Persister como ponto unico de verdade
**Cross-link:** ADR-0003, Persister source, schema, integration test.
**Outcome:** Doc canonico atualizado.
**Next:** Stage 20.6 — smoke regression.

---

## [2026-05-20 01:34 UTC] Stage 20.4 — completion

**Context:** Cross-pipeline regression guard fechando Gap 6 mecanicamente.
**Outcome:**
  - tests/integration/test_tft_baselines_y_true_alignment.py criado
  - 3 test cases: parametrico (5 pares decision_idx,h), trading-day arithmetic,
    boundary symmetry
  - 7 tests passing
**Principle:** §"Aceite" Stage 20.4 do plano + §4.6 (mechanical > procedural:
  garantia via shared Persister, nao via convencao).
**Outcome:** pytest tests/integration/test_tft_baselines_y_true_alignment.py
  → 7 passed.
**Next:** Stage 20.5 — atualizar MULTI_HORIZON.md.

---

## [2026-05-20 01:33 UTC] Stage 20.3 — completion

**Context:** Migrar `_emit_oos_rows` em run_baselines_use_case para usar
MultiHorizonPredictionPersister. Tighten FACT_OOS_PREDICTIONS_SCHEMA
para decision_idx em required_columns (apos ambos writers emitirem).
**Changes:**
  - `_emit_oos_rows` constroi RunContext per-split e
    `dataset_timestamps = [pd.Timestamp(t) for t in timestamps]` (full df).
  - decision_idx = i (indice de chronological df) — baselines nao tem
    encoder offset.
  - Removida calendar arithmetic (Timedelta(days=h-1)) + check defensiva
    `target_ts <= decision_ts` (Persister enforce target_pos > decision_idx).
  - Removido comentario 244-248 enganoso (apontava para train_tft linha
    790, semantica antiga).
  - FACT_OOS_PREDICTIONS_SCHEMA.required_columns += decision_idx.
**Tests adaptados (behavior change esperada, Gap 6 fix):**
  - test_historical_mean_skips_rows_without_warmup_window: esperado 19
    (era 20). Razao: ultima decision (i=49) nao tem target h=1 valido
    (target_idx=50 out of bounds). Comportamento mais correto.
  - test_baseline_y_true_skips_rows_beyond_horizon_at_end_of_dataset:
    last decision_ts (i=19, n=20) emite zero rows (era {1}). Penultimate
    (i=18) emite h=1 mas nao h=2. Reflete Opcao (a) precisa.
**Principle:** §4.6 (refactor + fix intencional). Behavior change e o
  Gap 6 fix per ADR-0003.
**Outcome:** pytest tests/ → 543 passed. Stage 20 core migration completa
  (both writers usam Persister, schema decision_idx required).
**Next:** Stage 20.4 — cross-pipeline regression test.

---

## [2026-05-20 01:29 UTC] Stage 20.2 — completion

**Context:** Migrar `_persist_fact_oos_predictions` em train_tft_model_use_case
para usar MultiHorizonPredictionPersister em ambos call-sites (multi-horizon
matrix path + legacy 1-horizon path).
**Changes:**
  - Adicionado parametro `max_encoder_length: int` em
    `_persist_fact_oos_predictions`.
  - Call-site em execute() (linha ~1393) passa
    `metadata_config.get("max_encoder_length", TFT_TRAINING_DEFAULTS[...])`.
  - decision_start_offset = max(max_encoder_length - 1, 0). Sample i mapeia
    para decision_idx = decision_start_offset + i (no split_df coords).
  - Ambos paths usam Persister.build_record() + record.to_dict().
  - IncompletePredictionWindowError → continue (AGENT_CORE skip).
  - Removida calendar arithmetic (Timedelta(days=h-1)). Removido tail(n)
    workaround.
**Tests adaptados:**
  - test_persist_fact_oos_predictions_keeps_horizon_index_alignment_per_split:
    timestamps estendidos para 40 dias (acomoda h=30); max_encoder_length=1;
    assertions atualizadas — val_h7 target = val_ts[7] (2025-01-17, era
    2025-01-16 calendar); test_h30 target = test_ts[30] (2025-03-12, era
    2025-03-11 calendar). Decision_idx=0 verificado.
  - test_persist_fact_oos_predictions_applies_quantile_guardrail_columns:
    timestamps estendidos para 5 dias; max_encoder_length=1.
  - test_persists_dim_run_identity_fields_when_analytics_repo_is_enabled:
    df estendido para 7 timestamps; split_config nao-overlapping (val
    20240601-20240602, test 20250102-20250103); training_config com
    max_encoder_length=0.
**Principle:** §4.3 (refactor preservando contrato: Persister centraliza
  a convencao Opcao (a)) + §4.6 (Stage 20.2 e refactor MAS aplica fix Gap 6
  intencional — timestamps em trading-days, year=decision_day).
**Outcome:** .venv/bin/pytest tests/ → 543 passed. Commit em sequencia.
**Next:** Stage 20.3 — migrar run_baselines_use_case + tighten schema required.

---

## [2026-05-20 01:21 UTC] Stage 20.1 — decision (deviation from plan)

**Context:** Stage 20.1 — adicionar decision_idx em FACT_OOS_PREDICTIONS_SCHEMA.
Plano §"Aceite" 20.1: "em columns + required_columns".
**Question/Issue:** Adicionar decision_idx a required_columns em 20.1 quebra
17 testes de baselines + repo (validate_table_payload rejeita rows sem decision_idx).
Baselines so emite decision_idx apos 20.3; train_tft so apos 20.2.
**Options:**
  (a) Tighten required em 20.1, aceitar testes red entre 20.1 e 20.3
  (b) decision_idx em columns only em 20.1; tighten para required em 20.3
**Choice:** (b) — adicionar em columns mas NAO em required_columns em 20.1.
**Principle:** §1 (Principio operacional, "Valide tests apos cada task") +
§4.5 (Liberdade quando ganho concreto): tests green em cada commit boundary.
Same end-state apos 20.3 (mecanico, schema-level). Plan literal era ideal
sem considerar acoplamento de migracao em multiplos commits.
**Outcome:** schema atualizado; comentario explicito no codigo explica
tightening em 20.3. Tests: 543 passed (525 anteriores + 18 novos).
**Next:** 20.1 commit, depois 20.2.

---

## [2026-05-20 01:21 UTC] Stage 20.1 — completion

**Context:** Stage 20.1 — Persister + schema + tests.
**Outcome:**
  - src/domain/services/multi_horizon_prediction_persister.py criado (133 LOC)
    com PredictionRecord, RunContext, IncompletePredictionWindowError,
    MultiHorizonPredictionPersister.build_record() (keyword-only).
  - FACT_OOS_PREDICTIONS_SCHEMA + FACT_INFERENCE_PREDICTIONS_SCHEMA com
    decision_idx: int64 em columns (consistencia com C1 da auditoria).
  - tests/unit/domain/services/test_multi_horizon_prediction_persister.py
    com 15 testes (h=1/h=7, boundary, calendar gaps, run_context propagation,
    immutability, keyword-only).
  - Schema tests atualizados (decision_idx assertions).
  - Fixtures _oos_row + test_append_fact_oos_predictions atualizados.
**Principle:** §4.1 "Match precedente do projeto" — QuantileGuardrailService
  pattern (dataclass(frozen=True) + @staticmethod).
**Outcome:** .venv/bin/pytest tests/ → 543 passed, 0 failed.
**Next:** Stage 20.2 — migrar train_tft_model_use_case.

---

## [2026-05-20 01:12 UTC] Stage 20.0 — completion

**Context:** Inicio de Stage 20. Verificacao de pre-condicoes conforme §2 do plano.
**Outcome:**
  - main local == origin/main (576f4fd Merge PR #43)
  - working dir clean
  - .venv/bin/pytest tests/ -q → 525 passed (45 warnings, conhecidos)
  - ADR-0003/0004/0005 + B_architectural_debt + tft_y_true_investigation presentes
  - ADR-0003 §Decision confirma Opcao (a): y_true = dataset.target_return[decision_idx + h - 1]
  - Tag checkpoint-pre-option-b existe (verificada)
  - data/analytics_archive_pre_phase_b/ = 851M (baseline pre-execucao)
  - Branch criado: feat/stage-20-multi-horizon-prediction-persister
**Principle:** §2 do plano (pre-condicoes).
**Next:** Stage 20.1 — criar MultiHorizonPredictionPersister + schema + tests.

---
