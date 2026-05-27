---
title: Phase B Architectural Debt Assessment (post-F.0)
scope: Diagnostico de debito arquitetural revelado durante execucao do Stage F.0
  (POST_CLOSURE_FIXES_CHECKLIST). Identifica 3 god objects que produziram
  pattern "ajusta aqui, esquece la" durante a iteracao do smoke F.1 e propoe
  3 Stages de refator de habilitacao (PHASE_B §Stages 20-22) antes do fix do
  Gap 6 (TFT y_true convention). Paralelo a A_code_audit.md mas escopo diferente:
  A_code_audit cobriu acoes do codigo herdado pre-Phase B; este cobre debito
  produzido pela propria execucao de Stages 1-14.
update_when:
  - veredicto de algum modulo P1 (M5-refresh, M-train_tft, M-validate_quality) mudar
  - novo god object for identificado durante PHASE_B / POST_CLOSURE iterations
  - extracao mergeada → atualizar status do modulo correspondente
canonical_for: [phase_b_architectural_debt_audit, god_object_findings, post_f0_refactor_rationale]
---

# Phase B — Architectural Debt Assessment (post-F.0)

**Status:** remediado/superseded em 2026-05-23. Os 3 modulos YELLOW foram
enderecados por PHASE_B §Stages 20-22; o fix do Gap 6 foi fechado na
remediacao R-20/R-23. Este arquivo permanece como diagnostico historico da
divida arquitetural post-F.0, nao como bloqueio ativo da Phase B.
**Criado:** 2026-05-19
**Pre-requisito:** [`A_audit_closure_2026-05-17.md`](../phase-a/A_audit_closure_2026-05-17.md) §"Findings post-closure".
**Bloqueante para:** fix do Gap 6 (TFT y_true convention) — ver
[`POST_CLOSURE_FIXES_CHECKLIST.md`](../../../05_checklists/phase-a/POST_CLOSURE_FIXES_CHECKLIST.md) §"Ondas futuras".

Este documento e paralelo a [`A_code_audit.md`](../phase-a/A_code_audit.md) com escopo
diferente:

- `A_code_audit.md` (Phase A) cobriu **codigo herdado pre-Phase B** com
  veredictos M1-M7. Fechado parcialmente em 2026-05-17.
- Este doc cobre **debito arquitetural produzido pela propria execucao
  de Stages 1-14 + F.0**. Nao reabre Phase A; registra honestamente que
  a serie de fixes corretos individualmente acumulou peso em 3 modulos
  que agora geram pattern "ajusta aqui, esquece la".

A descoberta foi observavel apenas quando todas as 14 Stages rodaram
juntas no E2E (smoke F.1), porque cada Stage isoladamente parecia
cirurgico — o problema e a **sequencia**, nao o conteudo de qualquer
Stage individual.

---

## Como usar

Para cada modulo identificado, preencher:
- **Veredicto:** GREEN / YELLOW / RED
- **Achados:** evidencia objetiva (LOC, metodos, heatmap de commits, casos clinicos)
- **Acao:** Stage proposto em [`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../../../05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md)

Veredictos:
- `GREEN` — sem debito arquitetural relevante.
- `YELLOW` — god object identificado; nao bloqueia experimento mas degrada
  sustentabilidade e amplia blast-radius de fixes futuros.
- `RED` — debito que ja produz regressao funcional ou bloqueia fix de bug
  conhecido sem refator previo.

Atualmente todos os 3 modulos identificados sao **YELLOW**: nao bloqueiam
F.1 conceitualmente, mas tornam o fix do Gap 6 (proximo Stage de codigo)
estruturalmente arriscado se nao precedido por refator.

---

## Resumo

| Modulo | Veredicto | LOC | Metodos / Builders / Checks | Stage de fix |
|---|---|---|---|---|
| M5-refresh — `refresh_analytics_store_use_case.py` | YELLOW | 2.568 | 47 metodos / **26 builders** | PHASE_B §Stage 22 (ADR-0005) |
| M-train_tft persistencia — `train_tft_model_use_case.py` | YELLOW | 1.482 | 30 metodos / persistencia embutida | PHASE_B §Stage 20 (ADR-0003) |
| M-validate_quality — `validate_analytics_quality_use_case.py` | YELLOW | 1.152 | 11 metodos / **~20 checks** num `execute()` | PHASE_B §Stage 21 (ADR-0004) |

Top 3 use_cases somam **5.202 LOC = 39% do total** de `src/use_cases/`
(13.379 LOC). `src/domain/` (camada esperada para regras) tem apenas
**2.238 LOC** — domain services existem (`quantile_contract_analyzer`,
`scope_spec`, etc.) mas sao importados em **apenas 10 lugares** dos
use_cases; a maioria da logica de negocio vive enterrada em metodos
privados dos god objects.

**Nenhum P0 bloqueante.** Os 3 modulos sao YELLOW porque:
- nao bloqueiam o experimento conceitualmente (smoke F.1 v2 ja
  produziu metricas validas em 22-25 de 27 checks);
- bloqueiam **fix limpo do Gap 6** porque a convencao `target_timestamp`
  esta distribuida em 5 arquivos sem dono unico (M-train_tft +
  run_baselines + validate_quality + refresh + build_tft_dataset).

---

## Evidencia transversal

### E1 — Heatmap de modificacoes (ultimos 60 commits)

| Arquivo | Modificacoes |
|---|---|
| `validate_analytics_quality_use_case.py` | 5 |
| `run_baselines_use_case.py` | 4 |
| `refresh_analytics_store_use_case.py` | 3 |
| `tests/unit/use_cases/test_validate_analytics_quality_use_case.py` | 4 |
| `tests/unit/use_cases/test_refresh_analytics_store_use_case.py` | 3 |

3 arquivos de producao concentram **12 modificacoes = 20% dos commits**.

### E2 — Caso clinico do "ajusta aqui, esquece la"

- Commit `584373e` (2026-05-18 20:25): "feat(analytics-quality): gate
  alinhamento target_timestamp TFT vs baselines (F.0.3)"
- Commit `ab79996` (2026-05-18 20:42): "fix(analytics-quality): F.0.3
  enrich oos_view com parent_sweep_id via dim_run join"

Mensagem do segundo commit: *"Versao inicial do gate F.0.3 nao fazia o
join e silenciosamente skipava o check com motivo
'missing_group_columns'."*

Causa-raiz: autor do check novo nao tinha visibilidade do schema
implicito navegado pelos outros 19 checks na mesma classe.

### E3 — Coautoria forcada via re-aberturas cross-Stage

Mesmos 5 builders de `refresh_analytics_store` reabertos por:
- Stage 5 (cohort-aware groupby)
- Stage 6 (parent_sweep_id no output) — uma semana depois
- Stage 9 (filtros probabilisticos)
- Stage 11 (gate degeneracao)

E os Gaps 4 e 5 do smoke F.1 (cross-Stage 9↔12 e 10↔12) sao instancias do
mesmo fenomeno: cada Stage individual e cirurgico mas a sequencia gera
regressao porque nenhum dos autores tinha visibilidade dos 19/26 vizinhos.

### E4 — Cobertura de testes

| Modulo | Producao LOC | Testes LOC | Ratio |
|---|---|---|---|
| `refresh_analytics_store_use_case.py` | 2.568 | 2.487 | 0.97 |
| `validate_analytics_quality_use_case.py` | 1.152 | 1.863 | 1.62 |
| `train_tft_model_use_case.py` | 1.482 | 1.172 | 0.79 |

Total: src 28.982 LOC vs tests 18.269 LOC (63% ratio). Refator e
auditavel — alta cobertura de teste mitiga risco de regressao na
migracao.

---

## Mapeamento detalhado

### M5-refresh — `refresh_analytics_store_use_case.py` (YELLOW)

**Achados:**

- 2.568 LOC, 47 metodos, **26 builders gold** (`_build_gold_*`) inline.
- Mistura I/O, scope resolution, normalizacao de `parent_sweep_id`,
  builders, pinball_loss, IQR, prob_up — 5 categorias de responsabilidade
  numa unica classe `RefreshAnalyticsStoreUseCase`.
- Builders por categoria semantica:
  - Ranking (3): `runs_long`, `ranking_by_config`, `consistency_topk`.
  - Pairwise (3): `dm_pairwise_results`, `mcs_results`,
    `paired_oos_intersection_by_horizon`.
  - Quantile (4): `quantile_guardrail_audit`, `quantile_degeneracy_report`,
    `prediction_metrics_by_run_split_horizon` (+ single_contract),
    `prediction_metrics_by_config`.
  - Descriptive (3): `ic95`, `feature_set_impact`, `model_decision_final`.
  - Confidence (2): `confidence_calibrated_by_horizon`,
    `metrics_by_config_n_oos_contract`.
  - Outros (~11): variantes e sub-builders.
- Stages 5, 6, 9, 11 reabriram subsets sobrepostos sem refator estrutural.

**Acao:** PHASE_B §Stage 22 (ADR-0005 — GoldBuilders modulares por
categoria). Extracao para `src/domain/services/gold_builders/` com 5-6
modulos. `refresh_analytics_store` reduz a orquestrador (~400 LOC).

### M-train_tft persistencia — `train_tft_model_use_case.py` (YELLOW)

**Achados:**

- 1.482 LOC, 30 metodos. Responsabilidades empilhadas:
  feature selection, split temporal, warmup policy, pretrain quality
  gate, ablacao, git commit collect, library versions, **persistencia
  multi-horizonte de `fact_oos_predictions`** (linhas 770-810).
- Linha 778: `split_tail = split_df.tail(n)` — recupera **decoder_end_day**
  como `timestamp_utc`. Bug raiz do Gap 6 (descrito em
  [`tft_y_true_investigation_2026-05-18.md`](../phase-a/tft_y_true_investigation_2026-05-18.md)).
- Convencao `target_timestamp` distribuida em 5 arquivos sem dono unico:
  - [`train_tft_model_use_case.py:778`](../../../src/use_cases/train_tft_model_use_case.py#L778) (TFT persistence)
  - [`run_baselines_use_case.py:240`](../../../src/use_cases/run_baselines_use_case.py#L240) (baseline persistence)
  - [`validate_analytics_quality_use_case.py`](../../../src/use_cases/validate_analytics_quality_use_case.py) (F.0.3 alignment gate)
  - [`refresh_analytics_store_use_case.py`](../../../src/use_cases/refresh_analytics_store_use_case.py) (builders pareados)
  - [`build_tft_dataset_use_case.py:563`](../../../src/use_cases/build_tft_dataset_use_case.py#L563) (target_return convention)
- [`docs/03_modeling/MULTI_HORIZON.md:28-29`](../../../03_modeling/MULTI_HORIZON.md#L28-L29)
  define `target_timestamp` sem fixar anchor. Ambiguidade permitiu
  divergencia TFT↔baseline a coexistir ate o smoke F.1 expor.

**Acao:** PHASE_B §Stage 20 (ADR-0003 — `MultiHorizonPredictionPersister`).
Extracao para `src/domain/services/multi_horizon_prediction_persister.py`
materializando a convencao canonica como **um objeto do dominio**, nao
linha de codigo enterrada. Pre-requisito do fix do Gap 6.

### M-validate_quality — `validate_analytics_quality_use_case.py` (YELLOW)

**Achados:**

- 1.152 LOC, 11 metodos. `execute()` chama `self._record(checks, ...)`
  em **~20 lugares** distintos (cada um um check).
- Stage F.0 adicionou 3 modificacoes em sequencia (F.0.3 gate novo,
  F.0.6 filtro, F.0.7 filtro). Caso clinico E2 (`584373e` + `ab79996`)
  evidencia que checks novos colidem com schema implicito dos vizinhos.
- Lista de checks heterogeneos numa unica funcao gigante torna review
  cross-Stage praticamente impossivel sem reler ~1.000 LOC.

**Acao:** PHASE_B §Stage 21 (ADR-0004 — `QualityCheckRegistry`).
Extracao para `src/domain/services/quality_checks/` com 4-5 modulos por
categoria semantica (cardinality, alignment, calibration, contracts).
Cada check vira classe pequena com `applies_when(scope)` e
`run(data) -> CheckResult`.

---

## Findings post-closure (descobertos via Stage F.0 2026-05-18→19)

Esses 3 modulos YELLOW **nao invalidam** o closure mapping dos Stages
1-14 em si — todos os P0 transversais do A_code_audit permanecem
fechados. Sao debito **estrutural produzido pela serie de fixes
corretos**, observavel apenas no E2E.

Plano de remediacao historico:

1. **PHASE_B §Stage 20** (ADR-0003) — `MultiHorizonPredictionPersister`.
2. **PHASE_B §Stage 21** (ADR-0004) — `QualityCheckRegistry`.
3. **PHASE_B §Stage 22** (ADR-0005) — `GoldBuilders` modulares.
4. **Apos Stage 20**: fix do Gap 6 (contrato canonico consolidado na
   Opcao (d), registrado em ADR-0003 e validado em R-20/R-23); blast-radius
   reduzido por `MultiHorizonPredictionPersister`.
5. **Apos Stage 22**: re-rodada F.1 v4 PASS pleno (Stage R-23), com smoke
   regression byte-identical garantido pelos Aceites de cada Stage.

Promocao registrada em
[`POST_CLOSURE_FIXES_CHECKLIST.md`](../../../05_checklists/phase-a/POST_CLOSURE_FIXES_CHECKLIST.md)
§"Stage F.A" (origem: F.0 iteration); execucao em
[`PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../../../05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md)
§Stages 20-22 (porte: pre-experiment engineering).
