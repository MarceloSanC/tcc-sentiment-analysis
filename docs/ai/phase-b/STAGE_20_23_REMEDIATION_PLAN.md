---
title: Stage 20-23 Remediation Plan (audit REQUEST_CHANGES — 2026-05-20, rev 2)
scope: Plano de correcao dos 7 RED levantados pela auditoria independente das PRs #44-#47, ajustado apos meta-auditoria empirica. Organizado por Stage; cada Stage termina com checkpoint humano antes de fechar a PR. NAO e plano autonomo — execucao em sessao supervisionada, um Stage por vez. Cobre hotfix de CI ruff em main, extracao do fix de bug pre-existente para PR isolada, e a decisao arquitetural sobre dependency ordering de gold builders (omitida no plano original §22).
update_when:
  - Stage de remediacao completado (PR mergeada ou explicitamente reclassificada)
  - decisao tomada em ponto bloqueante (RED-1 option a/b/c/d; arquitetura R-22.2.bis)
  - novo RED descoberto durante remediacao
  - Marcelo retorna e altera prioridades
canonical_for: [stage_20_23_remediation_plan, audit_response_2026_05_20]
---

# Stage 20-23 — Plano de Remediacao (audit REQUEST_CHANGES)

**Status:** auditoria independente concluida em 2026-05-20 ~18:15 UTC.
**Veredicto:** REQUEST_CHANGES. 7 RED bloqueiam merge das PRs #44-#47.
**Executor:** Claude (sessao supervisionada) + Marcelo (decisoes-chave).
**Modo:** **NAO autonomo.** Um Stage por vez; checkpoint humano antes de
proximo Stage; cada PR so fecha apos validacao completa.

---

## 0. Sumario executivo

| RED | Descricao | Stage afetado | PR | Bloqueio para merge |
|---|---|---|---|---|
| RED-1 | Gap 6 nao fechado: TFT y_true(target_ts=X) == baseline y_true(target_ts=X+1) em 663/685 linhas test/h=1 | 20 | #44 | sim |
| RED-2 | Teste cross-pipeline 20.4 e hollow (passa mesmo input pelo Persister duas vezes) | 20 | #44 | sim |
| RED-3 | Stage 21 skeleton: QualityCheckRegistry zero imports em producao | 21 | #45 | sim |
| RED-4 | Skeleton-only nao autorizado pelo plano §21.2-21.8 | 21 | #45 | sim |
| RED-5 | Stage 22 skeleton: god-object intacto (2.579 LOC vs meta 400) | 22 | #46 | sim |
| RED-6 | Stage 23 fix em arquivo alvo de Stage 22 (escopo creep + divida latente) | 22/23 | #47 (e novo #48) | yes (extracao) |
| RED-7 | F.1 marcado [x] com max_epochs=1; aceite literal exige >= 5 | 23 | #47 | sim |
| CI | 3 ruff errors pre-existentes em main bloqueiam lint-type-unit em todas as PRs | todas | todas | sim (critério F.1 #5) |

Tempo estimado para fechar todos: **45-65h Claude ativo** distribuidas por sessoes.
Estimativa original (28-46h) foi otimista; revisao incorpora dependency
ordering em gold builders (omitido no plano original), test coverage expandido
(scope_mode matrix, builder deps, propagation) e adapter integration tests.

---

## 1. Principio operacional (diferente do plano original)

**SUPERVISIONADO, NAO AUTONOMO.** Cada Stage de remediacao tem 3 fases:

1. **Investigacao + decisao** (sessao com Marcelo presente)
2. **Implementacao + validacao** (Claude pode executar sozinho)
3. **Checkpoint pre-merge** (revisao por outra sessao Claude OU Marcelo)

**Authoring de review prompts:** quando chegar ao fim da fase 2 de cada
Stage, invoca skill [`staged-implementation-protocol`](../../../.claude/skills/staged-implementation-protocol/SKILL.md)
§"Phase D step 12" para gerar o review prompt da PR alvo. Skill encoda
as 5 audit dimensions mandatorias + active gap detection.

Nao pre-authorar todos os 6 review prompts agora — diff entre Stages
pode mudar gaps relevantes; authoring just-in-time e mais barato e mais
preciso.

**Sem stacking novo.** As PRs ja stacked (#44 → #45 → #46 → #47) ficam stacked
ate Marcelo decidir mergear. Fixes adicionados via novos commits nas branches
existentes; conflitos resolvidos via merge (NUNCA rebase/amend/force-push).

**Um Stage por vez.** Nao iniciar Stage N+1 ate Stage N estar:
- aprovado em revisao independente, OU
- explicitamente reclassificado por Marcelo (e.g., "Stage 21 fica skeleton, abre Stage 21.bis depois").

**Cada PR so fecha quando:**
- todos os RED daquela PR resolvidos OU YELLOW-aceitos
- `pytest tests/` verde
- `ruff check` sem erros novos (pre-existentes admissiveis se nao bloqueiam)
- smoke de validacao especifico do Stage executado
- log de execucao atualizado

---

## 2. Pre-condicoes

Antes de iniciar qualquer Stage de remediacao:

```bash
# Estado atual
git status                                  # working dir clean
git branch --show-current                   # confirmar branch alvo
git fetch origin
git log --oneline -5                        # sincronizado

# Testes baseline
.venv/bin/pytest tests/ -q | tail -3        # esperado: 559 passed

# Archive integrity
du -sh data/analytics_archive_pre_phase_b/  # esperado: 851M

# CI status
gh pr checks 44 2>&1 | tail -3              # lint-type-unit FAILURE (3 ruff)
gh pr checks 45 2>&1 | tail -3
gh pr checks 46 2>&1 | tail -3
gh pr checks 47 2>&1 | tail -3
```

Se qualquer item divergir, parar e investigar.

---

## 3. Ordem de execucao recomendada

Sequencial. Nao iniciar um Stage ate o anterior estar fechado/aprovado.

```
Pre-Stage CI  (CI ruff fix em main + merge mechanics)  — ~1 h
  ↓
Stage R-20   (PR #44 closure: Opcao d + adapter test)  — 6-10 h
  ↓
Stage R-E    (PR #48 extract suffix fix + regression)  — 1-2 h
  ↓
Stage R-21   (PR #45 migration + 2-mode regression)    — 10-16 h
  ↓
Stage R-22   (PR #46 migration + dependency arch)      — 16-24 h
  ↓
Stage R-23   (PR #47 closure: max_epochs=5 + golden)   — 3-5 h
```

Letra "R-" para distinguir de Stages originais. Nomenclatura interna; nao
muda Stage numbers em commits/PRs.

---

# Pre-Stage CI — Hotfix lint-type-unit em main

**Classification:** **Operational** (3 fixes mecanicos de ruff).
**Implementer target:** **Codex** (find/replace ruff `--fix --unsafe-fixes` +
remover 1 variavel unused; nada decisional).

### Protected files (NAO TOCAR)
- Tudo exceto:
  - `src/use_cases/validate_analytics_quality_use_case.py` (2 erros UP037)
  - `tests/unit/use_cases/test_run_baselines_test_pipeline_use_case.py` (1 erro F841)

**Objetivo:** main com CI verde para que cada PR subsequente possa ser
mergeada sem o critério #5 do F.1 ("CI verde em PRs mergeados") falhar.

**Erros pre-existentes em main:**
- `src/use_cases/validate_analytics_quality_use_case.py:94` — UP037 (quoted `"pd.DataFrame"`)
- `src/use_cases/validate_analytics_quality_use_case.py:95` — UP037 (idem)
- `tests/unit/use_cases/test_run_baselines_test_pipeline_use_case.py:219` — F841 (`config_silver` unused)

**Procedimento:**

1. Branch nova direto de main:
   ```bash
   git checkout main && git pull
   git checkout -b fix/lint-pre-existing-ruff-errors
   ```

2. Aplicar `ruff check --fix --unsafe-fixes src/ tests/` para UP037 (2 ocorrencias).

3. Remover linha 219 (`config_silver = _load_dim(silver)`) em
   `test_run_baselines_test_pipeline_use_case.py` (variavel realmente nunca
   usada; o teste exercita o write via `pipeline.execute`).

4. Validar:
   ```bash
   .venv/bin/ruff check src/ tests/        # Found 0 errors
   .venv/bin/pytest tests/ -q              # 559 passed
   ```

5. Commit + push + PR contra main:
   ```bash
   git commit -m "fix(lint): remove pre-existing ruff errors (UP037, F841)"
   gh pr create --base main --title "fix(lint): clean pre-existing ruff errors blocking CI"
   ```

**Aceite:** PR mergeada em main; CI verde em main.

## Pre-Stage CI.bis — Reconciliacao das branches stacked

Apos Pre-Stage CI mergeada em main, as 4 PRs `feat/stage-2X-*` ainda
apontam para o main antigo (sem o ruff fix). Cada branch precisa
incorporar main via merge (NAO rebase) para que o CI delas reflita
a base limpa.

```bash
git checkout main && git pull
for b in \
    feat/stage-20-multi-horizon-prediction-persister \
    feat/stage-21-quality-check-registry \
    feat/stage-22-gold-builders-modular \
    feat/stage-23-f1-pass-pleno; do
  git checkout "$b"
  git merge main --no-edit                  # incorpora ruff fix; SEM rebase
  .venv/bin/pytest tests/ -q | tail -3      # confirmar 559+ passed
  .venv/bin/ruff check src/ tests/ | tail -3  # 0 errors
  git push                                  # CI roda; deve ficar verde
done
```

Resultado esperado: 4 PRs com `lint-type-unit` verde apos o push.

---

# Stage R-20 — Correcao PR #44 (RED-1, RED-2)

**Branch:** `feat/stage-20-multi-horizon-prediction-persister` (existente).
**PR:** #44.
**RED a resolver:** RED-1 (Gap 6 substantivo), RED-2 (hollow test).
**Tempo estimado:** 6-10h.
**Classification:** depende de Opcao escolhida em R-20.0 — Opcao (a) =
**policy-introducing** (requer R-20.pre doc PR); Opcoes (b), (c), (d) =
**operational** (code PR direto com ADR amendado no mesmo PR).
**Implementer target:** **Claude** (nova sessao). Decisao Opcao a/b/c/d
exige raciocinio sobre trade-offs semantica vs custo; nao mecanico.

### Protected files (NAO TOCAR)
- `src/use_cases/validate_analytics_quality_use_case.py` (territorio R-21)
- `src/use_cases/refresh_analytics_store_use_case.py` (territorio R-22; ex
  R-E pega o suffix fix)
- `src/domain/services/quality_checks/**` (territorio R-21)
- `src/domain/services/gold_builders/**` (territorio R-22)
- `docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (territorio R-23)
- `data/analytics_archive_pre_phase_b/**` (intocavel sempre)

### Required reading (contexto antes de R-20.0)
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §3.5 — construcao de `target_return` em `build_tft_dataset_use_case.py:563`
  (Inputs/Outputs/Calculos/Quality gates). Le isto antes de avaliar
  Opcoes a/b/c/d.
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §4.6 — contrato de `fact_oos_predictions`, convencao `decision_idx`,
  comportamento de `IncompletePredictionWindowError`.
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §4.12 — full enumeration dos 3 baselines (specs, offsets, run_id).
  Util ao ajustar `run_baselines_use_case.py:259`.

## R-20.0 — Investigacao + decisao (BLOQUEADOR; precisa Marcelo)

**Contexto auditado:** off-by-one entre TFT e baseline em y_true e
estrutural. Confirmado **empiricamente** em parquets de smoke (662/684
linhas test/h=1 mostram `TFT[i].y_true == baseline[i+1].y_true`; 0/685
casam direto). pytorch_forecasting popula decoder com
`target_return[decision_idx + h]`, mas ADR-0003 Opcao (a) prescreve
`target_return[decision_idx + h - 1]` (h=1 = return D → D+1).

Localizacao precisa do bug:
[`src/adapters/pytorch_forecasting_tft_trainer.py:480`](../../src/adapters/pytorch_forecasting_tft_trainer.py#L480)
`actuals = torch.cat([y[0] for _, y in iter(dataloader)], dim=0)`. `y[0]`
e o decoder target do TimeSeriesDataSet ([linha 297](../../src/adapters/pytorch_forecasting_tft_trainer.py#L297)).
Para sample com encoder_end=D, decoder[h-1] = target_return at split_df
position `D + h`.

Construcao da coluna target em
[`src/use_cases/build_tft_dataset_use_case.py:563`](../../src/use_cases/build_tft_dataset_use_case.py#L563):
`df["target_return"] = np.log(df["close"].shift(-1) / df["close"])`
i.e. `target_return[t] = log(close[t+1]/close[t])` (forward 1-step).

**4 opcoes (decisao final e do Marcelo):**

| Opcao | O que muda | Custo | Preserva semantica financeira? |
|---|---|---|---|
| (a) | ADR-0003 + baselines passam a usar `target_return[i + h]`; "h=1" passa a significar "return D+1→D+2" | Baixo (3-4 arquivos) | Nao — abandona convencao "next-day return after decision" |
| (b) | TFT adapter ignora decoder do pytorch_forecasting; le `target_return[D+h-1]` direto do split_df | Alto (invasivo no adapter; modelo treinado em decoder[d+h] mas avaliado em target[d+h-1] — desalinhamento loss/metrica) | Sim — ADR intacto, mas y_pred e y_true descreven dias diferentes |
| (c) | Documentar como limitacao conhecida; manter codigo, marcar Gap 6 como RESOLVED PARTIAL | Zero | Nao — bug fica visivel em qualquer comparacao pareada |
| **(d)** | **Shift target_return formula em build_tft_dataset_use_case para `log(close[t]/close[t-1])` (backward 1-step); baseline tambem usa `[i + h]`; ADR formula muda mas semantica preserva** | **Medio (1 arquivo dataset + 1 arquivo baseline + ADR + testes + rebuild de `dataset_tft_AAPL.parquet`)** | **Sim — h=1 continua "next-day return after decision" = `log(close[D+1]/close[D])`** |

**Verificacao matematica de Opcao (d):**
- Com `target_return[t] = log(close[t]/close[t-1])` (backward):
- Decoder[h-1] em encoder_end=D fetcha posicao D+h → valor = `log(close[D+h]/close[D+h-1])`
- Para h=1: decoder[0] = `log(close[D+1]/close[D])` = "next-day return after decision" ✓
- Baseline y_true_idx vira `i + h` em vez de `i + h - 1`; valor identico ao decoder

**Custo real de Opcao (d) (corrigido apos esclarecimento):**

NAO requer retreino de modelos. Phase B (treino confirmatorio real) ainda
nao rodou; F.2 pre-registro nem foi feito. Os unicos "modelos" que existem
hoje sao artefatos de smoke (max_epochs=1, hidden_size=16) que sao throwaway.

O que efetivamente custa:
1. Rebuild `data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet` (~30s, mecanico).
2. Smoke regression Stage R-23 com nova formula (ja planejado independente).
3. `data/analytics_archive_pre_phase_b/` permanece como esta — registro
   historico pre-Phase B; sua semantica antiga vira metadado documentado,
   nao debito ativo.

**Recomendacao desta sessao:** Opcao (d). Justificativa: e a unica que
casa codigo + semantica financeira documentada em MULTI_HORIZON.md +
preserva ADR-0003 §Decision intent ("h=1 = next-day return after decision").
Custo medio (rebuild dataset + ajustar baselines) e justificado pela
correcao substantiva do Gap 6.

**Decision point:** Marcelo escolhe (a), (b), (c) ou (d). Sem decisao,
nao prosseguir R-20.1.

**Stage classification por opcao (per skill `staged-implementation-protocol`):**

| Opcao | Classification | Sequencia |
|---|---|---|
| (a) | **Policy-introducing** — muda semantica de h ("next-day after decision" → "return D+1→D+2"); reescreve ADR-0003 §Decision item 3 + MULTI_HORIZON.md §"Convencao canonica" + impacta 20_method.md se F.2 afetado | **Doc PR primeiro** (R-20.pre): ADR + MULTI_HORIZON.md mergeados antes do code PR. Code PR (R-20) cita ADR pos-merge. |
| (b) | **Operational** — preserva semantica; muda apenas adapter (le target_return direto do split_df) | Code PR direto; ADR §Consequences amendada no mesmo PR |
| (c) | **Operational** (documentacao apenas) — Gap 6 marcado RESOLVED PARTIAL | Code PR direto; ADR §Consequences amendada |
| (d) | **Operational** — preserva semantica "next-day return"; muda formula da coluna (mecanico) | Code PR direto; ADR formula item 3 amendada no mesmo PR (mudanca de formula, nao de semantica) |

## R-20.1 — Implementacao do fix (depende de R-20.0)

**Persister call sites (referencia para impacto por opcao):**

Verificado via `grep MultiHorizonPredictionPersister.build_record`:
- `src/use_cases/run_baselines_use_case.py:268` (baseline path)
- `src/use_cases/train_tft_model_use_case.py:820` (TFT multi-horizon path)
- `src/use_cases/train_tft_model_use_case.py:858` (TFT legacy 1-horizon path)

Impacto por Opcao:
- **(a):** ajustar `run_baselines_use_case.py:259` (1 site). Sites TFT inalterados (recebem y_true_m do adapter).
- **(b):** modificar adapter `pytorch_forecasting_tft_trainer.py:480`. Sites TFT recebem dados diferentes mas codigo dos sites nao muda.
- **(c):** sem mudanca de codigo.
- **(d):** ajustar `build_tft_dataset_use_case.py:563` (formula) + `run_baselines_use_case.py:259` (indexacao). Sites TFT inalterados (adapter le coluna atualizada automaticamente).

**Se (d) escolhida (recomendada):**

1. Modificar
   [`src/use_cases/build_tft_dataset_use_case.py:563`](../../src/use_cases/build_tft_dataset_use_case.py#L563):
   ```python
   # ANTES (forward, problema):
   df["target_return"] = np.log(df["close"].shift(-1) / df["close"])
   # DEPOIS (backward, correto per Opcao d):
   df["target_return"] = np.log(df["close"] / df["close"].shift(1))
   df = df.dropna(subset=["target_return"]).reset_index(drop=True)
   ```

2. Rebuild dataset (CLI canonico do projeto):
   ```bash
   .venv/bin/python -m src.main_dataset_tft --asset AAPL
   ```

3. Modificar
   [`src/use_cases/run_baselines_use_case.py:259`](../../src/use_cases/run_baselines_use_case.py#L259):
   ```python
   # ANTES:
   y_true_idx = i + h_int - 1
   # DEPOIS:
   y_true_idx = i + h_int
   ```
   (compensa o shift; valor numerico identico ao TFT decoder agora.)

4. Atualizar
   [`docs/01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md`](../../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md):
   - §Decision item 3: "y_true = dataset.target_return[decision_idx + h]"
     (formula muda; interpretacao "next-day return after decision" preserva
     porque target_return passa a ser backward-indexed).
   - §Consequences: documentar que target_return semantics shifted
     backward; archive pre-Phase B mantem formula antiga (registro
     historico).

5. Atualizar
   [`docs/03_modeling/MULTI_HORIZON.md`](../../03_modeling/MULTI_HORIZON.md)
   §"Convencao canonica" para refletir nova formula + indexacao.

6. Atualizar
   [`src/domain/services/multi_horizon_prediction_persister.py`](../../src/domain/services/multi_horizon_prediction_persister.py)
   docstring class — referencia ao ADR atualizado.

7. Atualizar tests afetados:
   - `tests/unit/use_cases/test_run_baselines_use_case.py` — recalcular
     valores esperados (e.g., `test_historical_mean_skips_rows_without_warmup_window`)
   - `tests/unit/use_cases/test_train_tft_model_use_case.py` — recalcular
     `test_persist_fact_oos_predictions_keeps_horizon_index_alignment_per_split`
     com nova indexacao

**Se (a), (b), ou (c) escolhida:** ver versao previa do plano para detalhes.
Resumo: (a) muda semantica financeira; (b) cria desalinhamento training
loss vs metrica reportada; (c) deixa Gap 6 em aberto.

## R-20.2 — Adapter integration test focado (substitui hollow test)

**Critica recebida da meta-auditoria:** smoke completo (max_epochs=1) e
caro em CI (~minutos por test). Alternativa proposta na meta-auditoria
de "mockar o trainer" foi rejeitada — bug vive no adapter, mockar o
trainer esconde exatamente o que precisa ser detectado.

**Abordagem correta:** adapter integration test SEM treino do modelo.
Constroi `TimeSeriesDataSet` direto, instancia dataloader, le `y[0]`,
verifica indexacao.

**Arquivo novo:** `tests/integration/test_tft_decoder_target_indexing.py`.

```python
def test_decoder_target_value_matches_persisted_y_true(tmp_path):
    """Adapter integration: verify that for sample with encoder_end=D,
    dataloader y[0][i, h-1] equals dataset.target_return at the index
    that ADR-0003 (after Opcao d) prescribes for that (decision_idx, h).
    NO model training; only dataset construction + dataloader iteration.
    """
    # 1. Synthetic split_df com 30 dias + close known + target_return
    #    construido per nova formula (log(close[t]/close[t-1]))
    # 2. TimeSeriesDataSet(split_df, target="target_return",
    #    max_encoder_length=10, max_prediction_length=7, ...)
    # 3. dataloader = ds.to_dataloader(train=False, batch_size=N)
    # 4. actuals = torch.cat([y[0] for _, y in iter(dataloader)], dim=0)
    # 5. Para cada sample i e horizon h em {1, 7}:
    #    expected = split_df["target_return"].iloc[
    #        (max_encoder_length - 1) + i + h
    #    ]
    #    assert actuals[i, h-1] == pytest.approx(expected, abs=1e-9)
```

Custo CI: ~5-10 segundos por test (sem treino).

**Substituir tambem** `tests/integration/test_tft_baselines_y_true_alignment.py`
(que era hollow):

```python
def test_tft_and_baselines_produce_equal_y_true_for_same_target_ts(tmp_path):
    """Cross-pipeline guard: TFT adapter + baseline use case both produce
    y_true values that match at same (target_timestamp_utc, horizon).
    Detects regression of Gap 6 off-by-one.
    """
    # 1. Synthetic dataset (30 dias, close + target_return per Opcao d)
    # 2. Construir TimeSeriesDataSet + dataloader; extrair actuals (NAO treinar)
    # 3. Para cada sample i, extrair (decision_idx, h, y_true) emitido pelo TFT path
    # 4. Rodar baseline _emit_oos_rows sobre mesmo dataset (zero_return)
    # 5. Join por (target_timestamp_utc, horizon); assert y_true igual
```

Custo CI: ~15-30 segundos.

**Aceite R-20.2:**
- Adapter test detecta o off-by-one se R-20.1 nao for aplicado.
- Cross-pipeline test detecta o off-by-one e divergencia de range.
- Ambos passam apos R-20.1 implementado.

## R-20.2.bis — Tests faltantes em Persister (coverage gaps confirmados)

Auditoria empirica revelou cobertura ausente:

1. **Year boundary test** — partition `(asset, feature_set_name, year)` ja
   muda comportamento se decision_day esta em Dec e target_ts em Jan
   (e.g., decision 2023-12-29, h=7 → target 2024-01-08). Atual
   `test_year_reflects_decision_day` so verifica 1 ponto Dec/Jan; precisa
   cobertura de:
   ```python
   def test_year_partition_uses_decision_day_year_across_boundary(self):
       """decision 2023-12-29, h=5 → year column = 2023 (decision_day.year),
       not target.year=2024. Verifies the Stage 20 semantic shift documented
       in MULTI_HORIZON.md and Persister.build_record year=int(ts.year).
       """
       ts = [pd.Timestamp(t, tz="UTC") for t in
             pd.date_range("2023-12-26", periods=15, freq="D")]
       record = MultiHorizonPredictionPersister.build_record(
           decision_idx=3, h=5, ..., dataset_timestamps=ts, run_context=rc
       )
       assert record.year == 2023
       assert pd.Timestamp(record.target_timestamp_utc).year == 2024
   ```

2. **decision_start_offset verification** — atualmente `max_encoder_length - 1`
   e usado no TFT call-site sem teste unit explicito. Adicionar:
   ```python
   def test_persist_uses_decision_start_offset_equals_max_encoder_minus_one(self):
       """Sample i in matrix maps to decision_idx = max_encoder_length - 1 + i
       per pytorch_forecasting TimeSeriesDataSet convention. Regression
       guard if matrix iteration semantics change.
       """
       # split_df com 100 timestamps; max_encoder=30
       # 1 sample in matrix → expected decision_idx = 29
       # Persister output should have decision_idx=29 for i=0
   ```

Arquivos: `tests/unit/domain/services/test_multi_horizon_prediction_persister.py`
+ `tests/unit/use_cases/test_train_tft_model_use_case.py`.

## R-20.3 — Re-smoke + reproducao da auditoria

Reproduzir o experimento do auditor:

```bash
rm -rf data/analytics/silver/* data/analytics/gold/*
.venv/bin/python -m src.main_train_tft --asset AAPL ... --parent-sweep-id phase_a_smoke_20260518_v2
.venv/bin/python -m src.main_baselines_test_pipeline ...
.venv/bin/python -m src.main_refresh_analytics_store ...

# Diff TFT vs baseline em fact_oos_predictions
.venv/bin/python -c "
import pandas as pd
from pathlib import Path
df = pd.concat([pd.read_parquet(p) for p in Path('data/analytics/silver/fact_oos_predictions').rglob('*.parquet')])
tft = df[df['model_version'].str.contains('B$')].set_index(['target_timestamp_utc', 'horizon'])
base = df[df['model_version'].str.startswith('baseline_zero')].set_index(['target_timestamp_utc', 'horizon'])
join = tft[['y_true']].join(base[['y_true']], lsuffix='_tft', rsuffix='_base', how='inner')
diff = (join['y_true_tft'] - join['y_true_base']).abs()
print(f'Rows joined: {len(join)}; equal (<1e-9): {(diff<1e-9).sum()}; off-by-one expected: {663 if FIX_NOT_APPLIED else 0}')
"
```

Esperado pos-fix: `equal == len(join)` para todas as linhas que ambos
emitem (intersecao).

## R-20.4 — Validacao final

```bash
.venv/bin/pytest tests/ -q                                            # 559+ passed
.venv/bin/pytest tests/integration/test_tft_baselines_y_true_alignment.py -v  # passa com pipelines reais
.venv/bin/ruff check src/ tests/                                       # 0 errors
du -sh data/analytics_archive_pre_phase_b/                             # 851M
```

## R-20.5 — Update log + push

- Atualizar [`docs/07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md`](../../07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md)
  com entry Stage R-20 (decisao tomada, opcao escolhida, evidencia
  empirica do fix).
- Commit + push (commits novos; sem rebase).

**Aceite Stage R-20:**
- RED-1 resolvido (TFT.y_true == baseline.y_true em mesma target_ts).
- RED-2 resolvido (teste cross-pipeline e nao-hollow e detectaria regressao).
- Decisao Opcao (a/b/c/d) registrada em ADR + log.
- pytest verde; ruff verde; archive intacto.
- **Wait for review** (sessao separada audita PR #44; aprova ou pede mudancas).
- Marcelo merge PR #44.

---

# Stage R-E — Extracao do fix `suffixes=("", "_dim")` para PR isolada (RED-6)

**Classification:** **Operational** (extracao mecanica de fix existente + 1 regression test).
**Implementer target:** **Codex** (mecanico).
**Tempo estimado:** 1-2h.

### Protected files (NAO TOCAR)
- Tudo exceto:
  - `src/use_cases/refresh_analytics_store_use_case.py` (somente o trecho
    `_build_gold_oos_quality_report` ~linha 1486 onde o merge ocorre;
    `suffixes=("", "_dim")` adiciona ao argumento)
  - `tests/unit/use_cases/test_refresh_analytics_store_use_case.py` (adicionar
    regression test)

**Objetivo:** o fix de bug em `_build_gold_oos_quality_report`
([`src/use_cases/refresh_analytics_store_use_case.py:1486`](../../src/use_cases/refresh_analytics_store_use_case.py#L1486)
onde merge call ocorre; argumento `suffixes=("", "_dim")` na linha 1490
da branch Stage 23) introduzido em Stage 23 e mecanicamente correto mas
violou escopo (Stage 23 era smoke + closure, nao fix de bug em arquivo
que Stage 22 deveria refatorar). Extrair para PR isolada contra main.

**Branch nova:**
```bash
git checkout main && git pull
git checkout -b fix/refresh-asset-merge-collision
```

## R-E.1 — Copiar o fix

- Aplicar a mesma mudanca de
  [`src/use_cases/refresh_analytics_store_use_case.py:1486`](../../src/use_cases/refresh_analytics_store_use_case.py#L1486)
  (merge call `df = df.merge(...)` recebe argumento `suffixes=("", "_dim")`
  na linha 1490 da branch Stage 23) — copiar do commit `69fabc3` da branch
  `feat/stage-23-f1-pass-pleno`.

## R-E.2 — Teste de regressao

Adicionar a
[`tests/unit/use_cases/test_refresh_analytics_store_use_case.py`](../../tests/unit/use_cases/test_refresh_analytics_store_use_case.py):

```python
def test_build_gold_oos_quality_report_preserves_asset_after_dim_run_merge(tmp_path):
    """RED-6 regression: merge collision on `asset` column previously
    produced asset=None in the report; downstream gold_quality_statistics_report
    then showed statistics_ready=False. Fix uses suffixes=("", "_dim").
    """
    # Fixtures: fact_oos_predictions com asset=AAPL + dim_run com asset=AAPL
    # Run _build_gold_oos_quality_report
    # Assert report['asset'].notna().all() and (report['asset'] == 'AAPL').all()
```

## R-E.3 — Validacao + PR

```bash
.venv/bin/pytest tests/unit/use_cases/test_refresh_analytics_store_use_case.py -v
.venv/bin/ruff check src/ tests/
```

PR contra main:
```bash
gh pr create --base main --title "fix(refresh): preserve asset column on dim_run merge (regression test)"
```

**Aceite R-E:**
- PR mergeada em main.
- Apos merge, branches `feat/stage-22-*` e `feat/stage-23-*` ja contem
  o mesmo fix (via stacking) — git resolve graciosamente como no-op no
  merge final.

---

# Stage R-21 — Migracao completa de QualityCheckRegistry (PR #45)

**Branch:** `feat/stage-21-quality-check-registry`.
**PR:** #45.
**RED a resolver:** RED-3 (skeleton dead code), RED-4 (escopo nao
autorizado).
**Tempo estimado:** 10-16h.
**Classification:** **Operational** (refactor mecanico; preservar
bit-identical em 2 scope_modes). Aceite §"Bit-identical regression PASS"
e medida objetiva.
**Implementer target:** **Codex** (migrar 30 checks para classes seguindo
padrao do skeleton existente; mecanico). Excecao: se checks tem edge
cases sutis (skip conditions, scope filters), Claude pode ajudar em
R-21.5 (orchestrator).

### Protected files (NAO TOCAR)
- `src/adapters/pytorch_forecasting_tft_trainer.py` (territorio R-20)
- `src/use_cases/build_tft_dataset_use_case.py` (territorio R-20)
- `src/use_cases/train_tft_model_use_case.py` (territorio R-20)
- `src/use_cases/run_baselines_use_case.py` (territorio R-20)
- `src/use_cases/refresh_analytics_store_use_case.py` (territorio R-22)
- `src/domain/services/gold_builders/**` (territorio R-22)
- `src/domain/services/multi_horizon_prediction_persister.py` (R-20)
- `docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (territorio R-23)
- `data/analytics_archive_pre_phase_b/**` (intocavel sempre)

### Required reading (contexto antes de R-21.1)
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §4 — full enumeration dos quality gates aplicados por tabela silver.
  Use para mapear cada check do monolito ao seu cluster (cardinality /
  alignment / calibration / contracts). Cada subsecao 4.X lista
  "Quality gates aplicados" com line refs ao
  `validate_analytics_quality_use_case.py`.

## R-21.0 — Decisao escopo (BLOQUEADOR; Marcelo confirma)

**Opcoes:**

| Opcao | O que acontece com #45 | Risco |
|---|---|---|
| (1) Migracao completa | Implementar tasks §21.2-21.8 do plano original | Tempo (~12h) |
| (2) Fechar #45 como WONTFIX, abrir Stage 21.bis depois | Skeleton fica em main como dead code | Codigo morto persistente; visibilidade ruim |
| (3) Fechar #45 sem merge; abrir Stage 21.bis com escopo completo | Trabalho ja feito vira pendente | Ok mas gera fluxo de duas PRs no historico |

**Recomendacao baseada em "um stage por vez":** (1). User declarou querer
migracao apropriada; skeleton em main viola "nao deixar half-finished".

## R-21.1 — Migrar cluster cardinality (per plano §21.2)

Arquivo novo: `src/domain/services/quality_checks/cardinality.py`.

Migrar 6 checks de [`validate_analytics_quality_use_case.py`](../../src/use_cases/validate_analytics_quality_use_case.py):

| Check | Linha origem | Subclass nova |
|---|---|---|
| `cardinality_config_fold_seed` | ~784 | `CardinalityConfigFoldSeedCheck` |
| `min_samples_by_split` | ~805 | `MinSamplesBySplitCheck` |
| `run_id_execution_consistency` | ~383 | `RunIdExecutionConsistencyCheck` |
| `inference_predictions_continuity` | ~361 | `InferencePredictionsContinuityCheck` |
| `feature_contrib_local_continuity` | ~379 | `FeatureContribLocalContinuityCheck` |
| `required_metrics_nan` | ~448 | `RequiredMetricsNanCheck` |

Cada subclass implementa `applies_when(scope) -> bool` + `run(snapshot) -> CheckResult`.

Mover testes correspondentes de [`tests/unit/use_cases/test_validate_analytics_quality_use_case.py`](../../tests/unit/use_cases/test_validate_analytics_quality_use_case.py)
para `tests/unit/domain/services/quality_checks/test_cardinality.py`.

**Verificacao bit-identical:** outputs (passed, detail) byte-byte iguais
aos do codigo monolitico para mesmo input.

## R-21.2 — Migrar cluster alignment (per plano §21.3)

Arquivo: `src/domain/services/quality_checks/alignment.py`.

| Check | Linha origem | Subclass |
|---|---|---|
| `tft_baselines_timestamp_subset_alignment` | ~92 (helper) + ~339 (record) | `TftBaselinesTimestampSubsetAlignmentCheck` |
| `oos_pairwise_target_alignment` | ~639 | `OosPairwiseTargetAlignmentCheck` |

Move testes correspondentes.

## R-21.3 — Migrar cluster calibration (per plano §21.4)

Arquivo: `src/domain/services/quality_checks/calibration.py`.

| Check | Linha origem | Subclass |
|---|---|---|
| `gold_confidence_calibrated_by_horizon` | ~957 | `GoldConfidenceCalibratedByHorizonCheck` |
| `dm_mcs_persisted_executable` | ~901 | `DmMcsPersistedExecutableCheck` |
| `gold_metrics_by_config_n_oos_contract` | ~1015 | `GoldMetricsByConfigNOosContractCheck` |

## R-21.4 — Migrar cluster contracts (per plano §21.5)

Arquivo: `src/domain/services/quality_checks/contracts.py`.

Identificar restantes (~15 checks): `official_contract_quantile_attention`,
`baseline_present_per_candidate_sweep`, `referential_integrity`, etc.
Listar comprehensively via `grep -n "self._record(" src/use_cases/validate_analytics_quality_use_case.py`.

## R-21.5 — Orchestrator + Dependency Injection (per plano §21.6 + audit)

Reduzir `ValidateAnalyticsQualityUseCase.execute()` para orchestrator
fino. **DI explicito:** registry como parametro do constructor (default
factory para producao; injectable em testes).

```python
# src/domain/services/quality_checks/__init__.py
def _build_default_registry() -> QualityCheckRegistry:
    reg = QualityCheckRegistry()
    # Ordem de registro = ordem original do _record no monolito (preserva
    # shape do `checks` list para bit-identical regression).
    reg.register(CardinalityConfigFoldSeedCheck())
    reg.register(MinSamplesBySplitCheck())
    # ... resto na ordem original
    return reg

DEFAULT_QUALITY_REGISTRY = _build_default_registry()


# src/use_cases/validate_analytics_quality_use_case.py
class ValidateAnalyticsQualityUseCase:
    def __init__(
        self,
        *,
        analytics_silver_dir: Path,
        analytics_gold_dir: Path | None = None,
        scope_spec: ScopeSpec | None = None,
        # ... outros parametros existentes
        registry: QualityCheckRegistry | None = None,  # NEW: DI
    ) -> None:
        # ...
        self._registry = registry or DEFAULT_QUALITY_REGISTRY

    def execute(self) -> AnalyticsQualityResult:
        scope_spec = self._resolve_scope_spec()
        snapshot = self._load_snapshot(scope_spec)
        results = [
            check.run(snapshot)
            for check in self._registry.applicable(scope_spec)
        ]
        return AnalyticsQualityResult(
            passed=all(r.passed for r in results),
            checks=[{"check": r.name, "passed": r.passed, "detail": r.detail}
                    for r in results],
        )
```

`_load_snapshot(scope_spec)` consolida todos os parquet reads atuais
espalhados no monolito.

**Por que DI:** testes podem instanciar use case com `registry=mock_registry`
sem monkey-patch do modulo `__init__`. Padrao consistente com
`MultiHorizonPredictionPersister` style (servico injectable) e com o
restante da Clean Arch do projeto.

**Aceite §R-21.5:** `validate_analytics_quality_use_case.py` reduz para
~200-300 LOC (de 1.152); DI via `registry=` parametro funciona em testes.

## R-21.6 — Doc canonico (per plano §21.7)

Adicionar a [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md)
secao "Quality check extension point" com exemplo:

```python
class MyNewCheck(QualityCheck):
    name = "my_new_check"
    def applies_when(self, scope): return scope.scope_mode == "cohort_decision"
    def run(self, snapshot): ...

# Em quality_checks/__init__.py, registrar antes do orchestrator chamar.
```

## R-21.7 — Smoke bit-identical multi-mode (per plano §21.8 + audit)

**Critica recebida:** smoke single-run cobre apenas 1 path. Auditoria
empirica confirmou que apenas 2 checks tem branch explicito por
scope_mode ([linhas 107 e 1021](../../src/use_cases/validate_analytics_quality_use_case.py)),
mas esses 2 paths nao seriam exercitados por smoke em `cohort_decision`
apenas.

**Procedimento (matriz 2-mode):**

Para CADA scope_mode em `{cohort_decision, global_health}`:

1. Checkout main (pre-merge R-21). Wipe silver/gold. Rodar smoke +
   `main_refresh_analytics_store` + `validate_analytics_quality` com
   `--scope-mode <mode>`. Salvar output em
   `/tmp/quality_pre_r21_<mode>.json`.
2. Checkout branch R-21 HEAD. Wipe silver/gold. Rodar mesmo smoke
   com mesmo scope_mode.
3. Comparar:
   ```python
   import json
   for mode in ["cohort_decision", "global_health"]:
       pre = json.load(open(f"/tmp/quality_pre_r21_{mode}.json"))
       post = json.load(open(f"/tmp/quality_post_r21_{mode}.json"))
       assert pre["checks"] == post["checks"], f"Bit-identical FAIL ({mode})"
   ```

Specifically verifica que:
- Sob `global_health`: checks 107 (`tft_baselines_timestamp_subset_alignment`)
  e 1021 (`baseline_present_per_candidate_sweep`) emitem
  `"skipped(not_cohort_decision)"` em ambos pre e post — texto exato,
  passed identico.
- Sob `cohort_decision`: comportamento concreto identico bit-byte.

Tempo CI: ~2x do single-mode smoke (~2-3 min total).

**Aceite R-21.7:** bit-identical PASS em ambos os modes. Quaisquer
diferencas devem ser cosmeticas (e.g., field ordering); valores e ordem
dos checks identicos.

## R-21.7.bis — Tests faltantes em registry/orchestrator (audit gaps)

Adicionar a `tests/unit/domain/services/quality_checks/`:

1. **applies_when matrix por scope_mode:**
   ```python
   @pytest.mark.parametrize("check_class,scope_mode,expected_runs", [
       (TftBaselinesTimestampSubsetAlignmentCheck, "cohort_decision", True),
       (TftBaselinesTimestampSubsetAlignmentCheck, "global_health", False),
       (BaselinePresentPerCandidateSweepCheck, "cohort_decision", True),
       (BaselinePresentPerCandidateSweepCheck, "global_health", False),
       (CardinalityConfigFoldSeedCheck, "cohort_decision", True),
       (CardinalityConfigFoldSeedCheck, "global_health", True),
       # ... resto na matriz
   ])
   def test_applies_when_per_scope_mode(check_class, scope_mode, expected_runs):
       check = check_class()
       scope = ScopeSpec(scope_mode=scope_mode)
       assert check.applies_when(scope) == expected_runs
   ```

2. **Order de registro preservada** — assert `[c.name for c in registry]`
   coincide com ordem do `_record` no monolito original (snapshot list).

3. **Orchestrator com snapshot parcial:**
   ```python
   def test_orchestrator_handles_missing_optional_tables(tmp_path):
       """Snapshot sem fact_inference_predictions: checks que dependem dele
       devem reportar 'skipped(no_inference_predictions)' nao crashar.
       """
   ```

4. **`_load_snapshot` helper:**
   ```python
   def test_load_snapshot_loads_all_required_tables(tmp_path):
       # Setup silver com tabelas conhecidas; assert snapshot.tables contains
       # exatamente o conjunto esperado
   ```

Custo: ~10-15 unit tests, ~segundos no total.

## R-21.8 — Atualizar ADR + log + validacao

- Atualizar [`docs/01_architecture/decisions/ADR-0004-quality-check-registry.md`](../../01_architecture/decisions/ADR-0004-quality-check-registry.md)
  status de "Accepted" para "Implemented (Stage R-21, PR #45)".
- Atualizar log com Stage R-21 completion.
- pytest verde; ruff verde; archive intacto.

**Aceite Stage R-21:**
- 30 checks migrados; orchestrator reduzido.
- Bit-identical regression PASS.
- ADR atualizado.
- **Wait for review.**
- Marcelo merge PR #45.

---

# Stage R-22 — Migracao completa de GoldBuilders (PR #46)

**Branch:** `feat/stage-22-gold-builders-modular`.
**PR:** #46.
**RED a resolver:** RED-5 (skeleton dead code).
**Tempo estimado:** 16-24h.
**Classification:** **Operational** com **policy edge case** em R-22.2.bis
(introduz conceito `requires_gold` + topology test; ANALYTICS_STORE_ARCHITECTURE.md
deve registrar como extension point). R-22.0.bis (rename Snapshots) +
R-22.1 (migracao mecanica de 26 builders) + R-22.3 (smoke byte-identical)
sao operational puros.
**Implementer target:** **Claude** (nova sessao). Justificativa: dependency
ordering (R-22.2.bis) exige raciocinio arquitetural sobre Opcao A/B/C
hibrida. Codex pode auxiliar em R-22.1 (migracao mecanica de cada builder)
mas orchestrator + topology test precisam decisao.

### Protected files (NAO TOCAR)
- `src/use_cases/validate_analytics_quality_use_case.py` (territorio R-21)
- `src/domain/services/quality_checks/**` (territorio R-21)
- `src/adapters/pytorch_forecasting_tft_trainer.py` (territorio R-20)
- `src/use_cases/build_tft_dataset_use_case.py` (territorio R-20)
- `src/use_cases/train_tft_model_use_case.py` (territorio R-20)
- `src/use_cases/run_baselines_use_case.py` (territorio R-20)
- `src/domain/services/multi_horizon_prediction_persister.py` (R-20)
- `docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (territorio R-23)
- `data/analytics_archive_pre_phase_b/**` (intocavel sempre)

### Required reading (contexto antes de R-22.1)
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §6 — full enumeration dos 26 gold builders (`gold_runs_long`,
  `gold_ranking_by_config`, ... `gold_quality_statistics_report`) com
  file:line do monolito `refresh_analytics_store_use_case.py`. Use como
  mapa para migracao mecanica de cada builder.
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §6.2.4-6.2.5 + §6.3.10 — calculos detalhados de DM pairwise + MCS
  (loss matrix, Holm adjustment). Critico para preservar byte-identical
  apos modularizacao do cluster `pairwise.py`.
- [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../02_data/DATA_PIPELINE_WALKTHROUGH.md)
  §6.3.15 — calculo de `gold_quality_statistics_report` (Tier 3 com
  deps em quality_report + dm_results + mcs_results + win_rate_results).
  Use para validar correta propagacao via `BuildContext.gold_outputs`
  per R-22.2.bis.

**Pre-condicao:** Stage R-E (suffix fix em PR #48) mergeada em main.
Stage R-21 mergeada (para evitar drift entre validate e refresh modulares).

## R-22.0 — Decisao escopo

Mesmas 3 opcoes do R-21.0. Recomendacao: (1) migracao completa, mesma
razao.

## R-22.0.bis — Rename AnalyticsSnapshot para evitar DRY violation

Auditoria detectou: `quality_checks/base.py` e `gold_builders/base.py`
ambos definem classe `AnalyticsSnapshot` com shapes diferentes (Quality
tem scope_spec + scoped_run_ids; Gold tem so tables). Mesmo nome,
modulos diferentes — confusao garantida.

**Acao:**
- `src/domain/services/quality_checks/base.py`: renomear
  `class AnalyticsSnapshot` → `class QualityCheckSnapshot`. Atualizar
  imports em `__init__.py` e em testes.
- `src/domain/services/gold_builders/base.py`: renomear
  `class AnalyticsSnapshot` → `class GoldBuilderSnapshot`. Atualizar
  imports.

Nao unificar em um modulo `_shared/` — Quality e Gold tem semanticas
diferentes (scope-aware vs nao) e merge prematuro adicionaria
acoplamento desnecessario.

**Aceite R-22.0.bis:** grep por `class AnalyticsSnapshot` retorna 0
hits; cada cluster usa nome explicito.

## R-22.1 — Migrar 5 clusters (per plano §22.2-22.6)

Cada cluster vira arquivo proprio:

| Cluster | Arquivo | Builders |
|---|---|---|
| ranking | `src/domain/services/gold_builders/ranking.py` | runs_long, ranking_by_config, consistency_topk |
| pairwise | `src/domain/services/gold_builders/pairwise.py` | dm_pairwise_results, mcs_results, paired_oos_intersection_by_horizon |
| quantile | `src/domain/services/gold_builders/quantile.py` | quantile_guardrail_audit, quantile_degeneracy_report, prediction_metrics_by_run_split_horizon, prediction_metrics_by_config |
| descriptive | `src/domain/services/gold_builders/descriptive.py` | ic95, feature_set_impact, model_decision_final, oos_consolidated, oos_quality_report |
| confidence | `src/domain/services/gold_builders/confidence.py` | confidence_calibrated_by_horizon, metrics_by_config_n_oos_contract, prediction_calibration, prediction_risk, prediction_generalization_gap, prediction_robustness_by_horizon, feature_impact_by_horizon, feature_contrib_local_summary, quality_statistics_report, quality_run_sweep_summary, win_rate_pairwise_results |

Cada builder vira `class XGoldBuilder(GoldBuilder)`:
- `output_table: str` = nome do parquet
- `requires: tuple[str, ...]` = tabelas silver/dim consumidas
- `build(snapshot, ctx) -> pd.DataFrame` = logica IDENTICA ao
  `_build_gold_X` do monolito (mesmas joins, groupbys, ordem de colunas).

Mover testes correspondentes para `tests/unit/domain/services/gold_builders/test_<cluster>.py`.

**Importante:** o fix `suffixes=("", "_dim")` (de R-E) ja em main precisa
ser preservado no builder modularizado (e.g.,
`OosQualityReportGoldBuilder.build`). Verificar via teste de regressao
adicionado em R-E.

## R-22.2 — Orchestrator + requires enforcement + DI (per plano §22.7 + audit)

**Pre-requisito:** definir excecao em
[`src/domain/services/gold_builders/base.py`](../../src/domain/services/gold_builders/base.py)
(append apos as classes existentes):

```python
class GoldBuilderRequirementError(RuntimeError):
    """Raised by the orchestrator when a builder's `requires` or
    `requires_gold` is not satisfied by the current snapshot/ctx.
    Fail-fast: indica bug em ordem de registro ou snapshot incompleto.
    """
```

Exportada via `gold_builders/__init__.py`.


```python
# src/domain/services/gold_builders/__init__.py
def _build_default_registry() -> GoldBuildersRegistry:
    reg = GoldBuildersRegistry()
    # Ordem topologica: Tier 1 → Tier 2 → Tier 3 (ver R-22.2.bis).
    # Tier 1 (silver/dim only):
    reg.register(RunsLongGoldBuilder())
    reg.register(RankingByConfigGoldBuilder())
    # ... resto Tier 1
    # Tier 2 (consume Tier 1 outputs via BuildContext):
    reg.register(PredictionMetricsByConfigGoldBuilder())
    # ... resto Tier 2
    # Tier 3:
    reg.register(QualityStatisticsReportGoldBuilder())
    # ...
    return reg

DEFAULT_GOLD_REGISTRY = _build_default_registry()


# src/use_cases/refresh_analytics_store_use_case.py
class RefreshAnalyticsStoreUseCase:
    def __init__(
        self,
        *,
        analytics_silver_dir: Path,
        analytics_gold_dir: Path,
        scope_spec: ScopeSpec,
        primary_quantile_contract: PrimaryQuantileContract = "post_guardrail",
        registry: GoldBuildersRegistry | None = None,  # NEW: DI
    ) -> None:
        # ...
        self._registry = registry or DEFAULT_GOLD_REGISTRY

    def execute(self):
        snapshot = self._load_silver_dim_snapshot(self._scope_spec)
        ctx = BuildContext(
            scope_spec=self._scope_spec,
            primary_quantile_contract=self._primary_quantile_contract,
        )
        outputs = {}
        for builder in self._registry.applicable(self._scope_spec):
            # Requires enforcement: fail rapido com mensagem clara
            missing = [t for t in builder.requires if not snapshot.has(t)]
            if missing:
                raise GoldBuilderRequirementError(
                    f"{builder.output_table} requires {builder.requires}, "
                    f"snapshot missing: {missing}"
                )
            df = builder.build(snapshot, ctx)
            path = self.analytics_gold_dir / f"{builder.output_table}.parquet"
            self._safe_write(df, path)
            outputs[builder.output_table] = str(path)
            # Tier 2/3 propagation: outputs deste builder podem ser
            # consumidos por subsequentes via ctx (ver R-22.2.bis)
            ctx.gold_outputs[builder.output_table] = df
        self._normalize_partitions()
        return outputs
```

**Mudancas vs plano original:**
- DI via `registry=` parameter.
- `requires` enforcement antes de chamar `build()`.
- `ctx.gold_outputs` acumula DataFrames para Tier 2/3 (ver R-22.2.bis).

Helpers compartilhados (`_safe_write`, `_load_silver_dim_snapshot`,
`_normalize_partitions`) permanecem no use case.

**Aceite §R-22.2:** `refresh_analytics_store_use_case.py` reduz de 2.579
para ~400 LOC; `requires` enforced; DI funciona.

## R-22.2.bis — DECISAO ARQUITETURAL: dependency ordering entre builders

**Gap critico do plano original §22 (e omissao da minha versao inicial):**
26 builders nao sao independentes. Levantamento por leitura de assinaturas
em `refresh_analytics_store_use_case.py`:

**Tier 1 — silver/dim only (14 builders):** `runs_long`, `ranking_by_config`,
`consistency_topk`, `ic95`, `feature_set_impact`,
`prediction_metrics_by_run_split_horizon`, `quantile_guardrail_audit`,
`quantile_degeneracy_report`, `oos_consolidated`, `oos_quality_report`,
`dm_pairwise_results`, `mcs_results`, `paired_oos_intersection_by_horizon`,
`win_rate_pairwise_results`, `prediction_risk`, `feature_contrib_local_summary`.

**Tier 2 — consume Tier 1 outputs (7 builders):**
`prediction_metrics_by_config(metrics_run_split_h)`,
`prediction_metrics_by_horizon(metrics_run_split_h)`,
`prediction_calibration(metrics_run_split_h)`,
`prediction_generalization_gap(metrics_run_split_h)`,
`prediction_robustness_by_horizon(metrics_run_split_h)`,
`feature_impact_by_horizon(fact_model_artifacts, metrics_run_split_h)`.

**Tier 3 — consume Tier 1+2 (3 builders):**
`quality_statistics_report(quality_report, dm_results, mcs_results, win_rate_results)`,
`quality_run_sweep_summary(quality_report, quality_statistics, dim_run)`,
`model_decision_final(metrics_by_config, robustness_by_horizon, generalization_gap, dm_results, mcs_results, win_rate_results, paired_intersection, ...)`.

**Sem mecanica de propagacao, Tier 2/3 falham silenciosamente.** Plano
original §22 disse "ordem de execucao no orchestrator garante dependencia"
mas nao especificou COMO. Atual `for builder in registry.applicable(scope)`
funciona apenas se cada builder le tudo de `snapshot` (silver/dim) — nao
funciona para Tier 2/3.

**3 opcoes (Marcelo decide):**

| Opcao | Mecanica | Pro | Contra |
|---|---|---|---|
| (A) | `BuildContext.gold_outputs: dict[str, pd.DataFrame]` acumula outputs em ordem topologica (registro manual em ordem). Tier 2/3 leem via `ctx.gold_outputs["gold_runs_long"]`. | Sem topo-sort automatico; ordem explicita no registro; sem I/O extra | Disciplina manual: novo builder em posicao errada quebra silenciosamente |
| (B) | Cada builder declara `requires_gold: tuple[str, ...]` adicional. Orchestrator faz topo-sort programatico. | Robusto a re-ordenamento de registro; deteccao automatica de ciclos | Mais complexo; topo-sort em runtime |
| (C) | Builders Tier 2/3 re-leem parquets do disco (escritos pelo orchestrator entre builds). | Sem mudanca no contrato GoldBuilder; mecanica simples | ~25 parquet reads extras (lento); ordem de escrita ainda importa; acopla a I/O |

**Recomendacao desta sessao:** (A) + (B) hibrido.
- `BuildContext.gold_outputs` acumula (Opcao A) — limpo, sem I/O extra
- `requires_gold` declarado para documentacao e detecao em testes
- Topo-sort em testes (`tests/unit/.../test_registry_topology.py`) que
  valida ordem de registro vs `requires_gold` — falha rapida se ordem
  divergir

```python
# Em base.py
@dataclass
class BuildContext:
    scope_spec: ScopeSpec | None = None
    primary_quantile_contract: PrimaryQuantileContract = "post_guardrail"
    gold_outputs: dict[str, pd.DataFrame] = field(default_factory=dict)  # NEW

class GoldBuilder(ABC):
    output_table: str = ""
    requires: tuple[str, ...] = ()           # silver/dim tables
    requires_gold: tuple[str, ...] = ()      # NEW: other gold tables
    # ...
```

```python
# Tier 2 example
class PredictionMetricsByConfigGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_metrics_by_config"
    requires = ("dim_run",)  # nothing from silver beyond dim_run
    requires_gold = ("gold_prediction_metrics_by_run_split_horizon",)

    def build(self, snapshot, ctx):
        metrics = ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"]
        # ... compute
```

Orchestrator enforca `requires_gold` mesma forma que `requires`:
```python
missing_gold = [t for t in builder.requires_gold if t not in ctx.gold_outputs]
if missing_gold:
    raise GoldBuilderRequirementError(
        f"{builder.output_table} requires_gold {builder.requires_gold}, "
        f"missing: {missing_gold}. Likely a registration order bug."
    )
```

**Aceite R-22.2.bis:**
- `BuildContext` tem `gold_outputs` dict.
- `GoldBuilder` tem `requires_gold` attribute.
- Orchestrator enforca ambos.
- Teste de topologia em `tests/unit/domain/services/gold_builders/test_topology.py`
  verifica que `(requires_gold de B) ⊂ (output_table de builders antes de B na ordem de registro)`.

## R-22.3 — Smoke byte-identical (per plano §22.8)

Procedimento:
1. Checkout main (apos R-E + R-20 + R-21 mergeados). Wipe silver/gold.
   Rodar smoke completo. Copiar `data/analytics/gold/` para `/tmp/gold_pre_r22/`.
2. Checkout branch R-22 HEAD. Wipe gold (silver preservar se nao mudou).
   Rodar `main_refresh_analytics_store`.
3. Comparar parquets byte-byte via script:

```python
import pandas as pd
from pathlib import Path

GOLD_TABLES = [
    "gold_runs_long", "gold_ranking_by_config", ..., # 22+ tabelas
]
pre = Path("/tmp/gold_pre_r22")
post = Path("data/analytics/gold")
for tbl in GOLD_TABLES:
    pre_df = pd.read_parquet(pre / f"{tbl}.parquet").sort_values(list(...)).reset_index(drop=True)
    post_df = pd.read_parquet(post / f"{tbl}.parquet").sort_values(list(...)).reset_index(drop=True)
    pd.testing.assert_frame_equal(pre_df, post_df, check_exact=False, rtol=1e-9)
    print(f"{tbl}: OK")
```

Quaisquer diferencas: investigar (sorting? float precision?); corrigir
builder ate match exato.

## R-22.3.bis — Tests faltantes em gold builders (audit gaps)

Adicionar a `tests/unit/domain/services/gold_builders/`:

1. **BuildContext.primary_quantile_contract propagation** — verificar
   que builders Cat A respeitam `ctx.primary_quantile_contract` (raw vs
   post_guardrail produzem outputs distintos):
   ```python
   @pytest.mark.parametrize("contract", ["raw", "post_guardrail"])
   def test_prediction_metrics_uses_primary_quantile_contract(contract):
       builder = PredictionMetricsByRunSplitHorizonGoldBuilder()
       ctx = BuildContext(primary_quantile_contract=contract)
       # Snapshot fixture com quantile_p10/p90 e _post_guardrail
       df = builder.build(snapshot, ctx)
       # Assert que columns/values refletem o contract escolhido
   ```

2. **`requires` enforcement** — orchestrator deve falhar rapido se
   snapshot esta incompleto:
   ```python
   def test_orchestrator_raises_on_missing_required_silver_table():
       reg = GoldBuildersRegistry()
       reg.register(RunsLongGoldBuilder())  # requires=("dim_run",)
       use_case = RefreshAnalyticsStoreUseCase(..., registry=reg)
       # Setup silver SEM dim_run
       with pytest.raises(GoldBuilderRequirementError, match="dim_run"):
           use_case.execute()
   ```

3. **Dependency ordering (topology test):**
   ```python
   def test_registry_topology_respects_requires_gold():
       """For each builder B with requires_gold=(X,Y),
       X and Y must appear in registry BEFORE B (output_table).
       """
       reg = DEFAULT_GOLD_REGISTRY
       seen: set[str] = set()
       for b in reg:
           for dep in b.requires_gold:
               assert dep in seen, (
                   f"{b.output_table} requires {dep} but {dep} not yet "
                   f"registered. Registry order violates topology."
               )
           seen.add(b.output_table)
   ```

4. **Asset preservation regression (sobrevivencia do fix R-E):**
   ```python
   def test_oos_quality_report_builder_preserves_asset_after_dim_run_merge():
       """Regression guard: bug originalmente fixado em monolito via
       suffixes=('', '_dim'). Versao modular deve manter o fix.
       """
       builder = OosQualityReportGoldBuilder()
       # Fixture: fact_oos_predictions[asset=AAPL] + dim_run[asset=AAPL]
       df = builder.build(snapshot, ctx)
       assert df['asset'].notna().all()
       assert (df['asset'] == 'AAPL').all()
   ```

Custo: ~15-20 unit tests, ~segundos no total.

## R-22.4 — Atualizar ADR + log + validacao

- ADR-0005 status para "Implemented (Stage R-22)".
- Log entry Stage R-22 completion.
- pytest verde; archive intacto.

**Aceite Stage R-22:**
- 26 builders migrados; orchestrator reduzido a ~400 LOC.
- Byte-identical regression PASS em todas as gold tables.
- ADR-0005 atualizado.
- **Wait for review.**
- Marcelo merge PR #46.

---

# Stage R-23 — Closure de PR #47 (RED-7)

**Branch:** `feat/stage-23-f1-pass-pleno`.
**PR:** #47.
**RED a resolver:** RED-7 (max_epochs=1 → ≥5).
**Tempo estimado:** 3-5h.
**Classification:** **Closure gate** (nao Stage de codigo; validacao
empirica de F.1). Atualizacoes de docs/checklists apenas; smoke
deterministico.
**Implementer target:** **Claude** (executa smoke + valida 6/6 + atualiza
docs; precisa raciocinio para confirmar criterios e gerar golden file
test).

### Protected files (NAO TOCAR)
- Todo `src/` (Stage e gate de validacao, nao de codigo).
- `data/analytics_archive_pre_phase_b/**` (intocavel sempre).
- Permitido modificar:
  - `docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (nota F.1 textual
    apenas; checkbox `[x]` permanece per memory)
  - `docs/05_checklists/phase-a/POST_CLOSURE_FIXES_CHECKLIST.md` (texto Gap 6)
  - `docs/07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md` (secao nova post-R-23)
  - `docs/07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md` (entries R-23.*)
  - Possivel novo `tests/integration/test_f1_golden_smoke.py` (golden file)
  - `docs/07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md` (status final)

**Pre-condicao:** Stages R-20, R-E, R-21, R-22 mergeadas. CI verde em main.

## R-23.0 — Pre-condicao (sem toggle de checkbox; ver memory)

**Branch context:** R-23.0 acontece em `feat/stage-23-f1-pass-pleno`,
onde F.1 `[x]` foi marcado prematuramente em commit `69fabc3`.

**Politica do projeto** (memory `feedback_checklist_vs_docs`):
> "checklists sao imutaveis apos completos — sao registro historico de
> decisoes e progresso, nao fontes de verdade de definicoes"

Toggle do checkbox `[x] → [~] → [x]` gera ruido git e viola imutabilidade.
**Marker `[x]` permanece** em todo o ciclo de remediacao. Correcoes
fluem via NOTA TEXTUAL anexada ao item, em commit unico ao fim de R-23.3.

Nesta task R-23.0, apenas registrar em log
[`docs/07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md`](../../07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md)
que F.1 marker permanece [x] mas sua nota sera atualizada em R-23.3
apos re-smoke max_epochs>=5 confirmar 6/6.

Em [`docs/05_checklists/phase-a/POST_CLOSURE_FIXES_CHECKLIST.md`](../../05_checklists/phase-a/POST_CLOSURE_FIXES_CHECKLIST.md)
(este NAO e checkbox toggling — e atualizacao de descricao textual em
nota que ja era texto livre):
- Texto "Gap 6 RESOLVIDO" → "Gap 6 RESOLVIDO (Opcao d aplicada em
  Stage R-20 mergeado YYYY-MM-DD; off-by-one corrigido empiricamente;
  validado em R-23.1)".

## R-23.1 — Re-smoke com max_epochs >= 5 (aceite literal F.1)

```bash
rm -rf data/analytics/silver/* data/analytics/gold/*
du -sh data/analytics_archive_pre_phase_b/  # 851M verificar

.venv/bin/python -m src.main_train_tft \
  --asset AAPL \
  --features "BASELINE_FEATURES,TECHNICAL_FEATURES,SENTIMENT_FEATURES,FUNDAMENTAL_FEATURES" \
  --max-epochs 5 --max-encoder-length 60 --max-prediction-length 7 \
  --evaluation-horizons "[1, 7]" \
  --parent-sweep-id "phase_a_smoke_$(date +%Y%m%d)_v3" \
  --seed 20260517 --prediction-mode quantile

.venv/bin/python -m src.main_baselines_test_pipeline \
  --asset AAPL --config-json config/sweeps/explicit/phase_a_smoke_20260518_v2.json \
  --overwrite-on-collision

.venv/bin/python -m src.main_refresh_analytics_store \
  --scope-mode cohort_decision \
  --scope-sweep-prefixes "phase_a_smoke_" \
  --primary-quantile-contract post_guardrail \
  --fail-on-quality
```

**Tempo esperado:** ~10-30 min CPU (max_epochs=5 com features completas).

**Verificacao:** exit 0; failed_checks=[]; total_checks=27 (ou maior se
novos checks adicionados em R-21).

## R-23.2 — Validacao 6 criterios F.1 com evidencia

Per [`docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../../05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md)
§"Stage final F.1":

| # | Criterio | Verificar como |
|---|---|---|
| 1 | `% p10==p90 < 5%` | `block_quantile_degeneracy_gate` em failed_checks=[] |
| 2 | Zero violacoes probabilisticas | MPIW post_guardrail >= 0; crossing rate raw OK |
| 3 | Gold cohort-aware | `gold_quality_statistics_report` mostra parent_sweep_id populado |
| 4 | pytest tests/ PASS | `.venv/bin/pytest tests/ -q` → 0 failed |
| 5 | CI verde em PRs mergeados | gh pr checks para Stages R-20, R-E, R-21, R-22 (apos merges) — todos PASS |
| 6 | Refresh + quality 0 FAIL | exit 0 do main_refresh_analytics_store |

**Todos 6 devem PASS** para marcar F.1 `[x]`. Documentar em
[`docs/07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md`](../../07_reports/phase-gates/phase-a/smoke_confirmatory_2026-05-18.md)
secao "F.1 v3 PASS pleno (post-Stage R-23, YYYY-MM-DD)" substituindo a
secao Stage 20 atual (que foi inadequada).

## R-23.3 — Atualizar nota de satisfacao F.1 (checkbox permanece `[x]`)

Em [`docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md`](../../05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md):

**Checkbox `[x]` permanece** (per memory `feedback_checklist_vs_docs`).
Atualizar APENAS o texto da nota de satisfacao do F.1 num COMMIT UNICO,
substituindo a nota anterior (que referenciava smoke max_epochs=1) por:

> "satisfeito YYYY-MM-DD via smoke v3 (max_epochs=5, 6/6 criterios
> reproduzidos em sessao R-23 commit <hash>). Marker inicial em commit
> `69fabc3` foi prematuro porque smoke v2 tinha max_epochs=1 (aceite
> literal exige >=5); revalidacao em R-23.1 confirmou substancia."

F.3 mantem `[~]` (dependencia em F.2 pre-registro, separado).

## R-23.4 — Update log + validacao final

- Log entry Stage R-23 completion com hash do smoke run + comandos.
- Confirmar archive 851M.
- pytest verde.

**Aceite Stage R-23:**
- F.1 `[x]` legitimamente.
- Smoke max_epochs=5 reproduz **0 FAIL** (contagem total pode mudar pos-R-21 se checks consolidarem; aceite e ausencia de FAIL, nao count fixo).
- 5/6 criterios PASS objetivamente + criterio 5 PASS pos-merge.
- **Wait for review.**
- Marcelo merge PR #47.

---

# 4. Logging

Append-only em [`docs/07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md`](../../07_reports/phase-gates/phase-b/option_b_execution_log_2026-05-19.md).

Cada Stage de remediacao tem entries:
- `Stage R-N.X — decision` (escolhas-chave; e.g., Opcao a/b/c em R-20.0)
- `Stage R-N.X — completion` (task fechada)
- `Stage R-N — closure` (PR mergeada)

Formato identico ao log existente (§7.2 do plano original).

---

# 4.bis Rollback procedure (post-merge)

Se um Stage de remediacao for mergeado em main e descoberto bug subtil
depois, NAO usar rebase/force-push. Use `git revert <merge-commit>`:

```bash
git checkout main && git pull
git log --merges --oneline | head -10   # achar o merge commit do Stage
git revert -m 1 <merge-commit-sha>      # -m 1 mantem main como parent
# editar mensagem de revert se necessario
git push                                # main vai com revert; CI roda
```

Branches dos Stages stacked permanecem (preservadas como historico).
Marcelo pode reabrir nova PR contra o main pos-revert se quiser
re-tentar com fix.

**NAO USE:** `git reset --hard`, `git push --force`, `git push --force-with-lease`
em main. Revert preserva historia e e reversivel.

---

# 5. Abort conditions (remediacao)

| # | Condicao | Acao |
|---|---|---|
| 1 | Decisao R-20.0 (Opcao a/b/c) sem Marcelo disponivel | Pausar; nao implementar default. |
| 2 | Migracao R-21/R-22 quebra bit-identical e diff nao resolve em 3 tentativas | PR `[BLOCKED]`; analisar com Marcelo. |
| 3 | Smoke R-23 max_epochs=5 quebra algum check que passava em max_epochs=1 | Investigar; pode ser flaky ou bug real. |
| 4 | Archive `data/analytics_archive_pre_phase_b/` modificado | ABORT imediato; PR `[BLOCKED: archive_corruption]`. |
| 5 | Tempo total excede 60h ativas sem fechamento de R-22 | Negociar com Marcelo: reescopar ou pausar. |

---

# 6. DO / DO NOT

## DO

- Ler RED especifico antes de iniciar cada Stage R-X.
- Decisoes-chave (R-20.0, R-21.0, R-22.0) requerem Marcelo presente.
- Bit-identical / byte-identical regression em R-21 e R-22 e mandatoria.
- Cada Stage termina com `pytest tests/ -q` verde + archive intacto.
- Novos commits via NOVOS COMMITS (nao amend/rebase/force-push).
- Wait for review entre Stages.

## DO NOT

- NUNCA iniciar Stage R-N+1 antes de Stage R-N estar mergeado/aprovado.
- NUNCA refatorar "while I'm here" alem do escopo declarado de cada R-X.
- NUNCA marcar F.1 `[x]` em PHASE_B sem todos 6 criterios PASS substantivamente.
- NUNCA mover suffix fix de R-E para dentro de R-22 (extracao e o ponto).
- NUNCA bypassar bit-identical em R-21 ou byte-identical em R-22 mesmo se "looks the same".
- NUNCA aplicar Opcao (c) em R-20.0 sem registrar formal em ADR-0003 §Consequences.

---

# 7. Sumario final

| Stage | Branch / PR | Objetivo | Aceite | Tempo |
|---|---|---|---|---|
| Pre-Stage CI | `fix/lint-pre-existing-ruff-errors` / novo PR | CI verde em main + reconciliacao de branches stacked | 0 ruff errors; 4 PRs com lint verde apos merge main into branches | 1h |
| Stage R-20 | `feat/stage-20-...` / #44 | RED-1, RED-2 fechados (Opcao d default) | TFT.y_true == baseline.y_true; adapter integration test + cross-pipeline test detectam regressao | 6-10h |
| Stage R-E | `fix/refresh-asset-merge-collision` / novo PR | RED-6 extracao | Suffix fix em PR isolada com regression test | 1-2h |
| Stage R-21 | `feat/stage-21-...` / #45 | RED-3, RED-4 fechados | 30 checks migrados; orchestrator ~250 LOC; bit-identical PASS em 2 scope_modes; DI + tests de applies_when matrix | 10-16h |
| Stage R-22 | `feat/stage-22-...` / #46 | RED-5 fechado + R-22.2.bis arquitetura deps | 26 builders migrados; refresh ~400 LOC; byte-identical PASS; BuildContext.gold_outputs + requires_gold + topology test | 16-24h |
| Stage R-23 | `feat/stage-23-...` / #47 | RED-7 fechado | F.1 max_epochs=5 → 6/6 PASS; checkbox `[x]` preservado (per memory checklist-immutability); nota textual atualizada em commit unico; golden file test | 3-5h |
| **Total** | | | 7 RED resolvidos; 4 PRs prontas para merge | **37-58h** |

**Sucesso desta remediacao:** F.1 `[x]` legitimo + Gap 6 fechado em
substancia (Opcao d empiricamente validada) + Stages 21+22 com god-objects
desmontados (registries com DI, dependency ordering documentada e
enforcada) + CI verde + Marcelo confortavel mergeando PRs #44-#47 em
sequencia.

**O que NAO esta no escopo desta remediacao (separado, futuro):**
- F.2 pre-registro (documento separado em `docs/06_pre_registration/`).
  Sem F.2 mergeado, F.3 fica `[~]` e "Data de abertura da Fase B" no
  `A_code_audit.md` permanece em branco. Estado tecnico pos-R-23: Fase B
  PRONTA para abertura formal, aguardando F.2.
- Treino confirmatorio Phase B (gate F.4 hipotetico, pos-F.2).

**Comece executando §2 (pre-condicoes) e depois Pre-Stage CI.**
