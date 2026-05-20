---
title: Stage 20-23 Autonomous Execution Plan (Option B — Refactor + Gap 6 fix)
scope: Plano detalhado para execucao autonoma de Stages 20-23 sem supervisao direta. Extrai 3 god objects (Persister, QualityCheckRegistry, GoldBuilders) e fecha F.1 PASS pleno (Gap 6 fix). Documenta pre-condicoes, rubrica de decisao, higiene de dados, error handling, logging, abort conditions e PR/commit discipline. Destinado a Claude session em modo unsupervised.
update_when:
  - decisao tomada e registrada em option_b_execution_log_2026-05-19.md
  - Stage progride (task fechada / Stage mergeado)
  - abort condition gatilhada
  - Marcelo retorna e revisa
canonical_for: [stage_20_23_autonomous_plan, option_b_refactor_execution, gap_6_closure_plan]
---

# Stage 20-23 — Plano de Execucao Autonoma (Opcao B)

**Status:** ready for execution (pre-conditions a verificar)
**Criado:** 2026-05-19
**Executor:** Claude (autonomous session)
**Owner:** Marcelo (reviewer no retorno)

---

## Para quem le este documento

Voce e uma sessao Claude que vai executar **Stages 20, 21, 22 e 23**
do PHASE_B_IMPLEMENTATION_CHECKLIST.md sem supervisao direta de Marcelo
por uma janela de ~7-14 dias. Este documento e seu mapa operacional
completo. **Leia o documento inteiro antes de tocar em qualquer arquivo.**

Quando uma decisao for ambigua, aplique a **rubrica de decisao** (§4),
registre em **log de execucao** (§7) e prossiga. Nao pause para
confirmacao humana exceto nas **abort conditions** (§8).

---

## 0. Resumo executivo

| Stage | Objetivo | Branch | PRs | Tasks | Esforco esperado |
|---|---|---|---|---|---|
| 20 | Extrair `MultiHorizonPredictionPersister` | `feat/stage-20-multi-horizon-prediction-persister` | 1 | 7 | 5-10h Claude ativo |
| 21 | Extrair `QualityCheckRegistry` | `feat/stage-21-quality-check-registry` | 1 | 9 | 8-14h |
| 22 | Modularizar `GoldBuilders` | `feat/stage-22-gold-builders-modular` | 1 | 9 | 12-20h |
| 23 | Smoke F.1 PASS pleno + closure | `feat/stage-23-f1-pass-pleno` | 1 | 6 | 3-5h + tempo de smoke |

Total esperado: ~30-50h Claude ativo, distribuido em 4 PRs sequenciais.
Calendar realista (incluindo smoke runs + review eventual de Marcelo):
**7-14 dias**.

---

## 1. Principio operacional

**EXECUCAO CONTINUA.** Nao pause para confirmacao humana. Em cada turn:

1. Verifique pre-condicoes da task atual
2. Execute
3. Valide (tests + smoke quando aplicavel)
4. Commit + push
5. Registre em log se houve decisao nao-obvia
6. Avance para proxima task

**Pause apenas se:**
- Abort condition gatilhada (§8)
- 3 tentativas mal-sucedidas de fix do mesmo erro na mesma task
- Decisao critica nao prevista neste plano que mude semantica de
  comportamento documentado (ex: alterar contrato de Persister
  acordado em ADR-0003)

**NAO MEXA NO HISTORICO GIT:**
- Sem `git rebase` (interativo ou nao)
- Sem `git commit --amend`
- Sem `git push --force` ou `git push -f`
- Sem `git reset --hard` em commits ja pushados
- Sem squash de commits

Erros se corrigem com **NOVOS COMMITS**. Se um commit X.Y introduziu
bug, commit X.Y.fix (mensagem clara) corrige. Multiple fix-up commits
sao OK. PR final pode parecer "ruidosa" mas e historica honesta.

---

## 2. Pre-condicoes (verificar antes de iniciar Stage 20.0)

Execute esta sequencia. **Se qualquer falhar, abra PR `[BLOCKED]` ou
Issue descrevendo o problema e pare.**

```bash
# 2.1 Estado da arvore
git status
# Esperado: working dir clean
# Se modificacoes pendentes: nao deveriam existir. Verificar e
# decidir se commitar (em qual branch?) antes de prosseguir.

git branch --show-current
# Esperado: main

git fetch origin
git log --oneline main..origin/main
# Esperado: vazio (main local == remote). Se diferente:
# git pull origin main (rebase ou merge per padrao do repo)

# 2.2 F.0 + F.A mergeados
git log --oneline -20 main | grep -E "Stage F\.0|Stage F\.A|ADR-000[345]"
# Esperado: ver commits de F.0 (PR #40 ja mergeado) + F.A (ADRs 003/004/005)

# 2.3 Documentos de referencia disponiveis
test -f docs/01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md
test -f docs/01_architecture/decisions/ADR-0004-quality-check-registry.md
test -f docs/01_architecture/decisions/ADR-0005-gold-builders-modularization.md
test -f docs/07_reports/phase-gates/B_architectural_debt_2026-05-19.md
test -f docs/07_reports/tft_y_true_investigation_2026-05-18.md
# Todos devem existir

# 2.4 Tag checkpoint (criar se nao existir)
git tag -l "checkpoint-pre-option-b"
# Se vazio:
git tag -a checkpoint-pre-option-b -m "Pre-Option-B autonomous execution starting point"
git push origin checkpoint-pre-option-b

# 2.5 Tests green em main
.venv/bin/pytest tests/ -q 2>&1 | tail -5
# Esperado: "X passed", zero failed
# Se red: abortar. Marcelo investiga manualmente.

# 2.6 Data archive intacto (NAO TOCAR durante TODO Stage 20-23)
ls data/analytics_archive_pre_phase_b/ 2>/dev/null | head -3
# Esperado: existe e tem conteudo
# Se ausente: REGISTRAR em log mas prosseguir (talvez foi movido ou
# renomeado por outra sessao; Marcelo confirma na volta).

# 2.7 Decisao ADR-0003 Fix 3
grep -A 2 "Opcao (a)\|Opcao (b)\|Option (a)\|Option (b)" \
  docs/01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md
# Esperado: ver "Opcao (a)" como Decision (h=1 = next-day return after
# decision). Se mudou para (b), AJUSTAR Stage 20.1 (semantica do
# build_record) antes de comecar.

# 2.8 Verificar log nao existe ainda
test ! -f docs/07_reports/option_b_execution_log_2026-05-19.md && \
  echo "Log file ainda nao existe — sera criado no primeiro commit do Stage 20"
```

Se tudo passa: prossiga para §3 antes de Stage 20.0.

---

## 3. Convencoes de PR e commit

### Commit messages

Padrao do projeto (validado em `git log`):

```
<tipo>(<escopo>): <imperativo, max 70 chars> (Stage X.Y)

<corpo: por que, o que mudou, links para arquivos relevantes>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Tipos validos: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`.

Escopos comuns:
- `feat(domain)` — novo domain service / entity
- `refactor(train-tft)` — mudancas em train_tft_model_use_case
- `refactor(baselines)` — mudancas em run_baselines_use_case
- `refactor(analytics-quality)` — mudancas em validate_analytics_quality
- `refactor(refresh)` — mudancas em refresh_analytics_store
- `test(unit)` ou `test(integration)` — testes
- `docs(modeling)` — docs em 03_modeling/
- `docs(adr)` — ADRs

### Branch naming

```
<tipo>/stage-<n>-<short-objetivo-kebab>
```

Exemplos:
- `feat/stage-20-multi-horizon-prediction-persister`
- `feat/stage-21-quality-check-registry`
- `feat/stage-22-gold-builders-modular`
- `feat/stage-23-f1-pass-pleno`

### PR pattern

- 1 Stage = 1 PR
- Base = `main` (PRs nao stackeadas — cada Stage e atomico)
- Aguardar merge de Stage N antes de iniciar Stage N+1
- Titulo: `Stage X — <objetivo>`
- Corpo: Summary / Tasks completas / Test plan / Cross-link

### Quando algo da errado num PR aberto

1. Push commit fix novo (`<tipo>(<escopo>): fix <descricao> (Stage X.Y)`)
2. Comentar no PR explicando o que falhou e que commit corrigiu
3. NAO use `git commit --amend` ou squash
4. PR vai mostrar timeline honesta

---

## 4. Rubrica de decisao

Quando enfrentar escolha ambigua, aplique em ordem:

### 4.1. Principios do projeto (top priority)

Citados em `CLAUDE.md`, na memoria do Claude para este projeto, e em `B_architectural_debt_2026-05-19.md`:

- **No overengineering.** Prefira a solucao mais simples viavel.
  Servico stateful so se inevitavel. Sem "Leis" novas de governance.
  Sem checklists de review novos.
- **Mechanical > procedural.** Coluna no schema, gate tecnico, tipo
  explicito. Nao processo manual ou convencao "todo mundo deve
  lembrar".
- **Preservar comportamento bit-for-bit em refactor.** Stage 20-22
  sao reorganizacoes estruturais. Output dos pipelines deve ser
  byte-identical antes vs depois. Exceto onde Stage 20.1 aplica
  Opcao (a) do Fix 3, que MUDA `timestamp_utc` e `target_timestamp_utc`
  intencionalmente (Gap 6 fix embarcado).
- **Match precedente do projeto.** Novo domain service segue padrao
  de `src/domain/services/quantile_guardrail_service.py` (60 LOC,
  `@dataclass(frozen=True)` + `staticmethod`). Novo schema follows
  `src/infrastructure/schemas/analytics_store_schema.py` patterns.

### 4.2. Convencoes do projeto

- Frontmatter em docs canonicos (`title`, `scope`, `update_when`,
  `canonical_for`).
- Imports relativos dentro de `src/`; absolute `from src.X import Y`.
- `pytest tests/unit/` e `tests/integration/` espelham `src/`.
- Tipos: `from __future__ import annotations` no topo; `| None` em
  vez de `Optional[T]`.

### 4.3. Sem precedente claro

- Pratica mainstream Python (PEP 8, stdlib idiomatico).
- `@dataclass(frozen=True)` para records imutaveis.
- ABC + `@abstractmethod` para protocolos de servico.
- Tipo `Protocol` so se ABC seria forced inheritance desnecessaria.

### 4.4. Em duvida persistente

**Opcao conservadora**: menos extracao, mais reutilizacao do codigo
existente. Refactor minimo. Registrar dilema em log.

### 4.5. Liberdade para melhorar

Se ver solucao melhor que **realmente compensa** (nao "parece elegante",
mas tem ganho concreto: menos LOC, melhor testabilidade objetiva,
deteccao de bug que o plano nao previu), **tome**. Mas:

- **Registre** em log explicando o ganho concreto sobre alternativa do
  plano.
- **Nao mude API publica** definida em ADRs sem registrar como
  "DECISAO ARQUITETURAL" (Marcelo pode reverter no retorno).

Liberdade nao e licenca para reescrever ad hoc. E reconhecimento de
que o plano foi escrito antes do codigo ser tocado e pode ser ingenuo
em detalhes.

### 4.6. Hierarquia em conflito refactor-vs-fix

Conflito tipico: **byte-identical regression vs limpeza estrutural**.

- Stages 20, 21, 22 sao **refactors**: preservar bit-for-bit. Exceto
  Stage 20.1 onde `decision_idx` materializa Opcao (a) — output muda
  em `timestamp_utc` e `target_timestamp_utc` mas e fix intencional.
- Stage 23 e **fix + closure**: comportamento muda, gates de quality
  ate aqui falhos passam.

Na duvida sobre se uma task e refactor ou fix: **trate como refactor**
e prepare-se para reverter se Marcelo discordar no retorno.

---

## 5. Higiene de dados (smoke regression)

### 5.1 Wipe procedure

Antes de cada smoke run:

```bash
# Verifica que archive existe (NAO TOCAR)
test -d data/analytics_archive_pre_phase_b/ || {
  echo "ALERTA: archive ausente. Investigar antes de prosseguir."
  exit 1
}

# Conta tamanho do archive para verificacao posterior
du -sh data/analytics_archive_pre_phase_b/ > /tmp/archive_size_pre.txt

# Wipe test data
rm -rf data/analytics/silver/* 2>/dev/null || echo "wipe silver: partial or failed"
rm -rf data/analytics/gold/* 2>/dev/null || echo "wipe gold: partial or failed"

# Verifica archive NAO foi tocado
du -sh data/analytics_archive_pre_phase_b/ > /tmp/archive_size_post.txt
diff /tmp/archive_size_pre.txt /tmp/archive_size_post.txt || {
  echo "ERRO CRITICO: archive foi modificado. ABORTAR."
  # Registrar em log, abrir PR [BLOCKED]
  exit 1
}
```

### 5.2 Se wipe falhar (perms, files in use)

NAO insistir com `sudo` ou `chmod`. Em vez disso:

1. Registrar em log: "wipe silver/gold failed: <reason>; usando filter approach"
2. Gerar `NEW_PARENT_SWEEP_ID` unico para esta rodada de smoke:
   ```bash
   NEW_PARENT_SWEEP_ID="stage_${STAGE_NUMBER}_smoke_$(date +%Y%m%d_%H%M%S)"
   ```
3. Passar para os pipelines:
   ```bash
   .venv/bin/python -m src.main_train_tft \
     --parent-sweep-id "$NEW_PARENT_SWEEP_ID" \
     ... outros args
   .venv/bin/python -m src.main_baselines_test_pipeline \
     --config-json config/sweeps/explicit/phase_a_smoke_20260518_v2.json \
     --output-subdir-suffix "_${NEW_PARENT_SWEEP_ID}" \
     ... 
   .venv/bin/python -m src.main_refresh_analytics_store \
     --scope-mode cohort_decision \
     --scope-sweep-prefixes "$NEW_PARENT_SWEEP_ID" \
     ...
   ```
4. Validar quality checks filtrando por NEW_PARENT_SWEEP_ID; nao misturar com runs falhos anteriores

### 5.3 NUNCA tocar archive

```
data/analytics_archive_pre_phase_b/   ← INTOCAVEL durante TODA execucao Stage 20-23
```

Verificacao final apos cada smoke:
```bash
du -sh data/analytics_archive_pre_phase_b/
# Comparar com tamanho registrado em log no inicio do Stage
```

Se diff: **ABORT** + PR [BLOCKED].

---

## 6. Error handling

### 6.1 Test failures (`pytest`)

Procedimento:

1. **Ler erro completo** (`pytest tests/unit/path/test_X.py::test_Y -xvs`)
2. **Diagnosticar**:
   - Test estava passando antes do meu commit? → Regressao. Fix codigo.
   - Test novo escrito por mim? → Pode ser test mal escrito. Releia spec.
   - Test em arquivo nao tocado por mim? → Investigar; flag em log.
3. **Fix minimo**. Nunca:
   - Hackear assertion para passar (`assert x == x` etc.)
   - Comentar ou `@pytest.mark.skip` sem registrar em log como
     "DECISAO: skip <test> porque <razao concreta>"
   - Mudar test para passar com codigo errado
4. **Re-rodar**. Se passa: prosseguir.
5. **Falha persiste**: ate **3 tentativas no mesmo erro**. Cada
   tentativa = approach diferente, nao mesma mudanca de novo.
6. **Apos 3 fails**:
   - Registrar em log: "BLOCKED on test <X> at Stage X.Y after 3 attempts"
   - Commitar estado atual com mensagem `[WIP-BLOCKED]`
   - Pular para proxima task **se nao dependente**; se dependente:
     pular Stage inteiro, ir para proximo
   - PR final: marcar `[BLOCKED]` no titulo

### 6.2 Smoke failures

Procedimento:

1. Confirmar wipe ou filter foi aplicado (§5)
2. `.venv/bin/pytest tests/ -q` green primeiro (descarta bug obvio)
3. Comparar output esperado vs obtido:
   - Para byte-identical (Stages 21, 22): comparar parquets por coluna
     com pandas `assert_frame_equal` ou script de diff
   - Para gate PASS (Stages 20, 23): ler `validate_analytics_quality`
     output, identificar check falho
4. Diff em coluna nova introduzida pelo Stage: esperado, registrar
5. Diff em coluna preservada: regressao real, investigar
6. Apos 3 tentativas de debug em sessao: abort com PR [BLOCKED]

### 6.3 Schema conflicts

Stages 20-22 NAO devem alterar schema das tabelas existentes em
`analytics_store_schema.py`, **EXCETO**:

- **Stage 20**: adicionar coluna `decision_idx: int64` em
  `FACT_OOS_PREDICTIONS_SCHEMA` (ja previsto em ADR-0003 e na
  investigacao Gap 6). Tambem adicionar em
  `FACT_INFERENCE_PREDICTIONS_SCHEMA` para consistencia (C1 da
  auditoria do prompt de outra sessao).

Qualquer outra mudanca de schema = abortar e registrar.

### 6.4 Sem progresso real

Se uma sessao Claude inteira (4h ativas) gerou **0 commits novos**, isso
e sinal de bloqueio nao-reconhecido. Procedimento:

1. Registrar em log: "Sessao improdutiva, status atual: <descricao>"
2. Commitar qualquer trabalho parcial com `[WIP]` na mensagem
3. Marcar PR como `[BLOCKED]` se houver
4. Parar e aguardar Marcelo

---

## 7. Logging

### 7.1 Arquivo

```
docs/07_reports/option_b_execution_log_2026-05-19.md
```

Criar este arquivo no **primeiro commit do Stage 20.0** com frontmatter:

```markdown
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
```

### 7.2 Formato de entry

```markdown
## [YYYY-MM-DD HH:MM UTC] Stage X.Y — <kind>

<kind ∈ {decision | error | completion | abort | observation}>

**Context:** o que estava sendo feito; arquivo(s) tocado(s)
**Question/Issue:** o que ficou ambiguo OU erro encontrado
**Options:** alternativas consideradas (se decisao)
**Choice:** o que foi decidido / como foi fixado
**Principle:** qual rubrica orientou (§4.X)
**Outcome:** commit hash / test resultado / proximo passo

---
```

### 7.3 Regras

- **Append-only.** Nao edite entries passadas. Erros em entries antigas
  se corrigem com NOVA entry "(correcao da entry de YYYY-MM-DD HH:MM)".
- **Timestamp em UTC.** Use `date -u +"%Y-%m-%d %H:%M"`.
- **Cite principio sempre** (§4.1 ate §4.5). Se nenhum aplicou: "ad hoc".
- **Sumario por Stage**: ao fechar cada Stage (apos PR mergeada),
  adicionar entry "Stage X completed" com resumo (tasks, decisoes
  totais, problemas).

### 7.4 Exemplo de entry

```markdown
## [2026-05-20 14:32 UTC] Stage 20.1 — decision

**Context:** Criando MultiHorizonPredictionPersister em
src/domain/services/multi_horizon_prediction_persister.py.
**Question/Issue:** Persister.build_record() deve retornar dict (sem tipo)
ou dataclass PredictionRecord?
**Options:**
  (a) dict[str, Any] — match com row append atual em train_tft (rows.append({...}))
  (b) PredictionRecord dataclass frozen + .to_dict() — mais type-safe
**Choice:** (b) PredictionRecord dataclass + .to_dict()
**Principle:** §4.1 "Match precedente do projeto" — QuantileGuardrailService
  retorna QuantileGuardrailResult dataclass, nao dict. Consistencia.
**Outcome:** commit ab12cd3. Tests passam.

---
```

---

## 8. Abort conditions (hard stop)

Em qualquer das condicoes abaixo, **PARE** execucao, abra PR atual com
tag `[BLOCKED: <razao>]` no titulo, registre em log com kind=abort,
e aguarde Marcelo.

| # | Condicao | Acao adicional |
|---|---|---|
| 1 | `pytest tests/` red apos 3 tentativas em mesma task | PR `[BLOCKED: test_X.Y]` |
| 2 | Smoke byte-identical falha + debug nao resolve em sessao | PR `[BLOCKED: smoke_diff]` |
| 3 | Schema change necessaria fora do escopo (nao `decision_idx`) | PR `[BLOCKED: schema]`, doc o que precisaria |
| 4 | CI red em main apos merge de Stage anterior | Investigar antes de iniciar proximo Stage |
| 5 | `data/analytics_archive_pre_phase_b/` foi tocado | PR `[BLOCKED: archive_corruption]`, ALTA PRIORIDADE |
| 6 | Working dir diverge inesperadamente (untracked desconhecidos, stash orfao) | Investigar; nao limpar sem entender origem |
| 7 | Sessao Claude inteira (4h) sem 1 commit novo | PR `[WIP]`, registrar, parar |
| 8 | Time budget (definido pelo Marcelo) excedido | PR `[BLOCKED: time]` em qualquer task em curso |

**EM ABORT:**
- NAO `git push --force`, NAO `git rebase`, NAO `git reset --hard`
- Open PR mesmo incompleta com tag `[BLOCKED]`
- Branch WIP fica preservada
- Marcelo decide rollback ou continuacao no retorno

---

# Stage 20 — MultiHorizonPredictionPersister extraction

**Branch:** `feat/stage-20-multi-horizon-prediction-persister`
**Base:** `main`
**Esforco esperado:** 5-10h Claude ativo
**Bloqueia:** Stage 23 (fix Gap 6 depende de Persister)

## Objetivo

Extrair logica de persistencia de `fact_oos_predictions` multi-horizonte
hoje duplicada e divergente em:

- `src/use_cases/train_tft_model_use_case.py:778-810` (TFT trainer)
- `src/use_cases/run_baselines_use_case.py:225-280` (baseline runner)

para um domain service unico que materializa a convencao canonica
(ADR-0003 §Decision, Opcao (a)):

- `timestamp_utc = decision_day = dataset_timestamps[decision_idx]`
- `target_timestamp_utc = dataset_timestamps[decision_idx + h]`
- `y_true = dataset.target_return[decision_idx + h - 1]`
- `decision_idx: int64` nova coluna em `fact_oos_predictions`

## Pre-condicoes especificas

- `pytest tests/` green em main
- Branch criado de main atualizado:
  ```bash
  git checkout main && git pull
  git checkout -b feat/stage-20-multi-horizon-prediction-persister
  ```

## Tasks

### Task 20.0 — Verificar ADR-0003 e criar log

**Objetivo:** confirmar ADR ainda fixa Opcao (a); criar arquivo de log.

```bash
grep "Opcao (a)\|Option (a)" docs/01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md
# Esperado: ver "Opcao (a)" como decisao explicita
```

Criar `docs/07_reports/option_b_execution_log_2026-05-19.md` com
frontmatter (§7.1) e primeira entry:

```markdown
## [TIMESTAMP UTC] Stage 20.0 — completion

**Context:** Inicio de Stage 20. Verificacao de pre-condicoes.
**Outcome:** ADR-0003 confirma Opcao (a). pytest tests/ green. Tag
  checkpoint-pre-option-b verificada. Archive intacto.
**Next:** Stage 20.1 — criar Persister.

---
```

**Commit:** `chore(log): inicializar option_b_execution_log + verificar pre-condicoes (Stage 20.0)`

**Aceite:** arquivo de log criado, entry inicial registrada, ADR confirmado.

### Task 20.1 — Criar MultiHorizonPredictionPersister + tests + schema

**Objetivo:** criar domain service + atualizar schema.

**Arquivos:**
- `src/domain/services/multi_horizon_prediction_persister.py` (novo)
- `tests/unit/domain/services/test_multi_horizon_prediction_persister.py` (novo)
- `src/infrastructure/schemas/analytics_store_schema.py` (modificar:
  adicionar `decision_idx: int64` em `FACT_OOS_PREDICTIONS_SCHEMA` e
  `FACT_INFERENCE_PREDICTIONS_SCHEMA`)
- `tests/unit/infrastructure/schemas/test_analytics_store_schema.py`
  (atualizar para refletir nova coluna)

**Padrao a seguir:** `src/domain/services/quantile_guardrail_service.py`
(60 LOC, `@dataclass(frozen=True)` + `staticmethod`).

**Estrutura esperada:**

```python
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence
import pandas as pd


class IncompletePredictionWindowError(ValueError):
    """Raised when decision_idx + h exceeds dataset bounds."""


@dataclass(frozen=True)
class RunContext:
    schema_version: int
    run_id: str
    model_version: str
    asset: str
    feature_set_name: str
    config_signature: str
    split: str
    fold: str
    seed: int


@dataclass(frozen=True)
class PredictionRecord:
    schema_version: int
    run_id: str
    model_version: str
    asset: str
    feature_set_name: str
    config_signature: str
    split: str
    fold: str
    seed: int
    horizon: int
    decision_idx: int
    timestamp_utc: str
    target_timestamp_utc: str
    y_true: float
    y_pred: float
    error: float
    abs_error: float
    sq_error: float
    quantile_p10: float | None
    quantile_p50: float | None
    quantile_p90: float | None
    quantile_p10_post_guardrail: float | None
    quantile_p50_post_guardrail: float | None
    quantile_p90_post_guardrail: float | None
    quantile_guardrail_applied: int
    year: int

    def to_dict(self) -> dict:
        return asdict(self)


class MultiHorizonPredictionPersister:
    """
    Centralizes convention for fact_oos_predictions rows.

    Per ADR-0003 (Opcao (a)):
    - timestamp_utc = dataset_timestamps[decision_idx]
    - target_timestamp_utc = dataset_timestamps[decision_idx + h]
    - y_true = dataset.target_return[decision_idx + h - 1]
      (because target_return[t] = log(close[t+1]/close[t]) per
       build_tft_dataset_use_case.py:563)

    Raises IncompletePredictionWindowError if decision_idx + h
    exceeds dataset bounds (skip-rule per AGENT_CORE).
    """

    @staticmethod
    def build_record(
        *,
        decision_idx: int,
        h: int,
        y_true: float,
        y_pred: float,
        q10: float | None,
        q50: float | None,
        q90: float | None,
        q10_post: float | None,
        q50_post: float | None,
        q90_post: float | None,
        guardrail_applied: bool,
        dataset_timestamps: Sequence[pd.Timestamp],
        run_context: RunContext,
    ) -> PredictionRecord:
        if h < 1:
            raise ValueError(f"h must be >= 1, got {h}")
        if decision_idx < 0:
            raise ValueError(f"decision_idx must be >= 0, got {decision_idx}")
        target_pos = decision_idx + h
        if target_pos >= len(dataset_timestamps):
            raise IncompletePredictionWindowError(
                f"decision_idx={decision_idx} + h={h} = {target_pos} "
                f">= len(timestamps)={len(dataset_timestamps)}"
            )
        ts = pd.Timestamp(dataset_timestamps[decision_idx])
        target_ts = pd.Timestamp(dataset_timestamps[target_pos])
        err = float(y_pred - y_true)
        return PredictionRecord(
            schema_version=run_context.schema_version,
            run_id=run_context.run_id,
            model_version=run_context.model_version,
            asset=run_context.asset,
            feature_set_name=run_context.feature_set_name,
            config_signature=run_context.config_signature,
            split=run_context.split,
            fold=run_context.fold,
            seed=run_context.seed,
            horizon=int(h),
            decision_idx=int(decision_idx),
            timestamp_utc=ts.isoformat(),
            target_timestamp_utc=target_ts.isoformat(),
            y_true=float(y_true),
            y_pred=float(y_pred),
            error=err,
            abs_error=abs(err),
            sq_error=err * err,
            quantile_p10=q10,
            quantile_p50=q50,
            quantile_p90=q90,
            quantile_p10_post_guardrail=q10_post,
            quantile_p50_post_guardrail=q50_post,
            quantile_p90_post_guardrail=q90_post,
            quantile_guardrail_applied=1 if guardrail_applied else 0,
            year=int(ts.year),
        )
```

**Tests obrigatorios:**

- `test_build_record_h1_returns_next_day_target_ts` — h=1, decision_idx=10 → target_ts == timestamps[11]
- `test_build_record_h7_returns_7_ahead` — h=7, decision_idx=10 → target_ts == timestamps[17]
- `test_y_true_uses_decision_idx_plus_h_minus_1` — explicit check that callsite passes target_return correctly
- `test_boundary_raises_incomplete_window` — decision_idx + h >= len(timestamps) → raises
- `test_trading_day_arithmetic_not_calendar` — fixture com weekend gaps; verificar target_ts pula corretamente
- `test_run_context_propagated` — todos os campos de RunContext aparecem no record
- `test_to_dict_matches_schema_columns` — record.to_dict().keys() == set de colunas de FACT_OOS_PREDICTIONS_SCHEMA

**Decision points possiveis:**

- Q: PredictionRecord ou dict puro? → A: PredictionRecord (§4.1, match
  QuantileGuardrailService).
- Q: `dataset_timestamps` como `Sequence` ou `np.ndarray`? → A:
  `Sequence[pd.Timestamp]` (mais flexivel; np.ndarray e Sequence).
- Q: Onde calcular `error`/`abs_error`/`sq_error`? → A: dentro de
  build_record (consolida; evita duplicacao em callsites).
- Q: `decision_idx` deve ser keyword-only? → A: sim (`*` no signature).
  Match com signature pattern de QuantileGuardrailService.

**Commit:** `feat(domain): introduce MultiHorizonPredictionPersister domain service (Stage 20.1)`

**Aceite:**
- Arquivo existe, 100-200 LOC.
- Tests passam (`.venv/bin/pytest tests/unit/domain/services/test_multi_horizon_prediction_persister.py -v`).
- Schema `FACT_OOS_PREDICTIONS_SCHEMA` tem `decision_idx: int64` em columns + required_columns.
- Test do schema atualizado verifica nova coluna.

### Task 20.2 — Migrar train_tft_model_use_case

**Objetivo:** substituir linhas 770-810 por chamada ao Persister.

**Arquivos:**
- `src/use_cases/train_tft_model_use_case.py` (modificar)
- `tests/unit/use_cases/test_train_tft_model_use_case.py` (adaptar)

**O que mudar:**

Atual (linhas 770-810, abreviado):

```python
n = min(len(y_true_m), len(y_pred_m), ..., len(split_df))
split_tail = split_df.tail(n).reset_index(drop=True)  # BUG: decoder_end
for i in range(n):
    ts = pd.to_datetime(split_tail.loc[i, "timestamp"], utc=True)
    for h_idx, h in enumerate(horizons):
        y_true = float(y_true_m[i][h_idx])
        y_pred = float(y_pred_m[i][h_idx])
        ...
        target_ts = ts + pd.Timedelta(days=max(int(h) - 1, 0))  # BUG: calendar
        err = float(y_pred - y_true)
        rows.append({
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            ...
        })
```

Novo (com Persister):

```python
n = min(len(y_true_m), len(y_pred_m), ..., len(split_df))
# decision_idx maps sample i to time_idx in original dataset
# For pytorch_forecasting: sample i corresponds to encoder_end at
# offset (max_encoder_length - 1) + i in split_df
# Decision day = encoder_end = split_df.iloc[max_encoder_length - 1 + i]
decision_start_offset = max_encoder_length - 1
dataset_timestamps = pd.to_datetime(split_df["timestamp"], utc=True).tolist()

run_context = RunContext(
    schema_version=ANALYTICS_SCHEMA_VERSION,
    run_id=run_id,
    model_version=str(model_version),
    asset=str(asset_id),
    feature_set_name=str(feature_set_name),
    config_signature=str(config_signature),
    split=str(split_name),
    fold=str(fold_name) if fold_name is not None else "none",
    seed=int(seed) if seed is not None else 0,
)

for i in range(n):
    decision_idx = decision_start_offset + i
    for h_idx, h in enumerate(horizons):
        if h_idx >= len(y_true_m[i]) or h_idx >= len(y_pred_m[i]):
            continue
        try:
            record = MultiHorizonPredictionPersister.build_record(
                decision_idx=decision_idx,
                h=int(h),
                y_true=float(y_true_m[i][h_idx]),
                y_pred=float(y_pred_m[i][h_idx]),
                q10=float(q10_m[i][h_idx]) if q10_m[i][h_idx] is not None else None,
                # ... idem q50, q90
                q10_post=...,  # post-guardrail values
                q50_post=...,
                q90_post=...,
                guardrail_applied=...,
                dataset_timestamps=dataset_timestamps,
                run_context=run_context,
            )
        except IncompletePredictionWindowError:
            continue  # boundary case, skip
        rows.append(record.to_dict())
```

**Decision points:**

- **Q: como mapear sample i para decision_idx?**
  - Verificar: pytorch_forecasting TimeSeriesDataSet com
    `max_encoder_length` + `max_prediction_length` gera samples onde
    sample 0 tem encoder_end = primeiro valid encoder_end no split.
  - **Action:** ler `pytorch_forecasting_tft_trainer.py` para entender
    a sample iteration. Se nao for `decision_start_offset = max_encoder_length - 1`,
    ajustar e registrar em log.
  - **Validacao empirica:** rodar smoke pequeno (1 epoch, dataset
    sintetico 50 dias) e verificar que `timestamp_utc` produzido para
    sample 0, h=1 corresponde ao esperado.
- **Q: split_df pode nao ter coluna "timestamp"?**
  - Se nao: usar coluna correta do dataset (provavelmente "timestamp"
    ou "ds" ou "date" — verificar `build_tft_dataset_use_case.py`).
- **Q: max_encoder_length disponivel no escopo?**
  - Verificar variavel ou recuperar de config.

**Commit:** `refactor(train-tft): use MultiHorizonPredictionPersister for fact_oos_predictions (Stage 20.2)`

**Aceite:**
- Linhas 770-810 reduzidas a ~30 LOC (call ao Persister).
- `pytest tests/unit/use_cases/test_train_tft_model_use_case.py -v` green.
- Schema produzido tem coluna `decision_idx`.
- **NOTA:** Testes que validavam `timestamp_utc == decoder_end` agora
  vao FALHAR — fix do Gap 6 embutido. Atualizar esses testes para
  esperar `decision_day`. Registrar em log.

### Task 20.3 — Migrar run_baselines_use_case

**Objetivo:** substituir linhas 225-280 por chamada ao Persister.

**Arquivos:**
- `src/use_cases/run_baselines_use_case.py` (modificar)
- `tests/unit/use_cases/test_run_baselines_use_case.py` (adaptar)

**O que mudar:**

Atual (linhas 225-280, abreviado):

```python
for i in range(len(target_returns)):
    decision_ts = timestamps[i]
    history = target_returns[:i]
    prediction = self._compute_prediction(...)
    if prediction is None:
        skipped_warmup += 1
        continue
    y_pred, q10, q50, q90 = prediction
    for h in horizons:
        h_int = int(h)
        if h_int < 1:
            continue
        target_ts = decision_ts + pd.Timedelta(days=max(h_int - 1, 0))  # BUG
        if target_ts <= decision_ts and h_int > 1:
            continue
        # WRONG COMMENT removed
        y_true_idx = i + h_int - 1  # WRONG
        if y_true_idx >= len(target_returns):
            continue
        y_true_value = target_returns[y_true_idx]
        ...
        rows.append({...})
```

Novo:

```python
run_context = RunContext(
    schema_version=ANALYTICS_SCHEMA_VERSION,
    run_id=run_id,
    model_version=f"baseline_{baseline_name}",
    asset=str(asset),
    feature_set_name="baseline",
    config_signature=str(config_signature),
    split=str(split_name),
    fold="none",
    seed=0,
)

dataset_timestamps_list = [pd.Timestamp(t) for t in timestamps]

for i in range(len(target_returns)):
    history = target_returns[:i]
    prediction = self._compute_prediction(...)
    if prediction is None:
        skipped_warmup += 1
        continue
    y_pred, q10, q50, q90 = prediction
    for h in horizons:
        h_int = int(h)
        if h_int < 1:
            continue
        # y_true index per ADR-0003: target_return[decision_idx + h - 1]
        y_true_idx = i + h_int - 1
        if y_true_idx >= len(target_returns):
            continue
        y_true_value = target_returns[y_true_idx]
        if not np.isfinite(y_true_value):
            continue
        guardrail = QuantileGuardrailService.enforce_monotonic_triplet(q10, q50, q90)
        try:
            record = MultiHorizonPredictionPersister.build_record(
                decision_idx=i,
                h=h_int,
                y_true=float(y_true_value),
                y_pred=float(y_pred),
                q10=q10, q50=q50, q90=q90,
                q10_post=guardrail.p10_post,
                q50_post=guardrail.p50_post,
                q90_post=guardrail.p90_post,
                guardrail_applied=guardrail.applied,
                dataset_timestamps=dataset_timestamps_list,
                run_context=run_context,
            )
        except IncompletePredictionWindowError:
            continue
        rows.append(record.to_dict())
```

**Commit:** `refactor(baselines): use MultiHorizonPredictionPersister for fact_oos_predictions (Stage 20.3)`

**Aceite:**
- Linhas 225-280 reduzidas a ~30 LOC.
- Comentario enganoso 244-248 removido.
- `pytest tests/unit/use_cases/test_run_baselines_use_case.py -v` green
  (com testes adaptados para nova convencao timestamp_utc).

### Task 20.4 — Cross-pipeline regression test

**Objetivo:** teste que pareia decision_idx entre TFT e baselines e
verifica y_true igual.

**Arquivo:**
- `tests/integration/test_tft_baselines_y_true_alignment.py` (novo)

**Conteudo:**

```python
"""Regression guard: TFT and baseline pipelines produce same y_true for
same (decision_idx, h). Closes Gap 6 mechanically.
"""
import pandas as pd
import numpy as np
from src.domain.services.multi_horizon_prediction_persister import (
    MultiHorizonPredictionPersister, RunContext,
)


def test_tft_baseline_y_true_aligned_by_decision_idx():
    """For same decision_idx and h, both pipelines should reference same target_return."""
    # Synthetic timestamps + target_returns
    n = 50
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    target_returns = np.random.RandomState(42).normal(0, 0.01, n).tolist()

    rc = RunContext(
        schema_version=2,
        run_id="test",
        model_version="m",
        asset="AAPL",
        feature_set_name="f",
        config_signature="s",
        split="test",
        fold="none",
        seed=42,
    )

    # Pretend TFT and baseline both predict for decision_idx=10, h=7
    decision_idx = 10
    h = 7

    # Both should use target_returns[decision_idx + h - 1] = target_returns[16]
    expected_y_true = target_returns[decision_idx + h - 1]

    tft_record = MultiHorizonPredictionPersister.build_record(
        decision_idx=decision_idx, h=h,
        y_true=expected_y_true, y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=timestamps, run_context=rc,
    )

    baseline_record = MultiHorizonPredictionPersister.build_record(
        decision_idx=decision_idx, h=h,
        y_true=expected_y_true, y_pred=0.001,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=timestamps, run_context=rc,
    )

    # Both records reference SAME target_timestamp_utc and same y_true
    assert tft_record.target_timestamp_utc == baseline_record.target_timestamp_utc
    assert tft_record.timestamp_utc == baseline_record.timestamp_utc
    assert tft_record.y_true == baseline_record.y_true == expected_y_true
    assert tft_record.decision_idx == baseline_record.decision_idx == decision_idx
```

(Adicionar mais testes: range de decisions, varios horizons, boundary.)

**Commit:** `test(integration): cross-pipeline y_true alignment guard (Stage 20.4)`

**Aceite:** novo test green.

### Task 20.5 — Atualizar MULTI_HORIZON.md

**Objetivo:** doc canonico de convencao alinhado com ADR-0003.

**Arquivo:** `docs/03_modeling/MULTI_HORIZON.md`

**Mudancas:**

1. Nova secao §"Convencao canonica de target_timestamp e y_true"
   apos §"Conceitos-chave":

```markdown
## Convencao canonica de target_timestamp e y_true

Fixada em [ADR-0003](../01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md) Opcao (a):

- **Anchor:** `timestamp_utc = decision_day` (ultimo dia do encoder).
- **target_timestamp:** `target_timestamp_utc = dataset_timestamps[decision_idx + h]`,
  indexado em **trading days** (consulta no dataset, nao `Timedelta`).
- **y_true:** `dataset.target_return[decision_idx + h - 1]`. Convencao
  financeira: `h=1` significa "return realizado no dia seguinte a
  decisao", consistente com literatura.
- **decision_idx:** coluna `int64` em `fact_oos_predictions` que mapeia
  diretamente para `time_idx` do dataset. Materializa o invariante no
  schema (nao apenas no codigo).

Persistencia: ambos os pipelines (TFT trainer e baseline runner) usam
`MultiHorizonPredictionPersister.build_record()` para emitir registros.
Convencao deixa de ser implicita e distribuida; vira objeto explicito
do dominio.
```

2. Em §"Conceitos-chave", remover ambiguidade da linha 28-29 atual:

   Antes:
   ```
   - target_timestamp: timestamp do periodo alvo previsto. Chave de
     alinhamento em comparacoes pareadas; calculado por horizonte.
   ```
   Depois:
   ```
   - target_timestamp: dataset_timestamps[decision_idx + h]. Ver §"Convencao canonica de target_timestamp e y_true".
   ```

3. Em §"Persistencia (contrato por linha de predicao)", adicionar
   linha sobre `decision_idx`:

```markdown
| `decision_idx` | `int64` | indice no dataset original (time_idx) do dia de decisao |
```

**Commit:** `docs(modeling): convencao canonica de target_timestamp per ADR-0003 (Stage 20.5)`

**Aceite:** doc atualizado; cross-link bidirecional com ADR-0003.

### Task 20.6 — Smoke regression

**Objetivo:** verificar que Stage 20 nao quebra smoke F.1 (exceto o
fix do Gap 6 que muda timestamp_utc/target_timestamp_utc intencionalmente).

**Procedimento:**

1. Wipe silver/gold (§5.1)
2. Rodar smoke:
   ```bash
   .venv/bin/python -m src.main_train_tft \
     --asset AAPL \
     --features "BASELINE_FEATURES,TECHNICAL_FEATURES,SENTIMENT_FEATURES,FUNDAMENTAL_FEATURES" \
     --max-epochs 5 --max-encoder-length 60 --max-prediction-length 7 \
     --evaluation-horizons "[1, 7]" \
     --parent-sweep-id "stage_20_smoke_$(date +%Y%m%d)" \
     --seed 20260517 --prediction-mode quantile
   .venv/bin/python -m src.main_baselines_test_pipeline \
     --asset AAPL --config-json config/sweeps/explicit/phase_a_smoke_20260518_v2.json \
     --overwrite-on-collision
   .venv/bin/python -m src.main_refresh_analytics_store \
     --scope-mode cohort_decision \
     --scope-sweep-prefixes "stage_20_smoke_" \
     --primary-quantile-contract post_guardrail \
     --fail-on-quality
   ```
3. Verificar:
   - `tft_baselines_timestamp_subset_alignment` PASS (esperava que falhasse pre-Stage 20; agora deve passar)
   - `oos_pairwise_target_alignment` PASS (Gap 6 fix embutido)
   - Demais checks: idem ou melhor que F.1 v2
4. Atualizar `docs/07_reports/smoke_confirmatory_2026-05-18.md`
   secao nova "Post-Stage 20 (YYYY-MM-DD)":
   - Listar checks PASS/FAIL
   - Diferenca vs F.1 v2

**Commit:** `test(smoke): F.1 regression post-Stage 20 (Stage 20.6)`

**Aceite:**
- Smoke roda sem erro fatal
- Alignment gates passam
- Relatorio atualizado

**Se smoke falhar:** §6.2.

### Task 20.7 — Open PR

**Commit final:** ja feitos.

**PR:**
```bash
git push -u origin feat/stage-20-multi-horizon-prediction-persister

gh pr create --base main \
  --title "Stage 20 — MultiHorizonPredictionPersister extraction" \
  --body "$(cat <<'EOF'
## Summary

Stage 20 extrai logica de persistencia multi-horizonte de
fact_oos_predictions para domain service unico, materializando a
convencao canonica (ADR-0003 Opcao (a)) e fixando o Gap 6 mecanicamente.

## Tasks

- 20.0 — Pre-conditions + log inicializado
- 20.1 — MultiHorizonPredictionPersister + decision_idx schema
- 20.2 — Migrar train_tft_model_use_case
- 20.3 — Migrar run_baselines_use_case
- 20.4 — Cross-pipeline regression test
- 20.5 — Atualizar MULTI_HORIZON.md
- 20.6 — Smoke regression: alignment gates PASS

## Test plan

- pytest tests/ green
- Cross-pipeline alignment test passes
- F.1 smoke alignment gates PASS

## Cross-link

- ADR-0003
- B_architectural_debt §M-train_tft persistencia
- tft_y_true_investigation_2026-05-18.md
- option_b_execution_log_2026-05-19.md (decisoes registradas)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Aguardar merge** antes de iniciar Stage 21. Verificar CI green em main.

**Aceite:** PR mergeada em main, main green.

---

# Stage 21 — QualityCheckRegistry

**Branch:** `feat/stage-21-quality-check-registry`
**Base:** `main` (apos Stage 20 mergeado)
**Esforco esperado:** 8-14h Claude ativo

## Objetivo

Extrair os ~20 quality checks hoje em `validate_analytics_quality_use_case.py`
para um registry de classes `QualityCheck` por categoria semantica.
Refactor PURO — comportamento bit-for-bit identico.

## Pre-condicoes

```bash
git checkout main && git pull
test ! -z "$(git log --oneline -10 | grep 'Stage 20')"
.venv/bin/pytest tests/ -q  # green
git checkout -b feat/stage-21-quality-check-registry
```

## Tasks

### Task 21.0 — Verificar ADR-0004 + log entry

```bash
grep "Accepted" docs/01_architecture/decisions/ADR-0004-quality-check-registry.md
```

Adicionar entry de log "Stage 21.0 — completion: pre-conditions OK".

**Commit:** `chore(log): Stage 21 start, pre-conditions verified (Stage 21.0)`

### Task 21.1 — Base + Registry skeleton

**Arquivos novos:**
- `src/domain/services/quality_checks/__init__.py`
- `src/domain/services/quality_checks/base.py`
- `tests/unit/domain/services/quality_checks/__init__.py`
- `tests/unit/domain/services/quality_checks/test_base.py`

**Conteudo de base.py:**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.domain.services.scope_spec import ScopeSpec


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class AnalyticsSnapshot:
    """Snapshot of analytics tables loaded for quality validation.

    Holds DataFrames keyed by table name. Schema implicito que cada
    check antes navegava ad-hoc agora vive aqui.
    """
    tables: dict[str, pd.DataFrame]
    scope_spec: ScopeSpec
    scoped_run_ids: set[str] | None = None

    def has(self, table: str) -> bool:
        return table in self.tables and not self.tables[table].empty

    def get(self, table: str) -> pd.DataFrame:
        return self.tables.get(table, pd.DataFrame())


class QualityCheck(ABC):
    name: str  # override em subclasses

    @abstractmethod
    def applies_when(self, scope: ScopeSpec) -> bool: ...

    @abstractmethod
    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult: ...


class QualityCheckRegistry:
    def __init__(self) -> None:
        self._checks: list[QualityCheck] = []

    def register(self, check: QualityCheck) -> None:
        self._checks.append(check)

    def applicable(self, scope: ScopeSpec) -> list[QualityCheck]:
        return [c for c in self._checks if c.applies_when(scope)]

    def __len__(self) -> int:
        return len(self._checks)
```

**Tests:**
- `test_registry_register_and_query`
- `test_applies_when_filter`
- `test_check_result_immutable`

**Commit:** `feat(domain): introduce QualityCheck base + Registry (Stage 21.1)`

### Task 21.2 — Migrar cluster cardinality

**Arquivo:** `src/domain/services/quality_checks/cardinality.py`

Migrar:
- `cardinality_config_fold_seed`
- `min_samples_by_split`
- `run_id_execution_consistency`
- `inference_predictions_continuity`
- `feature_contrib_local_continuity`
- `required_metrics_nan`

Cada um vira `class XCheck(QualityCheck)` com `applies_when` e `run`.

**Tests:** mover de `test_validate_analytics_quality_use_case.py` para
`tests/unit/domain/services/quality_checks/test_cardinality.py`.

**Verificacao:** outputs (passed, detail) devem ser **bit-for-bit
identicos** aos de antes para o mesmo input.

**Commit:** `refactor(analytics-quality): migrate cardinality checks to Registry (Stage 21.2)`

### Tasks 21.3, 21.4, 21.5 — Migrar clusters alignment, calibration, contracts

Analogo a 21.2. Um arquivo por categoria:
- `alignment.py` — tft_baselines_timestamp_subset_alignment, oos_pairwise_target_alignment
- `calibration.py` — gold_confidence_calibrated_by_horizon, dm_mcs_persisted_executable, gold_metrics_by_config_n_oos_contract
- `contracts.py` — official_contract_quantile_attention, baseline_present_per_candidate_sweep, referential_integrity

**Commits:**
- `refactor(analytics-quality): migrate alignment checks (Stage 21.3)`
- `refactor(analytics-quality): migrate calibration checks (Stage 21.4)`
- `refactor(analytics-quality): migrate contracts checks (Stage 21.5)`

### Task 21.6 — Orchestrator

**Arquivo:** `src/use_cases/validate_analytics_quality_use_case.py` (modificar)

Reduzir `execute()` para:

```python
def execute(self) -> AnalyticsQualityResult:
    scope_spec = self._resolve_scope_spec()
    snapshot = self._load_snapshot(scope_spec)
    results: list[dict] = []
    for check in self._registry.applicable(scope_spec):
        result = check.run(snapshot)
        results.append({
            "name": result.name,
            "passed": result.passed,
            "detail": result.detail,
        })
    return AnalyticsQualityResult(checks=results, ...)
```

Helper `_load_snapshot(scope_spec) -> AnalyticsSnapshot` carrega todos
os parquets necessarios. Registry default vem do
`quality_checks/__init__.py`.

**Commit:** `refactor(analytics-quality): execute() becomes registry orchestrator (Stage 21.6)`

**Aceite:** `validate_analytics_quality_use_case.py` reduz para ~200-300 LOC (de 1.152).

### Task 21.7 — ANALYTICS_STORE_ARCHITECTURE.md

Documentar registry como ponto de extensao oficial. Adicionar exemplo
"Como adicionar novo quality check" (criar classe, registrar).

**Commit:** `docs(architecture): document QualityCheckRegistry as extension point (Stage 21.7)`

### Task 21.8 — Smoke regression (bit-identical)

**Procedimento:**
1. Wipe silver/gold + rodar smoke pre-Stage 21 (de checkpoint
   `stage_20_smoke_`), salvar output.
2. Rodar smoke post-Stage 21 (com mesmo input).
3. Comparar `validate_analytics_quality` output: nomes + passed + detail
   de cada check **identicos**.

Script de diff:

```bash
python -c "
import json
pre = json.load(open('/tmp/quality_pre_21.json'))
post = json.load(open('/tmp/quality_post_21.json'))
assert pre['checks'] == post['checks'], 'Bit-identical regression FAIL'
print('Bit-identical PASS')
"
```

**Commit:** `test(smoke): Stage 21 bit-identical regression (Stage 21.8)`

### Task 21.9 — Open PR

Padrao Stage 20.7. Aguardar merge antes de Stage 22.

---

# Stage 22 — GoldBuilders modular

**Branch:** `feat/stage-22-gold-builders-modular`
**Base:** `main` (apos Stage 21 mergeado)
**Esforco esperado:** 12-20h Claude ativo (maior dos 3 refactors)

## Objetivo

Extrair 26 builders gold de `refresh_analytics_store_use_case.py`
(~2.568 LOC) para `src/domain/services/gold_builders/` organizado por
categoria. Refactor PURO — output byte-identical em todas as 10+ tabelas
gold.

## Pre-condicoes

```bash
git checkout main && git pull
test ! -z "$(git log --oneline -20 | grep 'Stage 21')"
.venv/bin/pytest tests/ -q
git checkout -b feat/stage-22-gold-builders-modular
```

## Tasks

### Task 22.0 — Verificar ADR-0005 + log

Padrao Stages 20.0/21.0.

### Task 22.1 — Base + skeleton

**Arquivos novos:**
- `src/domain/services/gold_builders/__init__.py`
- `src/domain/services/gold_builders/base.py`
- `src/domain/services/gold_builders/_metrics.py` (helpers: pinball,
  IQR, prob_up — antes em refresh)

**base.py:**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
from src.domain.services.scope_spec import ScopeSpec


@dataclass
class BuildContext:
    """Context for building gold tables.

    Holds scope spec, primary quantile contract, runtime params.
    """
    scope_spec: ScopeSpec
    primary_quantile_contract: str  # "raw" | "post_guardrail"
    # ... outros params necessarios


@dataclass
class AnalyticsSnapshot:
    """Silver tables + dim_run + dim_features loaded.

    Reuse from quality_checks if possible OR define here separately.
    """
    tables: dict[str, pd.DataFrame]


class GoldBuilder(ABC):
    output_table: str  # nome do parquet gerado
    requires: list[str]  # silver/dim tables required

    @abstractmethod
    def build(
        self,
        snapshot: AnalyticsSnapshot,
        ctx: BuildContext,
    ) -> pd.DataFrame: ...
```

### Tasks 22.2 — 22.6: Migrar categorias

Cada uma vira `.py` proprio:

- **22.2 — `ranking.py`:** `runs_long`, `ranking_by_config`, `consistency_topk`
- **22.3 — `pairwise.py`:** `dm_pairwise_results`, `mcs_results`, `paired_oos_intersection_by_horizon`
- **22.4 — `quantile.py`:** `quantile_guardrail_audit`, `quantile_degeneracy_report`, `prediction_metrics_by_run_split_horizon` (+ single_contract variant), `prediction_metrics_by_config`
- **22.5 — `descriptive.py`:** `ic95`, `feature_set_impact`, `model_decision_final`
- **22.6 — `confidence.py`:** `confidence_calibrated_by_horizon`, `metrics_by_config_n_oos_contract`

Para CADA categoria:
- Mover funcao `_build_gold_X` do refresh para classe
  `XGoldBuilder(GoldBuilder)` no novo arquivo
- Manter lógica IDENTICA (mesmas joins, mesmos groupbys, mesma ordem
  de colunas)
- Tests existentes em `test_refresh_analytics_store_use_case.py`
  espelham para `tests/unit/domain/services/gold_builders/test_<cat>.py`
- Verificar byte-identical (smoke) apos cada categoria

**Commits:** um por categoria, padrao
`refactor(refresh): migrate <category> gold builders (Stage 22.X)`

**Decision points possiveis:**

- Q: builders que dependem de outros (ex: ranking que usa runs_long)?
- A: ordem de execucao no orchestrator garante dependencia. Cada
  builder ainda e independente, mas `requires` declara dependencia de
  silver/dim, nao de outros gold.

- Q: e se um builder mistura categorias?
- A: pertencer a uma categoria primaria; documentar em log se
  ambiguidade real.

### Task 22.7 — Orchestrator

Reduzir `RefreshAnalyticsStoreUseCase` para orquestrador:

```python
def execute(self):
    snapshot = self._load_silver_dim_snapshot(self._scope_spec)
    ctx = BuildContext(...)
    for builder in self._gold_builders_registry.applicable(self._scope_spec):
        df = builder.build(snapshot, ctx)
        path = self._path_for(builder.output_table)
        self._safe_write(df, path)
    self._normalize_partitions()  # etc — pos-build housekeeping
```

`refresh_analytics_store_use_case.py` reduz de 2.568 para ~400 LOC.

**Commit:** `refactor(refresh): execute() becomes builders orchestrator (Stage 22.7)`

### Task 22.8 — Smoke regression byte-identical

**Procedimento:**

1. Wipe silver/gold.
2. Checkout main (pre-Stage 22).
3. Rodar smoke; copiar `data/analytics/gold/` para `/tmp/gold_pre_22/`.
4. Checkout branch Stage 22 (com todos os commits 22.0-22.7).
5. Wipe gold (silver pode permanecer se silver nao mudou).
6. Rodar refresh apenas (`main_refresh_analytics_store`).
7. Comparar `/tmp/gold_pre_22/*.parquet` com `data/analytics/gold/*.parquet`.

Script de diff:

```python
import pandas as pd
from pathlib import Path

GOLD_TABLES = [
    "gold_runs_long", "gold_ranking_by_config", "gold_consistency_topk",
    "gold_ic95", "gold_feature_set_impact",
    "gold_prediction_metrics_by_run_split_horizon",
    "gold_prediction_metrics_by_config",
    "gold_quantile_guardrail_audit", "gold_quantile_degeneracy_report",
    "gold_dm_pairwise_results", "gold_mcs_results",
    "gold_paired_oos_intersection_by_horizon",
    "gold_model_decision_final",
    "gold_confidence_calibrated_by_horizon",
    "gold_metrics_by_config_n_oos_contract",
]

pre = Path("/tmp/gold_pre_22")
post = Path("data/analytics/gold")

for tbl in GOLD_TABLES:
    pre_files = sorted(pre.rglob(f"{tbl}/**/*.parquet"))
    post_files = sorted(post.rglob(f"{tbl}/**/*.parquet"))
    if not pre_files and not post_files:
        continue
    df_pre = pd.concat([pd.read_parquet(f) for f in pre_files]).sort_index(axis=1)
    df_post = pd.concat([pd.read_parquet(f) for f in post_files]).sort_index(axis=1)
    # Sort by all columns for stable comparison
    df_pre = df_pre.sort_values(list(df_pre.columns)).reset_index(drop=True)
    df_post = df_post.sort_values(list(df_post.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(df_pre, df_post, check_exact=False, rtol=1e-9)
    print(f"{tbl}: OK")
```

Se falhar em alguma tabela: identificar diferenca de colunas/linhas;
debugar; corrigir builder; re-rodar smoke.

**Commit:** `test(smoke): Stage 22 byte-identical gold regression (Stage 22.8)`

### Task 22.9 — Open PR

Padrao. Aguardar merge antes de Stage 23.

---

# Stage 23 — F.1 PASS pleno (Gap 6 closure)

**Branch:** `feat/stage-23-f1-pass-pleno`
**Base:** `main` (apos Stage 22 mergeado)
**Esforco esperado:** 3-5h Claude ativo + tempo de smoke completo (1-2h)

## Objetivo

Rodar smoke F.1 completo (criterios oficiais) com Stages 20-22 em main.
Validar todos os 6 criterios PASS. Fechar F.1 + abrir Phase B
formalmente.

## Pre-condicoes

```bash
git checkout main && git pull
.venv/bin/pytest tests/ -q
# Tag final dos refactors
git tag -a stage-22-merged -m "Final state of Option B refactors"
git push origin stage-22-merged
git checkout -b feat/stage-23-f1-pass-pleno
```

## Tasks

### Task 23.0 — Pre-condicao + log

Verificar:
- `git log --oneline -30 main | grep "Stage 2[012]"` mostra 3 Stages mergeados
- `pytest tests/` green
- CI green em main (ver Actions ou equivalente do projeto)

### Task 23.1 — Full smoke F.1 v3

Rodar smoke conforme criterio oficial de F.1:

```bash
# Wipe silver/gold (§5.1)
rm -rf data/analytics/silver/* data/analytics/gold/*

# Config para F.1: max_epochs >= 5, n_rows >= 1000
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
  --scope-sweep-prefixes "phase_a_smoke_$(date +%Y%m%d)_v3" \
  --primary-quantile-contract post_guardrail \
  --fail-on-quality
```

### Task 23.2 — Validar 6 criterios F.1

Verificar conforme `PHASE_B_IMPLEMENTATION_CHECKLIST.md` §"Stage final F.1":

| # | Criterio | Como verificar |
|---|---|---|
| 1 | `% p10==p90 < 5%` | gate `block_quantile_degeneracy_gate` PASS |
| 2 | Zero violacoes probabilisticas | MPIW post_guardrail >= 0; crossing rate raw OK |
| 3 | Gold cohort-aware | todas as 10+ tabelas com `parent_sweep_id` |
| 4 | `pytest tests/` PASS | rodar local |
| 5 | CI verde em PRs mergeados | revisar Actions |
| 6 | Refresh + quality sem erro | `validate_analytics_quality` exit 0; **0 checks FAIL** |

Em 23.2, **TODOS os 6 devem PASS**. Se algum FAIL: §6.

### Task 23.3 — Atualizar smoke_confirmatory_2026-05-18.md

Adicionar secao final "F.1 v3 PASS pleno (post-Stage 22, YYYY-MM-DD)":

- Tabela 6 criterios todos PASS
- Comparativo metricas vs F.1 v2
- Confirmacao Gap 6 fechado (oos_pairwise_target_alignment PASS)

**Commit:** `docs(smoke): registrar F.1 v3 PASS pleno post-Stages 20-22 (Stage 23.3)`

### Task 23.4 — Marcar F.3 satisfeito

**Arquivos:**
- `docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md`:
  - F.1 task `- [ ]` → `- [x]` com nota "(satisfeito YYYY-MM-DD via smoke v3)"
  - F.3 task `- [~]` → `- [x]`
  - "Data de abertura da Fase B" preenchida
- `docs/07_reports/phase-gates/A_code_audit.md` §"Gate de saida da Fase A":
  - Todos os checkboxes restantes `- [x]`
  - Data de abertura da Fase B preenchida
- `docs/07_reports/phase-gates/A_audit_closure_2026-05-17.md`:
  - Status `pendente` → `fechado` se aplicavel
  - Nota final: "Fase B aberta YYYY-MM-DD"

**Commit:** `docs(checklist): marcar F.3 satisfeito + abertura Fase B (Stage 23.4)`

### Task 23.5 — Atualizar POST_CLOSURE_FIXES_CHECKLIST.md

- Tasks F.0.0 a F.0.9: marcar `[x]` (mergeadas via PR #40)
- Stage F.A: marcar `[x]` (mergeado nesta sessao via PR de F.A)
- Stages 20-22: marcar como mergeadas em PHASE_B (cross-link)
- Gap 6: marcar como **resolvido** via Stage 20 (fix embutido na
  convencao Opcao (a) do Persister)

**Commit:** `docs(checklist): fechar POST_CLOSURE F.0 + F.A + resolucao Gap 6 (Stage 23.5)`

### Task 23.6 — Open PR final

```bash
gh pr create --base main \
  --title "Stage 23 — F.1 PASS pleno + Fase B aberta" \
  --body "$(cat <<'EOF'
## Summary

Stage 23 fecha F.1 (smoke confirmatorio) pleno post-Stages 20-22.
Todos os 6 criterios oficiais PASS. Gap 6 fechado via Persister
(Stage 20). Fase B abre formalmente.

## Tasks

- 23.1 — Full smoke F.1 v3 executado
- 23.2 — 6 criterios PASS validados
- 23.3 — Relatorio atualizado
- 23.4 — F.3 marcado, Fase B aberta em A_code_audit
- 23.5 — POST_CLOSURE fechado (F.0, F.A, Gap 6 resolvido)

## Test plan

- F.1 v3: 6/6 PASS
- pytest tests/ green
- CI verde

## Cross-link

- B_architectural_debt_2026-05-19.md
- ADR-0003/0004/0005
- option_b_execution_log_2026-05-19.md
- smoke_confirmatory_2026-05-18.md §"F.1 v3 PASS pleno"

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Aceite:** PR mergeada; main reflete F.1 PASS pleno + Fase B aberta.

---

## 9. Validacao na volta do Marcelo

Checklist para Marcelo executar quando retornar:

```
[ ] git log --oneline main | head -40   # ver historia da execucao
[ ] git tag | grep -E "checkpoint-pre-option-b|stage-22-merged"
[ ] Ler docs/07_reports/option_b_execution_log_2026-05-19.md inteiro
[ ] Revisar PRs mergeadas: Stage 20, 21, 22, 23
[ ] pytest tests/ green
[ ] Smoke F.1 v3 6/6 PASS? Revisar smoke_confirmatory_2026-05-18.md
[ ] data/analytics_archive_pre_phase_b/ intacto:
    diff $(du -sh data/analytics_archive_pre_phase_b/) <inicial>
[ ] Decisoes registradas em log: validar cada uma (especialmente as
    com kind=decision e Principle="liberdade para melhorar")
[ ] Stage F.3 [x]? Data de abertura Fase B preenchida?
[ ] Reverter qualquer decisao do Claude que voce discordar:
    git revert <commit-hash>  (NUNCA reset --hard)
```

Se voce voltar e algum PR esta `[BLOCKED]`:
- Ler entry de log com `kind=abort`
- Decidir: continuar manualmente, reverter, ou abandonar

---

## 10. Rollback procedure

Se quiser voltar tudo:

```bash
# Soft rollback (preserve work, just go back to checkpoint)
git checkout main
git reset --soft checkpoint-pre-option-b   # WIP files vao para staging
git stash                                   # set aside
# Branches Stages 20-23 permanecem; podem ser auditadas

# Hard rollback (delete all Option B work)
# CUIDADO: irreversivel
git checkout main
git reset --hard checkpoint-pre-option-b
git push --force-with-lease origin main     # APENAS Marcelo decide isso
# Deletar branches stage-2X
for b in feat/stage-20-* feat/stage-21-* feat/stage-22-* feat/stage-23-*; do
    git branch -D "$b" 2>/dev/null
    git push origin --delete "$b" 2>/dev/null
done
```

**Claude NAO executa rollback hard autonomamente.** Apenas Marcelo no
retorno.

---

## 11. DO e DO NOT explicitos

### DO

- Ler ADRs 0003/0004/0005 ANTES de comecar cada Stage correspondente
- Match precedentes do projeto (QuantileGuardrailService, scope_spec)
- Registrar TODA decisao nao-obvia em log
- Verificar archive intacto apos cada smoke
- Smoke regression apos cada Stage antes de abrir PR
- Aguardar merge de Stage N antes de iniciar Stage N+1
- Fix-up commits via novos commits (jamais amend/rebase/force-push)
- Conservative em duvida; liberdade quando ganho concreto

### DO NOT

- NUNCA tocar `data/analytics_archive_pre_phase_b/`
- NUNCA `git reset --hard` em commits ja pushados
- NUNCA `git push --force` ou `--force-with-lease`
- NUNCA `git rebase` ou `git commit --amend`
- NUNCA squash de commits em PR mergeada (squash-merge OK via gh; mas
  nao reescrever historia)
- NUNCA hackear test para passar
- NUNCA `@pytest.mark.skip` sem registrar em log
- NUNCA inflar PR com refactor "while I'm here" alem do escopo do Stage
- NUNCA assumir comportamento sem ler codigo real do projeto
- NUNCA criar arquivo `.bak` ou commitar arquivos temporarios
- NUNCA mudar `MULTI_HORIZON.md` ou outro doc canonico sem registrar
  como "DECISAO ARQUITETURAL" em log

---

## 12. Sumario final

| Stage | Branch | PR title | Aceite |
|---|---|---|---|
| 20 | `feat/stage-20-multi-horizon-prediction-persister` | `Stage 20 — MultiHorizonPredictionPersister extraction` | Alignment gates PASS; tests green |
| 21 | `feat/stage-21-quality-check-registry` | `Stage 21 — QualityCheckRegistry` | Bit-identical regression PASS |
| 22 | `feat/stage-22-gold-builders-modular` | `Stage 22 — GoldBuilders modular` | Byte-identical 10 gold tables PASS |
| 23 | `feat/stage-23-f1-pass-pleno` | `Stage 23 — F.1 PASS pleno + Fase B aberta` | 6/6 criterios PASS; F.3 [x]; Fase B aberta |

**Comece executando §2 (pre-condicoes). Boa execucao.**
