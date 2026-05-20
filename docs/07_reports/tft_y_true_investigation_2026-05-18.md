---
title: TFT y_true Multi-Horizon Alignment Investigation
date: 2026-05-18
trigger: Stage F.0 closure amendment, "Gap 6 — TFT y_true convention bug"
status: investigation_complete
classification: blocker_for_F.1_PASS
canonical_for: [tft_y_true_convention, oos_alignment_bug, timestamp_utc_semantic]
---

# Investigação: convenção de y_true multi-horizonte TFT vs baselines

## TL;DR (veredicto)

**Hipótese vencedora: nova H8** — distinta de H1–H7 originais. **Ambos** os
pipelines (TFT e baseline runner) operam internamente de forma coerente, mas
usam **convenções incompatíveis** para o campo `timestamp_utc` de
`fact_oos_predictions`. A diferença é **exatamente 6 trading days** (=
`max_prediction_length - 1`), persistente em **todas** as 1244 linhas test+val
inspecionadas, em **todos** os horizontes.

Verificação empírica, alinhamento 100% determinístico:

| Pipeline | `y_true(h=1)` | `y_true(h=7)` |
|---|---|---|
| **TFT (BTSF)** | `target_return[ts_idx - 6]` | `target_return[ts_idx]` |
| **Baselines (3)** | `target_return[ts_idx]` | `target_return[ts_idx + 6]` |

`ts_idx` = `time_idx` em [dataset_tft_AAPL](data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet)
para o `timestamp_utc` gravado em [fact_oos_predictions](data/analytics/silver/fact_oos_predictions/).

Consequência operacional: para o **mesmo** `target_timestamp_utc`, TFT e
baseline reportam `y_true` referentes a **dias completamente diferentes do
dataset** — diferença máxima medida = `0.138` (escala típica de retornos = 0.01).
**0 de 685** linhas com `target_ts` coincidente em h=1 têm `y_true` igual.

O sintoma originalmente reportado ("Gap 6 — sobram 6 target_ts em TFT vs
baseline para h=7") é apenas a manifestação de borda do bug. **O escopo real é
todas as linhas, todos os horizontes**, não apenas as 6 da fronteira.

## Hipóteses originais — refutação ponto-a-ponto

| H | Verdict | Evidência |
|---|---|---|
| H1 (PF retorna y mal-construída) | ❌ | y do dataloader varia corretamente entre passos do decoder; `actuals[s, k]` ≠ `actuals[s, 0]` para k ≠ 0 (h=1=−0.010, h=7=+0.013 no mesmo sample). |
| H2 (extração `y_true_m[i][h_idx]` errada) | ❌ | `actuals_sel.tolist()` preserva shape `[N, 2]` (h=1, h=7); `y_true_m[i][h_idx]` extrai consistentemente o passo correto. |
| H3 (convenção do projeto difere) | ⚠ Parcial | [MULTI_HORIZON.md](docs/03_modeling/MULTI_HORIZON.md) define `target_timestamp` como "timestamp do período alvo previsto" e diz que é "calculado por horizonte", **mas não fixa a âncora** (decision_day, decoder_start, decoder_end). A ambiguidade do doc deixou o bug passar. |
| H4 (Stage X introduziu) | ✅ Parcial | Persistência de TFT está em `f555e0e1` (2026-04-01, antes de Stage 9–14). Baseline runner é `a8c8944` (Stage 12) tentou alinhar com TFT e errou a interpretação. Nem TFT nem baseline mudaram convenção de y_true em Stage F.0 — bug é pré-existente. |
| H5 (PF upgrade) | ❌ | Comportamento do `to_dataloader` é consistente com semântica documentada do pacote em 1.6.1. |
| H6 (TFT certo, baseline errado) | ⚠ | TFT é **internamente consistente** quanto a y_true (decoder out alinha com decoder time steps), mas usa `timestamp_utc = decoder_end_day` que **não corresponde** a "decision_ts" no sentido natural. Baseline interpreta `timestamp_utc = decision_day` — também internamente consistente — porém divergente do TFT. Nem TFT nem baseline está claramente "certo"; ambos diferem em convenção e ambos têm bugs adicionais (ver Apêndice B). |
| H7 (ambos errados, outra convenção) | ❌ na forma proposta | Não há uma "terceira convenção" (cumulative return, log-return acumulado) em jogo. |

**H8 vencedora (formulação proposta)**:

> O trainer TFT persiste `timestamp_utc = decoder_end_day` (último dia da janela
> de decoder) por ler `split_tail = split_df.tail(n)`, enquanto o baseline
> runner persiste `timestamp_utc = decision_day` (`timestamps[i]` da iteração
> direta). A diferença sistemática é `max_prediction_length − 1` trading days.
> O comentário em [run_baselines_use_case.py:231-248](src/use_cases/run_baselines_use_case.py#L231-L248)
> afirma alinhar-se à convenção TFT mas implementa convenção diferente.

## Evidência empírica

### E1 — Determinismo do offset (todas as 1244 linhas)

Para cada linha de `fact_oos_predictions` (TFT e baseline), procuramos o offset
`δ ∈ [-8, +8]` tal que `y_true == target_return[ts_idx + δ]`:

```
=== TFT mapping (BTSF run) ===
test  h=1: n=685  100% match em δ=-6
test  h=7: n=685  100% match em δ=0
val   h=1: n=437  100% match em δ=-6
val   h=7: n=437  100% match em δ=0

=== Baseline mapping (run 1934b90...) ===
test  h=1: n=685  100% match em δ=0
test  h=7: n=679  100% match em δ=+6
val   h=1: n=437  100% match em δ=0
val   h=7: n=437  100% match em δ=+6
```

Nenhum outro offset produz match. O sistema é estritamente determinístico e
gerou os mesmos δ para 3 baseline runs distintos (`naive_zero`,
`historical_mean_rolling`, `historical_quantiles_rolling`).

### E2 — TFT vs baseline com `target_ts` igual

Pareando `(target_timestamp_utc, horizon)` entre TFT-BTSF e
`baseline/run=1934b90...` no split test:

| h | overlap | y_true match (tol 1e-7) | max abs diff |
|---|---|---|---|
| 1 | 685/685 (100%) | **0/685** | 0.1379 |
| 7 | 679/685 (TFT-only=6, BL-only=0) | **0/679** | (não medido, esperado ≥ 0.1) |

Exemplo (test, h=1): `target_ts = 2023-04-10`
- TFT y_true = +0.015523 (= `target_return[ts_idx 3338 - 6]`)
- Baseline y_true = -0.007620 (= `target_return[ts_idx 3338]`)
- A diferença é por **shift de 6 trading days**, confirmando E1.

### E3 — Sample concreto com decoder_end mapping

```
Sample i=300 (test, h=7):
  timestamp_utc  = 2024-06-18, ts_idx = 3638
  target_ts      = 2024-06-24 (= ts + 6 calendar)
  y_true(TFT)    = -0.0217481
  target_return[3638] = -0.0217481  ✓  (decoder step 6 = decoder_end)
  target_return[3644] = -0.0163882  ✗
  target_return[3632] = +0.0701314  ✗

Mesmo sample, h=1:
  y_true(TFT)    = +0.0701314
  target_return[3638 - 6 = 3632] = +0.0701314  ✓  (decoder step 0)
```

Conclusão: o **vetor `actuals` do dataloader está correto** (passo k=0 ≠ passo
k=6). O bug é no mapeamento `(sample s) → timestamp_utc` em
[train_tft_model_use_case.py:778](src/use_cases/train_tft_model_use_case.py#L778)
(`split_tail = split_df.tail(n)`), que recupera o **último** dia da janela do
decoder (encoder_end + `max_prediction_length`), não o dia de decisão.

### E4 — Fórmula calendar-vs-trading no target_ts

[train_tft_model_use_case.py:796](src/use_cases/train_tft_model_use_case.py#L796):
```python
target_ts = ts + pd.Timedelta(days=max(int(h) - 1, 0))
```

Bug secundário: usa **calendar days** (`Timedelta(days=...)`), não trading
days, em um dataset trading-day. Para h=7 em sample com `ts = 2025-12-30`,
emite `target_ts = 2026-01-05`, mas o decoder está predizendo o passo no
time_idx 4022 (do próprio dia 2025-12-30, dado o offset detectado em E1).
Resultado: `target_timestamp_utc` em TFT **nunca é trading-day-aligned para h>1**.
A intersecção pareada com baselines (que também usa calendar days, mas a
partir de uma âncora diferente) só funciona quando `(h−1) calendar days`
coincide com algum trading day real — o que para h=7 nunca ocorre (6 dias
corridos da última 6ª-feira pulam um fim de semana).

Baseline runner sofre o mesmo bug em [run_baselines_use_case.py:240](src/use_cases/run_baselines_use_case.py#L240).

### E5 — Convenção do target_return (off-by-one suplementar)

[build_tft_dataset_use_case.py:563](src/use_cases/build_tft_dataset_use_case.py#L563):
```python
df["target_return"] = np.log(df["close"].shift(-1) / df["close"])
```

`target_return[t] = log(close[t+1]/close[t])` = retorno realizado **de t para
t+1**. Em PF, decoder no passo k prediz target em `time_idx = encoder_end + 1 + k`.
Então a predição h=1 (k=0) é para "o retorno realizado de `(encoder_end + 1)`
para `(encoder_end + 2)`" — i.e., **dois dias após o encoder_end**, não um.

Equivalentemente: h=k é o **(k+1)-ésimo** retorno após a decisão, não o k-ésimo.

Este é um **terceiro** ponto de divergência semântica entre TFT e baseline.
Baseline assume h=1 = 1-step-ahead a partir de decision_day (consistente com
literatura financeira). Pode ser bug de target-encoding ou convenção
intencional não documentada.

## Implicações

### Para F.1 (smoke confirmatório)

**F.1 não pode PASSAR como está**. Qualquer métrica pareada (DM, MCS, common-N
overlap) entre TFT e baselines em `gold_paired_oos_intersection_by_horizon`
está medindo erro contra **valores verdadeiros referidos a dias diferentes do
dataset**. RMSE/MAE per-run/split/horizon **internos a um pipeline** continuam
matematicamente corretos (y_true e y_pred do mesmo pipeline são consistentes
entre si), mas qualquer comparação TFT-vs-baseline é falsa.

Em outras palavras: as métricas absolutas no fact/gold para um run individual
estão "OK relativamente a si mesmas" mas mal-rotuladas no eixo de tempo; as
métricas pareadas estão totalmente erradas.

### Para resultados acumulados Stages 1–14

Persistência de TFT em `f555e0e1` data de **2026-04-01**. Todos os sweeps
desde então (~6 semanas até hoje) gravaram `timestamp_utc` como decoder_end.
Impacto:

- **Não invalida** rankings RMSE/MAE entre runs TFT-vs-TFT (todos compartilham
  a mesma convenção).
- **Invalida** comparações TFT-vs-baseline que existem em `gold_dm_pairwise_results.parquet`,
  `gold_paired_oos_intersection_by_horizon`, e plots de paired-OOS.
- **Invalida** quaisquer afirmações de "o modelo prevê retorno do dia D dado
  decisão em D-1" — o que está sendo predito é "retorno do dia D+1 dado
  decisão em D-7" (combinação dos bugs E1 + E5).
- evidência_log.md de 2026-04-24 (mencionada no contexto) já declarou
  invalidação de histórico por bugs anteriores; **este bug deve ser somado** à
  invalidação, não tratado como questão isolada.

### Para o baseline runner (Stage 12)

O Stage 12 PR #30 introduziu `y_true_idx = i + h - 1` (linha 249) **com
intenção** de alinhar com TFT (comentário linhas 244-248). O alinhamento
foi feito por convicção do autor sobre a semântica do TFT, não por verificação
empírica. **O comentário está errado**: TFT não usa `actuals_matrix[i, h-1]
como target h-1 ahead` no sentido em que o baseline assumiu; o `actuals` está
correto, mas o mapeamento sample→ts persiste decoder_end, não decision_day.

Stage F.0.2 (`evaluation_start_offset_days`/`evaluation_end_offset_days`)
aumentou a confusão: o trim ajustou as contagens (685=685, 679=685) e fez
parecer que estavam alinhados, mas só alinhou **o conjunto de target_timestamps
emitidos**, não os y_true sob esses timestamps.

### Para fact_oos_quality_report e gates

O gate de monotonicidade de `target_timestamp` por (run, split, horizon) em
[MULTI_HORIZON.md §"Validacoes obrigatorias"](docs/03_modeling/MULTI_HORIZON.md)
continua válido — `target_ts = ts + (h-1) cal days` é monotonic-increasing
por construção. Mas o gate não detecta o desalinhamento decoder_end vs
decision_day porque cada pipeline isoladamente é monotônico.

## Recomendação de fix (NÃO aplicar nesta sessão)

Três correções coordenadas, com flag de migração:

**Fix 1 (TFT trainer): persistir decision_ts, não decoder_end_ts**

```python
# train_tft_model_use_case.py:778
# ANTES:
split_tail = split_df.tail(n).reset_index(drop=True)
# DEPOIS (proposto):
# A decisão para sample s = encoder_end day = test_df row offset (s + max_encoder_length - 1).
# Para alinhar com baseline runner, queremos ts = timestamps[s + max_encoder_length - 1]
# (i.e., último dia do encoder, primeiro dia "fechado" antes da janela de decoder).
# Adicionar `decision_idx_offset = max_encoder_length - 1` ao split_predictions e
# usar `split_df.iloc[max_encoder_length-1 : max_encoder_length-1 + n]` em vez de tail(n).
```

**Fix 2 (TFT trainer + baseline): target_ts em trading days, não calendar**

Ambos use_cases (linhas 796 do TFT e 240 do baseline) devem mapear:
```python
target_idx = decision_idx + h    # ou h-1, conforme convenção fixada em Fix 3
target_ts = trading_day_at(target_idx)  # consulta no dataset, não Timedelta
```

Em PF a tradução é direta: o `time_idx` é índice em trading-day-space; basta
consultar `dataset_df.iloc[decision_idx + h, "timestamp"]`.

**Fix 3 (convenção de h=k): decidir e documentar em
[MULTI_HORIZON.md](docs/03_modeling/MULTI_HORIZON.md)**

Opções:
- **(a) h=1 = next-day return after decision** (literatura financeira clássica).
  Implica retreinar com `target_return[t] = log(close[t]/close[t-1])` (shift(0),
  retorno realizado *do* dia t) **ou** reescrever decoder mapping para emitir
  `y_true = target_return[encoder_end + h - 1]` em vez de `target_return[encoder_end + h]`.
- **(b) h=1 = 2nd-day return after decision** (TFT atual). Documentar
  explicitamente e renomear: h=1 já é "h+2" semanticamente.

Recomendação: (a). É a convenção alinhada com baseline runner e com horizontes
operacionais documentados em
[STRATEGIC_DIRECTION.md §5 (Fase B)](docs/00_overview/STRATEGIC_DIRECTION.md).

## Blast radius do fix

Se aplicado:

- **Reescrita obrigatória de `fact_oos_predictions`** para todos os runs
  históricos com `timestamp_utc` ressincronizado para decision_day, ou
  invalidação total via flag `schema_version` bump.
- **Refresh do analytics store completo** (silver→gold derivados).
- **Re-treino do TFT** se Fix 3 escolher (a) via dataset rebuild — é caro
  (custou ~40h GPU acumuladas em sweeps até hoje). Alternativa de (a) via
  reindexação do decoder evita re-treino mas precisa de teste unit que verifique
  o shift.
- **Recomputação de toda comparação TFT-vs-baseline** em
  `gold_dm_pairwise_results`, `gold_paired_oos_intersection_by_horizon`,
  `gold_model_decision_final`, plots de OOS pareado, e seções §H1/H2a/H2b
  do living-paper.
- **Validação cruzada**: implementar teste de regressão que pareie
  `(decision_idx, h)` entre TFT e baseline e confirme `y_true_TFT(decision_idx, h)
  == y_true_baseline(decision_idx, h)` em ≥ 99.9% das linhas (margem de
  tolerância apenas para arredondamento float).

Sem o fix, o smoke F.1 não pode ser declarado PASS para qualquer hipótese
que exija pairing TFT-vs-baseline.

## Sugestão de Stage

**Stage F.0.10 (proposed)**: "Reconciliação de convenção timestamp_utc / target_ts
multi-horizonte (TFT e baselines)".

Sub-tasks:
1. Decidir e documentar convenção canônica em
   [MULTI_HORIZON.md](docs/03_modeling/MULTI_HORIZON.md) (Fix 3) — PR doc-only,
   precede código.
2. Implementar Fix 1 (TFT decision_idx mapping) + Fix 2 (target_ts trading-day)
   + Fix 3 (convenção h) em PR código separado.
3. Teste de regressão pareando decision_idx entre TFT e baseline.
4. Bump `schema_version` em `fact_oos_predictions` (atual + 1).
5. Refresh analytics store completo + re-rodada do smoke F.1 confirmatório.
6. Atualizar `docs/07_reports/phase-gates/A_audit_closure_2026-05-17.md` para
   referenciar este bug como achado pós-closure (já existe seção "Gap 6";
   estender para corrigir o escopo).
7. Adicionar no `evidence_log` (data 2026-05-18) a invalidação de resultados
   pareados TFT-vs-baseline anteriores.

Aceite formal: Marcelo decide se Fix 3 escolhe (a) ou (b), e se re-treino é
aceito antes de F.1.

## Apêndice A — Provas reproduzíveis

Script principal (executável em `.venv/bin/python`):

```python
import pandas as pd, numpy as np, glob
ds = pd.read_parquet('data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet')
ds['ts'] = pd.to_datetime(ds['timestamp']).dt.tz_localize(None).dt.normalize()
ts_to_idx = dict(zip(ds['ts'], ds['time_idx']))
target = ds['target_return'].to_numpy()

oos = pd.concat([pd.read_parquet(f) for f in
    sorted(glob.glob('data/analytics/silver/fact_oos_predictions/**/*.parquet', recursive=True))],
    ignore_index=True)
oos['ts'] = pd.to_datetime(oos['timestamp_utc']).dt.tz_localize(None).dt.normalize()

for feat in oos['feature_set_name'].unique():
    for split in ['test','val']:
        for h in [1, 7]:
            sl = oos[(oos['feature_set_name']==feat) & (oos['split']==split) & (oos['horizon']==h)]
            for off in range(-8, 9):
                m = sum(1 for _,r in sl.iterrows()
                        if 0 <= ts_to_idx.get(r['ts'], -1)+off < len(target)
                        and np.isclose(r['y_true'], target[ts_to_idx[r['ts']]+off], atol=1e-7))
                if m > 0:
                    print(f'{feat} {split} h={h} n={len(sl)}: offset={off:+d} matches={m}')
```

Output:
```
BTSF      test h=1 n=685  offset=-6  matches=685
BTSF      test h=7 n=685  offset= 0  matches=685
BTSF      val  h=1 n=437  offset=-6  matches=437
BTSF      val  h=7 n=437  offset= 0  matches=437
baseline  test h=1 n=2055 offset= 0  matches=2055     (3 runs × 685)
baseline  test h=7 n=2037 offset=+6  matches=2037     (3 runs × 679)
baseline  val  h=1 n=1311 offset= 0  matches=1311     (3 runs × 437)
baseline  val  h=7 n=1311 offset=+6  matches=1311
```

## Apêndice B — Bugs secundários adicionais identificados

Durante a investigação:

1. **Calendar days no target_ts (E4)**: `pd.Timedelta(days=h-1)` em dataset
   trading-day. Bug está em ambos pipelines (TFT linha 796, baseline linha 240).
2. **target_return convention (E5)**: `shift(-1)` cria off-by-one em
   interpretação financeira do horizonte. Pode ser intencional, mas não
   documentado.
3. **`split_tail = split_df.tail(n)` recupera decoder_end, não decision_day**:
   bug raiz de H8.
4. **Comentário enganoso em [run_baselines_use_case.py:244-248](src/use_cases/run_baselines_use_case.py#L244-L248)**:
   afirma alinhar-se com TFT mas implementa convenção diferente.

## Apêndice C — Por que a sessão anterior viu apenas "Gap 6"

A sessão Stage F.0 verificou apenas a **interseção do conjunto de
target_timestamps** entre TFT e baseline. Para h=1 o conjunto coincidiu
(685=685) — concluíram "alinhamento perfeito". Não verificaram **valores de
y_true por target_ts**. O sintoma observado para h=7 (679 vs 685) foi tratado
como problema de borda quando, na verdade, é a única manifestação visível em
contagem de um bug que afeta **todas as 685 linhas** (e todos os runs
históricos do TFT).

Aprendizado para gates futuros: validação de OOS pareada precisa verificar
**equidade de y_true por target_ts**, não apenas overlap de conjunto.
