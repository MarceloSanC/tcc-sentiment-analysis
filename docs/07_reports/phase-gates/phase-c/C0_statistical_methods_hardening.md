---
title: "C.0 — Statistical Methods Hardening (Skeleton)"
scope: |
  Relatorio-esqueleto de hardening metodologico dos testes/metricas estatisticas
  do gold legacy do projeto (DM, MCS, Holm, PICP, MPIW, pinball, win-rate,
  prob_up, confidence_calibrated, VaR/ES, gold_model_decision_final, top-50).
  Define inventario, status por item, perguntas metodologicas, criterios de
  promocao a confirmatorio e plano de testes. NAO substitui nem invalida a
  Phase B confirmatoria, que ja foi fechada com sidecars proprios. Este
  documento e living durante a Phase C e sera preenchido progressivamente
  conforme cada item percorrer o ciclo TODO_RESEARCH -> ... -> PROMOTED ou
  -> DESCRIPTIVE_ONLY / HEURISTIC_ONLY.
status: draft_skeleton
created_at: 2026-05-27
canonical_for:
  - c0_statistical_hardening
  - gold_legacy_audit_status
  - phase_c_method_audit_protocol
update_when:
  - novo teste/metrica entrar no inventario
  - status de algum item mudar (CODE_VERIFIED, METHOD_SPECIFIED,
    GAP_CONFIRMED, TEST_PLAN_READY, IMPLEMENTED, VALIDATED,
    PROMOTED_CONFIRMATORY, DESCRIPTIVE_ONLY, HEURISTIC_ONLY)
  - pesquisa primaria fixar definicao canonica de um item
  - test plan de um item P0 estiver pronto para execucao
  - politica de contencao de escopo for revisada
---

# C.0 — Statistical Methods Hardening (Skeleton)

> **Tipo de documento:** living skeleton. Cada item do inventario tem um
> dossie padronizado e um status formal; campos `TODO` sao preenchidos
> conforme a pesquisa metodologica avanca. Este arquivo nao tira conclusao
> sozinho.

## 1. Resumo executivo

C.0 e uma etapa de **hardening metodologico**. Objetivo: tornar os testes e
metricas estatisticas do **gold legacy** academicamente defensaveis antes
de qualquer uso confirmatorio futuro.

Pontos firmes:

- **C.0 nao bloqueia a Phase B**, que ja foi fechada
  ([B_confirmatory_2026-05-25.md](../phase-b/B_confirmatory_2026-05-25.md))
  com sidecars confirmatorios proprios:
  - H1 (calibracao) -> `phase_b_marginal_coverage.parquet` +
    `phase_b_tier_verdict.parquet`;
  - H2a/H2b (TFT vs baselines em pinball) -> `phase_b_dm_family_6.parquet`
    + `phase_b_tier_verdict.parquet`.
- **C.0 bloqueia apenas o uso futuro dos gold statistical artifacts como
  evidencia confirmatoria** (em particular `gold_dm_pairwise_results`,
  `gold_mcs_results`, `gold_win_rate_pairwise_results`,
  `gold_model_decision_final`, `gold_prediction_risk`,
  `gold_prediction_calibration` e seus derivados quando usados para
  selecionar/declarar vencedor).
- **Phase B sidecars continuam a fonte confirmatoria atual** para H1/H2a/H2b.
  Se houver conflito numerico/conceitual com gold legacy, vale Phase B
  sidecar para o claim correspondente.
- C.0 **nao corrige codigo** neste PR. Este arquivo e apenas o guia de
  trabalho. Implementacao vive em PRs subsequentes (C.0.5, C.0.6).

Insumos externos relevantes (registrados, ainda nao internalizados):

- Auditoria metodologica externa
  ([`auditoria_metodologica_forecasting_financeiro.md`](../../external-reviews/auditoria_metodologica_forecasting_financeiro.md))
  fornece R1-R17 e verdicts preliminares por item; nao foi auditoria sobre o
  codigo real, apenas sobre o
  [`DATA_PIPELINE_WALKTHROUGH.md`](../../../02_data/DATA_PIPELINE_WALKTHROUGH.md).
  C.0 reverifica os achados contra implementacao verdadeira em `src/`.

## 2. Escopo

### In scope

- DM gold (`gold_dm_pairwise_results`)
- MCS gold (`gold_mcs_results`)
- Holm gold (ajuste aplicado a DM gold)
- top-50 filter aplicado antes de DM/MCS/win-rate gold
- PICP (gold prediction metrics + calibration)
- MPIW (gold prediction metrics + calibration)
- pinball loss (gold prediction metrics, todas variantes)
- win-rate gold (`gold_win_rate_pairwise_results`)
- prob_up gold (`gold_prediction_metrics_*`)
- confidence_calibrated gold (`gold_prediction_metrics_*`,
  `gold_prediction_calibration`)
- VaR / ES gold (`gold_prediction_risk`)
- `gold_model_decision_final`
- Phase B DM family-6 sidecar (apenas para registro de contraste com gold
  legacy; **nao reavaliar** a Phase B em si)

### Out of scope

- Reexecutar a Phase B.
- Alterar conclusoes ja publicadas da Phase B.
- Implementar correcoes de codigo neste PR (este arquivo e skeleton).
- Fazer pesquisa bibliografica completa neste arquivo inicial; aqui apenas
  registramos campos TODO e o que ja sabemos.
- Promover qualquer metrica a status confirmatorio sem paper/spec/testes.
- Reescrever historia dos artefatos legacy (nao desfazer Phase B).

## 3. Politica de status

Cada item do inventario tem exatamente **um** status corrente. A unica
direcao normal e do topo para o fundo da lista; reabertura (ex.
`PROMOTED_CONFIRMATORY` -> `GAP_CONFIRMED`) e permitida e deve ser
registrada com data.

| Status | Significado |
|---|---|
| `TODO_RESEARCH` | Ainda nao foi confirmado o uso atual no codigo nem definida a referencia canonica. |
| `CODE_LOCATED` | Implementacao atual mapeada no repositorio (arquivos/funcoes/linhas). Ainda nao validada matematicamente contra paper. |
| `CODE_VERIFIED` | Codigo verificado linha a linha contra a formula declarada nos docs internos; sem afirmacao sobre adequacao academica. |
| `METHOD_SPECIFIED` | Paper/documentacao primaria identificada; variante e suposicoes explicitas registradas neste documento. |
| `GAP_CONFIRMED` | Foi formalmente confirmado um gap entre o que o codigo faz e o que o paper/uso pretendido exige. Pode ser bug, limitacao ou misuso. |
| `TEST_PLAN_READY` | Test plan unitario + contratual escrito e revisado neste documento. Ainda nao implementado. |
| `READY_FOR_IMPLEMENTATION` | Decisao metodologica e plano de testes aprovados; PR de implementacao pode ser aberto. Exige `METHOD_SPECIFIED` + decisao explicita + `TEST_PLAN_READY`. **Nao exige** passagem por `GAP_CONFIRMED` quando a decisao for rebaixar (`DESCRIPTIVE_ONLY`/`HEURISTIC_ONLY`) ou quando a verificacao concluiu "sem gap" sob a definicao canonica adotada. |
| `IMPLEMENTED` | PR mergeado em `main`; codigo reflete a decisao C.0. |
| `VALIDATED` | Testes unitarios + contratuais passam; sidecar/parquet de validacao gerado e revisado. |
| `PROMOTED_CONFIRMATORY` | Item pode ser usado como evidencia confirmatoria em pre-registro futuro; entrou na familia oficial de inferencia. |
| `DESCRIPTIVE_ONLY` | Item util como estatistica descritiva, **sem** uso confirmatorio; documentado como tal nos docs canonicos. |
| `HEURISTIC_ONLY` | Item util apenas como heuristica operacional/intuitiva; nao serve como inferencia, mesmo descritiva formal. Renomeacao recomendada. |

Regra de avanco: nao pular etapas. Caminhos validos para
`READY_FOR_IMPLEMENTATION`:

- **Com gap:** `METHOD_SPECIFIED` -> `GAP_CONFIRMED` -> `TEST_PLAN_READY` -> `READY_FOR_IMPLEMENTATION`.
- **Sem gap (validacao):** `METHOD_SPECIFIED` -> decisao "no-gap" registrada -> `TEST_PLAN_READY` -> `READY_FOR_IMPLEMENTATION`.
- **Rebaixamento (descritivo/heuristico):** `METHOD_SPECIFIED` -> decisao
  explicita de rebaixar -> `TEST_PLAN_READY` (para invariantes + bloqueios de
  uso) -> `READY_FOR_IMPLEMENTATION` -> `IMPLEMENTED` -> `VALIDATED` ->
  `DESCRIPTIVE_ONLY` / `HEURISTIC_ONLY`. Nao passa por `GAP_CONFIRMED` nem
  por `PROMOTED_CONFIRMATORY`.

`GAP_CONFIRMED` so e usado quando houver gap matematico/inferencial
formalmente confirmado contra a definicao canonica adotada.

## 4. Matriz principal de inventario

| Item | Categoria atual esperada | Uso atual no projeto | Implementacao localizada | Docs que mencionam | Risco metodologico conhecido | Pesquisa primaria necessaria | Testes necessarios | Status inicial | Prioridade |
|---|---|---|---|---|---|---|---|---|---|
| Diebold-Mariano gold | unknown (suspeito descriptive) | `gold_dm_pairwise_results` alimenta `gold_model_decision_final` | [`pairwise.py:_compute_dm_pairwise_from_loss_matrix`](../../../../src/domain/services/gold_builders/pairwise.py) (linhas 67-104) | [`STATISTICAL_TESTS.md`](../../../04_evaluation/STATISTICAL_TESTS.md), `DATA_PIPELINE_WALKTHROUGH.md` | loss=squared_error fixa; HAC lag hard-coded `min(max(1, n^{1/3}), 10)`; two-sided fixo; sem HLN; aplicado pos top-50; Holm sem split_signature | DM 1995, HLN 1997, Newey-West 1987; lag policy; alinhamento por target_timestamp/decision_timestamp | Unit: HAC numerico; contrato: alinhamento estrito por (asset, parent_sweep_id, split_signature, split, horizon, target_timestamp); fixture com benchmark externo | CODE_LOCATED | P0 |
| MCS gold | unknown (suspeito descriptive) | `gold_mcs_results.selected_in_mcs_alpha_0_05` e mergeado em `gold_model_decision_final` como coluna diagnostica | [`pairwise.py:_compute_mcs_from_loss_matrix`](../../../../src/domain/services/gold_builders/pairwise.py) (linhas 107-180) | `STATISTICAL_TESTS.md`, `DATA_PIPELINE_WALKTHROUGH.md` | B=300 baixo; block_len=5 hard-coded; loss=squared_error; verificar se split_signature esta sempre presente/populada upstream; alimentado por loss_matrix pos top-50 | Hansen-Lunde-Nason 2011; Kunsch 1989 (block bootstrap); justificativa block_len; sensibilidade B | Unit: MCS contra pacote externo em fixture pequena; contrato: saida preserva split_signature e cohort | CODE_LOCATED | P0 |
| Holm gold | unknown (suspeito descriptive) | Ajuste aplicado em `gold_dm_pairwise_results.pvalue_adj_holm` | [`pairwise.py:_apply_holm_adjustment_for_dm`](../../../../src/domain/services/gold_builders/pairwise.py) (linhas 183-214) | `STATISTICAL_TESTS.md` | groupby = [asset, parent_sweep_id, split, horizon]; **split_signature ausente**; familia pode subcorrigir multiplicidade real do claim | Holm 1979; definicao formal de familia para cada claim | Unit: formula Holm contra valores manuais; contrato: familia inclui todos os testes que sustentam o mesmo claim; teste de invariancia split_signature | CODE_LOCATED | P0 |
| top-50 filter | heuristica de engenharia (a remover ou explicitar como exploratorio) | `_select_top_configs_for_pairwise(..., max_configs=50)` aplicado antes de DM/MCS/win-rate (linhas 299, 351, 396) | [`pairwise.py:43-64`](../../../../src/domain/services/gold_builders/pairwise.py) | `DATA_PIPELINE_WALKTHROUGH.md` lacuna A.65 | **Selecao pos-test:** filtra top-50 por `mean(squared_error)` no proprio test split; viola hipotese de universo pre-definido; cap silencioso | Inferencia seletiva / data snooping (Romano-Wolf, White 2000) | Contrato: ou (a) universo pre-declarado, ou (b) flag explicita `exploratory_only=True` propagada ate `gold_model_decision_final` | CODE_LOCATED | P0 |
| PICP | descriptive (potencial confirmatory com bandas + condicional) | `gold_prediction_metrics_*.picp`, `gold_prediction_calibration` | [`quantile.py`](../../../../src/domain/services/gold_builders/quantile.py) (`_build_metrics_single_contract`, linha 102; `covered_80` derivado e agg em ~244-261) | `METRICS_DEFINITIONS.md`, `CALIBRATION_AND_RISK.md`, `STATISTICAL_TESTS.md` | `coverage_nominal=0.80` hard-coded; nao usa contrato dinamico de quantis; sem teste de cobertura condicional/independencia | Christoffersen 1998; Gneiting-Balabdaoui-Raftery 2007; Khosravi et al. 2011 | Unit: PICP contra fixture controlada; contrato: nominal derivado de `quantile_levels` reais; teste de Christoffersen (cobertura condicional + independencia) | CODE_LOCATED | P1 |
| MPIW | descriptive | `gold_prediction_metrics_*.mpiw` | [`quantile.py:245`](../../../../src/domain/services/gold_builders/quantile.py) `mpiw=("pred_interval_width", "mean")` | `METRICS_DEFINITIONS.md`, `CALIBRATION_AND_RISK.md` | Sozinho nao mede calibracao; degeneracao da quantil produz MPIW=0 nao informativo; comparacao cross-asset exige normalizacao | Gneiting-Raftery 2007 (interval score / Winkler); Khosravi et al. 2011 | Unit: MPIW = mean(p90 - p10) em fixture; contrato: interval/Winkler score adicional; bloqueio quando degeneracy_rate alto | CODE_LOCATED | P1 |
| pinball loss | descriptive (confirmatory na Phase B sidecar) | `gold_prediction_metrics_*.{pinball_q10,q50,q90,mean_pinball}` | [`quantile.py:79`](../../../../src/domain/services/gold_builders/quantile.py) `_pinball_loss`; agg em linhas 188-244 | `METRICS_DEFINITIONS.md`, `STATISTICAL_TESTS.md` | Quantis hard-coded q=0.1, 0.5, 0.9; pesos uniformes implicitos em mean_pinball; sem ressalva sobre triplet degenerado | Koenker-Bassett 1978; Gneiting 2011 | Unit: pinball contra formula canonica; contrato: quantis lidos do contrato real (`quantile_levels`); flag degeneracao | CODE_LOCATED | P1 |
| win-rate gold | descriptive | `gold_win_rate_pairwise_results` e mergeado em `gold_model_decision_final` como coluna diagnostica | [`pairwise.py:382`](../../../../src/domain/services/gold_builders/pairwise.py) (`output_table = "gold_win_rate_pairwise_results"`; calculo em linhas ~410-445) | `STATISTICAL_TESTS.md`, `DATA_PIPELINE_WALKTHROUGH.md` | Sem variancia/p-value; risco de consumidor downstream tratar coluna como sinal de selecao; aplicado pos top-50 | Sign test / block bootstrap para inferencia; tratamento de empates | Unit: win_rate e win_rate_ex_ties em fixture; contrato: nao usar como criterio confirmatorio em decision_final; tie_rate reportado | CODE_LOCATED | P2 |
| prob_up | heuristic (renomear) | `gold_prediction_metrics_*.{prob_up,prob_down}` | [`quantile.py:_prob_up_from_quantiles`](../../../../src/domain/services/gold_builders/quantile.py) (linha 85), agg em linha 247 | `METRICS_DEFINITIONS.md`, `DATA_PIPELINE_WALKTHROUGH.md` lacuna A.67 | CDF piecewise-linear ad hoc entre q10 e q90; fallback 1/0/0.5 quando width=0 mascara degeneracao | Gneiting-Raftery 2007 (proper scoring); literatura de quantile-to-CDF | Unit: prob_up retorna NaN se q90 <= q10; calibracao empirica vs y_true>0 (reliability/Brier) | CODE_LOCATED | P2 |
| confidence_calibrated | heuristic (renomear) | `gold_prediction_metrics_*.confidence_calibrated`, `gold_prediction_calibration` | [`quantile.py:267`](../../../../src/domain/services/gold_builders/quantile.py) `calibration_term * width_term` | `METRICS_DEFINITIONS.md` ("Confidence Proxy") | Produto ad hoc; nao e proper score; depende da escala do target; nome sugere calibracao estatistica | Gneiting et al. 2007 (proper scores: CRPS, WIS, interval score) | Unit: formula determinada; contrato: nao entra como criterio confirmatorio em decision_final; renomear coluna para `heuristic_coverage_width_score` em PR futuro | CODE_LOCATED | P2 |
| VaR / ES gold | unknown (suspeito tail_error, nao risco financeiro) | `gold_prediction_risk.{var_10, es_10_approx}` | [`confidence.py:62-109`](../../../../src/domain/services/gold_builders/confidence.py) | `METRICS_DEFINITIONS.md`, `CALIBRATION_AND_RISK.md` | TODO: verificar se variavel base e `y_pred` (quantil), erro (`y_pred - y_true`), ou retorno; aud externa supos `error`, codigo agora parece usar `q10`; sem backtesting de excedencias; nivel alpha implicito | Jorion 2007 (VaR); Acerbi-Tasche 2002 (ES); Fissler-Ziegel 2016 (joint elicitability); Christoffersen 1998 / Kupiec 1995 (backtesting) | Unit: VaR/ES sobre fixture com distribuicao conhecida; contrato: variavel-alvo declarada (retorno/perda) e nivel alpha explicito; backtesting de cobertura | CODE_LOCATED | P1 |
| gold_model_decision_final | unknown (depende de inputs P0) | Rollup diagnostico por (`asset`, `horizon`); consumido por plots oficiais | [`confidence.py:_build_model_decision_final`](../../../../src/domain/services/gold_builders/confidence.py) (linha 463); `requires_gold` em linhas 824-831; `output_table = "gold_model_decision_final"` (linha 822) | `ANALYTICS_STORE_ARCHITECTURE.md`, `METRICS_DEFINITIONS.md`, `PLOT_INTERPRETATION.md` | Ordenacao por `rank_rmse, rank_mae` (pontual); pinball/PICP/MPIW entram como colunas; `win_rate_ex_ties_mean` aparece sem disclaimer; herda riscos de DM/MCS/Holm/top-50; sem pre-registro do criterio | Scorecard confirmatorio exige criterio pre-declarado separado | Contrato: nao usar como evidencia confirmatoria sem pre-registro; teste de invariancia que congele criterio pos-promocao | CODE_LOCATED | P0 |
| Phase B DM family-6 | confirmatory (Phase B fechada, **apenas H2a/H2b**) | `phase_b_dm_family_6.parquet` (sidecar), 6 testes one-sided HAC+HLN, Holm sobre familia 6 | [`dm_tft_vs_baseline.py`](../../../../src/domain/services/dm_tft_vs_baseline.py), [`holm_family_6.py`](../../../../src/domain/services/holm_family_6.py), [`compute_phase_b_tier_metrics_use_case.py`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py) | `STATISTICAL_TESTS.md` §"Unidade Estatistica DM Em Walk-forward", `preregistration_phase_b.md` E1.8, `B_confirmatory_2026-05-25.md` §4.1 | Item ja **pre-registrado e fechado**; risco C.0 e apenas registro/duplicacao acidental; H1 vive em `phase_b_marginal_coverage.parquet` + `phase_b_tier_verdict.parquet`, nao aqui | Nenhuma — referenciar Phase B; nao reabrir | C.0 nao testa; apenas verifica que gold legacy nao seja confundido com este sidecar | PROMOTED_CONFIRMATORY (escopo Phase B) | N/A |

Onde diz "TODO" ou "suspeito", **nao tomar como verdade**. Quem promover ou
rebaixar precisa preencher o dossie correspondente em §7 com evidencia.

## 5. Priorizacao

| Prioridade | Criterio |
|---|---|
| **P0** | Itens cujo uso incorreto pode **contaminar claim confirmatorio** (DM gold, MCS gold, Holm gold, top-50, gold_model_decision_final). Resolver antes de qualquer claim publicado a partir de gold legacy. |
| **P1** | Metricas probabilisticas centrais para o TCC/artigo (PICP, MPIW, pinball, VaR/ES). Resolver antes de claims probabilisticos novos que nao sejam Phase B. |
| **P2** | Heuristicas/descritivos uteis, mas nao confirmatorios (win-rate, prob_up, confidence_calibrated). Resolver para harmonizacao documental e renomear se necessario. |
| **P3** | Documentacao, walkthrough e limpeza pos-decisao C.0. |

Nenhum item P1 deve avancar para implementacao ou promocao confirmatoria antes
de os itens P0 estarem em pelo menos `METHOD_SPECIFIED`.

## 5.1 Politica de vencedor final e scorecard de modelos

A conclusao final do TCC pode eleger um **vencedor primario**, mas essa eleicao
nao deve ser inferida implicitamente de `gold_model_decision_final` nem de uma
leitura livre de varias metricas apos observar os resultados.

Regra recomendada:

1. Definir ex-ante a metrica primaria do claim principal. Para o escopo atual,
   a candidata preferencial e `pinball_loss_post_guardrail`, por estar alinhada
   ao objetivo probabilistico e ja ter caminho confirmatorio na Phase B.
2. Usar DM + Holm para testar se o ganho na metrica primaria contra baselines
   e estatisticamente sustentado, com familia de hipoteses declarada.
3. Usar PICP como gate/diagnostico de calibracao e MPIW como medida de
   sharpness, sempre em conjunto. PICP isolado nao escolhe vencedor; MPIW
   isolado tambem nao escolhe vencedor.
4. Usar MCS como evidencia complementar: pertencer ao confidence set superior
   significa "nao rejeitado como estatisticamente inferior", nao "venceu".
5. Reportar RMSE/MAE, PICP, MPIW, MCS, win-rate e demais metricas como
   **perfil comparativo** dos modelos. Essas metricas explicam trade-offs e
   caracteristicas de cada modelo, mas nao trocam o vencedor primario depois
   que os resultados foram vistos.

Artefato recomendado para a conclusao:

- `model_comparison_confirmatory_scorecard_phase_c.parquet`

Esse sidecar deve ser curado a partir de artefatos persistidos e da regra
metodologica pre-registrada. Ele nao substitui os sidecars Phase B nem promove
automaticamente o gold legacy. O conteudo minimo esperado e:

- chaves de escopo: `asset`, `horizon`, `cohort_id`, `parent_sweep_id`,
  `split_signature`;
- identificacao do modelo: `model_family`, `feature_set_name`,
  `config_signature`, `baseline_name` quando aplicavel;
- metrica primaria: `primary_metric_name`, `primary_metric_value`,
  `primary_metric_delta_vs_baseline`, `primary_metric_relative_delta`;
- teste confirmatorio: `dm_stat`, `pvalue`, `p_adj_holm`,
  `holm_reject_0_05`, `claim_supported`;
- diagnosticos probabilisticos: `picp`, `picp_band_status`, `mpiw`,
  `mean_pinball`;
- diagnosticos auxiliares: `rmse`, `mae`, `mcs_selected_alpha_0_05`,
  `win_rate_ex_ties_mean`;
- decisao interpretativa: `primary_winner`, `profile_summary`,
  `main_tradeoff`, `limitations_note`.

## 6. Dossies por item

Template padronizado. Cada subsecao e auto-suficiente para referencia
cruzada e pode ser linkada de PRs futuros (`C0_statistical_methods_hardening.md#diebold-mariano-gold`).

---

### Diebold-Mariano gold

**Objetivo estatistico pretendido**
TODO: declarar se o claim e pontual (squared error) ou probabilistico
(pinball/CRPS/WIS). A versao gold atual roda em squared_error mesmo quando
o projeto reivindica resultado probabilistico.

**Uso atual no projeto**
- Alimenta `gold_dm_pairwise_results` (loss matrix wide por `target_timestamp_utc`).
- Resultado agregado por `_build_model_decision_final` em `dm_significant_wins`,
  `dm_significant_losses` e `dm_net_wins`, depois mergeado em
  `gold_model_decision_final` como colunas diagnosticas. Nao e criterio de
  ordenacao; a ordenacao final do gold e por `rank_rmse, rank_mae`
  ([`confidence.py:588-631`](../../../../src/domain/services/gold_builders/confidence.py),
  [`confidence.py:717-818`](../../../../src/domain/services/gold_builders/confidence.py)).
- **Nao confundir** com `phase_b_dm_family_6.parquet` (sidecar Phase B confirmatorio).

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/pairwise.py`](../../../../src/domain/services/gold_builders/pairwise.py)
- Funcao: `_compute_dm_pairwise_from_loss_matrix`
- Linhas aproximadas: 67-104
- Evidencia (confirmada por leitura linha-a-linha em 2026-05-28):
  - loss = `squared_error`: `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2` em `_pairwise_preprocess` (linha 280)
  - n minimo = 5: `if n < 5: continue` (linha 78)
  - HAC Bartlett: lag `int(min(max(1, n ** (1 / 3)), 10))` (linha 82); kernel `weight = 1.0 - (k / (lag + 1))` (linha 87)
  - estatistica = `mean_d / math.sqrt(var_mean)` (linha 92); pvalue = `2.0 * (1.0 - _norm_cdf(abs(stat)))`, two-sided (linha 93)
  - top-50 filter aplicado **antes** por `DmPairwiseResultsGoldBuilder.build()` (linha 299: `g = _select_top_configs_for_pairwise(g, max_configs=50)`)
  - **sem HLN small-sample correction** (confirmado — nao ha chamada a HLN em nenhum ponto do fluxo)
  - **sem versao one-sided** (confirmado)
  - Holm correction aplicada em `DmPairwiseResultsGoldBuilder.build()` (linha 333): `return _apply_holm_adjustment_for_dm(pd.concat(rows, ...))` — saida `gold_dm_pairwise_results` ja inclui colunas `pvalue_adj_holm` e `significant_adj_0_05`; porem `_build_model_decision_final` usa `pvalue_two_sided` (nao `pvalue_adj_holm`) para calcular `dm_net_wins` (conf. confidence.py L605)

**Definicao canonica esperada**
TODO: Diebold-Mariano 1995 (R1); HLN 1997 (R2) para small-sample; Newey-West
1987 (R3) para HAC. Declarar lag policy explicita.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Phase B usa `target_timestamp` com
  dedup operationally-latest cross-fold; gold legacy faz pivot wide com
  `dropna(how="any")` (linha ~285 de `pairwise.py`). Equivalencia ainda nao
  formalizada.
- Ha intersecao temporal exata por `target_timestamp` quando aplicavel?
  Aparentemente sim via pivot, mas falta validar que folds/seeds nao sao
  duplicados.
- Ha controle de multiplas comparacoes quando necessario? Apenas Holm, e a
  familia possivelmente esta mal definida (ver Holm gold).
- A perda usada corresponde a hipotese do TCC? Nao — `squared_error` para
  claims sobre quantis e modelo probabilistico TFT.
- A metrica e confirmatoria, descritiva ou heuristica? **TODO**; status atual
  do TCC trata sidecar Phase B como confirmatorio e nao usa DM gold para
  H1/H2a/H2b.
- O uso atual bate com o que o nome sugere? Parcialmente — formula e DM
  reconhecivel, mas o nome `dm_pairwise` sugere generalidade que nao se
  sustenta sob top-50 e loss desalinhada.

**Riscos conhecidos**
- Inferencia seletiva via top-50 pos-test.
- Loss desalinhada de claim probabilistico.
- Ausencia de HLN gera p-values otimistas em amostras pequenas.
- Two-sided sem one-sided perde poder para hipotese direcional.

**Criterio para promocao a confirmatorio**
TODO. Sugestao inicial: (1) loss primaria pre-declarada por pre-registro,
(2) universo pre-definido (sem top-50 por test), (3) HLN aplicado, (4)
one-sided quando claim direcional, (5) Holm com familia formalmente
definida, (6) sensibilidade ao lag reportada.

**Testes unitarios necessarios**
- TODO: formula HAC contra calculo manual em serie controlada.
- TODO: comparacao com pacote externo (statsmodels/arch) em fixture pequena.
- TODO: equivalencia com `dm_tft_vs_baseline.py` (Phase B) sob mesma
  configuracao (loss, lag, HLN off, two-sided).

**Testes de contrato/integridade necessarios**
- TODO: nao executar se top-50 estiver ativo sem flag `exploratory_only`.
- TODO: `pvalue_adj_holm` so existe quando familia foi declarada e
  groupby inclui split_signature.
- TODO: alinhamento por `(asset, parent_sweep_id, split_signature, split, horizon, target_timestamp_utc)` validado.

**Docs impactados se corrigido**
- TODO: `STATISTICAL_TESTS.md`, `DATA_PIPELINE_WALKTHROUGH.md`,
  `METRICS_DEFINITIONS.md`, `PLOT_INTERPRETATION.md` (se DM aparece em plot
  oficial).

**Status**
`CODE_LOCATED` (arquivo, funcao e linhas confirmados por leitura direta em 2026-05-28; evidencia atualizada com nota sobre Holm em L333).

---

### MCS gold

**Objetivo estatistico pretendido**
TODO. Variante range t-stat com moving block bootstrap.

**Uso atual no projeto**
- `gold_mcs_results.selected_in_mcs_alpha_0_05` e renomeado para
  `mcs_selected_alpha_0_05` e mergeado em `gold_model_decision_final` como
  coluna diagnostica. Nao e criterio de ordenacao; a ordenacao final do gold e
  por `rank_rmse, rank_mae`
  ([`confidence.py:633-647`](../../../../src/domain/services/gold_builders/confidence.py),
  [`confidence.py:717-818`](../../../../src/domain/services/gold_builders/confidence.py)).

**Implementacao atual localizada**
- Arquivo: `src/domain/services/gold_builders/pairwise.py`
- Funcao: `_compute_mcs_from_loss_matrix`
- Linhas aproximadas: 107-180
- Evidencia:
  - alpha=0.05; bootstrap_samples=300; block_len=5; random_seed=42
  - moving block bootstrap com wrap-around (linha 124-130)
  - estatistica `tr_stat = nanmax(|dbar / sqrt(var)|)` (linha 153)
  - elimina por `argmax(losses_mean)` (linha 170)
  - tambem alimentado pelo top-50 (linha 351)
  - saida lista `config_label`, `selected_in_mcs_alpha_0_05`, `mean_loss`
  - `_pairwise_group_cols` inclui `split_signature` quando a coluna existe
    ([`pairwise.py:25-29`](../../../../src/domain/services/gold_builders/pairwise.py))
    e o MCS propaga essa chave na saida
    ([`pairwise.py:368-374`](../../../../src/domain/services/gold_builders/pairwise.py)).

**Definicao canonica esperada**
TODO: Hansen, Lunde, Nason 2011 (R8); Kunsch 1989 (R9) para block bootstrap.

**Perguntas metodologicas**
- A unidade estatistica esta correta?
- Ha intersecao temporal exata? Loss matrix herda de `_pairwise_preprocess` (mesma chave do DM).
- B=300 e suficiente para inferencia em alpha=0.05?
- block_len=5 e adequado para a serie de losses pareadas? Sem justificativa documental.
- A perda usada corresponde a hipotese do TCC? `squared_error` (idem DM gold).
- A metrica e confirmatoria, descritiva ou heuristica? **TODO**.
- O uso atual bate com o que o nome sugere? Sim como variante MCS, mas "selecionado em MCS" e frequentemente mal interpretado como "vence".

**Riscos conhecidos**
- B=300 baixo; sensibilidade a block_len; loss desalinhada; pre-selecao top-50.
- Se `split_signature` nao vier populada do upstream, a familia MCS perde essa
  chave; quando a coluna existe, o codigo atual preserva `split_signature`.

**Criterio para promocao a confirmatorio**
TODO. Esboco: B >= 1000 (preferencialmente 5000), block_len justificado por
estimador (Politis-White), loss primaria declarada, universo pre-definido,
saida preserva `split_signature`, interpretacao no texto rebaixada de
"vencedor" para "nao rejeitado como pertencente ao set superior".

**Testes unitarios necessarios**
- TODO: replicacao em fixture pequena contra pacote conhecido (R-MCS, arch).
- TODO: invariancia ao seed dentro do erro de Monte Carlo declarado.
- TODO: comportamento com empates de mean_loss.

**Testes de contrato/integridade necessarios**
- TODO: top-50 nao pode estar ativo em coorte confirmatoria.
- TODO: schema de saida inclui `split_signature` e contagem de configs eliminadas em cada iteracao.

**Docs impactados se corrigido**
- TODO: `STATISTICAL_TESTS.md`, `DATA_PIPELINE_WALKTHROUGH.md`,
  `ANALYTICS_STORE_ARCHITECTURE.md`.

**Status**
`CODE_LOCATED` (arquivo, funcao e linhas confirmados por leitura direta em 2026-05-28).

---

### Holm gold

**Objetivo estatistico pretendido**
Controle de FWER sobre a familia de DM pairwise tests.

**Uso atual no projeto**
- Coluna `pvalue_adj_holm` em `gold_dm_pairwise_results`.
- `_build_model_decision_final` calcula `dm_net_wins` a partir de
  `pvalue_two_sided < 0.05`, nao de `pvalue_adj_holm`; portanto Holm gold fica
  disponivel no artefato DM, mas nao e hoje o criterio efetivo do rollup final.

**Implementacao atual localizada**
- Arquivo: `src/domain/services/gold_builders/pairwise.py`
- Funcao: `_apply_holm_adjustment_for_dm`
- Linhas aproximadas: 183-214
- Evidencia:
  - groupby = `[asset, parent_sweep_id, split, horizon]` (linha 191)
  - **`split_signature` NAO entra no groupby** (apesar de DM ter `split_signature` na grain de preprocessamento)
  - formula: ordena p-values; `(m - j + 1) * p`; `np.maximum.accumulate`; clip 1 (linhas 197-208)
  - flag `significant_adj_0_05` = `pvalue_adj_holm < 0.05` (linha 211-213)

**Definicao canonica esperada**
TODO: Holm 1979 (R7). Familia precisa estar declarada antes de olhar p-values.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Cada teste e um par (left_config, right_config) dentro de cada grupo.
- Ha intersecao temporal exata? Herdada do DM upstream.
- Ha controle de multiplas comparacoes quando necessario? Sim, mas familia possivelmente mal definida.
- A perda usada corresponde a hipotese do TCC? N/A — Holm e ajuste, nao perda.
- A metrica e confirmatoria, descritiva ou heuristica? Depende de quao bem a familia case com o claim final.
- O uso atual bate com o que o nome sugere? **Parcialmente** — Holm step-down esta correto na formula; a falha esta na composicao da familia.

**Riscos conhecidos**
- Omissao de `split_signature` mistura desenhos experimentais diferentes na mesma familia.
- Se claim final agrega sobre horizontes/baselines/assets/sweeps, ajuste so por `(asset, parent_sweep_id, split, horizon)` subcorrige.

**Criterio para promocao a confirmatorio**
TODO. Esboco: familia formalmente declarada por claim em pre-registro; Holm
aplicado exatamente sobre o conjunto dos testes que sustentam o mesmo
claim; teste de regressao que falha se `split_signature` sair do groupby
sem justificativa.

**Testes unitarios necessarios**
- TODO: formula Holm contra valores tabelados (literatura/scipy).
- TODO: invariancia ao reorder dos p-values.

**Testes de contrato/integridade necessarios**
- TODO: groupby inclui `split_signature`.
- TODO: validacao que para cada grupo, `m` igual a numero de pares no DM correspondente.

**Docs impactados se corrigido**
- TODO: `STATISTICAL_TESTS.md` (atualizar §"Multiple Testing"), `DATA_PIPELINE_WALKTHROUGH.md` (lacuna A.66).

**Status**
`CODE_LOCATED` (arquivo, funcao e linhas confirmados por leitura direta em 2026-05-28; groupby sem split_signature em L191 confirmado).

---

### top-50 filter

**Objetivo estatistico pretendido**
Nenhum. E uma reducao operacional de custo computacional. O risco e
metodologico: aplicar antes de DM/MCS muda a pergunta inferencial.

**Uso atual no projeto**
- `_select_top_configs_for_pairwise(grouped_oos, max_configs=50)` aplicado
  antes de DM (linha 299), MCS (linha 351) e win-rate (linha 396) em
  `pairwise.py`.
- Ranking por `groupby("config_label")["squared_error"].mean()` ascendente
  (linhas 57-62).
- Cap silencioso: nao ha sinalizacao em coluna do output de quais configs
  foram excluidas.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/pairwise.py`](../../../../src/domain/services/gold_builders/pairwise.py)
- Funcao: `_select_top_configs_for_pairwise`
- Linhas: 43-64
- Evidencia (confirmada por leitura linha-a-linha em 2026-05-28):
  - Retorna o proprio DataFrame sem filtro se `cfg_count <= max_configs` (linha 55)
  - Ranking: `grouped_oos.groupby("config_label")["squared_error"].mean().sort_values(ascending=True).head(max_configs)` (linhas 57-61) — menor squared_error = melhor
  - max_configs default = 50 (linha 46)
  - Aplicado em `DmPairwiseResultsGoldBuilder.build()` (linha 299), `McsResultsGoldBuilder.build()` (linha 351), `WinRatePairwiseResultsGoldBuilder.build()` (linha 396)
  - Configs excluidas nao aparecem nos outputs gold (cap silencioso confirmado — nao ha coluna indicando excluidas)

**Definicao canonica esperada**
TODO: nao ha; literatura exige universo pre-definido em test
comparisons (White 2000, Romano-Wolf 2005, Hansen 2005). Reducao
defensavel apenas se baseada em **validacao** (split=val) ou pre-registro.

**Perguntas metodologicas**
- A unidade estatistica esta correta? N/A — e filtro, nao teste.
- Ha intersecao temporal exata? N/A.
- Ha controle de multiplas comparacoes quando necessario? Filtro **invalida**
  o controle (universo nao e mais pre-definido).
- A perda usada corresponde a hipotese do TCC? Ranking usa `squared_error`,
  mesmo desalinhamento dos itens P0.
- A metrica e confirmatoria, descritiva ou heuristica? N/A — e filtro.
- O uso atual bate com o que o nome sugere? Nome de funcao
  (`_select_top_configs_for_pairwise`) e ate honesto; o problema e o
  acoplamento implicito ao test split.

**Riscos conhecidos**
- Inferencia pos-selecao em DM/MCS/win-rate todos derivados deste filtro.
- Cap silencioso (configs excluidas nao aparecem em gold).
- Phase B nao usa este filtro (sidecar opera sobre cohort fixo de
  baselines + TFT runs); por isso Phase B nao foi contaminada.

**Criterio para promocao a confirmatorio**
N/A — top-50 nao deve ser promovido. Decisoes possiveis:
1. **Remover** quando coorte for pre-declarada.
2. **Substituir** por filtro baseado em validacao (split=val).
3. **Marcar** explicitamente como `exploratory_only=True` quando aplicado, propagando flag ate `gold_model_decision_final` que deve rejeitar dados marcados como exploratorios.

**Testes unitarios necessarios**
- TODO: filtro reduz a >=50 configs e preserva o subset ranqueado.
- TODO: filtro nao se aplica se `max_configs` >= n_configs (idempotencia).

**Testes de contrato/integridade necessarios**
- TODO: contrato bloqueante: se top-50 ativo, nenhum claim confirmatorio
  pode ser derivado do DM/MCS/win-rate resultantes.
- TODO: persistir lista de configs excluidas em coluna auxiliar ou
  sidecar.

**Docs impactados se corrigido**
- TODO: `STATISTICAL_TESTS.md`, `DATA_PIPELINE_WALKTHROUGH.md` (lacuna A.65),
  `ANALYTICS_STORE_ARCHITECTURE.md`.

**Status**
`CODE_LOCATED` (arquivo, funcao e linhas confirmados por leitura direta em 2026-05-28; evidencia adicionada).

---

### PICP

**Objetivo estatistico pretendido**
Proporcao empirica de cobertura intervalar para [q10, q90] (nominal 0.80).

**Uso atual no projeto**
- Coluna `picp` em `gold_prediction_metrics_by_run_split_horizon` e
  agregados; usada por `gold_prediction_calibration`. Em
  `gold_model_decision_final`, PICP entra indiretamente como coluna agregada
  (`mean_picp_post_guardrail`) vinda de `gold_prediction_metrics_by_config`,
  nao como criterio de ordenacao nem de `academic_decision_ready`.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/quantile.py`](../../../../src/domain/services/gold_builders/quantile.py)
- Funcao: `_build_metrics_single_contract` (linhas 102-285)
- Evidencia (confirmada por leitura linha-a-linha em 2026-05-28):
  - `covered_80`: `valid["covered_80"] = ((valid["y_true"] >= valid[q10_col]) & (valid["y_true"] <= valid[q90_col])).astype(float)` (linhas 178-180)
  - `covered_80` so e computado para linhas `_prob_eligible` (Cat C filter, linhas 148-171): prediction_mode==quantile E raw q10 != raw q90
  - Linhas nao-elegiveis (point/degenerate) tem `covered_80` forcado a NaN (linhas 210-212)
  - agg: `picp=("covered_80", "mean")` (linha 244)
  - `coverage_nominal = 0.80` hard-coded: `agg["coverage_nominal"] = 0.80` (linha 260)
  - `coverage_error = picp - coverage_nominal` (linha 261)

**Definicao canonica esperada**
TODO: Christoffersen 1998 (R13) para cobertura condicional; Gneiting et al.
2007 (R12) para calibracao; Khosravi et al. 2011 (R14) para revisao.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Linha OOS.
- Ha intersecao temporal exata? N/A — metrica e por run.
- Ha controle de multiplas comparacoes? N/A.
- A perda usada corresponde a hipotese do TCC? PICP nao e perda, e
  estatistica de cobertura; para H1 a Phase B usa PICP post-guardrail
  contra bandas Tier 1/Tier 2.
- A metrica e confirmatoria, descritiva ou heuristica? Phase B usa PICP
  como confirmatoria via banda; gold legacy reporta descritivo.
- O uso atual bate com o que o nome sugere? Sim para [q10, q90]; **nao** se
  o sweep usar outros niveis (nominal hard-coded em 0.80).

**Riscos conhecidos**
- `coverage_nominal` hard-coded.
- Cobertura incondicional pode esconder clustering de falhas.
- Degeneracao quantilica (q10==q90) torna PICP nao informativo.

**Criterio para promocao a confirmatorio**
TODO. Esboco: nominal dinamico = `q_high - q_low`; reportar IC bootstrap;
adicionar teste de Christoffersen (cobertura condicional + independencia);
gate de degeneracao bloqueante.

**Testes unitarios necessarios**
- TODO: PICP em fixture com cobertura analitica conhecida.
- TODO: nominal dinamico para sweeps com `quantile_levels != [0.1, 0.5, 0.9]`.

**Testes de contrato/integridade necessarios**
- TODO: `coverage_nominal == q_high - q_low` lido do contrato.
- TODO: degeneracy_rate por (run, split, horizon) reportado em parquet
  acessivel ao `gold_model_decision_final`.

**Docs impactados se corrigido**
- TODO: `METRICS_DEFINITIONS.md`, `CALIBRATION_AND_RISK.md`,
  `DATA_PIPELINE_WALKTHROUGH.md` lacuna A.68.

**Status**
`CODE_LOCATED` (arquivo, funcao e linhas confirmados por leitura linha-a-linha em 2026-05-28; evidencia atualizada com Cat C filter e linhas exatas).

---

### MPIW

**Objetivo estatistico pretendido**
Largura media do intervalo preditivo (sharpness).

**Uso atual no projeto**
- Coluna `mpiw` em `gold_prediction_metrics_*` e
  `gold_prediction_calibration`.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/quantile.py`](../../../../src/domain/services/gold_builders/quantile.py)
- Funcao: `_build_metrics_single_contract` (linhas 102-285)
- Evidencia (confirmada por leitura linha-a-linha em 2026-05-28):
  - `pred_interval_width = q90_col - q10_col` por linha: `valid["pred_interval_width"] = valid[q90_col] - valid[q10_col]` (linha 177)
  - Sujeito ao mesmo Cat C filter que PICP: linhas nao-elegiveis tem `pred_interval_width` forcado a NaN (linhas 210-212)
  - agg: `mpiw=("pred_interval_width", "mean")` (linha 245)
  - Tambem emitido: `pred_interval_width=("pred_interval_width", "mean")` como coluna separada (linha 246)

**Definicao canonica esperada**
TODO: Gneiting-Raftery 2007 (R11), Khosravi et al. 2011 (R14). Interval
score / Winkler combina largura e cobertura.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Linha OOS.
- Ha intersecao temporal exata? N/A.
- Ha controle de multiplas comparacoes? N/A.
- A perda usada corresponde a hipotese do TCC? MPIW e estatistica
  descritiva; nao e perda.
- A metrica e confirmatoria, descritiva ou heuristica? Descritiva. Em
  Phase B reportada junto a PICP, nao como criterio isolado.
- O uso atual bate com o que o nome sugere? Sim.

**Riscos conhecidos**
- Comparar largura sem comparar cobertura recompensa modelos otimistas demais.
- MPIW=0 em quantis degenerados.
- Sem normalizacao para comparar cross-asset/escala.

**Criterio para promocao a confirmatorio**
TODO. Esboco: MPIW so entra junto a PICP via interval/Winkler score; nunca
isolado.

**Testes unitarios necessarios**
- TODO: MPIW em fixture com larguras conhecidas.
- TODO: interval score implementado (se promovido como score primario).

**Testes de contrato/integridade necessarios**
- TODO: MPIW invalido (NaN) quando degeneracy_rate alto.
- TODO: MPIW normalizado disponivel para comparacao cross-asset.

**Docs impactados se corrigido**
- TODO: `METRICS_DEFINITIONS.md`, `CALIBRATION_AND_RISK.md`.

**Status**
`CODE_LOCATED` (arquivo, funcao e linhas confirmados por leitura linha-a-linha em 2026-05-28).

---

### Pinball loss

**Objetivo estatistico pretendido**
Proper scoring rule para quantis; perda primaria para claims probabilisticos.

**Uso atual no projeto**
- Colunas `pinball_q10`, `pinball_q50`, `pinball_q90`, `mean_pinball` em
  `gold_prediction_metrics_*`.
- Phase B usa `pinball_loss_post_guardrail` como perda primaria em
  `phase_b_dm_family_6.parquet` (Emenda E1.8).

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/quantile.py`](../../../../src/domain/services/gold_builders/quantile.py)
- Funcao: `_pinball_loss` (linhas 79-82)
- Evidencia (re-verificada por leitura linha-a-linha em 2026-05-28):
  - Formula: `diff = y_true - y_pred_q; return np.maximum(q * diff, (q - 1.0) * diff)` (linhas 81-82) — equivale a `max(q*(y_true-y_pred_q), (q-1)*(y_true-y_pred_q))` ✓
  - Aplicada para q=0.1, q=0.5, q=0.9 (linhas 188-190 em `_build_metrics_single_contract`)
  - `pinball_mean_row = (q10 + q50 + q90) / 3.0` (linhas 191-193) — media simples com pesos iguais
  - agg: `mean_pinball=("pinball_mean_row", "mean")` (linha 243)
  - Pinball e calculado para TODAS as linhas elegiveis (Cat C filter aplica mascara NaN para nao-elegiveis antes do agg)

**Definicao canonica esperada**
TODO: Koenker-Bassett 1978 (R10), Gneiting 2011 (R15).

**Perguntas metodologicas**
- A unidade estatistica esta correta? Linha OOS.
- Ha intersecao temporal exata? N/A em agregacao; relevante apenas no DM.
- A perda usada corresponde a hipotese do TCC? Sim para claims
  probabilisticos (Phase B usa post-guardrail; gold legacy expoe ambas
  variantes via politica de variante quantilica).
- A metrica e confirmatoria, descritiva ou heuristica? Phase B
  confirmatoria; gold legacy descritivo enquanto DM gold roda
  squared_error.

**Riscos conhecidos**
- Quantis hard-coded q=0.1, 0.5, 0.9 podem nao bater com `quantile_levels` em sweeps futuros.
- Triplet degenerado tem pinball finito mas sem semantica probabilistica.

**Criterio para promocao a confirmatorio**
Promocao ja ocorrida no escopo Phase B (sidecar). Para gold legacy: usar como
loss primaria em DM gold corrigido (depende de §"Diebold-Mariano gold").

**Testes unitarios necessarios**
- TODO: pinball por linha contra calculo manual em fixture.
- TODO: invariancia ao reorder de quantis lidos do contrato.

**Testes de contrato/integridade necessarios**
- TODO: quantis lidos de `quantile_levels` real.
- TODO: flag de degeneracao por linha persistida.

**Docs impactados se corrigido**
- TODO: `METRICS_DEFINITIONS.md`, `STATISTICAL_TESTS.md`.

**Status**
`CODE_LOCATED` (re-verificado por leitura linha-a-linha em 2026-05-28; formula confirma; quantis q=0.1/0.5/0.9 hard-coded em L188-190, nao lidos de quantile_levels).

---

### Win-rate gold

**Objetivo estatistico pretendido**
Estatistica descritiva: proporcao de timestamps em que `loss_left < loss_right`.

**Uso atual no projeto**
- `gold_win_rate_pairwise_results` e consumido por
  `_build_model_decision_final`
  ([`confidence.py:649-694`](../../../../src/domain/services/gold_builders/confidence.py))
  apenas para gerar a coluna `win_rate_ex_ties_mean` mesclada no artefato.
- A ordenacao final do gold e por `rank_rmse, rank_mae`
  ([`confidence.py:785-818`](../../../../src/domain/services/gold_builders/confidence.py)).
- `academic_decision_ready`
  ([`confidence.py:803-813`](../../../../src/domain/services/gold_builders/confidence.py))
  depende apenas de `pairwise_ready_dm`, `pairwise_ready_mcs` e
  `target_exact_alignment` — **nao** depende de win-rate.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/pairwise.py`](../../../../src/domain/services/gold_builders/pairwise.py)
- Classe: `WinRatePairwiseResultsGoldBuilder`; `output_table = "gold_win_rate_pairwise_results"` (linha 382); `build()` (linha 385)
- Evidencia (re-verificada por leitura linha-a-linha em 2026-05-28):
  - Top-50 aplicado antes (linha 396) — usando a mesma `_select_top_configs_for_pairwise`
  - Loop de pares em linhas 409-444; cada par calcula `left_wins`, `right_wins`, `ties`
  - `left_win_rate = left_wins / n` (linha 434); `right_win_rate = right_wins / n` (linha 435)
  - `left_win_rate_ex_ties = left_wins / non_ties` (linha 436); `right_win_rate_ex_ties = right_wins / non_ties` (linha 437), onde `non_ties = max(1, left_wins + right_wins)` (linha 418)
  - Coluna `ties` presente na saida (linha 432); `tie_rate` NAO e computado diretamente (nao ha coluna tie_rate no output)
  - `split_signature` propagada condicionalmente (linha 422)

**Definicao canonica esperada**
TODO: nao e teste estatistico; para inferencia exige sign test/block
bootstrap com HAC.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Sim, timestamp pareado.
- Ha intersecao temporal exata? Herdada de `_pairwise_preprocess`.
- Ha controle de multiplas comparacoes? Nao.
- A perda usada corresponde a hipotese do TCC? Idem DM gold (squared_error).
- A metrica e confirmatoria, descritiva ou heuristica? **Descritiva** — nao
  e teste. No `gold_model_decision_final` aparece como **coluna mesclada**
  (`win_rate_ex_ties_mean`), nao como criterio de ordenacao ou prontidao;
  risco real e leitura indevida do artefato final.
- O uso atual bate com o que o nome sugere? Sim no calculo. O risco fica em
  consumidores downstream (plots, dashboards, narrativa em docs) que podem
  tratar a coluna como se fosse evidencia inferencial.

**Riscos conhecidos**
- Tratamento implicito de empates (`_ex_ties` exclui; sem coluna `tie_rate` visivel).
- Sem variancia/IC.
- Carona em top-50 e na loss `squared_error`.
- `win_rate_ex_ties_mean` aparece em `gold_model_decision_final` sem
  disclaimer; risco de plot/narrativa interpretar como evidencia
  confirmatoria.

**Criterio para promocao a confirmatorio**
N/A — manter descriptive. Se entrar em decisao, adicionar IC por block
bootstrap e teste de sinais formal.

**Testes unitarios necessarios**
- TODO: win_rate e win_rate_ex_ties em fixture controlada.
- TODO: `tie_rate` exposto.

**Testes de contrato/integridade necessarios**
- TODO: bloquear consumidores downstream (plots oficiais, docs canonicos,
  narrativa do paper) de tratar `win_rate_ex_ties_mean` como evidencia
  confirmatoria; reportar sempre acompanhado de `tie_rate` e n alinhado.
- TODO: se a coluna for usada como desempate, exigir flag explicita +
  IC por block bootstrap.

**Docs impactados se corrigido**
- TODO: `STATISTICAL_TESTS.md` (rebaixar win-rate explicitamente para
  descriptive); `DATA_PIPELINE_WALKTHROUGH.md`.

**Status**
`CODE_LOCATED` (re-verificado por leitura linha-a-linha em 2026-05-28; linhas corrigidas de ~410-445 para 409-444; nota sobre ausencia de tie_rate direto adicionada).

---

### prob_up

**Objetivo estatistico pretendido**
P(Y > 0) — probabilidade direcional. Identificacao com apenas q10/q50/q90 e
heuristica.

**Uso atual no projeto**
- Colunas `prob_up`, `prob_down` em `gold_prediction_metrics_*`.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/quantile.py`](../../../../src/domain/services/gold_builders/quantile.py)
- Funcao: `_prob_up_from_quantiles` (linhas 85-99)
- Evidencia (re-verificada por leitura linha-a-linha em 2026-05-28):
  - CDF piecewise-linear: `cdf0 = 0.1 + 0.8 * ((0.0 - q10) / safe_width)` (linha 90), clipada a [0.1, 0.9] (linha 91)
  - Hard bounds: se `q10 > 0`, `cdf0 = 0` (linha 93); se `q90 < 0`, `cdf0 = 1` (linha 94)
  - Quando width ≈ 0: `safe_width = NaN` (linha 88); fallback usa q50 como ancora: `q50 > 0 → cdf0=1.0; q50 < 0 → cdf0=0.0; q50=0 → cdf0=0.5` (linha 96) — entao `prob_up = 1 - cdf0`
  - Saida: `return pd.Series(1.0 - cdf0, ...)` (linha 99)
  - Row-level: `valid["prob_up_row"] = _prob_up_from_quantiles(valid[q10_col], valid[q50_col], valid[q90_col])` (linhas 194-196) sujeito ao Cat C filter (NaN para nao-elegiveis)
  - Aggregator: `prob_up=("prob_up_row", "mean")` (linha 247)
  - Variantes raw/post-guardrail computadas em `PredictionMetricsByRunSplitHorizonGoldBuilder` (quantile.py, linhas 288-404) e **passadas-through** por `PredictionCalibrationGoldBuilder` (descriptive.py, linhas 363-419) como colunas `prob_up_raw`, `prob_up_post_guardrail` em `gold_prediction_calibration`

**Definicao canonica esperada**
TODO: piecewise-linear CDF entre q10 e q90 nao e canonica; alternativas:
distribuicao paramétrica, grade de quantis densa, amostras preditivas.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Linha OOS.
- Ha controle de multiplas comparacoes? N/A.
- A perda usada corresponde a hipotese do TCC? N/A — prob_up nao e perda;
  Phase B nao usa prob_up para H1/H2a/H2b.
- A metrica e confirmatoria, descritiva ou heuristica? **Heuristic**.
- O uso atual bate com o que o nome sugere? **Nao** — o nome sugere
  probabilidade calibrada, mas e aproximacao CDF.

**Riscos conhecidos**
- Fallback 1/0/0.5 quando width=0 mascara degeneracao.
- Hard bounds quando q10>0 ou q90<0 forcam valores extremos sem ressalva.
- Sem calibracao empirica vs y_true>0.

**Criterio para promocao a confirmatorio**
Nao promover. Para uso operacional honesto: renomear coluna para
`prob_up_heuristic_from_q10_q90` em PR futuro; retornar NaN quando
q90 <= q10; opcional calibracao empirica reportada em coluna separada.

**Testes unitarios necessarios**
- TODO: prob_up = 0.5 quando q10 = -q90 e q50 = 0.
- TODO: prob_up = NaN quando q90 <= q10.

**Testes de contrato/integridade necessarios**
- TODO: bloquear consumo de prob_up por `gold_model_decision_final` e por
  plots oficiais sem disclaimer de heuristica.

**Docs impactados se corrigido**
- TODO: `METRICS_DEFINITIONS.md` (renomear); `DATA_PIPELINE_WALKTHROUGH.md`
  lacuna A.67.

**Status**
`CODE_LOCATED` (re-verificado por leitura linha-a-linha em 2026-05-28; evidencia expandida com fallback exato e nota sobre descriptive.py pass-through).

---

### confidence_calibrated

**Objetivo estatistico pretendido**
Score heuristico que combina cobertura e largura. **Nao e** proper score.

**Uso atual no projeto**
- Coluna `confidence_calibrated` em `gold_prediction_metrics_*` e
  `gold_prediction_calibration`.
- **Nao** e consumida por `gold_model_decision_final` —
  [`confidence.py:492-499`](../../../../src/domain/services/gold_builders/confidence.py)
  mostra que `primary_metric_map` inclui apenas `mean_pinball*`,
  `mean_picp` e `mean_mpiw` (variantes do contrato primario). O risco
  esta em docs, plots e quality checks que possam interpretar a coluna
  como metrica de calibracao estatistica.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/quantile.py`](../../../../src/domain/services/gold_builders/quantile.py)
- Funcao: `_build_metrics_single_contract` (linhas 102-285)
- Evidencia (re-verificada por leitura linha-a-linha em 2026-05-28):
  - `calibration_term = (1.0 - (agg["coverage_error"].abs() / agg["coverage_nominal"])).clip(lower=0.0, upper=1.0)` (linhas 263-265)
  - `width_term = 1.0 / (1.0 + agg["pred_interval_width"].clip(lower=0.0))` (linha 266)
  - `agg["confidence_calibrated"] = calibration_term * width_term` (linha 267)
  - `agg["coverage_error"] = agg["picp"] - agg["coverage_nominal"]` (linha 261) — depende do picp agregado e do coverage_nominal=0.80
  - Tambem passada-through por `PredictionCalibrationGoldBuilder` (descriptive.py, linhas 363-419) como `confidence_calibrated_raw` e `confidence_calibrated_post_guardrail` em `gold_prediction_calibration`

**Definicao canonica esperada**
TODO: nao tem definicao canonica equivalente; substituicao indicada e CRPS
(Gneiting-Raftery 2007), WIS (Bracher et al. 2021) ou interval score
(Gneiting-Raftery 2007).

**Perguntas metodologicas**
- A unidade estatistica esta correta? Run-level agregado.
- A perda usada corresponde a hipotese do TCC? Nao e perda; e score ad hoc.
- A metrica e confirmatoria, descritiva ou heuristica? **Heuristic**.
- O uso atual bate com o que o nome sugere? **Nao** — nome sugere calibracao estatistica.

**Riscos conhecidos**
- Dependencia da escala do target (`width_term`).
- Combinacao multiplicativa sem fundamento teorico.
- Nome induz interpretacao incorreta.

**Criterio para promocao a confirmatorio**
Nao promover. Decisao recomendada: renomear para
`heuristic_coverage_width_score`; **impedir uso como evidencia
confirmatoria em docs canonicos, plots oficiais e quality checks**
(consumidores downstream, nao `gold_model_decision_final`, que ja nao
consome a coluna). Substituir por interval/Winkler score ou CRPS para
qualquer claim confirmatorio futuro.

**Testes unitarios necessarios**
- TODO: valor analitico em fixture (coverage_error=0, pred_interval_width=0 -> score=1).
- TODO: monotonicidade local (mais cobertura -> maior score; menor largura -> maior score).

**Testes de contrato/integridade necessarios**
- TODO: enforcement nos consumidores downstream — plots oficiais,
  `living-paper`, quality checks de calibracao — para que a coluna nunca
  seja apresentada/interpretada como metrica de calibracao estatistica.
  Renomeacao da coluna (ver acima) faz parte do contrato.

**Docs impactados se corrigido**
- TODO: `METRICS_DEFINITIONS.md` (renomear "Confidence Proxy" -> "Heuristic Coverage-Width Score"); `CALIBRATION_AND_RISK.md`.

**Status**
`CODE_LOCATED` (re-verificado por leitura linha-a-linha em 2026-05-28; formula e linhas confirmadas; nota sobre pass-through em descriptive.py adicionada).

---

### VaR / ES gold

**Objetivo estatistico pretendido**
Value-at-Risk e Expected Shortfall sobre a distribuicao preditiva do alvo.

**Uso atual no projeto**
- `gold_prediction_risk.var_10`, `gold_prediction_risk.es_10_approx`.

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/confidence.py`](../../../../src/domain/services/gold_builders/confidence.py)
- Classe: `PredictionRiskGoldBuilder`; `output_table = "gold_prediction_risk"` (linha 36); `build()` (linha 39)
- Evidencia (re-verificada por leitura linha-a-linha em 2026-05-28):
  - Variaveis base: `post_q10 = "quantile_p10_post_guardrail"` (linha 65), `post_q50 = "quantile_p50_post_guardrail"` (linha 66)
  - `var_10_row = df[post_q10]` (linha 70) — identico ao q10_post_guardrail
  - `es_10_approx_row = 1.125 * df[post_q10] - 0.125 * df[post_q50]` (linha 71)
  - clip: `es_10_approx_row = np.minimum(es_10_approx_row, var_10_row)` (linha 72)
  - Tambem ha: `expected_move_row = y_pred.abs()` (linha 59) e `downside_risk_row = max(-y_pred, 0)` (linha 60) — **nao ha `max_drawdown`** (ao contrario do que a auditoria externa citou)
  - agg por 8 colunas `[run_id, asset, feature_set_name, config_signature, split, fold, seed, horizon]` (linhas 87-100); colunas `var_10=("var_10_row","mean")` e `es_10_approx=("es_10_approx_row","mean")` (linhas 108-109)
- **Discrepancia com auditoria externa (RESOLVIDA por leitura direta):**
  - Auditoria supos: VaR/ES calculados sobre `error = y_pred - y_true`; tambem cita `max_drawdown`
  - Codigo real: usa exclusivamente `quantile_p10_post_guardrail` e `quantile_p50_post_guardrail`; nao ha `max_drawdown` no builder atual
  - Explicacao: auditoria foi feita sobre `DATA_PIPELINE_WALKTHROUGH.md` (que pode descrever versao anterior), nao sobre o codigo atual

**Definicao canonica esperada**
TODO: Jorion 2007 (VaR), Acerbi-Tasche 2002 (ES), Fissler-Ziegel 2016
(elicitabilidade conjunta), Christoffersen 1998 / Kupiec 1995
(backtesting de cobertura).

**Perguntas metodologicas**
- A unidade estatistica esta correta? Linha OOS; agregado por run.
- Ha intersecao temporal exata? N/A.
- A perda usada corresponde a hipotese do TCC? Nao ha claim de risco
  financeiro no TCC ate o momento; tabela tem natureza descritiva.
- A metrica e confirmatoria, descritiva ou heuristica? **Descritiva**
  enquanto nao houver backtesting.
- O uso atual bate com o que o nome sugere? **Parcialmente** — VaR/ES sao
  calculados sobre o quantil predito do retorno, o que e razoavel como
  risco predito; falta nivel alpha explicitamente declarado e backtesting.

**Riscos conhecidos**
- Sem backtesting de excedencias.
- ES como `1.125 * q10 - 0.125 * q50` e interpolacao linear assumindo
  forma especifica de cauda; nao e ES universal.
- Nome `var_10` sugere alpha=10%; declarar isso na coluna/doc.

**Criterio para promocao a confirmatorio**
TODO. Esboco: variavel-alvo declarada (retorno ou perda; sinal explicito);
alpha explicito; backtesting de cobertura (Kupiec/Christoffersen); par
(VaR, ES) avaliado por proper scoring (Fissler-Ziegel 2016).

**Testes unitarios necessarios**
- TODO: VaR/ES em fixture com quantis conhecidos.
- TODO: violacao de monotonicidade quantilica -> NaN (atualmente clip pode mascarar).

**Testes de contrato/integridade necessarios**
- TODO: variante quantilica = post_guardrail (atualmente sim; documentar).
- TODO: schema com `alpha=0.10` explicito.

**Docs impactados se corrigido**
- TODO: `CALIBRATION_AND_RISK.md`, `METRICS_DEFINITIONS.md`,
  `DATA_PIPELINE_WALKTHROUGH.md`.

**Status**
`CODE_LOCATED` (re-verificado por leitura linha-a-linha em 2026-05-28; discrepancia com auditoria externa resolvida — codigo usa quantis post-guardrail, nao error; max_drawdown ausente no builder atual).

---

### gold_model_decision_final

**Objetivo estatistico pretendido**
Rollup final de diagnosticos por (asset, horizon). Nao deve ser tratado como
scorecard confirmatorio final sem regra de vencedor pre-registrada.

**Uso atual no projeto**
- `gold_model_decision_final` consumido por plots oficiais
  ([`generate_prediction_analysis_plots_use_case.py:819`](../../../../src/use_cases/generate_prediction_analysis_plots_use_case.py)).

**Implementacao atual localizada**
- Arquivo: [`src/domain/services/gold_builders/confidence.py`](../../../../src/domain/services/gold_builders/confidence.py)
- Funcao: `_build_model_decision_final` (linha 463); classe `ModelDecisionFinalGoldBuilder` (linha 821)
- `output_table = "gold_model_decision_final"` (linha 822)
- Evidencia (confirmada por leitura linha-a-linha em 2026-05-28):
  - `requires_gold`: linhas 824-832 (corrigido de 824-831): `gold_prediction_metrics_by_config`, `gold_prediction_robustness_by_horizon`, `gold_prediction_generalization_gap`, `gold_dm_pairwise_results`, `gold_mcs_results`, `gold_win_rate_pairwise_results`, `gold_paired_oos_intersection_by_horizon`
  - `primary_metric_map`: linhas 492-499 — inclui `mean_pinball_q10/q50/q90/{primary}`, `mean_mean_pinball_{primary}`, `mean_picp_{primary}`, `mean_mpiw_{primary}`; **nao inclui `confidence_calibrated`**
  - DM processing: linhas 588-631 (winners/losers por `pvalue_two_sided < 0.05`, nao `pvalue_adj_holm`)
  - MCS summary: linhas 633-647; renomeia `selected_in_mcs_alpha_0_05` → `mcs_selected_alpha_0_05`
  - Win-rate processing: linhas 649-694; agrega `win_rate_ex_ties_mean` por config
  - `rank_rmse`: linhas 785-789 (`mean_rmse.rank(ascending=True)`)
  - `rank_mae`: linhas 790-794 (`mean_mae.rank(ascending=True)`) — corrigido de L816
  - `academic_decision_ready`: linhas 803-813 — depende de `pairwise_ready_dm & pairwise_ready_mcs & target_exact_alignment`; **nao** depende de win-rate
  - sort_values final: linhas 815-818 por `["asset","parent_sweep_id","horizon","rank_rmse","rank_mae"]`
- **Nao consome** `gold_prediction_calibration` nem `confidence_calibrated`; calibracao entra como `mean_picp_{primary}` / `mean_mpiw_{primary}` via `gold_prediction_metrics_by_config` (L492-499)
- Win-rate entra como coluna (`win_rate_ex_ties_mean`), nao como criterio de ordenacao nem de `academic_decision_ready`
- Consumidor de plots: [`generate_prediction_analysis_plots_use_case.py:819`](../../../../src/use_cases/generate_prediction_analysis_plots_use_case.py) (confirmado por grep em 2026-05-28)

**Definicao canonica esperada**
TODO: nao ha definicao canonica; e regra de engenharia. Para uso
academico exige pre-registro do criterio. Para a conclusao do TCC, preferir o
sidecar `model_comparison_confirmatory_scorecard_phase_c.parquet`, que separa
vencedor primario de perfil comparativo dos modelos.

**Perguntas metodologicas**
- A unidade estatistica esta correta? Rollup multi-input por (asset, horizon);
  nao e teste estatistico por si so.
- Ha controle de multiplas comparacoes? Indireto via Holm gold (com gaps).
- A perda usada corresponde a hipotese do TCC? Mistura objetivos: ordenacao
  por `rank_rmse, rank_mae` (pontual) enquanto pinball/PICP/MPIW entram
  como colunas auxiliares. `confidence_calibrated` **nao** participa.
- A metrica e confirmatoria, descritiva ou heuristica? **Unknown** — herda
  status dos inputs.
- O uso atual bate com o que o nome sugere? Nome sugere decisao final
  reproduzivel a partir de gold. Implementacao depende de dezenas de
  colunas heterogeneas.

**Riscos conhecidos**
- Herda riscos dos inputs efetivamente consumidos: DM gold, MCS gold,
  Holm gold (via `pvalue_adj_holm` em DM), top-50, e os agregados de
  pinball/picp/mpiw lidos de `gold_prediction_metrics_by_config`.
- `win_rate_ex_ties_mean` aparece na tabela final como coluna; embora
  nao seja criterio de ordenacao nem de prontidao, expoe risco de
  consumidores downstream interpretarem como inferencial.
- Sem pre-registro do criterio.
- Plots oficiais (`generate_prediction_analysis_plots_use_case.py:819`)
  podem mostrar "modelo vencedor" sem disclaimer de que a ordenacao e por
  RMSE/MAE, nao por loss probabilistica primaria.
- Risco de confundir tabela de rollup com scorecard confirmatorio final. O
  scorecard final deve declarar explicitamente qual metrica elege o vencedor e
  quais metricas apenas caracterizam trade-offs.

**Criterio para promocao a confirmatorio**
TODO. Esboco: bloqueio explicito enquanto qualquer input P0 estiver em
`TODO_RESEARCH`. Promocao exige que todos os inputs P0 estejam
`PROMOTED_CONFIRMATORY` **e** criterio de combinacao esteja pre-registrado.

**Testes unitarios necessarios**
- TODO: reproduzir decisao a partir de inputs sinteticos.
- TODO: invariancia ao reorder de configs.

**Testes de contrato/integridade necessarios**
- TODO: contrato bloqueante: nao publicar decision_final como evidencia se
  qualquer input P0 nao esta `PROMOTED_CONFIRMATORY`.
- TODO: persistir `decision_criterion_hash` e `decision_criterion_version`.

**Docs impactados se corrigido**
- TODO: `ANALYTICS_STORE_ARCHITECTURE.md`, `PLOT_INTERPRETATION.md`,
  `METRICS_DEFINITIONS.md`.

**Status**
`CODE_LOCATED` (confirmado por leitura linha-a-linha em 2026-05-28; correcoes: requires_gold L824-832 nao 824-831; rank_mae em L790-794 nao L816; rank_rmse L785-789 correto; plots consumer L819 confirmado por grep).

---

### Phase B DM family-6 (referencia, nao reabrir)

**Objetivo estatistico pretendido**
6 testes DM one-sided HAC+HLN com Holm-6 (3 baselines × 2 horizontes),
unidade `target_timestamp_utc` com dedup operationally-latest cross-fold.

**Uso atual no projeto**
- `phase_b_dm_family_6.parquet` em
  `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/`.
- Evidencia confirmatoria primaria **para H2a/H2b apenas**. H1
  (calibracao) vem de `phase_b_marginal_coverage.parquet` e
  `phase_b_tier_verdict.parquet` (banda Tier 1/Tier 2 de PICP +
  cobertura marginal por quantil); o sidecar DM family-6 **nao**
  sustenta H1.

**Implementacao atual localizada**
- [`src/domain/services/dm_tft_vs_baseline.py`](../../../../src/domain/services/dm_tft_vs_baseline.py) — **caminho confirmado** por `ls` em 2026-05-28 ✓
- [`src/domain/services/holm_family_6.py`](../../../../src/domain/services/holm_family_6.py) — **caminho confirmado** ✓
- [`src/use_cases/compute_phase_b_tier_metrics_use_case.py`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py) — **caminho confirmado** ✓
- Pre-registro: `docs/06_pre_registration/phase-b/preregistration_phase_b.md` (Emenda E1.8).
- Relatorio: `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` §4.1.
- Nota C.0.1: metodologia nao auditada nesta sessao (escopo Phase B fechado); apenas caminhos verificados.

**Definicao canonica esperada**
Diebold-Mariano 1995 + HLN 1997 + Newey-West 1987 + Holm 1979.

**Perguntas metodologicas**
Ja respondidas e fixadas pelo pre-registro Phase B. C.0 nao reabre.

**Riscos conhecidos**
N efetivo em h=7 (~937 timestamps dedupados); power limitado para deltas
<1% (declarado em `B_confirmatory_2026-05-25.md` §7, item 2).

**Criterio para promocao a confirmatorio**
Ja promovido no escopo Phase B.

**Testes unitarios necessarios**
N/A em C.0; ja cobertos pelos PRs Phase B.

**Testes de contrato/integridade necessarios**
- Apenas em C.0: garantir que **nenhuma** correcao em gold legacy
  sobrescreva ou substitua o sidecar Phase B.

**Docs impactados se corrigido**
N/A.

**Status**
`PROMOTED_CONFIRMATORY` (escopo Phase B). Listado aqui para evitar
contaminacao acidental durante o hardening do gold legacy.

---

## 7. Seccao especial: Phase B sidecars vs gold legacy

| Aspecto | Phase B sidecar | Gold legacy |
|---|---|---|
| Localizacao | `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/` | `data/analytics/gold/` |
| DM tabela | `phase_b_dm_family_6.parquet` (6 rows) | `gold_dm_pairwise_results.parquet` (todos os pares pos top-50) |
| Loss | `pinball_loss_post_guardrail` | `squared_error` |
| HAC | Newey-West, lag `max(h-1, 1)` | Bartlett, lag `min(max(1, n^{1/3}), 10)` |
| Small-sample | HLN aplicado | Sem HLN |
| Tail do teste | One-sided `H_A: TFT < baseline` | Two-sided |
| Familia Holm | 6 testes (3 baselines × 2 horizontes), declarada ex-ante | groupby `(asset, parent_sweep_id, split, horizon)`; sem `split_signature` |
| Top-50 | N/A — coorte pre-declarada (15 TFT + 45 baseline) | Aplicado por default |
| Unidade estatistica | `target_timestamp_utc` cross-fold com dedup operationally-latest | `target_timestamp_utc` pivot wide com `dropna(how="any")` |
| Quem usa | `B_confirmatory_2026-05-25.md`, `phase_b_tier_verdict.parquet` | `gold_model_decision_final`, plots oficiais |

Regra geral durante C.0:

1. **Phase B sidecars sao fonte confirmatoria** em h=1 e h=7, com mapeamento
   por hipotese: H1 -> `phase_b_marginal_coverage.parquet` +
   `phase_b_tier_verdict.parquet`; H2a/H2b -> `phase_b_dm_family_6.parquet`
   + `phase_b_tier_verdict.parquet`.
2. **Gold legacy nao deve ser usado como evidencia confirmatoria** ate
   conclusao de C.0.
3. Se houver divergencia numerica entre os dois para o mesmo claim,
   prevalece o sidecar Phase B.
4. Correcoes em gold legacy nao devem sobrescrever ou apagar artefatos
   Phase B. Em particular, `analytics_archive_phase_b_20260524/` e
   read-only.
5. Documentar explicitamente em qualquer doc canonico (`STATISTICAL_TESTS.md`,
   `ANALYTICS_STORE_ARCHITECTURE.md`) que "gold legacy DM/MCS != Phase B
   family-6".

## 8. Seccao especial: heuristicas

Itens explicitamente **nao confirmatorios** sem redesign profundo:

- `prob_up` (`prob_down`)
- `confidence_calibrated`
- `win_rate` (em uso atual sem variancia/IC)

Estes itens **nao sao inuteis**. Classificacao por uso defensavel:

| Item | Uso operacional permitido | Uso descritivo permitido | Uso confirmatorio |
|---|---|---|---|
| `prob_up` | Sinal direcional em dashboard interno, com disclaimer de heuristica | Reportar distribuicao por horizonte em descricao do dataset, com NaN explicito em width=0 | **Proibido** sem CDF densa + calibracao empirica vs y_true>0 |
| `confidence_calibrated` | Score operacional para triagem visual de runs | Reportar como "heuristic coverage-width score" | **Proibido** sem substituicao por CRPS/WIS/interval score |
| `win_rate` | Apresentar lado a lado com DM/MCS em diagnostico | Tabelar com `tie_rate` explicito e n alinhado | **Proibido** como criterio de selecao sem IC/sign test |

Acoes recomendadas (para PRs futuros, fora de C.0):

- Renomear `prob_up` -> `prob_up_heuristic_from_q10_q90`.
- Renomear `confidence_calibrated` -> `heuristic_coverage_width_score`.
- Remover heuristicas como criterio de `gold_model_decision_final`.

## 9. Plano de execucao incremental

Cada fase abaixo gera um PR pequeno e tem criterio de conclusao explicito.
**Nao misturar fases no mesmo PR.**

### C.0.1 — Inventario e localizacao de codigo

- Conteudo: este arquivo (skeleton) + inspecao linha-a-linha para preencher
  campos `Implementacao atual localizada` de cada dossie ate `CODE_LOCATED`.
- Criterio de conclusao: todos os 13 itens com arquivo + funcao + linhas
  preenchidas; auditoria externa cross-checked contra codigo real.

### C.0.2 — Pesquisa primaria por item P0

- Conteudo: ler papers R1-R10, R13 e fixar definicao canonica + suposicoes
  + variantes aceitas por item P0 (DM, MCS, Holm, top-50,
  gold_model_decision_final).
- Criterio de conclusao: campos `Definicao canonica esperada` preenchidos;
  status P0 evolui para `METHOD_SPECIFIED`.

### C.0.3 — Especificacao metodologica por item P0

- Conteudo: para cada item P0, declarar:
  - loss primaria;
  - tail do teste;
  - groupby/familia formal;
  - politica de lag / B / block_len;
  - decisao sobre top-50 (remover / mover para validacao / marcar exploratorio);
  - quais decisoes derivam de pre-registro futuro.
- Criterio de conclusao: status P0 -> `GAP_CONFIRMED` para itens com gap;
  campos `Riscos conhecidos` e `Criterio para promocao a confirmatorio`
  preenchidos com decisao final.

### C.0.4 — Test plan P0

- Conteudo: preencher `Testes unitarios necessarios` e
  `Testes de contrato/integridade necessarios` por item P0 com listas
  executaveis (fixture, oracle, assertions).
- Criterio de conclusao: status P0 -> `TEST_PLAN_READY`.

### C.0.5 — Implementacao P0

- Conteudo: PRs separados por item; cada PR implementa correcao + testes do
  test plan + atualizacao dos docs canonicos (Documentation Sync Rules).
- Criterio de conclusao: status P0 -> `IMPLEMENTED` -> `VALIDATED`.
  Promocao a `PROMOTED_CONFIRMATORY` exige pre-registro futuro
  (`docs/06_pre_registration/phase-c/` ou similar).

### C.0.6 — Extensao para P1/P2

- Repete C.0.2 a C.0.5 para PICP, MPIW, pinball, VaR/ES, win-rate,
  prob_up, confidence_calibrated.
- Criterio de conclusao: cada item P1/P2 em `PROMOTED_CONFIRMATORY` ou
  formalmente em `DESCRIPTIVE_ONLY` / `HEURISTIC_ONLY` com renomeacao
  aplicada.

### C.0.7 — Scorecard final de comparacao de modelos

- Conteudo: especificar e materializar
  `model_comparison_confirmatory_scorecard_phase_c.parquet` como sidecar
  curado para a conclusao do TCC.
- Criterio de conclusao: regra de vencedor primario pre-registrada; colunas de
  perfil comparativo preenchidas; nenhum campo descritivo/heuristico usado como
  criterio confirmatorio sem status `PROMOTED_CONFIRMATORY`.

## 10. Regras de contencao de escopo

Regras explicitas para evitar que C.0 vire um projeto sem fim:

- **Nao corrigir todos os testes de uma vez.** Cada item percorre o ciclo
  isoladamente.
- **Nao promover metrica sem teste.** `PROMOTED_CONFIRMATORY` exige
  `VALIDATED`.
- **Nao misturar pesquisa bibliografica com implementacao no mesmo passo.**
  C.0.2 fecha antes de C.0.5.
- **Nao alterar claims da Phase B.** Phase B fechada; sidecars
  read-only.
- **Nao usar metricas heuristicas como inferencia estatistica.** Mesmo
  durante diagnostico exploratorio, reportar como heuristica.
- **Nao usar gold global/unscoped para decisao de modelo.** `cohort_decision`
  obrigatorio, conforme AGENT_CORE §Scope Governance.
- **Nao escrever neste arquivo afirmacoes sem evidencia.** Onde nao houver
  certeza, escrever `TODO: verificar`.
- **Nao deletar campos `TODO`** de um item ainda nao auditado para fingir
  que esta resolvido. Status formal e a unica medida de avanco.

## 11. Checklist final do arquivo

- [x] Todos os 13 itens inventariados em §4 com linha completa
- [x] Codigo localizado (arquivo, funcao, linhas) para todos os itens (C.0.1 concluido em 2026-05-28)
- [ ] Docs canonicos linkados para todos os itens
- [ ] Papers/documentacao primaria definidos para itens P0
- [ ] Papers/documentacao primaria definidos para itens P1
- [ ] Papers/documentacao primaria definidos para itens P2
- [ ] Gaps confirmados (`GAP_CONFIRMED`) para itens P0
- [ ] Test plan escrito (`TEST_PLAN_READY`) para itens P0
- [ ] Test plan escrito para itens P1
- [ ] Test plan escrito para itens P2
- [ ] Implementacao planejada por PR para cada item P0
- [ ] Validacao executada (`VALIDATED`) para cada item P0 corrigido
- [ ] Cada item promovido (`PROMOTED_CONFIRMATORY`) ou rebaixado
      (`DESCRIPTIVE_ONLY` / `HEURISTIC_ONLY`) explicitamente, com docs
      canonicos atualizados
- [ ] Renomeacao `prob_up` / `confidence_calibrated` aplicada (se C.0.6
      concluiu como HEURISTIC_ONLY)
- [ ] `gold_model_decision_final` declarado confirmatorio **ou**
      formalmente marcado como ferramenta operacional nao academica
- [ ] `model_comparison_confirmatory_scorecard_phase_c.parquet` especificado
      antes de qualquer conclusao final com "vencedor"
- [ ] Documentacao Sync Rules executada para cada PR de C.0.5/C.0.6

## 12. Cross-links

| Tema | Documento |
|---|---|
| Baseline agente IA | [`docs/ai/AGENT_CORE.md`](../../../ai/AGENT_CORE.md) |
| Indice IA | [`docs/ai/INDEX.md`](../../../ai/INDEX.md) |
| Testes estatisticos canonico | [`docs/04_evaluation/STATISTICAL_TESTS.md`](../../../04_evaluation/STATISTICAL_TESTS.md) |
| Metricas canonico | [`docs/04_evaluation/METRICS_DEFINITIONS.md`](../../../04_evaluation/METRICS_DEFINITIONS.md) |
| Calibracao e risco canonico | [`docs/04_evaluation/CALIBRATION_AND_RISK.md`](../../../04_evaluation/CALIBRATION_AND_RISK.md) |
| Explicabilidade canonico | [`docs/04_evaluation/EXPLAINABILITY.md`](../../../04_evaluation/EXPLAINABILITY.md) |
| Walkthrough pipeline | [`docs/02_data/DATA_PIPELINE_WALKTHROUGH.md`](../../../02_data/DATA_PIPELINE_WALKTHROUGH.md) |
| Pre-registro Phase B | [`docs/06_pre_registration/phase-b/preregistration_phase_b.md`](../../../06_pre_registration/phase-b/preregistration_phase_b.md) |
| Relatorio Phase B confirmatorio | [`docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md`](../phase-b/B_confirmatory_2026-05-25.md) |
| Auditoria metodologica externa | [`docs/07_reports/external-reviews/auditoria_metodologica_forecasting_financeiro.md`](../../external-reviews/auditoria_metodologica_forecasting_financeiro.md) |
| Analytics store architecture | [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../../../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md) |

## 13. C.0.1 — Log de verificacao (line-by-line)

Data da sessao de inspecao: **2026-05-28**. Todos os arquivos foram lidos integralmente antes do preenchimento dos dossiês. Nenhum arquivo em `src/` foi modificado.

---

### #1 — Diebold-Mariano gold

**Refs confirmados:**
- `pairwise.py:67-104` — funcao `_compute_dm_pairwise_from_loss_matrix`
- `pairwise.py:280` — `squared_error` calculado em `_pairwise_preprocess`
- `pairwise.py:78` — `if n < 5: continue`
- `pairwise.py:82` — lag Bartlett `int(min(max(1, n**(1/3)), 10))`
- `pairwise.py:87` — kernel Bartlett `weight = 1.0 - (k / (lag + 1))`
- `pairwise.py:92-93` — stat e pvalue two-sided
- `pairwise.py:299` — top-50 aplicado antes (em `DmPairwiseResultsGoldBuilder.build()`)
- `pairwise.py:333` — Holm aplicado dentro do builder: `_apply_holm_adjustment_for_dm(pd.concat(...))`

**Correcoes feitas:** Nenhuma nos bullets existentes. **Adicao**: nota sobre Holm aplicado em L333 (nao estava documentado no skeleton).

**Cross-check auditoria externa:** CONCORDA parcialmente. Auditoria (§3.1) descreveu corretamente a formula DM, o lag, o two-sided e o top-50. Auditoria nao citou a aplicacao do Holm dentro do builder (L333).

---

### #2 — MCS gold

**Refs confirmados:**
- `pairwise.py:107-180` — funcao `_compute_mcs_from_loss_matrix`
- `pairwise.py:110-113` — alpha=0.05, bootstrap_samples=300, block_len=5, random_seed=42
- `pairwise.py:124-130` — nested `_block_bootstrap_indices` (moving block, wrap-around)
- `pairwise.py:153` — `tr_stat = nanmax(|dbar/sqrt(var)|)`
- `pairwise.py:170` — eliminacao por `argmax(losses_mean)` (worst_local)
- `pairwise.py:351` — top-50 aplicado antes (em `McsResultsGoldBuilder.build()`)
- `pairwise.py:368-374` — split_signature propagada condicionalmente

**Correcoes feitas:** Nenhuma — todos os bullets do skeleton conferem.

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.3) identificou B=300 baixo, block_len=5 hard-coded, loss squared_error e top-50 propagar; tudo confirmado no codigo.

---

### #3 — Holm gold

**Refs confirmados:**
- `pairwise.py:183-214` — funcao `_apply_holm_adjustment_for_dm`
- `pairwise.py:191` — `group_cols = [c for c in ["asset", "parent_sweep_id", "split", "horizon"] ...]` — split_signature AUSENTE confirmado
- `pairwise.py:203-207` — formula: ordena p-values; `(m - j + 1) * pval`; `np.maximum.accumulate`; clip 1
- `pairwise.py:211-213` — `significant_adj_0_05 = pvalue_adj_holm < 0.05`

**Correcoes feitas:** Nenhuma — todos os bullets conferem.

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.2) identificou formula correta e familia potencialmente incorreta (split_signature ausente); confirmado em L191.

---

### #4 — top-50 filter

**Refs confirmados:**
- `pairwise.py:43-64` — funcao `_select_top_configs_for_pairwise`
- `pairwise.py:55` — retorno sem filtro se `cfg_count <= max_configs`
- `pairwise.py:57-61` — ranking por `squared_error.mean()` ascendente
- `pairwise.py:46` — `max_configs=50` default
- `pairwise.py:299, 351, 396` — aplicado antes de DM, MCS, win-rate respectivamente

**Correcoes feitas:** Nenhuma nos numeros existentes. Adicao: nota de cap silencioso confirmada (sem coluna indicando configs excluidas).

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.12) identificou o top-50 por test loss como risco de inferencia seletiva; confirmado em L57-61.

---

### #5 — PICP

**Refs confirmados:**
- `quantile.py:102-285` — funcao `_build_metrics_single_contract` (corpo completo)
- `quantile.py:148-171` — Cat C filter (prediction_mode==quantile AND raw p10!=p90)
- `quantile.py:178-180` — `covered_80 = (y_true >= q10_col) & (y_true <= q90_col)`
- `quantile.py:210-212` — mascara NaN para linhas nao-elegiveis
- `quantile.py:244` — agg `picp=("covered_80","mean")`
- `quantile.py:260` — `agg["coverage_nominal"] = 0.80` hard-coded
- `quantile.py:261` — `agg["coverage_error"] = agg["picp"] - agg["coverage_nominal"]`

**Correcoes feitas:** Skeleton dizia "agg em linhas ~188-261" — linhas exatas sao: row-level L178-180, agg L244, coverage_nominal L260.

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.4) identificou coverage_nominal=0.80 hard-coded e ausencia de cobertura condicional; confirmados.

---

### #6 — MPIW

**Refs confirmados:**
- `quantile.py:177` — `valid["pred_interval_width"] = valid[q90_col] - valid[q10_col]`
- `quantile.py:245` — agg `mpiw=("pred_interval_width","mean")`
- `quantile.py:246` — agg `pred_interval_width=("pred_interval_width","mean")` (coluna separada)
- Sujeito ao Cat C filter (NaN para nao-elegiveis)

**Correcoes feitas:** Skeleton dizia apenas "Linha aproximada: 245 + row-level derived columns"; adicionadas linhas exatas 177 e 246.

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.5) descreveu MPIW = mean(q90-q10); confirmado.

---

### #7 — Pinball loss

**Refs confirmados:**
- `quantile.py:79-82` — funcao `_pinball_loss`
- `quantile.py:81-82` — formula: `diff = y_true - y_pred_q; return np.maximum(q * diff, (q - 1.0) * diff)` — identica a canonica
- `quantile.py:188-190` — aplicada para q=0.1, 0.5, 0.9 (hard-coded)
- `quantile.py:191-193` — `pinball_mean_row = (q10 + q50 + q90) / 3.0`
- `quantile.py:243` — agg `mean_pinball=("pinball_mean_row","mean")`

**Correcoes feitas:** Skeleton dizia "Linha: 79; Formula ... (verifica exatamente)" — formula verificada e confirmada identica a canonica. Adicionados: linhas de aplicacao (188-190) e mean_pinball (191-193, 243).

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.6) considerou a formula correta; confirmado.

---

### #8 — win-rate gold

**Refs confirmados:**
- `pairwise.py:382` — `output_table = "gold_win_rate_pairwise_results"` (linha exata)
- `pairwise.py:385` — `def build(...)` — inicio do builder
- `pairwise.py:396` — top-50 aplicado antes
- `pairwise.py:409-444` — loop de pares; calculo de wins/ties/rates
- `pairwise.py:415-416` — `left_wins = (l < r).sum(); right_wins = (r < l).sum()`
- `pairwise.py:434-437` — `left_win_rate`, `right_win_rate`, `left_win_rate_ex_ties`, `right_win_rate_ex_ties`
- `pairwise.py:418` — `non_ties = max(1, left_wins + right_wins)`
- `pairwise.py:432` — `ties` no output; `tie_rate` NAO e calculado como coluna separada

**Correcoes feitas:** Linhas corridas de "~410-445" para "409-444"; adicionado `build()` em L385.

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.7) descreveu win_rate como descritivo auxiliar sem p-value; confirmado; nota sobre ties correto.

---

### #9 — prob_up

**Refs confirmados:**
- `quantile.py:85-99` — funcao `_prob_up_from_quantiles`
- `quantile.py:88` — `safe_width = width.where(width.abs() > 1e-12, np.nan)`
- `quantile.py:90` — `cdf0 = 0.1 + 0.8 * ((0.0 - q10) / safe_width)`
- `quantile.py:91` — `cdf0.clip(lower=0.1, upper=0.9)`
- `quantile.py:93` — hard bound: `q10 > 0 → cdf0 = 0.0`
- `quantile.py:94` — hard bound: `q90 < 0 → cdf0 = 1.0`
- `quantile.py:96` — fallback cdf0: `q50 > 0 → 1.0; q50 < 0 → 0.0; q50 = 0 → 0.5`
- `quantile.py:99` — `return 1.0 - cdf0`
- `quantile.py:194-196` — `valid["prob_up_row"] = _prob_up_from_quantiles(...)`
- `quantile.py:247` — agg `prob_up=("prob_up_row","mean")`
- `descriptive.py:363-419` — `PredictionCalibrationGoldBuilder` passa-through `prob_up_raw`, `prob_up_post_guardrail` para `gold_prediction_calibration` (L408-409)

**Correcoes feitas:** Skeleton descrevia localizacao corretamente (quantile.py). Adicoes: linhas exatas da funcao (85-99), detalhamento do fallback (cdf0, nao prob_up diretamente), confirmacao do pass-through em descriptive.py L363-419.

**Cross-check auditoria externa:** CONCORDA PARCIALMENTE. Auditoria (§3.8) descreveu CDF piecewise-linear e fallback 1/0/0.5; confirmado, porem os valores 1/0/0.5 referem-se a `cdf0` (nao a prob_up diretamente) — prob_up = 1 - cdf0, entao q50>0 → prob_up=0.0, q50<0 → prob_up=1.0. Auditoria nao identificou o descriptive.py pass-through.

---

### #10 — confidence_calibrated

**Refs confirmados:**
- `quantile.py:263-265` — `calibration_term = (1.0 - coverage_error.abs() / coverage_nominal).clip(0, 1)`
- `quantile.py:266` — `width_term = 1.0 / (1.0 + pred_interval_width.clip(lower=0.0))`
- `quantile.py:267` — `agg["confidence_calibrated"] = calibration_term * width_term`
- `descriptive.py:363-419` — pass-through para `gold_prediction_calibration` como `confidence_calibrated_raw` (L398) e `confidence_calibrated_post_guardrail` (L407)

**Correcoes feitas:** Skeleton tinha linha 267 correta. Adicoes: linhas exatas 263-265, 266; confirmacao do pass-through em descriptive.py.

**Cross-check auditoria externa:** CONCORDA. Auditoria (§3.9) descreveu corretamente a formula produto e a dependencia de escala; confirmado.

---

### #11 — VaR / ES gold

**Refs confirmados:**
- `confidence.py:35-36` — `class PredictionRiskGoldBuilder; output_table = "gold_prediction_risk"`
- `confidence.py:39` — `def build(...)`
- `confidence.py:59-60` — `expected_move_row = y_pred.abs(); downside_risk_row = max(-y_pred, 0)`
- `confidence.py:65-66` — `post_q10 = "quantile_p10_post_guardrail"; post_q50 = "quantile_p50_post_guardrail"`
- `confidence.py:70` — `var_10_row = df[post_q10]`
- `confidence.py:71` — `es_10_approx_row = 1.125 * df[post_q10] - 0.125 * df[post_q50]`
- `confidence.py:72` — `es_10_approx_row = np.minimum(es_10_approx_row, var_10_row)`
- `confidence.py:87-100` — group_cols (8 colunas)
- `confidence.py:108-109` — `var_10=("var_10_row","mean"); es_10_approx=("es_10_approx_row","mean")`
- **max_drawdown NAO existe** no builder atual

**Discrepancia VaR/ES (RESOLVIDA):** Auditoria externa afirmou que VaR/ES eram calculados sobre `error = y_pred - y_true`. O codigo atual usa exclusivamente `quantile_p10_post_guardrail` e `quantile_p50_post_guardrail`. A auditoria tambem citou `max_drawdown` que nao existe no builder atual. Conclusao: o walkthrough descrevia comportamento de versao anterior; o codigo atual usa quantis post-guardrail.

**Cross-check auditoria externa:** DIVERGE. Auditoria (§3.10) afirmou `error = y_pred - y_true` e `max_drawdown`; codigo atual usa quantis post-guardrail e nao tem max_drawdown. Discrepancia atribuida a walkthrough desatualizado.

---

### #12 — gold_model_decision_final

**Refs confirmados:**
- `confidence.py:463` — `def _build_model_decision_final(...)`
- `confidence.py:821-822` — `class ModelDecisionFinalGoldBuilder; output_table = "gold_model_decision_final"`
- `confidence.py:824-832` — `requires_gold` (correcao: skeleton dizia 824-831; linha 832 fecha o tuple)
- `confidence.py:492-499` — `primary_metric_map` (pinball, picp, mpiw; **sem confidence_calibrated**)
- `confidence.py:588-631` — DM processing (usa `pvalue_two_sided`, nao `pvalue_adj_holm`)
- `confidence.py:633-647` — MCS summary
- `confidence.py:649-694` — win-rate processing (`win_rate_ex_ties_mean`)
- `confidence.py:785-789` — `rank_rmse` (correcao: era citado como L785; rank_mae estava como L816)
- `confidence.py:790-794` — `rank_mae` (corrigido de L816)
- `confidence.py:803-813` — `academic_decision_ready` (pairwise_ready_dm & mcs & target_exact_alignment)
- `confidence.py:815-818` — `sort_values` final por rank_rmse, rank_mae
- `generate_prediction_analysis_plots_use_case.py:819` — consumidor de plots confirmado por grep

**Correcoes feitas:** requires_gold L824-831 → L824-832; rank_mae L816 → L790-794.

**Cross-check auditoria externa:** CONCORDA PARCIALMENTE. Auditoria (§3.14) identificou combinacao de DM/MCS/win-rate/gaps sem regra ex-ante; confirmado. Auditoria nao detalhou os numeros de linha.

---

### #13 — Phase B DM family-6 (referencia)

**Caminhos confirmados por `ls` em 2026-05-28:**
- `src/domain/services/dm_tft_vs_baseline.py` ✓
- `src/domain/services/holm_family_6.py` ✓
- `src/use_cases/compute_phase_b_tier_metrics_use_case.py` ✓

**Metodologia NAO auditada** (escopo Phase B fechado; status `PROMOTED_CONFIRMATORY` mantido).

**Cross-check auditoria externa:** N/A — auditoria nao cobriu o sidecar Phase B.
