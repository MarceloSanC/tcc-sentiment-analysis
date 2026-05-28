---
title: Living Paper - Results and Analysis
scope: Base textual para o Capitulo 5 do TCC (Resultados) e Results/Analysis do artigo. Consolida resultados por horizonte/split, analise de incerteza, comparacoes pareadas e investigacao de hipoteses (ex.: H1-H7 de quantis).
update_when:
  - novo resultado experimental relevante for gerado (Fase B/C)
  - hipotese investigada gerar nova evidencia
  - criterios de aceite (Gates A/B/C/D) forem atualizados
  - nova rodada confirmatoria for concluida
canonical_for: [living_paper_results, tcc_chapter_5_source, results_analysis_source, hypothesis_h1_h7]
---

# 30 - Results and Analysis

## Objetivo deste arquivo
Consolidar resultados, comparacoes e interpretacoes para TCC e artigo.

Detalhamento do objetivo:
- Centralizar resultados por horizonte, split e configuracao de experimento.
- Diferenciar desempenho pontual de desempenho probabilistico (incerteza e calibracao).
- Registrar comparacoes pareadas e evidencias de robustez estatistica sem retraining.
- Produzir interpretacoes tecnicas defensaveis, evitando afirmacoes sem suporte empirico.

Entrada e saida esperadas:
- Entrada: artefatos analiticos persistidos (silver/gold), metricas e logs de evidencias.
- Saida: secao de Resultados e Discussao do TCC e Results/Analysis do artigo.

## Status pos-Phase B confirmatoria (2026-05-25)

Este arquivo contem dois blocos de evidencia distintos:

1. **Phase B confirmatoria (cohort `phase_b_confirmatorio_20260524`)** — fecha
   H1/H2a/H2b para AAPL em h=1 e h=7 sob o protocolo sealed. Fonte primaria:
   [`B_confirmatory_2026-05-25.md`](../phase-gates/phase-b/B_confirmatory_2026-05-25.md)
   e sidecars em `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/`.
   Sintese reproduzida na proxima secao.
2. **Round 0 exploratoria + investigacao H1..H7 de crossing quantilico
   (2026-04-07/2026-04-24)** — diagnostico de pipeline, motivacao dos gates de
   qualidade e justificativa do guardrail monotonico. Permanece util como
   auditoria, **nao** sustenta claim confirmatorio. As "hipoteses H1..H7" listadas
   mais abaixo sao internas a essa investigacao de crossing e nao se confundem
   com H1/H2a/H2b/H3 do trabalho (definidas em [`00_outline.md`](00_outline.md) §7).

Limites de uso editorial:
- Round 0 entra no texto somente como exploracao de hiperparametros e
  diagnostico de pipeline; nao seleciona modelo final.
- Phase B sustenta H1/H2a/H2b apenas para a cohort, ativo, horizontes,
  candidato sealed, baselines pre-declarados e protocolo Holm-6 declarados
  ex-ante. **Nao** sustenta claim geral sobre o TFT em forecasting financeiro,
  nem sobre outros ativos, outros horizontes ou outros feature sets.
- Fase C / H3 (contribuicao de familias de features) permanece pendente; o
  texto deve marcar Capitulo 5 §5.6 como future work ate Fase C concluir.
- Comparacoes Round 0 entre feature sets nao sao apresentadas como selecao
  confirmatoria.

O snapshot silver/gold pre-2026-05-10 foi arquivado em
`data/analytics_archive_pre_phase_b/` (ver
[`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md)
§"Archive pre-Phase B"); o estado post-Phase B foi arquivado read-only em
`data/analytics_archive_phase_b_20260524/`. Ambos sao diagnostico/rastro
historico, nao evidencia confirmatoria — todo claim H1/H2a/H2b restrito a
runs da cohort `phase_b_confirmatorio_20260524`.

## Resultados confirmatorios — Phase B (cohort `phase_b_confirmatorio_20260524`)

Fonte canonica: [`B_confirmatory_2026-05-25.md`](../phase-gates/phase-b/B_confirmatory_2026-05-25.md).
Esta secao reproduz a sintese, sem recomputar nenhum numero. Veredito tier
mecanicamente derivado da politica pre-declarada
([`preregistration_phase_b.md`](../../06_pre_registration/phase-b/preregistration_phase_b.md)
§9) sobre o sidecar `phase_b_tier_verdict.parquet`.

### Parametros executados

- Cohort: `phase_b_confirmatorio_20260524`; ativo: AAPL; horizontes: h=1, h=7.
- Candidato TFT all-features sealed (config sha256
  `fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4`),
  15 runs (3 folds x 5 seeds).
- Baselines pre-declarados (15 runs cada, mesmas folds e seeds):
  `zero_return`, `historical_mean_rolling` (w=30),
  `historical_quantiles_rolling` (w=252).
- Total dim_run: 60.
- Perda primaria: `pinball_loss_post_guardrail` (Categoria A).
- N alinhado (test split, dedup operationally-latest cross-fold): 937 timestamps.

### Veredito tier por (hipotese, horizonte)

| Hipotese | h=1 | h=7 |
|---|---|---|
| H1 (calibracao probabilistica) | **Tier 2** | **Tier 1** |
| H2a (TFT > `zero_return` em pinball post-guardrail) | **Tier 1** | **Tier 1** |
| H2b (TFT > todos os baselines em pinball post-guardrail) | **Refutada** | **Refutada** |

### Sintese por hipotese

- **H1 (calibracao).** h=7 Tier 1: `cov_q10=0,112`, `cov_q50=0,505`,
  `cov_q90=0,910`, `|picp_error|=0,2%`. h=1 Tier 2: `|picp_error|=2,02%`,
  fora da banda Tier 1 ±2,0% por 0,02 ponto percentual; demais criterios
  marginais Tier 1 passam. Sustenta calibracao probabilistica do candidato
  sealed para o protocolo, com h=7 confirmatorio primario e h=1 confirmatorio
  secundario.
- **H2a (TFT > `zero_return`).** Tier 1 em h=1 e h=7. DM family-6:
  `p_adj_holm = 0,000` em ambos horizontes; `delta_pinball_rel ≈ +28,8%`
  (h=7) e `+29,9%` (h=1), ambos muito acima do threshold Tier 1 (≥ 3%).
  Sustenta dominancia do TFT sobre o baseline trivial.
- **H2b (TFT > todos os baselines).** Refutada em h=1 e h=7. Contra
  `historical_quantiles_rolling`: `p_adj_holm = 0,465` (h=1) e `0,685` (h=7),
  com `delta_pinball_rel ≈ −0,56%` e `−1,88%` (empate estatistico
  ligeiramente desfavoravel ao TFT). Vence os outros dois baselines
  ponturais com folga ampla. Resultado consistente com
  [`STRATEGIC_DIRECTION.md`](../../00_overview/STRATEGIC_DIRECTION.md) §3
  (baselines quantilicos rolling near-optimal em retornos near-random-walk
  de ativos liquidos); confirma um limite empirico declarado ex-ante, nao
  um achado negativo inesperado.

### Gate de degeneracao quantilica

`p10_eq_p90_rate = 0,000` em todos os grupos TFT quantilicos (test, val,
train, h=1 e h=7). Zero runs TFT excluidos de claim probabilistico. Sustenta
validade da variante quantilica usada em H1, H2a e H2b sob o protocolo Phase
B; nao substitui o gate de calibracao H1, que e avaliado separadamente.

### Sidecars que sustentam a decisao confirmatoria

| Hipotese | Sidecar primario |
|---|---|
| H1 | `phase_b_marginal_coverage.parquet` + `phase_b_tier_verdict.parquet` |
| H2a / H2b | `phase_b_dm_family_6.parquet` + `phase_b_tier_verdict.parquet` |

`gold_dm_pairwise_results` e `gold_mcs_results` legacy nao alimentam o
tier_verdict da Phase B. Reutilizacao desses gold legacy como evidencia
confirmatoria em rodadas futuras depende do hardening declarado em
[`C0_statistical_methods_hardening.md`](../phase-gates/phase-c/C0_statistical_methods_hardening.md).

### O que a Phase B confirma e o que nao confirma

- **Confirma (no escopo do protocolo sealed):** que o candidato TFT
  all-features escolhido produz, em AAPL, h=7, uma distribuicao preditiva
  marginalmente calibrada nas bandas Tier 1 e, em ambos horizontes, supera o
  baseline trivial `zero_return` em pinball post-guardrail com magnitude
  pratica relevante (~29%).
- **Nao confirma:** generalizacao para outros ativos, outros horizontes,
  outros feature sets, outras configuracoes de hiperparametro, outros
  baselines, outras epocas. Nao prova o limite final do desempenho do TFT em
  forecasting financeiro; rodadas futuras com escopo proprio e pre-registro
  proprio podem buscar novas evidencias.
- **Refuta (no escopo do protocolo sealed):** que o candidato sealed supere
  `historical_quantiles_rolling` em pinball post-guardrail sob a Holm-6 com
  α=0,05. Esse e o limite empirico do candidato avaliado, nao do TFT em
  geral.

### Limitacoes inerentes ao escopo Phase B

Per [`B_confirmatory_2026-05-25.md`](../phase-gates/phase-b/B_confirmatory_2026-05-25.md)
§7 e [`STRATEGIC_DIRECTION.md`](../../00_overview/STRATEGIC_DIRECTION.md) §3:

1. Single-asset (AAPL); sem claim cross-asset.
2. h=30 out-of-scope.
3. N_eff 937 (dedup cross-fold) limita poder para detectar |Δpinball| < ~1%.
4. Walk-forward sensibilidade B.4 declarada como Phase-B-bis (futuro).
5. Cross-family MCS/win-rate ausentes (`split_fingerprint` distintos entre
   TFT e baselines); DM family-6 substitui inferencia formal.
6. Bit-identity end-to-end cross-rerun nao verificada; reproducao
   estatistica via seeds fixas e suficiente para a validade declarada.

### Transicao para a Phase C

A Phase C (explicativa) e a rodada onde H3 (contribuicao de familias de
features via VSN, permutation, ablation) sera avaliada, junto com o
hardening C.0 dos testes/metricas gold legacy
([`C0_statistical_methods_hardening.md`](../phase-gates/phase-c/C0_statistical_methods_hardening.md)).
H3 nao foi respondida pela Phase B e nao pode ser inferida dos sidecars
Phase B. Reutilizar `gold_dm_pairwise_results`, `gold_mcs_results`,
`gold_model_decision_final` ou `gold_prediction_risk` como evidencia
confirmatoria em Phase C depende da conclusao de C.0.

---

## Investigacao exploratoria H1..H7 de crossing quantilico (Round 0)

A subsecao abaixo registra a investigacao exploratoria de Round 0 (sweep
`0_2_3`, AAPL) sobre causa raiz do crossing quantilico que motivou o gate
de degeneracao do Stage 11 e a politica de variante quantilica
post-guardrail. As "hipoteses H1..H7" abaixo sao internas a essa
investigacao de diagnostico e **nao** sao H1/H2a/H2b/H3 do trabalho
(reformuladas em [`00_outline.md`](00_outline.md) §7 e fechadas na Phase B).

### Validacao de hipoteses (quantis)

### Contextualizacao da investigacao
No momento desta investigacao, o projeto estava na etapa de consolidacao da avaliacao probabilistica do sweep explicito `0_2_3` (AAPL), com pipeline de treino/inferencia/analytics ja operacional e quality gate ativo. O objetivo especifico era explicar por que parte das linhas OOS ainda violava o contrato quantilico (`p10 <= p50 <= p90`) e como corrigir isso sem comprometer rigor estatistico.

O que estava sendo testado:
- consistencia de extracao e parser de quantis no codigo;
- consistencia de configuracao entre treino e inferencia;
- comportamento real do modelo em OOS (crossing e calibracao);
- sensibilidade a repeticoes/otimizacao e efeito de regime (OOD/volatilidade).

### Problemas observados
1. Violacoes de ordenacao quantilica em parte das previsoes OOS.
- Escopo `0_2_3`: 421 violacoes em 674.820 linhas (0,0624%), em 70/675 runs.

2. Degradacao probabilistica localizada.
- Runs com crossing apresentaram pior erro medio OOS (`rmse`/`mae`) que runs sem crossing.

3. Sensibilidade entre repeticoes de algumas configuracoes.
- Variabilidade da taxa de crossing entre seeds/folds em subgrupos especificos.

4. Evidencia de piora em regime mais OOD/volatil.
- Maior taxa de violacao em meses/timestamps com proxy de deslocamento de distribuicao e alta volatilidade local.

Com base nisso, as hipoteses H1..H7 abaixo foram usadas para identificar causa raiz e orientar mitigacoes.

### H1. Mapeamento errado na extracao de quantis (ordem do tensor interpretada incorretamente)
- Como validado:
  - Inspecao de `src/adapters/pytorch_forecasting_tft_inference_engine.py` e `src/adapters/pytorch_forecasting_tft_trainer.py`.
  - Confirmacao de extracao por proximidade de nivel (`argmin(abs(q_levels - alvo))`), e nao por posicao fixa.
  - No `0_2_3`, padrao esparso (nao massivo/sistematico).
- Conclusao:
  - Hipotese refutada como causa principal atual.

### H2. `quantile_levels` inconsistente entre treino e inferencia
- Como validado:
  - `fact_config` no `0_2_3` com `quantile_levels_json=[0.1,0.5,0.9]` em 675/675 runs.
  - Checagem de `config.json` dos 70 runs com violacao sem divergencias.
  - `LocalTFTInferenceModelLoader`: consistencia entre `training_config.quantile_levels` e `model.loss.quantiles`.
- Conclusao:
  - Hipotese refutada no escopo atual.

### H3. Crossing real de quantile regression
- Como validado:
  - Medicao de crossing por run e associacao com erro pontual OOS.
- Evidencia:
  - 421/674.820 violacoes (0,0624%), em 70/675 runs.
  - Runs com crossing tiveram pior erro medio de teste:
    - `test_rmse`: 0,020796 vs 0,018696
    - `test_mae`: 0,014826 vs 0,013536
  - Correlacao positiva entre taxa de crossing e erro OOS (`~0,145` com RMSE/MAE).
- Conclusao:
  - Hipotese suportada.

### H4. Instabilidade numerica/otimizacao
- Como validado:
  - Associacao de crossing com `learning_rate`, `batch_size` e proxies de estabilidade de treino.
  - Variabilidade por grupos de hiperparametros (5 seeds x 3 folds).
- Evidencia:
  - Sem suporte para narrativa simplificada "LR alto/treino curto/batch pequeno".
  - Componente real de sensibilidade entre repeticoes em alguns grupos (std de crossing ate ~0,02244; media ~0,00177).
- Conclusao:
  - Hipotese parcialmente suportada.

### H5. Problema de escala/inversao no pos-processamento
- Como validado:
  - Revisao de `_apply_scalers`, persistencia OOS e busca de `inverse_transform` em caminho de quantis.
- Evidencia:
  - Nao ha inverse transform aplicado em `quantile_p10/p50/p90` antes da gravacao.
- Conclusao:
  - Hipotese refutada como causa principal atual.

### H6. Mistura de horizonte/split no parser
- Como validado:
  - Revisao da montagem/persistencia por horizonte no trainer.
  - Checagem de violacoes por `horizon` e `max_prediction_length`.
- Evidencia:
  - `421/421` violacoes em `horizon=1`; zero em `horizon>1` no conjunto analisado.
- Conclusao:
  - Hipotese refutada para o problema atual.

### H7. Dados com regime shift forte
- Como validado:
  - Concentracao temporal de violacoes.
  - Proxy OOD: `z_train = |y_true - media_treino|/desvio_treino`.
  - Proxy de regime: volatilidade local rolling (21d).
- Evidencia:
  - Maior taxa de violacao em meses de 2019-2020 e em regioes mais OOD/volateis.
  - Ex.: `high_vol=0,000813` vs `low_vol=0,000542`.
- Conclusao:
  - Hipotese parcialmente suportada (evidencia correlacional, nao causal unica).

### Acoes corretivas (mapeadas por problema e hipoteses)

#### Problema A - Violacao do contrato quantilico (ordem/largura)
- Hipoteses relacionadas: H3 (suportada), H4 (parcial), H7 (parcial).
- Acoes:
1. Treino: introduzir penalizacao/parametrizacao monotona de quantis para reduzir crossing na origem.
2. Inferencia: aplicar guardrail monotono final (`p10<=p50<=p90`) com auditoria before/after (`crossing_rate`, `PICP`, `MPIW`, `pinball`).
3. Gate: elevar checks de ordem e largura para bloqueio de release quando violados acima do limiar definido.

#### Problema B - Instabilidade entre repeticoes (seed/fold/config)
- Hipoteses relacionadas: H4 (parcial).
- Acoes:
1. Selecao de modelo: priorizar ranking por robustez (media + IC95 + variancia), nao apenas melhor media de erro.
2. Protocolo finalista: rerun adicional dos candidatos finais para confirmar estabilidade probabilistica.
3. Governanca: penalizar no ranking configuracoes com variabilidade alta de crossing.

#### Problema C - Degradacao em regime OOD/volatil
- Hipoteses relacionadas: H7 (parcial), H3 (suportada).
- Acoes:
1. Recalibracao por horizonte em janela recente (rolling/expanding) para cobertura mais aderente ao regime atual.
2. Monitor de drift/regime com gatilho de retreino/recalibracao.
3. Validacao segmentada por regime no relatorio final (evitar mascaramento por media agregada).

#### Problema D - Risco de regressao de pipeline (causas refutadas)
- Hipoteses relacionadas: H1, H2, H5, H6 (refutadas no escopo atual).
- Acoes:
1. Manter hardening de parser/config com testes automatizados dedicados.
2. Incluir checks de consistencia treino-inferencia no load do modelo.
3. Preservar rastreabilidade no living-paper para reavaliar rapidamente se o contexto mudar.



### Criterios quantitativos de aceite (status de gate)

Status atual: `provisorio`.

Criticos para promocao:
- Gate A (contrato quantilico):
  - `crossing_bruto_rate <= 0.10%` (meta oficial futura: `<=0.05%`)
  - `crossing_pos_guardrail_rate = 0`
  - `negative_interval_width = 0`
- Gate B (impacto do guardrail; thresholds semânticos relativos definidos em
  `docs/04_evaluation/CALIBRATION_AND_RISK.md` §"Bandas de aceitacao";
  magnitudes calculadas a partir das colunas absolutas
  `delta_<metrica>_post_minus_raw` do `gold_prediction_metrics_by_run_split_horizon`):
  - `delta_pinball_rel <= +1.0%` (computado como `delta_mean_pinball_post_minus_raw / mean_pinball_raw`)
  - `abs(delta_picp) <= 0.02` (computado a partir de `delta_picp_post_minus_raw`)
  - `delta_mpiw_rel <= +5.0%` (computado como `delta_mpiw_post_minus_raw / mpiw_raw`)
- Gate C (variabilidade relevante, H4):
  - grupo instavel quando `std_invalid_rate > 0.005`
  - reprovar familia quando `>20%` dos grupos forem instaveis
- Gate D (regime/OOD, H7):
  - confirmar efeito em teste controlado com `p<0.05` e sinal consistente

Condicao para status `oficial`:
- 2 rodadas consecutivas aprovando Gates A/B/C/D e evidencias H3/H4/H7 completas no `evidence_log`.

### Sintese executiva
- Causas principais do problema atual: crossing real (H3) com componente de instabilidade (H4) e efeito de regime (H7).
- Causas investigadas e descartadas neste escopo: H1, H2, H5, H6.
- Direcao de correcao: atuar simultaneamente em modelagem (origem), guardrails de inferencia (contrato), e governanca estatistica (robustez por repeticao/regime).
