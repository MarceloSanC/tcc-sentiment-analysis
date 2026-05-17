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

## Status apos pivo metodologico de 2026-04-23/2026-04-25

Este arquivo contem evidencia exploratoria de diagnostico probabilistico da
Round 0, especialmente sobre crossing quantilico e qualidade de artefatos. Esses
achados permanecem uteis como auditoria e motivacao metodologica, mas nao devem
ser usados como evidencia confirmatoria de superioridade do modelo final.

Para o TCC/artigo:
- Round 0 entra como exploracao de hiperparametros, diagnostico de pipeline e
  justificativa dos gates de qualidade.
- Fase B deve preencher os resultados confirmatorios do candidato all-features
  pruned contra o baseline primario pre-declarado e baselines complementares.
- Fase C deve preencher a analise de contribuicao/explicabilidade via VSN,
  permutation importance e ablation explicativa.
- Comparacoes antigas entre feature sets nao devem ser apresentadas como selecao
  confirmatoria de modelo.

Adicionalmente, o snapshot silver/gold pré-2026-05-10 foi arquivado em
`data/analytics_archive_pre_phase_b/` (ver
`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` §"Archive pre-Phase B").
Esse archive é diagnóstico histórico, não evidência confirmatória — todo claim
H1/H2a/H2b/H3 do Capítulo 5 é restrito a runs gerados após o reset, com
`parent_sweep_id` da coorte confirmatória.

## Estrutura sugerida
- Configuracoes avaliadas
- Resultados principais por horizonte/split
- Analise de incerteza (quantis, cobertura)
- Comparacoes estatisticas (ex.: DM/MCS, quando aplicavel)
- Interpretabilidade e implicacoes praticas

## Validacao de hipoteses (quantis)

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
