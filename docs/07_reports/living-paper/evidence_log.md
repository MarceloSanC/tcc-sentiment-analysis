---
title: Evidence Log
scope: Log rastreavel de evidencias empiricas do projeto, com template padronizado (Data, Questao/Hipotese, Experimento, Configuracao-chave, Resultado, Interpretacao, Artefatos, Uso no texto). Toda afirmacao relevante no TCC/artigo deve ter entrada aqui.
update_when:
  - evidencia empirica nova for gerada (entrada nova)
  - hipotese for refutada/suportada (entrada de fechamento)
  - resultado experimental novo precisar de rastro auditavel
canonical_for: [evidence_log, empirical_evidence, hypothesis_tracking, experimental_traceability]
---

# Evidence Log

Objetivo: registrar evidencias de forma rastreavel para suportar afirmacoes no TCC e no artigo.

Detalhamento do objetivo:
- Vincular cada afirmacao relevante a uma evidencia observavel e reproduzivel.
- Registrar contexto de experimento (configuracao, periodo, ativo, split, horizonte) para auditoria.
- Facilitar escrita de resultados, discussoes e limitacoes com base em fatos.
- Evitar conclusoes baseadas apenas em memoria ou interpretacao nao documentada.

## Template de entrada
- Data:
- Questao/Hipotese:
- Experimento/Execucao:
- Configuracao-chave:
- Resultado observado:
- Interpretacao:
- Artefatos (paths):
- Uso no texto (TCC/Artigo/Ambos):

## Contexto da investigacao atual (2026-04-07)

Durante a rodada de analise do sweep explicito `0_2_3` (AAPL), o quality gate e as tabelas de avaliacao probabilistica indicaram inconsistencias em parte das previsoes quantilicas OOS, principalmente:

- violacoes de ordenacao de quantis (`quantile_p10 <= quantile_p50 <= quantile_p90`) em um subconjunto de linhas;
- poucos casos de largura de intervalo negativa (`quantile_p90 - quantile_p10 < 0`).

Neste ponto do projeto, os fluxos de treino, inferencia rolling e refresh analytics ja estavam operacionais e com artefatos gold/silver sendo gerados. A investigacao abaixo foi aberta para determinar se a causa estava em problema de contrato/layout da extracao de quantis ou em inconsistencias de configuracao entre treino e inferencia.

Hipoteses levantadas para explicar a causa raiz:

1. Mapeamento errado na extracao dos quantis (ordem de tensor interpretada incorretamente).
2. `quantile_levels` inconsistente entre treino e inferencia (lista/ordem divergente no artefato carregado).
3. Crossing real de quantile regression (quantis se cruzam em parte das amostras).
4. Instabilidade numerica/otimizacao (sensibilidade a seed/hiperparametros/regime).
5. Problema de escala/inversao no pos-processamento (desnormalizacao alterando ordenacao dos quantis).
6. Mistura de horizonte/split no parser (indices de matrizes atribuidos ao alvo errado).
7. Dados com regime shift forte (janelas fora da distribuicao de treino degradam o quantile head antes do ponto medio).

### Entradas
- Data: 2026-04-07
- Questao/Hipotese: H1 - Mapeamento errado na extracao dos quantis (tensor em uma ordem, codigo interpretando em outra).
- Experimento/Execucao:
  - Inspecao de codigo na inferencia e treino:
    - `src/adapters/pytorch_forecasting_tft_inference_engine.py`
    - `src/adapters/pytorch_forecasting_tft_trainer.py`
  - Checagem de dados no silver para sweep `0_2_3`:
    - `fact_oos_predictions` + `dim_run`.
- Configuracao-chave:
  - Regra de extracao observada no codigo: indices de p10/p50/p90 escolhidos por proximidade dos niveis (`argmin(abs(q_levels - alvo))`), nao por posicao fixa.
  - Escopo de dados analisado: runs com `parent_sweep_id` iniciando em `0_2_3_`.
- Resultado observado:
  - Em `0_2_3`: 421 linhas com ordem invalida de quantis, distribuidas em 70 runs.
  - Padrao observado esparso (nao massivo/sistematico).
- Interpretacao:
  - Se houvesse troca fixa de ordem (ex.: salvar q90 como p10), o problema apareceria de forma massiva na maior parte das linhas/runs.
  - A implementacao atual de mapeamento por nivel reduz risco de erro por layout/ordem de tensor.
  - H1 foi refutada como causa principal atual.
- Conclusao:
  - Status: refutada.
  - A causa principal atual nao e mapeamento errado de indices de quantis.
- Possiveis correcoes/melhorias:
  - Manter teste de contrato para garantir extracao por nivel (0.1/0.5/0.9) e nao por posicao fixa.
  - Incluir assertiva explicita de ordem dos niveis no ponto de extracao, com log de diagnostico em caso de divergencia.
  - Preservar monitor de taxa de crossing por run para detectar regressao de parser.
- Artefatos (paths):
  - `src/adapters/pytorch_forecasting_tft_inference_engine.py`
  - `src/adapters/pytorch_forecasting_tft_trainer.py`
  - `data/analytics/silver/fact_oos_predictions/asset=AAPL/**`
  - `data/analytics/silver/dim_run/asset=AAPL/**`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.

- Data: 2026-04-07
- Questao/Hipotese: H2 - `quantile_levels` inconsistente entre treino e inferencia (lista/ordem diferente ou metadado divergente em checkpoint/config).
- Experimento/Execucao:
  - Checagem de consistencia em dados persistidos:
    - `fact_config.quantile_levels_json` para `0_2_3`.
  - Checagem por artefato em runs com erro de ordem:
    - leitura de `config.json` dos 70 `model_version` associados aos runs invalidos.
  - Checagem de carregamento real de modelo:
    - `LocalTFTInferenceModelLoader().load(...)` e comparacao entre `training_config.quantile_levels` e `model.loss.quantiles`.
- Configuracao-chave:
  - Escopo: sweep `0_2_3`, ativo `AAPL`.
  - Fluxo atual de inferencia usa `model.loss.quantiles` para extracao.
- Resultado observado:
  - `fact_config` em `0_2_3`: `quantile_levels_json = [0.1, 0.5, 0.9]` em 675/675 linhas (1 valor distinto).
  - Runs com erro de ordem: 70 artefatos checados, 0 com `quantile_levels` fora de `[0.1, 0.5, 0.9]`.
  - Exemplo de carregamento real:
    - `training_config.quantile_levels = [0.1, 0.5, 0.9]`
    - `model.loss.quantiles = [0.1, 0.5, 0.9]`
    - `prediction_mode = quantile`
- Interpretacao:
  - Nao foi encontrada evidencia de divergencia treino vs inferencia em `quantile_levels` no escopo atual.
  - H2 foi refutada como causa principal atual.
- Conclusao:
  - Status: refutada.
  - Nao ha inconsistencia de `quantile_levels` entre treino e inferencia no escopo analisado.
- Possiveis correcoes/melhorias:
  - Validar na carga do modelo: `training_config.quantile_levels == model.loss.quantiles`, falhando cedo em caso de mismatch.
  - Persistir `quantile_levels` no metadata do run e no payload de inferencia para rastreabilidade.
  - Adicionar check automatizado no quality gate para detectar divergencia entre configuracao persistida e checkpoint carregado.
- Artefatos (paths):
  - `src/adapters/local_tft_inference_model_loader.py`
  - `src/adapters/pytorch_forecasting_tft_inference_engine.py`
  - `data/analytics/silver/fact_config/asset=AAPL/**`
  - `data/models/AAPL/**/config.json`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.


- Data: 2026-04-07
- Questao/Hipotese: H3 - Crossing real de quantile regression (o modelo aprendeu quantis que se cruzam em parte das amostras).
- Experimento/Execucao:
  - Escopo: runs `0_2_3_*` para AAPL.
  - Analise por run da taxa de violacao (`p10>p50` ou `p50>p90`) e associacao com erro pontual (RMSE/MAE OOS).
  - Checagem de efeito de tamanho amostral por run.
- Configuracao-chave:
  - Universo: 674.820 linhas OOS; 421 violacoes (0,0624%) em 70/675 runs.
- Resultado observado:
  - Runs com crossing tiveram pior erro de teste medio:
    - `test_rmse`: 0,020796 (com crossing) vs 0,018696 (sem crossing)
    - `test_mae`: 0,014826 (com crossing) vs 0,013536 (sem crossing)
  - Correlacao positiva (fraca a moderada) entre taxa de crossing e erro de teste:
    - corr(`invalid_rate`,`test_rmse`) ~ 0,1449
    - corr(`invalid_rate`,`test_mae`) ~ 0,1464
  - Nao houve diferenca relevante de tamanho amostral por run (`n` medio praticamente igual).
- Interpretacao:
  - Evidencia suporta H3 como mecanismo real presente: crossing quantilico ocorre no output do modelo e associa-se a piora de erro OOS.
  - O efeito e localizado (baixa taxa global), mas consistente em subgrupos de runs.
- Conclusao:
  - Status: suportada.
  - Existe crossing real de quantile regression no output do modelo, com associacao a pior erro OOS.
- Possiveis correcoes/melhorias:
  - Melhorar treinamento probabilistico: ajustar regularizacao/capacidade e priorizar calibracao por horizonte.
  - Avaliar perda com penalizacao de crossing ou parametrizacao monotona dos quantis.
  - Aplicar rearranjo monotono como guardrail de saida apenas com auditoria before/after (taxa crossing, PICP, MPIW, pinball).
  - Introduzir monitor por regime/periodo para identificar concentracao de crossing em janelas especificas.
- Artefatos (paths):
  - `data/analytics/silver/fact_oos_predictions/asset=AAPL/**`
  - `data/analytics/silver/fact_split_metrics/asset=AAPL/**`
  - `data/analytics/silver/dim_run/asset=AAPL/**`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.


- Data: 2026-04-07
- Questao/Hipotese: H4 - Instabilidade numerica/otimizacao (LR alto, treino curto, seed sensivel, batch pequeno).
- Experimento/Execucao:
  - Escopo: runs `0_2_3_*` para AAPL.
  - Analise de associacao entre crossing por run e hiperparametros (`learning_rate`, `batch_size`).
  - Analise de proxies de estabilidade de treino (`last_epoch`, `best_epoch`, `overfit_gap_last`).
  - Analise de sensibilidade por seeds/folds para mesmo grupo de hiperparametros (config canônica sem seed/split metadata).
- Configuracao-chave:
  - Agrupamento por 45 grupos de configuracao (15 execucoes por grupo = 5 seeds x 3 folds).
- Resultado observado:
  - Nao houve suporte para sub-hipoteses literais "LR alto" e "treino curto":
    - crossing maior em quartis mais baixos de LR (nao nos mais altos);
    - runs com crossing tiveram, em media, mais epocas executadas (nao menos).
  - Nao houve suporte para "batch pequeno": runs com crossing tenderam a batch maior (mediana 128).
  - Houve suporte para componente de sensibilidade/instabilidade entre repeticoes:
    - desvio-padrao medio da taxa de crossing entre seeds/folds por grupo ~ 0,00177;
    - alguns grupos (especialmente em `S`) com variabilidade alta (std ate ~0,02244).
- Interpretacao:
  - H4 fica parcialmente suportada: existe sensibilidade de estabilidade entre repeticoes, mas os gatilhos simplificados (LR alto/treino curto/batch pequeno) nao explicam os dados observados neste escopo.
- Conclusao:
  - Status: parcialmente suportada.
  - Ha instabilidade entre repeticoes (seed/fold/config), mas nao sustentada pelos gatilhos simplificados testados.
- Possiveis correcoes/melhorias:
  - Reduzir sensibilidade: aumentar robustez de selecao por estabilidade (mediana/IC95 e criterio de consistencia entre seeds).
  - Revisar espaco de busca com foco em combinacoes estaveis (nao apenas melhor media de erro).
  - Adotar protocolo de retreino para candidatos finais com repeticoes extras e auditoria de variancia.
  - Registrar no ranking final penalidade para configuracoes com alta variabilidade de crossing.
- Artefatos (paths):
  - `data/analytics/silver/fact_config/asset=AAPL/**`
  - `data/analytics/silver/fact_epoch_metrics/asset=AAPL/**`
  - `data/analytics/silver/fact_split_metrics/asset=AAPL/**`
  - `data/analytics/silver/fact_oos_predictions/asset=AAPL/**`
  - `data/analytics/silver/dim_run/asset=AAPL/**`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.

- Data: 2026-04-07
- Questao/Hipotese: H5 - Problema de escala/inversao no pos-processamento (desnormalizacao/reversao de scaler invertendo relacao entre quantis).
- Experimento/Execucao:
  - Inspecao de codigo de escala e persistencia:
    - `src/use_cases/train_tft_model_use_case.py`
    - `src/use_cases/run_tft_inference_use_case.py`
    - `src/adapters/pytorch_forecasting_tft_trainer.py`
  - Busca por uso de `inverse_transform`/desnormalizacao aplicada em `y_pred`/`quantile_*`.
- Configuracao-chave:
  - Em inferencia, `_apply_scalers` protege `target_return` e colunas estruturais (`time_idx`, `timestamp`, `asset_id`), aplicando scaler apenas nas features de entrada.
  - Quantis OOS sao persistidos diretamente dos tensores de saida do modelo (sem etapa de inverse transform no pos-processamento).
- Resultado observado:
  - Nao foi encontrado caminho de codigo que aplique `inverse_transform` em `quantile_p10/p50/p90` antes de persistir OOS.
  - O padrao de violacao e esparso, nao caracterizando inversao global apos desnormalizacao.
- Interpretacao:
  - A hipotese de inversao por pos-processamento de escala foi refutada como causa principal atual.
  - O problema remanescente e mais consistente com crossing pontual dos quantis produzidos pelo modelo.
- Conclusao:
  - Status: refutada.
  - Nao ha evidencia de inversao dos quantis causada por desnormalizacao/pos-processamento no pipeline atual.
- Possiveis correcoes/melhorias:
  - Manter cobertura de testes para protecao das colunas de alvo/quantis contra transformacoes indevidas.
  - Incluir validacao de integridade no ponto de persistencia OOS (ordem dos quantis e largura de intervalo).
  - Fortalecer logs de debug no caminho de escalonamento para facilitar investigacoes futuras.
- Artefatos (paths):
  - `src/use_cases/train_tft_model_use_case.py`
  - `src/use_cases/run_tft_inference_use_case.py`
  - `src/adapters/pytorch_forecasting_tft_trainer.py`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.

- Data: 2026-04-07
- Questao/Hipotese: H6 - Mistura de horizonte/split no parser (saida multi-horizonte/multi-head lida com indice errado).
- Experimento/Execucao:
  - Inspecao da montagem e persistencia de matrizes por horizonte:
    - selecao de horizontes no trainer (`selected_horizons`, `selected_idx`).
    - persistencia em `_persist_fact_oos_predictions` usando o mesmo `h_idx` para `y_true`, `y_pred`, `q10`, `q50`, `q90`.
  - Validacao empirica no silver:
    - distribuicao das violacoes por `horizon`.
    - cruzamento com `fact_config.max_prediction_length`.
- Configuracao-chave:
  - Em `0_2_3`, os runs com violacao estao em modelos `max_prediction_length=1`.
- Resultado observado:
  - `421/421` violacoes em `horizon=1`.
  - `0` violacoes em `horizon>1`.
  - `0` violacoes em runs com `max_prediction_length>1` (no conjunto analisado).
- Interpretacao:
  - A hipotese de mistura de horizonte/split no parser foi refutada para o problema atual observado.
  - Como hardening, manter teste dedicado para cenarios multi-horizonte (quando houver runs com `max_prediction_length>1` em producao analitica).
- Conclusao:
  - Status: refutada.
  - Nao ha evidencia de erro de parser por mistura de horizonte/split no problema observado.
- Possiveis correcoes/melhorias:
  - Manter testes unitarios/integracao especificos para alinhamento `h_idx` entre `y_true`, `y_pred`, `q10`, `q50`, `q90`.
  - Incluir validacao de sanidade por horizonte no quality gate para execucoes multi-horizonte.
  - Expandir suite com caso sintetico multi-head para prevenir regressao silenciosa.
- Artefatos (paths):
  - `src/adapters/pytorch_forecasting_tft_trainer.py`
  - `src/use_cases/train_tft_model_use_case.py`
  - `data/analytics/silver/fact_oos_predictions/asset=AAPL/**`
  - `data/analytics/silver/fact_config/asset=AAPL/**`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.


- Data: 2026-04-07
- Questao/Hipotese: H7 - Dados com regime shift forte (janelas fora da distribuicao de treino degradam o quantile head antes do ponto medio).
- Experimento/Execucao:
  - Escopo: runs `0_2_3_*` para AAPL.
  - Checagem 1 (concentracao temporal): taxa de violacao por mes (`target_timestamp_utc`).
  - Checagem 2 (proxy OOD por run): `z_train = |y_true - media_treino|/desvio_treino` usando janelas de treino por `run_id` (via `fact_run_snapshot`) e `target_return` do `dataset_tft`.
  - Checagem 3 (proxy de regime): taxa de violacao em timestamps de alta volatilidade local (rolling std 21d no `y_true` agregado por timestamp).
- Configuracao-chave:
  - Universo analisado: 674.820 linhas OOS no escopo `0_2_3`.
  - Violacoes observadas: 421 (taxa 0,0624%), concentradas em `horizon=1`.
- Resultado observado:
  - Concentracao temporal em meses de 2019-2020 (ex.: 2020-12, 2020-10, 2020-11), com taxa mensal superior ao baseline.
  - OOD proxy: `z_train` maior nas linhas invalidas (media ~0,971) vs validas (media ~0,818).
  - Taxa de violacao maior em amostras mais OOD:
    - `z>=2.0`: 0,000926 vs 0,000599
    - `z>=2.5`: 0,001073 vs 0,000606
    - `z>=3.0`: 0,000913 vs 0,000617
  - Regime proxy: taxa em alta volatilidade local maior que baixa volatilidade:
    - high_vol: 0,000813
    - low_vol: 0,000542
- Interpretacao:
  - Evidencia consistente com degradacao do componente quantilico em janelas mais fora da distribuicao de treino e em regime mais volatil.
  - Hipotese H7 fica parcialmente suportada (associacao observada), sem prova de causalidade unica.
- Conclusao:
  - Status: parcialmente suportada.
  - Ha associacao entre regime mais OOD/volatil e maior taxa de crossing, sem isolar causalidade unica.
- Possiveis correcoes/melhorias:
  - Recalibrar quantis por horizonte em janela recente (rolling/expanding) para reduzir misscoverage em mudanca de regime.
  - Adotar monitoramento de drift/regime com gatilhos de retreino/recalibracao.
  - Considerar ponderacao temporal ou treino com maior enfase em periodos recentes para melhorar adaptacao de caudas.
  - Segmentar analise de qualidade probabilistica por regime para evitar conclusoes agregadas enganosas.
- Artefatos (paths):
  - `data/analytics/silver/fact_oos_predictions/asset=AAPL/**`
  - `data/analytics/silver/dim_run/asset=AAPL/**`
  - `data/analytics/silver/fact_run_snapshot/asset=AAPL/**`
  - `data/processed/dataset_tft/AAPL/**`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.


- Data: 2026-04-07
- Questao/Hipotese: Decisao tecnica apos H7 (mitigacao de crossing quantilico).
- Experimento/Execucao:
  - Sintese de alternativas para reduzir crossing sem perder rastreabilidade estatistica.
- Configuracao-chave:
  - Estrategia em 2 camadas: (1) melhoria no treino/calibracao, (2) guardrail de saida.
- Resultado observado:
  - Solucao proposta A (origem): restricao/parametrizacao monotona + calibracao probabilistica por horizonte.
  - Solucao proposta B (saida): rearranjo monotono com auditoria de impacto antes/depois.
- Interpretacao:
  - A combinacao A+B reduz crossing estrutural e garante contrato final de quantis em producao.
- Artefatos (paths):
  - `docs/07_reports/living-paper/30_results_and_analysis.md`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.


- Data: 2026-04-08
- Questao/Hipotese: Definicao de criterios quantitativos de aceite para fechamento das hipoteses H3/H4/H7 no quality gate.
- Experimento/Execucao:
  - Consolidacao dos achados de H3 (crossing), H4 (instabilidade) e H7 (regime/OOD).
  - Definicao de thresholds operacionais para promocao de modelo/sweep com status de gate `provisorio` -> `oficial`.
- Configuracao-chave:
  - Gate A: contrato quantilico (`crossing_bruto_rate`, crossing pos-guardrail, largura negativa).
  - Gate B: impacto before/after do guardrail (`delta_pinball`, `delta_picp`, `delta_mpiw`).
  - Gate C: variabilidade relevante por grupo (`std_invalid_rate > 0.005`, limite de grupos instaveis >20%).
  - Gate D: teste controlado para regime/OOD com `p<0.05` e sinal consistente.
- Resultado observado:
  - Criterios documentados para decisao tecnica com thresholds explicitos.
  - Status inicial definido como `provisorio` ate conclusao das evidencias paper-ready pendentes.
- Interpretacao:
  - O framework reduz subjetividade de decisao e conecta diretamente hipoteses suportadas/parcialmente suportadas a regras de promocao.
  - Evita promocao baseada apenas em media de erro sem qualidade probabilistica/robustez.
- Conclusao:
  - Status: suportada (decisao de governanca).
  - Criterios quantitativos formalizados e rastreaveis para implementacao e escrita academica.
- Possiveis correcoes/melhorias:
  - Revisar thresholds apos cada rodada completa com base na distribuicao observada.
  - Migrar de `provisorio` para `oficial` somente apos 2 rodadas consecutivas aprovadas.
  - Incluir no pacote final de resultados os testes faltantes de H3/H4/H7 (IC95 e p-valores).
- Artefatos (paths):
  - `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md`
  - `docs/07_reports/living-paper/30_results_and_analysis.md`
  - `docs/07_reports/living-paper/evidence_log.md`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.

- Data: 2026-04-24
- Questao/Hipotese: Degeneracao de quantis nos sweeps 0_X_X — causa raiz e impacto nas conclusoes.
- Experimento/Execucao:
  - Inspecao de `data/analytics/silver/fact_oos_predictions/asset=AAPL/` (arquivos por feature_set e ano).
  - Medicao de `quantile_p10 == quantile_p50 == quantile_p90` por arquivo e feature_set.
  - Cruzamento com git log de `src/adapters/pytorch_forecasting_tft_trainer.py`.
  - Inspecao de `data/analytics/gold/gold_prediction_metrics_by_config.parquet` e
    `gold_dm_pairwise_results.parquet`.
- Configuracao-chave:
  - Escopo: todos os runs do analytics store (dim_run asset=AAPL), com enfase em
    runs pré-ScopeSpec (parent_sweep_id=NULL, criados 2026-03-08 a 2026-03-15) e
    runs 0_2_3 (criados 2026-04-02 a 2026-04-03).
  - Arquivo inspecionado: `fact_oos_predictions/asset=AAPL/feature_set_name=BTSF/year=2019/`.
- Resultado observado:
  - BTSF (2019 test): 36.278/42.493 linhas (85,4%) com p10 = p50 = p90 = y_pred (MPIW = 0).
  - B-only (2019 test): 73.033/74.278 linhas (98,3%) degeneradas.
  - Nas 14,6% nao-degeneradas do BTSF: PICP = 0,762 e MPIW = 0,033 — faixa razoavel
    para retornos com std = 0,015.
  - Gold PICP reportado (0,01–0,16) e artefato do MPIW=0 agregado, nao calibracao ruim.
  - Git: commit `f7901a4` (2026-04-03 13:07:40) removeu o fallback permissivo que causou isso.
    Código removido:
      ```
      except Exception:
          q10_np, q50_np, q90_np = None, None, None   # silencioso
      if q10_np is None: q10_np = preds_matrix.copy() # fallback = y_pred
      if q50_np is None: q50_np = preds_matrix.copy()
      if q90_np is None: q90_np = preds_matrix.copy()
      ```
  - Todos os runs pré-ScopeSpec rodaram com esse codigo. A maioria dos runs 0_2_3
    tambem (sweep iniciou 2026-04-02; fix commitado 13:07:40 do ultimo dia).
- Interpretacao:
  - O TFT nao colapsou probabilisticamente. O pipeline falhou silenciosamente ao
    extrair quantis e preencheu p10/p50/p90 com a predicao pontual. As metricas
    de RMSE/MAE/DA sao validas (vinham de `preds_matrix`, nao afetado pelo bug).
    As metricas de PICP/MPIW/pinball de todos os runs anteriores a `f7901a4` sao
    invalidas para qualquer claim probabilistico.
  - Pipeline corrigido: codigo atual lanca RuntimeError se quantis nao forem
    extraidos, sem fallback silencioso.
- Conclusao:
  - Status: causa confirmada (evidencia de codigo + git + dados).
  - Runs pré-ScopeSpec: RMSE/MAE/DA validos; PICP/MPIW/pinball invalidos.
  - Runs 0_2_3: idem (maioria sob codigo antigo).
  - Runs futuros (0_2_4 em diante): pipeline corrigido — PICP/MPIW/pinball validos.
- Possiveis correcoes/melhorias:
  - Gate de degeneracao obrigatorio antes de qualquer sweep: % linhas com MPIW=0 < 5%.
  - Implementar como check no quality gate (oos_quantile_block_a_acceptance ja detecta).
- Artefatos (paths):
  - `data/analytics/silver/fact_oos_predictions/asset=AAPL/**`
  - `data/analytics/gold/gold_prediction_metrics_by_config.parquet`
  - `src/adapters/pytorch_forecasting_tft_trainer.py` (commit `f7901a4`)
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos. Secao de Metodo (limitacao do protocolo exploratório) e Resultados
    (justificativa para rejeitar claims probabilisticos das rodadas 0_X_X).

- Data: 2026-04-24
- Questao/Hipotese: O que os sweeps exploratórios 0_X_X permitem concluir validamente?
- Experimento/Execucao:
  - Analise de `gold_prediction_metrics_by_config.parquet` (test, h=1): 6.900 configs,
    12 feature sets, sweep 0_2_3 (675 runs com parent_sweep_id) + pré-ScopeSpec (6.225 runs).
  - Analise de `gold_dm_pairwise_results.parquet`: 17.730 pares DM, sendo 5.239
    cross-feature-set (parent_sweep_ids distintos).
  - Analise de `gold_mcs_results.parquet`: 1.129 configs avaliadas por sweep.
  - Decomposicao de variancia: variância entre feature sets vs variância total de RMSE.
- Configuracao-chave:
  - Filtro: todos os runs com parent_sweep_id nao-nulo (0_2_3) + pré-ScopeSpec
    (parent_sweep_id=NULL, mas config_signature disjunto — zero overlap confirmado).
  - Metricas avaliadas: RMSE, MAE, DA (pontuais); PICP, MPIW, pinball (probabilisticas).
- Resultado observado (pontuais):
  - Delta RMSE entre melhor (BTSF, 0,018893) e pior (BF, 0,018973) feature set: 0,42%.
  - Feature set explica < 0,01% da variancia total de RMSE
    (ratio between/total = 0,0063; R² ≈ 0,00%).
  - Variancia ENTRE configs dentro do mesmo feature set: RMSE range 83–87% do minimo —
    ~200x maior que a variancia entre feature sets.
  - DM cross-feature-set: 0/5.239 pares significativos apos Holm (todos p-adj = 1,0).
  - MCS: 82–100% das configs incluidas no MCS por sweep (teste nao rejeita quase nenhum modelo).
  - DA media: 51–52% em todos os feature sets.
  - RMSE medio: ~0,019 (consistente com literatura para ativos liquidos de alta eficiencia).
- Resultado observado (probabilisticas):
  - PICP medio por feature set: 0,012–0,161 (nominal: 0,80). Invalido.
  - Causa: 85–98% das predicoes com MPIW = 0 (degeneracao por fallback de pipeline).
  - Ver entrada anterior (2026-04-24, degeneracao de quantis) para causa raiz.
- Interpretacao:
  - Para metricas pontuais: os sweeps 0_X_X demonstram de forma consistente que a
    escolha de feature set nao e o driver de performance — hiperparametros dominam.
    Isso e evidencia exploratoria pre-registro, declaravel como tal no TCC.
  - Para metricas probabilisticas: nenhuma conclusao defensavel pode ser extraida.
  - A estrategia all-features motivada por esses dados e metodologicamente correta:
    se feature set nao importa pontualmente e nao e possivel comparar probabilisticamente,
    o principio de parcimonia aponta para treinar com todas as familias e deixar o
    modelo (VSN) decidir o que usar.
- Conclusao:
  - Status: conclusoes delimitadas confirmadas.
  - Claim valido para TCC: "Testaram-se 12 combinacoes de familias de indicadores.
    Nenhuma diferenca estatisticamente significativa foi detectada em RMSE/MAE entre
    feature sets (DM Holm; 5.239 pares cross-FS, todos p-adj=1,0). Delta maximo
    entre melhor e pior feature set: 0,42% do RMSE medio. A variancia dominante e
    explicada por hiperparametros (~200x maior). Com base nessa evidencia exploratoria,
    optou-se por treinar o candidato final com todas as familias de indicadores
    disponiveis."
  - Claims invalidos: qualquer coisa sobre PICP, MPIW, pinball ou calibracao
    probabilistica desses sweeps.
- Possiveis correcoes/melhorias:
  - Rodar os top-k configs nao-degenerados (MPIW > 0) com o pipeline corrigido
    para ter uma baseline de calibracao exploratoria antes da Fase B.
- Artefatos (paths):
  - `data/analytics/gold/gold_prediction_metrics_by_config.parquet`
  - `data/analytics/gold/gold_dm_pairwise_results.parquet`
  - `data/analytics/gold/gold_mcs_results.parquet`
  - `data/analytics/silver/dim_run/asset=AAPL/`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos. Secao de Metodo (justificativa da estrategia all-features) e Resultados
    (analise exploratoria — nao confirmatoria).

- Data: 2026-04-08
- Questao/Hipotese: Implementacao modular do Bloco A (contrato quantilico) para uso recorrente no quality gate.
- Experimento/Execucao:
  - Criacao de servico de dominio reutilizavel para analise/evaluacao de crossing e largura de intervalo.
  - Integracao do servico no `ValidateAnalyticsQualityUseCase` com check dedicado.
  - Inclusao de escopo opcional por `parent_sweep_prefixes`, `splits` e `horizons`.
- Configuracao-chave:
  - Thresholds default do Bloco A:
    - `max_crossing_bruto_rate=0.001` (0.10%)
    - `max_negative_interval_width_count=0`
    - `max_crossing_post_guardrail_rate=0.0`
    - `require_post_guardrail=False`
- Resultado observado:
  - Novo check `oos_quantile_block_a_acceptance` disponivel no quality gate.
  - Implementacao validada por testes unitarios de dominio e de use case.
- Interpretacao:
  - A validacao do Bloco A deixou de ser ad-hoc e passou a ser componente rastreavel, reusavel e escopavel.
- Conclusao:
  - Status: implementada.
  - Pronta para suportar analises 0_2_3 vs 0_2_4 sem duplicacao de logica.
- Possiveis correcoes/melhorias:
  - Expor parametros de escopo/threshold via CLI do `main_refresh_analytics_store`.
  - Persistir resumo do Bloco A em tabela gold dedicada para comparacao historica.
- Artefatos (paths):
  - `src/domain/services/quantile_contract_analyzer.py`
  - `src/use_cases/validate_analytics_quality_use_case.py`
  - `tests/unit/domain/services/test_quantile_contract_analyzer.py`
  - `tests/unit/use_cases/test_validate_analytics_quality_use_case.py`
  - `docs/06_runbooks/RUN_REFRESH_ANALYTICS.md`
- Uso no texto (TCC/Artigo/Ambos):
  - Ambos.

## 2026-05-01 — Phase A / M6: estado real do Analytics Store

**Escopo:** auditoria read-only do Analytics Store antes da Fase B.

**Evidência:**
- `fact_oos_predictions` possui 7.709.077 linhas e 6.901 `run_id` rastreáveis em `dim_run`.
- 6.247/6.923 runs em `dim_run` possuem `parent_sweep_id=NULL`.
- A degeneração raw em massa (`quantile_p10 == quantile_p90`) está concentrada em `parent_sweep_id=NULL`: 7.034.257/7.709.077 linhas.
- Sub-sweeps `0_2_3_*` apresentam 0% de `quantile_p10 == quantile_p90` no estado atual, mas permanecem exploratórios.
- Tabelas gold agregadas como `gold_prediction_metrics_by_config`, `gold_prediction_calibration` e `gold_prediction_robustness_by_horizon` não preservam `parent_sweep_id`.

**Interpretação:** o Analytics Store é legível e auditável, mas claims confirmatórios da Fase B exigem filtro de coorte explícito e não devem usar agregados gold globais sem `parent_sweep_id`.

**Ação:** M6 marcado como `YELLOW` em `docs/07_reports/phase-gates/phase-a/A_code_audit.md`; M5 deve decidir se as gold agregadas recebem `parent_sweep_id` ou se a Fase B usa artefatos scoped-only.

## 2026-05-14 — Stage 3: reset silver/gold pré-Phase B

**Escopo:** arquivamento do silver/gold histórico antes de abrir Phase B confirmatória.

**Evidência:** `data/analytics/{silver,gold}` movidos para `data/analytics_archive_pre_phase_b/` e protegidos como read-only (`chmod -R a-w`). Snapshot histórico abrange runs `parent_sweep_id=NULL` (6.247 runs com 91,25% de degenerescência) e `0_2_3_*` (exploratórios pós-fix `f7901a4`).

**Interpretação:** o archive é rollback/diagnóstico, não evidência confirmatória. Runs pós-2026-05-10 começam de silver vazio, com Stages 1-2 garantindo idempotência e invariante de `config_signature`.

**Artefatos (paths):**
- `data/analytics_archive_pre_phase_b/silver/**`
- `data/analytics_archive_pre_phase_b/gold/**`
- `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` §"Archive pre-Phase B"
- `docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md` §Stage 3

**Uso no texto (TCC/Artigo/Ambos):** Ambos. Capítulo 4 (Método — política de archive) e Capítulo 6 (Limitações — fronteira temporal dos claims).

## 2026-05-16 — Stage 8: política de variante quantilica (raw + post-guardrail; Cat A/B/C)

**Escopo:** definição da policy metodológica raw vs post-guardrail por tabela gold, com categorização A (dual obrigatório), B (post-only) e C (raw-only), e flag `--primary-quantile-contract` (default `post_guardrail`).

**Evidência:** Stages 8.1-8.8 mergeados; PR #21 (`8ab1d39`) propaga policy para `docs/04_evaluation/METRICS_DEFINITIONS.md` §"Variante quantilica", `docs/04_evaluation/CALIBRATION_AND_RISK.md`, `docs/00_overview/GLOSSARY.md` (entrada "variante quantilica") e `docs/07_reports/living-paper/20_method.md` §"Política de variante quantilica". Justificativa acadêmica: Chernozhukov, Fernández-Val & Galichon (2010); Gneiting & Raftery (2007); Jorion (2007); Acerbi & Tasche (2002).

**Interpretação:** H1/H2a/H2b passam a usar `post_guardrail` como variante primária; H3 idem por conservadorismo. Métricas de risco (`var_10`, `es_10_approx`) são post-guardrail apenas. Checks de patologia (`_pred_interval_negative`, gate Stage 11) permanecem raw apenas.

**Artefatos (paths):**
- `docs/04_evaluation/METRICS_DEFINITIONS.md` §Variante quantilica
- `docs/07_reports/living-paper/20_method.md` §Política de variante quantilica
- `src/use_cases/refresh_analytics_store_use_case.py` (colunas `*_raw`/`*_post_guardrail`/`delta_*_post_minus_raw`)

**Uso no texto (TCC/Artigo/Ambos):** Ambos. Capítulo 4 (Método) e Capítulo 5 (Resultados — Gate B).

## 2026-05-25 — Phase B confirmatoria: fechamento da cohort `phase_b_confirmatorio_20260524`

**Escopo:** rodada confirmatoria da Phase B (TFT all-features sealed vs 3
baselines pre-declarados, AAPL, h=1 e h=7), protocolo sealed do pre-registro
F.2 com Emendas E1-E1.8.

**Configuracao-chave:**
- `parent_sweep_id = phase_b_confirmatorio_20260524`
- Config sealed sha256 `fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4`
- Dataset sha256 (AAPL) `aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298`
- Commit selo pre-registro F.2: `067cb32`
- 15 runs TFT (3 folds x 5 seeds) + 45 runs baseline (15 por baseline)
- Baselines: `zero_return`, `historical_mean_rolling` (w=30), `historical_quantiles_rolling` (w=252)
- N alinhado test split (dedup operationally-latest cross-fold): 937 timestamps
- Perda primaria: `pinball_loss_post_guardrail`
- DM family-6: HAC Newey-West com `lag = max(h-1, 1)`, HLN, one-sided, Holm-6

**Resultado observado (tier verdict mecanico, sidecar `phase_b_tier_verdict.parquet`):**

| Hipotese | h=1 | h=7 |
|---|---|---|
| H1 (calibracao) | tier_2 (PICP error 2,02%) | tier_1 (PICP error 0,23%) |
| H2a (vs `zero_return`) | tier_1 (p_adj_holm=0,000; Δrel=+29,9%) | tier_1 (p_adj_holm=0,000; Δrel=+28,8%) |
| H2b (vs todos baselines) | refutada (p_adj_holm vs hist_q = 0,465; Δrel=−0,56%) | refutada (p_adj_holm vs hist_q = 0,685; Δrel=−1,88%) |

Gate de degeneracao quantilica: `p10_eq_p90_rate = 0,000` em todos os grupos
TFT quantilicos (test, val, train; h=1 e h=7); zero runs excluidos.

**Interpretacao:**
- No protocolo Phase B, o candidato TFT all-features sealed entrega evidencia
  confirmatoria primaria (Tier 1) para calibracao em h=7 e para dominancia
  sobre `zero_return` em ambos horizontes; evidencia secundaria (Tier 2) para
  calibracao em h=1.
- H2b refutada por empate estatistico com `historical_quantiles_rolling`:
  resultado consistente com [`STRATEGIC_DIRECTION.md`](../../00_overview/STRATEGIC_DIRECTION.md)
  §3 e §4.4 (baselines quantilicos rolling near-optimal em retornos
  near-random-walk de ativos liquidos). Confirma um limite empirico
  declarado ex-ante, **nao** prova o limite final do TFT em forecasting
  financeiro.
- O resultado **fecha o ciclo confirmatorio para a cohort sealed** (ativo
  AAPL, h=1 e h=7, candidato sealed, baselines pre-declarados, protocolo
  Holm-6). Outras configuracoes de TFT, outros feature sets, outros ativos
  ou outros horizontes podem produzir resultados diferentes em rodadas
  futuras com pre-registro proprio.
- H3 (contribuicao de familias de features) e Phase C **nao** sao tocadas
  por esta rodada.

**Conclusao:**
- Status: H1 suportada (h=7 primaria, h=1 secundaria); H2a suportada em h=1
  e h=7; H2b refutada em h=1 e h=7 (no escopo do protocolo). Veredito final
  e mecanico, sem reframing pos-observacao.

**Artefatos (paths):**
- `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` (relatorio
  primario, 420 LOC, cobre parametros, tier_verdict, DM family-6, MCS
  within-family, gate degeneracao, robustez, limitacoes, conclusao, apendice
  DM-18 sensibilidade e apendice decisoes autonomas Sessao-B).
- `docs/06_pre_registration/phase-b/preregistration_phase_b.md` (pre-registro
  F.2 + Emendas E1 a E6).
- `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/phase_b_marginal_coverage.parquet`
- `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/phase_b_dm_family_6.parquet`
- `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/phase_b_dm_family_18_sensitivity.parquet`
- `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/phase_b_delta_pinball.parquet`
- `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/phase_b_tier_verdict.parquet`
- Archive read-only: `data/analytics_archive_phase_b_20260524/`.

**Uso no texto (TCC/Artigo/Ambos):** Ambos. Capitulo 5 (Resultados §5.2-§5.5)
e Capitulo 6 (Conclusoes §6.1, §6.3). Reportar somente os claims permitidos
no escopo da cohort; nao generalizar para outros ativos, outros horizontes,
outros feature sets ou outras configuracoes.

## 2026-05-25 — Decisao C.0: gold legacy DM/MCS/Holm/top-50/VaR-ES como hardening pendente

**Escopo:** delimitar uso editorial de `gold_dm_pairwise_results`,
`gold_mcs_results`, `gold_win_rate_pairwise_results`,
`gold_model_decision_final`, `gold_prediction_risk` no texto.

**Evidencia:** auditoria metodologica externa
([`auditoria_metodologica_forecasting_financeiro.md`](../external-reviews/auditoria_metodologica_forecasting_financeiro.md))
+ inventario C.0
([`C0_statistical_methods_hardening.md`](../phase-gates/phase-c/C0_statistical_methods_hardening.md))
confirmam que DM/MCS gold operam sobre `squared_error` em vez da perda
primaria pinball; top-50 cap silencioso antes do pairwise gera inferencia
pos-selecao; Holm gold nao preserva `split_signature`; MCS hard-codeia
`B=300, block_len=5`; `prob_up`, `confidence_calibrated` sao heuristicas
nao canonicas; VaR/ES operam sobre quantis post-guardrail sem backtesting
de excedencias.

**Interpretacao:** essas tabelas permanecem disponiveis para diagnostico
exploratorio dentro de uma cohort, mas **nao** sustentam claim
confirmatorio. A Phase B substitui inferencialmente esses artefatos por
sidecars dedicados; Phase C / C.0 conduzira o hardening antes de qualquer
reuso confirmatorio.

**Conclusao:** o Capitulo 5 do TCC nao deve apresentar
`gold_model_decision_final`, `gold_dm_pairwise_results`,
`gold_mcs_results`, `prob_up`, `confidence_calibrated`, `var_10`,
`es_10_approx` ou `win_rate_ex_ties_mean` como evidencia confirmatoria.
Esses artefatos podem ser referenciados como diagnostico ou descritivo,
sempre rotulados como tal.

**Artefatos (paths):**
- `docs/07_reports/phase-gates/phase-c/C0_statistical_methods_hardening.md`
- `docs/07_reports/external-reviews/auditoria_metodologica_forecasting_financeiro.md`

**Uso no texto (TCC/Artigo/Ambos):** Ambos. Aplica-se a todo Capitulo 5 que
toque essas tabelas, e ao Capitulo 6 §6.4 (trabalhos futuros).
