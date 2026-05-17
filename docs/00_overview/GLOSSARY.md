---
title: Glossary
scope: Definicoes canonicas dos termos tecnicos recorrentes (PICP, MPIW, pinball, DM, MCS, VSN, scope_mode, run_id etc.). Vocabulario de uma linha por termo. Para formulas e thresholds, ver 04_evaluation/METRICS_DEFINITIONS.md.
update_when:
  - novo termo tecnico for introduzido no projeto
  - definicao semantica de termo existente mudar
  - sigla nova for adotada no codigo ou nos docs
canonical_for: [glossary, terminology, vocabulary]
---

# Glossary

Definicoes canonicas dos termos tecnicos recorrentes neste projeto.
Objetivo: garantir que banca, orientador, revisor e agentes de IA usem a mesma
semantica ao longo de toda a documentacao, codigo e texto do TCC.

Organizacao: termos agrupados por dominio. Cada entrada tem: definicao, uso no
projeto, e referencia quando aplicavel.

---

## A — Analytics e Artefatos

**analytics store**
Conjunto de tabelas Parquet (silver) e DuckDB (gold) que materializam metricas,
predicoes e comparacoes de forma reproduzivel sem retrain. Estruturado em camadas:
silver (fonte primaria imutavel) e gold (derivados reconstruiveis).
Referencia: `docs/01_architecture/`.

**artefato**
Qualquer arquivo persistido que permite reproducao de uma analise ou treinamento:
checkpoint de modelo, Parquet de predicoes OOS, gold de metricas, pre-registro.
Decisoes finais dependem apenas de artefatos — sem retrain para reanalise.

---

## C — Calibracao e Contrato Probabilistico

**calibracao probabilistica**
Correspondencia entre a probabilidade nominal declarada pelo modelo e a frequencia
empirica observada. Um modelo bem calibrado em p90 deve ter y_true abaixo de p90
em aproximadamente 90% dos casos. Medida por PICP.

**cohort / coorte**
Conjunto de runs de um mesmo sweep (mesmo `parent_sweep_id` ou frozen set) que
compartilham protocolo homogeneo de treino/validacao/teste. Comparacoes estatisticas
validas ocorrem apenas dentro de coortes comparaveis.

**cohort_decision** (scope_mode)
Modo de escopo no qual validacoes e comparacoes sao filtradas para a coorte oficial
declarada. Bloqueante para decisoes finais. Ver `global_health` para contraparte.
Referencia: `docs/ai/AGENT_CORE.md`.

**coverage_error**
`PICP - cobertura_nominal`. Positivo = over-coverage (intervalos conservadores);
negativo = under-coverage (intervalos otimistas). Usado como metrica de calibracao.

**crossing (quantile crossing)**
Violacao da ordenacao monotonica de quantis: ocorre quando p10 > p50 ou p50 > p90
na saida do modelo. Causa: quantile regression aprendeu quantis que se cruzam em
algumas amostras (H3 do evidence log suportada). Mitigado pelo guardrail monotono
pos-inferencia (ver `variante quantilica`).

**variante quantilica (raw vs post-guardrail)**
Duas variantes paralelas dos quantis previstos persistidas em `fact_oos_predictions`:
(a) **raw** = saida direta do modelo (`quantile_p10`, `quantile_p50`, `quantile_p90`),
podendo violar monotonicidade; (b) **post-guardrail** = rearranjo monotono via
`QuantileGuardrailService.enforce_monotonic_triplet` (sort por linha), garantindo
`p10 <= p50 <= p90`. Toda metrica probabilistica e persistida em ambas variantes
com sufixos `_raw` / `_post_guardrail`. **Variante primaria das hipoteses H1, H2a,
H2b: `post_guardrail`** — declarada via flag `--primary-quantile-contract`
(default `post_guardrail`) e persistida como coluna `primary_quantile_contract`
em `gold_model_decision_final`. Justificativa metodologica em
`docs/07_reports/living-paper/20_method.md` §Politica de variante quantilica.
Categorizacao por tabela (A=dual, B=post-only, C=raw-only) em
`docs/04_evaluation/METRICS_DEFINITIONS.md` §Variante quantilica.

---

## D — Dados e Dataset

**dataset_fingerprint**
Hash SHA-256 do dataset de entrada (asset, timestamp_min, timestamp_max, row_count,
close_sum, volume_sum, parquet_file_hash). Garante que runs diferentes usaram
exatamente os mesmos dados de entrada.

**drift (de modelo)**
Degradacao gradual da performance quando a distribuicao dos dados em producao se
afasta da distribuicao de treino. Relevante para monitoramento pos-deployment.

---

## F — Features

**familias de features**
Agrupamento semantico de features por origem:
- `PRICE`: preco, volume, retorno log (OHLCV derivados).
- `TECHNICAL`: indicadores tecnicos calculados sobre OHLCV (RSI, MACD, Bollinger).
- `SENTIMENT`: sentimento de noticias agregado por sessao de mercado.
- `FUNDAMENTAL`: indicadores fundamentalistas (P/L, P/VPA, EPS, margem etc.).

**feature_set_hash**
SHA-256 da lista ordenada de features usadas em um run (`"|".join(features_ordered)`).
Garante rastreabilidade exata do conjunto de entrada sem armazenar nomes redundantes.

**feature_set_name**
Nome semantico do grupo de features (ex: `BASELINE_FEATURES`, `ALL_FEATURES`).
Mapeado para listas concretas em `src/infrastructure/`.

**poda minima (all-features com poda minima)**
Pre-processamento baseado em principio (nao em OOS): remover features com correlacao
par-a-par > 0,95; missing > 30% em treino; derivadas redundantes. Decisao deve ser
documentada antes de olhar resultados. Ver `STRATEGIC_DIRECTION.md` §4.2.

**time_varying_known_reals**
Parametro do TFT: features cujos valores sao conhecidos em tempo de previsao (ex:
calendario, feriados planejados). Features derivadas de preco/volume/retorno NAO
pertencem a esta categoria — pertencer incorretamente gera leakage temporal.

**time_varying_unknown_reals**
Parametro do TFT: features cujos valores futuros sao desconhecidos em tempo de
previsao (ex: retorno, volume, RSI). Features de preco e indicadores tecnicos devem
estar aqui.

---

## G — Governance e Escopo

**frozen_timestamp**
Timestamp de congelamento do conjunto de avaliacao. Declarado antes da analise para
prevenir data snooping. Artefato obrigatorio no pre-registro.

**global_health** (scope_mode)
Modo de escopo no qual validacoes cobrem todos os runs do asset, sem restricao de
coorte. Usado para monitoramento de saude da plataforma. Falhas em `global_health`
nao bloqueiam automaticamente decisoes de `cohort_decision`.
Referencia: `docs/ai/AGENT_CORE.md`.

**guardrail monotonico**
Pos-processamento que reordena quantis para garantir p10 <= p50 <= p90. Aplicado
na persistencia de predicoes. Auditado via `gold_quantile_guardrail_audit`:
`delta_pinball`, `delta_picp`, `delta_mpiw` before/after. As metricas calculadas
sobre quantis pos-guardrail sao primarias para H1/H2a/H2b — ver
`variante quantilica` na secao C.

**leakage temporal**
Uso de informacoes futuras no treino ou estimacao de features. Formas comuns:
(1) feature calculada com dados apos o timestamp corrente; (2) scaler `fit` com
dados de validacao ou teste; (3) HPO que usa o test set como criterio de selecao.
Invalida conclusoes de OOS se presente.

**pre-registro**
Documento versionado (commitado antes da rodada confirmatoria) que declara:
candidato, baselines, metrica primaria, regra de desempate, threshold de relevancia
pratica, PICP bands de elegibilidade. Garante que decisoes metodologicas foram
tomadas antes de ver os resultados. Referencia: `docs/08_governance/`.

**scope_spec**
Objeto de dominio que encapsula o modo de escopo (`scope_mode`), prefixos de sweep,
splits e horizontes. Evita comparacoes globais acidentais em fluxos de decisao.
Implementado em `src/domain/services/scope_spec.py`.

---

## H — Horizontes e Splits

**h+N (horizonte de previsao)**
Numero de passos a frente previsto. `h+1` = 1 dia util a frente; `h+7` = 7 dias;
`h+30` = 30 dias.

**OOS (Out-of-Sample)**
Dados do conjunto de teste que nao foram usados em treino ou validacao. Unico
conjunto valido para reportar performance generalizavel. Sinonimo de test split
neste projeto.

**split temporal**
Divisao cronologica do dataset em: `train` (ajuste), `val` (early stopping e HPO),
`test` (avaliacao OOS final). Splits nao se sobrepoem.

**split_fingerprint**
SHA-256 das listas de timestamps ordenadas por split (train/val/test). Garante
comparabilidade: runs com mesmo `split_fingerprint` foram avaliados no mesmo OOS.

**target_timestamp**
Timestamp do periodo alvo (futuro) previsto. Chave de alinhamento em comparacoes
pareadas. Exemplo: em h+1 com input em 2024-01-02, `target_timestamp` e 2024-01-03.

---

## M — Metricas

**coverage_error** ver secao C.

**DA (Directional Accuracy)**
Percentual de previsoes em que o sinal da predicao coincide com o sinal do retorno
realizado. Metrica de acuracia direcional; nao mede magnitude.

**DM test (Diebold-Mariano)**
Teste de hipotese para igualdade de acuracia preditiva entre dois modelos (H0:
E[L(e_A)] = E[L(e_B)]). Requer alinhamento temporal exato. Versao recomendada:
Harvey, Leybourne & Newbold (1997) com correcao HAC para amostras pequenas.

**MAE (Mean Absolute Error)**
`mean(|y_pred - y_true|)`. Metrica de erro pontual robusta a outliers.

**MCS (Model Confidence Set)**
Procedimento (Hansen, Lunde & Nason 2011) que identifica o conjunto de modelos nao
rejeitaveis como igualmente bons ao nivel alpha. Controla FWER por construcao.
Resultado: conjunto de modelos, nao ranking unico.

**MPIW (Mean Prediction Interval Width)**
`mean(p90 - p10)`. Largura media do intervalo de predicao. Analisar sempre com PICP:
intervalo estreito com PICP baixo significa modelo sistematicamente errado.

**PICP (Prediction Interval Coverage Probability)**
`mean(p10 <= y_true <= p90)`. Para o intervalo [p10, p90] o nominal e 80%.
Modelo bem calibrado tem PICP proximo de 0,80.

**pinball loss (quantile loss)**
`L_q(y, q_hat) = max(q*(y - q_hat), (q-1)*(y - q_hat))`. Funcao de perda para
regressao quantilica. **Metrica primaria** neste projeto para avaliacao probabilistica.

**RMSE (Root Mean Squared Error)**
`sqrt(mean((y_pred - y_true)^2))`. Metrica de erro pontual; sensivel a outliers.
Metrica secundaria neste projeto (pinball e primaria).

---

## P — Pipeline e Execucao

**parent_sweep_id**
Identificador do sweep pai de um run (ex: `0_2_3_`, `1_0_0_`). Usado para filtrar
coortes nas tabelas gold. Campo obrigatorio no grao de decisao.

**run_id**
Identidade logica deterministica de um experimento:
`sha256(asset|feature_set_hash|trial_number|fold|seed|model_version|config_signature|split_signature|pipeline_version)`.
Estavel: o mesmo experimento sempre gera o mesmo `run_id`.

**sweep**
Conjunto de experimentos com variacao sistematica de hiperparametros ou configuracoes.
Rodadas `0_X_X` sao exploratorias (HPO, sensibilidade); rodadas `1_X_X` sao
confirmatorias (candidato unico vs baselines).

**VSN weights (Variable Selection Network)**
Pesos de selecao dinamica de features produzidos pelo TFT (Lim et al. 2021).
Indicam, por timestep, quanto cada feature foi usada. Base para analise de
contribuicao nativa. Interpretacao: "o modelo priorizou esta feature" — nao
"esta feature causa o retorno".

---

## R — Robustez e Reproducibilidade

**fold**
Divisao temporal cruzada para estimativa de variancia. Neste projeto: K=3 folds;
splits nao vazam informacao futura.

**seed**
Semente de geracao de numeros aleatorios para reproducao de resultados estocasticos.
Fixada por run e registrada em `fact_config`. Fixada tambem em inferencia para
resultados bit-exact.

---

## Referencia de Siglas

| Sigla | Significado |
|---|---|
| DA | Directional Accuracy |
| DM | Diebold-Mariano test |
| ES | Expected Shortfall |
| HAC | Heteroskedasticity and Autocorrelation Consistent |
| HPO | Hyperparameter Optimization |
| MAE | Mean Absolute Error |
| MCS | Model Confidence Set |
| MPIW | Mean Prediction Interval Width |
| OOS | Out-of-Sample |
| PICP | Prediction Interval Coverage Probability |
| RMSE | Root Mean Squared Error |
| TFT | Temporal Fusion Transformer |
| VaR | Value at Risk |
| VSN | Variable Selection Network |
