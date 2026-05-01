---
title: Phase A Code Audit (Pre-Experiment Gate)
scope: Gate de entrada da Fase A (sanidade) com veredictos por modulo P0 (M1-M7: train_use_case, trainer, build_dataset, inference, refresh_analytics, estado do Analytics Store, capacidade operacional da Fase B). Bloqueante para qualquer treino confirmatorio da Fase B.
update_when:
  - veredicto de algum modulo P0 mudar (GREEN/YELLOW/RED)
  - novo achado for registrado
  - nova questao for adicionada a um modulo
  - status de gate de saida (criterio de Fase B) mudar
canonical_for: [phase_a_code_audit, pre_phase_b_gate, p0_module_verdicts, analytics_store_state_audit, phase_b_pipeline_capability]
---

# Phase A — Code Audit (Pre-Experiment Gate)

**Status:** pendente
**Criado:** 2026-04-24
**Bloqueante para:** qualquer treino confirmatorio da Fase B

Este documento e o gate de entrada da Fase A (sanidade). Nenhum experimento
confirmatorio pode ser iniciado antes de todos os modulos P0 receberem veredicto
GREEN ou YELLOW (com acao documentada).

Referencia de contexto: `docs/00_overview/STRATEGIC_DIRECTION.md` §5 (Fase A).

---

## Como usar

Para cada modulo P0, preencher:
- **Veredicto:** GREEN / YELLOW / RED
- **Achados:** o que foi verificado e o que foi encontrado (nao o que se esperava)
- **Acao:** se YELLOW ou RED, o que precisa ser corrigido antes da Fase B

Veredictos:
- `GREEN` — sem problemas. Fase B pode prosseguir neste modulo.
- `YELLOW` — problema menor com acao documentada. Fase B pode prosseguir apos acao.
- `RED` — problema bloqueante. Fase B nao pode comecar.

## Ordem recomendada de execucao

Executar os modulos na ordem abaixo para reduzir risco com menor custo. Esta
ordem nao altera o gate de saida: todos os modulos M1-M7 continuam obrigatorios
antes da Fase B.

1. **M6 — Estado real do Analytics Store.**
   Verifica primeiro se os dados persistidos sao isolaveis e auditaveis por
   coorte. Se o estado silver/gold estiver inconsistente, os demais achados
   precisam ser interpretados com esse contexto.
2. **M7 — Capacidade operacional basica, sem treino caro.**
   Checar se existem caminhos para candidato unico, baselines, escopo
   descartavel e multi-horizonte antes de gastar tempo com execucao end-to-end.
3. **M2 e M3 — quantis e dataset/leakage.**
   Validar riscos que um smoke test pode nao detectar: fallback multi-horizonte,
   degeneracao de quantis, split/scaler/features e disponibilidade temporal.
4. **M7 — smoke test completo.**
   Rodar o teste end-to-end reduzido apenas depois dos riscos centrais de
   quantis e dataset terem sido avaliados.
5. **M1, M4 e M5 — orquestracao, inferencia e refresh.**
   Fechar os riscos de pipeline completo, consistencia treino/inferencia,
   reproducibilidade, guardrails, metricas e escopo estatistico.

---

## Modulos P0

---

### M1 — `train_tft_model_use_case.py`

**Por que e critico:** orquestra o pipeline inteiro. E onde a sequencia
split → scaler fit → dataset build pode introduzir leakage silencioso.
O maior arquivo do projeto (1.442 linhas).

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | `known_real_cols` so contem features genuinamente futuro-conhecidas (calendario)? Nao ha override via config que injete features de preco/retorno? | `train_tft_model_use_case.py:1187` | |
| Q2 | O scaler interno do pytorch-forecasting (`TimeSeriesDataSet`) e fit apenas em `train_df`? O `from_dataset(training, val_df)` usa os parametros do training sem re-fit? | `pytorch_forecasting_tft_trainer.py:221-236` | |
| Q3 | A warmup policy remove corretamente os primeiros N dias antes de calcular splits, garantindo que features com `warmup_count > 0` nao produzam NaN no inicio do treino? | `train_tft_model_use_case.py:_apply_warmup_policy_to_split` | |
| Q4 | Ha algum caminho de codigo onde `test_df` e usado como criterio de HPO (mesmo indiretamente, como via `mean_test_rmse` sendo o objetivo padrao)? | `run_tft_optuna_search_use_case.py:49,176-184` | |

**Veredicto geral:** ___

**Achados:**

**Acao (se YELLOW/RED):**

---

### M2 — `pytorch_forecasting_tft_trainer.py`

**Por que e critico:** executa o treino real do TFT. O bug do fallback foi
corrigido em `f7901a4` (2026-04-03), mas a correcao nao foi validada
end-to-end com uma config all-features + multi-horizonte.

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | Early stopping: o `monitor` e `val_loss` (correto) ou `train_loss`/metrica de test (invalido)? | `pytorch_forecasting_tft_trainer.py`: procurar `EarlyStopping` / `ModelCheckpoint` | GREEN |
| Q2 | `QuantileLoss(quantiles=[0.1, 0.5, 0.9])`: os pesos sao uniformes? Algum override reduz o peso de p10/p90 a zero (causaria colapso de gradiente)? | `pytorch_forecasting_tft_trainer.py:254` | GREEN |
| Q3 | O novo fallback `_manual_forward_quantiles_and_actuals` funciona para `max_prediction_length > 1`? O cubo `[N, H, Q]` e indexado corretamente para H > 1? | `pytorch_forecasting_tft_trainer.py:358-430` | YELLOW |
| Q4 | Gate de degeneracao: apos o treino, ha verificacao de que `% linhas com p10==p90 < 5%` no OOS? Ou o pipeline passa silenciosamente com quantis degenerados? | `train_tft_model_use_case.py`: buscar check de MPIW | RED |

**Veredicto geral:** RED

**Achados:**

- Q1: `ModelCheckpoint` e `EarlyStopping` monitoram `val_loss` com
  `mode="min"`; a selecao de melhor epoca tambem usa `val_loss`. Nao foi
  encontrado uso de `train_loss` ou metrica de test como criterio de parada ou
  checkpoint.
- Q2: o modo quantil instancia `QuantileLoss(quantiles=list(cfg.quantile_levels))`
  e os defaults do trainer sao `(0.1, 0.5, 0.9)`. A introspeccao local de
  `QuantileLoss(quantiles=[0.1,0.5,0.9])` mostrou `quantiles=[0.1, 0.5, 0.9]`
  e nenhum atributo de peso/ponderacao customizada. Nao ha override no trainer
  zerando p10/p90.
- Q3: por inspecao estatica, `_manual_forward_quantiles_and_actuals` aceita
  saida 3D e trata dois layouts esperados: `[batch, horizon, quantile]` e
  `[batch, quantile, horizon]` (via `swapaxes`). Depois seleciona p10/p50/p90
  por indice de quantil e retorna matrizes `[N, H]`. O caminho principal
  `_extract_quantiles` tambem espera `[batch, horizon, quantile]`.
- Q3: o trainer seleciona horizontes por `selected_idx = [h - 1]`, corta
  `preds_matrix`/`actuals_matrix` pelo menor `horizon_dim` e persiste detalhes
  em `quantile_p*_matrix`. A persistencia em
  `_persist_fact_oos_predictions` ja possui teste unitario para alinhamento
  `h=1,7,30` por split.
- Q3: apesar da logica estar coerente por leitura e haver teste para
  persistencia multi-horizonte, nao foi encontrado teste unitario dedicado
  exercitando `_manual_forward_quantiles_and_actuals` com `H>1`, nem smoke
  end-to-end real com `max_prediction_length >= 7`. Portanto o ponto fica
  YELLOW ate validacao por teste dedicado ou smoke M7.
- Q4: `train_tft_model_use_case.py` aplica `QuantileGuardrailService` antes de
  persistir OOS e grava colunas `quantile_p10_post_guardrail`,
  `quantile_p50_post_guardrail`, `quantile_p90_post_guardrail` e
  `quantile_guardrail_applied`. Isso corrige ordenacao monotona, mas nao
  detecta nem bloqueia degeneracao de largura zero (`p10==p90`).
- Q4: `QuantileContractAnalyzer` e `ValidateAnalyticsQualityUseCase` checam
  ordem dos quantis, largura negativa e crossing pos-guardrail; nao ha regra
  equivalente para `% quantile_p10 == quantile_p90` ou
  `% quantile_p10 == quantile_p50 == quantile_p90`. Assim, uma coorte com
  quantis colapsados pode passar por treino/persistencia e so ser descoberta
  por diagnostico posterior, repetindo a classe de risco que motivou a Fase A.
- Validacao executada:
  `tests/unit/adapters/test_pytorch_forecasting_tft_trainer.py`,
  `tests/unit/use_cases/test_train_tft_model_use_case.py::test_persist_fact_oos_predictions_keeps_horizon_index_alignment_per_split`,
  `tests/unit/use_cases/test_train_tft_model_use_case.py::test_persist_fact_oos_predictions_applies_quantile_guardrail_columns`,
  `tests/unit/domain/services/test_quantile_contract_analyzer.py`,
  `tests/unit/use_cases/test_validate_analytics_quality_use_case.py::test_validate_analytics_quality_block_a_check_passes_on_minimal_valid_dataset` e
  `tests/unit/use_cases/test_validate_analytics_quality_use_case.py::test_validate_analytics_quality_block_a_scope_filters_parent_sweep_prefix`
  passaram (`18 passed`, 1 warning de deprecacao de parametros antigos de
  Block A).

**Acao (se YELLOW/RED):**

- Antes da Fase B, adicionar gate bloqueante de degeneracao quantilica
  **condicional ao `prediction_mode` declarado em `fact_config`** (source of
  truth, nao inferir do output):
  - **Runs com `prediction_mode='quantile'`:** reportar `n_rows`, `% p10==p90`
    e `% p10==p50==p90` por `parent_sweep_id`/split/horizonte, com limiar rigido
    apenas quando `n_rows` for suficiente (mesma regra operacional do M7-Q6).
    Degeneracao em massa bloqueia a Fase B porque viola o contrato declarado:
    o run prometeu intervalo probabilistico e entregou ponto disfarcado.
  - **Runs com `prediction_mode='point'`:** quantis colapsados
    (`p10==p50==p90==y_pred`) sao permitidos apenas como campos de
    compatibilidade de schema e devem ser explicitamente excluidos de
    metricas/claims probabilisticos (PICP, MPIW, pinball, intervalo p10-p90).
    Claims pontuais (RMSE, MAE, DA) permanecem validos para esses runs.
- Decidir onde esse gate deve viver: preferencialmente em
  `QuantileContractAnalyzer`/`ValidateAnalyticsQualityUseCase` para permitir
  validacao scoped por coorte; alternativamente como verificacao pos-treino em
  `TrainTFTModelUseCase` antes de aceitar a rodada.
- Em M5, garantir que metricas probabilisticas em
  `refresh_analytics_store_use_case.py` (PICP, MPIW, pinball, calibracao) sejam
  computadas apenas sobre runs com `prediction_mode='quantile'` nao-degenerados.
  Sem esse filtro, runs `point` (ou `quantile` degenerados antes do gate)
  arrastam metricas globais. Considerar coluna auxiliar
  `is_quantile_genuine` em `fact_oos_predictions` ou filtro por NULL nas
  colunas quantilicas para runs `point`.
- Adicionar teste unitario dedicado para o fallback
  `_manual_forward_quantiles_and_actuals` com saida `H>1`, cobrindo pelo menos
  o layout `[batch, horizon, quantile]` e, idealmente, o layout alternativo
  `[batch, quantile, horizon]`. O teste deve validar valores especificos por
  `(n, h, q)` com cubo sintetico, nao apenas ausencia de excecao.
- Executar no M7 um smoke real com `max_prediction_length >= 7` e
  `evaluation_horizons=[1,7]` para validar que o caminho completo produz
  `fact_oos_predictions` com `h=7`, timestamps coerentes e quantis nao
  degenerados.

---

### M3 — `build_tft_dataset_use_case.py`

**Por que e critico:** constroi o dataset que alimenta o TFT. Leakage aqui
contamina todos os runs — nao pode ser detectado por metricas OOS.

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | O `sklearn_indicator_normalizer` (`fit()`) e chamado ANTES do split temporal ou DEPOIS? Se antes, leakage de normalizacao garantido. | `build_tft_dataset_use_case.py`: buscar `normalizer.fit` / `sklearn_indicator_normalizer` | |
| Q2 | Features com `warmup_count > 0` no `feature_registry`: o dataset exclui os primeiros N dias dessas features corretamente, ou os inclui com NaN? | `feature_registry.py`: verificar `warmup_count`; `build_tft_dataset_use_case.py`: verificar aplicacao | |
| Q3 | Todas as features de retorno/preco estao em `time_varying_unknown_reals`, nao em `time_varying_known_reals`? Confirmar que o `known_real_cols` do trainer bate com o que foi definido aqui. | `build_tft_dataset_use_case.py:228-229` | |
| Q4 | Se a Fase B exigir rebuild do dataset all-features, fundamentals e sentiment respeitam disponibilidade temporal real (as-of date, publication/collection timestamp e corte da sessao)? | feature registry + fontes processadas; ver tambem M7 | |

**Veredicto geral:** ___

**Achados:**

**Acao (se YELLOW/RED):**

---

### M4 — `run_tft_inference_use_case.py` + `pytorch_forecasting_tft_inference_engine.py`

**Por que e critico:** inferencia deve reproduzir exatamente o protocolo de
treino. Divergencias silenciosas (config, scaler, seed) invalidam a
comparacao treino vs inferencia.

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | O modelo e carregado com os mesmos `known_real_cols` e `feature_cols` usados no treino? Ou sao recalculados dinamicamente e podem divergir se a config mudar? | `run_tft_inference_use_case.py`: buscar `known_real_cols` / `feature_cols` no loader | |
| Q2 | Seed de inferencia: o modelo e colocado em `eval()` mode (desliga dropout)? Ou ha dropout ativo produzindo quantis diferentes entre runs? | `pytorch_forecasting_tft_inference_engine.py`: buscar `.eval()` / `torch.no_grad()` | |
| Q3 | O guardrail monotonico (`quantile_p*_post_guardrail`) e aplicado na inferencia da mesma forma que no treino? Ou so e aplicado no refresh do analytics store? | `run_tft_inference_use_case.py`: buscar `guardrail` / `post_guardrail` | |
| Q4 | A seed de inferencia e registrada em `fact_config` ou `dim_run` para reproducibilidade? | `run_tft_inference_use_case.py` / `parquet_analytics_run_repository.py` | |

**Veredicto geral:** ___

**Achados:**

**Acao (se YELLOW/RED):**

---

### M5 — `refresh_analytics_store_use_case.py`

**Por que e critico:** calcula todas as metricas do paper. PICP calculado
sobre quantis raw vs pos-guardrail produz numeros diferentes. Scope incorreto
mistura sweeps na mesma metrica.

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | PICP e calculado sobre `quantile_p10_post_guardrail`/`quantile_p90_post_guardrail` ou sobre `quantile_p10`/`quantile_p90` raw? As duas colunas existem — qual e usada como primaria? | `refresh_analytics_store_use_case.py`: buscar `picp` / `in_interval` | |
| Q2 | As tabelas gold `gold_prediction_metrics_by_config`, `gold_prediction_calibration` e `gold_prediction_robustness_by_horizon` recebem `parent_sweep_id` no output? Se nao, como filtrar escopo na analise final? | Confirmado: nenhuma das tres tem `parent_sweep_id`. Verificar se ha filtro pre-calculo ou so pos-calculo. | |
| Q3 | O `ScopeSpec` com `scope_mode=cohort_decision` e passado corretamente em todos os paths de calculo que alimentam decisao estatistica (DM, MCS, win-rate)? Ou so no quality gate? | `refresh_analytics_store_use_case.py`: buscar `scope_spec` / `ScopeSpec` | |

**Veredicto geral:** ___

**Achados:**

**Acao (se YELLOW/RED):**

---

### M6 — Estado real do Analytics Store

**Por que e critico:** a Fase B depende de decisoes estatisticas por coorte.
Se silver/gold misturam runs historicos, pre-ScopeSpec, runs degenerados ou
escopos diferentes, as conclusoes ficam invalidas mesmo com codigo correto.

**Questoes a verificar:**

| # | Questao | Fonte/consulta | Veredicto |
|---|---|---|---|
| Q1 | Todo `run_id` em `fact_oos_predictions`, `fact_config`, `fact_split_metrics` e tabelas relacionadas possui entrada correspondente em `dim_run`? | queries silver | GREEN |
| Q2 | Cada `run_id` possui no maximo um `parent_sweep_id` efetivo? Ha runs orfaos ou ambiguos? | `dim_run`, `fact_config` | YELLOW |
| Q3 | As tabelas gold usadas para decisao preservam ou recebem filtro inequivoco de coorte antes do calculo? | gold + `refresh_analytics_store_use_case.py` | YELLOW |
| Q4 | Runs pre-ScopeSpec e runs `0_2_3` degenerados podem ser excluidos de claims probabilisticos por filtro reproduzivel? | silver/gold + `evidence_log.md` | YELLOW |
| Q5 | Existem particoes parciais, duplicatas por chave logica ou arquivos corrompidos em `fact_oos_predictions`? | parquet scan | GREEN |
| Q6 | Ha algum calculo gold ou query de decisao que agregue por `target_timestamp`, `feature_set_name`, `split` ou `horizon` sem preservar/filtrar `parent_sweep_id`, misturando coortes distintas? | gold + refresh use case | YELLOW |

**Veredicto geral:** YELLOW

**Achados:**

- Inventario lido em modo read-only: 9.874 arquivos Parquet em
  `data/analytics`, incluindo 7.709.077 linhas em `fact_oos_predictions`.
- Q1: `fact_oos_predictions` (6.901 `run_id`), `fact_config` (6.899),
  `fact_split_metrics` (6.901), `fact_epoch_metrics` (6.901),
  `fact_model_artifacts` (6.913), `fact_run_snapshot` (6.901),
  `fact_split_timestamps_ref` (1.358), `bridge_run_features` (6.899) e
  `fact_failures` (10) nao possuem `run_id` ausente em `dim_run`.
- Q2: nao ha `run_id` duplicado em `dim_run` e nao ha `run_id` com multiplos
  valores efetivos de `parent_sweep_id` entre `dim_run`, `fact_config`,
  `fact_split_metrics`, `fact_epoch_metrics` e `fact_run_snapshot`. Porem,
  6.247/6.923 runs em `dim_run` possuem `parent_sweep_id=NULL` e devem ser
  tratados como historico pre-ScopeSpec/orfao para decisoes por coorte.
- Q3/Q6: tabelas estatisticas pareadas preservam `parent_sweep_id`
  (`gold_dm_pairwise_results`, `gold_mcs_results`,
  `gold_win_rate_pairwise_results`, `gold_paired_oos_intersection_by_horizon`,
  `gold_quality_statistics_report`, `gold_model_decision_final`). Porem,
  tabelas gold agregadas usadas em analise descritiva ou ranking nao preservam
  `parent_sweep_id`, incluindo `gold_prediction_metrics_by_config`,
  `gold_prediction_calibration`, `gold_prediction_metrics_by_horizon`,
  `gold_prediction_robustness_by_horizon`,
  `gold_prediction_generalization_gap`, `gold_ranking_by_config`,
  `gold_ic95_by_config_metric`, `gold_feature_set_impact` e
  `gold_feature_impact_by_horizon`.
- Q3/Q6: no estado atual, nao ha mistura de `parent_sweep_id` quando o grao
  inclui `asset + feature_set_name + config_signature + split + horizon` ou
  `asset + feature_set_name + config_signature + horizon`. Ha mistura quando o
  grao remove `config_signature`: 24 grupos em
  `asset + feature_set_name + split + horizon` combinam mais de um grupo de
  `parent_sweep_id`.
- Q4: os runs pre-ScopeSpec sao filtraveis por `parent_sweep_id=NULL` e
  concentram a degeneracao raw atual: 7.034.257/7.709.077 linhas
  (`91,25%`) com `quantile_p10 == quantile_p90`, todas no grupo
  `parent_sweep_id=NULL`. Os runs `0_2_3` sao filtraveis por prefixo de
  `parent_sweep_id`; no estado atual nao apresentam `p10==p90` em massa, mas
  continuam exploratorios e tiveram violacoes quantilicas registradas no
  `evidence_log.md`.
- Q5: nao foram encontrados arquivos Parquet ilegíveis em
  `fact_oos_predictions`; nao ha duplicatas por
  `run_id + split + horizon + target_timestamp_utc`, nem por
  `run_id + split + horizon + timestamp_utc + target_timestamp_utc`, nem por
  `asset + feature_set_name + config_signature + split + fold + seed + horizon
  + target_timestamp_utc`.
- Q5: `fact_oos_predictions` possui 421 linhas com ordenacao raw invalida de
  quantis, mas as colunas pos-guardrail possuem 0 violacoes de ordem e 0 linhas
  com `quantile_p10_post_guardrail == quantile_p90_post_guardrail`.
- Follow-up de timestamp: o store atual possui apenas `horizon=1` em
  `fact_oos_predictions`; nesse caso `timestamp_utc == target_timestamp_utc`
  em 7.709.077/7.709.077 linhas. A inspecao de
  `train_tft_model_use_case.py` indica que, no caminho de treino,
  `timestamp_utc` e o timestamp do primeiro alvo do decoder, nao o timestamp de
  decisao; `target_timestamp_utc` e calculado como `timestamp_utc + (horizon-1)`
  dias. Portanto, a igualdade em `h=1` e esperada pela implementacao atual e
  nao prova leakage sozinha. O contrato ainda precisa ser validado em M7 com
  `h=7`, pois nao ha `h=7` historico persistido no store atual.
- Follow-up de degeneracao `0_2_3`: todos os sub-sweeps com prefixo
  `0_2_3_*` apresentaram `0%` de linhas com `quantile_p10 == quantile_p90` e
  `0%` de linhas com `quantile_p10 == quantile_p50 == quantile_p90` no estado
  atual do store. A degeneracao em massa esta concentrada em
  `parent_sweep_id=NULL`.
- Follow-up de runs sem OOS: existem 22 runs em `dim_run` sem linhas em
  `fact_oos_predictions`; 10 possuem registro em `fact_failures` e 12 estao
  com `status=ok`, todos com `parent_sweep_id=NULL`. Esse achado reforca que
  historico pre-ScopeSpec deve ser tratado como diagnostico, nao evidencia
  confirmatoria.
- Follow-up de cobertura temporal: `fact_split_timestamps_ref` cobre 1.358 de
  6.923 runs (`19,62%`). Essa cobertura parcial nao bloqueia M6, mas nao deve
  ser assumida como fonte completa para auditorias historicas.
- Follow-up de schema Parquet: leitura agregada via PyArrow dataset falha em
  alguns caminhos por heterogeneidade de schema entre arquivos/particoes
  (`string` vs `dictionary`, `int64` vs `int32`). Leitura por arquivo fisico
  com `ParquetFile.read(columns=...)` funciona. Consumidores que dependem de
  leitura dataset-level devem normalizar schema ou ler por arquivo.

**Acao (se YELLOW/RED):**

- Antes da Fase B, nao usar tabelas gold agregadas sem `parent_sweep_id` para
  claims confirmatorios. Usar tabelas run-level/cohort-aware ou regenerar gold
  com `ScopeSpec`/filtro de coorte explicito.
- Tratar `parent_sweep_id=NULL` como historico global-health/pre-ScopeSpec:
  pode apoiar diagnostico historico, mas nao claims probabilisticos ou decisao
  estatistica confirmatoria.
- Para qualquer analise de Fase B, declarar no pre-registro o filtro de coorte
  reproduzivel (`parent_sweep_id`/escopo da rodada) e validar que os outputs
  usados preservam esse escopo ate a tabela final.
- Avaliar em M5 se as tabelas gold agregadas devem receber `parent_sweep_id`
  no output ou se devem ser substituidas por artefatos scoped-only para a
  rodada confirmatoria.
- Validar em M7, com smoke test multi-horizonte real, que `h=7` produz
  `target_timestamp_utc` coerente com o alvo previsto e que os quantis nao
  degeneram. O store historico atual nao e suficiente para fechar esse ponto.
- Antes da Fase C, auditar explicitamente `fact_inference_predictions`,
  `fact_inference_runs` e `fact_feature_contrib_local`; M6 nao certifica essas
  tabelas para analise de explicabilidade local.

---

### M7 — Capacidade operacional para Fase B

**Por que e critico:** a Fase B exige executar candidato unico vs baselines com
seeds/folds/horizontes e persistencia comparavel. Se o pipeline nao suporta isso
de forma reproduzivel, a Fase B nao deve comecar.

**Questoes a verificar:**

| # | Questao | Fonte/consulta | Veredicto |
|---|---|---|---|
| Q1 | O pipeline consegue executar 1 candidato explicito `all-features` com N seeds x K folds sem depender de selecao por feature set? | `main_train_tft`, sweeps | YELLOW |
| Q2 | Baselines podem ser persistidos em `fact_oos_predictions` no mesmo grao do TFT, com `run_id`, `parent_sweep_id`, `split`, `horizon`, `target_timestamp` e quantis quando aplicavel? | schema/repositorios | RED |
| Q3 | A poda minima all-features esta implementada ou pode ser congelada por lista pre-registrada sem usar OOS? | feature registry/pre-registro | YELLOW |
| Q4 | `h=1` e `h=7` funcionam end-to-end com quantis nao degenerados apos o fix `f7901a4`? | smoke test multi-horizonte | YELLOW |
| Q5 | Um smoke test reduzido consegue treinar, inferir, persistir e atualizar gold sem violar gates criticos? | execucao curta | RED |
| Q6 | O smoke test multi-horizonte produz quantis nao degenerados em volume suficiente para validar o caminho H>1? Reportar `n_rows`, `% p10==p90`, `% p10==p50==p90` e exemplos por horizonte. Se `n_rows >= 1000`, aplicar `% p10==p90 < 5%` como gate. Se `n_rows < 1000`, tratar como diagnostico e exigir validacao com volume maior antes da liberacao da Fase B. | smoke test + `quantile_contract_analyzer` | RED |

**Nota sobre volume do smoke test:** se for necessario aumentar `n_rows`, usar
janela OOS maior ou multiplos ativos reais que satisfaçam os mesmos criterios de
disponibilidade e qualidade de dados. Nao usar ativos sinteticos para validar o
gate academico principal.

**Nota condicional sobre rebuild do dataset:** se a Fase B exigir rebuild do
dataset all-features, validar a disponibilidade temporal minima de fundamentals
e sentiment antes da rodada confirmatoria (as-of date de fundamentals,
publication/collection timestamp de sentimento e corte da sessao usada para
construir o alvo). Se nao houver rebuild e as features ja estiverem congeladas,
auditar pelo dataset/feature registry e registrar a decisao.

**Veredicto geral:** RED

**Achados:**

- Q1: `main_train_tft` aceita config explicita, `--seed`,
  `--parent-sweep-id`, `--max-prediction-length` e
  `--evaluation-horizons`. `main_tft_param_sweep` suporta `replica_seeds` e
  `walk_forward.folds`, e propaga `parent_sweep_id` como `sweep_name`.
  Portanto, e operacionalmente possivel rodar um candidato unico com N seeds x
  K folds usando uma configuracao fixa e uma string de features all-features.
  O veredicto e YELLOW porque o caminho confirmatorio preferencial ainda precisa
  ser escolhido e validado: explicit-config deve ser avaliado primeiro; OFAT
  com `max_runs=1` deve ser tratado como fallback.
- Q1: `main_train_tft` sozinho executa uma rodada unica; a orquestracao N seeds
  x K folds fica em `main_tft_param_sweep`/`RunTFTModelAnalysisUseCase`.
- Q2: existe contrato documental em `docs/04_evaluation/BASELINES.md`, mas nao
  foi encontrada implementacao de baselines estatisticos (`zero_return`,
  random walk, media historica, AR(1), EWMA-vol ou quantis historicos) que
  persista previsoes no mesmo grao de `fact_oos_predictions`. As referencias a
  "baseline" no sweep se referem a configuracao default do TFT/OFAT, nao a
  baseline estatistico comparavel.
- Q3: a lista all-features pode ser derivada por criterios pre-definidos e
  congelada por tokens de grupo
  (`BASELINE_FEATURES`, `TECHNICAL_FEATURES`, `SENTIMENT_FEATURES`,
  `FUNDAMENTAL_FEATURES`, `MOMENTUM_LIQUIDITY_FEATURES`,
  `VOLATILITY_ROBUST_FEATURES`, `REGIME_FEATURES`,
  `SENTIMENT_DYNAMICS_FEATURES`, `FUNDAMENTAL_DERIVED_FEATURES`). O dataset
  atual de AAPL possui todas as 55 features desses grupos. Porem, nao foi
  encontrado calculo operacional da poda por criterios de
  `STRATEGIC_DIRECTION.md` §4.2 (disponibilidade temporal, missing > 30%,
  correlacao > 0,95 e redundancia derivada). Para AAPL, isso pode ser feito
  uma vez antes do pre-registro; para multiplos ativos, o ideal e transformar
  essa verificacao em quality gate deterministico do dataset.
- Q4: testes unitarios confirmam propagacao de `evaluation_horizons=[1,7,30]`
  no CLI e alinhamento de persistencia para `h=7`/`h=30` em
  `_persist_fact_oos_predictions`. Isso valida o contrato de persistencia em
  unidade, mas nao substitui smoke test end-to-end com treino real e
  `max_prediction_length >= 7`.
- Q5/Q6: smoke test reduzido nao foi executado nesta primeira passagem do M7.
  O veredicto RED aqui significa bloqueio por dependencia sequencial, nao falha
  executada: pela ordem recomendada da Fase A, o smoke test completo deve
  ocorrer depois de M2/M3 e depois da implementacao dos baselines estatisticos
  de Q2. Executar smoke confirmatorio sem baseline persistido ainda nao valida
  a capacidade completa da Fase B.
- Validacao executada nesta passagem: testes unitarios direcionados de
  multi-horizonte, guardrail, CLI de horizontes e sweep builder passaram
  (`6 passed`).

**Acao (se YELLOW/RED):**

- Antes da Fase B, implementar ou adicionar um runner de baselines que persista
  `zero_return`/random walk, media historica, AR(1), EWMA-vol e/ou quantis
  historicos no mesmo grao de `fact_oos_predictions`, com `run_id`,
  `parent_sweep_id`, `split`, `horizon`, `target_timestamp_utc` e quantis
  quando aplicavel.
- Avaliar a pipeline explicit-config como caminho preferencial para Round 1:
  uma configuracao fixa all-features, N seeds, K folds e `parent_sweep_id`
  unico. Usar OFAT com `max_runs=1` apenas como fallback se explicit-config nao
  suportar o caso confirmatorio.
- Validar por unit test ou execucao controlada que `main_tft_param_sweep
  --max-runs 1` com N seeds x K folds gera N x K runs esperados, e nao apenas
  1 run total, caso OFAT seja usado como fallback.
- Calcular a lista all-features podada para AAPL aplicando os criterios de
  `STRATEGIC_DIRECTION.md` §4.2 (disponibilidade temporal, missing > 30%,
  correlacao > 0,95 e redundancia derivada) e congelar a lista no
  pre-registro. Se a Fase B incluir multiplos ativos, preferir automatizar essa
  verificacao como quality gate de dataset antes do pre-registro.
- Apos M2/M3, executar smoke test multi-horizonte real com
  `max_prediction_length >= 7`, `evaluation_horizons=[1,7]`, escopo
  descartavel e `parent_sweep_id` proprio. Reportar `n_rows`,
  `% p10==p90`, `% p10==p50==p90` e exemplos por horizonte antes de liberar a
  Fase B.
- Deixar `h=30` fora do smoke inicial por padrao; incluir apenas se o
  pre-registro decidir que o horizonte suplementar possui `N_effective`
  defensavel.
- Tratar inferencia rolling multi-horizonte como dependencia de M4 e refresh do
  Analytics Store sob escopo Fase B como dependencia de M5.

---

## Gate de saida da Fase A

Criterio para abrir a Fase B:

- [ ] M1 GREEN ou YELLOW com acao concluida
- [ ] M2 GREEN ou YELLOW com acao concluida
- [ ] M3 GREEN ou YELLOW com acao concluida
- [ ] M4 GREEN ou YELLOW com acao concluida
- [ ] M5 GREEN ou YELLOW com acao concluida
- [ ] M6 GREEN ou YELLOW com acao concluida
- [ ] M7 GREEN ou YELLOW com acao concluida
- [ ] Nenhum modulo com veredicto RED em aberto

**Data de abertura da Fase B:** ___
**Responsavel:** Marcelo

---

## Modulos P1 (revisar antes da analise final, nao bloqueantes para Fase B)

| Modulo | Questao principal | Status |
|---|---|---|
| `run_tft_optuna_search_use_case.py` | Desabilitar ou remover `mean_test_rmse` como `objective_metric` para evitar uso acidental de test no HPO. | pendente |
| `feature_registry.py` | Todas as features all-features tem `anti_leakage_tag` e `warmup_count` corretos? Alguma feature nova sem tag? | pendente |
| `validate_analytics_quality_use_case.py` | O Block A (degeneracao) esta configurado como bloqueante (nao apenas warning) para MPIW=0 > 5%? | pendente |
| `sklearn_indicator_normalizer.py` | Confirmar que nunca e instanciado com dados alem do treino. | pendente |

---

## Registro de achados adicionais

_(preencher durante a auditoria com qualquer achado fora das questoes pre-definidas)_
