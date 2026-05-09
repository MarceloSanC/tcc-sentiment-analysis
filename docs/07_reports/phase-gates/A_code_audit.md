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
5. **M4 — inferencia.**
   Validar consistencia treino/inferencia, dropout, guardrail e rastreabilidade
   do artefato.
6. **M5 — refresh analytics.**
   Resolver mismatch `gold_metrics_by_config_n_oos_contract`, filtro
   `prediction_mode` e contrato raw vs post-guardrail.
7. **M1 — orquestracao final.**
   Consolidar riscos de orquestracao com contexto de M4/M5 fechado.

## Implementacao apos auditoria

8. PRs pequenos por contrato: gate de degeneracao (M2), baselines (M7-Q2),
   `effective_date` (M3) e correcoes M5 quando aplicavel.
9. Rebuild scoped + revalidacao depois do ultimo PR necessario.
10. Smoke confirmatorio (`max_epochs >= 5`, `n_rows >= 1000`).
11. Pre-registro com decisao raw vs post-guardrail fundamentada.
12. Fase B.

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
| Q1 | `known_real_cols` so contem features genuinamente futuro-conhecidas (calendario)? Nao ha override via config que injete features de preco/retorno? | `train_tft_model_use_case.py:1187` | GREEN |
| Q2 | O scaler interno do pytorch-forecasting (`TimeSeriesDataSet`) e fit apenas em `train_df`? O `from_dataset(training, val_df)` usa os parametros do training sem re-fit? | `pytorch_forecasting_tft_trainer.py:221-236` | GREEN |
| Q3 | A warmup policy remove corretamente os primeiros N dias antes de calcular splits, garantindo que features com `warmup_count > 0` nao produzam NaN no inicio do treino? | `train_tft_model_use_case.py:_apply_warmup_policy_to_split` | GREEN |
| Q4 | Ha algum caminho de codigo onde `test_df` e usado como criterio de HPO (mesmo indiretamente, como via `mean_test_rmse` sendo o objetivo padrao)? | `run_tft_optuna_search_use_case.py:49,176-184` | RED |

**Veredicto geral:** RED

**Achados:**

- Q1: `known_real_cols` e fixado em
  `["time_idx", "day_of_week", "month"]` quando essas colunas existem no
  dataset. Nao ha leitura de `training_config` para sobrescrever
  `known_real_cols`, nem caminho observado que injete preco, retorno,
  sentimento, fundamentos ou indicadores tecnicos como future-known. Esse
  contrato tambem bate com M3-Q3.
- Q1: `feature_cols` e resolvido por `_resolve_features(...)` a partir dos
  grupos/colunas explicitamente solicitados, mas sempre e passado ao trainer
  como `time_varying_unknown_reals`; a lista de known reals permanece separada
  e hardcoded no use case de treino.
- Q2: a normalizacao externa de features tecnicas em
  `_apply_split_feature_normalization(...)` e fitada apenas com `train_df`
  (`StandardScaler.fit(...)` sobre serie de treino) e depois aplicada em
  `train_df`, `val_df` e `test_df`. Nao ha fit em `val_df` ou `test_df`.
- Q2: o scaler interno do `pytorch-forecasting` e fitado no
  `TimeSeriesDataSet(train_df, ...)`; validacao e teste sao criados via
  `TimeSeriesDataSet.from_dataset(training, val_df/test_df, predict=False,
  stop_randomization=True)`. Esse e o caminho correto para reaproveitar os
  parametros do dataset de treino sem refit em OOS.
- Q3: a warmup policy roda antes do split temporal: `execute(...)` resolve
  `feature_cols`, executa `_run_pretrain_quality_gate(...)`, calcula
  `known_real_cols`, aplica `_apply_warmup_policy_to_split(...)` e so depois
  chama `_apply_time_split(...)`. Portanto, quando `warmup_policy=drop_leading`,
  o `train_start` efetivo e ajustado antes de construir `train_df`/`val_df`/
  `test_df`.
- Q3: `warmup_policy=strict_fail` aborta quando ha leading null warmup no
  periodo de treino solicitado; `drop_leading` ajusta `train_start` para a
  primeira data valida sugerida por `FeatureWarmupInspector` e registra
  `effective_train_start`, `required_warmup_count`, `warmup_features` e
  `warmup_applied` no metadata. Testes unitarios cobrem strict fail,
  drop-leading e revalidacao de amostra minima apos o ajuste.
- Q4: o caminho padrao de Optuna usa `objective_metric="robust_score"`; quando
  `robust_score` nao esta disponivel, o fallback usa `mean_val_rmse` e
  `std_val_rmse`. O default, portanto, nao usa test set como objetivo.
- Q4: apesar do default seguro, `RunTFTOptunaSearchUseCase` aceita
  `objective_metric="mean_test_rmse"` e `"joint_val_test_rmse"`; a CLI
  `main_tft_optuna_sweep.py` tambem expõe essas opcoes em
  `--objective-metric`. Nesses modos, `_objective_from_summary(...)` usa
  `mean_test_rmse` diretamente ou mistura `mean_val_rmse` com
  `mean_test_rmse`. Isso e leakage metodologico para HPO/model selection se
  usado em qualquer rodada exploratoria que alimente decisao confirmatoria.
- Q4: `RunTFTModelAnalysisUseCase` tambem ordena rankings por
  `robust_score`, `mean_val_rmse`, `mean_test_rmse`, `mean_val_mae` e
  `mean_test_mae`. Isso e aceitavel para relatorio diagnostico, mas o ranking
  usado para escolha de hiperparametros confirmatorios precisa excluir metricas
  de test.
- Validacao executada nesta passagem:
  - inspecao de codigo com `rg`/`sed` em `train_tft_model_use_case.py`,
    `pytorch_forecasting_tft_trainer.py`, `run_tft_optuna_search_use_case.py`,
    `run_tft_model_analysis_use_case.py` e `main_tft_optuna_sweep.py`;
  - inspecao de testes unitarios de warmup em
    `tests/unit/use_cases/test_train_tft_model_use_case.py`.
  - suite dirigida passou:
    `tests/unit/use_cases/test_train_tft_model_use_case.py`,
    `tests/unit/adapters/test_pytorch_forecasting_tft_trainer.py`,
    `tests/unit/domain/services/test_explicit_config_sweep_analysis_service.py`
    e `tests/unit/domain/services/test_tft_sweep_experiment_builder.py`
    (`41 passed`).

**Acao (se YELLOW/RED):**

- Antes da Fase B, remover ou bloquear `mean_test_rmse` e
  `joint_val_test_rmse` como `objective_metric` em Optuna/HPO. O contrato
  aceitavel para HPO deve permitir apenas objetivo baseado em validacao
  (`robust_score`/`mean_val_rmse`) ou outro criterio pre-registrado que nao use
  test set.
- Atualizar `main_tft_optuna_sweep.py` e `RunTFTOptunaSearchUseCase` para
  rejeitar objetivos baseados em test set quando `scope_mode`/rodada estiver
  marcada como exploratoria para suporte a Fase B ou confirmatoria. Alternativa
  mais simples: remover essas duas opcoes da CLI e da lista suportada.
- Garantir no pre-registro que qualquer hiperparametro herdado de Round 0 foi
  escolhido por criterio de validacao/robustez, nao por `mean_test_rmse`.

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
| Q1 | O `sklearn_indicator_normalizer` (`fit()`) e chamado ANTES do split temporal ou DEPOIS? Se antes, leakage de normalizacao garantido. | `build_tft_dataset_use_case.py`: buscar `normalizer.fit` / `sklearn_indicator_normalizer` | GREEN |
| Q2 | Features com `warmup_count > 0` no `feature_registry`: o dataset exclui os primeiros N dias dessas features corretamente, ou os inclui com NaN? | `feature_registry.py`: verificar `warmup_count`; `build_tft_dataset_use_case.py`: verificar aplicacao | YELLOW |
| Q3 | Todas as features de retorno/preco estao em `time_varying_unknown_reals`, nao em `time_varying_known_reals`? Confirmar que o `known_real_cols` do trainer bate com o que foi definido aqui. | `build_tft_dataset_use_case.py:228-229` | GREEN |
| Q4 | Se a Fase B exigir rebuild do dataset all-features, fundamentals e sentiment respeitam disponibilidade temporal real (as-of date, publication/collection timestamp e corte da sessao)? | feature registry + fontes processadas; ver tambem M7 | YELLOW |

**Veredicto geral:** YELLOW

**Achados:**

- Q1: `BuildTFTDatasetUseCase` nao instancia nem chama
  `SklearnTechnicalIndicatorNormalizer`; a busca por
  `SklearnTechnicalIndicatorNormalizer`, `TechnicalIndicatorNormalizer`,
  `normalizer.fit`, `normalizer.transform` e `sklearn_indicator_normalizer`
  encontrou apenas a interface/adaptador legado e seus testes unitarios.
- Q1: a normalizacao efetiva de features tecnicas ocorre depois do split em
  `TrainTFTModelUseCase._apply_split_feature_normalization`: o `StandardScaler`
  e ajustado apenas em `train_df` e aplicado depois em `train_df`, `val_df` e
  `test_df`. Portanto nao ha evidencia de scaler fitado no dataset inteiro no
  caminho atual.
- Q2: o `feature_registry` declara `warmup_count` para features rolling e o
  `DatasetQualityGate` recebe `FEATURE_WARMUP_BARS`, ignorando NaNs iniciais
  dentro da janela de warmup ao avaliar razao de nulos. Os testes cobrem esse
  comportamento (`test_quality_gate_ignores_nan_inside_feature_warmup_window`
  e `test_quality_gate_still_rejects_nan_after_feature_warmup_window`).
- Q2: o build do dataset nao remove fisicamente as primeiras linhas com warmup;
  ele mantem os NaNs iniciais e delega a seguranca do split para o treino
  (`warmup_policy` em `TrainTFTModelUseCase`). Isso e aceitavel se a Fase B
  usar `strict_fail`/`drop_leading` com registro no pre-registro, mas nao
  certifica o parquet final como pronto para qualquer consumidor arbitrario.
- Q3: no caminho de treino, `known_real_cols` e fixado em
  `["time_idx", "day_of_week", "month"]` quando essas colunas existem. O
  trainer passa esse conjunto para `time_varying_known_reals` e passa
  `feature_cols` para `time_varying_unknown_reals`. Nao foi encontrado override
  de config que injete preco, retorno, tecnicos, sentiment ou fundamentals em
  `known_real_cols`.
- Q4: fundamentals possuem `reported_date` no schema e o build calcula
  `effective_date` a partir de `reported_date` ou, quando ausente, usa fallback
  `fiscal_date_end + 45 dias`; o merge diario e `pd.merge_asof(...,
  direction="backward")`. No estado atual de AAPL, 17/81 reports possuem
  `reported_date` nulo, o primeiro dia do dataset com fundamentals
  (`2010-04-20`) coincide com o primeiro `effective_day`, e nao foi encontrado
  uso de `revenue` antes de sua menor data efetiva observada.
- Q4: sentiment preserva `published_at` no `scored_news` e a engenharia diaria
  possui guard de causalidade por `trading_day_from_timestamp`. No estado atual
  de AAPL, a contagem de `daily_sentiment.n_articles` bate com a contagem de
  `scored_news` por dia util efetivo usando `close_hour=16:00` e
  `weekends=false` para dias uteis; as 269 divergencias encontradas estao em
  linhas de fim de semana do `daily_sentiment`, que nao entram no dataset TFT
  diario de candles.
- Q4: apesar disso, o dataset final nao preserva `effective_date` de
  fundamentals nem timestamps/artigos de origem do sentiment. A disponibilidade
  temporal real fica auditavel apenas cruzando com `fundamentals_AAPL.parquet`
  e `scored_news_AAPL.parquet`, nao a partir de `dataset_tft_AAPL.parquet`
  isoladamente. Por isso Q4 permanece YELLOW ate a lista all-features da Fase B
  ser congelada e a auditoria source-level ser anexada ao pre-registro.
- Validacao executada:
  `tests/unit/use_cases/test_build_tft_dataset_use_case.py`,
  `tests/unit/domain/services/test_dataset_quality_gate.py`,
  `tests/unit/infrastructure/schemas/test_feature_validation_schema.py`,
  `tests/unit/infrastructure/schemas/test_feature_registry.py` e
  `tests/unit/test_main_dataset_tft.py` passaram (`25 passed`).

**Acao (se YELLOW/RED):**

- Para a Fase B, declarar no pre-registro a politica de warmup usada
  (`strict_fail` ou `drop_leading`) e registrar `required_warmup_count`,
  `warmup_features`, `effective_train_start` e `feature_list_ordered`.
- Antes de congelar o candidato all-features, gerar um artefato curto de
  auditoria source-level para AAPL confirmando:
  `reported_date`/fallback usado em fundamentals, `effective_day` por report,
  ausencia de uso antes da disponibilidade efetiva, e consistencia de
  `daily_sentiment` contra `scored_news.published_at` com a politica
  `open_hour=09:30`, `close_hour=16:00`, `weekends=false`.
- Se a Fase B incluir outros ativos alem de AAPL, repetir a auditoria
  source-level por ativo antes do pre-registro. O sistema pode ser multi-asset,
  mas cada modelo/rodada confirmatoria deve ter dataset e disponibilidade
  temporal auditados por ativo.
- Preservar `effective_date` (renomeado para
  `fundamentals_effective_date` para nao colidir com nomenclatura temporal
  existente) como coluna do dataset final, removendo o `df.drop(...)` em
  [build_tft_dataset_use_case.py:519](src/use_cases/build_tft_dataset_use_case.py#L519).
  Mudanca de 1-2 linhas; rebuild do dataset (~segundos para AAPL, <1 min
  multi-asset) deve ser bundled com as outras alteracoes resultantes da Fase A
  (gate de degeneracao M2, baselines persistidos M7, eventuais fixes M1/M4/M5)
  para evitar dois ciclos de rebuild + revalidacao. Beneficio: rastreabilidade
  direta no dataset (sem cruzar com fontes processadas), citacao direta no TCC
  e auditoria multi-asset por inspecao de coluna em vez de script ad-hoc.
- Documentar no pre-registro a justificativa da escolha "+45 dias" como
  fallback de `reported_date` quando ausente, incluindo: motivacao
  (compromisso entre prazos SEC de 40 dias para 10-Q e 60 dias para 10-K),
  cobertura de impacto (proporcao de reports com fallback aplicado vs
  `reported_date` original; em AAPL = 17/81 = 21%) e analise de sensibilidade
  caso a janela mude. A escolha hoje so aparece em
  [build_tft_dataset_use_case.py:118](src/use_cases/build_tft_dataset_use_case.py#L118)
  sem documentacao metodologica em nenhum doc canonico.

---

### M4 — `run_tft_inference_use_case.py` + `pytorch_forecasting_tft_inference_engine.py`

**Por que e critico:** inferencia deve reproduzir exatamente o protocolo de
treino. Divergencias silenciosas (config, scaler, seed) invalidam a
comparacao treino vs inferencia.

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | O modelo e carregado com os mesmos `known_real_cols` e `feature_cols` usados no treino? Ou sao recalculados dinamicamente e podem divergir se a config mudar? | `run_tft_inference_use_case.py`: buscar `known_real_cols` / `feature_cols` no loader | GREEN |
| Q2 | Seed de inferencia: o modelo e colocado em `eval()` mode (desliga dropout)? Ou ha dropout ativo produzindo quantis diferentes entre runs? | `pytorch_forecasting_tft_inference_engine.py`: buscar `.eval()` / `torch.no_grad()` | GREEN |
| Q3 | O guardrail monotonico (`quantile_p*_post_guardrail`) e aplicado na inferencia da mesma forma que no treino? Ou so e aplicado no refresh do analytics store? | `run_tft_inference_use_case.py`: buscar `guardrail` / `post_guardrail` | GREEN |
| Q4 | A seed de inferencia e registrada em `fact_config` ou `dim_run` para reproducibilidade? | `run_tft_inference_use_case.py` / `parquet_analytics_run_repository.py` | YELLOW |

**Veredicto geral:** RED

RED qualificado: bloqueia o uso de inferencia rolling multi-horizonte e
exemplos locais `h=7`/`h=30` na Fase C, mas nao bloqueia sozinho a Fase B
confirmatoria enquanto H1/H2 usarem `fact_oos_predictions`, que ja e
multi-horizonte no caminho de treino/OOS.

**Achados:**

- Q1: a inferencia nao recalcula `feature_cols` por config dinamica. O loader
  le `features.json` e retorna `feature_cols` do artefato treinado; o use case
  valida que todas essas features existem no `dataset_tft` antes de chamar o
  engine. Features extras no dataset apenas geram log informativo, nao entram
  no modelo.
- Q1: `known_real_cols` nao aparece como lista explicita no loader, mas a
  reconstrucao do dataset de inferencia usa `dataset_parameters.pkl` via
  `TimeSeriesDataSet.from_parameters(..., predict=False,
  stop_randomization=True)`. Esse e o caminho correto para preservar a
  especificacao do `TimeSeriesDataSet` do treino, incluindo known/unknown
  reals, normalizadores internos e estrutura temporal.
- Q1: o use case tambem aplica `scalers.pkl` carregado do artefato e protege
  colunas estruturais (`time_idx`, `timestamp`, `asset_id`,
  `target_return`) contra transformacao. Isso alinha inferencia com a
  normalizacao split-aware do treino.
- Q2: o loader chama `model.eval()` apos carregar o checkpoint. O engine usa
  dataloader com `train=False`, `stop_randomization=True` e, no fallback de
  forward manual, chama `model.eval()` e `torch.no_grad()`. Nao ha evidencia de
  dropout ativo no caminho de inferencia.
- Q3: a persistencia de inferencia aplica o mesmo
  `QuantileGuardrailService.enforce_monotonic_triplet` usado no treino e grava
  `quantile_p10_post_guardrail`, `quantile_p50_post_guardrail`,
  `quantile_p90_post_guardrail` e `quantile_guardrail_applied` em
  `fact_inference_predictions`. Teste unitario confirma a presenca dessas
  colunas na persistencia.
- Q4: nao ha `seed` em `fact_inference_runs` nem em
  `fact_inference_predictions`, e inferencia nao cria `dim_run`/`fact_config`.
  A semente de treino fica indiretamente no `config.json`/`training_config` do
  artefato e no `model_version`, mas nao e materializada no Analytics Store de
  inferencia. Como o caminho esta em `eval()` e deve ser deterministico, isso
  nao e bloqueio imediato; para rastreabilidade da Fase C, a seed/config do
  modelo deveria ser persistida ou referenciada explicitamente.
- Achado fora das quatro perguntas, mas critico para o direcionamento atual:
  `PytorchForecastingTFTInferenceEngine.infer(...)` recebe
  `max_prediction_length`, mas registra sempre `horizon=1` e usa apenas o
  primeiro decoder step (`arr[:, 0]` / `decoder_time_idx[:, 0]`). Portanto, a
  inferencia rolling atual nao entrega `h=7`/`h=30`, mesmo quando o modelo foi
  treinado com `max_prediction_length >= 7`. Isso conflita com o contrato P2
  de multiplos horizontes e deve ser corrigido antes de usar inferencia rolling
  como evidencia operacional ou exemplo local multi-horizonte no TCC. Esse
  achado nao invalida `fact_oos_predictions` para metricas confirmatorias.
- Achado relacionado a explicabilidade local: `fact_feature_contrib_local`
  usa metodo `local_magnitude_signed_v1`, baseado em magnitude da feature no
  `inference_slice`, nao VSN/permutation/ablation. Esse metodo pode ser util
  como placeholder operacional, mas nao deve ser usado como evidencia
  academica principal de contribuicao na Fase C. Conforme
  `docs/04_evaluation/EXPLAINABILITY.md`, a evidencia primaria deve vir de VSN
  weights agregados, permutation importance por familia e ablation explicativa.
- Validacao executada nesta passagem:
  `tests/unit/use_cases/test_run_tft_inference_use_case.py`,
  `tests/unit/adapters/test_pytorch_forecasting_tft_inference_engine.py`,
  `tests/unit/adapters/test_local_tft_inference_model_loader.py` e
  `tests/unit/infrastructure/schemas/test_analytics_store_schema.py`
  passaram (`50 passed`).

**Acao (se YELLOW/RED):**

- Se a Fase B/C exigir casos ilustrativos via inferencia rolling para
  `h=7`/`h=30`, implementar persistencia multi-horizonte real no engine de
  inferencia: uma linha por (`inference_run_id`, `decision_timestamp_utc`,
  `target_timestamp_utc`, `horizon`) para os horizontes habilitados ou
  pre-registrados. Caso contrario, registrar como future work e manter a Fase B
  confirmatoria em `fact_oos_predictions`.
- Se a persistencia multi-horizonte de inferencia rolling for implementada,
  adicionar testes unitarios cobrindo `max_prediction_length=7` e validando que
  o engine persiste `horizon=1` e `horizon=7`, com `target_timestamp_utc`
  coerente com a semantica decidida no pre-registro.
- Nao criar coluna nova de rastreabilidade sem necessidade. No minimo,
  documentar em contrato/ADR que `model_version + model_path + config.json` e a
  fonte canonica de seed/config para inferencia; se isso for insuficiente para
  auditoria query-only, materializar `model_run_id`/`training_run_id` ou
  `training_seed` no Analytics Store de inferencia.
- Antes da Fase C, marcar `local_magnitude_signed_v1` como diagnostico
  operacional ilustrativo, nao evidencia academica primaria. A evidencia de
  contribuicao para o TCC deve vir de: VSN weights agregados (global por
  horizonte e por regime), permutation importance por familia com bootstrap
  IC95 e ablation explicativa. Considerar adicionar `evidence_tier`
  (`primary`/`illustrative`) ou criar tabela separada para evidencia academica
  primaria.

---

### M5 — `refresh_analytics_store_use_case.py`

**Por que e critico:** calcula todas as metricas do paper. PICP calculado
sobre quantis raw vs pos-guardrail produz numeros diferentes. Scope incorreto
mistura sweeps na mesma metrica.

**Questoes a verificar:**

| # | Questao | Arquivo:linha | Veredicto |
|---|---|---|---|
| Q1 | PICP e calculado sobre `quantile_p10_post_guardrail`/`quantile_p90_post_guardrail` ou sobre `quantile_p10`/`quantile_p90` raw? As duas colunas existem — qual e usada como primaria? | `refresh_analytics_store_use_case.py`: buscar `picp` / `in_interval` | RED |
| Q2 | As tabelas gold `gold_prediction_metrics_by_config`, `gold_prediction_calibration` e `gold_prediction_robustness_by_horizon` recebem `parent_sweep_id` no output? Se nao, como filtrar escopo na analise final? | Confirmado: nenhuma das tres tem `parent_sweep_id`. Verificar se ha filtro pre-calculo ou so pos-calculo. | RED |
| Q3 | O `ScopeSpec` com `scope_mode=cohort_decision` e passado corretamente em todos os paths de calculo que alimentam decisao estatistica (DM, MCS, win-rate)? Ou so no quality gate? | `refresh_analytics_store_use_case.py`: buscar `scope_spec` / `ScopeSpec` | YELLOW |
| Q4 | O check `gold_metrics_by_config_n_oos_contract` falhou com `mismatch_with_run_level=13800` no estado historico atual (smoke M7 evidenciou isso). Quais sao as causas (runs pre-ScopeSpec entrando em agregacao, recomputacao parcial, mudanca de regra entre versoes)? Esse check e global ou pode ser scoped via `ScopeSpec`? E corrigivel via re-refresh ou exige migracao de schema? | `gold_prediction_metrics_by_config.parquet` + `refresh_analytics_store_use_case.py` (busca pelo check de `n_oos_contract`) | RED |
| Q5 | Filtro de metricas probabilisticas por `prediction_mode='quantile'` AND `is_quantile_genuine` (proposto em M2): runs `point` ou `quantile` degenerados antes do gate de M2 entrariam silenciosamente em PICP/MPIW/pinball/calibracao globais. O `refresh_analytics_store_use_case.py` filtra por modo? Se nao, runs historicos com `parent_sweep_id=NULL` (91% degenerados) podem ter contaminado metricas globais. | `refresh_analytics_store_use_case.py` + `fact_config.prediction_mode` | RED |
| Q6 | Invariante estrutural vigente desde `9f0ccec` (2026-03-10): `config_signature` herda escopo porque `compute_config_signature(...)` hashia o `training_config` canonico, e `_build_trainer_config(...)` preserva `parent_sweep_id`, `fold`, `trial_number` e `seed` (via `TFT_TRAINING_DEFAULTS`). Existe teste de regressao garantindo que configs iguais com esses campos diferentes geram assinaturas diferentes? | `analytics_store_schema.py:37`, `train_tft_model_use_case.py:1437`, `model_artifact_schema.py:13-24` | YELLOW |

**Veredicto geral:** RED

**Achados:**

- Q1: as metricas oficiais de `gold_prediction_metrics_by_run_split_horizon`
  usam quantis raw por default. A funcao
  `_build_gold_prediction_metrics_by_run_split_horizon(...)` recebe
  `quantile_columns=("quantile_p10", "quantile_p50", "quantile_p90")` e calcula
  `picp`, `mpiw`, `pred_interval_width`, `pinball_q*` e `mean_pinball` sobre
  essas colunas. O refresh tambem calcula `gold_quantile_guardrail_audit`
  comparando raw vs post-guardrail, mas esse audit nao substitui a tabela
  oficial de metricas.
- Q1: no estado atual, `gold_quantile_guardrail_audit` contem 4 linhas do smoke
  M7 e mostra diferencas materiais entre raw e post-guardrail: `picp`,
  `mpiw`, `mean_pinball`, `coverage_error` e `confidence_calibrated` mudaram em
  4/4 linhas; `crossing_before_count=402`, `crossing_after_count=0`,
  `negative_width_before_count=326` e `negative_width_after_count=0`. Portanto,
  a decisao raw vs post-guardrail nao e cosmetica; muda metricas do paper.
- Q1: `gold_prediction_metrics_by_run_split_horizon` possui 4 linhas com
  `mpiw < 0`, todas compatíveis com o smoke M7 raw crossing. Isso confirma que
  as metricas oficiais atuais ainda podem carregar intervalos raw invalidos.
- Q2: `gold_prediction_metrics_by_config` (14.189 linhas),
  `gold_prediction_calibration` (14.189 linhas) e
  `gold_prediction_robustness_by_horizon` (96.628 linhas) nao possuem
  `parent_sweep_id` no output. As tres sao derivadas de
  `gold_prediction_metrics_by_run_split_horizon`, que preserva
  `parent_sweep_id` (1.354 linhas nao-nulas; 13 coortes), mas
  perdem o campo ao agregar/selecionar colunas.
- Q2: como `RefreshAnalyticsStoreUseCase` nao recebe `ScopeSpec`, essas tabelas
  sao geradas globalmente. Sem `parent_sweep_id`, nao ha filtro pos-calculo
  inequivoco para separar coorte confirmatoria de historico pre-ScopeSpec ou
  smoke. Elas podem ser usadas como diagnostico global, mas nao como fonte
  direta para claims confirmatorios.
- Q3: os artefatos estatisticos pareados preservam `parent_sweep_id`:
  `gold_dm_pairwise_results`, `gold_mcs_results`,
  `gold_win_rate_pairwise_results`, `gold_paired_oos_intersection_by_horizon`,
  `gold_quality_statistics_report` e `gold_model_decision_final`. Isso reduz o
  risco para DM/MCS/win-rate, desde que a analise filtre explicitamente a
  coorte. Porem, o refresh em si nao aplica `ScopeSpec`; o escopo existe no
  quality gate (`ValidateAnalyticsQualityUseCase`) e nos campos persistidos,
  nao como contrato de geracao scoped das gold tables.
- Q3: decisao arquitetural pendente: manter refresh global e exigir
  `parent_sweep_id` nas tabelas agregadas, ou implementar refresh scoped que
  gera artefatos gold isolados por coorte. Para a Fase B inicial, o caminho
  mais barato e suficiente tende a ser refresh global + tabelas agregadas
  cohort-aware. Refresh scoped completo pode ficar como future work se o custo
  de cache/invalidation nao se justificar agora.
- Q4: o check `gold_metrics_by_config_n_oos_contract` passa em modo
  `global_health`: `non_positive_n_oos=0`, `mismatch_with_run_level=0`.
  Portanto, o mismatch de 13.800 nao e causado por corrupcao global de
  `n_oos`, recomputacao parcial ou mudanca de regra entre versoes.
- Q4: o mesmo check falha em `cohort_decision` para o smoke M7:
  `mismatch_with_run_level=13800`. Causa-raiz: assimetria de chave de filtro.
  O quality gate filtra `gold_prediction_metrics_by_run_split_horizon` por
  `parent_sweep_id`, `split` e `horizon`; em
  `gold_prediction_metrics_by_config`, como `parent_sweep_id` nao esta no
  output, o gate filtra apenas por `split`/`horizon`. O escopo `val,test` +
  `h=1,7` deixa 13.804 grupos em `gold_prediction_metrics_by_config`; apenas
  4 pertencem ao smoke M7, e os outros 13.800 viram mismatch. A informacao de
  coorte continua recuperavel por `config_signature` para runs gerados pelo
  contrato atual com `parent_sweep_id` nao nulo (ver Q6), mas o gate nao faz
  esse join.
- Q4: o problema e corrigivel por contrato/codigo, nao por purge ou re-refresh
  isolado. As opcoes corretas sao tornar o output agregado explicitamente
  cohort-aware com `parent_sweep_id`, ou manter o schema atual e obrigar
  consumidores/gates a derivarem as `config_signature` da coorte via
  `dim_run`/`fact_config`.
- Q5: `fact_config.prediction_mode` existe e, no estado atual lido, todos os
  6.900 registros estao como `quantile`. O refresh nao filtra por
  `prediction_mode` nem por `is_quantile_genuine` antes de calcular PICP/MPIW/
  pinball/calibracao. Tambem nao ha coluna `is_quantile_genuine` persistida no
  contrato atual.
- Q5: isso significa que runs historicos configurados como `quantile`, mas com
  outputs degenerados, entram nas metricas probabilisticas globais se nao forem
  removidos por escopo. O M6 ja mostrou que `parent_sweep_id=NULL` concentra
  91,25% de linhas raw degeneradas; logo, metricas probabilisticas globais
  pre-ScopeSpec nao devem ser usadas como evidencia.
- Q6: a invariante estrutural foi confirmada no codigo para o contrato atual e
  tem data de inicio conhecida. `compute_config_signature(...)` remove apenas
  chaves temporais explicitas (`created_at`, `started_at`, `ended_at`,
  `timestamp`) e hashia o JSON canonico restante. Desde `9f0ccec`
  (2026-03-10), `_build_trainer_config(...)` preserva `parent_sweep_id`,
  `fold`, `trial_number` e `seed` (via `TFT_TRAINING_DEFAULTS`). Portanto, em
  runs gerados por esse caminho com `parent_sweep_id` nao nulo, dois sweeps com
  `parent_sweep_id` distinto geram `config_signature` distinto mesmo se os
  hiperparametros forem iguais.
- Q6: escopo da invariante: vale plenamente para Round `0_2_3_*` e para a
  Fase B futura, assumindo `parent_sweep_id` populado. Nao deve ser usada para
  reinterpretar o historico pre-ScopeSpec com `parent_sweep_id=NULL`: esse
  bloco atravessa a mudanca de `9f0ccec` e segue tratado como diagnostico
  global, nao evidencia confirmatoria.
- Q6: essa invariante explica o overlap zero observado em M6 quando o grao
  inclui `config_signature` em runs pos-contrato com coorte populada. Risco
  residual: mudancas futuras em `_build_trainer_config` ou
  `compute_config_signature` podem quebrar essa propriedade sem aviso. Falta
  teste de regressao explicito.
- Validacao executada nesta passagem:
  - inspecao de codigo com `rg`/`sed` em `refresh_analytics_store_use_case.py`
    e `validate_analytics_quality_use_case.py`;
  - leitura read-only de Parquets em `data/analytics/gold` e `data/analytics/silver`;
  - execucao read-only de `ValidateAnalyticsQualityUseCase` em `global_health`
    e em `cohort_decision` para o prefixo
    `phase_a_m7_smoke_20260501_tft_only`.

**Acao (se YELLOW/RED):**

- Antes da Fase B, decidir no pre-registro e implementar no refresh qual e o
  contrato primario das metricas probabilisticas: raw, post-guardrail ou ambos
  com colunas separadas e nomenclatura explicita. Enquanto essa decisao nao
  estiver fechada, PICP/MPIW/pinball das gold tables agregadas nao devem
  sustentar claims confirmatorios.
- Tornar as tabelas gold usadas para ranking/calibracao/robustez
  cohort-aware. Opcao preferida para a Fase B: incluir `parent_sweep_id` no
  output de `gold_prediction_metrics_by_config`,
  `gold_prediction_calibration`, `gold_prediction_robustness_by_horizon` e
  tabelas agregadas correlatas. Para runs pos-`9f0ccec` com
  `parent_sweep_id` populado, isso nao deve mudar a cardinalidade de runs
  gerados pelo caminho oficial, porque `config_signature` ja inclui
  `parent_sweep_id`; o beneficio e permitir filtro direto e reduzir bugs de
  consumidores. Historico `parent_sweep_id=NULL` continua fora de claims
  confirmatorios.
- Alternativa aceitavel: manter o schema agregado atual e fazer todos os
  consumidores/gates recuperarem escopo por `config_signature` derivada de
  `dim_run`/`fact_config`. Essa opcao evita mudanca de schema, mas e mais
  fragil porque cada consumidor precisa lembrar do join.
- Ajustar `gold_metrics_by_config_n_oos_contract` para operar sobre a mesma
  semantica de escopo dos dois lados da comparacao. Se as gold agregadas
  receberem `parent_sweep_id`, filtrar diretamente por esse campo nos dois
  lados. Se o schema atual for mantido, filtrar
  `gold_prediction_metrics_by_config` por
  `config_signature.isin(scoped_config_signatures)`, onde
  `scoped_config_signatures` vem de `dim_run`/`fact_config` filtradas pela
  coorte.
- Implementar ou materializar `is_quantile_genuine` apos o gate de degeneracao
  de M2, e fazer o refresh filtrar metricas probabilisticas por
  `prediction_mode='quantile'` e `is_quantile_genuine=True` quando calcular
  PICP/MPIW/pinball/calibracao. Runs `point` ou `quantile` degenerados devem
  continuar podendo alimentar metricas pontuais, mas nao metricas
  probabilisticas.
- Alternativa aceitavel para a primeira implementacao: calcular
  `is_quantile_genuine` dinamicamente no refresh a partir de
  `prediction_mode='quantile'` e `quantile_p10 != quantile_p90`, em vez de
  persistir uma nova coluna imediatamente. Persistir o campo so deve ser
  necessario se multiplos consumidores precisarem da mesma regra materializada.
- Adicionar teste de regressao para a invariante estrutural de
  `config_signature`: duas configs iguais devem gerar assinaturas diferentes
  quando `parent_sweep_id`, `fold`, `trial_number` ou `seed` forem diferentes.
  Esse teste deve documentar que a invariante foi introduzida em `9f0ccec`
  (2026-03-10) e protege a semantica de escopo usada pelas tabelas gold
  agregadas.
- Para os resultados ja persistidos, tratar `gold_prediction_metrics_by_config`,
  `gold_prediction_calibration` e `gold_prediction_robustness_by_horizon` como
  diagnostico historico/global, nao como evidencia confirmatoria. Para qualquer
  analise antes do fix, usar `gold_prediction_metrics_by_run_split_horizon`
  filtrado por coorte ou recalcular metricas em artefato scoped.

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
- Q3/Q6: nota sobre o overlap zero entre coortes quando `config_signature` esta
  no grao: para runs pos-`9f0ccec` (2026-03-10) com `parent_sweep_id` nao nulo,
  isso e consequencia do contrato atual, nao coincidencia empirica.
  `config_signature` hashia o `training_config` canonico, e o caminho oficial
  preserva `parent_sweep_id`, `fold`, `trial_number` e `seed` na configuracao
  hashada (ver M5-Q6). Runs com `parent_sweep_id` distintos devem ter
  `config_signature` distintos enquanto essa invariante for preservada. O
  bloco pre-ScopeSpec com `parent_sweep_id=NULL` continua sendo tratado como
  diagnostico global.
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

- Antes da Fase B, nao usar tabelas gold agregadas sem mecanismo cohort-aware
  para claims confirmatorios. Usar tabelas run-level, tabelas agregadas com
  `parent_sweep_id` direto, ou filtro por `config_signature` derivada da coorte
  em `dim_run`/`fact_config`.
- Tratar `parent_sweep_id=NULL` como historico global-health/pre-ScopeSpec:
  pode apoiar diagnostico historico, mas nao claims probabilisticos ou decisao
  estatistica confirmatoria.
- Para qualquer analise de Fase B, declarar no pre-registro o filtro de coorte
  reproduzivel (`parent_sweep_id`/escopo da rodada) e validar que os outputs
  usados preservam esse escopo ate a tabela final.
- M5 concluiu que as tabelas gold agregadas precisam se tornar cohort-aware
  antes de claims confirmatorios, seja por `parent_sweep_id` direto no output,
  seja por join/filtro obrigatorio via `config_signature` da coorte.
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
| Q4 | `h=1` e `h=7` funcionam end-to-end com quantis nao degenerados apos o fix `f7901a4`? | smoke test multi-horizonte | GREEN (com nota: validado com `n=402`, `max_epochs=1`; re-validar em rodada confirmatoria com volume e convergencia maiores) |
| Q5 | Um smoke test reduzido consegue executar treino, persistencia OOS, refresh sob scope e quality gate produzindo diagnostico acionavel (sem travamentos, sem dados corrompidos, com falhas do gate vinculadas a causas identificaveis)? | execucao curta | YELLOW |
| Q6 | O smoke test multi-horizonte produz quantis nao degenerados em volume suficiente para validar o caminho H>1? Reportar `n_rows`, `% p10==p90`, `% p10==p50==p90` e exemplos por horizonte. Se `n_rows >= 1000`, aplicar `% p10==p90 < 5%` como gate. Se `n_rows < 1000`, tratar como diagnostico e exigir validacao com volume maior antes da liberacao da Fase B. | smoke test + `quantile_contract_analyzer` | YELLOW |

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

Motivo do RED: o caminho TFT-only multi-horizonte funciona end-to-end (treino,
persistencia, refresh, quality gate detectando problemas reais), mas a Fase B
ainda nao esta operacionalmente pronta porque (a) Q2 baselines estatisticos
nao foram implementados; (b) decisao raw vs post-guardrail como contrato
primario do gate continua aberta; (c) `gold_metrics_by_config_n_oos_contract`
falha com 13.800 mismatches historicos que precisam ser tratados em M5.

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
- Q4: smoke test TFT-only multi-horizonte executado em 2026-05-01 com
  `parent_sweep_id=phase_a_m7_smoke_20260501_tft_only`,
  `max_prediction_length=7`, `evaluation_horizons=[1,7]`, `max_epochs=1`,
  `seed=20260501`, features all-features via tokens de grupo e artefatos de
  modelo em `/tmp/phase_a_m7_smoke_models`. O treino concluiu e persistiu uma
  rodada `status=ok` com `run_id=9f9697b03a8fa3a538096064453a53acade18cf3ac4a19742cc845be32f52dee`,
  `feature_set_name=BTSFDVRYQ`, 55 features em `bridge_run_features`, 2 linhas
  em `fact_split_metrics`, 3 linhas em `fact_split_timestamps_ref` e 402 linhas
  em `fact_oos_predictions`.
- Q4: o smoke validou o caminho TFT de treino e persistencia para `h=1` e
  `h=7`. Para `h=7`, `target_timestamp_utc - timestamp_utc = 6` dias em todas
  as linhas, coerente com a convencao atual de decoder step
  (`target_timestamp_utc = timestamp_utc + (horizon - 1)`), mas essa semantica
  ainda deve ser explicitada no pre-registro se o texto usar linguagem "h+7".
- Q6: o smoke produziu 402 linhas OOS, abaixo do limiar `n_rows >= 1000` para
  gate rigido. Resultado diagnostico por split/horizonte:
  `val,h=1`: 99 linhas, `% p10==p90=0%`, `% p10==p50==p90=0%`;
  `val,h=7`: 99 linhas, `% p10==p90=0%`, `% p10==p50==p90=0%`;
  `test,h=1`: 102 linhas, `% p10==p90=0%`, `% p10==p50==p90=0%`;
  `test,h=7`: 102 linhas, `% p10==p90=0%`, `% p10==p50==p90=0%`.
- Q6: os quantis brutos do smoke apresentaram cruzamento em 402/402 linhas
  (`p10 > p50` ou `p50 > p90`), mas as colunas pos-guardrail
  (`quantile_p10_post_guardrail`, `quantile_p50_post_guardrail`,
  `quantile_p90_post_guardrail`) tiveram 0 violacoes de ordem e 0 linhas com
  `quantile_p10_post_guardrail == quantile_p90_post_guardrail`. Isso confirma
  que o guardrail funciona no caminho de persistencia, mas tambem reforca que
  metricas probabilisticas e graficos devem usar explicitamente os quantis
  pos-guardrail ou reprovar o run se raw crossing for tratado como bloqueante.
- Q5: smoke TFT-only com refresh/quality gate executado em 2026-05-01 (cobriu
  treino, persistencia OOS, refresh sob scope e quality gate; nao cobriu
  baselines estatisticos nem inferencia rolling). O refresh com
  `--fail-on-quality --skip-prediction-plots --block-a-scope-sweep-prefixes
  phase_a_m7_smoke_20260501_tft_only --block-a-splits val,test
  --block-a-horizons 1,7 --block-a-require-post-guardrail` rodou em ~6,5
  minutos (18:20:22 -> 18:27:04) e produziu 24 tabelas gold. Quality gate
  retornou `Exit code 1` com 2 falhas informativas, ambas com diagnostico util:
  - `oos_quantile_block_a_acceptance` falhou com `crossing_bruto_rate=1.00 >
    max=0.001` e `negative_interval_width_count=326 > max=0`, mas
    `crossing_post_guardrail_rate=0.00`. **Falha compativel com hipotese de
    modelo nao-convergido em smoke de 1 epoch**, mas ainda nao provada como
    explicacao unica; requer revalidacao com `max_epochs >= 5` para
    distinguir entre falta de convergencia e fenomeno estrutural (regressao
    quantilica que nao aprende monotonia mesmo com convergencia). O gate
    atual aplica raw como contrato primario. Empiricamente urgente: decisao
    raw vs post-guardrail como contrato primario das metricas precisa entrar
    no pre-registro como item P0.
  - `gold_metrics_by_config_n_oos_contract` falhou com `mismatch_with_run_level=13800`.
    O smoke adicionou apenas 402 linhas, entao a falha e historica/global
    e nao causada pelo smoke. **Cabe a M5 investigar a causa raiz** (runs
    pre-ScopeSpec entrando na agregacao, recomputacao parcial, mudanca de
    regra entre versoes, ou similar). Esse achado e input direto para
    questao adicional em M5.
- Q5: o smoke confirma que `main_refresh_analytics_store` aceita os flags de
  scope (`--block-a-scope-sweep-prefixes`, `--block-a-splits`,
  `--block-a-horizons`, `--block-a-require-post-guardrail`) e produz output
  scoped no detail dos checks (`scope_mode=cohort_decision`,
  `scope_parent_sweep_prefixes=['phase_a_m7_smoke_20260501_tft_only']`). O
  scope foi aplicado ao Block A; o check `gold_metrics_by_config_n_oos_contract`
  parece ser global, o que tambem precisa ser verificado em M5.
- Q5: tempo de refresh sob scope (~6,5 min para o estado atual com 9.874
  arquivos Parquet) nao bloqueia, mas e baseline a considerar para iteracoes
  da Fase B (ciclo smoke + refresh + inspecao ~10 min).
- Q5 continua YELLOW (nao GREEN) porque (a) Q2 baselines persistidos ainda
  nao foi resolvido e o smoke completo confirmatorio depende disso; (b)
  decisao raw vs post-guardrail para o gate continua aberta; (c) a falha do
  `gold_metrics_by_config_n_oos_contract` precisa ser tratada em M5 antes
  da Fase B.
- Q6: os exemplos do smoke confirmam que o crossing raw nao e ruido numerico:
  magnitudes da ordem de `1e-3` a `5e-3` em retorno diario, com `y_true`
  tipico da ordem de `1e-2`. Exemplos: `test/h=1` com
  `p10=0.005561, p50=-0.003333, p90=0.001804` (p10 > p90, p10 > p50);
  `val/h=1` com `p10=-0.000948, p50=-0.005937, p90=0.003714` (p10 > p50).
  E fenomeno real do modelo nao-convergido (1 epoch), nao bug numerico ou
  do parser.
- Validacao executada nesta passagem: testes unitarios direcionados de
  multi-horizonte, guardrail, CLI de horizontes e sweep builder passaram
  (`6 passed`).
- Validacao executada nesta passagem adicional: smoke TFT-only com dados reais
  AAPL, `h=1,7`, all-features, `max_epochs=1`, persistindo no silver sob
  `parent_sweep_id` descartavel + refresh scoped completo com 24 tabelas
  gold geradas e quality gate executado.

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
- Antes da liberacao da Fase B, repetir o smoke multi-horizonte com volume
  suficiente (`n_rows >= 1000`) ou justificar formalmente por que o volume menor
  e apenas diagnostico. Reportar `n_rows`, `% p10==p90`,
  `% p10==p50==p90`, crossing bruto, crossing pos-guardrail e exemplos por
  horizonte.
- Repetir o smoke multi-horizonte com `max_epochs >= 5` para verificar se o
  crossing raw (`100%` em 1 epoch) diminui com convergencia ou se persiste
  como fenomeno estrutural. Se persistir alto com convergencia, e sinal de
  problema de regularizacao, learning rate ou regressao quantilica que precisa
  ser tratado antes da Fase B.
- **Item P0 do pre-registro (decisao em aberto, nao antecipar agora):**
  contrato quantilico primario das metricas e graficos. O smoke evidenciou
  que raw crossing pode ser 100% mesmo com post-guardrail 0%, mas ainda nao
  ha evidencia de qual e a causa estrutural (1 epoch vs problema de
  regressao quantilica). Antes de decidir, executar teste adicional com
  `max_epochs >= 5` e verificar se raw crossing diminui. So depois desse
  teste, o pre-registro deve fixar entre as opcoes: (a) raw como gate
  rigido (exige convergencia forte), (b) post-guardrail como contrato
  primario com raw como diagnostico (reviewer pode questionar calibracao
  artificial; mitigado se pre-registrado), (c) hibrido com thresholds
  distintos para raw e post-guardrail.
- **Cross-link com M5:** falha de `gold_metrics_by_config_n_oos_contract` com
  13.800 mismatches no estado historico precisa ser tratada em M5. O smoke
  evidenciou que o check e global (nao scoped) e expoe inconsistencia
  pre-existente entre `gold_prediction_metrics_by_config.n_oos` e a soma de
  `n_samples` do nivel run.
- Deixar `h=30` fora do smoke inicial por padrao; incluir apenas se o
  pre-registro decidir que o horizonte suplementar possui `N_effective`
  defensavel.
- Tratar inferencia rolling multi-horizonte como dependencia de M4 e refresh do
  Analytics Store sob escopo Fase B como dependencia de M5.

**Nota operacional:** refresh sob scope custa ~6,5 min no estado atual do
Analytics Store (9.874 arquivos Parquet). Iteracoes da Fase B (smoke + refresh
+ inspecao) tem ciclo aproximado de 10 min. Nao bloqueia, mas e baseline a
considerar para planejamento.

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
| `run_tft_optuna_search_use_case.py` | Desabilitar ou remover `mean_test_rmse` como `objective_metric` para evitar uso acidental de test no HPO. | promovido para M1-Q4 (P0) |
| `feature_registry.py` | Todas as features all-features tem `anti_leakage_tag` e `warmup_count` corretos? Alguma feature nova sem tag? | pendente |
| `validate_analytics_quality_use_case.py` | O Block A (degeneracao) esta configurado como bloqueante (nao apenas warning) para MPIW=0 > 5%? | pendente |
| `sklearn_indicator_normalizer.py` | Confirmar que nunca e instanciado com dados alem do treino. | pendente |

---

## Registro de achados adicionais

_(preencher durante a auditoria com qualquer achado fora das questoes pre-definidas)_
