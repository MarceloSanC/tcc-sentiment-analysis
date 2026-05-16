---
title: Run Refresh Analytics
scope: Refresh do analytics store via `src.main_refresh_analytics_store`, incluindo flag `--fail-on-quality` (gate de saude global) e flags de escopo por sweep/coorte. Comando canonico para regenerar tabelas gold.
update_when:
  - novo flag de escopo for adicionado ao main_refresh_analytics_store
  - politica de fail-on-quality mudar
  - novo modo de refresh (incremental, full) for introduzido
canonical_for: [run_refresh_analytics, fail_on_quality_command, scope_filter_flags]
---

# Refresh Analytics Store

## Command
Padrao:
`python -m src.main_refresh_analytics_store --fail-on-quality`

Escopado para Bloco A (exemplo `0_2_3`, val/test, h=1):
`python -m src.main_refresh_analytics_store --fail-on-quality --block-a-scope-sweep-prefixes 0_2_3_ --block-a-splits val,test --block-a-horizons 1 --block-a-max-crossing-bruto-rate 0.001 --block-a-max-negative-interval-width-count 0 --block-a-max-crossing-post-guardrail-rate 0.0`

Refresh scoped por coorte declarada (exemplo `round_1_`):
`python -m src.main_refresh_analytics_store --scope-mode cohort_decision --scope-sweep-prefixes round_1_`

## Expected
- gold tables updated
- quality checks pass (`failed_checks=[]`)
- check `oos_quantile_block_a_acceptance` presente no resultado de quality

## Quantile Block A (Implemented)

Contexto:
- O quality gate identificou inconsistencias probabilisticas OOS (crossing e largura negativa) no escopo de analise `0_2_3`.
- Para tornar a validacao reutilizavel no fluxo de testes estatisticos, o Bloco A foi implementado de forma modular no dominio e integrado ao quality gate.

Implementacao tecnica:
- Servico de dominio: `src/domain/services/quantile_contract_analyzer.py`
  - `QuantileContractAnalyzer.filter_scope(...)`: recorte por `parent_sweep_prefixes`, `splits`, `horizons`.
  - `QuantileContractAnalyzer.analyze(...)`: calcula `crossing_bruto_count/rate`, `negative_interval_width_count/rate` e `crossing_post_guardrail_rate` (quando colunas post-guardrail existirem).
  - `QuantileContractAnalyzer.evaluate_block_a(...)`: aplica thresholds de aceite (Bloco A) e retorna status + detalhe auditavel.
- Integracao no quality gate:
  - `src/use_cases/validate_analytics_quality_use_case.py`
  - Novo check: `oos_quantile_block_a_acceptance`

Como usar:
- Uso padrao (global):
  - `python -m src.main_refresh_analytics_store --fail-on-quality`
- Uso com escopo de analise (programatico, via `ValidateAnalyticsQualityUseCase`):
  - `block_a_parent_sweep_prefixes=["0_2_3_"]`
  - `block_a_splits=["val", "test"]`
  - `block_a_horizons=[1]`

Criterios atuais (provisorio):
- `crossing_bruto_rate <= 0.10%`
- `crossing_pos_guardrail_rate = 0` (quando houver colunas post-guardrail)
- `negative_interval_width_count = 0`

Validacao automatizada:
- `tests/unit/domain/services/test_quantile_contract_analyzer.py`
- `tests/unit/use_cases/test_validate_analytics_quality_use_case.py`



### Parametros CLI de Bloco A
- `--block-a-scope-sweep-prefixes`: prefixos de `parent_sweep_id` para recorte da avaliacao.
- `--block-a-splits`: splits considerados no Bloco A (ex.: `val,test`).
- `--block-a-horizons`: horizontes considerados (ex.: `1,7,30`).
- `--block-a-max-crossing-bruto-rate`: limiar de crossing bruto.
- `--block-a-max-negative-interval-width-count`: limite de largura negativa.
- `--block-a-max-crossing-post-guardrail-rate`: limiar de crossing apos guardrail.
- `--block-a-require-post-guardrail`: exige colunas pós-guardrail para aprovar o Bloco A.


### Parametros CLI de Refresh Scoped
- `--scope-mode`: modo de escopo do refresh (`global_health` ou `cohort_decision`).
- `--scope-sweep-prefixes`: prefixos de `parent_sweep_id` usados para recortar silver antes de gerar gold.
- `--scope-splits`: splits usados para recortar silver antes de gerar gold (ex.: `val,test`).
- `--scope-horizons`: horizontes usados para recortar silver antes de gerar gold (ex.: `1,7,30`).

Quando nenhuma flag `--scope-*` e passada, o refresh preserva o comportamento
global historico. Quando `--scope-mode cohort_decision` e usado, pelo menos um
filtro de coorte deve ser declarado.

## Scope Semantics (Governance)
- `--fail-on-quality` sem filtros e um gate global de saude do asset/dataset.
- Filtros `--scope-*` escopam a leitura silver do refresh e materializam gold
  reduzido para a coorte declarada.
- Filtros `--block-a-*` atualmente escopam o check `oos_quantile_block_a_acceptance`; outros checks permanecem globais.
- Para decisao estatistica por coorte (ex.: sweep `0_2_3`), use sempre o mesmo escopo em tabelas/plots/report e declare o escopo no resultado.
- Nao usar quality global isolado para concluir vencedor de coorte especifica.
