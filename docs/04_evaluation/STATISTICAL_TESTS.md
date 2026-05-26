---
title: Statistical Tests
scope: Protocolo estatistico para comparacao OOS entre modelos. Define DM (Diebold-Mariano), MCS (Model Confidence Set), pairwise win-rate, ajuste Holm para multiplas comparacoes e regras obrigatorias de alinhamento temporal por `target_timestamp`.
update_when:
  - novo teste estatistico for adicionado (Giacomini-White, Clark-West etc.)
  - politica de multiple testing mudar (Holm, Bonferroni, FDR)
  - regras de alinhamento OOS forem revisadas
  - thresholds default de significancia mudarem
canonical_for: [statistical_tests, dm_test, mcs, win_rate, multiple_testing, oos_alignment]
---

# Statistical Tests

Objetivo: definir protocolo estatistico para comparacao entre modelos usando
somente dados OOS persistidos e alinhados temporalmente.

## Canonical Implementation
- `src/use_cases/refresh_analytics_store_use_case.py`
- `src/use_cases/validate_analytics_quality_use_case.py`

## Required Preconditions
- Comparacao somente em OOS (`split=test` ou split explicitamente definido)
- Alinhamento exato por:
  - `asset`
  - `split`
  - `horizon`
  - `target_timestamp`
- Intersecao temporal explicita antes de qualquer teste pareado
- Escopo declarado (`cohort_decision` para decisao final)

## Tests

### 1) Diebold-Mariano (DM)
- Hipotese nula: igualdade de acuracia preditiva esperada entre dois modelos.
- Base: series de perdas pareadas por `target_timestamp`.
- Saida minima:
  - estatistica DM
  - `pvalue`
  - `pvalue_adj_holm` (controle de comparacoes multiplas)

### 2) Model Confidence Set (MCS)
- Objetivo: conjunto de modelos nao rejeitados como melhores ao nivel `alpha`.
- Nivel default implementado: `alpha=0.05`.
- Saida minima:
  - `selected_in_mcs_alpha_0_05`
  - `aligned_timestamps`
  - `n_configs`

### 3) Pairwise Win-Rate
- Compara vitorias por timestamp entre pares de modelos.
- Reporta win-rate com e sem empates e tamanhos de amostra alinhada.

## Multiple Testing
- Ajuste obrigatorio para DM pairwise:
  - Holm (`pvalue_adj_holm`)
- Resultado significativo:
  - `pvalue_adj_holm < 0.05`

## Unidade Estatistica DM Em Walk-forward Com Folds Sobrepostos

Para a Phase B confirmatoria, a unidade estatistica primaria do DM
TFT-vs-baseline e um `target_timestamp_utc` unico por par
(`TFT`, `baseline`, `horizon`). Essa regra evita contar duas vezes o mesmo
evento de mercado quando folds walk-forward possuem janelas OOS sobrepostas.

Operacionalizacao declarada na Emenda E1.8 do pre-registro:
- manter, por timestamp, o fold operacionalmente mais recente cujo
  `train_end_utc < target_timestamp_utc - horizon`;
- em empate defensivo, ordenar por nome do fold;
- agregar seeds por media aritmetica dos diferenciais de perda `d_t`;
- usar `pinball_loss_post_guardrail` como perda primaria;
  para baselines pontuais, a Emenda E1.8 fixa a convencao degenerada
  `q10=q50=q90=y_pred`;
- aplicar HAC Newey-West com lag `max(horizon - 1, 1)`;
- aplicar correcao Harvey-Leybourne-Newbold;
- reportar como pvalue primario `pvalue_one_sided_less` para H1:
  `E[d_t] < 0`;
- aplicar Holm-Bonferroni sobre a familia primaria de 6 testes
  (`2 horizontes x 3 baselines`).

O pos-processador `src.main_compute_phase_b_tier_metrics` materializa esse
protocolo em sidecars fora de `data/analytics/gold/`, preservando gold como
artefato canonico do refresh e deixando a interpretacao cientifica para E5.

## Scope and Horizon Rules
- Nunca misturar horizontes no mesmo teste.
- Nunca comparar coortes com `split_signature` incompativel.
- Resultado global nao substitui resultado scoped para decisao de vencedor.

## Canonical Output Tables
- `gold_dm_pairwise_results.parquet`
- `gold_mcs_results.parquet`
- `gold_win_rate_pairwise_results.parquet`
- `gold_paired_oos_intersection_by_horizon.parquet`
- `gold_quality_statistics_report.parquet`

## Validation Gate
O quality gate deve confirmar que os testes sao executaveis com dados persistidos:
- check `dm_mcs_persisted_executable` valido
- intersecoes temporais presentes e nao vazias
