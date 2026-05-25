# Phase B — PR-A4: Tier Classification Pipeline (E5 pre-processador)

Você é uma sessão Claude entrando em modo **engenharia operacional** para
implementar o pipeline de pré-processamento E5 da Phase B confirmatória.

Sua missão é **construir um pós-processador analítico** que lê silver/gold
existentes da Phase B, executa o protocolo estatístico declarado em F.2 §7
(versão operacionalizada pela emenda E1.8 desta PR), e escreve **4 parquets
sidecar** auditáveis em
`data/analytics/reports/phase_b/cohort=<parent_sweep_id>/`, prontos para a
Sessão-B consumir em E5.

**Você NÃO** redige relatório, **NÃO** escolhe tier, **NÃO** decide veredito
científico. Sua saída é mecânica e determinística; toda interpretação fica
para Sessão-B. **Você NÃO** modifica `RefreshAnalyticsStoreUseCase`, gold
builders existentes, nem `analytics_store_schema.py` (escopo deliberadamente
reduzido per veredito da auditoria de plano).

Modo: **AUTÔNOMO** com pause obrigatório:
1. Após implementação completa + testes + emenda E1.8 escrita, **antes**
   de rodar o pipeline contra a cohort real Phase B
   (`phase_b_confirmatorio_20260524`).

## Contexto verificado (confirme via comandos antes de iniciar)

Estado em main esperado:
- PR-A3 mergeada (commit `dae01c6`). Cohort silver Phase B unificada:
  `data/analytics/silver/dim_run/asset=AAPL/sweep_id=phase_b_confirmatorio_20260524/`
  com 60 dim_run rows (15 TFT + 45 baseline), `dim_run.fold` populado nos
  baselines (`wf_1/wf_2/wf_3`), `dim_run.seed` populado nos baselines.
- Refresh global mais recente: `failed_checks=[]`, `passed=True`, 27/27 checks.
- 25 gold tables materializadas em `data/analytics/gold/`.
- pytest 609 passed; ruff clean; archive
  `data/analytics_archive_pre_phase_b/` intacto (851M).
- Dataset sha256
  `aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298`
  (invariante).

Decisões científicas fechadas (não redecidir):
- **Unidade estatística DM** = `target_timestamp_utc` único (Estratégia A
  refinada). NÃO concatenar fold-seeds como observações.
- **Resolução de duplicatas fold** = "operationally latest fold" (manter
  previsão do fold cujo `train_end` é mais recente E < target_ts - horizon).
- **Agregação de seeds** = média aritmética dos loss_diffs por timestamp
  (após dedup de fold).
- **Loss function** = `pinball_loss_post_guardrail` (F.2 §6 contrato primário).
- **DM test** = Diebold-Mariano (1995) + correção Harvey-Leybourne-Newbold
  (1997) + HAC Newey-West com lag = `max(horizon - 1, 1)`.
- **Teste unilateral** = H1: `E[d_t] < 0` (TFT melhor). pvalue_one_sided_less
  é o pvalue primário; bilateral fica como coluna suplementar.
- **Holm scope** = exatamente 6 testes (2 horizontes × 3 baselines). NÃO
  reutiliza `pvalue_adj_holm` de `gold_dm_pairwise_results` (que tem scope
  ampliado por design legacy do builder).
- **Sensibilidade D (Holm sobre 18)** = sidecar separado marcado
  `analysis_role="sensitivity_conservative"`. Não é primário.

Justificativa metodológica completa: ver análise estatística referenciada
no Required reading + texto da Emenda E1.8 nesta especificação.

## Required reading (na ordem; NÃO PULE)

1. **Emenda E1.8 draft** (seção "Emenda E1.8 (texto-base)" desta especificação)
   — protocolo estatístico operacionalizado que esta PR formaliza.
2. [`docs/06_pre_registration/preregistration_phase_b.md`](../06_pre_registration/preregistration_phase_b.md)
   §3 (H1/H2a/H2b), §6 (post-guardrail), §7 (família Holm 6), §9 (Tier 1/2
   gates), §14 (emendas anteriores E1, E1.6, E1.7).
3. [`docs/04_evaluation/STATISTICAL_TESTS.md`](../04_evaluation/STATISTICAL_TESTS.md)
   — DM, HAC, Holm canonical references.
4. [`docs/04_evaluation/CALIBRATION_AND_RISK.md`](../04_evaluation/CALIBRATION_AND_RISK.md)
   §"Calibration Acceptance Thresholds" — bandas Tier 1/Tier 2.
5. [`docs/04_evaluation/METRICS_DEFINITIONS.md`](../04_evaluation/METRICS_DEFINITIONS.md)
   §"Variante quantilica" — categoria A/B/C, leitura `*_post_guardrail`.
6. [`src/use_cases/run_baselines_use_case.py`](../../src/use_cases/run_baselines_use_case.py)
   — padrão clean arch a espelhar (use case + persister + interface).
7. [`src/use_cases/run_baselines_test_pipeline_use_case.py`](../../src/use_cases/run_baselines_test_pipeline_use_case.py)
   — padrão de orchestrator-as-use-case.
8. [`src/domain/services/gold_builders/pairwise.py`](../../src/domain/services/gold_builders/pairwise.py)
   — referência (NÃO modificar): `_apply_holm_adjustment_for_dm` aplica Holm
   sobre família ampla por design legacy; nosso pipeline NÃO reutiliza.
9. [`src/adapters/parquet_analytics_run_repository.py`](../../src/adapters/parquet_analytics_run_repository.py)
   — padrão de adapter parquet (use como modelo para sidecar writer).
10. [`config/sweeps/explicit/phase_b_confirmatorio_20260524.json`](../../config/sweeps/explicit/phase_b_confirmatorio_20260524.json)
    — folds canônicos (wf_1/wf_2/wf_3) + periods.
11. Notas de revisão E0-E4 em
    [`docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md`](../05_checklists/PHASE_B_EXECUTION_CHECKLIST.md)
    — contexto do que foi executado pela Sessão-A.

## Comandos proibidos (sempre)

- `git rebase` (qualquer forma)
- `git commit --amend`
- `git push --force` / `git push -f`
- Escrita em `data/analytics_archive_pre_phase_b/` (read-only)
- `--no-verify` em commit hooks
- **Re-executar treino/baseline/refresh** — outputs Sessão-A são canônicos.
- Modificar tabelas em `data/analytics/gold/` (apenas leitura).
- Modificar `data/analytics/silver/` (apenas leitura).

## Protected files (NÃO modificar)

- `src/use_cases/refresh_analytics_store_use_case.py` (orquestrador canônico
  do gold refresh; nosso pipeline é separado).
- `src/domain/services/gold_builders/*.py` (incluindo `pairwise.py` —
  `_apply_holm_adjustment_for_dm` permanece com scope ampliado legacy).
- `src/infrastructure/schemas/analytics_store_schema.py` (cobre silver/dim
  apenas; adicionar gold schemas aqui requer redesenho de política — fora
  de escopo desta PR).
- `data/analytics/silver/` e `data/analytics/gold/` (apenas leitura).
- `data/analytics_archive_pre_phase_b/` (read-only).
- `config/sweeps/explicit/phase_b_confirmatorio_20260524.json` (selada).
- `docs/05_checklists/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (imutável).
- ADRs em `docs/01_architecture/decisions/`.

**Permitido criar:**
- `src/domain/services/phase_b_tier_policy.py` (constantes versionadas).
- `src/domain/services/fold_dedup_resolver.py` (regra operationally-latest).
- `src/domain/services/marginal_coverage_calculator.py`.
- `src/domain/services/dm_tft_vs_baseline.py`.
- `src/domain/services/holm_family_6.py`.
- `src/domain/services/tier_classifier_per_hypothesis.py`.
- `src/use_cases/compute_phase_b_tier_metrics_use_case.py`.
- `src/adapters/parquet_phase_b_tier_sidecar_writer.py`.
- `src/interfaces/phase_b_tier_sidecar_writer.py`.
- `src/main_compute_phase_b_tier_metrics.py` (CLI).
- `tests/unit/domain/services/test_*.py` (cobertura nova).
- `tests/integration/test_compute_phase_b_tier_metrics_use_case.py`.
- `docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md`.
- Sidecar output dir: `data/analytics/reports/phase_b/cohort=<id>/` (apenas
  no momento da execução real, não durante implementação dos tests).

**Permitido modificar:**
- `docs/06_pre_registration/preregistration_phase_b.md` (§14 anexar Emenda
  E1.8 + §7 nota cross-link à E1.8).
- `docs/04_evaluation/STATISTICAL_TESTS.md` (adicionar seção "Unidade
  estatística DM em walk-forward com folds sobrepostos").
- `docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md` (Stage E5 reescrito
  para usar o novo CLI — atualização já consta nesta PR mas pode precisar
  ajuste fino após implementação).
- `docs/ai/PHASE_B_SESSION_B_PROMPT.md` (instruções de uso do CLI — já
  ajustado nesta PR mas pode precisar refinamento).

## Skills a invocar (proativo)

- [`module-implementation-clean-arch`](../../.claude/skills/module-implementation-clean-arch/SKILL.md)
  — padrão arquitetural por camada (domain puro / use case / adapter /
  interface segregada / CLI thin).
- [`orchestrator-design`](../../.claude/skills/orchestrator-design/SKILL.md)
  — use case como thin orchestrator sobre domain services.
- [`project-scope-principles`](../../.claude/skills/project-scope-principles/SKILL.md)
  — escopo reduzido; resistir a feature creep arquitetural.
- [`staged-implementation-protocol`](../../.claude/skills/staged-implementation-protocol/SKILL.md)
  — commits sequenciais com aceite explícito por etapa.

## Decisões metodológicas fixadas (do not redecide)

Estas decisões foram tomadas com base em análise estatística formal
(referenciada em Required reading). NÃO redecida durante implementação.

| Decisão | Valor | Origem |
|---|---|---|
| Unidade estatística DM | 1 `target_timestamp_utc` único por par (TFT, baseline, horizon) | Estratégia A refinada |
| Resolução de duplicatas de fold | "operationally latest fold" (`train_end` mais recente E < `target_ts - horizon`) | Análise estatística |
| Agregação de seeds | média aritmética dos `d_t` (loss_diff) por timestamp | Análise estatística |
| Loss function primária | `pinball_loss_post_guardrail` | F.2 §6 |
| HAC kernel | Newey-West, lag = `max(horizon - 1, 1)` | Diebold-Mariano (1995); análise para h=7 |
| Small-sample correction | Harvey-Leybourne-Newbold (1997), aplicado sempre | F.2 §7 + HLN canonical |
| Teste primário | unilateral H1: `E[d_t] < 0` | Análise (TFT melhor) |
| Holm family scope (primário) | 6 testes = 2 horizontes × 3 baselines | F.2 §7 |
| Holm family scope (sensibilidade) | 18 testes = 3 folds × família 6 (apêndice, secundário) | Sensibilidade D |
| Coverage marginal cálculo | `mean(y_true <= y_hat_quantile_X_post_guardrail)` | F.2 §9 + categoria A |
| Delta pinball relativo | `(mean_pinball_baseline_post - mean_pinball_TFT_post) / mean_pinball_baseline_post` | F.2 §9 |
| Tier bands H1 (calibração) | `coverage_q10 ∈ [0.07, 0.13]`, `coverage_q50 ∈ [0.45, 0.55]`, `coverage_q90 ∈ [0.87, 0.93]`, `|coverage_error| ≤ 0.02`, `mpiw > 0` | F.2 §9 Tier 1 |
| Tier bands H2 (significância) | `pvalue_adj_holm < 0.05`, `delta_pinball_rel ≥ 0.03`, `mpiw > 0` | F.2 §9 Tier 1 |
| H2a baseline primário | `baseline_zero_return_v1` | F.2 §5 |
| H2b critério | TFT vence TODOS baselines | F.2 §3 |
| Identificador de baseline | `dim_run.model_version` (NÃO `feature_set_name`) | PR-A3 validation |

## Decisões técnicas com defaults (corrija se prefere diferente; senão assume)

| Decisão técnica | Default assumido |
|---|---|
| Implementação DM | `statsmodels.stats.diagnostic` + NeweyWest HAC; HLN manual |
| Implementação Holm | função pura nova em `holm_family_6.py` (NÃO usar `statsmodels.stats.multitest`) |
| Política de NaN em `y_true` ou `y_hat` | drop row antes de calcular `d_t` |
| Política de timestamps sem cobertura cross-baseline | excluir do suporte comum |
| Sidecar output path | `data/analytics/reports/phase_b/cohort=<parent_sweep_id>/` |
| Nome dos 4 sidecars | `phase_b_marginal_coverage.parquet`, `phase_b_dm_family_6.parquet`, `phase_b_delta_pinball.parquet`, `phase_b_tier_verdict.parquet` (+ sensibilidade: `phase_b_dm_family_18_sensitivity.parquet`) |

## Estrutura de arquivos por camada

### Domain (regras puras, sem I/O; testáveis sem fixtures de filesystem)

**`src/domain/services/phase_b_tier_policy.py`** (novo)
- Dataclass `PhaseBTierPolicy` com constantes F.2 §9 hardcoded:
  - `tier_1_coverage_q10_band: tuple[float, float] = (0.07, 0.13)`
  - `tier_1_coverage_q50_band: tuple[float, float] = (0.45, 0.55)`
  - `tier_1_coverage_q90_band: tuple[float, float] = (0.87, 0.93)`
  - `tier_1_picp_error_max: float = 0.02`
  - `tier_1_delta_pinball_rel_min: float = 0.03`
  - `tier_1_dm_pvalue_max: float = 0.05`
  - `tier_2_*_band` análogos (Tier 2 mais permissivo)
  - `tier_2_dm_pvalue_max: float = 0.10`
- Função `default_phase_b_policy() -> PhaseBTierPolicy` retorna instância
  imutável (frozen dataclass).
- **NÃO** carregar de YAML; constantes versionadas no código por segurança
  científica (impede ajuste pós-resultado sem rastro git).

**`src/domain/services/fold_dedup_resolver.py`** (novo)
- Função pura
  `select_operationally_latest_fold(candidates: list[RunInFoldCandidate], target_ts: pd.Timestamp, horizon: int) -> RunInFoldCandidate | None`.
- `RunInFoldCandidate` dataclass: `run_id`, `fold_name`, `train_end_utc`,
  `model_version`, `seed`.
- Regra: filtrar candidatos com `train_end_utc < target_ts - pd.Timedelta(days=horizon)`;
  retornar aquele com `train_end_utc` maior. None se nenhum elegível.
- Determinístico (sem random); idempotente.
- Truth table testável: ≥6 cenários (1 fold elegível, múltiplos, nenhum,
  empate de train_end → tiebreak por fold_name ordenado, etc).

**`src/domain/services/marginal_coverage_calculator.py`** (novo)
- Função pura
  `compute_marginal_coverage(fact_oos_filtered: pd.DataFrame, run_ids: set[str], quantile_levels: list[float]) -> pd.DataFrame`.
- Output: 1 row por `(run_id, split, horizon)` com colunas
  `coverage_q10`, `coverage_q50`, `coverage_q90`, `n_obs`, `coverage_error_picp`
  (= `picp_q10_q90 - 0.80`).
- Coverage_qX = `mean(y_true <= y_hat_quantile_X_post_guardrail)` para
  X ∈ {10, 50, 90}.
- Filtra split ∈ {val, test} se quiser; deixa caller decidir.

**`src/domain/services/dm_tft_vs_baseline.py`** (novo)
- Função pura
  `build_loss_diff_series(fact_oos: pd.DataFrame, dim_run: pd.DataFrame, tft_run_ids: set[str], baseline_run_ids: set[str], fold_periods: dict[str, FoldPeriod], horizon: int, split: str) -> pd.Series`.
  - Retorna Series indexada por `target_timestamp_utc` com valores `d_t`.
  - Aplica `fold_dedup_resolver` por timestamp.
  - Agrega seeds via média.
- Função pura
  `compute_dm_hln_hac(d_series: pd.Series, horizon: int, alpha: float = 0.05) -> DMResult`.
  - HAC Newey-West lag = `max(horizon - 1, 1)`.
  - HLN small-sample correction (Harvey, Leybourne, Newbold 1997):
    `DM_HLN = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n)`.
  - Retorna dataclass `DMResult(dm_stat, pvalue_two_sided, pvalue_one_sided_less, hac_lag_used, hln_applied, n_obs)`.
- Função pública
  `compute_dm_family(fact_oos, dim_run, tft_run_ids, baselines_by_model_version: dict[str, set[str]], fold_periods, horizons: list[int], split: str = "test") -> pd.DataFrame`.
  - Output: 1 row por `(horizon, model_version)` com colunas DMResult + meta.
  - Para o sweep Phase B Family 6: 2 horizontes × 3 model_versions = 6 rows.

**`src/domain/services/holm_family_6.py`** (novo)
- Função pura
  `apply_holm_one_sided(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame`.
- Input: Series de N pvalues unilaterais. Output: DataFrame com colunas
  `pvalue_one_sided`, `pvalue_adj_holm`, `significant_adj_0_05`.
- Implementação manual (não `statsmodels.multitest`) para controle total +
  testes determinísticos.
- **Não** chama nada em `pairwise.py:_apply_holm_adjustment_for_dm`.

**`src/domain/services/tier_classifier_per_hypothesis.py`** (novo)
- 3 funções puras (uma por hipótese):
  - `classify_h1(coverage_q10: float, coverage_q50: float, coverage_q90: float, picp_error: float, mpiw: float, policy: PhaseBTierPolicy) -> TierVerdict` (não usa DM nem delta).
  - `classify_h2a(dm_pvalue_adj_holm_zero_return: float, delta_pinball_rel_zero_return: float, mpiw: float, policy: PhaseBTierPolicy) -> TierVerdict` (usa apenas baseline primário).
  - `classify_h2b(dm_pvalues_adj_holm_by_baseline: dict[str, float], delta_pinball_rel_by_baseline: dict[str, float], mpiw: float, policy: PhaseBTierPolicy) -> TierVerdict` (TFT vence TODOS).
- Dataclass `TierVerdict(hypothesis: str, horizon: int, tier: Literal["tier_1", "tier_2", "refutado"], criteria_passed: dict[str, bool], numerical_inputs: dict[str, float], justification: str)`.
- Truth tables 100% determinísticas (sem float comparison ambígua; use
  `<=` consistente).

### Interfaces (contratos segregados)

**`src/interfaces/phase_b_tier_sidecar_writer.py`** (novo)
```python
from typing import Protocol

class PhaseBTierSidecarWriter(Protocol):
    def write_marginal_coverage(self, df: pd.DataFrame, cohort_id: str) -> Path: ...
    def write_dm_family_6(self, df: pd.DataFrame, cohort_id: str) -> Path: ...
    def write_dm_family_18_sensitivity(self, df: pd.DataFrame, cohort_id: str) -> Path: ...
    def write_delta_pinball(self, df: pd.DataFrame, cohort_id: str) -> Path: ...
    def write_tier_verdict(self, df: pd.DataFrame, cohort_id: str) -> Path: ...
```

### Adapter (I/O parquet)

**`src/adapters/parquet_phase_b_tier_sidecar_writer.py`** (novo)
- Implementa `PhaseBTierSidecarWriter`.
- Output path: `data/analytics/reports/phase_b/cohort=<parent_sweep_id>/<name>.parquet`.
- Cria dir se não existir; idempotent (sobrescreve).
- Validação de schema mínimo (colunas obrigatórias + tipos) antes de
  escrever.

### Use case (thin orchestrator)

**`src/use_cases/compute_phase_b_tier_metrics_use_case.py`** (novo)
- Constructor injection:
  - `silver_dir: Path` (lê `dim_run`, `fact_oos_predictions` via path globs).
  - `gold_dir: Path` (lê `gold_prediction_calibration` filtrado por cohort).
  - `sidecar_writer: PhaseBTierSidecarWriter`.
  - `policy: PhaseBTierPolicy` (default `default_phase_b_policy()`).
- Método único `execute(asset: str, parent_sweep_id: str) -> PhaseBTierMetricsResult`:
  1. Carrega silver: `dim_run` + `fact_oos_predictions` filtrado por
     `parent_sweep_id`.
  2. Identifica TFT runs (`feature_set_name != 'baseline'`) e baseline runs
     agrupados por `model_version`.
  3. Resolve fold periods do `dim_run.fold` (baselines) + lookup
     split_fingerprint → fold (TFT, via `fact_run_snapshot` se necessário).
  4. Calcula coverage marginal via `marginal_coverage_calculator` (split=test).
  5. Calcula DM family 6 via `dm_tft_vs_baseline.compute_dm_family`.
  6. Calcula DM family 18 (sensibilidade): mesma função mas grupos por
     `(horizon, baseline, fold)` em vez de agregar folds; aplica Holm sobre 18.
  7. Aplica `holm_family_6` sobre os 6 pvalues unilaterais.
  8. Calcula delta_pinball_rel agregando `gold_prediction_calibration.mean_pinball_post_guardrail`
     via mediana sobre seeds+folds por (horizon, model_version) — TFT vs cada baseline.
  9. Classifica H1/H2a/H2b por horizonte via `tier_classifier_per_hypothesis`.
  10. Persiste 5 sidecars via writer.
  11. Retorna `PhaseBTierMetricsResult(cohort_id, sidecar_paths: dict, verdicts: list[TierVerdict])`.
- **Não loga interpretação** — apenas counts/paths.

### CLI

**`src/main_compute_phase_b_tier_metrics.py`** (novo)
- Argparse mínimo:
  - `--asset` (required)
  - `--parent-sweep-id` (required)
  - `--silver-dir` (default `data/analytics/silver`)
  - `--gold-dir` (default `data/analytics/gold`)
  - `--output-dir` (default `data/analytics/reports/phase_b`)
- Resolve dependencies, instancia adapter + use case, chama `execute()`.
- Loga resumo (counts dos sidecars, paths, count de verdicts por tier sem
  emitir prosa). Exit 0 mesmo com tier=refutado (não é erro operacional).

### Tests

**Unit tests (domain puros)** em `tests/unit/domain/services/`:
- `test_phase_b_tier_policy.py`: bandas Tier 1/2 imutáveis (frozen).
- `test_fold_dedup_resolver.py`: ≥6 cenários (1 elegível, múltiplos,
  nenhum < target-h, empate train_end, tiebreak determinístico, NaN).
- `test_marginal_coverage_calculator.py`: fixture sintética 100 obs,
  expected coverage_q10 ≈ 0.10 com seed fixa.
- `test_dm_tft_vs_baseline.py`: fixture sintética 2 folds × 2 seeds,
  validar dedup correto + agregação seed mean + DM stat hand-calculated
  contra `statsmodels` referência.
- `test_holm_family_6.py`: caso clássico Holm (Holm 1979 exemplo) —
  6 pvalues, verificar ordem + ajuste correto.
- `test_tier_classifier_per_hypothesis.py`: truth tables H1/H2a/H2b —
  cenários: tier_1 borderline, tier_2 borderline, refutado, NaN handling.

**Integration test** em `tests/integration/`:
- `test_compute_phase_b_tier_metrics_use_case.py`: monta silver+gold
  sintético em `tmp_path` (1 candidato TFT 5 seeds × 3 folds, 3 baselines
  análogos), executa use case end-to-end via adapter parquet real (writer
  para `tmp_path/reports/`). Verifica:
  - 5 sidecars criados com schemas esperados.
  - `phase_b_dm_family_6.parquet` tem 6 rows (2h × 3 baselines).
  - `phase_b_tier_verdict.parquet` tem 6 rows (3 hyps × 2h).
  - `pvalue_adj_holm` ≤ 1.0 e monotonic per Holm.
- Sem rodar contra Phase B real (deixa para pause point pós-implementação).

### Documentação

**`docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md`** (novo runbook curto)
- Padrão dos outros runbooks: header YAML + comando canônico + outputs
  esperados + cross-links.

**`docs/04_evaluation/STATISTICAL_TESTS.md`** (modificar; adicionar seção)
- Nova seção "Unidade estatística DM em walk-forward com folds sobrepostos".
- Justifica unidade timestamp único, regra dedup operationally-latest,
  HAC lag, HLN, teste unilateral, Holm-6.
- Cross-link a Emenda E1.8 no pré-registro.

**`docs/06_pre_registration/preregistration_phase_b.md`** (modificar)
- §14 anexar texto da Emenda E1.8 (ver seção seguinte desta especificação).
- §7 adicionar nota curta: "Operacionalização do protocolo declarada na
  Emenda E1.8 (§14), por mediação do pós-processador
  `main_compute_phase_b_tier_metrics`."

## Emenda E1.8 (texto-base)

A ser anexada em `docs/06_pre_registration/preregistration_phase_b.md` §14
(no topo da lista de emendas, antes da E1.7). Texto-base — refinar se
necessário durante implementação, mas preservar substância:

```markdown
### 2026-MM-DD — Emenda E1.8: operacionalizacao do protocolo estatistico DM/Holm (unidade timestamp + dedup + seed mean + HAC + HLN + one-sided + Holm-6)

- Decisao alterada: nenhuma cientifica do pre-registro original; este texto
  OPERACIONALIZA o protocolo estatistico declarado em §7 ("familia unica
  Holm-Bonferroni sobre 6 testes = 2 horizontes x 3 baselines") que estava
  declarado mas nao especificado em nivel de implementacao.
- Hiperparametros, candidato top-1 trial 19 do E0, 5 seeds, 3 folds, 3
  baselines pre-declarados, bandas Tier 1/Tier 2: TODOS INALTERADOS.
- Hash da config JSON inalterado:
  sha256 = fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4

#### Especificacao operacional do teste Diebold-Mariano

A unidade estatistica primaria do teste Diebold-Mariano sera o
`target_timestamp_utc` no periodo OOS. Para cada par modelo-baseline
{candidato_TFT, baseline_i} e cada horizonte h em {1, 7}, sera construida
uma unica serie temporal de diferenciais de perda

  d_t = pinball_loss_post_guardrail_TFT(t, h) - pinball_loss_post_guardrail_baseline_i(t, h)

usando apenas timestamps presentes em ambos os metodos (suporte comum OOS).

##### Resolucao de duplicatas inter-fold (regra operationally-latest)

Walk-forward folds tem janelas OOS sobrepostas: wf_1 test cobre 2022-2023,
wf_2 test cobre 2023-2024, wf_3 test cobre 2024-2025 — portanto 2023 eh
coberto por wf_1+wf_2 e 2024 por wf_2+wf_3. Concatenar todas as previsoes
cria pseudo-observacoes duplicadas do mesmo evento de mercado, o que infla
artificialmente o tamanho amostral e subestima incerteza.

Regra de dedup, declarada antes de inspecionar resultados:

  Para cada target_timestamp_utc, manter a previsao do fold/window
  operacionalmente mais proximo do uso real: o fold cujo train_end e
  mais recente antes do forecast_origin = target_timestamp_utc - h dias.
  Tiebreak (caso impossivel mas defensivo): ordem alfabetica de fold.name.

Isso simula o que seria feito em producao: no tempo (t - h), usa-se o modelo
treinado com a janela mais atual disponivel sem usar informacao futura.

##### Agregacao de seeds

Quando multiplos seeds produzirem previsoes para o mesmo timestamp sob a
mesma configuracao temporal (mesmo fold escolhido pela regra acima), a
aleatoriedade de seed sera integrada por **media aritmetica dos
diferenciais de perda d_t** no timestamp, antes de calcular DM. Outras
agregacoes (mediana, IQR-trimmed mean) ficam como sensibilidade no
apendice se reportadas.

##### Teste estatistico

O teste primario sera Diebold-Mariano com:
- erro-padrao robusto HAC Newey-West com lag minimo = max(h - 1, 1)
  (previsoes multi-step induzem autocorrelacao residual em h - 1 lags);
- correcao Harvey-Leybourne-Newbold (1997) para amostras finitas:
  DM_HLN = DM * sqrt((n + 1 - 2h + h(h-1)/n) / n);
- hipotese alternativa **unilateral**: H1: E[d_t] < 0 (TFT tem menor
  pinball loss esperado);
- nivel de significancia alpha = 0.05.

A coluna pvalue primaria sera `pvalue_one_sided_less`. A coluna
`pvalue_two_sided` sera reportada como suplementar (informativa, nao
alimenta decisao tier).

##### Correcao para multiplos testes

A correcao Holm-Bonferroni (Holm 1979) sera aplicada sobre a **familia
unica primaria de 6 testes**, definida por:

  {(h, baseline) : h in {1, 7}, baseline in {zero_return,
                   historical_mean_rolling, historical_quantiles_rolling}}

Resultado significativo exige `pvalue_adj_holm < 0.05` (sobre o pvalue
unilateral). Esta correcao sera computada pelo modulo
`src/domain/services/holm_family_6.py` (PR-A4), nao pela coluna
`pvalue_adj_holm` de `gold_dm_pairwise_results` (que aplica Holm sobre
escopo ampliado por design legacy do gold builder e nao corresponde a
familia 6 declarada aqui).

##### Sensibilidade

Como analise de sensibilidade conservadora (apendice do relatorio E6),
sera tambem reportada a familia expandida de 18 testes (3 folds x 6 testes
primarios), com Holm sobre 18. Nao alimenta decisao tier; serve apenas
como robustness check.

#### Tabelas sidecar produzidas (cohort `phase_b_confirmatorio_20260524`)

O CLI `main_compute_phase_b_tier_metrics` produz 5 parquets em
`data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/`:

- `phase_b_marginal_coverage.parquet`: coverage_q10/q50/q90 por (run_id,
  split, horizon).
- `phase_b_dm_family_6.parquet`: 6 rows (2h x 3 baselines) com dm_stat,
  pvalue_one_sided, pvalue_two_sided, pvalue_adj_holm, n_obs_effective,
  hac_lag_used, hln_applied, direction, dedup_rule, seed_aggregation.
- `phase_b_dm_family_18_sensitivity.parquet`: 18 rows (3 folds x 6) com
  analysis_role="sensitivity_conservative".
- `phase_b_delta_pinball.parquet`: 6 rows com delta_mean_pinball_rel por
  (horizon, baseline).
- `phase_b_tier_verdict.parquet`: 6 rows (3 hipoteses x 2 horizontes) com
  tier in {tier_1, tier_2, refutado}, criteria_passed_dict, numerical_inputs,
  justification.

#### Justificativa

Tres opcoes alternativas foram consideradas e descartadas:

- Concatenacao naive fold-seed: infla tamanho amostral via pseudo-observacoes
  duplicadas; viola pressuposto DM de independencia/estacionariedade da
  serie d_t (Diebold-Mariano 1995).
- DM por fold-seed + meta-analise (~15 DMs agregados): seeds compartilham
  timestamps e baseline; meta-analise exigiria modelo dependencia explicito
  (fixed/random effects) sem precedente claro no pre-registro.
- Familia Holm de 18 testes como primario: diverge de F.2 §7 literal (6
  testes); mantido apenas como sensibilidade conservadora.

A estrategia A refinada (unidade timestamp + dedup operationally-latest +
seed mean) eh a mais alinhada com a definicao classica DM e mais defensavel
academicamente (Diebold 2015 — twenty years later; ver
STATISTICAL_TESTS.md).

#### Cross-link

- PR-A4 + commits.
- `src/use_cases/compute_phase_b_tier_metrics_use_case.py` orchestrator.
- `src/main_compute_phase_b_tier_metrics.py` CLI.
- `docs/04_evaluation/STATISTICAL_TESTS.md` documentacao canonica.
- `docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md` runbook.

Sem impacto nas hipoteses cientificas H1, H2a, H2b nem nos gates Tier 1/Tier 2.
Apenas formaliza COMO os 6 pvalues unilaterais sao computados a partir dos
dados, deixando a interpretacao para Sessao-B em E5.
```

## Sequência de commits (incrementais, com Aceite por commit)

### Commit 1: domain puros + tests unit
- Arquivos:
  - `src/domain/services/phase_b_tier_policy.py`
  - `src/domain/services/fold_dedup_resolver.py`
  - `src/domain/services/marginal_coverage_calculator.py`
  - `src/domain/services/dm_tft_vs_baseline.py`
  - `src/domain/services/holm_family_6.py`
  - `src/domain/services/tier_classifier_per_hypothesis.py`
  - `tests/unit/domain/services/test_phase_b_tier_policy.py`
  - `tests/unit/domain/services/test_fold_dedup_resolver.py`
  - `tests/unit/domain/services/test_marginal_coverage_calculator.py`
  - `tests/unit/domain/services/test_dm_tft_vs_baseline.py`
  - `tests/unit/domain/services/test_holm_family_6.py`
  - `tests/unit/domain/services/test_tier_classifier_per_hypothesis.py`
- Aceite: `pytest tests/unit/domain/services/test_phase_b*.py
  test_fold_dedup* test_marginal_coverage* test_dm_tft* test_holm_family_6*
  test_tier_classifier* -v` → todos PASS; ruff clean.

### Commit 2: interface + adapter sidecar writer
- Arquivos:
  - `src/interfaces/phase_b_tier_sidecar_writer.py`
  - `src/adapters/parquet_phase_b_tier_sidecar_writer.py`
- Aceite: ruff clean; writer não-trivialmente importável (smoke).

### Commit 3: use case orchestrator + integration test
- Arquivos:
  - `src/use_cases/compute_phase_b_tier_metrics_use_case.py`
  - `tests/integration/test_compute_phase_b_tier_metrics_use_case.py`
- Aceite: `pytest tests/integration/test_compute_phase_b_tier_metrics_use_case.py -v`
  → PASS; 5 sidecars criados em tmp_path; verdict count = 6.

### Commit 4: CLI entrypoint + runbook
- Arquivos:
  - `src/main_compute_phase_b_tier_metrics.py`
  - `docs/06_runbooks/RUN_PHASE_B_TIER_CLASSIFICATION.md`
- Aceite: `python -m src.main_compute_phase_b_tier_metrics --help` →
  ajuda corretamente; ruff clean.

### Commit 5: Emenda E1.8 + STATISTICAL_TESTS.md
- Arquivos:
  - `docs/06_pre_registration/preregistration_phase_b.md` (§14 + §7)
  - `docs/04_evaluation/STATISTICAL_TESTS.md` (nova seção)
- Aceite: emenda E1.8 anexada com data preenchida; §7 com nota cross-link;
  STATISTICAL_TESTS.md com nova seção sobre unidade DM em walk-forward.

### ⏸️ PAUSE OBRIGATÓRIO — antes de rodar contra Phase B real

Reporte a Marcelo:
> "PR-A4 pronta para review. 5 commits empilhados. Tests unit + integration
> passing. Aguardo (a) review dos diffs especialmente do `fold_dedup_resolver`,
> `dm_tft_vs_baseline`, e texto da emenda E1.8; (b) autorização para
> rodar o CLI contra cohort real `phase_b_confirmatorio_20260524` antes
> de abrir o PR contra main."

Após autorização:

### Commit 6 (pós-pause): smoke run contra cohort real + ajustes finais
- Rodar:
  ```bash
  .venv/bin/python -m src.main_compute_phase_b_tier_metrics \
    --asset AAPL \
    --parent-sweep-id phase_b_confirmatorio_20260524
  ```
- Validar que os 5 sidecars são criados em
  `data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/`.
- Validar shapes esperados (6/18/6/6 rows; coverage_marginal com 60 rows).
- Validar que `phase_b_dm_family_6.parquet` tem 6 rows com pvalue_one_sided ≤ 1.0
  e pvalue_adj_holm monotonic.
- NÃO interpretar veredito; apenas confirmar mecânica.
- Se algum sidecar tiver shape inesperado, debugar + ajuste mínimo + commit
  separado "fix(phase-b-tier): ajuste pós smoke contra cohort real".

### Push + PR-A4

```bash
git push -u origin <branch-name>
gh pr create --base main --head <branch-name> \
  --title "feat(phase-b-tier): pipeline E5 pos-processador (Emenda E1.8)" \
  --body "<corpo detalhado: motivacao + emenda E1.8 link + decisoes
  metodologicas + estrutura clean arch + tests + smoke real PASS>"
```

## Init commands

Execute na ordem antes de começar:

```bash
cd /home/marcelo/Code/financial-time-series-forecasting
git checkout main && git pull --ff-only
git log --oneline -5  # confirmar PR-A3 mergeada (commit dae01c6 ou similar)

# Branch nova
git checkout -b feat/phase-b-tier-pipeline-<YYYYMMDD>

# Estado inicial
.venv/bin/pytest tests/ -q \
  --ignore=tests/integration/test_quality_registry_bit_identical_archive.py \
  --ignore=tests/integration/test_gold_builders_byte_identical_archive.py 2>&1 | tail -3
.venv/bin/ruff check src/ tests/ 2>&1 | tail -3

# Confirmar cohort intacto
du -sh data/analytics_archive_pre_phase_b/   # 851M
ls data/analytics/silver/dim_run/asset=AAPL/sweep_id=phase_b_confirmatorio_20260524/dim_run.parquet
ls data/analytics/gold/*.parquet | wc -l     # ≥25

# Hash dataset invariante
sha256sum data/processed/dataset_tft/AAPL/dataset_tft_AAPL.parquet
# esperado: aee6b3ed7d931ff278353d647656effe4f338782292c801d35581f541c6c1298
```

## Sucesso definido

PR-A4 entrega:
1. **6 novos arquivos domain** com tests unit (≥6 cenários cada) — todos PASS.
2. **1 interface + 1 adapter** parquet sidecar writer.
3. **1 use case** thin orchestrator + integration test PASS.
4. **1 CLI entrypoint** funcional.
5. **Emenda E1.8** anexada em §14 do pré-registro com texto completo;
   §7 com nota cross-link.
6. **Nova seção** em `STATISTICAL_TESTS.md`.
7. **Runbook** `RUN_PHASE_B_TIER_CLASSIFICATION.md`.
8. **Smoke run** contra cohort Phase B real produz 5 sidecars corretos.
9. **PR aberta** com CI verde (lint + pytest pleno).
10. `data/analytics/silver/` e `data/analytics/gold/` inalterados; archive
    intacto 851M; dataset sha256 invariante.

Após merge PR-A4, Sessão-B retoma E5 lendo os sidecars + interpretando
verdicts (não recalcula nada).

## Constraints inegociáveis

- NUNCA modificar `RefreshAnalyticsStoreUseCase` ou gold builders existentes.
- NUNCA modificar `analytics_store_schema.py` (sidecars são FORA de gold/
  por design).
- NUNCA reutilizar `pvalue_adj_holm` de `gold_dm_pairwise_results` (scope
  errado — calcular do zero via `holm_family_6.py`).
- NUNCA re-executar treino/baseline/refresh — silver/gold são canônicos.
- NUNCA decidir tier sob inspeção dos dados (cherry-picking) — tier_classifier
  aplica truth tables mecânicas; veredito não pode depender de qual cenário
  dá Tier 1 vs Tier 2.
- NUNCA "ajustar" defaults metodológicos durante implementação para fazer
  testes passarem. Se um teste sintético força redecisão, reporte a
  Marcelo, NÃO mude o domain service.
- NUNCA tocar `data/analytics_archive_pre_phase_b/`.

## Notas de revisão template (preencher após PR-A4 mergeada)

Adicionar em `PHASE_B_EXECUTION_CHECKLIST.md` Stage E5 (não nas Notas E0-E4):

```markdown
#### PR-A4 (pré-E5): pipeline tier metrics

- 2026-MM-DD HH:MM UTC: PR-A4 mergeada (commit <sha>).
- Branch: feat/phase-b-tier-pipeline-<YYYYMMDD>.
- Smoke contra cohort real: 5 sidecars criados em
  data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/.
- Sidecars shapes confirmados: marginal_coverage=60 rows, dm_family_6=6 rows,
  dm_family_18_sensitivity=18 rows, delta_pinball=6 rows, tier_verdict=6 rows.
- Emenda E1.8 anexada em §14 do preregistration_phase_b.md (data: 2026-MM-DD).
- Sessao-B retoma E5 daqui (consome sidecars; nao recalcula).
```
