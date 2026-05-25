---
title: Run Phase B Tier Classification
scope: Execucao do pos-processador PR-A4 para gerar sidecars E5 da Phase B confirmatoria. Nao reexecuta treino, baselines nem refresh gold.
update_when:
  - flags de `src.main_compute_phase_b_tier_metrics` mudarem
  - nomes ou schemas dos sidecars Phase B mudarem
  - protocolo E1.8 de DM/Holm/tier mudar
canonical_for: [phase_b_tier_classification_cli, phase_b_e5_sidecars]
---

# Phase B Tier Classification

## Command

```bash
.venv/bin/python -m src.main_compute_phase_b_tier_metrics \
  --asset AAPL \
  --parent-sweep-id phase_b_confirmatorio_20260524
```

O comando le somente:
- `data/analytics/silver/`
- `data/analytics/gold/`

O comando escreve somente em:
- `data/analytics/reports/phase_b/cohort=<parent_sweep_id>/`

## Expected Outputs

Para `phase_b_confirmatorio_20260524`, os sidecars esperados sao:

| Arquivo | Shape esperado | Papel |
|---|---:|---|
| `phase_b_marginal_coverage.parquet` | 60 rows | Coverage marginal q10/q50/q90 dos runs TFT por split/horizonte |
| `phase_b_dm_family_6.parquet` | 6 rows | Familia primaria Holm-6: 2 horizontes x 3 baselines |
| `phase_b_dm_family_18_sensitivity.parquet` | 18 rows | Sensibilidade conservadora: 3 folds x familia 6 |
| `phase_b_delta_pinball.parquet` | 6 rows | Delta pinball relativo por horizonte/baseline |
| `phase_b_tier_verdict.parquet` | 6 rows | Veredito mecanico por hipotese/horizonte |

## Validation

```bash
.venv/bin/python <<'PY'
import pandas as pd
from pathlib import Path

base = Path('data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524')
expected = {
    'phase_b_marginal_coverage.parquet': 60,
    'phase_b_dm_family_6.parquet': 6,
    'phase_b_dm_family_18_sensitivity.parquet': 18,
    'phase_b_delta_pinball.parquet': 6,
    'phase_b_tier_verdict.parquet': 6,
}
for name, rows in expected.items():
    df = pd.read_parquet(base / name)
    assert len(df) == rows, (name, len(df), rows)
print('SMOKE PASS')
PY
```

## Scope Semantics

Escopo: `cohort_decision`.

Os sidecars sao por `parent_sweep_id` e existem para a Sessao-B interpretar E5. O CLI nao decide conclusao cientifica, nao altera silver/gold, e nao substitui `main_refresh_analytics_store`.

## Cross-links

- Pre-registro Phase B: `docs/06_pre_registration/preregistration_phase_b.md` §7, §9 e §14 Emenda E1.8.
- Protocolo estatistico: `docs/04_evaluation/STATISTICAL_TESTS.md`.
- Checklist operacional: `docs/05_checklists/PHASE_B_EXECUTION_CHECKLIST.md` Stage E5.
- Implementacao: `src/use_cases/compute_phase_b_tier_metrics_use_case.py`.
