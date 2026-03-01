# Rodando Analise de Modelo (Sweeps)

Este guia documenta o uso do pipeline de analise de hiperparametros via:

- `python -m src.main_tft_param_sweep`

Ele executa treinos OFAT (one-factor-at-a-time), consolida resultados e gera relatorios/plots.

## 1) Pre-requisitos

1. Dataset TFT do ativo ja gerado:
- `data/processed/dataset_tft/{ASSET}/dataset_tft_{ASSET}.parquet`

2. Ambiente com dependencias de treino e plots:
- `pytorch_forecasting`, `lightning`, `matplotlib`, etc.

3. Arquivo de configuracao JSON para o sweep:
- por padrao: `config/model_analysis.default.json`
- recomendado: criar arquivos em `config/sweeps/*.json`

## 2) Estrutura minima do JSON

Exemplo:

```json
{
  "output_subdir": "0_0_sweep_all_params_robust_vs_baseline",
  "features": "BASELINE_FEATURES",
  "replica_seeds": [7, 11, 13, 42, 123],
  "walk_forward": {
    "enabled": true,
    "folds": [
      {
        "name": "wf_1",
        "train_start": "20100101",
        "train_end": "20161231",
        "val_start": "20170101",
        "val_end": "20181231",
        "test_start": "20190101",
        "test_end": "20201231"
      }
    ]
  },
  "training_config": {
    "max_encoder_length": 120,
    "max_prediction_length": 1,
    "batch_size": 64,
    "max_epochs": 20,
    "learning_rate": 0.0005,
    "hidden_size": 32,
    "attention_head_size": 2,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "seed": 42,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0
  },
  "split_config": {
    "train_start": "20100101",
    "train_end": "20221231",
    "val_start": "20230101",
    "val_end": "20241231",
    "test_start": "20250101",
    "test_end": "20251231"
  },
  "param_ranges": {
    "max_encoder_length": [120, 150, 180, 210]
  }
}
```

## 3) Execucao basica

```bash
python -m src.main_tft_param_sweep --asset AAPL --config-json config/sweeps/SEU_SWEEP.json
```

Comportamento:

1. Treina as variacoes OFAT por fold e seed.
2. Salva `sweep_runs.csv/json`.
3. Gera relatorios agregados (`config_ranking`, `param_impact`, `summary`).
4. Gera plots automaticamente (via `main_generate_sweep_plots`).

## 4) Regerar apenas artefatos (sem treinar)

Use quando os modelos/runs ja existem e voce quer recomputar relatorios/plots:

```bash
python -m src.main_generate_sweep_plots --sweep-dir data/models/AAPL/sweeps/SEU_SWEEP
```

## 5) Merge incremental de testes (`--merge-tests`)

Use para adicionar novos runs ao mesmo sweep sem perder historico:

```bash
python -m src.main_tft_param_sweep --asset AAPL --config-json config/sweeps/SEU_SWEEP.json --output-subdir SEU_SWEEP --merge-tests
```

### Regras de seguranca do merge

Para evitar inconsistencias, ao usar `--merge-tests`:

1. Apenas estes campos podem mudar:
- `param_ranges`
- `walk_forward.folds`
- `replica_seeds`

2. Restricoes:
- `param_ranges`: valores antigos nao podem ser removidos (somente manter/adicionar).
- `folds`: folds antigos nao podem ser removidos nem alterados (somente manter/adicionar novos folds).
- `replica_seeds`: seeds antigas nao podem ser removidas (somente manter/adicionar).

3. Qualquer outra mudanca de configuracao bloqueia a execucao com erro.

## 6) Principais artefatos gerados

No diretorio:
- `data/models/{ASSET}/sweeps/{output_subdir}/`

Arquivos principais:

- `analysis_config.json`: configuracao efetiva do sweep
- `sweep_runs.csv`: base principal de runs (coracao da analise de performance)
- `summary.json`: resumo executivo
- `config_ranking.csv`: ranking por configuracao
- `param_impact_detail.csv`: impacto por run
- `param_impact_summary.csv`: impacto agregado por parametro
- `drift_ks_psi_by_fold.csv`: drift por fold
- `drift_ks_psi_overall_summary.json`: resumo global de drift

Estrutura por fold:

- `folds/{fold_name}/sweep_runs.csv`
- `folds/{fold_name}/config_ranking.csv`
- `folds/{fold_name}/param_impact_*.csv`
- `folds/{fold_name}/plots/*.png`
- `folds/{fold_name}/models/{version}/...`

## 7) Leitura rapida dos resultados

1. Parametro com melhor impacto medio:
- veja `param_impact_summary.csv`
- menor `avg_delta_rmse_vs_baseline` e mais estavel

2. Melhor configuracao robusta:
- veja `config_ranking.csv`
- criterio principal: menor `robust_score` (`mean_val_rmse + std_val_rmse`)

3. Estabilidade temporal:
- compare resultados por fold (`folds/*/config_ranking.csv`)
- use `drift_ks_psi_by_fold.csv` para contexto de regime

## 8) Observacoes praticas

1. `sweep_runs.csv` e a base principal para ranking/impacto, mas:
- `all_models_ranked.csv` depende dos artefatos em `models/`
- drift depende de dataset + split (nao so de `sweep_runs.csv`)

2. Se um sweep falhar parcialmente:
- use `--continue-on-error` para seguir
- depois regenere artefatos com `main_generate_sweep_plots`

3. Para comparacao justa entre sweeps:
- compare apenas intersecao de folds/seeds/params iguais


## 9) HPO com Optuna (top-k para sweep)

Use quando quiser buscar hiperparametros com estrategia bayesiana e depois
rodar somente os melhores na pipeline de sweep tradicional.

Comando:

```bash
python -m src.main_tft_optuna_sweep --asset AAPL --config-json config/optuna/default_tft_optuna_sweep.json
```

Artefatos gerados em:
- `data/models/{ASSET}/optuna/{output_subdir}/...`

Arquivos principais:
- `optuna_summary.json`
- `optuna_best_trial.json`
- `optuna_top_k_configs.json`
- `optuna_trials.csv`
- `optuna_trials.json`
- `optuna_study.db`

Fluxo recomendado:
1. Executar Optuna para obter top-k configuracoes (`optuna_top_k_configs.json`).
2. Levar as configuracoes para um sweep dedicado e gerar relatorios/plots finais.

