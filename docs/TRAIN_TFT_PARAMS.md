# Treino TFT - Parametros

Este documento centraliza os parametros customizaveis do `src.main_train_tft`.
Os defaults e limites sao definidos em `src/infrastructure/schemas/model_artifact_schema.py`.

## Minimo Para Rodar

| Parametro | Tipo | Default | Valores validos | Obrigatorio | Exemplo | Impacto no treino |
|---|---|---:|---|---|---|---|
| `--asset` | `str` | - | simbolo de ativo | Sim | `--asset AAPL` | Define qual dataset sera carregado |
| `--features` | `str` (CSV) | `DEFAULT_TFT_FEATURES` | grupos (`BASELINE_FEATURES`, `TECHNICAL_FEATURES`, `SENTIMENT_FEATURES`, `FUNDAMENTAL_FEATURES`) e/ou colunas existentes | Nao | `--features BASELINE_FEATURES,TECHNICAL_FEATURES` | Define o conjunto de entrada do modelo |
| `--train-start` | `str` | `20100101` | `yyyymmdd` | Nao | `--train-start 20100101` | Janela de treino |
| `--train-end` | `str` | `20221231` | `yyyymmdd` | Nao | `--train-end 20221231` | Janela de treino |
| `--val-start` | `str` | `20230101` | `yyyymmdd` | Nao | `--val-start 20230101` | Janela de validacao |
| `--val-end` | `str` | `20241231` | `yyyymmdd` | Nao | `--val-end 20241231` | Janela de validacao |
| `--test-start` | `str` | `20250101` | `yyyymmdd` | Nao | `--test-start 20250101` | Janela de teste |
| `--test-end` | `str` | `20251231` | `yyyymmdd` | Nao | `--test-end 20251231` | Janela de teste |

## Parametros Avancados

| Parametro | Tipo | Default | Valores validos | Obrigatorio | Exemplo | Impacto no treino |
|---|---|---:|---|---|---|---|
| `--max-encoder-length` | `int` | `60` | `>= 2` | Nao | `--max-encoder-length 90` | Contexto historico usado pelo TFT |
| `--max-prediction-length` | `int` | `1` | `>= 1` e `<= max_encoder_length` | Nao | `--max-prediction-length 1` | Horizonte de previsao |
| `--batch-size` | `int` | `64` | `>= 1` | Nao | `--batch-size 128` | Custo por passo e estabilidade |
| `--max-epochs` | `int` | `20` | `>= 1` | Nao | `--max-epochs 40` | Duracao maxima do treino |
| `--learning-rate` | `float` | `0.001` | `> 0` | Nao | `--learning-rate 0.0005` | Velocidade de otimizacao |
| `--hidden-size` | `int` | `16` | `>= 1` | Nao | `--hidden-size 32` | Capacidade do modelo |
| `--attention-head-size` | `int` | `2` | `>= 1` | Nao | `--attention-head-size 4` | Resolucao da atencao |
| `--dropout` | `float` | `0.1` | `[0, 1]` | Nao | `--dropout 0.2` | Regularizacao |
| `--hidden-continuous-size` | `int` | `8` | `>= 1` | Nao | `--hidden-continuous-size 16` | Representacao de variaveis continuas |
| `--seed` | `int` | `42` | inteiro | Nao | `--seed 7` | Reprodutibilidade |
| `--early-stopping-patience` | `int` | `5` | `>= 0` | Nao | `--early-stopping-patience 8` | Parada antecipada |
| `--early-stopping-min-delta` | `float` | `0.0` | `>= 0` | Nao | `--early-stopping-min-delta 0.0001` | Sensibilidade da parada antecipada |
| `--run-ablation` | flag | `False` | booleano | Nao | `--run-ablation` | Executa ablacoes adicionais (mais custo/tempo) |

## Observacoes

- Sem `--features`, o treino usa `DEFAULT_TFT_FEATURES` disponivel no dataset.
- `--features` aceita grupos e colunas combinadas.
- A validacao de ranges e feita antes do treino. Em caso invalido, o processo falha com mensagem clara.
- A configuracao efetiva usada no treino e salva no `metadata.json` em `training_config`.

## Configuracao Via JSON (`--config-json`)

Voce pode passar um arquivo JSON com parametros de treino:

```bash
python -m src.main_train_tft --asset AAPL --config-json config/train_tft_aapl.json
```

Exemplo de `config/train_tft_aapl.json`:

```json
{
  "features": ["BASELINE_FEATURES", "TECHNICAL_FEATURES"],
  "run_ablation": false,
  "training_config": {
    "max_encoder_length": 60,
    "max_prediction_length": 1,
    "batch_size": 64,
    "max_epochs": 25,
    "learning_rate": 0.001,
    "hidden_size": 16,
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
  }
}
```

Regras de precedencia:

- `defaults` <- `JSON` <- `CLI`
- Se um parametro for informado no CLI, ele sobrescreve o valor do JSON.
- `features` no JSON aceita lista ou string CSV.

## Exemplos De Uso

Execucao minima:

```bash
python -m src.main_train_tft --asset AAPL
```

Selecao de features por colunas:

```bash
python -m src.main_train_tft --asset AAPL --features close,volume,rsi_14,ema_50,sentiment_score,news_volume
```

Selecao de features por grupos:

```bash
python -m src.main_train_tft --asset AAPL --features BASELINE_FEATURES,TECHNICAL_FEATURES
```

Mistura de grupos + features customizadas:

```bash
python -m src.main_train_tft --asset AAPL --features BASELINE_FEATURES,ema_50,sentiment_score
```

Ajuste de hiperparametros:

```bash
python -m src.main_train_tft --asset AAPL --max-epochs 30 --batch-size 128 --learning-rate 0.0005
```

Split temporal explicito:

```bash
python -m src.main_train_tft --asset AAPL --train-start 20100101 --train-end 20201231 --val-start 20210101 --val-end 20221231 --test-start 20230101 --test-end 20251231
```

Ablacao (desligada por padrao):

```bash
python -m src.main_train_tft --asset AAPL --run-ablation
```

Execucao com JSON:

```bash
python -m src.main_train_tft --asset AAPL --config-json config/train_tft_aapl.json
```
