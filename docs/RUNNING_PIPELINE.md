# Executando o Pipeline

O sistema é dividido em partes modulares. Abaixo, como executar cada uma.

## 1. Coleta e análise de sentimento

Coleta notícias via Finnhub e aplica FinBERT:

```
python src/main.py
```

> Configuração: `TICKERS` em `src/main.py`

## 2. Coleta de candles (OHLCV)

Baixa dados históricos de preço via YFinance:

```
python -m src.main_candles --asset PETR4.SA
```

> Configuração: `config/data_sources.yaml`

## 3. Formatação e linting

Mantenha o código padronizado:

```
make format
make lint
```

## 4. Verificação de tipo

Valide anotações com `mypy`:

```
make type-check
```