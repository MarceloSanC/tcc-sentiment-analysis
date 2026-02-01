# Executando o Pipeline

Este projeto é dividido em **5 pipelines principais**, executados nesta ordem:

1) **Candles (OHLCV)** → baixa e persiste preços históricos  
2) **Sentimento (raw news)** → coleta notícias brutas e salva em parquet raw  
3) **Scoring de notícias** → infere sentimento por notícia (parquet processed)  
4) **Features de sentimento** → agrega sentimento diário a partir do scored news  
5) **Indicadores técnicos** → calcula indicadores técnicos a partir dos candles

> **Regra importante (consistência temporal):** o projeto padroniza timestamps em **UTC** para evitar ambiguidade e *lookahead bias*.


## Pré-requisitos

### 1) Variáveis de ambiente (Finnhub)

Crie um `.env` (ou exporte no seu shell) com:

```bash
FINNHUB_API_KEY="SUA_CHAVE_AQUI"
```

### 2) Configuração de ativos e janelas temporais

A janela de execução (start/end) e os ativos ficam em:

- `config/data_sources.yaml`

Exemplo:

```yaml
assets:
  - symbol: "AAPL"
    name: "Apple Inc"
    start_date: "2024-01-01"
    end_date: "2024-12-31"

data_sources:
  candles:
    provider: "yfinance"
    interval: "1d"

  sentiment:
    enabled: true
    provider: "finnhub"
    aggregation: "daily"
```


## 1) Coletar Candles (OHLCV)

Baixa candles via **YFinance** e persiste em:

- `data/raw/market/candles/{ASSET}/candles_{ASSET}_1d.parquet`

Executar:

```bash
python -m src.main_candles --asset AAPL
```

**Observações**

O período é lido de `config/data_sources.yaml`.
- Este pipeline deve rodar **antes** do sentimento.


## 2) Rodar Pipeline de Sentimento (raw news)

Coleta notícias via **Alpha Vantage** e persiste o dataset raw em parquet.

Executar:

```bash
python -m src.main_news_dataset --asset AAPL
```

**Observações**
- O período (start/end) também é lido de `config/data_sources.yaml`.
- A execução é incremental e deduplica por URL/article_id.


## 3) Rodar Scoring de Notícias (FinBERT → parquet processed)

Lê data/raw/news/{ASSET} e gera:

- `data/processed/scored_news/{ASSET}/scored_news_{ASSET}.parquet` (sentimento por notícia)

Executar:

```bash
python -m src.main_sentiment --asset AAPL
```

**Observações**
- O período (start/end) é lido de `config/data_sources.yaml`.
- O pipeline ignora notícias já pontuadas (skip por article_id).


## 4) Gerar Features de Sentimento (agregação diária)

Lê `data/processed/scored_news/{ASSET}` e gera:

- `data/processed/sentiment_daily/{ASSET}/daily_sentiment_{ASSET}.parquet` (agregado diário)

Executar:

```bash
python -m src.main_sentiment_features --asset AAPL
```

**Observações**
- O período (start/end) é lido de `config/data_sources.yaml`.
- Reprocessamento é idempotente por dia.


## 5) Gerar Indicadores Técnicos

Lê candles já enriquecidos e gera indicadores técnicos em:

- `data/processed/technical_indicators/{ASSET}/technical_indicators_{ASSET}.parquet`

Executar:

```bash
python -m src.main_technical_indicators --asset AAPL
```

Para sobrescrever um arquivo existente:

```bash
python -m src.main_technical_indicators --asset AAPL --overwrite
```


## Rotinas de Qualidade

### Formatação e lint

```bash
make format
make lint
```

### Tipagem

```bash
make type-check
```

### Testes

Rodar unit (sem integration):

```bash
make test
```

Rodar integration:

```bash
make test-integration
```

Rodar tudo:

```bash
make test-all
```
