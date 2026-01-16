# Executando o Pipeline

Este projeto é dividido em **3 pipelines principais**, executados nesta ordem:

1) **Candles (OHLCV)** → baixa e persiste preços históricos  
2) **Sentimento** → coleta notícias e enriquece candles com sentimento diário  
3) **Features** → calcula indicadores técnicos e gera features normalizadas

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
- O período é lido de `config/data_sources.yaml`.
- Este pipeline deve rodar **antes** do sentimento (porque o sentimento enriquece candles já existentes).


## 2) Rodar Pipeline de Sentimento (enriquecimento dos candles)

Coleta notícias via **Finnhub**, infere sentimento com **FinBERT**, agrega por dia (UTC) e persiste no mesmo parquet de candles.

Executar:

```bash
python -m src.main_sentiment --asset AAPL
```

**Observações**
- O período (start/end) também é lido de `config/data_sources.yaml`.
- Se `data_sources.sentiment.enabled: false`, o pipeline encerra sem executar.


## 3) Gerar Features Técnicas

Lê candles já enriquecidos e gera features em:

- `data/processed/features/{ASSET}/features_{ASSET}.parquet`

Executar:

```bash
python -m src.main_features --asset AAPL
```

Para sobrescrever um arquivo existente:

```bash
python -m src.main_features --asset AAPL --overwrite
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
