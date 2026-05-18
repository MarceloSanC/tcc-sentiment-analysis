---
title: Data Sources
scope: Fontes externas usadas no projeto (yfinance para candles, finnhub/alpha vantage para news, alpha vantage para fundamentals) e caminho canonico de ingestao de cada uma.
update_when:
  - nova fonte de dados externa for integrada
  - fonte existente for substituida ou descontinuada
  - politica de incremental/refetch mudar
  - chave de API ou autenticacao mudar
  - fallback de `reported_date` ausente para fundamentals mudar (valor de
    dias ou metodo)
canonical_for: [data_sources, external_apis, ingestion_path, yfinance, finnhub, alpha_vantage, fundamentals_fallback_policy]
---

# Data Sources

## Purpose
Definir as fontes de dados usadas no projeto e o caminho canonico de ingestao
de cada uma, com rastreabilidade para implementacao e validacao.

## Canonical Sources

### Market candles (OHLCV)
- Origem: `yfinance`
- Camada: `data/raw/market/candles/{ASSET}/`
- Contrato de schema: `src/infrastructure/schemas/candle_parquet_schema.py`
- Pipeline:
  - `src/main_candles.py`
  - `src/use_cases/fetch_candles_use_case.py`
  - `src/adapters/yfinance_candle_fetcher.py`
  - `src/adapters/parquet_candle_repository.py`

### News (raw)
- Origens: `Finnhub`, `Alpha Vantage`
- Camada: `data/raw/news/{ASSET}/`
- Contrato de schema: `src/infrastructure/schemas/news_parquet_schema.py`
- Pipeline:
  - `src/main_news_dataset.py`
  - `src/use_cases/fetch_news_use_case.py`
  - `src/adapters/finnhub_news_fetcher.py`
  - `src/adapters/alpha_vantage_news_fetcher.py`
  - `src/adapters/parquet_news_repository.py`

### Sentiment (processed)
- Origem: scoring FinBERT sobre news raw
- Camadas:
  - `data/processed/scored_news/{ASSET}/`
  - `data/processed/sentiment_daily/{ASSET}/`
- Contratos de schema:
  - `src/infrastructure/schemas/scored_news_parquet_schema.py`
  - `src/infrastructure/schemas/daily_sentiment_parquet_schema.py`
- Pipeline:
  - `src/main_sentiment.py`
  - `src/main_sentiment_features.py`
  - `src/use_cases/infer_sentiment_use_case.py`
  - `src/use_cases/sentiment_feature_engineering_use_case.py`

### Technical indicators (processed)
- Origem: calculo causal sobre OHLCV
- Camada: `data/processed/technical_indicators/{ASSET}/`
- Contratos de schema:
  - `src/infrastructure/schemas/technical_indicator_parquet_schema.py`
  - `src/infrastructure/schemas/technical_indicators_schema.py`
- Pipeline:
  - `src/main_technical_indicators.py`
  - `src/use_cases/technical_indicator_engineering_use_case.py`
  - `src/adapters/technical_indicator_calculator.py`

### Fundamentals (processed)
- Origem: Alpha Vantage fundamentals
- Camada: `data/processed/fundamental_indicators/{ASSET}/`
- Contrato de schema: `src/infrastructure/schemas/fundamental_parquet_schema.py`
- Pipeline:
  - `src/main_fundamentals.py`
  - `src/use_cases/fetch_fundamentals_use_case.py`

#### Fallback `reported_date` ausente (`fiscal_date_end + 45 dias`)

Quando um report de fundamentals nao tem `reported_date` populado pela fonte
externa (caso comum em Alpha Vantage para reports antigos; ~17 de 81 reports
de AAPL no estado auditado em `A_code_audit.md` §M3-Q4), o build do dataset
TFT usa fallback de 45 dias corridos apos `fiscal_date_end`:

```python
# src/use_cases/build_tft_dataset_use_case.py:117-118
reported_date = r.fiscal_date_end + timedelta(days=45)
```

**Motivacao SEC**: a SEC exige que reports anuais (Form 10-K) sejam arquivados
em 60 a 90 dias apos o fim do periodo fiscal e que reports trimestrais (Form
10-Q) sejam arquivados em 40 a 45 dias apos o fim do trimestre fiscal,
dependendo do tipo de filer (large accelerated, accelerated ou
non-accelerated). O valor de 45 dias e estimativa **conservadora
intermediaria** — provavelmente posterior a publicacao real para large
accelerated filers (AAPL incluso, prazo 60 dias 10-K / 40 dias 10-Q), mas
seguramente anterior a 10-Ks de qualquer filer e a 10-Qs de filers
non-accelerated (prazo 45 dias). Usar valor mais alto (60+ dias) reduziria
coverage no inicio da serie; valor mais baixo (30 dias) introduziria risco
de leakage para reports anuais.

**Cobertura empirica** (AAPL, audit §M3-Q4 linhas 401-403): 17 de 81 reports
usam fallback; primeira data efetiva no dataset coincide com primeiro
`effective_day` observado; nenhum uso de `revenue` antes da menor data
efetiva.

**Sensibilidade**: substituir 45 por 30 ou 60 dias muda a primeira data
disponivel de cada feature fundamental em ate 15 dias, podendo deslocar a
fronteira de warmup mas sem introduzir leakage se a substituicao for em
direcao conservadora (mais dias). O valor 45 esta pre-registrado para
Phase B; alteracao futura exige emenda datada ao pre-registro.

**Auditabilidade**: pos-Stage 14, a coluna `fundamentals_effective_date` no
`dataset_tft_<ASSET>.parquet` permite verificar diretamente, por linha, qual
foi a data efetiva usada — `fundamentals_effective_date == fiscal_date_end
+ 45 dias` indica linha onde o fallback foi aplicado. Defense-in-depth
assert no proprio build garante `fundamentals_effective_date <= date` em
todas as linhas.

**Referencia**: SEC filing deadlines —
https://www.sec.gov/files/quick-edgar-tutorial.pdf
(secao sobre prazos 10-K/10-Q).

### TFT dataset (processed)
- Origem: uniao temporal de candles + technical + sentiment + fundamentals
- Camada: `data/processed/dataset_tft/{ASSET}/`
- Contratos de schema:
  - `src/infrastructure/schemas/tft_dataset_schema.py`
  - `src/infrastructure/schemas/tft_dataset_parquet_schema.py`
- Pipeline:
  - `src/main_dataset_tft.py`
  - `src/use_cases/build_tft_dataset_use_case.py`
  - `src/adapters/parquet_tft_dataset_repository.py`

### Inference outputs
- Camada: `data/processed/inference_tft/{ASSET}/`
- Contrato de schema: `src/infrastructure/schemas/tft_inference_parquet_schema.py`
- Pipeline:
  - `src/main_infer_tft.py`
  - `src/use_cases/run_tft_inference_use_case.py`
  - `src/adapters/parquet_tft_inference_repository.py`

### Analytics store (silver + gold)
- Silver (source of truth analitico): facts/dims em Parquet
- Gold (derivados para decisao): tabelas agregadas reconstruiveis
- Contrato de schema: `src/infrastructure/schemas/analytics_store_schema.py`
- Pipeline:
  - `src/main_refresh_analytics_store.py`
  - `src/use_cases/refresh_analytics_store_use_case.py`
  - `src/use_cases/validate_analytics_quality_use_case.py`
  - `src/adapters/parquet_analytics_run_repository.py`

## Operational Validation
- Runbook end-to-end: `docs/06_runbooks/RUN_DATASET.md`
- Quick smoke:
  - `python -m src.main_candles --asset AAPL`
  - `python -m src.main_dataset_tft --asset AAPL`
  - `python -m src.main_train_tft --asset AAPL`
  - `python -m src.main_infer_tft --asset AAPL --model-path <MODEL_VERSION> --start 20260101 --end 20260228`
- Quality gate:
  - `python -m src.main_refresh_analytics_store --fail-on-quality`

## Checklist Cross-Reference
- Tracking operacional: `docs/05_checklists/CHECKLISTS.md`
- Checklist de analytics: `docs/05_checklists/ANALYTICS_STORE_CHECKLIST.md`
