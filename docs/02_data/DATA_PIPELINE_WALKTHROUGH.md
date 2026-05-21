---
title: Data Pipeline Walkthrough
scope: Walk-through pedagogico do pipeline de dados, de ingestao raw ate tabelas gold. Para cada celula (stage x scope) lista inputs, outputs e calculos com referencias exatas (file:line). Complementa DATA_FLOW.md (paths/ordem) e DATA_SOURCES.md (fontes externas).
update_when:
  - novo escopo (candles/news/sentiment/fundamentals/etc.) for adicionado ao pipeline
  - nova coluna for adicionada ao schema de qualquer stage
  - formula ou referencia file:line de calculo existente mudar
  - novo stage (alem de raw/processed/silver/gold) for introduzido
canonical_for: [data_walkthrough, pipeline_io_inventory, pipeline_calculations_inventory]
---

# Data Pipeline Walkthrough

## 0. Como ler este doc

Este documento e um walk-through **exaustivo** do pipeline de dados, organizado em duas dimensoes:
- **Eixo vertical (stages)**: `raw` -> `processed` -> `silver (analytics)` -> `gold (analytics)`.
- **Eixo horizontal (scopes)**: `candles`, `technical_indicators`, `news`, `sentiment`, `fundamentals`, `dataset_tft`, `inference`, `baselines`, `analytics`.

Cada celula (`stage x scope`) usa o mesmo bloco padrao:

```
### N.M <scope>
**Entrypoint** / **Use case** / **Adapter(s)** / **Schema**

#### Inputs
| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |

#### Outputs
| # | Coluna | Dtype | Ref schema |

#### Calculos
| # | Output col | Inputs (cols) | Formula | Ref implementacao |

#### Quality gates aplicados

#### Lacunas (link para §A)
```

Toda formula tem ref `file:line`. Quando uma celula nao aplica calculo (pass-through ou pure I/O), e dito explicitamente. Lacunas conhecidas estao em §A.

Documentos relacionados:
- [`01_architecture/DATA_FLOW.md`](../01_architecture/DATA_FLOW.md): paths e ordem operacional (mais alto nivel)
- [`02_data/DATA_SOURCES.md`](DATA_SOURCES.md): fontes externas e contratos de ingestao
- [`02_data/DATA_CONTRACTS.md`](DATA_CONTRACTS.md): `run_id`, fingerprints, `schema_version`
- [`02_data/FEATURE_SETS.md`](FEATURE_SETS.md): registry de features
- [`01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md): camadas silver/gold

## 1. Visao geral

```
yfinance/finnhub/alpha_vantage   (fontes externas)
        |
        v
[raw]    data/raw/market/candles/{ASSET}/        +  data/raw/news/{ASSET}/
        |                                            |
        v                                            v
[processed] data/processed/technical_indicators/   data/processed/scored_news/
            data/processed/sentiment_daily/        data/processed/fundamentals/
                          |
                          v
            data/processed/dataset_tft/{ASSET}/    (join de tudo + feature engineering)
                          |
                          v
            data/processed/inference_tft/{ASSET}/  (predict do TFT)
                          |
                          v
[silver]  data/analytics/silver/{dim_run, fact_*}/asset={ASSET}/
                          |
                          v
[gold]    data/analytics/gold/gold_*.parquet
```

Matriz de celulas (legenda: OK existe / PT pass-through / -- nao aplica):

| Stage \ Scope | candles | technical | news | sentiment | fundamentals | dataset_tft | inference | baselines | analytics |
|---|---|---|---|---|---|---|---|---|---|
| raw       | OK | -- | OK | -- | PT | -- | -- | -- | -- |
| processed | PT (mutado in-place por sentiment merge) | OK | -- | OK (scored + daily) | OK | OK | OK | -- | -- |
| silver    | -- | -- | -- | -- | -- | -- | -- | OK (§5) | OK (§5) |
| gold      | -- | -- | -- | -- | -- | -- | -- | -- | OK (§6) |

## 2. Stage: raw

### 2.1 candles

**Entrypoint**: [`src/main_candles.py`](../../src/main_candles.py)
**Use case**: [`src/use_cases/fetch_candles_use_case.py`](../../src/use_cases/fetch_candles_use_case.py)
**Adapters**: [`src/adapters/yfinance_candle_fetcher.py`](../../src/adapters/yfinance_candle_fetcher.py), [`src/adapters/parquet_candle_repository.py`](../../src/adapters/parquet_candle_repository.py)
**Schema**: [`src/infrastructure/schemas/candle_parquet_schema.py`](../../src/infrastructure/schemas/candle_parquet_schema.py)

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| `--asset` (CLI) | argv | — | `src/main_candles.py:29` |
| `config/data_sources.yaml` | filesystem | `assets[*].symbol`, `assets[*].data_period.{start_date,end_date}` | `src/main_candles.py:21-38`, `src/utils/asset_periods.py:12-13` |
| `config/data_paths.yaml` | filesystem | `data.raw.candles` | `src/utils/path_resolver.py:24` |
| Env `DATA_ROOT` (opt) | env | — | `src/utils/path_resolver.py:15-16` |
| yfinance API | HTTP | `Open, High, Low, Close, Volume` | `src/adapters/yfinance_candle_fetcher.py:55-75` |
| Existing parquet (count only) | `data/raw/market/candles/{SYMBOL}/candles_{SYMBOL}_1d.parquet` | (file size) | `src/use_cases/fetch_candles_use_case.py:46-52` |

#### Outputs

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `timestamp` | datetime UTC midnight (no dtype declarado — gap §A.1) | `candle_parquet_schema.py:4` |
| 2 | `open` | float32 | `candle_parquet_schema.py:5,13` |
| 3 | `high` | float32 | `candle_parquet_schema.py:6,14` |
| 4 | `low` | float32 | `candle_parquet_schema.py:7,15` |
| 5 | `close` | float32 | `candle_parquet_schema.py:8,16` |
| 6 | `volume` | int64 | `candle_parquet_schema.py:9,17` |

Path de saida: `data/raw/market/candles/{SYMBOL}/candles_{SYMBOL}_1d.parquet` — `src/adapters/parquet_candle_repository.py:55-57,139`.
Report DQ adicional: `data/raw/market/candles/{SYMBOL}/reports/candles_report_{TS}.json` — `src/main_candles.py:81`, profile `candles` em `src/domain/services/data_quality_profiles.py:34-54`.

#### Calculos

| # | Output col | Inputs (cols) | Formula | Ref implementacao |
|---|---|---|---|---|
| 1 | `end_inclusive` (interno) | `end` (UTC datetime) | `end + 1d if end.time()==00:00 else combine(end.date()+1d, 00:00, UTC)` | `src/use_cases/fetch_candles_use_case.py:20-30` |
| 2 | `timestamp` | yfinance index | `combine(ts_utc.date(), 00:00, UTC)` | `src/adapters/yfinance_candle_fetcher.py:82-88` |
| 3 | `open/high/low/close` | yfinance cols | `float(row["Open"|"High"|"Low"|"Close"])` | `src/adapters/yfinance_candle_fetcher.py:93-96` |
| 4 | `volume` | yfinance col | `int(row["Volume"])` | `src/adapters/yfinance_candle_fetcher.py:97` |
| 5 | (invariante) | `close, volume` | assert `close>0`, `volume>=0` | `src/entities/candle.py:15-20` |

Ordering: `df.sort_values("timestamp")` no write e no read — `src/adapters/parquet_candle_repository.py:137, 85-90`.
Dtype enforcement: `df.astype(CANDLE_PARQUET_DTYPES)` — `src/adapters/parquet_candle_repository.py:136`.

#### Quality gates aplicados

- Profile `candles` ([`data_quality_profiles.py:34-54`](../../src/domain/services/data_quality_profiles.py)): null counts, dup keys em `timestamp`, ranges `[open,high,low,close,volume]>=0`, OHLC comparison (`high>=low`, `high>=open`, `high>=close`, `low<=open`, `low<=close`), business-day inventory — computado em `data_quality_reporter.py:28-199`.

#### Lacunas (links para §A)

- §A.1 (dtype `timestamp`), §A.2 (`update_sentiment` muta in-place), §A.3 (`provider/interval` decorativos), §A.4 (`CANDLE_PARQUET_COLUMNS` e `set`, ordem nao deterministica), §A.5 (sem incremental append — rewrite total), §A.6 (DQ report nao consulta calendario de feriados).

### 2.2 news

**Entrypoint**: [`src/main_news_dataset.py`](../../src/main_news_dataset.py)
**Use case**: [`src/use_cases/fetch_news_use_case.py`](../../src/use_cases/fetch_news_use_case.py)
**Adapter ativo**: [`src/adapters/alpha_vantage_news_fetcher.py`](../../src/adapters/alpha_vantage_news_fetcher.py), [`src/adapters/parquet_news_repository.py`](../../src/adapters/parquet_news_repository.py)
**Schema**: [`src/infrastructure/schemas/news_parquet_schema.py`](../../src/infrastructure/schemas/news_parquet_schema.py)

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| `--asset` (CLI) | argv | — | `src/main_news_dataset.py:144` |
| `ALPHAVANTAGE_API_KEY` | env | — | `src/main_news_dataset.py:115` |
| `config/data_sources.yaml` | filesystem | `data_sources.news_dataset.{build_enabled,provider,dataset_start,dataset_end,safety_margin}`, `assets[*].{symbol,data_period}` | `src/main_news_dataset.py:97-115` |
| `config/data_paths.yaml` | filesystem | `data.raw.news` (lookup key `raw_news` — gap §A.7) | `src/main_news_dataset.py:125`, `src/utils/path_resolver.py:41` |
| Alpha Vantage API | HTTPS `function=NEWS_SENTIMENT` | `feed[].{time_published,title,summary,source,url}` | `src/adapters/alpha_vantage_news_fetcher.py:103-162` |
| Existing parquet | `data/raw/news/{SYMBOL}/news_{SYMBOL}.parquet` (cursor) | `published_at` | `src/adapters/parquet_news_repository.py:94-111` |

#### Outputs

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `asset_id` | string | `news_parquet_schema.py:5,19` |
| 2 | `article_id` | string | `news_parquet_schema.py:6,20` |
| 3 | `published_at` | datetime64[ns, UTC] | `news_parquet_schema.py:7` (handled at `parquet_news_repository.py:147`) |
| 4 | `headline` | string | `news_parquet_schema.py:8,21` |
| 5 | `summary` | string | `news_parquet_schema.py:9,22` |
| 6 | `source` | string | `news_parquet_schema.py:10,23` |
| 7 | `url` | string | `news_parquet_schema.py:11,24` |
| 8 | `language` | string (hard-coded `"en"`) | `news_parquet_schema.py:12,25` |

Path: `data/raw/news/{SYMBOL}/news_{SYMBOL}.parquet` — `parquet_news_repository.py:178`.
Report DQ: profile `news_raw` em `data_quality_profiles.py`.

#### Calculos

| # | Output col | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | `published_at` | `feed[].time_published` | parse `YYYYMMDDTHHMM` ou `YYYYMMDDTHHMMSS` com `tzinfo=UTC` | `alpha_vantage_news_fetcher.py:60-82` |
| 2 | `article_id` | `url`, `time_published`, `headline` | `url or f"{time_published}:{headline[:80]}"` | `alpha_vantage_news_fetcher.py:170`; fallback enforce `article_id := url` em `parquet_news_repository.py:65-92` |
| 3 | `headline/summary` (fallback) | upstream vazio | `if both empty: " "` | `alpha_vantage_news_fetcher.py:165-167` |
| 4 | `asset_id` | input asset | `asset_id.split(".")[0].upper()` | `parquet_news_repository.py:53-55` |
| 5 | `language` | (constante) | hard-coded `"en"` | `alpha_vantage_news_fetcher.py:181` |
| 6 | (cursor logic) | `published_at` da batch | `cursor = max(published_at)`; advance 1 dia em batch vazio | `fetch_news_use_case.py:159-238` |
| 7 | (dedup) | `article_id` | `drop_duplicates(subset=["article_id"], keep="last")` | `parquet_news_repository.py:170-171` |

Throttle: 1.1 s entre requests — `alpha_vantage_news_fetcher.py:32,48-58`. Error handling `Note`/`Information` raise — `:132-135`.

#### Quality gates aplicados

- Profile `news_raw` em `data_quality_profiles.py` (verificacao em §A.8).

#### Lacunas

- §A.7 (`paths.get("raw_news")` lookup errado), §A.8 (profile DQ news_raw nao auditado), §A.9 (finnhub_news_fetcher dead code), §A.10 (sqlite_news_repository import quebrado), §A.11 (Alpha Vantage `topics`, `overall_sentiment_score`, `ticker_sentiment` descartados), §A.12 (`sort`/`limit` da YAML decorativos), §A.13 (`language` hard-coded `"en"`).

## 3. Stage: processed

### 3.1 technical_indicators

**Entrypoint**: [`src/main_technical_indicators.py`](../../src/main_technical_indicators.py)
**Use case**: [`src/use_cases/technical_indicator_engineering_use_case.py`](../../src/use_cases/technical_indicator_engineering_use_case.py)
**Adapters**: [`src/adapters/technical_indicator_calculator.py`](../../src/adapters/technical_indicator_calculator.py), [`src/adapters/parquet_technical_indicator_repository.py`](../../src/adapters/parquet_technical_indicator_repository.py)
**Schemas**: [`src/infrastructure/schemas/technical_indicators_schema.py`](../../src/infrastructure/schemas/technical_indicators_schema.py), [`src/infrastructure/schemas/technical_indicator_parquet_schema.py`](../../src/infrastructure/schemas/technical_indicator_parquet_schema.py)

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| `--asset` / `--overwrite` (CLI) | argv | — | `src/main_technical_indicators.py:31-42` |
| `config/data_paths.yaml` | filesystem | `data.raw.candles`, `data.processed.technical_indicators` | `src/utils/path_resolver.py:24-28` |
| Candles parquet | `data/raw/market/candles/{SYMBOL}/candles_{SYMBOL}_1d.parquet` | `timestamp,open,high,low,close,volume` | `src/adapters/parquet_candle_repository.py:75-103` |
| Existing output (coverage skip) | `data/processed/technical_indicators/{ASSET}/technical_indicators_{ASSET}.parquet` | `timestamp` | `src/main_technical_indicators.py:87-93` |

#### Outputs

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `asset_id` | string | `technical_indicator_parquet_schema.py:3-12` |
| 2 | `timestamp` | datetime (no dtype declarado — gap §A.14) | `technical_indicator_parquet_schema.py:5,12` |
| 3 | `rsi_14` | float32 | `technical_indicators_schema.py:6` |
| 4 | `ema_10` | float32 | `technical_indicators_schema.py:8` |
| 5 | `ema_50` | float32 | `technical_indicators_schema.py:9` |
| 6 | `ema_100` | float32 | `technical_indicators_schema.py:10` |
| 7 | `ema_200` | float32 | `technical_indicators_schema.py:11` |
| 8 | `macd` | float32 | `technical_indicators_schema.py:14` |
| 9 | `macd_signal` | float32 | `technical_indicators_schema.py:15` |
| 10 | `volatility_20d` | float32 | `technical_indicators_schema.py:18` |
| 11 | `candle_range` | float32 | `technical_indicators_schema.py:21` |
| 12 | `candle_body` | float32 | `technical_indicators_schema.py:22` |

Path: `data/processed/technical_indicators/{ASSET}/technical_indicators_{ASSET}.parquet` — `parquet_technical_indicator_repository.py:33,65`.
Coercao dtype: `pd.to_numeric(..., errors="coerce").astype("float32")` — `parquet_technical_indicator_repository.py:50-54`.

#### Calculos

| # | Output col | Inputs (cols) | Formula | Ref |
|---|---|---|---|---|
| 1 | `rsi_14` | `close` | `pandas_ta.rsi(close, length=14)` (Wilder RSI) | `technical_indicator_calculator.py:46` |
| 2 | `macd` | `close` | `ta.macd(close)["MACD_12_26_9"]` = `EMA12(close) - EMA26(close)` | `technical_indicator_calculator.py:48-49` |
| 3 | `macd_signal` | `close` | `ta.macd(close)["MACDs_12_26_9"]` = `EMA9(macd)` | `technical_indicator_calculator.py:48,50` |
| 4 | `ema_10` | `close` | `pandas_ta.ema(close, length=10)` | `technical_indicator_calculator.py:52-53` |
| 5 | `ema_50` | `close` | `pandas_ta.ema(close, length=50)` | `technical_indicator_calculator.py:52-53` |
| 6 | `ema_100` | `close` | `pandas_ta.ema(close, length=100)` | `technical_indicator_calculator.py:52-53` |
| 7 | `ema_200` | `close` | `pandas_ta.ema(close, length=200)` | `technical_indicator_calculator.py:52-53` |
| 8 | `volatility_20d` | `close` | `close.pct_change().rolling(20).std()` (stdev de retornos **simples**) | `technical_indicator_calculator.py:55` |
| 9 | `candle_range` | `high, low` | `high - low` | `technical_indicator_calculator.py:56` |
| 10 | `candle_body` | `open, close` | `abs(close - open)` | `technical_indicator_calculator.py:57` |

Upsert quando `--overwrite=False`: `concat([old, new]).drop_duplicates(["asset_id","timestamp"], keep="last")` — `parquet_technical_indicator_repository.py:59-63`.
Coverage skip: `existing_start <= requested_start AND existing_end >= requested_end` — `src/main_technical_indicators.py:106-123`.

#### Quality gates aplicados

- Profile `technical_indicators` em `data_quality_profiles.py:94-100`: SOMENTE null/dup/business-day. **Sem `value_ranges`/`validation_rules`** — inf/NaN/explosao silenciosa (§A.15).

#### Lacunas

- §A.14 (dtype `timestamp`), §A.15 (DQ profile fraco), §A.16 (sem leakage guard — calcula em todo o range; trim ocorre downstream), §A.17 (`volatility_20d` usa **simple** returns; `formula_desc` no schema nao pin), §A.18 (`SklearnTechnicalIndicatorNormalizer` dead code), §A.19 (warmup rows nao tratadas).

### 3.2 scored_news

**Entrypoint**: [`src/main_sentiment.py`](../../src/main_sentiment.py)
**Use case**: [`src/use_cases/infer_sentiment_use_case.py`](../../src/use_cases/infer_sentiment_use_case.py)
**Adapters**: [`src/adapters/finbert_sentiment_model.py`](../../src/adapters/finbert_sentiment_model.py), [`src/adapters/parquet_news_repository.py`](../../src/adapters/parquet_news_repository.py), [`src/adapters/parquet_scored_news_repository.py`](../../src/adapters/parquet_scored_news_repository.py)
**Schema**: [`src/infrastructure/schemas/scored_news_parquet_schema.py`](../../src/infrastructure/schemas/scored_news_parquet_schema.py)

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| `--asset` (CLI) | argv | — | `src/main_sentiment.py:36-44` |
| `config/data_paths.yaml` | filesystem | `data.raw.news` (key `news_dataset`), `data.processed.news_scored` (key `processed_news_scored`) | `src/main_sentiment.py:57-62` |
| News parquet | `data/raw/news/{SYMBOL}/news_{SYMBOL}.parquet` | `asset_id,article_id,published_at,headline,summary,source,url,language` | `parquet_news_repository.py:190-238` |
| Existing scored (skip) | `data/processed/scored_news/{SYMBOL}/scored_news_{SYMBOL}.parquet` | `article_id` | `parquet_scored_news_repository.py:202-215` |
| FinBERT model | HuggingFace Hub `ProsusAI/finbert` | weights + tokenizer | `finbert_sentiment_model.py:33,48-49` |
| Device | `torch.cuda.is_available()` | — | `finbert_sentiment_model.py:42-46` |

Parametros: `batch_size=16` (default), `max_length=512`, padding+truncation — `finbert_sentiment_model.py:35-36, 105-111`.

#### Outputs

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `asset_id` | string | `scored_news_parquet_schema.py:3,16` |
| 2 | `article_id` | string | `scored_news_parquet_schema.py:4,17` |
| 3 | `published_at` | datetime64[ns, UTC] | `scored_news_parquet_schema.py:5` (cast em `parquet_scored_news_repository.py:107`) |
| 4 | `sentiment_score` | float64 ∈ [-1, +1] | `scored_news_parquet_schema.py:6,18`; range guard em `src/entities/scored_news_article.py:51-52` |
| 5 | `confidence` | Float64 nullable ∈ [0, 1] | `scored_news_parquet_schema.py:7,19`; range guard `:55-59` |
| 6 | `model_name` | string | `scored_news_parquet_schema.py:8,20` |

Path: `data/processed/scored_news/{SYMBOL}/scored_news_{SYMBOL}.parquet` — `parquet_scored_news_repository.py:136`.

#### Calculos

| # | Output col | Inputs (cols) | Formula | Ref |
|---|---|---|---|---|
| 1 | `text` (interno) | `headline, summary` | `" ".join([h, s] if non-empty) or " "` | `finbert_sentiment_model.py:91-98` |
| 2 | `logits, probs` (interno) | `text` (tokens) | `probs = softmax(model(tokens).logits, dim=1)` (label order `[neg, neu, pos]`) | `finbert_sentiment_model.py:114-118` |
| 3 | `sentiment_score` | `probs` | `P(pos) - P(neg) = probs[:,2] - probs[:,0]` | `finbert_sentiment_model.py:117` |
| 4 | `confidence` | `sentiment_score` | `abs(sentiment_score)` | `finbert_sentiment_model.py:84` |
| 5 | `model_name` | (constante) | literal `"ProsusAI/finbert"` | `finbert_sentiment_model.py:33,85` |
| 6 | (skip filter) | scored ids existentes | `[a for a in articles if a.article_id not in scored_ids]` | `infer_sentiment_use_case.py:94-95` |
| 7 | (dedup) | `article_id` | `drop_duplicates(subset=["article_id"], keep="last")` + sort por `published_at` | `parquet_scored_news_repository.py:129-130` |

#### Quality gates aplicados

- Profile `scored_news` em `data_quality_profiles.py` (a verificar §A.20).

#### Lacunas

- §A.20 (profile DQ scored_news nao auditado), §A.21 (probs de 3 classes descartadas), §A.22 (`confidence=|score|` nao calibrada — colapsa quando neg≈pos mesmo com neu alto), §A.23 (sem `revision`/commit do HF), §A.24 (sem assert contra `model.config.id2label`), §A.25 (truncation 512 silenciosa), §A.26 (`language` nao propagada).

### 3.3 sentiment_daily

**Entrypoint**: [`src/main_sentiment_features.py`](../../src/main_sentiment_features.py)
**Use case**: [`src/use_cases/sentiment_feature_engineering_use_case.py`](../../src/use_cases/sentiment_feature_engineering_use_case.py)
**Adapters**: [`src/adapters/parquet_scored_news_repository.py`](../../src/adapters/parquet_scored_news_repository.py) (read), [`src/adapters/parquet_daily_sentiment_repository.py`](../../src/adapters/parquet_daily_sentiment_repository.py) (write)
**Domain**: [`src/domain/services/sentiment_aggregator.py`](../../src/domain/services/sentiment_aggregator.py), [`src/domain/time/trading_calendar.py`](../../src/domain/time/trading_calendar.py)
**Schema**: [`src/infrastructure/schemas/daily_sentiment_parquet_schema.py`](../../src/infrastructure/schemas/daily_sentiment_parquet_schema.py)

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| `--asset` (CLI) | argv | — | `src/main_sentiment_features.py:41-49` |
| `config/data_sources.yaml` | filesystem | `assets[*].{open_hour,close_hour,weekends}` (TradingDayPolicy) | `src/domain/time/trading_calendar.py:40-52`, `src/main_sentiment_features.py:60` |
| `config/data_paths.yaml` | filesystem | `data.processed.news_scored` (key `processed_news_scored`), `data.processed.sentiment_daily` (key `processed_sentiment_daily`) | `src/main_sentiment_features.py:63-68` |
| Scored news parquet | `data/processed/scored_news/{SYMBOL}/scored_news_{SYMBOL}.parquet` | `asset_id,article_id,published_at,sentiment_score,confidence,model_name` | `parquet_scored_news_repository.py:148-200` |

#### Outputs

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `asset_id` | string | `daily_sentiment_parquet_schema.py:3,14` |
| 2 | `day` | datetime64[ns, UTC] | `daily_sentiment_parquet_schema.py:4` (cast em `parquet_daily_sentiment_repository.py:99`) |
| 3 | `sentiment_score` | float64 | `daily_sentiment_parquet_schema.py:5,16` |
| 4 | `n_articles` | int64 | `daily_sentiment_parquet_schema.py:6,17` |
| 5 | `sentiment_std` | Float64 nullable | `daily_sentiment_parquet_schema.py:7,18` |

Path: `data/processed/sentiment_daily/{SYMBOL}/daily_sentiment_{SYMBOL}.parquet` — `parquet_daily_sentiment_repository.py:131`.

#### Calculos

| # | Output col | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | `day` (bucketing) | `published_at`, TradingDayPolicy | `trading_day_from_timestamp(ts, policy)` — `if ts.time() > close_hour: next_day`; `if not weekends: roll forward to Monday` | `trading_calendar.py:62-82`; aggregator `sentiment_aggregator.py:60-64` |
| 2 | `sentiment_score` | `sentiment_score` (per-article) | `mean(scores_of_day)` | `sentiment_aggregator.py:70-78` |
| 3 | `n_articles` | `article_id` | `len(scores_of_day)` | `sentiment_aggregator.py:75` |
| 4 | `sentiment_std` | `sentiment_score` | `pstdev(scores) if len>1 else 0.0` (population stdev) | `sentiment_aggregator.py:76-78` |
| 5 | (no-news fill) | trading days em `[start, end]` | `if day_has_no_articles: emit (sentiment_score=0.0, n_articles=0, sentiment_std=0.0)`, respeitando `weekends` | `sentiment_feature_engineering_use_case.py:173-208` |
| 6 | (causality guard) | tudo | (a) cada artigo mapeia para day in window; (b) cada day em window; (c) `n_articles` bate com fonte; (d) sinteticos = (0,0,0); (e) sem days faltando | `sentiment_feature_engineering_use_case.py:114-171` |

#### Quality gates aplicados

- Profile `sentiment_daily` em `data_quality_profiles.py:81-89`.

#### Lacunas

- §A.27 (agregacao fixa em mean — `confidence` ignorada), §A.28 (`model_name` nao propagado), §A.29 (sem calendario de feriados US), §A.30 (`sentiment_score=0.0` no-news colide com neutro real — `has_news` so existe downstream), §A.31 (`sentiment_std=0.0` para `n_articles=1` igual a sintetico).

### 3.4 fundamentals

**Entrypoint**: [`src/main_fundamentals.py`](../../src/main_fundamentals.py)
**Use case**: [`src/use_cases/fetch_fundamentals_use_case.py`](../../src/use_cases/fetch_fundamentals_use_case.py)
**Adapters**: [`src/adapters/alpha_vantage_fundamental_fetcher.py`](../../src/adapters/alpha_vantage_fundamental_fetcher.py), [`src/adapters/parquet_fundamental_repository.py`](../../src/adapters/parquet_fundamental_repository.py)
**Schema**: [`src/infrastructure/schemas/fundamental_parquet_schema.py`](../../src/infrastructure/schemas/fundamental_parquet_schema.py)

Nota: o codigo persiste em `data/processed/fundamentals/{ASSET}/`, **nao** em `data/processed/fundamental_indicators/` como diz a doc atual (§A.32). Indicadores derivados (`net_margin`, `leverage_ratio`, etc.) sao calculados em `dataset_tft`, nao aqui.

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| `--asset` (CLI) | argv | — | `src/main_fundamentals.py:40,68` |
| `config/data_sources.yaml` | filesystem | `assets[*].{symbol,data_period}` (chaves `enabled/provider/report_types` declaradas mas **ignoradas** — §A.33) | `src/main_fundamentals.py:71-78` |
| `ALPHAVANTAGE_API_KEY` | env | — | `src/main_fundamentals.py:80-82` |
| `config/data_paths.yaml` | filesystem | `data.processed.fundamentals` | `src/utils/path_resolver.py:33-35` |
| Alpha Vantage API (4 endpoints) | HTTPS `function=INCOME_STATEMENT \| BALANCE_SHEET \| CASH_FLOW \| EARNINGS` | varios — ver mapping abaixo | `alpha_vantage_fundamental_fetcher.py:134-137,153-156,159-173` |
| Existing parquet (coverage) | `data/processed/fundamentals/{ASSET}/fundamentals_{ASSET}.parquet` | `fiscal_date_end` | `src/main_fundamentals.py:50-61` |

**Mapping de campos por endpoint** (`alpha_vantage_fundamental_fetcher.py:107-131`):

| Endpoint | Field externo | Coluna output |
|---|---|---|
| INCOME_STATEMENT | `totalRevenue` | `revenue` |
| INCOME_STATEMENT | `netIncome` | `net_income` |
| BALANCE_SHEET | `totalShareholderEquity` | `total_shareholder_equity` |
| BALANCE_SHEET | `totalLiabilities` | `total_liabilities` |
| CASH_FLOW | `operatingCashflow` | `operating_cash_flow` |
| EARNINGS | `reportedDate` | `reported_date` |
| (todos) | `fiscalDateEnding` | `fiscal_date_end` |

Throttle hard-coded: 12.5 s entre requests (free-tier) — `alpha_vantage_fundamental_fetcher.py:30`.

#### Outputs

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `asset_id` | string | `fundamental_parquet_schema.py:4,19` |
| 2 | `report_type` | string ∈ {`annual`, `quarterly`} | `fundamental_parquet_schema.py:5,20` |
| 3 | `fiscal_date_end` | datetime64[ns, UTC] | `fundamental_parquet_schema.py:6` (cast em `parquet_fundamental_repository.py:115-117`) |
| 4 | `reported_date` | datetime64[ns, UTC] (ou NaT) | `fundamental_parquet_schema.py:7` (cast `:118-121`) |
| 5 | `revenue` | float64 | `fundamental_parquet_schema.py:8,21` |
| 6 | `net_income` | float64 | `fundamental_parquet_schema.py:9,22` |
| 7 | `operating_cash_flow` | float64 | `fundamental_parquet_schema.py:10,23` |
| 8 | `total_shareholder_equity` | float64 | `fundamental_parquet_schema.py:11,24` |
| 9 | `total_liabilities` | float64 | `fundamental_parquet_schema.py:12,25` |
| 10 | `source` | string (literal `"alpha_vantage"`) | `fundamental_parquet_schema.py:13,26` |

Path: `data/processed/fundamentals/{ASSET}/fundamentals_{ASSET}.parquet` — `parquet_fundamental_repository.py:48-54,150`.

#### Calculos

| # | Output col | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | (throttle) | — | `sleep(_MIN_INTERVAL - (now - last_ts))` se positivo | `alpha_vantage_fundamental_fetcher.py:46-56` |
| 2 | `fiscal_date_end`, `reported_date` | strings da API | `datetime.strptime(v, "%Y-%m-%d").date() if v else None` | `alpha_vantage_fundamental_fetcher.py:89-96` |
| 3 | numericos | strings da API | `float(v) if v not in (None, "", "None", "null", "NaN") else None` | `alpha_vantage_fundamental_fetcher.py:98-105` |
| 4 | (merge endpoints) | dicts por endpoint | `dict.setdefault((report_type, fiscal_date_end), {})`; later overwrites earlier (income → balance → cash_flow → earnings) | `alpha_vantage_fundamental_fetcher.py:107-131,153-173` |
| 5 | `source` | (constante) | `"alpha_vantage"` | `alpha_vantage_fundamental_fetcher.py:188` |
| 6 | `asset_id` | input | `asset_id.split(".")[0].upper()` | `parquet_fundamental_repository.py:45-46,91` |
| 7 | (window filter) | `fiscal_date_end, report_type` | `start.date() <= fiscal_date_end <= end.date() AND report_type in {"annual","quarterly"}` | `fetch_fundamentals_use_case.py:60-65` |
| 8 | (dedup+sort) | `report_type, fiscal_date_end` | `drop_duplicates(keep="last").sort_values([type, date])` | `parquet_fundamental_repository.py:141-144` |

**Nao** ha calculo de `effective_date` aqui. O fallback `reported_date = fiscal_date_end + 45d` mora em `dataset_tft` (§3.5).

#### Quality gates aplicados

- Profile `fundamentals` em `data_quality_profiles.py:102-116`: `key_cols=["report_type","fiscal_date_end"]`, `validation_rules=[report_type_in_set]`, `business_days=False`.

#### Lacunas

- §A.32 (path doc vs codigo), §A.33 (config keys ignoradas), §A.34 (throttle/timeout hard-coded), §A.35 (`source` literal — sem provenance multi-version), §A.36 (`asset_id/report_type` persistidos mas o `merge_asof` downstream mistura annual+quarterly), §A.37 (`_is_period_covered` so olha `fiscal_date_end`, ignora `reported_date` faltando).

### 3.5 dataset_tft (consolidacao + feature engineering)

**Entrypoint**: [`src/main_dataset_tft.py`](../../src/main_dataset_tft.py)
**Use case**: [`src/use_cases/build_tft_dataset_use_case.py`](../../src/use_cases/build_tft_dataset_use_case.py)
**Adapter (write)**: [`src/adapters/parquet_tft_dataset_repository.py`](../../src/adapters/parquet_tft_dataset_repository.py)
**Adapters (read)**: candle/technical/sentiment/fundamentals (acima)
**Schemas**: [`src/infrastructure/schemas/tft_dataset_parquet_schema.py`](../../src/infrastructure/schemas/tft_dataset_parquet_schema.py), [`src/infrastructure/schemas/feature_registry.py`](../../src/infrastructure/schemas/feature_registry.py), [`src/infrastructure/schemas/feature_validation_schema.py`](../../src/infrastructure/schemas/feature_validation_schema.py)

**Esta e a celula mais densa do pipeline** — 62 colunas finais, joins de 4 fontes, feature engineering pesado.

#### Inputs (por upstream)

Path resolution: `src/main_dataset_tft.py:117-138` via `src/utils/path_resolver.py:24-37`.

**(a) candles** (base do LEFT JOIN, virando `df`)
- Path: `data/raw/market/candles/{ASSET}/...` — read via `parquet_candle_repository.py:55-57`
- Cols: `timestamp, open, high, low, close, volume` — `build_tft_dataset_use_case.py:69-86`
- Filtro window: `timestamp ∈ [start_utc, end_utc]` — `:477`
- Inject `asset_id` (constante) — `:471`
- Derive `date` (trading day) — `:472-476` (helper, descartada no save em `:569-570`)

**(b) technical_indicators** (LEFT JOIN on `timestamp`)
- Path: `data/processed/technical_indicators/{ASSET}/...`
- Cols indicador (wide): `volatility_20d,rsi_14,candle_body,macd_signal,ema_100,macd,ema_10,ema_200,ema_50,candle_range`
- Join: `df.merge(indicators_df, on="timestamp", how="left")` — `:480`
- Fill: nenhum explicito (warmup gating pelo DatasetQualityGate)

**(c) sentiment_daily** (LEFT JOIN on `date`)
- Path: `data/processed/sentiment_daily/{ASSET}/...` — read via `parquet_daily_sentiment_repository.py:143-197`
- Cols: `day, sentiment_score, n_articles → news_volume, sentiment_std`
- Rename `n_articles → news_volume` — `:99-111`
- Join: `df.merge(sentiment_df, on="date", how="left")` — `:483-484`
- Fill: `news_volume.fillna(0).int64`, `sentiment_score.fillna(0.0).float64`, `sentiment_std.fillna(0.0).float64` — `:489-504`

**(d) fundamentals** (MERGE_ASOF on `date ↔ fundamentals_effective_date`)
- Path: `data/processed/fundamentals/{ASSET}/...` — read via `parquet_fundamental_repository.py` (com `include_latest_before_start=True`)
- Build `effective_date` (renamed → `fundamentals_effective_date`):
  - `reported_date_eff = reported_date or (fiscal_date_end + timedelta(days=45))` — `build_tft_dataset_use_case.py:116-118` (fallback 45d, ver [DATA_SOURCES.md §reported_date](DATA_SOURCES.md))
  - `effective_ts = datetime.combine(reported_date_eff, policy.close_hour, tzinfo=UTC)` — `:124`
  - `effective_day = trading_day_from_timestamp(effective_ts, policy)` — `:125-127`
- Join: `pd.merge_asof(df, fund_df, left_on="date", right_on="fundamentals_effective_date", direction="backward")` — `:518-524`
- Anti-leakage guard: raise se `fundamentals_effective_date > date` — `:525-535` + `:401-416`
- Fill (sem fundamentals): inject as NA — `:536-544`

#### Outputs (schema completo — 62 colunas)

Path: `data/processed/dataset_tft/{ASSET}/dataset_tft_{ASSET}.parquet` — `parquet_tft_dataset_repository.py:49-69`.
Base columns enforced no write: `asset_id, timestamp, time_idx, day_of_week, month, target_return` — `tft_dataset_parquet_schema.py:3-18`.

| # | Coluna | Dtype | Origem | Ref |
|---|---|---|---|---|
| 1 | `asset_id` | string | constante | `build_tft_dataset_use_case.py:471` |
| 2 | `timestamp` | datetime64[ns, UTC] | candle | `:84,473` |
| 3 | `open` | float64 (era float32 na fonte — §A.38) | candle | `:78` |
| 4 | `high` | float64 | candle | `:78` |
| 5 | `low` | float64 | candle | `:78` |
| 6 | `close` | float64 | candle | `:78` |
| 7 | `volume` | float64 (era int64 na fonte) | candle | `:80` |
| 8 | `volatility_20d` | float32 | technical | `:480`; registry `feature_registry.py:58-65` |
| 9 | `rsi_14` | float32 | technical | `feature_registry.py:66-73` |
| 10 | `candle_body` | float32 | technical | `feature_registry.py:74-80` |
| 11 | `macd_signal` | float32 | technical | `feature_registry.py:81-88` |
| 12 | `ema_100` | float32 | technical | `feature_registry.py:89-96` |
| 13 | `macd` | float32 | technical | `feature_registry.py:97-104` |
| 14 | `ema_10` | float32 | technical | `feature_registry.py:105-112` |
| 15 | `ema_200` | float32 | technical | `feature_registry.py:113-120` |
| 16 | `ema_50` | float32 | technical | `feature_registry.py:121-128` |
| 17 | `candle_range` | float32 | technical | `feature_registry.py:129-135` |
| 18 | `sentiment_score` | float64 | sentiment | `build_tft_dataset_use_case.py:495-499` |
| 19 | `news_volume` | int64 | sentiment (rename) | `:489-493` |
| 20 | `sentiment_std` | float64 | sentiment (era nullable Float64 — §A.39) | `:500-504` |
| 21 | `has_news` | int64 | derived | `:494` |
| 22 | `log_return_1d` | float64 | derived | `:154` |
| 23 | `log_return_5d` | float64 | derived | `:155` |
| 24 | `log_return_21d` | float64 | derived | `:156` |
| 25 | `momentum_5d` | float64 | derived | `:158` |
| 26 | `momentum_21d` | float64 | derived | `:159` |
| 27 | `momentum_63d` | float64 | derived | `:160` |
| 28 | `reversal_1d` | float64 | derived | `:161` |
| 29 | `reversal_5d` | float64 | derived | `:162` |
| 30 | `drawdown_lookback` | float64 | derived | `:163-164` |
| 31 | `amihud_illiquidity_proxy` | float64 | derived | `:166-167` |
| 32 | `volume_zscore` | float64 | derived | `:169-175` |
| 33 | `volume_spike_flag` | int64 | derived | `:176` |
| 34 | `volatility_parkinson` | float64 | derived | `:178-185` |
| 35 | `volatility_garman_klass` | float64 | derived | `:187-194` |
| 36 | `downside_semivolatility` | float64 | derived | `:199-202` |
| 37 | `vol_of_vol` | float64 | derived | `:204-208` |
| 38 | `volatility_regime` | float64 (∈ {0,1,2,NaN}) | derived | `:211-218` |
| 39 | `trend_regime` | float64 (∈ {-1,0,+1,NaN}) | derived | `:220-229` |
| 40 | `stress_tail_return_flag` | float64 (∈ {0,1,NaN}) | derived | `:231-236` |
| 41 | `sentiment_lag_1` | float64 | derived | `:246` |
| 42 | `sentiment_lag_3` | float64 | derived | `:247` |
| 43 | `sentiment_lag_5` | float64 | derived | `:248` |
| 44 | `sentiment_ema` | float64 | derived | `:249` |
| 45 | `sentiment_surprise` | float64 | derived | `:250-251` |
| 46 | `sentiment_x_volatility` | float64 | derived | `:252` |
| 47 | `sentiment_x_volume` | float64 | derived | `:253` |
| 48 | `fundamentals_effective_date` | datetime64[ns, UTC] | as-of join | `:515-524` (dtype nao em `TFT_DATASET_DTYPES` — §A.40) |
| 49 | `revenue` | float64 (nullable) | fundamentals as-of | `:131` |
| 50 | `net_income` | float64 (nullable) | fundamentals as-of | `:132` |
| 51 | `operating_cash_flow` | float64 (nullable) | fundamentals as-of | `:133` |
| 52 | `total_shareholder_equity` | float64 (nullable) | fundamentals as-of | `:134` |
| 53 | `total_liabilities` | float64 (nullable) | fundamentals as-of | `:135` |
| 54 | `net_margin` | float64 | derived | `:279` |
| 55 | `leverage_ratio` | float64 | derived | `:280` |
| 56 | `cashflow_efficiency` | float64 | derived | `:281` |
| 57 | `revenue_yoy_growth` | float64 | derived | `:283` |
| 58 | `net_income_yoy_growth` | float64 | derived | `:284` |
| 59 | `day_of_week` | int64 | derived | `:553` |
| 60 | `month` | int64 | derived | `:554` |
| 61 | `target_return` | float64 | derived (TARGET) | `:563` |
| 62 | `time_idx` | int64 | derived | `:573` |

#### Calculos (derived features)

| # | Output col | Inputs (cols) | Formula | Ref |
|---|---|---|---|---|
| 1 | `date` (helper) | `timestamp` | `trading_day_from_timestamp(ts, policy)` | `build_tft_dataset_use_case.py:472-476` |
| 2 | `fundamentals_effective_date` | `reported_date, fiscal_date_end, close_hour` | `trading_day_from_timestamp(combine(reported_date or fiscal_date_end+45d, close_hour, UTC), policy)` | `:115-127` |
| 3 | `has_news` | `news_volume` | `(news_volume > 0).astype(int64)` | `:494` |
| 4 | `log_return_1d` | `close` | `log(close / close.shift(1))` | `:154` |
| 5 | `log_return_5d` | `close` | `log(close / close.shift(5))` | `:155` |
| 6 | `log_return_21d` | `close` | `log(close / close.shift(21))` | `:156` |
| 7 | `momentum_5d` | `close` | `close / close.shift(5) - 1` | `:158` |
| 8 | `momentum_21d` | `close` | `close / close.shift(21) - 1` | `:159` |
| 9 | `momentum_63d` | `close` | `close / close.shift(63) - 1` | `:160` |
| 10 | `reversal_1d` | `close` | `-close.pct_change(1)` | `:161` |
| 11 | `reversal_5d` | `momentum_5d` | `-momentum_5d` | `:162` |
| 12 | `drawdown_lookback` | `close` | `close / rolling_max(close, 63, min_periods=63) - 1` | `:163-164` |
| 13 | `amihud_illiquidity_proxy` | `close, volume` | `where(volume>0, abs(pct_change(close)) / volume, NaN)` | `:166-167` |
| 14 | `volume_zscore` | `volume` | `(volume - mean(volume.shift(1), 20)) / std(volume.shift(1), 20)` (ddof=0); NaN se std<=0 | `:169-175` |
| 15 | `volume_spike_flag` | `volume_zscore` | `int(volume_zscore > 3.0)` | `:176` |
| 16 | `volatility_parkinson` | `high, low` | `sqrt(rolling_mean(ln(high/low)^2, 20) / (4·ln2))`, clip min 0 | `:178-185` |
| 17 | `volatility_garman_klass` | `high, low, close, open` | `sqrt(rolling_mean(0.5·ln(high/low)^2 - (2·ln2-1)·ln(close/open)^2, 20))`, clip min 0 | `:187-194` |
| 18 | `downside_semivolatility` | `close` | `sqrt(rolling_mean(min(pct_change(close), 0)^2, 20))`, clip min 0 | `:199-202` |
| 19 | `vol_of_vol` | `volatility_20d` (ou proxy) | `rolling_std(volatility_20d, 20, ddof=0)` (proxy = `pct_change(close).rolling(20).std()` se faltar) | `:204-208` |
| 20 | `volatility_regime` | `volatility_20d` | bucket tercil contra trailing q33/q66 em 63d shifted by 1 → {0,1,2,NaN} | `:211-218` |
| 21 | `trend_regime` | `ema_10, ema_50` | `spread = ema_10 - ema_50`; `deadband = 0.10 · rolling_std(spread.shift(1), 63, ddof=0)`; `+1` if `spread>deadband`, `-1` if `spread<-deadband`, `0` else | `:220-229` |
| 22 | `stress_tail_return_flag` | `close` | `int(pct_change(close,1) <= rolling_quantile(pct_change(close,1).shift(1), 63, 0.10))` | `:231-236` |
| 23 | `sentiment_lag_1/3/5` | `sentiment_score` | `sentiment_score.shift(N)` for N ∈ {1,3,5} | `:246-248` |
| 24 | `sentiment_ema` | `sentiment_score` | `sentiment_score.ewm(span=10, adjust=False).mean()` | `:249` |
| 25 | `sentiment_surprise` | `sentiment_score` | `sentiment_score - sentiment_score.shift(1).rolling(5, min_periods=5).mean()` | `:250-251` |
| 26 | `sentiment_x_volatility` | `sentiment_score, volatility_20d` | produto | `:252` |
| 27 | `sentiment_x_volume` | `sentiment_score, volume` | produto | `:253` |
| 28 | `net_margin` | `net_income, revenue` | `_safe_ratio(net_income, revenue)` (NaN se denom 0/NaN) | `:279`; helper `:257-263` |
| 29 | `leverage_ratio` | `total_liabilities, total_shareholder_equity` | `_safe_ratio(total_liabilities, total_shareholder_equity)` | `:280` |
| 30 | `cashflow_efficiency` | `operating_cash_flow, revenue` | `_safe_ratio(operating_cash_flow, revenue)` | `:281` |
| 31 | `revenue_yoy_growth` | `revenue` (daily as-of) | `revenue.pct_change(252, fill_method=None)` | `:283` |
| 32 | `net_income_yoy_growth` | `net_income` (daily as-of) | `net_income.pct_change(252, fill_method=None)` | `:284` |
| 33 | `day_of_week` | `timestamp` | `timestamp.dt.dayofweek.astype(int64)` | `:553` |
| 34 | `month` | `timestamp` | `timestamp.dt.month.astype(int64)` | `:554` |
| 35 | **`target_return`** (TARGET) | `close` | `log(close.shift(-1) / close)` → log return do proximo dia; ultima row dropada via `dropna(subset=["target_return"])` | `:563-566` |
| 36 | `time_idx` | (linha) | `range(len(df))` apos sort por `timestamp` e drop do target NaN | `:557,573` |

Anti-leakage validators (mesmo run): `:287-416` — checa `candle_range=high-low`, `candle_body=|close-open|`, `has_news`, `volume_spike_flag`, volatilidades nao-negativas, valores discretos de regime, sentiment lags shift correto, sentiment_x_volume, ratios fundamentalistas, e fundamentals as-of (`fundamentals_effective_date <= date`).

#### Quality gates aplicados

- DatasetQualityGate: `:588-594`, config em `config/quality/dataset_quality.yaml`, warmups em `feature_validation_schema.py:11-15`.
- DQ report com extra section `quality_gate` — `src/main_dataset_tft.py:209-216`.

#### Lacunas

- §A.38 (dtype OHLCV muda no join), §A.39 (`sentiment_std` perde nullable), §A.40 (`fundamentals_effective_date` sem dtype no `TFT_DATASET_DTYPES`), §A.41 (`tft_dataset_schema.py` dead code — `sector/market_cap/rsi`), §A.42 (`sentiment_ema/surprise` sobre `sentiment_score=0.0` imputado), §A.43 (sem `revenue/net_income_yoy_growth` no anti-leakage validator), §A.44 (DQ quality gate inclui `fundamentals_effective_date` no `_model_feature_columns` — pode falhar nan-ratio), §A.45 (`fundamentals` as-of mistura annual+quarterly).

### 3.6 inference

**Entrypoint**: [`src/main_infer_tft.py`](../../src/main_infer_tft.py)
**Use case**: [`src/use_cases/run_tft_inference_use_case.py`](../../src/use_cases/run_tft_inference_use_case.py)
**Adapters**: [`src/adapters/parquet_tft_inference_repository.py`](../../src/adapters/parquet_tft_inference_repository.py), [`src/adapters/local_tft_inference_model_loader.py`](../../src/adapters/local_tft_inference_model_loader.py), [`src/adapters/pytorch_forecasting_tft_inference_engine.py`](../../src/adapters/pytorch_forecasting_tft_inference_engine.py)
**Schema (parquet silver)**: [`src/infrastructure/schemas/tft_inference_parquet_schema.py`](../../src/infrastructure/schemas/tft_inference_parquet_schema.py)
**Schema (analytics silver)**: [`src/infrastructure/schemas/analytics_store_schema.py`](../../src/infrastructure/schemas/analytics_store_schema.py)

#### Inputs

| Nome | Caminho/Fonte | Coluna(s) consumidas | Ref |
|---|---|---|---|
| dataset_tft parquet | `data/processed/dataset_tft/{ASSET}/dataset_tft_{ASSET}.parquet` | obrigatorias: `{timestamp, time_idx, asset_id, target_return}` + features do modelo (`model_bundle.feature_cols`) | `run_tft_inference_use_case.py:415,507-510` |
| model artifact dir | `models/{ASSET}/runs/{VERSION}` | `metadata.json, config.json, features.json, checkpoints/best.ckpt, scalers.pkl, dataset_parameters.pkl` | `local_tft_inference_model_loader.py:88-232` |
| CLI/JSON | argv | `asset, model_path, start, end, overwrite, batch_size, dataset_dir, inference_dir, auto_refresh, allow_missing_quantiles, inference_mode` | `src/main_infer_tft.py:58-137` |
| `paths["analytics_silver"]` | `config/data_paths.yaml` | — | `src/main_infer_tft.py:316` |

#### Outputs

**Output 1: silver parquet** `data/processed/inference_tft/{ASSET}/inference_tft_{ASSET}.parquet` (logical PK `(asset_id, timestamp, model_version)`)

| # | Coluna | Dtype | Ref schema |
|---|---|---|---|
| 1 | `asset_id` | string | `tft_inference_parquet_schema.py:3,19` |
| 2 | `timestamp` (= `target_timestamp` — §A.46) | datetime64[ns, UTC] | `:4` (cast em `parquet_tft_inference_repository.py:151`) |
| 3 | `model_version` | string | `:5,20` |
| 4 | `model_path` | string (canonica project-relative) | `:6,21` |
| 5 | `feature_set_name` | string | `:7,22` |
| 6 | `features_used_csv` | string | `:8,23` |
| 7 | `prediction` | float64 | `:9,24` |
| 8 | `quantile_p10` | float64 nullable | `:10,25` |
| 9 | `quantile_p50` | float64 nullable | `:11,26` |
| 10 | `quantile_p90` | float64 nullable | `:12,27` |
| 11 | `inference_run_id` | string | `:13,28` |
| 12 | `created_at` | datetime64[ns, UTC] | `:14` (cast `:152`) |

**Output 2: analytics silver tabelas** — ver §5 (`fact_inference_runs`, `fact_inference_predictions`, `fact_feature_contrib_local`).

#### Calculos

| # | Output col | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | `asset` (normalizado) | input | `asset_id.split(".")[0].upper()` | `run_tft_inference_use_case.py:246-248` |
| 2 | window | input | `end_utc = end_date or now(UTC)`; `start_utc = start_date or (latest_existing + 1d)` | `:274-303,337` |
| 3 | `inference_run_id` | clock | `datetime.now(UTC).strftime("%Y%m%d_%H%M%S")` | `:535` |
| 4 | eligible idx | `time_idx` | `idx >= max_encoder_length AND time_idx contiguous` | `:322-358,473-490` |
| 5 | slice + scale | feature cols | `iloc[first_target - max_encoder_length : last_target + 1]`; `scaler.transform` per col (skip `{time_idx,timestamp,asset_id,target_return}`) | `:250-272,532-533` |
| 6 | TFT predict | scaled slice + model | `TimeSeriesDataSet.from_parameters(...).predict(mode="prediction")` + `mode="quantiles"` (fallback `raw` → forward+`to_quantiles`) | `pytorch_forecasting_tft_inference_engine.py:179-350` |
| 7 | `target_timestamp` | `time_idx`, `timestamp_by_time_idx` map | `target_timestamp = timestamp[time_idx]` | `pytorch_forecasting_tft_inference_engine.py:352-371` |
| 8 | `decision_timestamp` | idem | `decision_timestamp = timestamp[time_idx - 1]` | `:364,372` |
| 9 | `prediction` (point) | quantile tensor | `_to_1d_first_step(arr)` — se 3D pega quantile mediana; depois decoder step 0 | `:248,419-425` |
| 10 | `quantile_p10/p50/p90` | model quantile output | nearest(0.1)/nearest(0.5)/nearest(0.9) de `loss.quantiles` (default `[0.1,0.5,0.9]`); decoder step 0 | `:260-306,427-545` |
| 11 | `horizon` | (constante) | literal `1` (§A.47 — descarta horizontes >1 mesmo se modelo treinado com `max_prediction_length>1`) | `:373`; downstream coerce em `run_tft_inference_use_case.py:118,193` |
| 12 | filter target_ts | input window | `keep iff to_utc(target_timestamp).isoformat() in requested_target_timestamps` | `:552-556` |
| 13 | inference_mode `last_point` | targets | `keep iff target_timestamp == max(target_timestamps)` | `:558-566` |
| 14 | dedup vs existing | `(asset, timestamp, model_version)` | quando `overwrite=False`: drop matches; conta `skipped_existing` | `:613-629` |
| 15 | quantile guardrail | `(q10, q50, q90)` | sort triple; `applied=True` se mudou ordem; se algum NaN/non-finite: `applied=False`, mantem | `quantile_guardrail_service.py:18-60`; aplicado em `:122` |
| 16 | upsert silver | output rows | `drop_duplicates(["asset_id","timestamp","model_version"], keep="last")` + sort | `parquet_tft_inference_repository.py:214-228` |

#### Quality gates aplicados

- Strict quantile validation: se `prediction_mode=="quantile"` e `strict_quantiles=True`, qualquer record com p10/p50/p90 None raise — `:595-611`.
- Feature compatibility: `missing = model_feature_cols \ dataset.columns`; `excess = dataset.columns ∩ IMPLEMENTED_FEATURES − expected_set` — `:305-320,512-530`.
- Invariante `model_version ↔ single model_path` — `parquet_tft_inference_repository.py:158-210`.

#### Lacunas

- §A.46 (silver parquet `timestamp` e ambiguamente `target_timestamp`; nao tem `decision_timestamp` nem `horizon` nem `guardrail_post`), §A.47 (`horizon=1` hard-coded), §A.48 (`inference_run_id` so com precisao de segundo — possivel colisao), §A.49 (`decision_idx` nao populado em `fact_inference_predictions` apesar de declarado), §A.50 (refresh dispara nova build sem `trading_day_policy`/`quality_gate_config` — `effective_date` pode divergir).

## 4. Stage: silver (analytics)

Conceitos: ver [`docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](../01_architecture/ANALYTICS_STORE_ARCHITECTURE.md).
Schemas: [`src/infrastructure/schemas/analytics_store_schema.py`](../../src/infrastructure/schemas/analytics_store_schema.py).
Writer: [`src/adapters/parquet_analytics_run_repository.py`](../../src/adapters/parquet_analytics_run_repository.py).

**Producers**:
- Treino: [`src/use_cases/train_tft_model_use_case.py`](../../src/use_cases/train_tft_model_use_case.py) — emite `dim_run`, `fact_run_snapshot`, `fact_config`, `fact_epoch_metrics`, `fact_split_metrics`, `fact_oos_predictions`, `fact_model_artifacts`, `fact_failures`.
- Inferencia: [`src/use_cases/run_tft_inference_use_case.py`](../../src/use_cases/run_tft_inference_use_case.py) — emite `dim_run`, `fact_run_snapshot`, `fact_config`, `fact_inference_runs`, `fact_inference_predictions`, `fact_feature_contrib_local`.
- Baselines: [`src/use_cases/run_baselines_use_case.py`](../../src/use_cases/run_baselines_use_case.py) — emite `dim_run`, `fact_run_snapshot`, `fact_config`, `fact_oos_predictions` (com `model_version="baseline:*"`).
  - **Detalhe completo dos calculos dos baselines: ver §5 (2a passada).**

### 4.1 dim_run

- **Grain**: 1 linha por `run_id` (hash deterministico de `(asset, feature_set_hash, trial_number, fold, seed, model_version, config_signature, split_signature, pipeline_version)`)
- **Schema**: `analytics_store_schema.py:154-202` (`partition_by=("asset", "parent_sweep_id")`)
- **Writer**: `parquet_analytics_run_repository.py:170-188` (`upsert_dim_run`)
- **PK helper**: `compute_run_id(...)` em `analytics_store_schema.py:101-124`

#### Outputs

| Coluna | Dtype |
|---|---|
| `schema_version` | int64 |
| `run_id` | string |
| `execution_id` | string |
| `parent_sweep_id` | string |
| `trial_number` | int64 |
| `fold` | string |
| `seed` | int64 |
| `asset` | string |
| `feature_set_name` | string |
| `feature_set_hash` | string |
| `feature_list_ordered_json` | string |
| `config_signature` | string |
| `split_fingerprint` | string |
| `model_version` | string |
| `checkpoint_path_final` | string |
| `checkpoint_path_best` | string |
| `git_commit` | string |
| `pipeline_version` | string |
| `library_versions_json` | string |
| `hardware_info_json` | string |
| `status` | string ∈ `{ok, failed, partial_failed}` |
| `duration_total_seconds` | float64 |
| `eta_recorded_seconds` | float64 |
| `retries` | int64 |
| `created_at_utc` | string |

#### Inputs (no producer de treino)

- Asset/version/artifacts_dir/trainer_config (com `parent_sweep_id`, `trial_number`, `fold`, `seed`, `pipeline_version`, `duration_total_seconds`, `eta_recorded_seconds`, `retries`, `feature_set_name`) — `train_tft_model_use_case.py:540-602`
- `feature_cols = _resolve_features(...)` — `:1186`
- `feature_set_hash = compute_feature_set_hash(feature_cols)` — `analytics_store_schema.py:33-34`, invocado `:1187`
- `config_signature = compute_config_signature(trainer_config)` — `analytics_store_schema.py:37-41`, invocado `:1251`
- `split_signature = compute_split_fingerprint(...)` — `analytics_store_schema.py:44-55`, invocado `:1246-1250`
- `git_commit` via `_collect_git_commit` — `:501-506`
- `library_versions_json` via `_collect_library_versions` — `:509-517`
- `hardware_info_json` via `_collect_hardware_info` — `:519-536`
- `status` derivado por `derive_run_status` — `analytics_store_schema.py:127-140`

#### Quality gates aplicados

- Schema-level: `required_columns` + `status` whitelist — `analytics_store_schema.py:747-750`, gate `validate_table_payload` em `:713-772`
- `run_id_execution_consistency` — `validate_analytics_quality_use_case.py:382-399`
- `referential_integrity` (outros facts referenciam `dim_run.run_id`) — `:401-430`
- `required_tables_presence` — `:326-342`
- `cardinality_config_fold_seed` — `:775-784`
- `baselines_share_parent_sweep_id_with_candidates` — `:1017-1056`
- `tft_baselines_timestamp_subset_alignment` — `:92-170, 1058-1074`
- `official_contract_quantile_attention` — `:1076-1145`

### 4.2 fact_run_snapshot

- **Grain**: 1 linha por `run_id` (snapshot do dataset/splits)
- **Schema**: `analytics_store_schema.py:204-252` (`partition_by=("asset","parent_sweep_id")`, append-only)
- **Writer**: `parquet_analytics_run_repository.py:190-208`

#### Outputs

| Coluna | Dtype |
|---|---|
| `schema_version` | int64 |
| `run_id` | string |
| `asset` | string |
| `parent_sweep_id` | string |
| `dataset_start_utc` | string |
| `dataset_end_utc` | string |
| `train_start_utc` | string |
| `train_end_utc` | string |
| `val_start_utc` | string |
| `val_end_utc` | string |
| `test_start_utc` | string |
| `test_end_utc` | string |
| `warmup_policy` | string |
| `required_warmup_count` | int64 |
| `warmup_applied` | string |
| `effective_train_start_utc` | string |
| `n_samples_train` | int64 |
| `n_samples_val` | int64 |
| `n_samples_test` | int64 |
| `dataset_fingerprint` | string |
| `split_fingerprint` | string |

#### Inputs

- `df, train_df, val_df, test_df` (cada um com `timestamp`, opcional `close, volume`) — `train_tft_model_use_case.py:625-639`
- `dataset_fingerprint = compute_dataset_fingerprint(...)` — `analytics_store_schema.py:58-77`, invocado `:631-639`
- `warmup_meta` de `_apply_warmup_policy_to_split` — `:354-436`

#### Quality gates

- `min_samples_by_split` (`n_samples_train/val/test`) — `validate_analytics_quality_use_case.py:786-805`
- `referential_integrity` — `:401-430`

### 4.3 fact_config

- **Grain**: 1 linha por `run_id`
- **Schema**: `analytics_store_schema.py:254-306`
- **Writer**: `parquet_analytics_run_repository.py:210-228`

#### Outputs (colunas)

`schema_version, run_id, asset, parent_sweep_id, prediction_mode, loss_name, quantile_levels_json, evaluation_horizons_json, max_encoder_length, max_prediction_length, batch_size, max_epochs, learning_rate, hidden_size, attention_head_size, dropout, hidden_continuous_size, early_stopping_patience, early_stopping_min_delta, scaler_type, training_config_json, dataset_parameters_json, search_space_json, objective_name, objective_direction` — dtypes em `:257-282`.

#### Inputs / calculos

- `trainer_config` direto — `train_tft_model_use_case.py:1012-1043`
- `loss_name, quantile_levels` extraidos via `_extract_loss_metadata(training_result.model)` — `:712-729`
- `scaler_type` via `_infer_scaler_type(dataset_parameters)` — `:731-743`

#### Quality gates

- `prediction_mode` whitelist `{point, quantile}` — `analytics_store_schema.py:752-755`
- `oos_horizon_coverage` consome `evaluation_horizons_json` — `validate_analytics_quality_use_case.py:562-598`

### 4.4 fact_epoch_metrics

- **Grain**: 1 linha por `(run_id, epoch)`
- **Schema**: `analytics_store_schema.py:308-337` (`partition_by=("asset","sweep_id","fold")`, append-only)
- **Writer**: `parquet_analytics_run_repository.py:268-285`

#### Outputs

| Coluna | Dtype |
|---|---|
| `schema_version` | int64 |
| `run_id` | string |
| `asset` | string |
| `parent_sweep_id` | string |
| `fold` | string |
| `epoch` | int64 |
| `train_loss` | float64 |
| `val_loss` | float64 |
| `epoch_time_seconds` | float64 |
| `best_epoch` | int64 |
| `stopped_epoch` | int64 |
| `early_stop_reason` | string |

#### Inputs

- `training_result.history: list[dict]` (`epoch, train_loss, val_loss, epoch_time_seconds, best_epoch, stopped_epoch, early_stop_reason`) — `train_tft_model_use_case.py:899-924`

#### Quality gates

- `required_metrics_nan` em `train_loss, val_loss` — `validate_analytics_quality_use_case.py:432-448`

### 4.5 fact_split_metrics

- **Grain**: 1 linha por `(run_id, split)`
- **Schema**: `analytics_store_schema.py:339-367`
- **Writer**: `parquet_analytics_run_repository.py:230-247`

#### Outputs

| Coluna | Dtype |
|---|---|
| `schema_version` | int64 |
| `run_id` | string |
| `asset` | string |
| `parent_sweep_id` | string |
| `split` | string |
| `rmse` | float64 |
| `mae` | float64 |
| `mape` | float64 |
| `smape` | float64 |
| `directional_accuracy` | float64 |
| `n_samples` | int64 |

#### Inputs / calculos

- `training_result.split_metrics: dict[split → dict[metric → float]]` — `train_tft_model_use_case.py:947-967`
- `split_counts` — `:1383-1387`

#### Quality gates

- `required_metrics_nan` em `rmse, mae, directional_accuracy, n_samples` — `validate_analytics_quality_use_case.py:432-448`

### 4.6 fact_oos_predictions

- **Grain**: 1 linha por `(run_id, split, horizon, timestamp_utc, target_timestamp_utc)`
- **Schema**: `analytics_store_schema.py:369-426` (`partition_by=("asset","feature_set_name","year")`, append-only)
- **Writer**: `parquet_analytics_run_repository.py:287-304`
- **Persister**: `src/domain/services/multi_horizon_prediction_persister.py`

#### Outputs

| # | Coluna | Dtype |
|---|---|---|
| 1 | `schema_version` | int64 |
| 2 | `run_id` | string |
| 3 | `model_version` | string |
| 4 | `asset` | string |
| 5 | `feature_set_name` | string |
| 6 | `split` | string |
| 7 | `fold` | string |
| 8 | `seed` | int64 |
| 9 | `horizon` | int64 |
| 10 | `decision_idx` | int64 |
| 11 | `timestamp_utc` | string |
| 12 | `target_timestamp_utc` | string |
| 13 | `y_true` | float64 |
| 14 | `y_pred` | float64 |
| 15 | `error` | float64 |
| 16 | `abs_error` | float64 |
| 17 | `sq_error` | float64 |
| 18 | `quantile_p10` | float64 |
| 19 | `quantile_p50` | float64 |
| 20 | `quantile_p90` | float64 |
| 21 | `quantile_p10_post_guardrail` | float64 |
| 22 | `quantile_p50_post_guardrail` | float64 |
| 23 | `quantile_p90_post_guardrail` | float64 |
| 24 | `quantile_guardrail_applied` | int64 |
| 25 | `year` | int64 |

#### Inputs / calculos

- `split_frames: dict[split → DataFrame]` com `timestamp` — `train_tft_model_use_case.py:771-777`
- `split_predictions[split]` com matrizes (`y_true_matrix, y_pred_matrix, quantile_p10_matrix, quantile_p50_matrix, quantile_p90_matrix, horizons`) — `:791-797`
- Anchor convention: `decision_idx = max(max_encoder_length - 1, 0) + i` — `:768`
- `target_pos = decision_idx + h`; raise `IncompletePredictionWindowError` se `target_pos >= len(timestamps)` — `multi_horizon_prediction_persister.py:96-105`
- `error = y_pred - y_true` — `:108,125`
- `abs_error = abs(error)` — `:126`
- `sq_error = error * error` — `:127`
- `year = int(target_ts.year)` — `:135`
- Guardrail: `QuantileGuardrailService.enforce_monotonic_triplet(q10, q50, q90)` — `quantile_guardrail_service.py:19-60`; gera as 4 colunas `*_post_guardrail` e `quantile_guardrail_applied`

#### Quality gates

- `temporal_consistency` (parse, `target>=decision`, monotonic por grupo) — `validate_analytics_quality_use_case.py:451-490,639-644`
- `oos_unique_key` (sem dups em PK) — `:492-506,645-650`
- `oos_numeric_types` — `:508-527,651-656`
- `oos_horizon_coverage` (esperado de `fact_config.evaluation_horizons_json`) — `:561-598,657-662`
- `oos_supervised_nulls` — `:529-539,663-668`
- `oos_interval_width_non_negative` (`q90 - q10 >= 0`) — `:541-551,669-674`
- `oos_quantile_order` (`q10 <= q50 <= q90`) — `:553-559,675-680`
- `oos_pairwise_target_alignment` (TFT vs baseline em mesmo `target_timestamp`) — `:600-637,714-719`
- `oos_quantile_block_a_acceptance` via `QuantileContractAnalyzer.evaluate_block_a` — `:682-699`; thresholds em `quantile_contract_analyzer.py:9-15` (`max_crossing_bruto_rate=0.001, max_negative_interval_width_count=0, max_crossing_post_guardrail_rate=0.0`)
- `block_quantile_degeneracy_gate` — `:700-713`; metric `p10_eq_p90_rate = count / n_rows`; falha se `mode=='quantile' AND n_rows>=min AND p10_eq_p90_rate >= max` — `quantile_contract_analyzer.py:292,330,367-369`

### 4.7 fact_failures

- **Grain**: 1 linha por `(run_id, failed_at_utc, stage)`
- **Schema**: `analytics_store_schema.py:428-459`
- **Writer**: `parquet_analytics_run_repository.py:403-421`

#### Outputs

`schema_version, run_id, execution_id, asset, failed_at_utc, stage, error_type, error_message, trace_hash, traceback_excerpt, entrypoint, cmdline, stdout_truncated, stderr_truncated` — dtypes em `analytics_store_schema.py:431-446`.

#### Inputs / calculos

- `Exception + traceback.format_exc()` (trunc 4000 chars) — `train_tft_model_use_case.py:1131-1149`
- `trace_hash = sha256(tb_text.encode("utf-8")).hexdigest()` — `:1144`

### 4.8 fact_model_artifacts

- **Grain**: 1 linha por `run_id`
- **Schema**: `analytics_store_schema.py:461-492`
- **Writer**: `parquet_analytics_run_repository.py:306-324`

#### Outputs

`schema_version, run_id, training_run_id, asset, model_version, checkpoint_path_final, checkpoint_path_best, config_path, scaler_path, encoder_path, feature_importance_json, attention_summary_json, logs_ref_json` — dtypes em `:464-479`.

#### Inputs

- `artifacts_dir` tree (`model_state.pt, checkpoints/best.ckpt, config.json, scalers.pkl, dataset_parameters.pkl, history.csv, metrics.json, split_metrics.json, metadata.json, plots/loss_curve.png`) — `train_tft_model_use_case.py:1060-1083`
- `training_result.feature_importance` (list of dicts) — `:1071,1113`
- Escreve `training_run_id` no `metadata.json` — `:1085-1092`

#### Quality gates

- `official_contract_quantile_attention` requer `feature_importance_json, attention_summary_json` nao-vazios em runs candidatas oficiais — `validate_analytics_quality_use_case.py:1076-1145`

### 4.9 fact_inference_runs

- **Grain**: 1 linha por `inference_run_id`
- **Schema**: `analytics_store_schema.py:494-528`
- **Writer**: `parquet_analytics_run_repository.py:345-363`

#### Outputs

`schema_version, run_id, training_run_id, inference_run_id, model_version, asset, inference_start_utc, inference_end_utc, overwrite, batch_size, status, inferred_count, skipped_count, upserts_count, duration_seconds` — dtypes em `:497-515`.

#### Quality gates

- `inference_predictions_continuity` — `validate_analytics_quality_use_case.py:344-361`

### 4.10 fact_inference_predictions

- **Grain**: 1 linha por `(inference_run_id, horizon, timestamp_utc, target_timestamp_utc)`
- **Schema**: `analytics_store_schema.py:530-586` (`partition_by=("asset","model_version","year")`, append-only)
- **Writer**: `parquet_analytics_run_repository.py:365-382`

#### Outputs

`schema_version, inference_run_id, run_id, training_run_id, model_version, asset, feature_set_name, features_used_csv, model_path, split, horizon, decision_idx, timestamp_utc, target_timestamp_utc, y_true, y_pred, error, abs_error, sq_error, quantile_p10, quantile_p50, quantile_p90, quantile_p10_post_guardrail, quantile_p50_post_guardrail, quantile_p90_post_guardrail, quantile_guardrail_applied, year, created_at_utc` — dtypes em `:533-565`.

#### Inputs / calculos

- TFT inference output consumido por `_emit_oos_rows` em `run_tft_inference_use_case.py:191`
- Mapping: `split="inference"`, `horizon = int(getattr(r, "horizon", 1) or 1)`, `y_true=None`, `error=None`, `abs_error=None`, `sq_error=None`
- `decision_idx` **NAO POPULADO** apesar de declarado no schema — §A.49

### 4.11 fact_feature_contrib_local

- **Grain**: 1 linha por `(inference_run_id, horizon, timestamp_utc, target_timestamp_utc, feature_name)`
- **Schema**: `analytics_store_schema.py:588-635`
- **Writer**: `parquet_analytics_run_repository.py` (anexar `append_fact_feature_contrib_local`)

#### Outputs

`schema_version, inference_run_id, run_id, training_run_id, model_version, asset, feature_set_name, split, horizon, timestamp_utc, target_timestamp_utc, feature_name, feature_rank, contribution, abs_contribution, contribution_sign, method, year, created_at_utc`.

#### Calculos (top-k local feature contribution)

- Para cada record `r`, lookup row do dataset_tft em `target_timestamp`
- Para cada feature `f` com valor `v`:
  - `w = |v| / sum(|v_f|)` (denom default 1 ou contagem se sum=0)
  - `contribution = prediction * w * sign(v)`
- Sort desc por `|contribution|`; top_k = `min(5, max(1, len(feature_cols)))`; rank desde 1
- `method = "local_magnitude_signed_v1"`
- Ref: `run_tft_inference_use_case.py:160-244,681-692`

### 4.12 silver.baselines

**A ser completado na 2a passada** — ver §5.

## 5. Baselines (silver writers)

Baselines sao modelos estatisticos/ingenuos cujas predicoes sao escritas em `fact_oos_predictions` (silver) para comparacao pareada com o TFT.

**Entrypoints**:
- Sweep JSON (canonico): [`src/main_baselines_test_pipeline.py`](../../src/main_baselines_test_pipeline.py)
- Ad-hoc debug: [`src/main_run_baselines.py`](../../src/main_run_baselines.py)
- (Nota: `main_rebuild_explicit_sweep_predictions.py` e TFT-only — hard-asserta `test_type=="explicit_configs"`, nao toca baselines.)

**Use cases**: [`src/use_cases/run_baselines_test_pipeline_use_case.py`](../../src/use_cases/run_baselines_test_pipeline_use_case.py) (orquestracao fold x seed x baseline), [`src/use_cases/run_baselines_use_case.py`](../../src/use_cases/run_baselines_use_case.py) (per-baseline persistence)

### 5.0 Orquestracao e selecao

**Quais baselines rodam**:
- Sweep JSON: `config['baselines']` e `list[{name, config}]`; validado em `_extract_baseline_specs` — `run_baselines_test_pipeline_use_case.py:97-120`. Ausente → warning + no-op (`:178-189`).
- Debug CLI: `--baselines` csv, default = todos em `BASELINE_SPECS` — `main_run_baselines.py:106-115`; unknown → `argparse.ArgumentTypeError` (`:34-46`).

**Conjunto suportado** (`run_baselines_use_case.py:28,46-58`): apenas 3 baselines:
- `zero_return`
- `historical_mean_rolling`
- `historical_quantiles_rolling`

**Tabelas silver emitidas por run** (`run_baselines_use_case.py:486-638`):
1. `dim_run` — via `_persist_dim_run` (`:288-329`)
2. `fact_run_snapshot` — via `_persist_fact_run_snapshot` (`:331-384`)
3. `fact_config` — via `_persist_fact_config` (`:386-425`)
4. `bridge_run_features` — via `_persist_bridge_run_features` (`:427-442`) (single row `feature_name=f"baseline:{name}"`)
5. `fact_oos_predictions` — via `_emit_oos_rows` (`:191-286`) → `append_fact_oos_predictions` (`:614-616`)

**NAO** emite: `fact_training_runtime`, `fact_feature_importance`, `fact_attention_summary` (exemptos por gate em `validate_analytics_quality_use_case.py:1076-1099`).

**Convencao `parent_sweep_id`** (para alinhamento pareado):
- `parent_sweep_id_root = config['output_subdir']` — `run_baselines_test_pipeline_use_case.py:191-195`
- Multi-fold: `f"{root}__{fold.name}"` — `:228-231`
- Single-fold: usa root direto (`fold.name == "single"`)
- Exigido (raise se vazio): `run_baselines_use_case.py:501-504`

**Alinhamento de timestamps com TFT** (contrato F.0.2/F.0.3):
- Offsets espelham os drops do trainer TFT: `evaluation_start_offset_days = max(max_encoder_length - 1, 0)`; `evaluation_end_offset_days = max(max_prediction_length, 0)` — `run_baselines_test_pipeline_use_case.py:200-212`
- Aplicados no slicing: `idxs = idxs[n_start:]` depois `idxs = idxs[:-n_end]` — `run_baselines_use_case.py:223-228`
- Defesa: gate `tft_baselines_timestamp_subset_alignment` (`validate_analytics_quality_use_case.py:92-170, 1058-1074`) verifica set-equality de `target_timestamp_utc` por `(asset, parent_sweep_id, split, horizon)`

**`run_id` deterministico** (`_compute_run_id`, `run_baselines_use_case.py:98-116`):
- `sha256({"kind":"baseline", "baseline_name", "asset", "parent_sweep_id", "seed", "window"})`

**Hashes**:
- `feature_set_name = "baseline"` (literal) — `:546`
- `feature_set_hash = sha256("baseline|{name}|window={window}")` — `:547`
- `config_signature = sha256({"baseline":..., "window":..., "horizons":...})` — `:548-553`

**Overwrite (Lei 2)**: flag CLI `--overwrite-on-collision` (`main_run_baselines.py:164-168`) propaga como `overwrite_on_collision` kwarg em todo `append_*`. Default `False` → colisao raise.

### 5.1 Baseline: zero_return

| Campo | Valor |
|---|---|
| **Spec** | `BaselineSpec(name="zero_return", prediction_mode="point", window=None)` — `run_baselines_use_case.py:47` |
| **`model_version`** | `"baseline_zero_return_v1"` — `:554` |
| **`loss_name`** | `"baseline_zero_return"` — `:404` |
| **Inputs (dataset_tft)** | `timestamp, target_return` — `:147-152` |
| **Calibration window** | nenhuma |
| **Hiperparametros** | nenhum (`window=None`) |

#### Calculo

| # | Output | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | `y_pred` | (none) | `0.0` para todo horizon | `run_baselines_use_case.py:174-175` (returns `(0.0, 0.0, 0.0, 0.0)`) |
| 2 | `q10/q50/q90` | (degenerado) | `(0.0, 0.0, 0.0)` | `:175` |

`prediction_mode="point"` → Stage 9/11 degeneracy gates skipam.

### 5.2 Baseline: historical_mean_rolling

| Campo | Valor |
|---|---|
| **Spec** | `BaselineSpec(name="historical_mean_rolling", prediction_mode="point", window=30)` — `run_baselines_use_case.py:33,48-52` |
| **`model_version`** | `"baseline_historical_mean_rolling_v1"` — `:554` |
| **`loss_name`** | `"baseline_historical_mean_rolling"` — `:404` |
| **Inputs (dataset_tft)** | `timestamp, target_return` — `:147-157` |
| **Calibration window** | `30` (DEFAULT_MEAN_WINDOW, overridable) — `:33` |
| **Justificativa** | "30 dias para media: suaviza ruido diario sem capturar regime de medio prazo" — `:30-32` |
| **History rule** | `history = target_returns[:i]` (strictly past) — `:243` |
| **Warmup** | `len(finite_history) < window` → return `None`, row skipped, `skipped_warmup++` — `:178-180, 249-251` |

#### Calculo

| # | Output | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | `y_pred` | window past returns | `mean(target_return[i-window : i])` (exclui `i`); mesmo valor para todo `h` | `run_baselines_use_case.py:180-183` |
| 2 | `q10/q50/q90` | (degenerado) | `(mean, mean, mean)` | `:183` |

### 5.3 Baseline: historical_quantiles_rolling

| Campo | Valor |
|---|---|
| **Spec** | `BaselineSpec(name="historical_quantiles_rolling", prediction_mode="quantile", window=252)` — `run_baselines_use_case.py:34,53-57` |
| **`model_version`** | `"baseline_historical_quantiles_rolling_v1"` — `:554` |
| **`loss_name`** | `"baseline_historical_quantiles_rolling"` — `:404` |
| **`objective_name`** | `"mean_pinball"` (vs `"rmse"` dos point) — `:422` |
| **`quantile_levels_json`** | `[0.1, 0.5, 0.9]` — `:405` |
| **Inputs (dataset_tft)** | `timestamp, target_return` — `:147-157` |
| **Calibration window** | `252` (DEFAULT_QUANTILE_WINDOW, ~1 ano de pregoes) — `:34` |
| **Justificativa** | "252 dias para quantis: aproxima 1 ano de pregoes (anualidade financeira)" — `:30-34` |
| **History rule** | `history = target_returns[:i]` (strictly past) — `:243` |
| **Warmup** | igual ao mean (252 rows) — `:178-180, 249-251` |

#### Calculo

| # | Output | Inputs | Formula | Ref |
|---|---|---|---|---|
| 1 | `q10` | window past returns | `np.percentile(window_slice, 10)` | `run_baselines_use_case.py:186` |
| 2 | `q50` | window | `np.percentile(window_slice, 50)` | `:187` |
| 3 | `q90` | window | `np.percentile(window_slice, 90)` | `:188` |
| 4 | `y_pred` | window | `q50` (mediana dobra como point) | `:184-188` (returns `(q50, q10, q50, q90)`) |

### 5.4 Loop comum de emissao (todos os baselines)

`_emit_oos_rows` (`run_baselines_use_case.py:191-286`):
- Iteracao: split outer → `idx i` → horizon inner — `:215-285`
- Offset slicing: `:223-228`
- History strictly past: `:243`
- `_compute_prediction` per baseline (static): `:159-189`
- `RunContext` per split (`fold="none"` hard-coded): `:230-240`
- `MultiHorizonPredictionPersister.build_record` (mesmo do TFT): `:254-285`
- Anchor convention (mesma do TFT): `decision_idx = i`, `y_true = target_return[decision_idx + h - 1]` (ADR-0003 Opcao (a)) — `:258-262`; `target_timestamp = dataset_timestamps[decision_idx + h]` — `multi_horizon_prediction_persister.py:96-107`
- Quantile guardrail aplicado (defensivo): `QuantileGuardrailService.enforce_monotonic_triplet` — `:266`
- Skip rules: `y_true` nao-finito (`:263-265`), `IncompletePredictionWindowError` (`:283-284`)

### 5.5 Diferencas de `fact_config` vs TFT

`_persist_fact_config` (`run_baselines_use_case.py:386-425`):

| Campo | Valor (baseline) |
|---|---|
| `prediction_mode` | `"point"` ou `"quantile"` per spec (`:403`) |
| `loss_name` | `"baseline_{name}"` (`:404`) |
| `quantile_levels_json` | `[0.1, 0.5, 0.9]` se quantile; `[]` se point (`:405`) |
| `evaluation_horizons_json` | `horizons_sorted` (`:406`) |
| `max_encoder_length` | reutilizado como `spec.window` (`:407`) |
| `max_prediction_length` | `max(horizons)` (`:408`) |
| `batch_size, max_epochs, learning_rate, hidden_size, attention_head_size, dropout, hidden_continuous_size, early_stopping_patience, early_stopping_min_delta` | todos `0` (`:410-417`) |
| `scaler_type` | `"none"` (`:418`) |
| `training_config_json` | `{"baseline": name, "window": window}` (`:419`) |
| `objective_name` | `"mean_pinball"` se quantile, `"rmse"` se point (`:422`) |

### 5.6 Diferencas de `dim_run` vs TFT

`_persist_dim_run` (`run_baselines_use_case.py:288-329`):
- `execution_id=None, trial_number=None, fold=None, seed=None` (note: `seed` e hashed no `run_id` mas NULL no row — `:305-309`)
- `feature_set_name="baseline"`, `feature_set_hash=sha256("baseline|{name}|window={window}")` (`:546-547`)
- `feature_list_ordered_json=json.dumps([baseline_name])` (`:313`) — "feature" e o proprio baseline
- `model_version=f"baseline_{name}_v1"` (`:554`)
- `checkpoint_path_*=None` (sem artefato)
- `library_versions_json=None, hardware_info_json=None, git_commit=None`
- `status="ok"`, `duration_total_seconds=0.0`, `eta_recorded_seconds=0.0`, `retries=0` (`:323-326`)

### 5.7 Quality gates aplicados aos baselines

- **`baselines_share_parent_sweep_id_with_candidates`** (cohort_decision; `validate_analytics_quality_use_case.py:1017-1056`): para cada `parent_sweep_id`, requer `n_baselines > 0` se `n_candidates > 0`. Deteccao por `feature_set_name=="baseline" OR model_version.startswith("baseline_")` (`:1030-1034`).
- **`tft_baselines_timestamp_subset_alignment`** (cohort_decision; `:92-170, 1058-1074`): join `fact_oos_predictions × dim_run` + verifica set-equality `tft_ts == baseline_ts` por `(asset, parent_sweep_id, split, horizon)`. Falha → retorna `symdiff` cardinality.
- **Artifact-exemption** (`:1076-1099`): exclui `run_id`s de baseline do contrato `feature_importance/attention`, mas ainda exige p10/p50/p90.
- **Probabilistic-metric exclusion** (`:925-927`): point baselines (`zero_return`, `historical_mean_rolling`) filtrados de metricas PICP-dependentes.

### 5.8 Lacunas dos baselines

- §A.51 (3 baselines documentados nao implementados: `random_walk, AR(1), EWMA-vol` — promessas em `docs/04_evaluation/BASELINES.md` e na docstring de `run_baselines_use_case.py:27`; ausentes do `BASELINE_SPECS`; raise se chamados)
- §A.52 (`zero_return` e `historical_mean_rolling` emitem triplet quantilico degenerado mas `prediction_mode="point"` — Stage 11 degeneracy gate so flagaria se tratado como quantile)
- §A.53 (`fold="none"` hard-coded em `RunContext` para baselines — fold identity vive so no sufixo do `parent_sweep_id`)
- §A.54 (`seed=0` no `RunContext` mas `seed=None` no `dim_run` row — `seed` e hashed no `run_id` mas `dim_run.seed` retorna NULL)
- §A.55 (`required_warmup_count=0` no `fact_run_snapshot` independente do `window` — warmup real (252 do quantiles) so aparece via `skipped_warmup` counter, nao persistido)
- §A.56 (`evaluation_start_offset_days` mal-nomeado — operacao e em rows, nao dias; coincide so em dataset diario)
- §A.57 (`main_rebuild_explicit_sweep_predictions.py` e TFT-only — nao ha shortcut para rebuild de baselines em sweep explicit)
- §A.58 (anchor convention ADR-0003 nao asserted runtime — guardado apenas pelo `tft_baselines_timestamp_subset_alignment` gate em silver)

## 6. Stage: gold (analytics)

**Entrypoint**: [`src/main_refresh_analytics_store.py`](../../src/main_refresh_analytics_store.py)
**Use case**: [`src/use_cases/refresh_analytics_store_use_case.py`](../../src/use_cases/refresh_analytics_store_use_case.py)
**Writer**: `_safe_write` em `refresh_analytics_store_use_case.py:190-193`
**Schemas**: **NENHUM declarado** (ver §A.59)
**Scope domain**: [`src/domain/services/scope_spec.py`](../../src/domain/services/scope_spec.py)

### 6.0 Achados arquiteturais (LER ANTES das tabelas)

**LACUNA G1 (architectural)** — §A.59: `analytics_store_schema.py` declara **zero** `GOLD_*_SCHEMA` constants. O registry `ANALYTICS_TABLE_SCHEMAS` (`:674-691`) enumera apenas silver. Gold tables nao tem ref de schema, dtype contract, partition policy, nem PK declarada.

**LACUNA G2 (writer divergence)** — §A.60: O writer unico de gold e `_safe_write` em `refresh_analytics_store_use_case.py:190-193`:
```python
def _safe_write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
```
**Sem PK/dtype validation, sem particionamento**. Gold = single flat parquet `<analytics_gold_dir>/<table>.parquet`. NAO usa o caminho `ParquetAnalyticsRunRepository._write_with_overwrite_policy` (silver).

**LACUNA G3 (update policy)** — §A.61: Toda escrita gold e **full overwrite** por construcao (`df.to_parquet` em single file).

**LACUNA G4** — §A.62: Todas as 24 tabelas gold emitidas estao "written but not declared".

### 6.1 Inventario de tabelas gold emitidas

Listagem ordenada por linha em `refresh_analytics_store_use_case.py` (`execute` ~lines 2416-2563):

| # | Tabela | Builder | Emit |
|---|---|---|---|
| 1 | `gold_runs_long` | `_build_gold_runs_long` (`:267-294`) | `:2416-2419` |
| 2 | `gold_ranking_by_config` | `_build_gold_ranking_by_config` (`:296`) | `:2420-2423` |
| 3 | `gold_consistency_topk` | `_build_gold_consistency_topk` (`:339`) | `:2424-2427` |
| 4 | `gold_ic95_by_config_metric` | `_build_gold_ic95` (`:392`) | `:2428-2431` |
| 5 | `gold_feature_set_impact` | `_build_gold_feature_set_impact` (`:429`) | `:2432-2435` |
| 6 | `gold_oos_consolidated` | `_build_gold_oos_consolidated` (`:1305`) | `:2436-2440` |
| 7 | `gold_prediction_metrics_by_run_split_horizon` | `_build_gold_prediction_metrics_by_run_split_horizon` (`:655`) | `:2442-2453` |
| 8 | `gold_quantile_guardrail_audit` | `_build_gold_quantile_guardrail_audit` (`:769`) | `:2455-2457` |
| 9 | `gold_quantile_degeneracy_report` | `_build_gold_quantile_degeneracy_report` (`:880`) | `:2459-2466` |
| 10 | `gold_prediction_metrics_by_config` | `_build_gold_prediction_metrics_by_config` (`:933`) | `:2467-2470` |
| 11 | `gold_prediction_metrics_by_horizon` | `_build_gold_prediction_metrics_by_horizon` (`:965`) | `:2471-2474` |
| 12 | `gold_prediction_calibration` | `_build_gold_prediction_calibration` (`:997`) | `:2475-2478` |
| 13 | `gold_prediction_risk` | `_build_gold_prediction_risk` (`:1226`) | `:2479-2482` |
| 14 | `gold_prediction_generalization_gap` | `_build_gold_prediction_generalization_gap` (`:1025`) | `:2483-2486` |
| 15 | `gold_prediction_robustness_by_horizon` | `_build_gold_prediction_robustness_by_horizon` (`:1063`) | `:2487-2490` |
| 16 | `gold_feature_impact_by_horizon` | `_build_gold_feature_impact_by_horizon` (`:1113`) | `:2491-2497` |
| 17 | `gold_feature_contrib_local_summary` | `_build_gold_feature_contrib_local_summary` (`:2262`) | `:2498-2501` |
| 18 | `gold_oos_quality_report` | `_build_gold_oos_quality_report` (`:1463`) | `:2502-2506` |
| 19 | `gold_dm_pairwise_results` | `_build_gold_dm_pairwise_results` (`:1568`) + `_apply_holm_adjustment_for_dm` (`:2228`) | `:2507-2513` |
| 20 | `gold_mcs_results` | `_build_gold_mcs_results` (`:1642`) + `_compute_mcs_from_loss_matrix` (`:1387`) | `:2514-2518` |
| 21 | `gold_win_rate_pairwise_results` | `_build_gold_win_rate_pairwise_results` (`:2048`) | `:2519-2523` |
| 22 | `gold_paired_oos_intersection_by_horizon` | `_build_gold_paired_oos_intersection_by_horizon` (`:1718`) | `:2524-2532` |
| 23 | `gold_model_decision_final` | `_build_gold_model_decision_final` (`:1811`) | `:2533-2545` |
| 24 | `gold_quality_statistics_report` | `_build_gold_quality_statistics_report` (`:2141`) | `:2546-2555` |
| 25 | `gold_quality_run_sweep_summary` | `_build_gold_quality_run_sweep_summary` (`:1190`) | `:2556-2563` |

(Total: 25 tabelas — preview do agente dizia 24, a contagem real e 25.)

### 6.2 Formulas canonicas (statistical core)

Reusadas em multiplas gold tables. Cada formula com `file:line`.

#### 6.2.1 `_pinball_loss(y_true, y_pred_q, quantile)` — `:470-473`

```
diff = y_true - y_pred_q
pinball = max(q * diff, (q - 1) * diff)
```

Aplicada per row em `:575-577` para q=0.1, q=0.5, q=0.9. Mean per (run, split, horizon) gera colunas `pinball_q10`, `pinball_q50`, `pinball_q90`, `mean_pinball` (`:619-622`).

#### 6.2.2 PICP e MPIW — `:563-567, 623-625`

```
pred_interval_width = q90 - q10
covered_80 = (y_true >= q10) AND (y_true <= q90)
PICP = mean(covered_80)             # cobertura empirica
MPIW = mean(pred_interval_width)    # largura media
coverage_nominal = 0.80
coverage_error = PICP - coverage_nominal   # :641
```

Mascara probabilistica: cols probabilisticas (`pinball_*`, `picp`, `mpiw`, `prob_up`) NaN-fied para rows com `prediction_mode != "quantile"` OR quantiles degenerados (`q10==q90`) — `:534-557, 597-599`.

`confidence_calibrated = calibration_term * width_term` (`:643-645`):
- `calibration_term = clip(1 - |coverage_error| / coverage_nominal, 0, 1)`
- `width_term = 1 / (1 + clip(pred_interval_width, 0, +inf))`

#### 6.2.3 `_prob_up_from_quantiles(q10, q50, q90)` — `:475-493`

CDF piecewise-linear approximation (anchors `CDF(q10)=0.1`, `CDF(q90)=0.9`):
```
width = q90 - q10
cdf0 = clip(0.1 + 0.8 * ((0 - q10) / width), 0.1, 0.9)
# Hard bounds:
cdf0 = 0.0 if q10 > 0
cdf0 = 1.0 if q90 < 0
# Fallback (zero-width):
fallback = 1.0 if q50 > 0 else 0.0 if q50 < 0 else 0.5
cdf0 = fallback if cdf0 is NaN
prob_up = 1.0 - cdf0
```

#### 6.2.4 Diebold-Mariano — `_compute_dm_pairwise_from_loss_matrix:1327-1364`

Per-par `(left, right)` em loss matrix `L[t, config]` (loss = squared_error per timestamp):
```
d_t = L[t, left] - L[t, right]            (loss differential)
d   = d_t[isfinite(d_t)]                  # filtra NaN
n   = len(d)
if n < 5: skip

mean_d = mean(d)
d_c    = d - mean_d
lag    = min(max(1, n^(1/3)), 10)         # Bartlett kernel bandwidth

gamma_0 = sum(d_c^2) / n
hac     = gamma_0
for k in 1..lag:
    cov_k  = sum(d_c[k:] * d_c[:-k]) / n
    weight = 1 - k/(lag+1)                # Bartlett (triangular) kernel
    hac   += 2 * weight * cov_k

var_mean = hac / n
dm_stat  = mean_d / sqrt(var_mean)
p_value  = 2 * (1 - Phi(|dm_stat|))       # two-sided normal CDF
```

Selecao top-50 configs por `mean(squared_error)` antes do pairwise — `_select_top_configs_for_pairwise:1367-1384`.

Loss matrix construido em `_build_gold_dm_pairwise_results:1568-1640`:
- Filtra `split=="test"` e `status=="ok"` (`:1597-1598`)
- Por grupo `(asset, parent_sweep_id, split_signature, split, horizon)`
- `config_label = f"{feature_set_name}|{config_signature}"` (`:1608`)
- `loss = (y_pred - y_true)^2` (`:1609`)
- Mean per `(target_timestamp, config_label)` (`:1615-1619`)
- Pivot to wide; `dropna(axis=0, how="any")` (alinhamento estrito) — `:1620-1621`
- DM stat por par; output cols: `left_config, right_config, n, mean_loss_diff_left_minus_right, dm_stat, pvalue_two_sided, asset, parent_sweep_id, split_signature, split, horizon, aligned_timestamps, n_configs`

#### 6.2.5 Holm step-down adjustment — `_apply_holm_adjustment_for_dm:2228-2259`

Per grupo `(asset, parent_sweep_id, split, horizon)`:
```
ordered = sort_ascending(p_values)
m = len(ordered)
for j in 1..m:                                # 1-indexed
    adj[j] = (m - j + 1) * ordered[j]
adj = cumulative_max(adj)                    # monotonic step-down
adj = min(adj, 1.0)                          # clip to [0, 1]
significant_adj_0_05 = (pvalue_adj_holm < 0.05)
```

Output cols adicionadas a `gold_dm_pairwise_results`: `pvalue_adj_holm, significant_adj_0_05`.

#### 6.2.6 MCS (Model Confidence Set) — `_compute_mcs_from_loss_matrix:1387-1460`

Hansen et al. (2011), variante range t-stat com block bootstrap.

Parametros (hard-coded):
- `alpha = 0.05`
- `bootstrap_samples = 300`
- `block_len = 5` (moving block bootstrap, wrap-around)
- `random_seed = 42`

Algoritmo:
```
1. mean_loss = mean(L[:, j]) for each config j; sort ascending
2. active = list(range(n_configs))                   # all configs initially
3. while len(active) > 1:
       sub = L[:, active]
       n_obs, n_models = sub.shape

       # observed pairwise mean differentials
       dbar[i,j] = mean(sub[:,i] - sub[:,j])

       # block bootstrap (moving block, wrap-around)
       for b in 1..300:
           idx = block_bootstrap_indices(n_obs)      # blocks of length 5
           boot[b,i,j] = mean(sub[idx, i] - sub[idx, j])

       var[i,j] = var(boot[:,i,j], ddof=1)           # bootstrap variance
       t[i,j]    = |dbar[i,j] / sqrt(var[i,j])|

       tr_stat = nanmax(t)                            # range t-statistic
       boot_centered = boot - dbar
       tr_boot[b]    = nanmax(|boot_centered[b,:,:] / sqrt(var)|)

       p_value = mean(tr_boot >= tr_stat)
       if p_value >= alpha: break

       # eliminate worst config in active set
       losses_mean = mean(sub, axis=0)
       active.pop(argmax(losses_mean))

4. selected_in_mcs_alpha_0_05 = (config in active)
```

Output cols: `config_label, selected_in_mcs_alpha_0_05, mean_loss`. Loss matrix mesma construcao do DM (squared_error per `(target_timestamp, config_label)`).

#### 6.2.7 Win rate pairwise — `_build_gold_win_rate_pairwise_results:2048`

Por par `(left, right)`: `win_rate = mean(loss_left < loss_right)` ao longo dos `target_timestamp`s alinhados. Estatistica simples sem ajuste de variancia.

### 6.3 Detalhe por gold table (resumo)

Para nao explodir o doc, cada tabela e resumida; o codigo do builder e fonte da verdade.

#### 6.3.1 gold_prediction_metrics_by_run_split_horizon — `:496-663`

- **Grain**: 1 linha por `(run_id, asset, feature_set_name, config_signature, split, fold, seed, horizon)`
- **Reads**: `dim_run, fact_oos_predictions, fact_config`
- **Cols agregados** (`:608-628`): `n_samples, n_probabilistic_samples, rmse, mae, mape, smape, directional_accuracy, bias, pinball_q10, pinball_q50, pinball_q90, mean_pinball, picp, mpiw, pred_interval_width, prob_up`
- **Cols derivados**: `is_quantile_genuine, coverage_nominal=0.80, coverage_error, prob_down, confidence_calibrated`
- **Variantes**: existe `gold_prediction_metrics_by_run_split_horizon` (acima) + `_build_gold_quantile_guardrail_audit` (`:769-878`) que compara metricas raw vs post-guardrail
- **Quantile contract**: usa as cols `quantile_p10/p50/p90` (RAW) ou `*_post_guardrail` per `primary_quantile_contract` (constructor arg) — `:1838-1888`

#### 6.3.2 gold_prediction_metrics_by_config — `:933-963`

- **Grain**: 1 linha por `(asset, feature_set_name, config_signature, split, horizon)` (drop fold/seed/run_id)
- **Reads**: `gold_prediction_metrics_by_run_split_horizon`
- **Agregacao**: mean / std / iqr / min / max sobre seeds e folds (em particular `n_runs_*` para contagem)

#### 6.3.3 gold_prediction_metrics_by_horizon — `:965-995`

- **Grain**: agregado por `horizon` somente (sem config)
- **Reads**: idem above

#### 6.3.4 gold_prediction_calibration — `:997-1023`

- **Grain**: por `(config_signature, split, horizon)`
- **Cols**: `picp` (raw e post_guardrail), `coverage_error`, `mpiw_*`, `mean_pinball_*` — `:1008-1013`

#### 6.3.5 gold_prediction_generalization_gap — `:1025-1061`

- **Grain**: por `(config_signature, horizon)` com `test` vs `val`
- **Cols**: `gap_mean_pinball_test_minus_val, gap_picp_test_minus_val, gap_mpiw_test_minus_val, gap_rmse_test_minus_val, gap_mae_test_minus_val, gap_directional_accuracy_test_minus_val` (`:1886-1888`)

#### 6.3.6 gold_prediction_robustness_by_horizon — `:1063-1111`

- **Grain**: por `(config_signature, horizon)`
- **Metricas**: dispersao das metricas across seeds (std, range, etc.)

#### 6.3.7 gold_prediction_risk — `:1226-1303`

- **Grain**: por `(asset, parent_sweep_id, config_signature, split, horizon)`
- **Reads**: `dim_run + fact_oos_predictions`
- **Cols** (tail / extreme): VaR, ES (expected shortfall), max_drawdown (todos calculados sobre `error = y_pred - y_true`)

#### 6.3.8 gold_oos_consolidated — `:1305-1461`

- **Grain**: 1 linha por row de `fact_oos_predictions` enriquecido com `dim_run.parent_sweep_id` + flags derivados (`is_baseline, is_quantile_mode`)
- **Uso**: feed unificado para os demais agregados gold

#### 6.3.9 gold_oos_quality_report — `:1463-1566`

- **Grain**: por `(asset, parent_sweep_id, split, horizon)`
- **Cols**: contagens de rows, NaN counts, range checks, alinhamento de target_timestamp

#### 6.3.10 gold_dm_pairwise_results — §6.2.4 + §6.2.5

- **Grain**: 1 linha por `(asset, parent_sweep_id, split_signature, split, horizon, left_config, right_config)`
- **Filtro**: somente `split="test"`, `status="ok"`, top-50 configs
- **Cols**: `n, mean_loss_diff_left_minus_right, dm_stat, pvalue_two_sided, pvalue_adj_holm, significant_adj_0_05, aligned_timestamps, n_configs`

#### 6.3.11 gold_mcs_results — §6.2.6

- **Grain**: 1 linha por `(asset, parent_sweep_id, split, horizon, config_label)`
- **Cols**: `config_label, selected_in_mcs_alpha_0_05, mean_loss`

#### 6.3.12 gold_win_rate_pairwise_results — §6.2.7, `:2048-2139`

- **Grain**: idem DM, mas sem variance/p-value
- **Cols**: `left_config, right_config, n, win_rate_left, win_rate_right`

#### 6.3.13 gold_paired_oos_intersection_by_horizon — `:1718-1809`

- **Grain**: por `(asset, parent_sweep_id, split, horizon, run_id_tft, run_id_baseline)`
- **Cols**: `intersection_count, tft_only_count, baseline_only_count, union_count, jaccard_index`
- **Uso**: feed para DM/MCS so quando intersection_count > 0

#### 6.3.14 gold_model_decision_final — `:1811-2046`

- **Grain**: 1 linha por `(asset, parent_sweep_id, split, horizon)` (com config selecionada)
- **Reads**: `metrics_by_config + robustness_by_horizon + generalization_gap + dm_results + mcs_results + win_rate + paired_intersection`
- **Logica**: criterio composto — config selecionada deve estar em MCS, ter DM significativo vs baseline (Holm-adjusted), generalization_gap dentro de threshold
- **Output**: a decisao final que sustenta `STRATEGIC_DIRECTION` claims

#### 6.3.15 gold_quality_statistics_report — `:2141-2261`

- **Grain**: por `(asset, parent_sweep_id, split, horizon)`
- **Cols**: agregados de quality (`dm_min_pvalue` em `:2171`, `mcs_*`, etc.)

#### 6.3.16 gold_quality_run_sweep_summary — `:1190-1224`

- **Grain**: por `(asset, parent_sweep_id)`
- **Reads**: `quality_report + quality_statistics + dim_run`
- **Cols**: contagem de runs ok/failed/partial, breakdown por status

#### 6.3.17 gold_feature_set_impact — `:429`

- **Grain**: por `(feature_set_name, split, horizon)`
- **Logica**: delta de metrica vs baseline (within same parent_sweep_id)

#### 6.3.18 gold_feature_impact_by_horizon — `:1113-1188`

- **Grain**: por `(asset, parent_sweep_id, feature_set_name, horizon)`
- **Reads**: `dim_run + gold_prediction_metrics_by_run_split_horizon`

#### 6.3.19 gold_feature_contrib_local_summary — `:2262-2416`

- **Grain**: por `(asset, parent_sweep_id, feature_set_name, horizon, feature_name)`
- **Reads**: `fact_feature_contrib_local + dim_run`
- **Agregacao**: `mean(contribution)`, `mean(abs_contribution)`, `feature_rank` agregado, sign consistency (% positive)

#### 6.3.20 gold_runs_long — `:267-294`

- **Grain**: long-format de `(dim_run JOIN fact_split_metrics)`
- **Cols** (`:271-294`): `run_id, asset, feature_set_name, feature_set_hash, config_signature, model_version, parent_sweep_id, trial_number, fold, seed, split, rmse, mae, ...`
- **Uso**: base para `ranking_by_config, consistency_topk, ic95`

#### 6.3.21 gold_ranking_by_config — `:296-338`

- **Grain**: por `(config_signature, split)` com rank de RMSE/MAE

#### 6.3.22 gold_consistency_topk — `:339-390`

- **Grain**: medida de consistencia top-k across seeds por config

#### 6.3.23 gold_ic95_by_config_metric — `:392-428`

- **Grain**: per `(config_signature, split, metric_name)` com IC95 bootstrap

#### 6.3.24 gold_quantile_guardrail_audit — `:769-878`

- **Grain**: per `(run_id, split, horizon)`
- **Cols**: comparacao RAW vs POST-GUARDRAIL de `pinball_*, picp, mpiw, coverage_error` — `:1008-1013`

#### 6.3.25 gold_quantile_degeneracy_report — `:880-931`

- **Grain**: per `(run_id, split, horizon)`
- **Cols**: contagem de `p10==p90` rows, `p10_eq_p90_rate`, gate result

### 6.4 Quality gates do refresh (silver validation)

`ValidateAnalyticsQualityUseCase` consome o silver ANTES de gerar gold. Gates relevantes:
- `required_tables_presence` — `:326-342`
- `referential_integrity` (todo fact referencia `dim_run.run_id`) — `:401-430`
- `temporal_consistency, oos_unique_key, oos_numeric_types, oos_horizon_coverage, oos_supervised_nulls, oos_interval_width_non_negative, oos_quantile_order` — `:451-559, 639-680`
- `oos_pairwise_target_alignment, oos_quantile_block_a_acceptance, block_quantile_degeneracy_gate` — `:600-713`
- `min_samples_by_split, cardinality_config_fold_seed, baselines_share_parent_sweep_id_with_candidates, tft_baselines_timestamp_subset_alignment, official_contract_quantile_attention` — `:775-1145`

### 6.5 Lacunas do gold

- §A.59 (zero `GOLD_*_SCHEMA` declarado — todo o gold e schemaless)
- §A.60 (`_safe_write` sem PK/dtype/partition — divergencia silver vs gold)
- §A.61 (overwrite total por design)
- §A.62 (25 tabelas "written but not declared")
- §A.63 (DM lag selection hard-coded `min(max(1, n^(1/3)), 10)`)
- §A.64 (MCS params hard-coded: `alpha=0.05, bootstrap_samples=300, block_len=5, random_seed=42`)
- §A.65 (DM top-50 cap aplicado SILENCIOSAMENTE — `_select_top_configs_for_pairwise:1367-1384` — configs alem do top-50 nao aparecem em gold_dm)
- §A.66 (Holm grouping fixo em `(asset, parent_sweep_id, split, horizon)` — nao agrupa por `split_signature`)
- §A.67 (`_prob_up_from_quantiles` fallback `0.5` quando width = 0 mascara modelo degenerado — pode esconder problema)
- §A.68 (`coverage_nominal = 0.80` hard-coded — assume contrato fixo p10/p90; se sweep usar quantis diferentes, `coverage_error` esta errado mas nao falha)

## A. Lacunas conhecidas

Lista canonica de lacunas encontradas durante este mapeamento. Cada item linka da celula relevante.

### A.1 — `timestamp` em `candle_parquet_schema.py:12-18` sem dtype declarado
`CANDLE_PARQUET_COLUMNS` declara `timestamp` mas nao tem entry em `CANDLE_PARQUET_DTYPES`. Persistido como `datetime` inferido por pyarrow.

### A.2 — `update_sentiment` muta candle parquet in-place
`parquet_candle_repository.py:206-262` adiciona `sentiment_score, sentiment_std, n_articles` ao parquet de candles — **nao** declaradas em `CANDLE_PARQUET_COLUMNS`. Quebra "raw imutavel".

### A.3 — `provider/interval` (yfinance) decorativos
`config/data_sources.yaml:36-38` declara mas `yfinance_candle_fetcher.py:59` hard-codeia `interval="1d"`.

### A.4 — `CANDLE_PARQUET_COLUMNS` e `set`
Ordem de colunas no parquet nao deterministica (CPython mantem insertion-order mas nao e contrato).

### A.5 — Sem incremental append no candle
Cada run rewrite total — `parquet_candle_repository.py:114-148`. TODO `:277-295`.

### A.6 — DQ candles ignora calendario de feriados
`data_quality_reporter.py:175-181` reporta missing business days mesmo em feriados validos.

### A.7 — `paths.get("raw_news")` lookup errado
`main_news_dataset.py:125` busca `"raw_news"`; `path_resolver.py:41` expoe como `"news_dataset"`. Cai no fallback `data/raw/news`. `main_sentiment.py:57` usa o nome correto.

### A.8 — Profile DQ `news_raw` nao auditado
Referenciado em `main_news_dataset.py:216` mas conteudo nao foi inspecionado nesta varredura.

### A.9 — `finnhub_news_fetcher.py` dead code
Implementa `NewsFetcher` mas nenhum entrypoint instancia.

### A.10 — `sqlite_news_repository.py:6` import quebrado
`from src.entities.news import News` — entidade nao existe (real e `entities/news_article.py`). Adapter morto.

### A.11 — Alpha Vantage descarta `topics, overall_sentiment_score, ticker_sentiment`
`alpha_vantage_news_fetcher.py:149-162` so consome 5 campos do feed.

### A.12 — `sort/limit` em data_sources.yaml decorativos
Linhas 45-46 declaradas; hard-coded em `alpha_vantage_news_fetcher.py:108-109`.

### A.13 — `language` hard-coded `"en"`
`alpha_vantage_news_fetcher.py:181`, `parquet_news_repository.py:89,139`.

### A.14 — `timestamp` em `technical_indicator_parquet_schema.py:8-10` sem dtype declarado
Idem A.1, no schema de indicadores.

### A.15 — DQ profile `technical_indicators` sem `value_ranges`/`validation_rules`
`data_quality_profiles.py:94-100`. Inf/NaN/explosao silenciosa.

### A.16 — Sem leakage guard em `technical_indicators`
`technical_indicator_engineering_use_case.py:30-39` calcula sobre o range inteiro de candles. Trim depende do split downstream.

### A.17 — `volatility_20d` usa simple returns
`technical_indicator_calculator.py:55` = `pct_change().rolling(20).std()`. `formula_desc` no schema (`technical_indicators_schema.py:60-64`) so diz `"source": "close"` — nao pin.

### A.18 — `SklearnTechnicalIndicatorNormalizer` dead code
Existe em `sklearn_indicator_normalizer.py` mas nao usado pelo pipeline.

### A.19 — Warmup rows nao tratadas no stage
NaNs leading de EMA/RSI/volatility ficam no parquet processed.

### A.20 — Profile DQ `scored_news` nao auditado
Referenciado em `main_sentiment.py:96` mas conteudo nao inspecionado.

### A.21 — Probs 3 classes descartadas
`finbert_sentiment_model.py:114-118` calcula `P(neg), P(neu), P(pos)` mas so persiste `P(pos) - P(neg)`.

### A.22 — `confidence = |sentiment_score|` nao calibrada
Colapsa para 0 quando `P(neg) ≈ P(pos)` mesmo com `P(neu)` alto.

### A.23 — Sem `revision`/commit do FinBERT
`finbert_sentiment_model.py:48-51` chama `from_pretrained` sem `revision=`. Reprodutibilidade.

### A.24 — Sem assert contra `model.config.id2label`
Pipeline assume ordem `[neg, neu, pos]` — `finbert_sentiment_model.py:116`.

### A.25 — Truncation 512 tokens silenciosa
Sem contador de truncadas.

### A.26 — `language` nao propagada para scored_news
Schema scored nao tem coluna `language`.

### A.27 — Sentiment_daily agregacao fixa em mean
TODO `sentiment_aggregator.py:91-95`. `aggregate_daily` (`:53-78`) consome apenas `asset_id, published_at, sentiment_score` dos `ScoredNewsArticle`; `confidence` e `model_name` sao ignorados (presentes no input mas nao lidos).

### A.28 — `model_name` nao propagado para daily
Schema daily nao tem coluna; reprodutibilidade.

### A.29 — Sentiment_daily sem calendario de feriados US
`trading_calendar.py` so faz weekend filter.

### A.30 — `sentiment_score=0.0` no-news colide com neutro real
`has_news` so existe em dataset_tft (`build_tft_dataset_use_case.py:494`); daily nao distingue.

### A.31 — `sentiment_std=0.0` para `n_articles=1` igual ao sintetico
Nao da pra diferenciar via std.

### A.32 — Path stage fundamentals: doc vs codigo
`docs/02_data/DATA_SOURCES.md:70` diz `fundamental_indicators/`; codigo usa `fundamentals/`.

### A.33 — Config keys `data_sources.fundamentals.*` ignoradas
`enabled, provider, report_types` declaradas mas nao consumidas — provider hard-coded, report_types fallback default.

### A.34 — Throttle/timeout hard-coded em fundamentals
`_MIN_INTERVAL=12.5s`, `timeout=30s`.

### A.35 — `source` literal `"alpha_vantage"` sem provenance
Sem versao da call/codigo do fetcher quando o file tem dados mesclados ao longo do tempo.

### A.36 — `merge_asof` mistura annual+quarterly silenciosamente
`build_tft_dataset_use_case.py:113-143` ignora `report_type` no as-of.

### A.37 — `_is_period_covered` so olha `fiscal_date_end`
`main_fundamentals.py:44-61` nao checa se `reported_date` esta presente.

### A.38 — Dtype OHLCV muda no join (float32 → float64)
`build_tft_dataset_use_case.py:78-80`. `TFT_DATASET_DTYPES` nao enforce.

### A.39 — `sentiment_std` perde nullable no dataset_tft
`Float64 → float64` apos `fillna(0.0)`.

### A.40 — `fundamentals_effective_date` sem dtype em `TFT_DATASET_DTYPES`
Persiste por inferencia do `merge_asof`.

### A.41 — `tft_dataset_schema.py` dead code
Declara `sector, market_cap, rsi` — nao batem com o pipeline real (e `rsi_14`).

### A.42 — `sentiment_ema/surprise` sobre `0.0` imputado
`build_tft_dataset_use_case.py:495-499` fillna antes de `:249-251`. Contamina com zeros estruturais.

### A.43 — `revenue_yoy_growth/net_income_yoy_growth` sem anti-leakage validator
Validators em `:287-416` nao checam estas derivacoes.

### A.44 — DQ quality gate inclui `fundamentals_effective_date`
`main_dataset_tft.py:75-77` `_model_feature_columns` nao exclui — pode falhar nan-ratio em coluna datetime.

### A.45 — `revenue/net_income_yoy_growth` use as-of merged daily series
`pct_change(252)` sobre serie diaria carregada do as-of fundamentals — significa que mudanca de fundamental dispara YoY pulse, nao YoY real ano-a-ano.

### A.46 — Silver inference parquet omite campos
Sem `target_timestamp`, `decision_timestamp`, `horizon`, `quantile_*_post_guardrail`, `quantile_guardrail_applied`. So `timestamp` (= target_timestamp por construcao).

### A.47 — `horizon=1` hard-coded em inference engine
`pytorch_forecasting_tft_inference_engine.py:373`. Modelos com `max_prediction_length>1` so persistem o primeiro decoder step.

### A.48 — `inference_run_id` so com precisao de segundo
`run_tft_inference_use_case.py:535` — possibilidade de colisao em runs no mesmo segundo.

### A.49 — `decision_idx` declarado mas nao populado em `fact_inference_predictions`
Schema `analytics_store_schema.py:546-551`; writer em `run_tft_inference_use_case.py:124-154` nao emite. Comentario do schema explicita "not required here yet because the inference writer is migrated post-Stage 20".

### A.50 — Refresh auto dispara nova build sem policies
`main_infer_tft.py:257-267` chama `BuildTFTDatasetUseCase` sem `trading_day_policy`/`quality_gate_config`. `effective_date` pode divergir do original.

### A.51 — 3 baselines documentados nao implementados
`docs/04_evaluation/BASELINES.md:3,34-35` e `docs/00_overview/STRATEGIC_DIRECTION.md:83,252,257` listam `random_walk, AR(1), EWMA-vol` como canonicos; docstring de `run_baselines_use_case.py:27` os marca "Follow-up YELLOW (Stage 12-bis)". Nao implementados; raise se chamados (`:510-515` e `run_baselines_test_pipeline_use_case.py:113-117`).

### A.52 — Baselines point emitem triplet quantilico degenerado
`zero_return` retorna `(0,0,0)` (`:175`); `historical_mean_rolling` retorna `(mean,mean,mean)` (`:183`). `prediction_mode="point"` escapa o Stage 11 degeneracy gate.

### A.53 — `fold="none"` hard-coded em RunContext de baselines
`run_baselines_use_case.py:238`. Fold identity vive so no sufixo do `parent_sweep_id`. Downstream join em `fact_oos_predictions.fold` nao matchea TFT com fold explicito.

### A.54 — `seed=0` no RunContext mas `seed=None` em dim_run
`run_baselines_use_case.py:239` (RunContext default `seed=int(seed) if seed is not None else 0`) vs `:309` (`"seed": None` no dim_run row). `seed` e hashed no `run_id` (`:107-116`) mas query a `dim_run.seed` retorna NULL para baselines.

### A.55 — `required_warmup_count=0` em fact_run_snapshot independente de window
`run_baselines_use_case.py:375`. Warmup real (252 do quantiles) so aparece via `skipped_warmup` counter (nao persistido — `:625`).

### A.56 — `evaluation_start_offset_days` mal-nomeado
`run_baselines_use_case.py:223-228` opera em indices de row, nao dias. Coincide apenas em dataset diario; intraday futuro quebraria a semantica.

### A.57 — Rebuild de baselines em sweep explicit nao tem shortcut
`main_rebuild_explicit_sweep_predictions.py:46-50` hard-asserta `test_type=="explicit_configs"` (TFT-only). Re-invocar `main_baselines_test_pipeline` ou `main_run_baselines` e o unico caminho.

### A.58 — Anchor ADR-0003 sem assert runtime nos baselines
Comentario em `run_baselines_use_case.py:258` documenta `y_true = target_return[decision_idx + h - 1]`; persister usa `target_timestamp = dataset_timestamps[decision_idx + h]` (`multi_horizon_prediction_persister.py:96-107`). Sem assertion runtime amarrando os dois — guarda apenas no gate silver.

### A.59 — Zero `GOLD_*_SCHEMA` declarados
`analytics_store_schema.py` registry `ANALYTICS_TABLE_SCHEMAS` (`:674-691`) lista apenas silver. Gold schemaless.

### A.60 — `_safe_write` (gold) sem PK/dtype/partition
`refresh_analytics_store_use_case.py:189-193`. Diverge do silver (`ParquetAnalyticsRunRepository._write_with_overwrite_policy`).

### A.61 — Gold full overwrite por construcao
Cada `df.to_parquet(path, index=False)` em single file. Sem upsert, sem append.

### A.62 — 25 tabelas gold "written but not declared"
Toda a §6.1 e gap A.59 materializada.

### A.63 — DM lag selection hard-coded
`refresh_analytics_store_use_case.py:1342`: `lag = min(max(1, n^(1/3)), 10)`. Sem opcao de override.

### A.64 — MCS params hard-coded
`refresh_analytics_store_use_case.py:1387-1394`: `alpha=0.05, bootstrap_samples=300, block_len=5, random_seed=42`. Nao configuraveis via CLI/YAML.

### A.65 — DM top-50 cap silencioso
`_select_top_configs_for_pairwise:1367-1384` filtra para top-50 configs por `mean(squared_error)` antes do pairwise — configs alem do top-50 nao aparecem em `gold_dm_pairwise_results`. Nao reportado em metadado.

### A.66 — Holm grouping fixo
`_apply_holm_adjustment_for_dm:2236` agrupa por `(asset, parent_sweep_id, split, horizon)`. Nao usa `split_signature` mesmo quando disponivel. Pode misturar p-values de splits diferentes do mesmo logical split.

### A.67 — `_prob_up_from_quantiles` fallback mascara degeneracao
`:489-491`. Quando `width=0`, retorna `1.0`/`0.0`/`0.5` em vez de NaN. Esconde modelo degenerado em `prob_up` agregada.

### A.68 — `coverage_nominal = 0.80` hard-coded
`:640`. Assume contrato fixo p10/p90. Se sweep usar quantis diferentes (e.g. p05/p95), `coverage_error = picp - 0.80` esta errado mas nao falha — gold publica numero incorreto silenciosamente.

---

## B. Referencias canonicas

- Source of truth (codigo): listada em cada celula
- Source of truth (decisao): `01_architecture/decisions/ADR-*.md`
- Source of truth (operacao): `06_runbooks/`
- Tracking de incidentes/dividas: `docs/ai/STAGE_20_23_REMEDIATION_PLAN.md` (escopo de remediation atual)
- Map original (skeleton + decisoes): conversa de 2026-05-20 com o usuario (este doc e o entregavel)
