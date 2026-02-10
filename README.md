# TCC: Sentiment Analysis for Financial News and Market Features

> Student: Marcelo Santos  
> Program: Engenharia Mecatronica - UFSC  
> Advisor: Dr. Pedro Paulo 
> Year: 2025

This project builds a data pipeline that collects market and news data, scores sentiment with FinBERT, aggregates daily sentiment, computes technical indicators, fetches fundamentals, and produces datasets for modeling (TFT-ready).

The codebase follows Clean Architecture to keep fetchers, repositories, domain logic, and use cases decoupled and testable.

---

## Current Stage (Feb 2026)
The pipeline is operational end-to-end with structured outputs in `data/processed/`:
- Raw market candles (parquet + report)
- Raw news (parquet + report)
- Scored news (FinBERT)
- Daily sentiment aggregates
- Technical indicators
- Fundamentals (Alpha Vantage)
- TFT dataset (feature set)

Legacy experiments live under `data/legacy/` and are not part of the current pipeline.

---

## Project Structure (high level)
```
config/
  data_paths.yaml
  data_sources.yaml
data/
  raw/                 # candles + raw news
  processed/           # scored_news, sentiment_daily, indicators, fundamentals, tft dataset
  reports/             # experiments (ablation, feature selection, stability)
docs/                  # guides and checklists
notebooks/             # research and experiments
src/
  adapters/            # fetchers, repositories, model adapters
  domain/              # services and time utilities
  entities/            # core entities
  infrastructure/      # parquet schemas
  interfaces/          # ports (interfaces)
  use_cases/           # application logic
  main_*.py            # orchestration entrypoints
tests/                 # unit + integration tests
```

For the full structure see `docs/PROJECT_STRUCTURE.md`.

---

## Requirements
- Windows 10/11
- Python 3.13+
- Git (optional)
- Make (via GnuWin32 or similar)

---

## Setup
```powershell
git clone https://github.com/MarceloSanC/tcc-sentiment-analysis.git
cd tcc-sentiment-analysis
```

Install Make:
```
winget install GnuWin32.Make
```

Create and activate a venv:
```
python -m venv .venv
.\setup.ps1
```

Install dependencies:
```
make install
```

---

## Configuration
Pipeline configuration is centralized in:
- `config/data_sources.yaml` (API providers, symbols, time ranges)
- `config/data_paths.yaml` (raw/processed output paths)

Do not store API keys in this repository. Use environment variables or `.env`.

---

## How to Run
Orchestrators live under `src/` and are exposed via Make targets.

For the complete step-by-step instructions (including training), see `docs/RUNNING_PIPELINE.md`.

Run a single step:

## Configuration
Pipeline configuration is centralized in:
- `config/data_sources.yaml` (API providers, symbols, time ranges)
- `config/data_paths.yaml` (raw/processed output paths)

Do not store API keys in this repository. Use environment variables or `.env`.

---

## How to Run
Orchestrators live under `src/` and are exposed via Make targets.

Run a single step:
```
make run-candles ASSET=AAPL
make run-news-raw ASSET=AAPL
make run-sentiment ASSET=AAPL
make run-sentiment-feat ASSET=AAPL
make run-indicators ASSET=AAPL
```

Run the full pipeline (candles -> news -> sentiment -> daily sentiment -> indicators):
make run-candles ASSET=AAPL
make run-news-raw ASSET=AAPL
make run-sentiment ASSET=AAPL
make run-sentiment-feat ASSET=AAPL
make run-indicators ASSET=AAPL

Run the full pipeline (candles -> news -> sentiment -> daily sentiment -> indicators):
```
make run-all ASSET=AAPL
```

Other entrypoints:
```
python -m src.main_fundamentals --asset AAPL
python -m src.main_dataset_tft --asset AAPL
```

Logs are written to `logs/pipeline.log`.

---

## Outputs
Example output paths (per asset):
- `data/raw/market/candles/<ASSET>/`
- `data/raw/news/<ASSET>/`
- `data/processed/scored_news/<ASSET>/`
- `data/processed/sentiment_daily/<ASSET>/`
- `data/processed/technical_indicators/<ASSET>/`
- `data/processed/fundamentals/<ASSET>/`
- `data/processed/dataset_tft/<ASSET>/`

Each step also writes a JSON report under a `reports/` subfolder.

---

## Tests
Unit tests only (no external APIs):
```
make run-all ASSET=AAPL
```

Other entrypoints:
```
python -m src.main_fundamentals --asset AAPL
python -m src.main_dataset_tft --asset AAPL
```

Logs are written to `logs/pipeline.log`.

---

## Outputs
Example output paths (per asset):
- `data/raw/market/candles/<ASSET>/`
- `data/raw/news/<ASSET>/`
- `data/processed/scored_news/<ASSET>/`
- `data/processed/sentiment_daily/<ASSET>/`
- `data/processed/technical_indicators/<ASSET>/`
- `data/processed/fundamentals/<ASSET>/`
- `data/processed/dataset_tft/<ASSET>/`

Each step also writes a JSON report under a `reports/` subfolder.

---

## Docs
Recommended starting points:
- `docs/GETTING_STARTED.md`
- `docs/RUNNING_PIPELINE.md`
- `docs/TROUBLESHOOTING.md`

---

## References
- Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
- Martin, R. C. (2017). Clean Architecture.

---

## Contact
Marcelo Santos  
marcelo.santos.c@grad.ufsc.br  
Engenharia Mecatronica - UFSC
