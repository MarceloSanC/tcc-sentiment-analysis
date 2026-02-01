# Etapas MVP

- [x] **Etapa 0: Setup e Valida??o do Ambiente**

**Objetivo**: Garantir que o ambiente est? configurado, documentado e reproduz?vel.

**Componentes**:

- `setup.ps1`, `Makefile`, `pyproject.toml`, `GIT_GUIDE.md`, `docs/`
- Ambiente virtual com depend?ncias (`torch`, `pytorch-forecasting`, `yfinance`, `pytest`, `ruff`, `black`)

**Crit?rio de aceite**:

- [x] `make test` roda sem erro (mesmo com 0 testes)
- [x] `make format` e `make lint` executam sem falha
- [x] `python -c "import pytorch_forecasting"` funciona

**Justificativa (TCC)**:

> Reprodutibilidade ? condi??o necess?ria para valida??o cient?fica (Peng, 2011). A automa??o via Makefile e configura??o declarativa em pyproject.toml garantem que o experimento possa ser replicado por terceiros.

---

- [x] **Etapa 1: Dom?nio (Entidades e Interfaces)**

**Objetivo**: Estabelecer o n?cleo conceitual do sistema, independente de tecnologia.

**Componentes**:

- [x] `src/entities/candle.py` (`Candle`)
- [x] `src/entities/news_article.py` (`NewsArticle`)
- [x] `src/entities/scored_news_article.py` (`ScoredNewsArticle`)
- [x] `src/entities/daily_sentiment.py` (`DailySentiment`)
- [x] `src/entities/technical_indicator_set.py` (`TechnicalIndicatorSet`)
- [x] Interfaces: `CandleRepository`, `NewsRepository`, `ScoredNewsRepository`, `SentimentModel`, `TechnicalIndicatorRepository`

**Crit?rio de aceite**:

- [x] Entidades s?o `@dataclass(frozen=True)`
- [x] Interfaces s?o abstratas (`ABC`) com m?todos assinados
- [x] Nenhuma depend?ncia externa (ex: `pandas`, `requests`) nas camadas internas

**Justificativa (TCC)**:

> A Clean Architecture (Martin, 2017) prop?e que o dom?nio seja independente de frameworks e infraestrutura. Isso aumenta a testabilidade e facilita a manuten??o ? crit?rios essenciais em sistemas de decis?o financeira (IEEE Std 1012-2016).

---

- [x] **Etapa 2: Coleta e Persist?ncia de Pre?os (OHLCV)**

**Objetivo**: Coletar e persistir candles di?rios de forma robusta, test?vel e idempotente.

**Componentes**:

- [x] `src/adapters/yfinance_candle_fetcher.py`
- [x] `src/adapters/parquet_candle_repository.py`
- [x] `src/use_cases/fetch_candles_use_case.py`
- [x] `src/main_candles.py`
- [x] `config/data_sources.yaml`

**Crit?rio de aceite**:

- [x] `python -m src.main_candles --asset AAPL` gera `data/raw/market/candles/{ASSET}/candles_{ASSET}_1d.parquet`
- [x] Colunas: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- [x] Tipos coerentes (`float32`, `int64`, `datetime64[ns, UTC]`)
- [x] Suporta retries e backoff exponencial
- [x] Testes unit?rios com mocks passam

**Justificativa (TCC)**:

> A qualidade dos dados ? o principal fator de sucesso em modelos preditivos (Dhar, 2013). A persist?ncia em Parquet assegura efici?ncia, preserva??o de tipos e compatibilidade com fluxos de ML (Zaharia et al., 2018).

---

- [x] **Etapa 3: Indicadores T?cnicos (a partir de OHLCV)**

**Objetivo**: Calcular indicadores t?cnicos brutos para uso posterior no dataset do TFT.

**Componentes**:

- [x] `src/adapters/technical_indicator_calculator.py`
- [x] `src/adapters/parquet_technical_indicator_repository.py`
- [x] `src/use_cases/technical_indicator_engineering_use_case.py`
- [x] `src/main_technical_indicators.py`

**Crit?rio de aceite**:

- [x] Gera `data/processed/technical_indicators/{ASSET}/technical_indicators_{ASSET}.parquet`
- [x] Formato wide: uma linha por `timestamp`
- [x] Colunas de indicadores consistentes e dtypes v?lidos
- [x] Ordena??o temporal garantida

**Justificativa (TCC)**:

> Indicadores t?cnicos sintetizam padr?es de tend?ncia, momentum e volatilidade e s?o inputs cl?ssicos para modelos de s?ries financeiras. Separar indicadores de features finais evita vazamento conceitual e facilita extens?o futura.

---

- [x] **Etapa 4: Coleta de Not?cias (raw)**

**Objetivo**: Coletar e persistir not?cias brutas do ativo, de forma incremental e audit?vel.

**Componentes**:

- [x] `src/adapters/finnhub_news_fetcher.py`
- [x] `src/adapters/alpha_vantage_news_fetcher.py`
- [x] `src/adapters/parquet_news_repository.py`
- [x] `src/use_cases/fetch_news_use_case.py`
- [x] `src/main_news_dataset.py`

**Crit?rio de aceite**:

- [x] Gera `data/raw/news/{ASSET}/news_{ASSET}.parquet`
- [x] Suporta incremental com cursor e dedup
- [x] Campos de dom?nio validados (URL, published_at UTC, source)

**Justificativa (TCC)**:

> A separa??o entre dados brutos e processados preserva auditabilidade e reprocessamento controlado ? requisito central em pipelines de dados financeiros.

---

- [x] **Etapa 5: Sentimento (scoring + agrega??o di?ria)**

**Objetivo**: Inferir sentimento por not?cia e agregar por dia para uso no dataset do TFT.

**Componentes**:

- [x] `src/adapters/finbert_sentiment_model.py`
- [x] `src/use_cases/infer_sentiment_use_case.py`
- [x] `src/use_cases/sentiment_feature_engineering_use_case.py`
- [x] `src/adapters/parquet_scored_news_repository.py`
- [x] `src/adapters/parquet_daily_sentiment_repository.py`
- [x] `src/domain/services/sentiment_aggregator.py`
- [x] `src/main_sentiment.py`
- [x] `src/main_sentiment_features.py`

**Crit?rio de aceite**:

- [x] `sentiment_score` varia entre `-1.0` (negativo) e `+1.0` (positivo)
- [x] Scoring gera `data/processed/scored_news/{ASSET}/scored_news_{ASSET}.parquet`
- [x] Reprocessamento ? idempotente por `article_id`
- [x] Agrega??o di?ria persiste `data/processed/sentiment_daily/{ASSET}/daily_sentiment_{ASSET}.parquet`
- [x] Cada dia cont?m `sentiment_score` agregado e `news_volume`
- [x] Agrega??o di?ria determin?stica (mesmo input ? mesmo output)

**Justificativa (TCC)**:

> Estudos emp?ricos demonstram que sentimento coletivo explica parte da volatilidade n?o capturada por indicadores t?cnicos (Bollen et al., 2011; Oliveira et al., 2023). A agrega??o di?ria reduz ru?do e alinha horizonte com candles di?rios.

---

- [ ] **Etapa 6: Indicadores Fundamentalistas**

**Objetivo**: Coletar indicadores fundamentalistas para enriquecer o dataset do TFT.

**Componentes** (planejados):

- [ ] Adapter de fundamentals (ex: Alpha Vantage, Finnhub ou outro provedor)
- [ ] Reposit?rio parquet dedicado (raw/processed)
- [ ] Use case e orquestrador CLI

**Crit?rio de aceite**:

- [ ] Dataset fundamentalista salvo com schema est?vel
- [ ] Alinhamento temporal com candles e sentimento

**Justificativa (TCC)**:

> Vari?veis fundamentalistas adicionam informa??o macro e microecon?mica ao modelo, complementando sinais t?cnicos e de sentimento.

---

- [ ] **Etapa 7: Pr?-processamento e Dataset TFT**

**Objetivo**: Unificar candles, indicadores t?cnicos, sentimento agregado e fundamentals em um dataset de treino do TFT.

**Componentes** (planejados):

- [ ] `FeatureAssembler` (join e alinhamento temporal)
- [ ] Normaliza??o/escala por janela temporal
- [ ] Gera??o de alvo (`t+1` retorno) e lags
- [ ] Persist?ncia `data/processed/dataset_tft/{ASSET}/dataset_tft_{ASSET}.parquet`

**Crit?rio de aceite**:

- [ ] Dataset cont?m features sincronizadas e alvo
- [ ] Sem leakage temporal
- [ ] Schema consistente e validado

**Justificativa (TCC)**:

> A prepara??o correta do dataset evita vieses de look-ahead e garante que o TFT receba sinais coerentes no tempo.

---

- [ ] **Etapa 8: Treinamento do TFT e An?lise de Features**

**Objetivo**: Treinar o modelo e produzir an?lises de contribui??o de features.

**Componentes** (planejados):

- [ ] Adapter de treino TFT (pytorch-forecasting)
- [ ] Use case de treino e valida??o temporal
- [ ] Rotinas de explainability (ex: SHAP ou aten??o do TFT)

**Crit?rio de aceite**:

- [ ] Treina modelo com alvo `retorno_t+1`
- [ ] Salva checkpoint e artefatos (scalers, config)
- [ ] Relat?rio de import?ncia das features

**Justificativa (TCC)**:

> A interpretabilidade ? requisito em finan?as para validar se o modelo aprendeu rela??es plaus?veis, n?o apenas correla??es esp?rias.

---

- [ ] **Etapa 9: Infer?ncia do Modelo em Novos Dados**

**Objetivo**: Executar infer?ncia com o modelo treinado em dados novos e registrar outputs.

**Componentes** (planejados):

- [ ] Pipeline de infer?ncia (candles ? features ? previs?o)
- [ ] Persist?ncia de previs?es e metadados

**Crit?rio de aceite**:

- [ ] Gera previs?es para janelas recentes
- [ ] Loga confian?a/quantis do TFT

**Justificativa (TCC)**:

> A infer?ncia operacional valida o uso do modelo em ambiente pr?ximo ao real e permite avalia??o cont?nua de performance.
