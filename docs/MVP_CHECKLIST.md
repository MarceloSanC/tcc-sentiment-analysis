# Etapas MVP

- [x] **Etapa 0: Setup e Validação do Ambiente**

**Objetivo**: Garantir que o ambiente está configurado, documentado e reproduzível.

**Componentes**:

- `setup.ps1`, `Makefile`, `pyproject.toml`, `GIT_GUIDE.md`, `docs/`
- Ambiente virtual com dependências (`torch`, `pytorch-forecasting`, `yfinance`, `pytest`, `ruff`, `black`)

**Critério de aceite**:

- [x] `make test` roda sem erro (mesmo com 0 testes)
- [x] `make format` e `make lint` executam sem falha
- [x] `python -c "import pytorch_forecasting"` funciona

**Justificativa (TCC)**:

> Reprodutibilidade é condição necessária para validação científica (Peng, 2011). A automação via Makefile e configuração declarativa em pyproject.toml garantem que o experimento possa ser replicado por terceiros.

---

- [x] **Etapa 1: Domínio (Entidades e Interfaces)**

**Objetivo**: Estabelecer o núcleo conceitual do sistema, independente de tecnologia.

**Componentes**:

- [x] `src/entities/candle.py` (`Candle`)
- [x] `src/entities/news_article.py` (`NewsArticle`)
- [x] `src/entities/scored_news_article.py` (`ScoredNewsArticle`)
- [x] `src/entities/daily_sentiment.py` (`DailySentiment`)
- [x] `src/entities/technical_indicator_set.py` (`TechnicalIndicatorSet`)
- [x] Interfaces: `CandleRepository`, `NewsRepository`, `ScoredNewsRepository`, `SentimentModel`, `TechnicalIndicatorRepository`

**Critério de aceite**:

- [x] Entidades são `@dataclass(frozen=True)`
- [x] Interfaces são abstratas (`ABC`) com métodos assinados
- [x] Nenhuma dependência externa (ex: `pandas`, `requests`) nas camadas internas

**Justificativa (TCC)**:

> A Clean Architecture (Martin, 2017) propõe que o domínio seja independente de frameworks e infraestrutura. Isso aumenta a testabilidade e facilita a manutenção — critérios essenciais em sistemas de decisão financeira (IEEE Std 1012-2016).

---

- [x] **Etapa 2: Coleta e Persistência de Preços (OHLCV)**

**Objetivo**: Coletar e persistir candles diários de forma robusta, testável e idempotente.

**Componentes**:

- [x] `src/adapters/yfinance_candle_fetcher.py`
- [x] `src/adapters/parquet_candle_repository.py`
- [x] `src/use_cases/fetch_candles_use_case.py`
- [x] `src/main_candles.py`
- [x] `config/data_sources.yaml`

**Critério de aceite**:

- [x] `python -m src.main_candles --asset AAPL` gera `data/raw/market/candles/{ASSET}/candles_{ASSET}_1d.parquet`
- [x] Colunas: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- [x] Tipos coerentes (`float32`, `int64`, `datetime64[ns, UTC]`)
- [x] Suporta retries e backoff exponencial
- [x] Testes unitários com mocks passam

**Justificativa (TCC)**:

> A qualidade dos dados é o principal fator de sucesso em modelos preditivos (Dhar, 2013). A persistência em Parquet assegura eficiência, preservação de tipos e compatibilidade com fluxos de ML (Zaharia et al., 2018).

---

- [x] **Etapa 3: Indicadores Técnicos (a partir de OHLCV)**

**Objetivo**: Calcular indicadores técnicos brutos para uso posterior no dataset do TFT.

**Componentes**:

- [x] `src/adapters/technical_indicator_calculator.py`
- [x] `src/adapters/parquet_technical_indicator_repository.py`
- [x] `src/use_cases/technical_indicator_engineering_use_case.py`
- [x] `src/main_technical_indicators.py`

**Critério de aceite**:

- [x] Gera `data/processed/technical_indicators/{ASSET}/technical_indicators_{ASSET}.parquet`
- [x] Formato wide: uma linha por `timestamp`
- [x] Colunas de indicadores consistentes e dtypes válidos
- [x] Ordenação temporal garantida

**Justificativa (TCC)**:

> Indicadores técnicos sintetizam padrões de tendência, momentum e volatilidade e são inputs clássicos para modelos de séries financeiras. Separar indicadores de features finais evita vazamento conceitual e facilita extensão futura.

---

- [x] **Etapa 4: Coleta de Notícias (raw)**

**Objetivo**: Coletar e persistir notícias brutas do ativo, de forma incremental e auditável.

**Componentes**:

- [x] `src/adapters/finnhub_news_fetcher.py`
- [x] `src/adapters/alpha_vantage_news_fetcher.py`
- [x] `src/adapters/parquet_news_repository.py`
- [x] `src/use_cases/fetch_news_use_case.py`
- [x] `src/main_news_dataset.py`

**Critério de aceite**:

- [x] Gera `data/raw/news/{ASSET}/news_{ASSET}.parquet`
- [x] Suporta incremental com cursor e dedup
- [x] Campos de domínio validados (URL, published_at UTC, source)

**Justificativa (TCC)**:

> A separação entre dados brutos e processados preserva auditabilidade e reprocessamento controlado — requisito central em pipelines de dados financeiros.

---

- [x] **Etapa 5: Sentimento (scoring + agregação diária)**

**Objetivo**: Inferir sentimento por notícia e agregar por dia para uso no dataset do TFT.

**Componentes**:

- [x] `src/adapters/finbert_sentiment_model.py`
- [x] `src/use_cases/infer_sentiment_use_case.py`
- [x] `src/use_cases/sentiment_feature_engineering_use_case.py`
- [x] `src/adapters/parquet_scored_news_repository.py`
- [x] `src/adapters/parquet_daily_sentiment_repository.py`
- [x] `src/domain/services/sentiment_aggregator.py`
- [x] `src/main_sentiment.py`
- [x] `src/main_sentiment_features.py`

**Critério de aceite**:

- [x] `sentiment_score` varia entre `-1.0` (negativo) e `+1.0` (positivo)
- [x] Scoring gera `data/processed/scored_news/{ASSET}/scored_news_{ASSET}.parquet`
- [x] Reprocessamento é idempotente por `article_id`
- [x] Agregação diária persiste `data/processed/sentiment_daily/{ASSET}/daily_sentiment_{ASSET}.parquet`
- [x] Cada dia contém `sentiment_score` agregado e `news_volume`
- [x] Agregação diária determinística (mesmo input → mesmo output)
- [x] Validação explícita de causalidade: sentimento diário usa apenas notícias do próprio dia (sem notícia futura)

**Justificativa (TCC)**:

> Estudos empíricos demonstram que sentimento coletivo explica parte da volatilidade não capturada por indicadores técnicos (Bollen et al., 2011; Oliveira et al., 2023). A agregação diária reduz ruído e alinha horizonte com candles diários.

---

- [x] **Etapa 6: Indicadores Fundamentalistas**

**Objetivo**: Coletar indicadores fundamentalistas para enriquecer o dataset do TFT.

**Componentes**:

- [x] Adapter de fundamentals (Alpha Vantage)
- [x] Repositório parquet dedicado (processed)
- [x] Use case e orquestrador CLI (`main_fundamentals`)

**Critério de aceite**:

- [x] Dataset fundamentalista salvo com schema estável
- [x] Alinhamento temporal com candles e sentimento

**Justificativa (TCC)**:

> Variáveis fundamentalistas adicionam informação macro e microeconômica ao modelo, complementando sinais técnicos e de sentimento.

---

- [x] **Etapa 7: Pré-processamento e Dataset TFT**

**Objetivo**: Unificar candles, indicadores técnicos, sentimento agregado e fundamentals em um dataset de treino do TFT.

**Componentes**:

- [x] `BuildTFTDatasetUseCase` (join e alinhamento temporal)
- [x] Normalização/escala por janela temporal
- [x] Geração de alvo (`t+1` retorno) e lags
- [x] Persistência `data/processed/dataset_tft/{ASSET}/dataset_tft_{ASSET}.parquet`

**Critério de aceite**:

- [x] Dataset contém features sincronizadas e alvo
- [x] Sem leakage temporal
- [x] Schema consistente e validado
- [ ] Política explícita de missing por feature (ex.: `news_volume=0`) 
- [x] Validador de warmup para detectar e alertar `null` iniciais por feature no período de treino

**Justificativa (TCC)**:

> A preparação correta do dataset evita vieses de look-ahead e garante que o TFT receba sinais coerentes no tempo.

---

- [ ] **Etapa 8: Treinamento do TFT e Análise de Features**

**Objetivo**: Treinar o modelo e produzir análises de contribuição de features.

**Componentes** (planejados):

- [x] Adapter de treino TFT (pytorch-forecasting)
- [x] Use case de treino e validação temporal
- [ ] Rotinas de explainability (ex: SHAP ou atenção do TFT)
- [x] Estratégia de experimentos para relevância (ablação + permutation importance)

**Critério de aceite**:

- [x] Treina modelo com alvo `retorno_t+1`
- [x] Avalia em test e salva métricas separadas (`train`, `val`, `test`)
- [ ] Salva checkpoint e artefatos (scalers, config, `metadata.json` com split efetivo)
- [x] Relatório de importância das features
- [x] Split temporal (train/val/test) sem leakage
- [ ] Normalização ajustada no treino e aplicada no val/test
- [x] Early stopping + checkpoint do melhor `val_loss`
- [x] Relevância em duas camadas: ablação controlada (baseline vs +sentimento vs +fundamentals) e permutation importance no conjunto de teste
- [ ] Exporta artefatos de análise (`feature_importance.csv`, gráfico comparativo por experimento)

**Justificativa (TCC)**:

> A interpretabilidade é requisito em finanças para validar se o modelo aprendeu relações plausíveis, não apenas correlações espúrias.

---

- [ ] **Etapa 9: Inferência do Modelo em Novos Dados**

**Objetivo**: Executar inferência com o modelo treinado em dados novos e registrar outputs.

**Componentes** (planejados):

- [ ] Pipeline de inferência (candles → features → previsão)
- [ ] Persistência de previsões e metadados

**Critério de aceite**:

- [ ] Gera previsões para janelas recentes
- [ ] Loga confiança/quantis do TFT

**Justificativa (TCC)**:

> A inferência operacional valida o uso do modelo em ambiente próximo ao real e permite avaliação contínua de performance.

