.PHONY: help install test lint format type-check clean check-python-version

check-python-version:
	@python -c "import sys; v = sys.version_info; assert v.major == 3 and v.minor >= 13, f'Python 3.13+ required, got {v.major}.{v.minor}'; print('Python 3.13+ OK')"

help:
	@echo "Comandos disponíveis:"
	@echo "  install     		- Instala dependências (dev)"
	@echo "  test               - Testes (sem integração externa)"
	@echo "  test-integration   - Testes com APIs externas"
	@echo "  test-all           - Todos os testes"
	@echo "  lint        		- Verifica estilo e imports"
	@echo "  format      		- Formata código com black + ruff"
	@echo "  type-check  		- Verifica anotações de tipo"
	@echo "  clean       		- Remove arquivos temporários"
	@echo "  run-candles        - Executa main_candles (ex: make run-candles ASSET=AAPL)"
	@echo "  run-news-raw       - Executa main_news_dataset (ex: make run-news-raw ASSET=AAPL)"
	@echo "  run-sentiment      - Executa main_sentiment (ex: make run-sentiment ASSET=AAPL)"
	@echo "  run-sentiment-feat - Executa main_sentiment_features (ex: make run-sentiment-feat ASSET=AAPL)"
	@echo "  run-indicators     - Executa main_technical_indicators (ex: make run-indicators ASSET=AAPL)"
	@echo "  run-all            - Executa todos os orquestradores (ex: make run-all ASSET=AAPL)"

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ \
		-m "not integration" \
		--cov=src \
		--cov-report=term-missing

test-integration:
	python -m pytest tests/ -m integration

test-all:
	python -m pytest tests/ --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff check --diff  # mostra o que seria corrigido

format:
	black --config pyproject.toml src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f .coverage
	rm -rf htmlcov/ 2>/dev/null || true
.PHONY: run-candles run-news-raw run-sentiment run-sentiment-feat run-indicators run-all

run-candles:
	python -m src.main_candles --asset $(ASSET)

run-news-raw:
	python -m src.main_news_dataset --asset $(ASSET)

run-sentiment:
	python -m src.main_sentiment --asset $(ASSET)

run-sentiment-feat:
	python -m src.main_sentiment_features --asset $(ASSET)

run-indicators:
	python -m src.main_technical_indicators --asset $(ASSET)

run-all:
	$(MAKE) run-candles ASSET=$(ASSET)
	$(MAKE) run-news-raw ASSET=$(ASSET)
	$(MAKE) run-sentiment ASSET=$(ASSET)
	$(MAKE) run-sentiment-feat ASSET=$(ASSET)
	$(MAKE) run-indicators ASSET=$(ASSET)
