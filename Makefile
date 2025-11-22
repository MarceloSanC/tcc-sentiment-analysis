.PHONY: help install test lint format type-check clean check-python-version

check-python-version:
	@python -c "import sys; v = sys.version_info; assert v.major == 3 and v.minor >= 13, f'Python 3.13+ required, got {v.major}.{v.minor}'; print('Python 3.13+ OK')"

help:
	@echo "Comandos disponíveis:"
	@echo "  install     - Instala dependências (dev)"
	@echo "  test        - Roda testes com cobertura"
	@echo "  lint        - Verifica estilo e imports"
	@echo "  format      - Formata código com black + ruff"
	@echo "  type-check  - Verifica anotações de tipo"
	@echo "  clean       - Remove arquivos temporários"

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff check --diff  # mostra o que seria corrigido

format:
	black src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f .coverage
	rm -rf htmlcov/ 2>/dev/null || true