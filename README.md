# TCC: Análise de Sentimento em Notícias Financeiras para Previsão de Tendências de Mercado

> **Aluno**: Marcelo Santos  
> **Curso**: Engenharia Mecatrônica — UFSC  
> **Orientador**: [Nome do orientador]  
> **Ano**: 2025  

Este projeto implementa um sistema automatizado para:
- Coletar notícias financeiras de ações (via API **Finnhub**),
- Inferir sentimento (*positivo*, *neutro*, *negativo*) com **FinBERT**,
- Calcular médias móveis de sentimento,
- Analisar correlação entre sentimento e retorno de mercado.

O sistema segue os princípios da **Clean Architecture**, garantindo **desacoplamento**, **testabilidade** e **extensibilidade** — permitindo futuras expansões com análise técnica, modelos de ML, etc.

---

## Estrutura do Projeto (Clean Architecture)
```
src/
├── main.py
├── adapters/
│   ├── finbert_sentiment_model.py
│   ├── finnhub_news_fetcher.py
│   └── sqlite_news_repository.py
├── entities/
│   └── news.py
├── interfaces/
│   ├── news_fetcher.py
│   ├── news_repository.py
│   └── sentiment_model.py
├── services/
│   ├── finbert.py
│   ├── market_data_fetcher.py
│   ├── news_search.py
│   ├── sentiment_aggregator.py
│   └── sentiment_market_analyzer.py
├── tcc_sentiment_analysis.egg-info/
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
└── use_cases/
    ├── fetch_news_use_case.py
    └── infer_sentiment_use_case.py
```

---

## Requisitos

- **Windows 10/11**
- **Python 3.13+** ([download](https://python.org))
- **Git** (opcional, para versionamento)

> *Recomenda-se uso de ambiente virtual.*

---

## Configuração (Passo a Passo)

### 1. Clone ou baixe o projeto

```powershell
git clone https://github.com/seu-usuario/tcc-sentiment-analysis.git
cd tcc-sentiment-analysis
```

### 2. Instale o make

```
winget install GnuWin32.Make
```

### 3. Configure o ambiente

Habilita execução de scripts (só uma vez no PC)
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Configura PATH, Python e ativa o venv (se existir)

```
.\setup.ps1
```

### 4. Crie e ative o ambiente virtual

```
python -m venv .venv
.\setup.ps1
```

### 5. Instale dependências

```
make install
```
ou manualmente:

```
pip install -e ".[dev]"
```

## Como Executar

### Pipeline completo (notícias → sentimento → análise)

```
python src/main.py
```

### Comandos úteis (make)

Roda testes + cobertura
```
make test
```
Formata código ( black + ruff )
```
make format
```
Verifica estilo e imports
```
make lint
```
Valida anotações de tipo ( mypy )
```
make type-check
```
Limpa arquivos temporários
```
make clean
```

## Testes
Testes unitários isolados (sem dependência de rede, banco ou modelo):
```
make test
```
Saída esperada:
```
----------- coverage: platform win32, python 3.13.3 -----------
Name                              Stmts   Miss  Cover
-------------------------------------------------------
src/entities/news.py                 12      0   100%
src/use_cases/fetch_news_use_case.py 28      0   100%
src/use_cases/infer_sentiment_use_case.py 24  0   100%
-------------------------------------------------------
TOTAL                                64      0   100%
```

## Banco de Dados

 - SQLite: data/tcc_sentiment.db
 - Tabela única: news
    - Campos: ticker, published_at, title, source, url, sentiment, confidence
    - sentiment começa como NULL (notícia bruta) e é atualizado após inferência.


## Referências
- Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
- Martin, R. C. (2017). Clean Architecture.

##  Contato

Marcelo Santos
marcelo.santos.c@grad.ufsc.br

Engenharia Mecatrônica — UFSC
