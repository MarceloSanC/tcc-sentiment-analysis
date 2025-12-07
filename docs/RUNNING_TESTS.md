# 🧪 Executando Testes

O projeto usa `pytest` para testes unitários e de integração.

## Rodar todos os testes

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
...
TOTAL                                64      0   100%
```

## Rodar testes específicos

- Por arquivo:
```
python -m pytest tests/unit/test_use_cases/test_fetch_news_use_case.py -v
```

- Por marcação (ex: testes de integração):
```
pytest tests/integration/ -v
```

## Regras de teste

- Testes unitários: `tests/unit/` — não usam rede, banco ou modelo pesado
- Testes de integração: `tests/integration/` — usam APIs reais (marcados com `@pytest.mark.integration`)
- Todos os testes devem ser executáveis offline (exceto os de integração)