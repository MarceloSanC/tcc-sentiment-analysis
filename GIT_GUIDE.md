# Guia de Git — TCC Sentiment Analysis

Este guia padroniza o uso do Git no projeto, garantindo histórico limpo, reprodutível e adequado à apresentação.

> **Objetivo**: Commits **atômicos**, **descritivos** e **reversíveis**, alinhados à evolução funcional do sistema.

---

## Branches

| Branch | Propósito | Status |
|--------|-----------|--------|
| `main` | Versão estável | Protegida — só recebe `merge` de PRs testados |
| `feat/sentiment-analysis-v1` | Desenvolvimento da v1.0 (análise de sentimento + correlação) | Ativa — onde novos commits são feitos |
| `chore/setup` | Configuração inicial (pyproject.toml, Makefile, etc.) | Finalizada |

> **Regra**:  
> - Sempre crie uma branch nova para novas funcionalidades ou refatorações grandes.  
> - Nunca commite direto em `main`.

---

## Padrão de mensagens de commit

Usamos **[Conventional Commits](https://www.conventionalcommits.org/)** com ajustes para o TCC:

```
<tipo>(<escopo>): <título curto>
```
```
[opcional] Corpo detalhado (justificativa, impacto, referência teórica).
```
```
[opcional] Footer (ex: "Closes #1", "Refs Artigo Bollen 2011")
```

### Tipos comuns:
| Tipo | Quando usar |
|------|-------------|
| `feat` | Nova funcionalidade |
| `refactor` | Mudança de estrutura **sem alterar comportamento**  |
| `test` | Adição ou melhoria de testes |
| `chore` | Configuração, CI, arquivos de build |
| `docs` | Documentação (README, este guia, comentários) |
| `fix` | Correção de bug |

### Escopos sugeridos:
- `news`, `sentiment`, `pipeline`, `db`, `arch`, `ci`, `deps`

### Exemplos válidos:
```text
feat(sentiment): adiciona inferência com FinBERT via adapter

- FinBERTSentimentModel implementa SentimentModel
- Usa yiyanghkust/finbert-tone com pipeline do transformers
- Retorna sentiment (Positive/Negative/Neutral) + confidence
- Próximo passo: integrar no InferSentimentUseCase

Refs Bollen (2011): uso de sentimento coletivo em previsão
```
```
chore(arch): inicializa estrutura Clean Architecture

- Cria src/{entities,interfaces,use_cases,adapters}
- Adiciona pyproject.toml com black, ruff, pytest
- setup.ps1 para configuração no Windows
```
```
test(use_cases): adiciona testes para FetchNewsUseCase com mock

- Testa busca incremental com base na última data salva
- Mock de NewsRepository e NewsFetcher
- 100% cobertura em lógica de negócio
```

## Fluxo de trabalho recomendado

### 1. Atualizar main antes de começar
```
git switch main
git pull origin main
```

### 2. Criar branch para nova funcionalidade
```
git switch -c vX.Y/nome-da-funcionalidade
```

### 3. Desenvolver e commitar por etapa lógica
```
# Adicione só os arquivos da etapa atual
git add src/use_cases/fetch_news_use_case.py src/interfaces/news_fetcher.py

# Commit com mensagem descritiva
git commit -m "feat(news): implementa FetchNewsUseCase com busca incremental"
```

> Não commitar tudo de uma vez.
Um commit = uma unidade testável de funcionalidade. 

### 4. Sincronizar com main
```
git switch feat/minha-branch
git rebase main      # mantém histórico linear
# ou, se preferir merge:
git merge main
```

### 5. Enviar para GitHub
```
git push -u origin feat/minha-branch
```

### 6. Abrir Pull Request
- Revise o diff por commit.
- Comente decisões de arquitetura.
- Use como base para a seção "Metodologia de Desenvolvimento" do TCC.

## Comandos úteis

| Açao | Comando |
|--|--|
| Ver status |  git status |
| Ver diff do que vai commitar | git diff --staged |
| Salvar alterações temporárias | git stash -u|
| Recuperar stash | git stash pop |
| Histórico legível | git log --oneline --graph |
| Desfazer último commit (mantendo mudanças) | git reset --soft HEAD~1 |
|Desfazer commit + mudanças (cuidado!) | git reset --hard HEAD~1  |