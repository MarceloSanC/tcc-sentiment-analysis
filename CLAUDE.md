# CLAUDE.md

Este arquivo é apenas um **ponteiro durável**. Ele não contém regras próprias e
não deve duplicar conteúdo operacional — toda a fonte de verdade vive nos docs
canônicos referenciados abaixo. Se algo aqui conflitar com eles, os docs
canônicos prevalecem.

## Sempre, no início de cada tarefa

1. **Carregue o baseline** — `docs/ai/AGENT_CORE.md`.
   Regras universais (always-on): data rules inegociáveis, scope governance,
   documentation sync, política de resposta especialista e ordem de prioridade.

2. **Consulte o índice** — `docs/ai/INDEX.md`.
   Entrypoint canônico: catálogo de skills em `docs/ai/skills/*`, ordem de
   carregamento e fronteiras de source-of-truth.

3. **Selecione skills** — carregue apenas as skills locais aplicáveis à tarefa
   atual (evite misturar contexto de skills irrelevantes).

## Rastreabilidade (por `AGENT_CORE`)

- Declare, **por passo**, quais skills foram usadas.
- `AGENT_CORE` é baseline implícito e não precisa ser repetido.
- Se nenhuma skill específica se aplicar, declare
  `none specific (AGENT_CORE only)`.

## Ordem de prioridade (resumo — detalhe em `AGENT_CORE`)

1. Restrições de runtime (system/developer).
2. `docs/ai/AGENT_CORE.md`.
3. Skill da tarefa (`docs/ai/skills/<name>/SKILL.md`).
4. Instruções pontuais do prompt.

## Roteamento documental adicional

- Manifesto de documentação do produto: `docs/INDEX.md`.
- Direção estratégica: `docs/00_overview/STRATEGIC_DIRECTION.md`.
- Runbooks operacionais: `docs/06_runbooks/*`.
- Governança / versionamento / commits / PRs: `docs/08_governance/GOVERNANCE_AND_VERSIONING.md`.

---

> Equivalente Claude Code do `AGENTS.md` deste repositório. Mantenha-o mínimo:
> ao adicionar/renomear/remover skills ou mudar a ordem de carregamento, atualize
> `docs/ai/INDEX.md` — **não** este arquivo.
