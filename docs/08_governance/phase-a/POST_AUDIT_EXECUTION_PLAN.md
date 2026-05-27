---
title: Post-Audit Execution Plan
scope: Plano temporario e de alto nivel para orientar as etapas apos o A_code_audit ate o fechamento do escopo experimental do TCC/artigo. Nao substitui pre-registro, runbooks, checklists ou contratos tecnicos detalhados.
update_when:
  - A_code_audit for concluido
  - criterio de passagem para Fase B mudar
  - escopo de ativos da rodada confirmatoria mudar
  - pre-registro da Fase B for criado ou atualizado
  - Fase B ou Fase C forem concluidas
  - todos os documentos canonicos das etapas 0-8 forem criados (sinal para arquivar este plano)
canonical_for: [post_audit_plan, phase_b_preparation_plan, post_phase_a_execution]
---

# Post-Audit Execution Plan

Status: temporario e sujeito a revisao.

Este documento mapeia, em alto nivel, o caminho esperado apos a conclusao do
`docs/07_reports/phase-gates/phase-a/A_code_audit.md`. Ele existe para evitar perda de
contexto entre a auditoria pre-experimento e a rodada confirmatoria.

Este documento **nao** e:
- pre-registro da Fase B;
- especificacao final de baselines;
- runbook de execucao;
- checklist tecnico detalhado;
- fonte final para escrita do TCC.

Cada etapa abaixo deve ser destrinchada e especificada no documento canonico
apropriado quando o projeto chegar naquela etapa.

---

## 1) Decisao de escopo: modelo single-asset, sistema multi-asset

O projeto deve distinguir tres niveis:

- **Modelo:** single-asset. Cada modelo TFT e treinado, avaliado e versionado para
  um ativo especifico.
- **Pipeline/sistema:** multi-asset. A mesma pipeline deve conseguir processar
  dados, treinar modelos, persistir predicoes e atualizar metricas para diferentes
  ativos.
- **Estudo/TCC:** AAPL e o ativo piloto inicial. A inclusao de outros ativos na
  analise final depende de criterios pre-registrados de disponibilidade, qualidade
  de dados, cobertura OOS e comparabilidade.

Consequencia:
- Resultados de um modelo treinado para um ativo nao devem ser generalizados para
  outros ativos.
- Se a Fase B incluir mais de um ativo, cada ativo deve ter coorte, artefatos,
  metricas e conclusoes identificaveis.
- Qualquer claim agregado multi-asset exige regra de agregacao pre-registrada.

---

## 2) Sequencia apos concluir o A_code_audit

### Etapa 0 - Fechar o gate da Fase A

Fonte atual: `docs/07_reports/phase-gates/phase-a/A_code_audit.md`

Saida esperada:
- M1-M7 com veredito `GREEN` ou `YELLOW` com acao concluida.
- Nenhum `RED` aberto.
- Gaps bloqueantes convertidos em PRs pequenos e rastreaveis.

Nao iniciar Fase B enquanto houver `RED` ou `YELLOW` com acao pendente.

### Etapa 1 - Corrigir gaps pos-auditoria

Fonte primaria futura: PRs especificos + `A_code_audit.md`

Escopo:
- corrigir problemas encontrados em M1-M7;
- manter mudancas pequenas, auditaveis e proporcionais ao achado;
- evitar refator preventivo sem relacao com o gate.

Status atual: mapeado em alto nivel; detalhes dependem dos achados reais.

### Etapa 1.5 - Isolar coorte canonica para a Fase B

Fonte primaria futura:
- `data/analytics/selection/frozen_candidates_round1.csv` (ou equivalente)
- referencia do `parent_sweep_id` da Fase B + `frozen_timestamp`

Escopo:
- definir o conjunto de runs/coortes elegiveis para a rodada confirmatoria;
- declarar regra de filtro reproduzivel para excluir runs pre-ScopeSpec, runs
  degenerados anteriores ao fix `f7901a4` e qualquer outro grupo identificado
  como nao-comparavel pelo M6;
- congelar `frozen_timestamp` antes de qualquer calculo confirmatorio;
- registrar a decisao no pre-registro da Fase B (Etapa 4).

Criterio de saida: filtro de coorte aplicavel via `ScopeSpec` retorna apenas o
conjunto declarado, sem mistura silenciosa com runs historicos.

Status atual: depende dos achados de M6; decisao concreta tomada apos auditoria.

### Etapa 2 - Implementar e persistir baselines comparaveis

Fontes atuais:
- `docs/04_evaluation/BASELINES.md`
- `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`
- `docs/03_modeling/SWEEPS_AND_SELECTION.md`

Escopo esperado:
- random walk/zero-return;
- media historica;
- AR(1);
- EWMA-vol e/ou quantis historicos para baseline probabilistico;
- persistencia no mesmo grao de `fact_oos_predictions`;
- identificacao clara por `run_id`, `parent_sweep_id`, `asset`, `split`,
  `horizon` e `target_timestamp`.

Status atual: contrato conceitual mapeado; plano de implementacao ainda nao
detalhado.

### Etapa 3 - Decidir escopo de ativos da rodada confirmatoria

Fonte primaria futura: pre-registro da Fase B.

Decisao minima:
- AAPL permanece como ativo piloto.

Se outros ativos reais forem incluidos:
- criterios de inclusao devem ser definidos antes da rodada;
- cada ativo deve ter dados suficientes e cobertura temporal comparavel;
- disponibilidade das familias de features deve ser documentada;
- a analise deve separar resultados por ativo;
- qualquer analise agregada deve ter regra pre-declarada.

Status atual: decisao conceitual definida (modelo single-asset, sistema
multi-asset); criterios concretos de inclusao ainda nao mapeados.

### Etapa 4 - Criar pre-registro da Fase B

Fonte futura:
- `docs/08_governance/preregistration_round1_<date>.md`

Conteudo esperado:
- ativos incluidos;
- candidato unico por ativo;
- lista de features e regras de poda;
- hiperparametros;
- seeds, folds e janelas;
- baseline primario e complementares;
- metrica primaria;
- criterios de calibracao;
- thresholds de relevancia pratica;
- regras de decisao por horizonte e por ativo;
- tratamento de h+30;
- politica para resultados refutarem H1/H2a/H2b/H3.

Imutabilidade (criterio anti-p-hacking):
- o commit do pre-registro deve preceder o primeiro commit de implementacao da
  Fase B;
- mudancas pos-commit so via emenda datada no proprio arquivo, nunca por
  sobrescrita silenciosa;
- o commit hash do pre-registro deve ser referenciado nos artefatos finais da
  Fase B para rastreabilidade.

Status atual: previsto no roadmap; documento ainda nao existe.

### Etapa 5 - Executar Fase B confirmatoria

Fontes atuais:
- `docs/00_overview/STRATEGIC_DIRECTION.md`
- `docs/03_modeling/MULTI_HORIZON.md`
- `docs/03_modeling/SWEEPS_AND_SELECTION.md`
- `docs/04_evaluation/STATISTICAL_TESTS.md`
- `docs/04_evaluation/CALIBRATION_AND_RISK.md`

Saidas esperadas:
- predicoes OOS persistidas para candidato e baselines;
- gold artifacts atualizados;
- comparacao pareada por `target_timestamp`;
- DM/HAC/HLN e Holm-Bonferroni quando aplicavel;
- tabelas por ativo, split e horizonte;
- evidencia registrada em `docs/07_reports/living-paper/evidence_log.md`.

Status atual: metodologia mapeada; execucao e configs finais dependem do
pre-registro.

### Etapa 6 - Atualizar resultados confirmatorios

Fontes a atualizar:
- `docs/07_reports/living-paper/30_results_and_analysis.md`
- `docs/07_reports/living-paper/evidence_log.md`
- `docs/07_reports/living-paper/tcc_mapping.md`

Status atual: parcialmente mapeado; depende dos resultados da Fase B.

### Etapa 7 - Especificar e executar Fase C

Fontes atuais:
- `docs/04_evaluation/EXPLAINABILITY.md`
- `docs/00_overview/STRATEGIC_DIRECTION.md`

Escopo esperado:
- VSN weights agregados;
- permutation importance por familia;
- ablation explicativa;
- estabilidade por seed/fold quando aplicavel;
- exemplos locais apenas ilustrativos.

Status atual: metodologia conceitual mapeada; plano operacional e artefatos ainda
nao detalhados.

### Etapa 8 - Fechar escopo experimental do TCC/artigo

Condicoes minimas:
- Fase A concluida;
- pre-registro commitado;
- Fase B executada e analisada;
- Fase C executada ou explicitamente rebaixada de escopo;
- `30_results_and_analysis.md` atualizado;
- `40_limitations_and_conclusion.md` atualizado;
- `tcc_mapping.md` revisado.

Somente depois disso o texto LaTeX deve ser tratado como etapa de consolidacao,
nao como fonte primaria de decisoes.

---

## 3) Itens ainda nao detalhados

Os itens abaixo estao mapeados como proximas etapas, mas ainda precisam de
especificacao propria quando forem atacados:

- plano tecnico de implementacao dos baselines;
- criterios concretos para incluir ativos alem de AAPL;
- matriz final de seeds, folds, janelas e horizontes;
- regra de agregacao multi-asset, se houver;
- formato dos artefatos da Fase C;
- comandos finais de execucao da Fase B;
- cronograma de escrita apos resultados.

---

## 4) Politica de mudanca deste documento

Este documento deve ser atualizado de forma leve, apenas para manter a ordem de
execucao e dependencias claras. Detalhes operacionais devem migrar para o
documento canonico apropriado quando a etapa correspondente comecar.

Quando o pre-registro da Fase B estiver criado e commitado, este plano deve ser
revisado: partes dele podem ser marcadas como concluidas, substituidas por links
ou arquivadas.

### Ciclo de vida e morte planejada

Este e um documento **temporario por design**. Seu papel e servir como ponte
entre o A_code_audit e os documentos canonicos das fases seguintes. Quando
todos os documentos canonicos das Etapas 0-8 existirem (em particular: pre-
registro commitado, runbooks de baseline e Fase B, especificacao da Fase C,
30_results e 40_limitations consolidados), este plano deve ser:

1. revisado uma ultima vez para garantir que nenhum conteudo unico foi perdido
   (em especial a distincao single-asset modelo / multi-asset sistema da
   Secao 1, que deve migrar para `STRATEGIC_DIRECTION.md` ou `PROJECT_OVERVIEW.md`
   antes da remocao);
2. arquivado em `docs/_archive/` ou removido com nota no commit explicando a
   superseção pelos documentos canonicos.

Nao manter este documento ativo apos a conclusao da Fase B/C. Plano temporario
que vira permanente por inercia perde valor e cria risco de divergencia com os
documentos canonicos.
