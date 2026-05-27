---
title: Project Overview
scope: Entrada executiva curta. Resumo de objetivo, escopo (em/fora), entregaveis com status, criterios de sucesso, estrutura de diretorios e proximos passos. Para detalhes estrategicos, ver STRATEGIC_DIRECTION.md.
update_when:
  - status de entregaveis mudar (tabela de entregaveis)
  - criterios de sucesso forem redefinidos
  - estrutura de diretorios docs/ mudar
  - escopo (em/fora) for alterado em STRATEGIC_DIRECTION
canonical_for: [project_overview, deliverables_status, docs_structure_summary]
---

# Project Overview

Este e o ponto de entrada da documentacao. Leia este arquivo antes de qualquer outro.

Ultima atualizacao: 2026-05-01.

---

## Objetivo

Avaliar rigorosamente a calibracao probabilistica de modelos Temporal Fusion
Transformer (TFT) especificos por ativo, aplicados a previsao multi-horizonte de
retornos, e caracterizar a contribuicao relativa de familias de indicadores
(preco, tecnico, fundamentalista, sentimento) para a previsao — sob protocolo
estatistico explicito, com pre-registro versionado e artefatos reproduziveis sem
retrain. AAPL e o ativo piloto inicial; a pipeline deve suportar repeticao do
mesmo protocolo para outros ativos com dados suficientes.

O projeto serve simultaneamente como:
- Base para um TCC de graduacao/pos-graduacao em ciencia de dados / financas quantitativas.
- Plataforma de experimentacao reproducivel em forecasting financeiro com TFT.

**Claim central:** Nao "provar que TFT vence o mercado", mas documentar com rigor o
que o modelo faz, o que cada familia de indicadores contribui, e onde o modelo falha.
Veja `docs/00_overview/STRATEGIC_DIRECTION.md` §4.4 para claims aceitos vs proibidos.

---

## Escopo

**Em escopo:**
- Previsao supervisionada de retorno diario com modelos single-asset por ativo.
- Pipeline multi-asset capaz de processar dados e treinar modelos especificos
  para diferentes ativos.
- AAPL como ativo piloto inicial para validar pipeline, auditoria e protocolo.
- Horizontes h+1 e h+7 como principais; h+30 como suplementar se N efetivo suficiente.
- Previsao quantilica (p10, p50, p90) com guardrail monotonico e auditoria before/after.
- Comparacao com baseline primario pre-declarado e baselines simples complementares
  (random walk/zero-return, media historica, AR(1), EWMA-vol e/ou quantis historicos).
- Analise de contribuicao de features via VSN weights, permutation importance e ablation.
- Analytics Store reproduzivel com rastreabilidade de experimentos sem retrain.

**Fora de escopo (neste projeto):**
- Generalizacao para ativos ou periodos nao avaliados sob protocolo pre-registrado.
- Inferencia causal sobre impacto de indicadores no retorno.
- Estrategia de trading, otimizacao de portfolio, inferencia intradiaria.
- Classificacao de regimes de mercado (Wyckoff, HMM) — registrado como future work.
- Comparacao exaustiva de arquiteturas de modelos profundos.

---

## Entregaveis

| Entregavel | Descricao | Status |
|---|---|---|
| Pipeline de dados | Ingestao, features (PRICE/TECHNICAL/SENTIMENT/FUNDAMENTAL), dataset TFT | Operacional |
| Treinamento TFT | Sweep exploratório Round 0; candidato unico Round 1 | Round 0 concluido |
| Analytics Store | Parquet silver + DuckDB gold; metricas reproduziveis sem retrain | Operacional |
| Baselines | Baseline primario pre-declarado + baselines complementares conforme `04_evaluation/BASELINES.md` | A implementar (Fase A) |
| Pre-registro | Documento versionado antes da rodada confirmatoria | A criar (Fase A) |
| Auditoria de leakage | Verificacao de `time_varying_known_reals`, scalers, HPO vs OOS | A executar (Fase A) |
| Analise probabilistica | Calibracao marginal por quantil, PICP intervalar, MPIW, reliability diagram, DM/MCS, Holm-Bonferroni | A executar (Fase B) |
| Analise de contribuicao | VSN, permutation importance, ablation explicativa por familia | A executar (Fase C) |
| Plano pos-auditoria | Mapa temporario das etapas apos A_code_audit ate fechamento do escopo experimental | Criado |
| TCC / Artigo | Documento academico com resultados, limitacoes e conclusoes defensaveis | Em escrita |

---

## Criterios de sucesso

- Candidato TFT produz calibracao marginal por quantil dentro de tolerancia
  pre-declarada para p10/p50/p90 e PICP intervalar adequado para p10-p90 em pelo
  menos um horizonte h in {1, 7} (H1 calibracao).
- Pinball loss menor que o baseline primario pre-declarado com significancia
  estatistica em pelo menos um horizonte (H2a superioridade primaria).
- Pinball loss menor que todos os baselines simples pre-declarados com
  significancia estatistica em pelo menos um horizonte (H2b evidencia forte
  adicional).
- Contribuicao de familias de indicadores e heterogenea e detectavel consistentemente
  em >= 2 dos 3 metodos de analise (H3 heterogeneidade de contribuicao).
- Todos os resultados reproduziveis a partir de artefatos persistidos, sem retrain.
- Decisao final rastreavel por `run_id`, `model_version`, `feature_set_hash`, `horizon`.

---

## Estrutura de diretorios de documentacao

```
docs/
  00_overview/          <- COMECE AQUI
    PROJECT_OVERVIEW.md   <- este arquivo
    STRATEGIC_DIRECTION.md <- objetivo, pivo metodologico, roadmap, claims
    GLOSSARY.md           <- definicoes de termos tecnicos recorrentes

  01_architecture/      <- Clean Architecture, data flow, ADRs
  02_data/              <- fontes, contratos, feature sets, quality gates
  03_modeling/          <- parametros de treino, inferencia, multi-horizonte
  04_evaluation/        <- metricas, baselines, testes, calibracao, explicabilidade
  05_checklists/        <- gates de implementacao e validacao
  06_runbooks/          <- comandos operacionais, exemplos, troubleshooting
  07_reports/           <- living-paper (TCC/artigo), reviews externos, plots
    living-paper/
      00_outline.md       <- tese, pergunta, hipoteses, objetivos (leia antes de escrever)
      10_problem_...md    <- cap. revisao de literatura / related work
      20_method.md        <- cap. metodo
      30_results_...md    <- cap. resultados e analise
      40_limitations.md   <- cap. conclusoes e limitacoes
      evidence_log.md     <- registro rastreavel de evidencias por hipotese
    external-reviews/   <- reviews externos de metodologia e governanca
  08_governance/        <- git/versionamento, pre-registro, roadmap, TODO
  ai/                   <- AGENT_CORE, skills, workflow (para agentes de IA)
```

---

## Decisoes de design canonicas

| Decisao | Escolha | Referencia |
|---|---|---|
| Camada de dados primarios | Parquet (silver, source of truth) | `01_architecture/` ADR-0001 |
| Camada analitica | DuckDB reconstruivel + gold Parquet | `01_architecture/` ADR-0001 |
| Alinhamento OOS pareado | inner join por `asset+split+horizon+target_timestamp` | ADR-0002 |
| Modelo | TFT (pytorch-forecasting) em modo quantilico | `03_modeling/TRAINING_PIPELINE.md` |
| Candidato final | all-features poda minima + hiperparametros Round 0 | `STRATEGIC_DIRECTION.md` §4.2 |
| Baselines | Primario fixado no pre-registro; complementares canonicos documentados | `04_evaluation/BASELINES.md` |
| Calibracao | Calibracao marginal por quantil + PICP intervalar | `04_evaluation/CALIBRATION_AND_RISK.md` |
| Analise de features | VSN + permutation + ablation (nao selecao por OOS) | `04_evaluation/EXPLAINABILITY.md` |
| Escopo de decisao | `scope_mode=cohort_decision` com `scope_sweep_prefix` explicito | `AGENT_CORE.md` |
| Governanca anti-p-hacking | Pre-registro versionado antes de rodada confirmatoria | `STRATEGIC_DIRECTION.md` §5 (A.3) |

---

## Proximos passos

O projeto esta na transicao de Round 0 (exploratoria de hiperparametros) para
Fase A do roadmap (sanidade pre-experimento principal).

Ver detalhes em: `docs/00_overview/STRATEGIC_DIRECTION.md` §5.
Gate tecnico de entrada da Fase B: `docs/07_reports/phase-gates/phase-a/A_code_audit.md`.
Plano temporario pos-auditoria: `docs/08_governance/phase-a/POST_AUDIT_EXECUTION_PLAN.md`.

Ordem de leitura recomendada:
1. `docs/00_overview/STRATEGIC_DIRECTION.md` — objetivo, pivo, roadmap
2. `docs/00_overview/GLOSSARY.md` — termos tecnicos
3. `docs/07_reports/living-paper/00_outline.md` — tese, hipoteses, proximos passos
4. `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md` — metricas e gates
5. `docs/ai/AGENT_CORE.md` — regras de governanca de escopo (para agentes IA)
