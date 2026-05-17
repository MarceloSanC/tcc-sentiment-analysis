---
title: TCC Mapping (chapter -> source files)
scope: Mapa mecanico capitulo/secao do TCC LaTeX (text/) -> arquivo-fonte primaria do living paper ou doc canonico. Coluna Status (pronto/parcial/pendente) vira backlog vivo de escrita. Inclui gaps P0 antes da escrita final.
update_when:
  - novo capitulo/secao for adicionado a estrutura do TCC
  - status de uma secao mudar (parcial -> pronto, pendente -> parcial)
  - fonte primaria de uma secao mudar
  - novo gap P0 for identificado
canonical_for: [tcc_mapping, tcc_writing_backlog, chapter_to_source_routing]
---

# TCC Mapping - capitulos -> arquivos-fonte

Mapa direto entre os capitulos do texto LaTeX (`text/`) e os artefatos do
repositorio que devem ser usados como fonte primaria para escrever cada secao.

Este arquivo e backlog de escrita e rastreabilidade. Ele nao substitui o texto
do TCC; indica onde buscar a fonte primaria e qual lacuna ainda existe.

Status por linha:
- `pronto` - fonte existe e esta suficientemente atualizada para escrita v1.
- `parcial` - fonte existe, mas precisa de atualizacao, consolidacao ou evidencia.
- `pendente` - fonte ainda nao existe ou depende de experimento futuro.

Importante: `Status fonte` avalia a maturidade da documentacao-fonte, nao a
prontidao do texto LaTeX. O campo `Status .tex` registra se o capitulo em
`text/` ja esta alinhado ao pivo metodologico de 2026-04-23/2026-04-25.

Regra de leitura:
- Round 0 e evidencia exploratoria de hiperparametros.
- Fase B sera a evidencia confirmatoria do candidato all-features vs baselines.
- Fase C sera a evidencia de contribuicao/explicabilidade.

---

## Capitulo 1 - Resumo (`text/1_Resumo/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 1.4 Resumo | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/00_outline.md` secao 2 e secao 6 | parcial | revisar apos Fase B/C |
| 1.5 Abstract | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/00_outline.md` secao 2 e secao 6 | parcial | revisar apos Fase B/C |

Observacao: resumo e abstract so ficam `pronto` depois das Fases B/C, porque
dependem dos resultados e conclusoes finais.

## Capitulo 2 - Introducao (`text/2_Introducao/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 2.1 Contextualizacao e motivacao | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 | `docs/07_reports/living-paper/00_outline.md` secao 11 | pronto | v1 existe; revisar linguagem |
| 2.2 Problema de pesquisa | `docs/07_reports/living-paper/00_outline.md` secao 3 | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 1 | pronto | v1 existe; revisar linguagem |
| 2.3 Justificativa | `docs/07_reports/living-paper/00_outline.md` secao 6 | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 1 | pronto | v1 existe; revisar linguagem |
| 2.4 Objetivos | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/00_outline.md` secao 9 | parcial | revisar H1/H2a/H2b/H3 |
| 2.5 Delimitacao | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4.4 | `docs/07_reports/living-paper/00_outline.md` secao 5 | pronto | v1 existe; revisar pivot |
| 2.6 Estrutura do trabalho | `docs/07_reports/living-paper/chapter_structure_freeze.md` | - | pronto | v1 existe |

Observacao: 2.4 permanece `parcial` ate a sincronizacao do texto LaTeX com
H1/H2a/H2b/H3.

## Capitulo 3 - Fundamentacao Teorica (`text/3_Revisao_Teorica/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 3.1 Previsao de series financeiras | `docs/07_reports/living-paper/10_problem_and_related_work.md` | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 | parcial | v1 existe; expandir base teorica |
| 3.2 Modelagem com deep learning | `docs/07_reports/living-paper/10_problem_and_related_work.md` | `text/7_Referencias/7_Referencias.bib` | pendente | v1 existe; revisar profundidade |
| 3.3 Temporal Fusion Transformer | `docs/07_reports/living-paper/10_problem_and_related_work.md` | `docs/00_overview/GLOSSARY.md`, `docs/03_modeling/MULTI_HORIZON.md` | parcial | v1 existe; revisar TFT/quantis |
| 3.4 Predicao probabilistica e quantilica | `docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/04_evaluation/CALIBRATION_AND_RISK.md` | `docs/00_overview/GLOSSARY.md` | pronto | v1 existe; revisar PICP/calibracao |
| 3.5 Avaliacao OOS e testes estatisticos | `docs/04_evaluation/STATISTICAL_TESTS.md` | `docs/01_architecture/decisions/ADR-0002-oos-alignment.md` | pronto | v1 existe; revisar H2a/H2b |
| 3.6 Reprodutibilidade e rastreabilidade | `docs/08_governance/GOVERNANCE_AND_VERSIONING.md`, `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` | `docs/08_governance/EXPERIMENT_TRACKING_POLICY.md` | pronto | v1 existe; revisar governanca |

Gap real: `10_problem_and_related_work.md` precisa ser expandido antes do
fechamento do Capitulo 3. Nao criar doc tecnico em `03_modeling/` para revisao
de literatura; a fonte primaria deve permanecer no living paper.

## Capitulo 4 - Metodo (`text/4_Metodo/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 4.1 Desenho da pesquisa | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4 | `docs/07_reports/living-paper/20_method.md` | parcial | revisar pivot |
| 4.2 Dados e pipelines | `docs/01_architecture/DATA_FLOW.md`, `docs/02_data/DATA_SOURCES.md` | `docs/06_runbooks/RUN_DATASET.md` | pronto | v1 existe; revisar leakage |
| 4.3 Engenharia de features | `docs/02_data/FEATURE_SETS.md`, `docs/05_checklists/FEATURES_SET_CHECKLIST.md` | `src/infrastructure/schemas/feature_registry.py` | pronto | revisar all-features pruned |
| 4.4 Configuracao do TFT | `docs/03_modeling/TRAINING_PIPELINE.md` | `src/adapters/pytorch_forecasting_tft_trainer.py` | pronto | v1 existe; revisar quantis |
| 4.5 Protocolo experimental | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 5, `docs/03_modeling/MULTI_HORIZON.md`, `docs/03_modeling/SWEEPS_AND_SELECTION.md` | `docs/07_reports/phase-gates/A_code_audit.md` | parcial | reescrever para Fase A/B/C |
| 4.6 Baselines | `docs/04_evaluation/BASELINES.md` | `docs/04_evaluation/STATISTICAL_TESTS.md` | parcial (runner pendente; ver `docs/07_reports/phase-gates/A_code_audit.md` §M7-Q2) | inserir secao/ajustar texto |
| 4.7 Metricas e calibracao | `docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/04_evaluation/CALIBRATION_AND_RISK.md` | `docs/04_evaluation/STATISTICAL_TESTS.md` | pronto | revisar PICP/calibracao; propagar politica raw vs post-guardrail (Stage 8; ver `docs/07_reports/living-paper/20_method.md` §"Política de variante quantilica") |
| 4.8 Explicabilidade e contribuicao | `docs/04_evaluation/EXPLAINABILITY.md` | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 4.3 | parcial | inserir contrato local/global |
| 4.9 Analytics Store e governanca | `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`, `docs/02_data/DATA_CONTRACTS.md` | `docs/01_architecture/decisions/ADR-0001-analytics-store.md`, `docs/05_checklists/ANALYTICS_STORE_CHECKLIST.md` | pronto | propagar para `.tex`: refresh scoped via ScopeSpec (Stage 7) e cohort-aware groupby (Stages 4-6) |
| 4.10 Criterios de qualidade | `docs/02_data/QUALITY_GATES.md`, `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md` secao 6 | `src/use_cases/validate_analytics_quality_use_case.py` | pronto | v1 existe; revisar gates |

Observacao: 4.5 e 4.6 so ficam `pronto` apos pre-registro e implementacao dos
baselines persistidos da Fase A.

## Capitulo 5 - Resultados (`text/5_Resultados/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 5.1 Cenarios experimentais | `docs/07_reports/living-paper/30_results_and_analysis.md`, `docs/03_modeling/SWEEPS_AND_SELECTION.md` | `docs/07_reports/ROUND0_023_FROZEN_SHORTLIST.md` | parcial | reescrever Round 0 como exploratoria |
| 5.2 Resultados agregados por horizonte/split | `docs/07_reports/living-paper/30_results_and_analysis.md` | Fase B a gerar | pendente | pendente Fase B |
| 5.3 Analise da incerteza e calibracao | `docs/04_evaluation/CALIBRATION_AND_RISK.md` | Fase B a gerar; `docs/07_reports/living-paper/evidence_log.md` | parcial | pendente Fase B |
| 5.4 Comparacoes entre candidato e baselines | `docs/07_reports/living-paper/30_results_and_analysis.md` | `docs/04_evaluation/BASELINES.md`, `docs/04_evaluation/STATISTICAL_TESTS.md` | pendente | pendente Fase B |
| 5.5 Analise estatistica pareada | `docs/04_evaluation/STATISTICAL_TESTS.md` | Fase B a gerar | pendente | pendente Fase B |
| 5.6 Interpretabilidade e implicacoes | `docs/04_evaluation/EXPLAINABILITY.md` | Fase C a gerar | pendente | pendente Fase C |
| 5.7 Ameacas a validade | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 e secao 4.4 | `docs/07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md`, `docs/07_reports/living-paper/evidence_log.md` | pronto | revisar apos Fase B/C |

Observacao: resultados da Round 0 entram apenas como exploracao e motivacao de
hiperparametros. Claims confirmatorios devem aguardar Fase B.

## Capitulo 6 - Conclusoes (`text/6_Conclusoes/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 6.1 Sintese dos achados | Fase B + Fase C a gerar | `docs/07_reports/living-paper/30_results_and_analysis.md` | pendente | pendente Fase B/C |
| 6.2 Contribuicoes | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4 | `docs/07_reports/living-paper/00_outline.md` secao 6 | parcial | revisar apos resultados |
| 6.3 Limitacoes | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 e secao 4.4 | `docs/07_reports/living-paper/40_limitations_and_conclusion.md` | parcial | revisar apos Fase B/C |
| 6.4 Trabalhos futuros | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/40_limitations_and_conclusion.md` | pronto | revisar apos conclusoes |

Observacao: `40_limitations_and_conclusion.md` ainda precisa consolidar
limitacoes antes de Capitulo 6 ser considerado pronto.

## Apendices (`text/8_Apendices/`)

| Conteudo sugerido | Fonte | Status |
|---|---|---|
| Auditoria pre-experimento da Fase A | `docs/07_reports/phase-gates/A_code_audit.md` | parcial |
| Pre-registro Fase B | `docs/08_governance/preregistration_round1_<date>.md` | pendente |
| Glossario tecnico | `docs/00_overview/GLOSSARY.md` | pronto |
| Comandos de reproducao | `docs/06_runbooks/RUN_PIPELINE_QUICKSTART.md` e demais runbooks | pronto |

## Gaps P0 Antes Da Escrita Final

- Expandir `docs/07_reports/living-paper/10_problem_and_related_work.md`.
- Preencher `docs/07_reports/phase-gates/A_code_audit.md`.
- Criar pre-registro versionado da Fase B.
- Implementar e persistir baselines comparaveis.
- Rodar Fase B para resultados confirmatorios.
- Rodar Fase C para contribuicao/explicabilidade.
- Consolidar `docs/07_reports/living-paper/40_limitations_and_conclusion.md`.
