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
| 1.4 Resumo | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/00_outline.md` secao 2 e secao 6 | parcial | v2 sincronizado (AAPL piloto, single-asset, post-guardrail, DM/HAC+Holm); revisar apos Fase B/C |
| 1.5 Abstract | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/00_outline.md` secao 2 e secao 6 | parcial | v2 sincronizado (AAPL piloto, single-asset, post-guardrail, DM/HAC+Holm); revisar apos Fase B/C |

Observacao: resumo e abstract so ficam `pronto` depois das Fases B/C, porque
dependem dos resultados e conclusoes finais.

## Capitulo 2 - Introducao (`text/2_Introducao/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 2.1 Contextualizacao e motivacao | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 | `docs/07_reports/living-paper/00_outline.md` secao 11 | pronto | v1 existe; revisar linguagem |
| 2.2 Problema de pesquisa | `docs/07_reports/living-paper/00_outline.md` secao 3 | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 1 | pronto | v1 existe; revisar linguagem |
| 2.3 Justificativa | `docs/07_reports/living-paper/00_outline.md` secao 6 | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 1 | pronto | v1 existe; revisar linguagem |
| 2.4 Objetivos | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/00_outline.md` secao 9 | parcial | v2 sincronizado com 00_outline.md §9 (H1/H2a/H2b/H3, baselines, DM/HAC+Holm, VSN/permutacao/ablacao) |
| 2.5 Delimitacao | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4.4 | `docs/07_reports/living-paper/00_outline.md` secao 5 | pronto | v2 sincronizado (AAPL piloto, single-asset, h+1/h+7, parent_sweep_id) |
| 2.6 Estrutura do trabalho | `docs/07_reports/living-paper/chapter_structure_freeze.md` | - | pronto | v1 existe |

Observacao: 2.4 sincronizada com H1/H2a/H2b/H3 em 2026-05-17.

## Capitulo 3 - Fundamentacao Teorica (`text/3_Revisao_Teorica/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 3.1 Previsao de series financeiras | `docs/07_reports/living-paper/10_problem_and_related_work.md` | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 | pronto | v2 sincronizado (HME/HMA, R^2 OOS, AAPL piloto, implicacoes para desenho) |
| 3.2 Modelagem com deep learning | `docs/07_reports/living-paper/10_problem_and_related_work.md` | `text/7_Referencias/7_Referencias.bib` | pronto | v2 sincronizado (Lim & Zohren 21, leakage temporal, escolha instrumental) |
| 3.3 Temporal Fusion Transformer | `docs/07_reports/living-paper/10_problem_and_related_work.md` | `docs/00_overview/GLOSSARY.md`, `docs/03_modeling/MULTI_HORIZON.md` | pronto | v2 sincronizado (Lim et al. 21, VSN/LSTM/atencao/multi-quantile head, guardrail) |
| 3.4 Predicao probabilistica e quantilica | `docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/04_evaluation/CALIBRATION_AND_RISK.md` | `docs/00_overview/GLOSSARY.md` | pronto | v2 sincronizado (Koenker & Bassett 78, Gneiting & Raftery 07, Chernozhukov et al. 10, Jorion 07, Acerbi & Tasche 02) |
| 3.5 Avaliacao OOS e testes estatisticos | `docs/04_evaluation/STATISTICAL_TESTS.md` | `docs/01_architecture/decisions/ADR-0002-oos-alignment.md` | pronto | v2 sincronizado (DM 95, HLN 97, MCS Hansen et al. 11, Holm, parent_sweep_id) |
| 3.6 Reprodutibilidade e rastreabilidade | `docs/08_governance/GOVERNANCE_AND_VERSIONING.md`, `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` | `docs/08_governance/EXPERIMENT_TRACKING_POLICY.md` | pronto | v2 sincronizado (Pineau et al. 21, run_id deterministico, ADR-0001, pre-registro) |

Observacao: Capitulo 3 foi expandido em 2026-05-17 (PR
`docs/tcc-expand-cap3-and-literature`). Living paper `10_md` agora carrega prosa autonoma
de revisao (~2493 palavras, 6 subsecoes); `.tex` reescrito com `\cite{}` ativo para 15
chaves nucleares. Sem duplicacao de definicoes formais (pinball, PICP, MPIW, VSN, MCS,
run_id) -- cross-link a docs canonicos em 04_evaluation/ e 00_overview/.

## Capitulo 4 - Metodo (`text/4_Metodo/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 4.1 Desenho da pesquisa | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4 | `docs/07_reports/living-paper/20_method.md` | parcial | revisar pivot |
| 4.2 Dados e pipelines | `docs/01_architecture/DATA_FLOW.md`, `docs/02_data/DATA_SOURCES.md` | `docs/06_runbooks/RUN_DATASET.md` | pronto | v1 existe; revisar leakage |
| 4.3 Engenharia de features | `docs/02_data/FEATURE_SETS.md`, `docs/05_checklists/FEATURES_SET_CHECKLIST.md` | `src/infrastructure/schemas/feature_registry.py` | pronto | revisar all-features pruned |
| 4.4 Configuracao do TFT | `docs/03_modeling/TRAINING_PIPELINE.md` | `src/adapters/pytorch_forecasting_tft_trainer.py` | pronto | v1 existe; revisar quantis |
| 4.5 Protocolo experimental | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 5, `docs/03_modeling/MULTI_HORIZON.md`, `docs/03_modeling/SWEEPS_AND_SELECTION.md` | `docs/07_reports/phase-gates/A_code_audit.md` | parcial | v2 sincronizado (pre-registro, parent_sweep_id, fronteira 2026-05-10); aguardar baselines persistidos para fechar |
| 4.6 Baselines | `docs/04_evaluation/BASELINES.md` | `docs/04_evaluation/STATISTICAL_TESTS.md` | parcial (runner pendente; ver `docs/07_reports/phase-gates/A_code_audit.md` §M7-Q2) | v1 inserida no `.tex`; runner pendente (M7-Q2) |
| 4.7 Metricas e calibracao | `docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/04_evaluation/CALIBRATION_AND_RISK.md` | `docs/04_evaluation/STATISTICAL_TESTS.md` | pronto | v2 sincronizado com Stage 8 (raw vs post-guardrail; Cat A/B/C cross-link a METRICS_DEFINITIONS.md) |
| 4.8 Explicabilidade e contribuicao | `docs/04_evaluation/EXPLAINABILITY.md` | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 4.3 | parcial | v1 inserida no `.tex` (VSN/permutacao/ablacao; hierarquia metodologica) |
| 4.9 Analytics Store e governanca | `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`, `docs/02_data/DATA_CONTRACTS.md` | `docs/01_architecture/decisions/ADR-0001-analytics-store.md`, `docs/05_checklists/ANALYTICS_STORE_CHECKLIST.md` | pronto | v2 sincronizado (ScopeSpec, cohort-aware groupby, parent_sweep_id) |
| 4.10 Criterios de qualidade | `docs/02_data/QUALITY_GATES.md`, `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md` secao 6 | `src/use_cases/validate_analytics_quality_use_case.py` | pronto | v1 existe; revisar gates |

Observacao: 4.5 e 4.6 so ficam `pronto` apos pre-registro e implementacao dos
baselines persistidos da Fase A.

## Capitulo 5 - Resultados (`text/5_Resultados/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 5.1 Cenarios experimentais | `docs/07_reports/living-paper/30_results_and_analysis.md`, `docs/03_modeling/SWEEPS_AND_SELECTION.md` | `docs/07_reports/ROUND0_023_FROZEN_SHORTLIST.md` | parcial | v2 sincronizado (Round 0 rotulada como diagnostico exploratorio; numeros removidos) |
| 5.2 Resultados agregados por horizonte/split | `docs/07_reports/living-paper/30_results_and_analysis.md` | Fase B a gerar | pendente | placeholder Fase B declarado no `.tex` |
| 5.3 Analise da incerteza e calibracao | `docs/04_evaluation/CALIBRATION_AND_RISK.md` | Fase B a gerar; `docs/07_reports/living-paper/evidence_log.md` | parcial | placeholder Fase B declarado no `.tex` |
| 5.4 Comparacoes entre candidato e baselines | `docs/07_reports/living-paper/30_results_and_analysis.md` | `docs/04_evaluation/BASELINES.md`, `docs/04_evaluation/STATISTICAL_TESTS.md` | pendente | placeholder Fase B declarado no `.tex` |
| 5.5 Analise estatistica pareada | `docs/04_evaluation/STATISTICAL_TESTS.md` | Fase B a gerar | pendente | placeholder Fase B declarado no `.tex` |
| 5.6 Interpretabilidade e implicacoes | `docs/04_evaluation/EXPLAINABILITY.md` | Fase C a gerar | pendente | placeholder Fase C declarado no `.tex` |
| 5.7 Ameacas a validade | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 e secao 4.4 | `docs/07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md`, `docs/07_reports/living-paper/evidence_log.md` | pronto | revisar apos Fase B/C |

Observacao: resultados da Round 0 entram apenas como exploracao e motivacao de
hiperparametros. Claims confirmatorios devem aguardar Fase B.

## Capitulo 6 - Conclusoes (`text/6_Conclusoes/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 6.1 Sintese dos achados | Fase B + Fase C a gerar | `docs/07_reports/living-paper/30_results_and_analysis.md` | pendente | pendente Fase B/C |
| 6.2 Contribuicoes | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4 | `docs/07_reports/living-paper/00_outline.md` secao 6 | parcial | revisar apos resultados |
| 6.3 Limitacoes | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 e secao 4.4 | `docs/07_reports/living-paper/40_limitations_and_conclusion.md` | parcial | v1 preenchida (8 limitacoes: single-asset, AAPL piloto, sem causalidade, 91,25% historico, reset 2026-05-10, h+30 suplementar, Round 0 exploratoria, interpretabilidade local) |
| 6.4 Trabalhos futuros | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/40_limitations_and_conclusion.md` | pronto | v1 preenchida (4 direcoes: mais ativos, regimes, causalidade, portfolio/trading) |

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

- ~~Expandir `docs/07_reports/living-paper/10_problem_and_related_work.md`.~~ (2026-05-17, branch `docs/tcc-expand-cap3-and-literature`)
- Preencher `docs/07_reports/phase-gates/A_code_audit.md`.
- Criar pre-registro versionado da Fase B.
- Implementar e persistir baselines comparaveis.
- Rodar Fase B para resultados confirmatorios.
- Rodar Fase C para contribuicao/explicabilidade.
- Consolidar `docs/07_reports/living-paper/40_limitations_and_conclusion.md`.
