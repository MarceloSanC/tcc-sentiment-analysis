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
- Fase B fechou em 2026-05-25 como evidencia confirmatoria do candidato
  all-features sealed vs baselines pre-declarados, escopo cohort
  `phase_b_confirmatorio_20260524` (AAPL, h=1 e h=7). Tier verdict
  H1/H2a/H2b por horizonte em
  [`B_confirmatory_2026-05-25.md`](../phase-gates/phase-b/B_confirmatory_2026-05-25.md).
- Fase C e pendente; cobre H3 (contribuicao/explicabilidade) e o hardening
  C.0 dos testes/metricas gold legacy
  ([`C0_statistical_methods_hardening.md`](../phase-gates/phase-c/C0_statistical_methods_hardening.md)).

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
| 2.1 Contextualizacao e motivacao | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 | `docs/07_reports/living-paper/00_outline.md` secao 11 | pronto | v2 sincronizado: paragrafo adicional sobre calibracao de expectativas (R^2 OOS 0,1-1% para retornos diarios; \cite{GU2020,RAPACH2013}) justificando avaliacao probabilistica como prioritaria |
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
| 4.3 Engenharia de features | `docs/02_data/FEATURE_SETS.md`, `docs/05_checklists/FEATURES_SET_CHECKLIST.md` | `src/infrastructure/schemas/feature_registry.py` | pronto | v2 sincronizado: poda minima baseada em principio (corr>0,95; missing>30%; derivadas redundantes; disponibilidade temporal) declarada conforme STRATEGIC_DIRECTION §4.2; lista final congelada no pre-registro |
| 4.4 Configuracao do TFT | `docs/03_modeling/TRAINING_PIPELINE.md` | `src/adapters/pytorch_forecasting_tft_trainer.py` | pronto | v1 existe; revisar quantis |
| 4.5 Protocolo experimental | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 5, `docs/03_modeling/MULTI_HORIZON.md`, `docs/03_modeling/SWEEPS_AND_SELECTION.md` | `docs/07_reports/phase-gates/phase-a/A_code_audit.md` | parcial | v2 sincronizado (pre-registro, parent_sweep_id, fronteira 2026-05-10); aguardar baselines persistidos para fechar |
| 4.6 Baselines | `docs/04_evaluation/BASELINES.md` | `docs/04_evaluation/STATISTICAL_TESTS.md` | parcial (runner pendente; ver `docs/07_reports/phase-gates/phase-a/A_code_audit.md` §M7-Q2) | v1 inserida no `.tex`; runner pendente (M7-Q2) |
| 4.7 Metricas e calibracao | `docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/04_evaluation/CALIBRATION_AND_RISK.md` | `docs/04_evaluation/STATISTICAL_TESTS.md` | pronto | v2 sincronizado com Stage 8 (raw vs post-guardrail; Cat A/B/C cross-link a METRICS_DEFINITIONS.md) |
| 4.8 Explicabilidade e contribuicao | `docs/04_evaluation/EXPLAINABILITY.md` | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 4.3 | parcial | v1 inserida no `.tex` (VSN/permutacao/ablacao; hierarquia metodologica) |
| 4.9 Analytics Store e governanca | `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`, `docs/02_data/DATA_CONTRACTS.md` | `docs/01_architecture/decisions/ADR-0001-analytics-store.md`, `docs/05_checklists/ANALYTICS_STORE_CHECKLIST.md` | pronto | v2 sincronizado (ScopeSpec, cohort-aware groupby, parent_sweep_id) |
| 4.10 Criterios de qualidade | `docs/02_data/QUALITY_GATES.md`, `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md` secao 6 | `src/use_cases/validate_analytics_quality_use_case.py`, `docs/07_reports/living-paper/30_results_and_analysis.md` §"Criterios quantitativos de aceite" | pronto | v2 sincronizado: subsecao "Gates quantitativos de aceitacao" com thresholds A/B/C/D explicitos (crossing<=0,10%; |dPICP|<=0,02; etc.) e regra de promocao hierarquica |

Observacao: 4.5 e 4.6 so ficam `pronto` apos pre-registro e implementacao dos
baselines persistidos da Fase A.

## Capitulo 5 - Resultados (`text/5_Resultados/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 5.1 Cenarios experimentais | `docs/07_reports/living-paper/30_results_and_analysis.md`, `docs/03_modeling/SWEEPS_AND_SELECTION.md` | `docs/07_reports/ROUND0_023_FROZEN_SHORTLIST.md` | parcial | v2 sincronizado (Round 0 rotulada como diagnostico exploratorio; numeros removidos) |
| 5.2 Resultados agregados por horizonte/split | `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md`, `docs/07_reports/living-paper/30_results_and_analysis.md` §"Resultados confirmatorios — Phase B" | sidecars `phase_b_marginal_coverage.parquet`, `phase_b_tier_verdict.parquet` | parcial | atualizar `.tex` com tabela tier por (hipotese, horizonte) e parametros sealed |
| 5.3 Analise da incerteza e calibracao | `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` §2-§3, `docs/04_evaluation/CALIBRATION_AND_RISK.md` | `phase_b_marginal_coverage.parquet`, `phase_b_tier_verdict.parquet`, `docs/07_reports/living-paper/evidence_log.md` | parcial | propagar PICP/MPIW/cov_qτ Phase B para `.tex` |
| 5.4 Comparacoes entre candidato e baselines | `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` §3, `docs/07_reports/living-paper/30_results_and_analysis.md` | `docs/04_evaluation/BASELINES.md` | parcial | propagar tabela descritiva Phase B (3 baselines x 2 horizontes) para `.tex` |
| 5.5 Analise estatistica pareada | `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` §4.1 (DM family-6), §11 (DM-18 sensibilidade) | `phase_b_dm_family_6.parquet`, `phase_b_dm_family_18_sensitivity.parquet`, `phase_b_delta_pinball.parquet`, `docs/04_evaluation/STATISTICAL_TESTS.md` | parcial | propagar DM family-6 + apendice DM-18 para `.tex` |
| 5.6 Interpretabilidade e implicacoes | `docs/04_evaluation/EXPLAINABILITY.md` | Fase C a gerar; `docs/07_reports/phase-gates/phase-c/C0_statistical_methods_hardening.md` | pendente | placeholder Fase C declarado no `.tex` |
| 5.7 Ameacas a validade | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 e secao 4.4, `docs/07_reports/living-paper/40_limitations_and_conclusion.md` §"Limitacoes especificas da Phase B confirmatoria" | `docs/07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md`, `docs/07_reports/living-paper/evidence_log.md` | pronto | propagar limitacoes Phase B para `.tex` |

Observacao: resultados da Round 0 entram apenas como exploracao e motivacao de
hiperparametros. Claims confirmatorios Phase B estao disponiveis em
[`B_confirmatory_2026-05-25.md`](../phase-gates/phase-b/B_confirmatory_2026-05-25.md)
e sintetizados em
[`30_results_and_analysis.md`](30_results_and_analysis.md) §"Resultados
confirmatorios — Phase B". Fase C / H3 permanece pendente; secao 5.6 fica
como future work no `.tex` ate a Phase C concluir.

## Capitulo 6 - Conclusoes (`text/6_Conclusoes/`)

| Secao | Fonte primaria | Fonte complementar | Status fonte | Status .tex |
|---|---|---|---|---|
| 6.1 Sintese dos achados | `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` §8, `docs/07_reports/living-paper/30_results_and_analysis.md` §"Resultados confirmatorios — Phase B" | Fase C a gerar (H3) | parcial | propagar sintese Phase B (Tier 1 H2a; Tier 1/Tier 2 H1; H2b refutada) para `.tex`; bloco Fase C / H3 permanece TODO |
| 6.2 Contribuicoes | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 e secao 4 | `docs/07_reports/living-paper/00_outline.md` secao 6, `docs/07_reports/living-paper/40_limitations_and_conclusion.md` §"Contribuicoes metodologicas" | parcial | v2 sincronizado: subsecao "Contribuicoes metodologicas" preenchida (5 contribuicoes: variante quantilica, governanca de coorte, camadas reconstruiveis, hierarquia VSN/permutation/ablation, pre-registro); subsecao "Contribuicoes empiricas" pode ser parcialmente preenchida com Phase B (calibracao Tier 1 h=7; dominancia vs `zero_return`); contribuicao empirica completa aguarda Fase C |
| 6.3 Limitacoes | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 3 e secao 4.4, `docs/07_reports/living-paper/40_limitations_and_conclusion.md` §"Limitacoes especificas da Phase B confirmatoria" e §"O que a Phase B nao conclui" | sidecars Phase B | parcial | v2 expandida (limitacoes Phase B explicitas: escopo cohort, H2b refutada localmente, H1 h=1 Tier 2, N_eff 937, cross-family ausentes, sem generalizacao); propagar para `.tex` |
| 6.4 Trabalhos futuros | `docs/00_overview/STRATEGIC_DIRECTION.md` secao 2 | `docs/07_reports/living-paper/40_limitations_and_conclusion.md` | pronto | v1 preenchida (4 direcoes: mais ativos, regimes, causalidade, portfolio/trading) |

Observacao: `40_limitations_and_conclusion.md` ainda precisa consolidar
limitacoes antes de Capitulo 6 ser considerado pronto.

## Apendices (`text/8_Apendices/`)

| Conteudo | Fonte primaria | Status |
|---|---|---|
| Apendice A - Protocolo operacional e contratos de qualidade | `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`, `docs/04_evaluation/METRICS_DEFINITIONS.md`, `docs/04_evaluation/CALIBRATION_AND_RISK.md`, `docs/04_evaluation/STATISTICAL_TESTS.md`, `docs/07_reports/living-paper/30_results_and_analysis.md` §"Criterios quantitativos de aceite" | pronto: v1 inserida no `.tex` (run_id/fingerprints, camadas silver/gold, variante quantilica raw/post-guardrail, gates A/B/C/D, protocolo estatistico, pre-registro) |
| Apendice B - Glossario tecnico de termos recorrentes | `docs/00_overview/GLOSSARY.md`, `docs/04_evaluation/METRICS_DEFINITIONS.md` | pronto: v1 inserida no `.tex` (calibracao, governanca, modelo/pipeline, testes estatisticos) |
| Apendice C - Comandos de reproducao | `docs/06_runbooks/RUN_PIPELINE_QUICKSTART.md` e demais runbooks | pronto: v1 inserida no `.tex` (referencias a runbooks de dataset, treino, inferencia, refresh analytics) |
| Auditoria pre-experimento da Fase A | `docs/07_reports/phase-gates/phase-a/A_code_audit.md`, `docs/07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md` | pronto: pode ser referenciada do Apendice A |
| Pre-registro Fase B | `docs/06_pre_registration/phase-b/preregistration_phase_b.md` (commit selo `067cb32`) | pronto: pode entrar como apendice adicional apos decisao editorial |
| Relatorio Phase B confirmatoria | `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md` | pronto: fonte primaria de §5.2-§5.5 e §6.1 |
| C.0 hardening Phase C | `docs/07_reports/phase-gates/phase-c/C0_statistical_methods_hardening.md` | pendente: living skeleton; referenciar como future work em §5.6 e §6.4 |

## Gaps P0 Antes Da Escrita Final

- ~~Expandir `docs/07_reports/living-paper/10_problem_and_related_work.md`.~~ (2026-05-17, branch `docs/tcc-expand-cap3-and-literature`)
- ~~Preencher `docs/07_reports/phase-gates/phase-a/A_code_audit.md`.~~ (Phase A fechada 2026-05-23)
- ~~Criar pre-registro versionado da Fase B.~~ (commit selo `067cb32`,
  `docs/06_pre_registration/phase-b/preregistration_phase_b.md`)
- ~~Implementar e persistir baselines comparaveis.~~ (cohort `phase_b_confirmatorio_20260524`, 45 runs baseline)
- ~~Rodar Fase B para resultados confirmatorios.~~ (relatorio
  `docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md`, 2026-05-25)
- Rodar Fase C para contribuicao/explicabilidade (H3).
- Concluir C.0 hardening do gold legacy (`docs/07_reports/phase-gates/phase-c/C0_statistical_methods_hardening.md`).
- Propagar resultados Phase B para `text/5_Resultados/*.tex` e `text/6_Conclusoes/*.tex` (apos validacao editorial).
- ~~Consolidar `docs/07_reports/living-paper/40_limitations_and_conclusion.md`.~~ (atualizado pos-Phase B em 2026-05-27)
