---
title: Documentation Index
scope: Indice canonico de toda documentacao do projeto. Lista cada arquivo com escopo de uma linha para roteamento de atualizacoes.
update_when:
  - novo arquivo de documentacao for criado
  - arquivo de documentacao for renomeado, movido ou removido
  - escopo de um arquivo existente mudar significativamente
canonical_for: [documentation_routing, docs_inventory]
---

# Documentation Index

This index is the single entrypoint for project documentation and the **canonical
routing manifest** used by humans and AI agents to decide which documents to update
after a change.

## How To Use This Docs Set

- **Start at `docs/00_overview/STRATEGIC_DIRECTION.md`** for current objective,
  methodological pivot, and execution roadmap (supersedes earlier plans).
- Then `docs/00_overview/PROJECT_OVERVIEW.md` for scope and goals.
- Use the `Sections` listing below to locate canonical docs by topic.
- Each canonical doc starts with a YAML frontmatter declaring its `scope`,
  `update_when` and `canonical_for` topics — read frontmatter first before editing.

## Documentation Routing (for agents and humans)

After any change that modifies code behavior, contracts, metrics, commands or
methodology, follow the rule in `docs/ai/AGENT_CORE.md` §Documentation Sync Rules
to identify impacted docs via grep-based verification.

The `Sections` listing below is the routing manifest: each link gives a
one-line scope so you can decide in seconds whether the doc is impacted.

## AI Playbook
- `ai/INDEX.md`: AI playbook entrypoint.
- `ai/AGENT_CORE.md`: always-on baseline rules for AI agents.
- `ai/skills/workflow-governance/SKILL.md`: workflow guidance for implementation
  tasks.

---

## Sections

### `00_overview/` — Direcao e Escopo
Define o que o projeto e, qual o objetivo atual, o que esta fora de escopo e o
vocabulario canonico. Documentos aqui orientam toda decisao posterior.

- [`00_overview/PROJECT_OVERVIEW.md`](00_overview/PROJECT_OVERVIEW.md) : entrada executiva curta — objetivo, escopo,
  entregaveis, criterios de sucesso, estrutura de docs.
- [`00_overview/STRATEGIC_DIRECTION.md`](00_overview/STRATEGIC_DIRECTION.md) : direcao estrategica viva — pivo
  metodologico, hipoteses (H1/H2a/H2b/H3), claims aceitos vs proibidos, roadmap
  fases A-D. **Documento mais critico do projeto.**
- [`00_overview/GLOSSARY.md`](00_overview/GLOSSARY.md) : definicoes canonicas dos termos tecnicos
  recorrentes (PICP, MPIW, pinball, DM, MCS, VSN, scope_mode etc.).

### `01_architecture/` — Arquitetura de Software
Estrutura, fluxo de dados e decisoes arquiteturais (ADRs). Sem detalhes de
execucao operacional.

- [`01_architecture/SYSTEM_OVERVIEW.md`](01_architecture/SYSTEM_OVERVIEW.md) : estrutura de diretorios `src/` e
  Clean Architecture do projeto.
- [`01_architecture/DATA_FLOW.md`](01_architecture/DATA_FLOW.md) : fluxo end-to-end raw -> processed -> silver
  -> gold com paths concretos e ordem operacional.
- [`01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`](01_architecture/ANALYTICS_STORE_ARCHITECTURE.md) : camadas, contratos e
  principios da analytics store (Parquet silver + Parquet/DuckDB gold).
- [`01_architecture/decisions/ADR-0001-analytics-store.md`](01_architecture/decisions/ADR-0001-analytics-store.md) : decisao Parquet
  como source of truth + DuckDB reconstruivel.
- [`01_architecture/decisions/ADR-0002-oos-alignment.md`](01_architecture/decisions/ADR-0002-oos-alignment.md) : decisao de inner
  join estrito por `target_timestamp` em testes pareados.

### `02_data/` — Dados, Contratos e Features
Documenta fontes externas, contratos de dados, feature sets e quality gates.
Critico para rigor anti-leakage.

- [`02_data/DATA_SOURCES.md`](02_data/DATA_SOURCES.md) : fontes externas (yfinance, finnhub, alpha
  vantage), caminho canonico de ingestao de cada uma e politica de fallback para
  `reported_date` ausente em fundamentals.
- [`02_data/DATA_CONTRACTS.md`](02_data/DATA_CONTRACTS.md) : `run_id`, fingerprints, `schema_version`,
  chaves logicas e regras de evolucao de schema.
- [`02_data/FEATURE_SETS.md`](02_data/FEATURE_SETS.md) : registry de features por familia
  (PRICE/TECHNICAL/SENTIMENT/FUNDAMENTAL), warmup e tags anti-leakage.
- [`02_data/QUALITY_GATES.md`](02_data/QUALITY_GATES.md) : checks aplicados durante refresh do analytics
  store (integridade, cobertura, contratos quantilicos).
- [`02_data/DATA_PIPELINE_WALKTHROUGH.md`](02_data/DATA_PIPELINE_WALKTHROUGH.md) : walk-through pedagogico
  exaustivo do pipeline (raw -> processed -> silver -> gold) — inputs/outputs/calculos com refs file:line
  por celula (stage x scope) e lacunas conhecidas.

### `03_modeling/` — Treino, Inferencia e Estrategia
Como os modelos sao treinados, inferidos e versionados. Politica de sweeps e
multi-horizonte.

- [`03_modeling/TRAINING_PIPELINE.md`](03_modeling/TRAINING_PIPELINE.md) : referencia operacional dos parametros
  CLI do `src.main_train_tft` (defaults, ranges, exemplos).
- [`03_modeling/INFERENCE_PIPELINE.md`](03_modeling/INFERENCE_PIPELINE.md) : referencia operacional dos parametros
  CLI do `src.main_infer_tft`.
- [`03_modeling/MULTI_HORIZON.md`](03_modeling/MULTI_HORIZON.md) : politica de horizontes h+1/h+7/h+30,
  contrato de persistencia por linha, validacoes obrigatorias.
- [`03_modeling/SWEEPS_AND_SELECTION.md`](03_modeling/SWEEPS_AND_SELECTION.md) : familias de sweeps (Optuna e
  explicit-config), governanca de coorte, status atual da Round 0/1.
- [`03_modeling/P2_INFERENCE_EXPLAINABILITY_CONTRACT.md`](03_modeling/P2_INFERENCE_EXPLAINABILITY_CONTRACT.md)
  : schema versionado da resposta de inferencia explicavel para producao
  (multi-horizonte, risco, contribuicao local).

### `04_evaluation/` — Metricas, Baselines, Estatistica e Explicabilidade
Nucleo academico do projeto. Toda metrica, baseline, teste ou claim de
interpretabilidade usado no TCC deve estar definido aqui.

- [`04_evaluation/METRICS_DEFINITIONS.md`](04_evaluation/METRICS_DEFINITIONS.md) : formulas e implementacao das
  metricas pontuais (RMSE, MAE, DA, bias) e probabilisticas (pinball, PICP, MPIW,
  cobertura por quantil); **politica canonica de variante quantilica (raw vs
  post-guardrail) por tabela gold** (categorizacao A/B/C, sufixos `_raw`/`_post_guardrail`).
- [`04_evaluation/CALIBRATION_AND_RISK.md`](04_evaluation/CALIBRATION_AND_RISK.md) : calibracao marginal por quantil,
  PICP intervalar, bandas de aceitacao, metricas de risco (VaR, ES); calibracao
  e risco usam **post-guardrail como variante primaria** (sincronizado com
  METRICS_DEFINITIONS.md §Variante quantilica).
- [`04_evaluation/STATISTICAL_TESTS.md`](04_evaluation/STATISTICAL_TESTS.md) : DM/MCS/win-rate, ajuste Holm,
  alinhamento OOS por `target_timestamp`.
- [`04_evaluation/BASELINES.md`](04_evaluation/BASELINES.md) : contrato canonico de baselines admissiveis
  (random walk, AR(1), media historica, EWMA-vol, quantis historicos), regras de
  comparabilidade e decisao H2a/H2b.
- [`04_evaluation/EXPLAINABILITY.md`](04_evaluation/EXPLAINABILITY.md) : hierarquia VSN > permutation > ablation,
  linguagem permitida vs proibida, estabilidade minima.
- [`04_evaluation/PLOT_INTERPRETATION.md`](04_evaluation/PLOT_INTERPRETATION.md) : criterios pass/fail para os 8
  graficos canonicos da analise (heatmap, boxplot, DM matrix, calibration, etc.).

### `05_checklists/` — Gates e Tracking
Checklists rastreiam execucao e progresso. Nao sao fonte de definicao
conceitual — apontam para os docs canonicos.

- [`05_checklists/CHECKLISTS.md`](05_checklists/CHECKLISTS.md) : checklist agregada do MVP (etapas 0-N de
  desenvolvimento, com checkboxes de progresso).
- [`05_checklists/ANALYTICS_STORE_CHECKLIST.md`](05_checklists/ANALYTICS_STORE_CHECKLIST.md) : tracking de implementacao
  da analytics store (silver/gold/quality).
- [`05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md`](05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md) : tracking de
  implementacao de metricas, predicoes OOS e gates de aceite probabilistico.
- [`05_checklists/FEATURES_SET_CHECKLIST.md`](05_checklists/FEATURES_SET_CHECKLIST.md) : tracking de features
  implementadas por componente (build_dataset, technical, sentiment, fundamentals).

### `06_runbooks/` — Comandos e Execucao
Documentacao operacional command-first. Nao explica teoria; mostra como rodar.

- [`06_runbooks/RUN_PIPELINE_QUICKSTART.md`](06_runbooks/RUN_PIPELINE_QUICKSTART.md) : setup do ambiente local e
  primeiros passos (clone, venv, deps, sanity check).
- [`06_runbooks/RUN_DATASET.md`](06_runbooks/RUN_DATASET.md) : ordem operacional dos 7 sub-pipelines de
  ingestao e construcao do dataset TFT.
- [`06_runbooks/RUN_TRAINING.md`](06_runbooks/RUN_TRAINING.md) : comando canonico de treino TFT.
- [`06_runbooks/RUN_INFERENCE.md`](06_runbooks/RUN_INFERENCE.md) : comando e fluxo de inferencia rolling.
- [`06_runbooks/RUN_SWEEPS.md`](06_runbooks/RUN_SWEEPS.md) : execucao de sweeps Optuna e explicit-config,
  com secao de purge.
- [`06_runbooks/RUN_REFRESH_ANALYTICS.md`](06_runbooks/RUN_REFRESH_ANALYTICS.md) : refresh do analytics store com
  `--fail-on-quality` e flags de escopo.
- [`06_runbooks/phase-b/RUN_PHASE_B_TIER_CLASSIFICATION.md`](06_runbooks/phase-b/RUN_PHASE_B_TIER_CLASSIFICATION.md) :
  pos-processador E5 da Phase B para sidecars DM/Holm/tier.
- [`06_runbooks/RUN_TESTS.md`](06_runbooks/RUN_TESTS.md) : execucao de testes unitarios e de integracao.
- [`06_runbooks/TROUBLESHOOTING.md`](06_runbooks/TROUBLESHOOTING.md) : solucao de problemas comuns no setup
  Windows e PATH.
- [`06_runbooks/P2_INFERENCE_EXPLAINABILITY_RUNBOOK.md`](06_runbooks/P2_INFERENCE_EXPLAINABILITY_RUNBOOK.md) : operacao da
  inferencia explicavel em producao.

### `06_pre_registration/` — Pre-registros cientificos
Pre-registros operacionais versionados, selados antes de rodadas
confirmatorias. Cada pre-registro vira `canonical_for` da rodada que cobre.

- [`06_pre_registration/phase-b/preregistration_phase_b.md`](06_pre_registration/phase-b/preregistration_phase_b.md) : pre-registro F.2
  da Fase B confirmatoria (TFT all-features, AAPL); fixa hipoteses,
  candidato, contrato quantilico, baselines, coorte, protocolo
  estatistico e bandas tier 1/tier 2.

### `07_reports/` — Pesquisa, Evidencias e TCC
Camada cientifica e narrativa. Living paper como fonte do TCC, evidence log
como rastro empirico, phase-gates como auditorias datadas.

- [`07_reports/ROUND0_023_FROZEN_SHORTLIST.md`](07_reports/ROUND0_023_FROZEN_SHORTLIST.md) : shortlist congelada da
  Round 0 (sweep 0_2_3) com champion provisorio + fallback.
- [`07_reports/SWEEP_0_2_3_PLOTS_ANALYSIS.md`](07_reports/SWEEP_0_2_3_PLOTS_ANALYSIS.md) : log de analise dos plots do
  sweep 0_2_3 (datado, exploratorio).
- [`07_reports/external-reviews/PROJECT_FULL_TECHNICAL_REVIEW_2026-03-30.md`](07_reports/external-reviews/PROJECT_FULL_TECHNICAL_REVIEW_2026-03-30.md)
  : snapshot de revisao tecnica em 2026-03-30 (historico, congelado).
- [`07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md`](07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md)
  : review externo de governanca de escopo e rigor metodologico (2026-04-09).

#### `07_reports/templates/` — templates de relatorio
- [`07_reports/templates/REPORT_TEMPLATE_FINAL.md`](07_reports/templates/REPORT_TEMPLATE_FINAL.md)
  : template em branco para relatorios finais por rodada.
- [`07_reports/templates/RESULTS_SUMMARY_YYYYMMDD.md`](07_reports/templates/RESULTS_SUMMARY_YYYYMMDD.md)
  : template em branco para resumo de resultados datado por execucao.

#### `07_reports/living-paper/` — base do TCC e do artigo
- [`07_reports/living-paper/00_outline.md`](07_reports/living-paper/00_outline.md) : tese, pergunta, hipoteses
  (H1/H2a/H2b/H3), objetivos, recorte, contribuicoes.
- [`07_reports/living-paper/10_problem_and_related_work.md`](07_reports/living-paper/10_problem_and_related_work.md) : base para
  Capitulo 3 (Fundamentacao Teorica).
- [`07_reports/living-paper/20_method.md`](07_reports/living-paper/20_method.md) : base para Capitulo 4 (Metodo).
- [`07_reports/living-paper/30_results_and_analysis.md`](07_reports/living-paper/30_results_and_analysis.md) : base para Capitulo 5
  (Resultados); inclui analise de hipoteses H1-H7 da investigacao de quantis.
- [`07_reports/living-paper/40_limitations_and_conclusion.md`](07_reports/living-paper/40_limitations_and_conclusion.md) : base para
  Capitulo 6 (Conclusoes e Limitacoes).
- [`07_reports/living-paper/evidence_log.md`](07_reports/living-paper/evidence_log.md) : log rastreavel de evidencias
  empiricas por hipotese, com artefatos e uso no texto.
- [`07_reports/living-paper/tcc_mapping.md`](07_reports/living-paper/tcc_mapping.md) : mapa capitulo/secao do TCC
  -> arquivo-fonte primaria, com status pronto/parcial/pendente.
- [`07_reports/living-paper/chapter_structure_freeze.md`](07_reports/living-paper/chapter_structure_freeze.md) : estrutura
  congelada de capitulos ABNT do TCC.
- [`07_reports/living-paper/WRITING_PROTOCOL.md`](07_reports/living-paper/WRITING_PROTOCOL.md) : protocolo operacional
  para sessoes de escrita do TCC e artigo.

#### `07_reports/phase-gates/` — auditorias datadas de ciclo de vida
- [`07_reports/phase-gates/phase-a/A_code_audit.md`](07_reports/phase-gates/phase-a/A_code_audit.md) : auditoria pre-Fase B com
  veredictos por modulo (M1-M7).
- [`07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md`](07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md) :
  fechamento do gate da Fase A.
- [`07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md`](07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md) :
  relatorio confirmatorio final da Phase B.

### `08_governance/` — Processo, Versao e Pre-registro
Controle de processo: branch/commit/PR, versionamento, reprodutibilidade,
pre-registros cientificos.

- [`08_governance/GOVERNANCE_AND_VERSIONING.md`](08_governance/GOVERNANCE_AND_VERSIONING.md) : fluxo canonico de
  branch/commit/PR, politica de versionamento (`pipeline_version`,
  `schema_version`, `model_version`, `sweep_id`) e politica de reprodutibilidade.
- [`08_governance/EXPERIMENT_TRACKING_POLICY.md`](08_governance/EXPERIMENT_TRACKING_POLICY.md) : politica de IDs e
  rastreabilidade de experimentos (run_id, execution_id, fingerprints).
- [`08_governance/phase-a/POST_AUDIT_EXECUTION_PLAN.md`](08_governance/phase-a/POST_AUDIT_EXECUTION_PLAN.md) : plano temporario
  das etapas apos o A_code_audit ate o fechamento do escopo experimental.
- [`08_governance/ROADMAP_AND_TODO.md`](08_governance/ROADMAP_AND_TODO.md) : padrao oficial para uso de TODOs e
  registro de melhorias tecnicas planejadas.

### `ai/` — Regras para Agentes de IA
Documentacao especifica para agentes de IA. Nao deve conter documentacao
cientifica do projeto.

- [`ai/INDEX.md`](ai/INDEX.md) : entrypoint da documentacao para agentes.
- [`ai/AGENT_CORE.md`](ai/AGENT_CORE.md) : baseline always-on com regras globais
  (Expert Response Policy, Non-Negotiable Data Rules, Scope Governance,
  Documentation Sync Rules).
- [`ai/skills/workflow-governance/SKILL.md`](ai/skills/workflow-governance/SKILL.md)
  : workflow padrao para tarefas de implementacao e validacao.

### Documentos raiz
- [`CONTRIBUTING.md`](CONTRIBUTING.md) : checklist de pull request e regras de contribuicao.

---

## Documentation Rules
- One topic, one source of truth.
- Cross-link; avoid duplicated definitions.
- Keep runbooks command-first and verifiable.
- Any metric or table used in decisions must have a documented definition.
- Every canonical doc starts with YAML frontmatter declaring `scope`,
  `update_when` and `canonical_for`.
- Historical/frozen documents may use `update_when: nunca`; when grep returns
  matches in those files, treat them as historical references unless the task is
  explicitly to annotate, move, supersede or correct that snapshot.
