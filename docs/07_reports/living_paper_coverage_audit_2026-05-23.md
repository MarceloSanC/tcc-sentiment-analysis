---
title: Living Paper Coverage Audit (2026-05-15 -> 2026-05-23)
date: 2026-05-23
author: GPT-5 Codex (auditoria supervisionada)
scope: Identificar conteudo TCC/artigo executado no ciclo R-20..R-23 + F.1/F.2/F.3 que ainda nao esta refletido no living paper.
status: draft - aguarda triagem de Marcelo
---

# Resumo executivo

- 5 gaps P0 (bloqueiam Fase B ou a escrita fiel do protocolo antes da Fase B): GAP-01 a GAP-05.
- 2 gaps P1 (necessarios antes da entrega TCC): GAP-06 e GAP-07.
- 2 gaps P2 (nice-to-have): GAP-08 e GAP-09.

# Metodologia da auditoria

Step 1 - Skills used: `workflow-governance`, `documentation-maintenance`.
Inventario feito por `git log --since="2026-05-15" --until="2026-05-23 23:59:59"` e `gh pr list --state merged --search "merged:>=2026-05-15"`. A lista confirmou PRs #50 a #59 no periodo, alem de PRs relacionados #39-#49.

Step 2 - Skills used: `tcc-writing-core`, `documentation-authoring`, `tcc-metodo-writer`, `tcc-resultados-writer`, `tcc-conclusoes-writer`.
Foram lidos os 9 arquivos do living paper com linhas numeradas: `00_outline.md`, `10_problem_and_related_work.md`, `20_method.md`, `30_results_and_analysis.md`, `40_limitations_and_conclusion.md`, `evidence_log.md`, `tcc_mapping.md`, `chapter_structure_freeze.md`, `WRITING_PROTOCOL.md`.

Step 3 - Skills used: `documentation-maintenance`, `tcc-metodo-writer`, `tcc-resultados-writer`.
Cada gap abaixo foi classificado somente quando havia: (a) fonte primaria do trabalho executado e (b) trecho do living paper ausente, desatualizado ou contraditorio. Itens puramente de housekeeping foram descartados.

# Gaps identificados

## GAP-01 - Convencao `target_return` / Gap 6 / Opcao (d) nao entrou no Metodo

- Tipo: metodo | decisao | rastreabilidade
- Fonte do conteudo: `docs/01_architecture/decisions/ADR-0003-multi-horizon-prediction-persister.md:66-84`; `docs/07_reports/option_b_execution_log_2026-05-19.md:543-564`; `docs/02_data/DATA_PIPELINE_WALKTHROUGH.md:585`
- Onde deveria estar: `docs/07_reports/living-paper/20_method.md` secao Dados e pipelines / Protocolo experimental; `docs/07_reports/living-paper/evidence_log.md` nova entrada Stage R-20
- Estado atual no living paper: ausente. `20_method.md` resume pipeline/protocolo apenas em bullets gerais (`docs/07_reports/living-paper/20_method.md:37-46`) e nao fixa `target_return[t] = log(close[t]/close[t-1])`, `decision_idx + h`, nem a historia de off-by-one.
- Urgencia: P0
- Justificativa: e decisao metodologica central. Define o alvo supervisionado, o significado de h+1/h+7, a comparabilidade TFT vs baselines e a correcao de uma falha que invalidaria comparacao pareada.
- Esboco do que precisa ser escrito: registrar no Metodo a convencao final do alvo, explicando que a formula backward preserva "next-day return after decision" e alinha TFT/baselines por `target_timestamp_utc`; no `evidence_log`, registrar o bug, a opcao escolhida e os testes/smoke que fecharam o risco.

## GAP-02 - Pre-registro F.2 nao foi propagado como protocolo Phase B

- Tipo: metodo | rastreabilidade
- Fonte do conteudo: `docs/06_pre_registration/preregistration_phase_b.md:55-71`, `docs/06_pre_registration/preregistration_phase_b.md:190-204`, `docs/06_pre_registration/preregistration_phase_b.md:437-463`, `docs/06_pre_registration/preregistration_phase_b.md:467-489`
- Onde deveria estar: `docs/07_reports/living-paper/20_method.md` secao Protocolo experimental; `docs/07_reports/living-paper/tcc_mapping.md` Capitulo 4 linhas 4.5/4.6; `docs/07_reports/living-paper/evidence_log.md`
- Estado atual no living paper: desatualizado. `20_method.md` nao contem o protocolo F.2 detalhado; `tcc_mapping.md` ainda marca 4.5/4.6 como parciais e diz aguardar pre-registro/baselines (`docs/07_reports/living-paper/tcc_mapping.md:84-92`).
- Urgencia: P0
- Justificativa: F.2 fixa escopo AAPL, splits, seeds/folds, ScopeSpec, JSON explicito, operacao confirmatoria e politica de emendas. Sem isso, o living paper nao e fonte suficiente para escrever o metodo confirmatorio.
- Esboco do que precisa ser escrito: consolidar o protocolo Phase B no Metodo sem duplicar o pre-registro inteiro: ativo, splits, coorte, seeds/folds, `scope_mode=cohort_decision`, comando de refresh e politica de emendas.

## GAP-03 - Conjunto oficial de baselines Phase B contradiz o living paper

- Tipo: metodo | decisao
- Fonte do conteudo: `docs/06_pre_registration/preregistration_phase_b.md:244-269`; `docs/02_data/DATA_PIPELINE_WALKTHROUGH.md:1030-1058`
- Onde deveria estar: `docs/07_reports/living-paper/20_method.md` secao Baselines; `docs/07_reports/living-paper/10_problem_and_related_work.md` secao 3.5; `docs/07_reports/living-paper/tcc_mapping.md` linha 4.6
- Estado atual no living paper: contradiz/desatualizado. `10_problem_and_related_work.md` ainda apresenta random walk, AR(1), media historica, EWMA-vol e quantis empiricos como baselines admissiveis (`docs/07_reports/living-paper/10_problem_and_related_work.md:276-285`); `20_method.md` repete lista ampla similar (`docs/07_reports/living-paper/20_method.md:86-94`). F.2 declara Phase B com apenas `zero_return`, `historical_mean_rolling` e `historical_quantiles_rolling`, deixando `random_walk` distinto, `AR(1)` e `EWMA-vol` fora de escopo.
- Urgencia: P0
- Justificativa: a familia de testes, H2a/H2b e a comparacao justa dependem do conjunto exato de baselines. O texto teorico pode citar alternativas, mas precisa separar "baselines conceituais" de "baselines oficiais Phase B".
- Esboco do que precisa ser escrito: ajustar o Metodo para declarar os 3 baselines oficiais e, na revisao/metodo, rebaixar AR(1)/EWMA/random_walk distinto para baselines planejados/futuros.

## GAP-04 - Holm familia unica e criterios Tier 1/Tier 2 nao estao no living paper

- Tipo: metodo | decisao | criterio de aceitacao
- Fonte do conteudo: `docs/06_pre_registration/preregistration_phase_b.md:310-344`; `docs/06_pre_registration/preregistration_phase_b.md:368-433`
- Onde deveria estar: `docs/07_reports/living-paper/20_method.md` secao Protocolo estatistico; `docs/07_reports/living-paper/30_results_and_analysis.md` secao criterios de aceite; `docs/07_reports/living-paper/evidence_log.md`
- Estado atual no living paper: desatualizado. O living paper cobre DM/HAC/MCS/Holm em nivel geral (`docs/07_reports/living-paper/10_problem_and_related_work.md:265-296`) e os gates A-D antigos (`docs/07_reports/living-paper/30_results_and_analysis.md:182-205`), mas nao registra a familia unica de 6 testes nem a politica Tier 1/Tier 2 do pre-registro.
- Urgencia: P0
- Justificativa: isso determina como H1/H2a/H2b serao aceitas, rebaixadas para evidencia secundaria ou refutadas. E criterio de decisao cientifica, nao detalhe operacional.
- Esboco do que precisa ser escrito: acrescentar a familia Holm unica (`2 horizontes x 3 baselines = 6 testes`), criterios Tier 1/Tier 2, regra de refutacao e gate de degeneracao como exclusao automatica de claim probabilistico.

## GAP-05 - Gate de saida da Fase A esta fechado, mas o backlog do living paper ainda diz que nao

- Tipo: rastreabilidade
- Fonte do conteudo: `docs/07_reports/phase-gates/A_code_audit.md:1235-1299`; `docs/07_reports/phase-gates/A_audit_closure_2026-05-17.md:442-462`
- Onde deveria estar: `docs/07_reports/living-paper/tcc_mapping.md` secao Gaps P0; possivelmente `docs/07_reports/living-paper/evidence_log.md`
- Estado atual no living paper: contradiz/desatualizado. `tcc_mapping.md` ainda lista "Preencher A_code_audit", "Criar pre-registro versionado da Fase B" e "Implementar e persistir baselines comparaveis" como P0 abertos (`docs/07_reports/living-paper/tcc_mapping.md:131-139`), enquanto `A_code_audit.md` marca F.1 + F.2 satisfeitos, todos M1-M7 fechados e abertura da Fase B em 2026-05-23.
- Urgencia: P0
- Justificativa: `tcc_mapping.md` e backlog de escrita e fonte de roteamento. Se ficar stale, uma sessao futura pode reabrir trabalho ja fechado ou escrever texto com status errado.
- Esboco do que precisa ser escrito: atualizar somente o living paper/backlog para registrar Fase A gate fechado, mantendo Fase B/C resultados como pendentes.

## GAP-06 - F.1 v4 PASS pleno nao entrou no `evidence_log`

- Tipo: evidencia | rastreabilidade
- Fonte do conteudo: `docs/07_reports/smoke_confirmatory_2026-05-18.md:12-43`; `docs/07_reports/option_b_execution_log_2026-05-19.md:101-126`
- Onde deveria estar: `docs/07_reports/living-paper/evidence_log.md`; opcionalmente `docs/07_reports/living-paper/30_results_and_analysis.md` como evidencia operacional, nao resultado confirmatorio
- Estado atual no living paper: ausente. O `evidence_log.md` termina com entrada Stage 8 de 2026-05-16 (`docs/07_reports/living-paper/evidence_log.md:521-534`) e nao tem entrada F.1/R-23.
- Urgencia: P1
- Justificativa: o smoke nao e resultado cientifico de superioridade, mas e evidencia de validade operacional do protocolo: 6/6 criterios PASS, 27 checks, zero falhas, 25 gold tables, cohort-aware.
- Esboco do que precisa ser escrito: entrada curta de evidencia "F.1 v4 PASS pleno" com parametros, checks, artefatos e uso no texto como validacao pre-experimento.

## GAP-07 - Justificativa academica dos limiares dos gates esta no `.tex`, mas nao voltou ao living paper

- Tipo: metodo | rastreabilidade
- Fonte do conteudo: `text/4_Metodo/4_Metodo.tex:63-76`; `text/8_Apendices/8_Apendices.tex:28-39`; PR #55 (`b550281`)
- Onde deveria estar: `docs/07_reports/living-paper/20_method.md` secao Criterios de qualidade; `docs/07_reports/living-paper/30_results_and_analysis.md` secao Criterios quantitativos; `docs/07_reports/living-paper/tcc_mapping.md`
- Estado atual no living paper: desatualizado. `30_results_and_analysis.md` lista os numeros dos gates (`docs/07_reports/living-paper/30_results_and_analysis.md:182-205`), mas nao inclui a classificacao em tres categorias metodologicas nem a justificativa de compromisso ex-ante/auditabilidade que foi escrita direto em `text/`.
- Urgencia: P1
- Justificativa: living paper deve ser a fonte canonica do LaTeX; aqui ha drift inverso (`text/` mais rico que living paper). O conteudo e substantivo porque justifica por que os thresholds sao aceitaveis apesar de nao serem todos canonizados pela literatura.
- Esboco do que precisa ser escrito: incorporar ao living paper a classificacao: convencoes estatisticas, limiares operacionais exploratorios e contratos de governanca pre-registrados.

## GAP-08 - Arquitetura de qualidade/reprodutibilidade R-21/R-22 esta fora do living paper

- Tipo: metodo | rastreabilidade
- Fonte do conteudo: `docs/01_architecture/decisions/ADR-0004-quality-check-registry.md:15-22`; `docs/07_reports/option_b_execution_log_2026-05-19.md:339-448`; `docs/01_architecture/decisions/ADR-0005-gold-builders-modularization.md:117-202`; `docs/07_reports/option_b_execution_log_2026-05-19.md:243-328`
- Onde deveria estar: `docs/07_reports/living-paper/20_method.md` secao Analytics Store e governanca; `docs/07_reports/living-paper/evidence_log.md` como evidencia de reproducibilidade operacional
- Estado atual no living paper: ausente/parcial. `20_method.md` so diz "gates de integridade temporal, consistencia contratual e monotonicidade" (`docs/07_reports/living-paper/20_method.md:44-46`); nao registra que os 27 checks viraram registry com sentinel bit-identical, nem que 25 gold builders foram modularizados com dependencias e sentinel byte-identical.
- Urgencia: P2
- Justificativa: os detalhes de LOC e nomes de classes sao implementacao, mas os contratos de sentinela bit/byte-identical sustentam a alegacao de reprodutibilidade e nao regressao dos artefatos analiticos.
- Esboco do que precisa ser escrito: um paragrafo de metodo/reprodutibilidade, sem listar todos os arquivos, explicando que quality checks e gold builders foram modularizados com testes de equivalencia contra o archive.

## GAP-09 - Divida tecnica de derivacao de `parent_sweep_id`/JSON fisico nao esta em limitacoes

- Tipo: limitacao | rastreabilidade
- Fonte do conteudo: `docs/07_reports/smoke_confirmatory_2026-05-18.md:34-38`; `docs/06_pre_registration/preregistration_phase_b.md:481-489`
- Onde deveria estar: `docs/07_reports/living-paper/40_limitations_and_conclusion.md` secao Limitacoes ou trabalhos futuros; opcionalmente `20_method.md` nota operacional
- Estado atual no living paper: ausente. `40_limitations_and_conclusion.md` lista limitacoes de escopo e fronteira temporal (`docs/07_reports/living-paper/40_limitations_and_conclusion.md:32-60`), mas nao inclui a fragilidade operacional de derivacao de coorte por arquivo JSON fisico.
- Urgencia: P2
- Justificativa: nao muda as hipoteses, mas afeta reproducibilidade operacional e merece registro como divida tecnica pos-Phase B se o texto defender rastreabilidade forte.
- Esboco do que precisa ser escrito: declarar como limitacao operacional: ambos pipelines devem consumir o mesmo JSON fisico ate que `parent_sweep_id` seja derivado de forma unica pelo sistema.

# Itens revisados e descartados

- Politica raw vs post-guardrail: ja esta bem coberta em `20_method.md:48-148`, `10_problem_and_related_work.md:220-249`, `evidence_log.md:521-534` e `40_limitations_and_conclusion.md:80-85`.
- DM/HAC/MCS em nivel teorico: ja esta coberto em `10_problem_and_related_work.md:265-296`; o gap e apenas a familia unica de 6 testes e Tier 1/Tier 2 do F.2.
- PR #56 cleanup `gold_builder_adapters`: descartado como housekeeping/test hygiene; sem decisao metodologica nova.
- Stage R-E suffix fix em `dim_run` merge: descartado como bugfix interno; sua relevancia ja entra indiretamente em F.1 PASS.
- ABNT, lista de siglas/simbolos e `\hyperref`/`\nameref`: descartados como formatacao/editorial. Apenas a justificativa dos limiares dos gates virou GAP-07.
- Checklists `docs/05_checklists/*`: revisados apenas como contexto; nao ha recomendacao de editar checkboxes fechados.

# Recomendacoes de propagacao

- Ordem sugerida: primeiro GAP-01 a GAP-04 em `20_method.md`; depois GAP-05 em `tcc_mapping.md`; depois GAP-06 e GAP-07 em `evidence_log.md`/`30_results_and_analysis.md`.
- Cascata: ao fechar GAP-02/GAP-03/GAP-05, atualizar `tcc_mapping.md` linhas 4.5/4.6 e "Gaps P0" para refletir F.2 mergeado, baselines MVP implementados e Gate de saida da Fase A fechado.
- Nao editar `text/` nesta rodada de propagacao inicial. O fluxo correto e living paper primeiro, depois outra sessao decide se propaga para LaTeX.
- Nao editar checklists fechados. Se algum status antigo aparecer em checklist, tratar como historico imutavel e corrigir apenas documentos living/canonicos.

# Notas de rodape

- `gh pr list` falhou inicialmente por rede bloqueada e passou com permissao de rede. A lista confirmou PRs #50-#59 merged entre 2026-05-21 e 2026-05-23.
- Ha uma ambiguidade em F.2: `preregistration_phase_b.md:69-71` diz walk-forward desativado para Fase B confirmatoria, mas `preregistration_phase_b.md:198-204` fala em 3 folds e 15 runs. Antes de propagar texto definitivo, Marcelo deve confirmar se "folds" ali sao folds confirmatorios ou sensibilidade B.4.
- Nao encontrei evidencia de execucao confirmatoria Phase B: a secao `Execucao` do pre-registro ainda esta vazia (`docs/06_pre_registration/preregistration_phase_b.md:570-581`). Portanto nenhum gap de resultado confirmatorio foi listado.
