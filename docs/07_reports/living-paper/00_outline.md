---
title: Living Paper Outline
scope: Ancora conceitual do projeto de escrita do TCC e artigo. Consolida tese central, problema, pergunta, recorte, contribuicoes, hipoteses (H1/H2a/H2b/H3) e objetivos. Toda alteracao de escopo deve ser refletida primeiro em STRATEGIC_DIRECTION e depois aqui.
update_when:
  - tese, problema, pergunta de pesquisa ou recorte mudarem
  - hipoteses (H1/H2a/H2b/H3) forem reformuladas
  - objetivos especificos mudarem
  - secao de contribuicoes esperadas for revisada
canonical_for: [living_paper_outline, tcc_thesis, research_question, hypotheses, paper_objectives, paper_contributions]
---

# Living Paper - Outline

Status: draft vivo para alimentar TCC (ABNT) e artigo estilo ML.
Ultima revisao estrategica: 2026-05-01 (alinhado com `docs/00_overview/STRATEGIC_DIRECTION.md`).
Protocolo operacional de escrita: `docs/07_reports/living-paper/WRITING_PROTOCOL.md`.

## Objetivo do arquivo
Este arquivo funciona como ancora conceitual do projeto de escrita.
Seu papel e consolidar as definicoes centrais (tese, problema, pergunta, recorte,
contribuicoes e hipoteses) antes da escrita detalhada dos capitulos.

Uso esperado:
- Entrada: alinhamentos de orientacao, escopo tecnico do repositorio e decisoes metodologicas.
- Saida: base coerente para Introducao do TCC e Introduction/Contributions do artigo.
- Regra: qualquer alteracao de escopo deve ser atualizada primeiro em
  `docs/00_overview/STRATEGIC_DIRECTION.md` e depois propagada para este arquivo.

## 1) Visao geral
Este repositorio suporta uma pesquisa de previsao de series temporais financeiras
com foco em calibracao probabilistica, contribuicao de familias de indicadores,
e reprodutibilidade experimental — usando Temporal Fusion Transformer (TFT) como
modelo principal.

## 2) Tese central

Este trabalho sustenta que e possivel avaliar rigorosamente a qualidade probabilistica
de previsoes de retorno de ativo com TFT, caracterizando a contribuicao relativa de
familias de indicadores (preco, tecnico, fundamentalista, sentimento) por horizonte,
sob um protocolo experimental reproduzivel e com claims explicitamente delimitados ao
escopo do estudo (ativos avaliados, periodos analisados, sem generalizacao para
ativos/periodos nao avaliados ou afirmacoes causais). AAPL e o ativo piloto
inicial; cada modelo e especifico para um ativo.

O diferencial do trabalho nao e "prever o mercado com alta acuracia", mas:
- produzir um protocolo experimental reproduzivel e auditavel para forecasting
  financeiro com TFT e predicao quantilica;
- reportar calibracao probabilistica honesta (calibracao marginal por quantil,
  PICP intervalar, MPIW, reliability diagram) por horizonte, comparada a baselines
  explicitos;
- quantificar, de forma metodologicamente defensavel, quais familias de indicadores
  contribuem para a previsao do modelo — sem confundir contribuicao preditiva com
  causalidade.

## 3) Problema cientifico
Em previsao financeira, modelos podem aparentar bom desempenho sem garantir:
- robustez fora da amostra;
- comparabilidade justa entre configuracoes (temporal alignment, scope governance);
- quantificacao confiavel da incerteza preditiva (calibracao real vs PICP superficial);
- separacao entre "o modelo usou esta feature" e "esta feature causa o retorno".

## 4) Pergunta de pesquisa

Pergunta principal:
- Em series temporais de retorno de ativos avaliados sob o protocolo (AAPL como
  piloto inicial, horizontes h+1 e h+7), um TFT especifico por ativo treinado
  com todas as familias de indicadores disponiveis produz previsoes
  probabilisticas calibradas e estatisticamente superiores a baselines simples?

Pergunta secundaria:
- Quais familias de indicadores (preco, tecnico, fundamentalista, sentimento)
  contribuem de forma heterogenea e consistente para a previsao do modelo, segundo
  VSN weights nativos, permutation importance e ablation explicativa?

## 5) Recorte (delimitacao)
- Tarefa: previsao supervisionada de series temporais financeiras.
- Modelo principal: TFT (sem foco central em comparar arquiteturas distintas).
- Escopo de ativos: cada modelo e single-asset; a pipeline e multi-asset. AAPL
  e o ativo piloto inicial. Outros ativos reais podem entrar na analise final
  apenas se forem definidos no pre-registro com criterios de disponibilidade,
  qualidade e comparabilidade.
- Horizonte principal: h+1 e h+7; h+30 como suplementar se N_effective suficiente.
- Entradas: familias de indicadores — PRICE, TECHNICAL, SENTIMENT, FUNDAMENTAL.
  Candidato principal: "all-features com poda minima baseada em principio (nao em OOS)".
- Incerteza: previsao quantilica (p10, p50, p90) com guardrail monotonico.
- Avaliacao: metricas pontuais e probabilisticas + comparacao pareada OOS com
  alinhamento temporal estrito por `target_timestamp`.
- Reprodutibilidade: rastreabilidade por `run_id`, configuracao, split e artefatos
  em Parquet/DuckDB; decisoes reproduziveis sem retrain.
- Governança de coorte: toda comparação estatística confirmatória (DM, MCS,
  win-rate, ranking de seleção) é restrita a runs com mesmo `parent_sweep_id`,
  conforme `docs/03_modeling/SWEEPS_AND_SELECTION.md` e
  `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`.
- Fora de escopo: estrategia de trading em producao, otimizacao de carteira,
  inferencia intradiaria, generalizacao para ativos/periodos nao avaliados,
  inferencia causal, classificacao de regimes de mercado (future work).

## 6) Contribuicoes esperadas
- Protocolo reproduzivel de avaliacao probabilistica para forecasting financeiro
  com TFT, incluindo governance de escopo, pre-registro e artefatos versionados.
- Analise de calibracao probabilistica (calibracao marginal por quantil, PICP
  intervalar, MPIW, reliability diagram) comparada a baselines explicitos por
  horizonte.
- Caracterizacao da contribuicao relativa de familias de indicadores via tres
  metodos complementares: VSN weights, permutation importance, ablation explicativa.
- Analytics Store reproduzivel que permite comparacao entre configuracoes sem retrain.

## 7) Hipoteses de trabalho (v3 - 2026-04-25)

- **H1 (calibracao):** O candidato TFT all-features produz, para pelo menos
  um horizonte h in {1, 7}, calibracao marginal por quantil dentro de tolerancia
  pre-declarada para p10/p50/p90 e PICP intervalar adequado para p10-p90.

- **H2a (superioridade primaria):** Para pelo menos um horizonte h in {1, 7},
  o candidato TFT all-features apresenta pinball loss menor que o baseline
  primario pre-declarado de forma estatisticamente significativa.

- **H2b (evidencia forte adicional):** Para pelo menos um horizonte h in {1, 7},
  o candidato TFT all-features apresenta pinball loss menor que todos os
  baselines simples pre-declarados de forma estatisticamente significativa.

- **H3 (heterogeneidade de contribuicao):** A contribuicao agregada das familias
  de indicadores (PRICE, TECHNICAL, SENTIMENT, FUNDAMENTAL) e heterogenea entre
  horizontes, detectavel de forma consistente em pelo menos dois dos tres metodos
  (VSN weights, permutation importance, ablation explicativa).

Nota: qualquer hipotese pode ser refutada. Refutacao honesta e resultado valido
e academicamente preferivel a forcar claims frageis. A analise de por que uma
hipotese foi refutada tem valor explicativo proprio.

## 8) Mapeamento de uso
- Uso TCC: texto-base para Introducao (tema, problema, justificativa, objetivos e
  delimitacoes); hipoteses para Metodo e Resultados.
- Uso Artigo: Abstract/Introduction/Contributions/Hypotheses.

## 9) Objetivos do trabalho (v3 - 2026-04-25)

Objetivo geral:
- Avaliar rigorosamente a calibracao probabilistica de um Temporal Fusion Transformer
  aplicado a previsao multi-horizonte de retornos por ativo, e caracterizar a
  contribuicao relativa de familias de indicadores sob um protocolo estatistico
  explicito, reproduzivel e com pre-registro versionado.

Objetivos especificos:
1. Estruturar um fluxo integrado de dados financeiros (preco, indicadores tecnicos,
   sentimento, fundamentals) com controle de consistencia temporal e mitigacao de
   leakage.
2. Implementar treinamento e inferencia com TFT em modo quantilico, com rastreabilidade
   completa de configuracoes, artefatos e seeds.
3. Definir e aplicar protocolo experimental reproduzivel com pre-registro versionado,
   particionamento temporal, multiplas sementes, alinhamento estrito por
   `target_timestamp` e comparacao contra baselines explicitos.
4. Avaliar calibracao probabilistica com metricas completas (calibracao marginal
   por quantil, PICP intervalar, MPIW, pinball, reliability diagram) e comparar
   com baselines via DM com HAC e Holm-Bonferroni.
5. Caracterizar a contribuicao relativa de familias de indicadores via VSN weights,
   permutation importance e ablation explicativa, reportando convergencias e
   divergencias entre os tres metodos.

## 10) Proximo passo imediato (Fase A do roadmap)
Ver `docs/00_overview/STRATEGIC_DIRECTION.md` §5 (Fase A) para o roadmap completo.

Antes de escrever qualquer capitulo de Resultados:
- [ ] `A_code_audit.md` concluido com M1-M7 GREEN ou YELLOW com acao concluida
- [ ] Engenharia pré-Fase B concluída por `docs/05_checklists/phase-b/PHASE_B_IMPLEMENTATION_CHECKLIST.md` (Stages 1-11 da Fase B).
- [ ] Auditoria de leakage concluida (`docs/07_reports/external-reviews/leakage_audit_<date>.md`)
- [ ] Baselines persistidos na coorte (`fact_oos_predictions` com mesmos grao)
- [ ] Pre-registro versionado commitado (`docs/08_governance/preregistration_round1_<date>.md`)
- [ ] Seed de inferencia fixa documentada
- [ ] Escopo de ativos definido no pre-registro (AAPL piloto; outros ativos apenas
      se criterios de inclusao forem declarados)

## 11) Base da Introducao (v2 sincronizada com `text/2_Introducao`)
- Contextualizacao: forecasting financeiro como problema desafiador sob HME/HMA
  e dinamica adaptativa de mercado. R² OOS tipico de 0,1-1% para retornos diarios
  de acoes liquidas (Gu, Kelly & Xiu 2020) — expectativa calibrada.
- Problema: comparacoes pouco auditaveis entre configuracoes; foco excessivo em
  erro pontual sem calibracao probabilistica; confusao entre predicao e inferencia
  causal.
- Justificativa: necessidade de protocolo reproduzivel com TFT, avaliacao OOS
  calibrada, analise de contribuicao de features metodologicamente defensavel.
- Delimitacao: foco em TFT + covariaveis estruturadas + avaliacao pontual/probabilistica;
  modelos single-asset por ativo; AAPL como piloto inicial; sem claim causal,
  sem generalizacao para ativos/periodos nao avaliados, sem trading.
- Estrutura do trabalho: cap. 3 (fundamentacao), cap. 4 (metodo), cap. 5 (resultados),
  cap. 6 (conclusoes).

Referencias nucleares da Introducao:
- `FAMA1991`, `LO2004`
- `LIMZOHREN2021`, `LIMETAL2021TFT`
- `GNEITING2007`, `KOENKERBASSETT1978`
- `DIEBOLDMARIANO1995`, `HARVEY1997`
- `GU2020`, `RAPACH2013`
- `PINEAU2021REPRODUCIBILITY`
