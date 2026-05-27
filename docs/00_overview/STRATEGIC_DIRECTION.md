---
title: Strategic Direction
scope: Direcao estrategica viva do projeto. Define objetivo cientifico atual, pivo metodologico, hipoteses (H1/H2a/H2b/H3), claims aceitos vs proibidos, roadmap fases A-D e calibracao de expectativas. Documento mais critico do projeto — supersede planos anteriores.
update_when:
  - mudar objetivo cientifico, escopo ou claim central
  - alterar hipoteses (H1/H2a/H2b/H3) ou metrica primaria
  - mudar definicao do candidato unico (poda, hiperparametros, baselines)
  - alterar fases do roadmap (A/B/C/D) ou criterios de gate
  - mudar lista de claims aceitos vs proibidos
  - calibracao de expectativas (Secao 3) ficar obsoleta com nova evidencia
canonical_for: [strategic_direction, hypotheses, methodological_pivot, roadmap, claims_policy]
---

# Strategic Direction

Status: vivo (living). Ultima revisao estrategica: 2026-05-01.

Este documento registra o direcionamento estrategico atual do projeto apos
reavaliacao critica do escopo, dos objetivos e do caminho de execucao. Ele
**supersede** a abordagem descrita no antigo `POST_ROUND1_MODEL_SELECTION_CHECKLIST.md`
(removido) e deve ser consultado antes de qualquer decisao sobre o que
entra ou nao entra no TCC e no artigo.

Ordem de leitura recomendada apos este documento:
1. `docs/00_overview/PROJECT_OVERVIEW.md`
2. `docs/07_reports/living-paper/00_outline.md`
3. `docs/04_evaluation/BASELINES.md`
4. `docs/04_evaluation/CALIBRATION_AND_RISK.md`
5. `docs/04_evaluation/EXPLAINABILITY.md`
6. `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md`
7. `docs/05_checklists/ANALYTICS_STORE_CHECKLIST.md`

---

## 1) Por que este documento existe

A abordagem original do projeto girava em torno de:
- comparar multiplos conjuntos de features (sweep explicito por `feature_set`);
- encontrar a melhor combinacao de features via desempenho OOS;
- promover o "vencedor" como modelo de producao em uma Round 1 confirmatoria.

Essa abordagem tem dois defeitos metodologicos de fundo para um trabalho
academico defensavel:

1. **Selecao de features por desempenho OOS e uma forma mascarada de p-hacking.**
   Testar N combinacoes de familias e escolher a melhor pelo pinball/RMSE OOS
   inflaciona a taxa de falso positivo. Mesmo com correcoes (Holm, FDR, MCS),
   o desenho nao e defensavel como "provamos que a feature set X melhora".

2. **Confunde predicao com inferencia.** "Quais indicadores importam para o retorno"
   e uma pergunta associacional/causal; "qual modelo prediz melhor" e uma pergunta
   preditiva. Responder as duas com o mesmo experimento compromete ambas.

Alem disso, varios claims implicitos ("provar que TFT funciona em mercado",
"boa acuracia em prever retorno de AAPL") sao academicamente fragilizados
pela realidade empirica do problema (ver Secao 3).

Este documento fixa o novo enquadramento do projeto, o caminho de execucao
pragmatico a partir do estado atual e os claims que aceitamos ou rejeitamos
no texto final.

---

## 2) Objetivo do trabalho (reformulado)

**Claim unico e defensavel do TCC/artigo:**

> "Sob este protocolo experimental, para cada ativo incluido no estudo (AAPL
> como ativo piloto inicial), avaliamos em que medida um Temporal Fusion
> Transformer (TFT) especifico por ativo produz previsoes probabilisticas
> calibradas de retorno em multiplos horizontes e caracterizamos a contribuicao
> relativa de familias de indicadores (preco, tecnico, fundamentalista,
> sentimento) para a previsao. Nao fazemos claims causais nem de generalizacao
> para ativos ou periodos nao avaliados sob o protocolo."

**Objetivo geral:**
- Avaliar rigorosamente a **calibracao probabilistica** de um TFT aplicado a previsao
  multi-horizonte de retornos, e caracterizar a **contribuicao relativa** de familias
  de indicadores para a previsao, sob um protocolo estatistico explicito.

**Objetivos especificos:**
- Treinar um unico candidato "all-features" com hiperparametros otimizados e
  compara-lo com baselines simples (random walk, media historica, AR(1), EWMA-vol
  para quantis) em multiplos horizontes, separadamente por ativo.
- Reportar metricas pontuais e probabilisticas (RMSE, DA, pinball, PICP, MPIW)
  por horizonte, com intersecao temporal exata por `target_timestamp` e controle
  de multiplas comparacoes.
- Quantificar contribuicao de familias de features via tres medidas complementares:
  VSN weights nativos do TFT, permutation importance por familia, e ablation
  **explicativa** (N+1 runs removendo cada familia, sem selecao por desempenho).
- Publicar um pre-registro operacional versionado antes da rodada confirmatoria
  e um pacote de artefatos que permita reproducao sem retrain.

**O que saiu de escopo (anteriormente considerado, agora explicitamente fora):**
- Selecao de modelo por comparacao de conjuntos de features como evidencia primaria.
- Modelo de classificacao de regimes (acumulacao/distribuicao/reversao). Permanece
  como *future work*; nao entra neste TCC.
- Generalizacao para ativos ou periodos nao avaliados sob protocolo pre-registrado.
- Inferencia causal sobre o efeito de indicadores no retorno.
- Afirmacoes operacionais de trading ou portfolio.

---

## 3) Calibracao de expectativas (realismo do problema)

Antes de qualquer claim, fixar referencias empiricas honestas:

- **R-squared OOS tipico para retornos diarios de acoes liquidas: 0,1%-1%** (Gu,
  Kelly & Xiu 2020; Rapach & Zhou 2013). Nao e 10%+.
- **Directional accuracy > 54-55% em h+1 para ativo liquido single-asset e
  alerta vermelho.** Sinaliza mais provavelmente leakage ou overfit do que
  descoberta real.
- **Ativos liquidos e amplamente acompanhados tendem a ter edge preditivo
  marginal.** AAPL e o ativo piloto inicial e esta nessa categoria. Isso limita
  o ceiling esperado de qualquer modelo.
- **Modelos sao single-asset por ativo.** A pipeline pode processar multiplos
  ativos, mas cada modelo, coorte e conclusao deve ser rastreavel por ativo.
- **Multi-horizonte por ativo tem amostra efetiva pequena.** Em h+30 com
  ~3 anos OOS, as janelas se sobrepoem e o N nao-sobreposto e ~25. DM/MCS sob
  N=25 tem poder baixo. Isso precisa aparecer como limitacao.
- **TFT nao e dominante em retornos de acoes liquidas.** TFT costuma brilhar em
  series com padroes estruturais fortes (energia, trafego, varejo). Para retornos,
  o ganho sobre baselines simples frequentemente nao e estatisticamente significativo.
  Nao assumir superioridade; **testar** e reportar com honestidade.
- **Contribuicao de feature != causalidade.** VSN weights, attention e permutation
  importance respondem "o que o modelo usou", nao "o que causou o retorno".
  Todo texto que apresentar esses resultados deve explicitar essa diferenca.

**Consequencia pratica:** o melhor claim defensavel NAO e "TFT tem acuracia X".
E "TFT produz intervalos probabilisticos calibrados em regime Y com contribuicao
dominante das familias A, B em horizonte h". Calibracao probabilistica defensavel
e resultado publicavel mesmo quando o ponto medio nao supera naive.

---

## 4) Pivo metodologico (the core change)

### 4.1 De "comparar feature sets" para "candidato unico + analise de contribuicao"

| Dimensao | Abordagem antiga | Abordagem nova |
|---|---|---|
| Modelo principal | melhor entre N feature sets | 1 candidato "all-features" com poda minima |
| Selecao de features | por desempenho OOS (p-hacking) | nativa via VSN + analise explicativa |
| Papel dos sweeps | selecao de vencedor | exploracao de vizinhanca de hiperparametros |
| Round 1 confirmatoria | re-comparar vencedores | confirmar candidato unico vs baselines |
| Contribuicao de feature | implicita (conjunto vs conjunto) | explicita (VSN + permutation + ablation) |
| Escopo estatistico | comparacao K-wise com correcao | comparacao candidato-vs-baselines |

### 4.2 Definicao do candidato unico "all-features poda minima"

- **Inclui** todas as familias disponiveis: PRICE, TECHNICAL, SENTIMENT, FUNDAMENTAL.
- **Aplica poda unica baseada em principio (nao em OOS):**
  - remover features com correlacao par-a-par > 0,95;
  - remover features com > 30% missing em treino;
  - remover derivadas redundantes (ex.: `MA_5` e `EMA_5` juntos: manter apenas uma,
    criterio a priori documentado).
  - remover qualquer feature sem disponibilidade temporal comprovada no momento
    da inferencia;
  - para fundamentals, respeitar data real de publicacao/disponibilidade, nao
    apenas data de fechamento do periodo contabil;
  - para sentimento, respeitar timestamp de coleta/publicacao e corte da sessao
    usada para construir o alvo;
  - congelar lista final de features e regras de poda no pre-registro antes da
    rodada confirmatoria.
- **Hiperparametros:** tomados da vizinhanca das top-3 configuracoes da Round 0
  que ja incluiram conjuntos amplos de features. Sem re-otimizacao por feature set.
- **Treinamento:** mesma semente e mesmo split para candidato e baselines.

### 4.3 Estrategia de contribuicao de features (substitui selecao)

Tres medidas **complementares**, reportadas juntas:

1. **VSN weights** (nativo do TFT, Lim et al. 2021): agregados globalmente,
   por horizonte, e por regime (alta/baixa volatilidade).
2. **Permutation importance por familia:** embaralha todas as features de uma
   familia e mede queda de pinball. Reporta bootstrap IC95.
3. **Ablation explicativa:** treina N+1 runs (um com todas, N removendo cada
   familia). Reporta Δ-pinball e Δ-PICP com IC95.
   **Importante:** ablation e *analise*, nao criterio de selecao do modelo de
   producao. Nao escolhemos o modelo "sem SENTIMENT" mesmo que ele seja
   marginalmente melhor em pinball.

Cada uma dessas medidas responde a pergunta "quais familias importam" de angulo
diferente. Convergencia entre as tres fortalece o claim; divergencia vira
limitacao declarada.

Hierarquia de evidencia para contribuicao: global e por horizonte > agregada por
regime > local pontual. Explicacoes locais sao ilustrativas, nao generalizaveis,
e nunca devem ser escritas como causalidade.

### 4.4 Claims aceitos vs proibidos

**Aceitos (defensaveis):**
- "TFT+all-features produz intervalo p10-p90 com PICP X no horizonte h=1 para
  AAPL no periodo Y, com pinball menor/maior que baseline Z em Δ (IC95 [a, b])."
- "A familia SENTIMENT contribui Δ-pinball = W ± IC95 em ablation para h=1 em AAPL."
- "VSN weights agregados indicam que features da familia TECHNICAL dominam o peso
  medio de selecao dinamica do modelo em regimes de baixa volatilidade."

**Proibidos (nao serao escritos):**
- "Features fundamentalistas causam o retorno."
- "TFT supera o mercado / funciona para prever mercado."
- "O modelo tem acuracia de X% em prever retorno da AAPL" (sem baseline e sem IC).
- "Este resultado vale para ativos ou periodos nao avaliados."
- "Features selecionadas pelo nosso metodo sao as melhores features para
  previsao de retorno."

**Resultados validos mesmo com hipoteses refutadas:**
- Se o TFT nao superar o baseline primario, o trabalho ainda pode concluir sobre
  limites empiricos do TFT nos ativos avaliados sob protocolo OOS auditavel.
- Se o ponto medio nao melhorar RMSE/DA, mas os intervalos forem calibrados, o
  resultado sustenta contribuicao probabilistica, nao superioridade pontual.
- Se a calibracao falhar, o resultado sustenta diagnostico metodologico sobre
  risco, cobertura e limites do pipeline atual.
- Se VSN, permutation e ablation divergirem, a conclusao correta e instabilidade
  de contribuicao preditiva, nao irrelevancia definitiva de uma familia.

---

## 5) Roadmap pragmatico (fases A-D)

Cada fase tem escopo, criterio de aceite e artefato versionado.

### Fase A - Sanidade (bloqueante, ~2 semanas)

**Status:** concluida em 2026-05-23. Gate de saida satisfeito em
`docs/07_reports/phase-gates/phase-a/A_code_audit.md`; detalhes em
`docs/07_reports/phase-gates/phase-a/A_audit_closure_2026-05-17.md`.

**Artefato de gate:** `docs/07_reports/phase-gates/phase-a/A_code_audit.md`
Preencher veredictos por modulo antes de abrir qualquer treino confirmatorio.
O artefato detalha o gate operacional em modulos M1-M7, incluindo auditoria do
estado do Analytics Store e capacidade operacional da Fase B.

**A.0 Gate de degeneracao de quantis** (1 dia, obrigatorio antes de qualquer treino confirmatorio)
- Verificar nos primeiros runs de validacao do candidato all-features:
  `% linhas com p10 == p90 (MPIW=0)` deve ser `< 5%` no OOS.
- Se ultrapassar 5%: revisar `_extract_quantiles` no adapter de treino e confirmar
  que o pipeline usa o codigo pos-`f7901a4`.
- Identificar nos sweeps `0_2_3` os top-k configs com MPIW > 0 (nao-degenerados) como
  ponto de partida de hiperparametros para a Fase B.
- **Artefato:** registro da taxa de degeneracao no primeiro run confirmatorio,
  em `docs/07_reports/phase-gates/phase-a/A_code_audit.md` (M2-Q4).

**A.1 Auditoria de leakage** (3 dias)
- Inspecionar `time_varying_known_reals` no TFT: features derivadas de
  preco/volume/retorno **nao podem** estar ali.
- Confirmar que `StandardScaler` e demais scalers foram `fit` apenas em treino.
- Auditar se HPO das rodadas `0_1_x` usou `val` (ok) ou `test` (invalida o OOS
  atual como holdout limpo).
- **Artefato:** `docs/07_reports/external-reviews/leakage_audit_<date>.md` com
  conclusao binaria: sem leakage detectado / leakage detectado + escopo de invalidacao.

**A.2 Baselines persistidos na coorte** (3 dias)
- Implementar e persistir em `fact_oos_predictions` com mesmo grao dos candidatos TFT:
  random walk, media historica (janela longa), AR(1), EWMA de volatilidade
  (para quantis probabilisticos honestos).
- O baseline primario deve ser declarado no pre-registro antes da rodada
  confirmatoria. Recomendacao default a confirmar no pre-registro: random
  walk/zero-return para retorno; complementares: media historica, AR(1),
  EWMA-vol e/ou quantis historicos.
- **Artefato:** `gold_dm_pairwise_results` e `gold_mcs_results` atualizados
  incluindo baselines como candidatos comparaveis.

**A.3 Pre-registro operacional versionado** (1 dia, **bloqueante** para Fase B)
- **Artefato:** `docs/08_governance/preregistration_round1_<date>.md`.
- Conteudo obrigatorio: candidato unico declarado (features exatas, hiperparametros,
  janela de treino, seed), metrica primaria por horizonte (proposta: pinball loss
  agregado como primaria; DA e RMSE como secundarias), baselines oficiais,
  regra de desempate, threshold de relevancia pratica (ex.: Δpinball >= 3%),
  criterio de calibracao minima (calibracao marginal por quantil e PICP intervalar
  pre-declarados), politica para conflito entre horizontes.
- **Criterio de aceite:** commit hash do pre-registro referenciado no artefato
  final de Fase B. Qualquer mudanca pos-commit deve ser justificada em emenda
  datada no proprio arquivo.

**A.4 Higiene de inferencia** (1 dia)
- Seed de inferencia fixa e registrada.
- Dropout desligado em OOS (ou, se mantido por design, documentado e justificado).
- **Criterio de aceite:** reproducao bit-exact de `fact_oos_predictions` a partir
  de checkpoint + seed.

**A.5 Planejamento pos-auditoria**
- O plano temporario para as etapas apos o `A_code_audit` esta em
  `docs/08_governance/phase-a/POST_AUDIT_EXECUTION_PLAN.md`.
- Esse plano nao substitui o pre-registro da Fase B; ele apenas mapeia a ordem
  esperada ate o fechamento do escopo experimental.

### Fase B - Experimento principal confirmatorio (~3 semanas)

**Status:** concluida substantivamente em 2026-05-25 e preparada para merge
na PR-B em 2026-05-26. Relatorio final em
`docs/07_reports/phase-gates/phase-b/B_confirmatory_2026-05-25.md`;
pre-registro final em `docs/06_pre_registration/phase-b/preregistration_phase_b.md`.

Um unico experimento entra no paper como evidencia confirmatoria. Rodadas `0_X_X`
sao recaracterizadas no texto como "analise exploratoria de hiperparametros",
nao evidencia primaria.

**B.1 Execucao**
- Treinar candidato unico "all-features poda minima" com 5 seeds x 3 folds.
- Treinar/persistir baselines com mesmas seeds e folds.
- Cobrir h=1 e h=7 como horizontes principais. h=30 e suplementar por padrao
  e so entra em claim principal se N_effective e dependencia temporal permitirem
  inferencia minimamente defensavel (ver B.3).

**B.2 Metricas e decisao estatistica**
- Metricas por horizonte: pinball (primaria recomendada para pre-registro),
  calibracao marginal por quantil, PICP intervalar, MPIW, RMSE, DA, coverage_error.
- **Calibracao como gate de elegibilidade**: candidato so entra em comparacao
  estatistica se a calibracao marginal por quantil estiver nas bandas
  pre-declaradas (proposta: p10 in [0,07; 0,13]; p50 in [0,45; 0,55];
  p90 in [0,87; 0,93]) e o PICP do intervalo p10-p90 atender o criterio
  pre-registrado. Candidatos fora sao reportados como "rejeitados por
  calibracao" com justificativa.
- DM pairwise TFT vs cada baseline, com HAC/HLN para amostras pequenas.
- Holm-Bonferroni sobre o conjunto de DMs reportados.
- Reliability diagram com bootstrap (1000 resamples) por quantil e por horizonte.

**B.3 Transparencia de amostra**
- Reportar `n_common` por horizonte. Se N_effective de h+30 < 100, declarar como
  limitacao e manter h+30 apenas como resultado suplementar.

**B.4 Sensibilidade de janela temporal (escopo reduzido)**
- Rodar **2** janelas alternativas (uma curta, uma longa) com o candidato unico
  **declarado no pre-registro**. Nao otimizar por janela.
- Objetivo: documentar sensibilidade, nao selecionar.

### Fase C - Analise de contribuicao de features (~2 semanas)

Executada **sobre o candidato unico ja confirmado em B**.

**C.1 VSN weights agregados**
- Global, por horizonte, e por regime (alta/baixa volatilidade local).
- Tabela e figura por horizonte.

**C.2 Permutation importance por familia**
- Embaralhar features de cada familia, medir Δ-pinball.
- Bootstrap IC95 sobre 500-1000 resamples.
- Tabela por familia x horizonte.

**C.3 Ablation explicativa**
- N+1 runs: 1 com todas as familias, N removendo uma familia por vez.
- Mesmos seeds/folds do candidato.
- Tabela de Δ-pinball e Δ-PICP com IC95 por familia.

**C.4 Explicabilidade local ilustrativa**
- 3-5 casos (exemplos de alta acuracia e alta surpresa) com VSN weights e
  attention locais.
- Rotulados explicitamente como "instancia ilustrativa, nao generalizavel".

**Criterio de aceite da Fase C:** os tres metodos (C.1, C.2, C.3) sao reportados
lado a lado. Convergencia entre eles vira claim; divergencia vira limitacao declarada.

### Fase D - Escrita e defesa (~2 semanas)

- Atualizar `docs/07_reports/living-paper/30_results_and_analysis.md` com
  resultados das Fases B e C.
- Atualizar `docs/07_reports/living-paper/00_outline.md` com as hipoteses
  reformuladas (ver Secao 7 abaixo).
- Atualizar `docs/07_reports/living-paper/40_limitations_and_conclusion.md`
  com limitacoes **explicitas**: modelos single-asset por ativo, periodos
  avaliados, nao-causal, amostra efetiva em h+30 baixa, ativos liquidos com edge
  esperado baixo.
- Registrar todas as entradas relevantes em
  `docs/07_reports/living-paper/evidence_log.md`.
- **Future work:** regime classification, analise cross-asset mais ampla,
  inferencia causal, portfolio, trading.

---

## 6) Decisoes estrategicas tomadas em 2026-04-23

Registradas aqui para rastreabilidade academica:

1. **Round 1 confirmatoria baseada em "feature sets vencedores" esta suspensa.**
   A Round 1 sera confirmatoria do candidato unico vs baselines, nao re-comparacao
   entre conjuntos de features.

2. **Rodadas `0_X_X` sao reclassificadas como exploratorias** no texto final.
   Servem para identificar vizinhanca de hiperparametros, nao como evidencia
   primaria comparativa.

3. **PICP e calibracao probabilistica sao alvo primario.** RMSE e DA sao secundarios.
   Em AAPL, espera-se que o ganho em RMSE sobre random walk seja pequeno e
   frequentemente nao significativo; calibracao e onde esta a contribuicao real.

4. **Modelo de classificacao de regimes sai do escopo do TCC.** Mantido como
   future work com referencia a Hamilton (1989) e literatura de regime shifts.

5. **Cada modelo e especifico por ativo.** AAPL e o ativo piloto inicial; a
   pipeline e multi-asset, mas nenhum claim sera generalizado para ativos ou
   periodos nao avaliados sob protocolo pre-registrado.

6. **Pre-registro e bloqueante.** Nenhuma rodada confirmatoria comeca sem o
   arquivo em `docs/08_governance/preregistration_round1_<date>.md` commitado.

---

## 7) Hipoteses reformuladas (para `00_outline.md`)

As hipoteses H1-H3 do `00_outline.md` atual estao formuladas em torno de
"configuracoes com features enriquecidas melhoram OOS". Isso deve ser
substituido no proximo update do outline por:

- **H1 (reformulada):** O candidato TFT all-features produz, para pelo menos
  um horizonte h in {1, 7}, calibracao marginal por quantil dentro de tolerancia
  pre-declarada para p10/p50/p90 e PICP intervalar adequado para p10-p90.
- **H2a (reformulada, primaria):** Para pelo menos um horizonte h in {1, 7}, o
  candidato TFT all-features apresenta pinball loss menor que o baseline primario
  pre-declarado, de forma estatisticamente significativa (DM com HAC/HLN,
  Holm-corrigido quando aplicavel).
- **H2b (reformulada, evidencia forte adicional):** Para pelo menos um horizonte
  h in {1, 7}, o candidato TFT all-features apresenta pinball loss menor que todos
  os baselines simples pre-declarados (random walk/zero-return, media historica,
  AR(1), EWMA-vol e/ou quantis historicos), de forma estatisticamente significativa.
- **H3 (reformulada):** A contribuicao agregada das familias de features
  (PRICE, TECHNICAL, SENTIMENT, FUNDAMENTAL) e **heterogenea** entre horizontes,
  detectavel consistentemente em pelo menos dois dos tres metodos (VSN,
  permutation, ablation).

Nota: qualquer uma dessas hipoteses pode ser refutada, e isso e resultado valido.
Refutar honestamente e academicamente preferivel a forcar um claim frageis.

---

## 8) Rastreabilidade e supersessao

**Este documento supersede:**
- `docs/05_checklists/POST_ROUND1_MODEL_SELECTION_CHECKLIST.md` (removido em 2026-04-23).

**Este documento reformula (deve ser propagado):**
- `docs/07_reports/living-paper/00_outline.md` - hipoteses H1/H2a/H2b/H3, objetivos
  e recorte precisam ser reescritos conforme Secao 2 e Secao 7 deste arquivo.
- `docs/00_overview/PROJECT_OVERVIEW.md` - deve permanecer sincronizado com
  H1/H2a/H2b/H3, escopo e roadmap deste documento.
- `docs/07_reports/living-paper/tcc_mapping.md` - deve mapear capitulos do TCC
  para fontes primarias atualizadas e distinguir Round 0 exploratoria, Fase B
  confirmatoria e Fase C explicativa.
- `docs/07_reports/living-paper/30_results_and_analysis.md` - deve marcar Round 0
  como evidencia exploratoria e reservar claims confirmatorios para Fase B.
- `docs/07_reports/living-paper/40_limitations_and_conclusion.md` - deve
  consolidar limitacoes antes da escrita final das conclusoes.
- `docs/08_governance/phase-a/POST_AUDIT_EXECUTION_PLAN.md` - deve manter, de forma
  temporaria e revisavel, a sequencia planejada apos a conclusao do A_code_audit.
- `docs/05_checklists/CHECKLISTS.md` e `docs/INDEX.md` - referencias ao
  checklist removido foram atualizadas para apontar para este arquivo.

**Este documento depende de (continuam validos):**
- `docs/04_evaluation/BASELINES.md` - contrato canonico de baselines.
- `docs/04_evaluation/CALIBRATION_AND_RISK.md` - semantica de calibracao
  marginal por quantil, PICP intervalar, MPIW e risco.
- `docs/04_evaluation/EXPLAINABILITY.md` - contrato canonico de contribuicao
  preditiva e explicabilidade.
- `docs/05_checklists/PREDICTIONS_AND_METRICS_CHECKLIST.md` - catalogo de metricas.
- `docs/05_checklists/ANALYTICS_STORE_CHECKLIST.md` - contrato de dados.
- `docs/07_reports/phase-gates/phase-a/A_code_audit.md` - gate operacional da Fase A.
- `docs/07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md`
  - review externo cujas recomendacoes P0 permanecem em vigor.
- `docs/ai/AGENT_CORE.md` - regras globais de governanca de escopo.

---

## 9) Como usar este documento

- **Antes de iniciar qualquer rodada experimental final:** ler Secao 2 (objetivo),
  Secao 4 (pivo metodologico) e Secao 5 (fase atual).
- **Antes de escrever secao de resultados/discussao:** ler Secao 4.4 (claims
  aceitos vs proibidos) e Secao 3 (calibracao de expectativas).
- **Antes de propor novo experimento:** confirmar que ele nao reintroduz selecao
  por desempenho OOS entre conjuntos de features.
- **Quando surgir novo direcionamento estrategico:** atualizar este documento
  primeiro; depois propagar para outline, checklists e runbooks.

---

## 10) Referencias principais

- Gu, S.; Kelly, B.; Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies.
- Rapach, D. E.; Zhou, G. (2013). *Forecasting Stock Returns*. Handbook of Economic Forecasting.
- Lim, B.; Arik, S. O.; Loeff, N.; Pfister, T. (2021). *Temporal Fusion Transformers for interpretable multi-horizon time series forecasting*. International Journal of Forecasting.
- Diebold, F. X.; Mariano, R. S. (1995). *Comparing predictive accuracy*. Journal of Business & Economic Statistics.
- Harvey, D.; Leybourne, S.; Newbold, P. (1997). *Testing the equality of prediction mean squared errors*. International Journal of Forecasting.
- Hansen, P. R.; Lunde, A.; Nason, J. M. (2011). *The Model Confidence Set*. Econometrica.
- Gneiting, T.; Raftery, A. E. (2007). *Strictly proper scoring rules, prediction, and estimation*. JASA.
- Koenker, R.; Bassett, G. (1978). *Regression quantiles*. Econometrica.
- Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle*. Econometrica.
- Pineau, J. et al. (2021). *Improving reproducibility in machine learning research*. JMLR.
