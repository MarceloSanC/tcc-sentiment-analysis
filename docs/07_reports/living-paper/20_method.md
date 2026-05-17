---
title: Living Paper - Method
scope: Base textual para o Capitulo 4 do TCC (Metodo) e Method do artigo. Descreve dados, features, modelo TFT, treinamento, inferencia, protocolo experimental, metricas e rastreabilidade.
update_when:
  - protocolo experimental for revisado (splits, seeds, folds)
  - novo metodo for incorporado (ex.: nova analise estatistica)
  - descricao de pipelines mudar
  - secao de reprodutibilidade for atualizada
canonical_for: [living_paper_method, tcc_chapter_4_source, experimental_protocol_source]
---

# 20 - Method

## Objetivo deste arquivo
Descrever de forma reproduzivel o metodo: dados, features, modelo, treinamento, inferencia e avaliacao.

Detalhamento do objetivo:
- Registrar o metodo com nivel de detalhe suficiente para reproducao tecnica.
- Documentar o fluxo completo do pipeline real: dados -> features -> treino TFT -> inferencia -> analytics.
- Definir protocolo experimental e criterios de comparacao justa entre configuracoes.
- Tornar explicito onde estao os mecanismos de rastreabilidade (run_id, fingerprints, contratos de schema).

Entrada e saida esperadas:
- Entrada: documentacao tecnica do repositorio (`README.md`, checklists e runbooks em `docs/`).
- Saida: secao de Metodo pronta para TCC e secoes Data/Method/Experimental Setup do artigo.

## Estrutura sugerida
- Dados e pipelines
- Engenharia de features
- Modelo (TFT)
- Protocolo experimental (splits, seeds, folds)
- Metricas pontuais e probabilisticas
- Reprodutibilidade e rastreabilidade (run_id, fingerprints, analytics store)

## Rascunho inicial

## Base do Método (v1 sincronizada com `text/4_Metodo`)
- Desenho: pesquisa quantitativa e experimental-computacional com foco em validade metodológica.
- Pipeline: ingestão -> features -> dataset -> treino TFT -> inferência -> atualização analítica.
- Dados: combinação de mercado, sentimento agregado, indicadores técnicos e fundamentais.
- Persistência: camadas imutáveis e auditáveis para dados primários de execução + camada analítica derivada e reconstruível para comparação.
- Engenharia: famílias de features com controle anti-leakage e governança de warmup.
- Modelo: TFT em modo pontual e quantílico, com rastreabilidade por assinaturas/fingerprints.
- Protocolo: splits temporais explícitos, múltiplas sementes e alinhamento estrito por `target_timestamp` em comparações pareadas.
- Métricas: pontuais (RMSE, MAE, DA, viés) e probabilísticas (quantis, cobertura, largura de intervalo).
- Qualidade: gates de integridade temporal, consistência contratual e monotonicidade de quantis.

## Política de variante quantilica (raw vs post-guardrail)

### O problema

O TFT em modo quantilico emite, por linha OOS, um trio `(q10, q50, q90)` cuja
monotonicidade `q10 <= q50 <= q90` **não é garantida pelo treinamento** —
quantile crossing é patologia conhecida de regressores quantilicos treinados
independentemente. No dataset histórico do projeto (smoke M7, ver
`docs/07_reports/phase-gates/A_code_audit.md` §M5-Q1), `gold_prediction_metrics_by_run_split_horizon`
contém linhas com `mpiw < 0` (intervalo "negativo", matematicamente
impossível como intervalo de predição) e `crossing_before_count = 402` num
amostra de 4 linhas auditadas. O efeito sobre métricas probabilísticas é
material: `picp`, `mpiw`, `mean_pinball`, `coverage_error` e
`confidence_calibrated` mudaram em 4/4 linhas entre raw e o trio rearranjado.

### A solução de pós-processamento

A pipeline aplica um **guardrail monotônico** (`QuantileGuardrailService.enforce_monotonic_triplet`,
implementado como `sorted([q10, q50, q90])` por linha) antes da persistência
em `fact_oos_predictions`, materializando as colunas paralelas
`quantile_p*_post_guardrail`. Esse procedimento corresponde ao **rearranjo
monotônico** estudado por Chernozhukov, Fernández-Val & Galichon (2010,
*Econometrica*, "Quantile and Probability Curves Without Crossing"), que
provam: para qualquer função de perda convexa simétrica em relação à mediana
— **incluindo pinball loss** — a perda esperada do estimador rearranjado é
**menor ou igual** à perda esperada do estimador original, com igualdade se
e somente se o estimador original já é monotônico. Isto é, o rearranjo é
melhoria estatisticamente válida, não cosmética.

### Decisão metodológica

Persistimos **ambas variantes em paralelo** no analytics store (`*_raw` e
`*_post_guardrail`) e declaramos explicitamente qual é primária por hipótese:

- **H1 (calibração marginal e PICP intervalar):** **post-guardrail**. "Calibração"
  exige objeto avaliado ser intervalo válido (`p10 <= p90`); raw com `mpiw < 0`
  não admite interpretação como intervalo. PICP raw sob crossing é matematicamente
  computável mas semanticamente indefinido.
- **H2a/H2b (pinball loss menor que baselines, DM com HAC/HLN, Holm-corrigido):**
  **post-guardrail**. Baselines admissíveis para o estudo (random walk, AR(1),
  média histórica, EWMA-vol, quantis empíricos historicos — ver `BASELINES.md`)
  são monotônicos por construção. Comparar TFT raw vs baseline monotônico
  penalizaria o TFT pelo crossing e geraria comparação não-pareada por construção
  de estimador. A simetria exige que ambos os lados passem pelo mesmo
  pós-processamento; como o rearranjo é trivial sobre estimadores já
  monotônicos (idempotente), aplicar-lo a ambos é sem custo conceitual mas
  preserva paridade.
- **H3 (contribuição relativa de famílias de features):** **post-guardrail**
  por conservadorismo. A análise complementar de ablation (Δ-pinball entre runs
  com e sem família) **pode** reportar variante raw em apêndice como
  diagnóstico de impacto sobre o aprendizado quantilico antes do mascaramento
  por rearranjo; a decisão fica no pré-registro.

### Por que reportar ambas

Persistir apenas a variante primária esconderia a magnitude do efeito do
guardrail e fragilizaria o claim metodológico. O paper reporta, em apêndice:

1. **Taxa de crossing pré-guardrail** (`% linhas com p10 > p90` por run/horizon).
2. **Δ-pinball, Δ-PICP, Δ-MPIW** entre raw e post-guardrail por run, agregado
   por config (média e percentil 95). Magnitudes pequenas dão evidência de que
   o ganho frente a baselines não vem do pós-processamento; magnitudes grandes
   limitam o claim a "o sistema entregue (TFT + guardrail) tem propriedade X",
   não "o TFT tem propriedade X".
3. **Pinball avaliada como scoring rule** segue Gneiting & Raftery (2007,
   *JASA*, "Strictly Proper Scoring Rules, Prediction, and Estimation"): o
   objeto avaliado é o estimador entregue, não a saída crua. Sob esse
   enquadramento, post-guardrail é o estimador (a pipeline TFT+sort), e
   pinball pós-guardrail é a perda do estimador pre-registrado.

### Restrições de aplicabilidade da variante

Nem toda métrica admite ambas variantes; algumas têm semântica anclada em
uma só (categorização completa em
`docs/04_evaluation/METRICS_DEFINITIONS.md` §"Variante quantilica"):

- **Métricas de risco financeiro** (`var_10`, `es_10_approx`): apenas
  **post-guardrail**. Value-at-Risk exige monotonicidade da função quantil
  por definição (Jorion 2007, *Value at Risk*; Acerbi & Tasche 2002,
  *Journal of Banking & Finance*). Sob crossing, `var_10_raw = mean(p10)`
  não é o quantil 10% — é o número que saiu da última camada do modelo na
  posição p10, sem propriedades de risk measure (monotonicidade,
  subaditividade) garantidas.
- **Checks de patologia** (`pred_interval_negative` no quality report; gate
  de degeneração do Stage 11): apenas **raw**. Post-guardrail tornaria
  `(p90 - p10) < 0` tautologicamente zero e mascararia `p10 == p90` herdado.
  O propósito desses checks é detectar a doença, não o efeito do remédio.

### Rastreabilidade

A variante usada por `gold_model_decision_final` (tabela de seleção do paper)
é declarada via flag `--primary-quantile-contract` e persistida como coluna
homônima no output, com default `post_guardrail`. Plots herdam a mesma flag.
Mudança de contrato primário entre versões do paper exige re-refresh com a
flag alternativa — não retreina o modelo.

## Critério de inclusão para inferências probabilísticas

Métricas probabilísticas (PICP, MPIW, pinball, coverage_error, prob_up/down,
confidence_calibrated) são reportadas apenas sobre rows elegíveis:
`prediction_mode == 'quantile'` em `fact_config` **e** `quantile_p10 ≠
quantile_p90` no **quantil bruto** (Categoria C de
`docs/04_evaluation/METRICS_DEFINITIONS.md` §"Variante quantilica": detectar
colapso emitido pelo modelo antes do guardrail mascarar). Métricas pontuais
(RMSE, MAE, MAPE, sMAPE, DA, viés) são calculadas sobre todas as rows
independentemente — não dependem do objeto quantilico.

Por grupo `(run_id, split, horizon)`, o predicado `is_quantile_genuine`
(permissivo: ≥ 1 row elegível) marca contribuição para o caminho
probabilístico; claims H1/H2a/H2b filtram por `is_quantile_genuine == True`
antes da agregação. O **gate estrito** de degeneração (`% rows com p10==p90
≥ 5%` bloqueia o run inteiro) é regra complementar de promoção (Stage 11 do
`PHASE_B_IMPLEMENTATION_CHECKLIST.md`), não de inclusão.

**Motivação empírica:** auditoria pré-Phase B
(`docs/07_reports/phase-gates/A_code_audit.md` §M5-Q5) registrou **91,25%
de degenerescência quantilica** em runs históricos com `parent_sweep_id=NULL`.
Métricas probabilísticas agregadas sobre intervalos de largura zero são
matematicamente sem interpretação. Esse achado justifica (a) o reset do
silver/gold pré-Phase B (Stage 3) e (b) reportar `n_probabilistic_samples`
junto a todo claim probabilístico do paper, distinguindo amostra avaliável
de amostra total.
