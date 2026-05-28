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
`docs/07_reports/phase-gates/phase-a/A_code_audit.md` §M5-Q1), `gold_prediction_metrics_by_run_split_horizon`
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

A variante usada pelo refresh do analytics store (e portanto por todas as
tabelas gold legacy, incluindo `gold_model_decision_final` como diagnostico
exploratorio) é declarada via flag `--primary-quantile-contract` e persistida
como coluna homônima no output, com default `post_guardrail`. Plots herdam a
mesma flag. Mudança de contrato primário entre versões do paper exige
re-refresh com a flag alternativa — não retreina o modelo. A decisao
confirmatoria Phase B consome diretamente `pinball_loss_post_guardrail` via
sidecar `phase_b_dm_family_6.parquet`, sem depender de
`gold_model_decision_final` (ver §"Protocolo confirmatorio da Phase B
(sealed)" abaixo).

A política assume silver pos-Stage 8 (colunas `quantile_p*_post_guardrail`
materializadas no momento da escrita). Runs anteriores ao reset documentado
em `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` §"Archive pre-Phase B"
(2026-05-10) ficam com `*_post_guardrail = NaN` e não são elegíveis para
claims confirmatórios.

## Protocolo confirmatorio da Phase B (sealed)

A Phase B opera em escopo unico e fechado, declarado ex-ante no
pre-registro [`preregistration_phase_b.md`](../../06_pre_registration/phase-b/preregistration_phase_b.md)
e operacionalizado pela Emenda E1.8 (DM family-6).

Parametros sealed do protocolo confirmatorio:

- **Cohort:** `parent_sweep_id = phase_b_confirmatorio_20260524`.
- **Ativo:** AAPL (single-asset; sem claim de generalizacao para outros tickers).
- **Horizontes confirmatorios:** h=1 e h=7. h=30 fica explicitamente out-of-scope.
- **Candidato:** TFT all-features (config sealed sha256
  `fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4`,
  derivado do top-1 robust_score do sweep heritage `phase_b_hpo_20260523`).
- **Replicas TFT:** 15 = 3 folds walk-forward x 5 seeds.
- **Baselines pre-declarados:** `zero_return`, `historical_mean_rolling` (w=30),
  `historical_quantiles_rolling` (w=252); 15 replicas cada, mesmas folds e seeds.
- **Total dim_run cohort:** 60 (15 TFT + 45 baselines).
- **Perda primaria:** `pinball_loss_post_guardrail` (Categoria A;
  variante primaria declarada por hipotese conforme secao anterior).
- **DM family-6:** 2 horizontes x 3 baselines, HAC Newey-West com
  `lag = max(h-1, 1)`, correcao Harvey-Leybourne-Newbold (HLN),
  one-sided `H_A: TFT < baseline`, Holm-Bonferroni sobre a familia de 6.
- **Bandas Tier 1 / Tier 2:** declaradas em §9 do pre-registro e em
  [`CALIBRATION_AND_RISK.md`](../../04_evaluation/CALIBRATION_AND_RISK.md);
  a politica de promocao e mecanica, sem reframing pos-observacao.

### Sidecars confirmatorios da Phase B

A decisao confirmatoria nao consome o gold legacy de
`data/analytics/gold/`. O pos-processador
[`src.main_compute_phase_b_tier_metrics`](../../06_runbooks/phase-b/RUN_PHASE_B_TIER_CLASSIFICATION.md)
materializa 5 sidecars em
`data/analytics/reports/phase_b/cohort=phase_b_confirmatorio_20260524/`:

| Sidecar | Papel confirmatorio |
|---|---|
| `phase_b_marginal_coverage.parquet` | Calibracao marginal q10/q50/q90 dos runs TFT (sustenta H1). |
| `phase_b_tier_verdict.parquet` | Veredito mecanico tier por (hipotese, horizonte). |
| `phase_b_dm_family_6.parquet` | Familia primaria DM/Holm (sustenta H2a e H2b). |
| `phase_b_dm_family_18_sensitivity.parquet` | Sensibilidade conservadora 3 folds x familia 6; nao alimenta tier. |
| `phase_b_delta_pinball.parquet` | Δpinball relativo TFT vs cada baseline por horizonte. |

### Separacao entre Phase B sidecars e gold legacy

`gold_dm_pairwise_results`, `gold_mcs_results`, `gold_win_rate_pairwise_results`
e `gold_model_decision_final` permanecem disponiveis no analytics store, mas
**nao** sustentam claim confirmatorio Phase B. Essas tabelas (i) operam sobre
`squared_error` em vez da perda primaria pinball post-guardrail, (ii) aplicam
filtro top-50 antes do DM/MCS (inferencia pos-selecao), (iii) Holm legacy nao
preserva `split_signature`, e (iv) MCS hard-codeia `B=300, block_len=5`. Em
escopo Phase B, essas tabelas servem como diagnostico exploratorio
within-family, nao como fonte de inferencia cross-family. A revisao
metodologica formal dessas tabelas vive em
[`C0_statistical_methods_hardening.md`](../phase-gates/phase-c/C0_statistical_methods_hardening.md)
e e pre-requisito para qualquer reuso confirmatorio futuro.

A Phase B fecha o ciclo confirmatorio **somente** para a cohort sealed.
Rodadas futuras com outro feature set, outro ativo, outro horizonte ou
melhoria arquitetural devem ter pre-registro proprio, cohort propria e nao
podem reinterpretar a Phase B retroativamente.

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
(`docs/07_reports/phase-gates/phase-a/A_code_audit.md` §M5-Q5) registrou **91,25%
de degenerescência quantilica** em runs históricos com `parent_sweep_id=NULL`.
Métricas probabilísticas agregadas sobre intervalos de largura zero são
matematicamente sem interpretação. Esse achado justifica (a) o reset do
silver/gold pré-Phase B (Stage 3) e (b) reportar `n_probabilistic_samples`
junto a todo claim probabilístico do paper, distinguindo amostra avaliável
de amostra total.
