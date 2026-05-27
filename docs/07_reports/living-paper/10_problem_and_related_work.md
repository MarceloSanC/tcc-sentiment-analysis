---
title: Living Paper - Problem and Related Work
scope: Base textual para o Capitulo 3 do TCC (Fundamentacao Teorica) e Related Work do artigo. Cobre previsao de series financeiras, modelagem com deep learning, TFT, predicao quantilica, avaliacao OOS e reprodutibilidade.
update_when:
  - revisao de literatura for expandida
  - novas referencias bibliograficas centrais forem incorporadas
  - delimitacao do escopo da revisao mudar
canonical_for: [living_paper_related_work, tcc_chapter_3_source, literature_review_source]
---

# 10 - Problem and Related Work

## Objetivo deste arquivo
Registrar, em linguagem de artigo, o contexto do problema e a revisao de literatura orientada ao recorte da pesquisa.

Detalhamento do objetivo:
- Definir o estado da arte relevante ao recorte (forecasting financeiro, TFT/transformers, previsao quantilica, avaliacao OOS).
- Explicitar lacunas tecnicas/metodologicas que justificam o trabalho.
- Posicionar claramente a contribuicao do projeto frente a trabalhos relacionados.
- Evitar revisao ampla sem foco no problema de pesquisa definido no `00_outline.md`.

Entrada e saida esperadas:
- Entrada: tese central, pergunta de pesquisa e recorte.
- Saida: narrativa de revisao pronta para alimentar a Fundamentacao Teorica (TCC) e Related Work (artigo).

## Estrutura
A revisao segue as seis subsecoes congeladas do Capitulo 3 ABNT
(`chapter_structure_freeze.md`):

1. Previsao de series temporais financeiras
2. Modelagem com deep learning para series temporais
3. Temporal Fusion Transformer (TFT)
4. Predicao probabilistica e quantilica
5. Avaliacao de modelos em ambiente OOS e testes estatisticos
6. Reprodutibilidade experimental e rastreabilidade

Cada subsecao desenvolve, em prosa, (a) o estado da arte relevante e (b) as
implicacoes metodologicas para o desenho experimental adotado. Definicoes
formais (formulas, thresholds, contratos de tabela) **nao sao redefinidas**
aqui — sao recuperadas dos documentos canonicos via cross-link textual, em
linha com o principio "one topic, one source of truth" declarado em
`docs/INDEX.md`.

## 3.1 Previsao de series temporais financeiras

A previsao de retornos em mercados financeiros e historicamente tratada como
um problema de fronteira entre teoria informacional e evidencia empirica. Em
sua segunda revisao da hipotese de mercados eficientes, Fama (1991) consolida
a tese de que pre-cos refletem rapidamente toda a informacao publica
disponivel, de modo que retornos anormais sustentaveis dependem ou de
informacao privada, ou de exposicao a fatores de risco nao precificados, ou
de fricoes de mercado especificas. O corolario metodologico relevante para
este trabalho e direto: previsibilidade estavel de retornos lineares e a
excecao, nao a regra, em ativos liquidos.

Em contraponto, Lo (2004) formula a hipotese de mercados adaptativos, que
recolhe os mesmos fatos estilizados sob uma logica evolucionaria: eficiencia
e ineficiencia coexistem ao longo do tempo em funcao de competicao entre
estrategias, mudancas de regime e adaptacao de agentes. A consequencia
pratica nao e a refutacao de Fama, mas o reconhecimento de que sinais
preditivos quando existem tendem a ser contextuais, multivariados e
temporalmente localizados, em vez de globalmente persistentes. Dois
fenomenos amplamente documentados reforcam essa leitura: nao estacionariedade
de primeira e segunda ordem em retornos diarios, e clustering de
volatilidade.

Evidencia empirica recente recalibra a expectativa de magnitude do sinal.
Gu, Kelly e Xiu (2020), em revisao sistematica de aprendizado de maquina
aplicado a apreca-mento de ativos, reportam R^2 fora da amostra tipico entre
0,1% e 1% para retornos diarios de acoes liquidas mesmo com modelos
flexiveis e amplo conjunto de preditores. Rapach e Zhou (2013), em survey
sobre previsao de retornos agregados, chegam a magnitudes compativeis e
discutem como pequenos R^2 OOS podem ainda ser economicamente relevantes
quando combinados a alocacao informada. Essa literatura ancora uma
expectativa metodologica central para o presente trabalho: ganhos preditivos
modestos sao a norma, e qualquer protocolo que reporte ganhos
sistematicamente altos deve ser tratado com suspeita de leakage ou
data snooping antes da interpretacao economica.

Implicacao para o desenho experimental. A combinacao de retornos pouco
previsiveis, regimes nao estacionarios e R^2 OOS pequenos torna a robustez
metodologica mais informativa do que o numero pontual de erro: comparacao
pareada contra baselines explicitos, alinhamento temporal estrito,
avaliacao quantilica e correcao para multiplos testes deixam de ser
ornamento academico e passam a ser condicao necessaria para que o claim
"o modelo superou o baseline" carregue significado. O escopo congelado
deste trabalho (modelos single-asset, AAPL como piloto inicial, horizontes
h+1 e h+7) reflete diretamente essa restricao: nao se afirma generalizacao
inter-ativos ou inter-periodos que a literatura nao sustentaria.

## 3.2 Modelagem com deep learning para series temporais

Lim e Zohren (2021) sistematizam, em survey dedicado, o estado da arte em
forecasting de series temporais com aprendizado profundo, organizando
familias de arquiteturas (recorrentes, convolucionais, baseadas em atencao
e hibridas) e papeis distintos que podem desempenhar (previsao pontual,
multi-horizonte, probabilistica, multivariada). Dois pontos da revisao sao
particularmente pertinentes a este trabalho. Primeiro, modelos profundos
oferecem flexibilidade representacional para capturar nao linearidades,
interacoes entre covariaveis heterogeneas e dependencias temporais de
diferentes escalas, atributos diretamente relevantes para forecasting
financeiro com indicadores tecnicos, sentimento e fundamentos. Segundo,
essa flexibilidade vem acompanhada de risco aumentado de overfitting e de
sensibilidade a leakage temporal sutil, sobretudo quando o protocolo
experimental nao e desenhado com rigor.

A consequencia metodologica reforca a discussao do item anterior: a
adocao de deep learning em forecasting financeiro nao se justifica por
premissa de superioridade universal, mas por aderencia ao tipo de dado
(multivariado, multi-escala, multi-horizonte) e por adequacao a um
protocolo de avaliacao que controla explicitamente as fontes de inflacao
artificial de desempenho. Nao basta declarar splits temporais; e preciso
verificar leakage de features (scaler com `fit` em validacao/teste,
indicadores tecnicos calculados com janela contendo dia atual, sentimento
agregado fora da sessao de mercado etc.), leakage de selecao
(hiperparametros escolhidos com o conjunto de teste) e leakage de
alinhamento (intersecao temporal incompleta em comparacoes pareadas).

A definicao formal de leakage temporal utilizada neste trabalho esta em
`docs/00_overview/GLOSSARY.md` §G; o tratamento operacional aparece em
`docs/02_data/FEATURE_SETS.md` (tags anti-leakage por familia) e em
`docs/02_data/QUALITY_GATES.md` (checks de integridade temporal). A revisao
externa registrada em
`docs/07_reports/external-reviews/claude_scope_governance_review_2026-04-09.md`
formaliza a auditoria recorrente de leakage como pre-requisito para
qualquer claim confirmatorio.

Implicacao para o desenho. A escolha por uma arquitetura profunda neste
trabalho e instrumental: serve para representar interacoes entre familias
de features e horizontes em um mesmo modelo, com mecanismos nativos de
selecao de variaveis e atencao temporal. Ela nao dispensa, mas torna mais
critica, a disciplina experimental de pre-registro, cohort governance e
alinhamento OOS exato.

## 3.3 Temporal Fusion Transformer (TFT)

O Temporal Fusion Transformer, proposto por Lim, Arik, Loeff e Pfister
(2021), e um modelo de previsao multi-horizonte que combina componentes
recorrentes e de atencao com mecanismos explicitos de selecao de variaveis
e cabeca quantilica. Sua arquitetura organiza a inferencia em estagios
funcionais distintos: (i) processamento de variaveis estaticas via
encoders especificos; (ii) Variable Selection Networks (VSN) que atribuem
pesos dinamicos por timestep e por familia de feature, separando entradas
estaticas, conhecidas no futuro (calendario, feriados planejados) e
desconhecidas no futuro (preco, retorno, indicadores tecnicos);
(iii) blocos recorrentes LSTM no estagio encoder/decoder que carregam
estado temporal entre passos; (iv) atencao multi-head que captura
dependencias de longo prazo entre periodos relevantes; e (v) cabeca
multi-quantile que emite, por horizonte, multiplos quantis condicionais do
alvo em um unico passo de inferencia.

Tres caracteristicas justificam a escolha do TFT como modelo principal
deste trabalho. Primeiro, o TFT integra covariaveis estruturadas
heterogeneas em um unico estimador, alinhado ao recorte de familias
PRICE/TECHNICAL/SENTIMENT/FUNDAMENTAL descrito em
`docs/02_data/FEATURE_SETS.md`. Segundo, ele e nativamente multi-horizonte,
o que evita a necessidade de re-treinar modelos independentes por horizonte
e simplifica a politica de persistencia adotada em
`docs/03_modeling/MULTI_HORIZON.md` (uma linha por horizonte, mesmo
checkpoint). Terceiro, sua cabeca quantilica produz, sem reajuste posterior,
o objeto probabilistico exigido pelas hipoteses H1, H2a e H2b do trabalho:
quantis condicionais por linha OOS, base para pinball loss e para metricas
de calibracao.

A interpretabilidade nativa do TFT, via VSN weights e atencao por timestep,
e tratada neste trabalho como base do nivel mais forte de evidencia para
analise de contribuicao de features, conforme hierarquia metodologica
declarada em `docs/04_evaluation/EXPLAINABILITY.md` (global por horizonte
> agregada por regime > local pontual). O proprio Lim et al. (2021)
enfatiza, no entanto, que pesos de selecao indicam que variaveis o modelo
usou sob um protocolo, ativo e periodo, e nao quais variaveis causam o
alvo — distincao incorporada explicitamente como linguagem proibida no
mesmo documento canonico.

Limites conhecidos. A capacidade representacional do TFT amplifica o risco
discutido em §3.2: hiperparametros, seed, particionamento e composicao de
features tem impacto material sobre a saida quantilica, incluindo
patologias como quantile crossing e colapso de quantis (`p10 == p90`)
observadas no analytics store deste projeto. Esses fenomenos motivam o
pos-processamento por guardrail monotonico (definicao operacional em
`docs/00_overview/GLOSSARY.md` §G e justificativa metodologica em
`docs/07_reports/living-paper/20_method.md` §Politica de variante
quantilica), que e parte integral do estimador entregue.

## 3.4 Predicao probabilistica e quantilica

A regressao quantilica, introduzida por Koenker e Bassett (1978), e a base
formal para estimacao de quantis condicionais sem assumir forma parametrica
para a distribuicao do erro. Em vez de minimizar o erro quadratico medio,
estima-se cada quantil de interesse por minimizacao da pinball loss,
funcao assimetrica que penaliza desvios em direcoes opostas com pesos
distintos. A formula canonica de pinball por linha e por agregado esta em
`docs/04_evaluation/METRICS_DEFINITIONS.md` §Probabilistic Metrics; aqui
basta destacar que a perda media de pinball entre quantis e a metrica
primaria deste trabalho para avaliacao probabilistica, conforme declarado
em `docs/00_overview/GLOSSARY.md` §M.

A justificativa para tratar pinball como metrica primaria, e nao apenas
intervalos como PICP, vem do trabalho seminal de Gneiting e Raftery (2007)
sobre scoring rules. Os autores estabelecem o conceito de regra
estritamente propria, isto e, uma funcao de avaliacao cuja minimizacao em
esperanca ocorre se e somente se a previsao reportada coincide com a
distribuicao verdadeira do alvo. Pinball loss e estritamente propria para
o quantil correspondente, o que significa que reportar quantis honestos e
estrategia dominante para o predictor; em contraste, metricas de cobertura
agregada como PICP podem ser otimizadas com estrategias degeneradas
(intervalos artificialmente largos) e portanto nao sao adequadas como
funcao objetivo primaria, ainda que sejam complementos necessarios para
diagnostico de calibracao.

Avaliacao probabilistica neste trabalho distingue formalmente duas formas
de calibracao, conforme `docs/04_evaluation/CALIBRATION_AND_RISK.md`:
(a) calibracao intervalar, medida por PICP do par (p10, p90) contra a
cobertura nominal de 0,80; e (b) calibracao marginal por quantil, medida
pela taxa empirica `coverage_qtau = mean(y_true <= quantile_p_tau)` contra
o nominal `tau`. Largura media (MPIW) e analisada conjuntamente com PICP:
intervalo estreito com cobertura baixa indica modelo sistematicamente
errado, nao "preciso".

Patologia recorrente em regressores quantilicos treinados independentemente
e o quantile crossing, situacao em que `p10 > p50` ou `p50 > p90` em parte
das linhas. No analytics store deste projeto, a auditoria pre-Phase B
documentada em `docs/07_reports/phase-gates/phase-a/A_code_audit.md` §M5-Q1
registrou 402 crossings em amostra de 4 linhas e ate quatro linhas com MPIW
negativo. Chernozhukov, Fernandez-Val e Galichon (2010) demonstram, em
artigo no `Econometrica`, que o rearranjo monotonico da funcao quantil
estimada produz estimador com perda esperada menor ou igual a do estimador
original para qualquer perda convexa simetrica em torno da mediana,
incluindo pinball, com igualdade somente quando o estimador original ja e
monotonico. Esse resultado teorico ancora a decisao deste trabalho de
declarar a variante `post_guardrail` (rearranjo via sort por linha) como
primaria para H1, H2a e H2b, com a variante `raw` reportada em paralelo
para auditar a magnitude do efeito; a politica completa esta em
`docs/04_evaluation/METRICS_DEFINITIONS.md` §Variante quantilica.

Aplicacao a risco. Quando os quantis previstos servem de base para
metricas de risco financeiro, a monotonicidade nao e apenas conveniencia
estatistica: e pre-requisito de definicao. Jorion (2007), em referencia
classica sobre Value-at-Risk, define VaR como quantil da distribuicao de
perda, exigindo que o estimador da funcao quantil seja monotonico para
preservar a propria semantica de "perda maxima ao nivel `alpha`". Acerbi
e Tasche (2002), em discussao sobre coerencia de medidas de risco,
formalizam propriedades (monotonicidade, subaditividade, homogeneidade,
invariancia translacional) que Expected Shortfall satisfaz como medida
coerente sob ordenacao bem definida dos quantis. Sob crossing, nem VaR
nem ES preservam essas propriedades; por isso, as metricas de risco
deste trabalho usam exclusivamente a variante `post_guardrail`,
classificada como Categoria B em
`docs/04_evaluation/METRICS_DEFINITIONS.md` §Variante quantilica.

## 3.5 Avaliacao de modelos em ambiente OOS e testes estatisticos

Comparacao entre modelos sob criterio OOS exige, antes de qualquer teste
estatistico, alinhamento temporal exato. Diferentes modelos podem ter
inferencias persistidas sob diferentes coberturas (por falha pontual de
ingestao, por exigencia de warmup distinto, por filtro de regime), e
calcular medias de erro sobre amostras nao identicas confunde diferenca de
modelagem com diferenca de cobertura. O regime formal adotado neste
trabalho exige intersecao exata por `(asset, split, horizon,
target_timestamp)` antes de qualquer estatistica pareada; a decisao
arquitetural correspondente esta documentada em ADR-0002
(`docs/01_architecture/decisions/ADR-0002-oos-alignment.md`), e o protocolo
estatistico em `docs/04_evaluation/STATISTICAL_TESTS.md`.

O teste de Diebold e Mariano (1995), referencia canonica para comparacao
de acuracia preditiva, opera sobre a serie de perdas pareadas entre dois
modelos e testa a hipotese nula de igualdade de perda esperada. Sua
aplicacao direta, no entanto, e conhecidamente conservadora ou
anti-conservadora em amostras pequenas, dependendo da estrutura de
autocorrelacao das perdas. Harvey, Leybourne e Newbold (1997) propoem a
correcao HLN para a estatistica DM, baseada em ajuste de variancia HAC e
em distribuicao t com graus de liberdade adaptados ao tamanho efetivo da
amostra; essa correcao e o procedimento default usado pelos relatorios
gold deste projeto para horizontes h+1 e h+7 sob amostra OOS limitada.

Comparacoes multiplas entre o candidato TFT all-features e os baselines
admissiveis (random walk, AR(1), media historica rolling, EWMA-vol,
quantis empiricos historicos, conforme `docs/04_evaluation/BASELINES.md`)
inflam a taxa de erro tipo I quando reportadas sem correcao. O ajuste
adotado e Holm-Bonferroni, controle de Family-Wise Error Rate por
passos descendentes, persistido como `pvalue_adj_holm` nos artefatos gold.
A regra de decisao para H2a (superioridade contra baseline primario) e
H2b (superioridade contra todos os baselines simples) usa
`pvalue_adj_holm < 0,05` como criterio formal de significancia, conforme
`docs/04_evaluation/STATISTICAL_TESTS.md` §Multiple Testing.

Quando se deseja, em vez de um vencedor pontual, identificar o conjunto de
modelos estatisticamente indistinguiveis do melhor, recorre-se ao Model
Confidence Set proposto por Hansen, Lunde e Nason (2011). O procedimento
controla FWER por construcao e retorna o conjunto de configuracoes nao
rejeitadas como melhores ao nivel `alpha`, em vez de forcar um ranking
unico. Este trabalho usa MCS com `alpha = 0,05` no fluxo confirmatorio da
Fase B como leitura complementar a DM pareado: divergencia entre as duas
leituras (por exemplo, um modelo ganha em DM mas o MCS aceita varios
modelos) e ela mesma evidencia metodologica relevante, registrada em
`gold_dm_pairwise_results.parquet` e `gold_mcs_results.parquet`.

A governanca de coorte e parte do protocolo estatistico, nao um detalhe
operacional. Conforme `docs/ai/AGENT_CORE.md` §Scope Governance e
`docs/03_modeling/SWEEPS_AND_SELECTION.md`, toda comparacao confirmatoria
e restrita a runs com o mesmo `parent_sweep_id` (mesma coorte de
hiperparametros e mesma fronteira temporal). Comparacoes globais sem
restricao de coorte sao usadas apenas como saude da plataforma
(`global_health`) e nao podem ser reportadas como evidencia confirmatoria.

## 3.6 Reprodutibilidade experimental e rastreabilidade

Pineau et al. (2021), em relatorio sobre o programa de reprodutibilidade
da NeurIPS 2019, documentam que uma fracao significativa de trabalhos
publicados em conferencias de ponta em aprendizado de maquina permanece
dificil ou impossivel de reproduzir mesmo com codigo aberto, em razao de
omissoes em (i) dados exatos de entrada, (ii) sementes e versoes de
biblioteca, (iii) hiperparametros, e (iv) procedimento exato de splits e
selecao de modelo. Os autores propoem como mitigacao um conjunto minimo de
artefatos obrigatorios, incluindo descricao explicita do pipeline,
controle de aleatoriedade e disponibilidade de checkpoints. A consequencia
para trabalhos academicos com pretensao de defensabilidade e clara:
reprodutibilidade deixa de ser virtude operacional opcional e passa a
condicao necessaria de validade.

Esse posicionamento ancora a politica adotada neste trabalho.
Reprodutibilidade aqui significa que decisoes finais sao reproduziveis a
partir de artefatos persistidos, sem necessidade de re-treinamento.
Operacionalmente, isso implica tres camadas. Primeiro, identidade logica
deterministica por run: o `run_id` definido em
`docs/00_overview/GLOSSARY.md` §P combina hashes de asset,
`feature_set_hash`, `trial_number`, fold, seed, `model_version`,
`config_signature`, `split_signature` e `pipeline_version` em um SHA-256
estavel, de modo que o mesmo experimento sempre produz o mesmo
identificador. Segundo, fingerprints de dataset, configuracao e split
asseguram que comparacoes pareadas estao sobre entradas identicas (ver
`docs/02_data/DATA_CONTRACTS.md`). Terceiro, separacao entre camadas
imutaveis (fonte primaria, silver) e camadas reconstruiveis (gold)
permite que reanalises rodem sem retrain, conforme decisao registrada em
ADR-0001 (`docs/01_architecture/decisions/ADR-0001-analytics-store.md`).

O pre-registro versionado e o componente final da politica de
reprodutibilidade. Conforme `docs/00_overview/GLOSSARY.md` §G e
`docs/08_governance/EXPERIMENT_TRACKING_POLICY.md`, o pre-registro fixa,
antes da execucao da rodada confirmatoria: candidato, baselines, metrica
primaria, contrato de variante quantilica (`primary_quantile_contract`),
bandas de calibracao, regra de desempate, threshold de relevancia
pratica. O ato de versionar e commitar essa declaracao antes de observar
os resultados elimina graus de liberdade que, na ausencia de
pre-registro, geram p-hacking e selecao adversa de baselines.

Implicacao para o desenho. A combinacao de identidade deterministica,
fingerprints, camadas imutaveis e pre-registro versionado permite que as
conclusoes do TCC sejam auditadas e reconstruidas em sessoes futuras
exclusivamente a partir do repositorio em sua forma persistida. Essa
propriedade nao garante que as conclusoes sejam corretas, mas garante
que sejam falsificaveis no sentido operacional: qualquer revisor pode
re-executar os procedimentos sobre os mesmos artefatos e chegar aos
mesmos numeros, e qualquer divergencia e por si so um achado.

## Referencias nucleares do capitulo

- `FAMA1991`, `LO2004`
- `LIMZOHREN2021`, `LIMETAL2021TFT`
- `KOENKERBASSETT1978`, `GNEITING2007`, `CHERNOZHUKOV2010REARRANGEMENT`
- `JORION2007`, `ACERBI2002`
- `DIEBOLDMARIANO1995`, `HARVEY1997`, `HANSEN2011MCS`
- `GU2020`, `RAPACH2013`
- `PINEAU2021REPRODUCIBILITY`
