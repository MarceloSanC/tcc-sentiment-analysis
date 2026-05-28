# Auditoria metodologica do DATA_PIPELINE_WALKTHROUGH.md

Data: 2026-05-25

Escopo: auditoria metodologica baseada apenas no documento DATA_PIPELINE_WALKTHROUGH.md anexado e em pesquisa externa na literatura. Sem acesso ao codigo-fonte real, portanto esta auditoria avalia o uso descrito, nao certifica a implementacao do repositorio.

## 1. Resumo executivo

Verdict geral: o walkthrough descreve um pipeline de avaliacao relativamente maduro, com metricas e testes relevantes para forecasting financeiro. Entretanto, para dissertacao/artigo, varios elementos precisam de ressalva forte antes de sustentar claims confirmatorios: DM/MCS usam squared error mesmo quando o projeto parece probabilistico; ha top-50 por desempenho antes de testes pareados; Holm aparentemente nao inclui split_signature; MCS tem bootstrap e block length hard-coded; PICP assume nominal 0.80 fixo; prob_up_from_quantiles e confidence_calibrated sao heuristicas nao canonicas; VaR/ES sobre erro de previsao nao devem ser apresentados como risco financeiro.

A parte mais defensavel e manter pinball loss, PICP/MPIW e DM/MCS como instrumentos, mas rebaixar varias saidas para diagnostico exploratorio ate que: (i) a familia de testes seja pre-declarada, (ii) o top-50 seja removido ou feito em validacao, (iii) a perda primaria seja alinhada ao claim, e (iv) a degeneracao de quantis seja tratada como gate bloqueante.

## 2. Tabela de verdicts

| Item | Referencia canonica | Uso descrito no documento | Verdict | Impacto | Acao recomendada |
|---|---|---|---|---|---|
| Diebold-Mariano test (DM) | Diebold & Mariano 1995; HLN 1997; Newey-West 1987 | DM por pares em loss matrix L[t, config], loss=squared_error; alinhamento estrito por target_timestamp; HAC/Bartlett com lag=min(max(1,n^(1/3)),10); p-value normal two-sided; top-50 por MSE antes do pairwise. | Parcialmente correto / metodologicamente fragil | Alto | Manter apenas com ressalvas; adicionar HLN ou justificar assintotico; parametrizar lag; reportar sensibilidade; usar loss primaria adequada; remover top-50 por test ou fazer pre-filtro por validacao. |
| Holm step-down / Holm-Bonferroni | Holm 1979 | Ajuste por p-values ordenados; grupo descrito como (asset, parent_sweep_id, split, horizon); lacuna diz que split_signature nao entra. | Formula correta; familia possivelmente incorreta ou incompleta | Alto | Corrigir familia: incluir split_signature e todos os testes que sustentam o mesmo claim; ajustar tambem por multiplos horizontes/assets se o claim for agregado. |
| Model Confidence Set (MCS) | Hansen, Lunde & Nason 2011; Kunsch 1989 | Variante range t-stat com moving block bootstrap; alpha=0.05; B=300; block_len=5; seed=42; elimina pior mean_loss; loss=squared_error. | Parcialmente correto / fragil | Alto | Manter como variante reconhecida, mas aumentar B, justificar block length, relatar caminho de eliminacao, evitar top-50 por test, incluir split_signature nos outputs e interpretar como conjunto nao rejeitado, nao como vencedor. |
| PICP | Christoffersen 1998; Gneiting et al. 2007; Khosravi et al. 2011 | PICP=mean(y_true entre q10 e q90); coverage_nominal=0.80 hard-coded; mascara metricas quando point ou q10==q90. | Parcialmente correto | Alto | Correto so para intervalo central p10-p90. Tornar nominal dinamico, adicionar testes de cobertura condicional/independencia e reportar n_probabilistic_samples/degeneracao. |
| MPIW | Gneiting & Raftery 2007; Khosravi et al. 2011 | MPIW=mean(q90-q10); pred_interval_width agregado; guardrail compara raw vs post. | Correto como descritivo; insuficiente sozinho | Medio/alto | Manter com PICP e score proprio de intervalo; adicionar MPIW normalizado e interval/Winkler score; nunca tratar menor largura como melhor sem cobertura. |
| Pinball / quantile loss | Koenker & Bassett 1978; Gneiting 2011 | Formula max(q*diff,(q-1)*diff) para q=.1,.5,.9; mean_pinball por run/split/horizon. | Correto | Medio | Manter; usar como score primario para claims probabilisticos; reportar por quantil e media; nao confundir triplo degenerado com distribuicao probabilistica. |
| Win rate pairwise | Metrica descritiva; teste formal exigiria sign test/bootstrap/HAC | win_rate=mean(loss_left<loss_right) em timestamps alinhados; sem variancia/p-value. | Correto como descritivo; nao e teste | Medio | Manter apenas como auxiliar; se entrar em decisao final, adicionar IC/teste pareado com dependencia temporal e tratamento de empates. |
| prob_up_from_quantiles | Nao canonico com apenas q10/q50/q90; probabilidade requer CDF ou suposicao distribucional | Aproxima CDF(0) linearmente entre q10 e q90; hard bounds; fallback em zero-width retorna 1/0/.5. | Metodologicamente fragil | Alto | Renomear como heuristica; retornar NaN em width=0; calibrar contra evento y>0; preferir CDF completa/grade de quantis/modelo paramétrico. |
| confidence_calibrated | Calibracao e sharpness: Gneiting et al. 2007; scores proprios: Gneiting & Raftery 2007 | Produto ad hoc: calibration_term * width_term; width_term depende da escala do target. | Metodologicamente fragil / nao canonico | Alto | Nao usar para claim confirmatorio; renomear como score heuristico; substituir por interval score, WIS/CRPS ou analise separada de cobertura e largura. |
| VaR / Expected Shortfall aproximados | VaR/ES: Gneiting 2011; Fissler & Ziegel 2016 | gold_prediction_risk calcula VaR, ES e max_drawdown sobre error=y_pred-y_true. | Equivocado se chamado de risco financeiro; valido so como tail-error descritivo | Alto | Renomear para erro extremo; para risco financeiro, calcular sobre perdas/retornos e fazer backtesting de VaR/ES com nivel, sinal e excedencias. |
| Top-50 antes de DM/MCS | Inferencia seletiva; literatura de forecast comparison exige universo pre-definido | Seleciona top-50 por mean(squared_error) antes do pairwise; lacuna diz cap silencioso. | Metodologicamente fragil | Alto | Nao usar em claims confirmatorios se selecao usa test. Pre-filtrar por validacao ou declarar como analise exploratoria. |
| Quantis degenerados | Probabilistic forecasting exige distribuicao/intervalos informativos; calibracao e sharpness | Doc reconhece q10==q90, mascara metricas probabilisticas, mas prob_up tem fallback e baselines point emitem triplet degenerado. | Risco critico para claims probabilisticos | Alto | Transformar degeneracy_rate em gate decisivo; invalidar PICP/MPIW/prob_up quando q10==q50==q90; reportar antes de qualquer conclusao. |
| Criterio final de modelo | Forecast selection deve declarar score e familia de inferencia ex ante | Modelo final combina MCS, DM Holm vs baseline, generalization_gap, robustez, win-rate e calibracao. | Defensavel so se pre-registrado; fragil se post-hoc | Alto | Definir hierarquia: score primario, testes confirmatorios, metricas auxiliares. Evitar misturar objetivos sem regra ex ante. |

## 3. Analise detalhada por item

### 1. Diebold-Mariano test (DM)

**Definicao canonica ou variante reconhecida.** O DM testa se a media do diferencial de perdas d_t = L_1,t - L_2,t e zero. A funcao de perda pode ser quadratica, absoluta ou outra, desde que seja escolhida de acordo com o alvo inferencial. Para series temporais, a variancia de d_t precisa considerar autocorrelacao; HAC/Bartlett e uma escolha reconhecida. A correcao Harvey-Leybourne-Newbold e uma modificacao amplamente usada para melhorar propriedades em amostras pequenas, especialmente em comparacoes de previsoes multi-step.

**Referencias principais.** R1, R2, R3, R4, R5, R6.

**Hipoteses e condicoes de uso correto.**
- As previsoes comparadas devem estar pareadas no mesmo alvo, mesmo horizonte e mesma janela OOS.
- A perda usada no teste deve ser a perda primaria do claim. MSE/squared error e adequado para previsao pontual orientada a media, mas nao basta para claims probabilisticos baseados em quantis.
- A serie d_t deve ser tratada como dependente no tempo. Para h-step e janelas sobrepostas, a autocorrelacao residual tende a aumentar.
- O universo de comparacoes deve ser definido antes de olhar o desempenho em test, ou a inferencia vira seletiva.
- Em modelos aninhados ou com parametros estimados, testes padrao podem ter distribuicoes nao padrao; isso exige ressalva ou teste especifico.

**O que o walkthrough diz que o pipeline faz.** O walkthrough descreve DM por par em uma matriz L[t, config], com squared_error por target_timestamp, filtro de NaN, n minimo de 5, HAC com kernel Bartlett e lag min(max(1,n^(1/3)),10), estatistica normal two-sided, alinhamento estrito por target_timestamp e cap top-50 por mean(squared_error) antes do pairwise. O grupo de DM inclui asset, parent_sweep_id, split_signature, split e horizon.

**Avaliacao.** A formula central e reconhecivel como DM assintotico com HAC. O uso de squared_error e metodologicamente aceitavel se o claim for pontual/MSE. Entretanto, para um projeto que seleciona modelos quantilicos/TFT probabilistico, DM em squared_error nao sustenta, sozinho, claim sobre superioridade probabilistica. A ausencia de HLN nao e bug matematico, mas exige justificativa e sensibilidade. O lag hard-coded e uma limitacao metodologica/documental. O maior problema e o top-50 por desempenho em test antes de DM/MCS: isso cria inferencia pos-selecao e tende a tornar p-values otimistas. A agregacao media por target_timestamp tambem precisa ser validada: se mistura folds/seeds que geraram previsoes para o mesmo timestamp, e necessario documentar se isso e uma unidade estatistica planejada ou apenas uma reducao operacional.

**Verdict.** Parcialmente correto, mas metodologicamente fragil para claims confirmatorios.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Usar DM confirmatorio somente com perda primaria pre-declarada, universo pre-definido, mesma intersecao temporal, bandwidth parametrizavel e sensibilidade. Adicionar HLN ou explicar por que a amostra e suficientemente grande. Para claims probabilisticos, rodar DM tambem sobre mean_pinball/WIS/interval score. Remover top-50 por test; se for necessario reduzir, fazer com validacao ou criterio ex ante.

**Tipo de problema.** Limitacao metodologica + risco de inferencia seletiva. Nao verificavel pelo anexo: se a implementacao real respeita exatamente estes grupos e se os outputs nao contem duplicatas ou mistura indevida de folds/seeds.

### 2. Holm step-down / Holm-Bonferroni

**Definicao canonica ou variante reconhecida.** O procedimento de Holm ordena p-values e controla a family-wise error rate com mais poder que Bonferroni simples. O ponto critico nao e apenas a formula: e a definicao da familia de hipoteses que sera protegida.

**Referencias principais.** R7.

**Hipoteses e condicoes de uso correto.**
- A familia precisa conter todos os testes que podem ser selecionados, reportados ou usados para sustentar o mesmo claim.
- Se o claim e “o modelo vence em pelo menos um horizonte”, a familia deve incluir horizontes. Se o claim e por ativo, por ativo pode ser defensavel, mas isso deve estar escrito.
- P-values misturados de splits ou assinaturas de split diferentes nao devem ser ajustados como se fossem uma unica familia homogenea.

**O que o walkthrough diz que o pipeline faz.** A formula descrita para pvalue_adj_holm esta correta: p-values ordenados, multiplicacao por m-j+1, cumulative max e clip a 1. A aplicacao e por grupo (asset, parent_sweep_id, split, horizon). A lacuna A.66 informa que split_signature nao entra no agrupamento, embora DM tenha split_signature na grain.

**Avaliacao.** A formula e correta. O problema e a familia. Se split_signature pode mudar, omiti-la no ajuste pode misturar p-values de desenhos experimentais diferentes. Por outro lado, se o resultado final da tese/artigo seleciona entre multiplos horizontes, baselines, assets, sweeps ou split_signatures, ajustar separadamente por asset/sweep/split/horizon pode subcorrigir a multiplicidade do claim final. Isso nao e bug matematico na formula de Holm, mas e um risco inferencial alto.

**Verdict.** Formula correta; uso/familia parcialmente correto ou nao verificavel apenas pelo anexo.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Declarar explicitamente a familia primaria. Para cada claim confirmatorio, ajustar todos os testes que poderiam sustentar aquele claim. Incluir split_signature no groupby, ou provar que parent_sweep_id+split+horizon identifica uma unica split_signature. Se houver claim agregado entre horizontes, aplicar Holm tambem no nivel dos horizontes ou reformular o claim por horizonte.

**Tipo de problema.** Problema de definicao de familia de testes + problema de documentacao.

### 3. Model Confidence Set (MCS)

**Definicao canonica ou variante reconhecida.** O MCS de Hansen, Lunde e Nason constroi um conjunto de modelos superiores que nao foram eliminados por testes sequenciais de equal predictive ability. O resultado nao e “este modelo e o melhor”; e “este conjunto contem modelos que nao foram estatisticamente descartados, dado o universo inicial, a perda e o nivel”. Variantes reconhecidas usam estatisticas Tmax/TR e bootstrap para dependencias temporais.

**Referencias principais.** R8, R9.

**Hipoteses e condicoes de uso correto.**
- O universo inicial M0 precisa ser pre-definido ou selecionado em dados independentes do test.
- A matriz de perdas deve ter observacoes comparaveis e alinhadas para todos os modelos avaliados.
- O bootstrap em bloco precisa ter block length justificado ou analisado por sensibilidade.
- O numero de reamostragens precisa ser suficiente para p-values estaveis; B=300 e baixo para conclusoes fortes em alpha=0.05.
- A interpretacao de selected_in_mcs_alpha_0_05 deve ser conservadora: sobreviver ao MCS nao equivale a superar todos os outros.

**O que o walkthrough diz que o pipeline faz.** O walkthrough descreve uma variante range t-stat, moving block bootstrap com wrap-around, alpha=0.05, B=300, block_len=5, seed=42, eliminacao do pior mean_loss e saida selected_in_mcs_alpha_0_05. A matriz de perdas e a mesma do DM, com squared_error. A grain resumida de gold_mcs_results nao lista split_signature, embora o texto diga que a loss matrix segue a construcao do DM.

**Avaliacao.** A estrutura geral e reconhecivel como MCS. Os principais riscos sao: B=300 baixo; block_len=5 hard-coded; perda squared_error desalinhada de claims probabilisticos; possivel perda de split_signature no output; e, principalmente, pre-selecao top-50 se aplicada ao universo do MCS com base no test. A selecao top-50 por test altera M0 apos observar perdas e enfraquece a interpretacao de cobertura do MCS.

**Verdict.** Parcialmente correto; defensavel como exploratorio ou confirmatorio fraco com ressalvas fortes.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Aumentar B para pelo menos 1000 e preferencialmente 5000 em resultados finais; parametrizar e justificar block_len; reportar n_obs, n_models, p-values por iteracao e modelos eliminados; preservar split_signature no output; rodar MCS no universo pre-definido ou pre-filtrado por validacao; considerar loss probabilistica alem de MSE.

**Tipo de problema.** Variante aceitavel com limitacoes metodologicas e risco de selecao.

### 4. PICP

**Definicao canonica ou variante reconhecida.** PICP e a proporcao de observacoes que caem dentro do intervalo preditivo. Para um intervalo central [q_alpha/2, q_1-alpha/2], a cobertura nominal e 1-alpha. A literatura de interval forecasts enfatiza que cobertura incondicional nao basta: a sequencia de hits tambem deve ser avaliada quanto a independencia/condicionalidade.

**Referencias principais.** R12, R13, R14.

**Hipoteses e condicoes de uso correto.**
- O intervalo precisa ser bem definido e ordenado: lower <= upper.
- coverage_nominal deve ser q_high - q_low; para q10-q90, e 0.80; para q05-q95, seria 0.90.
- Deve haver numero efetivo de amostras probabilisticas suficiente e reportado.
- Intervalos degenerados devem invalidar ou acionar alerta, nao apenas reduzir largura.

**O que o walkthrough diz que o pipeline faz.** O walkthrough define covered_80=(y_true>=q10 AND y_true<=q90), PICP=mean(covered_80), coverage_nominal=0.80 e coverage_error=PICP-coverage_nominal. Tambem diz que metricas probabilisticas sao mascaradas para prediction_mode!=quantile ou quantis degenerados q10==q90. A lacuna A.68 reconhece que coverage_nominal=0.80 e hard-coded e errado se o sweep usar quantis diferentes.

**Avaliacao.** A definicao esta correta para intervalo p10-p90. O hard-code de 0.80 e correto apenas sob esse contrato. O documento mostra consciencia da degeneracao, mas a validade das conclusoes depende da taxa de degeneracao e de como ela e reportada. PICP sozinho nao prova calibracao condicional e pode esconder clustering de falhas.

**Verdict.** Parcialmente correto.

**Impacto nas conclusoes academicas.** Alto para qualquer claim probabilistico.

**Recomendacao concreta.** Tornar coverage_nominal dinamico a partir dos quantis efetivos; reportar PICP com intervalo de confianca e n_probabilistic_samples; adicionar teste de cobertura condicional/independencia de hits; bloquear claims probabilisticos se degeneracy_rate for material.

**Tipo de problema.** Limite metodologico + possivel bug de configuracao quando quantis mudam.

### 5. MPIW

**Definicao canonica ou variante reconhecida.** MPIW e a largura media do intervalo. Mede sharpness, mas nao calibracao. Um intervalo infinitamente largo maximiza cobertura e um intervalo zero minimiza largura; portanto MPIW deve ser interpretado junto com PICP ou, melhor, por um score proprio de intervalo.

**Referencias principais.** R11, R12, R14.

**Hipoteses e condicoes de uso correto.**
- O lower e upper precisam ser coerentes e comparaveis na mesma escala.
- Comparar largura so e justo quando a cobertura nominal e a calibracao sao semelhantes.
- Em series com magnitudes diferentes, largura normalizada ajuda a comparabilidade.

**O que o walkthrough diz que o pipeline faz.** O walkthrough calcula pred_interval_width=q90-q10 e MPIW=mean(pred_interval_width), tambem com comparacao raw vs post-guardrail.

**Avaliacao.** A formula esta correta como estatistica descritiva. Ela nao deve ser usada isoladamente para selecionar modelos. Em quantis degenerados, MPIW=0 nao e evidencia de precisao: e evidencia de ausencia de incerteza representada.

**Verdict.** Correto como descritivo; insuficiente para claims isolados.

**Impacto nas conclusoes academicas.** Medio/alto.

**Recomendacao concreta.** Manter, mas adicionar interval score/Winkler score ou WIS. Reportar MPIW condicionado a cobertura aceitavel, normalizado quando comparar assets/targets diferentes, e separar raw/post-guardrail.

**Tipo de problema.** Limitacao metodologica se usado como criterio de selecao.

### 6. Pinball loss / quantile loss

**Definicao canonica ou variante reconhecida.** A perda pinball, tambem chamada check loss, e consistente para o quantil q. Para q=0.5, equivale a uma constante vezes MAE/mediana. A media sobre quantis e uma avaliacao limitada da distribuicao preditiva nos quantis reportados.

**Referencias principais.** R10, R11, R15.

**Hipoteses e condicoes de uso correto.**
- O quantil previsto deve corresponder ao nivel q usado na perda.
- A perda deve ser calculada OOS e agregada sobre amostras pareadas.
- Para usar mean_pinball como objetivo primario, os quantis e pesos devem ser pre-declarados.

**O que o walkthrough diz que o pipeline faz.** O walkthrough define diff=y_true-y_pred_q e pinball=max(q*diff,(q-1)*diff), aplicada para q=.1,.5,.9, com colunas pinball_q10, pinball_q50, pinball_q90 e mean_pinball.

**Avaliacao.** A formula esta correta. A restricao e inferencial: se o ranking final usa DM/MCS em squared_error, o mean_pinball vira apenas descritivo, apesar de ser a perda mais alinhada aos quantis. Quantis degenerados ainda podem ter pinball finito, mas isso nao os torna probabilisticos.

**Verdict.** Correto.

**Impacto nas conclusoes academicas.** Medio.

**Recomendacao concreta.** Manter e considerar mean_pinball como perda primaria para a familia probabilistica. Reportar por quantil e media; em caso de degeneracao, separar “qualidade de ponto” de “qualidade probabilistica”.

**Tipo de problema.** Sem bug matematico pelo anexo; risco de desalinhamento entre metrica e claim final.

### 7. Win rate pairwise

**Definicao canonica ou variante reconhecida.** Win rate pareado e a proporcao de instantes em que um modelo tem perda menor que outro. Ele e uma estatistica descritiva. Para inferencia, seria necessario um teste pareado ou bootstrap que considere dependencia temporal e tratamento de empates.

**Referencias principais.** R1, R3, R9 como contexto de comparacao pareada e dependencia temporal.

**Hipoteses e condicoes de uso correto.**
- Mesmo conjunto de timestamps, mesmo horizonte e mesma loss.
- Empates devem ser especificados: ignorar, contar meio ponto ou reportar tie_rate.
- Nao chamar de significancia estatistica sem p-value/IC adequado.

**O que o walkthrough diz que o pipeline faz.** O walkthrough define win_rate=mean(loss_left<loss_right) ao longo de target_timestamps alinhados e diz explicitamente que e estatistica simples sem ajuste de variancia.

**Avaliacao.** Esta correto como descritivo auxiliar. O risco aparece em gold_model_decision_final, que consome win-rate junto com DM/MCS/gaps/calibracao. Se win-rate for criterio necessario ou desempate pre-declarado, ok. Se for usado para claim de superioridade sem teste, nao e defensavel.

**Verdict.** Correto como metrica auxiliar; nao e teste estatistico.

**Impacto nas conclusoes academicas.** Medio.

**Recomendacao concreta.** Manter como diagnostico; adicionar tie_rate e IC por block bootstrap se quiser usar em decisao. Nao usar para claim confirmatorio sozinho.

**Tipo de problema.** Limitacao metodologica se interpretado como teste.

### 8. Probabilidade de alta derivada de quantis (prob_up_from_quantiles)

**Definicao canonica ou variante reconhecida.** P(Y>0) e uma probabilidade da distribuicao preditiva. Com apenas tres quantis, ela nao e identificada de forma unica sem assumir uma forma para a CDF entre e fora dos quantis. Interpolacao linear entre q10 e q90 e uma heuristica, nao uma definicao canonica.

**Referencias principais.** R11, R12, R15.

**Hipoteses e condicoes de uso correto.**
- Declarar explicitamente a suposicao de CDF piecewise-linear.
- Tratar zero-width como indefinido/degenerado, nao como probabilidade calibrada.
- Validar calibracao do evento binario y_true>0 com reliability diagram, Brier/log score ou binning.

**O que o walkthrough diz que o pipeline faz.** O walkthrough aproxima CDF(0) por 0.1 + 0.8*((0-q10)/(q90-q10)), clipa para [0.1,0.9], aplica hard bounds se q10>0 ou q90<0, e usa fallback 1/0/0.5 quando width=0. A lacuna A.67 reconhece que esse fallback mascara degeneracao.

**Avaliacao.** Como heuristica operacional, pode ser util. Como probabilidade calibrada de alta, e fragil. O uso nao considera q50 como ancora explicita da CDF, e o fallback em intervalos degenerados produz valores aparentemente informativos quando a distribuicao preditiva colapsou.

**Verdict.** Metodologicamente fragil.

**Impacto nas conclusoes academicas.** Alto se houver claims direcionais/probabilisticos.

**Recomendacao concreta.** Renomear para prob_up_heuristic_from_q10_q90; retornar NaN quando q90<=q10; calibrar empiricamente contra y_true>0; preferir estimar a CDF com grade de quantis mais densa, distribuicao paramétrica ou amostras preditivas.

**Tipo de problema.** Variante heuristica aceitavel somente com ressalva; fallback zero-width e risco de bug conceitual.

### 9. confidence_calibrated

**Definicao canonica ou variante reconhecida.** Na literatura, calibracao e sharpness sao propriedades distintas. Scores proprios, como CRPS, log score, pinball/interval score ou WIS, combinam aspectos de forma teoricamente fundamentada. Um produto ad hoc entre cobertura e largura nao e, por si, uma metrica canonica de calibracao.

**Referencias principais.** R11, R12, R13.

**Hipoteses e condicoes de uso correto.**
- Se for usado, deve ser rotulado como indice heuristico e nao como calibracao estatistica.
- A escala da largura precisa ser normalizada; caso contrario, o score depende da unidade do target.
- A formula deve ser monotonicamente coerente com o objetivo pre-declarado.

**O que o walkthrough diz que o pipeline faz.** O walkthrough define confidence_calibrated = calibration_term * width_term, com calibration_term=clip(1-|coverage_error|/coverage_nominal,0,1) e width_term=1/(1+clip(pred_interval_width,0,+inf)).

**Avaliacao.** A formula penaliza erro de cobertura e largura, mas nao e score proprio, nao tem interpretacao probabilistica canonica e depende da escala do retorno. Chamar isso de “confidence_calibrated” pode sugerir uma calibracao que a formula nao prova.

**Verdict.** Metodologicamente fragil / nao canonico.

**Impacto nas conclusoes academicas.** Alto se entrar na selecao final.

**Recomendacao concreta.** Nao usar para claims confirmatorios. Renomear para heuristic_coverage_width_score. Para selecao, usar mean_pinball, interval score/WIS ou CRPS; para diagnostico, reportar cobertura e largura separadamente.

**Tipo de problema.** Problema conceitual e de nomenclatura.

### 10. VaR / Expected Shortfall aproximados

**Definicao canonica ou variante reconhecida.** Value-at-Risk e quantil de perdas/retornos de uma carteira ou variavel de perda. Expected Shortfall e a perda media alem do quantil. VaR e quantil elicitable; ES isolado nao e elicitable, mas o par (VaR, ES) e conjuntamente elicitable. Backtesting exige nivel alpha, sinal, exceedance process e alvo de perda bem definido.

**Referencias principais.** R15, R16, R17.

**Hipoteses e condicoes de uso correto.**
- Definir se a variavel e retorno, perda positiva, P&L ou erro de previsao.
- Definir nivel de cauda (ex.: 1%, 2.5%, 5%) e convencao de sinal.
- Distinguir risco financeiro do ativo de risco de erro do modelo.
- Para claims de risco, aplicar testes de cobertura/exceedance e, para ES, avaliacao conjunta com VaR.

**O que o walkthrough diz que o pipeline faz.** O walkthrough diz que gold_prediction_risk calcula VaR, ES e max_drawdown sobre error = y_pred - y_true, por asset, parent_sweep_id, config_signature, split e horizon.

**Avaliacao.** Se a tabela for chamada de risco financeiro do modelo/ativo, o uso e equivocado: erro de previsao nao e retorno nem perda da carteira. Pode ser util como diagnostico de cauda do erro, mas precisa ser nomeado como forecast_error_var/es ou tail_error. Faltam no anexo nivel alpha, sinal e formula exata de ES, entao a implementacao e nao verificavel apenas pelo anexo.

**Verdict.** Equivocado para claim de risco financeiro; parcialmente aceitavel como metrica descritiva de erro extremo.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Renomear colunas/tabela para tail_error_metrics. Para VaR/ES financeiro, calcular sobre distribuicao preditiva de retornos/perdas e backtestar excedencias. Documentar alpha, sinal, cauda, tratamento de empates e se ES e media condicional alem do VaR.

**Tipo de problema.** Problema conceitual; implementacao detalhada nao verificavel apenas pelo anexo.

### 11. Intersecao temporal exata para DM/MCS/win-rate

**Definicao canonica ou variante reconhecida.** Comparacoes pareadas exigem que cada perda dos modelos concorrentes seja calculada para o mesmo alvo realizado. Em forecasting, o par deve compartilhar target timestamp, horizon, forecast origin/decision timestamp e split OOS.

**Referencias principais.** R1, R4, R5.

**Hipoteses e condicoes de uso correto.**
- A intersecao por target_timestamp e necessaria, mas nao suficiente se modelos usam decision_timestamp ou horizons diferentes.
- Se varias seeds/folds geram previsoes para o mesmo target, a unidade estatistica deve ser definida antes: timestamp agregado ou fold-seed separado.
- A dependencia temporal remanescente precisa de HAC ou bootstrap.

**O que o walkthrough diz que o pipeline faz.** O walkthrough usa pivot wide por target_timestamp e dropna(how=any), e tambem tem gold_paired_oos_intersection_by_horizon com intersection_count, union_count e jaccard_index. Existem gates de alinhamento de baselines/TFT. No entanto, a inferencia silver omite decision_timestamp/horizon em uma tabela e ha lacunas de horizon=1 hard-coded em inferencia.

**Avaliacao.** A intersecao estrita e uma boa salvaguarda minima. Restam riscos: forecast origins podem diferir; folds/seeds podem ser agregados de modo nao declarado; e target timestamp sozinho nao prova que o mesmo horizonte/decisao foi usado se os campos relevantes faltarem em alguns outputs.

**Verdict.** Parcialmente correto; nao suficiente sozinho.

**Impacto nas conclusoes academicas.** Alto para DM/MCS/win-rate.

**Recomendacao concreta.** Validar diretamente no codigo e nas tabelas que todo par compartilha target_timestamp, decision_timestamp, horizon, split_signature e target value. Preservar decision_timestamp/horizon em todos os fatos usados em pairwise.

**Tipo de problema.** Boa pratica incompleta; parte nao verificavel apenas pelo anexo.

### 12. Top-50 antes de DM/MCS

**Definicao canonica ou variante reconhecida.** Testes de performance OOS assumem que o conjunto de comparacoes foi definido independentemente das perdas testadas. Selecionar candidatos no proprio test set antes de testar cria inferencia seletiva e tende a enviesar p-values e conjuntos superiores.

**Referencias principais.** R1, R7, R8.

**Hipoteses e condicoes de uso correto.**
- Qualquer reducao de universo deve ser pre-declarada ou baseada em validacao/treino, nao no test final.
- Se a reducao for computacional, o relatorio deve dizer quantos modelos foram excluidos e por qual criterio.
- Claims confirmatorios devem se limitar ao universo efetivamente testado e explicitar exclusoes.

**O que o walkthrough diz que o pipeline faz.** O walkthrough diz que o DM seleciona top-50 configs por mean(squared_error) antes do pairwise e a lacuna A.65 informa que o cap e silencioso e configs alem do top-50 nao aparecem em gold_dm. O MCS usa a mesma loss matrix do DM, entao o risco provavelmente se propaga, embora o anexo nao prove a chamada exata.

**Avaliacao.** Este e um dos riscos mais graves. Mesmo que a formula de DM/MCS esteja correta, aplicar o teste apenas aos top-50 do test set muda a pergunta estatistica: deixa de ser “entre os modelos candidatos, quais superam o baseline?” e vira “entre os modelos que ja pareceram bons no test, quais ainda parecem bons?”.

**Verdict.** Metodologicamente fragil para confirmacao.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Mover top-k para validacao, ou declarar DM/MCS top-50 como exploratorio. Para artigo/dissertacao, resultados confirmatorios devem usar todos os candidatos pre-declarados ou uma reducao ex ante.

**Tipo de problema.** Risco de viés inferencial pos-selecao.

### 13. Quantis degenerados

**Definicao canonica ou variante reconhecida.** Uma previsao probabilistica precisa representar incerteza. Se q10=q50=q90, o intervalo central tem largura zero e a previsao se comporta como ponto, nao como distribuicao calibrada.

**Referencias principais.** R11, R12, R14.

**Hipoteses e condicoes de uso correto.**
- Degeneracao deve ser medida por run/split/horizon e tratada como falha de contrato probabilistico quando recorrente.
- PICP, MPIW e prob_up devem ser invalidados ou marcados como nao aplicaveis para linhas degeneradas.
- Claims probabilisticos devem ser condicionados a uma taxa aceitavel de quantis genuinos.

**O que o walkthrough diz que o pipeline faz.** O walkthrough menciona block_quantile_degeneracy_gate, gold_quantile_degeneracy_report, mascara de metricas para q10==q90 e lacuna A.67 sobre prob_up fallback que mascara degeneracao. Tambem diz que baselines point emitem triplet quantilico degenerado, mas prediction_mode=point os exclui de metricas PICP-dependentes.

**Avaliacao.** O documento reconhece corretamente o problema. O risco restante e de interpretacao e enforcement: se outputs finais ainda exibem prob_up/confidence_calibrated ou metricas agregadas sem expor degeneracy_rate, o leitor pode inferir probabilidades onde ha previsoes pontuais. Degeneracao afeta PICP, MPIW, calibracao, prob_up e qualquer claim de incerteza.

**Verdict.** Risco critico; parcialmente mitigado conforme descrito, mas nao verificavel apenas pelo anexo.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Fazer degeneracy_rate virar criterio bloqueante para claims probabilisticos. Retornar NaN para prob_up quando q90<=q10. Reportar percentuais q10==q50==q90 por run/split/horizon e incluir no resumo executivo dos experimentos.

**Tipo de problema.** Risco metodologico e operacional; parte implementacional nao verificavel.

### 14. Uso combinado para selecao final de modelo

**Definicao canonica ou variante reconhecida.** Uma decisao final pode combinar metodos estatisticos e metricas auxiliares, mas a hierarquia precisa ser pre-especificada. Em artigos, e recomendavel declarar um score primario, testes confirmatorios associados, criterios de robustez e criterios descritivos separadamente.

**Referencias principais.** R5, R8, R11, R12, R15.

**Hipoteses e condicoes de uso correto.**
- Score primario e familia de testes precisam ser definidos antes de olhar o test.
- Metricas de naturezas diferentes nao devem ser combinadas sem regra de decisao explicita.
- Win-rate e confidence_calibrated nao devem ter peso confirmatorio sem formalizacao.
- Se ha multiplos objetivos (MSE, pinball, calibracao, gap), o claim deve dizer exatamente em qual objetivo o modelo vence.

**O que o walkthrough diz que o pipeline faz.** gold_model_decision_final consome metrics_by_config, robustness, generalization_gap, dm_results, mcs_results, win_rate e paired_intersection. A logica exige config em MCS, DM significativo vs baseline com Holm e generalization_gap dentro de threshold. O documento tambem menciona calibracao e robustness em tabelas gold.

**Avaliacao.** A combinacao pode ser defensavel como regra de engenharia ou triagem se pre-declarada. Para claim academico, e fragil se foi desenhada apos ver resultados ou se mistura DM/MCS em squared_error com metricas probabilisticas sem objetivo primario unico. MCS e DM nao sao independentes; ambos usam a mesma matriz de perdas. Win-rate nao e teste. confidence_calibrated e heuristic.

**Verdict.** Defensavel com pre-registro; fragil caso contrario.

**Impacto nas conclusoes academicas.** Alto.

**Recomendacao concreta.** Escrever uma secao metodologica com: objetivo primario, loss primaria, familia de testes, criterio de desempate e metricas auxiliares. Separar conclusoes: “melhor em MSE”, “melhor em quantile loss”, “calibracao aceitavel”, “robusto entre seeds”.

**Tipo de problema.** Risco de multiplicidade, objetivo movel e interpretacao excessiva.

## 4. Claims seguros, com ressalva e a evitar

### Claims que sao seguros
- O pipeline, conforme descrito, possui inventario explicito de tabelas gold e formulas para pinball, PICP, MPIW, DM, Holm, MCS e win-rate.
- A formula de pinball loss descrita e a formula canonica para quantile loss.
- PICP=mean(y in [q10,q90]) e MPIW=mean(q90-q10) sao definicoes usuais para o intervalo p10-p90.
- DM com HAC/Bartlett e p-value two-sided e uma variante reconhecivel do teste assintotico de equal predictive accuracy.
- Holm step-down, pela formula descrita, e um ajuste valido de p-values quando a familia de testes esta corretamente definida.
- Win-rate pode ser reportado como metrica descritiva auxiliar.

### Claims que exigem ressalva
- “O TFT supera o baseline estatisticamente” - so com ressalva sobre perda primaria, familia Holm, top-50, HLN/bandwidth, intersecao e unidade estatistica.
- “O modelo e probabilisticamente calibrado” - exige PICP dinamico, hits/condicionalidade, taxa baixa de degeneracao e calibracao por evento/probabilidade.
- “selected_in_mcs_alpha_0_05 indica modelo superior” - a interpretacao correta e sobreviver ao MCS, nao vencer todos os modelos.
- “Probabilidade de alta” - apenas como heuristica derivada de quantis, salvo calibracao externa e distribuicao/CDF mais completa.
- “A decisao final e estatisticamente robusta” - apenas se a regra de combinacao foi definida antes e se metricas auxiliares nao forem tratadas como testes.

### Claims que nao devem ser feitos
- “O repositório implementa corretamente os testes” - nao ha acesso ao codigo real nesta auditoria.
- “O DM/MCS prova superioridade entre todos os modelos testados” se houve top-50 por test antes da inferencia.
- “coverage_nominal=0.80 esta sempre correto” se o sweep puder usar quantis diferentes de p10/p90.
- “MPIW zero indica alta confianca” quando q10=q50=q90; isso indica degeneracao probabilistica.
- “VaR/ES medem risco financeiro do ativo/carteira” se foram calculados sobre erro de previsao y_pred-y_true.
- “confidence_calibrated e uma metrica canonica de calibracao” - e um score heuristico conforme descrito.

## 5. Checklist final para validar diretamente no codigo
1. Confirmar no codigo se DM, MCS e win-rate usam exatamente a mesma chave de pareamento: asset, parent_sweep_id, split_signature, split, horizon, target_timestamp e, idealmente, decision_timestamp.
2. Verificar se ha duplicatas por target_timestamp/config_label antes do pivot e como folds/seeds sao agregados.
3. Checar se o top-50 e aplicado tambem ao MCS ou apenas ao DM, e se o criterio usa test loss.
4. Validar se pvalue_adj_holm agrupa com split_signature. Se nao, corrigir.
5. Verificar se a familia Holm usada no relatorio final cobre multiplos horizontes, assets, baselines e sweeps quando o claim for agregado.
6. Testar DM com HLN e sem HLN; comparar bandwidths: h-1, n^(1/3), Newey-West plug-in ou sensitivity grid.
7. Rodar DM/MCS tambem com mean_pinball ou interval score quando o claim for probabilistico.
8. Aumentar bootstrap_samples do MCS e testar block_len em {5, 10, 20} ou por regra/estimativa de block length.
9. Comparar implementacao MCS local com pacote reconhecido em um fixture pequeno, incluindo casos com empates e variancia zero.
10. Validar que coverage_nominal e calculado como q_high-q_low a partir do contrato real de quantis.
11. Conferir se q10<=q50<=q90 antes e depois do guardrail e reportar quantile_crossing_rate.
12. Calcular degeneracy_rate q10==q50==q90 e q10==q90 por run/split/horizon; definir threshold bloqueante.
13. Garantir que prob_up_from_quantiles retorna NaN quando q90<=q10, ou que o output seja explicitamente marcado como heuristic.
14. Validar calibracao de prob_up contra o evento y_true>0 com reliability diagram, Brier score e binning temporal.
15. Renomear confidence_calibrated ou substituir por score proprio; testar sensibilidade a escala do target.
16. Para VaR/ES, checar formula exata, alpha, sinal e variavel alvo; renomear para tail_error se calculado sobre error.
17. Verificar se baselines point com triplet degenerado nunca entram em PICP/MPIW/prob_up como probabilisticos.
18. Checar se horizon=1 hard-coded em inference invalida claims multi-horizon em gold.
19. Garantir que os outputs gold tenham schema, dtypes, PKs e metadados suficientes para reproduzir a inferencia.
20. Adicionar ao texto academico uma tabela com: universo de modelos, loss primaria, familia de testes, correcao de multiplicidade, n efetivo e regras de exclusao.

## 6. Referencias bibliograficas
- [R1] Diebold, F. X.; Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3), 253-263. DOI: 10.1080/07350015.1995.10524599. Link: https://doi.org/10.1080/07350015.1995.10524599
- [R2] Harvey, D.; Leybourne, S.; Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281-291. DOI: 10.1016/S0169-2070(96)00719-4. Link: https://doi.org/10.1016/S0169-2070(96)00719-4
- [R3] Newey, W. K.; West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708. DOI: 10.2307/1913610. Link: https://doi.org/10.2307/1913610
- [R4] West, K. D. (1996). Asymptotic inference about predictive ability. Econometrica, 64(5), 1067-1084. Link: https://www.jstor.org/stable/2171956
- [R5] Giacomini, R.; White, H. (2006). Tests of Conditional Predictive Ability. Econometrica, 74(6), 1545-1578. DOI: 10.1111/j.1468-0262.2006.00718.x. Link: https://doi.org/10.1111/j.1468-0262.2006.00718.x
- [R6] Clark, T. E.; McCracken, M. W. (2001). Tests of equal forecast accuracy and encompassing for nested models. Journal of Econometrics, 105(1), 85-110. Link: https://doi.org/10.1016/S0304-4076(01)00071-9
- [R7] Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2), 65-70. Link: https://www.jstor.org/stable/4615733
- [R8] Hansen, P. R.; Lunde, A.; Nason, J. M. (2011). The Model Confidence Set. Econometrica, 79(2), 453-497. DOI: 10.3982/ECTA5771. Link: https://doi.org/10.3982/ECTA5771
- [R9] Kunsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Annals of Statistics, 17(3), 1217-1241. DOI: 10.1214/aos/1176347265. Link: https://doi.org/10.1214/aos/1176347265
- [R10] Koenker, R.; Bassett, G. (1978). Regression Quantiles. Econometrica, 46(1), 33-50. DOI: 10.2307/1913643. Link: https://doi.org/10.2307/1913643
- [R11] Gneiting, T.; Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association, 102(477), 359-378. DOI: 10.1198/016214506000001437. Link: https://doi.org/10.1198/016214506000001437
- [R12] Gneiting, T.; Balabdaoui, F.; Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. Journal of the Royal Statistical Society: Series B, 69(2), 243-268. DOI: 10.1111/j.1467-9868.2007.00587.x. Link: https://doi.org/10.1111/j.1467-9868.2007.00587.x
- [R13] Christoffersen, P. F. (1998). Evaluating Interval Forecasts. International Economic Review, 39(4), 841-862. DOI: 10.2307/2527341. Link: https://doi.org/10.2307/2527341
- [R14] Khosravi, A.; Nahavandi, S.; Creighton, D.; Atiya, A. F. (2011). Comprehensive Review of Neural Network-Based Prediction Intervals and New Advances. IEEE Transactions on Neural Networks, 22(9), 1341-1356. DOI: 10.1109/TNN.2011.2162110. Link: https://doi.org/10.1109/TNN.2011.2162110
- [R15] Gneiting, T. (2011). Making and Evaluating Point Forecasts. Journal of the American Statistical Association, 106(494), 746-762. DOI: 10.1198/jasa.2011.r10138. Link: https://doi.org/10.1198/jasa.2011.r10138
- [R16] Fissler, T.; Ziegel, J. F. (2016). Higher order elicitability and Osband's principle. Annals of Statistics, 44(4), 1680-1707. DOI: 10.1214/16-AOS1439. Link: https://doi.org/10.1214/16-AOS1439
- [R17] Fissler, T.; Ziegel, J. F. (2016). Expected Shortfall is jointly elicitable with Value at Risk - Implications for backtesting. Risk, Jan. 2016, 58-61. Link: https://arxiv.org/abs/1507.00244

## 7. Observacao final

Esta auditoria nao substitui uma revisao do codigo nem uma reproducao dos resultados. Ela identifica o que e metodologicamente defensavel conforme descrito, o que e apenas heuristico e o que precisa ser verificado no codigo antes de entrar em uma dissertacao ou artigo.