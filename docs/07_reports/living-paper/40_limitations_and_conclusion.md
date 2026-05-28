---
title: Living Paper - Limitations and Conclusion
scope: Base textual para o Capitulo 6 do TCC (Conclusoes) e Discussion/Limitations/Conclusion do artigo. Consolida limitacoes (modelos single-asset por ativo, periodos avaliados, nao-causal, N efetivo h+30 baixo), ameacas a validade e trabalhos futuros.
update_when:
  - nova limitacao for identificada na Fase B/C
  - sintese de achados for gerada
  - lista de trabalhos futuros for revisada
canonical_for: [living_paper_conclusion, tcc_chapter_6_source, limitations_source, future_work_source]
---

# 40 - Limitations and Conclusion

## Objetivo deste arquivo
Registrar limitacoes, ameacas a validade e conclusoes tecnicas sem exagero de claim.

Detalhamento do objetivo:
- Delimitar de forma transparente o que os resultados permitem afirmar e o que nao permitem.
- Organizar ameacas a validade (dados, modelagem, avaliacao, generalizacao).
- Consolidar contribuicoes efetivas sem inflar escopo alem do que foi testado.
- Fechar com propostas de trabalhos futuros tecnicamente coerentes com as limitacoes observadas.

Entrada e saida esperadas:
- Entrada: achados do arquivo de resultados e registro de evidencias.
- Saida: Conclusoes do TCC e Discussion/Limitations/Conclusion do artigo.

## Estrutura sugerida
- Limitacoes do estudo
- Ameacas a validade
- Conclusoes
- Trabalhos futuros

## Limitacoes ja estabelecidas pelo direcionamento estrategico

Estas limitacoes devem aparecer no Capitulo 6, independentemente do resultado
da Fase B/C:

- Modelos single-asset por ativo: cada modelo e treinado e avaliado para um
  ativo especifico.
- Plataforma multi-asset: a pipeline pode ser repetida para outros ativos, mas
  resultados so valem para ativos e periodos efetivamente avaliados.
- Sem claim de generalizacao para ativos, mercados ou periodos nao avaliados
  sob protocolo pre-registrado.
- Sem inferencia causal: VSN, permutation importance e ablation indicam
  contribuicao preditiva condicional ao modelo, nao causalidade.
- Retornos diarios de ativo liquido tem edge esperado baixo; ausencia de
  superioridade contra baseline e resultado empiricamente plausivel.
- h+30 e suplementar por padrao devido a amostra efetiva pequena e sobreposicao
  temporal.
- Round 0 e exploratoria; claims confirmatorios dependem da Fase B
  pre-registrada.
- Interpretabilidade local e ilustrativa; conclusoes de contribuicao devem vir
  de evidencia global/por horizonte e estabilidade entre seeds/folds.
- Recorte temporal dos dados confirmatórios: claims H1/H2a/H2b/H3 são restritos
  a runs gerados após o reset silver/gold de 2026-05-10 (ver
  `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md` §"Archive pre-Phase B").
  O histórico anterior, com 91,25% de degenerescência quantilica em runs
  `parent_sweep_id=NULL` (auditado em
  `docs/07_reports/phase-gates/phase-a/A_code_audit.md` §M5-Q5 e em
  `docs/07_reports/living-paper/evidence_log.md` 2026-05-01), foi arquivado
  como diagnóstico e não alimenta resultados do Capítulo 5.

## Limitacoes especificas da Phase B confirmatoria

Estas limitacoes derivam diretamente do escopo pre-registrado da Phase B
([`preregistration_phase_b.md`](../../06_pre_registration/phase-b/preregistration_phase_b.md)
§12; [`B_confirmatory_2026-05-25.md`](../phase-gates/phase-b/B_confirmatory_2026-05-25.md)
§7) e devem ser explicitas no Capitulo 6:

1. **Escopo da cohort.** Os claims H1/H2a/H2b sao validos **apenas** para a
   cohort `phase_b_confirmatorio_20260524`: ativo AAPL, horizontes h=1 e h=7,
   candidato TFT all-features sealed (config sha256
   `fc83b56d7605bf60479b4d1ed4c745c1679702e5e5116f642f3c33061d795bd4`),
   baselines pre-declarados (`zero_return`, `historical_mean_rolling` w=30,
   `historical_quantiles_rolling` w=252), 15 replicas TFT + 45 replicas
   baseline, protocolo Holm-6 (one-sided HAC+HLN).
2. **H2b refutada localmente.** No protocolo Phase B, o candidato TFT sealed
   nao supera `historical_quantiles_rolling` em pinball post-guardrail
   (p_adj_holm = 0,465 em h=1 e 0,685 em h=7; delta_rel ≈ −0,6% / −1,9%).
   Esse resultado e **da cohort**, nao do TFT em geral. Rodadas futuras com
   outro candidato, outro feature set ou outro protocolo podem reverter esse
   veredito, desde que com pre-registro proprio.
3. **H1 em h=1 em Tier 2.** PICP error 2,02% fica fora da banda Tier 1
   ±2,0% por 0,02 ponto percentual. Sustenta evidencia secundaria, nao
   primaria, para calibracao em h=1.
4. **N efetivo limitado em h=7.** 937 timestamps dedupados cross-fold sobre
   ~3 anos OOS; HLN aplicado. Poder estatistico limitado para detectar
   |Δpinball| < ~1%.
5. **h=30 out-of-scope** per §8 do pre-registro; nao reportado.
6. **Cross-family MCS / win-rate ausentes.** `split_fingerprint` distintos
   entre TFT e baselines bloqueiam pairing automatico em `gold_win_rate_*`
   e `gold_mcs_results`. DM family-6 substitui inferencialmente. Resolver
   requer Phase-B-bis.
7. **Walk-forward folds restritos.** 3 folds declarados; sensibilidade B.4
   (janelas alternativas) declarada como Phase-B-bis (rodada futura).
8. **TFT nao e dominante em retornos de ativos liquidos** (consistente com
   STRATEGIC_DIRECTION §3). H2b refutada confirma esse limite empirico
   declarado ex-ante; nao e achado negativo inesperado.

## O que a Phase B nao conclui

Para evitar leitura excessiva dos resultados confirmatorios da Phase B, o
Capitulo 6 deve explicitar o que a Phase B nao conclui:

- A Phase B **nao** prova o limite final do desempenho do TFT em forecasting
  financeiro. Avalia um candidato sealed unico, sob protocolo unico, em uma
  cohort unica. Outras configuracoes de TFT, outros feature sets, outros
  ativos, outros horizontes e outras epocas podem produzir resultados
  diferentes.
- A Phase B **nao** responde H3 (contribuicao de familias de features). H3
  e foco da Phase C explicativa, ainda pendente.
- A Phase B **nao** valida `gold_dm_pairwise_results`, `gold_mcs_results`,
  `gold_model_decision_final` ou `gold_prediction_risk` como evidencia
  confirmatoria. Esses artefatos legacy permanecem em escopo de hardening
  metodologico
  ([`C0_statistical_methods_hardening.md`](../phase-gates/phase-c/C0_statistical_methods_hardening.md)).
- A Phase B **nao** sustenta claims de generalizacao para outros tickers,
  mercados ou periodos nao avaliados sob o protocolo sealed.
- A Phase B **nao** sustenta claims de causalidade entre features e retorno.

Rodadas futuras que busquem evidencia confirmatoria adicional sobre TFT
(novas hipoteses, outros feature sets, outros ativos, outros horizontes ou
melhorias arquiteturais) devem ser declaradas como **novos experimentos**
com pre-registro proprio, cohort propria e escopo claramente separado da
Phase B. Mistura de runs futuros com a cohort Phase B viola o veto
declarado em [`preregistration_phase_b.md`](../../06_pre_registration/phase-b/preregistration_phase_b.md)
§14.

## Resultados validos mesmo com hipoteses refutadas

- Se H2a for refutada, o trabalho ainda pode concluir sobre limites do TFT para
  os ativos avaliados sob protocolo OOS auditavel.
- Se o modelo nao melhorar RMSE/DA, mas produzir intervalos calibrados, a
  contribuicao e probabilistica, nao pontual.
- Se a calibracao falhar, a contribuicao vira diagnostico de risco, cobertura e
  limites do pipeline atual.
- Se VSN, permutation e ablation divergirem, a conclusao correta e instabilidade
  de contribuicao preditiva.

## Contribuicoes metodologicas (independentes de Fase B/C)

Estas contribuicoes ja estao consolidadas pelo desenho experimental e pela
infraestrutura analitica, e devem ser reportadas no Capitulo 6 §6.2 mesmo
antes do fechamento da Fase B/C. A lista canonica corresponde a §6.2 do
`text/6_Conclusoes/6_Conclusoes.tex`:

1. **Politica operacional de variante quantilica (raw + post-guardrail)** com
   categorizacao A/B/C por tabela analitica e variante primaria declarada por
   hipotese. Justificativa formal: Chernozhukov, Fernandez-Val & Galichon
   (2010); Gneiting & Raftery (2007); Jorion (2007); Acerbi & Tasche (2002).
   Documentacao canonica: `docs/04_evaluation/METRICS_DEFINITIONS.md` §Variante
   quantilica.

2. **Governanca de coorte por `parent_sweep_id`** como pre-condicao de
   comparabilidade em testes pareados. Documentacao canonica:
   `docs/01_architecture/ANALYTICS_STORE_ARCHITECTURE.md`,
   `docs/03_modeling/SWEEPS_AND_SELECTION.md` e
   `docs/ai/AGENT_CORE.md` §Scope Governance.

3. **Separacao entre camadas imutaveis e camadas reconstruiveis** no analytics
   store (ADR-0001), permitindo reanalise sem retrain. Fronteira temporal
   confirmatoria: pos-reset 2026-05-10.

4. **Hierarquia explicita de evidencia para contribuicao de features**
   (VSN -> permutation -> ablation), com regra de consistencia em pelo menos
   dois dos tres metodos. Documentacao canonica:
   `docs/04_evaluation/EXPLAINABILITY.md`.

5. **Pre-registro versionado como bloqueante** para rodada confirmatoria.
   Documentacao canonica: `docs/08_governance/EXPERIMENT_TRACKING_POLICY.md`.

## Trabalhos futuros

- Ampliar a avaliacao para mais ativos reais, mantendo modelos especificos por
  ativo e criterios de inclusao pre-registrados.
- Classificacao ou deteccao de regimes como estudo separado, com rotulos ou
  metodologia propria.
- Avaliacao de causalidade/econometria apenas com desenho especifico para esse
  objetivo.
- Aplicacao em portfolio/trading apenas apos validacao fora do escopo atual.
