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
  `docs/07_reports/phase-gates/A_code_audit.md` §M5-Q5 e em
  `docs/07_reports/living-paper/evidence_log.md` 2026-05-01), foi arquivado
  como diagnóstico e não alimenta resultados do Capítulo 5.

## Resultados validos mesmo com hipoteses refutadas

- Se H2a for refutada, o trabalho ainda pode concluir sobre limites do TFT para
  os ativos avaliados sob protocolo OOS auditavel.
- Se o modelo nao melhorar RMSE/DA, mas produzir intervalos calibrados, a
  contribuicao e probabilistica, nao pontual.
- Se a calibracao falhar, a contribuicao vira diagnostico de risco, cobertura e
  limites do pipeline atual.
- Se VSN, permutation e ablation divergirem, a conclusao correta e instabilidade
  de contribuicao preditiva.

## Trabalhos futuros

- Ampliar a avaliacao para mais ativos reais, mantendo modelos especificos por
  ativo e criterios de inclusao pre-registrados.
- Classificacao ou deteccao de regimes como estudo separado, com rotulos ou
  metodologia propria.
- Avaliacao de causalidade/econometria apenas com desenho especifico para esse
  objetivo.
- Aplicacao em portfolio/trading apenas apos validacao fora do escopo atual.
