---
title: TCC Chapter Structure Freeze (ABNT)
scope: Macroestrutura congelada de capitulos do TCC ABNT (cap 1 pre-textuais, cap 2 introducao, cap 3 fundamentacao teorica, cap 4 metodo, cap 5 resultados, cap 6 conclusoes). Define limites de conteudo por capitulo e arquivos LaTeX correspondentes.
update_when:
  - estrutura de capitulos do TCC mudar (novo capitulo, divisao de capitulo)
  - secoes congeladas dentro de um capitulo forem revisadas
  - novo arquivo LaTeX for criado em text/
canonical_for: [chapter_structure_freeze, tcc_abnt_structure, latex_chapter_layout]
---

# Chapter Structure Freeze (TCC ABNT)

Status: estrutura congelada para reescrita do conteudo, mantendo o formato ABNT atual.

## Objetivo do arquivo
Congelar a macroestrutura de capitulos antes da escrita detalhada para evitar deriva de escopo e retrabalho.

Detalhamento do objetivo:
- Preservar o formato ABNT ja adotado no projeto.
- Definir fronteiras claras de conteudo por capitulo.
- Garantir alinhamento com o escopo tecnico real (TFT, quantis, reproducibilidade).
- Servir como referencia de validacao antes de cada rodada de edicao dos arquivos em `text/`.

## Capitulo 1 - Elementos pre-textuais
Arquivos:
- `text/1_Resumo/1_4_Resumo.tex`
- `text/1_Resumo/1_5_Abstract.tex`

Escopo de conteudo:
- problema, objetivo, metodo, resultados principais, contribuicoes.
- alinhado ao escopo TFT + avaliacao quantilica + reprodutibilidade.

## Capitulo 2 - Introducao
Arquivo:
- `text/2_Introducao/2_Introducao.tex`

Secoes congeladas:
1. Contextualizacao e motivacao
2. Problema de pesquisa
3. Justificativa
4. Objetivos
5. Delimitacao do estudo
6. Estrutura do trabalho

## Capitulo 3 - Fundamentacao teorica
Arquivo:
- `text/3_Revisao_Teorica/3_Revisao_Teorica.tex`

Secoes congeladas:
1. Previsao de series temporais financeiras
2. Modelagem com deep learning para series temporais
3. Temporal Fusion Transformer (TFT)
4. Predicao probabilistica e quantilica
5. Avaliacao de modelos em ambiente OOS
6. Reprodutibilidade experimental e rastreabilidade

## Capitulo 4 - Metodo
Arquivo:
- `text/4_Metodo/4_Metodo.tex`

Secoes congeladas:
1. Desenho da pesquisa
2. Dados e pipelines de processamento
3. Engenharia de features
4. Configuracao do modelo TFT
5. Protocolo experimental (splits, seeds, folds)
6. Baselines
7. Metricas de avaliacao (pontuais e probabilisticas)
8. Explicabilidade e contribuicao
9. Analytics Store e governanca de experimento
10. Criterios de qualidade e validacao

## Capitulo 5 - Resultados e discussao
Arquivo:
- `text/5_Resultados/5_Resultados.tex`

Secoes congeladas:
1. Cenarios experimentais avaliados
2. Resultados agregados por horizonte e split
3. Analise da incerteza (quantis, cobertura, largura de intervalo)
4. Comparacoes entre configuracoes e feature sets
5. Analise estatistica pareada (quando aplicavel)
6. Interpretabilidade e implicacoes
7. Ameacas a validade dos resultados

## Capitulo 6 - Conclusoes
Arquivo:
- `text/6_Conclusoes/6_Conclusoes.tex`

Secoes congeladas:
1. Sintese dos achados
2. Contribuicoes do trabalho
3. Limitacoes
4. Trabalhos futuros

## Capitulo 7 - Referencias
Arquivo:
- `text/7_Referencias/7_Referencias.bib`

Escopo de atualizacao:
- substituir base de template por referencias aderentes ao recorte atual.

## Capitulo 8 - Apendices
Arquivo:
- `text/8_Apendices/8_Apendices.tex`

Escopo sugerido:
- tabelas completas de experimento, configuracoes, e detalhes operacionais.

## Capitulo 9 - Anexos
Arquivo:
- `text/9_Anexos/9_Anexos.tex`

Escopo sugerido:
- materiais complementares externos quando necessario.

---

## Regra de escrita (congelada)
- O texto deve refletir pipeline real do projeto: dados -> features -> treino TFT -> inferencia -> analytics -> avaliacao.
- Evitar retorno ao escopo antigo de classificacao de sentimento com CNN como tema central.
- Manter consistencia entre: titulo, problema, objetivos, metodo, resultados e conclusoes.
