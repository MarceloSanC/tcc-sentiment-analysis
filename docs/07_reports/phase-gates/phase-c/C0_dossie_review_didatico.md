---
title: "C.0 — Revisão didática dos dossiês (§6) + cross-check contra código"
scope: |
  Documento companion do C0_statistical_methods_hardening.md. Para cada um dos
  13 itens da §6 (Dossiês por item), registra: (a) explicação didática do
  teste/métrica; (b) decomposição elemento a elemento com o que o doc afirma,
  como deveria funcionar (com exemplo), o que esperar no código e a referência
  doc->código; (c) para cada ponto de atenção (⚠️), um aprofundamento didático
  embutido no próprio elemento (o que é, por que importa, quando cada escolha
  se aplica, o que implica para o claim do projeto); (d) cross-check do que NÃO
  está corretamente indicado/referenciado contra o código real; (e) um veredito.
  Este material é base de estudo para revisão manual da C.0.1 e insumo de
  redação do TCC. Será FUNDIDO depois com a pesquisa acadêmica online (GPT web)
  por item para então tomar decisão e preencher o C0 canônico. Este arquivo NÃO
  é canônico e NÃO toma decisão metodológica final.
status: in_progress
created_at: 2026-05-29
relates_to:
  - "C0_statistical_methods_hardening.md (§6 Dossiês por item) — fonte das refs verificadas"
  - "external-reviews/ (pesquisa acadêmica GPT web, a fundir por item)"
update_when:
  - cada item da §6 receber sua revisão didática (progresso 1/13 ... 13/13)
  - uma referência doc->código for confirmada/corrigida contra o src real
  - um ponto de atenção receber aprofundamento didático
---

# C.0 — Revisão didática dos dossiês (§6) + cross-check contra código

> **Para que serve.** Este documento traduz cada dossiê técnico da §6 do
> [`C0_statistical_methods_hardening.md`](C0_statistical_methods_hardening.md)
> em uma explicação didática, verifica elemento a elemento contra o código real
> em `src/`, e aprofunda cada ponto de atenção com o "quando e por que" que
> embasa as decisões futuras de C.0.2/C.0.3. Não substitui o dossiê nem decide
> nada: é estudo e cross-check, e será fundido com a pesquisa acadêmica online
> antes de qualquer preenchimento canônico.

> **Como ler cada item.**
> - **O que é (didático):** a ideia central em linguagem simples.
> - **Elementos:** cada componente do teste/métrica, com 4 campos fixos
>   (o que o doc afirma / como deveria funcionar com exemplo / o que esperar no
>   código / ref doc->código) e, **quando houver**, um bloco ⚠️ com
>   aprofundamento de extensão variável.
> - **Cross-check:** discrepâncias entre o que os campos "Uso atual no projeto"
>   e "Implementação atual localizada" afirmam e o que o código real mostra.
> - **Veredito:** 🟢 íntegro / 🟡 ressalvas / 🔴 referência quebrada.

> **Convenção de cross-check.** ✅ = ref do doc confirmada por leitura direta do
> código; ⚠️ = ponto metodológico (não é erro de referência); 🔴 = ref
> quebrada/divergente. Toda confirmação cita a linha real observada.

## Progresso

- [x] #1 — Diebold-Mariano gold
- [x] #2 — MCS gold
- [x] #3 — Holm gold
- [x] #4 — top-50 filter
- [ ] #5 — PICP
- [ ] #6 — MPIW
- [ ] #7 — Pinball loss
- [ ] #8 — Win-rate gold
- [ ] #9 — prob_up
- [ ] #10 — confidence_calibrated
- [ ] #11 — VaR / ES gold
- [ ] #12 — gold_model_decision_final
- [ ] #13 — Phase B DM family-6 (referencia)

---

## #1 — Diebold-Mariano gold (DM)

### O que é (didatico)

O teste de **Diebold-Mariano** compara a **acurácia preditiva de dois modelos**. A ideia:

1. Para cada instante *t*, mede o "erro" de cada modelo (uma *loss*).
2. Forma a **série da diferença** `d_t = loss_A(t) − loss_B(t)`.
3. Pergunta: a **média** de `d_t` é estatisticamente diferente de zero?
   - `média ≈ 0` → os dois modelos são igualmente bons (não da para distinguir).
   - `média < 0` → modelo A erra menos (A vence).

O truque fino: `d_t` é **autocorrelacionado** no tempo (erro de ontem se parece
com o de hoje), então não dá para usar a variância "ingênua". Usa-se uma
**variância HAC** (robusta a heterocedasticidade e autocorrelacao) para nao
subestimar o desvio e gerar p-value otimista demais.

### Elementos

**1. Loss = `squared_error`**
- **O que o doc afirma:** a perda é o erro quadrático `(y_pred − y_true)²`, calculada em `_pairwise_preprocess`.
- **Como deveria funcionar (exemplo):** se o modelo prevê 102 e o real é 100, a loss é `(102−100)² = 4`. Quanto maior, pior. O DM compara essas losses entre dois modelos timestamp a timestamp.
- **O que esperar no código:** uma coluna `squared_error` derivada de `y_pred` e `y_true`, **antes** do pivot da matriz de losses.
- **Ref do doc → código:** [`pairwise.py:280`](../../../../src/domain/services/gold_builders/pairwise.py#L280). ✅ **Confere:** `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2`.
- ⚠️ **Ponto de atenção — a loss casa com o claim?**
  O DM **não quebra** com modelo probabilístico: ele é **agnóstico a loss** — testa a média da diferença de *qualquer* loss por período. O que precisa casar é **a loss com o claim**:

  | Loss alimentada no DM | O que o teste passa a medir | Claim que sustenta |
  |---|---|---|
  | `squared_error` (atual) | acurácia **pontual** (do ponto previsto) | "modelo prevê melhor o valor central" |
  | `squared_error` só do **q50** | acurácia pontual **da mediana** | claim pontual mais honesto p/ modelo de quantis |
  | **pinball loss** (q10/q50/q90) | qualidade **da distribuição preditiva** | "modelo é melhor probabilisticamente" ← claim do TCC |
  | CRPS / WIS | idem, agregando quantis de forma mais principled | idem, ainda mais defensavel |

  Para o claim probabilístico do projeto, a loss certa é **pinball** (foi o que a
  Phase B fez no `phase_b_dm_family_6`). `squared_error` não está "errado" — ele
  responde a uma **pergunta diferente** (pontual). Rodar squared_error e depois
  afirmar "modelo probabilístico melhor" é o descasamento. O claim probabilistico
  ja esta coberto pela Phase B; o gold legacy DM, como esta, so embasaria claim
  **pontual**.

**2. Matriz de losses por timestamp (unidade estatística)**
- **O que o doc afirma:** monta uma "loss matrix wide por `target_timestamp_utc`" e usa `dropna(how="any")`.
- **Como deveria funcionar (exemplo):** uma tabela com linhas = timestamps, colunas = configs de modelo, células = loss média naquele timestamp. Para comparar dois modelos de forma justa, só valem timestamps em que **ambos** previram → daí o `dropna(how="any")` (descarta qualquer linha com buraco).
- **O que esperar no codigo:** um `pivot(index=target_timestamp_utc, columns=config_label, values=squared_error)` seguido de `dropna(axis=0, how="any")`.
- **Ref do doc → codigo:** [`pairwise.py:305-308`](../../../../src/domain/services/gold_builders/pairwise.py#L305). ✅ **Confere:** pivot + `dropna(axis=0, how="any")`.

**3. n mínimo = 5**
- **O que o doc afirma:** pares com menos de 5 timestamps em comum são pulados.
- **Como deveria funcionar (exemplo):** com 3 pontos não dá para estimar variância de forma confiável; o teste é abortado para aquele par.
- **O que esperar no codigo:** um guard `if n < 5: continue`.
- **Ref do doc → codigo:** [`pairwise.py:78`](../../../../src/domain/services/gold_builders/pairwise.py#L78). ✅ **Confere** exatamente.

**4. Variância HAC (kernel de Bartlett) + política de lag**
- **O que o doc afirma:** HAC Bartlett com lag `int(min(max(1, n^(1/3)), 10))` e peso `1 − k/(lag+1)`.
- **Como deveria funcionar (exemplo):** com n=1000 timestamps, `n^(1/3)≈10`, então usa lag 10 (teto). Soma a variância "crua" (`gamma0`) mais as autocovarianças até o lag 10, cada uma com peso decrescente (lag 1 pesa mais que lag 10). Isso "infla" a variância para refletir a autocorrelação e evita p-value falsamente pequeno.
- **O que esperar no código:** `gamma0` + loop `for k in range(1, lag+1)` somando `2 * weight * cov`.
- **Ref do doc → código:** [`pairwise.py:82`](../../../../src/domain/services/gold_builders/pairwise.py#L82) (lag) e [`:87`](../../../../src/domain/services/gold_builders/pairwise.py#L87) (peso). ✅ **Confere** exatamente.
- ⚠️ **Ponto de atenção — deveria seguir Newey-West clássico?**
  Há uma razão teórica específica para previsão. **Erros de previsão *h*-passos-à-frente** (sob otimalidade) seguem um processo **MA(h−1)** — autocorrelacionados até o lag *h−1* e não além. Por isso o padrao de livro-texto para DM e **lag = h − 1**.
  - **Phase B usou `max(h−1, 1)`** → exatamente essa regra (h=7 → lag 6; h=1 → lag 1). É a escolha canônica para DM de *h*-passos.
  - **Gold legacy usa `min(max(1, n^(1/3)), 10)`** → bandwidth automatica generica (cresce com o tamanho da amostra, comum na literatura HAC), mas **ignora o horizonte h** e tem teto arbitrario 10.

  As duas são HAC válidas, mas a regra `h−1` "sabe" da estrutura MA(h−1) dos
  erros de previsão; a `n^(1/3)` não. Para DM especificamente, `h−1` é a mais
  defensável e é a que a Phase B adotou. Risco do gold legacy: (a) inconsistência
  com a Phase B e (b) para h=7 pode usar lag até 10 onde o correto seria 6 →
  variância e p-value ligeiramente diferentes. Não é bug; é política de lag menos
  alinhada a teoria de previsao.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: adotar
  > **lag = `max(h−1, 1)`** (Newey-West com bandwidth da estrutura MA(h−1) dos erros
  > *h*-passos), substituindo a regra `min(max(1, n^(1/3)), 10)`. Alinha o DM gold à
  > Phase B e à teoria de previsão; elimina o teto arbitrário 10.

**5. Estatística DM e p-value**
- **O que o doc afirma:** `stat = mean_d / sqrt(var_mean)`; `pvalue = 2·(1 − Φ(|stat|))`, **two-sided**, normal.
- **Como deveria funcionar (exemplo):** se a diferença média for −0,5 e o erro-padrão 0,1, então `stat = −5` → p-value minúsculo → diferença significativa. Two-sided = testa "diferente de zero" nos dois sentidos.
- **O que esperar no código:** `stat = mean_d / math.sqrt(var_mean)` e `2.0*(1.0 - _norm_cdf(abs(stat)))`.
- **Ref do doc → código:** [`pairwise.py:92-93`](../../../../src/domain/services/gold_builders/pairwise.py#L92). ✅ **Confere** exatamente. Confirmei lendo a função inteira: **sem HLN** e **sem one-sided**.
- ⚠️ **Ponto de atenção — quando incluir HLN e quando usar one-sided?**

  **HLN (Harvey-Leybourne-Newbold 1997)** = correção de amostra pequena do DM.
  - **Problema que resolve:** o DM puro é N(0,1) só **assintoticamente**. Em amostra pequena ele **rejeita demais** → p-values otimistas → "significância" falsa.
  - **O que faz:** (1) multiplica a estatistica por `sqrt[(n + 1 − 2h + h(h−1)/n) / n]` (encolhe a estatistica) e (2) compara contra **t de Student com n−1 g.l.** em vez da normal.
  - **Quando incluir:** sempre que n for pequeno/moderado (regra de bolso: n < ~100–200). Em n grande o fator → 1 e t → normal, então o efeito some — mas como **não custa nada**, a prática moderna é **sempre aplicar**.
  - **No projeto:** Phase B aplicou HLN; gold legacy **não** → p-values do gold legacy são **anti-conservadores**, efeito maior em coortes pequenas.

  **One-sided vs two-sided** = direção da hipótese.
  - **Two-sided** (atual no gold): H₀ "acurácia igual" vs Hₐ "**diferente** (qualquer direção)". Use quando não há direção a priori.
  - **One-sided:** H₀ "A não é melhor que B" vs Hₐ "**A é melhor** que B". Use quando o claim é **direcional** *e pré-registrado* antes de ver os dados.
  - **Por que importa:** o one-sided tem **mais poder** para detectar o efeito na direção esperada, mas exige comprometer-se com a direção antes — senão vira p-hacking.
  - **Exemplo:** se `stat = 1,9`, two-sided (corte ±1,96) **não** rejeita; one-sided (corte 1,645) **rejeita**. Mesma evidência, conclusão diferente.
  - **No projeto:** o claim H2a/H2b é direcional ("TFT < baseline em pinball"), então Phase B usou **one-sided** corretamente. O gold legacy two-sided é "seguro" mas **perde poder** para um claim direcional.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: adotar
  > **teste one-sided** na direção pré-registrada do claim (TFT melhor que baseline),
  > alinhando o DM gold à Phase B e ganhando poder estatístico. Exige que a direção
  > seja fixada antes de ver os dados (já é o caso em H2a/H2b). **Decisão paralela
  > ainda pendente** no mesmo elemento: aplicar a correção **HLN** de amostra pequena
  > (Phase B já aplica; barata e sempre recomendada) — confirmar junto.

**6. top-50 aplicado *antes* do teste**
- **O que o doc afirma:** `_select_top_configs_for_pairwise(g, max_configs=50)` roda antes de montar a matriz.
- **Como deveria funcionar (exemplo):** de 200 configs, mantém só as 50 de menor `squared_error` médio **no próprio split de teste** — e só essas entram no DM. (É o problema de inferência seletiva do item #4.)
- **O que esperar no codigo:** chamada ao filtro logo no inicio do loop de grupos do builder.
- **Ref do doc → codigo:** [`pairwise.py:299`](../../../../src/domain/services/gold_builders/pairwise.py#L299). ✅ **Confere**.

**7. Holm persistido vs. p-value cru no decision_final**
- **O que o doc afirma:** o builder aplica Holm na saída (`gold_dm_pairwise_results` tem `pvalue_adj_holm`), **porém** `_build_model_decision_final` usa `pvalue_two_sided` (cru) para `dm_net_wins`.
- **Como deveria funcionar (exemplo):** existem **duas noções de "DM significativo"**: (a) a do parquet, ajustada por Holm; (b) a do decision_final, usando p<0,05 **cru**. Ao revisar, não confunda — a coluna `dm_net_wins` do artefato final **ignora a correção de multiplicidade**.
- **O que esperar no codigo:** Holm em `pairwise.py:333`; em `confidence.py` o agregado filtra por `pvalue_two_sided`.
- **Ref do doc → código:** [`pairwise.py:333`](../../../../src/domain/services/gold_builders/pairwise.py#L333) + [`confidence.py:605`](../../../../src/domain/services/gold_builders/confidence.py#L605). ✅ **Confere:** L605 lê `pvalue_two_sided`, L609 filtra `p >= 0.05` (cru); groupby em L599-601; colunas em L626-628.
- ⚠️ **Ponto de atenção — deveria usar Holm nos dois? O que implica?**
  O Holm controla o **FWER** (probabilidade de ≥1 falso positivo numa **família** de testes). Rodando DM em todos os pares de configs, a chance de "significância por sorte" explode; Holm contém isso.

  **Princípio:** qualquer número que vá **embasar ou parecer evidência** deve usar
  p-value corrigido, com a família bem definida. Se `dm_net_wins` for lido como
  sinal de superioridade, **deveria** usar Holm — caso contrário **superconta**
  vitórias que são ruído de múltiplas comparações (e ainda sobre o universo
  enviesado pelo top-50).

  **Mas há um "depende":** se `dm_net_wins` é **puramente diagnóstico/descritivo**
  (como a reinterpretação do C.0.1 agora afirma — "coluna diagnóstica"), então
  contagem com p cru é tolerável **desde que rotulada como não-inferencial**. O
  perigo só existe se alguém tratar como evidência.

  **Implicações da inconsistência atual:** (1) o sistema reporta **dois
  veredictos** de "DM significativo" (parquet ajustado vs decision_final cru);
  (2) se o decision_final for promovido a confirmatorio, a contagem com p cru
  **inflaria** a aparente superioridade; (3) a correcao Holm persistida tambem e
  suspeita (familia sem `split_signature` — item #3). **Nao e necessariamente
  "aplicar Holm em todo lugar"**: é **ser consistente e honesto** sobre o que cada
  numero e. Duas saidas defensaveis em C.0.3: (a) definir **uma familia correta** e
  aplicar Holm em tudo que alimenta claim; ou (b) marcar `dm_net_wins` como
  **diagnostico nao-inferencial** para que o p cru nao seja confundido com evidencia.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: adotar a
  > saída (a) — **definir uma família correta** (com `split_signature` no groupby — ver
  > item #3) e **aplicar Holm em tudo que alimenta claim**, inclusive em `dm_net_wins`
  > no `_build_model_decision_final` (hoje usa `pvalue_two_sided` cru). Elimina os dois
  > veredictos divergentes e impede a supercontagem de vitórias por ruído de múltiplas
  > comparações.

### Cross-check — o que NÃO está corretamente indicado/referenciado

Para o DM, **todas as referências de "Uso atual" e "Implementação atual localizada" conferem** com o código (verifiquei 280, 299, 333 em `pairwise.py` e 599-628 em `confidence.py`). Pontos a registrar:

1. **Nenhuma referência quebrada.** As linhas-âncora citadas estão corretas.
2. **Atribuição de função correta, mas atenção ao ler:** o campo declara "Função: `_compute_dm_pairwise_from_loss_matrix` (67-104)", mas várias evidências apontam para **outras funções/arquivos** (`_pairwise_preprocess:280`, `build():299/333`, `confidence.py:605`). Está **certo e explicitado** nos bullets — só não confunda "a função localizada" com "onde cada elemento vive".
3. **Sutileza Holm cru vs. ajustado (elemento 7)** é a coisa mais importante a carregar para C.0.3 — não é erro de referência, mas é o tipo de coisa que um leitor desatento do artefato final interpretaria errado. O dossiê captou bem.

### Veredito do item #1

🟢 **Íntegro.** Referências intactas; evidência fiel ao código. Pontos
metodologicos (loss, lag, HLN/one-sided, Holm) sao decisoes para C.0.2/C.0.3,
nao defeitos de localizacao. Decisões recomendadas registradas nos elementos 1, 4,
5 e 7 (pinball, lag `h−1`, one-sided + HLN, Holm sobre família correta) — pendentes
de confirmação com a pesquisa acadêmica do paper.

---

## #2 — MCS gold

### O que é (didatico)

O **Model Confidence Set (MCS)** e um procedimento iterativo de eliminacao de modelos. Dado
um conjunto de M modelos e uma função de loss, o MCS identifica o menor subconjunto de
modelos que *não pode ser rejeitado estatisticamente como contendo o melhor modelo*, ao
nível de significância alpha. O algoritmo repete: testa se todos os modelos ativos são
"igualmente bons" via estatística de range (TR = max dos t-stats pairwise); se a hipótese
nula for rejeitada (p < alpha), elimina o modelo com maior perda média e continua. Para
quando o conjunto remanescente não pode ser rejeitado → esse é o *confidence set*.

O truque central: ao contrário do DM (que compara dois modelos por vez), o MCS responde
a pergunta **coletiva** "qual é o conjunto mínimo de modelos estatisticamente superiores?"
Pertencer ao MCS significa "não foi possível rejeitar este modelo como inferior ao melhor"
— **não** que ele venceu. A dependência temporal da série de losses é tratada via
**block bootstrap** (em vez do kernel HAC do DM).

### Elementos

**1. Loss = squared_error (herdada via `_pairwise_preprocess`)**
- **O que o doc afirma:** MCS é alimentado pela mesma loss_matrix que o DM — squared_error
  calculada em `_pairwise_preprocess` e pivotada por `target_timestamp_utc`.
- **Como deveria funcionar (exemplo):** se o modelo prevê 102 e o real é 100, a loss é
  `(102-100)^2 = 4`. A loss_matrix é uma tabela timestamps × configs. O MCS usa essa
  matriz para decidir quais configs pertencem ao confidence set.
- **O que esperar no código:** `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2`
  em `_pairwise_preprocess`; a loss_matrix é o pivot dessa coluna.
- **Ref do doc → código:** [`pairwise.py:280`](../../../../src/domain/services/gold_builders/pairwise.py#L280).
  ✅ **Confere:** `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2`.
- ⚠️ **Ponto de atenção — loss desalinhada do claim probabilístico**
  Idêntico ao item #1 (DM): squared_error mede acurácia do ponto previsto, não qualidade
  distribucional. O MCS com squared_error responde "qual conjunto de modelos não pode ser
  rejeitado como inferior em erro pontual?" — pergunta diferente de "qual conjunto é
  melhor probabilisticamente?".

  Para o TCC que reivindica superioridade do TFT em predição probabilística, a loss coerente
  seria pinball (q10/q50/q90) ou CRPS/WIS. A Phase B usou pinball_loss_post_guardrail no DM;
  o gold MCS usa squared_error. Reportar "TFT está no confidence set superior" usando
  squared_error não sustenta claim probabilístico — sustenta claim pontual.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: substituir
  > `squared_error` por **pinball loss** (q10/q50/q90) como loss primária do MCS gold,
  > tornando o confidence set coerente com o claim probabilístico do TCC. Enquanto a
  > mudança não for implementada, reportar o gold MCS apenas como descritivo pontual.

**2. Matriz de losses por timestamp (estrutura de entrada)**
- **O que o doc afirma:** mesmo padrão do DM — pivot wide por `target_timestamp_utc`,
  seguido de `dropna(how="any")` (herdado de `_pairwise_preprocess` + builder).
- **Como deveria funcionar (exemplo):** com 3 modelos A, B, C e 50 timestamps: só os
  timestamps em que os 3 previram entram na loss_matrix (intersecção de cobertura). Se A
  tem 50, B tem 45, C tem 48 → apenas os ~43 timestamps comuns são usados. O MCS vê
  n_obs = 43 para todos os pares simultaneamente.
- **O que esperar no codigo:** `pivot(index="target_timestamp_utc", columns="config_label",
  values="squared_error")` + `dropna(axis=0, how="any")`.
- **Ref do doc → codigo:** [`pairwise.py:353-360`](../../../../src/domain/services/gold_builders/pairwise.py#L353).
  ✅ **Confere:** linhas 352-360 replicam exatamente o padrao do DM builder (pivot + dropna).

**3. top-50 aplicado antes do MCS**
- **O que o doc afirma:** `_select_top_configs_for_pairwise(g, max_configs=50)` aplicado
  antes de montar a loss_matrix (linha 351).
- **Como deveria funcionar (exemplo):** de 200 configs possíveis, só as 50 com menor
  squared_error médio no test split entram no MCS — o universo já é filtrado antes de
  qualquer estatística ser computada. O confidence set resultante é relativo a esse
  universo reduzido, não ao universo original.
- **O que esperar no codigo:** chamada ao filtro logo no inicio do loop de grupos do builder.
- **Ref do doc → codigo:** [`pairwise.py:351`](../../../../src/domain/services/gold_builders/pairwise.py#L351).
  ✅ **Confere:** `g = _select_top_configs_for_pairwise(g, max_configs=50)`.

**4. Parâmetros hard-coded: B=300, block_len=5, random_seed=42**
- **O que o doc afirma:** alpha=0.05; bootstrap_samples=300; block_len=5; random_seed=42.
- **Como deveria funcionar (exemplo):** B=300 significa que o p-value bootstrap tem
  resolução mínima de 1/300 ≈ 0.0033. Para alpha=0.05, a variância do estimador é
  `p(1-p)/B ≈ 0.05 × 0.95 / 300 ≈ 0.00016`, ou seja desvio-padrão ≈ 0.012. Um "p
  verdadeiro" de 0.045 pode ser estimado como 0.033 ou 0.057 dependendo do seed → modelo
  pode entrar ou sair do confidence set por acidente do Monte Carlo.
- **O que esperar no código:** os quatro parâmetros como defaults na assinatura da função.
- **Ref do doc → código:** [`pairwise.py:109-113`](../../../../src/domain/services/gold_builders/pairwise.py#L109).
  ✅ **Confere:** `alpha=0.05`, `bootstrap_samples=300`, `block_len=5`, `random_seed=42`
  como defaults; chamada em `McsResultsGoldBuilder.build()` (linha 363) sem kwargs →
  todos os defaults são usados.
- ⚠️ **Ponto de atenção — B=300 é baixo; block_len=5 não está justificado**

  **B=300 (número de amostras bootstrap)**

  A variância do estimador do p-value é `p(1-p)/B`. Com B=300 e p ≈ 0.05:
  - IC 95% ≈ [0.026, 0.074] — a faixa inclui tanto rejeição quanto não-rejeição em alpha=0.05.
  - Para modelos com performance próxima (os mais interessantes para o TCC), o p-value
    estará perto de 0.05 exatamente onde B=300 tem maior incerteza.
  - O `random_seed=42` garante **reprodutibilidade** (mesmos dados → mesmo resultado),
    mas não elimina a instabilidade: com outro seed igualmente válido, o conjunto selecionado
    poderia ser diferente.
  - O skeleton cita B >= 1000 (preferencialmente 5000) como critério para promoção a
    confirmatório. Mesmo para uso descritivo estável, B=300 é a fronteira inferior.

  **block_len=5 (tamanho do bloco de bootstrap)**

  O block bootstrap (Kunsch 1989) é consistente apenas se `block_len → ∞` mais lentamente
  que `n_obs → ∞`. O choice de block_len afeta diretamente a estimativa da variância de
  `dbar` e portanto o p-value:

  | Escolha de block_len | Quando razoável | Risco |
  |---|---|---|
  | 5 fixo (atual) | n_obs ~20-50, série de baixa autocorrelação | Subestima dependência quando n_obs grande ou série persistente |
  | n^(1/3) automático | Regra análoga ao HAC do DM | Varia com a amostra, mais adaptativo |
  | Politis-White 2004 (data-driven) | Caso geral | Mais defensável; mais complexo de implementar |

  Com block_len muito pequeno: o bootstrap subestima a autocorrelação → subestima a
  variância de `dbar` → estatística TR artificialmente grande → **rejeita H0 mais facilmente
  → confidence set mais restrito** (menos modelos sobrevivem). Se os modelos TFT e baseline
  fossem difíceis de separar com block_len correto mas ficassem separados com block_len=5,
  o claim de superioridade seria artefato do parâmetro.

  Sem justificativa documental para block_len=5 (sem referência a autocorrelação observada
  nas séries de losses, sem estimativa data-driven), o parâmetro é um hardcode de
  conveniência. Para promoção a confirmatório, a regra de block_len precisa ser declarada
  e justificada antes de ver os resultados.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*:
  > - **B:** aumentar de 300 para **5000**. Custo é apenas tempo de execução; sem impacto
  >   arquitetural. Elimina a instabilidade de Monte Carlo na fronteira de decisão.
  > - **block_len:** adotar **`n^(1/3)` adaptativo** como regra base (análoga ao lag HAC
  >   do DM, sem dependência de biblioteca externa); complementar com **análise de
  >   sensibilidade** (rodar com block_len = n^(1/3), 10, 20 e verificar se o confidence
  >   set muda — estabilidade é evidência positiva). Avaliar complexidade de adotar
  >   **Politis-White (2004) data-driven** como alternativa mais defensável se o MCS
  >   for promovido a confirmatório.

**5. Moving block bootstrap com wrap-around**
- **O que o doc afirma:** moving block bootstrap com wrap-around (linhas 124-130).
- **Como deveria funcionar (exemplo):** com n_obs=10 e block_len=5, o resampling sorteia
  um início aleatório — digamos 7. O bloco seria os índices [7, 8, 9, 0, 1] (módulo 10:
  wrap-around circular). Isso evita blocos "cortados" no final da série. Repete até
  acumular n_obs observações.
- **O que esperar no código:** `(start + o) % n_obs` dentro da geração de blocos; loop
  `while len(idx) < n_obs`.
- **Ref do doc → código:** [`pairwise.py:124-130`](../../../../src/domain/services/gold_builders/pairwise.py#L124).
  ✅ **Confere:** `block = [(start + o) % n_obs for o in range(block_len)]`; loop
  `while len(idx) < n_obs`.

**6. Variância de `dbar` estimada pelo bootstrap**
- **O que o doc afirma:** para cada par de modelos (i, j), a variância de `dbar[i,j]` é
  estimada pela dispersão das B=300 estimativas bootstrap: `var = np.var(boot, ddof=1)`.
  Pares com variância ≤ 1e-12 recebem `nan` para evitar divisão por zero no t-stat.
- **Como deveria funcionar (exemplo):** para o par (TFT, LSTM), o código calculou
  `dbar_boot[b]` — a média de (loss_TFT − loss_LSTM) — em cada uma das B=300 amostras.
  Suponha que essas 300 estimativas variem entre −0.2 e +0.4; a variância delas é ≈ 0.03.
  Esse valor vai para o denominador do t-stat: `T = dbar_obs / sqrt(0.03)`.

  Três detalhes que ajudam a ler o código:
  - **Por que dispersão bootstrap = variância de `dbar`:** porque é a própria definição —
    se você calculou `dbar` em B versões dos dados, a dispersão *é* o quanto `dbar` varia
    de amostra para amostra. Como cada amostra usa blocos contíguos, a autocorrelação
    temporal já fica capturada implicitamente, sem fórmula HAC.
  - **Variância da média, não da série:** `dbar[i,j]` é a *média* de `d_t` ao longo de n
    timestamps; `var[i,j]` mede o quanto essa média varia entre amostras. Com n=100 e
    Var(série) ≈ 1.0: Var(média) ≈ 0.01 — a média é muito mais estável do que cada ponto
    individual, e é esse valor menor que vai para o denominador, tornando o t-stat maior
    para diferenças consistentes.
  - **Guard → nan:** ocorre quando dois modelos têm losses idênticas (mesmo seed, mesmos
    dados) — `dbar_boot` nunca varia, `var = 0`. O `nan` faz o `nanmax` ignorar esse par
    ao calcular TR, que é o comportamento correto.
- **O que esperar no código:** `var = np.var(boot, axis=0, ddof=1)` produzindo uma matriz
  M×M (um valor por par); `var[var <= 1e-12] = np.nan` logo abaixo.
- **Ref do doc → código:** [`pairwise.py:147-148`](../../../../src/domain/services/gold_builders/pairwise.py#L147).
  ✅ **Confere:** exatamente `var = np.var(boot, axis=0, ddof=1)` e
  `var[var <= 1e-12] = np.nan`.
- ⚠️ **Ponto de atenção — estimador bootstrap vs HAC: caminhos diferentes para a mesma quantidade**
  O DM estima `Var(mean_d)` por fórmula analítica (kernel Bartlett sobre a autocovariância
  observada). O MCS estima a mesma quantidade pela dispersão empírica das amostras
  bootstrap. Ambas são válidas nos seus contextos:

  | | DM (HAC) | MCS (bootstrap) |
  |---|---|---|
  | Como estima | Fórmula: γ₀ + 2Σ wₖγₖ | Empiricamente: Var das B amostras |
  | Trata autocorrelação | Explicitamente via kernel | Implicitamente via estrutura de bloco |
  | Parâmetro crítico | lag (n^1/3) | block_len |
  | Escala | Um par de modelos | Matriz M×M simultaneamente |

  A abordagem bootstrap é a especificada por Hansen-Lunde-Nason (2011) para o MCS e está
  correta — mas a qualidade da estimativa depende diretamente de block_len. Ver ⚠️ no
  elemento 4 (block_len=5 sem justificativa).

**7. Estatística TR (range t-stat — o maior t-stat entre todos os pares de modelos ativos): `nanmax|dbar/sqrt(var)|`**
- **O que o doc afirma:** `tr_stat = nanmax(|dbar / sqrt(var)|)` (linha 153).
- **Como deveria funcionar (exemplo):** com 3 modelos ativos {A, B, C} e médias de loss
  {0.5, 0.8, 1.2}: calcula `T_{AB} = |dbar_{AB} / sqrt(var_{AB})|`, `T_{AC}`, `T_{BC}`.
  `TR = max(T_{AB}, T_{AC}, T_{BC})`. Intuitivamente: qual par tem a diferença de
  performance mais fortemente suportada pelos dados? Se TR for grande e o bootstrap
  confirmar (p < alpha), há evidência de que pelo menos um modelo é inferior.
- **O que esperar no código:** `tmat = np.abs(dbar / np.sqrt(var))` seguido de
  `tr_stat = float(np.nanmax(tmat))`.
- **Ref do doc → código:** [`pairwise.py:150-153`](../../../../src/domain/services/gold_builders/pairwise.py#L150).
  ✅ **Confere:** `tmat = np.abs(dbar / np.sqrt(var))` na linha 150;
  `tr_stat = float(np.nanmax(tmat))` na linha 153.

**8. Bootstrap TR e p-value**
- **O que o doc afirma:** (implícito; o skeleton menciona "range t-stat" e block bootstrap).
- **Como deveria funcionar (exemplo):** cada uma das B=300 amostras bootstrap produz
  `dbar_boot` (média das diferenças no bootstrap sample). A estatística bootstrap é
  **centrada**: `dbar_boot - dbar_obs` — remove o viés da média. Então
  `|boot_centered / sqrt(var)|` e `max` sobre todos os pares → `tr_boot[b]`. O p-value
  é `mean(tr_boot >= tr_stat_obs)`. Lógica: "qual fração das amostras bootstrap, sob a
  hipótese nula de igualdade, produz TR tão extremo quanto o observado?"
- **O que esperar no código:** `boot_centered = boot - dbar[None, :, :]`; `tboot =
  |boot_centered / sqrt(var)[None, :, :]|`; `tr_boot[b] = nanmax(tboot[b])`; `pvalue =
  mean(tr_boot >= tr_stat)`.
- **Ref do doc → código:** [`pairwise.py:154-166`](../../../../src/domain/services/gold_builders/pairwise.py#L154).
  ✅ **Confere:** exatamente esse padrão nas linhas 154-166; `tr_boot` é filtrado para
  finitos (linha 163) antes de computar o p-value.

**9. Critério de parada e eliminação**
- **O que o doc afirma:** elimina por `argmax(losses_mean)` (linha 170); para quando
  `pvalue >= alpha`.
- **Como deveria funcionar (exemplo):** passo corrente com 3 modelos {A, B, C} com médias
  {0.5, 0.8, 1.2}: se p < 0.05 → eliminar C (maior mean_loss no subset ativo); repetir
  com {A, B}. Se agora p >= 0.05 → parar → MCS = {A, B}. Interpretação: "não há
  evidência estatística para distinguir A de B entre si; C foi rejeitado como inferior."
- **O que esperar no código:** `losses_mean = np.mean(sub, axis=0)`;
  `worst_local = int(np.argmax(losses_mean))`; `active.pop(worst_local)`.
- **Ref do doc → código:** [`pairwise.py:167-171`](../../../../src/domain/services/gold_builders/pairwise.py#L167).
  ✅ **Confere:** `if pvalue >= alpha or not np.isfinite(pvalue): break` na linha 167;
  `losses_mean = np.mean(sub, axis=0)` na linha 169; `worst_local = int(np.argmax(
  losses_mean))` na linha 170; `active.pop(worst_local)` na linha 171.

**10. Saída: `selected_in_mcs_alpha_0_05` e split_signature propagada condicionalmente**
- **O que o doc afirma:** saída lista `config_label`, `selected_in_mcs_alpha_0_05`,
  `mean_loss`; `_pairwise_group_cols` inclui `split_signature` quando a coluna existe
  (pairwise.py:25-29); o MCS propaga essa chave na saída (pairwise.py:368-374).
- **Como deveria funcionar (exemplo):** config A tem `selected_in_mcs_alpha_0_05 = True`
  e mean_loss = 0.5; config C tem `False` e mean_loss = 1.2. "True" significa: A nao
  pôde ser rejeitado como pertencente ao melhor grupo — **nao** que A venceu.
  `split_signature` e adicionada ao output quando estava presente nos dados de entrada.
- **O que esperar no código:** output com 3 colunas base (config_label,
  selected_in_mcs_alpha_0_05, mean_loss); builder adiciona asset, parent_sweep_id,
  split_signature (condicional), split, horizon, aligned_timestamps, n_configs.
- **Ref do doc → código:** [`pairwise.py:174-179`](../../../../src/domain/services/gold_builders/pairwise.py#L174)
  (saída base) e [`pairwise.py:368-374`](../../../../src/domain/services/gold_builders/pairwise.py#L368)
  (split_signature condicional). ✅ **Confere:** linhas 174-179 saída com 3 colunas base;
  linhas 368-374 propagação condicional: `if "split_signature" in group_cols: mcs_df[
  "split_signature"] = keys[2]`.
- ⚠️ **Ponto de atenção — "selecionado" != "vencedor": risco de interpretação do artefato**
  O nome `selected_in_mcs_alpha_0_05` é tecnicamente correto, mas a leitura coloquial
  como "aprovado/vencedor" é um erro comum na literatura e em relatórios. O MCS tem uma
  assimetria importante:

  - **selected=True:** "não foi possível rejeitar este modelo como inferior ao melhor com
    os dados disponíveis" → sobreviveu as eliminações, mas o set pode conter modelos muito
    diferentes entre si.
  - **selected=False:** "foi rejeitado (p < alpha)" → há evidência de que é pior que algum
    modelo no set remanescente.

  Cenários relevantes para o TCC:

  | Cenário | MCS diz | Interpretação correta | Erro típico |
  |---|---|---|---|
  | TFT e baseline ambos `True` | "Não distinguível" | Não há evidência de superioridade do TFT via MCS | "TFT aprovado pelo MCS" (omite que baseline também passou) |
  | Apenas TFT `True` | TFT sobreviveu; baseline rejeitado | Evidência de superioridade do TFT em squared_error | Correto — mas loss desalinhada (ver elemento 1) |
  | Confidence set com 40/50 configs `True` | Pouca potência | MCS não discriminou; falta de poder, não equivalência | "40 modelos excelentes" |

  O risco prático: reportar "`mcs_selected_alpha_0_05 = True`" sem mencionar o tamanho do
  confidence set e se o baseline também está incluído. Para o TCC, a interpretação deve
  ser: "o confidence set a 5% contém os modelos {X, Y, Z}; os demais foram rejeitados como
  estatisticamente inferiores ao melhor com base em squared_error."

**11. Consumo em confidence.py: split_signature descartada no merge**
- **O que o doc afirma:** `selected_in_mcs_alpha_0_05` é renomeado para
  `mcs_selected_alpha_0_05` e mergeado em `gold_model_decision_final` como coluna
  diagnóstica (confidence.py:633-647). O dossiê aponta: "se `split_signature` não vier
  populada do upstream, a família MCS perde essa chave."
- **Como deveria funcionar (exemplo):** se `gold_mcs_results` contém duas linhas para
  config A — `split_signature="fold1_test"` com selected=True e `"fold2_test"` com
  selected=False — o merge deveria preservar essa granularidade ou agregar com regra
  explícita (ex.: `any()` ou `all()` sobre split_signatures).
- **O que esperar no código:** o consumer em `confidence.py` propagaria `split_signature`
  ou agregaria explicitamente antes do merge.
- **Ref do doc → código:** [`confidence.py:633-647`](../../../../src/domain/services/gold_builders/confidence.py#L633).
  ✅ **Confere a seleção de colunas** — linhas 642-644 selecionam explicitamente:
  `["asset", "parent_sweep_id", "split", "horizon", "config_label",
  "selected_in_mcs_alpha_0_05"]`. Depois, na linha 727:
  `out.merge(df.drop_duplicates(merge_cols), ...)` com `merge_cols = ["asset",
  "parent_sweep_id", "split", "horizon", "config_label"]`.
- ⚠️ **Ponto de atenção — split_signature propagada pelo builder mas descartada pelo consumer**
  O dossiê descreve o risco como condicional ("se não vier populada do upstream").
  O código revela algo mais específico: a chave **é propagada pelo builder**
  (pairwise.py:368-374) e está em `gold_mcs_results`, mas o **consumer a descarta
  ativamente** (confidence.py:642-644) — independentemente do que vem do upstream.

  O impacto prático é em duas etapas:

  1. **Descarte de split_signature (linhas 642-644):** `mcs_summary` não tem
     `split_signature`; qualquer informação sobre qual split gerou aquele veredicto MCS
     é perdida.
  2. **drop_duplicates sem split_signature (linha 727):** se existirem múltiplas linhas
     por `(asset, parent_sweep_id, split, horizon, config_label)` — uma por split_signature
     — o `drop_duplicates(merge_cols)` as colapsa em uma linha, escolhendo
     **arbitrariamente** qual `mcs_selected_alpha_0_05` fica (a que aparece primeiro no
     DataFrame após o `.copy()` da linha 644).

  Quando importa: em sweeps com múltiplos splits de test (walk-forward com vários folds),
  um config pode ser "selecionado" em alguns split_signatures e "não selecionado" em
  outros. O `gold_model_decision_final.mcs_selected_alpha_0_05` vai refletir apenas um
  deles — não necessariamente o mais representativo. Isso é não-determinístico em relação
  a ordenação do DataFrame.

  Comparação com o DM: o DM em `confidence.py:599-600` também agrupa por `["asset",
  "parent_sweep_id", "split", "horizon"]` sem split_signature — mas lá o resultado é uma
  **soma de vitórias/derrotas** (via groupby + iterrows), o que é uma agregação implícita.
  No MCS, o colapso é via `drop_duplicates`, que não agrega, apenas escolhe uma linha.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: incluir
  > `split_signature` na seleção de colunas de `mcs_summary` (`confidence.py:642-644`) e
  > no `merge_cols` do merge final (`confidence.py:720-722`), propagando a chave até
  > `gold_model_decision_final`. Não há razão técnica para descartá-la — o builder já a
  > fornece. Agregar com regra explícita (`all()` ou `any()`) quando múltiplos
  > split_signatures existirem, em vez de deixar o `drop_duplicates` escolher
  > arbitrariamente.

### Cross-check — o que NÃO está corretamente indicado/referenciado

Para o MCS, as referências de "Implementação atual localizada" são precisas e conferem
com o código. Pontos a registrar:

1. **Sem referências quebradas.** As linhas-âncora citadas estão corretas:
   - Função `_compute_mcs_from_loss_matrix`: linhas 107-180 ✅ (confirmo por leitura).
   - Parâmetros alpha/B/block_len/seed: linhas 109-113 ✅ exatos.
   - Block bootstrap: linhas 124-130 ✅ exatos.
   - TR stat: linha 153 ✅ exata (`float(np.nanmax(tmat))`).
   - Eliminação: linha 170 ✅ exata (`int(np.argmax(losses_mean))`).
   - Top-50: linha 351 ✅ exato.
   - split_signature: linhas 25-29 (group_cols) ✅ e linhas 368-374 (propagação) ✅.

2. **Risco de split_signature incorretamente descrito como problema upstream; o drop é downstream.**
   O dossiê diz: "Se `split_signature` não vier populada do upstream, a família MCS perde
   essa chave." Essa descrição está imprecisa: o builder **propaga corretamente**
   `split_signature` (pairwise.py:368-374) e ela existe em `gold_mcs_results`. O descarte
   acontece **no consumer** (`confidence.py:642-644`), de forma ativa e incondicional —
   independente do upstream. A correção pertence ao consumer, não ao upstream.

3. **`drop_duplicates(merge_cols)` sem split_signature (confidence.py:727) não está
   mencionado no dossiê.** Quando há múltiplos split_signatures, o colapso é
   não-determinístico em relação ao valor de `mcs_selected_alpha_0_05` persistido no
   artefato final. Isso é mais específico do que "a chave se perde."

4. **Ponto residual metodológico central: B=300 e block_len=5** são os riscos mais
   relevantes para promoção a confirmatório. O skeleton capturou ambos ("B=300 baixo;
   sensibilidade a block_len; loss desalinhada"), mas sem detalhar o mecanismo
   quantitativo. Os aprofundamentos acima complementam.

### Veredito do item #2

🟡 **Ressalvas.** Referências intactas; evidência fiel ao código. Dois pontos
que o dossiê não captura com precisão suficiente: (a) o descarte de
`split_signature` ocorre no consumer (`confidence.py:642-644`), não é
condicional ao upstream — é ativo e independente; (b) o `drop_duplicates`
na linha 727 pode colapsar registros de split_signatures distintos de forma
não-determinística. Riscos metodológicos (B=300, block_len=5, loss desalinhada)
são decisões para C.0.2/C.0.3, não defeitos de localização.

---

## #3 — Holm gold

### O que é (didático)

O **Holm-Bonferroni** é uma correção de **múltiplas comparações**. O problema que
ele resolve: se você roda 100 testes ao nível α=5%, espera-se ~5 "significativos"
**só por sorte**, mesmo sem nenhum efeito real. Quanto mais testes, maior a chance
de pelo menos um falso positivo. O Holm controla o **FWER** (*family-wise error
rate* — a probabilidade de cometer **≥1** falso positivo na **família** inteira de
testes).

O "truque" é um procedimento **step-down**: ordena os p-values do menor para o
maior e exige cada vez **menos**: o menor p-value precisa passar de `α/m`, o
próximo de `α/(m−1)`, …, o maior só de `α/1`. Equivalentemente (forma usada no
código), multiplica o *j*-ésimo menor p-value por `(m − j + 1)` e força a
sequência a ser **monotônica crescente**. É **uniformemente mais poderoso que
Bonferroni** (que multiplicaria todos por `m`) e **não assume independência**
entre os testes.

A sutileza que faz ou quebra o Holm não está na fórmula — está na **definição da
família**: *quais* testes pertencem ao mesmo conjunto. Holm só é tão bom quanto a
família que você alimenta. No gold legacy, a fórmula está correta; a família é que
é definida por um **groupby administrativo**, não pelo claim.

### Elementos

**1. Onde o Holm roda e sobre qual p-value**
- **O que o doc afirma:** o ajuste é aplicado dentro de `DmPairwiseResultsGoldBuilder.build()` (linha 333), sobre os DM results já concatenados, lendo a coluna `pvalue_two_sided`; a saída ganha `pvalue_adj_holm` e `significant_adj_0_05` no parquet `gold_dm_pairwise_results`.
- **Como deveria funcionar (exemplo):** o builder roda o DM por grupo, junta todas as linhas pairwise (cada uma com seu `pvalue_two_sided`) num único DataFrame e chama **uma vez** `_apply_holm_adjustment_for_dm` sobre o concat. Para 6 pares com p-values crus `{0.005, 0.02, 0.03, 0.04, 0.20, 0.50}`, o Holm devolve uma coluna nova de p-values **ajustados** ao lado dos crus.
- **O que esperar no código:** `return _apply_holm_adjustment_for_dm(pd.concat(rows, ignore_index=True))` no fim do `build()`; dentro da função, um guard que devolve o df intacto se não houver `pvalue_two_sided`.
- **Ref do doc → código:** [`pairwise.py:333`](../../../../src/domain/services/gold_builders/pairwise.py#L333) (chamada) e [`pairwise.py:184`](../../../../src/domain/services/gold_builders/pairwise.py#L184) (guard). ✅ **Confere:** L333 `return _apply_holm_adjustment_for_dm(pd.concat(rows, ignore_index=True))`; L184 `if dm_results.empty or "pvalue_two_sided" not in dm_results.columns: return dm_results`. Confirmei lendo a função inteira: o input é o **`pvalue_two_sided`** (cru, two-sided) do DM.
- ⚠️ **Ponto de atenção — Holm herda o p-value que recebe; aqui é two-sided**
  O Holm é **agnóstico à direção do teste**: corrige *qualquer* vetor de p-values. Ele recebe o `pvalue_two_sided` do DM gold (ver item #1, elemento 5). Isso significa que **todas as ressalvas do p-value de entrada são herdadas pelo ajustado**: se o claim do TCC é direcional ("TFT melhor que baseline em pinball"), o p two-sided é o "errado para o claim", e o `pvalue_adj_holm` apenas corrige a multiplicidade de um teste já desalinhado. A Phase B faz o caminho coerente: corrige `pvalue_one_sided` ([`holm_family_6.py:16`](../../../../src/domain/services/holm_family_6.py#L16)). **Quando importa:** sempre que o número ajustado for lido como evidência do claim — aí o desalinhamento two-sided/one-sided do insumo se propaga para a conclusão. **Quando é tolerável:** se o uso for puramente diagnóstico/exploratório. Não é defeito do Holm; é o insumo que precisa casar com o claim (decisão de C.0.2/C.0.3, ver item #1).

**2. Composição da família: groupby `[asset, parent_sweep_id, split, horizon]` — `split_signature` ausente**
- **O que o doc afirma:** o groupby da família é `[asset, parent_sweep_id, split, horizon]` (linha 191) e **`split_signature` NÃO entra**, apesar de o DM ter `split_signature` na grain de pré-processamento.
- **Como deveria funcionar (exemplo):** Holm controla o FWER dentro de **uma família** = o conjunto de testes que sustentam **um claim**, declarada **antes** de olhar os p-values. Se o claim é "config A bate config B em BTC, sweep S1, horizonte 7", a família correta é o conjunto de pares testados sob *exatamente* esse desenho. Misturar testes de desenhos diferentes (ou repetir o mesmo teste em vários folds) numa só família contamina a contagem `m`.
- **O que esperar no código:** ou reuso de `_pairwise_group_cols` (que **insere** `split_signature` na posição 2 quando a coluna existe — [`pairwise.py:25-29`](../../../../src/domain/services/gold_builders/pairwise.py#L25)), ou uma lista explícita coerente com a grain do DM.
- **Ref do doc → código:** [`pairwise.py:191`](../../../../src/domain/services/gold_builders/pairwise.py#L191). ✅ **Confere:** `group_cols = [c for c in ["asset", "parent_sweep_id", "split", "horizon"] if c in out.columns]` — lista **hard-coded**, **sem** `split_signature`, e **sem** reusar `_pairwise_group_cols`.
- ⚠️ **Ponto de atenção — família por grain administrativo, e a assimetria DM-inclui / Holm-exclui o `split_signature`**
  Este é o ponto central do item. Há uma **inconsistência dentro do próprio builder**:

  | Etapa | Como agrupa | `split_signature`? | Ref |
  |---|---|---|---|
  | DM (cálculo dos testes) | `_pairwise_group_cols(df)` | **incluído** (inserido na pos. 2 se existir) | [`pairwise.py:297`](../../../../src/domain/services/gold_builders/pairwise.py#L297) + [`:25-29`](../../../../src/domain/services/gold_builders/pairwise.py#L25) |
  | Holm (família p/ correção) | lista hard-coded | **excluído** | [`pairwise.py:191`](../../../../src/domain/services/gold_builders/pairwise.py#L191) |

  Ou seja: o DM **computa os testes por `split_signature`** (cada fold/desenho gera
  seu próprio conjunto de pares, e a coluna `split_signature` é gravada de volta em
  [`pairwise.py:316-317`](../../../../src/domain/services/gold_builders/pairwise.py#L316)),
  mas o Holm **pool**a todos os `split_signature` que compartilham
  `(asset, parent_sweep_id, split, horizon)` numa única família. E como
  `_pairwise_preprocess` filtra `split == "test"` ([`pairwise.py:269`](../../../../src/domain/services/gold_builders/pairwise.py#L269)),
  na prática `split` é constante e o que distingue os folds **é justamente o
  `split_signature` descartado**.

  **O erro tem duas direções, em eixos diferentes — e qual delas domina depende do claim:**

  | Eixo | O que a omissão faz | Efeito no FWER | Quando é o risco |
  |---|---|---|---|
  | **`split_signature`** (folds do mesmo desenho) | *pool*a folds: o **mesmo par** (A,B) vira várias linhas na mesma família (uma por fold) | `m` infla com hipóteses **repetidas, não distintas** → mistura desenhos; correção incoerente | walk-forward com vários folds de test |
  | **horizon / baseline / asset / sweep** | *particiona*: cada `(asset, sweep, split, horizon)` é uma família separada | se o **claim agrega** sobre esses eixos, a família real é maior → **subcorrige** (FWER global não controlado) | claim que afirma superioridade "no geral" |

  **Exemplo numérico (eixo `split_signature`).** BTC, sweep S1, horizonte 7, 4
  configs → 6 pares por fold. Com 2 folds (`split_signature` = `foldA`, `foldB`),
  o DM gera 6 + 6 = 12 linhas. O Holm, omitindo `split_signature`, vê **m = 12** e
  trata como 12 hipóteses distintas — mas a hipótese "A bate B" aparece **duas
  vezes** (uma por fold). Holm pressupõe `m` hipóteses **distintas**; aqui ele as
  conta repetidas. A correção fica **incoerente** (nem o FWER por-fold nem o
  agregado é o que se quer).

  **Quando cada escolha se aplica e por quê:**
  - **Incluir `split_signature` (corrigir por fold):** correto se cada fold é um
    desenho/claim independente. Mantém famílias coerentes, mas exige depois decidir
    como **combinar** vereditos entre folds.
  - **Omitir e agregar antes (um teste por par, combinando folds):** correto se o
    claim é "A bate B *agregando* os folds" — mas então é preciso **agregar os
    p-values por par antes** do Holm (ex.: meta-análise / um DM sobre a série
    concatenada), não simplesmente empilhar as linhas dos folds.
  - **O que o código faz hoje:** nem um nem outro — empilha as linhas dos folds e
    corrige como se fossem testes distintos. É a opção que **não** corresponde a
    nenhuma família bem-definida.

  **Comparação com a Phase B.** A Phase B **declara a família ex-ante** (6 testes:
  3 baselines × 2 horizontes) e aplica o Holm exatamente sobre esses 6 p-values,
  passados como `Series` para [`holm_family_6.py`](../../../../src/domain/services/holm_family_6.py#L7) — a família não é
  derivada de um groupby, é **escolhida** para casar com H2a/H2b. (A Phase B
  também não usa `split_signature` no groupby — mas lá é **intencional e
  documentado**, porque a família confirmatória é um único conjunto pré-registrado;
  ver skeleton §"Familia Holm".) O gold legacy faz o oposto: a família **emerge** de
  um grain administrativo que ninguém declarou como correspondendo a um claim.

  **Implicação para o TCC:** a fórmula Holm está certa, mas a família **não é
  derivada de um claim**, então o `pvalue_adj_holm` do gold legacy **não tem
  interpretação de FWER bem-definida**. Não dá para dizer "controlamos o FWER da
  comparação de modelos a 5%" porque "a comparação de modelos" (o claim) não
  corresponde à partição usada. Promover esse número a confirmatório exige primeiro
  **declarar a família** e fazer o groupby/agregação baterem com ela.

**3. Fórmula Holm step-down: ordena, `(m − j + 1)·p`, `maximum.accumulate`, clip em 1**
- **O que o doc afirma:** dentro de cada grupo, ordena os p-values; calcula `(m − j + 1) · p` para o *j*-ésimo menor; aplica `np.maximum.accumulate`; faz clip em 1 (linhas 197-208).
- **Como deveria funcionar (exemplo):** família de 3 pares com p-values crus `{0.01, 0.04, 0.04}` → `m = 3`.
  - Ordena: `0.01, 0.04, 0.04` (j = 1, 2, 3).
  - `(m − j + 1)·p`: j=1 → `3·0.01 = 0.03`; j=2 → `2·0.04 = 0.08`; j=3 → `1·0.04 = 0.04`.
  - Sequência crua: `[0.03, 0.08, 0.04]` — repare que o terceiro (`0.04`) é **menor** que o segundo (`0.08`): não-monotônico.
  - `np.maximum.accumulate`: `[0.03, 0.08, 0.08]` — o terceiro é puxado para cima até `0.08`.
  - clip em 1: inalterado → ajustados `{0.03, 0.08, 0.08}`.

  Resultado: só `p = 0.01` (ajustado `0.03`) cruza 5%. **Por que o `accumulate` é
  necessário:** sem ele, um p-value *maior* (menos significativo) poderia receber um
  ajustado *menor* que um p-value menor — absurdo lógico (não se pode rejeitar o teste
  mais fraco e não rejeitar o mais forte). A monotonicidade garante a coerência
  step-down. **Por que é melhor que Bonferroni:** Bonferroni multiplicaria todos por
  `m = 3` → `{0.03, 0.12, 0.12}`; o Holm é ≤ Bonferroni em todo ponto → mais poder,
  mesma garantia de FWER.
- **O que esperar no código:** `ordered = valid.sort_values()`; loop `(m - j + 1) * float(pval)`; `np.minimum(1.0, np.maximum.accumulate(adj_vals))`.
- **Ref do doc → código:** [`pairwise.py:203-207`](../../../../src/domain/services/gold_builders/pairwise.py#L203). ✅ **Confere exatamente:** L203 `ordered = valid.sort_values()`; L205-206 `for j, (_, pval) in enumerate(ordered.items(), start=1): adj_vals.append((m - j + 1) * float(pval))`; L207 `adj_vals = np.minimum(1.0, np.maximum.accumulate(adj_vals))`. É a **mesma fórmula** de [`holm_family_6.py:25-27`](../../../../src/domain/services/holm_family_6.py#L25).
- ⚠️ **Ponto de atenção — ordenação não-estável vs. empates (é benigno, mas vale registrar)**
  O gold legacy usa `valid.sort_values()` (quicksort, **não estável**); a Phase B usa
  `sort_values(kind="mergesort")` (estável). Com p-values **empatados**, a ordem
  interna das linhas empatadas pode diferir entre execuções. **Por que não é bug:**
  para dois empates em `p` nas posições `j` e `j+1`, as estatísticas cruas são
  `(m−j+1)·p` e `(m−j)·p`; após o `maximum.accumulate`, **ambas** viram
  `(m−j+1)·p` (o maior). Logo o **valor ajustado é o mesmo** independentemente da
  ordem interna dos empates — o `accumulate` neutraliza a instabilidade. **Quando
  poderia importar:** só se algum consumidor dependesse de *qual linha física*
  recebeu qual valor entre empates idênticos — não é o caso aqui. Registro por
  completude/determinismo, não como defeito.

**4. Tamanho da família `m` = nº de p-values válidos (`dropna`)**
- **O que o doc afirma:** (implícito na fórmula) `m` é o número de p-values válidos; `valid = pv.dropna()`, `m = int(len(valid))`, e o grupo é pulado se `m == 0`.
- **Como deveria funcionar (exemplo):** num grupo onde o DM produziu 6 linhas, mas 1 par teve `pvalue_two_sided` não-numérico (coerce → NaN), `valid` cai para 5 → `m = 5`. Pares com p NaN **não contam** na família nem recebem `pvalue_adj_holm` (ficam NaN). Na prática, o DM só grava linhas com p finito (item #1: pula `n < 5` e `var ≤ 0` **sem criar linha**), então `m` ≈ nº de pares que sobreviveram ao DM.
- **O que esperar no código:** `valid = pv.dropna()`; `m = int(len(valid))`; `if m == 0: continue`.
- **Ref do doc → código:** [`pairwise.py:198-202`](../../../../src/domain/services/gold_builders/pairwise.py#L198). ✅ **Confere:** L198 `pv = pd.to_numeric(g["pvalue_two_sided"], errors="coerce")`; L199 `valid = pv.dropna()`; L200 `m = int(len(valid))`; L201-202 `if m == 0: continue`.
- ⚠️ **Ponto de atenção — `m` é contado sobre o universo já filtrado pelo top-50**
  O `m` da família **não** é o número de pares do universo completo de configs — é o
  número de pares **que sobreviveram ao top-50** (item #4), pois o filtro roda
  *antes* do DM ([`pairwise.py:299`](../../../../src/domain/services/gold_builders/pairwise.py#L299)). Com >50 configs, `m` é
  no máximo `C(50,2) = 1225` pares, sobre as 50 configs de menor `squared_error`
  médio **no próprio split de teste**. **Por que importa:** mesmo que a família
  fosse perfeitamente definida (elemento 2), a correção de multiplicidade opera
  sobre um subconjunto **selecionado pelos dados** — a garantia de FWER é
  *condicional à seleção*, não sobre o espaço original de modelos. **Quando é
  aceitável:** se o top-50 for declarado como parte do desenho e o claim for
  explicitamente "entre as 50 melhores". **Quando não é:** se o número for lido como
  "controlamos o FWER da comparação de todos os modelos". Liga-se diretamente ao
  problema de inferência seletiva do item #4.

**5. Flag `significant_adj_0_05 = pvalue_adj_holm < 0.05`**
- **O que o doc afirma:** a flag booleana `significant_adj_0_05` é `pvalue_adj_holm < 0.05` (linhas 211-213).
- **Como deveria funcionar (exemplo):** par com `pvalue_adj_holm = 0.03` → `True`; com `0.06` → `False`; com `NaN` (par sem ajuste) → `False`, porque `NaN < 0.05` é `False` em pandas/numpy. Ou seja, ausência de ajuste é tratada como **não-significativo**, que é o comportamento conservador correto.
- **O que esperar no código:** `pd.to_numeric(out["pvalue_adj_holm"], errors="coerce") < 0.05`.
- **Ref do doc → código:** [`pairwise.py:211-213`](../../../../src/domain/services/gold_builders/pairwise.py#L211). ✅ **Confere:** `out["significant_adj_0_05"] = (pd.to_numeric(out["pvalue_adj_holm"], errors="coerce") < 0.05)`.
- ⚠️ **Ponto de atenção — α = 0,05 hard-coded (Phase B parametriza)**
  O corte `0.05` está **fixo** no código do gold legacy, ao passo que a Phase B
  expõe `alpha` como parâmetro ([`holm_family_6.py:8`](../../../../src/domain/services/holm_family_6.py#L8) e o `< float(alpha)` em
  [`:32`](../../../../src/domain/services/holm_family_6.py#L32)). **Quando importa:** se C.0.3 decidir um α diferente (ou
  análise de sensibilidade ao α), o gold legacy exige editar código em vez de passar
  um parâmetro; e o **nome da coluna** (`significant_adj_0_05`) embute o `0.05`, então
  mudar o α sem renomear geraria um artefato com nome enganoso. Não é erro de
  cálculo; é rigidez de configuração que vale anotar para a fase de especificação.

**6. Consumo downstream: persistido no parquet, ignorado pelo rollup, mas usado no plot**
- **O que o doc afirma:** `pvalue_adj_holm` existe em `gold_dm_pairwise_results`, mas `_build_model_decision_final` calcula `dm_net_wins` a partir de `pvalue_two_sided < 0.05` (cru), **não** de `pvalue_adj_holm`; portanto o Holm fica disponível no artefato DM mas não é o critério efetivo do rollup final.
- **Como deveria funcionar (exemplo):** existem **três** noções de "DM significativo" circulando — e elas podem **discordar** para o mesmo par: (a) `significant_adj_0_05` no parquet (ajustado por Holm); (b) `dm_net_wins` no `gold_model_decision_final` (p **cru** < 0,05); (c) a célula da figura *DM P-Value Matrix* (que **prefere** o Holm quando disponível).
- **O que esperar no código:** no rollup, leitura de `pvalue_two_sided` e filtro `p >= 0.05`; no plot, preferência por `pvalue_adj_holm`.
- **Ref do doc → código:** [`confidence.py:605`](../../../../src/domain/services/gold_builders/confidence.py#L605) + [`confidence.py:609`](../../../../src/domain/services/gold_builders/confidence.py#L609) (rollup usa p cru) e [`generate_prediction_analysis_plots_use_case.py:430`](../../../../src/use_cases/generate_prediction_analysis_plots_use_case.py#L430) (plot prefere Holm). ✅ **Confere:** rollup — L599-600 groupby `["asset", "parent_sweep_id", "split", "horizon"]`, L605 `p = pd.to_numeric(r.get("pvalue_two_sided"), ...)`, L609 `if pd.isna(p) or pd.isna(d) or p >= 0.05: continue`, sem qualquer leitura de `pvalue_adj_holm`; plot — L430 `pcol = "pvalue_adj_holm" if "pvalue_adj_holm" in df.columns else (...)`, sobre `dm_df = gold_dm_pairwise_results` (L428).
- ⚠️ **Ponto de atenção — o Holm não é dormente: o plot oficial consome a Holm de família mal-definida**
  O dossiê (e o item #1, elemento 7) deixa a impressão de que o Holm fica "disponível
  mas inerte" porque o `decision_final` o ignora. O código mostra que **não é bem
  assim**: a figura *DM P-Value Matrix* (`_build_fig_dm_pvalue_matrix`)
  **prefere ativamente** o `pvalue_adj_holm` sobre o `pvalue_two_sided`
  ([`generate_prediction_analysis_plots_use_case.py:430`](../../../../src/use_cases/generate_prediction_analysis_plots_use_case.py#L430)). Consequências:

  - **Três vereditos divergentes** para "DM significativo" — parquet (Holm),
    `decision_final` (cru), plot (Holm). O leitor que cruza a figura com o
    `gold_model_decision_final` pode ver **conclusões diferentes** sobre o mesmo par.
  - **O problema da família (elemento 2) propaga para a figura.** Como o plot usa o
    `pvalue_adj_holm`, a matriz de p-values exibida carrega a **família
    mal-definida** (sem `split_signature`, sobre o top-50). Se a figura for lida como
    evidência de superioridade, o defeito de composição da família vira um artefato
    visual.

  **Quando importa:** quando a figura ou o `decision_final` forem usados para
  embasar/ilustrar o claim — aí a divergência entre os três números e a família
  mal-definida deixam de ser "diagnóstico" e passam a confundir a evidência.
  **Quando é tolerável:** se tudo isso for explicitamente rotulado como descritivo
  não-inferencial. A decisão de unificar (qual p-value cada artefato usa, e sobre
  qual família) é de C.0.2/C.0.3.

### Cross-check — o que NÃO está corretamente indicado/referenciado

Todas as referências de "Implementação atual localizada" do dossiê de Holm
**conferem** com o código (li a função `_apply_holm_adjustment_for_dm` inteira,
183-214, mais a chamada em 333, o helper 25-29, o rollup em confidence.py 599-609 e
o plot em 430). Pontos a registrar:

1. **Nenhuma referência quebrada.** Linhas-âncora corretas: função 183-214 ✅;
   groupby sem `split_signature` em L191 ✅; fórmula `(m − j + 1)·p` +
   `maximum.accumulate` + clip em L203-207 ✅ (o dossiê cita "197-208"/"203-207" —
   ambos batem com o corpo real); flag `significant_adj_0_05 = pvalue_adj_holm <
   0.05` em L211-213 ✅.
2. **O dossiê descreve o groupby como "lista de 4 colunas" mas não menciona a
   assimetria com `_pairwise_group_cols`.** O ponto mais informativo não é só "falta
   `split_signature`": é que o **DM inclui** `split_signature` (via
   `_pairwise_group_cols`, L297 + L25-29) e o **Holm exclui** (lista hard-coded em
   L191, sem reusar o helper). A omissão é uma **divergência interna do builder**,
   não um esquecimento isolado. Isso reforça que `split_signature` **está populado**
   em `gold_dm_pairwise_results` (gravado em L316-317) — o descarte é ativo no Holm,
   não ausência upstream (mesmo padrão do achado do item #2 sobre o MCS).
3. **O campo "Uso atual no projeto" subdescreve o consumo.** Afirma que o Holm "não
   é hoje o critério efetivo do rollup final" — verdade para o `decision_final`, mas
   **omite** que a figura *DM P-Value Matrix* (`generate_prediction_analysis_plots_use_case.py:430`)
   **prefere** o `pvalue_adj_holm`. O Holm gold **é** consumido — por um plot, não
   pelo rollup. Discrepância de completude, registrada no elemento 6.
4. **"split" é praticamente constante na grain do Holm.** Como `_pairwise_preprocess`
   filtra `split == "test"` (L269), `split` no groupby de L191 não discrimina nada; o
   que de fato distinguiria desenhos seria o `split_signature` — exatamente o que está
   de fora. O dossiê não explicita isso.
5. **Ponto residual metodológico central:** a **definição da família** (elemento 2) é
   o que decide se o `pvalue_adj_holm` tem interpretação de FWER. O dossiê captou o
   risco ("família possivelmente mal definida"; "subcorrige"), mas o aprofundamento
   acima detalha as **duas direções** (pool por `split_signature` vs. partição por
   horizon/baseline/asset/sweep) e a condicionalidade ao top-50.

### Veredito do item #3

🟡 **Ressalvas.** Referências intactas e evidência fiel ao código; a **fórmula
Holm step-down está correta** (idêntica à da Phase B). As ressalvas são
metodológicas e de descrição, não de localização: (a) a **família é definida por um
grain administrativo** (`[asset, parent_sweep_id, split, horizon]`, sem
`split_signature`) que não corresponde a nenhum claim declarado — o DM **inclui**
`split_signature` mas o Holm **exclui**, uma divergência interna do builder; (b) o
dossiê subdescreve o consumo — o Holm gold **é** usado pela figura *DM P-Value
Matrix* (não é dormente), propagando a família mal-definida para um artefato visual;
(c) `m` é contado sobre o universo já filtrado pelo top-50. Decisões de declarar a
família, unificar qual p-value cada artefato consome e parametrizar α são de
C.0.2/C.0.3.

---

## #4 — top-50 filter

### O que é (didático)

O **top-50 filter** **não é um teste estatístico** — é um **pré-filtro de
engenharia** que roda **antes** do DM, do MCS e do win-rate. Dentro de cada grupo
(asset, sweep, [split_signature], split, horizonte), se houver **mais de 50
configs**, ele mantém apenas as **50 com menor `squared_error` médio no próprio
split de teste** e descarta o resto. A motivação legítima é **conter custo
combinatório**: o nº de pares cresce com `C(k,2)` (≈ k²/2), então 200 configs dão
~19.900 pares e 50 configs dão 1.225. O **problema** é o critério: filtrar pelo
**resultado do teste** e depois **testar sobre os mesmos dados** é **seleção sobre
o desfecho** — o pecado clássico de **inferência seletiva / data snooping**. E o
corte é **silencioso**: nada downstream registra que o universo foi podado.

### Elementos

**1. Critério de seleção: menor `mean(squared_error)` no test split**
- **O que o doc afirma:** "filtra top-50 por `mean(squared_error)` no próprio test
  split"; ranking `groupby("config_label")["squared_error"].mean().sort_values(
  ascending=True).head(max_configs)` (linhas 57-61) — menor squared_error = melhor.
- **Como deveria funcionar (exemplo):** de 200 configs, computa a média de
  `squared_error` de cada config sobre as linhas do grupo (test split), ordena
  crescente e mantém as 50 menores. Ex.: config A média 0,50; B 0,55; … a 50ª com
  0,90 entra; a 51ª com 0,91 é descartada. "Menor é melhor" porque squared_error é
  erro — quanto menor, mais acurado.
- **O que esperar no código:** `rank = grouped_oos.groupby("config_label",
  dropna=False)["squared_error"].mean().sort_values(ascending=True).head(max_configs)`,
  depois `keep = set(rank.index.tolist())` e `grouped_oos[...isin(keep)].copy()`.
- **Ref do doc → código:** [`pairwise.py:57-61`](../../../../src/domain/services/gold_builders/pairwise.py#L57)
  (rank) + [`pairwise.py:63-64`](../../../../src/domain/services/gold_builders/pairwise.py#L63)
  (keep + filtro). ✅ **Confere** (li a função inteira, 43-64): `rank = (
  grouped_oos.groupby("config_label", dropna=False)["squared_error"].mean()
  .sort_values(ascending=True).head(max_configs))`; `keep = set(rank.index.tolist())`;
  `return grouped_oos[grouped_oos["config_label"].isin(keep)].copy()`.
- ⚠️ **Ponto de atenção — inferência seletiva / data snooping (o ponto central do item)**
  Este é o coração do item #4. Selecionar as 50 melhores **no test set** e depois
  rodar DM/MCS/win-rate **no mesmo test set** significa **pré-triar pelo próprio
  desfecho** que se vai testar. As 50 configs mantidas são, por construção, as que
  **por acaso** foram bem nesta amostra de teste — incluindo as que foram bem por
  **sorte**. Os p-values/confidence sets subsequentes são computados sobre um
  universo **já enviesado** para baixa perda.

  **Por que infla a "significância":** imagine 200 configs todas **igualmente boas**
  (sem efeito real). Por puro ruído, ~algumas terão `squared_error` baixo nesta
  amostra. Ao manter só as 50 de menor perda e comparar, você está comparando
  "vencedoras de sorteio" — a diferença média parece maior do que é, e o teste
  rejeita H₀ com mais frequência do que o α nominal. É o mesmo mecanismo do
  "winner's curse".

  **Quando cada escolha se aplica e por quê:**

  | Situação | A seleção é aceitável? | Por quê |
  |---|---|---|
  | Universo **pré-declarado** (configs escolhidas **antes** de ver o test) | ✅ Sim | Não há uso do desfecho na seleção |
  | Seleção em **validation**, teste em **test** (out-of-sample) | ✅ Sim | A triagem e a inferência usam dados **disjuntos** |
  | Top-50 por **test loss**, rotulado **exploratório** | ⚠️ Tolerável | Desde que **nunca** vire claim |
  | Top-50 por **test loss**, lido como evidência de superioridade | 🔴 Não | Data snooping puro — α nominal não vale |

  **Correção principled** quando se precisa testar sobre um universo grande: usar um
  procedimento que controla o erro **sobre o universo inteiro**, não sobre o
  subconjunto pré-triado — o **Reality Check de White (2000)** ou o **stepdown de
  Romano-Wolf** (citados no dossiê). Eles foram desenhados exatamente para "testar
  muitos modelos e ainda controlar a taxa de falsos positivos".

  **Comparação com a Phase B:** a Phase B **não usa** este filtro — opera sobre um
  **cohort fixo de 6 testes pré-registrados** (3 baselines × 2 horizontes). A família
  é escolhida **antes** de ver os dados; não há seleção por desfecho.

  **Implicação para o TCC:** qualquer número de DM/MCS/win-rate do gold legacy é
  **condicional a um universo selecionado pelo teste**. Reportado como "superioridade"
  ele **superestima** — a garantia estatística não cobre a etapa de seleção.

**2. Idempotência / passthrough quando `cfg_count ≤ 50` (cap silencioso)**
- **O que o doc afirma:** "retorna o próprio DataFrame sem filtro se `cfg_count <=
  max_configs` (linha 55)".
- **Como deveria funcionar (exemplo):** com 30 configs e max=50, devolve `g`
  inalterado — o filtro "não morde". Com 200 configs, mantém 50 e descarta 150 — **sem
  log, sem flag, sem coluna** registrando que cortou. O comportamento muda
  silenciosamente conforme o nº de configs cruza 50.
- **O que esperar no código:** `cfg_count = int(grouped_oos["config_label"].nunique())`
  seguido de `if cfg_count <= max_configs: return grouped_oos`.
- **Ref do doc → código:** [`pairwise.py:54-56`](../../../../src/domain/services/gold_builders/pairwise.py#L54).
  ✅ **Confere:** `cfg_count = int(grouped_oos["config_label"].nunique())`; `if
  cfg_count <= max_configs: return grouped_oos`.
- ⚠️ **Ponto de atenção — cap silencioso: o corte é invisível downstream**
  O filtro **não deixa rastro**. Não há `exploratory_only` flag, não há coluna
  `n_configs_original` vs `n_configs_kept`, não há log. Downstream
  (`gold_model_decision_final`, via DM/MCS/win-rate) **não consegue distinguir** um
  resultado vindo do universo completo (≤50 configs, intacto) de um vindo de um
  universo truncado (>50, só o top-50). Pior: o campo `n_configs` gravado na saída
  ([`pairwise.py:324`](../../../../src/domain/services/gold_builders/pairwise.py#L324)
  no DM, [`:376`](../../../../src/domain/services/gold_builders/pairwise.py#L376) no
  MCS) registra o tamanho da **loss_matrix pós-filtro** — ou seja, **no máximo 50** —
  então **nem esse campo revela** que houve poda. Quem lê o artefato vê "n_configs=50"
  e não sabe se o universo original tinha 50 ou 500.

  **Quando importa:** sempre que o artefato for lido como evidência — o leitor não tem
  como saber que o universo foi podado **pelo test loss**. **Quando é tolerável:**
  exploração pura, **desde que rotulada**. O contrato sugerido pelo dossiê endereça
  exatamente isso: ou **(a)** universo pré-declarado, ou **(b)** flag explícita
  `exploratory_only=True` **propagada até** `gold_model_decision_final`.

**3. Aplicado *antes* de DM/MCS/win-rate — 3 callsites, `max_configs=50` por grupo**
- **O que o doc afirma:** "`_select_top_configs_for_pairwise(..., max_configs=50)`
  aplicado antes de DM/MCS/win-rate (linhas 299, 351, 396)".
- **Como deveria funcionar (exemplo):** em cada um dos três builders, a **primeira**
  operação dentro do loop de grupos é `g = _select_top_configs_for_pairwise(g,
  max_configs=50)`, **antes** do `by_ts`/`pivot`/`dropna` que monta a loss_matrix. E é
  **50 por grupo** `(asset, parent_sweep_id, [split_signature], split, horizonte)` — não
  50 global —, pois a chamada está dentro de `for keys, g in df.groupby(group_cols, ...)`.
- **O que esperar no código:** chamada idêntica nos três builders, logo após abrir o
  `for ... in df.groupby(group_cols, dropna=False)`.
- **Ref do doc → código:** [`pairwise.py:299`](../../../../src/domain/services/gold_builders/pairwise.py#L299)
  (DM), [`pairwise.py:351`](../../../../src/domain/services/gold_builders/pairwise.py#L351)
  (MCS), [`pairwise.py:396`](../../../../src/domain/services/gold_builders/pairwise.py#L396)
  (win-rate). ✅ **Confere:** os três contêm exatamente `g =
  _select_top_configs_for_pairwise(g, max_configs=50)` como primeira linha do corpo do
  loop de grupos, antes da construção da loss_matrix.
- ⚠️ **Ponto de atenção — `50` hard-coded em três lugares; o default existe mas é sobrescrito**
  A função expõe `max_configs` como parâmetro (default 50, [`pairwise.py:46`](../../../../src/domain/services/gold_builders/pairwise.py#L46)),
  mas **os três callsites passam `max_configs=50` explicitamente** — então o default
  **nunca** é usado e o valor está **duplicado três vezes**. Para mudar o cap (ou
  desligá-lo) é preciso editar três pontos, não um. E o "50" em si é **arbitrário/não
  documentado**: não há justificativa amarrada a orçamento de cálculo nem a um universo
  pré-registrado. **Quando importa:** numa promoção a confirmatório, o cap precisa ser
  **removido** ou **substituído** por seleção baseada em validação — não re-tunado. Um
  parâmetro central (uma fonte de verdade) tornaria a política auditável; hoje ela está
  espalhada.

**4. Guard de robustez (colunas ausentes / df vazio)**
- **O que o doc afirma:** (implícito) early return defensivo.
- **Como deveria funcionar (exemplo):** se faltar `config_label` ou `squared_error`,
  ou se `g` estiver vazio, devolve `g` inalterado — o filtro **não quebra** o pipeline,
  apenas vira no-op.
- **O que esperar no código:** `if grouped_oos.empty or "config_label" not in cols or
  "squared_error" not in cols: return grouped_oos`.
- **Ref do doc → código:** [`pairwise.py:48-53`](../../../../src/domain/services/gold_builders/pairwise.py#L48).
  ✅ **Confere:** guard que retorna o input quando vazio ou sem as colunas
  `config_label`/`squared_error`. Comportamento benigno (no-op), não mascara erro.

### Cross-check — o que NÃO está corretamente indicado/referenciado

As referências do dossiê para o top-50 (matriz §4 linha 145; dossiê §6; log C.0.1 §13)
**conferem** com o código (li a função inteira 43-64 e os três callsites 299/351/396).
Pontos a registrar:

1. **Nenhuma referência quebrada.** Linhas-âncora corretas: função 43-64 ✅; default
   `max_configs=50` em L46 ✅; passthrough `cfg_count <= max_configs` em L55 ✅; ranking
   `groupby.mean.sort_values.head` em L57-61 ✅; callsites 299/351/396 ✅. O log C.0.1
   (§13, #4) registra "Cross-check auditoria externa: CONCORDA — §3.12 identificou o
   top-50 por test loss como risco de inferência seletiva; confirmado em L57-61" — bate
   com a leitura.

2. **Nuance não capturada pelo dossiê — suporte da seleção ≠ suporte do teste.** O
   ranking (passo 1) calcula a média de `squared_error` de cada config sobre **as
   próprias linhas dela** no grupo (toda a cobertura daquele config). Mas o teste
   DM/MCS real (passo 3) roda sobre a **interseção** `dropna(axis=0, how="any")` das 50
   configs mantidas (só os timestamps em que **todas** as 50 previram). Logo, um config
   pode ser **selecionado** por uma média calculada sobre timestamps que **não
   sobrevivem** à interseção — a loss que **seleciona** não é exatamente a loss que
   **testa**. É secundário ao problema central de snooping, mas significa que "top-50 por
   test loss" é um pouco mais sutil do que a frase do dossiê sugere.

3. **`n_configs` registra o tamanho pós-filtro, não o original.** O campo gravado em
   `gold_dm_pairwise_results`/`gold_mcs_results` (`int(loss_matrix.shape[1])`, L324/L376)
   é **≤ 50** por construção — então o artefato **não** carrega o tamanho do universo
   original. O "cap silencioso" do dossiê é portanto ainda mais literal do que o texto
   indica: nem o único campo numérico de contagem revela a poda.

4. **Determinismo de empate (benigno).** `sort_values(ascending=True)` usa quicksort
   (não estável). Num **empate exato** entre o 50º e o 51º config (mesma média de
   `squared_error`), **qual** entra no corte poderia variar entre execuções. Empates
   exatos em média de floats são praticamente impossíveis, então é benigno — registro
   por completude/determinismo (mesmo espírito da nota de `sort` estável no item #3),
   não como defeito.

5. **Ponto residual central já bem capturado pelo dossiê.** "Seleção pos-test; viola
   hipótese de universo pré-definido; cap silencioso" + referências a data snooping
   (White 2000, Romano-Wolf) descrevem **corretamente** o risco P0. O aprofundamento
   acima detalha o mecanismo (winner's curse) e o "quando cada escolha se aplica".

### Veredito do item #4

🟡 **Ressalvas.** Referências **intactas** e evidência fiel ao código — a função faz
exatamente o que o dossiê afirma (seleção top-50 por `squared_error` no test split,
passthrough abaixo de 50, aplicada antes dos três builders). As ressalvas são (a)
**metodológicas e inerentes ao item** — é o mais grave da família pairwise: **seleção
sobre o desfecho do teste, sem rastro**, contaminando todo artefato pairwise downstream
e, por carona, `gold_model_decision_final`; e (b) **de completude do dossiê** — três
nuances não capturadas: suporte da seleção ≠ suporte do teste, `n_configs` registra o
tamanho pós-filtro (poda literalmente invisível), e o empate de `sort` não-estável
(benigno). Não é defeito de localização; a decisão (remover / mover para validação /
marcar exploratório com flag propagada) é de C.0.2/C.0.3. A Phase B, corretamente,
**não usa** este filtro.

---

