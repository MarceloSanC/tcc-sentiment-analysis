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
- [x] #5 — PICP
- [x] #6 — MPIW
- [ ] #7 — Pinball loss
- [x] #8 — Win-rate gold
- [ ] #9 — prob_up
- [ ] #10 — confidence_calibrated
- [x] #11 — VaR / ES gold
- [ ] #12 — gold_model_decision_final
- [x] #13 — Phase B DM family-6 (referencia)

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

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: **herdar a
  > decisão do item #1, elemento 5** — alimentar o Holm com o `pvalue_one_sided` (na
  > direção pré-registrada do claim) acrescido da correção **HLN**, em vez do
  > `pvalue_two_sided`. Assim o ajuste de multiplicidade corrige um teste já alinhado
  > ao claim, como na Phase B. Não é uma decisão própria do Holm: ele segue o que o
  > item #1 definir para o p-value de entrada.

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

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: **incluir
  > `split_signature` no groupby do Holm** — reusar `_pairwise_group_cols`
  > (`pairwise.py:25-29`) em vez da lista hard-coded de `pairwise.py:191`, alinhando a
  > família ao mesmo grain que o DM já usa para computar os testes. Adicionalmente,
  > **declarar a família a partir do claim** — idealmente apenas os pares
  > TFT-vs-baseline relevantes (à la Phase B), em vez de "todos contra todos" — o que
  > dá ao `pvalue_adj_holm` interpretação de FWER bem-definida e, de quebra, dispensa o
  > pré-filtro top-50 (ver elemento 4: as decisões #1 e #4 se resolvem juntas).

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

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: **remover o
  > filtro top-50** antes do DM/Holm (elimina o viés de inferência seletiva).
  > **Ressalva:** removê-lo mantendo "todos contra todos" faz o `m` explodir
  > (`C(N, 2)`) e torna o Holm conservador demais; por isso a remoção deve vir junto
  > com a **família-do-claim** (elemento 2) — testar só os pares relevantes
  > (TFT vs baselines) mantém o `m` pequeno e justificável. As decisões #1 e #4 se
  > resolvem juntas.

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

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: **unificar
  > os três artefatos num único `pvalue_adj_holm`** — concretamente, **aplicar Holm
  > também em B** (`_build_model_decision_final`, hoje em `pvalue_two_sided` cru). O
  > mecanismo limpo é fazer o rollup (B) e o plot (C) **consumirem a coluna
  > `significant_adj_0_05`/`pvalue_adj_holm` já computada pelo builder do DM** (single
  > source of truth, cálculo no write-time), em vez de cada artefato decidir a própria
  > significância. Assim parquet (A), rollup (B) e plot (C) reportam o mesmo veredito e
  > some a supercontagem de vitórias por p cru em B.

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
C.0.2/C.0.3. Decisões recomendadas registradas nos elementos 1, 2, 4 e 6
(one-sided + HLN herdados do item #1; `split_signature` na família + família-do-claim;
remoção do top-50; unificação em `pvalue_adj_holm` aplicando Holm também em B) —
pendentes de confirmação com a pesquisa acadêmica do paper.

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

## #5 — PICP

### O que é (didático)

**PICP** (*Prediction Interval Coverage Probability*) é simplesmente uma **taxa de
acerto do intervalo**. O modelo, a cada timestamp, emite um intervalo `[q10, q90]`
(o "miolo" de 80% da distribuição prevista). O PICP pergunta: **de todas as vezes
que o modelo deu esse intervalo, em quantas o valor real caiu dentro?** Se o modelo
está bem **calibrado**, o real deveria cair dentro ~**80%** das vezes — nem mais
(intervalo largo/medroso), nem menos (intervalo estreito/confiante demais).

O "truque" é só contar: para cada linha OOS marca **1** se `q10 ≤ y_true ≤ q90`,
**0** caso contrário, e tira a **média**. O número-alvo (`coverage_nominal`) é
**0,80** porque o intervalo vai do percentil 10 ao 90 (`0,90 − 0,10 = 0,80`). PICP
mede **calibração** (o intervalo cumpre o que promete?); sozinho **não** mede
*sharpness* (largura) — isso é o MPIW (item #6), e os dois só fazem sentido **juntos**.

### Elementos

**1. `covered_80` — indicador binário de cobertura por linha**
- **O que o doc afirma:** `covered_80` = 1 se `q10 ≤ y_true ≤ q90`, senão 0, por linha
  ([`quantile.py:178-180`](../../../../src/domain/services/gold_builders/quantile.py#L178)).
- **Como deveria funcionar (exemplo):** intervalo `[q10=95, q90=110]`. Se `y_true=100`
  → **dentro** → `covered_80=1`. Se `y_true=120` → **fora** → `0`. Se `y_true=95`
  exatamente na borda → **dentro** (limites inclusivos `>=`/`<=`) → `1`. PICP será a
  fração desses 1s.
- **O que esperar no código:** `((y_true >= q10_col) & (y_true <= q90_col)).astype(float)`,
  usando as **colunas do contrato sendo processado** (raw **ou** post-guardrail — a
  função roda duas vezes).
- **Ref do doc → código:** [`quantile.py:178-180`](../../../../src/domain/services/gold_builders/quantile.py#L178).
  ✅ **Confere** (li a função `_build_metrics_single_contract` inteira, 102-285):
  `valid["covered_80"] = ((valid["y_true"] >= valid[q10_col]) & (valid["y_true"] <=
  valid[q90_col])).astype(float)`. As bordas são **inclusivas**; `q10_col`/`q90_col`
  vêm de `quantile_columns` (raw em uma chamada, post-guardrail na outra — L305/L381).

**2. Filtro de elegibilidade Cat C (`_prob_eligible`) — quais linhas contam para o PICP**
- **O que o doc afirma:** `covered_80` só conta para linhas `_prob_eligible` (Cat C
  filter: `prediction_mode == quantile` **E** `q10_raw != q90_raw`,
  [`quantile.py:148-171`](../../../../src/domain/services/gold_builders/quantile.py#L148));
  linhas não-elegíveis (point/degeneradas) têm `covered_80` forçado a **NaN**
  ([`quantile.py:210-212`](../../../../src/domain/services/gold_builders/quantile.py#L210)).
- **Como deveria funcionar (exemplo):** uma run de **previsão pontual**
  (`prediction_mode != "quantile"`) não tem intervalo de verdade → é excluída do PICP.
  Uma run de quantis onde o modelo **colapsou** o intervalo (q10 cru == q90 cru, ex.
  ambos = 100) → degenerada → excluída. Só linhas com **intervalo genuíno** entram na
  média. As demais viram NaN e o `.mean()` as ignora.
- **O que esperar no código:** uma máscara `is_quantile_mode & is_non_degenerate`,
  depois um loop que põe NaN em `covered_80` (e nas outras colunas probabilísticas) onde
  a máscara é falsa.
- **Ref do doc → código:** [`quantile.py:169-171`](../../../../src/domain/services/gold_builders/quantile.py#L169)
  (máscara) + [`quantile.py:210-212`](../../../../src/domain/services/gold_builders/quantile.py#L210)
  (NaN). ✅ **Confere:** `is_quantile_mode = valid["prediction_mode"].astype(str)
  .str.lower() == "quantile"` (L169); `prob_eligible_mask = (is_quantile_mode &
  is_non_degenerate).astype(bool)` (L170); e o loop `for col in prob_row_cols: ...
  valid.loc[~prob_eligible_mask, col] = np.nan` (L210-212), com `covered_80` em
  `prob_row_cols` (L207).
- ⚠️ **Ponto de atenção — a degeneração é *filtrada*, mas a elegibilidade é julgada no contrato cru**
  O risco "q10==q90 torna PICP não-informativo" listado no dossiê **já está mitigado**:
  uma linha com intervalo colapsado **não entra** na média do PICP (vira NaN). Sem o
  filtro, um intervalo de largura zero quase nunca cobriria (`y_true` teria de bater
  exatamente no ponto) e arrastaria o PICP para baixo de forma espúria — o filtro evita
  isso. **Mas há uma sutileza que o dossiê não menciona:** a elegibilidade
  (`is_non_degenerate`) é sempre calculada nos **quantis CRUS**
  (`RAW_QUANTILE_COLUMNS`, [`quantile.py:162-166`](../../../../src/domain/services/gold_builders/quantile.py#L162)),
  **mesmo quando o contrato processado é o post-guardrail**. Ou seja: quem decide se uma
  linha conta para o `picp_post_guardrail` é a **degeneração crua** (o que o modelo
  emitiu), não a largura pós-guardrail.

  | Linha | q10/q90 crus | q10/q90 pós-guardrail | Entra no PICP? |
  |---|---|---|---|
  | genuína | 95 / 110 | 95 / 110 | ✅ sim (raw e post) |
  | colapsada na origem | 100 / 100 | 100 / 100 | ❌ não (raw e post) |
  | colapsada na origem, "aberta" pelo guardrail | 100 / 100 | 99 / 101 | ❌ **não** — exclusão pelo cru |

  **Quando importa:** se o guardrail "consertar" muitos intervalos colapsados, o
  `picp_post_guardrail` será calculado sobre um **denominador menor** (só as linhas que
  já eram genuínas no cru), não sobre todas as linhas com intervalo pós-guardrail válido.
  **Por que é defensável:** medir genuinidade pelo que o **modelo** produziu (não pelo
  remendo do guardrail) é a leitura mais honesta de "a run é mesmo probabilística?".
  **Implicação para o TCC:** ao reportar PICP, vale declarar que a base elegível é
  fixada pelos quantis **crus** — duas runs com o mesmo PICP podem ter denominadores
  (n elegível) diferentes. O `n_probabilistic_samples` (elemento 3) é o número a citar
  junto.

**3. Agregação: PICP = média de `covered_80` por grupo**
- **O que o doc afirma:** `picp = ("covered_80", "mean")`
  ([`quantile.py:244`](../../../../src/domain/services/gold_builders/quantile.py#L244)),
  agrupado por `(run_id, asset, feature_set_name, config_signature, split, fold, seed,
  horizon)`.
- **Como deveria funcionar (exemplo):** num grupo com 50 linhas elegíveis, se 40
  cobriram → `PICP = 40/50 = 0,80`. Como as não-elegíveis são NaN, o `.mean()` do pandas
  **as ignora** — o PICP é, portanto, a média **só sobre as linhas elegíveis**. O
  contador `n_probabilistic_samples` (= soma de `_prob_eligible`,
  [`quantile.py:233`](../../../../src/domain/services/gold_builders/quantile.py#L233))
  guarda quantas linhas entraram.
- **O que esperar no código:** `picp=("covered_80", "mean")` dentro do `.agg(...)`, e o
  groupby pelos campos de run/split/horizon.
- **Ref do doc → código:** [`quantile.py:244`](../../../../src/domain/services/gold_builders/quantile.py#L244)
  (agg) + [`quantile.py:214-227`](../../../../src/domain/services/gold_builders/quantile.py#L214)
  (group_cols). ✅ **Confere:** `picp=("covered_80", "mean")` na L244;
  `n_probabilistic_samples=("_prob_eligible", "sum")` na L233; group_cols nas L214-227.
- ⚠️ **Ponto de atenção — PICP é uma estimativa ruidosa: ponto sem intervalo de confiança**
  O PICP é uma **proporção binomial** estimada sobre `n` linhas; ele tem **erro
  amostral** que o gold legacy não reporta. Com cobertura verdadeira 0,80 e `n=50`
  linhas, o desvio-padrão da estimativa é `sqrt(0,80·0,20/50) ≈ 0,057` → uma faixa de
  95% ≈ **[0,69; 0,91]**. Ou seja: um modelo **perfeitamente calibrado** vai exibir PICP
  oscilando entre ~0,69 e ~0,91 só por flutuação amostral. Observar `PICP = 0,74` **não**
  prova descalibração; observar `0,82` **não** prova calibração.

  **Quando importa:** sempre que o número for usado para **declarar** "calibrado / não
  calibrado" — sem uma banda, a leitura vira binária e enganosa, ainda mais com `n`
  pequeno (poucas linhas OOS por run). **Como a Phase B trata:** o H1 confirmatório
  compara o PICP contra **bandas Tier 1/Tier 2** justamente para não tratar o ponto como
  veredicto exato. O gold legacy expõe o `coverage_error` (elemento 4) mas **sem IC** —
  é o ponto, não a faixa.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: reportar o
  > PICP **com IC** (bootstrap ou intervalo binomial de Wilson) e, para qualquer leitura
  > de calibração, compará-lo contra uma **banda** (no espírito dos Tiers da Phase B) em
  > vez do ponto isolado. Citar sempre `n_probabilistic_samples` ao lado.
- ⚠️ **Ponto de atenção — cobertura *incondicional* (marginal) esconde *clustering* de falhas**
  PICP é **cobertura marginal**: conta *quantas* falhas, ignora *quando* elas acontecem.
  Dois modelos com o mesmo PICP de 0,80 podem ser muito diferentes:

  | Modelo | 20% de falhas distribuídas como… | Qualidade real |
  |---|---|---|
  | A | espalhadas, ~independentes no tempo | calibração saudável |
  | B | **agrupadas** num período volátil (ex. todas numa semana de crash) | calibração ruim — falha justo quando o intervalo mais importa |

  O PICP **não distingue** A de B. Quem distingue é o **teste de Christoffersen (1998)**,
  que checa **cobertura condicional** + **independência** das falhas (analisa as
  *runs* de acertos/erros). O gold legacy não tem esse teste — só a contagem marginal.

  **Quando importa:** num claim de "modelo bem calibrado" para fins financeiros, falhas
  agrupadas (modelo B) são exatamente o pior caso (subestima risco em cluster) e ficam
  **invisíveis** no PICP. **Quando é tolerável:** como retrato descritivo rápido da
  cobertura média, o PICP marginal serve.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: para promover
  > o PICP além de descritivo, adicionar o **teste de Christoffersen** (cobertura
  > condicional + independência) como complemento — o PICP marginal vira o "primeiro
  > olhar", não o veredicto de calibração.

**4. `coverage_nominal = 0.80` hard-coded + `coverage_error`**
- **O que o doc afirma:** `coverage_nominal = 0.80` fixo
  ([`quantile.py:260`](../../../../src/domain/services/gold_builders/quantile.py#L260));
  `coverage_error = picp − coverage_nominal`
  ([`quantile.py:261`](../../../../src/domain/services/gold_builders/quantile.py#L261)).
- **Como deveria funcionar (exemplo):** `PICP = 0,83` → `coverage_error = +0,03`
  (cobrindo **a mais** — intervalos um pouco largos). `PICP = 0,72` →
  `coverage_error = −0,08` (cobrindo **a menos** — intervalos estreitos/confiantes
  demais). O sinal diz a **direção** da descalibração; o alvo de referência é 0,80.
- **O que esperar no código:** `agg["coverage_nominal"] = 0.80` (literal) e
  `agg["coverage_error"] = agg["picp"] - agg["coverage_nominal"]`.
- **Ref do doc → código:** [`quantile.py:260-261`](../../../../src/domain/services/gold_builders/quantile.py#L260).
  ✅ **Confere** exatamente: `agg["coverage_nominal"] = 0.80`; `agg["coverage_error"] =
  agg["picp"] - agg["coverage_nominal"]`.
- ⚠️ **Ponto de atenção — o nominal (e os níveis de quantil) são literais, não lidos do contrato**
  O `0,80` é coerente com o intervalo `[q10, q90]` (= `0,90 − 0,10`), **mas é um
  literal**: o código **não deriva** o nominal dos `quantile_levels` reais do sweep.
  E não é só o nominal — **todo o pipeline assume 10/50/90**: os nomes das colunas
  (`RAW_QUANTILE_COLUMNS = quantile_p10/p50/p90`,
  [`quantile.py:32-37`](../../../../src/domain/services/gold_builders/quantile.py#L32)),
  os quantis do pinball (0,1/0,5/0,9, item #7) e agora o `coverage_nominal=0,80`.

  | Cenário do sweep | O que acontece | Risco |
  |---|---|---|
  | quantis = {0,1; 0,5; 0,9} (atual) | nominal 0,80 bate com o intervalo | ✅ nenhum |
  | quantis = {0,05; 0,5; 0,95} **mas colunas ainda nomeadas p10/p90** | PICP de um intervalo de 90% comparado a nominal **0,80** | 🔴 `coverage_error` sistematicamente errado (~+0,10) |
  | quantis = {0,05; 0,95} com **outros nomes de coluna** | o guard de colunas faltantes ([`quantile.py:133-135`](../../../../src/domain/services/gold_builders/quantile.py#L133)) devolve DataFrame vazio | ⚠️ métrica simplesmente some, sem aviso explícito |

  **Quando importa:** qualquer sweep futuro com níveis ≠ {0,1; 0,5; 0,9}, ou uma promoção
  a confirmatório onde o nominal precisa ser **provadamente** o do contrato. **Quando é
  tolerável:** o escopo atual fixa 10/50/90, então hoje o literal está "certo por
  coincidência de configuração". O risco é de **drift silencioso** se a config mudar.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: derivar
  > `coverage_nominal` **dinamicamente** = `q_high − q_low` lido dos `quantile_levels`
  > reais do contrato, em vez do literal `0,80`; e validar por contrato que os níveis
  > batem com os nomes de coluna. Alinha o PICP a sweeps com outros níveis e elimina o
  > risco de `coverage_error` calculado contra o nominal errado.

**5. Saídas e consumo cross-file: `picp_raw`/`picp_post_guardrail` → calibration → decision_final**
- **O que o doc afirma:** coluna `picp` em `gold_prediction_metrics_by_run_split_horizon`
  e agregados; usada por `gold_prediction_calibration`; em `gold_model_decision_final`
  entra **indiretamente** como `mean_picp_post_guardrail` (de
  `gold_prediction_metrics_by_config`), **não** como critério de ordenação nem de
  `academic_decision_ready`.
- **Como deveria funcionar (exemplo):** o `picp` "nu" calculado no agg é **renomeado**
  pelo builder para **`picp_raw`** e **`picp_post_guardrail`** (o padrão dois-contratos),
  então **não existe coluna `picp` nua** no parquet. `gold_prediction_calibration`
  carrega as duas; `gold_prediction_metrics_by_config` faz a média → `mean_picp_raw`/
  `mean_picp_post_guardrail`; o `decision_final` escolhe o contrato **post_guardrail**
  (default) e renomeia para **`mean_picp`**, levado adiante como **coluna passiva** (sem
  efeito na ordenação ou no gate).
- **O que esperar no código:** o rename `{m: f"{m}_raw"}` / `{m: f"{m}_post_guardrail"}`
  no builder run/split/horizon; `picp_raw`/`picp_post_guardrail` mantidas em
  `gold_prediction_calibration`; `mean_picp_post_guardrail` mapeado para `mean_picp` no
  `decision_final`; ordenação por `rank_rmse, rank_mae`; `academic_decision_ready` sem
  PICP.
- **Ref do doc → código:** rename em
  [`quantile.py:335-337`](../../../../src/domain/services/gold_builders/quantile.py#L335)
  e [`:391-393`](../../../../src/domain/services/gold_builders/quantile.py#L391);
  `picp_raw`/`picp_post_guardrail` em
  [`descriptive.py:394`](../../../../src/domain/services/gold_builders/descriptive.py#L394)
  e [`:403`](../../../../src/domain/services/gold_builders/descriptive.py#L403)
  (`gold_prediction_calibration`); média em
  [`quantile.py:651-657`](../../../../src/domain/services/gold_builders/quantile.py#L651)
  (`mean_<m>`); `mean_picp_post_guardrail → mean_picp` em
  [`confidence.py:497`](../../../../src/domain/services/gold_builders/confidence.py#L497)
  + rename em [`:521`](../../../../src/domain/services/gold_builders/confidence.py#L521);
  ordenação em [`confidence.py:815-818`](../../../../src/domain/services/gold_builders/confidence.py#L815);
  `academic_decision_ready` em [`confidence.py:809-813`](../../../../src/domain/services/gold_builders/confidence.py#L809).
  ✅ **Confere:** o builder substitui `picp` por `picp_raw`/`picp_post_guardrail`
  (não há alias nu — `_set_alias_post_primary` só roda para `confidence_calibrated`,
  `prob_up`, `prob_down`, [`quantile.py:399-400`](../../../../src/domain/services/gold_builders/quantile.py#L399));
  `gold_prediction_calibration` carrega as duas (L394/L403); o `decision_final` mapeia
  `mean_picp_{post_guardrail}` → `mean_picp` (L497) e a ordenação é por `rank_rmse,
  rank_mae` (L815-818), com `academic_decision_ready` dependendo **só** de
  `pairwise_ready_dm`, `pairwise_ready_mcs`, `target_exact_alignment` (L809-813) — **sem
  PICP**.
- ⚠️ **Ponto de atenção — "picp" no doc ≠ `picp_raw`/`picp_post_guardrail` no parquet (e `mean_picp` no final)**
  O dossiê fala em "coluna `picp`", mas **nenhuma tabela gold persiste uma coluna `picp`
  nua**: ela aparece sempre **sufixada pelo contrato** (`picp_raw`, `picp_post_guardrail`)
  e, no `gold_model_decision_final`, como **`mean_picp`** (com `primary_quantile_contract`
  registrando qual contrato foi escolhido — default **post_guardrail**,
  [`confidence.py:472`](../../../../src/domain/services/gold_builders/confidence.py#L472)).
  **Quando importa:** quem consultar o parquet por `picp` não acha nada — precisa saber o
  sufixo; e ao ler "PICP" num relatório, é obrigatório perguntar **raw ou
  post-guardrail?** (o número de destaque do artefato final é o **post-guardrail**).
  **Quando é tolerável:** desde que o contrato seja sempre declarado junto. É imprecisão
  de **nomenclatura no dossiê**, não referência quebrada — o código está coerente.

### Cross-check — o que NÃO está corretamente indicado/referenciado

Todas as referências de "Implementação atual localizada" do dossiê de PICP **conferem**
com o código (li a função `_build_metrics_single_contract` inteira, 102-285, mais o
builder run/split/horizon 288-404, `gold_prediction_calibration` em descriptive.py
363-419, o agg por config em quantile.py 615-667 e o rollup em confidence.py 463-818).
Pontos a registrar:

1. **Nenhuma referência quebrada.** Linhas-âncora **exatas**: `covered_80` em L178-180 ✅;
   Cat C filter em L148-171 ✅; máscara NaN em L210-212 ✅; `picp=("covered_80","mean")`
   em L244 ✅; `coverage_nominal = 0.80` em L260 ✅; `coverage_error` em L261 ✅.

2. **"Coluna `picp`" é imprecisão de nomenclatura.** O parquet
   `gold_prediction_metrics_by_run_split_horizon` **não tem** coluna `picp` nua — só
   `picp_raw` e `picp_post_guardrail` (rename em quantile.py L335-337/L391-393); no
   `gold_model_decision_final` a coluna é `mean_picp`. Quem busca `picp` no artefato não
   encontra. Registrado no elemento 5 (completude, não ref quebrada).

3. **"`covered_80` só é computado para linhas `_prob_eligible`" descreve o efeito, não o
   mecanismo.** Na verdade `covered_80` é calculado para **todas** as linhas válidas
   (L178-180) e **depois** mascarado a NaN nas não-elegíveis (L210-212) — "computa e
   mascara", não "computa só para elegíveis". O efeito líquido é idêntico (só elegíveis
   entram na média), e o segundo bullet de evidência do dossiê já cita o mascaramento, então
   o conjunto é autoconsistente; registro a mecânica exata por precisão.

4. **Sutileza não capturada — elegibilidade julgada no contrato CRU mesmo para o PICP
   post-guardrail.** `is_non_degenerate` usa sempre `RAW_QUANTILE_COLUMNS` (L162-166),
   independentemente de o contrato processado ser raw ou post-guardrail. O dossiê diz que
   "degeneração torna PICP não-informativo", mas o código na verdade **filtra** a
   degeneração — com a assimetria raw-vs-post detalhada no elemento 2. É o achado mais
   informativo deste item.

5. **Afirmação de "Uso atual" confirmada integralmente.** PICP entra no
   `gold_model_decision_final` apenas como coluna `mean_picp` (de
   `mean_picp_post_guardrail`); **não** é chave de ordenação (ordena por `rank_rmse,
   rank_mae`, L815-818) **nem** compõe `academic_decision_ready` (só DM/MCS/alinhamento,
   L809-813). Bate com o que o dossiê declara.

6. **Pontos residuais metodológicos centrais já capturados pelo dossiê.**
   `coverage_nominal` hard-coded, cobertura incondicional escondendo clustering, e
   degeneração — os três constam dos "Riscos conhecidos". Os aprofundamentos acima
   detalham o mecanismo (ruído binomial sem IC, Christoffersen, drift de nível de quantil)
   e o "quando cada escolha se aplica".

### Veredito do item #5

🟡 **Ressalvas.** Referências **intactas** e evidência fiel ao código — o PICP é
calculado exatamente como o dossiê afirma (indicador `covered_80` inclusivo, filtro
Cat C de elegibilidade, média por grupo, nominal 0,80 e `coverage_error`). As ressalvas
são (a) **metodológicas** — nominal e níveis de quantil **hard-coded** (drift silencioso
se a config mudar), cobertura **marginal** sem teste de Christoffersen (clustering de
falhas invisível) e **ponto sem IC** (estimativa binomial ruidosa, ao contrário das
bandas Tier da Phase B); e (b) **de precisão/completude do dossiê** — não existe coluna
`picp` nua (só `picp_raw`/`picp_post_guardrail`, e `mean_picp` no final), o `covered_80`
é "computado e mascarado" (não "só para elegíveis"), e a elegibilidade é julgada nos
quantis **crus** mesmo para o `picp_post_guardrail` (assimetria não mencionada). Não há
defeito de localização; as decisões (nominal dinâmico, Christoffersen, IC/banda) são de
C.0.2/C.0.3. Decisões recomendadas registradas nos elementos 3 e 4 — pendentes de
confirmação com a pesquisa acadêmica do paper.

---

## #6 — MPIW

### O que é (didático)

**MPIW** (*Mean Prediction Interval Width* — largura média do intervalo preditivo) é
simplesmente a **largura média** do intervalo `[q10, q90]` que o modelo emite. Para
cada linha OOS calcula `largura = q90 − q10`; o MPIW é a **média** dessas larguras. Ele
mede **sharpness** (o quão "apertado"/confiante é o intervalo): MPIW menor = intervalos
mais estreitos.

O "truque" — e a armadilha — é que **sharpness sozinha não diz nada sobre honestidade**.
Um modelo pode reduzir o MPIW à vontade simplesmente **mentindo** (emitindo intervalos
absurdamente estreitos). Quem checa se o intervalo cumpre o que promete é o **PICP**
(item #5). Por isso MPIW e PICP **só fazem sentido juntos**: largura sem cobertura
premia o modelo otimista demais; cobertura sem largura premia o modelo medroso (intervalo
gigante que sempre acerta). O par largura×cobertura é o que um *interval score* (Winkler)
combina num número só.

### Elementos

**1. `pred_interval_width` — largura por linha (= q90 − q10)**
- **O que o doc afirma:** `pred_interval_width = q90_col − q10_col` por linha
  ([`quantile.py:177`](../../../../src/domain/services/gold_builders/quantile.py#L177)).
- **Como deveria funcionar (exemplo):** intervalo `[q10=95, q90=110]` → largura `15`.
  Um modelo mais "afiado" que devolve `[q10=99, q90=101]` → largura `2`. Quanto menor a
  largura, mais confiante o modelo se diz — mas só o PICP dirá se essa confiança é
  justificada.
- **O que esperar no código:** `valid["pred_interval_width"] = valid[q90_col] -
  valid[q10_col]`, usando as **colunas do contrato sendo processado** (raw **ou**
  post-guardrail — a função `_build_metrics_single_contract` roda duas vezes, L305/L381).
- **Ref do doc → código:** [`quantile.py:177`](../../../../src/domain/services/gold_builders/quantile.py#L177).
  ✅ **Confere** (li a função inteira, 102-285): `valid["pred_interval_width"] =
  valid[q90_col] - valid[q10_col]`. É uma subtração simples, **sem `abs()`** e **sem
  clip** nesta linha.
- ⚠️ **Ponto de atenção — quantis cruzados passam o filtro → largura pode ser negativa (no contrato cru)**
  A largura é `q90 − q10` **sem valor absoluto**. Se os quantis **cruzam** (`q10 > q90`,
  patologia comum em modelos de quantis sem monotonicidade imposta), a largura fica
  **negativa**. E o filtro de elegibilidade (elemento 2) **não pega isso**: ele exige
  apenas `q10_raw != q90_raw` ([`quantile.py:166`](../../../../src/domain/services/gold_builders/quantile.py#L166)),
  ou seja, exclui só a **igualdade exata** (degeneração), **não** o cruzamento. Uma linha
  com `q10_raw=110, q90_raw=95` tem `q10 ≠ q90` → é elegível → entra na média com largura
  **−15**.

  Que o cruzamento de fato ocorre nos quantis crus está provado pelo próprio código: o
  `QuantileGuardrailAuditGoldBuilder` computa `negative_width_before` justamente como
  `(quantile_p90 − quantile_p10) < 0`
  ([`quantile.py:515-517`](../../../../src/domain/services/gold_builders/quantile.py#L515)) —
  essa métrica só existe porque larguras negativas aparecem **antes** do guardrail.

  | Contrato | Cruzamento (q10>q90)? | O que entra no MPIW |
  |---|---|---|
  | **raw** (`mpiw_raw`) | possível | largura **negativa** contamina a média (puxa para baixo) |
  | **post_guardrail** (`mpiw_post_guardrail`) | o guardrail impõe monotonicidade | larguras ≥ 0 (cruzamento removido) |

  **Quando importa:** ler `mpiw_raw` como "sharpness" é enganoso se houver cruzamento — um
  MPIW artificialmente baixo (ou negativo) reflete **patologia**, não intervalos
  apertados. **Quando é tolerável:** o número de destaque do artefato final é o
  **post_guardrail** (ver elemento 4), onde o cruzamento já foi corrigido — então o risco
  vive no contrato cru. Mesmo assim, o `mpiw_raw` persistido carrega a distorção.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: para qualquer
  > leitura de sharpness, usar o **`mpiw_post_guardrail`** (cruzamento já corrigido) e/ou
  > **excluir da elegibilidade** as linhas com largura negativa (cruzamento), não só as de
  > largura zero. Reportar a `crossing_before_rate` (já calculada no guardrail audit) ao
  > lado do `mpiw_raw`.

**2. Filtro Cat C de elegibilidade — degeneração mascarada a NaN, julgada nos quantis CRUS**
- **O que o doc afirma:** sujeito ao **mesmo Cat C filter** do PICP; linhas não-elegíveis
  têm `pred_interval_width` forçado a NaN
  ([`quantile.py:210-212`](../../../../src/domain/services/gold_builders/quantile.py#L210)).
- **Como deveria funcionar (exemplo):** uma run de **previsão pontual**
  (`prediction_mode != "quantile"`) não tem intervalo → excluída. Uma run de quantis com
  intervalo **colapsado na origem** (q10_raw == q90_raw, ex. ambos = 100) → degenerada →
  excluída. Só linhas com intervalo genuíno entram na média; as demais viram NaN e o
  `.mean()` as ignora.
- **O que esperar no código:** a mesma máscara `prob_eligible_mask = is_quantile_mode &
  is_non_degenerate` do PICP, com `pred_interval_width` na lista `prob_row_cols` que é
  zerada a NaN.
- **Ref do doc → código:** [`quantile.py:208`](../../../../src/domain/services/gold_builders/quantile.py#L208)
  (`pred_interval_width` em `prob_row_cols`) + [`quantile.py:210-212`](../../../../src/domain/services/gold_builders/quantile.py#L210)
  (NaN) + [`quantile.py:162-171`](../../../../src/domain/services/gold_builders/quantile.py#L162)
  (elegibilidade). ✅ **Confere:** `pred_interval_width` é o 7º item de `prob_row_cols`
  (L208) e o loop `valid.loc[~prob_eligible_mask, col] = np.nan` (L210-212) o mascara; a
  elegibilidade vem de `is_non_degenerate = (p10_raw != p90_raw) & ...` (L166) e
  `is_quantile_mode` (L169).
- ⚠️ **Ponto de atenção — "MPIW=0 em degenerados" está mitigado; mas a elegibilidade é julgada no CRU mesmo para o post-guardrail**
  O risco listado no dossiê ("MPIW=0 em quantis degenerados") está **majoritariamente
  mitigado**: uma linha com intervalo colapsado **não entra** na média (vira NaN). Se uma
  run for **toda** degenerada, `n_probabilistic_samples = 0` e o MPIW vira **NaN** (não
  `0`) — então a frase do dossiê **superestima** o risco não-tratado: o filtro já o
  contém, e o resultado de uma degeneração total é NaN, não um zero enganoso.

  **Mas há a mesma sutileza do PICP (elemento 2 do item #5), e aqui ela é ainda mais
  visível:** a elegibilidade (`is_non_degenerate`) é **sempre** calculada nos
  **quantis CRUS** (`RAW_QUANTILE_COLUMNS`, [`quantile.py:162-166`](../../../../src/domain/services/gold_builders/quantile.py#L162)),
  **mesmo quando o contrato processado é o post-guardrail**. A largura, porém, é a do
  contrato corrente. Daí uma **assimetria** específica do MPIW:

  | Linha | q10/q90 crus | q10/q90 pós-guardrail | Entra no `mpiw_post_guardrail`? | Largura que entra |
  |---|---|---|---|---|
  | genuína | 95 / 110 | 95 / 110 | ✅ sim | 15 |
  | colapsada na origem | 100 / 100 | 100 / 100 | ❌ não (cru degenerado) | — |
  | genuína no cru, **colapsada pelo guardrail** | 95 / 110 | 100 / 100 | ✅ **sim** (cru não é degenerado) | **0** |

  **Quando importa:** se o guardrail colapsar intervalos genuínos (clampando q10_post ==
  q90_post para forçar monotonicidade), essas linhas **continuam elegíveis** (porque o cru
  era genuíno) e entram no `mpiw_post_guardrail` com **largura 0**, **puxando-o para
  baixo**. É a **assimetria inversa** à do PICP: lá o cru-degenerado é excluído mesmo
  quando o guardrail "abre" o intervalo; aqui o cru-genuíno é incluído mesmo quando o
  guardrail "fecha" o intervalo. **Por que é defensável:** medir genuinidade pelo que o
  **modelo** emitiu (cru) é coerente entre PICP e MPIW; mas a consequência sobre a *média
  de largura* é que zeros induzidos pelo guardrail entram no denominador.

**3. Agregação: MPIW = média de `pred_interval_width` (e a coluna gêmea idêntica)**
- **O que o doc afirma:** `mpiw = ("pred_interval_width", "mean")`
  ([`quantile.py:245`](../../../../src/domain/services/gold_builders/quantile.py#L245));
  **também emitido** `pred_interval_width = ("pred_interval_width", "mean")` como coluna
  separada ([`quantile.py:246`](../../../../src/domain/services/gold_builders/quantile.py#L246)).
- **Como deveria funcionar (exemplo):** num grupo com 3 linhas elegíveis de larguras
  {10, 14, 12} → `MPIW = 36/3 = 12`. Como as não-elegíveis são NaN, o `.mean()` do pandas
  **as ignora** — o MPIW é a média **só sobre as elegíveis**, agrupado por `(run_id,
  asset, feature_set_name, config_signature, split, fold, seed, horizon)`.
- **O que esperar no código:** `mpiw=("pred_interval_width", "mean")` dentro do `.agg(...)`,
  e o groupby pelos campos de run/split/horizon.
- **Ref do doc → código:** [`quantile.py:245`](../../../../src/domain/services/gold_builders/quantile.py#L245)
  (`mpiw`) + [`quantile.py:246`](../../../../src/domain/services/gold_builders/quantile.py#L246)
  (`pred_interval_width`) + [`quantile.py:214-227`](../../../../src/domain/services/gold_builders/quantile.py#L214)
  (group_cols). ✅ **Confere** exatamente: ambas as linhas agregam a **mesma** coluna-fonte
  `pred_interval_width` com `mean`.
- ⚠️ **Ponto de atenção — `mpiw` e `pred_interval_width` são numericamente IDÊNTICAS**
  As duas saídas (L245 e L246) aplicam `mean` sobre a **mesma** coluna por linha
  (`pred_interval_width`), logo produzem **exatamente o mesmo número**. Após o rename do
  builder, o parquet carrega **quatro** colunas em **dois pares idênticos**: `mpiw_raw` ==
  `pred_interval_width_raw` e `mpiw_post_guardrail` == `pred_interval_width_post_guardrail`.
  Não é erro — `pred_interval_width` existe porque é a coluna que **alimenta o
  `confidence_calibrated`** (elemento 4); `mpiw` é o **nome semântico** da métrica. **Quando
  importa:** um leitor do parquet pode supor que são **duas medidas diferentes** (ex.: uma
  normalizada, outra crua) e ler significado onde não há — são byte-a-byte iguais.
- ⚠️ **Ponto de atenção — MPIW sozinho não mede calibração (sharpness sem cobertura)**
  Este é o ponto metodológico central do item. MPIW responde "quão estreito?", **nunca**
  "quão honesto?". Dois modelos com larguras muito diferentes podem ser igualmente
  (des)calibrados, e o "melhor MPIW" pode ser o **pior** modelo:

  | Modelo | MPIW | PICP | Leitura correta |
  |---|---|---|---|
  | A | 5 (estreito) | 0,55 | sharp mas **descalibrado** — intervalo otimista demais, não cobre |
  | B | 20 (largo) | 0,80 | bem calibrado, porém **pouco informativo** (intervalo largo) |
  | C | 8 | 0,80 | calibrado **e** razoavelmente sharp ← o desejável |

  Olhar só o MPIW elegeria **A** ("menor largura"), que é o pior. Por isso a literatura
  combina os dois num **interval score / Winkler score** (penaliza largura **e** penaliza
  cada vez que o real cai fora), e o skeleton fixa: "MPIW só entra junto a PICP; nunca
  isolado". **Quando importa:** sempre que MPIW for usado para **comparar ou escolher**
  modelos. **Quando é tolerável:** como descrição de sharpness **reportada lado a lado com
  o PICP** do mesmo grupo. **Como a Phase B trata:** reporta MPIW junto ao PICP, nunca
  como critério isolado.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: nunca reportar
  > MPIW isolado; acoplá-lo **sempre** ao PICP do mesmo grupo e, para qualquer uso
  > comparativo/seletivo, preferir um **interval score (Winkler)** que combine largura e
  > cobertura num único número próprio (*proper*).

**4. Saídas e consumo cross-file: `mpiw_raw`/`mpiw_post_guardrail` → calibration → decision_final (`mean_mpiw`)**
- **O que o doc afirma:** coluna `mpiw` em `gold_prediction_metrics_*` e
  `gold_prediction_calibration`.
- **Como deveria funcionar (exemplo):** como toda métrica probabilística, o `mpiw` "nu" do
  agg é **renomeado** pelo builder para `mpiw_raw` e `mpiw_post_guardrail` (padrão
  dois-contratos) — **não existe coluna `mpiw` nua** no parquet. A média por config gera
  `mean_mpiw_raw`/`mean_mpiw_post_guardrail`, e o `decision_final` escolhe o contrato
  **post_guardrail** (default) renomeando para **`mean_mpiw`**.
- **O que esperar no código:** rename `{m: f"{m}_raw"}`/`{m: f"{m}_post_guardrail"}` no
  builder; `mpiw_raw`/`mpiw_post_guardrail` em `gold_prediction_calibration`;
  `mean_mpiw_{contrato}` → `mean_mpiw` no `decision_final`.
- **Ref do doc → código:** rename em
  [`quantile.py:335-337`](../../../../src/domain/services/gold_builders/quantile.py#L335)
  e [`:391-393`](../../../../src/domain/services/gold_builders/quantile.py#L391);
  `mpiw_raw`/`mpiw_post_guardrail` (e os gêmeos `pred_interval_width_*`) em
  [`descriptive.py:395-396`](../../../../src/domain/services/gold_builders/descriptive.py#L395)
  e [`:404-405`](../../../../src/domain/services/gold_builders/descriptive.py#L404)
  (`gold_prediction_calibration`); `mean_mpiw_{post_guardrail} → mean_mpiw` em
  [`confidence.py:498`](../../../../src/domain/services/gold_builders/confidence.py#L498);
  e `gap_mpiw_{post_guardrail}_test_minus_val → gap_mpiw_test_minus_val` em
  [`confidence.py:565`](../../../../src/domain/services/gold_builders/confidence.py#L565).
  ✅ **Confere:** `gold_prediction_calibration` carrega `mpiw_raw`/`mpiw_post_guardrail`
  (L395/L404); o `decision_final` mapeia `mean_mpiw_post_guardrail → mean_mpiw` (L498) e
  expõe também o gap test−val (L565). A ordenação final continua por `rank_rmse,
  rank_mae` (ver item #5, [`confidence.py:815-818`](../../../../src/domain/services/gold_builders/confidence.py#L815)),
  então `mean_mpiw` é **coluna passiva** (não é chave de ordenação nem entra no
  `academic_decision_ready`).
- ⚠️ **Ponto de atenção — o dossiê omite que MPIW chega ao `gold_model_decision_final` e alimenta o `confidence_calibrated`**
  O campo "Uso atual no projeto" do dossiê lista apenas `gold_prediction_metrics_*` e
  `gold_prediction_calibration`. Na prática o MPIW vai **mais longe**:
  - chega ao **`gold_model_decision_final`** como `mean_mpiw` (e como `gap_mpiw_test_minus_val`),
    [`confidence.py:498`](../../../../src/domain/services/gold_builders/confidence.py#L498)/[`:565`](../../../../src/domain/services/gold_builders/confidence.py#L565)
    — coluna passiva, mas presente no artefato final;
  - a **mesma** largura por linha alimenta o **`confidence_calibrated`** via `width_term =
    1 / (1 + pred_interval_width.clip(lower=0))`
    ([`quantile.py:266`](../../../../src/domain/services/gold_builders/quantile.py#L266) —
    ver item #10). Note o `clip(lower=0)`: lá a largura negativa (cruzamento, elemento 1)
    é **zerada**, então a patologia some no `confidence_calibrated` mas **permanece** no
    `mpiw_raw`;
  - aparece ainda no `gold_quantile_guardrail_audit` como `mpiw_before`/`mpiw_after`
    ([`quantile.py:469`](../../../../src/domain/services/gold_builders/quantile.py#L469)).

  **Quando importa:** ao rastrear "onde o MPIW influencia decisões", a leitura do dossiê
  subdimensiona o alcance — `mean_mpiw` está no artefato final e a largura compõe um score
  heurístico (`confidence_calibrated`).
- ⚠️ **Ponto de atenção — MPIW está em unidades absolutas; comparar cross-asset exige normalização**
  MPIW herda a **escala do alvo**: a largura de um intervalo para um ativo cotado em
  dezenas de milhares (ex.: BTC) é numericamente enorme perto da de um ativo em unidades
  pequenas ou de uma série de retornos. Comparar `mpiw` **entre ativos** sem normalizar é
  comparar laranjas com maçãs.

  | Comparação | MPIW cru serve? |
  |---|---|
  | mesmo ativo, dois modelos | ✅ sim — escala comum |
  | ativos diferentes / escalas diferentes | 🔴 não — precisa normalizar (ex.: dividir pela escala do alvo, ou usar largura relativa) |

  **Quando importa:** qualquer ranking ou narrativa que junte MPIW de **ativos
  diferentes**. **Quando é tolerável:** comparações **dentro do mesmo ativo/escala**.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: disponibilizar
  > um **MPIW normalizado** (ex.: pela escala/volatilidade do alvo, ou largura relativa ao
  > nível previsto) para qualquer comparação cross-asset; manter o MPIW absoluto apenas
  > para comparações intra-ativo.

### Cross-check — o que NÃO está corretamente indicado/referenciado

Todas as referências de "Implementação atual localizada" do dossiê de MPIW **conferem**
com o código (li a função `_build_metrics_single_contract` inteira, 102-285, mais o
builder run/split/horizon 288-404, `gold_prediction_calibration` em descriptive.py
363-419 e o rollup em confidence.py 463-572). Pontos a registrar:

1. **Nenhuma referência quebrada.** Linhas-âncora **exatas**: `pred_interval_width =
   q90 − q10` em L177 ✅; máscara Cat C em L208 + L210-212 ✅; `mpiw=("pred_interval_width",
   "mean")` em L245 ✅; `pred_interval_width=(...,"mean")` (coluna gêmea) em L246 ✅.

2. **"Uso atual" subdimensiona o alcance do MPIW.** O dossiê lista só
   `gold_prediction_metrics_*` e `gold_prediction_calibration`, mas o MPIW também: (a)
   chega ao **`gold_model_decision_final`** como `mean_mpiw` e `gap_mpiw_test_minus_val`
   ([`confidence.py:498`](../../../../src/domain/services/gold_builders/confidence.py#L498)/[`:565`](../../../../src/domain/services/gold_builders/confidence.py#L565));
   (b) a largura por linha alimenta o **`confidence_calibrated`**
   ([`quantile.py:266`](../../../../src/domain/services/gold_builders/quantile.py#L266));
   (c) aparece no `gold_quantile_guardrail_audit` como `mpiw_before`/`mpiw_after`
   ([`quantile.py:469`](../../../../src/domain/services/gold_builders/quantile.py#L469)).
   É coluna passiva no artefato final (ordenação por `rank_rmse, rank_mae`), mas presente.

3. **`mpiw` == `pred_interval_width` (colunas gêmeas idênticas) não é mencionado.** As duas
   saídas (L245/L246) agregam a mesma fonte com `mean` → mesmo número; o parquet carrega
   dois pares byte-a-byte iguais (`mpiw_*` e `pred_interval_width_*`). Registrado no
   elemento 3.

4. **Risco "MPIW=0 em degenerados" superestimado.** O Cat C filter **já mascara** linhas
   cruamente degeneradas (q10_raw == q90_raw) a NaN — o resultado de uma run toda
   degenerada é **NaN**, não `0`. O caminho residual de largura-zero é o **colapso pelo
   guardrail** de linhas cruamente genuínas (elemento 2), não a degeneração crua que o
   dossiê descreve.

5. **Cruzamento de quantis (largura negativa) não capturado pelo dossiê.** O filtro exclui
   só a igualdade exata, não o cruzamento (`q10 > q90`); larguras negativas entram no
   `mpiw_raw` (elemento 1), comprovado pela existência da métrica `negative_width_before`
   no guardrail audit ([`quantile.py:515-517`](../../../../src/domain/services/gold_builders/quantile.py#L515)).

6. **Pontos residuais metodológicos centrais já capturados pelo dossiê.** "Comparar
   largura sem cobertura", "MPIW=0 em degenerados" e "sem normalização cross-asset" constam
   dos "Riscos conhecidos". Os aprofundamentos acima detalham o mecanismo (sharpness sem
   cobertura via tabela A/B/C, assimetria de elegibilidade raw-vs-post, unidades absolutas)
   e o "quando cada escolha se aplica".

### Veredito do item #6

🟡 **Ressalvas.** Referências **intactas** e evidência fiel ao código — o MPIW é
calculado exatamente como o dossiê afirma (largura `q90 − q10` por linha, filtro Cat C,
média por grupo, coluna gêmea `pred_interval_width`). As ressalvas são (a)
**metodológicas** — MPIW isolado **não mede calibração** (sharpness sem cobertura; só faz
sentido com PICP / interval score), unidades absolutas exigem normalização cross-asset, e
o contrato cru pode conter **largura negativa** (cruzamento) que o filtro não pega; e (b)
**de precisão/completude do dossiê** — "Uso atual" subdimensiona o alcance (MPIW chega ao
`gold_model_decision_final` como `mean_mpiw`, alimenta o `confidence_calibrated` e aparece
no guardrail audit), `mpiw` e `pred_interval_width` são colunas **idênticas**, o risco
"MPIW=0 em degenerados" está **superestimado** (o filtro já o mitiga → NaN, não 0) e a
assimetria de elegibilidade julgada nos quantis **crus** mesmo para o `mpiw_post_guardrail`
não é mencionada. Não há defeito de localização; as decisões (interval/Winkler score, MPIW
normalizado, tratamento de cruzamento) são de C.0.2/C.0.3. Decisões recomendadas
registradas nos elementos 1, 3 e 4 — pendentes de confirmação com a pesquisa acadêmica do
paper.

---

## #13 — Phase B DM family-6 (referência)

> **Natureza deste item — leia primeiro.** Este é um item de **referência, não de
> auditoria**. O sidecar Phase B DM family-6 já está **pré-registrado e fechado**
> (`PROMOTED_CONFIRMATORY`, escopo Phase B), e o escopo de C.0 **não reabre** sua
> metodologia. Esta seção faz só duas coisas: (1) confirma que os **três caminhos de
> código** existem (a tarefa literal da C.0.1 para este item) e (2) confirma que cada
> **característica que o distingue do gold legacy** — pinball post-guardrail, lag `h−1`,
> HLN, one-sided, família-6 declarada, sem top-50, dedup por `target_timestamp_utc` — está
> de fato presente no código. **Não** se avalia se essas escolhas estão "certas": Phase B
> está fechada. O objetivo é deixar a âncora de contraste à prova de drift, para que
> ninguém confunda `gold_dm_pairwise_results` (gold legacy, item #1) com
> `phase_b_dm_family_6.parquet` (este sidecar) durante o hardening.

### O que é (didático)

O **Phase B DM family-6** é o artefato **confirmatório** que respondeu às hipóteses
**H2a/H2b** ("o TFT é melhor que os baselines em perda probabilística"). É a *mesma
maquinaria* do DM do item #1 — diferença de perdas por período, variância robusta a
autocorrelação, estatística e p-value — mas montada do jeito **defensável**: perda alinhada
ao claim (pinball), teste **direcional** (one-sided), correção de **amostra pequena** (HLN),
lag de previsão teórico (`h−1`), **família declarada antes** de ver os dados (6 testes) e
**sem** filtro top-50. Por isso ele aparece na §6: como **âncora de contraste** que mostra,
elemento a elemento, o que o gold legacy faz de diferente — e como **lembrete** de não
sobrescrever nem confundir os dois durante C.0.

### Elementos

**1. Os três caminhos de código existem e contêm os símbolos do pipeline confirmatório**
- **O que o doc afirma:** a implementação vive em [`dm_tft_vs_baseline.py`](../../../../src/domain/services/dm_tft_vs_baseline.py), [`holm_family_6.py`](../../../../src/domain/services/holm_family_6.py) e [`compute_phase_b_tier_metrics_use_case.py`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py) (skeleton §6 "caminhos confirmados por `ls` em 2026-05-28").
- **Como deveria funcionar (exemplo):** uma referência só é sólida se os arquivos existem **e** contêm as funções que o fluxo confirmatório realmente chama. Confirmar por `ls` prova só o arquivo; confirmar por leitura prova que `compute_dm_family`, `apply_holm_one_sided` e o use case que os orquestra estão lá.
- **O que esperar no código:** `compute_dm_family`/`compute_dm_hln_hac` no service de DM; `apply_holm_one_sided` no service de Holm; o use case importando ambos e escrevendo o sidecar.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:277`](../../../../src/domain/services/dm_tft_vs_baseline.py#L277) (`compute_dm_family`) e [`:224`](../../../../src/domain/services/dm_tft_vs_baseline.py#L224) (`compute_dm_hln_hac`); [`holm_family_6.py:7`](../../../../src/domain/services/holm_family_6.py#L7) (`apply_holm_one_sided`); [`compute_phase_b_tier_metrics_use_case.py:11-16`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L11) (imports). ✅ **Confere:** li os três arquivos inteiros — todos existem e expõem exatamente esses símbolos. Confirmação por **leitura direta** (mais forte que o `ls` do skeleton).

**2. Perda = pinball post-guardrail (gold legacy usa squared_error)**
- **O que o doc afirma:** §7 (tabela de contraste) — loss do sidecar = `pinball_loss_post_guardrail`; do gold = `squared_error`.
- **Como deveria funcionar (exemplo):** com `y_true=100`, `q10=95, q50=101, q90=108`: pinball_q50 = `max(0.5·(100−101), −0.5·(100−101)) = max(−0.5, 0.5) = 0.5`; pinball_q10 = `max(0.1·5, −0.9·5) = 0.5`; pinball_q90 = `max(0.9·(−8), −0.1·(−8)) = 0.8`; perda = `(0.5+0.5+0.8)/3 ≈ 0.6`. Mede a **qualidade da distribuição** (q10/q50/q90), não o erro do ponto. O gold legacy, para a mesma linha, usaria `(y_pred − 100)²` — pergunta **pontual**, não probabilística (ver item #1, elemento 1).
- **O que esperar no código:** uma função de pinball média sobre as três colunas `quantile_p{10,50,90}_post_guardrail`.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:49-54`](../../../../src/domain/services/dm_tft_vs_baseline.py#L49) (`mean_pinball_post_guardrail`) e [`:44-46`](../../../../src/domain/services/dm_tft_vs_baseline.py#L44) (`_pinball`). ✅ **Confere:** `(_pinball(y,q10,0.1)+_pinball(y,q50,0.5)+_pinball(y,q90,0.9))/3.0` sobre as colunas `quantile_p10/p50/p90_post_guardrail`; alimentada como `df["loss"]` em [`:147`](../../../../src/domain/services/dm_tft_vs_baseline.py#L147).

**3. HAC Newey-West com lag = `max(h−1, 1)` (gold usa `min(max(1, n^{1/3}), 10)`)**
- **O que o doc afirma:** §7 — HAC do sidecar = "Newey-West, lag `max(h−1, 1)`"; do gold = "Bartlett, lag `min(max(1, n^{1/3}), 10)`".
- **Como deveria funcionar (exemplo):** erros de previsão *h*-passos seguem MA(*h−1*), então o lag canônico é `h−1`. Para `h=7` → lag `max(6,1)=6`; para `h=1` → lag `max(0,1)=1`. O gold legacy, para `n≈1000`, escolheria lag 10 (o teto) — **ignorando** o horizonte. Mesmo kernel de Bartlett (peso `1 − k/(lag+1)`), bandwidth diferente (ver item #1, elemento 4).
- **O que esperar no código:** `lag = max(h−1, 1)` e o loop HAC com peso de Bartlett.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:233`](../../../../src/domain/services/dm_tft_vs_baseline.py#L233) (`lag = max(int(horizon) - 1, 1)`) e [`:243`](../../../../src/domain/services/dm_tft_vs_baseline.py#L243) (`weight = 1.0 - (k / (lag + 1.0))`). ✅ **Confere** exatamente; o lag usado é exportado em `hac_lag_used` ([`:315`](../../../../src/domain/services/dm_tft_vs_baseline.py#L315)).

**4. Correção HLN de amostra pequena aplicada (gold não tem)**
- **O que o doc afirma:** §7 — small-sample do sidecar = "HLN aplicado"; do gold = "Sem HLN".
- **Como deveria funcionar (exemplo):** HLN encolhe a estatística por `sqrt[(n + 1 − 2h + h(h−1)/n)/n]` (ver item #1, elemento 5). Para `n≈937, h=7`: fator `= (937 + 1 − 14 + 42/937)/937 ≈ 0,986` → `sqrt ≈ 0,993` → encolhe a estatística ~0,7% (efeito pequeno por `n` ser grande). Em coortes pequenas o efeito é maior; por isso a prática é **sempre aplicar**, como Phase B faz e o gold legacy **não**.
- **O que esperar no código:** cálculo do fator HLN, multiplicação da estatística e uma flag `hln_applied`.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:255`](../../../../src/domain/services/dm_tft_vs_baseline.py#L255) (`hln_factor`), [`:259`](../../../../src/domain/services/dm_tft_vs_baseline.py#L259) (`dm_hln = dm_stat * sqrt(hln_factor)`) e [`:272`](../../../../src/domain/services/dm_tft_vs_baseline.py#L272) (`hln_applied=True`). ✅ **Confere:** a estatística reportada em `DMResult.dm_stat` **já é a corrigida** (`dm_hln`, [`:268`](../../../../src/domain/services/dm_tft_vs_baseline.py#L268)).

**5. Teste one-sided `H_A: TFT < baseline`, e o Holm corrige o p-value one-sided (gold é two-sided)**
- **O que o doc afirma:** §7 — tail do sidecar = "One-sided `H_A: TFT < baseline`"; do gold = "Two-sided". E o Holm da família corrige o **one-sided** (cf. item #3, elemento 1, que ancorou em `holm_family_6.py:16`).
- **Como deveria funcionar (exemplo):** a série é `d_t = loss_TFT − loss_baseline`; negativo = TFT melhor. Se `dm_hln = −1,9`: one-sided-less `p = Φ(−1,9) ≈ 0,029` (significativo a 5%); two-sided `= 2·(1−Φ(1,9)) ≈ 0,057` (não significativo). Mesma evidência, veredicto diferente — o one-sided "ganha poder" para o claim direcional, e é **esse** `0,029` que entra no Holm.
- **O que esperar no código:** `pvalue_one_sided_less = Φ(dm_hln)`; e o use case passando essa coluna ao Holm.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:266`](../../../../src/domain/services/dm_tft_vs_baseline.py#L266) (`p_less = float(_norm_cdf(dm_hln))`; two-sided fica em [`:265`](../../../../src/domain/services/dm_tft_vs_baseline.py#L265)) e [`compute_phase_b_tier_metrics_use_case.py:244`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L244) (`apply_holm_one_sided(dm["pvalue_one_sided_less"])`). ✅ **Confere:** o Holm da família recebe **o one-sided**, não o two-sided (espelho exato do gap apontado no gold legacy, item #3).

**6. Família declarada ex-ante = 6 testes (3 baselines × 2 horizontes)**
- **O que o doc afirma:** §6 e §7 — "6 testes … Holm sobre família 6 (3 baselines × 2 horizontes), declarada ex-ante"; o gold legacy, ao contrário, usa um groupby administrativo `(asset, parent_sweep_id, split, horizon)` sem `split_signature` (item #3).
- **Como deveria funcionar (exemplo):** a família é o **produto cartesiano declarado na política**: `confirmatory_horizons = (1, 7)` × 3 `baseline_model_versions` = 6 linhas → o Holm corrige com `m = 6`. Não é "o que sobrou de um groupby"; é o conjunto fixado antes de ver os p-values.
- **O que esperar no código:** `compute_dm_family` iterando horizontes × baselines, e o use case montando a família-6 + Holm com `analysis_role="primary_family_6"`.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:287-290`](../../../../src/domain/services/dm_tft_vs_baseline.py#L287) (loop `horizon` × `baseline_model_version`); [`compute_phase_b_tier_metrics_use_case.py:227-248`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L227) (`_dm_family_6`, `analysis_role="primary_family_6"` em [`:247`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L247)); contagem em [`phase_b_tier_policy.py:23-28`](../../../../src/domain/services/phase_b_tier_policy.py#L23) (`confirmatory_horizons=(1,7)`; `baseline_model_versions` com **3** nomes). ✅ **Confere:** 3 × 2 = 6, declarados na política congelada — família **ex-ante**, ao contrário do groupby do gold.

**7. Sem top-50; coorte pré-declarada (gold aplica top-50 por default)**
- **O que o doc afirma:** §7 — top-50 do sidecar = "N/A — coorte pré-declarada (15 TFT + 45 baseline)"; do gold = "Aplicado por default". O item #4 mostra que o top-50 do gold **invalida** o universo pré-definido.
- **Como deveria funcionar (exemplo):** o sidecar separa os runs por `feature_set_name` (TFT vs `"baseline"`) e os baselines por `model_version` segundo a política — **todos** os runs da coorte entram, sem rankear por `squared_error` e cortar nos 50 melhores. O universo é o que foi declarado, não o que "venceu no test split".
- **O que esperar no código:** `_run_groups` montando TFT/baselines a partir do `dim_run` escopado, **sem** nenhuma chamada a `_select_top_configs_for_pairwise`/`head(50)`.
- **Ref do doc → código:** [`compute_phase_b_tier_metrics_use_case.py:212-225`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L212) (`_run_groups`; baselines vindos de `self.policy.baseline_model_versions` em [`:219`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L219)). ✅ **Confere a ausência de top-50:** não há filtro top-N em nenhum ponto do fluxo confirmatório (li `dm_tft_vs_baseline.py` e o use case inteiros). ⚠️ **Ressalva de escopo:** a contagem **"15 TFT + 45 baseline"** é um número **de dados** (depende da coorte materializada), **não derivável do código** — o código consome quaisquer runs presentes no `dim_run` escopado por `asset`+`parent_sweep_id`. Registro como não-verificável aqui (não contradito).

**8. Unidade estatística = `target_timestamp_utc` com dedup operationally-latest cross-fold (gold faz pivot wide + `dropna(how="any")`)**
- **O que o doc afirma:** §6 ("unidade `target_timestamp_utc` com dedup operationally-latest cross-fold") e §7 (mesma frase vs. gold = "pivot wide com `dropna(how="any")`").
- **Como deveria funcionar (exemplo):** o mesmo `target_timestamp_utc` pode ter sido previsto por vários folds (janelas walk-forward que se sobrepõem). Em vez de contá-lo várias vezes, o sidecar escolhe **um** fold por timestamp — o "operationally latest" (a janela mais recente cujo treino terminou antes de prever aquele ponto). A série `d_t` usa só os timestamps **comuns** a TFT e baseline. O gold legacy não faz dedup por fold: ele pivota a matriz de losses (timestamps × configs) e descarta linhas com qualquer buraco — unidade estatística **diferente**.
- **O que esperar no código:** uso de `select_operationally_latest_fold` por timestamp e interseção `common_ts`.
- **Ref do doc → código:** [`dm_tft_vs_baseline.py:198-207`](../../../../src/domain/services/dm_tft_vs_baseline.py#L198) (`select_operationally_latest_fold` para TFT e baseline), [`:182`](../../../../src/domain/services/dm_tft_vs_baseline.py#L182) (`common_ts = intersection`) e a tag [`:319`](../../../../src/domain/services/dm_tft_vs_baseline.py#L319) (`"dedup_rule": "operationally_latest_fold"`). ✅ **Confere:** o resolvedor é importado de `fold_dedup_resolver` ([`:11-15`](../../../../src/domain/services/dm_tft_vs_baseline.py#L11)) e aplicado por timestamp antes de formar `d_t`.

**9. Saída e escopo: `phase_b_dm_family_6.parquet` sustenta só H2a/H2b; H1 vive em outro sidecar**
- **O que o doc afirma:** §6 — "Evidência confirmatória primária **para H2a/H2b apenas**. H1 (calibração) vem de `phase_b_marginal_coverage.parquet` e `phase_b_tier_verdict.parquet`; o sidecar DM family-6 **não** sustenta H1."
- **Como deveria funcionar (exemplo):** o use case calcula coisas distintas para hipóteses distintas: H1 (cobertura/calibração) sai de `compute_marginal_coverage`; H2a/H2b (superioridade probabilística) saem da família-6 DM + delta de pinball. Cada uma vai para o seu sidecar. Ler `gold_dm_pairwise_results` (ou mesmo o family-6) como evidência de **calibração** seria um erro de mapeamento de hipótese.
- **O que esperar no código:** `write_dm_family_6` para o sidecar DM; `compute_marginal_coverage` + `write_marginal_coverage` para o caminho de H1.
- **Ref do doc → código:** [`compute_phase_b_tier_metrics_use_case.py:492`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L492) (`write_dm_family_6`) e [`:457`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L457)/[`:488`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L488) (`compute_marginal_coverage` → `write_marginal_coverage`). ✅ **Confere:** o family-6 não carrega H1; o caminho de calibração é separado.
- ⚠️ **Ponto de atenção — o risco real deste item é confusão, não metodologia**
  O único "risco C.0" deste item (skeleton §6: "listado aqui para evitar contaminação acidental") é **interpretativo/operacional**, não estatístico. Três armadilhas a carregar para C.0.2/C.0.3:
  - **Confundir artefatos:** `gold_dm_pairwise_results` (gold legacy, item #1 — squared_error, two-sided, sem HLN, lag `n^{1/3}`, top-50, Holm sobre family administrativa) **≠** `phase_b_dm_family_6.parquet` (este — pinball, one-sided, HLN, lag `h−1`, sem top-50, Holm sobre família-6 declarada). São DMs com **decisões opostas em cada eixo**; só o nome "DM" é comum. Qualquer doc canônico deve dizer isso explicitamente (regra §7.5 do skeleton).
  - **Existe um irmão de sensibilidade:** o mesmo use case também produz uma **família-18 conservadora** ([`compute_phase_b_tier_metrics_use_case.py:250-290`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L250), `analysis_role="sensitivity_conservative"`) — DM por fold, **não** o confirmatório primário. Não confundir o sidecar de sensibilidade com o family-6.
  - **Não sobrescrever:** correções no gold legacy não devem tocar este sidecar (skeleton §7.4: `analytics_archive_phase_b_*` é read-only). **Quando importa:** sempre que C.0 mexer em `pairwise.py`/`confidence.py` — o family-6 vive em `data/analytics/reports/phase_b/...`, fora do gold, e deve continuar intocado.

### Cross-check — o que NÃO está corretamente indicado/referenciado

Para o item #13, **as referências do skeleton conferem** e foram **fortalecidas** (de `ls` para leitura direta). Pontos a registrar:

1. **Nenhuma referência quebrada.** Os três arquivos existem e contêm os símbolos citados (`compute_dm_family`, `compute_dm_hln_hac`, `apply_holm_one_sided`, `_dm_family_6`). O skeleton confirmara só por `ls`; aqui confirmei por leitura do corpo inteiro de cada função.
2. **Todos os contrastes da tabela §7 batem com o código** — pinball post-guardrail, lag `h−1`, HLN, one-sided (com Holm sobre o one-sided), família-6 declarada (3×2 na política), ausência de top-50 e dedup `operationally_latest_fold`. A âncora de contraste está **íntegra**.
3. **Dois números são de dados, não de código, e ficam não-verificáveis aqui:** "**15 TFT + 45 baseline**" (§7) e "**~937 timestamps dedupados em h=7**" (§6, "Riscos conhecidos"). Dependem da coorte materializada; o código apenas consome os runs presentes no `dim_run` escopado. Não contraditos — apenas fora do alcance de uma verificação por leitura de código.
4. **Contexto adicional não mencionado na §6 (não é discrepância):** o use case também emite uma **família-18 de sensibilidade** (`analysis_role="sensitivity_conservative"`) e roda um **gate de integridade pré-escrita** (`_validate_pre_write_integrity`, [`:419-447`](../../../../src/use_cases/compute_phase_b_tier_metrics_use_case.py#L419)) que falha se o family-6/18 vier vazio ou com estatística NaN. Ambos **reforçam** a solidez da referência; registro como contexto, não como falha do dossiê.
5. **Escopo respeitado:** o skeleton declara "metodologia NÃO auditada (escopo Phase B fechado)" e esta seção **não** a reabriu — só confirmou existência de caminhos e presença das características de contraste.

### Veredito do item #13

🟢 **Íntegro (referência).** Os três caminhos existem e foram confirmados por **leitura
direta** (mais forte que o `ls` do skeleton); **cada** característica que distingue o sidecar
do gold legacy — pinball post-guardrail, lag `h−1`, HLN, one-sided com Holm sobre o
one-sided, família-6 declarada (3×2), sem top-50, dedup `operationally_latest_fold` — está
de fato no código, tornando a âncora de contraste da §7 fiel. Metodologia **não** reauditada,
por escopo (Phase B fechada). Único resíduo: dois números de **dados** ("15 TFT + 45
baseline"; "~937 timestamps em h=7") não são verificáveis por leitura de código — registrados,
não contraditos. O risco prático do item é **interpretativo** (não confundir/sobrescrever),
não estatístico (elemento 9 ⚠️).

---

## #11 — VaR / ES gold

### O que é (didático)

**VaR (Value-at-Risk)** e **ES (Expected Shortfall)** são duas formas de resumir o
"quão ruim pode ficar" em um nível de probabilidade. Imagine que o modelo prevê uma
**distribuição** para o retorno do próximo período (não só um número, mas "qual a cara
da incerteza"). Então:

- **VaR a 10%** = o **quantil 10%** dessa distribuição prevista. Em palavras: "há 10% de
  chance de o retorno ficar **igual ou pior** do que esse valor". É a fronteira da cauda.
- **ES a 10%** = a **média** da distribuição **dentro** dessa cauda de 10% (o "quão fundo"
  costuma ir quando passa da fronteira). Por construção o ES é **pelo menos tão extremo
  quanto** o VaR — olha mais para o fundo da cauda.

O "truque" do builder: ele **não estima** essas quantidades a partir de erros realizados
nem faz backtesting. Ele simplesmente **lê os quantis preditos** (`q10`, `q50`
post-guardrail), define `var_10 = q10` diretamente, aproxima o `es_10` por uma **fórmula
linear** sobre `q10` e `q50`, e tira a **média** por grupo de runs. É, portanto, um resumo
**descritivo do que o modelo prevê** como cauda — não uma medida de risco financeiro
validada contra a realidade.

### Elementos

**1. Variável-base = quantis preditos *post-guardrail* (não `error`, não retorno-perda)**
- **O que o doc afirma:** as variáveis-base são `post_q10 = "quantile_p10_post_guardrail"` e `post_q50 = "quantile_p50_post_guardrail"`; o uso de `error = y_pred − y_true` e a coluna `max_drawdown` (citados pela auditoria externa) **não** existem no builder atual.
- **Como deveria funcionar (exemplo):** para uma linha OOS o modelo prevê `q10_post = −0,032` (retorno de −3,2%) e `q50_post = +0,004`. São os quantis **da distribuição preditiva do alvo**, já passados pelo guardrail de monotonicidade. O VaR/ES é montado **sobre esses quantis preditos**, não sobre o erro de previsão.
- **O que esperar no código:** leitura das duas colunas `*_post_guardrail`, conversão numérica, e — se ausentes — `var_10_row`/`es_10_approx_row` = `NaN` com warning (sem `error`, sem `max_drawdown`).
- **Ref do doc → código:** [`confidence.py:65-66`](../../../../src/domain/services/gold_builders/confidence.py#L65) (nomes das colunas) e [`:67-72`](../../../../src/domain/services/gold_builders/confidence.py#L67) (ramo que as usa); ramo `else` em [`:73-85`](../../../../src/domain/services/gold_builders/confidence.py#L73) (NaN + warning). ✅ **Confere:** `post_q10 = "quantile_p10_post_guardrail"`, `post_q50 = "quantile_p50_post_guardrail"`; `df["var_10_row"] = df[post_q10]`; e no `else`, `df["var_10_row"] = np.nan` / `df["es_10_approx_row"] = np.nan`. **Não há** `y_pred − y_true` nem `max_drawdown` na função inteira (li 39-130). A discrepância da auditoria externa está **resolvida** pelo código real.
- ⚠️ **Ponto de atenção — "VaR sobre quantil predito" não é o VaR do risk-management**
  Há **duas coisas diferentes** chamadas "VaR":

  | "VaR" | Como se obtém | O que valida |
  |---|---|---|
  | **VaR predito** (atual) | lê o `q10` que o modelo **previu** | nada — é a saída do modelo, não confrontada com a realidade |
  | **VaR backtested** (risk-management) | conta quantas vezes a perda **realizada** furou o `q10` previsto (≈10%?) | a cobertura empírica via Kupiec/Christoffersen |

  O builder faz o primeiro. Isso é legítimo como **descrição do que o modelo acha da cauda**,
  mas **não** sustenta um claim de risco financeiro: um modelo pode prever `q10 = −3,2%`
  e, na prática, furar esse limite em 30% dos casos (cauda mal calibrada) — e o
  `gold_prediction_risk` jamais perceberia, porque nunca olha o realizado. **Quando é
  tolerável:** uso puramente descritivo ("nível de downside que o modelo tipicamente
  projeta"). **Quando não é:** qualquer afirmação de que "o modelo controla risco a 10%" —
  isso exige backtesting de excedências (ver elemento 5). A própria auditoria externa
  recomenda **renomear para erro/cauda descritiva** enquanto não houver backtesting com
  sinal, nível e excedências.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: manter o cálculo
  > **descritivo** sobre quantis preditos, mas **declarar explicitamente a variável-alvo**
  > (retorno do alvo, sinal "menor = pior") e tratar a coluna como **cauda predita
  > descritiva** — não como VaR financeiro — até existir backtesting de cobertura
  > (elemento 5). Alinha-se à recomendação da auditoria externa.

**2. VaR_10 = `q10` predito, diretamente (`var_10_row = df[post_q10]`)**
- **O que o doc afirma:** `var_10_row = df[post_q10]` — idêntico ao `q10_post_guardrail`.
- **Como deveria funcionar (exemplo):** se `q10_post = −0,032`, então `var_10_row = −0,032`. Não há transformação, sinal trocado nem escalonamento: o VaR a 10% **é** o décimo percentil predito do alvo. Lê-se como "10% de chance de o retorno ser ≤ −3,2%".
- **O que esperar no código:** atribuição direta `var_10_row = q10_post`, sem `abs()`, sem `−`, sem multiplicador.
- **Ref do doc → código:** [`confidence.py:70`](../../../../src/domain/services/gold_builders/confidence.py#L70). ✅ **Confere:** `df["var_10_row"] = df[post_q10]` (cópia direta).
- ⚠️ **Ponto de atenção — α=10% é uma cauda "fraca", e o nível está só no nome da coluna**
  Dois pontos práticos:
  - **Nível de cauda.** O VaR/ES de risco financeiro costuma usar **α = 1% ou 5%** (cauda
    severa). **α = 10%** é uma cauda **rasa** — captura o "downside típico", não eventos
    extremos. Não é errado, mas é uma escolha **conservadora-fraca**: subdimensiona o risco
    de eventos raros frente ao que um VaR a 1% mostraria. Para previsão a *h* passos com
    poucos quantis preditos (só q10/q50/q90), 10% é o menor nível **realmente predito** — ir
    a 1% exigiria extrapolar muito mais (e a fórmula do ES já extrapola; ver elemento 3).
  - **Nível implícito no nome.** `α = 0,10` aparece só no **nome** `var_10`/`es_10_approx`,
    não como metadado/coluna de schema. Um leitor que pegue a coluna sem ler a doc não tem
    como saber o nível, o sinal, nem que é "sobre o quantil predito". **Quando importa:**
    sempre que a tabela for exportada/consumida fora do contexto desta doc.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: **declarar α no
  > schema** (ex.: coluna/atributo `alpha = 0.10`) em vez de embutir só no nome, e avaliar
  > **complementar com α = 5%** (e 1% se a doc aceitar a extrapolação) para que a tabela
  > descreva também a cauda severa, não só o downside típico.

**3. ES_10 ≈ `1.125·q10 − 0.125·q50` (extrapolação linear da cauda)**
- **O que o doc afirma:** `es_10_approx_row = 1.125 * df[post_q10] - 0.125 * df[post_q50]`; é "interpolação linear assumindo forma específica de cauda; não é ES universal".
- **Como deveria funcionar (exemplo):** com `q10 = −0,032` e `q50 = +0,004`: `es = 1,125·(−0,032) − 0,125·(0,004) = −0,036 − 0,0005 = −0,0365`. O ES (−3,65%) fica **mais fundo** que o VaR (−3,2%), como esperado — ele "olha dentro da cauda".
- **O que esperar no código:** a combinação linear exata `1.125·q10 − 0.125·q50`.
- **Ref do doc → código:** [`confidence.py:71`](../../../../src/domain/services/gold_builders/confidence.py#L71). ✅ **Confere** exatamente: `df["es_10_approx_row"] = 1.125 * df[post_q10] - 0.125 * df[post_q50]`.
- ⚠️ **Ponto de atenção — de onde vêm `1.125` e `0.125`, e por que isso subdimensiona caudas gordas**
  Os coeficientes **não são mágicos**: saem de assumir que a **função-quantil** (o inverso
  da CDF) é uma **reta** entre o ponto `(0,50; q50)` e `(0,10; q10)` e que essa reta
  **continua** abaixo de 10% até 0. O ES a 10% é a média da função-quantil no intervalo
  `[0; 0,10]`:

  `ES₁₀ = (1/0,10) ∫₀^0,10 q(u) du`, com `q(u) = q10 + [(q50−q10)/0,40]·(u−0,10)`.

  Resolvendo a integral: `∫₀^0,10 q(u) du = 0,10·q10 − 0,0125·(q50−q10)`, então
  `ES₁₀ = q10 − 0,125·(q50−q10) = 1,125·q10 − 0,125·q50`. **É exatamente a fórmula do
  código** — ou seja, é uma **extrapolação linear da cauda inferior**.

  **Por que isso importa:** retornos financeiros têm **caudas gordas** (leptocúrticas) — a
  função-quantil **acelera** (fica mais íngreme) no extremo, não segue reta. A extrapolação
  linear, por isso, **subestima** o quão fundo a cauda vai. Exemplo concreto com a **normal
  padrão** (cauda "leve", o caso *fácil* para a aproximação): `q10 = −1,2816`, `q50 = 0`.
  - Fórmula: `ES = 1,125·(−1,2816) − 0,125·0 = −1,442`.
  - ES verdadeiro da normal a 10%: `−φ(z₀,₁)/0,10 = −0,1755/0,10 = −1,755`.
  - A aproximação dá **−1,44 vs −1,76 reais → subestima a severidade da cauda em ~18%**,
    e isso **na normal**; numa distribuição de cauda gorda o erro é **maior**.

  **Quando é aceitável:** como número descritivo aproximado, ciente de que **encolhe** o ES
  (otimista quanto à cauda). **Quando não é:** se o ES virar evidência de "controle de risco
  de cauda" — aí o viés sistemático de subestimação engana na direção perigosa (faz o risco
  parecer menor do que é). A própria nomenclatura `es_10_**approx**` já sinaliza honestamente
  que é aproximação.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: manter como
  > `es_10_approx` **descritivo** com a ressalva documentada de que **subestima a cauda**
  > (extrapolação linear). Se o ES for promovido a qualquer uso confirmatório, substituir a
  > extrapolação linear por estimativa que respeite a cauda (mais quantis preditos no extremo,
  > ou ES backtested) e avaliar o par (VaR, ES) por **proper scoring** elicitável conjuntamente
  > (Fissler-Ziegel 2016).

**4. Clip `es_10_approx = min(es_10_approx, var_10)` (garante ES ≤ VaR)**
- **O que o doc afirma:** `es_10_approx_row = np.minimum(es_10_approx_row, var_10_row)`; o clip "pode mascarar" violação de monotonicidade.
- **Como deveria funcionar (exemplo):** o ES de uma cauda inferior deve ser **≤** o VaR (mais fundo). O clip força isso. Quando `es` calculado já é ≤ `var` (caso normal, `q10 ≤ q50`), o clip **não faz nada**. Ele só age se `es > var`.
- **O que esperar no código:** um `np.minimum(es, var)` logo após a fórmula do ES.
- **Ref do doc → código:** [`confidence.py:72`](../../../../src/domain/services/gold_builders/confidence.py#L72). ✅ **Confere:** `df["es_10_approx_row"] = np.minimum(df["es_10_approx_row"], df["var_10_row"])`.
- ⚠️ **Ponto de atenção — o clip ativa exatamente sob *crossing* (`q10 > q50`); mascara em vez de sinalizar**
  Quando o clip realmente muda algo? `es > var` ⟺ `1,125·q10 − 0,125·q50 > q10` ⟺
  `0,125·q10 > 0,125·q50` ⟺ **`q10 > q50`**. Ou seja, o clip só age quando há **cruzamento
  de quantis** (o 10º percentil acima do 50º — uma violação de monotonicidade). Nesse caso
  a fórmula daria `ES > VaR` (sem sentido), e o clip força `ES = VaR`.

  **A nuance importante:** as colunas são **post-guardrail**, e o
  [`QuantileGuardrailService.enforce_monotonic_triplet`](../../../../src/domain/services/quantile_guardrail_service.py#L18)
  **ordena** o triplo (`sorted([p10,p50,p90])`) sempre que os três são finitos — então
  `q10_post ≤ q50_post` é **garantido** para valores finitos e o clip vira **no-op** na
  prática. O comentário em [`confidence.py:62-64`](../../../../src/domain/services/gold_builders/confidence.py#L62)
  é justamente isso: "use exclusively post-guardrail columns" porque o guardrail garante a
  monotonicidade que o VaR/ES exige (Jorion 2007; Acerbi & Tasche 2002).

  **Então onde está o risco?** É de **engenharia defensiva, não de bug atual**: se algum dia
  uma linha post-guardrail **não** estiver ordenada (guardrail não aplicado num caminho de
  dados, valor não-finito que escapou ao `sorted`, refactor futuro), o clip **silenciaria** o
  cruzamento (ES=VaR) em vez de **sinalizá-lo** (ex.: `NaN` + auditoria). O lugar que
  **sinaliza** corretamente é o `gold_quantile_guardrail_audit`, que mede `crossing_after`
  ([`quantile.py:511-514`](../../../../src/domain/services/gold_builders/quantile.py#L511)).
  O dossiê captou isto no TODO "violação de monotonicidade → NaN (atualmente clip pode
  mascarar)". **Quando importa:** só se a garantia do guardrail falhar upstream; hoje, com
  o triplo ordenado, o clip não tem efeito observável.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: como o guardrail
  > já garante `q10 ≤ q50` para valores finitos, **manter o clip como salvaguarda**, porém
  > **trocar "mascarar" por "sinalizar"** num eventual hardening: se `es > var` (cruzamento
  > residual) ou se algum dos quantis for não-finito, emitir `NaN` em vez de colapsar para
  > `var`, deixando a anomalia visível (consistente com o `crossing_after` do guardrail audit).

**5. Agregação = média por 8 colunas de grupo (`var_10 = mean(q10)`)**
- **O que o doc afirma:** agrega por `[run_id, asset, feature_set_name, config_signature, split, fold, seed, horizon]`; `var_10 = ("var_10_row", "mean")` e `es_10_approx = ("es_10_approx_row", "mean")`.
- **Como deveria funcionar (exemplo):** três linhas OOS de um grupo com `q10 = {−0,03; −0,05; −0,02}` → `var_10 = média = −0,0333`. O número final é a **média dos quantis preditos** ao longo dos timestamps daquele grupo — um "nível de downside típico que o modelo projeta".
- **O que esperar no código:** `df.groupby(group_cols).agg(..., var_10=("var_10_row","mean"), es_10_approx=("es_10_approx_row","mean"))`.
- **Ref do doc → código:** [`confidence.py:87-100`](../../../../src/domain/services/gold_builders/confidence.py#L87) (as 8 colunas de grupo) e [`:102-112`](../../../../src/domain/services/gold_builders/confidence.py#L102) (a agregação; `var_10`/`es_10_approx` em [`:108-109`](../../../../src/domain/services/gold_builders/confidence.py#L108)). ✅ **Confere:** `group_cols` lista exatamente as 8 colunas (quando presentes); `var_10=("var_10_row", "mean")`, `es_10_approx=("es_10_approx_row", "mean")`.
- ⚠️ **Ponto de atenção — média de quantis preditos ≠ VaR estimado; e o silêncio sobre excedências**
  Duas leituras possíveis, e a diferença é o coração do "descritivo vs confirmatório":

  - **O que o número É:** a **média temporal** do 10º percentil **predito**. Responde "em
    média, qual o downside que o modelo projeta?". É uma estatística-resumo da **saída do
    modelo**.
  - **O que o número NÃO é:** um VaR **estimado e validado**. Um VaR de verdade seria
    confrontado com os retornos **realizados**: contar a fração de vezes em que o realizado
    furou o `q10` previsto e testar se ≈ 10% (**Kupiec 1995** para a taxa de excedência;
    **Christoffersen 1998** para independência das excedências). **Nada disso existe** no
    builder — ele nunca toca `y_true` para a parte de risco.

  **Implicação concreta:** suponha que o modelo preveja consistentemente `q10 ≈ −3%`, mas
  na realidade os retornos furem −3% em **25%** dos dias. O `var_10 = −0,03` parecerá um
  "downside controlado", quando a cauda está **gravemente subcoberta**. Promover esse número
  a evidência de risco sem backtesting seria afirmar cobertura que nunca foi medida.

  > **Decisão recomendada** *(confirmar com pesquisa acadêmica do paper)*: classificar o item
  > como **`DESCRIPTIVE_ONLY`** enquanto não houver **backtesting de excedências**
  > (Kupiec 1995 / Christoffersen 1998) com nível α e sinal declarados; só então cogitar
  > promoção a confirmatório, com avaliação do par (VaR, ES) por scoring elicitável
  > (Fissler-Ziegel 2016). Coincide com o esboço de "Critério para promoção" do skeleton.

**6. Natureza terminal da tabela + colunas-companheiras baseadas em `y_pred`**
- **O que o doc afirma:** "Linha OOS; agregado por run"; tabela de "natureza descritiva". (O dossiê não lista consumidores downstream do `gold_prediction_risk`.)
- **Como deveria funcionar (exemplo):** diferentemente de PICP/MPIW/pinball (que sobem para o `gold_model_decision_final`), o `gold_prediction_risk` é uma **tabela-folha**: ninguém a consome para decisão. Quem quiser usá-la lê o parquet diretamente.
- **O que esperar no código:** nenhum builder com `requires_gold = (... "gold_prediction_risk" ...)`; o `_build_model_decision_final` **não** recebe `gold_prediction_risk` entre seus 7 inputs.
- **Ref do doc → código:** varredura repo-wide — a **única** referência a `gold_prediction_risk` em `src/` é o próprio `output_table` em [`confidence.py:36`](../../../../src/domain/services/gold_builders/confidence.py#L36); nenhum `requires_gold` a inclui; os inputs de `_build_model_decision_final` ([`confidence.py:824-832`](../../../../src/domain/services/gold_builders/confidence.py#L824)) **não** a contemplam. ✅ **Confere: tabela terminal, sem consumidor.** A living-paper inclusive **proíbe** usá-la como evidência ([`30_results_and_analysis.md:172`](../../../07_reports/living-paper/30_results_and_analysis.md#L172), [`40_limitations_and_conclusion.md:112`](../../../07_reports/living-paper/40_limitations_and_conclusion.md#L112)).
- ⚠️ **Ponto de atenção — a mesma tabela mistura colunas de base distinta (`y_pred` vs quantis)**
  O `gold_prediction_risk` carrega, lado a lado:
  - `var_10`, `es_10_approx` → base = **quantis preditos** (`q10`/`q50` post-guardrail).
  - `expected_move = mean(|y_pred|)` e `downside_risk = mean(max(−y_pred, 0))` → base = a
    **previsão pontual** `y_pred` ([`confidence.py:59-60`](../../../../src/domain/services/gold_builders/confidence.py#L59)).

  São **duas famílias de variáveis** numa só tabela. Um leitor que assuma "a tabela toda é
  sobre a distribuição preditiva" erra: `expected_move`/`downside_risk` ignoram a incerteza
  (usam só o ponto central). **Quando importa:** ao citar a tabela em texto — convém dizer
  *qual coluna* e *sobre qual base*. Não é defeito; é heterogeneidade a documentar. (O fato
  de a tabela ser terminal e proibida como evidência **reduz** o risco prático.)

### Cross-check — o que NÃO está corretamente indicado/referenciado

As referências de "Implementação atual localizada" do dossiê de VaR/ES **conferem** com o
código (li `PredictionRiskGoldBuilder.build()` inteiro, 39-130, mais o
`QuantileGuardrailService` e a varredura de consumidores). Pontos a registrar:

1. **Nenhuma referência quebrada.** Linhas-âncora **exatas**: `post_q10`/`post_q50` em
   L65-66 ✅; `var_10_row = df[post_q10]` em L70 ✅; `es_10_approx = 1.125·q10 − 0.125·q50`
   em L71 ✅; clip em L72 ✅; `expected_move_row`/`downside_risk_row` em L59-60 ✅; 8 colunas
   de grupo em L87-100 ✅; `var_10`/`es_10_approx` agg em L108-109 ✅. (A faixa "62-109"
   citada na tabela-índice da §5 do skeleton é o **miolo** VaR/ES; a função abre em L39.)

2. **Discrepância da auditoria externa: confirmada como RESOLVIDA.** A
   [`auditoria_metodologica`](../../../07_reports/external-reviews/auditoria_metodologica_forecasting_financeiro.md)
   (linhas 26 e 257) afirma que o builder calcula VaR/ES/`max_drawdown` sobre
   `error = y_pred − y_true`. O **código atual não faz nada disso**: usa exclusivamente
   `quantile_p10/p50_post_guardrail` e **não tem** `max_drawdown`. A auditoria descreve uma
   versão anterior (via `DATA_PIPELINE_WALKTHROUGH.md`), não o `src/` atual. O dossiê já
   marcou isso como resolvido; **confirmo por leitura direta**.

3. **`max_drawdown` ausente — e `DATA_PIPELINE_WALKTHROUGH.md` ainda descreve a versão antiga.**
   O walkthrough ([`:1470`](../../../02_data/DATA_PIPELINE_WALKTHROUGH.md#L1470), seção 6.3.7)
   e a auditoria que dele deriva citam `max_drawdown` e base `error`. O código não os tem.
   **Isto é drift doc→código a corrigir em C.0.3** (no walkthrough, fora do escopo desta
   revisão), não erro do skeleton — que já aponta a divergência.

4. **"Uso atual no projeto" do skeleton está completo e até conservador.** Lista só
   `gold_prediction_risk.{var_10, es_10_approx}` — e, de fato, **não há consumidor downstream**
   (tabela terminal). O classificador "unknown (suspeito tail_error, não risco financeiro)" do
   skeleton bate com o que o código mostra: número descritivo sobre quantil predito, sem
   backtesting.

5. **`expected_move`/`downside_risk` (base `y_pred`) não são mencionados no dossiê.** Vivem na
   mesma tabela e na mesma função, mas usam **base diferente** (previsão pontual, não quantis).
   Registrado no elemento 6 — é heterogeneidade da tabela, não erro de referência.

6. **Pontos residuais metodológicos centrais já capturados pelo skeleton.** "Sem backtesting",
   "ES como interpolação linear assumindo forma de cauda", "α=10% implícito no nome" constam dos
   "Riscos conhecidos". Os aprofundamentos acima detalham o **mecanismo** (a derivação dos
   coeficientes 1.125/0.125 e o viés de subestimação de cauda; o clip ativando só sob crossing;
   média de quantis ≠ VaR backtested) e o "quando cada escolha se aplica".

### Veredito do item #11

🟡 **Ressalvas.** Referências **intactas** e evidência fiel ao código — o VaR/ES é calculado
exatamente como o dossiê (e o skeleton já corrigido) afirma: `var_10 = q10_post`,
`es_10_approx = 1.125·q10 − 0.125·q50`, clip `min(es, var)`, média por grupo, **sem** `error`
e **sem** `max_drawdown` (discrepância da auditoria externa **confirmada como resolvida**). As
ressalvas são **metodológicas/de enquadramento**, não de localização: (a) é VaR/ES **sobre
quantil predito**, não VaR backtested — falta backtesting de excedências (Kupiec/Christoffersen);
(b) o ES é uma **extrapolação linear** que **subestima caudas gordas** (~18% até na normal); (c)
α=10% é cauda rasa e o nível vive só no nome da coluna; (d) o clip mascara (em vez de sinalizar)
um eventual cruzamento, hoje neutralizado pelo guardrail que **ordena** o triplo; (e) é tabela
**terminal**, sem consumidor, e a living-paper já a **proíbe** como evidência — o que **reduz** o
risco prático. Decisões recomendadas registradas nos elementos 1, 2, 3, 4 e 5 (declarar alvo/α,
manter descritivo, ressalvar subestimação de cauda, sinalizar em vez de mascarar, exigir
backtesting antes de promover) — pendentes de confirmação com a pesquisa acadêmica do paper.

---
