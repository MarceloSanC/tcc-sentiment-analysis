---
title: "C.0 — Revisao didatica dos dossies (§6) + cross-check contra codigo"
scope: |
  Documento companion do C0_statistical_methods_hardening.md. Para cada um dos
  13 itens da §6 (Dossies por item), registra: (a) explicacao didatica do
  teste/metrica; (b) decomposicao elemento a elemento com o que o doc afirma,
  como deveria funcionar (com exemplo), o que esperar no codigo e a referencia
  doc->codigo; (c) para cada ponto de atencao (⚠️), um aprofundamento didatico
  embutido no proprio elemento (o que e, por que importa, quando cada escolha
  se aplica, o que implica para o claim do projeto); (d) cross-check do que NAO
  esta corretamente indicado/referenciado contra o codigo real; (e) um veredito.
  Este material e base de estudo para revisao manual da C.0.1 e insumo de
  redacao do TCC. Sera FUNDIDO depois com a pesquisa academica online (GPT web)
  por item para entao tomar decisao e preencher o C0 canonico. Este arquivo NAO
  e canonico e NAO toma decisao metodologica final.
status: in_progress
created_at: 2026-05-29
relates_to:
  - "C0_statistical_methods_hardening.md (§6 Dossies por item) — fonte das refs verificadas"
  - "external-reviews/ (pesquisa academica GPT web, a fundir por item)"
update_when:
  - cada item da §6 receber sua revisao didatica (progresso 1/13 ... 13/13)
  - uma referencia doc->codigo for confirmada/corrigida contra o src real
  - um ponto de atencao receber aprofundamento didatico
---

# C.0 — Revisao didatica dos dossies (§6) + cross-check contra codigo

> **Para que serve.** Este documento traduz cada dossie tecnico da §6 do
> [`C0_statistical_methods_hardening.md`](C0_statistical_methods_hardening.md)
> em uma explicacao didatica, verifica elemento a elemento contra o codigo real
> em `src/`, e aprofunda cada ponto de atencao com o "quando e por que" que
> embasa as decisoes futuras de C.0.2/C.0.3. Nao substitui o dossie nem decide
> nada: e estudo e cross-check, e sera fundido com a pesquisa academica online
> antes de qualquer preenchimento canonico.

> **Como ler cada item.**
> - **O que e (didatico):** a ideia central em linguagem simples.
> - **Elementos:** cada componente do teste/metrica, com 4 campos fixos
>   (o que o doc afirma / como deveria funcionar com exemplo / o que esperar no
>   codigo / ref doc->codigo) e, **quando houver**, um bloco ⚠️ com
>   aprofundamento de extensao variavel.
> - **Cross-check:** discrepancias entre o que os campos "Uso atual no projeto"
>   e "Implementacao atual localizada" afirmam e o que o codigo real mostra.
> - **Veredito:** 🟢 integro / 🟡 ressalvas / 🔴 referencia quebrada.

> **Convencao de cross-check.** ✅ = ref do doc confirmada por leitura direta do
> codigo; ⚠️ = ponto metodologico (nao e erro de referencia); 🔴 = ref
> quebrada/divergente. Toda confirmacao cita a linha real observada.

## Progresso

- [x] #1 — Diebold-Mariano gold
- [x] #2 — MCS gold
- [ ] #3 — Holm gold
- [ ] #4 — top-50 filter
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

### O que e (didatico)

O teste de **Diebold-Mariano** compara a **acuracia preditiva de dois modelos**. A ideia:

1. Para cada instante *t*, mede o "erro" de cada modelo (uma *loss*).
2. Forma a **serie da diferenca** `d_t = loss_A(t) − loss_B(t)`.
3. Pergunta: a **media** de `d_t` e estatisticamente diferente de zero?
   - `media ≈ 0` → os dois modelos sao igualmente bons (nao da para distinguir).
   - `media < 0` → modelo A erra menos (A vence).

O truque fino: `d_t` e **autocorrelacionado** no tempo (erro de ontem se parece
com o de hoje), entao nao da para usar a variancia "ingenua". Usa-se uma
**variancia HAC** (robusta a heterocedasticidade e autocorrelacao) para nao
subestimar o desvio e gerar p-value otimista demais.

### Elementos

**1. Loss = `squared_error`**
- **O que o doc afirma:** a perda e o erro quadratico `(y_pred − y_true)²`, calculada em `_pairwise_preprocess`.
- **Como deveria funcionar (exemplo):** se o modelo preve 102 e o real e 100, a loss e `(102−100)² = 4`. Quanto maior, pior. O DM compara essas losses entre dois modelos timestamp a timestamp.
- **O que esperar no codigo:** uma coluna `squared_error` derivada de `y_pred` e `y_true`, **antes** do pivot da matriz de losses.
- **Ref do doc → codigo:** [`pairwise.py:280`](../../../../src/domain/services/gold_builders/pairwise.py#L280). ✅ **Confere:** `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2`.
- ⚠️ **Ponto de atencao — a loss casa com o claim?**
  O DM **nao quebra** com modelo probabilistico: ele e **agnostico a loss** — testa a media da diferenca de *qualquer* loss por periodo. O que precisa casar e **a loss com o claim**:

  | Loss alimentada no DM | O que o teste passa a medir | Claim que sustenta |
  |---|---|---|
  | `squared_error` (atual) | acuracia **pontual** (do ponto previsto) | "modelo preve melhor o valor central" |
  | `squared_error` so do **q50** | acuracia pontual **da mediana** | claim pontual mais honesto p/ modelo de quantis |
  | **pinball loss** (q10/q50/q90) | qualidade **da distribuicao preditiva** | "modelo e melhor probabilisticamente" ← claim do TCC |
  | CRPS / WIS | idem, agregando quantis de forma mais principled | idem, ainda mais defensavel |

  Para o claim probabilistico do projeto, a loss certa e **pinball** (foi o que a
  Phase B fez no `phase_b_dm_family_6`). `squared_error` nao esta "errado" — ele
  responde a uma **pergunta diferente** (pontual). Rodar squared_error e depois
  afirmar "modelo probabilistico melhor" e o descasamento. O claim probabilistico
  ja esta coberto pela Phase B; o gold legacy DM, como esta, so embasaria claim
  **pontual**.

**2. Matriz de losses por timestamp (unidade estatistica)**
- **O que o doc afirma:** monta uma "loss matrix wide por `target_timestamp_utc`" e usa `dropna(how="any")`.
- **Como deveria funcionar (exemplo):** uma tabela com linhas = timestamps, colunas = configs de modelo, celulas = loss media naquele timestamp. Para comparar dois modelos de forma justa, so valem timestamps em que **ambos** previram → dai o `dropna(how="any")` (descarta qualquer linha com buraco).
- **O que esperar no codigo:** um `pivot(index=target_timestamp_utc, columns=config_label, values=squared_error)` seguido de `dropna(axis=0, how="any")`.
- **Ref do doc → codigo:** [`pairwise.py:305-308`](../../../../src/domain/services/gold_builders/pairwise.py#L305). ✅ **Confere:** pivot + `dropna(axis=0, how="any")`.

**3. n minimo = 5**
- **O que o doc afirma:** pares com menos de 5 timestamps em comum sao pulados.
- **Como deveria funcionar (exemplo):** com 3 pontos nao da para estimar variancia de forma confiavel; o teste e abortado para aquele par.
- **O que esperar no codigo:** um guard `if n < 5: continue`.
- **Ref do doc → codigo:** [`pairwise.py:78`](../../../../src/domain/services/gold_builders/pairwise.py#L78). ✅ **Confere** exatamente.

**4. Variancia HAC (kernel de Bartlett) + politica de lag**
- **O que o doc afirma:** HAC Bartlett com lag `int(min(max(1, n^(1/3)), 10))` e peso `1 − k/(lag+1)`.
- **Como deveria funcionar (exemplo):** com n=1000 timestamps, `n^(1/3)≈10`, entao usa lag 10 (teto). Soma a variancia "crua" (`gamma0`) mais as autocovariancias ate o lag 10, cada uma com peso decrescente (lag 1 pesa mais que lag 10). Isso "infla" a variancia para refletir a autocorrelacao e evita p-value falsamente pequeno.
- **O que esperar no codigo:** `gamma0` + loop `for k in range(1, lag+1)` somando `2 * weight * cov`.
- **Ref do doc → codigo:** [`pairwise.py:82`](../../../../src/domain/services/gold_builders/pairwise.py#L82) (lag) e [`:87`](../../../../src/domain/services/gold_builders/pairwise.py#L87) (peso). ✅ **Confere** exatamente.
- ⚠️ **Ponto de atencao — deveria seguir Newey-West classico?**
  Ha uma razao teorica especifica para previsao. **Erros de previsao *h*-passos-a-frente** (sob otimalidade) seguem um processo **MA(h−1)** — autocorrelacionados ate o lag *h−1* e nao alem. Por isso o padrao de livro-texto para DM e **lag = h − 1**.
  - **Phase B usou `max(h−1, 1)`** → exatamente essa regra (h=7 → lag 6; h=1 → lag 1). E a escolha canonica para DM de *h*-passos.
  - **Gold legacy usa `min(max(1, n^(1/3)), 10)`** → bandwidth automatica generica (cresce com o tamanho da amostra, comum na literatura HAC), mas **ignora o horizonte h** e tem teto arbitrario 10.

  As duas sao HAC validas, mas a regra `h−1` "sabe" da estrutura MA(h−1) dos
  erros de previsao; a `n^(1/3)` nao. Para DM especificamente, `h−1` e a mais
  defensavel e e a que a Phase B adotou. Risco do gold legacy: (a) inconsistencia
  com a Phase B e (b) para h=7 pode usar lag ate 10 onde o correto seria 6 →
  variancia e p-value ligeiramente diferentes. Nao e bug; e politica de lag menos
  alinhada a teoria de previsao.

**5. Estatistica DM e p-value**
- **O que o doc afirma:** `stat = mean_d / sqrt(var_mean)`; `pvalue = 2·(1 − Φ(|stat|))`, **two-sided**, normal.
- **Como deveria funcionar (exemplo):** se a diferenca media for −0,5 e o erro-padrao 0,1, entao `stat = −5` → p-value minusculo → diferenca significativa. Two-sided = testa "diferente de zero" nos dois sentidos.
- **O que esperar no codigo:** `stat = mean_d / math.sqrt(var_mean)` e `2.0*(1.0 - _norm_cdf(abs(stat)))`.
- **Ref do doc → codigo:** [`pairwise.py:92-93`](../../../../src/domain/services/gold_builders/pairwise.py#L92). ✅ **Confere** exatamente. Confirmei lendo a funcao inteira: **sem HLN** e **sem one-sided**.
- ⚠️ **Ponto de atencao — quando incluir HLN e quando usar one-sided?**

  **HLN (Harvey-Leybourne-Newbold 1997)** = correcao de amostra pequena do DM.
  - **Problema que resolve:** o DM puro e N(0,1) so **assintoticamente**. Em amostra pequena ele **rejeita demais** → p-values otimistas → "significancia" falsa.
  - **O que faz:** (1) multiplica a estatistica por `sqrt[(n + 1 − 2h + h(h−1)/n) / n]` (encolhe a estatistica) e (2) compara contra **t de Student com n−1 g.l.** em vez da normal.
  - **Quando incluir:** sempre que n for pequeno/moderado (regra de bolso: n < ~100–200). Em n grande o fator → 1 e t → normal, entao o efeito some — mas como **nao custa nada**, a pratica moderna e **sempre aplicar**.
  - **No projeto:** Phase B aplicou HLN; gold legacy **nao** → p-values do gold legacy sao **anti-conservadores**, efeito maior em coortes pequenas.

  **One-sided vs two-sided** = direcao da hipotese.
  - **Two-sided** (atual no gold): H₀ "acuracia igual" vs Hₐ "**diferente** (qualquer direcao)". Use quando nao ha direcao a priori.
  - **One-sided:** H₀ "A nao e melhor que B" vs Hₐ "**A e melhor** que B". Use quando o claim e **direcional** *e pre-registrado* antes de ver os dados.
  - **Por que importa:** o one-sided tem **mais poder** para detectar o efeito na direcao esperada, mas exige comprometer-se com a direcao antes — senao vira p-hacking.
  - **Exemplo:** se `stat = 1,9`, two-sided (corte ±1,96) **nao** rejeita; one-sided (corte 1,645) **rejeita**. Mesma evidencia, conclusao diferente.
  - **No projeto:** o claim H2a/H2b e direcional ("TFT < baseline em pinball"), entao Phase B usou **one-sided** corretamente. O gold legacy two-sided e "seguro" mas **perde poder** para um claim direcional.

**6. top-50 aplicado *antes* do teste**
- **O que o doc afirma:** `_select_top_configs_for_pairwise(g, max_configs=50)` roda antes de montar a matriz.
- **Como deveria funcionar (exemplo):** de 200 configs, mantem so as 50 de menor `squared_error` medio **no proprio split de teste** — e so essas entram no DM. (E o problema de inferencia seletiva do item #4.)
- **O que esperar no codigo:** chamada ao filtro logo no inicio do loop de grupos do builder.
- **Ref do doc → codigo:** [`pairwise.py:299`](../../../../src/domain/services/gold_builders/pairwise.py#L299). ✅ **Confere**.

**7. Holm persistido vs. p-value cru no decision_final**
- **O que o doc afirma:** o builder aplica Holm na saida (`gold_dm_pairwise_results` tem `pvalue_adj_holm`), **porem** `_build_model_decision_final` usa `pvalue_two_sided` (cru) para `dm_net_wins`.
- **Como deveria funcionar (exemplo):** existem **duas nocoes de "DM significativo"**: (a) a do parquet, ajustada por Holm; (b) a do decision_final, usando p<0,05 **cru**. Ao revisar, nao confunda — a coluna `dm_net_wins` do artefato final **ignora a correcao de multiplicidade**.
- **O que esperar no codigo:** Holm em `pairwise.py:333`; em `confidence.py` o agregado filtra por `pvalue_two_sided`.
- **Ref do doc → codigo:** [`pairwise.py:333`](../../../../src/domain/services/gold_builders/pairwise.py#L333) + [`confidence.py:605`](../../../../src/domain/services/gold_builders/confidence.py#L605). ✅ **Confere:** L605 le `pvalue_two_sided`, L609 filtra `p >= 0.05` (cru); groupby em L599-601; colunas em L626-628.
- ⚠️ **Ponto de atencao — deveria usar Holm nos dois? O que implica?**
  O Holm controla o **FWER** (probabilidade de ≥1 falso positivo numa **familia** de testes). Rodando DM em todos os pares de configs, a chance de "significancia por sorte" explode; Holm contem isso.

  **Principio:** qualquer numero que va **embasar ou parecer evidencia** deve usar
  p-value corrigido, com a familia bem definida. Se `dm_net_wins` for lido como
  sinal de superioridade, **deveria** usar Holm — caso contrario **superconta**
  vitorias que sao ruido de multiplas comparacoes (e ainda sobre o universo
  enviesado pelo top-50).

  **Mas ha um "depende":** se `dm_net_wins` e **puramente diagnostico/descritivo**
  (como a reinterpretacao do C.0.1 agora afirma — "coluna diagnostica"), entao
  contagem com p cru e toleravel **desde que rotulada como nao-inferencial**. O
  perigo so existe se alguem tratar como evidencia.

  **Implicacoes da inconsistencia atual:** (1) o sistema reporta **dois
  veredictos** de "DM significativo" (parquet ajustado vs decision_final cru);
  (2) se o decision_final for promovido a confirmatorio, a contagem com p cru
  **inflaria** a aparente superioridade; (3) a correcao Holm persistida tambem e
  suspeita (familia sem `split_signature` — item #3). **Nao e necessariamente
  "aplicar Holm em todo lugar"**: e **ser consistente e honesto** sobre o que cada
  numero e. Duas saidas defensaveis em C.0.3: (a) definir **uma familia correta** e
  aplicar Holm em tudo que alimenta claim; ou (b) marcar `dm_net_wins` como
  **diagnostico nao-inferencial** para que o p cru nao seja confundido com evidencia.

### Cross-check — o que NAO esta corretamente indicado/referenciado

Para o DM, **todas as referencias de "Uso atual" e "Implementacao atual localizada" conferem** com o codigo (verifiquei 280, 299, 333 em `pairwise.py` e 599-628 em `confidence.py`). Pontos a registrar:

1. **Nenhuma referencia quebrada.** As linhas-ancora citadas estao corretas.
2. **Atribuicao de funcao correta, mas atencao ao ler:** o campo declara "Funcao: `_compute_dm_pairwise_from_loss_matrix` (67-104)", mas varias evidencias apontam para **outras funcoes/arquivos** (`_pairwise_preprocess:280`, `build():299/333`, `confidence.py:605`). Esta **certo e explicitado** nos bullets — so nao confunda "a funcao localizada" com "onde cada elemento vive".
3. **Subtileza Holm cru vs. ajustado (elemento 7)** e a coisa mais importante a carregar para C.0.3 — nao e erro de referencia, mas e o tipo de coisa que um leitor desatento do artefato final interpretaria errado. O dossie captou bem.

### Veredito do item #1

🟢 **Integro.** Referencias intactas; evidencia fiel ao codigo. Pontos
metodologicos (loss, lag, HLN/one-sided, Holm) sao decisoes para C.0.2/C.0.3,
nao defeitos de localizacao.

---

## #2 — MCS gold

### O que e (didatico)

O **Model Confidence Set (MCS)** e um procedimento iterativo de eliminacao de modelos. Dado
um conjunto de M modelos e uma funcao de loss, o MCS identifica o menor subconjunto de
modelos que *nao pode ser rejeitado estatisticamente como contendo o melhor modelo*, ao
nivel de significancia alpha. O algoritmo repete: testa se todos os modelos ativos sao
"igualmente bons" via estatistica de range (TR = max dos t-stats pairwise); se a hipotese
nula for rejeitada (p < alpha), elimina o modelo com maior perda media e continua. Para
quando o conjunto remanescente nao pode ser rejeitado → esse e o *confidence set*.

O truque central: ao contrario do DM (que compara dois modelos por vez), o MCS responde
a pergunta **coletiva** "qual e o conjunto minimo de modelos estatisticamente superiores?"
Pertencer ao MCS significa "nao foi possivel rejeitar este modelo como inferior ao melhor"
— **nao** que ele venceu. A dependencia temporal da serie de losses e tratada via
**block bootstrap** (em vez do kernel HAC do DM).

### Elementos

**1. Loss = squared_error (herdada via `_pairwise_preprocess`)**
- **O que o doc afirma:** MCS e alimentado pela mesma loss_matrix que o DM — squared_error
  calculada em `_pairwise_preprocess` e pivotada por `target_timestamp_utc`.
- **Como deveria funcionar (exemplo):** se o modelo preve 102 e o real e 100, a loss e
  `(102-100)^2 = 4`. A loss_matrix e uma tabela timestamps × configs. O MCS usa essa
  matriz para decidir quais configs pertencem ao confidence set.
- **O que esperar no codigo:** `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2`
  em `_pairwise_preprocess`; a loss_matrix e o pivot dessa coluna.
- **Ref do doc → codigo:** [`pairwise.py:280`](../../../../src/domain/services/gold_builders/pairwise.py#L280).
  ✅ **Confere:** `df["squared_error"] = (df["y_pred"] - df["y_true"]) ** 2`.
- ⚠️ **Ponto de atencao — loss desalinhada do claim probabilistico**
  Identico ao item #1 (DM): squared_error mede acuracia do ponto previsto, nao qualidade
  distribucional. O MCS com squared_error responde "qual conjunto de modelos nao pode ser
  rejeitado como inferior em erro pontual?" — pergunta diferente de "qual conjunto e
  melhor probabilisticamente?".

  Para o TCC que reivindica superioridade do TFT em predicao probabilistica, a loss coerente
  seria pinball (q10/q50/q90) ou CRPS/WIS. A Phase B usou pinball_loss_post_guardrail no DM;
  o gold MCS usa squared_error. Reportar "TFT esta no confidence set superior" usando
  squared_error nao sustenta claim probabilistico — sustenta claim pontual.

**2. Matriz de losses por timestamp (estrutura de entrada)**
- **O que o doc afirma:** mesmo padrao do DM — pivot wide por `target_timestamp_utc`,
  seguido de `dropna(how="any")` (herdado de `_pairwise_preprocess` + builder).
- **Como deveria funcionar (exemplo):** com 3 modelos A, B, C e 50 timestamps: so os
  timestamps em que os 3 previram entram na loss_matrix (intersecao de cobertura). Se A
  tem 50, B tem 45, C tem 48 → apenas os ~43 timestamps comuns sao usados. O MCS ve
  n_obs = 43 para todos os pares simultaneamente.
- **O que esperar no codigo:** `pivot(index="target_timestamp_utc", columns="config_label",
  values="squared_error")` + `dropna(axis=0, how="any")`.
- **Ref do doc → codigo:** [`pairwise.py:353-360`](../../../../src/domain/services/gold_builders/pairwise.py#L353).
  ✅ **Confere:** linhas 352-360 replicam exatamente o padrao do DM builder (pivot + dropna).

**3. top-50 aplicado antes do MCS**
- **O que o doc afirma:** `_select_top_configs_for_pairwise(g, max_configs=50)` aplicado
  antes de montar a loss_matrix (linha 351).
- **Como deveria funcionar (exemplo):** de 200 configs possiveis, so as 50 com menor
  squared_error medio no test split entram no MCS — o universo ja e filtrado antes de
  qualquer estatistica ser computada. O confidence set resultante e relativo a esse
  universo reduzido, nao ao universo original.
- **O que esperar no codigo:** chamada ao filtro logo no inicio do loop de grupos do builder.
- **Ref do doc → codigo:** [`pairwise.py:351`](../../../../src/domain/services/gold_builders/pairwise.py#L351).
  ✅ **Confere:** `g = _select_top_configs_for_pairwise(g, max_configs=50)`.

**4. Parametros hard-coded: B=300, block_len=5, random_seed=42**
- **O que o doc afirma:** alpha=0.05; bootstrap_samples=300; block_len=5; random_seed=42.
- **Como deveria funcionar (exemplo):** B=300 significa que o p-value bootstrap tem
  resolucao minima de 1/300 ≈ 0.0033. Para alpha=0.05, a variancia do estimador e
  `p(1-p)/B ≈ 0.05 × 0.95 / 300 ≈ 0.00016`, ou seja desvio-padrao ≈ 0.012. Um "p
  verdadeiro" de 0.045 pode ser estimado como 0.033 ou 0.057 dependendo do seed → modelo
  pode entrar ou sair do confidence set por acidente do Monte Carlo.
- **O que esperar no codigo:** os quatro parametros como defaults na assinatura da funcao.
- **Ref do doc → codigo:** [`pairwise.py:109-113`](../../../../src/domain/services/gold_builders/pairwise.py#L109).
  ✅ **Confere:** `alpha=0.05`, `bootstrap_samples=300`, `block_len=5`, `random_seed=42`
  como defaults; chamada em `McsResultsGoldBuilder.build()` (linha 363) sem kwargs →
  todos os defaults sao usados.
- ⚠️ **Ponto de atencao — B=300 e baixo; block_len=5 nao esta justificado**

  **B=300 (numero de amostras bootstrap)**

  A variancia do estimador do p-value e `p(1-p)/B`. Com B=300 e p ≈ 0.05:
  - IC 95% ≈ [0.026, 0.074] — a faixa inclui tanto rejeicao quanto nao-rejeicao em alpha=0.05.
  - Para modelos com performance proxima (os mais interessantes para o TCC), o p-value
    estara perto de 0.05 exatamente onde B=300 tem maior incerteza.
  - O `random_seed=42` garante **reproducibilidade** (mesmos dados → mesmo resultado),
    mas nao elimina a instabilidade: com outro seed igualmente valido, o conjunto selecionado
    poderia ser diferente.
  - O skeleton cita B >= 1000 (preferencialmente 5000) como criterio para promocao a
    confirmatorio. Mesmo para uso descritivo estavel, B=300 e a fronteira inferior.

  **block_len=5 (tamanho do bloco de bootstrap)**

  O block bootstrap (Kunsch 1989) e consistente apenas se `block_len → ∞` mais lentamente
  que `n_obs → ∞`. O choice de block_len afeta diretamente a estimativa da variancia de
  `dbar` e portanto o p-value:

  | Escolha de block_len | Quando razoavel | Risco |
  |---|---|---|
  | 5 fixo (atual) | n_obs ~20-50, serie de baixa autocorrelacao | Subestima dependencia quando n_obs grande ou serie persistente |
  | n^(1/3) automatico | Regra analoga ao HAC do DM | Varia com a amostra, mais adaptativo |
  | Politis-White 2004 (data-driven) | Caso geral | Mais defensavel; mais complexo de implementar |

  Com block_len muito pequeno: o bootstrap subestima a autocorrelacao → subestima a
  variancia de `dbar` → estatistica TR artificialmente grande → **rejeita H0 mais facilmente
  → confidence set mais restrito** (menos modelos sobrevivem). Se os modelos TFT e baseline
  fossem dificeis de separar com block_len correto mas ficassem separados com block_len=5,
  o claim de superioridade seria artefato do parametro.

  Sem justificativa documental para block_len=5 (sem referencia a autocorrelacao observada
  nas series de losses, sem estimativa data-driven), o parametro e um hardcode de
  conveniencia. Para promocao a confirmatorio, a regra de block_len precisa ser declarada
  e justificada antes de ver os resultados.

**5. Moving block bootstrap com wrap-around**
- **O que o doc afirma:** moving block bootstrap com wrap-around (linhas 124-130).
- **Como deveria funcionar (exemplo):** com n_obs=10 e block_len=5, o resampling sorteia
  um inicio aleatorio — digamos 7. O bloco seria os indices [7, 8, 9, 0, 1] (modulo 10:
  wrap-around circular). Isso evita blocos "cortados" no final da serie. Repete ate
  acumular n_obs observacoes.
- **O que esperar no codigo:** `(start + o) % n_obs` dentro da geracao de blocos; loop
  `while len(idx) < n_obs`.
- **Ref do doc → codigo:** [`pairwise.py:124-130`](../../../../src/domain/services/gold_builders/pairwise.py#L124).
  ✅ **Confere:** `block = [(start + o) % n_obs for o in range(block_len)]`; loop
  `while len(idx) < n_obs`.

**6. Estimador de variancia: `np.var(boot, ddof=1)` + guard 1e-12**
- **O que o doc afirma:** (implicito no dossier; o skeleton menciona "range t-stat" sem
  detalhar o estimador de variancia).
- **Como deveria funcionar (exemplo):** para um par (A, B), `dbar[A,B]` e a media das
  diferencas de losses no sample real. A variancia de `dbar[A,B]` e estimada pela dispersao
  das `dbar_{A,B}` bootstrap: `var[A,B] = Var(boot[:, A, B])` com ddof=1. O guard
  `var <= 1e-12 → nan` evita divisao por zero quando dois modelos tem losses identicas.
- **O que esperar no codigo:** `var = np.var(boot, axis=0, ddof=1)` seguido de
  `var[var <= 1e-12] = np.nan`.
- **Ref do doc → codigo:** [`pairwise.py:147-148`](../../../../src/domain/services/gold_builders/pairwise.py#L147).
  ✅ **Confere:** exatamente `var = np.var(boot, axis=0, ddof=1)` e
  `var[var <= 1e-12] = np.nan`.

  Esta abordagem (bootstrap-estimated variance) e distinta do HAC do DM. No DM, a
  variancia de `mean_d` e estimada por kernel Bartlett sobre a serie temporal. No MCS, e
  estimada pela dispersao das medias bootstrap — o block bootstrap captura a dependencia
  temporal implicitamente via estrutura de bloco. Ambas sao validas nos seus contextos;
  a qualidade do estimador do MCS depende da qualidade do block bootstrap (ver ⚠️ acima).

**7. Estatistica TR (range t-stat): `nanmax|dbar/sqrt(var)|`**
- **O que o doc afirma:** `tr_stat = nanmax(|dbar / sqrt(var)|)` (linha 153).
- **Como deveria funcionar (exemplo):** com 3 modelos ativos {A, B, C} e medias de loss
  {0.5, 0.8, 1.2}: calcula `T_{AB} = |dbar_{AB} / sqrt(var_{AB})|`, `T_{AC}`, `T_{BC}`.
  `TR = max(T_{AB}, T_{AC}, T_{BC})`. Intuitivamente: qual par tem a diferenca de
  performance mais fortemente suportada pelos dados? Se TR for grande e o bootstrap
  confirmar (p < alpha), ha evidencia de que pelo menos um modelo e inferior.
- **O que esperar no codigo:** `tmat = np.abs(dbar / np.sqrt(var))` seguido de
  `tr_stat = float(np.nanmax(tmat))`.
- **Ref do doc → codigo:** [`pairwise.py:150-153`](../../../../src/domain/services/gold_builders/pairwise.py#L150).
  ✅ **Confere:** `tmat = np.abs(dbar / np.sqrt(var))` na linha 150;
  `tr_stat = float(np.nanmax(tmat))` na linha 153.

**8. Bootstrap TR e p-value**
- **O que o doc afirma:** (implicito; o skeleton menciona "range t-stat" e block bootstrap).
- **Como deveria funcionar (exemplo):** cada uma das B=300 amostras bootstrap produz
  `dbar_boot` (media das diferencas no bootstrap sample). A estatistica bootstrap e
  **centrada**: `dbar_boot - dbar_obs` — remove o vies da media. Entao
  `|boot_centered / sqrt(var)|` e `max` sobre todos os pares → `tr_boot[b]`. O p-value
  e `mean(tr_boot >= tr_stat_obs)`. Logica: "qual fracao das amostras bootstrap, sob a
  hipotese nula de igualdade, produz TR tao extremo quanto o observado?"
- **O que esperar no codigo:** `boot_centered = boot - dbar[None, :, :]`; `tboot =
  |boot_centered / sqrt(var)[None, :, :]|`; `tr_boot[b] = nanmax(tboot[b])`; `pvalue =
  mean(tr_boot >= tr_stat)`.
- **Ref do doc → codigo:** [`pairwise.py:154-166`](../../../../src/domain/services/gold_builders/pairwise.py#L154).
  ✅ **Confere:** exatamente esse padrao nas linhas 154-166; `tr_boot` e filtrado para
  finitos (linha 163) antes de computar o p-value.

**9. Criterio de parada e eliminacao**
- **O que o doc afirma:** elimina por `argmax(losses_mean)` (linha 170); para quando
  `pvalue >= alpha`.
- **Como deveria funcionar (exemplo):** passo corrente com 3 modelos {A, B, C} com medias
  {0.5, 0.8, 1.2}: se p < 0.05 → eliminar C (maior mean_loss no subset ativo); repetir
  com {A, B}. Se agora p >= 0.05 → parar → MCS = {A, B}. Interpretacao: "nao ha
  evidencia estatistica para distinguir A de B entre si; C foi rejeitado como inferior."
- **O que esperar no codigo:** `losses_mean = np.mean(sub, axis=0)`;
  `worst_local = int(np.argmax(losses_mean))`; `active.pop(worst_local)`.
- **Ref do doc → codigo:** [`pairwise.py:167-171`](../../../../src/domain/services/gold_builders/pairwise.py#L167).
  ✅ **Confere:** `if pvalue >= alpha or not np.isfinite(pvalue): break` na linha 167;
  `losses_mean = np.mean(sub, axis=0)` na linha 169; `worst_local = int(np.argmax(
  losses_mean))` na linha 170; `active.pop(worst_local)` na linha 171.

**10. Saida: `selected_in_mcs_alpha_0_05` e split_signature propagada condicionalmente**
- **O que o doc afirma:** saida lista `config_label`, `selected_in_mcs_alpha_0_05`,
  `mean_loss`; `_pairwise_group_cols` inclui `split_signature` quando a coluna existe
  (pairwise.py:25-29); o MCS propaga essa chave na saida (pairwise.py:368-374).
- **Como deveria funcionar (exemplo):** config A tem `selected_in_mcs_alpha_0_05 = True`
  e mean_loss = 0.5; config C tem `False` e mean_loss = 1.2. "True" significa: A nao
  pôde ser rejeitado como pertencente ao melhor grupo — **nao** que A venceu.
  `split_signature` e adicionada ao output quando estava presente nos dados de entrada.
- **O que esperar no codigo:** output com 3 colunas base (config_label,
  selected_in_mcs_alpha_0_05, mean_loss); builder adiciona asset, parent_sweep_id,
  split_signature (condicional), split, horizon, aligned_timestamps, n_configs.
- **Ref do doc → codigo:** [`pairwise.py:174-179`](../../../../src/domain/services/gold_builders/pairwise.py#L174)
  (saida base) e [`pairwise.py:368-374`](../../../../src/domain/services/gold_builders/pairwise.py#L368)
  (split_signature condicional). ✅ **Confere:** linhas 174-179 saida com 3 colunas base;
  linhas 368-374 propagacao condicional: `if "split_signature" in group_cols: mcs_df[
  "split_signature"] = keys[2]`.
- ⚠️ **Ponto de atencao — "selecionado" != "vencedor": risco de interpretacao do artefato**
  O nome `selected_in_mcs_alpha_0_05` e tecnicamente correto, mas a leitura coloquial
  como "aprovado/vencedor" e um erro comum na literatura e em relatorios. O MCS tem uma
  assimetria importante:

  - **selected=True:** "nao foi possivel rejeitar este modelo como inferior ao melhor com
    os dados disponiveis" → sobreviveu as eliminacoes, mas o set pode conter modelos muito
    diferentes entre si.
  - **selected=False:** "foi rejeitado (p < alpha)" → ha evidencia de que e pior que algum
    modelo no set remanescente.

  Cenarios relevantes para o TCC:

  | Cenario | MCS diz | Interpretacao correta | Erro tipico |
  |---|---|---|---|
  | TFT e baseline ambos `True` | "Nao distinguivel" | Nao ha evidencia de superioridade do TFT via MCS | "TFT aprovado pelo MCS" (omite que baseline tambem passou) |
  | Apenas TFT `True` | TFT sobreviveu; baseline rejeitado | Evidencia de superioridade do TFT em squared_error | Correto — mas loss desalinhada (ver elemento 1) |
  | Confidence set com 40/50 configs `True` | Pouca potencia | MCS nao discriminou; falta de poder, nao equivalencia | "40 modelos excelentes" |

  O risco pratico: reportar "`mcs_selected_alpha_0_05 = True`" sem mencionar o tamanho do
  confidence set e se o baseline tambem esta incluido. Para o TCC, a interpretacao deve
  ser: "o confidence set a 5% contem os modelos {X, Y, Z}; os demais foram rejeitados como
  estatisticamente inferiores ao melhor com base em squared_error."

**11. Consumo em confidence.py: split_signature descartada no merge**
- **O que o doc afirma:** `selected_in_mcs_alpha_0_05` e renomeado para
  `mcs_selected_alpha_0_05` e mergeado em `gold_model_decision_final` como coluna
  diagnostica (confidence.py:633-647). O dossier aponta: "se `split_signature` nao vier
  populada do upstream, a familia MCS perde essa chave."
- **Como deveria funcionar (exemplo):** se `gold_mcs_results` contem duas linhas para
  config A — `split_signature="fold1_test"` com selected=True e `"fold2_test"` com
  selected=False — o merge deveria preservar essa granularidade ou agregar com regra
  explicita (ex.: `any()` ou `all()` sobre split_signatures).
- **O que esperar no codigo:** o consumer em `confidence.py` propagaria `split_signature`
  ou agregaria explicitamente antes do merge.
- **Ref do doc → codigo:** [`confidence.py:633-647`](../../../../src/domain/services/gold_builders/confidence.py#L633).
  ✅ **Confere a selecao de colunas** — linhas 642-644 selecionam explicitamente:
  `["asset", "parent_sweep_id", "split", "horizon", "config_label",
  "selected_in_mcs_alpha_0_05"]`. Depois, na linha 727:
  `out.merge(df.drop_duplicates(merge_cols), ...)` com `merge_cols = ["asset",
  "parent_sweep_id", "split", "horizon", "config_label"]`.
- ⚠️ **Ponto de atencao — split_signature propagada pelo builder mas descartada pelo consumer**
  O dossier descreve o risco como condicional ("se nao vier populada do upstream").
  O codigo revela algo mais especifico: a chave **e propagada pelo builder**
  (pairwise.py:368-374) e esta em `gold_mcs_results`, mas o **consumer a descarta
  ativamente** (confidence.py:642-644) — independentemente do que vem do upstream.

  O impacto pratico e em duas etapas:

  1. **Descarte de split_signature (linhas 642-644):** `mcs_summary` nao tem
     `split_signature`; qualquer informacao sobre qual split gerou aquele veredicto MCS
     e perdida.
  2. **drop_duplicates sem split_signature (linha 727):** se existirem multiplas linhas
     por `(asset, parent_sweep_id, split, horizon, config_label)` — uma por split_signature
     — o `drop_duplicates(merge_cols)` as colapsa em uma linha, escolhendo
     **arbitrariamente** qual `mcs_selected_alpha_0_05` fica (a que aparece primeiro no
     DataFrame apos o `.copy()` da linha 644).

  Quando importa: em sweeps com multiplos splits de test (walk-forward com varios folds),
  um config pode ser "selecionado" em alguns split_signatures e "nao selecionado" em
  outros. O `gold_model_decision_final.mcs_selected_alpha_0_05` vai refletir apenas um
  deles — nao necessariamente o mais representativo. Isso e nao-deterministico em relacao
  a ordenacao do DataFrame.

  Comparacao com o DM: o DM em `confidence.py:599-600` tambem agrupa por `["asset",
  "parent_sweep_id", "split", "horizon"]` sem split_signature — mas la o resultado e uma
  **soma de vitorias/derrotas** (via groupby + iterrows), o que e uma agregacao implicita.
  No MCS, o colapso e via `drop_duplicates`, que nao agrega, apenas escolhe uma linha.

### Cross-check — o que NAO esta corretamente indicado/referenciado

Para o MCS, as referencias de "Implementacao atual localizada" sao precisas e conferem
com o codigo. Pontos a registrar:

1. **Sem referencias quebradas.** As linhas-ancora citadas estao corretas:
   - Funcao `_compute_mcs_from_loss_matrix`: linhas 107-180 ✅ (confirmo por leitura).
   - Parametros alpha/B/block_len/seed: linhas 109-113 ✅ exatos.
   - Block bootstrap: linhas 124-130 ✅ exatos.
   - TR stat: linha 153 ✅ exata (`float(np.nanmax(tmat))`).
   - Eliminacao: linha 170 ✅ exata (`int(np.argmax(losses_mean))`).
   - Top-50: linha 351 ✅ exato.
   - split_signature: linhas 25-29 (group_cols) ✅ e linhas 368-374 (propagacao) ✅.

2. **Risco de split_signature descrito como upstream, mas o drop e downstream.**
   O dossier diz: "Se `split_signature` nao vier populada do upstream, a familia MCS perde
   essa chave." O codigo mostra que o drop acontece **no consumer** (`confidence.py:642-644`),
   nao no upstream. O builder propaga corretamente; e o `_build_model_decision_final` que
   descarta. A descricao do risco no dossier esta incompleta (nao e bug de localizacao —
   as linhas citadas estao certas — mas o mecanismo do risco e diferente do descrito).

3. **`drop_duplicates(merge_cols)` sem split_signature (confidence.py:727) nao esta
   mencionado no dossier.** Quando ha multiplos split_signatures, o colapso e
   nao-deterministico em relacao ao valor de `mcs_selected_alpha_0_05` persistido no
   artefato final. Isso e mais especifico do que "a chave se perde."

4. **Ponto residual metodologico central: B=300 e block_len=5** sao os riscos mais
   relevantes para promoção a confirmatorio. O skeleton capturou ambos ("B=300 baixo;
   sensibilidade a block_len; loss desalinhada"), mas sem detalhar o mecanismo
   quantitativo. Os aprofundamentos acima complementam.

### Veredito do item #2

🟡 **Ressalvas.** Referencias intactas; evidencia fiel ao codigo. Dois pontos
que o dossier nao captura com precisao suficiente: (a) o descarte de
`split_signature` ocorre no consumer (`confidence.py:642-644`), nao e
condicional ao upstream — e ativo e independente; (b) o `drop_duplicates`
na linha 727 pode colapsar registros de split_signatures distintos de forma
nao-deterministica. Riscos metodologicos (B=300, block_len=5, loss desalinhada)
sao decisoes para C.0.2/C.0.3, nao defeitos de localizacao.

---

