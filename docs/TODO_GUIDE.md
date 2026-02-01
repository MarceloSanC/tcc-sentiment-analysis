# TODO Guide

Este documento define o padr?o oficial para uso de TODOs no projeto.

## Objetivo dos TODOs

Os TODOs servem para:
- Registrar melhorias t?cnicas j? identificadas
- Preservar decis?es arquiteturais para itera??es futuras
- Evitar perda de contexto durante o desenvolvimento incremental
- Facilitar prioriza??o t?cnica consciente

TODOs **n?o** substituem issues, mas funcionam como *technical breadcrumbs*.

---

## Quando criar um TODO?

Use TODO quando:
- A melhoria ? conhecida, mas n?o cr?tica no momento
- A mudan?a impacta arquitetura, ci?ncia de dados ou qualidade
- A implementa??o pode ser feita sem quebrar contratos existentes

N?o use TODO para:
- Bugs cr?ticos
- C?digo tempor?rio
- Lembretes pessoais vagos

---

## Formato padr?o

```python
# TODO: <descri??o clara da melhoria>
# TODO (CleanArch): ...
# TODO (Leakage): ...
# TODO (TFT): ...
```

### Exemplos v?lidos:
```
# TODO(data-pipeline):
# Suportar persist?ncia incremental de candles
# (append ou upsert por timestamp)

# TODO(architecture):
# Expor pol?tica expl?cita de persist?ncia:
# overwrite | append | upsert
```

## Checklist melhorias t?cnicas

### Arquitetura & Clean Architecture

- [ ] Criar interface `FeatureSetRepository` para persist?ncia desacoplada
- [ ] Substituir uso de `dict` por `Mapping[str, float]` imut?vel internamente
- [ ] Garantir ordena??o temporal expl?cita em todos os `UseCases`
- [ ] Introduzir `ValueObject` para `AssetId`
- [ ] Isolar depend?ncias de ML (`sklearn`, `lightgbm`) exclusivamente em adapters
- [ ] ?? Persistir scalers por `asset_id` para evitar vazamento cross-asset
- [ ] ?? Incluir metadados de normaliza??o no dom?nio (ex: janela, scaler)
- [ ] ?? Separar *Feature Calculators* por tipo (technical, sentiment, static)

---

### Feature Engineering

- [ ] Criar `LagFeatureCalculator` (lags de pre?o, volume e indicadores)
- [ ] Criar `RollingFeatureCalculator` (m?dias m?veis, rolling stats)
- [ ] Implementar `TargetCalculator` (`log_return`, `direction`, `vol_target`)
- [ ] Adicionar `SentimentFeatureCalculator` desacoplado do c?lculo t?cnico
- [ ] Criar `StaticFeatureProvider` (setor, beta, market cap)
- [ ] ?? Tornar lista de indicadores configur?vel (n?o hardcoded)

---

### Normaliza??o & Data Leakage

- [ ] Implementar normaliza??o por janela temporal (rolling fit)
- [ ] Persistir par?metros de normaliza??o por ativo e feature
- [ ] Criar teste automatizado de *leakage* temporal
- [ ] Permitir troca entre `StandardScaler`, `RobustScaler`, `MinMaxScaler`
- [ ] ?? Definir pol?tica expl?cita para NaNs
- [ ] ?? Validar n?mero m?nimo de amostras antes do `fit`
- [ ] ?? Avaliar clipping de outliers extremos
- [ ] ?? Separar **`fit`** treino vs infer?ncia explicitamente

---

### Valida??o Estat?stica

- [ ] Implementar c?lculo autom?tico de VIF
- [ ] Adicionar PCA opcional com threshold configur?vel
- [ ] Gerar relat?rio autom?tico de correla??o entre features
- [ ] Criar alerta para features altamente colineares
- [ ] Persistir m?tricas estat?sticas em `reports/`

---

### Testes & Qualidade

- [ ] Criar testes unit?rios para cada `FeatureCalculator`
- [ ] Criar teste de integra??o do pipeline completo (raw ? features ? TFT)
- [ ] Mockar dados hist?ricos para testes determin?sticos
- [ ] Validar consist?ncia temporal (`timestamp` monot?nico)
- [ ] Adicionar valida??o de schema de sa?da
- [ ] ?? Testar determinismo do pipeline de features

---

### Performance & Escalabilidade

- [ ] Otimizar c?lculos com vetoriza??o adicional
- [ ] Avaliar uso de `numba` para indicadores customizados
- [ ] Paralelizar c?lculo por ativo
- [ ] Implementar cache de features intermedi?rias
- [ ] Suporte a m?ltiplos ativos simult?neos

---

### Reprodutibilidade Cient?fica

- [ ] Versionar datasets processados
- [ ] Registrar hash dos dados de entrada
- [ ] Salvar metadados do pipeline (par?metros, datas, vers?es)
- [ ] Criar manifesto de experimento (`experiment_manifest.json`)
- [ ] Integrar pipeline com MLflow (opcional)

---

## Backlog fora do MVP

- [ ] Enriquecer candles com sentimento agregado diretamente no parquet de candles
- [ ] Agrega??es adicionais de sentimento (min, max, mediana, trimmed mean)
- [ ] Backtesting e simula??o de portf?lio (sinais, drawdown, Sharpe)
- [ ] Suporte a previs?o multi-horizonte (t+1 ... t+H)
- [ ] M?ltiplos alvos financeiros (retorno, volatilidade, risco)
- [ ] Escalar para m?ltiplos ativos em paralelo
- [ ] Monitoramento de drift e qualidade das previs?es em produ??o
