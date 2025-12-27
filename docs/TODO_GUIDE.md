# TODO Guide

Este documento define o padrão oficial para uso de TODOs no projeto.

## Objetivo dos TODOs

Os TODOs servem para:
- Registrar melhorias técnicas já identificadas
- Preservar decisões arquiteturais para iterações futuras
- Evitar perda de contexto durante o desenvolvimento incremental
- Facilitar priorização técnica consciente

TODOs **não** substituem issues, mas funcionam como *technical breadcrumbs*.

---

## Quando criar um TODO?

Use TODO quando:
- A melhoria é conhecida, mas não crítica no momento
- A mudança impacta arquitetura, ciência de dados ou qualidade
- A implementação pode ser feita sem quebrar contratos existentes

Não use TODO para:
- Bugs críticos
- Código temporário
- Lembretes pessoais vagos

---

## Formato padrão

```python
# TODO: <descrição clara da melhoria>
# TODO (CleanArch): ...
# TODO (Leakage): ...
# TODO (TFT): ...
```

## Checklist melhorias técnicas

### Arquitetura & Clean Architecture

- [ ]  Criar interface `FeatureSetRepository` para persistência desacoplada
- [ ]  Substituir uso de `dict` por `Mapping[str, float]` imutável internamente
- [ ]  Garantir ordenação temporal explícita em todos os `UseCases`
- [ ]  Introduzir `ValueObject` para `AssetId`
- [ ]  Isolar dependências de ML (`sklearn`, `lightgbm`) exclusivamente em adapters
- [ ]  🆕 Persistir scalers por `asset_id` para evitar vazamento cross-asset
- [ ]  🆕 Incluir metadados de normalização no domínio (ex: janela, scaler)
- [ ]  🆕 Separar *Feature Calculators* por tipo (technical, sentiment, static)

---

### Feature Engineering

- [ ]  Criar `LagFeatureCalculator` (lags de preço, volume e indicadores)
- [ ]  Criar `RollingFeatureCalculator` (médias móveis, rolling stats)
- [ ]  Implementar `TargetCalculator` (`log_return`, `direction`, `vol_target`)
- [ ]  Adicionar `SentimentFeatureCalculator` desacoplado do cálculo técnico
- [ ]  Criar `StaticFeatureProvider` (setor, beta, market cap)
- [ ]  🆕 Tornar lista de indicadores configurável (não hardcoded)

---

### Normalização & Data Leakage

- [ ]  Implementar normalização por janela temporal (rolling fit)
- [ ]  Persistir parâmetros de normalização por ativo e feature
- [ ]  Criar teste automatizado de *leakage* temporal
- [ ]  Permitir troca entre `StandardScaler`, `RobustScaler`, `MinMaxScaler`
- [ ]  🆕 Definir política explícita para NaNs
- [ ]  🆕 Validar número mínimo de amostras antes do `fit`
- [ ]  🆕 Avaliar clipping de outliers extremos
- [ ]  🆕 Separar **`fit`** treino vs inferência explicitamente

---

### Validação Estatística

- [ ]  Implementar cálculo automático de VIF
- [ ]  Adicionar PCA opcional com threshold configurável
- [ ]  Gerar relatório automático de correlação entre features
- [ ]  Criar alerta para features altamente colineares
- [ ]  Persistir métricas estatísticas em `reports/`

---

### Testes & Qualidade

- [ ]  Criar testes unitários para cada `FeatureCalculator`
- [ ]  Criar teste de integração do pipeline completo (raw → features → TFT)
- [ ]  Mockar dados históricos para testes determinísticos
- [ ]  Validar consistência temporal (`timestamp` monotônico)
- [ ]  Adicionar validação de schema de saída
- [ ]  🆕 Testar determinismo do pipeline de features

---

### Performance & Escalabilidade

- [ ]  Otimizar cálculos com vetorização adicional
- [ ]  Avaliar uso de `numba` para indicadores customizados
- [ ]  Paralelizar cálculo por ativo
- [ ]  Implementar cache de features intermediárias
- [ ]  Suporte a múltiplos ativos simultâneos

---

### Reprodutibilidade Científica

- [ ]  Versionar datasets processados
- [ ]  Registrar hash dos dados de entrada
- [ ]  Salvar metadados do pipeline (parâmetros, datas, versões)
- [ ]  Criar manifesto de experimento (`experiment_manifest.json`)
- [ ]  Integrar pipeline com MLflow (opcional)