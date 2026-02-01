# src/entities/technical_indicator_set.py
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class TechnicalIndicatorSet:
    """
    Snapshot imutável de indicadores técnicos calculados
    para um ativo em um timestamp específico.

    Entidade de domínio:
    - Não conhece pandas, numpy ou ML frameworks
    - Não sabe como os indicadores foram calculados
    """

    asset_id: str
    timestamp: datetime
    indicators: Mapping[str, float]

    def __post_init__(self):
        # TODO(validation): validar tipos e valores finitos
        if not self.indicators:
            raise ValueError("TechnicalIndicatorSet cannot be empty")


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(validation):
# Validar que todos os valores são finitos
# (não NaN, inf ou -inf)

# TODO(modeling):
# Introduzir ValueObject para IndicatorName
# evitando uso de strings livres

# TODO(architecture):
# Tornar TechnicalIndicatorSet versionável
# (ex: indicator_schema_version)

# TODO(reproducibility):
# Incluir metadados opcionais:
# - parâmetros de cálculo
# - versão do pipeline
