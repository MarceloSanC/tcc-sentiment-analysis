# src/entities/feature_set.py
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class FeatureSet:
    """
    Representa um snapshot imutável de features calculadas
    para um ativo em um timestamp específico.

    Entidade de domínio:
    - Não conhece pandas, numpy ou ML frameworks
    - Não sabe como as features foram calculadas
    """

    asset_id: str
    timestamp: datetime
    features: Mapping[str, float]

    def __post_init__(self):
        # TODO(validation): validar tipos e valores finitos
        if not self.features:
            raise ValueError("FeatureSet cannot be empty")
        

# =========================
# TODOs — melhorias futuras
# =========================

# TODO(validation):
# Validar que todos os valores de features são finitos
# (não NaN, inf ou -inf)

# TODO(modeling):
# Introduzir ValueObject para FeatureName
# evitando uso de strings livres

# TODO(architecture):
# Tornar FeatureSet versionável
# (ex: feature_schema_version)

# TODO(reproducibility):
# Incluir metadados opcionais:
# - origem (technical, sentiment, static)
# - parâmetros de cálculo