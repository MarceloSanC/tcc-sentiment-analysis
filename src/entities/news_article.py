# src/entities/news_article.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """
    Entidade de domínio que representa uma notícia financeira associada a um ativo.

    Esta classe é intencionalmente simples e imutável, pois:
    - Não contém lógica de negócio
    - Não depende de APIs externas
    - Pode ser usada de forma segura em pipelines paralelos
    """

    asset_id: str
    published_at: datetime
    headline: str
    summary: str
    source: str

    # Metadados opcionais (extensões futuras)
    url: Optional[str] = None
    article_id: Optional[str] = None
    language: Optional[str] = "en"


# =========================
# TODOs — melhorias futuras
# =========================

# TODO (Feature Engineering):
# Adicionar classificação do tipo da notícia:
# ex: earnings, macro, guidance, legal, ESG
# Pode ser usada como variável categórica no TFT
