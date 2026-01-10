# src/interfaces/candle_repository.py
from abc import ABC, abstractmethod

from src.entities.candle import Candle
from src.entities.daily_sentiment import DailySentiment


class CandleRepository(ABC):
    """
    Interface de persistência para candles financeiros.

    Define o contrato mínimo para:
    - leitura de séries temporais de candles
    - persistência inicial de candles
    - atualização/enriquecimento dos candles com dados derivados
      (ex: sentimento agregado)

    Implementações concretas (Parquet, DB, API, etc.)
    devem respeitar este contrato sem expor detalhes de infraestrutura
    para os Use Cases.
    """

    @abstractmethod
    def load_candles(self, asset_id: str) -> list[Candle]:
        """
        Carrega candles persistidos para um ativo financeiro.

        Args:
            asset_id: Identificador do ativo (ex: AAPL, PETR4)

        Returns:
            Lista de candles ordenados temporalmente.

        Raises:
            FileNotFoundError: se não existir persistência para o ativo
            ValueError: se o schema persistido estiver inválido
        """
        ...

    @abstractmethod
    def save_candles(
        self,
        asset_id: str,
        candles: list[Candle],
    ) -> None:
        """
        Persiste candles para um ativo financeiro.

        Este método é destinado à persistência inicial ou
        sobrescrita completa da série temporal de candles.

        Não deve ser utilizado para enriquecimento incremental
        (ex: inclusão de sentimento).

        Args:
            asset_id: Identificador do ativo (ex: AAPL, PETR4)
            candles: Lista completa de candles a serem persistidos

        Raises:
            ValueError: se a lista de candles estiver vazia
        """
        ...

    @abstractmethod
    def update_sentiment(
        self,
        asset_id: str,
        daily_sentiments: list[DailySentiment],
    ) -> None:
        """
        Atualiza candles existentes com sentimento diário agregado.

        Este método deve:
        - preservar candles já persistidos
        - enriquecer cada candle com métricas de sentimento
          correspondentes à sua data
        - evitar recriação completa da série de candles

        Args:
            asset_id: Identificador do ativo (ex: AAPL, PETR4)
            daily_sentiments: Sentimento agregado por dia

        Raises:
            FileNotFoundError: se não houver candles persistidos
        """
        ...