import os
import time

import pandas as pd

from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries


class MarketDataFetcher:
    """
    Classe para baixar dados históricos de preço, volume e indicadores técnicos
    usando Alpha Vantage API, limitados aos intervalos de datas definidos em daily_average.csv.
    """

    def __init__(self, market_data_dir, daily_avg_path, api_key="V4DFTPBNMGI2UAP2"):
        """
        Inicializa a classe.

        Args:
            market_data_dir (str): Diretório para salvar os arquivos CSV.
            daily_avg_path (str): Caminho para o CSV gerado pela agregação de sentimento
                                     com colunas ['Ticker','Date','Average_Daily_Score'].
            api_key (str): Chave de API do Alpha Vantage.
        """
        self.market_data_dir = market_data_dir
        self.daily_avg_path = daily_avg_path
        self.api_key = api_key

        # Cria pastas de saída
        os.makedirs(self.market_data_dir, exist_ok=True)

        # Inicializa clientes da Alpha Vantage
        self.ts = TimeSeries(key=self.api_key, output_format="pandas")
        self.ti = TechIndicators(key=self.api_key, output_format="pandas")

        # Carrega e processa intervalos de data por ticker
        self.intervals = self._load_intervals()

    def _load_intervals(self):
        """
        Lê o CSV de média diária e extrai, para cada ticker, a menor e maior data.
        Retorna um dict {ticker: (start_datetime, end_datetime)}.
        """
        df = pd.read_csv(self.daily_avg_path, parse_dates=["Date"])
        intervals = {}
        self.tickers = df["Ticker"].unique().tolist()

        for ticker, group in df.groupby("Ticker"):
            if len(group) >= 3:
                start = group["Date"].min()
                end = group["Date"].max()
                intervals[ticker] = (start, end)
        return intervals

    def fetch_and_save(self):
        """
        Para cada ticker, busca preços e indicadores no intervalo definido e salva CSV.
        """
        for ticker in self.tickers:
            if ticker not in self.intervals:
                print(f"[!] Ticker {ticker} não possui intervalo válido. Pulando.")
                continue

            start_date, end_date = self.intervals[ticker]
            print(
                f"\n📥 Coletando dados para {ticker} de {start_date.date()} a {end_date.date()}..."
            )

            try:
                data, _ = self.ts.get_daily(symbol=ticker, outputsize="full")
                data.index = pd.to_datetime(data.index)
                data = data.loc[(data.index >= start_date) & (data.index <= end_date)]
                data.columns = ["Open", "High", "Low", "Close", "Volume"]
            except Exception as e:
                print(f"Erro ao baixar preços para {ticker}: {e}")
                continue

            # Busca indicadores técnicos
            def fetch_indicator(fn, **kwargs):
                try:
                    df_ind, _ = fn(symbol=ticker, **kwargs)
                    df_ind.index = pd.to_datetime(df_ind.index)
                    return df_ind.loc[
                        (df_ind.index >= start_date) & (df_ind.index <= end_date)
                    ]
                except Exception as e:
                    print(f"Erro ao baixar indicador para {ticker}: {e}")
                    return pd.DataFrame()

            rsi = fetch_indicator(self.ti.get_rsi, interval="daily", time_period=14)
            sma = fetch_indicator(self.ti.get_sma, interval="daily", time_period=14)
            ema = fetch_indicator(self.ti.get_ema, interval="daily", time_period=14)

            df = data.join([rsi, sma, ema])
            df.index.name = "Date"
            df.reset_index(inplace=True)

            filename = f"{ticker}_marketdata.csv"
            path = os.path.join(self.market_data_dir, filename)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"✅ Salvado: {path}")

            time.sleep(12)  # respeita limite da API
