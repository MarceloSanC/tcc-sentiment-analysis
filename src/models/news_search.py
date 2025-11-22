import os

from datetime import datetime, timedelta

import pandas as pd
import requests


class NewsSearch:
    """
    Classe para buscar, filtrar e salvar notícias de ações e ativos usando a API da Finnhub.
    """

    def __init__(self):
        # 🔐 Chave da API da Finnhub (substitua pela sua se necessário)
        self.API_KEY = "d0ls2p9r01qpni3125ngd0ls2p9r01qpni3125o0"

        # 📆 Número de dias passados a considerar na busca de notícias
        self.DAYS = 60

        # 📊 Lista de tickers de interesse
        self.TICKERS = tickers

        # Caminho para diretorio de notícias extraídas
        self.raw_path = raw_path

    def fetch_news(self, ticker, start_date, end_date):
        """
        Realiza a chamada à API da Finnhub e retorna as notícias para o ticker especificado.
        """
        if ticker.startswith("BINANCE:") or ticker.startswith("COINBASE:"):
            url = "https://finnhub.io/api/v1/crypto-news"
            params = {"token": self.API_KEY}

        elif ticker == "SP500":
            url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "token": self.API_KEY}

        else:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": start_date,
                "to": end_date,
                "token": self.API_KEY,
            }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list) and data:
                    return data
                else:
                    print(f"Nenhuma notícia encontrada para {ticker}.")
                    return []
            except ValueError:
                print(f"Erro ao decodificar JSON para {ticker}.")
                return []
        else:
            print(f"Erro ao buscar {ticker}: {response.status_code}")
            return []

    def load_existing_csv(self, file_path):
        """
        Carrega um CSV existente com notícias, se disponível.
        """
        if os.path.exists(file_path):
            try:
                return pd.read_csv(file_path, parse_dates=["Date"])
            except Exception as e:
                print(f"Erro ao carregar o arquivo existente: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def filter_new_news(self, new_news, df_existing_news):
        """
        Filtra notícias que já existem no CSV anterior.
        """
        if df_existing_news.empty:
            return new_news

        last_date = df_existing_news["Data"].max()
        filtered_news = []

        for news in new_news:
            news_date = datetime.fromtimestamp(news["datetime"])
            if news_date > last_date:
                filtered_news.append(news)

        return filtered_news

    def save_news_to_csv(self, filtered_news, ticker):
        """
        Salva novas notícias no CSV correspondente ao ticker.
        Cria o arquivo se ele não existir, senão adiciona ao final.
        """
        if not filtered_news:
            print(f"Nenhuma nova notícia para {ticker}.")
            return

        file_path = os.path.join(self.raw_path, f"news_{ticker}.csv")

        # Prepara dados para DataFrame
        data = [
            [
                ticker,
                datetime.fromtimestamp(n["datetime"]).strftime("%Y-%m-%d %H:%M"),
                n["headline"],
                n["source"],
                n["url"],
            ]
            for n in filtered_news
        ]

        df_new = pd.DataFrame(
            data, columns=["Ticker", "Date", "Title", "Source", "URL"]
        )

        # Salva, adicionando ou criando o arquivo
        mode = "a" if os.path.exists(file_path) else "w"
        headline = not os.path.exists(file_path)

        df_new.to_csv(
            file_path, mode=mode, header=headline, index=False, encoding="utf-8-sig"
        )
        print(f"✅ Arquivo atualizado: {file_path}")

    def fetch_and_save_news(self):
        """
        Executa a coleta, filtragem e salvamento de notícias para todos os tickers definidos.
        """
        today = datetime.today()
        start_date = (today - timedelta(days=self.DAYS)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")

        for ticker in self.TICKERS:
            print(f"\n🔍 Buscando notícias de {ticker}, de {start_date} a {end_date}")
            noticias = self.fetch_news(ticker, start_date, end_date)

            file_path = os.path.join(self.raw_path, f"noticias_{ticker}.csv")
            df_existente = self.load_existing_csv(file_path)
            filtered_news = self.filter_new_news(noticias, df_existente)

            # Agora só passa as filtradas, o método de salvar lida com ausência
            self.save_news_to_csv(filtered_news, ticker)
