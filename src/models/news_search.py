import os
import requests
import pandas as pd
from datetime import datetime, timedelta

class NewsSearch:
    """
    Classe para buscar, filtrar e salvar notícias de ações e ativos usando a API da Finnhub.
    """

    def __init__(self):
        # 🔐 Chave da API da Finnhub (substitua pela sua se necessário)
        self.API_KEY = "d0ls2p9r01qpni3125ngd0ls2p9r01qpni3125o0"
        
        # 📆 Número de dias passados a considerar na busca de notícias
        self.DIAS = 7

        # 📊 Lista de tickers de interesse
        self.TICKERS = [
            "AAPL",        # Apple
            "MSFT",        # Microsoft
            "TSLA",        # Tesla
            "GOOGL",       # Google
            "NVDA",        # Nvidia
        ]

    def buscar_noticias(self, ticker, data_inicio, data_fim):
        """
        Realiza a chamada à API da Finnhub e retorna as notícias para o ticker especificado.
        """
        if ticker.startswith("BINANCE:") or ticker.startswith("COINBASE:"):
            url = "https://finnhub.io/api/v1/crypto-news"
            params = { "token": self.API_KEY }

        elif ticker == "SP500":
            url = "https://finnhub.io/api/v1/news"
            params = { "category": "general", "token": self.API_KEY }

        else:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": data_inicio,
                "to": data_fim,
                "token": self.API_KEY
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

    def carregar_csv_existente(self, caminho_arquivo):
        """
        Carrega um CSV existente com notícias, se disponível.
        """
        if os.path.exists(caminho_arquivo):
            try:
                return pd.read_csv(caminho_arquivo, parse_dates=["Data"])
            except Exception as e:
                print(f"Erro ao carregar o arquivo existente: {e}")
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    def filtrar_novas_noticias(self, novas_noticias, df_existente):
        """
        Filtra notícias que já existem no CSV anterior.
        """
        if df_existente.empty:
            return novas_noticias

        ultima_data = df_existente["Data"].max()
        noticias_filtradas = []

        for noticia in novas_noticias:
            data_noticia = datetime.fromtimestamp(noticia["datetime"])
            if data_noticia > ultima_data:
                noticias_filtradas.append(noticia)

        return noticias_filtradas

    def salvar_noticias_csv(self, noticias, ticker):
        """
        Salva novas notícias no CSV, adicionando ao final ou criando o arquivo se necessário.
        """
        if not noticias:
            print(f"Nenhuma nova notícia para {ticker}.")
            return

        caminho_arquivo = f"C://Users//Marcelo//Documents//Code//tcc-sentiment-analysis//data//raw//noticias_{ticker}.csv"
        df_existente = self.carregar_csv_existente(caminho_arquivo)
        noticias_filtradas = self.filtrar_novas_noticias(noticias, df_existente)

        if not noticias_filtradas:
            print(f"Nenhuma nova notícia para {ticker}.")
            return

        dados = []
        for noticia in noticias_filtradas:
            data = datetime.fromtimestamp(noticia["datetime"]).strftime("%Y-%m-%d %H:%M")
            dados.append([ticker, data, noticia["headline"], noticia["source"], noticia["url"]])

        df_novo = pd.DataFrame(dados, columns=["Ticker", "Data", "Título", "Fonte", "URL"])

        if os.path.exists(caminho_arquivo):
            df_novo.to_csv(caminho_arquivo, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_novo.to_csv(caminho_arquivo, index=False, encoding="utf-8-sig")

        print(f"Arquivo atualizado: {caminho_arquivo}")

    def carregar_dados(self):
        """
        Executa a coleta, filtragem e salvamento de notícias para todos os tickers definidos.
        """
        hoje = datetime.today()
        inicio = (hoje - timedelta(days=self.DIAS)).strftime("%Y-%m-%d")
        fim = hoje.strftime("%Y-%m-%d")

        for ticker in self.TICKERS:
            print(f"\nBuscando notícias de {ticker}...")
            noticias = self.buscar_noticias(ticker, inicio, fim)
            caminho_arquivo = f"C://Users//Marcelo//Documents//Code//tcc-sentiment-analysis//data//raw//noticias_{ticker}.csv"
            df_existente = self.carregar_csv_existente(caminho_arquivo)
            noticias_filtradas = self.filtrar_novas_noticias(noticias, df_existente)
            self.salvar_noticias_csv(noticias_filtradas, ticker)
