import os

import pandas as pd


class SentimentAggregator:
    def __init__(self, inferred_path, results_path):
        # Caminho para o diretório onde estão os arquivos CSV com os sentimentos inferidos
        self.inferred_path = inferred_path
        self.results_path = results_path

        # Dicionário para converter os rótulos do FinBERT em valores numéricos
        self.label_to_score = {
            "Positive": 1,  # Sentimento positivo recebe score 1
            "Neutral": 0,  # Sentimento neutro recebe score 0
            "Negative": -1,  # Sentimento negativo recebe score -1
            "ERRO": 0,  # Se houve erro na inferência, será tratado como neutro
        }

    def load_data(self):
        """
        Carrega todos os arquivos CSV com os sentimentos inferidos.
        Retorna um único DataFrame consolidado com todas as notícias.
        """
        files = [
            f for f in os.listdir(self.inferred_path) if f.endswith(".csv")
        ]  # Lista os arquivos CSV
        all_data = []

        for file in files:
            ticker = file.replace("noticias_", "").replace(
                ".csv", ""
            )  # Extrai o ticker do nome do arquivo
            path = os.path.join(
                self.inferred_path, file
            )  # Caminho completo do arquivo CSV

            try:
                df = pd.read_csv(
                    path, parse_dates=["Date"]
                )  # Lê o CSV, convertendo a coluna 'Date'
                df["Ticker"] = ticker  # Adiciona coluna com o ticker da ação
                all_data.append(df)
            except Exception as e:
                print(
                    f"Erro ao ler {file}: {e}"
                )  # Mostra erro se não conseguir carregar

        # Concatena todos os DataFrames em um único
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()  # Retorna vazio se não houver dados

    def convert_to_score(self, df):
        """
        Converte os rótulos de sentimento textual em valores numéricos usando o dicionário.
        Remove qualquer linha que não pôde ser convertida (ex: valores ausentes).
        """
        df["Sentiment"] = (
            df["Sentiment"].astype(str).str.capitalize()
        )  # Normaliza capitalização
        df["Score"] = df["Sentiment"].map(self.label_to_score)  # Aplica mapeamento
        return df.dropna(subset=["Score"])  # Remove linhas com score nulo

    def aggregate_by_day(self, df):
        """
        Calcula a média de sentimento (score) por dia para cada ticker.
        """
        df["Date"] = pd.to_datetime(df["Date"])  # Garante tipo datetime
        df["Date"] = df["Date"].dt.date  # Mantém apenas a parte da data (sem hora)
        return (
            df.groupby(["Ticker", "Date"])["Score"]
            .mean()
            .reset_index(name="Average_Daily_Score")
        )

    def aggregate_by_biweekly_and_month(self, df):
        """
        Calcula a média de sentimento por quinzena (1-15, 16-31) e por mês completo.
        """
        df["Date"] = pd.to_datetime(df["Date"])  # Garante datetime
        df["YearMonth"] = df["Date"].dt.to_period(
            "M"
        )  # Agrupa por mês/ano (ex: 2025-05)
        df["Day"] = df["Date"].dt.day  # Extrai o dia do mês
        df["Biweekly"] = df["Day"].apply(
            lambda d: "1" if d <= 15 else "2"
        )  # Define quinzena

        # Calcula a média por quinzena
        quinzena = (
            df.groupby(["Ticker", "YearMonth", "Biweekly"])["Score"]
            .mean()
            .reset_index(name="Score_Medio_Quinzena")
        )

        # Calcula a média por mês completo
        mensal = (
            df.groupby(["Ticker", "YearMonth"])["Score"]
            .mean()
            .reset_index(name="Average_Monthly_Score")
        )

        return quinzena, mensal

    def process(self):
        """
        Executa o pipeline completo: carrega os dados, converte os sentimentos em scores,
        calcula médias por dia, quinzena e mês, e salva os resultados em CSV.
        """
        df = self.load_data()  # Etapa 1: carregar os arquivos CSV
        if df.empty:
            print("Nenhum dado encontrado.")  # Se não tiver dados, finaliza
            return

        df = self.convert_to_score(df)  # Etapa 2: converter rótulos em scores

        # Etapa 3: agregações e exibição de resultados
        print("\n[✔] Média por dia:")
        daily_avg = self.aggregate_by_day(df)
        print(daily_avg.head())

        print("\n[✔] Média por quinzena:")
        biweekly_avg, monthly_avg = self.aggregate_by_biweekly_and_month(df)
        print(biweekly_avg.head())

        print("\n[✔] Média por mês:")
        print(monthly_avg.head())

        # Etapa 4: salvar os resultados como CSV
        # Sobrepõe resultados anteriores
        daily_avg.to_csv(os.path.join(self.results_path, "daily_avg.csv"), index=False)
        biweekly_avg.to_csv(
            os.path.join(self.results_path, "biweekly_avg.csv"), index=False
        )
        monthly_avg.to_csv(
            os.path.join(self.results_path, "monthly_avg.csv"), index=False
        )
        print("\n[💾] Resultados salvos em CSV.")
