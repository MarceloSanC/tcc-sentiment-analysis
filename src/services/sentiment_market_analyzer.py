import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from scipy.stats import pearsonr


class SentimentMarketAnalyzer:
    """
    Analisa a relação entre sentimento médio diário e retorno do mercado
    em janelas móveis de 3 dias, para múltiplos tickers. Calcula:

    - Correlação de Pearson entre sentimento e retorno.
    - Acurácia direcional entre sinais de sentimento e retorno.
    - Regressão linear OLS: retorno ~ sentimento.

    Também gera gráficos temporais para cada métrica calculada.
    """

    def __init__(self, sentiment_csv, market_data_dir, output_dir):
        """
        Inicializa a classe.

        Args:
            sentiment_csv (str): Caminho para o CSV com colunas ['Ticker','Data','Score_Medio_Diario'].
            market_data_dir (str): Diretório contendo os arquivos *_marketdata_*.csv por ticker.
            output_dir (str): Diretório onde resultados (CSV e gráficos) serão salvos.
        """
        self.sentiment_csv = sentiment_csv
        self.market_data_dir = market_data_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        # Carrega todos os dados de sentimento diário
        self.df_sent = pd.read_csv(sentiment_csv, parse_dates=["Data"])

    def _pearson_corr(self, x, y):
        try:
            return pearsonr(x, y)
        except:
            return None, None

    def _directional_accuracy(self, df_window):
        mask = (df_window["Sent_Signal"] != 0) & (df_window["Ret_Signal"] != 0)
        if mask.sum() > 0:
            return (
                df_window.loc[mask, "Sent_Signal"] == df_window.loc[mask, "Ret_Signal"]
            ).mean()
        return None

    def _ols_metrics(self, x, y):
        try:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            beta = model.params["Score_Medio_Diario"]
            pval = model.pvalues["Score_Medio_Diario"]
            r2 = model.rsquared
            return beta, pval, r2
        except:
            return None, None, None

    def _analisar_ticker(self, ticker, df_sent, df_mkt, output_csv):
        """
        Executa análise de métricas para um único ticker.
        """
        df = pd.merge(df_sent, df_mkt, on="Data", how="inner")
        df.sort_values("Data", inplace=True)

        df["Return"] = (df["Close"] - df["Open"]) / df["Open"]
        df["Sent_Signal"] = df["Score_Medio_Diario"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        df["Ret_Signal"] = df["Return"].apply(
            lambda r: 1 if r > 0 else (-1 if r < 0 else 0)
        )

        resultados = []
        for i in range(2, len(df)):
            janela = df.iloc[i - 2 : i + 1]
            data_ref = df.iloc[i]["Data"]

            x = janela["Score_Medio_Diario"]
            y = janela["Return"]

            corr, pval = self._pearson_corr(x, y)
            acc = self._directional_accuracy(janela)
            beta, beta_pval, r_squared = self._ols_metrics(x, y)

            resultados.append(
                {
                    "Data": data_ref,
                    "PearsonCorr": corr,
                    "PearsonPval": pval,
                    "DirectionalAcc": acc,
                    "Beta": beta,
                    "BetaPval": beta_pval,
                    "R2": r_squared,
                }
            )

        df_result = pd.DataFrame(resultados)
        df_result.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✅ [{ticker}] Resultados salvos em: {output_csv}")

        return df_result

    def _plot_metrics(self, df_result, ticker):
        """
        Gera e salva gráficos de série temporal para cada métrica numérica.
        """
        img_dir = os.path.join(self.output_dir, "timeseries")
        os.makedirs(img_dir, exist_ok=True)

        metricas = [
            "PearsonCorr",
            "PearsonPval",
            "DirectionalAcc",
            "Beta",
            "BetaPval",
            "R2",
        ]
        for metrica in metricas:
            plt.figure(figsize=(10, 4))
            plt.plot(df_result["Data"], df_result[metrica], marker="o", linestyle="-")
            plt.title(f"{metrica} - {ticker}")
            plt.xlabel("Data")
            plt.ylabel(metrica)
            plt.grid(True)
            plt.tight_layout()

            img_path = os.path.join(img_dir, f"{ticker}_{metrica}.png")
            plt.savefig(img_path)
            plt.close()

        print(f"📊 [{ticker}] Gráficos salvos em: {img_dir}")

    def executar(self):
        """
        Executa a análise para todos os tickers presentes em media_diaria.csv,
        usando seus respectivos arquivos de mercado no diretório informado.
        """
        tickers = self.df_sent["Ticker"].unique()

        for ticker in tickers:
            print(f"\n📈 Processando {ticker}...")

            df_sent_ticker = self.df_sent[self.df_sent["Ticker"] == ticker]

            # Procura o arquivo de dados de mercado correspondente
            arquivos = [
                f
                for f in os.listdir(self.market_data_dir)
                if f.startswith(ticker) and f.endswith(".csv")
            ]

            if not arquivos:
                print(f"❌ Arquivo de mercado não encontrado para {ticker}. Pulando.")
                continue

            market_csv = os.path.join(self.market_data_dir, arquivos[0])
            df_mkt = pd.read_csv(market_csv, parse_dates=["Date"])

            output_csv = os.path.join(self.output_dir, f"{ticker}_metrics.csv")

            # Análise e gráficos
            df_result = self._analisar_ticker(
                ticker, df_sent_ticker, df_mkt, output_csv
            )
            self._plot_metrics(df_result, ticker)
