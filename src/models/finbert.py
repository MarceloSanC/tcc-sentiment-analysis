import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class FinBERT:
    def __init__(self, caminho_raw, caminho_saida, model_name="yiyanghkust/finbert-tone"):
        self.caminho_raw = caminho_raw
        self.caminho_saida = caminho_saida
        os.makedirs(self.caminho_saida, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def processar_arquivos(self, coluna_texto="Título"):
        arquivos = [f for f in os.listdir(self.caminho_raw) if f.endswith(".csv")]

        for nome_arquivo in arquivos:
            caminho_csv = os.path.join(self.caminho_raw, nome_arquivo)
            print(f"Processando: {caminho_csv}")

            df = pd.read_csv(caminho_csv)

            if coluna_texto not in df.columns:
                print(f"Aviso: arquivo '{nome_arquivo}' não possui coluna '{coluna_texto}'. Pulando.")
                continue

            sentimentos = []
            for texto in tqdm(df[coluna_texto].astype(str), desc="Classificando"):
                try:
                    resultado = self.sentiment_pipeline(texto)[0]
                    sentimentos.append(resultado["label"])  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
                except Exception as e:
                    print(f"Erro ao processar linha: {texto} -> {e}")
                    sentimentos.append("ERRO")

            df["Sentimento"] = sentimentos

            caminho_arquivo_saida = os.path.join(self.caminho_saida, nome_arquivo)
            df.to_csv(caminho_arquivo_saida, index=False, encoding="utf-8-sig")
            print(f"Salvo em: {caminho_arquivo_saida}")
