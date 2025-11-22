import os

import pandas as pd

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class FinBERT:
    """
    Classe responsável por aplicar análise de sentimento com o modelo FinBERT
    sobre arquivos CSV contendo textos de notícias financeiras.
    """

    def __init__(self, raw_path, output_path, model_name="yiyanghkust/finbert-tone"):
        """
        Inicializa o modelo FinBERT com seus componentes e prepara diretórios.

        Args:
            raw_path (str): Caminho para o diretório com os arquivos de entrada (.csv).
            output_path (str): Caminho onde os arquivos processados serão salvos.
            model_name (str): Nome ou caminho do modelo FinBERT no HuggingFace Hub.
        """
        self.raw_path = raw_path  # Diretório de entrada dos arquivos .csv
        self.output_path = (
            output_path  # Diretório onde os arquivos com sentimento serão salvos
        )

        os.makedirs(
            self.output_path, exist_ok=True
        )  # Cria o diretório de saída se não existir

        # Carrega o tokenizer e o modelo pré-treinado FinBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Cria um pipeline de análise de sentimento com o modelo e tokenizer carregados
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer
        )

    def infer_files(self, text_column="Title"):
        """
        Aplica a inferência de sentimento em todos os arquivos .csv no diretório especificado.

        Args:
            text_column (str): Nome da coluna de texto em que o sentimento será inferido.
        """
        # Lista todos os arquivos .csv no diretório de entrada
        files = [f for f in os.listdir(self.raw_path) if f.endswith(".csv")]

        # Itera sobre cada arquivo CSV encontrado
        for file_name in files:
            csv_path = os.path.join(
                self.raw_path, file_name
            )  # Caminho completo do arquivo
            print(f"Processando: {csv_path}")

            df = pd.read_csv(csv_path)  # Lê o arquivo CSV

            # Verifica se a coluna de texto existe no DataFrame
            if text_column not in df.columns:
                print(
                    f"Aviso: arquivo '{file_name}' não possui coluna '{text_column}'. Pulando."
                )
                continue

            sentiments = []  # Lista para armazenar os rótulos de sentimento

            # Aplica o modelo FinBERT para cada linha da coluna de texto
            for text in tqdm(df[text_column].astype(str), desc="Classificando"):
                try:
                    result = self.sentiment_pipeline(text)[
                        0
                    ]  # Retorna um dicionário: {"label": ..., "score": ...}
                    sentiments.append(
                        result["label"]
                    )  # Adiciona apenas o rótulo (POSITIVE, NEGATIVE ou NEUTRAL)
                except Exception as e:
                    print(f"Erro ao processar linha: {text} -> {e}")
                    sentiments.append("ERRO")  # Marca erro na linha com texto inválido

            # Adiciona a nova coluna "Sentimento" ao DataFrame
            df["Sentiment"] = sentiments

            # Define o caminho de saída para o novo arquivo processado
            output_file_path = os.path.join(self.output_path, file_name)

            # Salva o DataFrame modificado com os sentimentos
            # Sobreescreve arquivo com inferencias já existentes
            # Realiza a inferencia em todas as linhas em caminho_csv
            df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
            print(f"Salvo em: {output_file_path}")
