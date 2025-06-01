
from src.models.finbert import FinBERT
from src.models.news_search import NewsSearch

def main():
    print("Iniciando carregamento de dados...")

    buscador = NewsSearch()
    buscador.carregar_dados()

    print("\nCarregamento concluído.")

    caminho_raw = r"C:\Users\Marcelo\Documents\Code\tcc-sentiment-analysis\data\raw"
    caminho_saida = r"C:\Users\Marcelo\Documents\Code\tcc-sentiment-analysis\data\inferred"

    finbert = FinBERT(caminho_raw, caminho_saida)
    finbert.processar_arquivos()

if __name__ == "__main__":
    main()
