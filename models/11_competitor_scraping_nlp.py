import os
import urllib.request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Aviso: A biblioteca 'beautifulsoup4' não está instalada. Para executar a raspagem de dados, rode: pip install beautifulsoup4")
    BeautifulSoup = None

def run_web_scraping_nlp():
    print("=== CAPTURA DE DADOS (WEB SCRAPING) & MODELAGEM DE TEXTO (NLP) ===")
    
    output_dir = os.path.join("data", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Simulação / Execução de Web Scraping usando BeautifulSoup
    # Tentaremos raspar um HTML estático de exemplo de avaliações de e-commerce
    # Se a internet falhar ou a URL mudar, temos um fallback estruturado.
    url = "https://raw.githubusercontent.com/britneyscripts/batman/main/README.md" # Exemplo de URL estável
    
    reviews_text = []
    
    if BeautifulSoup is not None:
        try:
            print(f"Fazendo requisição GET para obter o conteúdo de: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                html = response.read()
                
            soup = BeautifulSoup(html, 'html.parser')
            # Extrair parágrafos simples do documento raspado para simular o corpus
            paragraphs = [p.get_text().strip() for p in soup.find_all(['p', 'li'])]
            # Filtrar textos curtos ou vazios
            reviews_text = [p for p in paragraphs if len(p) > 20][:15]
            print(f"  [OK] Raspagem concluída com sucesso! Extraídos {len(reviews_text)} textos do HTML.")
        except Exception as e:
            print(f"Aviso de rede: Não foi possível obter a URL externa ({e}). Usando dados locais pré-definidos...")
            
    # Fallback/Corpus local caso a raspagem falhe ou falte a biblioteca
    if not reviews_text:
        reviews_text = [
            "Excelente produto! A entrega foi super rápida e o preço das mídias digitais valeu a pena.",
            "O anúncio do produto no Google Search estava confuso, mas no final deu tudo certo.",
            "Péssimo atendimento na entrega. O suporte demorou semanas para responder no WhatsApp.",
            "Comprei o produto após ver um anúncio incrível no TikTok. Recomendo muito!",
            "O investimento em mídia digital no Meta Ads me convenceu a comprar este modelo de tênis.",
            "Produto bom, preço regular. A visualização de dados das compras está um pouco desatualizada.",
            "Propaganda enganosa no anúncio patrocinado do Google. O desconto era menor do que o prometido.",
            "Experiência incrível de unboxing. O produto superou todas as minhas expectativas técnicas.",
            "Demorou mais de uma semana para chegar. Pelo menos o preço na promoção de Black Friday foi baixo."
        ]
        print("Usando corpus de textos de exemplo locais (Anúncios e Avaliações de Usuários).")

    # 2. Pré-processamento e extração de atributos de texto: Bag of Words (BoW)
    print("\nConstruindo Matriz de Frequência de Palavras (Bag of Words - BoW)...")
    # Removendo stopwords em português básicas de forma direta
    stopwords_pt = ['a', 'o', 'e', 'do', 'da', 'de', 'em', 'um', 'uma', 'os', 'as', 'com', 'para', 'no', 'na', 'foi', 'ao']
    
    bow_vectorizer = CountVectorizer(stop_words=stopwords_pt)
    bow_matrix = bow_vectorizer.fit_transform(reviews_text)
    
    # Criar DataFrame da representação BoW
    df_bow = pd.DataFrame(
        bow_matrix.toarray(), 
        columns=bow_vectorizer.get_feature_names_out()
    )
    
    bow_path = os.path.join(output_dir, "nlp_bag_of_words_matrix.csv")
    df_bow.to_csv(bow_path, index=False)
    print(f"  [OK] Matriz Bag of Words salva em: {bow_path}")
    print(f"  Tamanho da matriz BoW: {df_bow.shape} (Textos x Vocabulário)")

    # 3. Modelagem de importância de termos: TF-IDF
    print("\nConstruindo Matriz TF-IDF (Term Frequency-Inverse Document Frequency)...")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords_pt)
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_text)
    
    # Criar DataFrame da representação TF-IDF
    df_tfidf = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=tfidf_vectorizer.get_feature_names_out()
    )
    
    tfidf_path = os.path.join(output_dir, "nlp_tfidf_matrix.csv")
    df_tfidf.to_csv(tfidf_path, index=False)
    print(f"  [OK] Matriz TF-IDF salva em: {tfidf_path}")
    
    # Mostrar palavras com maiores pontuações médias de TF-IDF no corpus
    mean_tfidf = df_tfidf.mean(axis=0).sort_values(ascending=False)
    print("\nPrincipais termos identificados no corpus (maiores médias de pontuação TF-IDF):")
    print(mean_tfidf.head(10))

if __name__ == "__main__":
    run_web_scraping_nlp()
