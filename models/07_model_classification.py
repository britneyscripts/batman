import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def run_classification():
    print("=== MODELAGEM DE CLASSIFICAÇÃO & GLM ===")
    
    # 1. Carregamento dos dados (Exemplo usando store_sales ou GA4 transformado)
    # Por padrão, busca a base processada de transações
    data_path = os.path.join("data", "data_processed", "eda_report_store_sales_transactions.txt") # Placeholder
    raw_sales_path = os.path.join("data", "data_raw", "store_sales_train.csv")
    
    print("Aviso: Carregando dados e simulando variável target 'conversao_alta' (Compra > 15 itens)...")
    
    # Para testes sem base externa pesada, geramos um subset simulado
    if os.path.exists(raw_sales_path):
        # Carrega uma pequena amostra para estruturação
        df = pd.read_csv(raw_sales_path, nrows=5000)
        df['target'] = (df['sales'] > df['sales'].median()).astype(int)
        X = df[['onpromotion']].fillna(0)
        y = df['target']
    else:
        # Geração sintética para manter o script funcional
        np.random.seed(42)
        n_samples = 1000
        X = pd.DataFrame({
            'page_views': np.random.poisson(5, n_samples),
            'session_duration': np.random.exponential(120, n_samples),
            'is_mobile': np.random.binomial(1, 0.6, n_samples)
        })
        # Regra de decisão com ruído para o target (conversão/compra)
        y = (0.3 * X['page_views'] + 0.005 * X['session_duration'] + 0.5 * X['is_mobile'] + np.random.normal(0, 1, n_samples) > 1.5).astype(int)

    # 2. Divisão de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Padronização de escala
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Modelagem comparativa (Mapeando os modelos do Bloco de Aprendizado de Máquina & GLM)
    models = {
        "Regressão Logística (GLM)": LogisticRegression(random_state=42),
        "K-Vizinhos Mais Próximos (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Máquina de Vetores de Suporte (SVM)": SVC(probability=True, random_state=42),
        "Classificador Bayesiano (Naive Bayes)": GaussianNB(),
        "Floresta Aleatória (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTreinando {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = model.score(X_test_scaled, y_test)
        auc = roc_auc_score(y_test, probs)
        
        results[name] = {"Acurácia": acc, "AUC-ROC": auc}
        
        print(f"  Acurácia: {acc:.4f} | AUC-ROC: {auc:.4f}")
        print("  Matriz de Confusão:")
        print(confusion_matrix(y_test, preds))
        
    # 4. Salvar resultados consolidados
    output_dir = os.path.join("data", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(output_dir, "classification_model_comparison.csv")
    results_df.to_csv(results_path)
    print(f"\nComparativo salvo com sucesso em: {results_path}")

if __name__ == "__main__":
    run_classification()
