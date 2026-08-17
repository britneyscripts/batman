import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_clustering_segmentation():
    print("=== MODELAGEM DE AGRUPAMENTO (K-MEANS) & PCA ===")
    
    # 1. Carregamento dos dados (Exemplo usando metadados de lojas da Rossmann ou Store Sales)
    stores_path = os.path.join("data", "data_raw", "store_sales_stores.csv")
    output_dir = os.path.join("data", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(stores_path):
        print(f"Carregando dados reais de lojas de: {stores_path}")
        df_raw = pd.read_csv(stores_path)
        # Transforma colunas categóricas simples em dummies para simular atributos numéricos de comportamento
        df_encoded = pd.get_dummies(df_raw, columns=['type', 'state'], drop_first=True)
        # Excluir id
        features_cols = [c for c in df_encoded.columns if c not in ['store_nbr', 'city']]
        X = df_encoded[features_cols].astype(float)
    else:
        print("Aviso: Dados reais de lojas não encontrados. Gerando atributos sintéticos de clientes...")
        # Gerar dados sintéticos de comportamento RFM (Recência, Frequência, Valor Monetário) + Acessos
        np.random.seed(42)
        n_customers = 500
        X = pd.DataFrame({
            'recency': np.random.randint(1, 365, n_customers),
            'frequency': np.random.poisson(10, n_customers) + 1,
            'monetary_value': np.random.exponential(250, n_customers) + 15,
            'discount_ratio': np.random.uniform(0.0, 0.4, n_customers),
            'satisfaction_score': np.random.normal(7.5, 1.5, n_customers).clip(1, 10)
        })

    # 2. Padronização dos atributos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Análise de Componentes Principais (PCA)
    # Reduz para 2 componentes principais para fins de visualização 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    print(f"Variância explicada pelo PCA: Componente 1 = {explained_var[0]:.2%}, Componente 2 = {explained_var[1]:.2%}")
    print(f"Variância acumulada total: {sum(explained_var):.2%}")
    
    # 4. Agrupamento com K-Means
    # Determinando o número de clusters (Exemplo fixado em 3 clusters para segmentação de perfis)
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Adicionar labels na base original
    X['cluster'] = cluster_labels
    
    # 5. Salvar resultados e estatísticas médias por cluster
    cluster_summary = X.groupby('cluster').mean()
    summary_path = os.path.join(output_dir, "customer_segmentation_summary.csv")
    cluster_summary.to_csv(summary_path)
    print(f"\nResumo médio das métricas de cada cluster salvo em: {summary_path}")
    print(cluster_summary)
    
    # 6. Plotar e salvar gráfico de dispersão com os Componentes do PCA coloridos por Cluster
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='viridis', alpha=0.8, s=80)
    plt.title("Segmentação de Clusters no Espaço Reduzido do PCA", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"Componente Principal 1 ({explained_var[0]:.1%})", fontsize=12)
    plt.ylabel(f"Componente Principal 2 ({explained_var[1]:.1%})", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Cluster")
    
    chart_path = os.path.join(output_dir, "segmentation_pca_plot.png")
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de dispersão dos clusters salvo em: {chart_path}")

if __name__ == "__main__":
    run_clustering_segmentation()
