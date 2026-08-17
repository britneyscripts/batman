import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    # Caminhos relativos a partir da pasta EDA/
    input_file = os.path.join("..", "data", "data_raw", "energy_PJM_Load_hourly.csv")
    output_dir = os.path.join("..", "data", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Iniciando EDA para o arquivo: {input_file}")
    
    if not os.path.exists(input_file):
        # Fallback para caminho a partir da raiz (caso o script seja rodado da raiz)
        input_file = os.path.join("data", "data_raw", "energy_PJM_Load_hourly.csv")
        output_dir = os.path.join("data", "data_processed")
        
    # 1. Carregamento dos dados
    df = pd.read_csv(input_file)
    
    # 2. Análise Estrutural e Sumarização
    summary = []
    summary.append(f"=== RELATÓRIO DE EDA: energy_PJM_Load_hourly.csv ===")
    summary.append(f"Número de Linhas: {df.shape[0]}")
    summary.append(f"Número de Colunas: {df.shape[1]}")
    summary.append("\n=== TIPOS DE DADOS E VALORES AUSENTES ===")
    
    null_counts = df.isnull().sum()
    for col in df.columns:
        null_percent = (null_counts[col] / len(df)) * 100
        summary.append(f" - {col}: {df[col].dtype} | Nulos: {null_counts[col]} ({null_percent:.2f}%)")
        
    summary.append("\n=== ESTATÍSTICA DESCRITIVA ===")
    summary.append(df.describe(include='all').to_string())
    
    # Salvar relatório textual em data_processed
    report_name = "eda_report_energy_PJM_Load_hourly.txt"
    report_path = os.path.join(output_dir, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"  [OK] Relatório salvo em: {report_path}")
    
    # 3. Geração de Gráficos de Visualização
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'datetime' in col.lower() or col == 'ds']
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    if date_cols and len(num_cols) > 0:
        # Se for série temporal, faz plot de linha
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df_sorted = df.sort_values(date_col)
        
        target_col = num_cols[0]
        plt.figure(figsize=(12, 6))
        plt.plot(df_sorted[date_col], df_sorted[target_col], color='#1a73e8', linewidth=1.5, label=target_col)
        plt.title(f"Série Temporal: {target_col} vs {date_col} (energy_PJM_Load_hourly.csv)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(date_col, fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        chart_name = "eda_chart_energy_PJM_Load_hourly.png"
        chart_path = os.path.join(output_dir, chart_name)
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Gráfico temporal salvo em: {chart_path}")
        
    elif len(num_cols) > 0:
        # Se não for série temporal, faz plot de distribuição (Histograma / KDE)
        target_col = num_cols[0]
        plt.figure(figsize=(10, 5))
        sns.histplot(df[target_col], kde=True, color='#1a73e8', bins=30)
        plt.title(f"Distribuição de {target_col} (energy_PJM_Load_hourly.csv)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel("Frequência", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        chart_name = "eda_chart_energy_PJM_Load_hourly.png"
        chart_path = os.path.join(output_dir, chart_name)
        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  [OK] Gráfico de distribuição salvo em: {chart_path}")

if __name__ == "__main__":
    run_eda()
