import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
except ImportError:
    print("Aviso: A biblioteca 'lifelines' não está instalada. Para executar este script de sobrevivência, rode: pip install lifelines")
    KaplanMeierFitter = None
    CoxPHFitter = None

def run_survival_analysis():
    print("=== MODELAGEM DE ANÁLISE DE SOBREVIVÊNCIA (RETENÇÃO/LTV) ===")
    
    # 1. Verificar se a biblioteca lifelines está disponível
    if KaplanMeierFitter is None or CoxPHFitter is None:
        print("Erro: Instale a biblioteca 'lifelines' para rodar as análises estatísticas de sobrevivência.")
        return
        
    output_dir = os.path.join("data", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Gerar dados de exemplo de sobrevivência de clientes (tempo até o churn)
    # n = 200 clientes acompanhados por 24 meses
    np.random.seed(42)
    n_samples = 200
    
    # tenure: tempo (meses) em que o cliente permaneceu ativo (ou tempo de acompanhamento)
    tenure = np.random.geometric(p=0.08, size=n_samples).clip(1, 24)
    # churned: indicador se o evento ocorreu (1 se deu churn, 0 se foi censurado - ainda ativo)
    churned = np.random.binomial(n=1, p=0.75, size=n_samples)
    
    # Variáveis explicativas (covariáveis)
    # cohort: 1 se o cliente veio de canais de mídia paga (Paid), 0 se veio orgânico (Organic)
    is_paid = np.random.binomial(n=1, p=0.5, size=n_samples)
    # ticket_medio: gasto médio mensal
    monthly_spend = np.random.normal(150, 45, size=n_samples).clip(30, 400)
    
    df = pd.DataFrame({
        'tenure': tenure,
        'churned': churned,
        'is_paid': is_paid,
        'monthly_spend': monthly_spend
    })
    
    print("\nVisão inicial do dataset de ciclo de vida do cliente (Sobrevivência):")
    print(df.head())
    
    # 3. Ajustando Estimador Kaplan-Meier (Curva de Sobrevivência Geral)
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df['tenure'], event_observed=df['churned'], label='Sobrevivência Geral')
    
    # Salvar a tabela da curva de sobrevivência em csv
    survival_table_path = os.path.join(output_dir, "survival_kaplan_meier_table.csv")
    kmf.survival_function_.to_csv(survival_table_path)
    print(f"\nTabela de probabilidade de retenção salva em: {survival_table_path}")
    
    # 4. Ajustar modelo de riscos proporcionais de Cox (Regressão multivariada de sobrevivência)
    cph = CoxPHFitter()
    cph.fit(df, duration_col='tenure', event_col='churned')
    
    # Salvar resumo estatístico da regressão de Cox
    summary_path = os.path.join(output_dir, "survival_cox_model_summary.csv")
    cph.summary.to_csv(summary_path)
    print(f"Resumo da Regressão de Cox salvo em: {summary_path}")
    print("\nResultados do Modelo de Riscos Proporcionais de Cox:")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    # 5. Plotar e salvar curvas comparativas (Mídia Paga vs Orgânica)
    plt.figure(figsize=(10, 6))
    
    ax = plt.subplot(111)
    # Kaplan-Meier estratificado por canal de aquisição (Paid vs Organic)
    kmf_paid = KaplanMeierFitter()
    kmf_paid.fit(df[df['is_paid'] == 1]['tenure'], df[df['is_paid'] == 1]['churned'], label='Tráfego Pago')
    kmf_paid.plot_survival_function(ax=ax, color='#ea4335', ci_show=True)
    
    kmf_org = KaplanMeierFitter()
    kmf_org.fit(df[df['is_paid'] == 0]['tenure'], df[df['is_paid'] == 0]['churned'], label='Tráfego Orgânico')
    kmf_org.plot_survival_function(ax=ax, color='#34a853', ci_show=True)
    
    plt.title("Curva de Retenção de Clientes (Kaplan-Meier) por Canal de Aquisição", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Tempo de Vida (Meses)", fontsize=12)
    plt.ylabel("Probabilidade de Sobrevivência (Retenção)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    chart_path = os.path.join(output_dir, "survival_curves_comparison_plot.png")
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Gráfico comparativo de curvas de sobrevivência salvo em: {chart_path}")

if __name__ == "__main__":
    run_survival_analysis()
