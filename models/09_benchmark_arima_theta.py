import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

def run_time_series_benchmark():
    print("=== BENCHMARK DE SÉRIES TEMPORAIS: ARIMA VS THETA VS PROPHET ===")
    
    # 1. Carregamento dos dados de séries temporais
    # Vamos usar os dados de energia ou a receita processada do mix de marketing
    energy_path = os.path.join("data", "data_raw", "energy_AEP_hourly.csv")
    output_dir = os.path.join("data", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(energy_path):
        print(f"Carregando dados de consumo elétrico: {energy_path}")
        df = pd.read_csv(energy_path, nrows=5000)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.sort_values('Datetime')
        # Reamostrar para média diária para agilizar a modelagem
        df = df.set_index('Datetime').resample('D').mean().dropna()
        target_col = 'AEP_MW'
        y = df[target_col]
    else:
        print("Aviso: Dados de energia não encontrados. Gerando série temporal sintética...")
        # Geração sintética de série diária com sazonalidade e tendência
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
        t = np.arange(365)
        trend = 50 + 0.1 * t
        seasonality = 10 * np.sin(2 * np.pi * t / 365) + 5 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 3, size=365)
        y = pd.Series(trend + seasonality + noise, index=dates)
        y.index.name = 'Date'
        
    # 2. Divisão de treino e teste (Holdout temporal)
    test_size = int(len(y) * 0.15)
    train, test = y.iloc[:-test_size], y.iloc[-test_size:]
    print(f"Observações de treino: {len(train)} | Observações de teste: {len(test)}")
    
    # 3. Modelagem ARIMA (p, d, q) - Exemplo simples ARIMA(1, 1, 1)
    print("\nAjustando modelo ARIMA(1, 1, 1)...")
    try:
        arima_model = ARIMA(train, order=(1, 1, 1))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=test_size)
    except Exception as e:
        print(f"Erro ao ajustar ARIMA: {e}")
        arima_forecast = pd.Series(train.mean(), index=test.index)
        
    # 4. Modelagem Método Theta
    print("Ajustando modelo Método Theta...")
    try:
        theta_model = ThetaModel(train)
        theta_fit = theta_model.fit()
        theta_forecast = theta_fit.forecast(steps=test_size)
    except Exception as e:
        print(f"Erro ao ajustar Theta: {e}")
        theta_forecast = pd.Series(train.mean(), index=test.index)
        
    # 5. Cálculo das métricas de erro (RMSE e MAPE)
    metrics = {}
    for name, forecast in [("ARIMA", arima_forecast), ("Theta", theta_forecast)]:
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mape = mean_absolute_percentage_error(test, forecast)
        metrics[name] = {"RMSE": rmse, "MAPE": mape}
        print(f"  [{name}] RMSE: {rmse:.4f} | MAPE: {mape:.4%}")
        
    # Salvar métricas comparativas
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = os.path.join(output_dir, "time_series_benchmark_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nMétricas salvas com sucesso em: {metrics_path}")
    
    # 6. Plotar e salvar gráfico comparativo das previsões
    plt.figure(figsize=(12, 6))
    plt.plot(train.index[-90:], train.iloc[-90:], label="Histórico de Treino (Últimos 90 dias)", color='#202124')
    plt.plot(test.index, test, label="Valores Reais (Teste)", color='#34a853', linewidth=2)
    plt.plot(test.index, arima_forecast, label=f"Previsão ARIMA (MAPE: {metrics.get('ARIMA', {}).get('MAPE', 0):.2%})", color='#ea4335', linestyle='--')
    plt.plot(test.index, theta_forecast, label=f"Previsão Método Theta (MAPE: {metrics.get('Theta', {}).get('MAPE', 0):.2%})", color='#f9ab00', linestyle='--')
    
    plt.title("Comparativo de Modelos Preditivos de Séries Temporais", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Valores", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    chart_path = os.path.join(output_dir, "time_series_benchmark_plot.png")
    plt.savefig(chart_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Gráfico comparativo salvo em: {chart_path}")

if __name__ == "__main__":
    run_time_series_benchmark()
