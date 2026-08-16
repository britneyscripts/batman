import os
import numpy as np
import pandas as pd


def apply_geometric_adstock(spend: np.ndarray, decay_rate: float) -> np.ndarray:
    """Aplica decaimento temporal geométrico (efeito residual de semanas anteriores)."""
    adstocked = np.zeros_like(spend)
    current_effect = 0.0
    for i, val in enumerate(spend):
        current_effect = val + current_effect * decay_rate
        adstocked[i] = current_effect
    return adstocked


def apply_hill_saturation(
    adstock: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    """Aplica curva de Hill para modelar saturação e retornos decrescentes."""
    # alpha: inclinação da curva
    # gamma: ponto de meia-saturação (inflexão)
    return (adstock**alpha) / (adstock**alpha + gamma**alpha)


def main():
    print("Iniciando geração do dataset sintético multicanal...")

    # Fixar semente para reprodutibilidade dos experimentos
    np.random.seed(42)

    # 1. Configurar linha temporal (3 anos = 156 semanas)
    n_weeks = 156
    dates = pd.date_range(start="2023-01-02", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    # 2. Simular Investimentos por Canal (Spend em R$)
    # Google Search: Investimento mais estável com leves oscilações
    spend_google_search = np.random.uniform(12000, 28000, size=n_weeks)

    # Meta Ads (Meta/Instagram): Flutuações maiores e picos de campanha
    spend_meta_ads = np.random.uniform(8000, 22000, size=n_weeks)

    # TikTok Ads: Canal complementar de topo de funil
    spend_tiktok_ads = np.random.uniform(3000, 12000, size=n_weeks)

    # 3. Simular Métricas de Volume (Impressões estimadas por CPM médio)
    # Ex: CPM Google R$ 35, CPM Meta R$ 25, CPM TikTok R$ 15
    impressions_google = (spend_google_search / 35.0) * 1000 * np.random.uniform(0.95, 1.05, size=n_weeks)
    impressions_meta = (spend_meta_ads / 25.0) * 1000 * np.random.uniform(0.90, 1.10, size=n_weeks)
    impressions_tiktok = (spend_tiktok_ads / 15.0) * 1000 * np.random.uniform(0.85, 1.15, size=n_weeks)

    # 4. Simular Transformações de Mídia (Adstock + Saturação)
    # Google Search: Decaimento rápido (decay=0.2), saturação média
    adstock_google = apply_geometric_adstock(spend_google_search, decay_rate=0.20)
    response_google = (
        apply_hill_saturation(adstock_google, alpha=2.0, gamma=25000) * 85000
    )

    # Meta Ads: Decaimento médio (decay=0.45), bom efeito de branding residual
    adstock_meta = apply_geometric_adstock(spend_meta_ads, decay_rate=0.45)
    response_meta = (
        apply_hill_saturation(adstock_meta, alpha=1.8, gamma=22000) * 60000
    )

    # TikTok Ads: Decaimento curto (decay=0.15), satura mais rápido
    adstock_tiktok = apply_geometric_adstock(spend_tiktok_ads, decay_rate=0.15)
    response_tiktok = (
        apply_hill_saturation(adstock_tiktok, alpha=1.5, gamma=12000) * 25000
    )

    # 5. Baseline (Tendência orgânica + Sazonalidade anual)
    # Tendência de crescimento orgânico leve (+R$ 200/semana)
    trend = 80000 + (t * 220)

    # Sazonalidade anual (picos no fim de ano / Q4)
    annual_seasonality = 18000 * np.sin(2 * np.pi * (t - 10) / 52)

    # Efeito pontual de Black Friday (semanas ~47 de cada ano)
    bf_effect = np.zeros(n_weeks)
    for bf_week in [47, 47 + 52, 47 + 104]:
        if bf_week < n_weeks:
            bf_effect[bf_week] = 45000

    baseline_revenue = trend + annual_seasonality + bf_effect

    # 6. Receita Total (Soma de Baseline + Respostas dos Canais + Ruído Aleatório)
    noise = np.random.normal(0, 4000, size=n_weeks)
    total_revenue = (
        baseline_revenue
        + response_google
        + response_meta
        + response_tiktok
        + noise
    )

    # 7. Montar o DataFrame Final
    df = pd.DataFrame(
        {
            # Coluna de tempo padronizada (ds para o Prophet, date para MMM)
            "date": dates,
            # Métricas de negócio (Receita total e pedidos simulados)
            "revenue": np.round(total_revenue, 2),
            "orders": np.round(total_revenue / 185.0).astype(
                int
            ),  # Ticket médio ~R$ 185
            # Investimentos (Spend)
            "spend_google_search": np.round(spend_google_search, 2),
            "spend_meta_ads": np.round(spend_meta_ads, 2),
            "spend_tiktok_ads": np.round(spend_tiktok_ads, 2),
            # Impressões (Métricas de exposição)
            "impressions_google_search": np.round(impressions_google).astype(
                int
            ),
            "impressions_meta_ads": np.round(impressions_meta).astype(int),
            "impressions_tiktok_ads": np.round(impressions_tiktok).astype(int),
            # Variável de controle
            "is_black_friday": (bf_effect > 0).astype(int),
        }
    )

    # 8. Garantir que a pasta data/ existe e salvar o arquivo
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_data.csv")

    df.to_csv(output_path, index=False)

    print(f" Dataset gerado com sucesso em: {output_path}")
    print(f" Total de registros: {len(df)} semanas (de {df['date'].min().strftime('%Y-%m-%d')} até {df['date'].max().strftime('%Y-%m-%d')})")
    print(f" Colunas geradas: {list(df.columns)}")
    print("\nPrévia dos primeiros registros:")
    print(df[["date", "revenue", "spend_google_search", "spend_meta_ads", "spend_tiktok_ads"]].head())


if __name__ == "__main__":
    main()
