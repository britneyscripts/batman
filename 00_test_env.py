import sys
import jax
import numpy as np
import pandas as pd
from prophet import Prophet

print(f" Python Version: {sys.version.split()[0]}")
print(f" JAX default backend: {jax.default_backend()}")

# Teste relâmpago do Prophet
df_dummy = pd.DataFrame(
    {
        "ds": pd.date_range("2024-01-01", periods=10, freq="D"),
        "y": np.random.randn(10) + 100,
    }
)
m = Prophet()
m.fit(df_dummy)
print(" Prophet inicializado e ajustado com sucesso!")

# Teste de importação do Meridian
try:
    import meridian

    print(f" Google Meridian carregado com sucesso (v{meridian.__version__})!")
except ImportError:
    print(
        "⚠️ Meridian ainda não encontrado no path. Verifique se o pip install finalizou."
    )