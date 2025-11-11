# -*- coding: utf-8 -*-
"""
Serie temporal diaria (ventas en un mes)
- Simula 30 días con tendencia + estacionalidad semanal (7 días) + ruido
- Grafica la serie y la tendencia (media móvil)
- Descompone en tendencia/estacionalidad/residuo (periodo=7)
- Guarda CSV y PNGs en 04_series_temporales/out_diario/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

OUT_DIR = "04_series_temporales/out_diario"
os.makedirs(OUT_DIR, exist_ok=True)

# -------- 1) Simulación: ventas diarias (30 días) --------
np.random.seed(42)
n = 30
idx = pd.date_range("2025-01-01", periods=n, freq="D")

# tendencia suave (p.ej., campaña publicitaria que aumenta ventas)
trend = 0.6 * np.arange(n)

# estacionalidad semanal: más ventas sábados-domingos
season = 8 * np.sin(2 * np.pi * np.arange(n) / 7.0)

# ruido aleatorio
noise = np.random.normal(0, 3, size=n)

# nivel base de ventas
base = 50

y = base + trend + season + noise
ts = pd.Series(y, index=idx, name="ventas")

csv_path = os.path.join(OUT_DIR, "ventas_diarias_30d.csv")
ts.to_csv(csv_path, header=True)
print(f"[OK] CSV guardado: {csv_path}")

# -------- 2) Gráfico de la serie + tendencia (media móvil 7 días) --------
ma7 = ts.rolling(window=7, center=True).mean()

plt.figure(figsize=(10,4))
plt.plot(ts.index, ts.values, label="ventas diarias")
plt.plot(ma7.index, ma7.values, label="tendencia (media móvil 7d)")
plt.title("Ventas diarias (30 días) con tendencia")
plt.xlabel("Fecha"); plt.ylabel("Unidades")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_serie_tendencia.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 01_serie_tendencia.png")

# -------- 3) Descomposición aditiva (periodo semanal=7) --------
# Nota: con 30 observaciones y period=7 hay ~4 ciclos; suficiente para ilustrar.
decomp = seasonal_decompose(ts, model="additive", period=7)
fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_descomposicion_semanal.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 02_descomposicion_semanal.png")

print("\nListo: evidencias en 04_series_temporales/out_diario/")
