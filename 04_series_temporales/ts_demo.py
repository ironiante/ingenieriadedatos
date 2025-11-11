# -*- coding: utf-8 -*-
"""
Punto 4 — Series Temporales (demo completa)
- Simula una serie mensual con tendencia + estacionalidad + ruido
- Grafica la serie y la descompone (trend/seasonal/resid)
- Prueba de estacionariedad (ADF)
- Grafica ACF/PACF
- Ajusta ARIMA y pronostica
- Exporta PNGs y CSVs a 04_series_temporales/out/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

OUT_DIR = "04_series_temporales/out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 1) Simulación de serie mensual ----------
np.random.seed(42)
n = 120                       # 10 años de datos mensuales
idx = pd.date_range("2015-01-01", periods=n, freq="MS")  # MS = Month Start

trend = 0.8 * np.arange(n)                     # tendencia lineal suave
season = 10 * np.sin(2*np.pi*np.arange(n)/12)  # estacionalidad anual
noise = np.random.normal(0, 5, size=n)         # ruido
y = 50 + trend + season + noise                # serie final

ts = pd.Series(y, index=idx, name="valor")
csv_raw = os.path.join(OUT_DIR, "serie_simulada_mensual.csv")
ts.to_csv(csv_raw, header=True)
print(f"[OK] Serie simulada guardada: {csv_raw}")

# ---------- 2) Gráfico base ----------
plt.figure(figsize=(10,4))
plt.plot(ts.index, ts.values)
plt.title("Serie mensual simulada")
plt.xlabel("Fecha"); plt.ylabel("valor")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_serie.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 01_serie.png")

# ---------- 3) Descomposición (aditiva) ----------
decomp = seasonal_decompose(ts, model="additive", period=12)
fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_descomposicion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 02_descomposicion.png")

# ---------- 4) Prueba ADF (estacionariedad) ----------
adf_stat, pvalue, usedlag, nobs, crit, icbest = adfuller(ts.dropna())
adf_resumen = pd.Series({
    "ADF_stat": adf_stat,
    "p_value": pvalue,
    "lags_usados": usedlag,
    "n_obs": nobs,
    "crit_1%": crit["1%"],
    "crit_5%": crit["5%"],
    "crit_10%": crit["10%"],
})
adf_csv = os.path.join(OUT_DIR, "03_adf.csv")
adf_resumen.to_csv(adf_csv, header=False)
print(f"[OK] 03_adf.csv (p_value={pvalue:.4f})")

# ---------- 5) ACF / PACF ----------
fig_acf = plt.figure(figsize=(8,3))
plot_acf(ts, lags=36)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_acf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 04_acf.png")

fig_pacf = plt.figure(figsize=(8,3))
plot_pacf(ts, lags=36, method="ywm")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_pacf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 05_pacf.png")

# ---------- 6) Ajuste ARIMA y pronóstico ----------
# Sugerido para serie con tendencia y estacionalidad:
# - Diferenciación regular d=1
# - Sin activar SARIMA (para mantenerlo simple en esta demo)
# Puedes cambiar (p,d,q) según tus ACF/PACF.
orden = (1, 1, 1)
modelo = ARIMA(ts, order=orden)
ajuste = modelo.fit()
resumen_txt = os.path.join(OUT_DIR, "06_arima_resumen.txt")
with open(resumen_txt, "w", encoding="utf-8") as f:
    f.write(ajuste.summary().as_text())
print(f"[OK] 06_arima_resumen.txt — ARIMA{orden}")

# Pronóstico 24 meses
h = 24
fc = ajuste.get_forecast(steps=h)
pred = fc.predicted_mean
ic = fc.conf_int(alpha=0.05)
pred.index = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
ic.index = pred.index

df_fc = pd.DataFrame({"forecast": pred})
df_fc["ic_inf"] = ic.iloc[:, 0].values
df_fc["ic_sup"] = ic.iloc[:, 1].values
fc_csv = os.path.join(OUT_DIR, "07_forecast_24m.csv")
df_fc.to_csv(fc_csv)
print(f"[OK] 07_forecast_24m.csv")

# Plot serie + pronóstico
plt.figure(figsize=(10,4))
plt.plot(ts.index, ts.values, label="histórico")
plt.plot(pred.index, pred.values, label="pronóstico")
plt.fill_between(pred.index, df_fc["ic_inf"], df_fc["ic_sup"], alpha=0.2, label="IC 95%")
plt.legend()
plt.title(f"ARIMA{orden} — Pronóstico 24 meses")
plt.xlabel("Fecha"); plt.ylabel("valor")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_forecast.png"), dpi=150, bbox_inches="tight")
plt.close()
print("[OK] 08_forecast.png")

print("\nTodo listo en 04_series_temporales/out/")
