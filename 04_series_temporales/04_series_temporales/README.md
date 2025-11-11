# Punto 4 — Simulación y Análisis de Series Temporales

**Script principal:** `ts_diarias.py`  
**Datos simulados:** ventas diarias (30 días) con tendencia suave + estacionalidad semanal (7 días) + ruido.

## Evidencias generadas
- `out_diario/ventas_diarias_30d.csv`
- `out_diario/01_serie_tendencia.png` — Serie y tendencia (media móvil 7d).
- `out_diario/02_descomposicion_semanal.png` — Descomposición: observed, trend, seasonal, resid.
- *(opcional si lo añadiste)* `out_diario/03_adf.csv` — Prueba ADF (estacionariedad).
- *(opcional)* `out_diario/04_seaborn_trend.png` — Línea con Seaborn.

## Conclusión
Se observa **tendencia creciente** y **estacionalidad semanal**; la descomposición separa claramente ambos componentes.  
(Con ADF, puedes comentar el p-value si lo generaste).
