# -*- coding: utf-8 -*-
"""
Análisis interactivo de Saber_11 (sobre el CSV limpio).
Funciones:
- stats: media, mediana, desviación estándar, moda (numéricas o categóricas)
- plots: histogramas, boxplots y dispersión (scatter)
Salidas en: 03_dataset_publico/out_saber11/ y /plots
"""

import os
from typing import List, Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

CLEAN_CSV = "03_dataset_publico/out_saber11/saber11_clean.csv"
OUT_DIR = "03_dataset_publico/out_saber11"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============== UTILIDADES BASE ===============

def load_data(path: str = CLEAN_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el CSV limpio en: {path}")
    df = pd.read_csv(path)
    return df

def get_numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()

def get_categorical_cols(df: pd.DataFrame) -> List[str]:
    num = set(get_numeric_cols(df))
    return [c for c in df.columns if c not in num]

def normalize_list(inp: str) -> List[str]:
    """
    Convierte 'a,b,c' -> ['a','b','c'] y limpia espacios.
    Si inp está vacío -> []
    """
    if not inp.strip():
        return []
    return [x.strip() for x in inp.split(",") if x.strip()]

# =============== ESTADÍSTICAS ===============

def stats_numeric(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Media, mediana y desviación estándar de columnas numéricas."""
    num_cols = get_numeric_cols(df)
    if not num_cols:
        raise ValueError("No hay columnas numéricas en el dataset.")
    if cols:
        cols = [c for c in cols if c in num_cols]
        if not cols:
            raise ValueError("Las columnas indicadas no son numéricas o no existen.")
    else:
        cols = num_cols
    desc = df[cols].describe().T  # incluye mean y std; 50% = mediana
    desc = desc.rename(columns={"mean": "media", "std": "desv_std", "50%": "mediana"})
    out = desc[["count", "media", "mediana", "desv_std", "min", "25%", "75%", "max"]]
    out_path = os.path.join(OUT_DIR, "stats_numericas.csv")
    out.round(3).to_csv(out_path)
    print(f"[OK] Estadísticas numéricas guardadas en: {out_path}")
    return out

def modes_numeric(df: pd.DataFrame, cols: Optional[List[str]] = None, k: int = 3) -> pd.DataFrame:
    """Modas (hasta k) para columnas numéricas."""
    num_cols = get_numeric_cols(df)
    if cols:
        cols = [c for c in cols if c in num_cols]
    else:
        cols = num_cols
    data = {}
    for c in cols:
        m = df[c].mode(dropna=True)
        data[c] = list(m.values)[:k] if not m.empty else [None]
    out = pd.DataFrame.from_dict(data, orient="index", columns=[f"mode_{i+1}" for i in range(k)])
    out_path = os.path.join(OUT_DIR, "modas_numericas.csv")
    out.to_csv(out_path)
    print(f"[OK] Modas numéricas guardadas en: {out_path}")
    return out

def modes_categorical(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Moda (y frecuencia) para columnas categóricas."""
    cat_cols = get_categorical_cols(df)
    if cols:
        cols = [c for c in cols if c in cat_cols]
    else:
        cols = cat_cols
    rows = []
    for c in cols:
        m = df[c].mode(dropna=True)
        if not m.empty:
            moda = m.iloc[0]
            freq = int((df[c] == moda).sum())
            rows.append({"col": c, "mode": moda, "freq": freq})
        else:
            rows.append({"col": c, "mode": None, "freq": 0})
    out = pd.DataFrame(rows).sort_values("freq", ascending=False)
    out_path = os.path.join(OUT_DIR, "modas_categoricas.csv")
    out.to_csv(out_path, index=False)
    print(f"[OK] Modas categóricas guardadas en: {out_path}")
    return out

# =============== GRÁFICOS ===============

def plot_hist(df: pd.DataFrame, cols: Optional[List[str]] = None, bins: int = 20):
    """Histogramas para columnas numéricas."""
    num_cols = get_numeric_cols(df)
    if cols:
        cols = [c for c in cols if c in num_cols]
    else:
        cols = num_cols[:12]  # limitar por legibilidad
    if not cols:
        raise ValueError("No hay columnas numéricas válidas para histograma.")
    ax = df[cols].hist(figsize=(12, 8), bins=bins)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "hist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Histograma guardado en: {path}")

def plot_box(df: pd.DataFrame, cols: Optional[List[str]] = None, by: Optional[str] = None):
    """
    Boxplots de columnas numéricas.
    - cols: lista de numéricas a graficar
    - by: agrupar por columna categórica (opcional)
    """
    num_cols = get_numeric_cols(df)
    if cols:
        cols = [c for c in cols if c in num_cols]
    else:
        cols = num_cols[:10]
    if not cols:
        raise ValueError("No hay columnas numéricas válidas para boxplot.")

    if by and by in df.columns and by in get_categorical_cols(df):
        # Boxplot por categoría (apilado largo)
        dfl = df[cols + [by]].melt(id_vars=by, var_name="feature", value_name="value")
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=dfl, x="feature", y="value", hue=by)
        plt.xticks(rotation=20)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"box_by_{by}.png")
    else:
        # Boxplot sin categoría
        plt.figure(figsize=(12, 7))
        dfl = df[cols].melt(var_name="feature", value_name="value")
        sns.boxplot(data=dfl, x="feature", y="value")
        plt.xticks(rotation=20)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "box.png")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Boxplot guardado en: {path}")

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None):
    """
    Dispersión entre dos columnas numéricas.
    - hue: columna categórica opcional para colorear puntos.
    """
    if x not in df.columns or y not in df.columns:
        raise ValueError("x o y no existen en el DataFrame.")
    if x not in get_numeric_cols(df) or y not in get_numeric_cols(df):
        raise ValueError("x e y deben ser numéricas.")

    plt.figure(figsize=(7, 6))
    if hue and hue in get_categorical_cols(df):
        sns.scatterplot(data=df, x=x, y=y, hue=hue, s=25, alpha=0.8)
        path = os.path.join(PLOTS_DIR, f"scatter_{x}_vs_{y}_by_{hue}.png")
    else:
        sns.scatterplot(data=df, x=x, y=y, s=25, alpha=0.8)
        path = os.path.join(PLOTS_DIR, f"scatter_{x}_vs_{y}.png")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Scatter guardado en: {path}")

# =============== MENÚ CLI ===============

def menu():
    df = load_data()
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"[INFO] Numéricas: {get_numeric_cols(df)}")
    print(f"[INFO] Categóricas: {get_categorical_cols(df)}")

    while True:
        print("\n=== MENU ANALISIS SABER_11 ===")
        print("1) Media/Mediana/Desv.Std (numéricas)")
        print("2) Modas numéricas")
        print("3) Modas categóricas")
        print("4) Histogramas (numéricas)")
        print("5) Boxplots (numéricas)   | opcional: agrupar por categórica")
        print("6) Dispersión (x vs y)    | opcional: hue categórica")
        print("0) Salir")
        op = input("Opción: ").strip()

        try:
            if op == "1":
                cols = normalize_list(input("Columnas numéricas (coma) o vacío para todas: "))
                res = stats_numeric(df, cols)
                print(res.round(3).head(10))

            elif op == "2":
                cols = normalize_list(input("Columnas numéricas (coma) o vacío para todas: "))
                res = modes_numeric(df, cols)
                print(res.head())

            elif op == "3":
                cols = normalize_list(input("Columnas categóricas (coma) o vacío para todas: "))
                res = modes_categorical(df, cols)
                print(res.head())

            elif op == "4":
                cols = normalize_list(input("Columnas numéricas (coma) o vacío para auto-selección: "))
                plot_hist(df, cols)

            elif op == "5":
                cols = normalize_list(input("Columnas numéricas (coma) o vacío para auto-selección: "))
                by = input("Agrupar por categórica (opcional): ").strip() or None
                plot_box(df, cols, by)

            elif op == "6":
                x = input("Columna X (numérica): ").strip()
                y = input("Columna Y (numérica): ").strip()
                hue = input("Hue (categórica, opcional): ").strip() or None
                plot_scatter(df, x, y, hue)

            elif op == "0":
                print("¡Listo!")
                break
            else:
                print("Opción inválida.")

        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    menu()
