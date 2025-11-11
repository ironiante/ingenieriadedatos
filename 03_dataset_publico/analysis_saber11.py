# -*- coding: utf-8 -*-
import os
from typing import List, Optional
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CLEAN_CSV = "03_dataset_publico/out_saber11/saber11_clean.csv"
OUT_DIR = "03_dataset_publico/out_saber11"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_df() -> pd.DataFrame:
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"No existe {CLEAN_CSV}")
    return pd.read_csv(CLEAN_CSV)

def num_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()

def cat_cols(df: pd.DataFrame) -> List[str]:
    n = set(num_cols(df)); return [c for c in df.columns if c not in n]

def parse_cols(inp: str) -> List[str]:
    return [c.strip() for c in inp.split(",") if c.strip()]

# ---------- ESTADÍSTICAS ----------
def stats_numeric(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = [c for c in (cols or num_cols(df)) if c in num_cols(df)]
    desc = df[cols].describe().T.rename(columns={"mean":"media","50%":"mediana","std":"desv_std"})
    out = desc[["count","media","mediana","desv_std","min","25%","75%","max"]]
    path = os.path.join(OUT_DIR, "stats_numericas.csv")
    out.to_csv(path)
    print(f"[OK] {path}")
    return out

def modes_numeric(df: pd.DataFrame, cols: Optional[List[str]] = None, k: int = 3) -> pd.DataFrame:
    cols = [c for c in (cols or num_cols(df)) if c in num_cols(df)]
    data = {}
    for c in cols:
        m = df[c].mode(dropna=True)
        data[c] = list(m.values)[:k] if not m.empty else [None]
    out = pd.DataFrame.from_dict(data, orient="index", columns=[f"mode_{i+1}" for i in range(k)])
    path = os.path.join(OUT_DIR, "modas_numericas.csv")
    out.to_csv(path)
    print(f"[OK] {path}")
    return out

def modes_categorical(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    cols_all = cat_cols(df)
    cols = [c for c in (cols or cols_all) if c in cols_all]
    rows = []
    for c in cols:
        m = df[c].mode(dropna=True)
        if not m.empty:
            moda = m.iloc[0]; freq = int((df[c]==moda).sum())
            rows.append({"col": c, "mode": moda, "freq": freq})
        else:
            rows.append({"col": c, "mode": None, "freq": 0})
    out = pd.DataFrame(rows).sort_values("freq", ascending=False)
    path = os.path.join(OUT_DIR, "modas_categoricas.csv")
    out.to_csv(path, index=False)
    print(f"[OK] {path}")
    return out

# ---------- GRÁFICOS ----------
def plot_hist(df: pd.DataFrame, cols: Optional[List[str]] = None, bins: int = 20):
    cols = [c for c in (cols or num_cols(df)[:12]) if c in num_cols(df)]
    df[cols].hist(figsize=(12,8), bins=bins); plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "hist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {path}")

def plot_box(df: pd.DataFrame, cols: Optional[List[str]] = None, by: Optional[str] = None):
    cols = [c for c in (cols or num_cols(df)[:10]) if c in num_cols(df)]
    if by and by in cat_cols(df):
        dfl = df[cols + [by]].melt(id_vars=by, var_name="feature", value_name="value")
        sns.boxplot(data=dfl, x="feature", y="value", hue=by)
        path = os.path.join(PLOTS_DIR, f"box_by_{by}.png")
    else:
        dfl = df[cols].melt(var_name="feature", value_name="value")
        sns.boxplot(data=dfl, x="feature", y="value")
        path = os.path.join(PLOTS_DIR, "box.png")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {path}")

def plot_scatter(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None):
    if x not in num_cols(df) or y not in num_cols(df): raise ValueError("x e y deben ser numéricas.")
    if hue and hue not in cat_cols(df): hue = None
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=25, alpha=0.8)
    name = f"scatter_{x}_vs_{y}" + (f"_by_{hue}" if hue else "") + ".png"
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[OK] {path}")

# ---------- MENÚ ----------
def menu():
    df = load_df()
    print(f"[INFO] Cargado: {df.shape[0]} filas x {df.shape[1]} cols")
    print("[INFO] Numéricas:", num_cols(df))
    print("[INFO] Categóricas:", cat_cols(df))
    while True:
        print("\n=== MENU ANALISIS SABER_11 ===")
        print("1) Media/Mediana/Desv.Std")
        print("2) Modas numéricas")
        print("3) Modas categóricas")
        print("4) Histogramas")
        print("5) Boxplots (opcional agrupar por categórica)")
        print("6) Dispersión (x vs y, opcional hue)")
        print("0) Salir")
        op = input("Opción: ").strip()
        try:
            if op=="1":
                cols = parse_cols(input("Cols numéricas (coma) o Enter: "))
                print(stats_numeric(df, cols).round(3).head(10))
            elif op=="2":
                cols = parse_cols(input("Cols numéricas (coma) o Enter: ")); print(modes_numeric(df, cols).head())
            elif op=="3":
                cols = parse_cols(input("Cols categóricas (coma) o Enter: ")); print(modes_categorical(df, cols).head())
            elif op=="4":
                cols = parse_cols(input("Cols numéricas (coma) o Enter: ")); plot_hist(df, cols)
            elif op=="5":
                cols = parse_cols(input("Cols numéricas (coma) o Enter: "))
                by = input("Agrupar por (categórica, opcional): ").strip() or None
                plot_box(df, cols, by)
            elif op=="6":
                x = input("X (numérica): ").strip(); y = input("Y (numérica): ").strip()
                hue = input("Hue (categórica, opcional): ").strip() or None
                plot_scatter(df, x, y, hue)
            elif op=="0":
                break
            else:
                print("Opción inválida.")
        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    menu()
