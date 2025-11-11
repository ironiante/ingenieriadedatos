# -*- coding: utf-8 -*-
"""
Punto 5 — Optimización Numérica
Incluye:
  A) SciPy local: minimize (BFGS/CG) sobre Rosenbrock y cuadrática convexa
  B) SciPy global: differential_evolution
  C) Algoritmo propio: Gradiente descendente en f(x,y) = (x-3)^2 + 0.5*(y+2)^2
Evidencias: CSV/PNG en 05_optimizacion/out/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Callable

from scipy.optimize import minimize, differential_evolution

OUT_DIR = "05_optimizacion/out"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- FUNCIONES OBJETIVO --------------------
def rosenbrock(xy: np.ndarray, a=1.0, b=100.0) -> float:
    """f(x,y) = (a - x)^2 + b*(y - x^2)^2 (no convexa, valle curvado)."""
    x, y = xy
    return (a - x)**2 + b*(y - x**2)**2

def quad_convexa(xy: np.ndarray) -> float:
    """f(x,y) = (x-1)^2 + 2*(y+2)^2 (convexa, mínimo global claro)."""
    x, y = xy
    return (x-1)**2 + 2*(y+2)**2

def simple_quad(xy: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    f(x,y) = (x-3)^2 + 0.5*(y+2)^2
    Devuelve f y gradiente ∇f para gradiente descendente.
    """
    x, y = xy
    f = (x-3)**2 + 0.5*(y+2)**2
    grad = np.array([2*(x-3), (y+2)])  # d/dx, d/dy
    return f, grad

# -------------------- UTILS PLOT --------------------
def plot_contour_with_path(f: Callable, path: np.ndarray, title: str, fname: str, xlim=(-4, 4), ylim=(-6, 4)):
    xs = np.linspace(*xlim, 300)
    ys = np.linspace(*ylim, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    plt.figure(figsize=(6,5))
    cs = plt.contour(X, Y, Z, levels=30)
    plt.clabel(cs, inline=True, fontsize=7)
    if path is not None and len(path) > 0:
        plt.plot(path[:,0], path[:,1], marker='o', ms=3)
        plt.scatter(path[0,0], path[0,1], c='k', s=40, label='inicio')
        plt.scatter(path[-1,0], path[-1,1], c='red', s=40, label='final')
        plt.legend()
    plt.title(title)
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")

def plot_convergence(vals: np.ndarray, title: str, fname: str):
    plt.figure(figsize=(6,3.6))
    plt.plot(np.arange(len(vals)), vals)
    plt.xlabel("iteración"); plt.ylabel("f(x)")
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out}")

# -------------------- A) SciPy local --------------------
def scipy_local(obj_name="rosenbrock", x0=None, method="BFGS"):
    f = rosenbrock if obj_name=="rosenbrock" else quad_convexa
    if x0 is None:
        x0 = np.array([-1.5, 2.0]) if obj_name=="rosenbrock" else np.array([4.0, -4.0])

    traj = []
    def cb(xk):
        traj.append(np.array(xk))

    res = minimize(lambda z: f(z), x0, method=method, callback=cb, options=dict(maxiter=1000, gtol=1e-8))
    traj = np.vstack([x0, *traj]) if len(traj)>0 else np.array([x0, res.x])

    # Evidencias
    name = f"local_{obj_name}_{method}".lower()
    plot_contour_with_path(f, traj, f"Local {obj_name} ({method})", f"{name}_contour.png")
    plot_convergence(np.array([f(p) for p in traj]), f"Convergencia {obj_name} ({method})", f"{name}_convergencia.png")

    pd.DataFrame(traj, columns=["x","y"]).assign(f=[f(p) for p in traj]).to_csv(
        os.path.join(OUT_DIR, f"{name}_trayectoria.csv"), index=False
    )
    print(res)
    return res

# -------------------- B) SciPy global --------------------
def scipy_global(obj_name="rosenbrock", bounds=((-3,3), (-3,3))):
    f = rosenbrock if obj_name=="rosenbrock" else quad_convexa
    res = differential_evolution(lambda z: f(z), bounds=bounds, maxiter=300, tol=1e-7, polish=True)
    # Para graficar, generamos un "camino" artificial: inicio aleatorio -> mínimo
    path = np.array([np.array([b[0], b[0]]) for b in [bounds[0]]])  # punto esquina
    path = np.vstack([path, res.x])

    name = f"global_{obj_name}".lower()
    plot_contour_with_path(f, path, f"Global {obj_name} (Differential Evolution)", f"{name}_contour.png")
    pd.DataFrame([res.x], columns=["x","y"]).assign(f=f(res.x)).to_csv(
        os.path.join(OUT_DIR, f"{name}_sol.csv"), index=False
    )
    print(res)
    return res

# -------------------- C) Gradiente descendente propio --------------------
def gradiente_descendente(lr=0.1, x0=np.array([-4.0, 3.0]), maxiter=200, tol=1e-8):
    """
    Minimiza f(x,y) = (x-3)^2 + 0.5*(y+2)^2 con paso fijo lr.
    """
    x = x0.astype(float).copy()
    path = [x.copy()]
    vals = []
    for k in range(maxiter):
        fval, grad = simple_quad(x)
        vals.append(fval)
        if np.linalg.norm(grad) < tol:
            break
        x -= lr * grad
        path.append(x.copy())

    path = np.array(path)
    vals = np.array(vals)

    # Evidencias
    plot_contour_with_path(lambda z: simple_quad(z)[0], path,
                           f"Gradiente descendente (lr={lr})", "gd_contour.png",
                           xlim=(-6, 6), ylim=(-8, 2))
    plot_convergence(vals, "Convergencia gradiente descendente", "gd_convergencia.png")

    pd.DataFrame(path, columns=["x","y"]).assign(f=vals.tolist()+[simple_quad(path[-1])[0]] if len(vals)<len(path) else vals)\
        .to_csv(os.path.join(OUT_DIR, "gd_trayectoria.csv"), index=False)
    print(f"[OK] mínimo aproximado en {path[-1]}, f={simple_quad(path[-1])[0]:.6f}")
    return path, vals

# -------------------- MENÚ CLI --------------------
def menu():
    while True:
        print("\n=== PUNTO 5 — OPTIMIZACIÓN ===")
        print("1) SciPy local (Rosenbrock / BFGS)")
        print("2) SciPy local (Cuadrática / CG)")
        print("3) SciPy global (Differential Evolution en Rosenbrock)")
        print("4) Gradiente descendente propio (función simple)")
        print("0) Salir")
        op = input("Opción: ").strip()

        if op == "1":
            scipy_local(obj_name="rosenbrock", method="BFGS")
        elif op == "2":
            scipy_local(obj_name="quad", method="CG")
        elif op == "3":
            scipy_global(obj_name="rosenbrock", bounds=((-3,3),(-3,3)))
        elif op == "4":
            try:
                lr = float(input("Learning rate (ej. 0.1): ").strip() or "0.1")
            except ValueError:
                lr = 0.1
            gradiente_descendente(lr=lr)
        elif op == "0":
            print("Listo.")
            break
        else:
            print("Opción inválida.")

if __name__ == "__main__":
    menu()
