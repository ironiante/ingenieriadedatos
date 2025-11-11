# -*- coding: utf-8 -*-
import sys
from typing import Optional, Tuple

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x, y, z = sp.symbols('x y z')  # símbolos por defecto


def parse_expr(expr_str: str) -> sp.Expr:
    """
    Parsea una expresión de texto a SymPy.
    Ej: "sin(x) + x**2"  |  "exp(-x) * cos(x)"
    """
    try:
        return sp.sympify(expr_str, {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                                     "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})
    except Exception as e:
        raise ValueError(f"Expresión inválida: {expr_str}. Detalle: {e}")


# ---------- CÁLCULO SIMBÓLICO ----------
def calcular_derivada(expr_str: str, var: str = "x", orden: int = 1) -> sp.Expr:
    expr = parse_expr(expr_str)
    v = sp.symbols(var)
    return sp.diff(expr, v, orden)


def calcular_integral(expr_str: str, var: str = "x",
                      a: Optional[float] = None, b: Optional[float] = None) -> sp.Expr:
    expr = parse_expr(expr_str)
    v = sp.symbols(var)
    if a is None or b is None:
        # Integral indefinida
        return sp.integrate(expr, v)
    # Integral definida
    return sp.integrate(expr, (v, a, b))


# ---------- SISTEMAS LINEALES (HASTA 3 VARIABLES) ----------
def resolver_sistema_lineal(A: np.ndarray, b_vec: np.ndarray) -> Tuple[sp.Matrix, dict]:
    """
    Resuelve A * [x,y,z]^T = b para 1..3 variables.
    Devuelve (matriz_sympy, solución en dict)
    """
    n_vars = A.shape[1]
    if n_vars == 1:
        vars_syms = sp.symbols('x')
    elif n_vars == 2:
        vars_syms = sp.symbols('x y')
    else:
        vars_syms = sp.symbols('x y z')

    A_sym = sp.Matrix(A)
    b_sym = sp.Matrix(b_vec)
    sol = A_sym.gauss_jordan_solve(b_sym) if n_vars <= 3 else None
    # gauss_jordan_solve devuelve (solución, parámetros) o lanza excepción si no hay única solución
    if isinstance(sol, tuple):
        sol_vec, params = sol
        # Convierte a dict legible
        sol_dict = {str(vars_syms[i]): sp.simplify(sol_vec[i]) for i in range(n_vars)}
        return A_sym, sol_dict
    else:
        raise ValueError("No se pudo resolver el sistema (¿matriz singular o mal dimensionada?)")


# ---------- GRÁFICA DE FUNCIONES ----------
def graficar_funcion(expr_str: str, var: str = "x", a: float = -5, b: float = 5, puntos: int = 400):
    """
    Grafica una función en el intervalo [a,b].
    """
    expr = parse_expr(expr_str)
    v = sp.symbols(var)
    f_num = sp.lambdify(v, expr, "numpy")

    xs = np.linspace(a, b, puntos)
    ys = f_num(xs)

    plt.figure()
    plt.plot(xs, ys)
    plt.title(f"y = {sp.simplify(expr)}")
    plt.xlabel(var)
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


# ---------- MENÚ CLI ----------
def menu():
    while True:
        print("\n=== Calculador Matemático Avanzado ===")
        print("1) Derivada")
        print("2) Integral (indefinida/definida)")
        print("3) Sistema lineal (hasta 3 variables)")
        print("4) Graficar función")
        print("0) Salir")
        opcion = input("Elige una opción: ").strip()

        try:
            if opcion == "1":
                expr = input("f(x) = ")
                var = input("Variable (default x): ").strip() or "x"
                orden = int(input("Orden (default 1): ").strip() or "1")
                res = calcular_derivada(expr, var, orden)
                print(f"f^{orden}({var}) = {sp.simplify(res)}")

            elif opcion == "2":
                expr = input("f(x) = ")
                var = input("Variable (default x): ").strip() or "x"
                tipo = input("¿Integral definida? (s/n, default n): ").strip().lower() or "n"
                if tipo == "s":
                    a = float(input("Límite inferior a = "))
                    b = float(input("Límite superior b = "))
                    res = calcular_integral(expr, var, a, b)
                else:
                    res = calcular_integral(expr, var)
                print(f"Resultado = {sp.simplify(res)}")

            elif opcion == "3":
                print("Dimensión del sistema (1, 2 o 3): ")
                n = int(input().strip())
                if n not in (1, 2, 3):
                    print("Dimensión inválida.")
                    continue
                print("Introduce la matriz A (fila por fila, valores separados por espacio)")
                A = []
                for i in range(n):
                    fila = list(map(float, input(f"Fila {i+1}: ").strip().split()))
                    if len(fila) != n:
                        print("Número de columnas incorrecto.")
                        break
                    A.append(fila)
                else:
                    print("Vector b (valores separados por espacio): ")
                    b_vals = list(map(float, input().strip().split()))
                    if len(b_vals) != n:
                        print("Dimensión de b incorrecta.")
                        continue
                    A = np.array(A, dtype=float)
                    b_vec = np.array(b_vals, dtype=float)
                    _, sol = resolver_sistema_lineal(A, b_vec)
                    print("Solución:", sol)
                    continue
                print("Error al capturar A/b.")

            elif opcion == "4":
                expr = input("f(x) = ")
                a = float(input("x_min (default -5): ") or -5)
                b = float(input("x_max (default 5): ") or 5)
                graficar_funcion(expr, "x", a, b)

            elif opcion == "0":
                print("¡Hasta luego!")
                sys.exit(0)

            else:
                print("Opción inválida.")

        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    menu()
