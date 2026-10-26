"""Ejercicio 2: Análisis algorítmico (Greedy Knapsack)."""

import os
import matplotlib.pyplot as plt

def knapsack(values: list[int], weights: list[int], W: int):
    n = len(values)
    K = [[0] * (W + 1) for _ in range(n + 1)]   # K[i][c]: best value with items 1..i and capacity c
    for i in range(1, n + 1):
        v, w = values[i - 1], weights[i - 1]
        for c in range(W + 1):
            K[i][c] = K[i - 1][c]                                   # skip item i
            if w <= c:
                K[i][c] = max(K[i][c], v + K[i - 1][c - w])         # take item i
    chosen, c = [], W                           # walk back through the table
    for i in range(n, 0, -1):
        if K[i][c] != K[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return K, K[n][W], chosen[::-1]

items, values, weights = ["A", "B", "C"], [60, 100, 120], [10, 20, 30]
K, best, chosen = knapsack(values, weights, 50)
print("capacity:", *[f"{c:4d}" for c in range(0, 51, 10)])
for i, row in enumerate(K):
    print(f"{'-' if i == 0 else '+' + items[i - 1]:>8s}:", *[f"{row[c]:4d}" for c in range(0, 51, 10)])
print(f"Best value = {best} with items {[items[i] for i in chosen]}  (greedy found 160)")


def knapsack_greedy(values, weights, W, key):
    """Voraz: recorre los objetos en el orden dado por 'key' y agrega los que caben."""
    order = sorted(range(len(values)), key=key)
    chosen, w_used, total = [], 0, 0
    for i in order:
        if w_used + weights[i] <= W:
            chosen.append(i)
            w_used += weights[i]
            total += values[i]
    return sorted(chosen), w_used, total


# Instancia diseñada: el criterio de menor peso obtiene un resultado peor
nombres = ["A", "B", "C"]
valores, pesos, capacidad = [10, 100, 120], [10, 20, 30], 50
metodos = {
    "Valor/peso": lambda i: -valores[i] / pesos[i],      # mayor razón v/w primero
    "Menor peso primero": lambda i: pesos[i],            # modificación pedida
}
filas = {}
for nombre, key in metodos.items():
    ch, w_used, total = knapsack_greedy(valores, pesos, capacidad, key)
    filas[nombre] = {"items": [nombres[i] for i in ch], "peso": w_used, "valor": total}
_, optimo, _ = knapsack(valores, pesos, capacidad)       # óptimo con la programación dinámica del documento

print(f"\nCapacidad W = {capacidad}")
for n, v, w in zip(nombres, valores, pesos):
    print(f"  Objeto {n}: valor={v}, peso={w}, razón v/w={v / w:.1f}")
print(f"\n{'Método':22s} {'Objetos':12s} {'Capacidad usada':>16} {'Valor total':>12}")
for nombre, r in filas.items():
    print(f"{nombre:22s} {str(r['items']):12s} {str(r['peso']) + ' de ' + str(capacidad):>16} {r['valor']:>12}")
print(f"{'Óptimo (DP)':22s} {'-':12s} {'-':>16} {optimo:>12}")

# ---- Visualización
nom = list(filas)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, campo, tit in ((ax[0], "valor", "Valor total"), (ax[1], "peso", "Capacidad usada")):
    bars = a.bar(nom, [filas[n][campo] for n in nom], color=["tab:green", "tab:purple"])
    for b, n in zip(bars, nom):
        a.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
               f"{b.get_height()}\n{{{', '.join(filas[n]['items'])}}}", ha="center", fontsize=9)
    a.set_title(tit)
    a.grid(axis="y", alpha=0.3)
ax[0].axhline(optimo, color="k", ls="--", lw=1)
ax[0].set_ylim(0, 270)
ax[1].axhline(capacidad, color="r", ls="--", lw=1)
ax[1].text(1.45, capacidad + 1, f"capacidad W = {capacidad}", ha="right", fontsize=8, color="r")
ax[1].set_ylim(0, 62)
fig.tight_layout()
os.makedirs("resultados", exist_ok=True)
fig.savefig("resultados/ej2_mochila.png", dpi=150)
plt.show()
