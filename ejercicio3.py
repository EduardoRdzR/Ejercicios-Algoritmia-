"""Ejercicio 3: Búsqueda local (2-opt para TSP)."""

import math
import os
import random
import statistics
import matplotlib.pyplot as plt
import tsp_utils as tsp   

cities = {"A": (0, 0), "B": (0, 3), "E": (2, 5), "D": (4, 3), "C": (4, 0)}

def tour_length(route: list[str]) -> float:
    return sum(math.dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
               for i in range(len(route)))

def two_opt_move(route: list[str], i: int, k: int) -> list[str]:
    """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]

route = ["A", "D", "E", "B", "C"]
new_route = two_opt_move(route, 1, 3)
print(f"Before: {route}  L = {tour_length(route):.2f}")
print(f"After:  {new_route}  L = {tour_length(new_route):.2f}")


rng = random.Random(3)
n_ciudades, n_inst = 30, 100
datos = {"Aleatoria": [], "Vecino más cercano": []}
for _ in range(n_inst):
    D = tsp.dist_matrix(tsp.random_points(n_ciudades, rng))
    t_rand = tsp.random_tour(n_ciudades, rng)
    t_nn = tsp.nearest_neighbour(D, rng.randrange(n_ciudades))     # nodo inicial aleatorio
    for nombre, t0 in (("Aleatoria", t_rand), ("Vecino más cercano", t_nn)):
        t, mejoras = tsp.two_opt_local_search(D, t0)
        datos[nombre].append((mejoras, tsp.tour_len(D, t0), tsp.tour_len(D, t)))

print(f"\n{n_inst} instancias de {n_ciudades} ciudades")
print(f"{'Solución inicial':20s} {'Mejoras media/mediana':>22} {'L inicial':>10} {'L final media/mediana':>22}")
for k, lst in datos.items():
    print(f"{k:20s} {statistics.mean(x[0] for x in lst):>14.1f} / {statistics.median(x[0] for x in lst):<5} "
          f"{statistics.mean(x[1] for x in lst):>10.1f} "
          f"{statistics.mean(x[2] for x in lst):>13.1f} / {statistics.median(x[2] for x in lst):.1f}")
a_, b_ = datos["Aleatoria"], datos["Vecino más cercano"]
print(f"NN necesitó menos mejoras en {sum(y[0] < x[0] for x, y in zip(a_, b_))}/{n_inst} instancias")
print(f"NN terminó con mejor tour en {sum(y[2] < x[2] for x, y in zip(a_, b_))}/{n_inst} instancias")

# ---- Visualización
os.makedirs("resultados", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].boxplot([[x[0] for x in datos[k]] for k in datos], tick_labels=list(datos))
ax[0].set_ylabel("Número de mejoras 2-opt")
ax[0].set_title("Mejoras hasta el óptimo local")
ax[1].boxplot([[x[2] for x in datos[k]] for k in datos], tick_labels=list(datos))
ax[1].set_ylabel("Longitud del tour final")
ax[1].set_title("Calidad del óptimo local alcanzado")
for x in ax:
    x.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("resultados/ej3_2opt.png", dpi=150)
plt.show()
