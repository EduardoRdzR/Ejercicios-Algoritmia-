"""Ejercicio 5: Razón de aproximación (MST Approximation)."""

import itertools
import math
import random
import os
import statistics
import time
import matplotlib.pyplot as plt
import tsp_utils as tsp  

def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
    """Prim's algorithm from city 0; returns the tree as {node: children}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)          # add the cheapest edge to the tree
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children

def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    """Walk around the tree; visiting each city once = shortcutting repeated cities."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]

def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))

random.seed(7)
points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(9)]

approx = preorder(mst_prim(points))
optimal = min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
              key=lambda t: tour_length(points, t))   # brute force: 8! = 40,320 tours

print(f"MST approximation: {approx}  L = {tour_length(points, approx):.2f}")
print(f"Optimal (brute):   {list(optimal)}  L = {tour_length(points, optimal):.2f}")
print(f"Ratio = {tour_length(points, approx) / tour_length(points, optimal):.3f}  (guaranteed <= 2)")

rng = random.Random(7)
N_SA, T0, Tf = 20_000, 2.0, 0.005
sched = tsp.make_schedules(T0, Tf, N_SA)["Exponencial (lento)"]
f = {k: [] for k in ("MST", "MST + 2-opt", "Búsqueda local", "Recocido simulado", "Óptimo")}
tiempo = {k: 0.0 for k in f}
w_sobre_opt = []
for _ in range(100):
    pts = tsp.random_points(9, rng, size=10)
    D = tsp.dist_matrix(pts)

    t0 = time.perf_counter()
    hijos = mst_prim(pts)                                   # función del documento
    t_mst = preorder(hijos)                                 # función del documento
    L_mst = tour_length(pts, t_mst)                         # función del documento
    w = sum(math.dist(pts[p], pts[c]) for p, kids in hijos.items() for c in kids)   # peso del árbol
    tiempo["MST"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    L_mst2 = tsp.tour_len(D, tsp.two_opt_local_search(D, t_mst)[0])
    tiempo["MST + 2-opt"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    L_ls = tsp.tour_len(D, tsp.two_opt_local_search(D, tsp.random_tour(9, rng))[0])
    tiempo["Búsqueda local"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    L_sa = tsp.simulated_annealing(D, tsp.random_tour(9, rng), sched, N_SA, rng)[0]
    tiempo["Recocido simulado"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    opt = tsp.held_karp(D)
    tiempo["Óptimo"] += time.perf_counter() - t0

    eps = 1e-9   # w(T) <= OPT <= MST+2opt <= MST <= 2 w(T) <= 2 OPT
    assert w <= opt + eps and opt <= L_mst2 + eps and L_mst2 <= L_mst + eps
    assert L_mst <= 2 * w + eps and L_mst <= 2 * opt + eps
    for k, v in zip(f, (L_mst, L_mst2, L_ls, L_sa, opt)):
        f[k].append(v)
    w_sobre_opt.append(w / opt)

ratio = {k: [a / b for a, b in zip(v, f["Óptimo"])] for k, v in f.items()}
print(f"\n{'Método':20s} {'Costo medio':>12} {'Razón media':>12} {'Razón máx':>10} {'Óptimo':>9} {'ms/inst':>8}")
for k in f:
    print(f"{k:20s} {statistics.mean(f[k]):>12.2f} {statistics.mean(ratio[k]):>12.3f} "
          f"{max(ratio[k]):>10.3f} {sum(r < 1 + 1e-9 for r in ratio[k]):>6}/100 {tiempo[k] * 10:>8.2f}")
print(f"\nPeso del MST / óptimo (media): {statistics.mean(w_sobre_opt):.3f}")
print(f"Instancias con razón MST > 1.5: {sum(r > 1.5 for r in ratio['MST'])}")

# ---- Visualización 1: dispersión de la razón de aproximación (MST) con el límite teórico de 2
os.makedirs("resultados", exist_ok=True)
plt.figure(figsize=(7, 4.5))
plt.scatter(range(1, 101), ratio["MST"], s=14, label="MST")
plt.axhline(2, color="r", ls="--", label="Límite teórico = 2")
plt.axhline(1, color="k", ls=":", label="Óptimo")
plt.ylim(0.95, 2.1)
plt.xlabel("Instancia")
plt.ylabel("Razón C_obtenido / C_óptimo")
plt.title("Razón de aproximación del MST en 100 instancias de 9 ciudades")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("resultados/ej5_dispersion.png", dpi=150)

# ---- Visualización 2: costo medio por método (barras)
nombres = list(f)
plt.figure(figsize=(8, 4.5))
plt.bar(nombres, [statistics.mean(f[k]) for k in nombres],
        color=["tab:blue", "tab:cyan", "tab:orange", "tab:green", "gray"])
plt.ylabel("Costo medio (longitud del tour)")
plt.title("Costo medio por método (100 instancias de 9 ciudades)")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("resultados/ej5_barras.png", dpi=150)
plt.show()
