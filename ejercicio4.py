"""Ejercicio 4: Ajuste de parámetros (Simulated Annealing)."""

import os
import random
import statistics
import time
import matplotlib.pyplot as plt
import tsp_utils as tsp   

n_ciudades, T0, Tf = 40, 15.0, 0.05
rng0 = random.Random(11)
D = tsp.dist_matrix(tsp.random_points(n_ciudades, rng0))
inicial = tsp.random_tour(n_ciudades, rng0)                 # solución inicial aleatoria
L0 = tsp.tour_len(D, inicial)
L_hc = tsp.tour_len(D, tsp.two_opt_local_search(D, inicial)[0])
print(f"{n_ciudades} ciudades | T0={T0}, Tf={Tf} | L inicial={L0:.1f} | 2-opt (búsqueda local)={L_hc:.1f}")

# (1) Esquemas de enfriamiento y curvas de convergencia (una corrida de N iteraciones por esquema)
N = 200_000
sch = tsp.make_schedules(T0, Tf, N)
temps = {k: [f(t) for t in range(1, N + 1)] for k, f in sch.items()}
cur = {}
for k, f in sch.items():
    _, hc, _ = tsp.simulated_annealing(D, inicial, f, N, random.Random(5), trace=True)
    cur[k] = hc
colores = dict(zip(sch, ["tab:green", "tab:blue", "tab:red"]))
ticks = [0, 50_000, 100_000, 150_000, 200_000]
os.makedirs("resultados", exist_ok=True)

# Gráfica: esquemas de enfriamiento (temperatura vs. iteraciones)
plt.figure(figsize=(7, 4.5))
for k in sch:
    plt.plot(temps[k], color=colores[k], label=k)
plt.yscale("log")
plt.xticks(ticks, ["0", "50k", "100k", "150k", "200k"])
plt.xlabel("Iteración")
plt.ylabel("Temperatura T")
plt.title("Esquemas de enfriamiento")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("resultados/ej4_esquemas.png", dpi=150)

# Gráfica: curvas de convergencia (costo vs. iteraciones y costo vs. temperatura)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for k in sch:
    ax[0].plot(cur[k], color=colores[k], lw=0.6, alpha=0.8, label=k)
    ax[1].plot(temps[k], cur[k], color=colores[k], lw=0.6, alpha=0.8, label=k)
ax[0].axhline(L_hc, color="k", ls="--", lw=1, label="2-opt (búsqueda local)")
ax[0].set_xticks(ticks)
ax[0].set_xticklabels(["0", "50k", "100k", "150k", "200k"])
ax[0].set_xlabel("Iteración")
ax[0].set_ylabel("Costo (longitud del tour)")
ax[0].set_title("Convergencia: costo vs. iteraciones")
ax[1].set_xscale("log")
ax[1].invert_xaxis()
ax[1].set_ylim(500, 1300)
ax[1].set_xlabel("Temperatura T (enfriando →)")
ax[1].set_ylabel("Costo (longitud del tour)")
ax[1].set_title("Convergencia: costo vs. temperatura")
for a_ in ax:
    a_.grid(alpha=0.3)
    a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig("resultados/ej4_convergencia.png", dpi=150)

# (2) Tiempo de ejecución vs. calidad final con distintos presupuestos de iteraciones
presupuestos, reps = [20_000, 50_000, 100_000, 200_000, 400_000], 5
curvas = {k: [] for k in sch}
for Nb in presupuestos:
    for k, f in tsp.make_schedules(T0, Tf, Nb).items():
        tiempos, costos = [], []
        for s in range(reps):
            t0 = time.perf_counter()
            c, _, _ = tsp.simulated_annealing(D, inicial, f, Nb, random.Random(100 + s))
            tiempos.append(time.perf_counter() - t0)
            costos.append(c)
        curvas[k].append((Nb, statistics.mean(tiempos), statistics.mean(costos), min(costos), max(costos)))
print(f"\nCosto final medio ({reps} corridas) y tiempo medio (s) por presupuesto de iteraciones")
print(f"{'Esquema':22s}" + "".join(f"{p // 1000:>8}k" for p in presupuestos))
for k, pts in curvas.items():
    print(f"{k:22s}" + "".join(f"{p[2]:>9.1f}" for p in pts) + "   (costo)")
    print(f"{'':22s}" + "".join(f"{p[1]:>9.2f}" for p in pts) + "   (tiempo s)")

fig, ax = plt.subplots(figsize=(7.5, 5))
for k, pts in curvas.items():
    ax.plot([p[1] for p in pts], [p[2] for p in pts], "o-", color=colores[k], label=k)
ax.axhline(L_hc, color="k", ls="--", lw=1, label="2-opt (búsqueda local)")
ax.set_xscale("log")
ax.set_xlabel("Tiempo de ejecución (s)")
ax.set_ylabel(f"Costo final (media de {reps} corridas)")
ax.set_title("Tiempo de ejecución vs. calidad de la solución final\n"
             "(cada punto = 20k, 50k, 100k, 200k, 400k iteraciones)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("resultados/ej4_tiempo_calidad.png", dpi=150)
plt.show()
