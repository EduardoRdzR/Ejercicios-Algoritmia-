"""Ejercicio 1: Aceleración paralela (Parallel Speedup)."""

import multiprocessing as mp
import random
import time
import os
import matplotlib.pyplot as plt  

def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fraction of random points that fall inside the unit quarter circle."""
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples

if __name__ == "__main__":
    seeds = list(range(8))  # 8 independent runs, one seed each

    # 1. Sequential Looping
    t0 = time.perf_counter()
    seq_results = [estimate_pi(s) for s in seeds]
    t_seq = time.perf_counter() - t0

    # 2. Parallel Pool Execution
    t0 = time.perf_counter()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        par_results = pool.map(estimate_pi, seeds)
    t_par = time.perf_counter() - t0

    print(f"Sequential wall-clock: {t_seq:.3f} s")
    print(f"Parallel wall-clock:   {t_par:.3f} s (Speedup: {t_seq/t_par:.2f}x)")
    print(f"pi estimate (mean of {len(seeds)} runs): {sum(par_results) / len(par_results):.5f}")

    procs = [1, 2, 4, 8]
    times = []
    for p in procs:
        runs = []
        for _ in range(3):  # mejor tiempo de 3 repeticiones
            t0 = time.perf_counter()
            with mp.Pool(processes=p) as pool:
                pool.map(estimate_pi, seeds, chunksize=1)
            runs.append(time.perf_counter() - t0)
        times.append(min(runs))
    speedup = [times[0] / t for t in times]

    print(f"\nNúcleos disponibles: {mp.cpu_count()}")
    print(f"{'Procesos':>9} {'Tiempo (s)':>11} {'Speedup':>8} {'Ideal':>6} {'Eficiencia':>11}")
    for p, t, s in zip(procs, times, speedup):
        print(f"{p:>9} {t:>11.3f} {s:>8.2f} {p:>6} {s / p:>10.0%}")

    # ---- Visualización
    plt.figure(figsize=(6, 4.2))
    plt.plot(procs, speedup, "o-", label="Speedup medido",color="r")
    plt.plot(procs, procs, "k--", label="Speedup ideal (lineal)")
    plt.xlabel("Número de procesos")
    plt.ylabel("Speedup  S(p) = T1 / Tp")
    plt.title("Aceleración paralela de estimate_pi")
    plt.xticks(procs)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs("resultados", exist_ok=True)
    plt.savefig("resultados/ej1_speedup.png", dpi=150)
    plt.show()
