"""Funciones compartidas de TSP: distancias, 2-opt, MST, Held-Karp y recocido simulado."""
import math
import random


def random_points(n, rng, size=100.0):
    return [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(n)]


def dist_matrix(points):
    return [[math.dist(p, q) for q in points] for p in points]


def tour_len(D, tour):
    return sum(D[tour[i]][tour[(i + 1) % len(tour)]] for i in range(len(tour)))


def random_tour(n, rng):
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def nearest_neighbour(D, start):
    n = len(D)
    tour, left = [start], set(range(n)) - {start}
    while left:
        nxt = min(left, key=lambda j: D[tour[-1]][j])
        tour.append(nxt)
        left.remove(nxt)
    return tour


def two_opt_move(tour, i, k):
    """Invierte tour[i..k]: quita las aristas (i-1,i) y (k,k+1)."""
    return tour[:i] + tour[i:k + 1][::-1] + tour[k + 1:]


def two_opt_delta(D, tour, i, k):
    n = len(tour)
    a, b = tour[i - 1], tour[i]
    c, d = tour[k], tour[(k + 1) % n]
    return D[a][c] + D[b][d] - D[a][b] - D[c][d]   # < 0 => el tour mejora


def two_opt_local_search(D, tour):
    """Heurística 2-opt (mejor vecino en cada paso) hasta un óptimo local.
    Devuelve (tour, número de mejoras)."""
    tour = tour[:]
    n = len(tour)
    improvements = 0
    while True:
        best_delta, best_move = -1e-12, None
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                delta = two_opt_delta(D, tour, i, k)
                if delta < best_delta:
                    best_delta, best_move = delta, (i, k)
        if best_move is None:
            return tour, improvements
        tour = two_opt_move(tour, *best_move)
        improvements += 1


def mst_prim(D):
    """MST con Prim desde la ciudad 0. Devuelve (hijos, peso total)."""
    n = len(D)
    best = {v: (D[0][v], 0) for v in range(1, n)}
    children = {v: [] for v in range(n)}
    total = 0.0
    while best:
        v = min(best, key=lambda u: best[u][0])
        w, parent = best.pop(v)
        total += w
        children[parent].append(v)
        for u in best:
            if D[v][u] < best[u][0]:
                best[u] = (D[v][u], v)
    return children, total


def preorder(children, root=0):
    return [root] + [c for ch in children[root] for c in preorder(children, ch)]


def held_karp(D):
    """Longitud del tour óptimo exacto, O(n^2 2^n)."""
    n = len(D)
    full = 1 << (n - 1)
    INF = float("inf")
    dp = [[INF] * (n - 1) for _ in range(full)]
    for j in range(n - 1):
        dp[1 << j][j] = D[0][j + 1]
    for mask in range(1, full):
        for j in range(n - 1):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            for k in range(n - 1):
                if (mask >> k) & 1:
                    continue
                nm, v = mask | (1 << k), dp[mask][j] + D[j + 1][k + 1]
                if v < dp[nm][k]:
                    dp[nm][k] = v
    return min(dp[full - 1][j] + D[j + 1][0] for j in range(n - 1))


def simulated_annealing(D, tour, schedule, iters, rng, trace=False):
    """Recocido simulado con 2-opt como movimiento. dE = L_actual - L_nuevo."""
    cur, n = tour[:], len(tour)
    cur_len = tour_len(D, cur)
    best_len = cur_len
    hist_cur, hist_best = [cur_len], [cur_len]
    for t in range(2, iters + 1):
        T = schedule(t)
        i, k = sorted(rng.sample(range(1, n), 2))
        delta = two_opt_delta(D, cur, i, k)
        dE = -delta
        if dE > 0 or (T > 1e-12 and rng.random() < math.exp(dE / T)):
            cur = two_opt_move(cur, i, k)
            cur_len += delta
            best_len = min(best_len, cur_len)
        if trace:
            hist_cur.append(cur_len)
            hist_best.append(best_len)
    return best_len, hist_cur, hist_best


def make_schedules(T0, Tf, N):
    return {
        "Lineal": lambda t: T0 - (T0 - Tf) * t / N,
        "Exponencial (lento)": lambda t: T0 * (Tf / T0) ** (t / N),
        "Exponencial (rápido)": lambda t: max(Tf, T0 * (Tf / T0) ** (t / (0.05 * N))),
    }
