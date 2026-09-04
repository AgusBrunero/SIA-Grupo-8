"""Métodos de selección (Step 5).

Contrato: select(population, k, ctx) -> list[Individual] de largo exactamente k.
Salvo elite, todos seleccionan CON reposición: un individuo puede salir varias veces.

Todos asumen fitness >= 0 (lo garantiza `1 - RMSE/255`). No se aplica el shift por
el mínimo que se suele usar con fitness negativo: ese shift le da al peor individuo
probabilidad exactamente 0 y, al converger la población, amplifica las diferencias
relativas hasta volver la ruleta determinista. Con fitness ya positivo no hace falta.
"""

from __future__ import annotations

import math

import numpy as np

from .context import Context
from .individual import Individual


def _weights(population: list[Individual]) -> np.ndarray:
    return np.clip(np.array([ind.fitness for ind in population], dtype=np.float64), 0.0, None)


def _pick(population, weights: np.ndarray, pointers: np.ndarray) -> list[Individual]:
    """Traduce punteros en [0,1) a individuos según la ruleta acumulada."""
    total = weights.sum()
    probs = np.full(len(population), 1.0 / len(population)) if total <= 0 else weights / total
    idx = np.clip(np.searchsorted(np.cumsum(probs), pointers), 0, len(population) - 1)
    return [population[i] for i in idx]


def _roulette_pointers(k: int, ctx: Context) -> np.ndarray:
    """k tiradas independientes."""
    return ctx.rng.random(k)


def _universal_pointers(k: int, ctx: Context) -> np.ndarray:
    """Un solo azar y k punteros equiespaciados: r_j = (r + j) / k."""
    return (ctx.rng.random() + np.arange(k)) / k


def elite(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Los k mejores. Si k > N cada individuo se repite n(i) = ceil((k-i)/N) veces,
    que es exactamente lo que da recorrer la lista ordenada en forma cíclica."""
    ordered = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    return [ordered[i % len(ordered)] for i in range(k)]


def roulette(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Probabilidad proporcional al fitness."""
    return _pick(population, _weights(population), _roulette_pointers(k, ctx))


def universal(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Ruleta con puntero estocástico: misma probabilidad, mucha menos varianza."""
    return _pick(population, _weights(population), _universal_pointers(k, ctx))


def _temperature(ctx: Context) -> float:
    cfg = ctx.params.get("boltzmann", {})
    t0, tmin = cfg.get("t0", 100.0), cfg.get("tmin", 1.0)
    decay = cfg.get("k", 0.01)
    return tmin + (t0 - tmin) * math.exp(-decay * ctx.generation)


def boltzmann(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Ruleta sobre exp(f/T). T alta al principio (exploración) y baja al final
    (explotación). Se resta el máximo sólo por estabilidad numérica: es un factor
    constante que se cancela al normalizar, igual que dividir por el promedio."""
    f = _weights(population)
    weights = np.exp((f - f.max()) / _temperature(ctx))
    return _pick(population, weights, _roulette_pointers(k, ctx))


def ranking(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Ruleta sobre el pseudo-fitness f'(i) = (N - rank) / N, con rank 1..N.
    Sólo importa el orden, no la distancia entre fitness: mantiene presión de
    selección cuando la población convergió y todos los fitness son parecidos."""
    ordered = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    n = len(ordered)
    pseudo = (n - np.arange(1, n + 1)) / n
    return _pick(ordered, pseudo, _roulette_pointers(k, ctx))


def tournament_det(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Torneo determinístico: M al azar sin reposición, gana el de mayor fitness.
    M controla la presión de selección (M=2 suave, M=N equivale a elite)."""
    m = min(ctx.params.get("tournament", {}).get("m", 4), len(population))
    chosen = []
    for _ in range(k):
        idx = ctx.rng.choice(len(population), size=m, replace=False)
        chosen.append(max((population[i] for i in idx), key=lambda ind: ind.fitness))
    return chosen


def tournament_prob(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    """Torneo probabilístico: 2 al azar; con probabilidad Th gana el mejor y con
    1-Th el peor. Th < 1 deja pasar individuos malos y preserva diversidad."""
    threshold = ctx.params.get("tournament", {}).get("threshold", 0.75)
    chosen = []
    for _ in range(k):
        i, j = ctx.rng.choice(len(population), size=2, replace=False)
        a, b = population[i], population[j]
        best, worst = (a, b) if a.fitness >= b.fitness else (b, a)
        chosen.append(best if ctx.rng.random() < threshold else worst)
    return chosen


METHODS = {
    "elite": elite,
    "roulette": roulette,
    "universal": universal,
    "boltzmann": boltzmann,
    "ranking": ranking,
    "tournament_det": tournament_det,
    "tournament_prob": tournament_prob,
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"selección '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    return METHODS[name]


def build(spec):
    """Arma un selector a partir de la config.

    Acepta el nombre de un método ("elite") o la selección combinada que pide la
    cátedra: A% con un método y (1-A)% con otro.

        {"method_a": "elite", "method_b": "roulette", "a_ratio": 0.6}
    """
    if isinstance(spec, str):
        return get(spec)

    method_a, method_b = get(spec["method_a"]), get(spec["method_b"])
    ratio = spec.get("a_ratio", 0.5)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"a_ratio debe estar en [0,1], no {ratio}")

    def combined(population, k, ctx):
        k_a = int(round(k * ratio))
        return method_a(population, k_a, ctx) + method_b(population, k - k_a, ctx)

    return combined
