"""Métodos de cruza (Step 6).

Contrato: crossover(parent_a, parent_b, ctx) -> (child_a, child_b).

Todos trabajan sobre el vector plano de genes visto como una matriz de UNIDADES:

    crossover_granularity = "gene"      -> unidad = 1 gen  (puede partir un triángulo)
    crossover_granularity = "triangle"  -> unidad = 10 genes (triángulos enteros)

La granularidad es el experimento central del TP: como el orden de los triángulos
define el z-order, dos padres buenos pueden codificar la misma imagen con los
triángulos en distinto orden (competing conventions). Cortar a nivel gen mezcla
mitades de triángulos distintos y suele ser destructivo.
"""

from __future__ import annotations

import numpy as np

from .context import Context
from .individual import GENES_PER_TRIANGLE, Individual


def _unit_size(ctx: Context) -> int:
    return GENES_PER_TRIANGLE if ctx.params.get("crossover_granularity") == "triangle" else 1


def _as_units(individual: Individual, unit: int) -> np.ndarray:
    return individual.genes.reshape(-1, unit)


def _children(units_a: np.ndarray, units_b: np.ndarray) -> tuple[Individual, Individual]:
    return Individual(units_a.ravel().copy()), Individual(units_b.ravel().copy())


def _swap(a: Individual, b: Individual, mask: np.ndarray, ctx: Context):
    """Intercambia las unidades marcadas en `mask` entre los dos padres."""
    unit = _unit_size(ctx)
    ua, ub = _as_units(a, unit).copy(), _as_units(b, unit).copy()
    ua[mask], ub[mask] = ub[mask].copy(), ua[mask].copy()
    return _children(ua, ub)


def one_point(a: Individual, b: Individual, ctx: Context):
    """Un punto de corte p: el hijo 1 toma [0,p) de A y [p,n) de B."""
    n = len(a.genes) // _unit_size(ctx)
    if n < 2:
        return a.copy(), b.copy()
    p = int(ctx.rng.integers(1, n))
    mask = np.arange(n) >= p
    return _swap(a, b, mask, ctx)


def two_point(a: Individual, b: Individual, ctx: Context):
    """Dos puntos de corte: se intercambia el segmento central."""
    n = len(a.genes) // _unit_size(ctx)
    if n < 3:
        return one_point(a, b, ctx)
    p1, p2 = sorted(ctx.rng.choice(np.arange(1, n), size=2, replace=False))
    mask = (np.arange(n) >= p1) & (np.arange(n) < p2)
    return _swap(a, b, mask, ctx)


def uniform(a: Individual, b: Individual, ctx: Context):
    """Cada unidad se intercambia con probabilidad p (default 0.5), de forma
    independiente. Es el más disruptivo: no preserva bloques contiguos."""
    n = len(a.genes) // _unit_size(ctx)
    p = ctx.params.get("crossover_uniform_p", 0.5)
    return _swap(a, b, ctx.rng.random(n) < p, ctx)


def annular(a: Individual, b: Individual, ctx: Context):
    """Cruza anular: el cromosoma se cierra en un anillo, se elige un punto de
    inicio y un largo L, y se intercambia ese segmento circular. A diferencia de
    un punto, el segmento puede envolver el final del cromosoma."""
    n = len(a.genes) // _unit_size(ctx)
    if n < 2:
        return a.copy(), b.copy()
    start = int(ctx.rng.integers(0, n))
    length = int(ctx.rng.integers(0, n // 2 + 1))
    mask = np.zeros(n, dtype=bool)
    mask[(start + np.arange(length)) % n] = True
    return _swap(a, b, mask, ctx)


METHODS = {
    "one_point": one_point,
    "two_point": two_point,
    "uniform": uniform,
    "annular": annular,
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"cruza '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
