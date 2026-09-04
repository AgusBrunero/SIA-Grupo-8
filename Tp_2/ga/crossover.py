"""Métodos de cruza.

Contrato: crossover(parent_a, parent_b, ctx) -> (child_a, child_b).

Trabajan sobre el vector plano de genes. `granularity` decide dónde puede caer el
corte: "gene" corta en cualquier posición (puede partir un triángulo al medio),
"triangle" corta sólo en múltiplos de GENES_PER_TRIANGLE (preserva triángulos
enteros). Comparar ambas es uno de los experimentos del TP.

MVP: sólo un punto. Dos puntos, uniforme y anular son el Step 6.
"""

from __future__ import annotations

import numpy as np

from .context import Context
from .individual import GENES_PER_TRIANGLE, Individual


def _cut_points(n_genes: int, ctx: Context) -> np.ndarray:
    """Posiciones válidas de corte según la granularidad configurada."""
    if ctx.params.get("crossover_granularity", "gene") == "triangle":
        return np.arange(1, n_genes // GENES_PER_TRIANGLE) * GENES_PER_TRIANGLE
    return np.arange(1, n_genes)


def one_point(a: Individual, b: Individual, ctx: Context) -> tuple[Individual, Individual]:
    options = _cut_points(len(a.genes), ctx)
    if len(options) == 0:
        return a.copy(), b.copy()
    p = int(ctx.rng.choice(options))
    child_a = np.concatenate([a.genes[:p], b.genes[p:]])
    child_b = np.concatenate([b.genes[:p], a.genes[p:]])
    return Individual(child_a), Individual(child_b)


METHODS = {
    "one_point": one_point,
    # TODO Step 6: two_point, uniform, annular
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"cruza '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
