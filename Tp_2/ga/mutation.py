"""Métodos de mutación.

Contrato: mutate(individual, ctx) -> Individual (muta in place y devuelve el mismo).
La perturbación es gaussiana N(0, sigma) sobre genes reales, con clamp a [0, 1].

MVP: sólo mutación de gen. Multigen, uniforme y no uniforme son el Step 7.
"""

from __future__ import annotations

import numpy as np

from .context import Context
from .individual import Individual


def _perturb(individual: Individual, idx, sigma: float, ctx: Context) -> None:
    idx = np.atleast_1d(idx)
    noise = ctx.rng.normal(0.0, sigma, size=idx.shape)
    individual.genes[idx] = np.clip(individual.genes[idx] + noise, 0.0, 1.0)
    individual.fitness = None


def gene(individual: Individual, ctx: Context) -> Individual:
    """Muta UN gen elegido al azar, con probabilidad `mutation_rate`."""
    if ctx.rng.random() >= ctx.params.get("mutation_rate", 0.1):
        return individual
    idx = ctx.rng.integers(0, len(individual.genes))
    _perturb(individual, idx, ctx.params.get("mutation_sigma", 0.1), ctx)
    return individual


METHODS = {
    "gene": gene,
    # TODO Step 7: multigene, uniform, non_uniform
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"mutación '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
