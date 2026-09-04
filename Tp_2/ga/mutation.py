"""Métodos de mutación (Step 7).

Contrato: mutate(individual, ctx) -> Individual (muta in place y lo devuelve).

La perturbación es siempre gaussiana N(0, sigma) sobre el gen real, con clamp a
[0,1]. Lo que cambia entre métodos es CUÁNTOS genes se consideran y con qué
probabilidad muta cada uno:

    gen         un único gen elegido al azar, con probabilidad pm
    multigen    M genes elegidos al azar, cada uno con probabilidad pm
    uniforme    los N genes, cada uno con probabilidad pm  (pm constante)
    no uniforme igual que uniforme, pero pm y sigma decrecen con las generaciones
"""

from __future__ import annotations

import numpy as np

from .context import Context
from .individual import Individual


def _perturb(individual: Individual, idx: np.ndarray, sigma: float, ctx: Context) -> None:
    if idx.size == 0:
        return
    noise = ctx.rng.normal(0.0, sigma, size=idx.shape)
    individual.genes[idx] = np.clip(individual.genes[idx] + noise, 0.0, 1.0)
    individual.fitness = None  # invalida el caché


def _rate_and_sigma(ctx: Context) -> tuple[float, float]:
    return ctx.params.get("mutation_rate", 0.1), ctx.params.get("mutation_sigma", 0.1)


def gene(individual: Individual, ctx: Context) -> Individual:
    rate, sigma = _rate_and_sigma(ctx)
    if ctx.rng.random() < rate:
        _perturb(individual, ctx.rng.integers(0, len(individual.genes), size=1), sigma, ctx)
    return individual


def multigene(individual: Individual, ctx: Context) -> Individual:
    """M genes candidatos; si no se configura M, se sortea entre 1 y N."""
    rate, sigma = _rate_and_sigma(ctx)
    n = len(individual.genes)
    m = ctx.params.get("mutation_genes") or int(ctx.rng.integers(1, n + 1))
    candidates = ctx.rng.choice(n, size=min(m, n), replace=False)
    _perturb(individual, candidates[ctx.rng.random(len(candidates)) < rate], sigma, ctx)
    return individual


def uniform(individual: Individual, ctx: Context) -> Individual:
    rate, sigma = _rate_and_sigma(ctx)
    n = len(individual.genes)
    _perturb(individual, np.flatnonzero(ctx.rng.random(n) < rate), sigma, ctx)
    return individual


def non_uniform(individual: Individual, ctx: Context) -> Individual:
    """pm y sigma decrecen linealmente con el avance de la corrida hasta un piso.
    Exploración amplia al principio, ajuste fino al final."""
    rate, sigma = _rate_and_sigma(ctx)
    floor = ctx.params.get("mutation_decay_floor", 0.1)
    factor = max(floor, 1.0 - ctx.progress)
    n = len(individual.genes)
    _perturb(individual, np.flatnonzero(ctx.rng.random(n) < rate * factor), sigma * factor, ctx)
    return individual


METHODS = {
    "gene": gene,
    "multigene": multigene,
    "uniform": uniform,
    "non_uniform": non_uniform,
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"mutación '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
