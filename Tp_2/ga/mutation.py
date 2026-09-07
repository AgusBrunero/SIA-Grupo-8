"""Métodos de mutación (Step 7).

Contrato: mutate(individual, ctx) -> Individual (muta in place y lo devuelve).

La perturbación es siempre gaussiana N(0, sigma) sobre el gen real, con clamp a
[0,1]. Lo que cambia entre métodos es CUÁNTOS genes se consideran y con qué
probabilidad muta cada uno:

    gen         un único gen elegido al azar, con probabilidad pm
    multigen    M genes elegidos al azar, cada uno con probabilidad pm
    uniforme    los N genes, cada uno con probabilidad pm  (pm constante)
    no uniforme igual que uniforme, pero pm y sigma decrecen con las generaciones

El quinto método, `zorder`, no perturba valores: intercambia dos triángulos de
lugar. Es el único que se mueve en la dimensión del orden de pintado. No entra al
barrido —es un extra documentado, no un eje—, pero está registrado en METHODS como
cualquier otro. Ver `zorder`.

Además hay una perilla ORTOGONAL, `mutation_zorder_rate`, que aplica ese mismo swap
encima de cualquiera de los otros métodos y está apagada por defecto. Ver `_swap_zorder`.
"""

from __future__ import annotations

import numpy as np

from .context import Context
from .individual import GENES_PER_TRIANGLE, Individual


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


def _swap_two(individual: Individual, ctx: Context) -> bool:
    """Intercambia dos triángulos enteros de lugar. Devuelve si hubo swap.

    Trabaja sobre bloques de `GENES_PER_TRIANGLE` genes, así que nunca parte un
    triángulo por la mitad: mueve el triángulo entero, con sus vértices y su color.
    """
    n = len(individual.genes) // GENES_PER_TRIANGLE
    if n < 2:
        return False
    bloques = individual.genes.reshape(n, GENES_PER_TRIANGLE)
    i, j = ctx.rng.choice(n, size=2, replace=False)
    bloques[[i, j]] = bloques[[j, i]]
    individual.fitness = None  # cambió el orden de pintado: hay que reevaluar
    return True


def zorder(individual: Individual, ctx: Context) -> Individual:
    """Mutación de z-order: intercambia dos triángulos de lugar en la lista.

    El locus de nuestro cromosoma **es** el z-order: qué triángulo tapa a cuál
    depende sólo de la posición en la lista. Los otros cuatro métodos perturban
    valores dentro de un locus fijo, así que ninguno puede reordenar. Y es
    justamente el eje del problema de *competing conventions*: dos individuos buenos
    pueden codificar imágenes parecidas con los triángulos en distinto orden, y ahí
    la cruza los destruye. Este operador es el único que se mueve en esa dimensión.

    A diferencia de los otros cuatro no perturba ningún valor, así que usado solo
    no puede refinar una imagen. Para combinarlo con un método de perturbación está
    la perilla `mutation_zorder_rate`, que aplica este mismo swap encima de cualquiera
    de ellos.
    """
    _swap_two(individual, ctx)
    return individual


def _swap_zorder(individual: Individual, ctx: Context) -> Individual:
    """Aplica el swap de z-order con probabilidad `mutation_zorder_rate`.

    Es una perilla ORTOGONAL: se aplica encima del método de mutación que se haya
    elegido. Por defecto vale 0.0, así que **no cambia el comportamiento de ninguna
    corrida que no la pida** — el barrido del análisis corre con ella apagada.
    """
    rate = ctx.params.get("mutation_zorder_rate", 0.0)
    if rate <= 0.0 or ctx.rng.random() >= rate:
        return individual
    _swap_two(individual, ctx)
    return individual


METHODS = {
    "gene": gene,
    "multigene": multigene,
    "uniform": uniform,
    "non_uniform": non_uniform,
    "zorder": zorder,
}


def get(name: str):
    """Devuelve el método pedido, envuelto con la perilla de z-order.

    El envoltorio es un no-op mientras `mutation_zorder_rate` sea 0 (el default),
    así que agregarlo no altera ningún resultado ya medido. Sobre `zorder` no se
    envuelve nada: la perilla haría el mismo swap dos veces, que puede deshacerlo.
    """
    if name not in METHODS:
        raise ValueError(f"mutación '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    metodo = METHODS[name]
    if metodo is zorder:
        return metodo

    def con_zorder(individual: Individual, ctx: Context) -> Individual:
        return _swap_zorder(metodo(individual, ctx), ctx)

    return con_zorder
