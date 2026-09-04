"""Métodos de selección.

Contrato: select(population, k, ctx) -> list[Individual] de largo exactamente k
(con repetición permitida, salvo elite que repite sólo si k > len(population)).

MVP: sólo elite. Ruleta, universal, Boltzmann, torneos y ranking son el Step 5.
"""

from __future__ import annotations

from .context import Context
from .individual import Individual


def elite(population: list[Individual], k: int, ctx: Context) -> list[Individual]:
    ordered = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    if k <= len(ordered):
        return ordered[:k]
    # si piden más de los que hay, se repite la lista ordenada (n(i) = ceil((k-i)/N))
    return [ordered[i % len(ordered)] for i in range(k)]


METHODS = {
    "elite": elite,
    # TODO Step 5: roulette, universal, boltzmann, tournament_det, tournament_prob, ranking
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"selección '{name}' no implementada. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
