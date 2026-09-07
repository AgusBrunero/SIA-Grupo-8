"""Estrategias de supervivencia (Step 8).

Contrato: replace(parents, children, n, ctx) -> list[Individual] de largo n.
El selector de sobrevivientes viaja en ctx.survivor_selector (lo inyecta el motor),
así la supervivencia usa los mismos métodos de selección configurables y no un
elitismo hardcodeado.

N = tamaño de población, K = cantidad de hijos generados.

    aditiva    padres e hijos compiten juntos: se seleccionan N del pool N+K
    exclusiva  los hijos desplazan a los padres: si K >= N se seleccionan N de los
               K hijos; si K < N pasan los K hijos y se completan N-K con padres

Los nombres los define el deck de AG de la cátedra (láminas 45 y 46), textual:

    Supervivencia Aditiva
        "Generando K hijos de K padres... La nueva generación se formará
         seleccionando N individuos del conjunto de [ N (individuos de la
         generación actual) + K (hijos) ]."

    Supervivencia Exclusiva
        "Generando K hijos de K padres...
         K > N : La nueva generación se genera seleccionando N de los K hijos
                 exclusivamente.
         K <= N: La nueva generación se conformará por los K hijos generados +
                 (N-K) individuos seleccionados de la generación actual."
"""

from __future__ import annotations

from .context import Context
from .individual import Individual


def additive(parents: list[Individual], children: list[Individual], n: int, ctx: Context):
    return ctx.survivor_selector(parents + children, n, ctx)


def exclusive(parents: list[Individual], children: list[Individual], n: int, ctx: Context):
    if len(children) >= n:
        return ctx.survivor_selector(children, n, ctx)
    return children + ctx.survivor_selector(parents, n - len(children), ctx)


METHODS = {
    "additive": additive,
    "exclusive": exclusive,
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"reemplazo '{name}' no implementado. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
