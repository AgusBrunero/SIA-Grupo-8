"""Estrategias de supervivencia (Step 8).

Contrato: replace(parents, children, n, ctx) -> list[Individual] de largo n.
El selector de sobrevivientes viaja en ctx.survivor_selector (lo inyecta el motor),
así la supervivencia usa los mismos métodos de selección configurables y no un
elitismo hardcodeado.

N = tamaño de población, K = cantidad de hijos generados.

    aditiva    padres e hijos compiten juntos: se seleccionan N del pool N+K
    exclusiva  los hijos desplazan a los padres: si K >= N se seleccionan N de los
               K hijos; si K < N pasan los K hijos y se completan N-K con padres

(!) Los nombres aditiva/exclusiva no están definidos en el enunciado; este es el
mapeo que asumimos, pendiente de confirmar con la cátedra. Si resultara al revés,
sólo hay que intercambiar las claves de METHODS: la mecánica no cambia.
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
