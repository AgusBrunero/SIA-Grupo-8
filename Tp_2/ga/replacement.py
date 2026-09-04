"""Estrategias de supervivencia.

Contrato: replace(parents, children, n, ctx) -> list[Individual] de largo n.
El selector de sobrevivientes viaja en ctx.survivor_selector (lo inyecta el motor).

MVP: sólo aditiva. La exclusiva es el Step 8.
"""

from __future__ import annotations

from .context import Context
from .individual import Individual


def additive(parents: list[Individual], children: list[Individual], n: int, ctx: Context):
    """Llenado completo: se compite en el pool N+K y sobreviven los n mejores."""
    return ctx.survivor_selector(parents + children, n, ctx)


METHODS = {
    "additive": additive,
    # TODO Step 8: exclusive
}


def get(name: str):
    if name not in METHODS:
        raise ValueError(f"reemplazo '{name}' no implementado. Disponibles: {sorted(METHODS)}")
    return METHODS[name]
