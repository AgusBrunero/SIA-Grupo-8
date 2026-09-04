"""Criterios de corte (Step 9).

Se evalúan todos por OR: corta el primero que se cumple, y el motor reporta cuál
fue. Un criterio devuelve True cuando hay que frenar.

    max_generations       cota dura, siempre activa
    max_seconds           cota de tiempo
    target_fitness        entorno a la solución (el "error mínimo" opcional del enunciado)
    stall_generations     CONTENIDO: el mejor fitness no mejora en G generaciones
    structure_generations ESTRUCTURA: la población casi no cambia en G generaciones
                          (menos de structure_epsilon de recambio genético)
"""

from __future__ import annotations

DEFAULTS = {
    "max_generations": 500,
    "max_seconds": None,
    "target_fitness": None,
    "stall_generations": None,
    "structure_generations": None,
    "structure_epsilon": 0.01,
}

CRITERIA = {
    "max_generations": lambda cfg, s: s.generation >= cfg["max_generations"],
    "max_seconds": lambda cfg, s: s.elapsed >= cfg["max_seconds"],
    "target_fitness": lambda cfg, s: s.best_fitness >= cfg["target_fitness"],
    "content": lambda cfg, s: s.stalled >= cfg["stall_generations"],
    "structure": lambda cfg, s: s.structure_stable >= cfg["structure_generations"],
}

#: cada criterio se ignora si su parámetro quedó en null
REQUIRES = {
    "max_generations": "max_generations",
    "max_seconds": "max_seconds",
    "target_fitness": "target_fitness",
    "content": "stall_generations",
    "structure": "structure_generations",
}


def resolve(stop_config: dict | None) -> dict:
    return {**DEFAULTS, **(stop_config or {})}


def check(cfg: dict, state) -> str | None:
    """Devuelve el nombre del criterio que se cumplió, o None."""
    for name, predicate in CRITERIA.items():
        if cfg.get(REQUIRES[name]) is not None and predicate(cfg, state):
            return name
    return None
