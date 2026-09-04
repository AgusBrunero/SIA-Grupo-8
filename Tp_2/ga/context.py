"""Contexto compartido por todos los operadores.

Es el contrato entre los módulos: cualquier operador (selección, cruza, mutación,
reemplazo, corte) recibe el mismo `Context` y saca de ahí lo que necesita —
generación actual, rng, hiperparámetros. Así se agregan métodos nuevos sin tocar
el motor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Context:
    rng: np.random.Generator
    params: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    max_generations: int = 0
    #: selector usado por las estrategias de reemplazo (inyectado por el motor)
    survivor_selector: Callable | None = None

    @property
    def progress(self) -> float:
        """Avance de la corrida en [0, 1]; lo usan los operadores no uniformes."""
        if self.max_generations <= 0:
            return 0.0
        return min(1.0, self.generation / self.max_generations)
