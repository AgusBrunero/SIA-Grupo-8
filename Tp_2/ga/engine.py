"""Motor genérico: población inicial + loop generacional.

El loop no conoce ningún método concreto; toma los operadores por nombre desde la
configuración. Agregar un método de selección/cruza/mutación/reemplazo nuevo es
registrarlo en el METHODS de su módulo, sin tocar este archivo.

    selección de padres -> cruza -> mutación -> evaluación -> reemplazo -> corte
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import crossover as crossover_mod
from . import mutation as mutation_mod
from . import replacement as replacement_mod
from . import selection as selection_mod
from .context import Context
from .fitness import FitnessEvaluator
from .individual import Individual, random_individual

DEFAULTS = {
    "triangles": 20,
    "canvas_size": 64,
    "background": [255, 255, 255],
    "population_size": 50,
    "offspring_size": 50,
    "selection_parents": "elite",
    "selection_survivors": "elite",
    "crossover": "one_point",
    "crossover_rate": 0.85,
    "crossover_granularity": "gene",
    "mutation": "gene",
    "mutation_rate": 0.5,
    "mutation_sigma": 0.1,
    "replacement": "additive",
    "stop": {"max_generations": 500, "max_seconds": None, "target_fitness": None, "stall_generations": None},
    "seed": None,
}


@dataclass
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    diversity: float
    evaluations: int
    elapsed: float


@dataclass
class Result:
    best: Individual
    history: list[GenerationRecord] = field(default_factory=list)
    stop_reason: str = ""
    generations: int = 0
    evaluations: int = 0
    elapsed: float = 0.0

    def history_rows(self) -> list[dict]:
        return [asdict(r) for r in self.history]


def _diversity(population: list[Individual]) -> float:
    """Desvío promedio por gen: proxy barato de diversidad genética."""
    return float(np.stack([ind.genes for ind in population]).std(axis=0).mean())


def _stop_reason(config, record, stalled) -> str | None:
    stop = {**DEFAULTS["stop"], **config.get("stop", {})}
    if record.generation >= stop["max_generations"]:
        return "max_generations"
    if stop["max_seconds"] is not None and record.elapsed >= stop["max_seconds"]:
        return "max_seconds"
    if stop["target_fitness"] is not None and record.best_fitness >= stop["target_fitness"]:
        return "target_fitness"
    if stop["stall_generations"] is not None and stalled >= stop["stall_generations"]:
        return "content"
    return None


def run(config: dict, target: np.ndarray, on_generation=None) -> Result:
    cfg = {**DEFAULTS, **config}
    rng = np.random.default_rng(cfg["seed"])
    ctx = Context(rng=rng, params=cfg, max_generations=cfg["stop"].get("max_generations", 0))

    select_parents = selection_mod.get(cfg["selection_parents"])
    ctx.survivor_selector = selection_mod.get(cfg["selection_survivors"])
    cross = crossover_mod.get(cfg["crossover"])
    mutate = mutation_mod.get(cfg["mutation"])
    replace = replacement_mod.get(cfg["replacement"])

    evaluator = FitnessEvaluator(target, cfg["background"])
    n, k = cfg["population_size"], cfg["offspring_size"]

    population = [random_individual(cfg["triangles"], rng) for _ in range(n)]
    evaluator.evaluate_all(population)

    started = time.perf_counter()
    result = Result(best=max(population, key=lambda i: i.fitness).copy())
    stalled = 0

    generation = 0
    while True:
        generation += 1
        ctx.generation = generation

        parents = select_parents(population, k + (k % 2), ctx)
        children = []
        for a, b in zip(parents[::2], parents[1::2]):
            if rng.random() < cfg["crossover_rate"]:
                c1, c2 = cross(a, b, ctx)
            else:
                c1, c2 = a.copy(), b.copy()
            children.extend((mutate(c1, ctx), mutate(c2, ctx)))
        children = children[:k]

        evaluator.evaluate_all(children)
        population = replace(population, children, n, ctx)

        fitnesses = np.array([ind.fitness for ind in population])
        best = max(population, key=lambda i: i.fitness)
        improved = best.fitness > result.best.fitness + 1e-12
        if improved:
            result.best = best.copy()
        stalled = 0 if improved else stalled + 1

        record = GenerationRecord(
            generation=generation,
            best_fitness=float(fitnesses.max()),
            mean_fitness=float(fitnesses.mean()),
            std_fitness=float(fitnesses.std()),
            diversity=_diversity(population),
            evaluations=evaluator.evaluations,
            elapsed=time.perf_counter() - started,
        )
        result.history.append(record)
        if on_generation is not None:
            on_generation(record, result.best)

        reason = _stop_reason(cfg, record, stalled)
        if reason:
            result.stop_reason = reason
            break

    result.generations = generation
    result.evaluations = evaluator.evaluations
    result.elapsed = time.perf_counter() - started
    return result
