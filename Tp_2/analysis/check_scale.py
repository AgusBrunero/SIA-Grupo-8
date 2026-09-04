"""¿Los hallazgos aguantan fuera de nuestra escala?

La grilla de `run_experiments.py` corre chico a propósito (20 triángulos, población
40, 300 generaciones) para poder barrer muchos ejes con varias semillas. Este script
re-chequea los ejes más frágiles a la escala que se usa en una corrida real —100
triángulos, población 100, 1500 generaciones— sobre las dos imágenes.

    python analysis/check_scale.py

Tarda del orden de 10-15 minutos: son 48 corridas largas.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga import engine
from ga.render import load_target

ROOT = Path(__file__).resolve().parent.parent

BASE = dict(
    triangles=100, canvas_size=64, population_size=100, offspring_size=100,
    selection_parents="tournament_det", selection_survivors="elite",
    crossover="uniform", crossover_granularity="gene", mutation="uniform",
    mutation_rate=0.02, replacement="additive", initialization="random",
    stop={"max_generations": 1500},
)

AXES = {
    "selección": ("selection_parents", ["elite", "roulette", "tournament_det", "ranking"]),
    "cruza": ("crossover", ["one_point", "uniform", "annular", "spatial"]),
    "inicialización": ("initialization", ["random", "grid"]),
}
IMAGES = ["images/japan.png", "images/pika.png"]
SEEDS = [1, 2]


def _run_one(job):
    axis, key, value, image, seed = job
    config = {**engine.DEFAULTS, **BASE, key: value, "seed": seed}
    target = load_target(str(ROOT / image), config["canvas_size"], config["background"])
    return axis, value, image, engine.run(config, target).best.fitness


def main() -> None:
    jobs = [
        (axis, key, value, image, seed)
        for axis, (key, values) in AXES.items()
        for value in values
        for image in IMAGES
        for seed in SEEDS
    ]
    print(f"{len(jobs)} corridas de {BASE['stop']['max_generations']} generaciones")

    results: dict[tuple, list[float]] = {}
    with ProcessPoolExecutor() as pool:
        for axis, value, image, fitness in pool.map(_run_one, jobs):
            results.setdefault((axis, image, value), []).append(fitness)

    for axis, (_key, values) in AXES.items():
        for image in IMAGES:
            print(f"\n{axis} · {Path(image).name}")
            rows = [(v, np.mean(results[(axis, image, v)]), np.std(results[(axis, image, v)])) for v in values]
            for value, mean, std in sorted(rows, key=lambda r: -r[1]):
                print(f"  {value:16s} {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
