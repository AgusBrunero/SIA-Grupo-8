"""Barrido de configuraciones para el análisis (Step 11).

Cada experimento varía UN eje a la vez sobre la misma configuración base y repite
cada variante con varias semillas, para poder reportar promedio y desvío en vez de
una corrida suelta.

    python analysis/run_experiments.py                 # todos los experimentos
    python analysis/run_experiments.py selection       # sólo uno o varios
    python analysis/run_experiments.py --quick         # grilla reducida, para probar

Salida: analysis/results/<experimento>.csv (una fila por generación, variante y
semilla) y analysis/results/summary.csv (una fila por variante).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga import engine
from ga.render import load_target

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).parent / "results"


def _run_one(job: tuple[str, str, int, dict]) -> list[dict]:
    experiment, variant, seed, config = job
    target = load_target(str(ROOT / config["image"]), config["canvas_size"], config["background"])
    result = engine.run(config, target)
    return [
        {"experiment": experiment, "variant": variant, "seed": seed, **row}
        for row in result.history_rows()
    ]


def build_jobs(spec: dict, names: list[str], quick: bool) -> list[tuple]:
    base = {**engine.DEFAULTS, **spec["base"]}
    seeds = spec["seeds"][:1] if quick else spec["seeds"]
    if quick:
        base = {**base, "stop": {**base["stop"], "max_generations": 30}}

    jobs = []
    for experiment in names:
        for variant, overrides in spec["experiments"][experiment].items():
            for seed in seeds:
                jobs.append((experiment, variant, seed, {**base, **overrides, "seed": seed}))
    return jobs


def read_all_results() -> list[dict]:
    """Todos los CSV ya guardados, para que summary.csv quede completo aunque se
    haya corrido un solo experimento."""
    rows = []
    for path in sorted(RESULTS.glob("*.csv")):
        if path.stem == "summary":
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                rows.append({**row, "generation": int(row["generation"])})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Última generación de cada corrida, agregada por variante."""
    import numpy as np

    finals: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        finals.setdefault((row["experiment"], row["variant"], row["seed"]), row)
        if row["generation"] > finals[(row["experiment"], row["variant"], row["seed"])]["generation"]:
            finals[(row["experiment"], row["variant"], row["seed"])] = row

    grouped: dict[tuple[str, str], list[dict]] = {}
    for (experiment, variant, _seed), row in finals.items():
        grouped.setdefault((experiment, variant), []).append(row)

    summary = []
    for (experiment, variant), runs in grouped.items():
        fitness = np.array([float(r["best_fitness"]) for r in runs])
        summary.append(
            {
                "experiment": experiment,
                "variant": variant,
                "runs": len(runs),
                "best_fitness_mean": round(float(fitness.mean()), 5),
                "best_fitness_std": round(float(fitness.std()), 5),
                "rmse_mean": round(float((1 - fitness.mean()) * 255), 3),
                "diversity_mean": round(float(np.mean([float(r["diversity"]) for r in runs])), 5),
                "seconds_mean": round(float(np.mean([float(r["elapsed"]) for r in runs])), 2),
            }
        )
    return sorted(summary, key=lambda r: (r["experiment"], -r["best_fitness_mean"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corre los experimentos de análisis")
    parser.add_argument("experiments", nargs="*", help="cuáles correr (default: todos)")
    parser.add_argument("--quick", action="store_true", help="grilla reducida para verificar")
    parser.add_argument("--spec", default="analysis/experiments.json")
    parser.add_argument("--workers", type=int, default=None, help="procesos en paralelo")
    args = parser.parse_args()

    spec = json.loads((ROOT / args.spec).read_text())
    names = args.experiments or list(spec["experiments"])
    unknown = set(names) - set(spec["experiments"])
    if unknown:
        parser.error(f"experimentos desconocidos: {sorted(unknown)}. Hay: {list(spec['experiments'])}")

    jobs = build_jobs(spec, names, args.quick)
    print(f"{len(jobs)} corridas ({len(names)} experimentos)")

    started = time.perf_counter()
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, rows in enumerate(pool.map(_run_one, jobs), start=1):
            all_rows.extend(rows)
            print(f"\r{done}/{len(jobs)} corridas | {time.perf_counter() - started:.0f}s", end="", flush=True)
    print()

    for experiment in names:
        rows = [r for r in all_rows if r["experiment"] == experiment]
        write_csv(RESULTS / f"{experiment}.csv", rows)
        print(f"  {RESULTS.name}/{experiment}.csv ({len(rows)} filas)")

    summary = summarize(read_all_results())
    write_csv(RESULTS / "summary.csv", summary)
    print(f"\n{'experimento':14s} {'variante':22s} {'fitness':>16s} {'RMSE':>7s} {'seg':>6s}")
    for row in summary:
        print(
            f"{row['experiment']:14s} {row['variant']:22s} "
            f"{row['best_fitness_mean']:.4f} ± {row['best_fitness_std']:.4f} "
            f"{row['rmse_mean']:7.2f} {row['seconds_mean']:6.1f}"
        )


if __name__ == "__main__":
    main()
