"""Barrido de configuraciones para el análisis (Step 11).

Cada experimento varía UN eje a la vez sobre la misma configuración base y repite
cada variante con varias semillas y sobre varios targets, para poder reportar
promedio y desvío en vez de una corrida suelta, y para ver si las conclusiones
sobreviven a un tipo de imagen distinto.

    python analysis/run_experiments.py                 # todos los experimentos
    python analysis/run_experiments.py selection       # sólo uno o varios
    python analysis/run_experiments.py --quick         # grilla reducida, para probar
    python analysis/run_experiments.py --clean         # borra resultados previos

Salida en `analysis/results/` (con `--quick`, en `analysis/results_quick/`):

    <experimento>.csv   una fila por generación, variante y semilla
    summary.csv         una fila por variante (última generación de cada corrida)
    manifest.json       QUÉ configuración produjo cada número

El manifiesto es la pieza que hace auditable el barrido: guarda el commit, la
config base ya resuelta contra los defaults del motor, los overrides de cada
variante y la config completa con la que corrió, más las semillas y los targets.
Sin eso, seis meses después un CSV es una columna de números sin procedencia — y
en la presentación hay que poder decir, de cada figura, con qué se corrió.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ga import engine
from ga.render import load_target

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = Path(__file__).resolve().parent

#: claves que no describen el método sino la corrida: no cuentan como "config fija"
RUN_KEYS = {"image", "seed"}


def results_dir(quick: bool) -> Path:
    """`--quick` escribe aparte: una grilla reducida no puede pisar la buena."""
    return ANALYSIS / ("results_quick" if quick else "results")


def git_commit() -> dict:
    """Commit y estado del árbol. Los números cambian con el código que los produjo
    (ej.: el redondeo de vértices a 2 decimales movió todos los fitness)."""
    def run(*args):
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "--short", "HEAD") or "desconocido",
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def _run_one(job: tuple[str, str, str, int, dict, str]) -> list[dict]:
    experiment, target_name, variant, seed, config, batch = job
    target = load_target(str(ROOT / config["image"]), config["canvas_size"], config["background"],
                         config.get("preserve_aspect", False))
    result = engine.run(config, target)
    return [
        {
            "experiment": experiment,
            "target": target_name,
            "variant": variant,
            "seed": seed,
            **row,
            "run_batch": batch,
        }
        for row in result.history_rows()
    ]


def resolve_base(spec: dict, quick: bool) -> dict:
    base = {**engine.DEFAULTS, **spec["base"]}
    if quick:
        base = {**base, "stop": {**base["stop"], "max_generations": 30}}
    return base


def variant_config(base: dict, overrides: dict) -> dict:
    """Config completa de una variante, sin las claves propias de cada corrida."""
    return {k: v for k, v in {**base, **overrides}.items() if k not in RUN_KEYS}


def build_jobs(spec: dict, names: list[str], quick: bool, batch: str) -> list[tuple]:
    base = resolve_base(spec, quick)
    seeds = spec["seeds"][:1] if quick else spec["seeds"]

    jobs = []
    for experiment in names:
        for target_name, image in spec["targets"].items():
            for variant, overrides in spec["experiments"][experiment].items():
                for seed in seeds:
                    config = {**base, "image": image, **overrides, "seed": seed}
                    jobs.append((experiment, target_name, variant, seed, config, batch))
    return jobs


def build_manifest(spec: dict, names: list[str], quick: bool, batch: str) -> dict:
    """Todo lo necesario para reconstruir el barrido y para rotular las figuras."""
    base = resolve_base(spec, quick)
    seeds = spec["seeds"][:1] if quick else spec["seeds"]

    experiments = {}
    for experiment in names:
        variants = spec["experiments"][experiment]
        # las claves que este experimento hace variar; el resto es "config fija"
        varied = sorted({key for overrides in variants.values() for key in overrides})
        experiments[experiment] = {
            "varied_keys": varied,
            "fixed_config": {k: v for k, v in base.items() if k not in varied and k not in RUN_KEYS},
            "variants": {
                variant: {"overrides": overrides, "config": variant_config(base, overrides)}
                for variant, overrides in variants.items()
            },
        }

    return {
        "run_batch": batch,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "quick": quick,
        "seeds": seeds,
        "targets": spec["targets"],
        "base_config": {k: v for k, v in base.items() if k not in RUN_KEYS},
        "engine_defaults": engine.DEFAULTS,
        "experiments": experiments,
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def read_all_results(out_dir: Path) -> list[dict]:
    """Todos los CSV ya guardados, para que summary.csv quede completo aunque se
    haya corrido un solo experimento."""
    rows = []
    for path in sorted(out_dir.glob("*.csv")):
        if path.stem == "summary":
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                rows.append({**row, "generation": int(row["generation"])})
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Última generación de cada corrida, agregada por variante.

    Además de media y desvío se reportan mediana e IQR: con 3 semillas la media
    sola invita a declarar ganadores que están dentro del ruido.
    """
    import numpy as np

    finals: dict[tuple, dict] = {}
    for row in rows:
        key = (row["experiment"], row["target"], row["variant"], row["seed"])
        if key not in finals or row["generation"] > finals[key]["generation"]:
            finals[key] = row

    grouped: dict[tuple, list[dict]] = {}
    for (experiment, target, variant, _seed), row in finals.items():
        grouped.setdefault((experiment, target, variant), []).append(row)

    summary = []
    for (experiment, target, variant), runs in grouped.items():
        fitness = np.array([float(r["best_global_fitness"]) for r in runs])
        q1, q3 = np.percentile(fitness, [25, 75])
        summary.append(
            {
                "experiment": experiment,
                "target": target,
                "variant": variant,
                "runs": len(runs),
                "best_fitness_mean": round(float(fitness.mean()), 5),
                "best_fitness_std": round(float(fitness.std()), 5),
                "best_fitness_median": round(float(np.median(fitness)), 5),
                "best_fitness_q1": round(float(q1), 5),
                "best_fitness_q3": round(float(q3), 5),
                "rmse_mean": round(float((1 - fitness.mean()) * 255), 3),
                "diversity_mean": round(float(np.mean([float(r["diversity"]) for r in runs])), 5),
                "evaluations_mean": round(float(np.mean([float(r["evaluations"]) for r in runs])), 1),
                "seconds_mean": round(float(np.mean([float(r["elapsed"]) for r in runs])), 2),
                "run_batch": sorted({r.get("run_batch", "") for r in runs})[-1],
            }
        )
    return sorted(summary, key=lambda r: (r["experiment"], r["target"], -r["best_fitness_mean"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corre los experimentos de análisis")
    parser.add_argument("experiments", nargs="*", help="cuáles correr (default: todos)")
    parser.add_argument("--quick", action="store_true", help="grilla reducida, escribe en results_quick/")
    parser.add_argument("--clean", action="store_true", help="borra los CSV previos antes de correr")
    parser.add_argument("--spec", default="analysis/experiments.json")
    parser.add_argument("--workers", type=int, default=None, help="procesos en paralelo")
    parser.add_argument("--rebuild-manifest", action="store_true",
                        help="regenera manifest.json para TODOS los ejes del spec, sin correr nada")
    args = parser.parse_args()

    spec = json.loads((ROOT / args.spec).read_text())
    names = args.experiments or list(spec["experiments"])
    unknown = set(names) - set(spec["experiments"])
    if unknown:
        parser.error(f"experimentos desconocidos: {sorted(unknown)}. Hay: {list(spec['experiments'])}")

    out_dir = results_dir(args.quick)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_manifest:
        # El manifiesto es determinístico dado el spec: no hace falta correr nada para
        # reconstruirlo. Sirve cuando un cambio en engine.DEFAULTS hace que la fusión
        # se niegue (por diseño) y el archivo queda con un solo eje.
        previo = {}
        if (out_dir / "manifest.json").exists():
            previo = json.loads((out_dir / "manifest.json").read_text())
        todos = [n for n in spec["experiments"] if (out_dir / f"{n}.csv").exists()]
        manifest = build_manifest(spec, todos, args.quick, previo.get("run_batch", "desconocido"))
        manifest["runs"] = previo.get("runs")
        manifest["reconstruido"] = True
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"manifest.json reconstruido con {len(todos)} ejes: {', '.join(todos)}")
        return
    if args.clean:
        for path in out_dir.glob("*.csv"):
            path.unlink()
        print(f"borrados los CSV previos de {out_dir.name}/")

    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = build_manifest(spec, names, args.quick, batch)
    jobs = build_jobs(spec, names, args.quick, batch)

    print(f"batch {batch} | commit {manifest['git']['commit']}"
          f"{' (árbol sucio)' if manifest['git']['dirty'] else ''}")
    print(f"{len(jobs)} corridas ({len(names)} experimentos, {len(spec['targets'])} targets, "
          f"{len(manifest['seeds'])} semillas)")

    started = time.perf_counter()
    all_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, rows in enumerate(pool.map(_run_one, jobs), start=1):
            all_rows.extend(rows)
            print(f"\r{done}/{len(jobs)} corridas | {time.perf_counter() - started:.0f}s", end="", flush=True)
    print()

    for experiment in names:
        rows = [r for r in all_rows if r["experiment"] == experiment]
        write_csv(out_dir / f"{experiment}.csv", rows)
        print(f"  {out_dir.name}/{experiment}.csv ({len(rows)} filas)")

    manifest["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    manifest["runs"] = len(jobs)

    # Correr un solo eje no puede dejar al manifiesto sin los demás: `build_report`
    # y `caso_final` leen de ahí la configuración de CADA eje, así que se fusiona
    # con lo que ya había siempre que la base y las semillas coincidan.
    previo_path = out_dir / "manifest.json"
    if previo_path.exists() and not args.clean:
        previo = json.loads(previo_path.read_text())
        if (previo.get("base_config") == manifest["base_config"]
                and previo.get("seeds") == manifest["seeds"]
                and previo.get("targets") == manifest["targets"]):
            fusionado = {**previo["experiments"], **manifest["experiments"]}
            manifest["experiments"] = fusionado
            manifest["run_batch_previo"] = previo.get("run_batch")
        else:
            print("(!) la base/semillas/targets cambiaron: el manifiesto se reemplaza entero")
    previo_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"  {out_dir.name}/manifest.json")

    all_saved = read_all_results(out_dir)
    summary = summarize(all_saved)
    write_csv(out_dir / "summary.csv", summary)

    batches = {row["run_batch"] for row in summary if row["run_batch"]}
    if len(batches) > 1:
        print(f"\n(!) summary.csv mezcla {len(batches)} batches: {sorted(batches)}. "
              f"Corré con --clean si querés una tanda limpia.")

    print(f"\n{'experimento':14s} {'target':10s} {'variante':22s} {'fitness':>16s} {'RMSE':>7s}")
    for row in summary:
        print(
            f"{row['experiment']:14s} {row['target']:10s} {row['variant']:22s} "
            f"{row['best_fitness_mean']:.4f} ± {row['best_fitness_std']:.4f} "
            f"{row['rmse_mean']:7.2f}"
        )


if __name__ == "__main__":
    main()
