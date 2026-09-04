"""CLI del compresor de imágenes por triángulos.

    python main.py                              # usa config.json
    python main.py --triangles 50 --generations 2000
    python main.py --image images/germany.png --gif
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from ga import artifact, engine
from ga.render import load_target, render

ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aproxima una imagen con triángulos usando AG")
    p.add_argument("--config", default="config.json", help="archivo de configuración (default: config.json)")
    p.add_argument("--image", help="imagen target (override)")
    p.add_argument("--triangles", type=int, help="cantidad de triángulos (override)")
    p.add_argument("--canvas", type=int, help="tamaño del canvas de trabajo (override)")
    p.add_argument("--population", type=int, help="tamaño de población (override)")
    p.add_argument("--offspring", type=int, help="cantidad de hijos por generación (override)")
    p.add_argument("--generations", type=int, help="máximo de generaciones (override)")
    p.add_argument("--seed", type=int, help="semilla (override)")
    p.add_argument("--out", default="output", help="directorio de salida (default: output)")
    p.add_argument("--tag", help="sufijo para distinguir corridas")
    p.add_argument("--render-size", type=int, default=512, help="resolución de la imagen final")
    p.add_argument("--gif", action="store_true", help="además, guarda un gif de la evolución")
    p.add_argument("--quiet", action="store_true", help="no imprime el progreso")
    p.add_argument(
        "--rebuild",
        metavar="TRIANGLES_JSON",
        help="reconstruye la imagen desde un triangles.json y termina (no corre el AG)",
    )
    return p.parse_args()


def rebuild(path: Path) -> None:
    """Camino inverso: archivo de triángulos -> imagen.

    Si al lado está el PNG que produjo la corrida, compara y reporta la
    diferencia: así se verifica que el archivo es autosuficiente.
    """
    document = artifact.load(path)
    image = artifact.render(document)
    out = path.parent / "rebuilt.png"
    image.save(out)
    print(f"reconstruido desde {path} -> {out}")
    print(f"  {len(document['triangles'])} triángulos, canvas {document['canvas']['width']}x{document['canvas']['height']}")

    original = path.parent / "best.png"
    if original.exists():
        import numpy as np

        diff = np.abs(
            np.asarray(image, dtype=np.int16) - np.asarray(Image.open(original).convert("RGB"), dtype=np.int16)
        )
        print(f"  diferencia máxima contra {original.name}: {int(diff.max())} (0 = idénticas)")


def build_config(args) -> dict:
    config = json.loads((ROOT / args.config).read_text())
    overrides = {
        "image": args.image,
        "triangles": args.triangles,
        "canvas_size": args.canvas,
        "population_size": args.population,
        "offspring_size": args.offspring,
        "seed": args.seed,
    }
    config.update({k: v for k, v in overrides.items() if v is not None})
    if args.generations is not None:
        config.setdefault("stop", {})["max_generations"] = args.generations
    return config


def main() -> None:
    args = parse_args()
    if args.rebuild:
        rebuild(Path(args.rebuild))
        return

    config = build_config(args)
    canvas = config.get("canvas_size", engine.DEFAULTS["canvas_size"])
    background = config.get("background", engine.DEFAULTS["background"])
    target = load_target(str(ROOT / config["image"]), canvas, background)

    name = Path(config["image"]).stem + (f"-{args.tag}" if args.tag else "")
    out_dir = ROOT / args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    last_print = [0.0]

    def on_generation(record, best):
        if args.gif and record.generation % 25 == 0:
            frames.append(render(best, 128, background))
        if not args.quiet and (time.perf_counter() - last_print[0] > 0.5):
            last_print[0] = time.perf_counter()
            print(
                f"\rgen {record.generation:5d} | best {record.best_fitness:.4f} "
                f"| mean {record.mean_fitness:.4f} | div {record.diversity:.4f} "
                f"| {record.elapsed:6.1f}s",
                end="",
                flush=True,
            )

    print(f"target: {config['image']}  triángulos: {config.get('triangles')}  canvas: {canvas}px")
    result = engine.run(config, target, on_generation=on_generation)
    if not args.quiet:
        print()

    # 1) la enumeración de triángulos: documento autosuficiente del que sale todo
    #    lo demás (el genotipo es independiente de la resolución)
    document = artifact.build(
        result.best,
        args.render_size,
        args.render_size,
        background,
        source_image=config["image"],
        fitness=result.best.fitness,
    )
    artifact.save(document, out_dir / "triangles.json")

    # 2) imagen generada, renderizada DESDE el documento: así el PNG entregado y
    #    el archivo describen la misma imagen por construcción
    best_image = artifact.render(document)
    best_image.save(out_dir / "best.png")

    # 3) comparación target vs. resultado
    side = Image.new("RGB", (args.render_size * 2, args.render_size), "white")
    side.paste(Image.open(ROOT / config["image"]).convert("RGB").resize((args.render_size,) * 2), (0, 0))
    side.paste(best_image, (args.render_size, 0))
    side.save(out_dir / "comparison.png")

    # 4) métricas por generación
    rows = result.history_rows()
    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 5) resumen reproducible de la corrida
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "config": config,
                "best_fitness": result.best.fitness,
                "rmse": (1 - result.best.fitness) * 255,
                "generations": result.generations,
                "evaluations": result.evaluations,
                "elapsed_seconds": round(result.elapsed, 2),
                "stop_reason": result.stop_reason,
            },
            indent=2,
        )
    )

    if frames:
        frames[0].save(out_dir / "evolution.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

    print(
        f"fitness {result.best.fitness:.4f} (RMSE {(1 - result.best.fitness) * 255:.2f}) "
        f"| {result.generations} generaciones | {result.evaluations} evaluaciones "
        f"| {result.elapsed:.1f}s | corte: {result.stop_reason}"
    )
    print(f"salida en {out_dir}")


if __name__ == "__main__":
    main()
