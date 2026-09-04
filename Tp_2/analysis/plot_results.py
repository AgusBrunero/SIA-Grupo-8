"""Figuras para la presentación (Step 11).

Lee analysis/results/<experimento>.csv y arma, por experimento, un panel con:
  - mejor fitness por generación (promedio entre semillas, banda = ± desvío)
  - diversidad genética por generación

    python analysis/plot_results.py            # todos los CSV que existan
    python analysis/plot_results.py selection
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
FIGURES = Path(__file__).parent / "figures"


def load(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """CSV -> {variante: {métrica: matriz (semillas x generaciones)}}"""
    raw: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            raw[(row["variant"], int(row["seed"]))].append(row)

    by_variant: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for (variant, _seed), rows in raw.items():
        rows.sort(key=lambda r: int(r["generation"]))
        for metric in ("best_fitness", "diversity"):
            by_variant[variant][metric].append([float(r[metric]) for r in rows])
    return {v: {m: np.array(runs) for m, runs in metrics.items()} for v, metrics in by_variant.items()}


def plot(name: str, data: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (variant, metrics) in enumerate(sorted(data.items())):
        color = colors[i % 10]
        for ax, metric, label in zip(axes, ("best_fitness", "diversity"), ("mejor fitness", "diversidad")):
            values = metrics[metric]
            generations = np.arange(1, values.shape[1] + 1)
            mean, std = values.mean(axis=0), values.std(axis=0)
            ax.plot(generations, mean, label=variant, color=color, linewidth=1.6)
            if values.shape[0] > 1:
                # la diversidad va en escala log: la banda no puede bajar de 0
                lower = np.maximum(mean - std, mean * 1e-2 if metric == "diversity" else -np.inf)
                ax.fill_between(generations, lower, mean + std, color=color, alpha=0.15)
            ax.set_xlabel("generación")
            ax.set_ylabel(label)

    axes[0].set_title(f"{name} — convergencia")
    axes[1].set_title(f"{name} — diversidad genética")
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafica los resultados de los experimentos")
    parser.add_argument("experiments", nargs="*", help="cuáles graficar (default: todos)")
    args = parser.parse_args()

    paths = [RESULTS / f"{n}.csv" for n in args.experiments] if args.experiments else [
        p for p in sorted(RESULTS.glob("*.csv")) if p.stem != "summary"
    ]
    if not paths:
        parser.error(f"no hay CSV en {RESULTS}. Corré primero analysis/run_experiments.py")

    for path in paths:
        if not path.exists():
            parser.error(f"falta {path}. Corré primero analysis/run_experiments.py {path.stem}")
        print(f"escrito {plot(path.stem, load(path))}")


if __name__ == "__main__":
    main()
