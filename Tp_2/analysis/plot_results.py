"""Figuras para la presentación (Step 11).

Lee analysis/results/<experimento>.csv y arma, por experimento, una fila por target
con dos paneles:
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

#: paleta propia (la de matplotlib por defecto no combina con la presentación)
PALETTE = ["#BC002D", "#2C6E8F", "#C98A00", "#4B7B4A", "#7A4A8C", "#B4553A", "#5A6470"]
PAPER, INK, GRID = "#FBF9F8", "#241E20", "#D9D1CE"

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK, "ytick.color": INK, "grid.color": GRID,
    "axes.titlesize": 12, "axes.labelsize": 10, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(path: Path) -> dict:
    """CSV -> {target: {variante: {métrica: matriz (semillas x generaciones)}}}"""
    raw: dict[tuple, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            raw[(row.get("target", "único"), row["variant"], int(row["seed"]))].append(row)

    nested: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (target, variant, _seed), rows in raw.items():
        rows.sort(key=lambda r: int(r["generation"]))
        for metric in ("best_fitness", "diversity"):
            nested[target][variant][metric].append([float(r[metric]) for r in rows])

    return {
        target: {v: {m: np.array(runs) for m, runs in metrics.items()} for v, metrics in variants.items()}
        for target, variants in nested.items()
    }


def plot(name: str, data: dict) -> Path:
    """Una fila por imagen target, dos columnas: convergencia y diversidad."""
    targets = sorted(data)
    fig, grid = plt.subplots(len(targets), 2, figsize=(13, 4.5 * len(targets)), squeeze=False)

    for axes, target in zip(grid, targets):
        for i, (variant, metrics) in enumerate(sorted(data[target].items())):
            color = PALETTE[i % len(PALETTE)]
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
                ax.grid(alpha=0.3)

        suffix = f" · imagen {target}" if target != "único" else ""
        axes[0].set_title(f"{name}{suffix} — convergencia")
        axes[1].set_title(f"{name}{suffix} — diversidad genética")
        axes[1].set_yscale("log")
        axes[0].legend(fontsize=8, frameon=False)

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
