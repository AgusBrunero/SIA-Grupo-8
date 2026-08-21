from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from run_benchmarks import load_config
from sokoban_solver import manhattan_hungarian, manhattan_hungarian_weighted, manhattan_simple, parse_board


def optimal_cost_by_level(benchmarks: Path) -> dict[str, float]:
    df = pd.read_csv(benchmarks)
    bfs = df[(df["algorithm"] == "bfs") & (df["success"] == True)]
    return bfs.groupby("level_id")["cost"].mean().to_dict()


def compute_informedness(config: dict, optima: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    heuristics = {
        "simple": manhattan_simple,
        "hungarian": manhattan_hungarian,
        "weighted": manhattan_hungarian_weighted,
    }
    for level in config["levels"]:
        level_id = level["id"]
        optimal = optima.get(level_id)
        if not optimal:
            continue
        _, targets, boxes, _ = parse_board(level["grid"])
        for name, fn in heuristics.items():
            h = fn(boxes, targets)
            rows.append(
                {
                    "level_id": level_id,
                    "heuristic": name,
                    "h": h,
                    "h_star": optimal,
                    "informedness": h / optimal,
                }
            )
    return rows


def plot_informedness(rows: list[dict], outdir: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for level_id, level_df in df.groupby("level_id"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(level_df["heuristic"], level_df["informedness"], color="#1abc9c")
        ymax = max(1.15, float(level_df["informedness"].max()) * 1.15)
        ax.set_ylim(0, ymax)
        ax.axhline(1.0, color="#c0392b", linestyle="--", linewidth=1, label="h = h* (límite admisible)")
        ax.legend()
        ax.set_ylabel("h / h*")
        ax.set_title(f"Informedness de heurísticas — {level_id}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(outdir / f"{level_id}_informedness.png", dpi=150)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Calcula informedness h/h* por heurística.")
    parser.add_argument("--config", type=Path, default=analysis_dir / "config.json")
    parser.add_argument("--benchmarks", type=Path, default=analysis_dir / "results" / "benchmarks.csv")
    parser.add_argument("--output", type=Path, default=analysis_dir / "results" / "informedness.csv")
    parser.add_argument("--outdir", type=Path, default=analysis_dir / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = compute_informedness(config, optimal_cost_by_level(args.benchmarks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["level_id", "heuristic", "h", "h_star", "informedness"])
        writer.writeheader()
        writer.writerows(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_informedness(rows, args.outdir)
    print(f"Guardado: {args.output}")
    for row in rows:
        print(
            f"{row['level_id']} {row['heuristic']}: "
            f"h={row['h']:.1f} h*={row['h_star']:.1f} "
            f"informedness={row['informedness']:.3f}"
        )


if __name__ == "__main__":
    main()
