from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from plot_results import aggregate, load_results, plot_level
from run_benchmarks import load_config, merge_rows, run_benchmarks, write_csv


def main() -> None:
    config = load_config(ANALYSIS_DIR / "config.json")
    config["methods"] = [{"name": "astar", "heuristics": ["simple", "hungarian"]}]
    config["levels"] = [level for level in config["levels"] if level["id"] == "level3"]
    if not config["levels"]:
        raise SystemExit("No se encontró level3 en config.json")
    config["levels"][0]["timeout_seconds"] = 180
    config["levels"][0]["repetitions"] = 3

    print("Re-ejecutando A* en level3 con timeout=180s")
    new_rows = run_benchmarks(config)

    csv_path = ANALYSIS_DIR / "results" / "benchmarks.csv"
    merged = merge_rows(csv_path, new_rows, level_id="level3", algorithm="astar")
    write_csv(merged, csv_path)
    print(f"CSV actualizado: {csv_path} ({len(merged)} filas)")

    figures = ANALYSIS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = aggregate(load_results(csv_path))
    for level_id in summary["level_id"].unique():
        plot_level(summary, level_id, figures)
    print(f"Figuras guardadas en {figures}")


if __name__ == "__main__":
    main()
