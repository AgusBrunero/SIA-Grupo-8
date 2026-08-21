from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sokoban_solver import SearchResult, solve_sokoban

CSV_FIELDS = [
    "timestamp",
    "level_id",
    "repetition",
    "algorithm",
    "heuristic",
    "success",
    "timeout",
    "cost",
    "expanded_nodes",
    "frontier_nodes_final",
    "frontier_nodes_max",
    "elapsed_time",
    "peak_memory_kb",
    "path",
]


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def result_to_row(
    result: SearchResult,
    *,
    level_id: str,
    repetition: int,
    save_path: bool,
) -> dict:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level_id": level_id,
        "repetition": repetition,
        "algorithm": result.algorithm,
        "heuristic": result.heuristic or "",
        "success": result.success,
        "timeout": result.timeout,
        "cost": result.cost if result.cost is not None else "",
        "expanded_nodes": result.expanded_nodes,
        "frontier_nodes_final": result.frontier_nodes_final,
        "frontier_nodes_max": result.frontier_nodes_max,
        "elapsed_time": f"{result.elapsed_time:.6f}",
        "peak_memory_kb": (
            f"{result.peak_memory_kb:.3f}" if result.peak_memory_kb is not None else ""
        ),
        "path": "",
    }
    if save_path and result.path is not None:
        row["path"] = " ".join(f"{dx},{dy}" for dx, dy in result.path)
    return row


def warmup_imports(config: dict) -> None:
    uses_hungarian = any(
        heuristic in {"hungarian", "weighted"}
        for method in config.get("methods", [])
        for heuristic in (method.get("heuristics") or [])
    )
    if uses_hungarian:
        from scipy.optimize import linear_sum_assignment  # noqa: F401


def run_benchmarks(config: dict) -> list[dict]:
    warmup_imports(config)
    save_path = bool(config.get("save_path", False))
    default_reps = int(config.get("repetitions", 5))
    default_timeout = config.get("timeout_seconds")
    default_max_expanded = config.get("max_expanded")
    measure_memory = bool(config.get("measure_memory", True))
    dead_square_pruning = bool(config.get("dead_square_pruning", True))
    rows: list[dict] = []

    for level in config["levels"]:
        level_id = level["id"]
        grid = level["grid"]
        reps = int(level.get("repetitions", default_reps))
        timeout = level.get("timeout_seconds", default_timeout)
        max_expanded = level.get("max_expanded", default_max_expanded)

        for method_cfg in config["methods"]:
            method = method_cfg["name"]
            for heuristic in method_cfg["heuristics"]:
                label = f"{method}" + (f"/{heuristic}" if heuristic else "")
                print(f"==> {level_id} | {label} | {reps} corridas")
                for repetition in range(1, reps + 1):
                    result = solve_sokoban(
                        grid,
                        method=method,
                        heuristic=heuristic,
                        timeout_seconds=timeout,
                        max_expanded=max_expanded,
                        measure_memory=measure_memory,
                        dead_square_pruning=dead_square_pruning,
                    )
                    status = "OK" if result.success else ("TIMEOUT" if result.timeout else "FAIL")
                    print(
                        f"    rep {repetition}/{reps}: {status} "
                        f"costo={result.cost} exp={result.expanded_nodes} "
                        f"t={result.elapsed_time:.4f}s"
                    )
                    rows.append(
                        result_to_row(
                            result,
                            level_id=level_id,
                            repetition=repetition,
                            save_path=save_path,
                        )
                    )
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(
    existing_path: Path,
    new_rows: list[dict],
    *,
    level_id: str,
    algorithm: str,
) -> list[dict]:
    old_rows: list[dict] = []
    if existing_path.exists():
        with existing_path.open(encoding="utf-8") as file:
            old_rows = list(csv.DictReader(file))
    kept = [
        row
        for row in old_rows
        if not (row.get("level_id") == level_id and row.get("algorithm") == algorithm)
    ]
    return kept + new_rows


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Corre benchmarks del solver de Sokoban.")
    parser.add_argument(
        "--config",
        type=Path,
        default=analysis_dir / "config.json",
        help="Archivo JSON de configuración",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=analysis_dir / "results" / "benchmarks.csv",
        help="CSV de salida (una fila por corrida)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = run_benchmarks(config)
    write_csv(rows, args.output)
    print(f"Guardado: {args.output} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
