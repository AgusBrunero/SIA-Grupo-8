from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def method_label(row: pd.Series) -> str:
    heuristic = str(row["heuristic"]).strip()
    if heuristic and heuristic.lower() != "nan":
        return f"{row['algorithm']}\n({heuristic})"
    return str(row["algorithm"])


def method_group(algorithm: str) -> str:
    if algorithm in {"bfs", "dfs"}:
        return "No informados"
    if algorithm == "greedy":
        return "Greedy"
    return "A*"


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["heuristic"] = df["heuristic"].fillna("")
    df["label"] = df.apply(method_label, axis=1)
    df["group"] = df["algorithm"].map(method_group)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["level_id", "algorithm", "heuristic", "label", "group"], dropna=False)
        .agg(
            success_rate=("success", "mean"),
            cost_mean=("cost", "mean"),
            expanded_mean=("expanded_nodes", "mean"),
            frontier_final_mean=("frontier_nodes_final", "mean"),
            frontier_max_mean=("frontier_nodes_max", "mean"),
            time_mean=("elapsed_time", "mean"),
            time_std=("elapsed_time", "std"),
            memory_mean=("peak_memory_kb", "mean"),
        )
        .reset_index()
    )
    grouped["time_std"] = grouped["time_std"].fillna(0.0)
    grouped["memory_mb_mean"] = grouped["memory_mean"] / 1024.0
    return grouped


def _ordered_labels(summary: pd.DataFrame) -> list[str]:
    order = {"No informados": 0, "Greedy": 1, "A*": 2}
    sorted_df = summary.sort_values(
        by=["group", "algorithm", "heuristic"],
        key=lambda col: col.map(order) if col.name == "group" else col,
    )
    return list(dict.fromkeys(sorted_df["label"]))


def _bar_positions(summary: pd.DataFrame) -> tuple[list[str], list[float]]:
    labels = _ordered_labels(summary)
    positions: list[float] = []
    x = 0.0
    last_group = None
    for label in labels:
        group = summary.loc[summary["label"] == label, "group"].iloc[0]
        if last_group is not None and group != last_group:
            x += 0.6
        positions.append(x)
        x += 1.0
        last_group = group
    return labels, positions


def plot_metric(
    summary: pd.DataFrame,
    level_id: str,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    output: Path,
    log_scale: bool = False,
    error_col: str | None = None,
) -> None:
    level = summary[summary["level_id"] == level_id].copy()
    if level.empty:
        return

    labels, positions = _bar_positions(level)
    values = [level.loc[level["label"] == label, value_col].iloc[0] for label in labels]
    errors = None
    if error_col is not None:
        errors = [level.loc[level["label"] == label, error_col].iloc[0] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(positions, values, yerr=errors, capsize=4, color="#3498db", ecolor="#2c3e50")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_frontier(summary: pd.DataFrame, level_id: str, output: Path) -> None:
    level = summary[summary["level_id"] == level_id].copy()
    if level.empty:
        return

    labels, positions = _bar_positions(level)
    width = 0.35
    finals = [level.loc[level["label"] == label, "frontier_final_mean"].iloc[0] for label in labels]
    maxes = [level.loc[level["label"] == label, "frontier_max_mean"].iloc[0] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([p - width / 2 for p in positions], finals, width, label="Frontera final", color="#9b59b6")
    ax.bar([p + width / 2 for p in positions], maxes, width, label="Frontera máxima", color="#e67e22")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Nodos en frontera")
    ax.set_title(f"Frontera final vs máxima — {level_id}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_level(summary: pd.DataFrame, level_id: str, outdir: Path) -> None:
    plot_metric(
        summary,
        level_id,
        value_col="expanded_mean",
        ylabel="Nodos expandidos (media)",
        title=f"Nodos expandidos — {level_id}",
        output=outdir / f"{level_id}_expanded.png",
        log_scale=True,
    )
    plot_metric(
        summary,
        level_id,
        value_col="time_mean",
        ylabel="Tiempo (s)",
        title=f"Tiempo de procesamiento — {level_id}",
        output=outdir / f"{level_id}_time.png",
        log_scale=True,
        error_col="time_std",
    )
    plot_metric(
        summary,
        level_id,
        value_col="cost_mean",
        ylabel="Costo de la solución",
        title=f"Costo de la solución — {level_id}",
        output=outdir / f"{level_id}_cost.png",
        log_scale=False,
    )
    plot_frontier(summary, level_id, outdir / f"{level_id}_frontier.png")
    plot_metric(
        summary,
        level_id,
        value_col="memory_mb_mean",
        ylabel="Memoria pico (MB)",
        title=f"Memoria pico — {level_id}",
        output=outdir / f"{level_id}_memory.png",
        log_scale=True,
    )


def parse_args() -> argparse.Namespace:
    analysis_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Grafica resultados de benchmarks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=analysis_dir / "results" / "benchmarks.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=analysis_dir / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_results(args.input)
    summary = aggregate(df)
    for level_id in summary["level_id"].unique():
        plot_level(summary, level_id, args.outdir)
    print(f"Figuras guardadas en {args.outdir}")


if __name__ == "__main__":
    main()
