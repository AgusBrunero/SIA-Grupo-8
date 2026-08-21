from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from run_benchmarks import load_config
from sokoban_solver import parse_board, solve_sokoban

WALL = "#7f8c8d"
FLOOR = "#34495e"
OUTLINE = "#2c3e50"
TARGET = "#e74c3c"
BOX = "#f39c12"
BOX_ON_TARGET = "#2ecc71"
PLAYER = "#3498db"
PATH = "#f1c40f"


def replay_path(
    grid: list[str],
    path: list[tuple[int, int]],
) -> tuple[set, set, frozenset, tuple[int, int], list[tuple[int, int]]]:
    walls, targets, boxes, player = parse_board(grid)
    trail = [player]
    boxes_set = set(boxes)
    for dx, dy in path:
        next_player = (player[0] + dy, player[1] + dx)
        if next_player in boxes_set:
            pushed = (next_player[0] + dy, next_player[1] + dx)
            boxes_set.remove(next_player)
            boxes_set.add(pushed)
        player = next_player
        trail.append(player)
    return walls, targets, frozenset(boxes_set), player, trail


def draw_board(
    ax,
    *,
    height: int,
    width: int,
    walls: set,
    targets: set,
    boxes: set,
    player: tuple[int, int],
    trail: list[tuple[int, int]],
    title: str,
) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color="#ecf0f1", fontsize=12, pad=8)

    for row in range(height):
        for col in range(width):
            cell = (row, col)
            color = WALL if cell in walls else FLOOR
            ax.add_patch(
                Rectangle((col, row), 1, 1, facecolor=color, edgecolor=OUTLINE, linewidth=0.8)
            )
            if cell in targets:
                ax.add_patch(Circle((col + 0.5, row + 0.5), 0.14, color=TARGET, zorder=2))

    if len(trail) >= 2:
        xs = [c + 0.5 for _, c in trail]
        ys = [r + 0.5 for r, _ in trail]
        ax.plot(xs, ys, color=PATH, linewidth=2.2, alpha=0.9, zorder=3, solid_capstyle="round")
        if len(trail) <= 20:
            for i, (row, col) in enumerate(trail):
                ax.text(
                    col + 0.5,
                    row + 0.22,
                    str(i),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#2c3e50",
                    zorder=5,
                    fontweight="bold",
                )
        else:
            step = max(1, len(trail) // 12)
            for i in range(0, len(trail), step):
                row, col = trail[i]
                ax.text(
                    col + 0.5,
                    row + 0.18,
                    str(i),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#ecf0f1",
                    zorder=5,
                )

    for row, col in boxes:
        fill = BOX_ON_TARGET if (row, col) in targets else BOX
        ax.add_patch(
            Rectangle(
                (col + 0.18, row + 0.18),
                0.64,
                0.64,
                facecolor=fill,
                edgecolor="#d35400" if fill == BOX else "#27ae60",
                linewidth=1.5,
                zorder=4,
            )
        )

    pr, pc = player
    ax.add_patch(Circle((pc + 0.5, pr + 0.5), 0.28, color=PLAYER, zorder=6))


def render_level(level_id: str, grid: list[str], outdir: Path) -> None:
    result = solve_sokoban(grid, method="astar", heuristic="hungarian", measure_memory=False)
    if not result.success or result.path is None:
        raise SystemExit(f"No hay solución para {level_id}")

    walls, targets, boxes, player, trail = replay_path(grid, result.path)
    height = len(grid)
    width = max(len(row) for row in grid)
    fig_w = max(6, width * 0.55)
    fig_h = max(4, height * 0.55 + 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#2c3e50")
    ax.set_facecolor("#2c3e50")
    draw_board(
        ax,
        height=height,
        width=width,
        walls=walls,
        targets=targets,
        boxes=set(boxes),
        player=player,
        trail=trail,
        title=f"{level_id} — A* hungarian, {result.cost} pasos (estado final + camino)",
    )
    fig.tight_layout()
    output = outdir / f"{level_id}_solution.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Guardado: {output} (costo={result.cost}, exp={result.expanded_nodes})")


def render_pipeline(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.6), facecolor="#2c3e50")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    ax.set_title("Pipeline de búsqueda", color="#ecf0f1", fontsize=13, pad=6)

    boxes = [
        (0.2, "ASCII\nconfig.json"),
        (2.35, "parse_board\nwalls/targets\n(player, boxes)"),
        (4.7, "Dead squares\nBFS inverso"),
        (6.85, "Búsqueda\nfrontera + visited"),
        (9.05, "SearchResult\nGUI / CSV / PNG"),
    ]
    for x, label in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.55),
                1.9,
                1.5,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor="#34495e",
                edgecolor="#1abc9c",
                linewidth=1.4,
            )
        )
        ax.text(x + 0.95, 1.3, label, ha="center", va="center", color="#ecf0f1", fontsize=8)
    for x1, x2 in [(2.1, 2.35), (4.25, 4.7), (6.6, 6.85), (8.75, 9.05)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 1.3),
                (x2, 1.3),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#ecf0f1",
                lw=1.2,
            )
        )
    fig.tight_layout()
    output = outdir / "search_pipeline.png"
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Guardado: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Renderiza tableros resueltos y el pipeline.")
    parser.add_argument("--config", type=Path, default=ANALYSIS_DIR / "config.json")
    parser.add_argument("--outdir", type=Path, default=ANALYSIS_DIR / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    render_pipeline(args.outdir)
    for level in config["levels"]:
        render_level(level["id"], level["grid"], args.outdir)


if __name__ == "__main__":
    main()
