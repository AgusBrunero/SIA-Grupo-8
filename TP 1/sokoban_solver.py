from __future__ import annotations

import heapq
import time
import tracemalloc
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

UNINFORMED_METHODS = {"bfs", "dfs"}
INFORMED_METHODS = {"greedy", "astar"}
VALID_METHODS = UNINFORMED_METHODS | INFORMED_METHODS
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
WEIGHTED_HUNGARIAN_FACTOR = 3
HEURISTIC_WEIGHTS = {
    "astar": 0.5,
    "greedy": 1.0,
    "bfs": 0.0,
}


@dataclass
class SearchResult:
    success: bool
    cost: int | None
    path: list[tuple[int, int]] | None
    expanded_nodes: int
    frontier_nodes_final: int
    frontier_nodes_max: int
    elapsed_time: float
    algorithm: str
    heuristic: str | None
    timeout: bool = False
    peak_memory_kb: float | None = None

    def to_dict(self, include_path: bool = True) -> dict:
        data = asdict(self)
        if not include_path:
            data.pop("path", None)
        return data


def parse_board(grid):
    walls, targets, boxes = set(), set(), set()
    player = None
    
    for row_index, row in enumerate(grid):
        for column_index, char in enumerate(row):
            if char == '#':
                walls.add((row_index, column_index))
            elif char == '.':
                targets.add((row_index, column_index))
            elif char == '$':
                boxes.add((row_index, column_index))
            elif char == '@':
                player = (row_index, column_index)
            elif char == '*':  
                targets.add((row_index, column_index))
                boxes.add((row_index, column_index))
            elif char == '+': 
                targets.add((row_index, column_index))
                player = (row_index, column_index)
                
    return walls, targets, frozenset(boxes), player

def manhattan_simple(boxes, targets):
    """
    Heuristic for A*: Manhattan distance
    From one box to its closest objective
    """
    return sum(
        min(abs(box_row - target_row) + abs(box_column - target_column) for target_row, target_column in targets)
        for box_row, box_column in boxes
    )

def manhattan_hungarian(boxes, targets):
    """
    Heuristic for A*: Hungarian with Manhattan distance
    Considers the shortest distance for every box to its own objective
    """
    from scipy.optimize import linear_sum_assignment

    boxes = list(boxes)
    targets = list(targets)

    cost_matrix = [
        [
            abs(box_row - target_row) + abs(box_column - target_column)
            for target_row, target_column in targets
        ]
        for box_row, box_column in boxes
    ]

    rows, columns = linear_sum_assignment(cost_matrix)
    return sum(
        cost_matrix[row][column] for row, column in zip(rows, columns)
    )


def manhattan_hungarian_weighted(boxes, targets):
    """
    Heurística no admisible: Hungarian multiplicada por 3.

    En A* (f = 0.5 h + 0.5 g) equivale a Weighted A* con w=3, o sea f ~ g + 3 h_hung.
    Puede sobreestimar el costo real; no garantiza optimalidad.
    """
    return WEIGHTED_HUNGARIAN_FACTOR * manhattan_hungarian(boxes, targets)

heuristics = {
    "hungarian": manhattan_hungarian,
    "simple": manhattan_simple,
    "weighted": manhattan_hungarian_weighted,
}

def is_blocked(box, walls, targets):
    if box in targets:
        return False
    row, column = box
    return ((row-1,column) in walls or (row+1,column) in walls) and ((row,column-1) in walls or (row,column+1) in walls)


def playable_floor(walls: set, origin: tuple, height: int, width: int) -> set:
    """Celdas no-muro alcanzables desde el jugador, ignorando cajas."""
    if origin is None:
        return set()
    seen = {origin}
    queue = deque([origin])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in DIRECTIONS:
            nxt = (row + d_row, col + d_col)
            nxt_row, nxt_col = nxt
            if not (0 <= nxt_row < height and 0 <= nxt_col < width):
                continue
            if nxt in walls or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def compute_dead_squares(walls, targets, floor: set) -> set:
    """
    Casillas desde las que una caja no puede llegar a ningún objetivo.

    BFS inverso (unpush): si la caja está en curr tras un empuje en dirección d,
    venía de prev = curr - d y el jugador tenía que estar en prev - d.
    Toda casilla de piso no alcanzada desde los goals es un dead square.
    """
    alive: set = set()
    queue: deque = deque()
    for target in targets:
        if target in floor:
            alive.add(target)
            queue.append(target)

    while queue:
        curr_row, curr_col = queue.popleft()
        for d_row, d_col in DIRECTIONS:
            prev = (curr_row - d_row, curr_col - d_col)
            player_cell = (curr_row - 2 * d_row, curr_col - 2 * d_col)
            if prev in alive:
                continue
            if prev not in floor or player_cell not in floor:
                continue
            alive.add(prev)
            queue.append(prev)

    return floor - alive


def _resolve_heuristic(method: str, heuristic: str | None) -> Callable | None:
    if method in UNINFORMED_METHODS:
        return None
    if heuristic is None or heuristic not in heuristics:
        raise ValueError("heuristic must be 'hungarian', 'simple' or 'weighted' for greedy/astar")
    return heuristics[heuristic]


def _node_priority(
    method: str,
    heuristic_fn: Callable | None,
    boxes,
    targets,
    g: int,
    heuristic_weight: float,
) -> float:
    if method == "bfs" or heuristic_fn is None:
        return float(g)
    h = heuristic_fn(boxes, targets)
    if method == "greedy":
        return float(h)
    return heuristic_weight * h + (1 - heuristic_weight) * g


def solve_sokoban(
    grid,
    method: str = "astar",
    heuristic: str | None = "hungarian",
    verbose: bool = False,
    timeout_seconds: float | None = None,
    max_expanded: int | None = None,
    measure_memory: bool = False,
    dead_square_pruning: bool = True,
) -> SearchResult:
    if method not in VALID_METHODS:
        raise ValueError("method must be 'astar', 'greedy', 'bfs' or 'dfs'")

    heuristic_fn = _resolve_heuristic(method, heuristic)
    reported_heuristic = None if method in UNINFORMED_METHODS else heuristic
    start_time = time.perf_counter()
    if measure_memory:
        tracemalloc.start()

    walls, targets, boxes, player = parse_board(grid)
    height = len(grid)
    width = max((len(row) for row in grid), default=0)
    floor = playable_floor(walls, player, height, width)
    dead_squares = compute_dead_squares(walls, targets, floor) if dead_square_pruning else set()
    start_state = (player, boxes)
    
    visited = {start_state}
    expanded_nodes = 0
    frontier_max = 0
    timed_out = False

    directions = DIRECTIONS
    counter = 0
    g = 0
    heuristic_weight = HEURISTIC_WEIGHTS.get(method, 0.0)

    if verbose:
        print(f"[{method.upper()}] Inicio: jugador={player}, cajas={sorted(boxes)}")

    if method == "dfs":
        frontier: deque | list = deque([(start_state, [])])
    else:
        f = _node_priority(method, heuristic_fn, boxes, targets, g, heuristic_weight)
        frontier = [(f, g, counter, start_state, [])]

    def finish(
        success: bool,
        path: list[tuple[int, int]] | None,
        frontier_len: int,
    ) -> SearchResult:
        peak_memory_kb = None
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory_kb = peak / 1024
        return SearchResult(
            success=success,
            cost=len(path) if path is not None else None,
            path=path,
            expanded_nodes=expanded_nodes,
            frontier_nodes_final=frontier_len,
            frontier_nodes_max=max(frontier_max, frontier_len),
            elapsed_time=time.perf_counter() - start_time,
            algorithm=method,
            heuristic=reported_heuristic,
            timeout=timed_out,
            peak_memory_kb=peak_memory_kb,
        )

    while frontier:
        frontier_max = max(frontier_max, len(frontier))

        if timeout_seconds is not None and (time.perf_counter() - start_time) >= timeout_seconds:
            timed_out = True
            break

        if method == "dfs":
            (player_position, boxes_positions), path = frontier.pop()
            g = len(path)
            priority = None
        else:
            f, g, _, (player_position, boxes_positions), path = heapq.heappop(frontier)
            priority = f

        if verbose:
            print(
                f"[{method.upper()}] Estado {expanded_nodes + 1}: "
                f"jugador={player_position}, cajas={sorted(boxes_positions)}, "
                f"movimientos={g}, prioridad={priority}, frontera={len(frontier)}"
            )

        if boxes_positions == targets:
            if verbose:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{method.upper()}] Solución encontrada: "
                    f"{len(path)} movimientos, {expanded_nodes} estados explorados, "
                    f"tiempo: {elapsed:.6f} segundos"
                )
            return finish(True, path, len(frontier))

        if max_expanded is not None and expanded_nodes >= max_expanded:
            timed_out = True
            break

        expanded_nodes += 1

        player_row, player_column = player_position

        for d_row, d_column in directions:
            new_player_row, new_player_column = player_row + d_row, player_column + d_column  
            
            if (new_player_row, new_player_column) in walls:
                continue

            new_boxes_positions = boxes_positions

            if (new_player_row, new_player_column) in boxes_positions:
                new_box_row, new_box_column = player_row + 2 * d_row, player_column + 2 * d_column 
                
                if (
                    (new_box_row, new_box_column) in walls
                    or (new_box_row, new_box_column) in boxes_positions
                    or (new_box_row, new_box_column) in dead_squares
                    or is_blocked((new_box_row, new_box_column), walls, targets)
                ):
                    continue
                
                new_boxes_positions = (boxes_positions - {(new_player_row, new_player_column)}) | {(new_box_row, new_box_column)}

            next_state = ((new_player_row, new_player_column), new_boxes_positions)

            if next_state not in visited:
                visited.add(next_state)
                dx, dy = d_column, d_row
                new_path = path + [(dx, dy)]

                if verbose:
                    action = "empuja una caja" if new_boxes_positions != boxes_positions else "se mueve"
                    print(f"[{method.upper()}]  -> {action} hacia {(new_player_row, new_player_column)}")

                new_g = g + 1
                counter += 1
                if method == "dfs":
                    frontier.append((next_state, new_path))
                else:
                    f = _node_priority(
                        method,
                        heuristic_fn,
                        new_boxes_positions,
                        targets,
                        new_g,
                        heuristic_weight,
                    )
                    heapq.heappush(frontier, (f, new_g, counter, next_state, new_path))

    if verbose:
        elapsed = time.perf_counter() - start_time
        status = "Timeout" if timed_out else "Sin solución"
        print(
            f"[{method.upper()}] {status}: {expanded_nodes} estados explorados, "
            f"tiempo: {elapsed:.6f} segundos"
        )

    return finish(False, None, len(frontier))
