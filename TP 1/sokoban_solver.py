import heapq
import time
from scipy.optimize import linear_sum_assignment
from collections import deque


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


def heuristic(boxes, targets):
    """
    Heuristic for A*: Hungarian with Manhattan distance
    Considers the shortest distance for every box to its own objective
    """
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

def is_blocked(box, walls, targets):
    if box in targets:
        return False
    row, column = box
    return ((row-1,column) in walls or (row+1,column) in walls) and ((row,column-1) in walls or (row,column+1) in walls)

def solve_sokoban(grid, method="astar", verbose=False):
    start_time = time.perf_counter()
    walls, targets, boxes, player = parse_board(grid)
    start_state = (player, boxes)
    
    visited = {start_state}
    explored = 0

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    counter = 0
    g = 0

    if verbose:
        print(f"[{method.upper()}] Inicio: jugador={player}, cajas={sorted(boxes)}")

    if method=="dfs":
        frontier = deque([(start_state,[])])
    else:
        weights = {
            "astar": 0.5,
            "greedy": 1,
            "bfs": 0,
        }
        if method not in weights:
            raise ValueError("method must be 'astar', 'greedy', 'bfs' or 'dfs'")
        heuristic_weight = weights[method]
        f = heuristic_weight * heuristic(boxes, targets)
        frontier = [(f, g, counter, start_state, [])]

    while frontier:
        if method=="dfs":
            (player_position, boxes_positions), path = frontier.pop()
            g=len(path)
        else:
            f, g, _, (player_position, boxes_positions), path = heapq.heappop(frontier)

        explored += 1

        if verbose:
            print(
                f"[{method.upper()}] Estado {explored}: "
                f"jugador={player_position}, cajas={sorted(boxes_positions)}, "
                f"movimientos={g}, prioridad={f}, frontera={len(frontier)}"
            )

        if boxes_positions == targets:
            if verbose:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{method.upper()}] Solución encontrada: "
                    f"{len(path)} movimientos, {explored} estados explorados, "
                    f"tiempo: {elapsed:.6f} segundos"
                )
            return path

        player_row, player_column = player_position

        for d_row, d_column in directions:
            new_player_row, new_player_column = player_row + d_row, player_column + d_column  
            
            if (new_player_row, new_player_column) in walls:
                continue

            new_boxes_positions = boxes_positions

            if (new_player_row, new_player_column) in boxes_positions:
                new_box_row, new_box_column = player_row + 2 * d_row, player_column + 2 * d_column 
                
                if (new_box_row, new_box_column) in walls or (new_box_row, new_box_column) in boxes_positions or is_blocked((new_box_row,new_box_column),walls,targets):
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
                if method=="dfs":
                    frontier.append((next_state,new_path))
                else:
                    f = heuristic_weight * heuristic(new_boxes_positions, targets)+ (1 - heuristic_weight) * new_g
                    heapq.heappush(frontier, (f, new_g, counter, next_state, new_path))

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f"[{method.upper()}] Sin solución: {explored} estados explorados, "
            f"tiempo: {elapsed:.6f} segundos"
        )

    return None  