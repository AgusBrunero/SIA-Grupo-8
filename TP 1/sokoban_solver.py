import heapq
from collections import deque
import time
from scipy.optimize import linear_sum_assignment

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

def solve_sokoban(grid, method="astar", verbose=False):# astar or bfs

    start_time = time.perf_counter()
    walls, targets, boxes, player = parse_board(grid)
    start_state = (player, boxes)
    
    visited = {start_state}
    explored = 0

    if verbose:
        print(f"[{method.upper()}] Inicio: jugador={player}, cajas={sorted(boxes)}")
    

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    if method == "astar":
        counter = 0
        h = heuristic(boxes, targets)
        frontier = [(h, 0, counter, start_state, [])]
    else:  # BFS
        frontier = deque([(start_state, [])])

    while frontier:
        if method == "astar":
            f, g, _, (player_position, boxes_positions), path = heapq.heappop(frontier)
        else:
            (player_position, boxes_positions), path = frontier.popleft()
            g = len(path)
            f = g

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

                if method == "astar":
                    counter += 1
                    f = (g + 1) + heuristic(new_boxes_positions, targets)
                    heapq.heappush(frontier, (f, g + 1, counter, next_state, new_path))
                else:
                    frontier.append((next_state, new_path))

    if verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f"[{method.upper()}] Sin solución: {explored} estados explorados, "
            f"tiempo: {elapsed:.6f} segundos"
        )

    return None  