import heapq
from collections import deque
import time

def parse_board(grid):
    walls, targets, boxes = set(), set(), set()
    player = None
    
    for r, row in enumerate(grid):
        for c, char in enumerate(row):
            if char == '#':
                walls.add((r, c))
            elif char == '.':
                targets.add((r, c))
            elif char == '$':
                boxes.add((r, c))
            elif char == '@':
                player = (r, c)
            elif char == '*':  
                targets.add((r, c))
                boxes.add((r, c))
            elif char == '+': 
                targets.add((r, c))
                player = (r, c)
                
    return walls, targets, frozenset(boxes), player


def heuristic(boxes, targets):
    """
    Heuristic for A*: Sum of Manhattan distances
    from each box to its closest target.
    """
    return sum(
        min(abs(br - tr) + abs(bc - tc) for tr, tc in targets)
        for br, bc in boxes
    )


def solve_sokoban(grid, method="astar", verbose=True):# astar or bfs

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
            f, g, _, (p_pos, b_pos), path = heapq.heappop(frontier)
        else:
            (p_pos, b_pos), path = frontier.popleft()
            g = len(path)
            f = g

        explored += 1

        if verbose:
            print(
                f"[{method.upper()}] Estado {explored}: "
                f"jugador={p_pos}, cajas={sorted(b_pos)}, "
                f"movimientos={g}, prioridad={f}, frontera={len(frontier)}"
            )

        if b_pos == targets:
            if verbose:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{method.upper()}] Solución encontrada: "
                    f"{len(path)} movimientos, {explored} estados explorados, "
                    f"tiempo: {elapsed:.6f} segundos"
                )
            return path

        pr, pc = p_pos

        for dr, dc in directions:
            nr, nc = pr + dr, pc + dc  
            
            if (nr, nc) in walls:
                continue

            new_b_pos = b_pos

            if (nr, nc) in b_pos:
                nnr, nnc = pr + 2 * dr, pc + 2 * dc 
                
                if (nnr, nnc) in walls or (nnr, nnc) in b_pos:
                    continue
                
                new_b_pos = (b_pos - {(nr, nc)}) | {(nnr, nnc)}

            next_state = ((nr, nc), new_b_pos)

            if next_state not in visited:
                visited.add(next_state)
                dx, dy = dc, dr
                new_path = path + [(dx, dy)]

                if verbose:
                    action = "empuja una caja" if new_b_pos != b_pos else "se mueve"
                    print(f"[{method.upper()}]  -> {action} hacia {(nr, nc)}")

                if method == "astar":
                    counter += 1
                    f = (g + 1) + heuristic(new_b_pos, targets)
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