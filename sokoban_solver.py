"""
sokoban_solver.py - Solveur de Sokoban automatique via l'algorithme BFS (Breadth-First Search).
Recherche la séquence de mouvements minimale pour résoudre un niveau donné.
"""

from collections import deque
import time

def parse_level(level_str):
    """
    Convertit une chaîne de caractères représeantant un niveau en état structuré :
    - walls : ensemble des coordonnées (r, c) des murs
    - goals : ensemble des coordonnées (r, c) des cibles
    - boxes : tuple ordonné des coordonnées (r, c) des caisses
    - player : coordonnées (r, c) du joueur
    """
    lines = level_str.strip().split('\n')
    walls = set()
    goals = set()
    boxes = []
    player = None

    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == '#':
                walls.add((r, c))
            elif char == '.':
                goals.add((r, c))
            elif char == '$':
                boxes.append((r, c))
            elif char == '*':
                goals.add((r, c))
                boxes.append((r, c))
            elif char == '@':
                player = (r, c)
            elif char == '+':
                goals.add((r, c))
                player = (r, c)

    return frozenset(walls), frozenset(goals), tuple(sorted(boxes)), player

def solve_sokoban(level_str):
    """
    Résout le niveau Sokoban avec BFS.
    Retourne la liste des directions ('U', 'D', 'L', 'R') et le nombre d'états explorés.
    """
    walls, goals, initial_boxes, initial_player = parse_level(level_str)
    
    # État représenté par (position_joueur, tuple_caisses)
    start_state = (initial_player, initial_boxes)
    
    # Files BFS : (état_actuel, chemin_de_mouvements)
    queue = deque([(start_state, "")])
    visited = {start_state}
    
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }
    
    nodes_explored = 0
    start_time = time.time()
    
    while queue:
        (player_pos, boxes), path = queue.popleft()
        nodes_explored += 1
        
        # Condition de victoire : toutes les caisses sont sur des cibles
        if set(boxes) == set(goals):
            elapsed = time.time() - start_time
            return path, nodes_explored, elapsed
            
        pr, pc = player_pos
        boxes_set = set(boxes)
        
        for move_name, (dr, dc) in directions.items():
            nr, nc = pr + dr, pc + dc
            
            # Ne peut pas traverser un mur
            if (nr, nc) in walls:
                continue
                
            # Si le joueur avance vers une caisse
            if (nr, nc) in boxes_set:
                nnr, nnc = nr + dr, nc + dc
                # La case derrière la caisse doit être libre (pas de mur ni de caisse)
                if (nnr, nnc) in walls or (nnr, nnc) in boxes_set:
                    continue
                    
                # Déplacement de la caisse
                new_boxes = list(boxes)
                new_boxes.remove((nr, nc))
                new_boxes.append((nnr, nnc))
                new_boxes = tuple(sorted(new_boxes))
                new_player = (nr, nc)
            else:
                new_boxes = boxes
                new_player = (nr, nc)
                
            next_state = (new_player, new_boxes)
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + move_name))
                
    return None, nodes_explored, time.time() - start_time

if __name__ == "__main__":
    # Exemple de niveau Sokoban simple
    sample_level = """
######
#  . #
# # $#
# .@$#
#  $ #
######
"""
    print("--- Résolution du niveau Sokoban ---")
    print(sample_level)
    
    solution, nodes, duration = solve_sokoban(sample_level)
    
    if solution:
        print(f"Solution trouvée en {duration:.4f} secondes !")
        print(f"Nombre d'états explorés : {nodes}")
        print(f"Nombre de mouvements : {len(solution)}")
        print(f"Séquence : {solution}")
    else:
        print("Aucune solution trouvée pour ce niveau.")
