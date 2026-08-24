import tkinter as tk
from tkinter import messagebox

import sokoban_solver

# (# = Wall, . = target, @ = Player, $ = Box, * = Box on target, + = Player on target, ' ' = empty)
LEVELS = [
    [
        "######",
        "#  . #",
        "# #$ #",
        "# @  #",
        "######"
    ],
    [
        "#####",
        "#   #",
        "# $ #",
        "# $ #",
        "#.@.#",
        "#####"
    ],
    [
"   #####   ",
"####   #   ",
"#  #$  ####",
"# $$      #",
"#@  #$ $# #",
"### #   # #",
" #  ##### #",
" #  ..... #",
" ##########"
    ]
]

class SokobanGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Sokoban - Python Game")
        
        self.current_level_idx = 0
        self.history = []
        self.moves_count = 0
        self.pushes_count = 0
        
        self.info_label = tk.Label(
            root, 
            text="", 
            font=("Helvetica", 12, "bold"), 
            bg="#2c3e50", 
            fg="#ecf0f1", 
            pady=10
        )
        self.info_label.pack(fill=tk.X)
        
        self.canvas = tk.Canvas(root, bg="#34495e")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.controls_label = tk.Label(
            root, 
            text="Arrows: Move | R: Reset | U: Undo | N: Next | A/B/G/D: A*/BFS/Greedy/DFS (hungarian)\nShift+tecla: manhattan simple | w: A* weighted (no admisible) | W: Greedy (no admisible)", 
            font=("Helvetica", 10), 
            bg="#2c3e50", 
            fg="#bdc3c7", 
            pady=5
        )
        self.controls_label.pack(fill=tk.X)
        
        self.root.bind("<Up>", lambda e: self.move(0, -1))
        self.root.bind("<Down>", lambda e: self.move(0, 1))
        self.root.bind("<Left>", lambda e: self.move(-1, 0))
        self.root.bind("<Right>", lambda e: self.move(1, 0))
        self.root.bind("r", lambda e: self.reset_level())
        self.root.bind("R", lambda e: self.reset_level())
        self.root.bind("u", lambda e: self.undo())
        self.root.bind("U", lambda e: self.undo())
        self.root.bind("n", lambda e: self.next_level())
        self.root.bind("N", lambda e: self.next_level())

        self.root.bind("a", lambda e: self.auto_solve(method="astar"))
        self.root.bind("b", lambda e: self.auto_solve(method="bfs"))
        self.root.bind("g", lambda e: self.auto_solve(method="greedy"))
        self.root.bind("d", lambda e: self.auto_solve(method="dfs"))
        self.root.bind("A", lambda e: self.auto_solve(method="astar", heuristic="simple"))
        self.root.bind("B", lambda e: self.auto_solve(method="bfs", heuristic="simple"))
        self.root.bind("G", lambda e: self.auto_solve(method="greedy", heuristic="simple"))
        self.root.bind("D", lambda e: self.auto_solve(method="dfs", heuristic="simple"))
        self.root.bind("w", lambda e: self.auto_solve(method="astar", heuristic="weighted"))
        self.root.bind("W", lambda e: self.auto_solve(method="greedy", heuristic="weighted"))
        
        self.load_level(self.current_level_idx)

    def load_level(self, idx):
        self.current_level_idx = idx % len(LEVELS)
        raw_grid = LEVELS[self.current_level_idx]
        
        self.height = len(raw_grid)
        self.width = max(len(row) for row in raw_grid)
        
        self.grid = []
        for row in raw_grid:
            padded_row = list(row.ljust(self.width, ' '))
            self.grid.append(padded_row)
            
        self.history = []
        self.moves_count = 0
        self.pushes_count = 0
        self.draw_board()

    def get_player_pos(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] in ('@', '+'):
                    return r, c
        return None

    def draw_board(self):
        self.canvas.delete("all")
        cell_size = 50
        
        self.canvas.config(width=self.width * cell_size, height=self.height * cell_size)
        
        colors = {
            '#': '#7f8c8d',
            ' ': '#34495e',
            '.': '#e74c3c',
            '$': '#f39c12',  
            '*': '#2ecc71',  
            '@': '#3498db',  
            '+': '#9b59b6'   
        }
        
        for r in range(self.height):
            for c in range(self.width):
                char = self.grid[r][c]
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                bg_color = colors['#'] if char == '#' else colors[' ']
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg_color, outline="#2c3e50")
                
                if char in ('.', '*', '+'):
                    self.canvas.create_oval(x1 + 18, y1 + 18, x2 - 18, y2 - 18, fill="#e74c3c", outline="")
                
                if char == '$':
                    self.canvas.create_rectangle(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#f39c12", outline="#d35400", width=2)
                    self.canvas.create_line(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#d35400")
                    self.canvas.create_line(x1 + 6, y2 - 6, x2 - 6, y1 + 6, fill="#d35400")
                elif char == '*':
                    self.canvas.create_rectangle(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#2ecc71", outline="#27ae60", width=2)
                    self.canvas.create_line(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#27ae60")
                    self.canvas.create_line(x1 + 6, y2 - 6, x2 - 6, y1 + 6, fill="#27ae60")
                    
                if char in ('@', '+'):
                    self.canvas.create_oval(x1 + 8, y1 + 8, x2 - 8, y2 - 8, fill="#3498db", outline="#2980b9", width=2)

        self.info_label.config(
            text=f"Level {self.current_level_idx + 1}/{len(LEVELS)} | Moves: {self.moves_count} | Push: {self.pushes_count}"
        )

    def move(self, dx, dy):
        p_pos = self.get_player_pos()
        if not p_pos:
            return
        
        pr, pc = p_pos
        nr, nc = pr + dy, pc + dx
        nnr, nnc = pr + 2 * dy, pc + 2 * dx
        
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return
            
        target_cell = self.grid[nr][nc]
        
        if target_cell == '#':
            return  
            
        pushed_box = False
        
        if target_cell in ('$', '*'):
            if not (0 <= nnr < self.height and 0 <= nnc < self.width):
                return
            beyond_cell = self.grid[nnr][nnc]
            if beyond_cell in ('#', '$', '*'):
                return 
                
            pushed_box = True
            self.save_state()
            
            self.grid[nnr][nnc] = '*' if beyond_cell == '.' else '$'
            self.grid[nr][nc] = '+' if target_cell == '*' else '@'
        else:
            self.save_state()
            self.grid[nr][nc] = '+' if target_cell == '.' else '@'
            
        current_player_cell = self.grid[pr][pc]
        self.grid[pr][pc] = '.' if current_player_cell == '+' else ' '
        
        self.moves_count += 1
        if pushed_box:
            self.pushes_count += 1
            
        self.draw_board()
        self.check_win()

    def save_state(self):
        state_copy = [row[:] for row in self.grid]
        self.history.append((state_copy, self.moves_count, self.pushes_count))

    def undo(self):
        if not self.history:
            return
        state, moves, pushes = self.history.pop()
        self.grid = state
        self.moves_count = moves
        self.pushes_count = pushes
        self.draw_board()

    def reset_level(self):
        self.load_level(self.current_level_idx)

    def next_level(self):
        self.load_level(self.current_level_idx + 1)

    def check_win(self):
        for row in self.grid:
            if '$' in row:
                return
        messagebox.showinfo("Victory !", f"Bravo ! Level {self.current_level_idx + 1} done in {self.moves_count} moves and {self.pushes_count} pushes !")
        self.next_level()

    def auto_solve(self, method="astar", heuristic="hungarian"):
        if getattr(self, "is_animating", False):
            return

        result = sokoban_solver.solve_sokoban(self.grid, method=method, heuristic=heuristic)

        if not result.success or result.path is None:
            detail = " (timeout)" if result.timeout else ""
            messagebox.showwarning(
                "Solve", f"No solution found for this level !{detail}"
            )
            return

        heuristic_label = result.heuristic or "-"
        memory = (
            f"{result.peak_memory_kb:.1f} KB"
            if result.peak_memory_kb is not None
            else "-"
        )
        messagebox.showinfo(
            "Solve",
            "\n".join(
                [
                    f"Método: {result.algorithm} ({heuristic_label})",
                    f"Costo: {result.cost}",
                    f"Nodos expandidos: {result.expanded_nodes}",
                    f"Frontera final: {result.frontier_nodes_final}",
                    f"Frontera máxima: {result.frontier_nodes_max}",
                    f"Tiempo: {result.elapsed_time:.4f} s",
                    f"Memoria pico: {memory}",
                ]
            ),
        )

        self.is_animating = True
        self.animate_solution(list(result.path))

    def animate_solution(self, path, delay=150):
        if not path:
            self.is_animating = False
            return

        dx, dy = path.pop(0)

        self.move(dx, dy)

        self.root.after(delay, lambda: self.animate_solution(path, delay))

if __name__ == "__main__":
    root = tk.Tk()
    game = SokobanGame(root)
    root.mainloop()