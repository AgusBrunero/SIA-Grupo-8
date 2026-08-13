"""
sokoban_game.py - Jeu Sokoban complet en Python avec Tkinter.
Contrôles : Touches fléchées (Haut, Bas, Gauche, Droite), 'R' pour réinitialiser, 'U' pour annuler.
"""

import tkinter as tk
from tkinter import messagebox

# Cartes des niveaux (Légende : # Mur, . Cible, @ Joueur, $ Caisse, * Caisse sur cible, + Joueur sur cible, ' ' Vide)
LEVELS = [
    [
        "  #####",
        "  #   #",
        "  #$  #",
        "###  $##",
        "#  $ $ #",
        "### #. #",
        "  # #. #",
        "  # .@.#",
        "  ######"
    ],
    [
        "####",
        "# .#",
        "#  ###",
        "#*@  #",
        "#  $ #",
        "#  ###",
        "####"
    ],
    [
        "######",
        "#  . #",
        "# # $#",
        "# .@$#",
        "#  $ #",
        "######"
    ]
]

class SokobanGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Sokoban - Jeu en Python")
        
        self.current_level_idx = 0
        self.history = []
        self.moves_count = 0
        self.pushes_count = 0
        
        # Interface graphique
        self.info_label = tk.Label(
            root, 
            text="", 
            font=("Helvetica", 12, "bold"), 
            bg="#2c3e50", 
            fg="#ecf0f1", 
            padding=10
        )
        self.info_label.pack(fill=tk.X)
        
        self.canvas = tk.Canvas(root, bg="#34495e")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.controls_label = tk.Label(
            root, 
            text="Flèches: Déplacer | R: Recommencer | U: Annuler | N: Niveau Suivant", 
            font=("Helvetica", 10), 
            bg="#2c3e50", 
            fg="#bdc3c7", 
            padding=5
        )
        self.controls_label.pack(fill=tk.X)
        
        # Raccourcis clavier
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
        
        self.load_level(self.current_level_idx)

    def load_level(self, idx):
        self.current_level_idx = idx % len(LEVELS)
        raw_grid = LEVELS[self.current_level_idx]
        
        self.height = len(raw_grid)
        self.width = max(len(row) for row in raw_grid)
        
        # Grid interne sous forme de liste 2D mutable
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
            '#': '#7f8c8d',  # Mur (Gris)
            ' ': '#34495e',  # Sol (Bleu nuit)
            '.': '#e74c3c',  # Cible (Rouge)
            '$': '#f39c12',  # Caisse (Orange)
            '*': '#2ecc71',  # Caisse sur cible (Vert)
            '@': '#3498db',  # Joueur (Bleu)
            '+': '#9b59b6'   # Joueur sur cible (Violet)
        }
        
        for r in range(self.height):
            for c in range(self.width):
                char = self.grid[r][c]
                x1, y1 = c * cell_size, r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                
                # Arrière-plan
                bg_color = colors['#'] if char == '#' else colors[' ']
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=bg_color, outline="#2c3e50")
                
                # Cible
                if char in ('.', '*', '+'):
                    self.canvas.create_oval(x1 + 18, y1 + 18, x2 - 18, y2 - 18, fill="#e74c3c", outline="")
                
                # Caisse
                if char == '$':
                    self.canvas.create_rectangle(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#f39c12", outline="#d35400", width=2)
                    self.canvas.create_line(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#d35400")
                    self.canvas.create_line(x1 + 6, y2 - 6, x2 - 6, y1 + 6, fill="#d35400")
                elif char == '*':
                    self.canvas.create_rectangle(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#2ecc71", outline="#27ae60", width=2)
                    self.canvas.create_line(x1 + 6, y1 + 6, x2 - 6, y2 - 6, fill="#27ae60")
                    self.canvas.create_line(x1 + 6, y2 - 6, x2 - 6, y1 + 6, fill="#27ae60")
                    
                # Joueur
                if char in ('@', '+'):
                    self.canvas.create_oval(x1 + 8, y1 + 8, x2 - 8, y2 - 8, fill="#3498db", outline="#2980b9", width=2)

        self.info_label.config(
            text=f"Niveau {self.current_level_idx + 1}/{len(LEVELS)} | Déplacements: {self.moves_count} | Poussées: {self.pushes_count}"
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
            return  # Obstacle mur
            
        pushed_box = False
        
        # Si la case destination contient une caisse
        if target_cell in ('$', '*'):
            if not (0 <= nnr < self.height and 0 <= nnc < self.width):
                return
            beyond_cell = self.grid[nnr][nnc]
            if beyond_cell in ('#', '$', '*'):
                return  # Caisse bloquée
                
            # Déplacer la caisse
            pushed_box = True
            self.save_state()
            
            # Nouvelle position de la caisse
            self.grid[nnr][nnc] = '*' if beyond_cell == '.' else '$'
            # Ancienne position de la caisse devient le joueur
            self.grid[nr][nc] = '+' if target_cell == '*' else '@'
        else:
            self.save_state()
            self.grid[nr][nc] = '+' if target_cell == '.' else '@'
            
        # Libérer la case initiale du joueur
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
        # La victoire est atteinte s'il ne reste aucune caisse hors cible ('$')
        for row in self.grid:
            if '$' in row:
                return
        messagebox.showinfo("Victoire !", f"Bravo ! Niveau {self.current_level_idx + 1} réussi en {self.moves_count} coups !")
        self.next_level()

if __name__ == "__main__":
    root = tk.Tk()
    game = SokobanGame(root)
    root.mainloop()
