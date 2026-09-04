import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

# ==============================================================================
# 1. ÉVALUATION ET CARTE DES ZONES ACTIVES
# ==============================================================================
class LogoEvaluator:
    def __init__(self, target_path, target_height=48):
        # Fond blanc forcé
        raw_img = Image.open(target_path).convert('RGBA')
        white_bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, raw_img).convert('RGB')

        w_orig, h_orig = composite.size
        aspect = w_orig / h_orig
        self.h = target_height
        self.w = int(target_height * aspect)

        self.target_img = composite.resize((self.w, self.h), Image.Resampling.BILINEAR)
        self.target_arr = np.array(self.target_img, dtype=np.float32)

        # Détection des pixels qui ne sont PAS blancs (zones d'intérêt : lettres)
        # On calcule la distance au blanc pur (255, 255, 255)
        diff_from_white = np.mean(np.abs(self.target_arr - 255.0), axis=2)
        non_white_coords = np.argwhere(diff_from_white > 20) # Indices (y, x)
        
        if len(non_white_coords) > 0:
            self.active_y = non_white_coords[:, 0] / (self.h - 1)
            self.active_x = non_white_coords[:, 1] / (self.w - 1)
        else:
            self.active_x = np.random.rand(100)
            self.active_y = np.random.rand(100)

    def render(self, individual):
        canvas = Image.new('RGB', (self.w, self.h), (255, 255, 255))
        overlay = Image.new('RGBA', (self.w, self.h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, 'RGBA')

        for g in individual.genes:
            pts = [
                (g[0] * self.w, g[1] * self.h),
                (g[2] * self.w, g[3] * self.h),
                (g[4] * self.w, g[5] * self.h)
            ]
            color = (int(g[6]), int(g[7]), int(g[8]), int(g[9]))
            draw.polygon(pts, fill=color)

        canvas.paste(overlay, (0, 0), overlay)
        return canvas

    def evaluate(self, individual):
        if individual.fitness is not None:
            return individual.fitness
        rendered_arr = np.array(self.render(individual), dtype=np.float32)
        # MSE
        mse = np.mean((self.target_arr - rendered_arr) ** 2)
        individual.fitness = 1.0 / (1.0 + mse)
        return individual.fitness


# ==============================================================================
# 2. GÉNOTYPE AVEC TAILLES MIXTES ET ANCRAGE INTELLIGENT
# ==============================================================================
class Individual:
    def __init__(self, num_triangles, evaluator=None, genes=None):
        self.num_triangles = num_triangles
        if genes is not None:
            self.genes = np.copy(genes)
        else:
            self.genes = np.zeros((num_triangles, 10), dtype=np.float32)
            for i in range(num_triangles):
                # 80% des triangles naissent directement sur les lettres bleues
                if evaluator is not None and np.random.rand() < 0.80:
                    idx = np.random.randint(0, len(evaluator.active_x))
                    cx, cy = evaluator.active_x[idx], evaluator.active_y[idx]
                else:
                    cx, cy = np.random.rand(), np.random.rand()

                # Mixte de tailles : 35% de grands triangles pour faire les troncs de lettres
                if i < int(num_triangles * 0.35):
                    rx = np.random.uniform(0.08, 0.22)
                    ry = np.random.uniform(0.15, 0.40) # Grands verticalement (pour les barres)
                else:
                    rx = np.random.uniform(0.03, 0.09)
                    ry = np.random.uniform(0.04, 0.12)

                angles = np.random.uniform(0, 2 * np.pi, 3)
                self.genes[i, 0] = np.clip(cx + rx * np.cos(angles[0]), 0.0, 1.0)
                self.genes[i, 1] = np.clip(cy + ry * np.sin(angles[0]), 0.0, 1.0)
                self.genes[i, 2] = np.clip(cx + rx * np.cos(angles[1]), 0.0, 1.0)
                self.genes[i, 3] = np.clip(cy + ry * np.sin(angles[1]), 0.0, 1.0)
                self.genes[i, 4] = np.clip(cx + rx * np.cos(angles[2]), 0.0, 1.0)
                self.genes[i, 5] = np.clip(cy + ry * np.sin(angles[2]), 0.0, 1.0)

                # Couleur échantillonnée
                if evaluator is not None:
                    px = int(np.clip(cx * (evaluator.w - 1), 0, evaluator.w - 1))
                    py = int(np.clip(cy * (evaluator.h - 1), 0, evaluator.h - 1))
                    self.genes[i, 6:9] = evaluator.target_arr[py, px]
                else:
                    self.genes[i, 6:9] = [0, 79, 139] # Bleu ITBA par défaut

                # Forte opacité pour créer des blocs solides
                self.genes[i, 9] = np.random.uniform(180, 250)

        self.fitness = None

    def clone(self):
        ind = Individual(self.num_triangles, genes=self.genes)
        ind.fitness = self.fitness
        return ind


# ==============================================================================
# 3. CROISEMENT ET MUTATIONS COMPLÈTES (Sommets + Translation + Échelle)
# ==============================================================================
def crossover_two_point(p1, p2):
    n = p1.num_triangles
    pt1, pt2 = sorted(np.random.choice(range(1, n), size=2, replace=False))
    c1 = np.vstack((p1.genes[:pt1], p2.genes[pt1:pt2], p1.genes[pt2:]))
    c2 = np.vstack((p2.genes[:pt1], p1.genes[pt1:pt2], p2.genes[pt2:]))
    return Individual(n, genes=c1), Individual(n, genes=c2)

def mutate_robust(ind, gen, max_gen, pm=0.25):
    mutated = False
    progress = gen / max_gen
    sigma = max(0.005, 0.07 * (1.0 - progress))

    for i in range(ind.num_triangles):
        if np.random.rand() < pm:
            mode = np.random.rand()
            
            # Type 1 : Translation globale du triangle (permet de caler une barre entière)
            if mode < 0.35:
                dx = np.random.normal(0, sigma)
                dy = np.random.normal(0, sigma)
                ind.genes[i, 0:6:2] = np.clip(ind.genes[i, 0:6:2] + dx, 0.0, 1.0)
                ind.genes[i, 1:6:2] = np.clip(ind.genes[i, 1:6:2] + dy, 0.0, 1.0)
            
            # Type 2 : Déplacement individuel des sommets (ajustement fin)
            elif mode < 0.75:
                ind.genes[i, 0:6] += np.random.normal(0, sigma, 6)
                ind.genes[i, 0:6] = np.clip(ind.genes[i, 0:6], 0.0, 1.0)
            
            # Type 3 : Redimensionnement / scaling par rapport au centre du triangle
            else:
                scale = np.random.uniform(0.85, 1.15)
                cx = np.mean(ind.genes[i, 0:6:2])
                cy = np.mean(ind.genes[i, 1:6:2])
                ind.genes[i, 0:6:2] = np.clip(cx + (ind.genes[i, 0:6:2] - cx) * scale, 0.0, 1.0)
                ind.genes[i, 1:6:2] = np.clip(cy + (ind.genes[i, 1:6:2] - cy) * scale, 0.0, 1.0)

            # Ajustement couleur & alpha plus discret
            if np.random.rand() < 0.2:
                ind.genes[i, 6:9] += np.random.normal(0, sigma * 80, 3)
                ind.genes[i, 6:9] = np.clip(ind.genes[i, 6:9], 0.0, 255.0)
                ind.genes[i, 9] = np.clip(ind.genes[i, 9] + np.random.normal(0, 15), 120.0, 255.0)

            mutated = True

    # Permutation occasionnelle du Z-order (essentiel pour superposer le blanc de coupure)
    if np.random.rand() < 0.10:
        i1, i2 = np.random.choice(ind.num_triangles, 2, replace=False)
        ind.genes[[i1, i2]] = ind.genes[[i2, i1]]
        mutated = True

    if mutated:
        ind.fitness = None

def select_tournament(pop, k, m=3):
    """Tournoi déterministe (plus dynamique que Boltzmann pour ce type de logo)"""
    selected = []
    n = len(pop)
    for _ in range(k):
        candidates = np.random.choice(n, size=m, replace=False)
        best_idx = max(candidates, key=lambda idx: pop[idx].fitness)
        selected.append(pop[best_idx].clone())
    return selected

def survival_additive(parents, children, n):
    pool = parents + children
    pool.sort(key=lambda ind: ind.fitness, reverse=True)
    return pool[:n]


# ==============================================================================
# 4. INTERFACE GRAPHIQUE TKINTER
# ==============================================================================
class GeneticAppTk:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("SIA - TP2 : Reconstitution Vectorielle ITBA")
        self.root.configure(bg="#18191d")

        self.evaluator = LogoEvaluator(config["target_path"], target_height=config["render_h"])
        self.w = self.evaluator.w
        self.h = self.evaluator.h
        self.scale = 4
        self.disp_w = self.w * self.scale
        self.disp_h = self.h * self.scale

        # Création de la population
        self.pop = [Individual(config["num_triangles"], evaluator=self.evaluator) for _ in range(config["pop_size"])]
        for ind in self.pop:
            self.evaluator.evaluate(ind)

        self.gen = 1
        self.fitness_history = []
        self.gif_frames = []
        self.is_running = True

        self._build_ui()
        self.root.after(50, self.step_evolution)

    def _build_ui(self):
        img_frame = tk.Frame(self.root, bg="#18191d")
        img_frame.pack(padx=20, pady=15)

        # Cible
        f_left = tk.Frame(img_frame, bg="#18191d")
        f_left.grid(row=0, column=0, padx=12)
        tk.Label(f_left, text="CIBLE ORIGINALE", font=("Segoe UI", 10, "bold"), fg="#888888", bg="#18191d").pack()
        self.target_lbl = tk.Label(f_left, bg="#ffffff", bd=1, relief="solid")
        self.target_lbl.pack(pady=4)
        
        target_disp = self.evaluator.target_img.resize((self.disp_w, self.disp_h), Image.Resampling.NEAREST)
        self.tk_target = ImageTk.PhotoImage(target_disp)
        self.target_lbl.config(image=self.tk_target)

        # Approximation
        f_right = tk.Frame(img_frame, bg="#18191d")
        f_right.grid(row=0, column=1, padx=12)
        tk.Label(f_right, text="APPROXIMATION (AG)", font=("Segoe UI", 10, "bold"), fg="#00e676", bg="#18191d").pack()
        self.best_lbl = tk.Label(f_right, bg="#ffffff", bd=1, relief="solid")
        self.best_lbl.pack(pady=4)

        # Texte métriques
        self.metrics_lbl = tk.Label(
            self.root,
            text="Initialisation...",
            font=("Consolas", 11, "bold"),
            fg="#ffffff",
            bg="#18191d"
        )
        self.metrics_lbl.pack(pady=4)

        # Graphe vectoriel
        self.graph_w = (self.disp_w * 2) + 24
        self.graph_h = 85
        self.canvas_graph = tk.Canvas(self.root, width=self.graph_w, height=self.graph_h, bg="#22232a", highlightthickness=0)
        self.canvas_graph.pack(padx=20, pady=10)

    def step_evolution(self):
        if not self.is_running or self.gen > self.config["max_generations"]:
            self.finalize()
            return

        # 1. Sélection par tournoi
        mating_pool = select_tournament(self.pop, self.config["k_offspring"], m=3)

        # 2. Croisement 2-points
        children = []
        for i in range(0, self.config["k_offspring"] - 1, 2):
            if np.random.rand() < self.config["pc"]:
                c1, c2 = crossover_two_point(mating_pool[i], mating_pool[i + 1])
            else:
                c1, c2 = mating_pool[i].clone(), mating_pool[i + 1].clone()
            children.extend([c1, c2])

        # 3. Mutation robuste
        for c in children:
            mutate_robust(c, self.gen, self.config["max_generations"], pm=self.config["pm"])
            self.evaluator.evaluate(c)

        # 4. Survie additive
        self.pop = survival_additive(self.pop, children, self.config["pop_size"])

        best = self.pop[0]
        self.fitness_history.append(float(best.fitness))

        # Affichage
        if self.gen % self.config["ui_step"] == 0 or self.gen == 1:
            best_pil = self.evaluator.render(best)

            if self.gen % self.config["gif_step"] == 0:
                self.gif_frames.append(best_pil.copy())

            best_disp = best_pil.resize((self.disp_w, self.disp_h), Image.Resampling.NEAREST)
            self.tk_best = ImageTk.PhotoImage(best_disp)
            self.best_lbl.config(image=self.tk_best)

            mse = (1.0 / best.fitness) - 1.0
            self.metrics_lbl.config(
                text=f"Gen: {self.gen:4d}/{self.config['max_generations']} | "
                     f"Fitness: {best.fitness:.6f} | MSE: {mse:.1f} | Triangles: {self.config['num_triangles']}"
            )
            self._update_graph()

        self.gen += 1
        self.root.after(1, self.step_evolution)

    def _update_graph(self):
        self.canvas_graph.delete("all")
        if len(self.fitness_history) < 2:
            return

        min_f = min(self.fitness_history)
        max_f = max(self.fitness_history)
        rng = max(max_f - min_f, 1e-6)

        points = []
        n_pts = len(self.fitness_history)
        for idx, val in enumerate(self.fitness_history):
            x = int((idx / (n_pts - 1)) * (self.graph_w - 10)) + 5
            y = self.graph_h - int(((val - min_f) / rng) * (self.graph_h - 20)) - 10
            points.extend([x, y])

        self.canvas_graph.create_line(points, fill="#00e5ff", width=2, smooth=True)

    def finalize(self):
        self.metrics_lbl.config(text=self.metrics_lbl.cget("text") + " [TERMINÉ]")
        if self.gif_frames:
            print("Génération du GIF...")
            self.gif_frames[0].save(
                "evolution_itba_letters.gif",
                save_all=True,
                append_images=self.gif_frames[1:],
                duration=65,
                loop=0
            )
            print("Exporté avec succès : 'evolution_itba_letters.gif'.")


if __name__ == "__main__":
    cfg = {
        "target_path": "itba.png",
        "render_h": 44,
        "num_triangles": 65,
        "pop_size": 90,
        "k_offspring": 65,
        "pc": 0.65,
        "pm": 0.30,
        "max_generations": 2000,
        "ui_step": 3,
        "gif_step": 8
    }

    root = tk.Tk()
    app = GeneticAppTk(root, cfg)
    root.mainloop()