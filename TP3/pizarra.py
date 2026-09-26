"""Pizarra interactiva del TP3: dibujá un dígito y mirá qué hace la red por dentro.

Pestaña "Pizarra"
  - Dibujás con el mouse y los modelos del Ej. 2 y del Ej. 3 predicen en tiempo real.
    (El del Ej. 2 nunca vio un 8: dibujá uno y fijate qué contesta.)
  - "Lo que ve la red": el dibujo llevado al formato del dataset (20×20 centrado por
    centro de masa en 28×28) + ruido gaussiano opcional.
  - Atribución (saliency, gradiente × entrada): qué píxeles empujan hacia la clase predicha.
  - Capa oculta: cada cuadradito es una neurona; click para ver sus pesos como imagen 28×28.

Pestaña "Entrenamiento en vivo"
  - Un MLP chico aprende delante tuyo: los pesos de cada neurona pasan de ruido a trazos,
    y la matriz de confusión se va llenando (sin more_digits la fila del 8 queda vacía).

Pestaña "Fraude · TinyModel" (Ej. 1)
  - Sliders para armar una transacción y ver la probabilidad de TinyModel en vivo, con la
    contribución de cada feature (peso × valor estandarizado).
  - Umbral + relación de costos FN/FP: matriz de confusión out-of-fold y el umbral de mínimo costo.

Uso (desde TP3/):  python pizarra.py        (antes: python ej2.py final && python ej3.py final)
"""
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import numpy as np

import digitos
from perceptron.mlp import MLP
from perceptron import optimizadores

MODELS = Path(__file__).parent / 'modelos'
MODEL_FILES = {'Ej. 2 · sin 8': 'ej2_final.npz', 'Ej. 3 · ≥98%': 'ej3_final.npz'}

# Paleta
BG, PANEL, PANEL2 = '#101318', '#181d24', '#222933'
INK, MUTED, FAINT = '#eceae4', '#9aa1ab', '#4a525e'
ACCENT, WARN = '#4f9cf0', '#f07a4f'
MODEL_COLORS = ['#f0a04f', '#4fd1a0', '#b58cf0']
FONT = ('Helvetica', 12)
FONT_B = ('Helvetica', 12, 'bold')
FONT_T = ('Helvetica', 15, 'bold')
FONT_BIG = ('Helvetica', 44, 'bold')

DRAW = 336          # resolución interna del lienzo de dibujo
BRUSH = 13          # radio del pincel en píxeles del lienzo


# ---------- imágenes numpy → Tk ----------

def hex_to_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def ramp(values, c0, c1):
    """Interpola entre dos colores (values en [0, 1])."""
    v = np.clip(values, 0, 1)[..., None]
    return (hex_to_rgb(c0) * (1 - v) + hex_to_rgb(c1) * v).astype(np.uint8)


def diverging(values):
    """values en [−1, 1]: azul negativo, fondo neutro, naranja positivo."""
    v = np.clip(values, -1, 1)[..., None]
    base, pos, neg = hex_to_rgb(PANEL2), hex_to_rgb('#f5a15a'), hex_to_rgb('#5aa8f5')
    out = np.where(v >= 0, base * (1 - v) + pos * v, base * (1 + v) + neg * (-v))
    return out.astype(np.uint8)


def photo(rgb: np.ndarray, scale: int) -> tk.PhotoImage:
    big = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    h, w, _ = big.shape
    return tk.PhotoImage(data=f'P6 {w} {h} 255 '.encode() + big.tobytes(), format='PPM')


# ---------- preprocesamiento estilo MNIST ----------

def resize_area(img, nh, nw, k=6):
    """Reduce img a (nh, nw) promediando bloques (anti-aliasing sin librerías extra)."""
    h, w = img.shape
    ys = np.minimum(((np.arange(nh * k) + 0.5) * h / (nh * k)).astype(int), h - 1)
    xs = np.minimum(((np.arange(nw * k) + 0.5) * w / (nw * k)).astype(int), w - 1)
    return img[np.ix_(ys, xs)].reshape(nh, k, nw, k).mean(axis=(1, 3))


def to_dataset_format(canvas: np.ndarray) -> np.ndarray:
    """Recorta el trazo, lo escala a 20 px en el lado mayor y lo centra por centro de masa,
    igual que las imágenes del dataset (centro de masa ≈ (14, 14))."""
    ys, xs = np.nonzero(canvas > 0.05)
    if len(ys) == 0:
        return np.zeros(digitos.N_PIXELS, dtype=np.float32)
    crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = resize_area(crop, nh, nw)
    img = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    img[top:top + nh, left:left + nw] = small
    img /= max(img.max(), 1e-6)
    yy, xx = np.mgrid[0:28, 0:28]
    mass = img.sum()
    dy, dx = round(14 - (img * yy).sum() / mass), round(14 - (img * xx).sum() / mass)
    return digitos.shift(img.reshape(1, -1), np.array([dx]), np.array([dy]))[0]


def load_models() -> dict[str, MLP]:
    models = {}
    for label, file in MODEL_FILES.items():
        if (MODELS / file).exists():
            models[label], _ = MLP.load(MODELS / file)
    return models


# ---------- widgets ----------

class ProbBars(tk.Canvas):
    """Barras de probabilidad por dígito para un modelo."""

    def __init__(self, master, title, color, width=250, height=300):
        super().__init__(master, width=width, height=height, bg=PANEL, highlightthickness=0)
        self.title, self.color, self.w, self.h = title, color, width, height

    def show(self, probs, true_label=None):
        self.delete('all')
        self.create_text(14, 16, text=self.title, anchor='w', fill=self.color, font=FONT_B)
        if probs is None:
            self.create_text(self.w / 2, self.h / 2, text='modelo no entrenado', fill=MUTED, font=FONT)
            return
        pred = int(np.argmax(probs))
        self.create_text(self.w - 14, 40, text=str(pred), anchor='e', fill=INK, font=FONT_BIG)
        self.create_text(14, 44, text=f'{probs[pred]:.0%} seguro', anchor='w', fill=MUTED, font=FONT)
        top, row = 82, (self.h - 92) / 10
        bar_x0, bar_x1 = 34, self.w - 56
        for d in range(10):
            y = top + d * row
            fill = self.color if d == pred else FAINT
            if true_label is not None and d == true_label:
                self.create_rectangle(4, y, self.w - 4, y + row - 3, outline=MUTED)
            self.create_text(18, y + row / 2, text=str(d), fill=INK if d == pred else MUTED, font=FONT_B)
            self.create_rectangle(bar_x0, y + 4, bar_x1, y + row - 6, fill=PANEL2, width=0)
            self.create_rectangle(bar_x0, y + 4, bar_x0 + (bar_x1 - bar_x0) * probs[d], y + row - 6, fill=fill, width=0)
            self.create_text(self.w - 10, y + row / 2, text=f'{probs[d]:.2f}', anchor='e', fill=MUTED, font=('Helvetica', 10))


class ImagePanel(tk.Frame):
    def __init__(self, master, title, size):
        super().__init__(master, bg=PANEL)
        tk.Label(self, text=title, bg=PANEL, fg=MUTED, font=FONT).pack(anchor='w', padx=8, pady=(6, 2))
        self.label = tk.Label(self, bg=PANEL, width=size, height=size)
        self.label.pack(padx=8, pady=(0, 8))
        self.caption = tk.Label(self, text='', bg=PANEL, fg=MUTED, font=('Helvetica', 10), wraplength=size)
        self.caption.pack(padx=8, pady=(0, 6))
        self._img = None

    def set(self, rgb, scale, caption=''):
        self._img = photo(rgb, scale)
        self.label.configure(image=self._img, width=rgb.shape[1] * scale, height=rgb.shape[0] * scale)
        self.caption.configure(text=caption)


# ---------- pestaña 1: pizarra ----------

class Pizarra(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.canvas_data = np.zeros((DRAW, DRAW), dtype=np.float32)
        yy, xx = np.mgrid[-BRUSH:BRUSH + 1, -BRUSH:BRUSH + 1]
        d = np.sqrt(xx ** 2 + yy ** 2) / BRUSH
        self.brush = np.clip(1.4 - d * 1.4, 0, 1) ** 0.7   # borde suave
        self.last = None
        self.true_label = None
        self.selected_neuron = None
        self.pending = None
        self.noise_seed = 0

        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky='n', padx=(16, 8), pady=16)
        tk.Label(left, text='Dibujá un dígito', bg=BG, fg=INK, font=FONT_T).pack(anchor='w')
        self.draw_canvas = tk.Canvas(left, width=DRAW, height=DRAW, bg='#050607', highlightthickness=1,
                                     highlightbackground=FAINT, cursor='pencil')
        self.draw_canvas.pack(pady=8)
        self.draw_canvas.bind('<Button-1>', self.on_press)
        self.draw_canvas.bind('<B1-Motion>', self.on_drag)
        self.draw_canvas.bind('<ButtonRelease-1>', lambda e: setattr(self, 'last', None))
        self.draw_canvas.bind('<Button-3>', lambda e: self.clear())
        self.draw_canvas.bind('<Button-2>', lambda e: self.clear())

        buttons = tk.Frame(left, bg=BG)
        buttons.pack(fill='x')
        self.button(buttons, 'Borrar', self.clear).pack(side='left')
        self.button(buttons, 'Muestra del test', self.sample).pack(side='left', padx=6)
        self.button(buttons, 'Un 8 del test', lambda: self.sample(8)).pack(side='left')

        noise = tk.Frame(left, bg=BG)
        noise.pack(fill='x', pady=(12, 0))
        tk.Label(noise, text='Ruido gaussiano σ', bg=BG, fg=MUTED, font=FONT).pack(anchor='w')
        self.noise = tk.DoubleVar(value=0.0)
        tk.Scale(noise, from_=0, to=0.6, resolution=0.02, orient='horizontal', variable=self.noise,
                 command=lambda _: self.schedule(), bg=BG, fg=INK, troughcolor=PANEL2, highlightthickness=0,
                 length=DRAW, sliderrelief='flat').pack(anchor='w')
        self.button(noise, 'Otro ruido', self.reseed).pack(anchor='w')

        tk.Label(left, text='Click derecho: borrar', bg=BG, fg=FAINT, font=('Helvetica', 10)).pack(anchor='w', pady=(10, 0))

        mid = tk.Frame(self, bg=BG)
        mid.grid(row=0, column=1, sticky='n', padx=8, pady=16)
        self.seen = ImagePanel(mid, 'Lo que ve la red (28×28)', 168)
        self.seen.pack(fill='x')
        self.saliency = ImagePanel(mid, 'Atribución: gradiente × entrada', 168)
        self.saliency.pack(fill='x', pady=10)
        focus = tk.Frame(mid, bg=BG)
        focus.pack(fill='x')
        tk.Label(focus, text='Modelo en foco', bg=BG, fg=MUTED, font=FONT).pack(anchor='w')
        self.focus = tk.StringVar()
        self.focus_box = ttk.Combobox(focus, textvariable=self.focus, state='readonly', width=22)
        self.focus_box.pack(anchor='w', pady=4)
        self.focus_box.bind('<<ComboboxSelected>>', lambda e: (setattr(self, 'selected_neuron', None), self.schedule()))

        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=2, sticky='n', padx=(8, 16), pady=16)
        self.bars_frame = tk.Frame(right, bg=BG)
        self.bars_frame.pack(anchor='w')
        self.bars: dict[str, ProbBars] = {}

        hidden = tk.Frame(right, bg=PANEL)
        hidden.pack(fill='x', pady=(10, 0))
        tk.Label(hidden, text='Capa oculta del modelo en foco · click en una neurona', bg=PANEL, fg=MUTED,
                 font=FONT).pack(anchor='w', padx=8, pady=(6, 2))
        row = tk.Frame(hidden, bg=PANEL)
        row.pack(anchor='w', padx=8, pady=(0, 8))
        self.hidden_canvas = tk.Canvas(row, width=340, height=170, bg=PANEL, highlightthickness=0)
        self.hidden_canvas.pack(side='left')
        self.hidden_canvas.bind('<Button-1>', self.on_neuron_click)
        self.neuron_panel = ImagePanel(row, 'Pesos de la neurona', 140)
        self.neuron_panel.pack(side='left', padx=(8, 0))
        self.hidden_image = None
        self.hidden_layout = None

        self.refresh_models()

    def button(self, master, text, command):
        return tk.Button(master, text=text, command=command, bg=PANEL2, fg=INK, activebackground=FAINT,
                         activeforeground=INK, relief='flat', font=FONT, padx=10, pady=4, highlightthickness=0, bd=0)

    # ----- modelos disponibles -----

    def refresh_models(self):
        names = list(self.app.models)
        for w in self.bars_frame.winfo_children():
            w.destroy()
        self.bars = {}
        for i, name in enumerate(names or ['(sin modelos)']):
            bars = ProbBars(self.bars_frame, name, MODEL_COLORS[i % len(MODEL_COLORS)], width=250 if len(names) < 3 else 196)
            bars.grid(row=0, column=i, padx=(0, 8))
            self.bars[name] = bars
        self.focus_box['values'] = names
        if names and self.focus.get() not in names:
            self.focus.set(names[-1])
        self.schedule()

    # ----- dibujo -----

    def stamp(self, x, y):
        r = BRUSH
        x0, y0 = int(x) - r, int(y) - r
        xs, ys = slice(max(x0, 0), min(x0 + 2 * r + 1, DRAW)), slice(max(y0, 0), min(y0 + 2 * r + 1, DRAW))
        bx = slice(xs.start - x0, xs.stop - x0)
        by = slice(ys.start - y0, ys.stop - y0)
        if xs.start < xs.stop and ys.start < ys.stop:
            np.maximum(self.canvas_data[ys, xs], self.brush[by, bx], out=self.canvas_data[ys, xs])

    def on_press(self, e):
        if self.true_label is not None:
            self.clear()
        self.last = (e.x, e.y)
        self.stamp(e.x, e.y)
        self.draw_canvas.create_oval(e.x - BRUSH, e.y - BRUSH, e.x + BRUSH, e.y + BRUSH, fill=INK, outline='')
        self.schedule()

    def on_drag(self, e):
        if self.last is None:
            return self.on_press(e)
        x0, y0 = self.last
        steps = max(1, int(np.hypot(e.x - x0, e.y - y0) / (BRUSH / 3)))
        for t in np.linspace(0, 1, steps + 1)[1:]:
            self.stamp(x0 + (e.x - x0) * t, y0 + (e.y - y0) * t)
        self.draw_canvas.create_line(x0, y0, e.x, e.y, fill=INK, width=2 * BRUSH, capstyle='round', smooth=True)
        self.last = (e.x, e.y)
        self.schedule()

    def clear(self):
        self.canvas_data[:] = 0
        self.true_label = None
        self.draw_canvas.delete('all')
        self.draw_canvas.configure(bg='#050607')
        self.schedule()

    def sample(self, digit=None):
        """Carga una imagen real de digits_test (con su etiqueta)."""
        X, y = self.app.test_set()
        rng = np.random.default_rng()
        idx = rng.choice(np.flatnonzero(y == digit)) if digit is not None else rng.integers(len(y))
        self.clear()
        self.true_label = int(y[idx])
        self.test_image = X[idx]
        self.draw_canvas.delete('all')
        self._sample_img = photo(ramp(X[idx].reshape(28, 28), '#050607', INK), DRAW // 28)
        self.draw_canvas.create_image(0, 0, image=self._sample_img, anchor='nw')
        self.draw_canvas.create_text(10, 10, text=f'digits_test · etiqueta real: {self.true_label}', anchor='nw',
                                     fill=ACCENT, font=FONT_B)
        self.schedule()

    def reseed(self):
        self.noise_seed += 1
        self.schedule()

    # ----- inferencia -----

    def schedule(self):
        # Agrupa los eventos del mouse: como mucho una inferencia cada ~30 ms
        if self.pending is None:
            self.pending = self.after(30, self.update_view)

    def current_input(self):
        x = self.test_image.copy() if self.true_label is not None else to_dataset_format(self.canvas_data)
        sigma = self.noise.get()
        if sigma > 0:
            x = digitos.add_noise(x.reshape(1, -1), sigma, seed=self.noise_seed)[0]
        return x

    def update_view(self):
        self.pending = None
        x = self.current_input()
        empty = x.max() == 0
        self.seen.set(ramp(x.reshape(28, 28), '#050607', INK), 6,
                      f'etiqueta real: {self.true_label}' if self.true_label is not None else 'escalado a 20 px y centrado')
        for name, bars in self.bars.items():
            model = self.app.models.get(name)
            bars.show(None if model is None or empty else model.forward(x.reshape(1, -1))[0], self.true_label)

        model = self.app.models.get(self.focus.get())
        if model is None or empty:
            self.saliency.set(ramp(np.zeros((28, 28)), PANEL2, PANEL2), 6, '')
            self.hidden_canvas.delete('all')
            return
        pred = int(model.predict(x.reshape(1, -1))[0])
        grad = model.input_gradient(x.reshape(1, -1), pred)[0]
        attribution = grad * x
        scale = np.abs(attribution).max() or 1
        self.saliency.set(diverging((attribution / scale).reshape(28, 28)), 6,
                          f'naranja: empuja hacia el {pred} · azul: lo aleja')
        self.draw_hidden(model, x)

    def draw_hidden(self, model, x):
        acts = model.hidden_activations(x.reshape(1, -1))[0][0]
        n = len(acts)
        cols = int(np.ceil(np.sqrt(n * 2)))
        rows = int(np.ceil(n / cols))
        cell = max(4, min(340 // cols, 170 // rows))
        grid = np.full(rows * cols, np.nan)
        grid[:n] = acts
        lo = 0 if model.hidden.name in ('relu', 'logistic') else -1
        hi = max(acts.max(), 1e-6) if model.hidden.name == 'relu' else 1
        vals = (grid.reshape(rows, cols) - lo) / (hi - lo)
        rgb = ramp(np.nan_to_num(vals), PANEL2, '#ffd166')
        if self.selected_neuron is not None and self.selected_neuron < n:
            r, c = divmod(self.selected_neuron, cols)
            rgb[r, c] = hex_to_rgb(ACCENT)
        self.hidden_image = photo(rgb, cell)
        self.hidden_layout = (cols, cell, n)
        self.hidden_canvas.configure(width=cols * cell, height=rows * cell)
        self.hidden_canvas.delete('all')
        self.hidden_canvas.create_image(0, 0, image=self.hidden_image, anchor='nw')
        if self.selected_neuron is not None and self.selected_neuron < n:
            j = self.selected_neuron
            w = model.W[0][:, j].reshape(28, 28)
            self.neuron_panel.set(diverging(w / (np.abs(w).max() or 1)), 5,
                                  f'neurona {j} · activación {acts[j]:+.2f}')

    def on_neuron_click(self, e):
        if not self.hidden_layout:
            return
        cols, cell, n = self.hidden_layout
        j = (e.y // cell) * cols + e.x // cell
        if j < n:
            self.selected_neuron = int(j)
            self.update_view()


# ---------- pestaña 2: entrenamiento en vivo ----------

class EntrenamientoVivo(tk.Frame):
    VAL_SIZE = 2000

    def __init__(self, master, app):
        super().__init__(master, bg=BG)
        self.app = app
        self.running = False
        self.model = None

        controls = tk.Frame(self, bg=BG)
        controls.grid(row=0, column=0, columnspan=2, sticky='w', padx=16, pady=(16, 8))
        self.dataset = tk.StringVar(value='digits (sin 8)')
        self.optimizer = tk.StringVar(value='adam')
        self.lr = tk.StringVar(value='0.001')
        self.hidden = tk.StringVar(value='32')
        self.speed = tk.IntVar(value=8)
        for label, var, values in [('Datos', self.dataset, ['digits (sin 8)', 'digits + more_digits']),
                                   ('Optimizador', self.optimizer, ['gd', 'momentum', 'rmsprop', 'adam']),
                                   ('Neuronas ocultas', self.hidden, ['16', '32', '64'])]:
            tk.Label(controls, text=label, bg=BG, fg=MUTED, font=FONT).pack(side='left', padx=(0, 4))
            ttk.Combobox(controls, textvariable=var, values=values, state='readonly',
                         width=max(len(v) for v in values) + 2).pack(side='left', padx=(0, 14))
        tk.Label(controls, text='η', bg=BG, fg=MUTED, font=FONT).pack(side='left', padx=(0, 4))
        tk.Entry(controls, textvariable=self.lr, width=7, bg=PANEL2, fg=INK, insertbackground=INK,
                 relief='flat').pack(side='left', padx=(0, 14))
        self.play_button = self.button(controls, '▶  Entrenar', self.toggle)
        self.play_button.pack(side='left')
        self.button(controls, 'Reiniciar', self.reset).pack(side='left', padx=6)
        self.button(controls, 'Usar en la pizarra', self.publish).pack(side='left')

        speed = tk.Frame(self, bg=BG)
        speed.grid(row=1, column=0, columnspan=2, sticky='w', padx=16)
        tk.Label(speed, text='Velocidad (mini-batches por cuadro)', bg=BG, fg=MUTED, font=FONT).pack(side='left')
        tk.Scale(speed, from_=1, to=60, orient='horizontal', variable=self.speed, bg=BG, fg=INK,
                 troughcolor=PANEL2, highlightthickness=0, length=260, sliderrelief='flat').pack(side='left', padx=8)
        self.status = tk.Label(speed, text='', bg=BG, fg=INK, font=FONT_B)
        self.status.pack(side='left', padx=16)

        self.weights = ImagePanel(self, 'Pesos de cada neurona oculta (28×28): de ruido a trazos', 520)
        self.weights.grid(row=2, column=0, sticky='nw', padx=(16, 8), pady=12)
        right = tk.Frame(self, bg=BG)
        right.grid(row=2, column=1, sticky='nw', padx=(8, 16), pady=12)
        self.curve = tk.Canvas(right, width=420, height=200, bg=PANEL, highlightthickness=0)
        self.curve.pack(anchor='w')
        self.confusion = tk.Canvas(right, width=420, height=330, bg=PANEL, highlightthickness=0)
        self.confusion.pack(anchor='w', pady=(10, 0))
        self.reset()

    button = Pizarra.button

    def reset(self):
        self.running = False
        self.play_button.configure(text='▶  Entrenar')
        combined = self.dataset.get() != 'digits (sin 8)'
        X, y = self.app.train_set(combined)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(y))
        self.X_val, self.y_val = self.app.test_set()
        self.X_val, self.y_val = self.X_val[:self.VAL_SIZE], self.y_val[:self.VAL_SIZE]
        self.X, self.Y = X[order], digitos.one_hot(y[order])
        self.model = MLP([784, int(self.hidden.get()), 10], hidden='tanh', output='softmax',
                         loss='cross_entropy', seed=0, dtype=np.float32)
        try:
            lr = float(self.lr.get())
        except ValueError:
            lr = 1e-3
        self.opt = optimizadores.build(self.optimizer.get(), lr)
        self.cursor, self.seen, self.epoch = 0, 0, 0
        self.acc_history = []
        self.draw()

    def toggle(self):
        self.running = not self.running
        self.play_button.configure(text='⏸  Pausa' if self.running else '▶  Entrenar')
        if self.running:
            self.tick()

    def tick(self):
        if not self.running:
            return
        for _ in range(self.speed.get()):
            batch = slice(self.cursor, self.cursor + 64)
            _, grads = self.model.gradients(self.X[batch], self.Y[batch])
            self.opt.step(self.model.params, grads)
            self.cursor += 64
            self.seen += 64
            if self.cursor >= len(self.X):
                self.cursor, self.epoch = 0, self.epoch + 1
        self.draw()
        self.after(15, self.tick)

    def publish(self):
        self.app.models['En vivo'] = self.model
        self.app.pizarra.refresh_models()
        self.app.notebook.select(0)

    def draw(self):
        pred = self.model.predict(self.X_val)
        acc = float(np.mean(pred == self.y_val))
        self.acc_history.append((self.seen, acc))
        self.status.configure(text=f'época {self.epoch} · {self.seen:,} imágenes vistas · accuracy en test {acc:.1%}')

        # Pesos de la primera capa en mosaico
        W = self.model.W[0]
        n = W.shape[1]
        cols = 8
        rows = int(np.ceil(n / cols))
        tiles = np.zeros((rows * 29 - 1, cols * 29 - 1))
        for j in range(n):
            r, c = divmod(j, cols)
            w = W[:, j].reshape(28, 28)
            tiles[r * 29:r * 29 + 28, c * 29:c * 29 + 28] = w / (np.abs(w).max() or 1)
        scale = max(1, min(520 // tiles.shape[1], 520 // tiles.shape[0]))
        self.weights.set(diverging(tiles), scale, 'naranja: la neurona se excita con tinta ahí · azul: se inhibe')

        self.draw_curve()
        self.draw_confusion(pred)

    def draw_curve(self):
        c = self.curve
        c.delete('all')
        w, h, pad = 420, 200, 34
        c.create_text(12, 14, text='Accuracy en digits_test', anchor='w', fill=MUTED, font=FONT)
        for level in (0.5, 0.9, 1.0):
            y = h - pad - (h - 2 * pad) * level
            c.create_line(pad, y, w - 10, y, fill=PANEL2)
            c.create_text(pad - 4, y, text=f'{level:.0%}', anchor='e', fill=FAINT, font=('Helvetica', 9))
        if len(self.acc_history) > 1:
            xs = np.array([s for s, _ in self.acc_history], float)
            ys = np.array([a for _, a in self.acc_history])
            xs = pad + (w - 10 - pad) * xs / max(xs[-1], 1)
            ys = h - pad - (h - 2 * pad) * ys
            c.create_line(*np.column_stack([xs, ys]).ravel(), fill=ACCENT, width=2)
        if self.dataset.get() == 'digits (sin 8)':
            y = h - pad - (h - 2 * pad) * 0.9
            c.create_text(w - 12, y - 8, text='techo ~90%: sin ochos', anchor='e', fill=WARN, font=('Helvetica', 10))

    def draw_confusion(self, pred):
        c = self.confusion
        c.delete('all')
        c.create_text(12, 14, text='Matriz de confusión (filas: real, columnas: predicho)', anchor='w', fill=MUTED, font=FONT)
        cm = np.zeros((10, 10))
        np.add.at(cm, (self.y_val, pred), 1)
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cell, x0, y0 = 27, 40, 40
        for i in range(10):
            c.create_text(x0 - 12, y0 + i * cell + cell / 2, text=str(i), fill=MUTED, font=FONT)
            c.create_text(x0 + i * cell + cell / 2, y0 - 10, text=str(i), fill=MUTED, font=FONT)
            for j in range(10):
                v = cm[i, j]
                color = ramp(np.array(v), PANEL2, ACCENT if i == j else WARN)
                c.create_rectangle(x0 + j * cell, y0 + i * cell, x0 + (j + 1) * cell - 2, y0 + (i + 1) * cell - 2,
                                   fill='#%02x%02x%02x' % tuple(color), width=0)
        recall8 = cm[8, 8]
        c.create_text(x0 + 10 * cell + 12, y0 + 8 * cell + cell / 2, text=f'recall 8: {recall8:.0%}', anchor='w',
                      fill=WARN if recall8 < 0.5 else INK, font=('Helvetica', 10, 'bold'))


# ---------- pestaña 3: fraude (Ej. 1) ----------

FRAUD_DIR = Path(__file__).parent / 'resultados' / 'generalizacion'
# (feature, etiqueta, mínimo, máximo, paso) según los rangos del EDA
FRAUD_FEATURES = [
    ('amount_usd', 'Monto (USD)', 1, 2000, 1),
    ('quantity_purchased', 'Unidades compradas', 1, 24, 1),
    ('session_duration_seconds', 'Duración de la sesión (s)', 5, 727, 1),
    ('days_since_last_purchase', 'Días desde la última compra', 0, 142, 1),
    ('account_age_days', 'Antigüedad de la cuenta (días)', 1, 3649, 1),
    ('items_viewed_before_purchase', 'Ítems vistos antes de comprar', 1, 29, 1),
]
FRAUD_DEFAULTS = {'amount_usd': 63, 'quantity_purchased': 5, 'session_duration_seconds': 287,
                  'days_since_last_purchase': 9, 'account_age_days': 1633, 'items_viewed_before_purchase': 8}


class FraudeSimulador(tk.Frame):
    """TinyModel del Ej. 1 (perceptrón sigmoide) + recomendación de umbral según costos."""

    def __init__(self, master):
        super().__init__(master, bg=BG)
        import pandas as pd
        model = pd.read_csv(FRAUD_DIR / 'modelo_final.csv').set_index('feature')
        self.w = model.loc[[f for f, *_ in FRAUD_FEATURES], 'peso'].to_numpy()
        self.mean = model.loc[[f for f, *_ in FRAUD_FEATURES], 'media'].to_numpy()
        self.std = model.loc[[f for f, *_ in FRAUD_FEATURES], 'desvio'].to_numpy()
        self.bias = float(model.loc['bias', 'peso'])
        self.sweep = pd.read_csv(FRAUD_DIR / 'barrido_umbral_oof.csv')

        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky='n', padx=(16, 8), pady=16)
        tk.Label(left, text='Simulá una transacción', bg=BG, fg=INK, font=FONT_T).pack(anchor='w')
        tk.Label(left, text='TinyModel: 6 pesos + bias, contra un BigModel caro', bg=BG, fg=MUTED,
                 font=FONT).pack(anchor='w', pady=(0, 8))
        self.vars = {}
        for feat, label, lo, hi, step in FRAUD_FEATURES:
            tk.Label(left, text=label, bg=BG, fg=MUTED, font=FONT).pack(anchor='w')
            var = tk.DoubleVar(value=FRAUD_DEFAULTS[feat])
            tk.Scale(left, from_=lo, to=hi, resolution=step, orient='horizontal', variable=var,
                     command=lambda _: self.update_view(), bg=BG, fg=INK, troughcolor=PANEL2,
                     highlightthickness=0, length=360, sliderrelief='flat').pack(anchor='w')
            self.vars[feat] = var
        buttons = tk.Frame(left, bg=BG)
        buttons.pack(anchor='w', pady=8)
        Pizarra.button(self, buttons, 'Transacción típica', lambda: self.preset(FRAUD_DEFAULTS)).pack(side='left')
        Pizarra.button(self, buttons, 'Perfil sospechoso', lambda: self.preset({
            'amount_usd': 100, 'quantity_purchased': 14, 'session_duration_seconds': 60,
            'days_since_last_purchase': 1, 'account_age_days': 20, 'items_viewed_before_purchase': 18})).pack(side='left', padx=6)

        mid = tk.Frame(self, bg=BG)
        mid.grid(row=0, column=1, sticky='n', padx=8, pady=16)
        self.result = tk.Canvas(mid, width=380, height=380, bg=PANEL, highlightthickness=0)
        self.result.pack()

        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=2, sticky='n', padx=(8, 16), pady=16)
        tk.Label(right, text='Umbral de detección', bg=BG, fg=MUTED, font=FONT).pack(anchor='w')
        self.threshold = tk.DoubleVar(value=0.89)
        tk.Scale(right, from_=0.01, to=0.99, resolution=0.01, orient='horizontal', variable=self.threshold,
                 command=lambda _: self.update_view(), bg=BG, fg=INK, troughcolor=PANEL2, highlightthickness=0,
                 length=400, sliderrelief='flat').pack(anchor='w')
        tk.Label(right, text='Costo de un fraude no detectado ÷ costo de bloquear a un cliente legítimo',
                 bg=BG, fg=MUTED, font=FONT, wraplength=400, justify='left').pack(anchor='w', pady=(8, 0))
        self.cost_ratio = tk.DoubleVar(value=5)
        tk.Scale(right, from_=1, to=30, resolution=1, orient='horizontal', variable=self.cost_ratio,
                 command=lambda _: self.update_view(), bg=BG, fg=INK, troughcolor=PANEL2, highlightthickness=0,
                 length=400, sliderrelief='flat').pack(anchor='w')
        Pizarra.button(self, right, 'Usar el umbral de mínimo costo', self.use_best).pack(anchor='w', pady=6)
        self.metrics = tk.Canvas(right, width=400, height=420, bg=PANEL, highlightthickness=0)
        self.metrics.pack(anchor='w', pady=(6, 0))
        self.update_view()

    def preset(self, values):
        for k, v in values.items():
            self.vars[k].set(v)
        self.update_view()

    def costs(self):
        s = self.sweep
        return s.umbral.to_numpy(), (self.cost_ratio.get() * s.fn + s.fp).to_numpy()

    def use_best(self):
        t, cost = self.costs()
        self.threshold.set(float(t[np.argmin(cost)]))
        self.update_view()

    def update_view(self):
        x = np.array([self.vars[f].get() for f, *_ in FRAUD_FEATURES])
        z = (x - self.mean) / self.std
        contrib = self.w * z
        h = contrib.sum() + self.bias
        p = 1 / (1 + np.exp(-h))
        t = self.threshold.get()
        flagged = p >= t

        c = self.result
        c.delete('all')
        c.create_text(16, 18, text='Probabilidad estimada de fraude', anchor='w', fill=MUTED, font=FONT)
        c.create_text(16, 62, text=f'{p:.1%}', anchor='w', fill=INK, font=FONT_BIG)
        verdict, color = ('MARCAR COMO FRAUDE', WARN) if flagged else ('APROBAR', '#4fd1a0')
        c.create_text(364, 62, text=verdict, anchor='e', fill=color, font=FONT_B)
        c.create_text(364, 84, text=f'umbral {t:.2f}', anchor='e', fill=MUTED, font=('Helvetica', 10))
        c.create_text(16, 118, text='Cuánto empuja cada feature (peso × valor estandarizado)', anchor='w',
                      fill=MUTED, font=('Helvetica', 10))
        mid_x, half = 250, 110
        lim = max(3.0, np.abs(contrib).max())
        for i, ((feat, label, *_), v) in enumerate(zip(FRAUD_FEATURES, contrib)):
            y = 142 + i * 36
            c.create_text(16, y + 10, text=label.split(' (')[0], anchor='w', fill=INK, font=('Helvetica', 10))
            c.create_line(mid_x, y, mid_x, y + 22, fill=FAINT)
            x1 = mid_x + half * v / lim
            c.create_rectangle(min(mid_x, x1), y + 3, max(mid_x, x1), y + 19,
                               fill=WARN if v > 0 else ACCENT, width=0)
            c.create_text(mid_x + (half + 6) * (1 if v >= 0 else -1) if abs(v) < lim * 0.6 else mid_x,
                          y + 11, text=f'{v:+.2f}', anchor='w' if v >= 0 else 'e', fill=MUTED, font=('Helvetica', 9))
        c.create_text(16, 366, text='naranja: sube la probabilidad · azul: la baja', anchor='w', fill=FAINT,
                      font=('Helvetica', 10))

        # Métricas out-of-fold al umbral elegido (Ej. 1, sin tocar test)
        row = self.sweep.iloc[(self.sweep.umbral - t).abs().argmin()]
        m = self.metrics
        m.delete('all')
        m.create_text(16, 18, text='Con este umbral, sobre las predicciones out-of-fold (6000 transacciones)',
                      anchor='w', fill=MUTED, font=('Helvetica', 10))
        cells = [('Fraudes detectados', row.tp, '#4fd1a0'), ('Falsas alarmas', row.fp, WARN),
                 ('Fraudes que pasan', row.fn, WARN), ('Aprobadas bien', row.tn, MUTED)]
        for k, (label, value, color) in enumerate(cells):
            cx, cy = 16 + (k % 2) * 190, 38 + (k // 2) * 62
            m.create_rectangle(cx, cy, cx + 180, cy + 54, fill=PANEL2, width=0)
            m.create_text(cx + 10, cy + 16, text=label, anchor='w', fill=MUTED, font=('Helvetica', 10))
            m.create_text(cx + 10, cy + 38, text=f'{int(value):,}', anchor='w', fill=color, font=FONT_T)
        m.create_text(16, 176, text=f'precision {row.precision:.2f}   recall {row.recall:.2f}   F1 {row.f1:.2f}',
                      anchor='w', fill=INK, font=FONT_B)

        # Costo total vs umbral
        ts, cost = self.costs()
        x0, x1, y0, y1 = 40, 386, 214, 396
        m.create_text(16, 200, text=f'Costo total relativo (FN × {self.cost_ratio.get():.0f} + FP)', anchor='w',
                      fill=MUTED, font=('Helvetica', 10))
        cmax = cost.max()
        pts = np.column_stack([x0 + (x1 - x0) * ts, y1 - (y1 - y0) * cost / cmax]).ravel()
        m.create_line(x0, y1, x1, y1, fill=FAINT)
        m.create_line(*pts, fill=ACCENT, width=2)
        best = int(np.argmin(cost))
        bx, by = x0 + (x1 - x0) * ts[best], y1 - (y1 - y0) * cost[best] / cmax
        m.create_oval(bx - 5, by - 5, bx + 5, by + 5, fill='#4fd1a0', outline=PANEL, width=2)
        m.create_text(bx, by - 14, text=f'mínimo en {ts[best]:.2f}', fill=INK, font=('Helvetica', 10))
        tx = x0 + (x1 - x0) * t
        m.create_line(tx, y0, tx, y1, fill=WARN, dash=(3, 3))
        for v in (0, 0.5, 1):
            m.create_text(x0 + (x1 - x0) * v, y1 + 12, text=f'{v:g}', fill=FAINT, font=('Helvetica', 9))


# ---------- app ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TP3 · Perceptrón multicapa en vivo')
        self.configure(bg=BG)
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook', background=BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=PANEL, foreground=MUTED, padding=(16, 8), font=FONT_B)
        style.map('TNotebook.Tab', background=[('selected', PANEL2)], foreground=[('selected', INK)])
        style.configure('TCombobox', fieldbackground=PANEL2, background=PANEL2, foreground=INK, arrowcolor=INK)

        self.models = load_models()
        self._test = None
        self._train = {}
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        self.pizarra = Pizarra(self.notebook, self)
        self.notebook.add(self.pizarra, text='Pizarra')
        self.notebook.add(EntrenamientoVivo(self.notebook, self), text='Entrenamiento en vivo')
        if (FRAUD_DIR / 'modelo_final.csv').exists():
            self.notebook.add(FraudeSimulador(self.notebook), text='Fraude · TinyModel')

    def test_set(self):
        if self._test is None:
            self._test = digitos.load('digits_test')
        return self._test

    def train_set(self, combined: bool):
        if combined not in self._train:
            X, y = digitos.load('digits')
            if combined:
                X2, y2 = digitos.load('more_digits')
                X, y, _ = digitos.combine_without_duplicates(X, y, X2, y2)
            self._train[combined] = (X, y)
        return self._train[combined]


if __name__ == '__main__':
    App().mainloop()
