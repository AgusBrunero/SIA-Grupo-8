"""Carga y preparación de los datasets de dígitos (Ej. 2 y 3).

Los CSV guardan cada imagen como un string "[0.0, 0.1, ...]" de 784 valores en [0, 1].
Parsearlos con ast.literal_eval (como hace digit_dataset_loader.py) tarda mucho, así que
se parsean una sola vez y se cachean en datasets/cache/*.npz.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = Path(__file__).parent / 'datasets'
CACHE = DATASETS / 'cache'
SIDE = 28
N_PIXELS = SIDE * SIDE
N_CLASSES = 10


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (X, y) con X de forma (n, 784) en float32 e y de forma (n,) en int."""
    cache = CACHE / f'{name}.npz'
    if cache.exists():
        data = np.load(cache)
        return data['X'], data['y']

    df = pd.read_csv(DATASETS / f'{name}.csv')
    # Sacar los corchetes y dejar que numpy parsee los números separados por coma
    flat = ','.join(df['image'].str.strip('[]'))
    X = np.array(flat.split(','), dtype=np.float32).reshape(len(df), N_PIXELS)
    y = df['label'].to_numpy(int)

    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def one_hot(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def unique_rows(X: np.ndarray) -> np.ndarray:
    """Índices de la primera aparición de cada imagen distinta."""
    _, first = np.unique(X, axis=0, return_index=True)
    return np.sort(first)


def combine_without_duplicates(X_a, y_a, X_b, y_b):
    """Concatena dos datasets y descarta las imágenes repetidas (dentro y entre ambos).

    Si no se quitan, las repetidas pesan doble y pueden caer una en train y otra en
    validación: la validación deja de medir generalización (leakage).
    """
    X = np.concatenate([X_a, X_b])
    y = np.concatenate([y_a, y_b])
    keep = unique_rows(X)
    return X[keep], y[keep], len(X) - len(keep)


def stratified_split(y: np.ndarray, val_fraction: float, seed: int):
    """Separa train/validación manteniendo la proporción de cada clase."""
    rng = np.random.default_rng(seed)
    val = np.concatenate([
        rng.permutation(idx)[:round(len(idx) * val_fraction)]
        for idx in (np.flatnonzero(y == c) for c in np.unique(y))
    ])
    train = np.setdiff1d(np.arange(len(y)), val)
    return rng.permutation(train), np.sort(val)


def class_counts(y: np.ndarray) -> np.ndarray:
    return np.bincount(y, minlength=N_CLASSES)


# ---------- data augmentation (solo sobre train) ----------

def shift(images: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Traslada cada imagen (dx, dy) píxeles rellenando con 0."""
    imgs = images.reshape(-1, SIDE, SIDE)
    out = np.zeros_like(imgs)
    for sx in np.unique(dx):
        for sy in np.unique(dy):
            sel = (dx == sx) & (dy == sy)
            if not sel.any():
                continue
            rolled = np.roll(imgs[sel], (sy, sx), axis=(1, 2))
            if sy > 0: rolled[:, :sy] = 0
            if sy < 0: rolled[:, sy:] = 0
            if sx > 0: rolled[:, :, :sx] = 0
            if sx < 0: rolled[:, :, sx:] = 0
            out[sel] = rolled
    return out.reshape(-1, N_PIXELS)


def rotate(images: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Rota cada imagen alrededor del centro (vecino más cercano, sin librerías extra)."""
    imgs = images.reshape(-1, SIDE, SIDE)
    n = len(imgs)
    c = (SIDE - 1) / 2
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    xx, yy = xx - c, yy - c
    t = np.deg2rad(angles_deg)[:, None, None]
    # Rotación inversa: para cada píxel destino busco de dónde viene
    src_x = np.rint(np.cos(t) * xx + np.sin(t) * yy + c).astype(int)
    src_y = np.rint(-np.sin(t) * xx + np.cos(t) * yy + c).astype(int)
    valid = (src_x >= 0) & (src_x < SIDE) & (src_y >= 0) & (src_y < SIDE)
    idx = np.arange(n)[:, None, None]
    out = np.where(valid, imgs[idx, np.clip(src_y, 0, SIDE - 1), np.clip(src_x, 0, SIDE - 1)], 0)
    return out.reshape(-1, N_PIXELS).astype(images.dtype)


def augment(X: np.ndarray, rng: np.random.Generator, max_shift: int = 2,
            max_angle: float = 12.0, noise: float = 0.0) -> np.ndarray:
    """Versión perturbada de X. Ángulos chicos: rotar mucho un 6 lo convierte en un 9."""
    n = len(X)
    out = rotate(X, rng.uniform(-max_angle, max_angle, n)) if max_angle > 0 else X
    if max_shift > 0:
        out = shift(out, rng.integers(-max_shift, max_shift + 1, n), rng.integers(-max_shift, max_shift + 1, n))
    if noise > 0:
        out = np.clip(out + rng.normal(0, noise, out.shape).astype(out.dtype), 0, 1)
    return out


def add_noise(X: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Ruido gaussiano N(0, σ²) recortado a [0, 1] (opcional de robustez)."""
    rng = np.random.default_rng(seed)
    return np.clip(X + rng.normal(0, sigma, X.shape).astype(X.dtype), 0, 1)


if __name__ == '__main__':
    for name in ['digits', 'digits_test', 'more_digits']:
        X, y = load(name)
        print(f'{name:12s} {X.shape}  rango=[{X.min():.2f}, {X.max():.2f}]  por clase={class_counts(y).tolist()}')
