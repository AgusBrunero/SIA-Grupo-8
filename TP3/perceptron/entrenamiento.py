"""Loop de entrenamiento genérico para el MLP.

- batch_size = 1 → online (SGD); 1 < batch_size < p → mini-batch; batch_size = p → batch (GD).
- Early stopping sobre validación: se guarda el mejor modelo, no el último (clase 13).
- Devuelve el historial por época (loss/accuracy en train y validación, tiempo, η) para que
  los scripts de experimentos lo guarden y el análisis lo grafique por separado.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np

from perceptron.mlp import MLP
from perceptron import optimizadores


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    optimizer: str = 'adam'
    lr: float = 1e-3
    weight_decay: float = 0.0          # λ de L2
    dropout: float = 0.0
    patience: int | None = None        # early stopping: épocas sin mejorar en validación
    adaptive_lr: dict | None = None    # {'a': ..., 'b': ..., 'k': ...}
    augment: dict | None = None        # kwargs de digitos.augment, se aplica a cada época
    seed: int = 0
    optimizer_kwargs: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def accuracy(model: MLP, X, y, chunk: int = 4096) -> float:
    return float(np.mean(predict(model, X, chunk) == y))


def balanced_accuracy(model: MLP, X, y) -> float:
    """Promedio del recall de cada clase presente: no premia acertar solo las clases grandes."""
    pred = predict(model, X)
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def predict(model: MLP, X, chunk: int = 4096):
    return np.concatenate([model.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def evaluate_loss(model: MLP, X, Y, chunk: int = 4096) -> float:
    total = sum(model.loss(model.forward(X[i:i + chunk]), Y[i:i + chunk]) * len(X[i:i + chunk])
                for i in range(0, len(X), chunk))
    return total / len(X)


def train(model: MLP, X, Y, cfg: TrainConfig, X_val=None, Y_val=None,
          augment_fn: Callable | None = None, verbose: bool = True,
          on_epoch: Callable | None = None) -> dict:
    """Entrena `model` in-place. Y (y Y_val) van en one-hot; las etiquetas se deducen con argmax."""
    rng = np.random.default_rng(cfg.seed)
    opt = optimizadores.build(cfg.optimizer, cfg.lr, cfg.weight_decay, **cfg.optimizer_kwargs)
    adaptive = optimizadores.AdaptiveLR(opt, **cfg.adaptive_lr) if cfg.adaptive_lr else None
    y = Y.argmax(axis=1)
    y_val = Y_val.argmax(axis=1) if Y_val is not None else None

    history = {k: [] for k in ['loss', 'acc', 'val_loss', 'val_acc', 'val_bacc', 'lr', 'time']}
    best = (np.inf, None, 0)  # (val_loss, params, época)
    start = time.perf_counter()
    p = len(X)

    for epoch in range(1, cfg.epochs + 1):
        X_ep = augment_fn(X, rng) if augment_fn else X
        order = rng.permutation(p)
        for i in range(0, p, cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            _, grads = model.gradients(X_ep[idx], Y[idx], cfg.dropout, rng)
            opt.step(model.params, grads)

        # Métricas sobre los datos originales (sin augmentation ni dropout)
        history['loss'].append(evaluate_loss(model, X, Y))
        history['acc'].append(accuracy(model, X, y))
        history['lr'].append(opt.lr)
        history['time'].append(time.perf_counter() - start)
        if X_val is not None:
            history['val_loss'].append(evaluate_loss(model, X_val, Y_val))
            history['val_acc'].append(accuracy(model, X_val, y_val))
            history['val_bacc'].append(balanced_accuracy(model, X_val, y_val))
        if adaptive:
            adaptive.end_epoch(history['loss'][-1])

        if verbose:
            val = f"  val_loss={history['val_loss'][-1]:.4f} val_acc={history['val_acc'][-1]:.4f}" if X_val is not None else ''
            print(f"  época {epoch:3d}/{cfg.epochs}  loss={history['loss'][-1]:.4f} acc={history['acc'][-1]:.4f}{val}"
                  f"  ({history['time'][-1]:.1f}s)", flush=True)
        if on_epoch:
            on_epoch(epoch, history)

        if X_val is not None:
            if history['val_loss'][-1] < best[0]:
                best = (history['val_loss'][-1], [q.copy() for q in model.params], epoch)
            elif cfg.patience and epoch - best[2] >= cfg.patience:
                if verbose:
                    print(f'  early stopping: {cfg.patience} épocas sin mejorar (mejor época {best[2]})')
                break
        if not np.isfinite(history['loss'][-1]):
            if verbose:
                print('  diverge: loss no finita, se corta')
            break

    # Con early stopping (o simplemente con validación) se restaura el mejor modelo
    if cfg.patience and best[1] is not None:
        for q, b in zip(model.params, best[1]):
            q[...] = b
    history['best_epoch'] = best[2]
    return history
