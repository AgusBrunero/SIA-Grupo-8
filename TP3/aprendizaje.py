"""Parte 1: capacidad de aprendizaje del perceptrón lineal vs. no lineal (sigmoide).

Se entrena con todas las muestras (sin split) y se mide el error de entrenamiento.
Uso (desde TP3/): python aprendizaje.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import load, fit_standardizer, standardize, FEATURES
from perceptron.perceptron import Neuron, linear, linear_derivative, sigmoid, sigmoid_derivative

OUT = Path(__file__).parent / 'resultados' / 'aprendizaje'
EPOCHS = 2000
SEEDS = range(5)
MODELS = {
    'lineal': (linear, linear_derivative, [0.01, 0.1, 0.5, 1.0]),
    'sigmoide': (sigmoid, sigmoid_derivative, [0.1, 1.0, 5.0, 20.0]),
}
# Todas las columnas menos target y etiqueta, para ver si descartar features limita la capacidad
ALL_FEATURES = FEATURES + ['timestamp', 'device_screen_resolution', 'time_since_last_login_s']
TARGET_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Paleta de referencia (skill dataviz): categórica azul/naranja, rampa ordinal azul, tinta neutra
COLORS = {'lineal': '#2a78d6', 'sigmoide': '#eb6834'}
LR_RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
DIVERGED = 1.0  # MSE por encima de esto se considera divergencia (el target está en [0, 1])


def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def train(X, y, model, lr, seed, epochs=EPOCHS):
    f, df, _ = MODELS[model]
    neuron = Neuron(X.shape[1], f, df, lr, seed=seed)
    history = np.empty(epochs)
    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(epochs):
            neuron.learn_batch(X, y)
            history[epoch] = mse(y, neuron.activate(X))
    return neuron, history


def ols_mse(X, y):
    # Mínimo global del MSE para cualquier modelo lineal: cota inferior del perceptrón lineal
    A = np.c_[X, np.ones(len(X))]
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return mse(y, A @ w)


def style(ax):
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def plot_lr_sweep(sweep, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        for color, (lr, history) in zip(LR_RAMP, sweep[model].items()):
            diverged = not np.isfinite(history[-1]) or history[-1] > DIVERGED
            label = f'lr={lr:g}' + (' (diverge)' if diverged else '')
            shown = np.where(history <= 1e3, history, np.nan)
            ax.plot(np.arange(1, EPOCHS + 1), shown, color=color, linewidth=2, label=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(5e-3, 10)
        ax.set_title(f'Perceptrón {model}', color=INK, loc='left')
        ax.set_xlabel('Época', color=MUTED)
        ax.legend(frameon=False, labelcolor=INK)
        style(ax)
    axes[0].set_ylabel('MSE de entrenamiento (log)', color=MUTED)
    fig.suptitle('Barrido de learning rate (semilla 0)', color=INK, x=0.06, ha='left')
    fig.tight_layout()
    fig.savefig(out / 'barrido_lr.png', dpi=150)
    plt.close(fig)


def plot_learning_curves(curves, best_lr, floor, baseline, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, EPOCHS + 1)
    for model, runs in curves.items():
        mean, std = runs.mean(axis=0), runs.std(axis=0)
        ax.plot(epochs, mean, color=COLORS[model], linewidth=2, label=f'{model} (lr={best_lr[model]:g})')
        ax.fill_between(epochs, mean - std, mean + std, color=COLORS[model], alpha=0.2, linewidth=0)
        ax.annotate(f'{model}: {mean[-1]:.4f}', (EPOCHS, mean[-1]), xytext=(6, 0),
                    textcoords='offset points', va='center', color=INK)
    ax.axhline(floor, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate(f'mejor lineal posible (OLS) = {floor:.4f}', (1.2, floor), xytext=(0, 4),
                textcoords='offset points', color=MUTED, fontsize=9)
    ax.axhline(baseline, color=MUTED, linestyle=':', linewidth=1)
    ax.annotate(f'predecir siempre la media = {baseline:.4f}', (1.2, baseline), xytext=(0, 4),
                textcoords='offset points', color=MUTED, fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Época', color=MUTED)
    ax.set_ylabel('MSE de entrenamiento (log)', color=MUTED)
    ax.set_title(f'Curva de aprendizaje: media ± desvío de {len(SEEDS)} semillas', color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK, loc='upper right')
    style(ax)
    fig.tight_layout()
    fig.savefig(out / 'curvas_aprendizaje.png', dpi=150)
    plt.close(fig)


def plot_predictions(y, preds, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    lo = min(p.min() for p in preds.values())
    hi = max(p.max() for p in preds.values())
    for ax, (model, pred) in zip(axes, preds.items()):
        ax.axhspan(lo - 0.1, 0, color=GRID, alpha=0.6, linewidth=0)
        ax.axhspan(1, hi + 0.1, color=GRID, alpha=0.6, linewidth=0)
        ax.scatter(y, pred, s=6, color=COLORS[model], alpha=0.35, linewidths=0)
        ax.plot([0, 1], [0, 1], color=INK, linewidth=1)
        outside = np.mean((pred < 0) | (pred > 1))
        ax.set_title(f'Perceptrón {model}: MSE={mse(y, pred):.4f}, fuera de [0,1]: {outside:.1%}',
                     color=INK, loc='left', fontsize=10)
        ax.set_xlabel('Probabilidad de BigModel (target)', color=MUTED)
        ax.set_ylim(lo - 0.1, hi + 0.1)
        style(ax)
    axes[0].set_ylabel('Salida del perceptrón', color=MUTED)
    fig.tight_layout()
    fig.savefig(out / 'prediccion_vs_bigmodel.png', dpi=150)
    plt.close(fig)


def error_by_bin(y, preds):
    bins = pd.cut(y, TARGET_BINS, include_lowest=True)
    table = pd.DataFrame({model: pd.Series(np.abs(y - p)).groupby(bins, observed=True).mean()
                          for model, p in preds.items()})
    table['n'] = pd.Series(y).groupby(bins, observed=True).size()
    table.index = [f'{lo:.1f}-{hi:.1f}' for lo, hi in zip(TARGET_BINS, TARGET_BINS[1:])]
    return table


def plot_error_by_bin(table, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(table))
    width = 0.38
    for i, model in enumerate(MODELS):
        bars = ax.bar(x + (i - 0.5) * (width + 0.02), table[model], width, color=COLORS[model], label=model)
        ax.bar_label(bars, fmt='%.3f', color=INK, fontsize=8, padding=2)
    ax.set_xticks(x, [f'{b}\n(n={n})' for b, n in zip(table.index, table['n'])])
    ax.set_xlabel('Rango de la probabilidad de BigModel', color=MUTED)
    ax.set_ylabel('Error absoluto medio', color=MUTED)
    ax.set_title('Dónde se equivoca cada perceptrón', color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(out / 'error_por_rango.png', dpi=150)
    plt.close(fig)


def summarize(model, lr, y, neuron, X, history, features):
    pred = neuron.activate(X)
    h = neuron.compute(X)
    row = {
        'modelo': model,
        'features': features,
        'lr': lr,
        'mse_final': mse(y, pred),
        'mae_final': float(np.mean(np.abs(y - pred))),
        'r2': 1 - mse(y, pred) / float(np.var(y)),
        'fuera_de_0_1': float(np.mean((pred < 0) | (pred > 1))),
        # Épocas hasta quedar a menos de 1% del error final: cuándo se satura el aprendizaje
        'epoca_1pct_final': int(np.argmax(history <= history[-1] * 1.01)) + 1,
        'mse_ep10': history[9], 'mse_ep100': history[99], 'mse_ep1000': history[999],
    }
    if model == 'sigmoide':
        # Neurona saturada: derivada de la sigmoide < 0.01 (|h| > ~4.6), casi no aprende de esas muestras
        row['neurona_saturada'] = float(np.mean(sigmoid_derivative(h) < 0.01))
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_raw, y, _ = load()
    X = standardize(X_raw, *fit_standardizer(X_raw))
    floor, baseline = ols_mse(X, y), float(np.var(y))

    sweep = {model: {lr: train(X, y, model, lr, seed=0)[1] for lr in lrs}
             for model, (_, _, lrs) in MODELS.items()}
    plot_lr_sweep(sweep, OUT)
    # Mejor lr: menor MSE promedio a lo largo de las épocas (premia converger rápido
    # y descarta los que divergen)
    best_lr = {model: min(runs, key=lambda lr: np.nan_to_num(runs[lr].mean(), nan=np.inf))
               for model, runs in sweep.items()}

    curves, preds, rows = {}, {}, []
    for model, lr in best_lr.items():
        runs = [train(X, y, model, lr, seed) for seed in SEEDS]
        curves[model] = np.array([history for _, history in runs])
        neuron, history = runs[0]
        preds[model] = neuron.activate(X)
        rows.append(summarize(model, lr, y, neuron, X, history, len(FEATURES)))

    # Control: con las 9 features, ¿cambia el techo de cada modelo?
    X_all_raw, _, _ = load(features=ALL_FEATURES)
    X_all = standardize(X_all_raw, *fit_standardizer(X_all_raw))
    for model, lr in best_lr.items():
        neuron, history = train(X_all, y, model, lr, seed=0)
        rows.append(summarize(model, lr, y, neuron, X_all, history, len(ALL_FEATURES)))

    plot_learning_curves(curves, best_lr, floor, baseline, OUT)
    plot_predictions(y, preds, OUT)
    table = error_by_bin(y, preds)
    plot_error_by_bin(table, OUT)

    summary = pd.DataFrame(rows)
    summary['std_mse_entre_semillas'] = summary['modelo'].map(
        {m: float(c[:, -1].std()) for m, c in curves.items()}).where(summary['features'] == len(FEATURES))
    summary.to_csv(OUT / 'resumen.csv', index=False)
    table.to_csv(OUT / 'error_por_rango.csv')

    pd.set_option('display.width', 200)
    print(f'MSE predecir la media: {baseline:.5f} | MSE óptimo lineal (OLS): {floor:.5f}')
    print('Finales del barrido de lr:', {m: {lr: round(float(h[-1]), 5) for lr, h in r.items()}
                                         for m, r in sweep.items()})
    print(summary.round(5).to_string(index=False))
    print(table.round(4).to_string())


if __name__ == '__main__':
    main()
