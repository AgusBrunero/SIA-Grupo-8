"""Opcional: perceptrón no lineal con ReLU. ¿Cambian las conclusiones de las partes 1 y 2?

Parte 1: mismo estudio de aprendizaje con todas las muestras, contra lineal y sigmoide.
Parte 2: misma partición dev/test y mismos folds que generalizacion.py.
Uso (desde TP3/): python relu.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import generalizacion as gen
from aprendizaje import ols_mse, style
from datos import load, fit_standardizer, standardize
from perceptron.perceptron import (Neuron, linear, linear_derivative, relu, relu_derivative,
                                   sigmoid, sigmoid_derivative)

OUT = Path(__file__).parent / 'resultados' / 'relu'
EPOCHS = 2000
SEEDS = range(5)
RELU_LRS = [0.01, 0.1, 0.5, 1.0, 2.0]
# lineal y sigmoide con el mejor lr de la Parte 1
MODELS = {
    'lineal': (linear, linear_derivative, 0.5),
    'sigmoide': (sigmoid, sigmoid_derivative, 20.0),
    'relu': (relu, relu_derivative, None),
}
SIGMOID_EPOCHS = 41  # elegido por k-fold en generalizacion.py
LINEAR_EPOCHS = 100  # con lr=0,5 el lineal converge en ~10 épocas (Parte 1)
CV_LRS = [0.1, 0.5]
TARGET_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEAD = 0.99  # neurona muerta: h <= 0 en (casi) todas las muestras, el gradiente es 0

COLORS = {'lineal': '#2a78d6', 'sigmoide': '#eb6834', 'relu': '#1baf7a'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def train_full(X, y, model, lr, seed):
    f, df, _ = MODELS[model]
    neuron = Neuron(X.shape[1], f, df, lr, seed=seed)
    history = np.empty(EPOCHS)
    with np.errstate(over='ignore', invalid='ignore'):
        for epoch in range(EPOCHS):
            neuron.learn_batch(X, y)
            history[epoch] = gen.mse(y, neuron.activate(X))
    return neuron, history


def describe(y, pred, h=None):
    stats = {**gen.regression(y, pred),
             'fuera_de_0_1': float(np.mean((pred < 0) | (pred > 1))),
             'mayor_a_1': float(np.mean(pred > 1))}
    if h is not None:
        # Solo tiene sentido para ReLU: muestras en la zona sin gradiente
        stats['h_no_positivo'] = float(np.mean(h <= 0))
    return stats


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def plot_relu_sweep(histories, table):
    fig, axes = plt.subplots(1, len(RELU_LRS), figsize=(16, 4), sharey=True)
    for ax, lr in zip(axes, RELU_LRS):
        for seed in SEEDS:
            ax.plot(np.arange(1, EPOCHS + 1), np.where(histories[lr, seed] <= 1e3, histories[lr, seed], np.nan),
                    color=COLORS['relu'], linewidth=1.5, alpha=0.8)
        dead = int(table[(table.lr == lr)].muerta.sum())
        ax.set_title(f'lr={lr:g}: {dead}/{len(SEEDS)} muertas', color=INK, loc='left', fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-2, 5)
        ax.set_xlabel('Época', color=MUTED)
        style(ax)
    axes[0].set_ylabel('MSE de entrenamiento (log)', color=MUTED)
    fig.suptitle(f'Perceptrón ReLU: barrido de learning rate, una línea por semilla ({len(SEEDS)})',
                 color=INK, x=0.02, ha='left')
    save(fig, 'barrido_lr_relu.png')


def plot_curves(curves, lrs, floor):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, EPOCHS + 1)
    for model, runs in curves.items():
        mean = runs.mean(axis=0)
        ax.plot(epochs, mean, color=COLORS[model], linewidth=2, label=f'{model} (lr={lrs[model]:g})')
        # Rango mín–máx entre semillas: media ± desvío se vuelve negativo en escala log
        ax.fill_between(epochs, runs.min(axis=0), runs.max(axis=0), color=COLORS[model], alpha=0.2, linewidth=0)
        ax.annotate(f'{model}: {mean[-1]:.4f}', (EPOCHS, mean[-1]), xytext=(6, -8 if model == 'relu' else 6),
                    textcoords='offset points', va='center', color=INK, fontsize=9)
    ax.axhline(floor, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate(f'mejor lineal posible (OLS) = {floor:.4f}', (1.2, floor), xytext=(0, 4),
                textcoords='offset points', color=MUTED, fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Época', color=MUTED)
    ax.set_ylabel('MSE de entrenamiento (log)', color=MUTED)
    ax.set_title(f'Curva de aprendizaje de las tres activaciones: media y rango de {len(SEEDS)} semillas',
                 color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK, loc='upper right')
    style(ax)
    save(fig, 'curvas_tres_activaciones.png')


def plot_predictions(y, preds):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    lo = min(p.min() for p in preds.values())
    hi = max(p.max() for p in preds.values())
    for ax, (model, pred) in zip(axes, preds.items()):
        ax.axhspan(lo - 0.1, 0, color=GRID, alpha=0.6, linewidth=0)
        ax.axhspan(1, hi + 0.1, color=GRID, alpha=0.6, linewidth=0)
        ax.scatter(y, pred, s=6, color=COLORS[model], alpha=0.35, linewidths=0)
        ax.plot([0, 1], [0, 1], color=INK, linewidth=1)
        outside = np.mean((pred < 0) | (pred > 1))
        ax.set_title(f'{model}: MSE={gen.mse(y, pred):.4f}, fuera de [0,1]: {outside:.1%}',
                     color=INK, loc='left', fontsize=10)
        ax.set_xlabel('Probabilidad de BigModel (target)', color=MUTED)
        ax.set_ylim(lo - 0.1, hi + 0.1)
        style(ax)
    axes[0].set_ylabel('Salida del perceptrón', color=MUTED)
    save(fig, 'prediccion_vs_bigmodel.png')


def error_by_bin(y, preds):
    bins = pd.cut(y, TARGET_BINS, include_lowest=True)
    table = pd.DataFrame({m: pd.Series(np.abs(y - p)).groupby(bins, observed=True).mean() for m, p in preds.items()})
    table['n'] = pd.Series(y).groupby(bins, observed=True).size()
    table.index = [f'{lo:.1f}-{hi:.1f}' for lo, hi in zip(TARGET_BINS, TARGET_BINS[1:])]
    return table


def part1():
    X_raw, y, _ = load()
    X = standardize(X_raw, *fit_standardizer(X_raw))

    rows, histories = [], {}
    for lr in RELU_LRS:
        for seed in SEEDS:
            neuron, history = train_full(X, y, 'relu', lr, seed)
            histories[lr, seed] = history
            with np.errstate(over='ignore', invalid='ignore'):
                h = neuron.compute(X)
                stats = describe(y, neuron.activate(X), h)
            rows.append({'lr': lr, 'semilla': seed, 'mse_final': stats['mse'],
                         'mse_promedio_epocas': float(np.nanmean(np.where(np.isfinite(history), history, np.nan))),
                         'h_no_positivo': stats['h_no_positivo'], 'mayor_a_1': stats['mayor_a_1'],
                         'muerta': stats['h_no_positivo'] >= DEAD})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / 'barrido_lr_relu.csv', index=False)
    plot_relu_sweep(histories, sweep)
    # Mejor lr: menor MSE promedio a lo largo de las épocas, promediado entre semillas
    lrs = {m: lr for m, (_, _, lr) in MODELS.items()}
    lrs['relu'] = float(sweep.groupby('lr').mse_promedio_epocas.mean().idxmin())

    curves, preds, summary = {}, {}, []
    for model, lr in lrs.items():
        runs = [train_full(X, y, model, lr, seed) for seed in SEEDS]
        curves[model] = np.array([hist for _, hist in runs])
        neuron, history = runs[0]
        preds[model] = neuron.activate(X)
        h = neuron.compute(X) if model == 'relu' else None
        summary.append({'modelo': model, 'lr': lr, **describe(y, preds[model], h),
                        'std_mse_entre_semillas': float(curves[model][:, -1].std()),
                        'epoca_1pct_final': int(np.argmax(history <= history[-1] * 1.01)) + 1})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'aprendizaje_resumen.csv', index=False)
    bins = error_by_bin(y, preds)
    bins.to_csv(OUT / 'error_por_rango.csv')
    plot_curves(curves, lrs, ols_mse(X, y))
    plot_predictions(y, preds)

    print('== Parte 1 ==')
    print(sweep.groupby('lr').agg(mse_final=('mse_final', 'mean'), std=('mse_final', 'std'),
                                  muertas=('muerta', 'sum'), h_no_positivo=('h_no_positivo', 'mean'),
                                  mayor_a_1=('mayor_a_1', 'mean')).round(5).to_string())
    print(summary.round(5).to_string(index=False))
    print(bins.round(4).to_string())


def part2():
    X_all, y, labels = load(features=gen.ALL_FEATURES)
    X = X_all[:, :len(gen.FEATURES)]
    # Misma secuencia aleatoria que generalizacion.py: misma partición y mismos folds
    rng = np.random.default_rng(gen.SEED)
    dev, test = gen.stratified_split(labels, gen.TEST_FRACTION, rng)
    folds = [(dev[tr], dev[va]) for tr, va in gen.stratified_kfold(labels[dev], gen.K, rng)]
    act = {'activation': relu, 'derivative': relu_derivative}

    rows = []
    for lr in CV_LRS:
        val = np.array([gen.train(X[tr], y[tr], lr, gen.MAX_EPOCHS, gen.SEED, X[va], y[va], **act)[2]
                        for tr, va in folds])
        epoch = int(np.argmin(val.mean(axis=0))) + 1
        rows.append({'lr': lr, 'mejor_epoca': epoch, 'mse_val': val.mean(axis=0)[epoch - 1],
                     'std_val': val[:, epoch - 1].std()})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / 'seleccion_hiperparametros_relu.csv', index=False)
    # A igual error (4 decimales) se prefiere el que necesita menos épocas, como en generalizacion.py
    best = cv.assign(r=cv.mse_val.round(4)).sort_values(['r', 'mejor_epoca']).iloc[0]
    lr, epochs = float(best.lr), int(best.mejor_epoca)

    oof = np.empty(len(y))
    for tr, va in folds:
        oof[va] = gen.train(X[tr], y[tr], lr, epochs, gen.SEED, **act)[0](X[va])
    sweep = pd.DataFrame([gen.classification(labels[dev], oof[dev], t) for t in gen.THRESHOLDS])
    sweep.to_csv(OUT / 'barrido_umbral_oof_relu.csv', index=False)
    thresholds = {'relu': {'max F1': float(sweep.loc[sweep.f1.idxmax(), 'umbral']),
                           'max F2': float(sweep.loc[sweep.f2.idxmax(), 'umbral'])},
                  # umbrales elegidos en generalizacion.py
                  'sigmoide': {'max F1': 0.89, 'max F2': 0.80}}

    configs = {'relu': (lr, epochs, act), 'sigmoide': (20.0, SIGMOID_EPOCHS, {})}
    rows = []
    for model, (m_lr, m_epochs, kwargs) in configs.items():
        predict, _, _, (neuron, mean, std) = gen.train(X[dev], y[dev], m_lr, m_epochs, gen.SEED, **kwargs)
        score_dev, score = predict(X[dev]), predict(X[test])
        base = {'modelo': model, 'lr': m_lr, 'epocas': m_epochs,
                'mse_train': gen.mse(y[dev], score_dev),
                **describe(y[test], score, neuron.compute(standardize(X[test], mean, std)) if kwargs else None),
                'roc_auc': gen.roc_auc(labels[test], score), 'pr_auc': gen.average_precision(labels[test], score)}
        for name, t in thresholds[model].items():
            c = gen.classification(labels[test], score, t)
            rows.append({**base, 'criterio': name, **{k: c[k] for k in ('umbral', 'precision', 'recall', 'f1', 'fp', 'fn')}})
    test_table = pd.DataFrame(rows)
    test_table.to_csv(OUT / 'test_relu_vs_sigmoide.csv', index=False)

    configs['lineal'] = (0.5, LINEAR_EPOCHS, {'activation': linear, 'derivative': linear_derivative})
    paired = paired_splits(X, y, labels, configs)
    paired.to_csv(OUT / 'particiones_pareadas.csv', index=False)
    plot_paired(paired)

    print('== Parte 2 ==')
    print(cv.round(5).to_string(index=False))
    print(test_table.round(4).to_string(index=False))
    print(paired.groupby('modelo')[['mse', 'roc_auc', 'pr_auc']].agg(['mean', 'std']).round(4).to_string())
    wide = paired.pivot(index='particion', columns='modelo', values='pr_auc')
    for model in ('relu', 'lineal'):
        diff = wide[model] - wide['sigmoide']
        print(f'PR-AUC {model} - sigmoide: media {diff.mean():+.4f}, desvío {diff.std():.4f}, '
              f'{model} gana en {int((diff > 0).sum())}/{len(diff)} particiones')


def paired_splits(X, y, labels, configs):
    # Los tres modelos se evalúan sobre las mismas particiones: las diferencias no dependen del azar del split
    rows = []
    for i in range(gen.N_SPLITS):
        tr, te = gen.stratified_split(labels, gen.TEST_FRACTION, np.random.default_rng(1000 + i))
        for model, (lr, epochs, kwargs) in configs.items():
            score = gen.train(X[tr], y[tr], lr, epochs, gen.SEED, **kwargs)[0](X[te])
            rows.append({'particion': i, 'modelo': model, 'mse': gen.mse(y[te], score),
                         'roc_auc': gen.roc_auc(labels[te], score),
                         'pr_auc': gen.average_precision(labels[te], score)})
    return pd.DataFrame(rows)


def plot_paired(paired):
    wide = paired.pivot(index='particion', columns='modelo', values='pr_auc')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    models = ['lineal', 'sigmoide', 'relu']
    for ax, col, title in ((axes[0], 'mse', 'MSE en test vs BigModel (fidelidad)'),
                           (axes[1], 'pr_auc', 'PR-AUC en test vs fraude real (detección)')):
        data = [paired.loc[paired.modelo == m, col] for m in models]
        parts = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={'color': INK})
        for patch, m in zip(parts['boxes'], models):
            patch.set_facecolor(COLORS[m])
            patch.set_edgecolor(COLORS[m])
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2, 3], models)
        ax.set_title(title, color=INK, loc='left', fontsize=10)
        style(ax)
        ax.grid(axis='x', visible=False)
    wins = int((wide['relu'] > wide['sigmoide']).sum())
    fig.suptitle(f'{gen.N_SPLITS} particiones 80/20 estratificadas, las mismas para los tres modelos '
                 f'(ReLU supera al sigmoide en PR-AUC en {wins}/{len(wide)})', color=INK, x=0.02, ha='left')
    save(fig, 'particiones_pareadas.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 250)
    part1()
    part2()


if __name__ == '__main__':
    main()
