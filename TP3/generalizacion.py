"""Parte 2: estudio de generalización del perceptrón sigmoide (TinyModel).

1. Se separa un test estratificado (20 %) que no se toca hasta el final.
2. Sobre el 80 % restante (dev) se hace k-fold estratificado para elegir lr, épocas,
   features y el umbral de fraude (con predicciones out-of-fold).
3. Se evalúa una sola vez en test: fidelidad a BigModel y detección de fraude real.

Uso (desde TP3/): python generalizacion.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import load, fit_standardizer, standardize, FEATURES
from perceptron.perceptron import Neuron, sigmoid, sigmoid_derivative

OUT = Path(__file__).parent / 'resultados' / 'generalizacion'
SEED = 42
TEST_FRACTION = 0.2
K = 5
LRS = [1.0, 5.0, 20.0]
MAX_EPOCHS = 500
ALL_FEATURES = FEATURES + ['timestamp', 'device_screen_resolution', 'time_since_last_login_s']
TRAIN_SIZES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
N_SPLITS = 50
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
BIG_MODEL_THRESHOLD = 0.85  # separa perfectamente flagged_fraud (edas/analisis_dataset.md §6)

# Paleta de referencia (skill dataviz)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


# ---------- particiones ----------

def stratified_split(labels, test_fraction, rng):
    test = np.concatenate([
        rng.permutation(idx)[:round(len(idx) * test_fraction)]
        for idx in (np.flatnonzero(labels == c) for c in np.unique(labels))
    ])
    return np.setdiff1d(np.arange(len(labels)), test), np.sort(test)


def random_split(n, test_fraction, rng):
    perm = rng.permutation(n)
    cut = round(n * test_fraction)
    return np.sort(perm[cut:]), np.sort(perm[:cut])


def stratified_kfold(labels, k, rng):
    # Cada clase se reparte por turnos entre los k folds: todos quedan con la misma proporción de fraude
    fold_of = np.empty(len(labels), dtype=int)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


# ---------- entrenamiento ----------

def mse(y, pred):
    return float(np.mean((y - pred) ** 2))


def train(X_tr, y_tr, lr, epochs, seed, X_val=None, y_val=None):
    # El escalado se ajusta solo con entrenamiento para no filtrar información de validación/test
    mean, std = fit_standardizer(X_tr)
    X_tr = standardize(X_tr, mean, std)
    neuron = Neuron(X_tr.shape[1], sigmoid, sigmoid_derivative, lr, seed=seed)
    train_hist, val_hist = np.empty(epochs), np.empty(epochs)
    X_val = None if X_val is None else standardize(X_val, mean, std)
    for epoch in range(epochs):
        neuron.learn_batch(X_tr, y_tr)
        train_hist[epoch] = mse(y_tr, neuron.activate(X_tr))
        if X_val is not None:
            val_hist[epoch] = mse(y_val, neuron.activate(X_val))
    predict = lambda X: neuron.activate(standardize(X, mean, std))
    return predict, train_hist, val_hist, (neuron, mean, std)


# ---------- métricas ----------

def roc_auc(labels, score):
    # Probabilidad de que un fraude tenga mayor score que una legítima (Mann-Whitney, con empates)
    ranks = pd.Series(score).rank().to_numpy()
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels, score):
    order = np.argsort(-score, kind='stable')
    hits = labels[order] == 1
    precision_at_k = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision_at_k[hits].mean())


def classification(labels, score, threshold):
    pred = score >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f = lambda b: (1 + b**2) * precision * recall / (b**2 * precision + recall) if precision + recall else 0.0
    return {'umbral': threshold, 'marcadas': tp + fp, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f(1), 'f2': f(2),
            'accuracy': (tp + tn) / len(labels)}


def regression(y, pred):
    return {'mse': mse(y, pred), 'mae': float(np.mean(np.abs(y - pred))),
            'r2': 1 - mse(y, pred) / float(np.var(y))}


# ---------- gráficos ----------

def style(ax):
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def plot_train_val(curves, lr, best_epoch):
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = np.arange(1, MAX_EPOCHS + 1)
    for key, color, label in (('train', BLUE, 'entrenamiento'), ('val', ORANGE, 'validación')):
        mean, std = curves[key].mean(axis=0), curves[key].std(axis=0)
        ax.plot(epochs, mean, color=color, linewidth=2, label=label)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    gap = curves['val'].mean(axis=0)[-1] - curves['train'].mean(axis=0)[-1]
    ax.annotate(f'curvas superpuestas: brecha final val − train = {gap:.5f}', (0.98, 0.5),
                xycoords='axes fraction', ha='right', color=INK, fontsize=9)
    ax.axvline(best_epoch, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate(f'mejor época = {best_epoch}', (best_epoch, curves['val'].mean(axis=0).max()),
                xytext=(6, 0), textcoords='offset points', color=MUTED, fontsize=9)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Época', color=MUTED)
    ax.set_ylabel('MSE vs BigModel (log)', color=MUTED)
    ax.set_title(f'Entrenamiento vs validación ({K}-fold estratificado, lr={lr:g}): media ± desvío',
                 color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    save(fig, 'curvas_train_val.png')


def plot_train_size(table):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color, label in (('train', BLUE, 'entrenamiento'), ('val', ORANGE, 'validación')):
        ax.errorbar(table['n_train'], table[f'mse_{key}'], yerr=table[f'std_{key}'], color=color,
                    linewidth=2, marker='o', markersize=5, capsize=3, label=label)
    ax.set_xscale('log')
    ax.set_xlabel('Muestras de entrenamiento (log)', color=MUTED)
    ax.set_ylabel('MSE vs BigModel', color=MUTED)
    ax.set_title('¿Cuántos datos necesita TinyModel? (media ± desvío entre folds)', color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    save(fig, 'tamano_entrenamiento.png')


def plot_split_variability(table):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [('mse_test', 'MSE en test vs BigModel'), ('ap_test', 'PR-AUC en test vs fraude real'),
               ('fraude_test', 'Proporción de fraude en test')]
    for ax, (col, title) in zip(axes, metrics):
        data = [table.loc[table.estrategia == s, col] for s in ('estratificado', 'aleatorio')]
        parts = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops={'color': INK})
        for patch, color in zip(parts['boxes'], (BLUE, ORANGE)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
        ax.set_xticks([1, 2], ['estratificado', 'aleatorio'])
        ax.set_title(title, color=INK, loc='left', fontsize=10)
        style(ax)
        ax.grid(axis='x', visible=False)
    fig.suptitle(f'Variabilidad según qué muestras caen en test ({N_SPLITS} particiones 80/20 por estrategia)',
                 color=INK, x=0.02, ha='left')
    save(fig, 'variabilidad_particion.png')


def plot_threshold(sweep, chosen):
    fig, ax = plt.subplots(figsize=(9, 5))
    for col, color, label in (('precision', BLUE, 'precision'), ('recall', ORANGE, 'recall'),
                              ('f1', AQUA, 'F1')):
        ax.plot(sweep['umbral'], sweep[col], color=color, linewidth=2, label=label)
    for name, t in chosen.items():
        ax.axvline(t, color=MUTED, linestyle='--', linewidth=1)
        ax.annotate(f'{name} = {t:.2f}', (t, 0.03), xytext=(-4, 0), textcoords='offset points',
                    rotation=90, ha='right', va='bottom', color=MUTED, fontsize=9)
    ax.set_xlabel('Umbral sobre la salida de TinyModel', color=MUTED)
    ax.set_ylabel('Métrica vs fraude real', color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.set_title('Elección del umbral con predicciones out-of-fold (dev)', color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK, loc='center left')
    style(ax)
    save(fig, 'umbral.png')


def plot_test_scores(score, labels, threshold):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 41)
    for c, color, label in ((0, BLUE, 'legítima'), (1, ORANGE, 'fraude')):
        ax.hist(score[labels == c], bins=bins, color=color, alpha=0.7, label=f'{label} (n={np.sum(labels == c)})')
    ax.axvline(threshold, color=INK, linestyle='--', linewidth=1)
    ax.annotate(f'umbral = {threshold:.2f}', (threshold, ax.get_ylim()[1] * 0.9), xytext=(6, 0),
                textcoords='offset points', color=INK, fontsize=9)
    ax.set_xlabel('Salida de TinyModel', color=MUTED)
    ax.set_ylabel('Transacciones', color=MUTED)
    ax.set_title('Test: distribución de la salida de TinyModel según la etiqueta real', color=INK, loc='left')
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    save(fig, 'test_distribucion.png')


# ---------- experimentos ----------

def select_hyperparameters(X, y, folds):
    rows, curves = [], {}
    for lr in LRS:
        runs = [train(X[tr], y[tr], lr, MAX_EPOCHS, SEED, X[va], y[va]) for tr, va in folds]
        curves[lr] = {'train': np.array([r[1] for r in runs]), 'val': np.array([r[2] for r in runs])}
        val = curves[lr]['val'].mean(axis=0)
        epoch = int(np.argmin(val)) + 1
        rows.append({'lr': lr, 'mejor_epoca': epoch, 'mse_val': val[epoch - 1],
                     'std_val': curves[lr]['val'][:, epoch - 1].std(),
                     'mse_train': curves[lr]['train'].mean(axis=0)[epoch - 1]})
    table = pd.DataFrame(rows)
    # A igual error (4 decimales) se prefiere el que necesita menos épocas
    best = table.assign(r=table.mse_val.round(4)).sort_values(['r', 'mejor_epoca']).iloc[0]
    return table, curves, float(best.lr), int(best.mejor_epoca)


def compare_features(X_dev_all, y, folds, lr, epochs):
    rows = []
    for name, cols in (('6 features', slice(0, len(FEATURES))), ('9 features', slice(None))):
        val = [mse(y[va], train(X_dev_all[tr][:, cols], y[tr], lr, epochs, SEED)[0](X_dev_all[va][:, cols]))
               for tr, va in folds]
        rows.append({'features': name, 'mse_val': np.mean(val), 'std_val': np.std(val)})
    return pd.DataFrame(rows)


def train_size_curve(X, y, labels, folds, lr, epochs, rng):
    rows = []
    for frac in TRAIN_SIZES:
        tr_mse, va_mse, n = [], [], []
        for tr, va in folds:
            sub = tr if frac == 1.0 else tr[stratified_split(labels[tr], 1 - frac, rng)[0]]
            predict = train(X[sub], y[sub], lr, epochs, SEED)[0]
            tr_mse.append(mse(y[sub], predict(X[sub])))
            va_mse.append(mse(y[va], predict(X[va])))
            n.append(len(sub))
        rows.append({'fraccion': frac, 'n_train': int(np.mean(n)), 'mse_train': np.mean(tr_mse),
                     'std_train': np.std(tr_mse), 'mse_val': np.mean(va_mse), 'std_val': np.std(va_mse)})
    return pd.DataFrame(rows)


def out_of_fold(X, y, folds, lr, epochs):
    oof = np.empty(len(y))
    for tr, va in folds:
        oof[va] = train(X[tr], y[tr], lr, epochs, SEED)[0](X[va])
    return oof


def split_variability(X, y, labels, lr, epochs):
    rows = []
    for i in range(N_SPLITS):
        rng = np.random.default_rng(1000 + i)
        for name, (tr, te) in (('estratificado', stratified_split(labels, TEST_FRACTION, rng)),
                               ('aleatorio', random_split(len(y), TEST_FRACTION, rng))):
            predict = train(X[tr], y[tr], lr, epochs, SEED)[0]
            score = predict(X[te])
            rows.append({'estrategia': name, 'particion': i, 'mse_test': mse(y[te], score),
                         'ap_test': average_precision(labels[te], score),
                         'fraude_test': labels[te].mean()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 200)
    rng = np.random.default_rng(SEED)
    X_all, y, labels = load(features=ALL_FEATURES)
    X = X_all[:, :len(FEATURES)]

    dev, test = stratified_split(labels, TEST_FRACTION, rng)
    folds_local = stratified_kfold(labels[dev], K, rng)
    # Índices de los folds expresados sobre el dataset completo
    folds = [(dev[tr], dev[va]) for tr, va in folds_local]
    print(f'dev={len(dev)} (fraude {labels[dev].mean():.3f}) | test={len(test)} (fraude {labels[test].mean():.3f})')

    # 1. Hiperparámetros por k-fold
    hp, curves, lr, epochs = select_hyperparameters(X, y, folds)
    hp.to_csv(OUT / 'seleccion_hiperparametros.csv', index=False)
    plot_train_val(curves[lr], lr, epochs)
    print(hp.round(5).to_string(index=False), f'\n-> lr={lr:g}, épocas={epochs}')

    feats = compare_features(X_all, y, folds, lr, epochs)
    feats.to_csv(OUT / 'comparacion_features.csv', index=False)
    print(feats.round(5).to_string(index=False))

    # 2. Tamaño del set de entrenamiento
    sizes = train_size_curve(X, y, labels, folds, lr, epochs, rng)
    sizes.to_csv(OUT / 'tamano_entrenamiento.csv', index=False)
    plot_train_size(sizes)
    print(sizes.round(5).to_string(index=False))

    # 3. Variabilidad según la partición (estratificada vs aleatoria)
    variability = split_variability(X, y, labels, lr, epochs)
    variability.to_csv(OUT / 'variabilidad_particion.csv', index=False)
    plot_split_variability(variability)
    print(variability.groupby('estrategia')[['mse_test', 'ap_test', 'fraude_test']]
          .agg(['mean', 'std', 'min', 'max']).round(4).to_string())

    # 4. Umbral con predicciones out-of-fold sobre dev (nunca con test)
    oof = out_of_fold(X, y, folds, lr, epochs)
    sweep = pd.DataFrame([classification(labels[dev], oof[dev], t) for t in THRESHOLDS])
    sweep.to_csv(OUT / 'barrido_umbral_oof.csv', index=False)
    # F1 pesa igual precision y recall; F2 pesa más el recall (un fraude que pasa cuesta más que una falsa alarma)
    chosen = {'max F1': float(sweep.loc[sweep.f1.idxmax(), 'umbral']),
              'max F2': float(sweep.loc[sweep.f2.idxmax(), 'umbral'])}
    plot_threshold(sweep, chosen)
    print('Umbrales candidatos:', chosen)
    print(sweep[sweep.umbral.isin(list(chosen.values()))].round(3).to_string(index=False))

    # 5. Modelo final: entrenado con todo dev, evaluado una sola vez en test
    predict, _, _, (neuron, mean, std) = train(X[dev], y[dev], lr, epochs, SEED)
    score_dev, score_test = predict(X[dev]), predict(X[test])
    reg = pd.DataFrame([{'conjunto': 'dev (entrenamiento)', **regression(y[dev], score_dev)},
                        {'conjunto': 'test', **regression(y[test], score_test)}])
    reg.to_csv(OUT / 'test_regresion.csv', index=False)
    print(reg.round(5).to_string(index=False))

    rows = []
    for name, t in chosen.items():
        rows.append({'modelo': f'TinyModel ({name})', 'roc_auc': roc_auc(labels[test], score_test),
                     'pr_auc': average_precision(labels[test], score_test),
                     **classification(labels[test], score_test, t)})
    big = y[test]
    rows.append({'modelo': 'BigModel (0,85)', 'roc_auc': roc_auc(labels[test], big),
                 'pr_auc': average_precision(labels[test], big),
                 **classification(labels[test], big, BIG_MODEL_THRESHOLD)})
    clf = pd.DataFrame(rows)
    clf.to_csv(OUT / 'test_clasificacion.csv', index=False)
    print(clf.round(4).to_string(index=False))

    # Fidelidad: ¿TinyModel toma la misma decisión que BigModel?
    t = chosen['max F1']
    agreement = float(np.mean((score_test >= t) == (big >= BIG_MODEL_THRESHOLD)))
    print(f'Acuerdo de decisión con BigModel en test (umbral {t:.2f}): {agreement:.4f}')
    plot_test_scores(score_test, labels[test], t)

    model = pd.DataFrame({'feature': FEATURES + ['bias'], 'peso': list(neuron.weights) + [neuron.bias],
                          'media': list(mean) + [np.nan], 'desvio': list(std) + [np.nan]})
    model.to_csv(OUT / 'modelo_final.csv', index=False)
    print(model.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
