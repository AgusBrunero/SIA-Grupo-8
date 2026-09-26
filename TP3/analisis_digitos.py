"""Análisis de los Ej. 2 y 3: solo lee resultados/digitos/ y modelos/ y genera gráficos y tablas.

Uso (desde TP3/):  python analisis_digitos.py
"""
from collections import defaultdict
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import digitos
from experimentos import load_runs, RESULTS, MODELS
from perceptron.mlp import MLP
from perceptron.entrenamiento import predict

# Paleta de referencia (skill dataviz): categórica en orden fijo, secuencial azul, divergente azul↔rojo
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
SEQUENTIAL = LinearSegmentedColormap.from_list('azul', ['#fcfcfb', '#cde2fb', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'])
DIVERGING = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f19a99', '#b8302f'])
OPTIMIZERS = ['gd', 'momentum', 'rmsprop', 'adam']


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def title(ax, text):
    ax.set_title(text, color=INK, loc='left', fontsize=11)


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def by_variant(runs, group):
    g = defaultdict(list)
    for r in runs:
        if r['group'] == group:
            g[r['variant']].append(r)
    return g


def summary_table(runs) -> pd.DataFrame:
    rows = []
    for (group, variant), rs in sorted({(r['group'], r['variant']): None for r in runs}.items()):
        rs = [r for r in runs if r['group'] == group and r['variant'] == variant]
        best = [max(r['history']['val_acc']) for r in rs]
        bacc = [max(r['history']['val_bacc']) for r in rs if r['history'].get('val_bacc')]
        row = {'grupo': group, 'variante': variant, 'semillas': len(rs),
               'val_acc_max_media': np.mean(best), 'val_acc_max_desvio': np.std(best),
               'val_acc_final_media': np.mean([r['history']['val_acc'][-1] for r in rs]),
               'val_bacc_max_media': np.mean(bacc) if bacc else None,
               'train_acc_final_media': np.mean([r['history']['acc'][-1] for r in rs]),
               'mejor_epoca_media': np.mean([r['history']['best_epoch'] for r in rs]),
               'epocas_corridas': np.mean([len(r['history']['acc']) for r in rs]),
               'tiempo_s_media': np.mean([r['history']['time'][-1] for r in rs]),
               'parametros': rs[0]['n_params']}
        if 'test_accuracy' in rs[0]:
            row['test_acc_media'] = np.mean([r['test_accuracy'] for r in rs])
            row['test_acc_desvio'] = np.std([r['test_accuracy'] for r in rs])
            row['test_recall_8_media'] = np.mean([r['test_recall_por_clase'][8] for r in rs])
            row['test_recall_5_media'] = np.mean([r['test_recall_por_clase'][5] for r in rs])
        rows.append(row)
    return pd.DataFrame(rows)


def mean_curve(rs, key):
    n = min(len(r['history'][key]) for r in rs)
    arr = np.array([r['history'][key][:n] for r in rs])
    return np.arange(1, n + 1), arr.mean(axis=0), arr.std(axis=0)


def plot_curves(ax, groups: dict, key='val_acc', labels=None):
    for i, (variant, rs) in enumerate(groups.items()):
        x, m, s = mean_curve(rs, key)
        color = SERIES[i % len(SERIES)]
        ax.plot(x, m, color=color, linewidth=2, label=labels[i] if labels else variant)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel('Época', color=MUTED)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    style(ax)


# ---------- Ej. 2 ----------

def ej2_learning_rate(runs, out):
    g = by_variant(runs, 'optimizador_lr')
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for i, opt in enumerate(OPTIMIZERS):
        pts = []
        for variant, rs in g.items():
            cfg = rs[0]['config']
            if cfg['optimizer'] == opt and not cfg.get('adaptive_lr'):
                best = [max(r['history']['val_acc']) for r in rs]
                pts.append((cfg['lr'], np.mean(best), np.std(best)))
        pts.sort()
        lr, m, s = map(np.array, zip(*pts))
        ax.errorbar(lr, m, yerr=s, color=SERIES[i], marker='o', markersize=7, linewidth=2, capsize=3, label=opt)
        rows += [{'optimizador': opt, 'lr': a, 'val_acc_max_media': b, 'desvio': c} for a, b, c in pts]
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Tasa de aprendizaje η (log)', color=MUTED)
    ax.set_ylabel('Accuracy máxima en validación', color=MUTED)
    title(ax, 'Ej. 2 · Tasa de aprendizaje por optimizador (media ± desvío, 3 semillas)')
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    save(fig, out, 'tasa_aprendizaje.png')

    # Zoom sobre la zona útil (las que no divergen)
    fig, ax = plt.subplots(figsize=(9, 5))
    df = pd.DataFrame(rows)
    for i, opt in enumerate(OPTIMIZERS):
        d = df[(df.optimizador == opt) & (df.val_acc_max_media > 0.5)]
        ax.errorbar(d.lr, d.val_acc_max_media, yerr=d.desvio, color=SERIES[i], marker='o', markersize=7,
                    linewidth=2, capsize=3, label=opt)
    ax.set_xscale('log')
    ax.set_xlabel('Tasa de aprendizaje η (log)', color=MUTED)
    ax.set_ylabel('Accuracy máxima en validación', color=MUTED)
    title(ax, 'Ej. 2 · Zoom en las η que convergen')
    ax.legend(frameon=False, labelcolor=INK)
    style(ax)
    save(fig, out, 'tasa_aprendizaje_zoom.png')
    return df


def ej2_optimizers(runs, out):
    g = by_variant(runs, 'optimizador_lr')
    best = {}
    for variant, rs in g.items():
        cfg = rs[0]['config']
        key = cfg['optimizer'] + (' adaptativo' if cfg.get('adaptive_lr') else '')
        score = np.mean([max(r['history']['val_acc']) for r in rs])
        if key not in best or score > best[key][0]:
            best[key] = (score, variant, rs)
    order = [k for k in ['gd', 'gd adaptativo', 'momentum', 'rmsprop', 'adam'] if k in best]
    groups = {best[k][1]: best[k][2] for k in order}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_curves(axes[0], groups, 'val_acc', labels=[f'{k} (η={best[k][2][0]["config"]["lr"]:g})' for k in order])
    axes[0].set_ylim(0.85, 0.98)
    axes[0].set_ylabel('Accuracy en validación', color=MUTED)
    title(axes[0], 'Ej. 2 · Cada optimizador con su mejor η')
    plot_curves(axes[1], groups, 'loss', labels=[f'{k}' for k in order])
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Pérdida en train (MSE, log)', color=MUTED)
    title(axes[1], 'Pérdida de entrenamiento')
    save(fig, out, 'optimizadores.png')


def ej2_architecture(runs, out):
    g = by_variant(runs, 'arquitectura')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for variant, rs in g.items():
        best = [max(r['history']['val_acc']) for r in rs]
        t = np.mean([r['history']['time'][-1] for r in rs])
        label = str(rs[0]['config']['layers'][1:-1])
        deep = len(rs[0]['config']['layers']) > 3
        color = SERIES[1] if deep else SERIES[0]
        axes[0].errorbar(rs[0]['n_params'], np.mean(best), yerr=np.std(best), color=color, marker='o', markersize=8, capsize=3)
        axes[0].annotate(label, (rs[0]['n_params'], np.mean(best)), xytext=(6, -12), textcoords='offset points',
                         color=INK, fontsize=8)
        axes[1].scatter(t, np.mean(best), color=color, s=60)
        axes[1].annotate(label, (t, np.mean(best)), xytext=(6, -12), textcoords='offset points', color=INK, fontsize=8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Cantidad de parámetros (log)', color=MUTED)
    axes[0].set_ylabel('Accuracy máxima en validación', color=MUTED)
    title(axes[0], 'Ej. 2 · Arquitectura (capas ocultas) · azul: 1 capa, naranja: varias')
    axes[1].set_xlabel('Tiempo de entrenamiento, 40 épocas (s)', color=MUTED)
    title(axes[1], 'Costo computacional')
    for ax in axes:
        style(ax)
    save(fig, out, 'arquitectura.png')


def ej2_batch(runs, out):
    g = by_variant(runs, 'batch_size')
    order = sorted(g, key=lambda v: g[v][0]['config']['batch_size'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    groups = {v: g[v] for v in order}
    labels = []
    for v in order:
        bs = g[v][0]['config']['batch_size']
        labels.append('online (1)' if bs == 1 else f'batch ({bs})' if bs > 1000 else f'mini-batch {bs}')
    plot_curves(axes[0], groups, 'val_acc', labels)
    axes[0].set_ylabel('Accuracy en validación', color=MUTED)
    axes[0].set_ylim(0.3, 1)
    title(axes[0], 'Ej. 2 · Tamaño de batch')
    times = [np.mean([r['history']['time'][-1] for r in g[v]]) for v in order]
    axes[1].bar(labels, times, color=[SERIES[i] for i in range(len(order))], width=0.6)
    for i, t in enumerate(times):
        axes[1].annotate(f'{t:.0f}s', (i, t), xytext=(0, 3), textcoords='offset points', ha='center', color=INK, fontsize=9)
    axes[1].set_ylabel('Tiempo, 40 épocas (s)', color=MUTED)
    title(axes[1], 'Costo: online hace una actualización por imagen')
    style(axes[1])
    save(fig, out, 'batch_size.png')


def ej2_other(runs, out):
    groups = ['activacion_oculta', 'perdida', 'beta', 'inicializacion', 'combinacion']
    titles = ['Activación oculta', 'Salida + pérdida', 'β', 'Inicialización', 'Combinación de los mejores']
    fig, axes = plt.subplots(1, len(groups), figsize=(22, 4.5))
    for ax, group, t in zip(axes, groups, titles):
        g = by_variant(runs, group)
        names = list(g)
        m = [np.mean([max(r['history']['val_acc']) for r in g[v]]) for v in names]
        s = [np.std([max(r['history']['val_acc']) for r in g[v]]) for v in names]
        ax.barh(range(len(names)), m, xerr=s, color=SERIES[0], height=0.55, capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([n.replace('relu + softmax/CE, ', '') for n in names], color=INK, fontsize=8)
        for i, v in enumerate(m):
            ax.annotate(f'{v:.4f}', (v, i), xytext=(4, 6), textcoords='offset points', color=INK, fontsize=8)
        ax.set_xlim(0.94, 0.98)
        title(ax, t)
        style(ax)
    axes[0].set_xlabel('Accuracy máxima en validación', color=MUTED)
    save(fig, out, 'otros_hiperparametros.png')


def train_val_curves(record, out, name, label):
    h = record['history']
    x = np.arange(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (tr, va, ylabel) in zip(axes, [('loss', 'val_loss', 'Pérdida'), ('acc', 'val_acc', 'Accuracy')]):
        ax.plot(x, h[tr], color=SERIES[0], linewidth=2, label='entrenamiento')
        ax.plot(x, h[va], color=SERIES[1], linewidth=2, label='validación')
        if h.get('best_epoch'):
            ax.axvline(h['best_epoch'], color=MUTED, linestyle='--', linewidth=1)
            ax.annotate(f'mejor época {h["best_epoch"]}', (h['best_epoch'], ax.get_ylim()[1]), xytext=(4, -14),
                        textcoords='offset points', color=MUTED, fontsize=9)
        ax.set_xlabel('Época', color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.legend(frameon=False, labelcolor=INK)
        style(ax)
    title(axes[0], f'{label} · entrenamiento vs validación')
    save(fig, out, name)


def confusion_plot(report, out, name, label):
    cm = np.array(report['confusion'], float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicho', color=MUTED)
    ax.set_ylabel('Real', color=MUTED)
    ax.tick_params(colors=MUTED)
    title(ax, f'{label} · digits_test · accuracy {report["accuracy"]:.2%}')
    save(fig, out, name)


def per_class_table(report) -> pd.DataFrame:
    return pd.DataFrame({'digito': range(10), 'precision': report['precision'], 'recall': report['recall'],
                         'f1': report['f1'], 'soporte': report['soporte']})


# ---------- Ej. 3 ----------

def ej3_ablation(runs, out):
    g = by_variant(runs, 'ablacion') | by_variant(runs, 'extras')
    names = sorted(by_variant(runs, 'ablacion')) + list(by_variant(runs, 'extras'))
    val = [np.mean([max(r['history']['val_bacc']) for r in g[v]]) for v in names]
    val_s = [np.std([max(r['history']['val_bacc']) for r in g[v]]) for v in names]
    test = [np.mean([r['test_accuracy'] for r in g[v]]) for v in names]
    test_s = [np.std([r['test_accuracy'] for r in g[v]]) for v in names]
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(names))
    ax.barh(y - 0.2, val, 0.38, xerr=val_s, color=SERIES[0], capsize=2, label='validación (balanced accuracy)')
    ax.barh(y + 0.2, test, 0.38, xerr=test_s, color=SERIES[1], capsize=2, label='digits_test (accuracy, solo referencia)')
    for i, (v, vs, t, ts) in enumerate(zip(val, val_s, test, test_s)):
        ax.annotate(f'{v:.2%}', (v + vs, i - 0.2), xytext=(5, 0), textcoords='offset points', va='center', color=INK, fontsize=8)
        ax.annotate(f'{t:.2%}', (t + ts, i + 0.2), xytext=(5, 0), textcoords='offset points', va='center', color=INK, fontsize=8)
    ax.axvline(0.98, color=MUTED, linestyle='--', linewidth=1)
    ax.annotate('objetivo 98%', (0.98, -0.75), xytext=(4, 0), textcoords='offset points', va='center', color=MUTED, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.8, 1.0)
    ax.legend(frameon=False, labelcolor=INK, loc='upper left', bbox_to_anchor=(0, -0.08), ncol=2)
    title(ax, 'Ej. 3 · Ablación: cada paso suma una técnica al anterior (media ± desvío, 3 semillas)')
    style(ax)
    save(fig, out, 'ablacion.png')


def recall_comparison(rep2, rep3, out):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(10)
    ax.bar(x - 0.2, rep2['recall'], 0.38, color=SERIES[0], label=f'Ej. 2 (accuracy {rep2["accuracy"]:.1%})')
    ax.bar(x + 0.2, rep3['recall'], 0.38, color=SERIES[1], label=f'Ej. 3 (accuracy {rep3["accuracy"]:.1%})')
    ax.annotate('0 ochos en digits.csv\n→ recall 0', (8 - 0.2, 0.02), xytext=(-60, 60), textcoords='offset points',
                color=INK, fontsize=9, arrowprops={'arrowstyle': '->', 'color': MUTED})
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Dígito', color=MUTED)
    ax.set_ylabel('Recall en digits_test', color=MUTED)
    ax.legend(frameon=False, labelcolor=INK, loc='lower left')
    title(ax, 'Recall por clase: el salto del Ej. 2 al Ej. 3 es sobre todo el 8 y el 5')
    style(ax)
    save(fig, out, 'recall_por_clase_ej2_vs_ej3.png')


def robustness(models: dict, out) -> pd.DataFrame:
    X, y = digitos.load('digits_test')
    sigmas = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for name, model in models.items():
        for s in sigmas:
            accs = [np.mean(predict(model, digitos.add_noise(X, s, seed) if s else X) == y) for seed in range(3)]
            rows.append({'modelo': name, 'sigma': s, 'accuracy': np.mean(accs), 'desvio': np.std(accs)})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={'width_ratios': [2, 1.3]})
    ax = axes[0]
    for i, name in enumerate(models):
        d = df[df.modelo == name]
        ax.errorbar(d.sigma, d.accuracy, yerr=d.desvio, color=SERIES[i], marker='o', markersize=7, linewidth=2,
                    capsize=3, label=name)
    ax.set_xlabel('σ del ruido gaussiano', color=MUTED)
    ax.set_ylabel('Accuracy en digits_test con ruido', color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=INK)
    title(ax, 'Robustez al ruido')
    style(ax)
    # Ejemplo de cómo se ve cada σ
    img = X[np.flatnonzero(y == 3)[0]]
    strip = np.concatenate([digitos.add_noise(img[None], s, 0)[0].reshape(28, 28) for s in sigmas[1:]], axis=1)
    axes[1].imshow(strip, cmap='gray', vmin=0, vmax=1)
    axes[1].set_xticks([14 + 28 * i for i in range(len(sigmas) - 1)])
    axes[1].set_xticklabels([f'σ={s}' for s in sigmas[1:]], color=MUTED, fontsize=8)
    axes[1].set_yticks([])
    title(axes[1], 'Cómo se ve el ruido')
    save(fig, out, 'robustez_ruido.png')
    return df


def weights_mosaic(model: MLP, out, name, label, n=48):
    W = model.W[0]
    order = np.argsort(-np.linalg.norm(W, axis=0))[:n]
    cols = 12
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2))
    for ax, j in zip(axes.ravel(), order):
        w = W[:, j].reshape(28, 28)
        lim = np.abs(w).max()
        ax.imshow(w, cmap=DIVERGING, vmin=-lim, vmax=lim)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(str(j), fontsize=7, color=MUTED)
    for ax in axes.ravel()[len(order):]:
        ax.axis('off')
    fig.suptitle(f'{label} · pesos de la primera capa (las {n} neuronas de mayor norma) · rojo: excita, azul: inhibe',
                 color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, out, name)


def saliency_grid(model: MLP, out, name, label):
    X, y = digitos.load('digits_test')
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for d in range(10):
        x = X[np.flatnonzero(y == d)[0]][None]
        pred = int(model.predict(x)[0])
        grad = model.input_gradient(x, pred)[0]
        attr = (grad * x[0]).reshape(28, 28)
        lim = np.abs(attr).max() or 1
        glim = np.abs(grad).max() or 1
        axes[0, d].imshow(x[0].reshape(28, 28), cmap='gray_r')
        axes[0, d].set_title(f'real {d} · pred {pred}', fontsize=9, color=INK if pred == d else SERIES[7])
        axes[1, d].imshow(grad.reshape(28, 28), cmap=DIVERGING, vmin=-glim, vmax=glim)
        axes[2, d].imshow(attr, cmap=DIVERGING, vmin=-lim, vmax=lim)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([])
    axes[0, 0].set_ylabel('imagen', color=MUTED)
    axes[1, 0].set_ylabel('saliency ∂h/∂x', color=MUTED)
    axes[2, 0].set_ylabel('gradiente × entrada', color=MUTED)
    fig.suptitle(f'{label} · atribución hacia la clase predicha (rojo: empuja a favor, azul: en contra)',
                 color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, out, name)


def errors_grid(model: MLP, out, name, label, n=30):
    X, y = digitos.load('digits_test')
    pred = predict(model, X)
    wrong = np.flatnonzero(pred != y)[:n]
    cols = 10
    rows = int(np.ceil(len(wrong) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5))
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(X[i].reshape(28, 28), cmap='gray_r')
        ax.set_title(f'{y[i]} → {pred[i]}', fontsize=9, color=INK)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f'{label} · primeros {len(wrong)} errores en digits_test (real → predicho)', color=INK,
                 fontsize=11, x=0.01, ha='left')
    save(fig, out, name)


def dataset_overview(out):
    names = ['digits', 'more_digits', 'digits_test']
    counts = {n: digitos.class_counts(digitos.load(n)[1]) for n in names}
    Xa, ya = digitos.load('digits')
    Xb, yb = digitos.load('more_digits')
    _, yc, removed = digitos.combine_without_duplicates(Xa, ya, Xb, yb)
    counts['digits + more_digits (sin duplicados)'] = digitos.class_counts(yc)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(10)
    keys = list(counts)
    for i, k in enumerate(keys):
        ax.bar(x + (i - 1.5) * 0.2, counts[k], 0.19, color=SERIES[i], label=f'{k} ({counts[k].sum():,})')
    ax.annotate('no hay 8', (8 - 0.3, 0), xytext=(-40, 40), textcoords='offset points', color=INK, fontsize=9,
                arrowprops={'arrowstyle': '->', 'color': MUTED})
    ax.set_xticks(x)
    ax.set_xlabel('Dígito', color=MUTED)
    ax.set_ylabel('Imágenes', color=MUTED)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    title(ax, f'Distribución de clases · {removed:,} imágenes de more_digits ya estaban en digits')
    style(ax)
    save(fig, out, 'distribucion_clases.png')
    pd.DataFrame(counts, index=range(10)).rename_axis('digito').to_csv(out / 'distribucion_clases.csv')

    X, y = digitos.load('digits_test')
    fig, axes = plt.subplots(3, 10, figsize=(12, 4))
    for d in range(10):
        for r in range(3):
            axes[r, d].imshow(X[np.flatnonzero(y == d)[r]].reshape(28, 28), cmap='gray_r')
            axes[r, d].set_xticks([]), axes[r, d].set_yticks([])
    fig.suptitle('Ejemplos de digits_test', color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, out, 'ejemplos.png')


def load_final(name, key='final'):
    path = RESULTS / name / f'{key}.json'
    return json.loads(path.read_text()) if path.exists() else None


def main():
    base = RESULTS
    dataset_overview(base / 'datos')

    out2 = base / 'ej2'
    runs2 = load_runs('ej2')
    summary_table(runs2).to_csv(out2 / 'resumen_barridos.csv', index=False)
    ej2_learning_rate(runs2, out2).to_csv(out2 / 'tasa_aprendizaje.csv', index=False)
    ej2_optimizers(runs2, out2)
    ej2_architecture(runs2, out2)
    ej2_batch(runs2, out2)
    ej2_other(runs2, out2)
    final2 = load_final('ej2')
    if final2:
        train_val_curves(final2, out2, 'final_train_vs_val.png', 'Ej. 2 · modelo final')
        confusion_plot(final2['test'], out2, 'final_confusion_test.png', 'Ej. 2 · modelo final')
        per_class_table(final2['test']).to_csv(out2 / 'final_metricas_por_clase.csv', index=False)

    out3 = base / 'ej3'
    if (out3 / 'corridas.jsonl').exists():
        runs3 = load_runs('ej3')
        summary_table(runs3).to_csv(out3 / 'resumen_ablacion.csv', index=False)
        ej3_ablation(runs3, out3)
    final3 = load_final('ej3')
    if final3:
        train_val_curves(final3, out3, 'final_train_vs_val.png', 'Ej. 3 · modelo final')
        confusion_plot(final3['test'], out3, 'final_confusion_test.png', 'Ej. 3 · modelo final')
        per_class_table(final3['test']).to_csv(out3 / 'final_metricas_por_clase.csv', index=False)
        if final2:
            recall_comparison(final2['test'], final3['test'], out3)

    # Opcionales: robustez e interpretabilidad (cargan los modelos guardados)
    models = {}
    for label, file in [('Ej. 2 final', 'ej2_final.npz'), ('Ej. 3 final', 'ej3_final.npz'),
                        ('Ej. 3 entrenado con ruido', 'ej3_final_ruido.npz')]:
        if (MODELS / file).exists():
            models[label], _ = MLP.load(MODELS / file)
    out_opt = base / 'opcionales'
    if models:
        robustness(models, out_opt).to_csv(out_opt / 'robustez_ruido.csv', index=False)
    if 'Ej. 3 final' in models:
        m = models['Ej. 3 final']
        weights_mosaic(m, out_opt, 'pesos_primera_capa.png', 'Ej. 3 final')
        saliency_grid(m, out_opt, 'atribucion.png', 'Ej. 3 final')
        errors_grid(m, out_opt, 'errores_ej3.png', 'Ej. 3 final')
    print('Listo: gráficos y tablas en', base)


if __name__ == '__main__':
    main()
