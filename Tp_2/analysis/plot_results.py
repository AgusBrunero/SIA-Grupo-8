"""Figuras para la presentación (Step 11).

Cada figura es **un solo gráfico, sin texto de mobiliario**: sin título, sin bloque
de configuración, con fondo transparente. La idea es que entren en una diapositiva
con el estilo de la presentación, no con el nuestro — el título lo pone la slide y
la configuración está anotada en el `informe.md` y el `config.md` de cada carpeta.

Se conservan sólo los elementos sin los cuales el gráfico no se puede leer: ejes
rotulados, leyenda y grilla tenue.

Este módulo expone las funciones de dibujo; quien arma las carpetas del informe es
`build_report.py`, que es el comando que hay que correr:

    python analysis/build_report.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent

#: paleta propia, consistente entre todas las figuras de un mismo eje
PALETTE = ["#BC002D", "#2C6E8F", "#C98A00", "#4B7B4A", "#7A4A8C", "#B4553A", "#5A6470",
           "#2F5D8C", "#9E7B2F", "#6E4A55"]

THEMES = {
    "light": {"ink": "#241E20", "grid": "#C9C1BE"},
    "dark": {"ink": "#F2EDEB", "grid": "#5A5457"},
}

METRICS = ("best_global_fitness", "best_fitness", "mean_fitness", "diversity",
           "evaluations", "elapsed")


def use_theme(name: str = "light") -> None:
    """Fondo SIEMPRE transparente; el tema sólo cambia el color de la tinta, para
    que las figuras sirvan tanto en slides claras como oscuras."""
    theme = THEMES[name]
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": theme["ink"], "axes.labelcolor": theme["ink"],
        "axes.edgecolor": theme["grid"], "grid.color": theme["grid"],
        "xtick.color": theme["ink"], "ytick.color": theme["ink"],
        "axes.labelsize": 13, "font.size": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200,
    })


def load(path: Path) -> dict:
    """CSV -> {target: {variante: {métrica: matriz (semillas x generaciones)}}}"""
    raw: dict[tuple, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            raw[(row.get("target", "único"), row["variant"], int(row["seed"]))].append(row)

    nested: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (target, variant, _seed), rows in raw.items():
        rows.sort(key=lambda r: int(r["generation"]))
        for metric in METRICS:
            if metric in rows[0]:
                nested[target][variant][metric].append([float(r[metric]) for r in rows])

    return {
        target: {v: {m: np.array(runs) for m, runs in ms.items()} for v, ms in variants.items()}
        for target, variants in nested.items()
    }


def colors_for(data: dict) -> dict[str, str]:
    """Mismo color para la misma variante en todas las figuras del eje."""
    variants = sorted({v for target in data.values() for v in target})
    return {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(variants)}


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)
    return out


def _band(ax, x, values, color, label=None, **kwargs):
    mean, std = values.mean(axis=0), values.std(axis=0)
    ax.plot(x, mean, color=color, label=label, linewidth=2.2, **kwargs)
    if values.shape[0] > 1:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    return mean


def convergencia(data_target: dict, colors: dict, out: Path) -> Path:
    """Mejor fitness acumulado por generación. Media entre semillas, banda ±σ."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, metrics in sorted(data_target.items()):
        best = metrics.get("best_global_fitness", metrics["best_fitness"])
        _band(ax, np.arange(1, best.shape[1] + 1), best, colors[variant], variant)
    ax.set_xlabel("generación"); ax.set_ylabel("mejor fitness")
    ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right",
                                   ncol=2 if len(data_target) > 5 else 1)
    return _save(fig, out)


def diversidad(data_target: dict, colors: dict, out: Path) -> Path:
    """Diversidad genética (desvío promedio por gen), en escala log."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, metrics in sorted(data_target.items()):
        div = metrics["diversity"]
        _band(ax, np.arange(1, div.shape[1] + 1), div, colors[variant], variant)
    ax.set_xlabel("generación"); ax.set_ylabel("diversidad genética")
    ax.set_yscale("log"); ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2 if len(data_target) > 5 else 1)
    return _save(fig, out)


def boxplot_final(data_target: dict, colors: dict, out: Path) -> Path:
    """Fitness final por variante, con los puntos crudos de cada semilla encima.

    Es el gráfico que pide la cátedra y el que muestra si un ganador es real: si
    las cajas se solapan, la diferencia está dentro del ruido.
    """
    items = sorted(data_target.items())
    finals = [m.get("best_global_fitness", m["best_fitness"])[:, -1] for _, m in items]
    labels = [v for v, _ in items]
    ink = plt.rcParams["text.color"]

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(labels)), 5))
    box = ax.boxplot(finals, tick_labels=labels, patch_artist=True, widths=0.55,
                     medianprops={"color": ink, "linewidth": 1.8})
    for patch, variant in zip(box["boxes"], labels):
        patch.set_facecolor(colors[variant]); patch.set_alpha(0.35)
        patch.set_edgecolor(colors[variant])
    for element in ("whiskers", "caps"):
        for line in box[element]:
            line.set_color(ink)
    for i, values in enumerate(finals, start=1):
        ax.plot(np.full(len(values), i), values, "o", color=ink, markersize=4, alpha=0.7)

    ax.set_ylabel("fitness final"); ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return _save(fig, out)


def best_vs_mean(data_target: dict, colors: dict, out: Path) -> Path:
    """Mejor y promedio de la población en el mismo panel.

    Cuando el promedio alcanza al mejor, la población convergió: todos los
    individuos son casi el mismo. Es la evidencia de convergencia prematura.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, metrics in sorted(data_target.items()):
        best = metrics.get("best_global_fitness", metrics["best_fitness"])
        x = np.arange(1, best.shape[1] + 1)
        _band(ax, x, best, colors[variant], variant)
        ax.plot(x, metrics["mean_fitness"].mean(axis=0), color=colors[variant],
                linewidth=1.2, linestyle="--", alpha=0.8)
    ax.set_xlabel("generación"); ax.set_ylabel("fitness")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right", ncol=2 if len(data_target) > 5 else 1,
              title="sólido: mejor · guionado: promedio")
    ax.get_legend().get_title().set_fontsize(10)
    return _save(fig, out)


def fitness_vs_evaluaciones(data_target: dict, colors: dict, out: Path) -> Path:
    """Mejor fitness contra evaluaciones de fitness, no contra generación.

    Una generación con K=2N cuesta el doble que una con K=N: compararlas por
    generación regala ventaja a la más cara. Este es el eje justo.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, metrics in sorted(data_target.items()):
        best = metrics.get("best_global_fitness", metrics["best_fitness"])
        ax.plot(metrics["evaluations"].mean(axis=0), best.mean(axis=0),
                color=colors[variant], label=variant, linewidth=2.2)
    ax.set_xlabel("evaluaciones de fitness"); ax.set_ylabel("mejor fitness")
    ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right",
                                   ncol=2 if len(data_target) > 5 else 1)
    return _save(fig, out)


def escalabilidad(data_target: dict, colors: dict, out: Path) -> Path:
    """Scatter: calidad alcanzada contra costo, con el tamaño del problema.

    Sólo tiene sentido en ejes donde las variantes cuestan distinto (triángulos,
    población). El eje X es el tiempo real de la corrida, que es lo que el usuario
    paga; las evaluaciones no alcanzan porque acá el costo extra está ADENTRO de
    cada evaluación, no en la cantidad.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variante, metrics in sorted(data_target.items()):
        best = metrics.get("best_global_fitness", metrics["best_fitness"])
        if "elapsed" not in metrics:
            continue
        segundos = metrics["elapsed"][:, -1]
        finales = best[:, -1]
        ax.scatter(segundos.mean(), finales.mean(), s=140, color=colors[variante],
                   alpha=0.85, edgecolor="none", label=variante, zorder=3)
        ax.errorbar(segundos.mean(), finales.mean(), yerr=finales.std(),
                    xerr=segundos.std(), color=colors[variante], alpha=0.4,
                    capsize=3, linewidth=1, zorder=2)
    ax.set_xlabel("segundos por corrida")
    ax.set_ylabel("fitness final")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10, ncol=2 if len(data_target) > 5 else 1)
    return _save(fig, out)


#: qué figuras se generan por cada eje y target
FIGURES = {
    "convergencia": convergencia,
    "diversidad": diversidad,
    "boxplot_final": boxplot_final,
    "mejor_vs_promedio": best_vs_mean,
    "fitness_vs_evaluaciones": fitness_vs_evaluaciones,
}

#: figuras que sólo tienen sentido donde las variantes cuestan distinto
FIGURAS_EXTRA = {
    "triangulos": {"escalabilidad": escalabilidad},
    "poblacion": {"escalabilidad": escalabilidad},
    "supervivencia": {"escalabilidad": escalabilidad},
}
