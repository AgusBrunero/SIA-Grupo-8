"""Ej. 1.2: agrupamiento de creadores con una red de Kohonen.

Uso: python agrupamiento.py [barridos | final]   (sin argumento corre ambos)
  barridos → resultados/kohonen/barridos.csv y barridos.png (un hiperparámetro por vez, 5 semillas)
  final    → modelo final: hit map, matriz U, temas, component planes, grilla sobre el PCA y CSVs
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import RegularPolygon

from datos import cargar_creadores, estandarizar, tema_dominante, variables_seleccionadas
from graficos import (AZUL, DIVERGENTE, ESTILO_TEMA, FONDO, GRILLA, NARANJA, SECUENCIAL,
                      TINTA, TINTA_SUAVE, guardar)
from models.kohonen import Kohonen, linear_decay
from models.pca import PCA

SALIDA = Path(__file__).parent / "resultados" / "kohonen"
SEMILLAS = range(5)

# η decae linealmente de eta0 a ETA_FINAL y el radio de radio0 a radio_final
# (1 = al final se siguen moviendo las 6 adyacentes; 0 = solo la BMU).
ETA_FINAL = 0.01
BASE = {"k": 5, "eta0": 0.5, "radio0": 3, "radio_final": 1, "epocas": 500, "init": "samples"}
VARIACIONES = {
    "k": [3, 4, 5, 6, 7, 8],
    "eta0": [0.05, 0.1, 0.5, 0.9],
    "radio0": [1, 2, 3, 5],
    "radio_final": [1, 0],
    "epocas": [10, 50, 200, 500, 1000],
    "init": ["samples", "random"],
}
# Elegida a partir de barridos.png (ver la justificación en el README).
FINAL = BASE

PLANOS = ["average_reach", "non_follower_reach_pct", "shares_per_1000", "saves_per_1000",
          "profile_visits_per_1000", "content_tutorial_pct", "content_showcase_pct",
          "argentina_audience_pct", "sponsored_posts_pct", "topic_football_pct",
          "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct"]


def entrenar(Z, config, semilla):
    som = Kohonen(Z.shape[1], config["k"], linear_decay(config["eta0"], ETA_FINAL),
                  linear_decay(config["radio0"], config["radio_final"]), init=config["init"], seed=semilla)
    som.train(Z, config["epocas"])
    som.process(Z)
    return som


def cargar():
    df = cargar_creadores()
    variables = variables_seleccionadas(df)
    Z, _, _ = estandarizar(df[variables])
    return df, variables, Z


# --------
# Barridos
# --------

def barridos(Z):
    filas = []
    for parametro, valores in VARIACIONES.items():
        for valor in valores:
            config = {**BASE, parametro: valor}
            for semilla in SEMILLAS:
                inicio = time.perf_counter()
                som = entrenar(Z, config, semilla)
                filas.append({"parametro": parametro, "valor": valor, "semilla": semilla,
                              **config, "error_cuantizacion": som.get_quantization_error(),
                              "error_topografico": som.get_topographic_error(),
                              "neuronas_muertas": som.get_dead_neurons(),
                              "segundos": time.perf_counter() - inicio})
            ultimas = pd.DataFrame(filas[-len(SEMILLAS):])
            print(f"{parametro}={valor!s:8s} EC={ultimas.error_cuantizacion.mean():.3f} "
                  f"ET={ultimas.error_topografico.mean():.3f} "
                  f"muertas={ultimas.neuronas_muertas.mean():.1f}")
    resultados = pd.DataFrame(filas)
    resultados.to_csv(SALIDA / "barridos.csv", index=False)
    return resultados


def grafico_barridos(resultados):
    metricas = {"error_cuantizacion": "error de cuantización",
                "error_topografico": "error topográfico",
                "neuronas_muertas": "neuronas muertas"}
    fig, axes = plt.subplots(len(metricas), len(VARIACIONES),
                             figsize=(3.2 * len(VARIACIONES), 2.4 * len(metricas)),
                             sharey="row")
    for j, parametro in enumerate(VARIACIONES):
        grupo = resultados[resultados.parametro == parametro]
        resumen = grupo.groupby("valor", sort=False)[list(metricas)].agg(["mean", "std"])
        x = np.arange(len(resumen))
        for i, metrica in enumerate(metricas):
            ax = axes[i, j]
            media, desvio = resumen[(metrica, "mean")], resumen[(metrica, "std")]
            base = [str(v) == str(BASE[parametro]) for v in resumen.index]
            ax.bar(x, media, yerr=desvio, width=0.6, capsize=3, ecolor=TINTA_SUAVE,
                   color=[NARANJA if b else AZUL for b in base])
            ax.set_xticks(x, [str(v) for v in resumen.index])
            ax.grid(axis="x", visible=False)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel(metricas[metrica])
            if i == len(metricas) - 1:
                ax.set_xlabel(parametro)
    fig.suptitle(f"Barridos de a un hiperparámetro (media ± desvío, {len(SEMILLAS)} semillas). "
                 f"En naranja, el valor de la configuración base {BASE}", fontsize=9)
    fig.tight_layout()
    guardar(fig, SALIDA / "barridos.png")


# ------------
# Modelo final
# ------------

def grilla_hex(ax, som, valores, cmap, vmin=None, vmax=None, textos=None, colores=None,
               fontsize=6):
    """Dibuja la grilla hexagonal (fila 0 arriba). `colores` pisa al colormap si se pasa."""
    valores = np.asarray(valores, dtype=float).ravel()
    vmin = np.nanmin(valores) if vmin is None else vmin
    vmax = np.nanmax(valores) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    for i, (x, y) in enumerate(som.positions):
        relleno = colores[i] if colores is not None else cmap(norm(valores[i]))
        ax.add_patch(RegularPolygon((x, -y), 6, radius=1 / np.sqrt(3), facecolor=relleno,
                                    edgecolor=FONDO, linewidth=2))
        if textos is not None and textos[i]:
            oscuro = colores is None and norm(valores[i]) > 0.6 and cmap is not DIVERGENTE
            ax.text(x, -y, textos[i], ha="center", va="center", fontsize=fontsize,
                    color=FONDO if oscuro else TINTA, linespacing=1.1)
    ax.set_xlim(-0.7, som.k + 0.2)
    ax.set_ylim(-(som.k - 1) * np.sqrt(3) / 2 - 0.7, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def grafico_hit_map(som, nombres):
    textos = ["\n".join(nombres[som.bmus == i]) for i in range(som.k ** 2)]
    fig, ax = plt.subplots(figsize=(2.3 * som.k, 2.1 * som.k))
    sm = grilla_hex(ax, som, som.get_hit_map(), SECUENCIAL, vmin=0, textos=textos, fontsize=5.5)
    fig.colorbar(sm, ax=ax, shrink=0.6, label="creadores en la neurona")
    ax.set_title("Hit map: creadores asignados a cada neurona")
    guardar(fig, SALIDA / "hit_map.png")


def grafico_u_matrix(som):
    u = som.get_u_matrix()
    fig, ax = plt.subplots(figsize=(1.5 * som.k + 1, 1.3 * som.k))
    sm = grilla_hex(ax, som, u, SECUENCIAL, textos=[f"{v:.1f}" for v in u.ravel()], fontsize=8)
    fig.colorbar(sm, ax=ax, shrink=0.7, label="distancia media a las vecinas")
    ax.set_title("Matriz U (oscuro = frontera entre grupos)")
    guardar(fig, SALIDA / "u_matrix.png")


def grafico_temas(som, temas):
    colores, textos = [], []
    for i in range(som.k ** 2):
        conteo = temas[som.bmus == i].value_counts()
        if conteo.empty:
            colores.append("#f0efec")
            textos.append("")
            continue
        colores.append(ESTILO_TEMA[conteo.index[0]][0] if conteo.iloc[0] > conteo.sum() / 2
                       else "#c3c2b7")
        textos.append("\n".join(f"{t} {n}" for t, n in conteo.items()))
    fig, ax = plt.subplots(figsize=(1.6 * som.k + 1, 1.4 * som.k))
    grilla_hex(ax, som, np.zeros(som.k ** 2), None, textos=textos, colores=colores, fontsize=6.5)
    for tema, (color, marcador) in ESTILO_TEMA.items():
        ax.scatter([], [], c=color, marker="h", s=120, label=f"mayoría {tema}")
    ax.scatter([], [], c="#c3c2b7", marker="h", s=120, label="sin mayoría")
    ax.scatter([], [], c="#f0efec", marker="h", s=120, edgecolors=GRILLA,
               label="sin creadores (muerta)")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title("Tema dominante de los creadores de cada neurona")
    guardar(fig, SALIDA / "temas.png")


def grafico_planos(som, variables, pca):
    """Component planes (pesos en desvíos estándar) + proyección de cada neurona en PC1 y PC2."""
    planos = {v: som.get_component_matrix(variables.index(v)) for v in PLANOS}
    proy = pca.transform(som.weights, 2)
    planos["PC1 (proyección)"] = proy[:, 0]
    planos["PC2 (proyección)"] = proy[:, 1]
    columnas = 5
    filas = -(-len(planos) // columnas)
    fig, axes = plt.subplots(filas, columnas, figsize=(3 * columnas, 2.8 * filas))
    for ax, (nombre, valores) in zip(axes.flat, planos.items()):
        limite = np.abs(valores).max()
        sm = grilla_hex(ax, som, valores, DIVERGENTE, vmin=-limite, vmax=limite)
        fig.colorbar(sm, ax=ax, shrink=0.8)
        ax.set_title(nombre, fontsize=8)
    for ax in axes.flat[len(planos):]:
        ax.set_visible(False)
    fig.suptitle("Component planes (pesos estandarizados: rojo = alto, azul = bajo)")
    fig.tight_layout()
    guardar(fig, SALIDA / "component_planes.png")


def grafico_grilla_en_pca(som, Z, temas, pca):
    """La grilla entrenada vista en el plano PC1–PC2: cómo se 'estira' sobre los datos."""
    datos = pca.transform(Z, 2)
    neuronas = pca.transform(som.weights, 2)
    fig, ax = plt.subplots(figsize=(10, 8))
    vecinas = np.argwhere(np.triu(np.isclose(som.grid_dist, 1)))
    for a, b in vecinas:
        ax.plot(*neuronas[[a, b]].T, color="#c3c2b7", lw=1, zorder=1)
    ax.scatter(*neuronas.T, s=25, c=TINTA_SUAVE, zorder=2, label="neurona")
    for tema, (color, marcador) in ESTILO_TEMA.items():
        m = (temas == tema).to_numpy()
        ax.scatter(*datos[m].T, c=color, marker=marcador, s=45, edgecolors=FONDO,
                   linewidths=1.5, zorder=3, label=tema)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio[1]:.1f} %)")
    ax.legend(frameon=False)
    ax.set_title("Grilla de Kohonen proyectada en PC1–PC2 (líneas = neuronas vecinas)")
    guardar(fig, SALIDA / "grilla_en_pca.png")


def perfiles(som, df, variables, Z, temas, pca):
    """Una fila por neurona: creadores, tema y las variables más altas / bajas de su peso."""
    filas = []
    proy = pca.transform(som.weights, 2)
    for i in range(som.k ** 2):
        miembros = df.index[som.bmus == i]
        peso = pd.Series(som.weights[i], index=variables).sort_values()
        filas.append({
            "fila": i // som.k, "columna": i % som.k, "creadores": len(miembros),
            "nombres": " ".join(miembros),
            "temas": temas[som.bmus == i].value_counts().to_dict() or "",
            "PC1": round(proy[i, 0], 2), "PC2": round(proy[i, 1], 2),
            "altas": ", ".join(f"{v} {z:+.1f}" for v, z in peso.tail(4)[::-1].items()),
            "bajas": ", ".join(f"{v} {z:+.1f}" for v, z in peso.head(4).items()),
        })
    return pd.DataFrame(filas)


def final(df, variables, Z):
    # Entre las semillas de la configuración final se queda la de menor error de cuantización.
    modelos = [entrenar(Z, FINAL, s) for s in SEMILLAS]
    som = min(modelos, key=lambda m: m.get_quantization_error())
    temas = tema_dominante(df)
    pca = PCA().fit(Z)
    nombres = df.index.to_numpy()

    grafico_hit_map(som, nombres)
    grafico_u_matrix(som)
    grafico_temas(som, temas)
    grafico_planos(som, variables, pca)
    grafico_grilla_en_pca(som, Z, temas, pca)

    proy = pca.transform(Z, 2)
    pd.DataFrame({"fila": som.clusters[:, 0], "columna": som.clusters[:, 1],
                  "tema_dominante": temas, "PC1": proy[:, 0].round(3),
                  "PC2": proy[:, 1].round(3)}, index=df.index).to_csv(SALIDA / "asignaciones.csv")
    tabla = perfiles(som, df, variables, Z, temas, pca)
    tabla.to_csv(SALIDA / "perfiles_neuronas.csv", index=False)
    np.savez(SALIDA / "modelo_final.npz", weights=som.weights, variables=variables, **FINAL)

    print(f"\nModelo final {FINAL}, semilla {SEMILLAS[modelos.index(som)]}")
    print(f"EC={som.get_quantization_error():.3f} ET={som.get_topographic_error():.3f} "
          f"muertas={som.get_dead_neurons()}")
    print(np.array2string(som.get_hit_map()))
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(tabla[["fila", "columna", "creadores", "temas", "PC1", "PC2", "altas"]].to_string())


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"
    df, variables, Z = cargar()
    if modo in ("barridos", "todo"):
        grafico_barridos(barridos(Z))
    if modo in ("final", "todo"):
        final(df, variables, Z)


if __name__ == "__main__":
    main()
