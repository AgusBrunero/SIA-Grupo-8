"""Ej. 1.1: EDA y PCA del dataset de creadores.

Uso: python exploracion.py  →  resultados/exploracion/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import (DERIVADAS, cargar_creadores, estandarizar, tema_dominante,
                   variables_seleccionadas)
from graficos import DIVERGENTE, ESTILO_TEMA, GRILLA, TINTA, TINTA_SUAVE
from graficos import guardar as _guardar
from models.pca import PCA

SALIDA = Path(__file__).parent / "resultados" / "exploracion"


def guardar(fig, nombre):
    _guardar(fig, SALIDA / nombre)


def heatmap(ax, matriz, filas, columnas, anotar=False):
    im = ax.imshow(matriz, cmap=DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columnas)), columnas, rotation=90)
    ax.set_yticks(range(len(filas)), filas)
    ax.grid(False)
    if anotar:
        for (i, j), v in np.ndenumerate(matriz):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=TINTA)
    return im


def grafico_distribuciones(df, variables):
    columnas = 6
    filas = -(-len(variables) // columnas)
    fig, axes = plt.subplots(filas, columnas, figsize=(2.6 * columnas, 1.9 * filas))
    for ax, var in zip(axes.flat, variables):
        ax.hist(df[var], bins=12, color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
        ax.set_title(var, fontsize=8)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)
    fig.suptitle("Distribución de cada variable seleccionada (n = 50)")
    fig.tight_layout()
    guardar(fig, "distribuciones.png")


def grafico_correlaciones(df, variables):
    numericas = df.select_dtypes("number").columns
    corr = df[numericas].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(11, 10))
    im = heatmap(ax, corr, numericas, numericas)
    # Las descartadas (ver seleccion.py) quedan subrayadas en las etiquetas.

    def subrayar(nombre):
        return "".join(f"{caracter}\u0332" for caracter in nombre)

    ax.set_xticklabels([nombre if nombre in variables else subrayar(nombre) for nombre in numericas],
                       rotation=90)
    ax.set_yticklabels([nombre if nombre in variables else subrayar(nombre) for nombre in numericas])
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlación de Pearson")
    ax.set_title("Correlaciones entre todas las variables (subrayadas: descartadas)")
    guardar(fig, "correlaciones.png")


def grafico_scree(pca):
    n = len(pca.eigenvalues)
    pcs = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(pcs, 100 * pca.explained_variance_ratio, color="#2a78d6", width=0.7,
           label="varianza explicada")
    ax.plot(pcs, 100 * np.cumsum(pca.explained_variance_ratio), color="#eb6834", lw=2,
            marker="o", ms=4, label="acumulada")
    # Criterio de Kaiser: sobre datos estandarizados, autovalor > 1 ⇔ PC explica más que una variable.
    ax.axhline(100 / n, color=TINTA_SUAVE, lw=1, ls="--", label="Kaiser (autovalor = 1)")
    acumulada = 100 * np.cumsum(pca.explained_variance_ratio)
    for x in (1, 2, 3):
        ax.annotate(f"{acumulada[x - 1]:.0f} %", (x, acumulada[x - 1]), textcoords="offset points",
                    xytext=(-6, 6), ha="right", fontsize=8, color=TINTA_SUAVE)
    ax.set_axisbelow(True)
    ax.set_xticks(pcs)
    ax.set_xlabel("componente principal")
    ax.set_ylabel("% de varianza")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Scree plot")
    guardar(fig, "scree.png")


def grafico_cargas(cargas, n_pcs):
    orden = cargas["PC1"].sort_values(ascending=False).index
    matriz = cargas.loc[orden].iloc[:, :n_pcs].to_numpy()
    fig, ax = plt.subplots(figsize=(4.5, 9))
    im = heatmap(ax, matriz, orden, cargas.columns[:n_pcs], anotar=True)
    ax.tick_params(axis="x", rotation=0)
    fig.colorbar(im, ax=ax, shrink=0.5, label="carga (correlación variable–PC)")
    ax.set_title("Cargas de las primeras componentes")
    guardar(fig, "cargas.png")


def grafico_biplot(proyecciones, cargas, temas, pca):
    fig, ax = plt.subplots(figsize=(11, 9))
    for tema, (color, marcador) in ESTILO_TEMA.items():
        m = temas == tema
        ax.scatter(proyecciones.loc[m, "PC1"], proyecciones.loc[m, "PC2"], c=color,
                   marker=marcador, s=45, edgecolors="#fcfcfb", linewidths=1.5, label=tema,
                   zorder=3)
    for nombre, (x, y) in proyecciones[["PC1", "PC2"]].iterrows():
        ax.annotate(nombre, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6,
                    color=TINTA_SUAVE)
    # Flechas = cargas escaladas al rango de los puntos (el largo relativo es lo que importa).
    escala = 0.9 * np.abs(proyecciones[["PC1", "PC2"]]).max().min()
    puntas = cargas[["PC1", "PC2"]] * escala
    for x, y in puntas.to_numpy():
        ax.arrow(0, 0, x, y, color="#e34948", width=0.01, head_width=0.12,
                 length_includes_head=True, alpha=0.75, zorder=2)
    # Variables casi colineales comparten una sola etiqueta para que no se pisen.
    grupos = []
    for var, punta in puntas.iterrows():
        grupo = next((g for g in grupos if np.linalg.norm(g[0] - punta.to_numpy()) < 0.7), None)
        if grupo:
            grupo[1].append(var)
        else:
            grupos.append((punta.to_numpy(), [var]))
    for (x, y), nombres in grupos:
        ax.text(x * 1.1, y * 1.1, "\n".join(nombres), fontsize=6.5, color="#a32a2a",
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="#fcfcfb", ec="none", alpha=0.8))
    ax.axhline(0, color=GRILLA, lw=1)
    ax.axvline(0, color=GRILLA, lw=1)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio[0]:.1f} %)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio[1]:.1f} %)")
    ax.legend(title="tema dominante", frameon=False, loc="upper left")
    ax.set_title("Biplot PC1–PC2: creadores (puntos) y variables (flechas)")
    guardar(fig, "biplot.png")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    df = cargar_creadores()
    variables = variables_seleccionadas(df)
    temas = tema_dominante(df)

    # --- EDA ---
    descripcion = df.select_dtypes("number").describe().T
    descripcion["asimetria"] = df.select_dtypes("number").skew()
    descripcion.round(3).to_csv(SALIDA / "descripcion.csv")
    grafico_distribuciones(df, variables)
    grafico_correlaciones(df, variables)

    print(f"Creadores: {len(df)} · columnas numéricas: {df.select_dtypes('number').shape[1]}"
          f" · vacíos: {int(df.isna().sum().sum())}")
    print("Descartadas por ser derivadas:")
    for var, formula in DERIVADAS.items():
        print(f"  {var:28s} = {formula}")
    print(f"Variables seleccionadas ({len(variables)}, ver seleccion.py): {variables}")
    print("Tema dominante:", temas.value_counts().to_dict())

    # --- PCA sobre las variables seleccionadas, estandarizadas ---
    Z, _, _ = estandarizar(df[variables])
    pca = PCA().fit(Z)
    pcs = [f"PC{i + 1}" for i in range(len(variables))]
    cargas = pd.DataFrame(pca.loadings(), index=variables, columns=pcs)
    proyecciones = pd.DataFrame(pca.transform(Z), index=df.index, columns=pcs)
    varianza = pd.DataFrame({
        "autovalor": pca.eigenvalues,
        "varianza_explicada": pca.explained_variance_ratio,
        "acumulada": np.cumsum(pca.explained_variance_ratio),
    }, index=pcs)

    varianza.round(4).to_csv(SALIDA / "pca_varianza.csv")
    cargas.round(4).to_csv(SALIDA / "pca_cargas.csv")
    proyecciones.assign(tema_dominante=temas).round(4).to_csv(SALIDA / "pca_proyecciones.csv")

    n_kaiser = int((pca.eigenvalues > 1).sum())
    grafico_scree(pca)
    grafico_cargas(cargas, max(n_kaiser, 4))
    grafico_biplot(proyecciones, cargas, temas, pca)

    print("\nVarianza explicada:")
    print(varianza.head(8).round(3).to_string())
    print(f"Componentes con autovalor > 1 (Kaiser): {n_kaiser}")
    for pc in pcs[:3]:
        orden = cargas[pc].sort_values()
        print(f"\n{pc} · cargas más negativas: {orden.head(4).round(2).to_dict()}")
        print(f"{pc} · cargas más positivas: {orden.tail(4)[::-1].round(2).to_dict()}")
    corr = np.corrcoef(proyecciones["PC1"], df["followers"])[0, 1]
    print(f"\nCorrelación PC1 vs followers: {corr:.3f}")
    print(f"Salidas en {SALIDA.relative_to(Path.cwd()) if SALIDA.is_relative_to(Path.cwd()) else SALIDA}")


if __name__ == "__main__":
    main()
