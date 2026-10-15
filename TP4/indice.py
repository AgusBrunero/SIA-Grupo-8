"""Ej. 1.3: índice de creadores con la regla de Oja.

Uso: python indice.py [convergencia | final]   (sin argumento corre ambos)
  convergencia → resultados/oja/convergencia.csv y .png: ‖w‖ y ángulo con PC1 por época según
                 η (constante o decreciente) y norma inicial, 5 semillas
  final        → índice final: pesos de Oja vs autovector PC1, ranking, índice vs followers y
                 por tema, CSVs
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datos import cargar_creadores, estandarizar, tema_dominante, variables_seleccionadas
from graficos import AZUL, ESTILO_TEMA, FONDO, GRILLA, NARANJA, TINTA_SUAVE, guardar
from models.oja import Oja
from models.pca import PCA

SALIDA = Path(__file__).parent / "resultados" / "oja"
SEMILLAS = range(5)
EPOCAS = 100


def eta_constante(eta):
    return lambda t, T: eta


def eta_inverso(eta0, tau):
    """η(t) = η0 / (1 + t/τ). Con τ = n (una época) a las 100 épocas vale η0/101."""
    return lambda t, T: eta0 / (1 + t / tau)


N = 50  # creadores: τ del decaimiento = una época
ETAS = {
    "η = 0,0001": eta_constante(1e-4),
    "η = 0,001": eta_constante(1e-3),
    "η = 0,01": eta_constante(1e-2),
    "η = 0,02": eta_constante(2e-2),
    "η = 0,05": eta_constante(5e-2),
    "η = 0,01 / (1 + t/n)": eta_inverso(1e-2, N),
}
NORMAS_INICIALES = [0.01, 1, 3, 10]  # con η = 0,001
# Elegida a partir de convergencia.png (ver resultados/oja/conclusiones.md).
FINAL = {"eta": "η = 0,01 / (1 + t/n)", "init_norm": 1.0, "epocas": EPOCAS, "semilla": 0}


def fijar_signo(w):
    """Mismo criterio que `PCA`: el peso de mayor módulo queda positivo. Así el signo de Oja
    se fija sin mirar a PC1 y la comparación es honesta."""
    return w * np.sign(w[np.abs(w).argmax()])


def angulo(w, pc1):
    """Ángulo (grados) entre w y el autovector PC1, ignorando el signo."""
    cos = np.clip(np.abs(w @ pc1) / np.linalg.norm(w), 0, 1)
    return np.degrees(np.arccos(cos))


def entrenar(Z, eta_f, init_norm, epocas, semilla):
    return Oja(Z.shape[1], eta_f, init_norm=init_norm, seed=semilla).train(Z, epocas)


def cargar():
    df = cargar_creadores()
    variables = variables_seleccionadas(df)
    Z, _, _ = estandarizar(df[variables])
    return df, variables, Z


# ------------
# Convergencia
# ------------

def convergencia(Z, pc1):
    """Una fila por (experimento, semilla, época) con ‖w‖ y ángulo con PC1."""
    corridas = [("eta", nombre, f, 1.0) for nombre, f in ETAS.items()]
    corridas += [("init_norm", f"‖w₀‖ = {n:g}", ETAS["η = 0,001"], n) for n in NORMAS_INICIALES]
    filas = []
    for grupo, nombre, eta_f, init_norm in corridas:
        for semilla in SEMILLAS:
            oja = entrenar(Z, eta_f, init_norm, EPOCAS, semilla)
            normas = np.linalg.norm(oja.history, axis=1)
            angulos = np.array([angulo(w, pc1) if np.isfinite(w).all() else np.nan
                                for w in oja.history])
            for e in range(EPOCAS):
                filas.append({"grupo": grupo, "experimento": nombre, "semilla": semilla,
                              "epoca": e + 1, "norma": normas[e], "angulo": angulos[e],
                              "divergio": oja.diverged})
    resultados = pd.DataFrame(filas)
    resultados.round(5).to_csv(SALIDA / "convergencia.csv", index=False)
    return resultados


def resumen_convergencia(resultados):
    """Por experimento: norma y ángulo finales (media entre semillas) y época en la que el
    ángulo baja de 1° por primera vez."""
    filas = []
    for (grupo, nombre), g in resultados.groupby(["grupo", "experimento"], sort=False):
        finales = g[g.epoca == EPOCAS]
        bajo_1 = g[g.angulo < 1].groupby("semilla").epoca.min()
        filas.append({
            "grupo": grupo, "experimento": nombre,
            "divergio": int(g.groupby("semilla").divergio.any().sum()),
            "norma_final": finales.norma.mean(), "angulo_final": finales.angulo.mean(),
            "angulo_final_max": finales.angulo.max(),
            "epocas_hasta_1_grado": bajo_1.mean() if len(bajo_1) == len(SEMILLAS) else np.nan,
        })
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(SALIDA / "convergencia_resumen.csv", index=False)
    return tabla


def grafico_convergencia(resultados):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    colores = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    for j, grupo in enumerate(["eta", "init_norm"]):
        datos = resultados[resultados.grupo == grupo]
        nombres = datos.experimento.unique()
        for color, nombre in zip(colores, nombres):
            for semilla in SEMILLAS:
                g = datos[(datos.experimento == nombre) & (datos.semilla == semilla)]
                etiqueta = nombre if semilla == 0 else None
                axes[0, j].plot(g.epoca, g.norma, color=color, lw=1, alpha=0.8, label=etiqueta)
                axes[1, j].plot(g.epoca, g.angulo, color=color, lw=1, alpha=0.8)
        axes[0, j].axhline(1, color=TINTA_SUAVE, lw=0.8, ls="--")
        axes[1, j].axhline(1, color=TINTA_SUAVE, lw=0.8, ls="--")
        axes[0, j].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in (axes[1, 0], axes[0, 1], axes[1, 1]):
        ax.set_yscale("log")
    axes[0, 0].set_ylabel("‖w‖ (línea = 1)")
    axes[1, 0].set_ylabel("ángulo con PC1 en grados (log; línea = 1°)")
    axes[1, 0].set_xlabel("época")
    axes[1, 1].set_xlabel("época")
    axes[0, 0].set_title("Según η (‖w₀‖ = 1). η = 0,05 diverge en las primeras épocas", fontsize=9)
    axes[0, 1].set_title("Según la norma inicial (η = 0,001)", fontsize=9)
    fig.suptitle(f"Convergencia de la regla de Oja ({len(SEMILLAS)} semillas por curva)")
    fig.tight_layout()
    guardar(fig, SALIDA / "convergencia.png")


# ------------
# Índice final
# ------------

def spearman(a, b):
    return np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]


def grafico_pesos(pesos):
    orden = pesos.sort_values("PC1").index
    y = np.arange(len(orden))
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(y + 0.2, pesos.loc[orden, "oja"], height=0.4, color=AZUL, label="Oja (w)")
    ax.barh(y - 0.2, pesos.loc[orden, "PC1"], height=0.4, color=NARANJA,
            label="PCA (autovector 1)")
    ax.set_yticks(y, orden)
    ax.axvline(0, color=GRILLA, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.set_xlabel("peso (vector unitario)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Pesos de Oja contra el autovector de PC1")
    guardar(fig, SALIDA / "pesos_vs_pc1.png")


def grafico_indice(tabla):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for tema, (color, marcador) in ESTILO_TEMA.items():
        g = tabla[tabla.tema_dominante == tema]
        estilo = dict(c=color, marker=marcador, s=45, edgecolors=FONDO, linewidths=1.5,
                      label=tema, zorder=3)
        axes[0].scatter(g.PC1, g.indice, **estilo)
        axes[1].scatter(g.followers, g.indice, **estilo)
        axes[2].scatter(np.full(len(g), list(ESTILO_TEMA).index(tema))
                        + np.random.default_rng(0).uniform(-0.18, 0.18, len(g)), g.indice, **estilo)
    limites = [tabla.indice.min() - 0.5, tabla.indice.max() + 0.5]
    axes[0].plot(limites, limites, color=GRILLA, lw=1, zorder=1)
    axes[0].set_xlabel("proyección sobre PC1 (PCA)")
    axes[0].set_ylabel("índice de Oja")
    axes[0].set_title("Índice vs PC1 (línea = identidad)", fontsize=9)
    axes[0].legend(frameon=False, title="tema dominante")

    for nombre, (x, y) in tabla[["followers", "indice"]].iterrows():
        axes[1].annotate(nombre, (x, y), xytext=(3, 2), textcoords="offset points",
                         fontsize=5.5, color=TINTA_SUAVE)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("followers (log)")
    axes[1].set_title("Índice vs followers (ρ de Spearman = "
                      f"{spearman(tabla.indice, tabla.followers):.2f})", fontsize=9)

    umbral = tabla.indice.nlargest(10).min()
    axes[2].axhline(umbral, color=TINTA_SUAVE, lw=0.8, ls="--")
    axes[2].text(3.45, umbral, "top 10", fontsize=7, color=TINTA_SUAVE, va="bottom", ha="right")
    axes[2].set_xticks(range(len(ESTILO_TEMA)), list(ESTILO_TEMA))
    axes[2].grid(axis="x", visible=False)
    axes[2].set_title("Índice por tema dominante", fontsize=9)
    for ax in axes[1:]:
        ax.set_ylabel("índice de Oja")
    fig.suptitle("El índice de Oja: igual a PC1, inverso al tamaño y ciego al tema")
    fig.tight_layout()
    guardar(fig, SALIDA / "indice.png")


def final(df, variables, Z):
    pca = PCA().fit(Z)
    pc1 = pca.components[:, 0]
    oja = entrenar(Z, ETAS[FINAL["eta"]], FINAL["init_norm"], FINAL["epocas"], FINAL["semilla"])
    w = fijar_signo(oja.weights)
    temas = tema_dominante(df)

    pesos = pd.DataFrame({"oja": w, "PC1": pc1, "diferencia": w - pc1}, index=variables)
    pesos.round(4).to_csv(SALIDA / "pesos.csv")
    grafico_pesos(pesos)

    indice = Z @ w
    proy1 = pca.transform(Z, 1)[:, 0]
    tabla = pd.DataFrame({
        "indice": indice, "PC1": proy1,
        "rank_indice": pd.Series(indice, index=df.index).rank(ascending=False).astype(int),
        "rank_PC1": pd.Series(proy1, index=df.index).rank(ascending=False).astype(int),
        "tema_dominante": temas,
        "followers": df.followers, "price_usd": df.price_usd,
        "saves_per_1000": df.saves_per_1000, "average_caption_words": df.average_caption_words,
    }, index=df.index).sort_values("indice", ascending=False)
    tabla.round(3).to_csv(SALIDA / "indice.csv")
    grafico_indice(tabla)

    # Correlación del índice con todas las variables numéricas (incluidas las derivadas).
    numericas = df.select_dtypes("number")
    corr = numericas.apply(lambda col: np.corrcoef(col, indice)[0, 1]).sort_values()
    corr.round(3).to_csv(SALIDA / "correlacion_indice_variables.csv", header=["correlacion"])
    np.savez(SALIDA / "modelo_final.npz", weights=w, variables=variables, **FINAL)

    print(f"Modelo final {FINAL}: ‖w‖ = {np.linalg.norm(w):.4f}, "
          f"ángulo con PC1 = {angulo(w, pc1):.3f}°, |w − PC1|_max = {np.abs(w - pc1).max():.4f}")
    print(f"Varianza del índice = {indice.var():.3f} (λ1 = {pca.eigenvalues[0]:.3f}, "
          f"{100 * indice.var() / Z.var(axis=0).sum():.1f} % del total)")
    print(f"Índice vs PC1: Pearson {np.corrcoef(indice, proy1)[0, 1]:.5f}, "
          f"Spearman {spearman(indice, proy1):.5f}, "
          f"posiciones distintas {(tabla.rank_indice != tabla.rank_PC1).sum()}")
    print(f"Índice vs followers: Pearson {corr['followers']:.3f}, "
          f"Spearman {spearman(indice, df.followers):.3f}")
    print("\nPesos ordenados:")
    print(pesos.sort_values("oja").round(3).to_string())
    print("\nCorrelación del índice con cada variable:")
    print(corr.round(2).to_string())
    print("\nRanking:")
    with pd.option_context("display.width", 200):
        print(tabla.round(2).to_string())
    print("\nÍndice por tema:")
    print(tabla.groupby("tema_dominante").indice.agg(["count", "mean", "min", "max"]).round(2))
    print("Top 10 por tema:", tabla.head(10).tema_dominante.value_counts().to_dict())


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    modo = sys.argv[1] if len(sys.argv) > 1 else "todo"
    df, variables, Z = cargar()
    if modo in ("convergencia", "todo"):
        pc1 = PCA().fit(Z).components[:, 0]
        resultados = convergencia(Z, pc1)
        tabla = resumen_convergencia(resultados)
        grafico_convergencia(resultados)
        print(tabla.round(4).to_string())
    if modo in ("final", "todo"):
        final(df, variables, Z)


if __name__ == "__main__":
    main()
