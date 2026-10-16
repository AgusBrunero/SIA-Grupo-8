"""Ej. 1.1: reducción a 15 variables con análisis de correlación y PCA.

Uso: python seleccion.py  →  resultados/seleccion/

Etapas (cada descarte queda en etapas.csv con su motivo):
  1. Derivadas (datos.DERIVADAS): 31 → 27.
  2. Ruido: variables cuya correlación máxima con el resto es compatible con el azar
     (test de permutación). Se conservan las que son un supuesto de negocio
     (datos.CONSERVAR_POR_DOMINIO). η² con el tema dominante se informa como respaldo.
  3. Correlación: en cada bloque conexo con |r| ≥ UMBRAL_BLOQUE se conserva la variable más
     central (mayor |r| medio con el resto del bloque).
  4. PCA hacia atrás (Jolliffe B2): mientras sobren variables, se toma la componente de menor
     autovalor (empezando por las restricciones "suman 100", de autovalor nulo) y se descarta
     la variable de mayor |coeficiente|: la que más pesa en la dirección de menor varianza es
     la más redundante.
  Los temas se protegen en todas las etapas: definen los cuatro clientes.

Después se compara el resultado con otras estrategias de selección de 15 variables.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agrupamiento import FINAL, SEMILLAS, entrenar
from datos import (CONSERVAR_POR_DOMINIO, DERIVADAS, TEMAS, VARIABLES, cargar_creadores,
                   estandarizar, tema_dominante)
from exploracion import heatmap
from graficos import guardar
from models.pca import PCA

SALIDA = Path(__file__).parent / "resultados" / "seleccion"
OBJETIVO = 15
UMBRAL_BLOQUE = 0.9
ALFA_RUIDO = 0.05
PERMUTACIONES = 2000


# ---------- Etapas ----------

def eta2(x, grupos):
    """Fracción de la varianza de x explicada por el grupo (η² de un ANOVA de un factor)."""
    x = pd.Series(np.asarray(x, dtype=float))
    g = x.groupby(np.asarray(grupos))
    return float((g.size() * (g.mean() - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())


def test_ruido(df, variables, temas, rng):
    """p-valor de max|r| con el resto bajo permutación de la variable, y η² con el tema."""
    X = df[variables].to_numpy(float)
    Z, _, _ = estandarizar(X)
    filas = []
    for j, var in enumerate(variables):
        otras = np.delete(Z, j, axis=1)
        observado = np.abs(Z[:, j] @ otras / len(Z)).max()
        nulos = np.array([np.abs(rng.permutation(Z[:, j]) @ otras / len(Z)).max()
                          for _ in range(PERMUTACIONES)])
        filas.append({"variable": var, "max_abs_r": observado,
                      "p_permutacion": (nulos >= observado).mean(),
                      "eta2_tema": eta2(X[:, j], temas)})
    return pd.DataFrame(filas).set_index("variable")


def bloques_correlacion(corr, umbral):
    """Componentes conexas del grafo |r| ≥ umbral (solo las de más de una variable)."""
    variables, vistos, bloques = list(corr.index), set(), []
    for inicio in variables:
        if inicio in vistos:
            continue
        bloque, pendientes = [], [inicio]
        while pendientes:
            v = pendientes.pop()
            if v in vistos:
                continue
            vistos.add(v)
            bloque.append(v)
            pendientes += [u for u in variables if u not in vistos and abs(corr.loc[v, u]) >= umbral]
        if len(bloque) > 1:
            bloques.append(bloque)
    return bloques


def direccion_redundante(pca, libres):
    """Dirección de menor varianza que involucra variables libres: (vector, autovalor, nombre).

    Las restricciones "suman 100" (temas, formatos) dan autovalores nulos. Si hay varias, su
    espacio nulo es una mezcla arbitraria: se busca ahí la combinación que no toca a las
    protegidas, así una restricción solo entre protegidas no fuerza un descarte.
    """
    nulas = pca.eigenvalues < 1e-9
    if nulas.any():
        N = pca.components[:, nulas]
        v = N[:, -1] if libres.all() else N @ np.linalg.svd(N[~libres])[2][-1]
        if np.abs(v * libres).max() > 1e-6:
            return v, 0.0, "espacio nulo"
    j = np.flatnonzero(~nulas)[-1]
    return pca.components[:, j], pca.eigenvalues[j], f"PC{j + 1}"


def pca_hacia_atras(Z, variables, n, protegidas=()):
    """Jolliffe B2. Devuelve (variables que quedan, registro de descartes)."""
    variables, registro = list(variables), []
    while len(variables) > n:
        pca = PCA().fit(Z[variables].to_numpy())
        libres = np.array([v not in protegidas for v in variables])
        v, autovalor, nombre = direccion_redundante(pca, libres)
        i = int(np.abs(v * libres).argmax())
        registro.append({"variable": variables[i], "componente": nombre,
                         "autovalor": autovalor, "coeficiente": v[i]})
        variables.pop(i)
    return variables, registro


# ---------- Criterios para comparar subconjuntos ----------

def varianza_retenida(R, todas, subconjunto):
    """Fracción de la varianza total de `todas` (estandarizadas) que se reproduce por regresión
    lineal sobre `subconjunto` (criterio RM de Cadima y Jolliffe, al cuadrado)."""
    s = [todas.index(v) for v in subconjunto]
    Rfs = R[:, s]
    return np.trace(Rfs @ np.linalg.pinv(R[np.ix_(s, s)]) @ Rfs.T) / len(todas)


def r2_componentes(Z, subconjunto, pca, k):
    """R² medio (ponderado por autovalor) de reconstruir las k primeras PCs de todas las
    variables a partir del subconjunto."""
    P = pca.transform(Z.to_numpy(), k)
    A = np.c_[np.ones(len(Z)), Z[subconjunto].to_numpy()]
    residuo = P - A @ np.linalg.lstsq(A, P, rcond=None)[0]
    r2 = 1 - (residuo ** 2).sum(axis=0) / (P ** 2).sum(axis=0)
    return float(np.average(r2, weights=pca.eigenvalues[:k]))


def pureza_som(Z, temas):
    """Fracción media (sobre SEMILLAS) de creadores cuyo tema coincide con el mayoritario de su
    neurona en el SOM final: si los temas quedan separados en el mapa del ejercicio 1.2."""
    purezas = []
    for semilla in SEMILLAS:
        bmus = entrenar(Z, FINAL, semilla).bmus
        purezas.append(sum(pd.Series(temas[bmus == i]).value_counts().iloc[0]
                           for i in np.unique(bmus)) / len(Z))
    return float(np.mean(purezas))


def evaluar(nombre, subconjunto, Z, R, todas, pca_todas, temas, ruido):
    Zs = Z[subconjunto].to_numpy()
    pca = PCA().fit(Zs)
    corr = np.abs(np.corrcoef(Zs, rowvar=False))
    np.fill_diagonal(corr, 0)
    kaiser = int((pca.eigenvalues > 1).sum())
    proy = pca.transform(Zs, 4)
    return {
        "estrategia": nombre,
        "varianza_retenida": varianza_retenida(R, todas, subconjunto),
        "r2_pc1_6_completo": r2_componentes(Z, subconjunto, pca_todas, 6),
        "max_abs_r": corr.max(),
        "kaiser": kaiser,
        "varianza_kaiser": pca.explained_variance_ratio[:kaiser].sum(),
        "varianza_pc1_pc2": pca.explained_variance_ratio[:2].sum(),
        "max_eta2_tema_pc1_4": max(eta2(proy[:, j], temas) for j in range(4)),
        "pureza_tema_som": pureza_som(Zs, temas),
        "cuatro_temas": all(t in subconjunto for t in TEMAS),
        "variables_ruido": ", ".join(v for v in ruido if v in subconjunto
                                     and v not in CONSERVAR_POR_DOMINIO) or "-",
        "faltan_ejes": ", ".join(v for v in ("average_reach", "non_follower_reach_pct",
                                             "argentina_audience_pct", "saves_per_1000")
                                 if v not in subconjunto) or "-",
        "descartadas": ", ".join(v for v in todas if v not in subconjunto),
    }


def pca_hacia_adelante(Z, variables, n):
    """Jolliffe B4: para PC1, PC2, … se toma la variable de mayor |coeficiente| aún no elegida."""
    pca, elegidas = PCA().fit(Z[variables].to_numpy()), []
    for j in range(len(variables)):
        orden = np.argsort(-np.abs(pca.components[:, j]))
        elegidas.append(next(variables[i] for i in orden if variables[i] not in elegidas))
        if len(elegidas) == n:
            return elegidas


def maxima_varianza(R, variables, n, fijas=()):
    """Avance codicioso + intercambios locales que maximizan `varianza_retenida`."""
    elegidas = list(fijas)
    while len(elegidas) < n:
        elegidas.append(max((v for v in variables if v not in elegidas),
                            key=lambda v: varianza_retenida(R, variables, elegidas + [v])))
    mejora = True
    while mejora:
        mejora, actual = False, varianza_retenida(R, variables, elegidas)
        for a, b in ((a, b) for a in elegidas if a not in fijas
                     for b in variables if b not in elegidas):
            candidata = [b if v == a else v for v in elegidas]
            if varianza_retenida(R, variables, candidata) > actual + 1e-9:
                elegidas, mejora = candidata, True
                break
    return elegidas


# ---------- Gráficos ----------

def grafico_correlaciones_finales(df, variables):
    corr = df[variables].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = heatmap(ax, corr, variables, variables, anotar=True)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlación de Pearson")
    ax.set_title(f"Correlaciones entre las {len(variables)} variables seleccionadas")
    guardar(fig, SALIDA / "correlaciones_seleccionadas.png")


def grafico_comparacion(tabla):
    columnas = [("varianza_retenida", "varianza de las 27 retenida"),
                ("r2_pc1_6_completo", "R² de PC1–PC6 (27 variables)"),
                ("max_abs_r", "máx. |r| interno"),
                ("pureza_tema_som", "pureza de tema en el SOM")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    y = np.arange(len(tabla))
    for ax, (col, titulo) in zip(axes, columnas):
        colores = ["#eb6834" if e == "elegida" else "#2a78d6" for e in tabla.index]
        ax.barh(y, tabla[col], color=colores, height=0.6)
        for yi, v in zip(y, tabla[col]):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.15)
        ax.set_title(titulo, fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, tabla.index)
    axes[0].invert_yaxis()
    fig.suptitle("Estrategias de selección de variables (naranja = elegida)")
    fig.tight_layout()
    guardar(fig, SALIDA / "comparacion.png")


# ---------- Main ----------

def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = cargar_creadores()
    temas = tema_dominante(df).to_numpy()
    etapas = [{"etapa": "1. derivada", "variable": v, "motivo": f"= {f}"} for v, f in DERIVADAS.items()]
    todas = [c for c in df.select_dtypes("number").columns if c not in DERIVADAS]

    # 2. Ruido
    ruido = test_ruido(df, todas, temas, rng)
    ruido.round(4).sort_values("p_permutacion", ascending=False).to_csv(SALIDA / "ruido.csv")
    sin_estructura = [v for v in ruido.index[ruido.p_permutacion > ALFA_RUIDO] if v not in TEMAS]
    restantes = [v for v in todas if v not in sin_estructura or v in CONSERVAR_POR_DOMINIO]
    for v in sin_estructura:
        fila = ruido.loc[v]
        dato = f"max|r| = {fila.max_abs_r:.2f} (p = {fila.p_permutacion:.2f}), η² tema = {fila.eta2_tema:.2f}"
        if v in CONSERVAR_POR_DOMINIO:
            etapas.append({"etapa": "2. ruido (se conserva)", "variable": v,
                           "motivo": f"{dato}; {CONSERVAR_POR_DOMINIO[v]}"})
        else:
            etapas.append({"etapa": "2. ruido", "variable": v, "motivo": dato})

    # 3. Bloques de correlación
    Z = pd.DataFrame(estandarizar(df[todas])[0], columns=todas, index=df.index)
    corr = Z[restantes].corr()
    for bloque in bloques_correlacion(corr, UMBRAL_BLOQUE):
        centralidad = corr.loc[bloque, bloque].abs().mean().sort_values(ascending=False)
        representante = centralidad.index[0]
        for v in centralidad.index[1:]:
            etapas.append({"etapa": "3. correlación", "variable": v,
                           "motivo": f"|r| = {abs(corr.loc[v, representante]):.3f} con "
                                     f"{representante} (representante del bloque)"})
            restantes.remove(v)

    # 4. PCA hacia atrás
    elegidas, registro = pca_hacia_atras(Z, restantes, OBJETIVO, protegidas=TEMAS)
    for r in registro:
        etapas.append({"etapa": "4. PCA (B2)", "variable": r["variable"],
                       "motivo": f"coef. {r['coeficiente']:+.2f} en {r['componente']} "
                                 f"(autovalor {r['autovalor']:.3f})"})
    etapas = pd.DataFrame(etapas)
    etapas.to_csv(SALIDA / "etapas.csv", index=False)
    assert elegidas == [v for v in VARIABLES], "datos.VARIABLES no coincide con la selección"

    # Comparación con otras estrategias
    R = Z[todas].corr().to_numpy()
    pca_todas = PCA().fit(Z[todas].to_numpy())
    candidatas = {
        "todas (27)": todas,
        "elegida": elegidas,
        "PCA atrás (B2) puro": pca_hacia_atras(Z, todas, OBJETIVO)[0],
        "PCA atrás, temas fijos": pca_hacia_atras(Z, todas, OBJETIVO, protegidas=TEMAS)[0],
        "PCA adelante (B4)": pca_hacia_adelante(Z, todas, OBJETIVO),
        "máx. varianza": maxima_varianza(R, todas, OBJETIVO),
        "máx. varianza, temas fijos": maxima_varianza(R, todas, OBJETIVO, fijas=TEMAS),
    }
    tabla = pd.DataFrame([evaluar(n, s, Z, R, todas, pca_todas, temas, sin_estructura)
                          for n, s in candidatas.items()]).set_index("estrategia")
    tabla.round(4).to_csv(SALIDA / "comparacion.csv")

    # Meseta: cuántos subconjuntos de 15 con los temas, a partir de las variables de la etapa 3,
    # retienen tanta varianza como la elección.
    libres = [v for v in restantes if v not in TEMAS]
    meseta = [varianza_retenida(R, todas, TEMAS + list(c))
              for c in combinations(libres, OBJETIVO - len(TEMAS))]

    grafico_correlaciones_finales(df, elegidas)
    grafico_comparacion(tabla)

    print(etapas.to_string(index=False))
    print(f"\nSeleccionadas ({len(elegidas)}): {elegidas}")
    print("\n" + tabla.drop(columns=["descartadas"]).round(3).to_string())
    print(f"\nSubconjuntos de 15 con los 4 temas desde la etapa 3: {len(meseta)}; varianza "
          f"retenida entre {min(meseta):.3f} y {max(meseta):.3f} (elegida "
          f"{tabla.loc['elegida', 'varianza_retenida']:.3f})")


if __name__ == "__main__":
    main()
