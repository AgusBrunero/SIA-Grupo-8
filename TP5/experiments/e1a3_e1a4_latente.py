"""Consigna 1a.3 y 1a.4: el espacio latente y la generación de letras nuevas.

1a.3  ¿Dónde queda cada uno de los 32 patrones en el latente de 2 dimensiones, y qué parte
      de esa disposición se repite entre semillas? Línea base: PCA 2D (decisión D8).
1a.4  ¿Qué decodifica la red en puntos del latente que no corresponden a ningún patrón?
      Se muestran todas las candidatas con su distancia de Hamming a la letra más cercana;
      cuál es "una letra nueva" lo decide el grupo (decisión D1).

Modelo: la arquitectura elegida en e1a2_arquitecturas (D57: 35–30–15–2–15–30–35, ocultas y
latente tanh, salida logística, BCE, Adam), con las semillas del config.

Sostén: Autoencoders diap. 32–33 (latente 2D y concept vector), diap. 34 (algoritmo del
Generative Autoencoder), diap. 7–16 (AE lineal y PCA).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1a3_e1a4_latente config/e1a3_e1a4.json

Salida en results/e1a3_e1a4/: entrenamiento.csv, latente.csv, pca.csv, estabilidad.csv,
grilla.csv, interpolaciones.csv, fig_*.png, config.json y el bloque e1a3_e1a4 de
docs/RESULTADOS.md.
"""
import json
import shutil
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA, TINTA_SECUNDARIA, aplicar_estilo, escribir_seccion
from experiments.e1a2_escalera import construir
from src.data.font import COLUMNAS, FILAS, cargar_font, distancias_hamming
from src.modelos.autoencoder import entrenar, evaluar
from src.modelos.generacion import grilla_latente, interpolar, vecino_mas_cercano
from src.modelos.metricas import binarizar, pixeles_incorrectos, resumir
from src.modelos.pca import ajustar_pca
from src.nn.optimizadores import crear


def entrenar_modelo(config: dict, semilla: int, X: np.ndarray):
    """Igual que una corrida de la arquitectura elegida en e1a2_arquitecturas con la misma semilla."""
    modelo = config['modelo']
    ae = construir(config, modelo, semilla)
    hiper = dict(modelo['optimizador'])
    optimizador = crear(hiper.pop('nombre'), **hiper)
    rng = np.random.default_rng(np.random.SeedSequence([semilla, 1]))
    historial = entrenar(ae, X, X, optimizador, config['epocas'], modelo['tamano_lote'], rng,
                         perdida=modelo['perdida'], codificacion=config['codificacion'],
                         reduccion=config['reduccion'])
    error, incorrectos = evaluar(ae, X, X, modelo['perdida'], config['codificacion'])
    resumen = resumir(incorrectos)
    fila = {'semilla': semilla, 'logro': resumen.cumple, 'max_pixeles': resumen.maximo,
            'total_pixeles': resumen.total, 'error_final': error, 'segundos': historial.segundos}
    return ae, fila


def distancias_entre_pares(Z: np.ndarray) -> np.ndarray:
    """Las 496 distancias euclídeas entre pares de patrones. No cambian si el latente se
    rota, se refleja o se traslada."""
    i, j = np.triu_indices(len(Z), k=1)
    return np.linalg.norm(Z[i] - Z[j], axis=1)


def vecino_en_latente(Z: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.argmin(axis=1)


def como_texto(muestra: np.ndarray) -> str:
    return ''.join('1' if pixel else '0' for pixel in muestra)


# ---------- figuras ----------

def dibujar_latente(ax, Z: np.ndarray, etiquetas: list[str], color: str) -> None:
    ax.scatter(Z[:, 0], Z[:, 1], s=14, color=color, zorder=3)
    for (z1, z2), etiqueta in zip(Z, etiquetas):
        ax.annotate(etiqueta, (z1, z2), xytext=(3, 3), textcoords='offset points', fontsize=10, color=TINTA)
    ax.set_xlabel('z1 (unidades del latente)')
    ax.set_ylabel('z2 (unidades del latente)')


def figura_latentes(latentes: dict[int, np.ndarray], etiquetas: list[str], ruta) -> None:
    fig, ejes = plt.subplots(1, len(latentes), figsize=(6 * len(latentes), 5.5))
    for ax, (semilla, Z) in zip(np.atleast_1d(ejes), latentes.items()):
        dibujar_latente(ax, Z, etiquetas, COLOR['AE'])
        ax.text(0.02, 0.98, f'semilla {semilla}', transform=ax.transAxes, va='top', fontweight='bold')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_pca(Z: np.ndarray, etiquetas: list[str], ruta) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    dibujar_latente(ax, Z, etiquetas, COLOR['PCA'])
    ax.set_xlabel('Primera componente principal')
    ax.set_ylabel('Segunda componente principal')
    fig.savefig(ruta)
    plt.close(fig)


def dibujar_muestra(ax, muestra: np.ndarray, epigrafe: str) -> None:
    ax.imshow(muestra.reshape(FILAS, COLUMNAS), cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    for borde in ax.spines.values():
        borde.set_visible(True)
        borde.set_color('#c3c2b7')
    ax.set_xlabel(epigrafe, fontsize=8, color=TINTA_SECUNDARIA, labelpad=2)


def figura_grilla(grilla: pd.DataFrame, n: int, ruta) -> None:
    """Todas las muestras de la grilla. Debajo de cada una: distancia de Hamming a la letra
    más cercana del dataset y cuál es esa letra."""
    fig, ejes = plt.subplots(n, n, figsize=(0.95 * n, 1.45 * n))
    for fila in grilla.itertuples():
        muestra = np.array([int(c) for c in fila.pixeles])
        dibujar_muestra(ejes[fila.fila, fila.columna], muestra, f'{fila.distancia} · {fila.vecino}')
    fig.supxlabel('z1 creciente →     (debajo de cada muestra: píxeles de distancia a la letra más cercana · esa letra)',
                  fontsize=10)
    fig.supylabel('z2 creciente →', fontsize=10)
    fig.tight_layout(h_pad=0.6, w_pad=0.2)
    fig.savefig(ruta)
    plt.close(fig)


def figura_interpolaciones(interpolaciones: pd.DataFrame, pares: list[list[str]], pasos: int, ruta) -> None:
    fig, ejes = plt.subplots(len(pares), pasos, figsize=(0.95 * pasos, 1.6 * len(pares)))
    for fila in interpolaciones.itertuples():
        muestra = np.array([int(c) for c in fila.pixeles])
        ax = ejes[[tuple(p) for p in pares].index(tuple(fila.par.split('→'))), fila.paso]
        dibujar_muestra(ax, muestra, f'{fila.distancia} · {fila.vecino}')
    for indice, par in enumerate(pares):
        ejes[indice, 0].set_ylabel(f'{par[0]} → {par[1]}', fontsize=10)
    fig.supxlabel('Posición sobre el segmento que une los dos códigos (izquierda: origen; derecha: destino)', fontsize=10)
    fig.tight_layout(h_pad=0.8, w_pad=0.2)
    fig.savefig(ruta)
    plt.close(fig)


# ---------- texto generado desde los datos ----------

def n(valor: float, decimales: int = 2) -> str:
    return f'{valor:.{decimales}f}'.replace('.', ',')


def texto(config, entrenamiento, pca_resumen, estabilidad, grilla, interpolaciones, etiquetas) -> str:
    semillas = config['semillas']
    lineas_entrenamiento = '; '.join(
        f'semilla {f.semilla}: máximo {f.max_pixeles} px, total {f.total_pixeles} px' for f in entrenamiento.itertuples())
    entre_semillas = estabilidad[estabilidad.tipo == 'AE–AE']
    con_hamming = estabilidad[estabilidad.tipo == 'AE–Hamming']
    con_pca = estabilidad[estabilidad.tipo == 'AE–PCA']
    pca_hamming = estabilidad[estabilidad.tipo == 'PCA–Hamming'].iloc[0]
    lineas_grilla = []
    for semilla, g in grilla.groupby('semilla'):
        conteo = g.distancia.value_counts().sort_index()
        reparto = ', '.join(f'{cantidad} a {distancia} px' for distancia, cantidad in conteo.items())
        lineas_grilla.append(f'- **Semilla {semilla}** ({len(g)} muestras; rango z1 [{n(g.z1.min())}; {n(g.z1.max())}], '
                             f'z2 [{n(g.z2.min())}; {n(g.z2.max())}]): {int((g.distancia == 0).sum())} son idénticas a una letra '
                             f'del dataset y {int((g.distancia > 0).sum())} no. Distancia a la letra más cercana: {reparto}. '
                             f'Letras distintas que aparecen como la más cercana: {g.vecino.nunique()} de 32.')
    lineas_inter = []
    for (semilla, par), camino in interpolaciones.groupby(['semilla', 'par'], sort=False):
        recorrido = ' → '.join(f'{f.distancia}·{f.vecino}' for f in camino.itertuples())
        lineas_inter.append(f'- Semilla {semilla}, {par}: {recorrido}.')
    p = pca_resumen.iloc[0]
    return f"""## 1a.3 y 1a.4 · Espacio latente y generación (e1a3_e1a4)

Modelo: arquitectura elegida en e1a2_arquitecturas (D57: 35–30–15–2–15–30–35, latente tanh, BCE, Adam η = 0,003), {config['epocas']} épocas, semillas {', '.join(map(str, semillas))}. Datos en `results/e1a3_e1a4/`.

### Entrenamiento

- Píxeles incorrectos al reconstruir los 32 patrones: {lineas_entrenamiento}. Las {int(entrenamiento.logro.sum())} de {len(entrenamiento)} semillas cumplen ≤ 1 px por patrón.

### 1a.3 · Latente (`fig_latente_ae.png`, `latente.csv`)

- La posición de cada patrón cambia de una semilla a otra. Para comparar sin depender de rotaciones, reflejos ni traslaciones se usan las 496 distancias entre pares de patrones en el latente (`estabilidad.csv`).
- Correlación de Pearson de esas distancias entre semillas: {', '.join(f'{f.a} y {f.b}: {n(f.correlacion)}' for f in entre_semillas.itertuples())}.
- Patrones (de 32) cuyo vecino más cercano en el latente es el mismo en las dos semillas: {', '.join(f'{f.a} y {f.b}: {int(f.mismo_vecino)}' for f in entre_semillas.itertuples())}.
- Correlación entre la distancia en el latente y la distancia de Hamming entre los patrones: {', '.join(f'{f.a}: {n(f.correlacion)}' for f in con_hamming.itertuples())}.

### Línea base: PCA 2D con SVD (`fig_latente_pca.png`, `pca.csv`)

- Las dos primeras componentes explican el {n(100 * p.varianza_c1, 1)} % y el {n(100 * p.varianza_c2, 1)} % de la varianza ({n(100 * (p.varianza_c1 + p.varianza_c2), 1)} % en total).
- Reconstruyendo con 2 componentes y umbral 0,5: máximo {int(p.max_pixeles)} px incorrectos en un patrón, {int(p.total_pixeles)} en total, y {int(p.patrones_con_error)} de 32 patrones con más de 1 px mal. El autoencoder, en la peor de sus semillas: {int(entrenamiento.total_pixeles.max())} en total.
- Correlación de las distancias entre pares: PCA con Hamming {n(pca_hamming.correlacion)}; AE con PCA {', '.join(f'{f.a}: {n(f.correlacion)}' for f in con_pca.itertuples())}.

### 1a.4 · Grilla decodificada (`fig_grilla_semilla*.png`, `grilla.csv`)

Grilla regular de {config['grilla']} × {config['grilla']} sobre el rectángulo que cubre los 32 códigos (algoritmo de Autoencoders diap. 34). Cada muestra se umbraliza en 0,5 y se compara con los 32 patrones.

{chr(10).join(lineas_grilla)}

### 1a.4 · Interpolaciones (`fig_interpolaciones_semilla*.png`, `interpolaciones.csv`)

{config['pasos_interpolacion']} puntos sobre el segmento entre los códigos de dos letras (Autoencoders diap. 33). En cada punto: distancia en píxeles a la letra más cercana · esa letra. Distancia de Hamming entre las letras de cada par: {', '.join(f"{a}–{b}: {int(d)}" for a, b, d in config['_distancias_pares'])}.

{chr(10).join(lineas_inter)}

No se señala ninguna muestra como "letra nueva": lo decide el grupo mirando las grillas.
"""


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    X, etiquetas = cargar_font()
    hamming = distancias_hamming(X)
    i, j = np.triu_indices(32, k=1)
    hamming_pares = hamming[i, j].astype(float)
    pares = config['pares_interpolacion']
    config['_distancias_pares'] = [(a, b, hamming[etiquetas.index(a), etiquetas.index(b)]) for a, b in pares]

    # --- PCA
    pca = ajustar_pca(X, 2)
    Z_pca = pca.proyectar(X)
    incorrectos_pca = pixeles_incorrectos(pca.reconstruir(Z_pca), X, config['codificacion'])
    resumen_pca = resumir(incorrectos_pca)
    pca_resumen = pd.DataFrame([{'varianza_c1': pca.varianza_explicada[0], 'varianza_c2': pca.varianza_explicada[1],
                                 'max_pixeles': resumen_pca.maximo, 'total_pixeles': resumen_pca.total,
                                 'patrones_con_error': resumen_pca.patrones_con_error}])
    pca_resumen.to_csv(salida / 'pca.csv', index=False)
    figura_pca(Z_pca, etiquetas, salida / 'fig_latente_pca.png')

    # --- Autoencoder, una vez por semilla
    filas, latentes, tablas_latente, grillas, interpolaciones = [], {}, [], [], []
    for semilla in config['semillas']:
        ae, fila = entrenar_modelo(config, semilla, X)
        print(fila, flush=True)
        filas.append(fila)
        Z = ae.codificar(X)
        latentes[semilla] = Z
        tablas_latente.append(pd.DataFrame({'modelo': 'AE', 'semilla': semilla, 'etiqueta': etiquetas,
                                            'z1': Z[:, 0], 'z2': Z[:, 1]}))
        # Grilla (diap. 34): valores de z elegidos directamente, decodificados
        puntos = grilla_latente(Z, config['grilla'])
        muestras = binarizar(ae.decodificar(puntos), config['codificacion'])
        vecinos, distancias = vecino_mas_cercano(muestras, X)
        grillas.append(pd.DataFrame({
            'semilla': semilla, 'fila': np.repeat(np.arange(config['grilla']), config['grilla']),
            'columna': np.tile(np.arange(config['grilla']), config['grilla']),
            'z1': puntos[:, 0], 'z2': puntos[:, 1], 'distancia': distancias,
            'vecino': [etiquetas[v] for v in vecinos], 'pixeles': [como_texto(m) for m in muestras]}))
        # Interpolaciones (diap. 33)
        for a, b in pares:
            camino = interpolar(Z[etiquetas.index(a)], Z[etiquetas.index(b)], config['pasos_interpolacion'])
            muestras = binarizar(ae.decodificar(camino), config['codificacion'])
            vecinos, distancias = vecino_mas_cercano(muestras, X)
            interpolaciones.append(pd.DataFrame({
                'semilla': semilla, 'par': f'{a}→{b}', 'paso': np.arange(len(camino)),
                'z1': camino[:, 0], 'z2': camino[:, 1], 'distancia': distancias,
                'vecino': [etiquetas[v] for v in vecinos], 'pixeles': [como_texto(m) for m in muestras]}))

    entrenamiento = pd.DataFrame(filas)
    assert len(entrenamiento) == len(config['semillas']) and entrenamiento.logro.all(), entrenamiento
    grilla = pd.concat(grillas)
    interpolacion = pd.concat(interpolaciones)
    assert (grilla.groupby('semilla').size() == config['grilla'] ** 2).all()
    assert (interpolacion.groupby(['semilla', 'par']).size() == config['pasos_interpolacion']).all()
    # Los extremos de cada interpolación son las dos letras del par
    extremos = interpolacion[interpolacion.paso.isin([0, config['pasos_interpolacion'] - 1])]
    assert (extremos.distancia == 0).all()

    # --- Qué se repite entre semillas
    estabilidad = []
    for a, b in combinations(config['semillas'], 2):
        estabilidad.append({'tipo': 'AE–AE', 'a': f'semilla {a}', 'b': f'semilla {b}',
                            'correlacion': np.corrcoef(distancias_entre_pares(latentes[a]),
                                                       distancias_entre_pares(latentes[b]))[0, 1],
                            'mismo_vecino': int((vecino_en_latente(latentes[a]) == vecino_en_latente(latentes[b])).sum())})
    for semilla, Z in latentes.items():
        estabilidad.append({'tipo': 'AE–Hamming', 'a': f'semilla {semilla}', 'b': 'Hamming',
                            'correlacion': np.corrcoef(distancias_entre_pares(Z), hamming_pares)[0, 1]})
        estabilidad.append({'tipo': 'AE–PCA', 'a': f'semilla {semilla}', 'b': 'PCA',
                            'correlacion': np.corrcoef(distancias_entre_pares(Z), distancias_entre_pares(Z_pca))[0, 1],
                            'mismo_vecino': int((vecino_en_latente(Z) == vecino_en_latente(Z_pca)).sum())})
    estabilidad.append({'tipo': 'PCA–Hamming', 'a': 'PCA', 'b': 'Hamming',
                        'correlacion': np.corrcoef(distancias_entre_pares(Z_pca), hamming_pares)[0, 1]})
    estabilidad = pd.DataFrame(estabilidad)

    # --- Guardar
    entrenamiento.to_csv(salida / 'entrenamiento.csv', index=False)
    tablas_latente.append(pd.DataFrame({'modelo': 'PCA', 'semilla': -1, 'etiqueta': etiquetas,
                                        'z1': Z_pca[:, 0], 'z2': Z_pca[:, 1]}))
    pd.concat(tablas_latente).to_csv(salida / 'latente.csv', index=False)
    estabilidad.to_csv(salida / 'estabilidad.csv', index=False)
    grilla.to_csv(salida / 'grilla.csv', index=False)
    interpolacion.to_csv(salida / 'interpolaciones.csv', index=False)

    figura_latentes(latentes, etiquetas, salida / 'fig_latente_ae.png')
    for semilla in config['semillas']:
        figura_grilla(grilla[grilla.semilla == semilla], config['grilla'], salida / f'fig_grilla_semilla{semilla}.png')
        figura_interpolaciones(interpolacion[interpolacion.semilla == semilla], pares,
                               config['pasos_interpolacion'], salida / f'fig_interpolaciones_semilla{semilla}.png')

    bloque = texto(config, entrenamiento, pca_resumen, estabilidad, grilla, interpolacion, etiquetas)
    escribir_seccion('e1a3_e1a4', bloque)
    print(bloque)


if __name__ == '__main__':
    main(sys.argv[1])
