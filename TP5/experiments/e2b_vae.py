"""Consigna 2a y 2b: dataset nuevo (caritas) y entrenamiento del VAE.

2a  ¿Cómo es el conjunto de datos construido? Se dibuja una muestra.
2b  ¿Aprende el VAE? Curvas de la reconstrucción L y de la regularización R por separado
    (J = L + λR), por semilla, y reconstrucciones de algunos patrones.

Sostén: Autoencoders diap. 74–89 (supuestos, KL cerrada, pérdida, reparametrización y
gradientes); convención del Colab de la diap. 90 (doc de cátedra §2.1); Adam de la
Clase 12.1 diap. 11–13; minibatch de la Clase 11 diap. 53.

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e2b_vae config/e2b.json

Salida en results/e2b/: entrenamiento.csv, curvas.csv, fig_dataset.png, fig_curvas.png,
fig_reconstrucciones.png, config.json y el bloque e2b de docs/RESULTADOS.md.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA_SECUNDARIA, aplicar_estilo, escribir_seccion
from src.data.emojis import LADO, PARAMETROS, generar_caritas
from src.modelos.vae import VAE, HistorialVAE, entrenar_vae
from src.nn.optimizadores import crear
from src.nn.perdidas import bce


def construir_vae(config: dict, semilla: int) -> VAE:
    modelo = config['modelo']
    return VAE(modelo['capas_encoder'], modelo['capas_decoder'], modelo['activacion_oculta'], semilla,
               beta=config['beta'], amplitud_inicial=config['amplitud_inicial'])


def entrenar_semilla(config: dict, semilla: int, X: np.ndarray, lam: float) -> tuple[VAE, HistorialVAE]:
    """Una corrida del VAE. La usan también los experimentos de la sesión 6."""
    vae = construir_vae(config, semilla)
    hiper = dict(config['optimizador'])
    optimizador = crear(hiper.pop('nombre'), **hiper)
    historial = entrenar_vae(vae, X, optimizador, config['epocas'], config['tamano_lote'], lam,
                             rng_lotes=np.random.default_rng(np.random.SeedSequence([semilla, 1])),
                             rng_epsilon=np.random.default_rng(np.random.SeedSequence([semilla, 4])))
    return vae, historial


def evaluar_vae(vae: VAE, X: np.ndarray) -> dict:
    """Medidas sin ruido de muestreo: se decodifica z = μ(x)."""
    mu, s = vae.codificar(X)
    salida = vae.decodificar(mu)
    L, _ = bce(salida, X, 'media')
    return {'L_con_z_igual_mu': L,
            'R': 0.5 * float(np.sum(np.exp(s) + mu ** 2 - 1.0 - s)) / len(X),
            'error_absoluto_por_pixel': float(np.abs(salida - X).mean()),
            'mu_media': float(mu.mean()), 'mu_desvio': float(mu.std()),
            'sigma_media': float(np.exp(s / 2.0).mean())}


def dibujar(ax, imagen: np.ndarray) -> None:
    ax.imshow(imagen.reshape(LADO, LADO), cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    for borde in ax.spines.values():
        borde.set_visible(True)
        borde.set_color('#c3c2b7')


def figura_dataset(X: np.ndarray, cantidad: int, ruta) -> None:
    columnas = 12
    filas = cantidad // columnas
    fig, ejes = plt.subplots(filas, columnas, figsize=(columnas * 0.8, filas * 0.8))
    for ax, imagen in zip(ejes.ravel(), X[:filas * columnas]):
        dibujar(ax, imagen)
    fig.supxlabel(f'Primeras {filas * columnas} caritas del dataset (16 × 16 píxeles, grises en [0, 1])', fontsize=10)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(ruta)
    plt.close(fig)


def figura_curvas(curvas: pd.DataFrame, ruta) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8))
    for columna, ax, etiqueta in (('L', ejes[0], 'Reconstrucción L (BCE sumada sobre los 256 píxeles,\npromedio por patrón, escala log)'),
                                  ('R', ejes[1], 'Regularización R (KL sumada sobre el latente,\npromedio por patrón)')):
        for _, curva in curvas.groupby('semilla'):
            ax.plot(curva.epoca, curva[columna], color=COLOR['VAE'], linewidth=1.2, alpha=0.7)
        ax.set_xlabel('Época (una línea por semilla)')
        ax.set_ylabel(etiqueta)
    ejes[0].set_yscale('log')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_reconstrucciones(X: np.ndarray, reconstrucciones: dict[int, np.ndarray], cantidad: int, ruta) -> None:
    filas = 1 + len(reconstrucciones)
    fig, ejes = plt.subplots(filas, cantidad, figsize=(cantidad * 0.85, filas * 0.95))
    for columna in range(cantidad):
        dibujar(ejes[0, columna], X[columna])
    ejes[0, 0].set_ylabel('original', fontsize=9, rotation=0, ha='right', va='center')
    for fila, (semilla, salida) in enumerate(reconstrucciones.items(), start=1):
        for columna in range(cantidad):
            dibujar(ejes[fila, columna], salida[columna])
        ejes[fila, 0].set_ylabel(f'semilla {semilla}', fontsize=9, rotation=0, ha='right', va='center', color=TINTA_SECUNDARIA)
    fig.supxlabel('Arriba: carita del dataset. Abajo: salida del decoder para z = μ(x), por semilla', fontsize=10)
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    fig.savefig(ruta)
    plt.close(fig)


def num(valor: float, decimales: int = 2) -> str:
    return f'{valor:.{decimales}f}'.replace('.', ',')


def pm(serie: pd.Series, decimales: int = 2) -> str:
    return f'{num(serie.mean(), decimales)} ± {num(serie.std(), decimales)}'


def texto(config: dict, X: np.ndarray, entrenamiento: pd.DataFrame, curvas: pd.DataFrame) -> str:
    modelo = config['modelo']
    epocas = config['epocas']
    cortes = [e for e in (1, 10, 100, epocas // 2, epocas) if e <= epocas]
    lineas = ['| Época | L (media ± desvío entre semillas) | R (media ± desvío entre semillas) |', '|---|---|---|']
    for epoca in cortes:
        c = curvas[curvas.epoca == epoca]
        lineas.append(f'| {epoca} | {pm(c.L)} | {pm(c.R)} |')
    ultimo_tramo = curvas[curvas.epoca > 0.9 * epocas].groupby('semilla')
    cambio_L = (ultimo_tramo.L.last() - ultimo_tramo.L.first())
    return f"""## 2a y 2b · Dataset de caritas y entrenamiento del VAE (e2b)

Datos en `results/e2b/`. Figuras: `fig_dataset.png`, `fig_curvas.png`, `fig_reconstrucciones.png`.

### 2a · Dataset

- {len(X)} caritas de {LADO} × {LADO} ({X.shape[1]} píxeles) en grises en [0, 1], construidas en código (`src/data/emojis.py`) con la semilla {config['dataset']['semilla']} a partir de {len(PARAMETROS)} parámetros continuos: {', '.join(PARAMETROS)}.
- Valor medio de un píxel: {num(X.mean(), 3)}. El {num(100 * ((X > 0) & (X < 1)).mean(), 1)} % de los píxeles es gris (ni 0 ni 1). No hay caritas repetidas ({len(np.unique(X, axis=0))} distintas de {len(X)}).

### 2b · Entrenamiento

- Encoder {'–'.join(map(str, modelo['capas_encoder'][:-1]))}–(μ y s de {modelo['capas_encoder'][-1]}), decoder {'–'.join(map(str, modelo['capas_decoder']))}, ocultas {modelo['activacion_oculta']}, salida logística. λ = {num(config['lambda'], 1)}, {config['optimizador']['nombre']} con η = {num(config['optimizador']['eta'], 3)}, lotes de {config['tamano_lote']}, {epocas} épocas, semillas {', '.join(map(str, config['semillas']))}.
- L y R durante el entrenamiento (con el ϵ sorteado en cada actualización):

{chr(10).join(lineas)}

- En el último 10 % de las épocas, L cambia {pm(cambio_L)} (media ± desvío entre semillas del valor final menos el inicial del tramo).
- Piso de L para estos datos: {num(bce(np.clip(X, 1e-12, 1 - 1e-12), X, 'media')[0])}. Con datos en grises la BCE no llega a 0 ni con una salida idéntica al dato (vale la entropía de los píxeles grises).
- Al terminar, decodificando z = μ(x) sin ruido: L = {pm(entrenamiento.L_con_z_igual_mu)}; error absoluto por píxel = {pm(entrenamiento.error_absoluto_por_pixel, 3)}; R = {pm(entrenamiento.R)}.
- Códigos de los {len(X)} patrones: desvío de μ = {pm(entrenamiento.mu_desvio)} (el prior tiene desvío 1); σ media = {pm(entrenamiento.sigma_media, 3)}.
- Tiempo por corrida: {pm(entrenamiento.segundos, 1)} s.
"""


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    X = generar_caritas(config['dataset']['n'], config['dataset']['semilla']).X
    figura_dataset(X, 72, salida / 'fig_dataset.png')

    filas, curvas, reconstrucciones = [], [], {}
    for semilla in config['semillas']:
        vae, historial = entrenar_semilla(config, semilla, X, config['lambda'])
        fila = {'semilla': semilla, **evaluar_vae(vae, X), 'segundos': historial.segundos}
        filas.append(fila)
        curvas.append(pd.DataFrame({'semilla': semilla, 'epoca': np.arange(1, config['epocas'] + 1),
                                    'L': historial.reconstruccion, 'R': historial.regularizacion}))
        reconstrucciones[semilla] = vae.decodificar(vae.codificar(X[:12])[0])
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)

    entrenamiento = pd.DataFrame(filas)
    curvas = pd.concat(curvas)
    assert len(entrenamiento) == len(config['semillas'])
    assert (curvas.groupby('semilla').size() == config['epocas']).all()
    entrenamiento.to_csv(salida / 'entrenamiento.csv', index=False)
    curvas.to_csv(salida / 'curvas.csv', index=False)
    figura_curvas(curvas, salida / 'fig_curvas.png')
    figura_reconstrucciones(X, reconstrucciones, 12, salida / 'fig_reconstrucciones.png')
    bloque = texto(config, X, entrenamiento, curvas)
    escribir_seccion('e2b', bloque)
    print(bloque)


if __name__ == '__main__':
    main(sys.argv[1])
