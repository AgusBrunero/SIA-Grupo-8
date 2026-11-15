"""Consigna 2b y 2c: el latente del AE común contra el del VAE, y generación de muestras.

2b  ¿Dónde quedan los datos en el latente de un AE y en el de un VAE entrenados con el
    mismo dataset y la misma arquitectura base? ¿Qué decodifica cada uno sobre el mismo
    rango del latente? ¿Qué cambia con λ (J = L + λR)?
2c  ¿Qué sale de decodificar z ~ N(0, I)? ¿A qué distancia está cada muestra generada de
    la imagen más cercana del dataset?

Sostén: Autoencoders diap. 34 (generar especificando z) y 36–37 (el problema del latente
del AE); diap. 74–78 (supuesto p(z) = N(0, I), KL y J = L + λR, ec. 13); diap. 90–91
(Colab: grilla del latente y muestras). El AE usa BCE y promedio por batch como el VAE,
para que solo cambie el latente (decisión D41).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e2c_latente_generacion config/e2c.json

Salida en results/e2c/: corridas.csv, muestras.csv, candidatas.csv, fig_latente.png,
fig_grilla.png, fig_muestras.png, fig_lambda.png, fig_candidatas_lambda_*.png,
fig_latente_ae_vae.png, fig_grilla_lambdas.png, fig_muestras_ae_vae.png (las tres últimas, con menos paneles, para proyectar),
config.json y el bloque e2c de docs/RESULTADOS.md.
"""
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA_SECUNDARIA, aplicar_estilo, escribir_seccion
from experiments.e2b_vae import dibujar, entrenar_semilla, evaluar_vae, num, pm
from src.data.relojes import LADO, generar_relojes
from src.modelos.autoencoder import Autoencoder, entrenar
from src.modelos.generacion import vecino_por_pixel
from src.nn.optimizadores import crear
from src.nn.perdidas import bce

RADIO = 3.0      # ‖z‖ ≤ 3: tres desvíos del prior N(0, I)


@dataclass
class Modelo:
    """Lo que hace falta de un AE o de un VAE ya entrenado: el código de un dato (z para el
    AE, μ(x) para el VAE) y el decoder."""
    codificar: Callable[[np.ndarray], np.ndarray]
    decodificar: Callable[[np.ndarray], np.ndarray]
    extras: dict


def etiqueta(lam: float | None) -> str:
    return 'AE' if lam is None else f'VAE, λ = {lam:g}'.replace('.', ',')


def entrenar_ae(config: dict, semilla: int, X: np.ndarray) -> Modelo:
    """AE común con la arquitectura base del VAE: mismo decoder, encoder con latente
    identidad de k neuronas, BCE promediada por batch, mismo optimizador y mismos lotes."""
    modelo = config['modelo']
    capas = modelo['capas_encoder']
    assert modelo['capas_decoder'] == capas[::-1], 'el AE es espejo: el decoder tiene que ser el encoder invertido'
    ae = Autoencoder(capas, [modelo['activacion_oculta']] * (len(capas) - 2) + ['identidad'], 'logistica',
                     semilla, beta=config['beta'], amplitud_inicial=config['amplitud_inicial'])
    hiper = dict(config['optimizador'])
    historial = entrenar(ae, X, X, crear(hiper.pop('nombre'), **hiper), config['epocas'], config['tamano_lote'],
                         rng=np.random.default_rng(np.random.SeedSequence([semilla, 1])),
                         perdida='bce', codificacion='01', reduccion='media')
    return Modelo(ae.codificar, ae.decodificar, {'R': np.nan, 'sigma_media': np.nan, 'segundos': historial.segundos})


def entrenar_vae_modelo(config: dict, semilla: int, X: np.ndarray, lam: float) -> Modelo:
    vae, historial = entrenar_semilla(config, semilla, X, lam)
    medidas = evaluar_vae(vae, X)
    return Modelo(lambda datos: vae.codificar(datos)[0], vae.decodificar,
                  {'R': medidas['R'], 'sigma_media': medidas['sigma_media'], 'segundos': historial.segundos})


def medir(modelo: Modelo, X: np.ndarray, horas: np.ndarray, Z_prior: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Medidas de una corrida. Devuelve (fila, códigos, generadas, vecino, distancia).
    horas_cubiertas (medida nuestra, como la distancia de D44): de las 12 horas del reloj (la 1, las 2, …),
    en cuántas cae el reloj del dataset más cercano a al menos una muestra generada. Mide variedad."""
    Z = modelo.codificar(X)
    salida = modelo.decodificar(Z)
    generadas = modelo.decodificar(Z_prior)
    vecino, distancia = vecino_por_pixel(generadas, X)
    dentro = np.all((Z_prior >= Z.min(axis=0)) & (Z_prior <= Z.max(axis=0)), axis=1)
    fila = {'L': bce(salida, X, 'media')[0],
            'error_absoluto_por_pixel': float(np.abs(salida - X).mean()),
            'codigo_desvio': float(Z.std()),
            'codigo_max_abs': float(np.abs(Z).max()),
            'fraccion_codigos_en_radio': float((np.linalg.norm(Z, axis=1) <= RADIO).mean()),
            'fraccion_prior_en_rectangulo': float(dentro.mean()),
            'distancia_vecino_media': float(distancia.mean()),
            'distancia_vecino_minima': float(distancia.min()),
            'horas_cubiertas': len(np.unique(np.floor(horas[vecino]))),
            **modelo.extras}
    return fila, Z, generadas, vecino, distancia


def mosaico(imagenes: np.ndarray, columnas: int) -> np.ndarray:
    """Pega imágenes de 16 × 16 en una sola, por filas, con una línea blanca entre ellas."""
    filas = len(imagenes) // columnas
    paso = LADO + 1
    lienzo = np.zeros((filas * paso - 1, columnas * paso - 1))
    for i, imagen in enumerate(imagenes[:filas * columnas]):
        f, c = divmod(i, columnas)
        lienzo[f * paso:f * paso + LADO, c * paso:c * paso + LADO] = imagen.reshape(LADO, LADO)
    return lienzo


def figura_latente(guardados: dict, horas: np.ndarray, ruta) -> None:
    """El color de cada punto es la hora que marca el reloj, en una escala cíclica (las 12 y las 11:59 casi igual)."""
    fig, ejes = plt.subplots(1, len(guardados), figsize=(3.6 * len(guardados) + 0.8, 3.9), layout='constrained')
    angulo = np.linspace(0, 2 * np.pi, 200)
    for ax, (lam, g) in zip(ejes, guardados.items()):
        puntos = ax.scatter(g['Z'][:, 0], g['Z'][:, 1], s=8, c=horas, cmap='twilight', vmin=0, vmax=12, alpha=0.8, linewidths=0)
        ax.plot(RADIO * np.cos(angulo), RADIO * np.sin(angulo), color=TINTA_SECUNDARIA, linewidth=1, linestyle='--')
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel(f'z₁ · {etiqueta(lam)}')
    ejes[0].set_ylabel('z₂')
    barra = fig.colorbar(puntos, ax=list(ejes), ticks=[0, 3, 6, 9, 12], fraction=0.025, pad=0.02)
    barra.ax.set_yticklabels(['12:00', '3:00', '6:00', '9:00', '12:00'])
    barra.set_label('Hora que marca el reloj')
    fig.supxlabel('Un punto por reloj del dataset (para el VAE, μ(x)). Línea punteada: ‖z‖ = 3, tres desvíos del prior N(0, I).\n'
                  'Cada panel tiene su propia escala.', fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def figura_grilla(guardados: dict, rango: float, n: int, ruta) -> None:
    valores = np.linspace(-rango, rango, n)
    grilla = np.array([[a, b] for b in valores[::-1] for a in valores])      # arriba es z₂ máximo
    fig, ejes = plt.subplots(1, len(guardados), figsize=(3.6 * len(guardados), 4.0))
    marcas = np.arange(0, n, max(1, (n - 1) // 4))
    for ax, (lam, g) in zip(ejes, guardados.items()):
        ax.imshow(mosaico(g['decodificar'](grilla), n), cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                  extent=(0, n, 0, n))
        ax.grid(visible=False)
        ax.set_xticks(marcas + 0.5, [num(valores[i], 1) for i in marcas])
        ax.set_yticks(marcas + 0.5, [num(valores[i], 1) for i in marcas])
        ax.set_xlabel(f'z₁ · {etiqueta(lam)}')
    ejes[0].set_ylabel('z₂')
    fig.supxlabel(f'Salida del decoder en una grilla de {n} × {n} puntos del latente, el mismo rango para todos', fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_muestras(guardados: dict, X: np.ndarray, cantidad: int, ruta) -> None:
    fig, ejes = plt.subplots(2 * len(guardados), cantidad, figsize=(cantidad * 0.85, 2 * len(guardados) * 1.05))
    for bloque, (lam, g) in enumerate(guardados.items()):
        for columna in range(cantidad):
            arriba, abajo = ejes[2 * bloque, columna], ejes[2 * bloque + 1, columna]
            dibujar(arriba, g['generadas'][columna])
            dibujar(abajo, X[g['vecino'][columna]])
            abajo.set_xlabel(num(g['distancia'][columna], 3), fontsize=7, color=TINTA_SECUNDARIA, labelpad=1)
        ejes[2 * bloque, 0].set_ylabel(f'{etiqueta(lam)}\ngenerada', fontsize=8, rotation=0, ha='right', va='center')
        ejes[2 * bloque + 1, 0].set_ylabel('más cercana\ndel dataset', fontsize=8, rotation=0, ha='right', va='center',
                                           color=TINTA_SECUNDARIA)
    fig.supxlabel('Los mismos z ~ N(0, I) para todos los modelos. Debajo de cada par: distancia media por píxel '
                  '(0 = idénticas)', fontsize=10)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.savefig(ruta)
    plt.close(fig)


def figura_lambda(corridas: pd.DataFrame, ruta) -> None:
    paneles = (('L', 'Reconstrucción L con z = μ(x)\n(BCE sumada sobre los píxeles, por patrón)'),
               ('R', 'Regularización R\n(KL, por patrón)'),
               ('distancia_vecino_media', 'Distancia media por píxel de una muestra\ngenerada a la imagen más cercana'))
    vae, ae = corridas[corridas.modelo == 'VAE'], corridas[corridas.modelo == 'AE']
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (columna, nombre) in zip(ejes, paneles):
        ax.scatter(vae['lambda'], vae[columna], s=14, color=COLOR['VAE'], alpha=0.35, linewidths=0)
        resumen = vae.groupby('lambda')[columna].agg(['mean', 'std'])
        ax.errorbar(resumen.index, resumen['mean'], yerr=resumen['std'], color=COLOR['VAE'], marker='o', markersize=6,
                    linewidth=2, capsize=4, label='VAE (media ± desvío entre semillas)')
        if columna != 'R':
            ax.axhline(ae[columna].mean(), color=COLOR['AE'], linewidth=2, label='AE (media entre semillas)')
        ax.set_xscale('log')
        ax.set_xticks(resumen.index, [f'{v:g}'.replace('.', ',') for v in resumen.index])
        ax.set_xlabel('λ (escala log)')
        ax.set_ylabel(nombre)
    for ax in ejes:
        ax.yaxis.set_major_formatter(lambda valor, _: f'{valor:g}'.replace('.', ','))
    ejes[2].legend(fontsize=9, loc='center left')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_candidatas(generadas: np.ndarray, lam: float, ruta) -> None:
    columnas = 8
    filas = len(generadas) // columnas
    fig, ejes = plt.subplots(filas, columnas, figsize=(columnas * 1.0, filas * 1.2))
    for numero, (ax, imagen) in enumerate(zip(ejes.ravel(), generadas), start=1):
        dibujar(ax, imagen)
        ax.set_xlabel(str(numero), fontsize=9, labelpad=1)
    fig.supxlabel(f'Candidatas generadas por el {etiqueta(lam)} desde z ~ N(0, I). El número es el de candidatas.csv', fontsize=10)
    fig.tight_layout(h_pad=0.3, w_pad=0.2)
    fig.savefig(ruta)
    plt.close(fig)


def texto(config: dict, corridas: pd.DataFrame, distancia_dataset: np.ndarray) -> str:
    n_semillas = len(config['semillas'])
    columnas = (('L', 'L con z = código', 2), ('error_absoluto_por_pixel', 'Error abs. por píxel', 3),
                ('codigo_desvio', 'Desvío de los códigos', 2), ('codigo_max_abs', 'Mayor valor absoluto de una coordenada de un código', 2),
                ('fraccion_codigos_en_radio', 'Códigos con ‖z‖ ≤ 3', 2),
                ('fraccion_prior_en_rectangulo', 'z ~ N(0, I) dentro del rectángulo de los códigos', 2),
                ('distancia_vecino_media', 'Distancia media al más cercano', 3),
                ('distancia_vecino_minima', 'Distancia mínima al más cercano', 3),
                ('horas_cubiertas', 'Horas (de 12) en las que cae el más cercano de alguna muestra', 1), ('R', 'R', 2), ('sigma_media', 'σ media', 3))
    grupos = {nombre: g for nombre, g in corridas.groupby('etiqueta', sort=False)}
    assert all(len(g) == n_semillas for g in grupos.values())
    lineas = ['| Medida | ' + ' | '.join(grupos) + ' |', '|---|' + '---|' * len(grupos)]
    for columna, nombre, decimales in columnas:
        celdas = ['—' if g[columna].isna().all() else pm(g[columna], decimales) for g in grupos.values()]
        lineas.append(f'| {nombre} | ' + ' | '.join(celdas) + ' |')
    gen = config['generacion']
    return f"""## 2b y 2c · Latente del AE contra el del VAE, y generación (e2c)

Datos en `results/e2c/`. Figuras: `fig_latente.png`, `fig_grilla.png`, `fig_muestras.png`, `fig_lambda.png`, `fig_candidatas_lambda_*.png`.

- Mismo dataset ({config['dataset']['n']} relojes) y misma arquitectura base ({'–'.join(map(str, config['modelo']['capas_encoder']))} y decoder espejo, ocultas {config['modelo']['activacion_oculta']}, salida logística, BCE, {config['optimizador']['nombre']}, lotes de {config['tamano_lote']}, {config['epocas']} épocas) para el AE común y para el VAE con λ = {', '.join(f'{v:g}'.replace('.', ',') for v in config['lambdas'])}.
- {n_semillas} semillas por modelo; cada celda es media ± desvío entre semillas. El código de un dato es z para el AE y μ(x) para el VAE.
- Generación: {gen['n_muestras']} valores de z ~ N(0, I), los mismos para todos los modelos, decodificados. La distancia entre dos imágenes es la media sobre los píxeles de la diferencia absoluta (0 = idénticas).

{chr(10).join(lineas)}

- Referencia dentro del dataset: la distancia de cada reloj al otro reloj más cercano es {num(distancia_dataset.mean(), 3)} en promedio (mínimo {num(distancia_dataset.min(), 3)}, máximo {num(distancia_dataset.max(), 3)}).
- Ninguna de las {len(corridas) * gen['n_muestras']} muestras generadas (todos los modelos y semillas) es idéntica a una imagen del dataset: la menor distancia es {num(corridas.distancia_vecino_minima.min(), 3)}.
- `fig_candidatas_lambda_*.png` trae {gen['n_candidatas']} candidatas numeradas por cada λ (semilla {config['semilla_figuras']}), con su z, su imagen más cercana y la distancia en `candidatas.csv`. La elección queda para el grupo.
"""


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    gen = config['generacion']
    relojes = generar_relojes(config['dataset']['n'], config['dataset']['semilla'])
    X, horas = relojes.X, relojes.parametros[:, 0]
    Z_prior = np.random.default_rng(gen['semilla']).standard_normal((gen['n_muestras'], config['modelo']['capas_encoder'][-1]))

    filas, muestras, guardados = [], [], {}
    for lam in [None] + config['lambdas']:
        for semilla in config['semillas']:
            modelo = entrenar_ae(config, semilla, X) if lam is None else entrenar_vae_modelo(config, semilla, X, lam)
            fila, Z, generadas, vecino, distancia = medir(modelo, X, horas, Z_prior)
            clave = {'modelo': 'AE' if lam is None else 'VAE', 'lambda': lam, 'etiqueta': etiqueta(lam), 'semilla': semilla}
            filas.append({**clave, **fila})
            muestras.append(pd.DataFrame({**clave, 'muestra': np.arange(1, len(Z_prior) + 1), 'z1': Z_prior[:, 0],
                                          'z2': Z_prior[:, 1], 'vecino': vecino, 'distancia': distancia}))
            if semilla == config['semilla_figuras']:
                guardados[lam] = {'Z': Z, 'decodificar': modelo.decodificar, 'generadas': generadas,
                                  'vecino': vecino, 'distancia': distancia}
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in filas[-1].items()}, flush=True)

    corridas, muestras = pd.DataFrame(filas), pd.concat(muestras)
    assert len(corridas) == (1 + len(config['lambdas'])) * len(config['semillas'])
    assert (muestras.groupby(['etiqueta', 'semilla']).size() == gen['n_muestras']).all()
    corridas.to_csv(salida / 'corridas.csv', index=False)
    muestras.to_csv(salida / 'muestras.csv', index=False)
    candidatas = muestras[(muestras.modelo == 'VAE') & (muestras.semilla == config['semilla_figuras'])
                          & (muestras.muestra <= gen['n_candidatas'])]
    candidatas.rename(columns={'muestra': 'numero'}).to_csv(salida / 'candidatas.csv', index=False)

    figura_latente(guardados, horas, salida / 'fig_latente.png')
    figura_grilla(guardados, gen['rango_grilla'], gen['n_grilla'], salida / 'fig_grilla.png')
    figura_muestras(guardados, X, gen['n_mostradas'], salida / 'fig_muestras.png')
    # Versiones con menos paneles, para proyectar: AE contra el VAE de λ = 1, y solo los tres λ
    ae_y_vae = {lam: guardados[lam] for lam in (None, 1.0) if lam in guardados}
    solo_vae = {lam: guardados[lam] for lam in config['lambdas']}
    if len(ae_y_vae) == 2:
        figura_latente(ae_y_vae, horas, salida / 'fig_latente_ae_vae.png')
        figura_muestras(ae_y_vae, X, gen['n_mostradas'], salida / 'fig_muestras_ae_vae.png')
    figura_grilla(solo_vae, gen['rango_grilla'], gen['n_grilla'], salida / 'fig_grilla_lambdas.png')
    figura_lambda(corridas, salida / 'fig_lambda.png')
    for lam in config['lambdas']:
        figura_candidatas(guardados[lam]['generadas'][:gen['n_candidatas']], lam, salida / f'fig_candidatas_lambda_{lam:g}.png')
    _, distancia_dataset = vecino_por_pixel(X, X, excluir_identicas=True)
    bloque = texto(config, corridas, distancia_dataset)
    escribir_seccion('e2c', bloque)
    print(bloque)


if __name__ == '__main__':
    main(sys.argv[1])
