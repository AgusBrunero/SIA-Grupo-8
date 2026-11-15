"""Consigna 2c · AE contra VAE en generación, con una comparación justa (punto 3 de la última
iteración de la presentación).

Preguntas:
  (a) ¿Qué decodifica cada modelo sobre su propio latente? Grilla de 11 × 11 que cubre, en
      cada modelo, del mínimo al máximo de sus propios códigos (z en el AE, μ(x) en el VAE).
  (b) Una muestra generada, ¿es un reloj? ¿Es una copia? Se compara cada muestra con un
      banco de 20.000 relojes nuevos del mismo generador (otra semilla, ninguno igual a los
      500) y con los 500 de entrenamiento. Referencia: la misma medida para 200 relojes reales
      que no están entre los 500 ni en el banco.
  (c) ¿Qué codifica cada eje del VAE? Correlación de Pearson de μ₁ y μ₂ con el seno y el coseno
      del ángulo de la hora (la hora es cíclica: las 12:00 y las 11:59 son casi la misma imagen).

Muestreo (decisión en docs/DECISIONES.md): el VAE decodifica z ~ N(0, I), su prior; el AE
decodifica z ~ N(media, covarianza) de sus propios 500 códigos. Cada modelo se muestrea de
donde viven sus códigos, con la misma cantidad de muestras y la misma medida.
Distancia: media sobre los píxeles de |a − b| (vecino_por_pixel, la misma de e2c; nuestra).
Sostén: Autoencoders diap. 33–37 (generar especificando z; el problema del latente del AE)
y 74–78 (prior p(z) = N(0, I), J = L + λR).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e2c_generacion_justa config/e2c_justa.json            (entrena, mide y analiza)
    .venv/bin/python -m experiments.e2c_generacion_justa config/e2c_justa.json analisis   (solo tablas y figuras)

Salida en results/e2c_justa/: distancias.csv, resumen.csv, correlaciones.csv,
fig_grillas_propias.png, fig_distancias.png, fig_correlaciones.png, config.json.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, RAMPA_AE, TINTA, TINTA_SECUNDARIA, aplicar_estilo
from experiments.e2c_latente_generacion import entrenar_ae, entrenar_vae_modelo, mosaico
from src.data.relojes import generar_relojes
from src.modelos.generacion import grilla_latente, vecino_por_pixel

ROTULOS = {'hora_seno': 'Hora: seno\ndel ángulo', 'hora_coseno': 'Hora: coseno\ndel ángulo'}
REALES = 'Relojes reales'


def factores(parametros: np.ndarray) -> dict[str, np.ndarray]:
    """La hora (columna 0, en [0, 12)) como ángulo: seno y coseno, para que las 12:00 y las 11:59 queden juntas."""
    angulo = 2 * np.pi * parametros[:, 0] / 12.0
    return {'hora_seno': np.sin(angulo), 'hora_coseno': np.cos(angulo)}


def muestrear(modelo: str, Z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """z para generar: el prior N(0, I) del VAE o la normal ajustada a los códigos del AE."""
    if modelo == 'VAE':
        return rng.standard_normal((n, Z.shape[1]))
    return rng.multivariate_normal(Z.mean(axis=0), np.cov(Z, rowvar=False), size=n)


def banco_sin_repetidas(config: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Banco y referencia sin imágenes idénticas a una de las 500 (ni la referencia idéntica al banco).

    Con relojes de 16 × 16, dos horas a menos de un minuto pueden dar la misma imagen: esas se
    sacan, para que "copia" quiera decir lo mismo que antes (idéntica a una de las 500).
    """
    banco = generar_relojes(config['banco']['n'], config['banco']['semilla']).X
    referencia = generar_relojes(config['referencia']['n'], config['referencia']['semilla']).X
    _, d = vecino_por_pixel(banco, X)
    banco = banco[d > 0]
    _, d_500 = vecino_por_pixel(referencia, X)
    _, d_banco = vecino_por_pixel(referencia, banco)
    referencia = referencia[(d_500 > 0) & (d_banco > 0)]
    print(f'banco: {len(banco)} relojes; referencia: {len(referencia)} (sin imágenes idénticas a las 500)', flush=True)
    return banco, referencia


def distancias(imagenes: np.ndarray, banco: np.ndarray, X: np.ndarray) -> pd.DataFrame:
    vecino_banco, d_banco = vecino_por_pixel(imagenes, banco)
    vecino_500, d_500 = vecino_por_pixel(imagenes, X)
    return pd.DataFrame({'d_banco': d_banco, 'vecino_banco': vecino_banco, 'd_500': d_500, 'vecino_500': vecino_500})


def resumir(tabla: pd.DataFrame, config: dict) -> pd.DataFrame:
    generadas = tabla[tabla.modelo != REALES]
    # Conteo: n_muestras por (modelo, semilla); la referencia, una sola vez
    assert (generadas.groupby(['modelo', 'semilla']).size() == config['generacion']['n_muestras']).all()
    # La referencia puede tener menos que config['referencia']['n']: se sacan las idénticas a las 500
    assert 0 < (tabla.modelo == REALES).sum() <= config['referencia']['n']
    real = tabla[tabla.modelo == REALES]
    # Umbral sin elegir a mano: el reloj real (fuera de los 500) más lejano del banco
    umbral = float(real.d_banco.max())
    por_semilla = generadas.groupby(['modelo', 'semilla']).agg(
        d_banco=('d_banco', 'mean'), d_500=('d_500', 'mean'),
        d_banco_med=('d_banco', 'median'), d_500_med=('d_500', 'median'),
        fuera=('d_banco', lambda d: float((d > umbral).mean()))).reset_index()
    filas = []
    for modelo, g in por_semilla.groupby('modelo', sort=False):
        filas.append({'modelo': modelo, 'semillas': len(g), 'muestras': len(g) * config['generacion']['n_muestras'],
                      'd_banco_media': g.d_banco.mean(), 'd_banco_desvio': g.d_banco.std(),
                      'd_500_media': g.d_500.mean(), 'd_500_desvio': g.d_500.std(),
                      'd_banco_mediana_media': g.d_banco_med.mean(), 'd_banco_mediana_desvio': g.d_banco_med.std(),
                      'd_500_mediana_media': g.d_500_med.mean(), 'd_500_mediana_desvio': g.d_500_med.std(),
                      'fuera_media': g.fuera.mean(), 'fuera_desvio': g.fuera.std()})
    # Los relojes reales son una sola muestra de 200: el "desvío" es entre relojes, no entre semillas
    filas.append({'modelo': REALES, 'semillas': np.nan, 'muestras': len(real),
                  'd_banco_media': real.d_banco.mean(), 'd_banco_desvio': np.nan,
                  'd_500_media': real.d_500.mean(), 'd_500_desvio': np.nan,
                  'd_banco_mediana_media': real.d_banco.median(), 'd_banco_mediana_desvio': np.nan,
                  'd_500_mediana_media': real.d_500.median(), 'd_500_mediana_desvio': np.nan,
                  'fuera_media': float((real.d_banco > umbral).mean()), 'fuera_desvio': np.nan})
    resumen = pd.DataFrame(filas).assign(umbral_max_reales=umbral)
    a, v = resumen.set_index('modelo').loc['AE'], resumen.set_index('modelo').loc['VAE']
    for medida in ('d_banco', 'd_500', 'd_banco_mediana', 'd_500_mediana', 'fuera'):
        diferencia = abs(a[f'{medida}_media'] - v[f'{medida}_media'])
        resumen[f'{medida}_ae_vs_vae'] = 'empatan' if diferencia <= max(a[f'{medida}_desvio'], v[f'{medida}_desvio']) else (
            'AE menor' if a[f'{medida}_media'] < v[f'{medida}_media'] else 'VAE menor')
    return resumen


def correlaciones(mus: dict[int, np.ndarray], parametros: np.ndarray) -> pd.DataFrame:
    filas = []
    for semilla, mu in mus.items():
        for k in range(mu.shape[1]):
            for nombre, valores in factores(parametros).items():
                filas.append({'semilla': semilla, 'eje': f'z{k + 1}', 'parametro': nombre,
                              'r': float(np.corrcoef(mu[:, k], valores)[0, 1])})
    tabla = pd.DataFrame(filas)
    assert (tabla.groupby(['eje', 'parametro']).size() == len(mus)).all()
    return tabla


# ---------- figuras ----------

def figura_grillas(guardados: dict, n: int, ruta) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    for ax, (modelo, g) in zip(ejes, guardados.items()):
        grilla = grilla_latente(g['Z'], n)
        ax.imshow(mosaico(g['decodificar'](grilla), n), cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                  extent=(0, n, 0, n))
        ax.grid(visible=False)
        z1, z2 = g['Z'][:, 0], g['Z'][:, 1]
        marcas = [0, n // 2, n - 1]
        ax.set_xticks([m + 0.5 for m in marcas], [f'{v:.1f}'.replace('.', ',') for v in np.linspace(z1.min(), z1.max(), n)[marcas]])
        ax.set_yticks([m + 0.5 for m in marcas], [f'{v:.1f}'.replace('.', ',') for v in np.linspace(z2.min(), z2.max(), n)[marcas]])
        ax.set_xlabel(f'z₁ · {modelo}')
        ax.set_ylabel('z₂')
    fig.supxlabel(f'Grilla de {n} × {n} puntos entre el mínimo y el máximo de los códigos de cada modelo '
                  '(z en el AE, μ(x) en el VAE). Cada panel tiene su propia escala.', fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_distancias(tabla: pd.DataFrame, umbral: float, ruta) -> None:
    colores = {'AE': COLOR['AE'], 'VAE': COLOR['VAE'], REALES: TINTA_SECUNDARIA}
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    for ax, (columna, nombre) in zip(ejes, (('d_banco', 'a la más cercana del banco\nde 20.000 relojes nuevos (¿es un reloj?)'),
                                            ('d_500', 'al más cercano de los\n500 de entrenamiento (¿es una copia?)'))):
        bordes = np.geomspace(tabla[columna].min() * 0.98, tabla[columna].max() * 1.02, 50)
        ax.set_xscale('log')
        for modelo, color in colores.items():
            datos = tabla[tabla.modelo == modelo][columna]
            ax.hist(datos, bins=bordes, density=True, histtype='step', linewidth=2, color=color,
                    label=f'{modelo} ({len(datos)})')
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5], ['0,01', '0,02', '0,05', '0,1', '0,2', '0,5'])
        ax.set_xlabel(f'Distancia media por píxel {nombre}\n(escala log)')
    ejes[0].axvline(umbral, color=TINTA, linestyle=':', linewidth=1.2, label='El reloj real más lejano del banco')
    ejes[0].set_ylabel('Densidad')
    ejes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_correlaciones(tabla: pd.DataFrame, ruta) -> None:
    """|r| por semilla y eje: el orden de los ejes cambia entre semillas, por eso no se promedia."""
    matriz = (tabla.assign(abs_r=tabla.r.abs()).pivot_table(index=['semilla', 'eje'], columns='parametro', values='abs_r')
              [list(ROTULOS)])
    fig, ax = plt.subplots(figsize=(6, 5.2))
    imagen = ax.imshow(matriz.to_numpy(), cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list('vae', ['#ffffff', COLOR['VAE']]),
                       vmin=0, vmax=1, aspect='auto')
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, f'{matriz.iat[i, j]:.2f}'.replace('.', ','), ha='center', va='center', fontsize=9,
                    color='white' if matriz.iat[i, j] > 0.6 else TINTA)
    ax.set_xticks(range(matriz.shape[1]), [ROTULOS[p] for p in matriz.columns], fontsize=9)
    ax.set_yticks(range(matriz.shape[0]), [f'semilla {s} · μ{"₁" if e == "z1" else "₂"}' for s, e in matriz.index], fontsize=9)
    for borde in range(2, matriz.shape[0], 2):
        ax.axhline(borde - 0.5, color='white', linewidth=3)
    ax.grid(visible=False)
    fig.colorbar(imagen, ax=ax, fraction=0.03, pad=0.02).set_label('|r| de Pearson entre μₖ(x)\ny la hora, sobre los 500 relojes')
    fig.savefig(ruta)
    plt.close(fig)


def por_semilla_relojes(tabla: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Medidas propias de los relojes, por modelo y semilla:
      horas_cubiertas   cuántas de las 12 horas aparecen entre las muestras (la hora entera del
                        reloj de las 500 más cercano a cada muestra; la medida de e2c, con muestreo justo)
      peor_vae          la distancia al banco de la peor muestra del VAE (todas las semillas)
      lejos_peor_vae    fracción de muestras más lejos del banco que la peor muestra del VAE
    El umbral "más lejos que el reloj real más lejano" no sirve con relojes: el banco es tan denso
    que los relojes reales quedan a ≈ 0 de él, y todas las muestras generadas lo superan."""
    horas = generar_relojes(config['dataset']['n'], config['dataset']['semilla']).parametros[:, 0]
    generadas = tabla[tabla.modelo != REALES].copy()
    generadas['hora'] = np.floor(horas[generadas.vecino_500.astype(int)]).astype(int)
    peor = float(generadas[generadas.modelo == 'VAE'].d_banco.max())
    medidas = generadas.groupby(['modelo', 'semilla']).agg(
        horas_cubiertas=('hora', 'nunique'), copias=('d_500', lambda d: int((d == 0).sum())),
        lejos_peor_vae=('d_banco', lambda d: float((d > peor).mean()))).reset_index()
    assert (medidas.groupby('modelo').size() == len(config['semillas'])).all()
    return medidas.assign(peor_vae=peor)


def fallas(tabla: pd.DataFrame, reconstrucciones: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fracción de muestras generadas que "fallan": quedan más lejos del banco que un umbral sacado de
    las reconstrucciones de relojes reales nuevos del mismo modelo y semilla (percentil 95, 99 o máximo).
    El umbral no depende de las muestras que se juzgan."""
    filas = []
    for (modelo, semilla), rec in reconstrucciones.groupby(['modelo', 'semilla']):
        generadas = tabla[(tabla.modelo == modelo) & (tabla.semilla == semilla)].d_banco
        assert len(generadas) == config['generacion']['n_muestras']
        for nombre, umbral in (('p95', rec.d_banco.quantile(0.95)), ('p99', rec.d_banco.quantile(0.99)), ('max', rec.d_banco.max())):
            filas.append({'modelo': modelo, 'semilla': semilla, 'umbral': nombre, 'valor_umbral': float(umbral),
                          'reconstrucciones': len(rec), 'fallan': float((generadas > umbral).mean())})
    tabla_fallas = pd.DataFrame(filas)
    assert (tabla_fallas.groupby(['modelo', 'umbral']).size() == len(config['semillas'])).all()
    return tabla_fallas


def analizar(config: dict, salida) -> None:
    """Tablas y figuras de (b) y (c) a partir de los CSV, sin reentrenar."""
    tabla = pd.read_csv(salida / 'distancias.csv')
    corr = pd.read_csv(salida / 'correlaciones.csv')
    resumen = resumir(tabla, config)
    resumen.to_csv(salida / 'resumen.csv', index=False)
    relojes = por_semilla_relojes(tabla, config)
    if (salida / 'reconstrucciones.csv').exists():
        fallas(tabla, pd.read_csv(salida / 'reconstrucciones.csv'), config).to_csv(salida / 'fallas.csv', index=False)
    relojes.to_csv(salida / 'relojes.csv', index=False)
    print(relojes.groupby('modelo')[['horas_cubiertas', 'copias', 'lejos_peor_vae']].agg(['mean', 'std']).round(3).to_string())
    figura_distancias(tabla, float(resumen.umbral_max_reales.iloc[0]), salida / 'fig_distancias.png')
    figura_correlaciones(corr, salida / 'fig_correlaciones.png')
    print(resumen.T.to_string())
    print(corr.assign(abs_r=corr.r.abs()).groupby(['eje', 'parametro']).abs_r.agg(['mean', 'std']).round(2).to_string())


def main(ruta_config: str, modo: str = 'todo') -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    if modo == 'analisis':
        analizar(config, salida)
        return
    shutil.copyfile(ruta_config, salida / 'config.json')
    relojes = generar_relojes(config['dataset']['n'], config['dataset']['semilla'])
    X = relojes.X
    banco, referencia = banco_sin_repetidas(config, X)
    # Relojes reales nuevos para reconstruir: miden qué tan lejos del banco queda lo que cada modelo
    # hace con un reloj real (umbral de falla que no sale de las muestras que se juzgan)
    rec = config['reconstruccion']
    reales = generar_relojes(rec['n'], rec['semilla']).X
    reales = reales[vecino_por_pixel(reales, X)[1] > 0]
    reconstrucciones, imagenes = [], {}
    gen = config['generacion']

    partes, guardados, mus = [], {}, {}
    for modelo in ('AE', 'VAE'):
        for semilla in config['semillas']:
            m = entrenar_ae(config, semilla, X) if modelo == 'AE' else entrenar_vae_modelo(config, semilla, X, config['lambda'])
            Z = m.codificar(X)
            rng = np.random.default_rng(np.random.SeedSequence([gen['semilla'], semilla]))
            z = muestrear(modelo, Z, gen['n_muestras'], rng)
            d = distancias(m.decodificar(z), banco, X)
            partes.append(d.assign(modelo=modelo, semilla=semilla, muestra=np.arange(1, len(z) + 1), z1=z[:, 0], z2=z[:, 1]))
            if modelo == 'VAE':
                mus[semilla] = Z
            _, d_rec = vecino_por_pixel(m.decodificar(m.codificar(reales)), banco)
            reconstrucciones.append(pd.DataFrame({'modelo': modelo, 'semilla': semilla, 'd_banco': d_rec}))
            if semilla == config['semilla_figuras']:
                guardados[modelo] = {'Z': Z, 'decodificar': m.decodificar}
                imagenes[modelo] = m.decodificar(z)
            print(f'{modelo} semilla {semilla}: d_banco {d.d_banco.mean():.4f}  d_500 {d.d_500.mean():.4f}', flush=True)
    partes.append(distancias(referencia, banco, X).assign(modelo=REALES, semilla=np.nan,
                                                          muestra=np.arange(1, len(referencia) + 1)))
    pd.concat(partes, ignore_index=True).to_csv(salida / 'distancias.csv', index=False)
    pd.concat(reconstrucciones, ignore_index=True).to_csv(salida / 'reconstrucciones.csv', index=False)
    # Muestras generadas de la semilla de figuras (para mostrar relojes del AE y del VAE)
    np.savez_compressed(salida / 'generadas_semilla_figuras.npz', **imagenes)
    correlaciones(mus, relojes.parametros).to_csv(salida / 'correlaciones.csv', index=False)
    figura_grillas(guardados, gen['n_grilla'], salida / 'fig_grillas_propias.png')
    analizar(config, salida)


if __name__ == '__main__':
    main(*sys.argv[1:])
