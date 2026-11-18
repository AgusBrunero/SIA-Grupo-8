"""Consigna 1b rehecha: Denoising Autoencoder con las dos mejores redes del AE.

Pregunta: con las dos primeras redes del orden de 1a.1 (eleccion.csv; D57 y D59), entrenadas
con ruido gaussiano o salt-and-pepper de un nivel dado, ¿cuánto ruido sacan cuando se las prueba
con el MISMO tipo y el MISMO nivel de ruido? ¿Cómo evoluciona eso época a época?

Ruido (Autoencoders diap. 22): gaussiano X̃ = X + σ·N(0, 1), sin recortar; salt-and-pepper,
cada píxel se invierte con probabilidad p. Los niveles los fija el config (D58).
Entrenamiento: entrada X̃ con ruido nuevo en cada época, salida esperada X (diap. 22). Se
entrena en tramos de `evaluar_cada` épocas: el optimizador y los generadores se conservan,
así que es el mismo entrenamiento que de un tirón; al final de cada tramo se prueba con
`realizaciones_curva` versiones ruidosas nuevas de cada letra.
Prueba final: `realizaciones_final` versiones ruidosas nuevas, del mismo tipo y nivel.
Los gráficos muestran solo el error a la salida (píxeles mal por letra) de cada nivel, con el
mismo ruido en entrenamiento y prueba (D61, feedback de la cátedra).
Sostén: Autoencoders diap. 20–22; redes y receta de D57/D59 (Autoencoders diap. 3–6,
Clase 10.2 diap. 23, Clase 12.1 diap. 11–13, Autoencoders diap. 51).

Uso, desde TP5/ y en primer plano:
    python3 -m experiments.e1b_ruidos config/e1b_ruidos.json <modelo> <tipo> <nivel>   (entrena una parte)
    python3 -m experiments.e1b_ruidos config/e1b_ruidos.json analisis                   (tablas, figuras y texto)

Salida de cada parte en results/e1b_ruidos/<modelo>/<tipo>_<nivel>/: corridas.csv y curvas.csv.
El análisis escribe en results/e1b_ruidos/: resumen.csv, curva_media.csv, fig_barras.png,
fig_curvas.png, config.json y el bloque e1b_ruidos de docs/RESULTADOS.md.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, aplicar_estilo, escribir_seccion
from experiments.e1a2_escalera import construir
from experiments.e1b_dae import RAMPA_DAE
from src.data.font import cargar_font
from src.data.ruido import RUIDOS
from src.modelos.autoencoder import entrenar, evaluar
from src.modelos.dae import corruptor, evaluar_con_ruido
from src.nn.optimizadores import crear

SALIDA = RAIZ / 'results' / 'e1b_ruidos'
NOMBRES = {'gaussiano': 'Gaussiano', 'sal_y_pimienta': 'Salt-and-pepper'}
SIMBOLO = {'gaussiano': 'σ', 'sal_y_pimienta': 'p'}
# Un tono del DAE por red, en el orden del config (la primera es la elegida para el AE)
TONOS = [COLOR['DAE'], RAMPA_DAE[6]]


def carpeta(modelo: str, tipo: str, nivel: float):
    return SALIDA / modelo / f'{tipo}_{nivel:g}'


def nombre_red(config: dict, modelo: str) -> str:
    capas = config['modelos'][modelo]['capas_encoder']
    return f"{modelo} {'–'.join(map(str, capas + capas[-2::-1]))}"


def medir(ae, X, nivel, realizaciones, rng, tipo) -> dict:
    tabla = evaluar_con_ruido(ae, X, nivel, realizaciones, rng, tipo)
    assert len(tabla) == len(X) * realizaciones
    return {'alterados_media': tabla.alterados.mean(), 'mal_media': tabla.mal.mean(),
            'entrada_exacta': (tabla.alterados == 0).mean(), 'exactos': (tabla.mal == 0).mean(),
            'hasta_1px': (tabla.mal <= 1).mean(), 'identidad_correcta': (tabla.predicho == tabla.patron).mean()}


def correr_parte(config: dict, modelo_id: str, tipo: str, nivel: float) -> None:
    assert modelo_id in config['modelos'] and tipo in RUIDOS and nivel in config['ruidos'][tipo], (modelo_id, tipo, nivel)
    salida = carpeta(modelo_id, tipo, nivel)
    salida.mkdir(parents=True, exist_ok=True)
    X, _ = cargar_font()
    modelo = config['modelos'][modelo_id]
    tramos = config['epocas'] // config['evaluar_cada']
    assert tramos * config['evaluar_cada'] == config['epocas']
    corridas, curvas = [], []
    for semilla in config['semillas']:
        ae = construir(config, modelo, semilla)
        hiper = dict(modelo['optimizador'])
        optimizador = crear(hiper.pop('nombre'), **hiper)
        rng_lotes = np.random.default_rng(np.random.SeedSequence([semilla, 1]))
        rng_ruido = np.random.default_rng(np.random.SeedSequence([semilla, 2]))
        rng_prueba = np.random.default_rng(np.random.SeedSequence([semilla, 3]))
        rng_curva = np.random.default_rng(np.random.SeedSequence([semilla, 4]))
        corromper = corruptor(nivel, rng_ruido, tipo)
        clave = {'modelo': modelo_id, 'tipo': tipo, 'nivel': nivel, 'semilla': semilla}
        segundos = 0.0
        curvas.append({**clave, 'epoca': 0, **medir(ae, X, nivel, config['realizaciones_curva'], rng_curva, tipo)})
        for tramo in range(1, tramos + 1):
            historial = entrenar(ae, X, X, optimizador, config['evaluar_cada'], modelo['tamano_lote'], rng_lotes,
                                 perdida=modelo['perdida'], codificacion=config['codificacion'],
                                 reduccion=config['reduccion'], corromper=corromper)
            segundos += historial.segundos
            curvas.append({**clave, 'epoca': tramo * config['evaluar_cada'],
                           **medir(ae, X, nivel, config['realizaciones_curva'], rng_curva, tipo)})
        final = medir(ae, X, nivel, config['realizaciones_final'], rng_prueba, tipo)
        _, incorrectos = evaluar(ae, X, X, modelo['perdida'], config['codificacion'])
        corridas.append({**clave, **final, 'total_px_sin_ruido': int(incorrectos.sum()),
                         'max_px_sin_ruido': int(incorrectos.max()), 'segundos': segundos})
        print(f"{modelo_id} {tipo} {SIMBOLO[tipo]}={nivel:g} semilla {semilla}: entrada {final['alterados_media']:.2f} px mal, "
              f"salida {final['mal_media']:.2f} px mal  {segundos:.1f} s", flush=True)
    corridas = pd.DataFrame(corridas)
    assert len(corridas) == len(config['semillas']) and not corridas.duplicated('semilla').any()
    corridas.to_csv(salida / 'corridas.csv', index=False)
    pd.DataFrame(curvas).to_csv(salida / 'curvas.csv', index=False)


def leer(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    corridas, curvas = [], []
    for modelo in config['modelos']:
        for tipo, niveles in config['ruidos'].items():
            for nivel in niveles:
                corridas.append(pd.read_csv(carpeta(modelo, tipo, nivel) / 'corridas.csv'))
                curvas.append(pd.read_csv(carpeta(modelo, tipo, nivel) / 'curvas.csv'))
    corridas, curvas = pd.concat(corridas, ignore_index=True), pd.concat(curvas, ignore_index=True)
    validar_conteos(config, corridas, curvas)
    return corridas, curvas


def validar_conteos(config: dict, corridas: pd.DataFrame, curvas: pd.DataFrame) -> None:
    """Una corrida por (modelo, tipo, nivel, semilla) y un punto de curva por época evaluada."""
    semillas = len(config['semillas'])
    celdas = len(config['modelos']) * sum(len(n) for n in config['ruidos'].values())
    claves = ['modelo', 'tipo', 'nivel']
    assert len(corridas) == celdas * semillas
    assert not corridas.duplicated(claves + ['semilla']).any()
    assert (corridas.groupby(claves).size() == semillas).all()
    puntos = config['epocas'] // config['evaluar_cada'] + 1
    assert len(curvas) == celdas * semillas * puntos
    assert (curvas.groupby(claves + ['semilla']).size() == puntos).all()
    assert not curvas.duplicated(claves + ['semilla', 'epoca']).any()


def resumir(config: dict, corridas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (modelo, tipo, nivel): media y desvío entre semillas, y en cuántas semillas la
    salida tiene menos píxeles mal que la entrada con ruido."""
    columnas = ['alterados_media', 'mal_media', 'entrada_exacta', 'exactos', 'hasta_1px', 'identidad_correcta',
                'total_px_sin_ruido']
    g = corridas.groupby(['modelo', 'tipo', 'nivel'], sort=False)
    tabla = g[columnas].mean().add_suffix('_m').join(g[columnas].std().add_suffix('_d'))
    tabla['semillas'] = g.size()
    tabla['semillas_que_mejoran'] = g.apply(lambda d: int((d.mal_media < d.alterados_media).sum()), include_groups=False)
    tabla = tabla.reset_index()
    assert (tabla.semillas == len(config['semillas'])).all()
    return tabla


def figura_barras(config: dict, resumen: pd.DataFrame, ruta) -> None:
    """Solo el error a la salida de cada nivel (mismo ruido en entrenamiento y prueba), una barra por red."""
    modelos = list(config['modelos'])
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={'width_ratios': [len(n) for n in config['ruidos'].values()]})
    ancho = 0.8 / len(modelos)
    for ax, tipo in zip(ejes, config['ruidos']):
        niveles = config['ruidos'][tipo]
        x = np.arange(len(niveles))
        for k, (modelo, color) in enumerate(zip(modelos, TONOS)):
            r = resumen[(resumen.modelo == modelo) & (resumen.tipo == tipo)].set_index('nivel').loc[niveles]
            m, d = r.mal_media_m, r.mal_media_d
            ax.bar(x + (k - (len(modelos) - 1) / 2) * ancho, m, ancho, yerr=[np.minimum(d, m), d], capsize=3,
                   color=color, label=nombre_red(config, modelo))
        ax.set_xticks(x, [f'{SIMBOLO[tipo]} = {coma(n, None)}' for n in niveles])
        ax.set_xlabel(f'Ruido {NOMBRES[tipo].lower()} (el mismo en entrenamiento y prueba)')
        ax.grid(axis='x', visible=False)
    ejes[0].set_ylabel('Píxeles incorrectos por letra a la salida\n(media ± desvío, 10 semillas)')
    fig.legend(*ejes[0].get_legend_handles_labels(), loc='upper center', ncol=len(modelos), bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_curvas(config: dict, curvas: pd.DataFrame, ruta) -> pd.DataFrame:
    """Px mal por letra con el mismo ruido, época a época: media entre semillas ± desvío. Filas: red; columnas: ruido."""
    media = curvas.groupby(['modelo', 'tipo', 'nivel', 'epoca']).agg(
        mal_m=('mal_media', 'mean'), mal_d=('mal_media', 'std')).reset_index()
    modelos = list(config['modelos'])
    fig, ejes = plt.subplots(len(modelos), 2, figsize=(14, 4.2 * len(modelos)), sharey=True, sharex=True, squeeze=False)
    for fila, modelo in enumerate(modelos):
        for columna, tipo in enumerate(config['ruidos']):
            ax = ejes[fila, columna]
            niveles = config['ruidos'][tipo]
            for nivel, color in zip(niveles, RAMPA_DAE[-len(niveles):]):
                c = media[(media.modelo == modelo) & (media.tipo == tipo) & (media.nivel == nivel)]
                ax.plot(c.epoca, c.mal_m, color=color, lw=1.8, label=f'{SIMBOLO[tipo]} = {coma(nivel, None)}')
                ax.fill_between(c.epoca, np.maximum(c.mal_m - c.mal_d, 0), c.mal_m + c.mal_d, color=color, alpha=0.18, lw=0)
            # Lineal hasta 0,01 (las curvas que llegan a 0 px) y logarítmica arriba
            ax.set_yscale('symlog', linthresh=0.01, linscale=0.5)
            ax.set_ylim(0, 30)
            ax.legend(title=f'{nombre_red(config, modelo)} · ruido {NOMBRES[tipo].lower()}', fontsize=9, title_fontsize=9, ncol=2)
            if fila == len(modelos) - 1:
                ax.set_xlabel('Época de entrenamiento')
            if columna == 0:
                ax.set_ylabel('Píxeles incorrectos por letra a la salida\n(mismo ruido; lineal hasta 0,01, log arriba)')
    fig.supxlabel('Media entre 10 semillas, banda ± desvío. Prueba cada 250 épocas con el mismo tipo y nivel de ruido del entrenamiento.',
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)
    return media


def coma(valor: float, decimales: int | None = 2) -> str:
    """Número con coma decimal; decimales=None lo escribe corto (0,05)."""
    return (f'{valor:g}' if decimales is None else f'{valor:.{decimales}f}').replace('.', ',')


def mejora(fila) -> str:
    """Semillas en que la salida tiene menos px mal que la entrada; sin sentido si la entrada no tiene ninguno."""
    if fila.alterados_media_m == 0:
        return '— (la entrada no tiene píxeles alterados)'
    return f'{fila.semillas_que_mejoran} de {fila.semillas}'


def pm(m: float, d: float, decimales: int = 2) -> str:
    return f'{m:.{decimales}f} ± {d:.{decimales}f}'.replace('.', ',')


def pct(m: float, d: float) -> str:
    return f'{100 * m:.1f} ± {100 * d:.1f} %'.replace('.', ',')


def texto(config: dict, resumen: pd.DataFrame, media: pd.DataFrame) -> str:
    bloques = []
    for modelo in config['modelos']:
        for tipo in config['ruidos']:
            lineas = [f'### {nombre_red(config, modelo)} · {NOMBRES[tipo]}', '',
                      f'| {SIMBOLO[tipo]} | Px alterados en la entrada | Px mal a la salida | Semillas en que la salida mejora la entrada | '
                      'Entradas exactas | Salidas exactas | Salidas con ≤ 1 px | Reconocida como la letra correcta |',
                      '|---|---|---|---|---|---|---|---|']
            for f in resumen[(resumen.modelo == modelo) & (resumen.tipo == tipo)].itertuples():
                lineas.append(f'| {coma(f.nivel, None)} | {pm(f.alterados_media_m, f.alterados_media_d)} | {pm(f.mal_media_m, f.mal_media_d)} | '
                              f'{mejora(f)} | {pct(f.entrada_exacta_m, f.entrada_exacta_d)} | '
                              f'{pct(f.exactos_m, f.exactos_d)} | {pct(f.hasta_1px_m, f.hasta_1px_d)} | '
                              f'{pct(f.identidad_correcta_m, f.identidad_correcta_d)} |')
            lineas += ['', 'Época a época (curva media entre semillas, px mal por letra con el mismo ruido):', '']
            for nivel in config['ruidos'][tipo]:
                c = media[(media.modelo == modelo) & (media.tipo == tipo) & (media.nivel == nivel)].set_index('epoca')
                minimo = c.mal_m.idxmin()
                epocas = sorted({config['evaluar_cada'], config['epocas'] // 10, config['epocas'] // 2, config['epocas']} & set(c.index))
                puntos = ', '.join(f'época {e}: {coma(c.mal_m[e])}' for e in epocas)
                lineas.append(f'- {SIMBOLO[tipo]} = {coma(nivel, None)}: {puntos}; mínimo {coma(c.mal_m[minimo])} '
                              f'en la época {minimo}.')
            bloques.append('\n'.join(lineas))
    redes = '; '.join(f"{nombre_red(config, m)} (ocultas {v['activacion_oculta']}, latente {v['activacion_latente']}, "
                      f"BCE, Adam η = {coma(v['optimizador']['eta'], None)})" for m, v in config['modelos'].items())
    return f"""## 1b rehecha · DAE con las dos mejores redes del AE, probado con el mismo ruido (e1b_ruidos)

Redes: las dos primeras del orden de 1a.1 (`results/e1a2_arquitecturas/eleccion.csv`, D59): {redes}. {config['epocas']} épocas, {len(config['semillas'])} semillas (0 a {config['semillas'][-1]}), ruido nuevo en cada época. **Cada modelo se prueba con el mismo tipo y nivel de ruido con que se entrenó**, sobre {config['realizaciones_final']} versiones ruidosas nuevas de cada letra ({config['realizaciones_final'] * 32} por corrida). Media ± desvío entre semillas. "Px alterados en la entrada": píxeles de la entrada con ruido, umbralizada en 0,5, distintos de la letra (lo que se obtiene sin hacer nada). Datos en `results/e1b_ruidos/`; figuras `fig_barras.png` y `fig_curvas.png`, que muestran solo el error a la salida (D61).

{chr(10).join(chr(10) + b for b in bloques)}
"""


def analizar(config: dict) -> None:
    aplicar_estilo()
    corridas, curvas = leer(config)
    resumen = resumir(config, corridas)
    resumen.to_csv(SALIDA / 'resumen.csv', index=False)
    figura_barras(config, resumen, SALIDA / 'fig_barras.png')
    media = figura_curvas(config, curvas, SALIDA / 'fig_curvas.png')
    media.to_csv(SALIDA / 'curva_media.csv', index=False)
    escribir_seccion('e1b_ruidos', texto(config, resumen, media))
    print(resumen[['modelo', 'tipo', 'nivel', 'alterados_media_m', 'mal_media_m', 'mal_media_d', 'semillas_que_mejoran',
                   'exactos_m']].to_string(index=False))


def main(ruta_config: str, *argumentos: str) -> None:
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    SALIDA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, SALIDA / 'config.json')
    if argumentos == ('analisis',):
        analizar(config)
    else:
        modelo, tipo, nivel = argumentos
        correr_parte(config, modelo, tipo, float(nivel))


if __name__ == '__main__':
    main(*sys.argv[1:])
