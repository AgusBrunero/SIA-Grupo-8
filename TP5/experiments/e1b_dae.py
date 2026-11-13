"""Consigna 1b: Denoising Autoencoder sobre font.h.

1b.1  ¿Qué arquitectura conviene para sacar ruido: con cuello de botella de 2, de 10, o
      sin cuello de botella? (decisión D2: arquitectura libre)
1b.2  ¿Cuánto ruido elimina el autoencoder según el nivel con que se entrenó y el nivel
      con que se lo prueba? Matriz nivel de entrenamiento × nivel de prueba.
La fila "entrenado con p = 0" de la arquitectura A es el autoencoder de 1a, evaluado con
el mismo protocolo (comparación justa).

Ruido: salt-and-pepper, cada píxel se invierte con probabilidad p (Autoencoders diap. 22).
Entrenamiento: entrada X̃ con ruido nuevo en cada época, salida esperada X (diap. 22).
Evaluación: siempre con realizaciones de ruido nuevas; para una semilla y un nivel de
prueba dados, todas las arquitecturas y niveles de entrenamiento ven el mismo ruido.
Sostén: Autoencoders diap. 6 y 20–22; receta del escalón E9 de e1a2.

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1b_dae config/e1b.json            (entrena, evalúa y analiza)
    .venv/bin/python -m experiments.e1b_dae config/e1b.json analisis   (solo tablas, figuras y texto)

Salida en results/e1b/: entrenamiento.csv, celdas.csv, confusion.csv, resumen.csv,
fig_*.png, config.json y el bloque e1b de docs/RESULTADOS.md.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from experiments.comun import RAIZ, TINTA, TINTA_SECUNDARIA, aplicar_estilo, escribir_seccion
from experiments.e1a2_escalera import construir
from src.data.font import cargar_font, distancias_hamming
from src.modelos.autoencoder import entrenar, evaluar
from src.modelos.dae import corruptor, evaluar_con_ruido
from src.modelos.metricas import resumir
from src.nn.optimizadores import crear

RAMPA_DAE = ['#fff8f4', '#fbd9c9', '#f6b294', '#f08a5d', '#eb6834', '#c44f20', '#963b16', '#6b290e']
N_PIXELES = 35


def correr(config: dict, salida) -> None:
    X, _ = cargar_font()
    niveles = config['niveles']
    entrenamientos, celdas, confusiones = [], [], []
    for arquitectura in config['arquitecturas']:
        modelo = dict(config['modelo'], capas_encoder=arquitectura['capas_encoder'])
        for p_entrenamiento in niveles:
            for semilla in config['semillas']:
                ae = construir(config, modelo, semilla)
                hiper = dict(modelo['optimizador'])
                optimizador = crear(hiper.pop('nombre'), **hiper)
                rng_lotes = np.random.default_rng(np.random.SeedSequence([semilla, 1]))
                rng_ruido = np.random.default_rng(np.random.SeedSequence([semilla, 2]))
                historial = entrenar(ae, X, X, optimizador, config['epocas'], modelo['tamano_lote'], rng_lotes,
                                     perdida=modelo['perdida'], codificacion='01', reduccion=config['reduccion'],
                                     corromper=corruptor(p_entrenamiento, rng_ruido))
                error, incorrectos = evaluar(ae, X, X, modelo['perdida'], '01')
                limpio = resumir(incorrectos)
                clave = {'arquitectura': arquitectura['id'], 'p_entrenamiento': p_entrenamiento, 'semilla': semilla}
                entrenamientos.append({**clave, 'error_sin_ruido': error, 'max_pixeles_sin_ruido': limpio.maximo,
                                       'total_pixeles_sin_ruido': limpio.total, 'segundos': historial.segundos})
                for indice, p_prueba in enumerate(niveles):
                    # Mismo ruido de prueba para toda arquitectura y nivel de entrenamiento
                    rng_prueba = np.random.default_rng(np.random.SeedSequence([semilla, 3, indice]))
                    tabla = evaluar_con_ruido(ae, X, p_prueba, config['realizaciones'], rng_prueba)
                    assert len(tabla) == len(X) * config['realizaciones']
                    celdas.append({**clave, 'p_prueba': p_prueba, 'n': len(tabla),
                                   'alterados_media': tabla.alterados.mean(), 'mal_media': tabla.mal.mean(),
                                   'exactos': (tabla.mal == 0).mean(), 'hasta_1px': (tabla.mal <= 1).mean(),
                                   'identidad_correcta': (tabla.predicho == tabla.patron).mean()})
                    conteo = tabla.groupby(['patron', 'predicho']).size().rename('cantidad').reset_index()
                    confusiones.append(conteo.assign(arquitectura=arquitectura['id'],
                                                     p_entrenamiento=p_entrenamiento, p_prueba=p_prueba))
                print(f"{arquitectura['id']} p_entrenamiento={p_entrenamiento:<4} semilla {semilla}  "
                      f"sin ruido: {limpio.total} px mal  {historial.segundos:.1f} s", flush=True)
    pd.DataFrame(entrenamientos).to_csv(salida / 'entrenamiento.csv', index=False)
    pd.DataFrame(celdas).to_csv(salida / 'celdas.csv', index=False)
    (pd.concat(confusiones).groupby(['arquitectura', 'p_entrenamiento', 'p_prueba', 'patron', 'predicho'])
     .cantidad.sum().reset_index().to_csv(salida / 'confusion.csv', index=False))


def resumir_celdas(config: dict, celdas: pd.DataFrame) -> pd.DataFrame:
    grupos = celdas.groupby(['arquitectura', 'p_entrenamiento', 'p_prueba'])
    # Conteo: cada celda tiene una fila por semilla, y cada fila las mismas reconstrucciones
    assert (grupos.size() == len(config['semillas'])).all()
    assert len(grupos) == len(config['arquitecturas']) * len(config['niveles']) ** 2
    assert (celdas.n == 32 * config['realizaciones']).all()
    resumen = grupos.agg(
        semillas=('semilla', 'size'), alterados=('alterados_media', 'mean'),
        mal_media=('mal_media', 'mean'), mal_desvio=('mal_media', 'std'),
        exactos_media=('exactos', 'mean'), exactos_desvio=('exactos', 'std'),
        hasta_1px_media=('hasta_1px', 'mean'), hasta_1px_desvio=('hasta_1px', 'std'),
        identidad_media=('identidad_correcta', 'mean'), identidad_desvio=('identidad_correcta', 'std'),
    ).reset_index()
    return resumen


# ---------- figuras ----------

def figura_matrices(config: dict, resumen: pd.DataFrame, columna: str, etiqueta: str, porcentaje: bool, ruta) -> None:
    """Una matriz nivel de entrenamiento × nivel de prueba por arquitectura."""
    niveles = config['niveles']
    fig, ejes = plt.subplots(1, len(config['arquitecturas']), figsize=(4.4 * len(config['arquitecturas']), 4.2))
    maximo = 100.0 if porcentaje else float(resumen[columna].max())
    for ax, arquitectura in zip(ejes, config['arquitecturas']):
        datos = resumen[resumen.arquitectura == arquitectura['id']].pivot(
            index='p_entrenamiento', columns='p_prueba', values=columna).loc[niveles, niveles].to_numpy()
        if porcentaje:
            datos = 100.0 * datos
        imagen = ax.imshow(datos, cmap=LinearSegmentedColormap.from_list('dae', RAMPA_DAE), vmin=0, vmax=maximo)
        for i in range(len(niveles)):
            for j in range(len(niveles)):
                texto = f'{datos[i, j]:.0f}' if porcentaje else f'{datos[i, j]:.1f}'.replace('.', ',')
                ax.text(j, i, texto, ha='center', va='center', fontsize=10,
                        color='white' if datos[i, j] > 0.55 * maximo else TINTA)
        ax.set_xticks(range(len(niveles)))
        ax.set_yticks(range(len(niveles)))
        ax.set_xticklabels([str(p).replace('.', ',') for p in niveles])
        ax.set_yticklabels([str(p).replace('.', ',') for p in niveles])
        ax.grid(visible=False)
        ax.set_xlabel(f"p de prueba\n{arquitectura['id']}: {arquitectura['nombre']}")
    ejes[0].set_ylabel('p de entrenamiento')
    barra = fig.colorbar(imagen, ax=ejes, fraction=0.025, pad=0.02)
    barra.set_label(etiqueta)
    fig.savefig(ruta)
    plt.close(fig)


def figura_ganancia(config: dict, resumen: pd.DataFrame, ruta) -> None:
    """Píxeles mal a la salida contra el nivel de prueba; una línea por nivel de entrenamiento."""
    niveles = config['niveles']
    colores = [RAMPA_DAE[i] for i in (2, 3, 4, 6, 7)]
    fig, ejes = plt.subplots(1, len(config['arquitecturas']), figsize=(4.4 * len(config['arquitecturas']), 4), sharey=True)
    for ax, arquitectura in zip(ejes, config['arquitecturas']):
        ax.plot(niveles, [N_PIXELES * p for p in niveles], color=TINTA_SECUNDARIA, linestyle='--', linewidth=1.5,
                label='Entrada ruidosa, sin red (35·p)')
        for p_entrenamiento, color in zip(niveles, colores):
            fila = resumen[(resumen.arquitectura == arquitectura['id']) & (resumen.p_entrenamiento == p_entrenamiento)]
            fila = fila.set_index('p_prueba').loc[niveles]
            ax.errorbar(niveles, fila.mal_media, yerr=fila.mal_desvio, color=color, linewidth=2, marker='o',
                        markersize=5, capsize=3, label=f'Entrenado con p = {str(p_entrenamiento).replace(".", ",")}')
        ax.set_xticks(niveles)
        ax.set_xticklabels([str(p).replace('.', ',') for p in niveles])
        ax.set_xlabel(f"p de prueba\n{arquitectura['id']}: {arquitectura['nombre']}")
    ejes[0].set_ylabel('Píxeles incorrectos a la salida, por patrón\n(media ± desvío entre semillas)')
    manijas, rotulos = ejes[0].get_legend_handles_labels()
    fig.legend(manijas, rotulos, loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_confusion(config: dict, confusion: pd.DataFrame, etiquetas: list[str], ruta) -> None:
    """Fila: letra original. Columna: letra del dataset más cercana a la salida. Solo errores."""
    p_e, p_p = config['confusion']['p_entrenamiento'], config['confusion']['p_prueba']
    fig, ejes = plt.subplots(1, len(config['arquitecturas']), figsize=(6.2 * len(config['arquitecturas']), 6.6))
    for ax, arquitectura in zip(ejes, config['arquitecturas']):
        c = confusion[(confusion.arquitectura == arquitectura['id']) & (confusion.p_entrenamiento == p_e)
                      & (confusion.p_prueba == p_p)]
        matriz = np.zeros((32, 32))
        matriz[c.patron, c.predicho] = c.cantidad
        np.fill_diagonal(matriz, 0)
        total = len(config['semillas']) * config['realizaciones']
        imagen = ax.imshow(100 * matriz / total, cmap=LinearSegmentedColormap.from_list('dae', RAMPA_DAE),
                           vmin=0, vmax=config['confusion']['maximo_porcentaje'])
        ax.set_xticks(range(32))
        ax.set_yticks(range(32))
        ax.set_xticklabels(etiquetas, fontsize=7, rotation=90)
        ax.set_yticklabels(etiquetas, fontsize=7)
        ax.grid(visible=False)
        ax.set_xlabel(f"Letra más cercana a la salida\n{arquitectura['id']}: {arquitectura['nombre']}")
    ejes[0].set_ylabel('Letra original')
    barra = fig.colorbar(imagen, ax=ejes, fraction=0.015, pad=0.02)
    barra.set_label(f'% de las {total} reconstrucciones de esa letra (sin la diagonal)')
    fig.savefig(ruta)
    plt.close(fig)


# ---------- texto generado desde los datos ----------

def pct(media: float, desvio: float) -> str:
    return f'{100 * media:.1f} ± {100 * desvio:.1f}'.replace('.', ',')


def num(valor: float, decimales: int = 2) -> str:
    return f'{valor:.{decimales}f}'.replace('.', ',')


def tabla_md(config: dict, resumen: pd.DataFrame, arquitectura: str, media: str, desvio: str, porcentaje: bool) -> str:
    niveles = config['niveles']
    lineas = ['| p de entrenamiento \\ p de prueba | ' + ' | '.join(str(p).replace('.', ',') for p in niveles) + ' |',
              '|---|' + '---|' * len(niveles)]
    r = resumen[resumen.arquitectura == arquitectura].set_index(['p_entrenamiento', 'p_prueba'])
    for p_e in niveles:
        celdas = [pct(r.loc[(p_e, p_p), media], r.loc[(p_e, p_p), desvio]) if porcentaje
                  else f'{num(r.loc[(p_e, p_p), media])} ± {num(r.loc[(p_e, p_p), desvio])}' for p_p in niveles]
        lineas.append(f'| {str(p_e).replace(".", ",")} | ' + ' | '.join(celdas) + ' |')
    return '\n'.join(lineas)


def texto(config, entrenamiento, resumen, confusion, etiquetas) -> str:
    niveles = config['niveles']
    total_celda = len(config['semillas']) * 32 * config['realizaciones']
    partes = [f"""## 1b · Denoising Autoencoder (e1b)

{len(config['arquitecturas'])} arquitecturas × {len(niveles)} niveles de entrenamiento × {len(config['semillas'])} semillas ({config['epocas']} épocas, receta de E9), cada una probada con {len(niveles)} niveles de ruido. Ruido salt-and-pepper: cada píxel se invierte con probabilidad p. Cada celda son {len(config['semillas'])} semillas × 32 patrones × {config['realizaciones']} realizaciones de ruido nuevas = {total_celda} reconstrucciones; se informa media ± desvío entre semillas. Datos en `results/e1b/`.

Figuras: `fig_hasta_1px.png`, `fig_exactos.png`, `fig_pixeles_mal.png` (matrices entrenamiento × prueba), `fig_ganancia.png` (píxeles mal a la salida contra píxeles alterados a la entrada), `fig_confusion.png`.

Píxeles alterados a la entrada, en promedio por patrón: {', '.join(f"p = {str(p).replace('.', ',')}: {num(35 * p)}" for p in niveles)} (35·p).

### Sin ruido en la prueba

"""]
    lineas = ['| Arquitectura | p de entrenamiento | Corridas con ≤ 1 px en los 32 patrones limpios | Px mal en total (media ± desvío) |', '|---|---|---|---|']
    for a in config['arquitecturas']:
        for p_e in niveles:
            e = entrenamiento[(entrenamiento.arquitectura == a['id']) & (entrenamiento.p_entrenamiento == p_e)]
            lineas.append(f"| {a['id']} ({a['nombre']}) | {str(p_e).replace('.', ',')} | {int((e.max_pixeles_sin_ruido <= 1).sum())} de {len(e)} | "
                          f"{num(e.total_pixeles_sin_ruido.mean(), 1)} ± {num(e.total_pixeles_sin_ruido.std(), 1)} |")
    partes.append('\n'.join(lineas) + '\n')
    for a in config['arquitecturas']:
        partes.append(f"\n### Arquitectura {a['id']}: {a['nombre']}\n\n% de reconstrucciones con ≤ 1 píxel incorrecto:\n\n"
                      + tabla_md(config, resumen, a['id'], 'hasta_1px_media', 'hasta_1px_desvio', True)
                      + '\n\nPíxeles incorrectos a la salida, por patrón:\n\n'
                      + tabla_md(config, resumen, a['id'], 'mal_media', 'mal_desvio', False) + '\n')
    # Comparación por nivel de prueba: mejor nivel de entrenamiento de cada arquitectura y el AE de 1a
    r = resumen.set_index(['arquitectura', 'p_entrenamiento', 'p_prueba'])
    lineas = ['| p de prueba | Entrada (px alterados) | AE de 1a (A, p = 0) | ' +
              ' | '.join(f"{a['id']}: mejor nivel de entrenamiento" for a in config['arquitecturas']) + ' |',
              '|---|---|---|' + '---|' * len(config['arquitecturas'])]
    for p_p in niveles[1:]:
        fila = [str(p_p).replace('.', ','), num(35 * p_p), f"{num(r.loc[('A', 0.0, p_p), 'mal_media'])} ± {num(r.loc[('A', 0.0, p_p), 'mal_desvio'])}"]
        for a in config['arquitecturas']:
            sub = resumen[(resumen.arquitectura == a['id']) & (resumen.p_prueba == p_p)]
            mejor = sub.loc[sub.mal_media.idxmin()]
            fila.append(f"{num(mejor.mal_media)} ± {num(mejor.mal_desvio)} (p = {str(mejor.p_entrenamiento).replace('.', ',')})")
        lineas.append('| ' + ' | '.join(fila) + ' |')
    partes.append('\n### Píxeles incorrectos a la salida por nivel de prueba\n\nPara cada arquitectura, el nivel de entrenamiento con menos píxeles mal en ese nivel de prueba (entre paréntesis).\n\n'
                  + '\n'.join(lineas) + '\n')
    # Confusiones
    p_e, p_p = config['confusion']['p_entrenamiento'], config['confusion']['p_prueba']
    lineas = []
    for a in config['arquitecturas']:
        c = confusion[(confusion.arquitectura == a['id']) & (confusion.p_entrenamiento == p_e) & (confusion.p_prueba == p_p)]
        errores = c[c.patron != c.predicho].sort_values('cantidad', ascending=False)
        por_letra = len(config['semillas']) * config['realizaciones']
        principales = '; '.join(f"'{etiquetas[f.patron]}' → '{etiquetas[f.predicho]}': {f.cantidad} de {por_letra}" for f in errores.head(5).itertuples())
        l_barra = int(c[(c.patron == etiquetas.index('l')) & (c.predicho == etiquetas.index('|'))].cantidad.sum())
        barra_l = int(c[(c.patron == etiquetas.index('|')) & (c.predicho == etiquetas.index('l'))].cantidad.sum())
        lineas.append(f"- **{a['id']}:** {int(errores.cantidad.sum())} de {int(c.cantidad.sum())} reconstrucciones quedan más cerca de otra letra que de la original. "
                      f"Más frecuentes: {principales}. 'l' → '|': {l_barra} de {por_letra}; '|' → 'l': {barra_l} de {por_letra}.")
    partes.append(f"\n### Confusiones entre letras (entrenado con p = {str(p_e).replace('.', ',')}, probado con p = {str(p_p).replace('.', ',')})\n\n"
                  "Se asigna a cada salida umbralizada la letra del dataset más cercana por distancia de Hamming. 'l' y '|' difieren en 2 píxeles.\n\n"
                  + '\n'.join(lineas) + '\n')
    return ''.join(partes)


def analizar(config: dict, salida) -> None:
    aplicar_estilo()
    _, etiquetas = cargar_font()
    entrenamiento = pd.read_csv(salida / 'entrenamiento.csv')
    celdas = pd.read_csv(salida / 'celdas.csv')
    confusion = pd.read_csv(salida / 'confusion.csv')
    assert len(entrenamiento) == len(config['arquitecturas']) * len(config['niveles']) * len(config['semillas'])
    resumen = resumir_celdas(config, celdas)
    resumen.to_csv(salida / 'resumen.csv', index=False)
    figura_matrices(config, resumen, 'hasta_1px_media', '% de reconstrucciones con ≤ 1 píxel incorrecto', True, salida / 'fig_hasta_1px.png')
    figura_matrices(config, resumen, 'exactos_media', '% de reconstrucciones exactas (0 píxeles incorrectos)', True, salida / 'fig_exactos.png')
    figura_matrices(config, resumen, 'mal_media', 'Píxeles incorrectos a la salida, por patrón', False, salida / 'fig_pixeles_mal.png')
    figura_ganancia(config, resumen, salida / 'fig_ganancia.png')
    figura_confusion(config, confusion, etiquetas, salida / 'fig_confusion.png')
    bloque = texto(config, entrenamiento, resumen, confusion, etiquetas)
    escribir_seccion('e1b', bloque)
    print(bloque)


def main(ruta_config: str, modo: str = 'todo') -> None:
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    if modo == 'todo':
        shutil.copyfile(ruta_config, salida / 'config.json')
        correr(config, salida)
    analizar(config, salida)


if __name__ == '__main__':
    main(*sys.argv[1:])
