"""Consigna 1a.2: arquitecturas y técnicas de optimización del autoencoder con latente 2D.

Pregunta: ¿qué configuración aprende los 32 patrones de font.h con latente de 2 dimensiones
y a lo sumo 1 píxel incorrecto por patrón, y qué aporta cada cambio? Se corre una escalera
de configuraciones (config/e1a2.json): cada escalón cambia una sola cosa respecto de otro
escalón, y se repite con todas las semillas del config.

Sostén: Autoencoders diap. 3–6 (arquitectura y ζ = X); Clase 10.2 (activaciones);
Clase 11 diap. 37 y 52–54 (error, épocas, batch/minibatch/online); Clase 12.1 diap. 5–13
(momentum, η adaptativo, RMSProp, Adam); Análisis de Datos diap. 9 y OptimizacionNOLineal
diap. 35–37 (Powell).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1a2_escalera config/e1a2.json

Salida en results/<experimento>/:
    corridas.csv   una fila por (escalón, semilla): si logró ≤ 1 px, en qué época, error,
                   máximo y total de píxeles, tiempo
    curvas.csv     error y píxeles por época (una fila cada `curva_cada` épocas)
    patrones.csv   por corrida y patrón: píxeles mal y posición (z1, z2) en el latente
    config.json    copia del config usado
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.font import a_bipolar, cargar_font
from src.modelos.autoencoder import Autoencoder, Historial, entrenar, evaluar
from src.modelos.metricas import binarizar, resumir, validar_codificacion
from src.nn.optimizadores import EtaAdaptativo, crear
from src.nn.perdidas import PERDIDAS
from src.nn.scipy_adapter import minimizar_powell

RAIZ = Path(__file__).resolve().parents[1]

# Un píxel mal está saturado si su salida quedó a más de esto del valor esperado (datos en {0, 1})
UMBRAL_SATURACION = 0.99

# Lo único que puede diferir entre un escalón y el escalón del que parte
CLAVES_COMPARABLES = ('capas_encoder', 'activacion_oculta', 'activacion_latente', 'activacion_salida',
                      'perdida', 'optimizador', 'eta_adaptativo', 'tamano_lote')


def cargar_datos(codificacion: str) -> tuple[np.ndarray, list[str]]:
    X, etiquetas = cargar_font()
    return (X if codificacion == '01' else a_bipolar(X)), etiquetas


def construir(config: dict, escalon: dict, semilla: int) -> Autoencoder:
    capas = escalon['capas_encoder']
    activaciones = [escalon['activacion_oculta']] * (len(capas) - 2) + [escalon['activacion_latente']]
    return Autoencoder(capas, activaciones, escalon['activacion_salida'], semilla,
                       beta=config['beta'], amplitud_inicial=config['amplitud_inicial'])


def correr(config: dict, escalon: dict, semilla: int, X: np.ndarray,
           etiquetas: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Una corrida. Devuelve (fila de corridas.csv, curva por época, tabla por patrón)."""
    # La pérdida es la del config, salvo que el escalón la cambie
    perdida, codificacion = escalon.get('perdida', config['perdida']), config['codificacion']
    validar_codificacion(escalon['activacion_salida'], codificacion)
    ae = construir(config, escalon, semilla)
    hiper = dict(escalon['optimizador'])
    nombre = hiper.pop('nombre')
    evaluaciones = iteraciones_powell = convergio_powell = None
    if nombre == 'powell':
        resultado = minimizar_powell(
            ae.redes, lambda: PERDIDAS[perdida](ae.reconstruir(X), X, 'suma')[0], **hiper)
        historial = Historial(segundos=resultado.segundos)
        evaluaciones = resultado.evaluaciones
        iteraciones_powell, convergio_powell = resultado.iteraciones, resultado.convergio
    else:
        optimizador = crear(nombre, **hiper)
        adaptativo = (EtaAdaptativo(optimizador, **escalon['eta_adaptativo'])
                      if escalon.get('eta_adaptativo') else None)
        # Semilla del orden de los lotes: derivada de la del experimento, distinta de la de los pesos
        rng = np.random.default_rng(np.random.SeedSequence([semilla, 1]))
        historial = entrenar(ae, X, X, optimizador, config['epocas'], escalon['tamano_lote'], rng,
                             perdida=perdida, codificacion=codificacion,
                             reduccion=config['reduccion'], eta_adaptativo=adaptativo)

    error, incorrectos = evaluar(ae, X, X, perdida, codificacion)
    resumen = resumir(incorrectos)
    # Saturación de la salida: píxeles mal cuya salida quedó pegada al valor opuesto, y la
    # θ' más grande entre los píxeles mal (solo con salida logística: θ' = 2βO(1 − O)).
    # Con error cuadrático el delta de salida es (ζ − O)θ'(h): si θ' ≈ 0 el píxel no aprende.
    salida = ae.reconstruir(X)
    mal = binarizar(salida, codificacion) != binarizar(X, codificacion)
    saturados = int((mal & (np.abs(salida - X) > UMBRAL_SATURACION)).sum())
    theta_prima_max = np.nan
    if escalon['activacion_salida'] == 'logistica' and mal.any():
        theta_prima_max = float((2.0 * config['beta'] * salida * (1.0 - salida))[mal].max())
    fila = {
        'escalon': escalon['id'], 'semilla': semilla, 'logro': resumen.cumple,
        'epoca_objetivo': historial.epoca_objetivo, 'error_final': error,
        'max_pixeles': resumen.maximo, 'total_pixeles': resumen.total,
        'patrones_con_error': resumen.patrones_con_error,
        'pixeles_mal_saturados': saturados, 'theta_prima_max_en_mal': theta_prima_max,
        'segundos': historial.segundos,
        'epocas': len(historial.error), 'evaluaciones': evaluaciones,
        'powell_iteraciones': iteraciones_powell, 'powell_convergio': convergio_powell,
        'n_pesos': ae.n_params(),
    }
    paso = config['curva_cada']
    epocas = np.arange(1, len(historial.error) + 1)
    guardar = (epocas % paso == 0) | (epocas == 1)
    curva = pd.DataFrame({
        'escalon': escalon['id'], 'semilla': semilla, 'epoca': epocas[guardar],
        'error': np.array(historial.error)[guardar],
        'max_pixeles': np.array(historial.max_pixeles, dtype=int)[guardar],
        'total_pixeles': np.array(historial.total_pixeles, dtype=int)[guardar],
        'eta': np.array(historial.eta)[guardar],
    })
    Z = ae.codificar(X)
    patrones = pd.DataFrame({
        'escalon': escalon['id'], 'semilla': semilla, 'patron': np.arange(len(etiquetas)),
        'etiqueta': etiquetas, 'pixeles_mal': incorrectos, 'z1': Z[:, 0], 'z2': Z[:, 1],
    })
    return fila, curva, patrones


def main(ruta_config: str) -> None:
    ruta = Path(ruta_config)
    config = json.loads(ruta.read_text(encoding='utf-8'))
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta, salida / 'config.json')
    X, etiquetas = cargar_datos(config['codificacion'])

    filas, curvas, patrones = [], [], []
    inicio = time.perf_counter()
    for escalon in config['escalones']:
        for semilla in config['semillas']:
            fila, curva, tabla = correr(config, escalon, semilla, X, etiquetas)
            filas.append(fila)
            curvas.append(curva)
            patrones.append(tabla)
            print(f"{escalon['id']:<28} semilla {semilla:>3}  logro={fila['logro']!s:<5} "
                  f"max_px={fila['max_pixeles']:>2}  total_px={fila['total_pixeles']:>3}  "
                  f"error={fila['error_final']:.4f}  {fila['segundos']:.1f} s", flush=True)

    corridas = pd.DataFrame(filas)
    # Conteo: exactamente una corrida por (escalón, semilla)
    esperadas = len(config['escalones']) * len(config['semillas'])
    assert len(corridas) == esperadas and not corridas.duplicated(['escalon', 'semilla']).any()
    corridas.to_csv(salida / 'corridas.csv', index=False)
    pd.concat(curvas).to_csv(salida / 'curvas.csv', index=False)
    pd.concat(patrones).to_csv(salida / 'patrones.csv', index=False)
    print(f'{esperadas} corridas en {time.perf_counter() - inicio:.0f} s → {salida}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
