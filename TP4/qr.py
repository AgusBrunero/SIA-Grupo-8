"""Ej. 2.1: QR deteriorados en Hopfield; Hebb y mejora opcional con pseudoinversa.

Instalar requirements-qr.txt y ejecutar: OPENBLAS_NUM_THREADS=1 python qr.py
El lector ZXing verifica el texto del QR; no participa del entrenamiento ni la dinámica.
Salida: resultados/hopfield/qr/. No se modifica ninguna imagen de entrada.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import zxingcpp

from graficos import AZUL, DIVERGENTE, NARANJA, guardar, plt
from models.hopfield import Hopfield

BASE = Path(__file__).parent
SALIDA = BASE / "resultados" / "hopfield" / "qr"
REPETICIONES = 30
REPETICIONES_ESTRUCTURADO = 10
RUIDOS = [0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
LADOS = [5, 9, 13]
REGLAS = ["hebb", "pseudoinverse"]


def cargar():
    archivos = sorted((BASE / "datos" / "hopfield").glob("*_menu_qr.png"))
    if len(archivos) != 5:
        raise ValueError("Se esperan los cinco archivos *_menu_qr.png")
    nombres, patrones = [], []
    for archivo in archivos:
        imagen = np.asarray(Image.open(archivo).convert("L"))
        if imagen.shape != (37, 37) or not np.isin(imagen, [0, 255]).all():
            raise ValueError(f"{archivo.name}: se espera una matriz binaria 37×37")
        nombres.append(archivo.name.removesuffix("_menu_qr.png"))
        patrones.append(np.where(imagen == 0, -1.0, 1.0).ravel())
    return nombres, np.array(patrones)


def imagen_lector(estado):
    """Negro=−1, blanco=+1; cuatro módulos de borde y escala entera sin interpolación."""
    pixeles = np.where(np.asarray(estado).reshape(37, 37) == -1, 0, 255).astype(np.uint8)
    con_borde = np.pad(pixeles, 4, constant_values=255)
    return np.repeat(np.repeat(con_borde, 8, axis=0), 8, axis=1)


@lru_cache(maxsize=4096)
def _leer(clave):
    estado = np.frombuffer(clave, dtype=np.int8)
    resultados = zxingcpp.read_barcodes(imagen_lector(estado), formats=zxingcpp.BarcodeFormat.QRCode)
    return tuple(resultado.text for resultado in resultados)


def leer(estado):
    return _leer(np.asarray(estado, dtype=np.int8).tobytes())


def deteriorar(patron, tipo, nivel, semilla):
    """Devuelve consulta y máscara afectada; el borrado blanco puede dejar bits intactos."""
    rng = np.random.default_rng(semilla)
    estado = np.asarray(patron).reshape(37, 37).copy()
    mascara = np.zeros((37, 37), dtype=bool)
    if tipo == "aleatorio":
        mascara.flat[rng.choice(estado.size, int(round(nivel * estado.size)), replace=False)] = True
        estado[mascara] *= -1
    elif tipo == "esquina_blanca":
        lado = int(nivel)
        fila = 0 if rng.integers(2) == 0 else 37 - lado
        columna = 0 if rng.integers(2) == 0 else 37 - lado
        mascara[fila:fila + lado, columna:columna + lado] = True
        estado[mascara] = 1
    elif tipo == "mancha_negra":
        lado = int(nivel)
        fila, columna = rng.integers(0, 38 - lado, size=2)
        mascara[fila:fila + lado, columna:columna + lado] = True
        estado[mascara] = -1
    elif tipo == "raya_blanca":
        # Una raya de ancho nivel, horizontal o vertical, ubicada al azar.
        ancho = int(nivel)
        inicio = rng.integers(0, 38 - ancho)
        if rng.integers(2):
            mascara[inicio:inicio + ancho, :] = True
        else:
            mascara[:, inicio:inicio + ancho] = True
        estado[mascara] = 1
    else:
        raise ValueError(f"Deterioro desconocido: {tipo}")
    return estado.ravel(), mascara.ravel()


def estabilidad(nombres, patrones, modelos):
    coincidencias = np.all(patrones == patrones[0], axis=0).sum()
    solapamientos = patrones @ patrones.T / patrones.shape[1]
    pd.DataFrame(solapamientos, index=nombres, columns=nombres).to_csv(SALIDA / "solapamientos.csv")
    filas = []
    for regla, modelo in modelos.items():
        for nombre, patron in zip(nombres, patrones):
            recuperado = modelo.recall(patron, seed=0)
            filas.append({"regla": regla, "qr": nombre, "memorias": len(patrones),
                          "punto_fijo": modelo.is_fixed_point(patron),
                          "bits_modificados": int(np.count_nonzero(patron != recuperado.state)),
                          "barridos": recuperado.sweeps,
                          "energia_inicial": recuperado.energies[0],
                          "energia_final": recuperado.energies[-1]})
    # Control: una sola memoria con Hebb para distinguir implementación de interferencia.
    for nombre, patron in zip(nombres, patrones):
        modelo = Hopfield().fit([patron])
        filas.append({"regla": "hebb_una_memoria", "qr": nombre, "memorias": 1,
                      "punto_fijo": modelo.is_fixed_point(patron), "bits_modificados": 0,
                      "barridos": modelo.recall(patron).sweeps,
                      "energia_inicial": modelo.energy(patron), "energia_final": modelo.energy(patron)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / "estabilidad.csv", index=False)
    print(f"Píxeles iguales en los cinco: {coincidencias}/{patrones.shape[1]}", flush=True)
    print(tabla.to_string(index=False), flush=True)


def ensayos():
    for nivel in RUIDOS:
        yield "aleatorio", nivel, REPETICIONES
    for tipo in ["esquina_blanca", "mancha_negra", "raya_blanca"]:
        for nivel in LADOS:
            yield tipo, nivel, REPETICIONES_ESTRUCTURADO


def experimentos(nombres, patrones, modelos, textos):
    filas, energias, ejemplos = [], [], {}
    for condicion, (tipo, nivel, repeticiones) in enumerate(ensayos()):
        for indice, (nombre, patron) in enumerate(zip(nombres, patrones)):
            for repeticion in range(repeticiones):
                semilla = condicion * 10000 + indice * 100 + repeticion
                consulta, mascara = deteriorar(patron, tipo, nivel, semilla)
                leidos_entrada = leer(consulta)
                for regla, modelo in modelos.items():
                    resultado = modelo.recall(consulta, seed=semilla)
                    assert np.all(np.diff(resultado.energies) <= 1e-8)
                    assert resultado.converged and modelo.is_fixed_point(resultado.state)
                    coincidencias = np.flatnonzero(np.all(patrones == resultado.state, axis=1))
                    inversos = np.any(np.all(-patrones == resultado.state, axis=1))
                    clase = ("correcto" if len(coincidencias) and coincidencias[0] == indice
                             else "otra_memoria" if len(coincidencias)
                             else "inverso_memoria" if inversos else "espurio")
                    leidos_salida = leer(resultado.state)
                    filas.append({"regla": regla, "qr": nombre, "deterioro": tipo, "nivel": nivel,
                                  "repeticion": repeticion, "semilla": semilla,
                                  "pixeles_afectados": int(mascara.sum()),
                                  "bits_invertidos": int(np.count_nonzero(consulta != patron)),
                                  "fraccion_invertida": float(np.mean(consulta != patron)),
                                  "resultado": clase, "barridos": resultado.sweeps,
                                  "actualizaciones": len(resultado.energies) - 1,
                                  "distancia_final": int(np.count_nonzero(resultado.state != patron)),
                                  "convergio": resultado.converged,
                                  "energia_inicial": resultado.energies[0],
                                  "energia_final": resultado.energies[-1],
                                  "entrada_decodifica_objetivo": textos[nombre] in leidos_entrada,
                                  "salida_decodifica_objetivo": textos[nombre] in leidos_salida,
                                  "salida_decodifica_otro": any(t != textos[nombre] for t in leidos_salida)})
                    for barrido, estado in enumerate(resultado.states):
                        energias.append({"regla": regla, "qr": nombre, "deterioro": tipo,
                                         "nivel": nivel, "semilla": semilla, "barrido": barrido,
                                         "energia": resultado.energies[barrido * patron.size]})
                    if nombre == "food" and repeticion == 0 and (
                        tipo == "aleatorio" and nivel == 0.30 or tipo != "aleatorio" and nivel == 9
                    ):
                        # Ejemplo fijado antes de ver los resultados, no elegido por éxito.
                        ejemplos[(tipo, regla)] = (patron, consulta, resultado, semilla)
                        np.savez(SALIDA / f"ejemplo_{tipo}_{regla}.npz", patron=patron,
                                 consulta=consulta, estados=resultado.states,
                                 energias=resultado.energies, semilla=semilla, nivel=nivel,
                                 regla=regla, qr=nombre)
                        Image.fromarray(imagen_lector(consulta)).save(
                            SALIDA / "lector" / f"entrada_{tipo}.png")
                        Image.fromarray(imagen_lector(resultado.state)).save(
                            SALIDA / "lector" / f"salida_{tipo}_{regla}.png")
        print(f"Terminado {tipo}, nivel={nivel}", flush=True)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(SALIDA / "recuperacion.csv", index=False)
    pd.DataFrame(energias).to_csv(SALIDA / "energia_por_barrido.csv", index=False)
    resumen = resultados.assign(exacto=resultados.resultado == "correcto").groupby(
        ["regla", "deterioro", "nivel"], as_index=False).agg(
        ensayos=("exacto", "size"), recuperacion_exacta=("exacto", "mean"),
        fraccion_invertida_media=("fraccion_invertida", "mean"),
        distancia_final_media=("distancia_final", "mean"),
        barridos_medios=("barridos", "mean"), barridos_maximos=("barridos", "max"),
        lectura_entrada=("entrada_decodifica_objetivo", "mean"),
        lectura_salida=("salida_decodifica_objetivo", "mean"))
    resumen.to_csv(SALIDA / "resumen.csv", index=False)
    return resultados, resumen, ejemplos


def graficos(nombres, patrones, resumen, ejemplos):
    fig, ax = plt.subplots(figsize=(6, 5))
    overlap = patrones @ patrones.T / patrones.shape[1]
    im = ax.imshow(overlap, cmap=DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(5), nombres, rotation=30)
    ax.set_yticks(range(5), nombres)
    ax.grid(False)
    for (i, j), valor in np.ndenumerate(overlap):
        ax.text(j, i, f"{valor:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="producto interno / 1369")
    ax.set_title("Solapamiento entre los cinco QR")
    guardar(fig, SALIDA / "solapamientos.png")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for columna, tipo in enumerate(["aleatorio", "esquina_blanca", "mancha_negra", "raya_blanca"]):
        for regla, color in zip(REGLAS, [NARANJA, AZUL]):
            tabla = resumen[(resumen.deterioro == tipo) & (resumen.regla == regla)]
            x = tabla.nivel * 100 if tipo == "aleatorio" else tabla.nivel
            axes[0, columna].plot(x, tabla.recuperacion_exacta * 100, "o-", color=color, label=regla)
            axes[1, columna].plot(x, tabla.lectura_salida * 100, "o-", color=color, label=regla)
        axes[1, columna].plot(x, tabla.lectura_entrada * 100, "k--", label="lector sin Hopfield")
        axes[0, columna].set_title(tipo.replace("_", " "))
        axes[1, columna].set_xlabel("bits invertidos solicitados (%)" if tipo == "aleatorio"
                                    else "lado del bloque / ancho de la raya (píxeles)")
        for fila in range(2):
            axes[fila, columna].set_ylim(-3, 103)
        axes[0, columna].legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel("recuperación exacta (%)")
    axes[1, 0].set_ylabel("lectura correcta del QR (%)")
    axes[1, -1].legend(frameon=False, fontsize=7)
    fig.suptitle("Cinco memorias simultáneas: Hebb, pseudoinversa y lector directo")
    fig.tight_layout()
    guardar(fig, SALIDA / "recuperacion.png")

    fig, axes = plt.subplots(4, 4, figsize=(10, 11))
    fig_e, ejes_e = plt.subplots(2, 4, figsize=(14, 6))
    for fila, tipo in enumerate(["aleatorio", "esquina_blanca", "mancha_negra", "raya_blanca"]):
        patron, consulta, hebb, _ = ejemplos[(tipo, "hebb")]
        pseudo = ejemplos[(tipo, "pseudoinverse")][2]
        for ax, estado, titulo in zip(axes[fila], [patron, consulta, hebb.state, pseudo.state],
                                     ["original food", "entrada", "salida Hebb", "salida pseudoinversa"]):
            ax.imshow(estado.reshape(37, 37), cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_title(titulo, fontsize=8)
            ax.axis("off")
        axes[fila, 0].text(-0.08, 0.5, tipo.replace("_", " "), transform=axes[fila, 0].transAxes,
                           rotation=90, ha="right", va="center", fontsize=8)
        for i, (regla, resultado, color) in enumerate([
            ("hebb", hebb, NARANJA), ("pseudoinverse", pseudo, AZUL)
        ]):
            eje = ejes_e[i, fila]
            eje.plot(np.arange(len(resultado.energies)) / 1369, resultado.energies, color=color)
            eje.set_title(f"{tipo.replace('_', ' ')} / {regla}", fontsize=8)
            eje.set_xlabel("barridos (incluye cada actualización)")
            if fila == 0:
                eje.set_ylabel("energía")
    fig.suptitle("Ejemplos prefijados: food, 30 % aleatorio o deterioro de tamaño 9")
    fig.tight_layout()
    guardar(fig, SALIDA / "ejemplos.png")
    fig_e.suptitle("Energía durante la recuperación; cada regla tiene su propia escala")
    fig_e.tight_layout()
    guardar(fig_e, SALIDA / "energia.png")


def main():
    (SALIDA / "lector").mkdir(parents=True, exist_ok=True)
    nombres, patrones = cargar()
    textos = {}
    for nombre, patron in zip(nombres, patrones):
        leidos = leer(patron)
        if len(leidos) != 1:
            raise RuntimeError(f"No se pudo leer exactamente un QR original: {nombre}")
        textos[nombre] = leidos[0]
        Image.fromarray(imagen_lector(patron)).save(SALIDA / "lector" / f"original_{nombre}.png")
    pd.DataFrame({"qr": nombres, "texto_original": [textos[n] for n in nombres]}).to_csv(
        SALIDA / "textos_originales.csv", index=False)
    modelos = {regla: Hopfield().fit(patrones, rule=regla) for regla in REGLAS}
    for regla, modelo in modelos.items():
        np.savez_compressed(SALIDA / f"modelo_{regla}.npz", weights=modelo.weights,
                            patrones=patrones, nombres=nombres, regla=regla)
    estabilidad(nombres, patrones, modelos)
    _, resumen, ejemplos = experimentos(nombres, patrones, modelos, textos)
    graficos(nombres, patrones, resumen, ejemplos)
    print(resumen.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
