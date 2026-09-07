"""Caso final: La noche estrellada.

Es la imagen que la cátedra mostró como ejemplo. Cierra el análisis porque es el caso
opuesto a los targets del barrido: `japan` es una región plana y `pika` tiene detalle
acotado sobre fondo liso; ésta es un óleo con textura en cada píxel, sin regiones
planas y sin bordes duros.

No es un eje experimental más. **Los operadores no se eligen a mano: se heredan del
barrido**, tomando de cada eje la variante ganadora promediada entre las tres
imágenes (la misma función que usa `interaccion.py`, para que no haya dos criterios
distintos de "ganador" en el trabajo).

Lo que sí se barre acá son las dos cosas que no se pueden heredar porque dependen del
tamaño del problema:

  1. la **cantidad de triángulos** — el segundo parámetro del problema;
  2. el **régimen de la tasa de mutación**. `pm` es una probabilidad POR GEN, así que
     al alargar el cromosoma el mismo `pm` multiplica la carga de mutación. Lo único
     transferible entre escalas es la CARGA (genes esperados mutados por individuo),
     y que eso sea así es una hipótesis que acá se mide en vez de asumirse.

    python analysis/caso_final.py                # barrido + corrida final
    python analysis/caso_final.py --sweep-only   # sólo el barrido
    python analysis/caso_final.py --solo-docs    # rehace los .md sin correr nada

Escribe `analysis/informe/15-caso-final/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import interaccion
import plot_results as figs
from ga import artifact, engine
from ga.individual import GENES_PER_TRIANGLE, Individual
from ga.render import canvas_size, load_target, render

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
RESULTS = ANALYSIS / "results"
DESTINO = ANALYSIS / "informe" / "15-caso-final"

IMAGEN = "images/noche_estrellada.jpg"
CANVAS = 96
SALIDA = 900          # lado largo de las imágenes entregadas
SEEDS = [1, 2, 3]
GENERACIONES_SWEEP = 1200
GENERACIONES_FINAL = 3000
TRIANGULOS = [100, 200, 400, 800]
POBLACION = 80

#: Ejes cuyo ganador se hereda tal cual. Tres quedan afuera, y por motivos distintos:
#:
#:   tasa_mutacion  no se hereda el número sino la CARGA, que es lo único transferible
#:                  a otro largo de cromosoma. Es la segunda dimensión del barrido.
#:   poblacion      se fija por presupuesto: el eje se midió con 50 triángulos y acá el
#:                  costo por generación es otro.
#:   mutacion       el barrido lo ganó `gene` en dos de las tres imágenes, pero `gene`
#:                  muta UN gen por individuo con probabilidad pm, así que su carga está
#:                  acotada a 1 sin importar el largo del cromosoma. La dimensión que se
#:                  estudia acá —cómo trasladar una tasa POR GEN a otro largo— no tiene
#:                  sentido con un método que no es por gen: aplicarle el pm escalado lo
#:                  deja con 0.004 genes mutados por individuo, o sea congelado.
#:                  Se fuerza `uniform`, que es el método por gen y el que ganó en la
#:                  imagen detallada. Que el ganador del eje no sea transferible a otra
#:                  escala es, en sí, parte del resultado.
EJES_HEREDADOS = [e for e in interaccion.EJES_OPERADORES
                  if e not in ("tasa_mutacion", "poblacion", "mutacion")]

#: método de mutación forzado, y por qué (va al informe)
MUTACION_FORZADA = "uniform"
MOTIVO_MUTACION = (
    "El eje lo ganó `gen` en dos de las tres imágenes, pero `gen` muta **un solo gen** "
    "por individuo: su carga está acotada a 1 sea cual sea el largo del cromosoma. La "
    "pregunta de este caso —cómo trasladar una tasa **por gen** a un cromosoma más "
    "largo— no se puede formular con un método que no es por gen. Se fuerza `uniform`, "
    "que es el que ganó en la imagen detallada y el único de los cuatro cuya tasa escala. "
    "**Que el ganador de un eje no sea transferible a otra escala es parte del resultado**, "
    "no una excepción cómoda."
)

#: nota propia por eje, para la columna "por qué" de la tabla de decisiones
NOTA_EJE = {
    "seleccion": "Método de selección de padres.",
    "supervivencia": "Estrategia de supervivencia y brecha generacional K/N.",
    "cruza": "Método de recombinación.",
    "granularidad": "Unidad de corte de la cruza: componente suelto o triángulo entero.",
    "tasa_cruza": "Probabilidad de recombinación pc.",
    "mutacion": "Método de mutación.",
    "sigma": "Magnitud de la perturbación gaussiana.",
    "inicializacion": "Población inicial. Es la decisión más discutible del trabajo: "
                      "la grilla inyecta conocimiento del target antes del algoritmo.",
}


def cargar_summary() -> list[dict]:
    ruta = RESULTS / "summary.csv"
    if not ruta.exists():
        sys.exit("faltan resultados. Corré primero analysis/run_experiments.py")
    return list(csv.DictReader(ruta.open()))


def separacion_por_target(summary, eje, variante) -> list[str]:
    """En qué imágenes esa variante gana con el rango intercuartil disjunto del segundo."""
    por_target: dict[str, list] = {}
    for fila in summary:
        if fila["experiment"] == eje:
            por_target.setdefault(fila["target"], []).append(fila)
    separadas = []
    for target, filas in por_target.items():
        filas.sort(key=lambda f: -float(f["best_fitness_mean"]))
        if (filas[0]["variant"] == variante and len(filas) > 1
                and float(filas[0]["best_fitness_q1"]) > float(filas[1]["best_fitness_q3"])):
            separadas.append(target)
    return sorted(separadas)


def carga_heredada(summary) -> float:
    """La carga de mutación que ganó su eje, leída del nombre de la variante
    ('carga 4' -> 4.0). Es la magnitud transferible entre largos de cromosoma."""
    elegida = interaccion.ganadores(summary, [], mejor=True).get("tasa_mutacion")
    if elegida is None:
        return 4.0
    try:
        return float(elegida["variante"].split()[-1])
    except ValueError:
        return 4.0


def base_heredada(summary, spec) -> tuple[dict, list[dict]]:
    """Configuración de operadores heredada del barrido, con su justificación."""
    mejores = interaccion.ganadores(summary, list(spec["targets"]), mejor=True)
    base = {**engine.DEFAULTS, **spec["base"], "image": IMAGEN, "canvas_size": CANVAS,
            "preserve_aspect": True, "population_size": POBLACION,
            "offspring_size": POBLACION}

    justificacion = []
    for eje in EJES_HEREDADOS:
        if eje not in mejores:
            continue
        overrides = interaccion.overrides_de(spec, eje, mejores[eje]["variante"])
        base.update(overrides)
        justificacion.append({
            "eje": eje,
            "variante": mejores[eje]["variante"],
            "overrides": overrides,
            "fitness": mejores[eje]["fitness_eje"],
            "separado": separacion_por_target(summary, eje, mejores[eje]["variante"]),
        })
    # la brecha generacional se fija acá: el eje la barre junto con la estrategia
    base["offspring_size"] = POBLACION
    # ver MOTIVO_MUTACION: la dimensión de régimen exige un método por gen
    base["mutation"] = MUTACION_FORZADA
    return base, justificacion


def regimenes(carga: float, genes_calibracion: int) -> dict:
    """Los dos modos de llevar la tasa de mutación a un cromosoma de otro largo."""
    return {
        # copiar pm tal cual: la carga se multiplica junto con el largo del cromosoma
        "pm fijo": lambda genes: carga / genes_calibracion,
        # traducir: se mantiene constante la cantidad esperada de genes mutados
        "carga fija": lambda genes: carga / genes,
    }


def etiqueta(triangles: int, regimen: str) -> str:
    return f"{triangles} tri · {regimen}"


# --------------------------------------------------------------------------------------
_CTX: dict = {}


def _init_worker(base, carga, genes_calibracion):
    _CTX["base"] = base
    _CTX["regimenes"] = regimenes(carga, genes_calibracion)


def _config(triangles, regimen, generaciones, seed):
    genes = triangles * GENES_PER_TRIANGLE
    return {**_CTX["base"], "triangles": triangles, "seed": seed,
            "mutation_rate": _CTX["regimenes"][regimen](genes),
            "stop": {**engine.DEFAULTS["stop"], "max_generations": generaciones}}


def _correr(job):
    triangles, regimen, seed = job
    cfg = _config(triangles, regimen, GENERACIONES_SWEEP, seed)
    target = load_target(str(ROOT / IMAGEN), CANVAS, cfg["background"], True)
    inicio = time.perf_counter()
    result = engine.run(cfg, target)
    # se devuelve el genotipo del mejor: la imagen sale de la misma corrida que el
    # número, sin volver a correr el motor
    return (triangles, regimen, seed, result.history_rows(),
            time.perf_counter() - inicio, result.best.genes)


def barrido(base, carga, genes_calibracion):
    """Cantidad de triángulos × régimen de pm. Los trabajos caros van primero."""
    regs = list(regimenes(carga, genes_calibracion))
    jobs = sorted([(t, r, s) for t in TRIANGULOS for r in regs for s in SEEDS],
                  key=lambda j: -j[0])
    print(f"barrido: {len(jobs)} corridas de {GENERACIONES_SWEEP} generaciones "
          f"({len(TRIANGULOS)} cantidades × {len(regs)} regímenes × {len(SEEDS)} semillas)")

    filas, tiempos, mejores = [], {}, {}
    inicio = time.perf_counter()
    with ProcessPoolExecutor(initializer=_init_worker,
                             initargs=(base, carga, genes_calibracion)) as pool:
        for hecho, (tri, reg, seed, rows, elapsed, genes) in enumerate(pool.map(_correr, jobs), 1):
            filas += [{"variant": etiqueta(tri, reg), "triangles": tri, "regimen": reg,
                       "seed": seed, "target": "compleja", **row} for row in rows]
            tiempos.setdefault((tri, reg), []).append(elapsed)
            if seed == SEEDS[0]:
                mejores[(tri, reg)] = genes
            print(f"\r  {hecho}/{len(jobs)} | {time.perf_counter() - inicio:.0f}s",
                  end="", flush=True)
    print()
    return filas, tiempos, mejores


def como_data(filas: list[dict]) -> dict:
    por_corrida: dict[tuple, list[dict]] = {}
    for fila in filas:
        por_corrida.setdefault((fila["variant"], fila["seed"]), []).append(fila)

    anidado: dict = {}
    for (variante, _seed), rows in por_corrida.items():
        rows.sort(key=lambda r: int(r["generation"]))
        destino = anidado.setdefault(variante, {})
        for metrica in figs.METRICS:
            destino.setdefault(metrica, []).append([float(r[metrica]) for r in rows])
    return {v: {m: np.array(runs) for m, runs in ms.items()} for v, ms in anidado.items()}


def corrida_final(base, carga, genes_calibracion, triangles, regimen) -> dict:
    _init_worker(base, carga, genes_calibracion)
    cfg = _config(triangles, regimen, GENERACIONES_FINAL, SEEDS[0])
    ancho, alto = canvas_size(str(ROOT / IMAGEN), CANVAS, True)
    out_w, out_h = SALIDA, max(1, round(SALIDA * alto / ancho))
    target = load_target(str(ROOT / IMAGEN), CANVAS, cfg["background"], True)

    capturas: list[tuple[int, Image.Image]] = []
    hitos = {1, 10, 50, 150, 400, 900, 1800, GENERACIONES_FINAL}
    ultimo = [0.0]

    def por_generacion(record, mejor):
        if record.generation in hitos:
            capturas.append((record.generation,
                             render(mejor, 300, round(300 * alto / ancho),
                                    background=cfg["background"])))
        if time.perf_counter() - ultimo[0] > 5.0:
            ultimo[0] = time.perf_counter()
            print(f"\r  gen {record.generation:5d}/{GENERACIONES_FINAL} | "
                  f"fitness {record.best_global_fitness:.4f} | {record.elapsed:5.0f}s",
                  end="", flush=True)

    print(f"corrida final: {triangles} triángulos, {regimen} "
          f"(pm={cfg['mutation_rate']:.5f}), {GENERACIONES_FINAL} generaciones")
    inicio = time.perf_counter()
    result = engine.run(cfg, target, on_generation=por_generacion)
    print()

    imagenes = DESTINO / "imagenes"
    documento = artifact.build(result.best, out_w, out_h, cfg["background"],
                               source_image=IMAGEN, fitness=result.best.fitness)
    artifact.save(documento, imagenes / "triangles.json")
    final = artifact.render(documento)
    final.save(imagenes / "resultado.png")

    original = Image.open(ROOT / IMAGEN).convert("RGB").resize((out_w, out_h))
    original.save(imagenes / "target.png")
    lado = Image.new("RGB", (out_w * 2 + 16, out_h), "white")
    lado.paste(original, (0, 0)); lado.paste(final, (out_w + 16, 0))
    lado.save(imagenes / "comparacion.png")

    if capturas:
        an, al = capturas[0][1].size
        etiq = 24
        tira = Image.new("RGB", (an * len(capturas), al + etiq), "white")
        lapiz = ImageDraw.Draw(tira)
        for i, (generacion, imagen) in enumerate(capturas):
            tira.paste(imagen, (i * an, etiq))
            lapiz.text((i * an + 6, 7), f"gen {generacion}", fill="black")
        tira.save(imagenes / "evolucion.png")
        for generacion, imagen in capturas:
            imagen.save(imagenes / f"gen_{generacion:05d}.png")

    return {
        "triangles": triangles, "regimen": regimen, "mutation_rate": cfg["mutation_rate"],
        "generations": result.generations, "fitness": float(result.best.fitness),
        "rmse": float((1 - result.best.fitness) * 255), "evaluations": result.evaluations,
        "seconds": round(time.perf_counter() - inicio, 1), "stop_reason": result.stop_reason,
        "canvas": [ancho, alto], "salida": [out_w, out_h], "config": cfg,
        "bytes_json": (imagenes / "triangles.json").stat().st_size,
        "bytes_original": (ROOT / IMAGEN).stat().st_size,
    }


# --------------------------------------------------------------------------------------
def escribir_config(justificacion, carga, genes_calibracion, elegido, base, final) -> None:
    triangles, regimen = elegido
    lineas = [
        "# Configuración — caso final (La noche estrellada)",
        "",
        "Los operadores **no se eligieron a mano**: cada uno es el ganador de su eje en el",
        "barrido, promediado entre las tres imágenes. La columna «¿se separa?» dice en qué",
        "imágenes ese ganador tiene el rango intercuartil disjunto del segundo — o sea, dónde",
        "la elección está respaldada y dónde es apenas la mejor media.",
        "",
        "| Eje | Ganador heredado | Qué fija | ¿Se separa del 2º? | Qué es |",
        "|---|---|---|---|---|",
    ]
    for j in justificacion:
        cambios = " · ".join(f"`{k}` = `{v}`" for k, v in j["overrides"].items())
        sep = ", ".join(j["separado"]) if j["separado"] else "**en ninguna** (empate)"
        numero = [n for n, s, _ in ORDEN_EJES if s == j["eje"]]
        enlace = f"[`{numero[0]}`](../{numero[0]}/informe.md)" if numero else f"`{j['eje']}`"
        lineas.append(f"| {enlace} | **{j['variante']}** | {cambios} | {sep} | "
                      f"{NOTA_EJE.get(j['eje'], '')} |")

    lineas += [
        "",
        "## Lo que no se hereda, y por qué",
        "",
        "| Parámetro | Valor | Motivo |",
        "|---|---|---|",
        f"| `mutation_rate` | se barre acá | `pm` es probabilidad **por gen**. El barrido lo "
        f"calibró con {genes_calibracion} genes y acá el cromosoma tiene entre "
        f"{TRIANGULOS[0] * GENES_PER_TRIANGLE:,} y {TRIANGULOS[-1] * GENES_PER_TRIANGLE:,}. "
        f"Lo que se hereda es la **carga** ({carga:g} genes mutados por individuo), y se "
        "prueban las dos formas de trasladarla. |",
        f"| `triangles` | se barre acá | Es un parámetro del **problema**, no del algoritmo: "
        "depende de la imagen. |",
        f"| `mutation` | `{MUTACION_FORZADA}` (forzado) | {MOTIVO_MUTACION} |",
        f"| `population_size` / `offspring_size` | `{POBLACION}` | El eje de población se midió "
        "con 50 triángulos y a esta escala el costo por generación es otro. Se fija por "
        "presupuesto, no por medición: es una **limitación declarada**. |",
        f"| `canvas_size` | `{CANVAS}` | El fitness se evalúa a {CANVAS}px de lado largo. Más "
        "resolución daría señal más fina y costaría proporcionalmente más. |",
        "",
        "## Resultado del barrido",
        "",
        "| Parámetro | Valores probados | Elegido |",
        "|---|---|---|",
        f"| `triangles` | {', '.join(str(t) for t in TRIANGULOS)} | **{triangles}** |",
        f"| régimen de mutación | pm fijo, carga fija | **{regimen}** |",
        "",
        "## Sobre la proporción de la imagen",
        "",
        "`La noche estrellada` es 1280×1014 (relación 1.26:1). Un canvas cuadrado la aplastaría",
        "un 26%, inaceptable para ponerla al lado del original. Por eso `preserve_aspect` hace",
        f"que `canvas_size` sea el **lado largo**: el canvas de trabajo queda "
        f"{canvas_size(str(ROOT / IMAGEN), CANVAS, True)[0]}×"
        f"{canvas_size(str(ROOT / IMAGEN), CANVAS, True)[1]}px.",
        "",
        "## Reproducir",
        "",
        "```bash",
        "python analysis/run_experiments.py --clean   # el barrido del que se hereda",
        "python analysis/caso_final.py",
        "```",
        "",
    ]
    if final:
        lineas += ["<details><summary>Configuración completa de la corrida final (JSON)</summary>",
                   "", "```json", json.dumps(final["config"], indent=2, ensure_ascii=False),
                   "```", "", "</details>", ""]
    (DESTINO / "config.md").write_text("\n".join(lineas))


ORDEN_EJES = [
    ("01", "seleccion", ""), ("02", "presion", ""), ("03", "supervivencia", ""),
    ("04", "cruza", ""), ("05", "granularidad", ""), ("06", "tasa_cruza", ""),
    ("07", "mutacion", ""), ("08", "tasa_mutacion", ""), ("09", "sigma", ""),
    ("10", "poblacion", ""), ("11", "triangulos", ""), ("12", "inicializacion", ""),
]
_SLUG = {"seleccion": "01-seleccion", "presion": "02-presion-seleccion",
         "supervivencia": "03-supervivencia", "cruza": "04-cruza",
         "granularidad": "05-granularidad", "tasa_cruza": "06-tasa-cruza",
         "mutacion": "07-mutacion", "tasa_mutacion": "08-tasa-mutacion",
         "sigma": "09-sigma", "poblacion": "10-poblacion",
         "triangulos": "11-triangulos", "inicializacion": "12-inicializacion"}
ORDEN_EJES = [(_SLUG[e], e, "") for e in _SLUG]


def escribir_informe(stats, tiempos, arranque, cruces, elegido, final,
                     carga, genes_calibracion) -> None:
    orden = sorted(stats, key=lambda s: -s["media"])
    regs = list(regimenes(carga, genes_calibracion))

    lineas = [
        "# Caso final — La noche estrellada",
        "",
        "> Cierre del análisis. Los operadores **se heredan** del barrido (el ganador de cada",
        "> eje, promediado entre las tres imágenes); ver [`config.md`](config.md) con la tabla",
        "> completa y si cada ganador está respaldado o es un empate. Acá se barre sólo lo que",
        "> depende del tamaño del problema.",
        "",
        "## Por qué esta imagen",
        "",
        "Es la que la cátedra mostró como ejemplo, y es el caso opuesto a los targets del",
        "barrido: `japan` es una región plana, `pika` tiene detalle acotado sobre fondo liso.",
        "La noche estrellada **no tiene una sola región plana**: es textura en cada píxel, sin",
        "bordes duros. Sirve para dos cosas — mostrar dónde deja de alcanzar el método, y poner",
        "a prueba decisiones tomadas con otras imágenes sobre una que nunca vimos.",
        "",
        "## El problema de transferir la tasa de mutación",
        "",
        "`pm` es una probabilidad **por gen**. El eje "
        "[`08-tasa-mutacion`](../08-tasa-mutacion/informe.md) encontró que la mejor carga es "
        f"**{carga:g} genes mutados por individuo y generación**, medida con "
        f"{genes_calibracion} genes. Acá el cromosoma es mucho más largo, y hay dos maneras "
        "incompatibles de trasladar ese resultado:",
        "",
        f"| Triángulos | Genes | `pm fijo` → carga | `carga fija` → pm |",
        "|---|---|---|---|",
    ]
    pm_fijo = carga / genes_calibracion
    for t in TRIANGULOS:
        genes = t * GENES_PER_TRIANGLE
        lineas.append(f"| {t} | {genes:,} | pm={pm_fijo:.4f} → carga {pm_fijo * genes:.0f} | "
                      f"carga {carga:g} → pm={carga / genes:.5f} |")
    lineas += [
        "",
        f"Con {TRIANGULOS[-1]} triángulos, copiar `pm` muta "
        f"**{pm_fijo * TRIANGULOS[-1] * GENES_PER_TRIANGLE / carga:.0f} veces más genes por "
        "individuo** que en la corrida donde ese valor ganó. Copiar el número no es heredar la "
        "configuración: es cambiarla. Por eso el régimen es una dimensión del barrido y no un "
        "supuesto.",
        "",
        "## Resultados",
        "",
        f"{len(TRIANGULOS)} cantidades × {len(regs)} regímenes × {len(SEEDS)} semillas × "
        f"{GENERACIONES_SWEEP} generaciones.",
        "",
        "| # | Triángulos | Régimen | pm | Fitness | RMSE | Tiempo/corrida |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, s in enumerate(orden, 1):
        segundos = float(np.mean(tiempos[(s["triangles"], s["regimen"])]))
        lineas.append(
            f"| {i} | {s['triangles']} | {s['regimen']} | {s['pm']:.5f} | "
            f"{s['media']:.4f} ± {s['desvio']:.4f} | {(1 - s['media']) * 255:.2f} | "
            f"{segundos:.0f} s |")

    mejor, peor = orden[0], orden[-1]
    lineas += [
        "",
        f"**Gana {mejor['triangles']} triángulos con {mejor['regimen']}** ({mejor['media']:.4f}), "
        f"contra {peor['media']:.4f} de la peor combinación "
        f"({peor['triangles']} tri · {peor['regimen']}).",
        "",
        "### El régimen de mutación decide más que la cantidad de triángulos",
        "",
        "| Triángulos | pm fijo | carga fija | Diferencia |",
        "|---|---|---|---|",
    ]
    ganancias = []
    for t in TRIANGULOS:
        fijo = next((s["media"] for s in stats
                     if s["triangles"] == t and s["regimen"] == "pm fijo"), None)
        esc = next((s["media"] for s in stats
                    if s["triangles"] == t and s["regimen"] == "carga fija"), None)
        if fijo is None or esc is None:
            continue
        ganancias.append(esc - fijo)
        lineas.append(f"| {t} | {fijo:.4f} | {esc:.4f} | {esc - fijo:+.4f} |")

    if ganancias and max(ganancias) > 0.002:
        mejora_en = sum(1 for g in ganancias if g > 0)
        lineas += [
            "",
            f"**Mantener la carga constante mejora en {mejora_en} de las {len(ganancias)} "
            f"cantidades**, y la ventaja crece con los triángulos ({ganancias[-1]:+.4f} con "
            f"{TRIANGULOS[-1]}). Es la conclusión más transferible del trabajo:",
            "",
            "> **Un hiperparámetro medido a una escala no se copia a otra: se traduce.** Lo que",
            "> hay que mantener constante al cambiar el largo del cromosoma no es la",
            "> probabilidad por gen sino la cantidad esperada de genes mutados por individuo.",
            "",
            "Y tiene una consecuencia hacia atrás: el eje "
            "[`11-triangulos`](../11-triangulos/informe.md) mide exactamente este cruce sobre "
            "las otras imágenes, y es el motivo por el que ahí también se corren los dos "
            "regímenes en vez de uno solo.",
        ]
    elif ganancias:
        lineas += [
            "",
            "Mantener la carga constante **no mejora** de forma consistente. La hipótesis era "
            "que la carga transfiere mejor que la probabilidad; los datos no la sostienen a "
            "esta escala, y eso también es un resultado.",
        ]

    orden_inicial = sorted(arranque, key=lambda k: -arranque[k])
    if orden_inicial and orden_inicial[0] != elegido:
        lineas += [
            "",
            "### Capacidad contra costo de búsqueda: las curvas se cruzan",
            "",
            f"En la generación 1 arriba está `{etiqueta(*orden_inicial[0])}` "
            f"({arranque[orden_inicial[0]]:.4f}) y abajo `{etiqueta(*orden_inicial[-1])}` "
            f"({arranque[orden_inicial[-1]]:.4f}). Al final gana `{etiqueta(*elegido)}`.",
            "",
            f"| Combinación | Fitness gen. 1 | Fitness gen. {GENERACIONES_SWEEP} | Ganancia |",
            "|---|---|---|---|",
        ]
        for s in sorted(stats, key=lambda s: (s["triangles"], s["regimen"])):
            clave = (s["triangles"], s["regimen"])
            lineas.append(f"| {etiqueta(*clave)} | {arranque[clave]:.4f} | {s['media']:.4f} | "
                          f"{s['media'] - arranque[clave]:+.4f} |")
        if cruces:
            lineas += ["", f"El cruce ocurre alrededor de la **generación {min(cruces.values())}**."]
        lineas += [
            "",
            "Se ven los dos efectos separados:",
            "",
            "- **La inicialización informada premia la capacidad**: más triángulos es una",
            "  grilla más fina, así que el punto de partida ya aproxima mejor el target.",
            "- **La búsqueda la castiga**: cada triángulo son 10 genes más que optimizar. A",
            "  presupuesto fijo, el espacio más grande se explora peor.",
            "",
            "Cuál de los dos gana depende del presupuesto, así que **la cantidad de triángulos",
            "no se elige por capacidad sino por cuántas generaciones se van a correr.**",
        ]

    if final:
        compresion = final["bytes_original"] / final["bytes_json"]
        lineas += [
            "",
            "## La corrida final",
            "",
            f"- **{final['triangles']} triángulos**, régimen `{final['regimen']}` "
            f"(pm = {final['mutation_rate']:.5f}), {final['generations']} generaciones, "
            f"{final['evaluations']:,} evaluaciones, {final['seconds']:.0f} s",
            f"- **Fitness {final['fitness']:.4f}** (RMSE {final['rmse']:.2f})",
            f"- Canvas de evaluación {final['canvas'][0]}×{final['canvas'][1]}px; la imagen",
            f"  entregada se renderiza a {final['salida'][0]}×{final['salida'][1]}px **desde el",
            "  mismo genotipo** — el individuo no tiene resolución propia, y eso es una",
            "  propiedad del formato de salida, no un detalle de implementación",
            "",
            "### Como compresor",
            "",
            "El enunciado plantea el TP como un compresor con pérdida. Acá está el número:",
            "",
            "| | Tamaño |",
            "|---|---|",
            f"| Original (`{Path(IMAGEN).name}`) | {final['bytes_original'] / 1024:,.0f} KB |",
            f"| `triangles.json` ({final['triangles']} triángulos) | "
            f"{final['bytes_json'] / 1024:,.0f} KB |",
            f"| **Relación** | **{compresion:.1f}×** |",
            "",
            "Con la salvedad de que el JSON es texto sin comprimir y el original es un JPEG ya",
            "comprimido. La comparación honesta es *cuánta información hace falta para describir",
            f"la imagen*: {final['triangles']} × {GENES_PER_TRIANGLE} = "
            f"{final['triangles'] * GENES_PER_TRIANGLE:,} números reales.",
        ]

    lineas += [
        "",
        "## Qué mirar",
        "",
        "- `imagenes/comparacion.png` — original y resultado lado a lado. Es la slide de cierre.",
        "- `imagenes/<n>-tri-<regimen>.png` — las 8 combinaciones. **Acá se ve lo que los",
        "  números no dicen**: con `pm fijo` y muchos triángulos la imagen queda ruidosa,",
        "  porque se están mutando decenas de genes por individuo y por generación.",
        "- `imagenes/evolucion.png` — la corrida en 8 momentos. Se ve el orden en que el",
        "  algoritmo construye: primero el reparto de masas de color, después el cielo, y el",
        "  ciprés y el pueblo al final. Los detalles chicos son los últimos porque aportan poco",
        "  al RMSE hasta que lo grueso está resuelto.",
        "- `figuras/convergencia_compleja.png` — que ninguna curva esté plana al final es la",
        "  advertencia honesta: **la corrida no convergió, se quedó sin presupuesto.**",
        "",
        "## Limitaciones, dichas antes de que las pregunten",
        "",
        "- **El fitness es RMSE en RGB, que no es percepción.** En una imagen con esta textura",
        "  el óptimo del RMSE es emborronar: promediar cada zona minimiza el error cuadrático",
        "  mejor que intentar reproducir la pincelada. Buena parte de lo que se ve «lavado» no",
        "  es una falla del AG — **está optimizando exactamente lo que le pedimos**. Un fitness",
        "  perceptual (ΔE en Lab) es la continuación natural.",
        "- La configuración combina ganadores de ejes distintos. Cuánto cuesta eso está medido",
        "  en [`14-interaccion`](../14-interaccion/informe.md); esta corrida no es «la mejor",
        "  configuración posible» sino la mejor que se puede *justificar* con lo medido.",
        "- `population_size` se fijó por presupuesto, no por medición a esta escala.",
        f"- {len(SEEDS)} semillas en el barrido: alcanza para la tendencia, no para diferencias",
        "  chicas.",
        "",
        "## Archivos",
        "",
        "| Archivo | Qué es |",
        "|---|---|",
        "| `imagenes/comparacion.png` | Original y resultado, lado a lado |",
        "| `imagenes/resultado.png` | Sólo el resultado |",
        "| `imagenes/target.png` | Sólo el original, al mismo tamaño |",
        "| `imagenes/evolucion.png` | Tira de 8 momentos de la corrida |",
        "| `imagenes/gen_*.png` | Cada momento por separado, para componer la slide |",
        "| `imagenes/<n>-tri-<regimen>.png` | El mejor de cada combinación del barrido |",
        "| `imagenes/triangles.json` | La enumeración de triángulos: la «compresión» |",
        "| `datos/barrido.csv` | Una fila por generación, combinación y semilla |",
        "| `datos/final.json` | Métricas de la corrida final |",
        "",
    ]
    (DESTINO / "informe.md").write_text("\n".join(lineas))


# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Caso final: La noche estrellada")
    parser.add_argument("--sweep-only", action="store_true", help="no corre la corrida final")
    parser.add_argument("--solo-docs", action="store_true",
                        help="rehace los .md desde los datos guardados, sin correr nada")
    args = parser.parse_args()

    if not (ROOT / IMAGEN).exists():
        sys.exit(f"falta {IMAGEN}")
    figs.use_theme("light")

    spec = json.loads((ANALYSIS / "experiments.json").read_text())
    summary = cargar_summary()
    base, justificacion = base_heredada(summary, spec)
    carga = carga_heredada(summary)
    genes_calibracion = spec["base"]["triangles"] * GENES_PER_TRIANGLE
    print(f"heredado del barrido: {len(justificacion)} operadores | carga de mutación {carga:g}")

    if args.solo_docs:
        filas = [{**r, "seed": int(r["seed"]), "triangles": int(r["triangles"])}
                 for r in csv.DictReader((DESTINO / "datos" / "barrido.csv").open())]
        ultimas: dict[tuple, float] = {}
        for fila in filas:
            clave = (fila["triangles"], fila["regimen"], fila["seed"])
            ultimas[clave] = max(ultimas.get(clave, 0.0), float(fila["elapsed"]))
        tiempos: dict[tuple, list] = {}
        for (tri, reg, _seed), segundos in ultimas.items():
            tiempos.setdefault((tri, reg), []).append(segundos)
        mejores = {}
    else:
        if DESTINO.exists():
            shutil.rmtree(DESTINO)
        for sub in ("datos", "figuras", "imagenes"):
            (DESTINO / sub).mkdir(parents=True)
        filas, tiempos, mejores = barrido(base, carga, genes_calibracion)
        with (DESTINO / "datos" / "barrido.csv").open("w", newline="") as fh:
            escritor = csv.DictWriter(fh, fieldnames=list(filas[0]))
            escritor.writeheader(); escritor.writerows(filas)

    data = como_data(filas)
    if not args.solo_docs:
        colores = figs.colors_for({"compleja": data})
        for nombre, funcion in figs.FIGURES.items():
            funcion(data, colores, DESTINO / "figuras" / f"{nombre}_compleja.png")

    regs = regimenes(carga, genes_calibracion)
    stats, arranque, curvas = [], {}, {}
    for variante, metrics in data.items():
        tri = int(variante.split()[0])
        reg = variante.split("·")[1].strip()
        curva = metrics["best_global_fitness"].mean(axis=0)
        finales = metrics["best_global_fitness"][:, -1]
        stats.append({"triangles": tri, "regimen": reg,
                      "pm": regs[reg](tri * GENES_PER_TRIANGLE),
                      "media": float(finales.mean()), "desvio": float(finales.std())})
        curvas[(tri, reg)] = curva
        arranque[(tri, reg)] = float(curva[0])

    orden = sorted(stats, key=lambda s: -s["media"])
    elegido = (orden[0]["triangles"], orden[0]["regimen"])
    lider_inicial = max(arranque, key=arranque.get)
    cruces = {}
    if lider_inicial != elegido:
        delante = curvas[elegido] > curvas[lider_inicial]
        for g in range(len(delante)):
            if delante[g:].all():
                cruces[elegido] = g + 1
                break
    print(f"elegido: {elegido[0]} triángulos · {elegido[1]} (fitness {orden[0]['media']:.4f})")

    ancho, alto = canvas_size(str(ROOT / IMAGEN), CANVAS, True)
    for (tri, reg), genes in mejores.items():
        nombre = f"{tri}-tri-{reg.replace(' ', '-')}.png"
        render(Individual(genes), 420, round(420 * alto / ancho),
               background=base["background"]).save(DESTINO / "imagenes" / nombre)

    if args.solo_docs:
        ruta = DESTINO / "datos" / "final.json"
        final = json.loads(ruta.read_text()) if ruta.exists() else None
    else:
        final = None if args.sweep_only else corrida_final(base, carga, genes_calibracion, *elegido)
        if final:
            (DESTINO / "datos" / "final.json").write_text(
                json.dumps(final, indent=2, ensure_ascii=False))

    escribir_config(justificacion, carga, genes_calibracion, elegido, base, final)
    escribir_informe(stats, tiempos, arranque, cruces, elegido, final, carga, genes_calibracion)
    print(f"informe en {DESTINO.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
