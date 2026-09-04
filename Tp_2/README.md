# TP 2 — Algoritmos Genéticos: aproximación de imágenes con triángulos

Motor de Algoritmos Genéticos implementado desde cero (sin librerías de AG) que
aproxima una imagen usando N triángulos traslúcidos sobre un canvas blanco.

- Enunciado y plan de trabajo: [`docs/TP2.md`](docs/TP2.md)
- Ejercicio 1 (ASCII art, sólo análisis): [`docs/ejercicio1.md`](docs/ejercicio1.md)

> **Estado: completo.** Los 6 métodos de selección (más la selección combinada
> A%/B%), las 4 cruzas, las 4 mutaciones, ambas estrategias de supervivencia, los 5
> criterios de corte, el pipeline de experimentos y la presentación.

## Setup

```bash
cd Tp_2
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python images/generate_samples.py           # genera las imágenes target de ejemplo
python main.py                              # corre con config.json
python main.py --triangles 50 --generations 2000
python main.py --image images/germany.png --tag prueba --gif
```

| Flag | Qué hace |
|---|---|
| `--config` | Archivo de configuración (default `config.json`) |
| `--image` | Imagen target |
| `--triangles` | Cantidad de triángulos (el 2º parámetro del problema) |
| `--canvas` | Resolución de trabajo para evaluar el fitness |
| `--population` / `--offspring` | Tamaño de población N y de descendencia K |
| `--generations` | Cota de generaciones |
| `--seed` | Semilla, para corridas reproducibles |
| `--tag` | Sufijo para no pisar corridas anteriores |
| `--render-size` | Resolución de la imagen final (default 512) |
| `--gif` | Además guarda un gif de la evolución |
| `--snapshots N` | Guarda el mejor individuo cada N generaciones y una tira comparativa |
| `--rebuild` | Reconstruye la imagen desde un `triangles.json` y termina |

## Salida

Cada corrida escribe en `output/<imagen>[-tag]/`:

| Archivo | Contenido |
|---|---|
| `best.png` | Imagen generada (el genotipo es independiente de la resolución: se puede renderizar a cualquier tamaño) |
| `comparison.png` | Target vs. resultado, lado a lado |
| `triangles.json` | Enumeración de los triángulos — la "compresión" de la imagen (ver abajo) |
| `metrics.csv` | Una fila por generación: mejor fitness, promedio, desvío, diversidad, evaluaciones, tiempo |
| `run.json` | Config completa + fitness final, RMSE, generaciones, evaluaciones, tiempo y motivo de corte |
| `evolution.gif` | Sólo con `--gif` |

## El archivo de triángulos

`triangles.json` es el output que pide el enunciado como enumeración de triángulos.
Está pensado como un **formato autosuficiente**: con el archivo solo —sin el código
que lo generó— se puede reconstruir la imagen, porque declara el tamaño del canvas,
el color de fondo, la regla de composición y el orden de pintado.

```jsonc
{
  "format_version": 1,
  "canvas": { "width": 512, "height": 512, "background": [255, 255, 255] },
  "compositing": "rgba-source-over",
  "paint_order": "El array `triangles` está en orden de pintado: ...",
  "source_image": "images/japan.png",
  "fitness": 0.9269,
  "triangles": [
    { "vertices": [[341.5, 213.97], [178.17, 121.14], [88.4, 402.8]], "color": [188, 0, 45, 120] }
  ]
}
```

El PNG entregado se renderiza **desde el documento**, no desde el individuo en
memoria, así que la imagen y el archivo describen la misma cosa por construcción.
Para verificarlo:

```bash
python main.py --rebuild output/japan/triangles.json
# reconstruido desde output/japan/triangles.json -> output/japan/rebuilt.png
#   20 triángulos, canvas 512x512
#   diferencia máxima contra best.png: 0 (0 = idénticas)
```

`test_ga.py` incluye el test de ida y vuelta (`TestArtifact`): individuo → documento
→ disco → documento → imagen, comparando píxel a píxel.

## Configuración

`config.json` — todo lo que no esté definido toma el default de `ga/engine.py`.

```jsonc
{
  "image": "images/japan.png",
  "triangles": 20,              // parámetro del problema
  "canvas_size": 64,            // resolución a la que se evalúa el fitness
  "background": [255, 255, 255],

  "population_size": 50,        // N
  "offspring_size": 50,         // K

  // un nombre, o la selección combinada A% / (1-A)% que pide la cátedra
  "selection_parents": { "method_a": "elite", "method_b": "tournament_det", "a_ratio": 0.5 },
  "selection_survivors": { "method_a": "elite", "method_b": "universal", "a_ratio": 0.5 },
  "tournament": { "m": 4, "threshold": 0.75 },
  "boltzmann": { "t0": 100.0, "tmin": 1.0, "k": 0.01 },

  "crossover": "uniform",
  "crossover_rate": 0.85,
  "crossover_granularity": "triangle",   // "gene" | "triangle"
  "crossover_uniform_p": 0.5,

  "mutation": "non_uniform",
  "mutation_rate": 0.1,
  "mutation_sigma": 0.15,
  "mutation_genes": null,                // M para multigen (null = al azar)
  "mutation_decay_floor": 0.1,           // piso del decaimiento en no uniforme

  "replacement": "additive",

  "stop": {                              // se evalúan por OR; null = desactivado
    "max_generations": 500,
    "max_seconds": null,
    "target_fitness": null,
    "stall_generations": null,           // contenido: sin mejorar en G generaciones
    "structure_generations": null,       // estructura: sin recambio en G generaciones
    "structure_epsilon": 0.01
  },
  "seed": 42
}
```

### Métodos disponibles

| Operador | Valores |
|---|---|
| `selection_parents` / `selection_survivors` | `elite`, `roulette`, `universal`, `boltzmann`, `ranking`, `tournament_det`, `tournament_prob`, o `{method_a, method_b, a_ratio}` |
| `crossover` | `one_point`, `two_point`, `uniform`, `annular`, `spatial` (parte por posición en el canvas, no por índice) |
| `mutation` | `gene`, `multigene`, `uniform`, `non_uniform` |
| `replacement` | `additive`, `exclusive` |
| `stop` | `max_generations`, `max_seconds`, `target_fitness`, `stall_generations` (contenido), `structure_generations` (estructura) |
| `initialization` | `random`, `grid` (un triángulo por celda, con el color que tiene el target ahí) |

## Diseño

- **Individuo**: lista ordenada de N triángulos; el orden define el z-order al pintar.
- **Genotipo**: vector plano de `N*10` floats en `[0,1]` —
  `[x1,y1,x2,y2,x3,y3,r,g,b,a]` por triángulo. Cruza y mutación operan sobre el
  `np.ndarray` sin conocer la semántica.
- **Fitness**: `1 - RMSE/255` contra el target, en `[0,1]` y a maximizar (ruleta,
  universal y Boltzmann necesitan valores positivos). Se cachea en el individuo.
- **Motor**: `ga/engine.py` no conoce ningún método concreto; los toma por nombre del
  `METHODS` de cada módulo. Agregar un método nuevo es registrarlo ahí.

```
ga/
├── individual.py    # representación y genotipo
├── render.py        # genotipo -> bitmap (Pillow)
├── fitness.py       # RMSE + caché
├── context.py       # contrato compartido por los operadores
├── selection.py     # 6 métodos + selección combinada
├── crossover.py     # 4 métodos, con granularidad gen/triángulo
├── mutation.py      # 4 métodos
├── replacement.py   # aditiva y exclusiva
├── stopping.py      # 5 criterios de corte, evaluados por OR
├── artifact.py      # formato de salida autosuficiente + decodificador
└── engine.py        # loop generacional + métricas
```

> (!) El enunciado no define cuál de las dos supervivencias es la "aditiva" y cuál
> la "exclusiva". Asumimos: **aditiva** = compiten los N padres con los K hijos y
> sobreviven N del pool N+K; **exclusiva** = los K hijos desplazan a los padres
> (si K < N se completa con padres). Consultado a la cátedra; si fuera al revés,
> se intercambian las claves de `METHODS` en `ga/replacement.py`.

## Experimentos

```bash
python analysis/run_experiments.py            # barrido completo
python analysis/run_experiments.py selection  # un solo eje
python analysis/run_experiments.py --quick    # grilla reducida, para verificar
python analysis/plot_results.py               # figuras a partir de los CSV
```

- Grilla: `analysis/experiments.json` (config base + un eje por experimento + semillas)
- Resultados: `analysis/results/<eje>.csv` (una fila por generación, variante y semilla)
  y `analysis/results/summary.csv` (una fila por variante)
- Figuras: `analysis/figures/<eje>.png` (convergencia y diversidad, promedio ± desvío)

Cada experimento varía **un solo eje** sobre la misma base y repite con varias
semillas, para reportar promedio y desvío en vez de una corrida suelta.

## Tests

```bash
python -m unittest test_ga.py -v
```

## Resultados

Con la configuración de `config.json` (inicialización en grilla, torneo determinístico
M=4, cruza uniforme por triángulo, mutación multigen pm=0.1, supervivencia aditiva,
N=K=50, 500 generaciones, canvas 64px):

| Target | Triángulos | Fitness | RMSE |
|---|---|---|---|
| `images/germany.png` | 10 | 0.966 | 8.6 |
| `images/japan.png` | 20 | 0.933 | 17.0 |
| `images/cross.png` | 15 | 0.911 | 22.7 |

Esa configuración se eligió midiendo **configuraciones completas** sobre las tres
imágenes, no combinando el ganador de cada eje: al hacer eso último el promedio cae de
0.939 a 0.890 (ver el hallazgo 3).

### Qué salió de los experimentos

Cada eje se corrió con 3 semillas, 300 generaciones, canvas 48px, N=K=40, y **sobre dos
tipos de imagen**: una plana (bandera de Japón) y una con detalle fino (Pikachu). Los
números están en `analysis/results/summary.csv`.

| Eje | Imagen plana | Imagen detallada |
|---|---|---|
| **Inicialización** | grilla **0.931** vs azar 0.899 | grilla **0.859** vs azar 0.842 |
| **Selección** | torneo det **0.899** > ranking 0.892 > ruleta 0.885 > Boltzmann 0.884 > universal 0.883 > elite 0.881 > torneo prob 0.880 | ranking **0.842** ≈ torneo det 0.842 > ruleta 0.838 > Boltzmann 0.836 > torneo prob 0.835 > elite 0.833 > universal 0.831 |
| **Supervivencia** | exclusiva K=2N **0.919** ≈ aditiva K=2N 0.914 > exclusiva K=N 0.901 ≈ aditiva K=N 0.899 | aditiva K=2N **0.859** ≈ exclusiva K=2N 0.857 > aditiva K=N 0.842 > exclusiva K=N 0.840 |
| **Cruza** | uniforme **0.907** > dos puntos 0.899 ≈ un punto 0.899 > anular 0.896 ≈ espacial 0.896 | uniforme **0.848** > anular 0.847 > dos puntos 0.846 > un punto 0.842 > espacial 0.841 |
| **Granularidad** | por triángulo **0.913** > por gen 0.907 | por triángulo **0.854** > por gen 0.848 |
| **Mutación** | multigen **0.914** > uniforme 0.913 > gen 0.905 > no uniforme 0.902 | multigen **0.853** > uniforme 0.851 > gen 0.851 > no uniforme 0.850 |
| **Tasa de mutación** | pm=0.02 **0.913** > 0.05 (0.911) > 0.005 (0.907) ≫ 0.2 (0.865) | pm=0.005 **0.853** > 0.02 (0.851) > 0.05 (0.848) ≫ 0.2 (0.836) |
| **Triángulos** | 10 → 0.904, 25 → 0.895, 50 → 0.859, 100 → 0.800 | 10 → 0.846, 25 → 0.844, 50 → 0.821, 100 → 0.780 |

### Hallazgos

**1 · La inicialización informada es la mejora más grande de todas.** Arrancar de una
grilla con los colores que el target tiene en cada celda gana en las dos imágenes, y no
es una ventaja inicial que se diluya: se sostiene hasta el final y además baja el desvío
entre semillas. Ninguna elección de operador mueve tanto la aguja.

**2 · La supervivencia exclusiva no es mala; lo malo es una combinación.** Con selección
de padres por elite y `k = N` —donde elite devuelve a toda la población— más supervivencia
exclusiva con K=N, el algoritmo se queda sin ninguna presión de selección y se vuelve una
caminata aleatoria. Ninguna de las dos piezas sola rompe nada: medido con torneo en los
padres, la exclusiva K=N rinde igual que la aditiva K=N. Lo que decide es **K**, no la
estrategia: con K=2N las dos son las mejores.

**3 · Los ganadores por eje no componen, y lo comprobamos dos veces.** Armar la
configuración con el mejor de cada experimento da 0.890 de promedio; la mejor
configuración completa da 0.939. La diferencia entre ambas es **sólo la tasa de
mutación** (0.02 contra 0.1): la tasa óptima depende de la escala a la que se corre, y
optimizarla en un experimento de 300 generaciones no la transfiere a uno de 500.

**4 · La granularidad de la cruza sí importa.** Cortar por triángulo gana en las dos
imágenes. En una tanda anterior nos había dado empate, pero esa tanda usaba una base sin
presión de selección en los padres; con una base sana la diferencia aparece.

**5 · El óptimo depende del tipo de imagen.** La tasa de mutación ideal es 0.02 en la
imagen plana y 0.005 en la detallada. Correr todo sobre un solo target habría escondido
esto.

**6 · La cruza espacial no funciona.** Partir por posición en el canvas en vez de por
índice sale última en las dos imágenes. La idea es buena en teoría —el índice es
arbitrario, la posición no— pero el cromosoma de largo fijo no la deja expresarse: el
hijo no puede cambiar cuántos triángulos hay por región. Lo reportamos como resultado
negativo.
