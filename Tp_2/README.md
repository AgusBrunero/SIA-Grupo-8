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
  "preserve_aspect": false,     // true = canvas_size es el lado LARGO y se respeta la proporción
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
  "mutation_zorder_rate": 0.0,           // swap de z-order; 0 = apagado (default)

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
| `mutation` | `gene`, `multigene`, `uniform`, `non_uniform`, `zorder` |
| `mutation_zorder_rate` | Perilla **ortogonal**, apagada por defecto (`0.0`): con probabilidad p aplica el swap de `zorder` **encima** del método elegido. Se aplica sobre cualquiera de los cuatro que perturban valores |
| `replacement` | `additive`, `exclusive` |
| `stop` | `max_generations`, `max_seconds`, `target_fitness`, `stall_generations` (contenido), `structure_generations` (estructura) |
| `initialization` | `random`, `grid` (un triángulo por celda, con el color que tiene el target ahí) |

> **`zorder` es un extra, no un eje del barrido.** Los cuatro primeros métodos perturban
> valores dentro de un locus fijo; `zorder` no toca ningún valor, sólo intercambia dos
> triángulos de lugar. Es el único operador que se mueve en la dimensión del orden de
> pintado, que es donde vive el problema de *competing conventions* que la cruza sufre.
> Usado solo no puede refinar una imagen, así que para combinarlo con perturbación está
> la perilla. **Ninguno de los dos entra al barrido**: las 945 corridas se midieron con
> `mutation_zorder_rate = 0.0` y sin `mutation: zorder`.

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

> **Supervivencia**, según las láminas 45 y 46 del deck de AG: **aditiva** = compiten
> los N padres con los K hijos y sobreviven N del pool N+K; **exclusiva** = con K > N se
> seleccionan N de los K hijos exclusivamente, y con K ≤ N pasan los K hijos más N−K
> individuos de la generación actual. Las definiciones textuales están citadas en el
> docstring de `ga/replacement.py`.

## Experimentos

```bash
python analysis/run_experiments.py --clean     # barrido completo, tanda limpia (~2 min)
python analysis/run_experiments.py selection   # un solo eje
python analysis/run_experiments.py --quick     # grilla reducida, escribe en results_quick/
python analysis/plot_results.py                # figuras a partir de los CSV
```

- Grilla: `analysis/experiments.json` (config base + un eje por experimento + semillas)
- Resultados: `analysis/results/<eje>.csv` (una fila por generación, variante y semilla)
  y `analysis/results/summary.csv` (una fila por variante)
- **Procedencia**: `analysis/results/manifest.json` — el commit con el que se corrió, la
  config base ya resuelta contra los defaults del motor, y para cada variante sus
  overrides y su configuración completa
- Figuras: `analysis/figures/<eje>_<target>.png`

Cada experimento varía **un solo eje** sobre la misma base y repite con varias semillas,
para reportar promedio, desvío y rango intercuartil en vez de una corrida suelta.

**Los resultados y las figuras se versionan** (igual que en el TP1): son el respaldo de
los números de este README y de la presentación. `--quick` escribe en `results_quick/`
para no pisarlos, y `--clean` borra la tanda anterior antes de correr — sin eso,
`summary.csv` puede mezclar corridas de escalas distintas.

### Caso final

```bash
python analysis/caso_final.py    # La noche estrellada (~20 min)
```

Cierre del análisis sobre la imagen que la cátedra mostró como ejemplo. La
configuración **no se barre**: se elige con lo que midieron los ocho ejes, y el informe
dice de qué eje sale cada decisión. Lo único que se barre es la cantidad de triángulos,
que es un parámetro del problema y depende de la imagen.

### Las figuras traen su propio contexto

Cada PNG es autocontenido: a la izquierda lleva la miniatura del target y el bloque de
hiperparámetros que **no** variaron en ese experimento, más el batch y el commit. El
rótulo se genera desde `manifest.json`, no está escrito a mano: si cambia la
configuración, la figura cambia sola.

Tres paneles por figura:

| Panel | Qué muestra |
|---|---|
| Convergencia | Mejor fitness por generación, media entre semillas con banda ±σ. **Línea sólida**: mejor global acumulado (monótono). **Punteada**: mejor de la población actual — con supervivencia exclusiva puede bajar, porque los hijos desplazan a los padres |
| Diversidad | Diversidad genética por generación, en escala log. Es el panel que explica *por qué* una configuración se estanca |
| Fitness final | Boxplot por variante, con los puntos crudos de cada semilla encima. Con 3 semillas, si las cajas se solapan no hay ganador |

## Tests

```bash
python -m unittest test_ga.py -v
```

## Resultados

**Todos los números de esta sección salen del batch `20260906T213411Z`**
(945 corridas · 12 ejes · 3 imágenes · 5 semillas · 800 generaciones).
Los datos crudos están en `analysis/results/<eje>.csv`, el agregado en
`analysis/results/summary.csv`, y `analysis/results/manifest.json` guarda la configuración
exacta con la que se corrió cada variante. El análisis completo, con figuras y lectura, está
en [`analysis/informe/`](analysis/informe/README.md).

### Cómo leer estos números

Con 5 semillas, una diferencia de milésimas no es un resultado. El criterio en todo el
análisis es el **rango intercuartil**: si el del primero se solapa con el del segundo, la
conclusión es que **empatan**, y así queda escrito. La última columna dice en qué imágenes el
ganador está realmente separado.

### Ganador por eje

| Eje | Gana en `plana` | Gana en `detallada` | Gana en `compleja` | ¿Separado del 2º? |
|---|---|---|---|---|
| Selección de padres | ranking | torneo det | torneo det | **en ninguna** |
| Presión de selección | torneo M=30 | torneo M=10 | torneo M=30 | plana |
| Supervivencia × K | exclusiva K=2N | exclusiva K=2N | aditiva K=2N | plana |
| Método de cruza | uniforme | un punto | uniforme | plana |
| Granularidad de cruza | corte por componente | corte por triangulo | corte por triangulo | **en ninguna** |
| Probabilidad de cruza | pc=1.00 | pc=0.85 | pc=0.50 | plana |
| Método de mutación | gen (carga 1) | uniforme (carga 4) | gen (carga 1) | plana |
| Carga de mutación | carga 2 | carga 4 | carga 2 | plana |
| Magnitud σ | sigma=0.20 | sigma=0.10 | sigma=0.10 | plana |
| Tamaño de población | N=K=120 | N=K=120 | N=K=120 | plana, compleja |
| Cantidad de triángulos | 25 tri - pm fijo | 50 tri - pm fijo | 100 tri - carga fija | **en ninguna** |
| Inicialización | grilla informada | grilla informada | grilla informada | plana, compleja |

**De 12 ejes, el ganador cambia entre imágenes en 10** — pero sólo en **6** ese cambio tiene
evidencia estadística, en el sentido de que el ganador de una imagen queda separado *por
debajo* en otra: cruza, presión, σ, supervivencia, tasa de cruza y triángulos.

### Cuánto importa cada eje

Tamaño del efecto = fitness del mejor menos el del peor, por imagen. El normalizado divide
por el error remanente `(1 − mejor_fitness)`, para que sea comparable entre imágenes que
alcanzan distinta calidad.

| Eje | plana | detallada | compleja | Normalizado (plana) |
|---|---|---|---|---|
| Magnitud σ | 0.0508 | 0.0259 | 0.0163 | 1.022 |
| Carga de mutación | 0.0511 | 0.0337 | 0.0222 | 0.950 |
| Inicialización | 0.0262 | 0.0061 | 0.0047 | 0.768 |
| Supervivencia × K | 0.0281 | 0.0185 | 0.0087 | 0.648 |
| Presión de selección | 0.0234 | 0.0120 | 0.0057 | 0.530 |
| Cantidad de triángulos | 0.0277 | 0.0313 | 0.0218 | 0.468 |
| Probabilidad de cruza | 0.0187 | 0.0137 | 0.0050 | 0.378 |
| Tamaño de población | 0.0166 | 0.0191 | 0.0073 | 0.326 |
| Método de mutación | 0.0179 | 0.0112 | 0.0058 | 0.318 |
| Método de cruza | 0.0143 | 0.0044 | 0.0029 | 0.310 |
| Selección de padres | 0.0086 | 0.0098 | 0.0059 | 0.145 |
| Granularidad de cruza | 0.0032 | 0.0028 | 0.0008 | 0.069 |

**La carga de mutación y σ son los dos ejes que más mueven la aguja** (0.051 en la imagen
plana, los mayores del barrido). La granularidad de la cruza es el que menos (0.003), y el
método de selección —los seis que el enunciado obliga a implementar— queda entre los últimos.

### Los hallazgos

**1 · Cuanta más superficie plana tiene la imagen, más importa qué operador elijas.**
El efecto medio de los 12 ejes cae de **0.0239** (plana) a **0.0157** (detallada) y **0.0089**
(compleja); normalizado por error remanente, de **0.494** a **0.131** y **0.104**. En una
imagen con regiones grandes y uniformes hay margen para que la búsqueda haga diferencia; en
una que es textura en todos lados, todas las configuraciones convergen a un resultado
parecido y la elección deja de pesar.

> No es un efecto de "estar más lejos de la asíntota": medido sobre las 945 corridas, la
> imagen plana es la que **más** sigue mejorando en las últimas 100 generaciones (0.383%
> contra 0.255% y 0.168%). La que menos margen tiene es justamente la que menos separa.

**2 · La ventaja de las poblaciones grandes es presupuesto, no método.** Por generación,
N=120 gana en las tres imágenes de forma monótona. A igual presupuesto de evaluaciones el
orden **se invierte por completo**, también monótono y también en las tres. N=120 necesita
entre 3 y 4 veces más evaluaciones sólo para *empatar* lo que N=15 logra con su presupuesto,
y lo supera recién con 8×.

**3 · Los ganadores por eje no componen.** Reemplazando un operador por vez desde la
configuración base, las diez mejoras individuales suman **+0.0414**; aplicadas todas juntas
rinden **+0.0164**. Se evapora el 60%. Un barrido de un factor por vez sirve para *entender*
qué hace cada operador, no para *elegir* una configuración.

**4 · El operador que más aporta no es un operador del algoritmo genético.** En ese mismo
experimento, la **inicialización informada** aporta +0.0123, el 75% del efecto conjunto; los
otros nueve operadores ganadores, sumados, aportan menos que ella. Y la selección de padres
aporta exactamente **+0.0000**. La grilla informada es conocimiento del target inyectado
*antes* de que el algoritmo arranque — por eso no está en la configuración base de ningún eje.

**5 · Un hiperparámetro medido a una escala no se copia a otra: se traduce.** `pm` es una
probabilidad por gen, así que al alargar el cromosoma el mismo `pm` multiplica la carga de
mutación. En el caso final, mantener constante la **carga** en vez del `pm` gana en las cuatro
cantidades de triángulos probadas, y la ventaja crece con el largo del cromosoma.

**6 · El ganador de un eje puede no ser transferible.** El método `gen` gana en dos de las
tres imágenes con 50 triángulos, pero muta **un solo gen** por individuo: su carga está
acotada a 1 sea cual sea el largo del cromosoma. Con 800 triángulos es estructuralmente
incapaz de competir.

### Criterios de corte

Están los cinco implementados, y la elección está medida en vez de argumentada: sobre las
945 corridas se reconstruye, para cada criterio, en qué generación habría disparado
y cuánto fitness habría costado (ver [`13-criterios-corte`](analysis/informe/13-criterios-corte/informe.md)).

| Criterio | ¿Dispara? | Generación (mediana) | Fitness perdido |
|---|---|---|---|
| contenido G=20 | 5% de las corridas | 441 | 0.0083 |
| contenido G=50 | 2% | 574 | 0.0045 |
| estructura G=20 | 0.2% | 602 | 0.0002 |
| **estructura G=50** | **nunca** | — | — |
| entorno fitness ≥ 0.90 | 64% | 322 | 0.0251 |

**Los criterios de contenido y estructura casi no disparan, y hay una razón de
representación:** los genes son reales y el criterio de estructura compara genomas por
igualdad exacta de bytes. Con mutación gaussiana la población nunca se congela — en un AG
binario dos individuos convergen a cadenas idénticas, acá convergen a cadenas *parecidas*.
El único operativo es el entorno a la solución, y su valor no es ahorrar cómputo sino permitir
comparar métodos a calidad igualada.

### Caso final: La noche estrellada

La imagen que la cátedra mostró como ejemplo. Los operadores **se heredan** del barrido (el
ganador de cada eje, promediado entre las tres imágenes); se barre sólo lo que depende del
tamaño del problema. Ver [`15-caso-final`](analysis/informe/15-caso-final/informe.md).

| | |
|---|---|
| Configuración | 800 triángulos, régimen `carga fija` (pm = 0.00025) |
| Resultado | **fitness 0.9442** (RMSE 14.23), 3000 generaciones, 240,080 evaluaciones, 804 s |
| Compresión | 604 KB (JPEG original) → 80 KB (`triangles.json`) = **7.6×** |

### Una corrida de ejemplo

`config.json` trae una configuración de demostración —no es la ganadora de ningún eje, para
eso está el informe— pensada para que una corrida termine en segundos:

```bash
python main.py                    # images/japan.png, 20 triángulos, 500 generaciones
# fitness 0.9333 (RMSE 17.01) | 23415 evaluaciones | corte: max_generations
```

## ¿Los hallazgos aguantan a otra escala?

La pregunta está respondida por dos vías, y la respuesta corta es **parcialmente**:

- El barrido ya corre a una escala razonable (50 triángulos, N=K=60, 800 generaciones), y el
  eje [`11-triangulos`](analysis/informe/11-triangulos/informe.md) lo estira de 10 a 200
  triángulos dentro del mismo barrido.
- El [caso final](analysis/informe/15-caso-final/informe.md) lleva la configuración heredada a
  100–800 triángulos sobre una imagen que el barrido nunca vio, y ahí aparece el límite:
  **la carga de mutación no se transfiere copiando `pm`**, y el método `gen` —ganador en dos
  de las tres imágenes a 50 triángulos— es estructuralmente incapaz de escalar.

O sea: los hallazgos sobre *qué operador* elegir aguantan; los hallazgos sobre *con qué
número* configurarlo, no — hay que traducirlos.

