# TP 2 — Algoritmos Genéticos (SIA, 2026 2Q)

Resumen del enunciado + plan de trabajo del grupo.
Enunciado original: [`../SIA - TP2 - 2026 2Q.pdf`](../SIA%20-%20TP2%20-%202026%202Q.pdf)

---

## 1. De qué se trata el TP

Implementar **desde cero un motor de Algoritmos Genéticos** (se pueden usar librerías
para manejo de imágenes, **no** para el AG) y usarlo para **aproximar una imagen con N
triángulos** de color uniforme (opcionalmente traslúcidos) sobre un canvas blanco.
Es, en la práctica, un compresor de imágenes con pérdida: la salida es la lista de
triángulos.

### Ejercicio 1 (no lo hacemos)
Es sólo teórico: pensar —sin implementar— cómo representar una imagen cuadrada en un
mapa de NxN caracteres ASCII con AG. Alcanza con un par de slides en la presentación
(individuo = matriz de caracteres, fitness = similitud entre el render del ASCII y la
imagen original). **Todo el trabajo real es el Ejercicio 2.**

### Ejercicio 2 — el TP

**Parámetros del problema** (no confundir con hiperparámetros del AG):
1. la imagen a procesar
2. la cantidad de triángulos

**Input**
- Imagen
- Cantidad de triángulos
- Hiperparámetros del AG (población, selección, cruza, mutación, corte, etc.)

**Output**
- Imagen generada
- Enumeración de triángulos (posición, color, …)
- Métricas para defender la implementación (fitness, error, generaciones, tiempo…)

**Obligatorio implementar**
- **Selección** (los 6): Elite, Ruleta, Universal, Boltzmann, Torneos (determinístico y
  probabilístico), Ranking.
- **Supervivencia / reemplazo**: aditiva y exclusiva (ambas).
- **Cruza**: al menos **2** de {un punto, dos puntos, uniforme, anular}.
- **Mutación**: al menos **2** de {gen, multigen, uniforme, no uniforme}.
- **Criterio(s) de corte**: decidir y justificar (max generaciones, estructura,
  contenido, etc.).
- **Justificar** la estructura del individuo y la función de aptitud.
- Justificar *qué* método de cruza/mutación conviene en *qué* circunstancia.

**Entregable (digital)**
- Código fuente
- Presentación
- README con cómo ejecutar el programa

**Opcionales**
- N triángulos como cota máxima + error mínimo como condición de corte.
- Otros polígonos u óvalos `(x, y, rx, ry, θ)`.
- Otros métodos de cruza y mutación.
- Experimentar con los "horizontes" de los AG vistos en clase.

**Consejo del enunciado**: empezar con imágenes simples — banderas, siluetas,
pictogramas, señales de tráfico, logos, iconos, símbolos, emojis.

---

## 2. Decisiones de diseño (las preguntas del enunciado, respondidas)

Estas son las respuestas que hay que fijar **antes** de experimentar. Sirven directo
para la presentación.

| Pregunta | Nuestra respuesta propuesta |
|---|---|
| ¿Qué es un individuo? | Una imagen candidata: una lista **ordenada** de N triángulos (el orden importa: define el z-order al pintar). |
| ¿Cuáles son sus genes? | Cromosoma real de largo `N*10`: por triángulo `(x1,y1,x2,y2,x3,y3,R,G,B,A)` normalizado a `[0,1]`. Un **gen** = un valor real; un **bloque** = un triángulo (10 genes). |
| ¿Cómo evalúo la aproximación? | Renderizo el genotipo a un bitmap y comparo píxel a píxel contra el target: `MSE` (o `RMSE`) en RGB. |
| ¿Qué es el fitness? | `fitness = 1 - RMSE/255` (o `1/(1+MSE)`), acotado a `[0,1]`. Debe ser **positivo y a maximizar** porque ruleta / universal / Boltzmann lo requieren. |
| ¿Cómo muta un individuo? | Perturbación gaussiana `N(0, σ)` sobre genes reales, con clamp a `[0,1]`. Variantes: 1 gen, varios genes, probabilidad uniforme, o σ/probabilidad decreciente con las generaciones (no uniforme). |
| ¿Cómo cruzo? | Sobre el vector de genes. **Cuidado**: cortar en medio de un triángulo genera hijos destructivos; por eso implementamos cruza a nivel **bloque/triángulo** además de a nivel gen, y comparamos. |
| ¿Cuál es la versión más simple? | Imagen 64x64, 10-20 triángulos, población 50, selección elite, cruza de un punto, mutación de un gen, corte por max generaciones. **Ese es el MVP del Step 4.** |
| ¿Qué imagen y cuántos triángulos para iterar rápido? | Imagen chica (64-128 px) y pocos triángulos: el costo por evaluación es `O(N_triángulos * área)`, y el costo total es `población * generaciones * eval`. Con imágenes grandes no se puede iterar. |
| ¿Alcanza implementar parcialmente? | Sí para *evaluar el motor*: con un método de cada tipo ya se puede medir. Pero la entrega exige los 6 de selección, 2 de reemplazo, ≥2 cruzas y ≥2 mutaciones. |

**Riesgo conocido a mencionar en la presentación**: *competing conventions* — dos
individuos buenos pueden codificar la misma imagen con los triángulos en distinto
orden, y la cruza entre ellos produce hijos malos. Es el argumento principal para
preferir cruza uniforme a nivel triángulo y/o mutación fuerte con poca cruza.

---

## 3. Estructura de archivos propuesta

```
Tp_2/
├── README.md                 # cómo ejecutar (entregable)
├── requirements.txt          # pillow, numpy, matplotlib
├── config.json               # imagen, N triángulos e hiperparámetros
├── main.py                   # CLI: lee config, corre el motor, escribe output/
├── ga/
│   ├── individual.py         # Triangle, Individual, random_individual
│   ├── render.py             # genotipo -> imagen (Pillow)
│   ├── fitness.py            # MSE/RMSE + caché
│   ├── selection.py          # elite, ruleta, universal, boltzmann, torneos, ranking
│   ├── crossover.py          # un punto, dos puntos, uniforme, anular
│   ├── mutation.py           # gen, multigen, uniforme, no uniforme
│   ├── replacement.py        # aditiva, exclusiva
│   ├── stopping.py           # criterios de corte
│   └── engine.py             # loop generacional + registro de métricas
├── analysis/
│   ├── run_experiments.py    # barrido de configuraciones -> results/*.csv
│   ├── plot_results.py       # figuras para la presentación
│   ├── results/
│   └── figures/
├── images/                   # targets (banderas, emojis, logos)
├── output/                   # imagen final, triangles.json, metrics.csv, gif
└── docs/
```

Convención heredada del TP 1: configuración en `config.json`, resultados en CSV,
figuras en `analysis/figures/`.

---

## 4. Plan por steps

**Estado al 2026-09-04: Steps 0-4 hechos** (setup, representación + render, fitness,
motor genérico con operadores inyectables, y MVP corriendo end to end con
elite / un punto / mutación de gen / supervivencia aditiva). Ver `../README.md`.
Los Steps 5-8 son los que se pueden repartir ahora.

Cada step tiene un entregable verificable. Los steps 5-8 son **paralelizables entre
integrantes** una vez cerrado el step 4 (todos comparten las mismas interfaces).

### Step 0 — Setup  ✅
- `venv` + `requirements.txt` (`pillow`, `numpy`, `matplotlib`).
- Crear el esqueleto de carpetas.
- Elegir 2-3 imágenes target simples y guardarlas en `images/` (ej.: bandera de Japón —
  trivial; bandera de Argentina o Alemania — franjas; un emoji — más difícil).
- **DoD**: `python main.py --help` corre.

### Step 1 — Representación y render  ✅
- `Triangle` / `Individual` con conversión **genotipo ↔ vector de floats** (`to_vector`
  / `from_vector`), así toda cruza y mutación trabaja sobre un `np.ndarray`.
- `render(individual, size) -> np.ndarray` con Pillow: canvas blanco + un
  `ImageDraw.polygon` por triángulo, con alpha (usar capa RGBA y `alpha_composite`).
- **DoD**: renderizar un individuo random y guardarlo como PNG.

### Step 2 — Fitness  ✅
- Cargar el target, escalarlo al tamaño de trabajo (parametrizable, arrancar en 64-128 px).
- `fitness(individual) -> float` en `[0,1]` a maximizar.
- Cachear el fitness en el individuo (no recalcular si no mutó): es el 90% del tiempo
  de cómputo.
- **DoD**: fitness de la imagen target contra sí misma = 1.0; de un canvas blanco < 1.

### Step 3 — Motor genérico (esqueleto)  ✅
- `engine.run(config)`: población inicial → loop {selección de padres → cruza →
  mutación → reemplazo → corte} → registro de métricas por generación.
- **Todas las estrategias se inyectan por config** (dict de nombre → función), sin
  `if` regados por el código.
- Firmas estables desde acá (contrato entre los integrantes):
  - `select(population, k, ctx) -> list[Individual]`
  - `crossover(parent_a, parent_b, rng) -> (child_a, child_b)`
  - `mutate(individual, ctx, rng) -> Individual`
  - `replace(parents, children, k, ctx) -> list[Individual]`
  - `should_stop(state) -> bool`
  - `ctx` lleva generación actual, total de generaciones, temperatura, rng, etc.
- **DoD**: corre 10 generaciones con selección elite + cruza dummy y el mejor fitness no baja.

### Step 4 — MVP end to end  ✅
- Config mínima: 64x64, 20 triángulos, población 50, elite + un punto + mutación de un
  gen + corte por max generaciones.
- **DoD**: sobre la bandera de Japón, el fitness sube visiblemente y la imagen de salida
  se parece al target. **Recién acá se paraleliza el trabajo.**

### Step 5 — Selección (los 6)
- Elite, Ruleta, Universal (puntero estocástico), Boltzmann
  (`T(t) = Tmin + (T0-Tmin)·e^(-k·t)`), Torneo determinístico (M individuos, gana el
  mejor), Torneo probabilístico (2 individuos, gana el mejor con prob. `Th≈0.75`),
  Ranking.
- Soportar **selección combinada**: `A%` con el método 1 + `(1-A)%` con el método 2.
- **DoD**: test que verifica que cada método devuelve exactamente `k` individuos y que
  ruleta/universal favorecen a los de mayor fitness (test estadístico simple).

### Step 6 — Cruza (≥2, implementar las 4 si da)
- Un punto, dos puntos, uniforme, anular, sobre el vector de genes.
- Además: **variante "por triángulo"** (el punto de corte cae siempre en múltiplos de
  10) → es el experimento interesante para la presentación.
- Probabilidad de cruza `pc` configurable.
- **DoD**: los hijos son genotipos válidos (longitud correcta, valores en `[0,1]`).

### Step 7 — Mutación (≥2, implementar las 4 si da)
- **Gen**: muta un gen al azar con prob. `pm`.
- **Multigen**: cada gen muta con prob. `pm`.
- **Uniforme**: `pm` constante durante toda la corrida.
- **No uniforme**: `pm` y/o `σ` decrecen con la generación (exploración → explotación).
- Perturbación gaussiana + clamp. Considerar `σ` distinto para coordenadas vs. color.
- **DoD**: con mutación sola (sin cruza) el fitness también mejora — es el sanity check
  de que la mutación es útil.

### Step 8 — Supervivencia / reemplazo
- **Aditiva**: se generan K hijos, se arma el pool `N + K` y se seleccionan N con los
  métodos de selección.
- **Exclusiva**: si `K ≥ N`, se seleccionan N de los K hijos; si `K < N`, pasan los K
  hijos y se completan `N-K` seleccionando de los padres.
- Igual que en selección, soportar combinación `B%` / `(1-B)%`.
- **DoD**: con aditiva + elite el mejor fitness es monótono no decreciente (elitismo).

### Step 9 — Criterios de corte
Implementar y poder combinarlos con OR:
- Máxima cantidad de generaciones (siempre activo, como cota).
- Tiempo máximo.
- **Contenido**: el mejor fitness no mejora en G generaciones.
- **Estructura**: menos del X% de la población cambia durante G generaciones.
- **Entorno a la solución**: fitness ≥ objetivo (es el opcional del "error mínimo").
- **DoD**: el motor reporta *por qué* cortó.

### Step 10 — I/O, CLI y métricas
- `config.json` con imagen, cantidad de triángulos e hiperparámetros; `main.py` lo lee
  (con overrides por CLI).
- Outputs: `output/best.png`, `output/triangles.json` (la "compresión": posición, color
  y alpha de cada triángulo), `output/metrics.csv` con una fila por generación
  (generación, mejor fitness, fitness promedio, desvío, diversidad, tiempo acumulado).
- Nice to have barato: GIF de la evolución — vende muchísimo en la presentación.
- **DoD**: una corrida deja todo lo que pide el enunciado en `output/`.

### Step 11 — Experimentación y análisis
Es lo que se defiende en la presentación. Un experimento = una config, varias semillas,
promedio ± desvío. Ejes a barrer:
- Cantidad de triángulos vs. fitness alcanzado vs. tiempo.
- Comparación de los 6 métodos de selección (fitness vs. generación).
- Aditiva vs. exclusiva.
- Cruza a nivel gen vs. a nivel triángulo.
- Mutación uniforme vs. no uniforme.
- Tamaño de población vs. velocidad de convergencia.
- Diversidad de la población en el tiempo (evidencia de convergencia prematura).
- **DoD**: `analysis/figures/` con las figuras que van a la presentación.

### Step 12 — Entregables
- `README.md`: setup, cómo correr, formato de `config.json`, cómo reproducir las figuras.
- Presentación: problema, representación y fitness (justificados), métodos
  implementados, resultados de los experimentos, conclusiones, + las slides del
  Ejercicio 1.

---

## 5. División sugerida (5 integrantes)

Steps 0-4 se hacen entre todos o de a dos, porque definen las interfaces. Después:

| Persona | Steps |
|---|---|
| 1 | Step 5 — selección (los 6 métodos + combinada) |
| 2 | Step 6 — cruza (4 métodos + variante por triángulo) |
| 3 | Step 7 — mutación (4 métodos) |
| 4 | Step 8 + 9 — reemplazo y criterios de corte |
| 5 | Step 10 — CLI, config, métricas, GIF |

Step 11 (experimentos) y 12 (presentación) entre todos, al final.

---

## 6. Hiperparámetros a exponer en `config.json`

```jsonc
{
  "image": "images/japan.png",
  "triangles": 50,
  "canvas_size": 128,
  "background": [255, 255, 255],

  "population_size": 100,
  "offspring_size": 100,          // K

  "selection": {
    "parents":  { "method_a": "elite", "method_b": "roulette", "a_ratio": 0.6 },
    "replacement": { "method_a": "tournament_det", "method_b": "universal", "a_ratio": 0.5 }
  },
  "tournament": { "m": 4, "threshold": 0.75 },
  "boltzmann":  { "t0": 100, "tmin": 1, "k": 0.01 },

  "crossover": { "method": "uniform", "rate": 0.85, "granularity": "triangle" },
  "mutation":  { "method": "multigen", "rate": 0.1, "sigma": 0.1, "non_uniform": true },

  "replacement_strategy": "additive",   // "additive" | "exclusive"

  "stop": {
    "max_generations": 5000,
    "max_seconds": 600,
    "target_fitness": 0.98,
    "content_generations": 200,
    "structure_generations": 200,
    "structure_epsilon": 0.01
  },

  "seed": 42
}
```

---

## 7. Restricción a no olvidar

> (!) Es posible utilizar librerías externas para el manejo de imágenes, pero **no** para
> la implementación de Algoritmos Genéticos.

Pillow / numpy / matplotlib: **sí**. DEAP, PyGAD o similares: **no**.
