# Configuración — caso final (La noche estrellada)

Los operadores **no se eligieron a mano**: cada uno es el ganador de su eje en el
barrido, promediado entre las tres imágenes. La columna «¿se separa?» dice en qué
imágenes ese ganador tiene el rango intercuartil disjunto del segundo — o sea, dónde
la elección está respaldada y dónde es apenas la mejor media.

| Eje | Ganador heredado | Qué fija | ¿Se separa del 2º? | Qué es |
|---|---|---|---|---|
| [`01-seleccion`](../01-seleccion/informe.md) | **torneo det** | `selection_parents` = `tournament_det` | **en ninguna** (empate) | Método de selección de padres. |
| [`03-supervivencia`](../03-supervivencia/informe.md) | **exclusiva K=2N** | `replacement` = `exclusive` · `offspring_size` = `120` | plana | Estrategia de supervivencia y brecha generacional K/N. |
| [`04-cruza`](../04-cruza/informe.md) | **uniforme** | `crossover` = `uniform` | plana | Método de recombinación. |
| [`05-granularidad`](../05-granularidad/informe.md) | **corte por triangulo** | `crossover` = `uniform` · `crossover_granularity` = `triangle` | **en ninguna** (empate) | Unidad de corte de la cruza: componente suelto o triángulo entero. |
| [`06-tasa-cruza`](../06-tasa-cruza/informe.md) | **pc=1.00** | `crossover_rate` = `1.0` | plana | Probabilidad de recombinación pc. |
| [`09-sigma`](../09-sigma/informe.md) | **sigma=0.20** | `mutation_sigma` = `0.2` | plana | Magnitud de la perturbación gaussiana. |
| [`12-inicializacion`](../12-inicializacion/informe.md) | **grilla informada** | `initialization` = `grid` | compleja, plana | Población inicial. Es la decisión más discutible del trabajo: la grilla inyecta conocimiento del target antes del algoritmo. |

## Lo que no se hereda, y por qué

| Parámetro | Valor | Motivo |
|---|---|---|
| `mutation_rate` | se barre acá | `pm` es probabilidad **por gen**. El barrido lo calibró con 500 genes y acá el cromosoma tiene entre 1,000 y 8,000. Lo que se hereda es la **carga** (2 genes mutados por individuo), y se prueban las dos formas de trasladarla. |
| `triangles` | se barre acá | Es un parámetro del **problema**, no del algoritmo: depende de la imagen. |
| `mutation` | `uniform` (forzado) | El eje lo ganó `gen` en dos de las tres imágenes, pero `gen` muta **un solo gen** por individuo: su carga está acotada a 1 sea cual sea el largo del cromosoma. La pregunta de este caso —cómo trasladar una tasa **por gen** a un cromosoma más largo— no se puede formular con un método que no es por gen. Se fuerza `uniform`, que es el que ganó en la imagen detallada y el único de los cuatro cuya tasa escala. **Que el ganador de un eje no sea transferible a otra escala es parte del resultado**, no una excepción cómoda. |
| `population_size` / `offspring_size` | `80` | El eje de población se midió con 50 triángulos y a esta escala el costo por generación es otro. Se fija por presupuesto, no por medición: es una **limitación declarada**. |
| `canvas_size` | `96` | El fitness se evalúa a 96px de lado largo. Más resolución daría señal más fina y costaría proporcionalmente más. |

## Resultado del barrido

| Parámetro | Valores probados | Elegido |
|---|---|---|
| `triangles` | 100, 200, 400, 800 | **800** |
| régimen de mutación | pm fijo, carga fija | **carga fija** |

## Sobre la proporción de la imagen

`La noche estrellada` es 1280×1014 (relación 1.26:1). Un canvas cuadrado la aplastaría
un 26%, inaceptable para ponerla al lado del original. Por eso `preserve_aspect` hace
que `canvas_size` sea el **lado largo**: el canvas de trabajo queda 96×76px.

## Reproducir

```bash
python analysis/run_experiments.py --clean   # el barrido del que se hereda
python analysis/caso_final.py
```

<details><summary>Configuración completa de la corrida final (JSON)</summary>

```json
{
  "triangles": 800,
  "initialization": "grid",
  "canvas_size": 96,
  "preserve_aspect": true,
  "background": [
    255,
    255,
    255
  ],
  "population_size": 80,
  "offspring_size": 80,
  "selection_parents": "tournament_det",
  "selection_survivors": "elite",
  "tournament": {
    "m": 4,
    "threshold": 0.75
  },
  "boltzmann": {
    "t0": 100.0,
    "tmin": 1.0,
    "k": 0.01
  },
  "crossover": "uniform",
  "crossover_rate": 1.0,
  "crossover_granularity": "triangle",
  "crossover_uniform_p": 0.5,
  "mutation": "uniform",
  "mutation_rate": 0.00025,
  "mutation_sigma": 0.2,
  "mutation_genes": null,
  "mutation_decay_floor": 0.1,
  "replacement": "exclusive",
  "stop": {
    "max_generations": 3000,
    "max_seconds": null,
    "target_fitness": null,
    "stall_generations": null,
    "structure_generations": null,
    "structure_epsilon": 0.01
  },
  "seed": 1,
  "image": "images/noche_estrellada.jpg"
}
```

</details>
