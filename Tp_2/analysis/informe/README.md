# Informe de análisis — TP2

Generado desde el batch `20260906T233756Z` (commit `5b59273`, árbol sucio), 105 corridas, 5 semillas por variante.

Una carpeta por eje experimental. Cada una trae el informe, la configuración
anotada, los datos, las figuras y las imágenes resultado.

```
NN-<eje>/
├── informe.md      qué se ve y qué es importante analizar
├── config.md       configuración fija, variantes y cómo reproducir
├── datos/          CSV crudo del eje + resumen agregado
├── figuras/        un gráfico por PNG, sin texto, fondo transparente
└── imagenes/       el mejor individuo de cada variante, renderizado
```

## Ejes

| Carpeta | Eje | Var. | Gana en `plana` | Gana en `detallada` | Gana en `compleja` | ¿Separado del 2º? |
|---|---|---|---|---|---|---|
| [`01-seleccion`](01-seleccion/informe.md) | Selección de padres | 7 | ranking | torneo det | torneo det | en ninguna |
| [`02-presion-seleccion`](02-presion-seleccion/informe.md) | Presión de selección | 7 | torneo M=30 | torneo M=10 | torneo M=30 | plana |
| [`03-supervivencia`](03-supervivencia/informe.md) | Supervivencia y brecha generacional | 6 | exclusiva K=2N | exclusiva K=2N | aditiva K=2N | plana |
| [`04-cruza`](04-cruza/informe.md) | Método de cruza | 5 | uniforme | un punto | uniforme | plana |
| [`05-granularidad`](05-granularidad/informe.md) | Granularidad de la cruza | 2 | corte por componente | corte por triangulo | corte por triangulo | en ninguna |
| [`06-tasa-cruza`](06-tasa-cruza/informe.md) | Probabilidad de cruza | 5 | pc=1.00 | pc=0.85 | pc=0.50 | plana |
| [`07-mutacion`](07-mutacion/informe.md) | Método de mutación | 4 | gen (carga 1) | uniforme (carga 4) | gen (carga 1) | plana |
| [`08-tasa-mutacion`](08-tasa-mutacion/informe.md) | Carga de mutación | 6 | carga 2 | carga 4 | carga 2 | plana |
| [`09-sigma`](09-sigma/informe.md) | Magnitud de la mutación (σ) | 5 | sigma=0.20 | sigma=0.10 | sigma=0.10 | plana |
| [`10-poblacion`](10-poblacion/informe.md) | Tamaño de población | 4 | N=K=120 | N=K=120 | N=K=120 | compleja, plana |
| [`11-triangulos`](11-triangulos/informe.md) | Cantidad de triángulos | 10 | 25 tri - pm fijo | 50 tri - pm fijo | 100 tri - carga fija | en ninguna |
| [`12-inicializacion`](12-inicializacion/informe.md) | Inicialización de la población | 2 | grilla informada | grilla informada | grilla informada | compleja, plana |
| [`16-seleccion-combinada`](16-seleccion-combinada/informe.md) | Selección combinada A%/B% | 7 | 25% elite + 75% torneo | torneo det (puro) | torneo det (puro) | en ninguna |

## Análisis transversales y cierre

| Carpeta | Qué es |
|---|---|
| [`13-criterios-corte`](13-criterios-corte/informe.md) | **Criterios de corte.** Cuál usar y por qué, medido: para cada criterio candidato se reconstruye, sobre las corridas del barrido, en qué generación habría disparado y cuánto fitness habría costado. Responde lo que el enunciado pide justificar. Lo genera `python analysis/criterios_corte.py`. |
| [`14-interaccion`](14-interaccion/informe.md) | **¿Los ganadores de cada eje componen?.** El barrido varía un factor por vez; esto mide si juntar los ganadores da la mejor configuración, reemplazando un operador por vez desde la base. Lo genera `python analysis/interaccion.py`. |
| [`15-caso-final`](15-caso-final/informe.md) | **Caso final: La noche estrellada.** La imagen que la cátedra mostró como ejemplo. Los operadores se **heredan** del barrido; se barre sólo lo que depende del tamaño del problema. Lo genera `python analysis/caso_final.py`. |

## Las tres imágenes

Se eligieron por **composición**, no por dificultad: cuánta superficie plana tienen
contra cuánta textura. Ése es el eje que ordena los resultados.

| Etiqueta | Imagen | Composición |
|---|---|---|
| `plana` | bandera de Japón | Regiones planas grandes, un solo borde curvo |
| `detallada` | Pikachu | Detalle fino sobre fondo liso, contornos negros duros |
| `compleja` | La noche estrellada | Textura en todo el lienzo, sin regiones planas |

**No están ordenadas por dificultad**, y conviene decirlo antes de que alguien lo
note: medido por el error que le queda a la mejor configuración, la más difícil es
`detallada`, no `compleja`.

| Imagen | Mejor fitness alcanzado | Error remanente |
|---|---|---|
| `plana` | 0.9658 | 0.034 |
| `detallada` | 0.8844 | **0.116** |
| `compleja` | 0.9180 | 0.082 |

Los contornos negros duros de Pikachu son justamente lo que un triángulo de color
uniforme no puede reproducir; un óleo sin bordes se aproxima razonablemente bien con
un borrón de triángulos, porque el RMSE premia el promedio local.

## Cobertura del enunciado

| Lo que pide | Dónde está |
|---|---|
| Justificar la estructura del individuo | `README.md` § Diseño, y `docs/TP2.md` |
| Justificar la función de aptitud | `README.md` § Diseño |
| Los 6 métodos de selección | [`01-seleccion`](01-seleccion/informe.md) (7, con la combinada aparte) |
| Supervivencia aditiva y exclusiva | [`03-supervivencia`](03-supervivencia/informe.md), cruzada con la brecha generacional |
| Al menos 2 métodos de cruza | [`04-cruza`](04-cruza/informe.md) (5) y [`05-granularidad`](05-granularidad/informe.md) |
| Al menos 2 métodos de mutación | [`07-mutacion`](07-mutacion/informe.md) (4), a carga comparable |
| **Decidir y justificar el criterio de corte** | [`13-criterios-corte`](13-criterios-corte/informe.md) |
| Qué operador conviene **en qué circunstancia** | Cada eje se corre sobre 3 imágenes de composición distinta; el resumen por eje dice si el ganador cambia entre ellas |
| Convergencia prematura (3 causas de la cátedra) | Presión de selección: [`02`](02-presion-seleccion/informe.md) · Tasa de mutación: [`08`](08-tasa-mutacion/informe.md) · Tamaño de población: [`10`](10-poblacion/informe.md) |
| Cantidad de triángulos vs. calidad y tiempo | [`11-triangulos`](11-triangulos/informe.md) |

## Cómo leer los números

Cada variante se corrió con 5 semillas. Con tan pocas repeticiones,
**una diferencia de dos milésimas en la media no es un resultado**. El criterio que usamos
en todo el informe es el rango intercuartil: si el del primero se solapa con el del
segundo, la conclusión es que empatan. Los boxplot son el gráfico que lo muestra.

La curva de convergencia usa el **mejor fitness acumulado**, que es monótono por
construcción. El mejor de la población actual puede bajar con supervivencia exclusiva,
porque los hijos desplazan a los padres; ese caso se ve en `mejor_vs_promedio`.

## Base común

Todos los ejes varían **un solo parámetro** sobre esta misma configuración base:

| Parámetro | Valor |
|---|---|
| color de fondo (`background`) | `[255, 255, 255]` |
| Boltzmann (`boltzmann`) | `{"t0": 100.0, "tmin": 1.0, "k": 0.01}` |
| canvas (px) (`canvas_size`) | `64` |
| cruza (`crossover`) | `one_point` |
| granularidad de cruza (`crossover_granularity`) | `gene` |
| probabilidad de cruza pc (`crossover_rate`) | `0.85` |
| p de cruza uniforme (`crossover_uniform_p`) | `0.5` |
| inicialización (`initialization`) | `random` |
| mutación (`mutation`) | `uniform` |
| piso de decaimiento (`mutation_decay_floor`) | `0.1` |
| M (multigen) (`mutation_genes`) | `None` |
| probabilidad de mutación pm (`mutation_rate`) | `0.008` |
| σ de mutación (`mutation_sigma`) | `0.1` |
| mutation_zorder_rate (`mutation_zorder_rate`) | `0.0` |
| descendencia K (`offspring_size`) | `60` |
| población N (`population_size`) | `60` |
| preserve_aspect (`preserve_aspect`) | `True` |
| supervivencia (`replacement`) | `additive` |
| selección de padres (`selection_parents`) | `tournament_det` |
| selección de sobrevivientes (`selection_survivors`) | `elite` |
| criterios de corte (`stop`) | `{"max_generations": 800}` |
| torneo (`tournament`) | `{"m": 4, "threshold": 0.75}` |
| triángulos (`triangles`) | `50` |

La base usa inicialización `random` **a propósito**: la inicialización informada es uno
de los ejes a estudiar, y si estuviera en la base contaminaría la medición de todos los
demás.

## Reproducir todo

```bash
python analysis/run_experiments.py --clean   # barrido completo (~2 min)
python analysis/build_report.py              # este informe (~1 min)
```
