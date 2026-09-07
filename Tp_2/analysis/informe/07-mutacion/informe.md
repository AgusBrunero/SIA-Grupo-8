# Método de mutación

> Eje `mutacion` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Compara los cuatro métodos de mutación **a carga comparable**. La mutación es la única fuente de material genético nuevo: la cruza sólo recombina lo que ya está.

La comparación honesta no es a `pm` igual sino a **cantidad esperada de genes mutados por individuo** igual, porque `pm` significa cosas distintas en cada método. Las variantes están calibradas a carga 4 sobre 500 genes — salvo `gen`, que por construcción muta a lo sumo **un** gen y no puede llegar a 4. Ese techo estructural es parte del resultado.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | gen (carga 1) | 0.9145 ± 0.0028 | 0.9121–0.9174 | 21.82 | 493 | 2.17e-03 |
| 2 | uniforme (carga 4) | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 3 | multigen M=8 (carga 4) | 0.9122 ± 0.0033 | 0.9117–0.9127 | 22.38 | 376 | 6.01e-03 |
| 4 | no uniforme (carga 4 -> 0.4) | 0.9087 ± 0.0024 | 0.9077–0.9099 | 23.29 | 346 | 8.79e-05 |

**No hay un ganador claro.** gen (carga 1) tiene la media más alta, pero su rango intercuartil se solapa con el de uniforme (carga 4): con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **no uniforme (carga 4 -> 0.4)**, que queda separado (0.9087).

- **Rango del eje**: 0.0058 de fitness entre el mejor y el peor (1.5 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 346; el más lento, en la 493.
- **Diversidad**: 2 de 4 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **gen (carga 1)** en la generación 31.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **no uniforme (carga 4 -> 0.4)** (0.0000); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 46,397. Comparadas todas a las **46,397 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | gen (carga 1) | 0.9139 |
| 2 | uniforme (carga 4) | 0.9129 |
| 3 | multigen M=8 (carga 4) | 0.9118 |
| 4 | no uniforme (carga 4 -> 0.4) | 0.9087 |

El orden **se mantiene**: *gen (carga 1)* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | uniforme (carga 4) | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | multigen M=8 (carga 4) | 0.8758 ± 0.0046 | 0.8746–0.8786 | 31.68 | 503 | 5.96e-03 |
| 3 | gen (carga 1) | 0.8735 ± 0.0024 | 0.8742–0.8749 | 32.27 | 575 | 2.33e-03 |
| 4 | no uniforme (carga 4 -> 0.4) | 0.8669 ± 0.0030 | 0.8657–0.8691 | 33.93 | 390 | 1.13e-04 |

**No hay un ganador claro.** uniforme (carga 4) tiene la media más alta, pero su rango intercuartil se solapa con el de multigen M=8 (carga 4): con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **no uniforme (carga 4 -> 0.4)**, que queda separado (0.8669).

- **Rango del eje**: 0.0112 de fitness entre el mejor y el peor (2.9 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 390; el más lento, en la 575.
- **Diversidad**: 2 de 4 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **gen (carga 1)** en la generación 44.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **no uniforme (carga 4 -> 0.4)** (0.0000); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 46,397. Comparadas todas a las **46,397 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | uniforme (carga 4) | 0.8777 |
| 2 | multigen M=8 (carga 4) | 0.8753 |
| 3 | gen (carga 1) | 0.8727 |
| 4 | no uniforme (carga 4 -> 0.4) | 0.8669 |

El orden **se mantiene**: *uniforme (carga 4)* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | gen (carga 1) | 0.9437 ± 0.0039 | 0.9426–0.9463 | 14.35 | 631 | 2.26e-03 |
| 2 | multigen M=8 (carga 4) | 0.9409 ± 0.0036 | 0.9401–0.9413 | 15.08 | 583 | 5.63e-03 |
| 3 | uniforme (carga 4) | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 4 | no uniforme (carga 4 -> 0.4) | 0.9258 ± 0.0040 | 0.9234–0.9297 | 18.91 | 493 | 7.98e-05 |

**gen (carga 1)** gana con el rango intercuartil **separado** del segundo (multigen M=8 (carga 4)): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0179 de fitness entre el mejor y el peor (4.6 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 493; el más lento, en la 631.
- **Diversidad**: 2 de 4 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **gen (carga 1)** en la generación 29.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **no uniforme (carga 4 -> 0.4)** (0.0000); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 46,397. Comparadas todas a las **46,397 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | gen (carga 1) | 0.9425 |
| 2 | multigen M=8 (carga 4) | 0.9401 |
| 3 | uniforme (carga 4) | 0.9391 |
| 4 | no uniforme (carga 4 -> 0.4) | 0.9258 |

El orden **se mantiene**: *gen (carga 1)* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Qué mirar

- `gen` está limitado a carga ≤ 1 por diseño. Si queda último, la explicación no es que el método sea malo sino que **no puede aplicar suficiente mutación** en un cromosoma de 500 genes. Es el mismo fenómeno que se mide en [`11-triangulos`](../11-triangulos/informe.md).
- `no uniforme` arranca en la misma carga y la decae con las generaciones: mirá si su curva arranca igual que `uniforme` y se aplana antes. Es exploración temprana y ajuste fino tardío.
- En `diversidad`, `no uniforme` debería mostrar el colapso más pronunciado al final, porque su propia tasa se apaga.
- `multigen` con M fijo concentra las mutaciones en menos individuos-gen que `uniforme` con la misma carga esperada: mirá si esa diferencia de *varianza* (no de media) se traduce en algo.

## Trampas y advertencias

- **`mutation_rate` no significa lo mismo en cada método.** En `gen` es la probabilidad de que ocurra la única mutación; en `uniforme` es la probabilidad **por gen** sobre los 500. Por eso cada variante lleva su propio pm calibrado y el nombre de la variante indica la carga.
- La magnitud comparable entre métodos es la carga esperada, no `pm`. El barrido de la carga está en [`08-tasa-mutacion`](../08-tasa-mutacion/informe.md).

## Figuras

Sin título ni texto adentro del PNG, fondo transparente: el rótulo lo pone la slide.

| Archivo | Qué muestra | Cómo se lee |
|---|---|---|
| `convergencia_<target>.png` | Mejor fitness acumulado por generación, media entre semillas con banda ±σ | Pendiente = velocidad; altura final = calidad; ancho de la banda = cuánto depende de la suerte |
| `diversidad_<target>.png` | Diversidad genética por generación, escala log | Cuándo y cuán rápido colapsa la población |
| `boxplot_final_<target>.png` | Fitness final por variante, con los puntos de cada semilla | **Si las cajas se solapan, no hay ganador** |
| `mejor_vs_promedio_<target>.png` | Mejor (sólido) y promedio (guionado) de la población | Cuando el promedio alcanza al mejor, la población convergió |
| `fitness_vs_evaluaciones_<target>.png` | Mejor fitness contra evaluaciones de fitness | Comparación a presupuesto de cómputo igualado, no a generaciones iguales |

## Imágenes resultado

El mejor individuo de cada variante, renderizado a 320px con la primera semilla del barrido (`seed=1`). El target original está como `<target>__TARGET.png`.

| Target | Variante | Fitness | Archivo |
|---|---|---|---|
| compleja | gen (carga 1) | 0.9133 | `imagenes/compleja__gen-carga-1.png` |
| compleja | multigen M=8 (carga 4) | 0.9127 | `imagenes/compleja__multigen-m8-carga-4.png` |
| compleja | no uniforme (carga 4 -> 0.4) | 0.9124 | `imagenes/compleja__no-uniforme-carga-4--0.4.png` |
| compleja | uniforme (carga 4) | 0.9119 | `imagenes/compleja__uniforme-carga-4.png` |
| detallada | gen (carga 1) | 0.8688 | `imagenes/detallada__gen-carga-1.png` |
| detallada | multigen M=8 (carga 4) | 0.8682 | `imagenes/detallada__multigen-m8-carga-4.png` |
| detallada | no uniforme (carga 4 -> 0.4) | 0.8621 | `imagenes/detallada__no-uniforme-carga-4--0.4.png` |
| detallada | uniforme (carga 4) | 0.8721 | `imagenes/detallada__uniforme-carga-4.png` |
| plana | gen (carga 1) | 0.9478 | `imagenes/plana__gen-carga-1.png` |
| plana | multigen M=8 (carga 4) | 0.9413 | `imagenes/plana__multigen-m8-carga-4.png` |
| plana | no uniforme (carga 4 -> 0.4) | 0.9314 | `imagenes/plana__no-uniforme-carga-4--0.4.png` |
| plana | uniforme (carga 4) | 0.9435 | `imagenes/plana__uniforme-carga-4.png` |

## Datos

- `datos/mutacion.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
