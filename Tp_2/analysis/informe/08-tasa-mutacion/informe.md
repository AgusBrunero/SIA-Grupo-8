# Carga de mutación

> Eje `tasa_mutacion` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Barre la **carga de mutación**: cuántos genes se espera que muten por individuo y generación. Se expresa como carga y no como `pm` a propósito, porque `pm` es probabilidad por gen y sólo tiene sentido junto al largo del cromosoma (acá 500 genes, así que carga = pm × 500).

Es la **causa 2 de convergencia prematura** que enumera la cátedra: probabilidad de mutación demasiado baja.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | carga 2 | 0.9143 ± 0.0024 | 0.9132–0.9157 | 21.86 | 456 | 2.55e-03 |
| 2 | carga 4 | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 3 | carga 10 | 0.9095 ± 0.0023 | 0.9077–0.9115 | 23.08 | 342 | 1.07e-02 |
| 4 | carga 0.5 | 0.9053 ± 0.0021 | 0.9053–0.9056 | 24.15 | 533 | 1.07e-03 |
| 5 | carga 25 | 0.9044 ± 0.0021 | 0.9029–0.9068 | 24.39 | 279 | 2.59e-02 |
| 6 | carga 50 | 0.8921 ± 0.0021 | 0.8913–0.8918 | 27.52 | 227 | 4.52e-02 |

**No hay un ganador claro.** carga 2 tiene la media más alta, pero su rango intercuartil se solapa con el de carga 4: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **carga 50**, que queda separado (0.8921).

- **Rango del eje**: 0.0222 de fitness entre el mejor y el peor (5.7 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 227; el más lento, en la 533.
- **Diversidad**: 2 de 6 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **carga 0.5** en la generación 21.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **carga 2** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 43,719. Comparadas todas a las **43,719 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | carga 2 | 0.9133 |
| 2 | carga 4 | 0.9122 |
| 3 | carga 10 | 0.9086 |
| 4 | carga 0.5 | 0.9053 |
| 5 | carga 25 | 0.9037 |
| 6 | carga 50 | 0.8915 |

El orden **se mantiene**: *carga 2* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | carga 4 | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | carga 2 | 0.8739 ± 0.0038 | 0.8707–0.8766 | 32.16 | 530 | 2.31e-03 |
| 3 | carga 10 | 0.8709 ± 0.0020 | 0.8693–0.8715 | 32.92 | 444 | 1.27e-02 |
| 4 | carga 0.5 | 0.8689 ± 0.0039 | 0.8690–0.8692 | 33.42 | 593 | 9.38e-04 |
| 5 | carga 25 | 0.8564 ± 0.0063 | 0.8518–0.8622 | 36.63 | 349 | 2.56e-02 |
| 6 | carga 50 | 0.8444 ± 0.0007 | 0.8441–0.8451 | 39.68 | 374 | 4.71e-02 |

**No hay un ganador claro.** carga 4 tiene la media más alta, pero su rango intercuartil se solapa con el de carga 2: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **carga 50**, que queda separado (0.8444).

- **Rango del eje**: 0.0337 de fitness entre el mejor y el peor (8.6 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 349; el más lento, en la 593.
- **Diversidad**: 2 de 6 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **carga 0.5** en la generación 20.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **carga 4** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 43,719. Comparadas todas a las **43,719 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | carga 4 | 0.8769 |
| 2 | carga 2 | 0.8718 |
| 3 | carga 10 | 0.8697 |
| 4 | carga 0.5 | 0.8689 |
| 5 | carga 25 | 0.8557 |
| 6 | carga 50 | 0.8431 |

El orden **se mantiene**: *carga 4* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | carga 2 | 0.9462 ± 0.0033 | 0.9431–0.9493 | 13.72 | 607 | 2.87e-03 |
| 2 | carga 10 | 0.9396 ± 0.0049 | 0.9394–0.9426 | 15.39 | 497 | 1.12e-02 |
| 3 | carga 4 | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 4 | carga 0.5 | 0.9276 ± 0.0036 | 0.9248–0.9307 | 18.46 | 665 | 1.30e-03 |
| 5 | carga 25 | 0.9189 ± 0.0059 | 0.9129–0.9239 | 20.68 | 489 | 2.98e-02 |
| 6 | carga 50 | 0.8950 ± 0.0062 | 0.8902–0.8984 | 26.76 | 511 | 4.86e-02 |

**carga 2** gana con el rango intercuartil **separado** del segundo (carga 10): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0511 de fitness entre el mejor y el peor (13.0 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 489; el más lento, en la 665.
- **Diversidad**: 2 de 6 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **carga 0.5** en la generación 26.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **carga 2** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 43,719. Comparadas todas a las **43,719 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | carga 2 | 0.9437 |
| 2 | carga 10 | 0.9380 |
| 3 | carga 4 | 0.9380 |
| 4 | carga 0.5 | 0.9276 |
| 5 | carga 25 | 0.9173 |
| 6 | carga 50 | 0.8935 |

El orden **se mantiene**: *carga 2* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Qué mirar

- Se espera una U: muy poca mutación no explora y la población se congela; demasiada destruye lo construido más rápido de lo que lo mejora. Mirá **dónde está el mínimo y cuán ancho es el valle** — si es ancho, no hace falta afinar.
- El extremo malo suele ser mucho más claro que el óptimo. Es más defendible decir «carga 50 es netamente peor» que «carga 4 es la mejor».
- En `diversidad` se ve el mecanismo directo: carga alta sostiene diversidad artificialmente sin que eso se traduzca en fitness. Es la mejor ilustración de que **diversidad no es calidad**.
- Con carga 0.5 la población casi no cambia: contrastalo con el criterio de estructura en [`13-criterios-corte`](../13-criterios-corte/informe.md).

## Trampas y advertencias

- La carga óptima depende del **presupuesto**: la que gana con 800 generaciones no es necesariamente la que gana con 3000.
- Y depende del **largo del cromosoma**: por eso este eje fija los triángulos en 50. El efecto de cambiar el largo se mide en [`11-triangulos`](../11-triangulos/informe.md).

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
| compleja | carga 0.5 | 0.9083 | `imagenes/compleja__carga-0.5.png` |
| compleja | carga 10 | 0.9115 | `imagenes/compleja__carga-10.png` |
| compleja | carga 2 | 0.9180 | `imagenes/compleja__carga-2.png` |
| compleja | carga 25 | 0.9070 | `imagenes/compleja__carga-25.png` |
| compleja | carga 4 | 0.9119 | `imagenes/compleja__carga-4.png` |
| compleja | carga 50 | 0.8960 | `imagenes/compleja__carga-50.png` |
| detallada | carga 0.5 | 0.8692 | `imagenes/detallada__carga-0.5.png` |
| detallada | carga 10 | 0.8693 | `imagenes/detallada__carga-10.png` |
| detallada | carga 2 | 0.8796 | `imagenes/detallada__carga-2.png` |
| detallada | carga 25 | 0.8653 | `imagenes/detallada__carga-25.png` |
| detallada | carga 4 | 0.8721 | `imagenes/detallada__carga-4.png` |
| detallada | carga 50 | 0.8452 | `imagenes/detallada__carga-50.png` |
| plana | carga 0.5 | 0.9248 | `imagenes/plana__carga-0.5.png` |
| plana | carga 10 | 0.9304 | `imagenes/plana__carga-10.png` |
| plana | carga 2 | 0.9431 | `imagenes/plana__carga-2.png` |
| plana | carga 25 | 0.9245 | `imagenes/plana__carga-25.png` |
| plana | carga 4 | 0.9435 | `imagenes/plana__carga-4.png` |
| plana | carga 50 | 0.8984 | `imagenes/plana__carga-50.png` |

## Datos

- `datos/tasa_mutacion.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
