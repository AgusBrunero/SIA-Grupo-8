# Presión de selección

> Eje `presion` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Mide la **presión de selección de forma controlada**: en vez de comparar métodos distintos, se toma un solo método y se mueve su parámetro de presión. En el torneo determinístico eso es **M** (cuántos compiten: M=2 es la presión mínima, M=N equivale a elite); en el probabilístico es **Th** (con qué probabilidad gana el mejor: Th=0.5 es azar puro, Th=1 es torneo determinístico de 2).

Es el experimento que aísla la **causa 1 de convergencia prematura** sin confundirla con otras diferencias entre métodos.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo M=30 | 0.9150 ± 0.0026 | 0.9130–0.9168 | 21.67 | 350 | 2.11e-03 |
| 2 | torneo M=10 | 0.9146 ± 0.0033 | 0.9146–0.9167 | 21.77 | 359 | 2.25e-03 |
| 3 | torneo M=4 | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 4 | torneo M=2 | 0.9118 ± 0.0019 | 0.9107–0.9137 | 22.49 | 375 | 5.46e-03 |
| 5 | torneo prob Th=0.75 | 0.9102 ± 0.0020 | 0.9089–0.9113 | 22.90 | 448 | 6.55e-03 |
| 6 | torneo prob Th=0.95 | 0.9094 ± 0.0029 | 0.9090–0.9099 | 23.10 | 412 | 5.85e-03 |
| 7 | torneo prob Th=0.55 | 0.9093 ± 0.0026 | 0.9081–0.9115 | 23.12 | 427 | 8.73e-03 |

**No hay un ganador claro.** torneo M=30 tiene la media más alta, pero su rango intercuartil se solapa con el de torneo M=10: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **torneo prob Th=0.55**, que queda separado (0.9093).

- **Rango del eje**: 0.0057 de fitness entre el mejor y el peor (1.4 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 350; el más lento, en la 448.
- **Diversidad**: 2 de 7 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **torneo M=30** en la generación 494.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo M=30** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo M=10 | 0.8803 ± 0.0029 | 0.8787–0.8804 | 30.51 | 500 | 2.86e-03 |
| 2 | torneo M=30 | 0.8803 ± 0.0060 | 0.8806–0.8852 | 30.53 | 499 | 2.27e-03 |
| 3 | torneo M=4 | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 4 | torneo M=2 | 0.8753 ± 0.0028 | 0.8739–0.8774 | 31.81 | 526 | 7.01e-03 |
| 5 | torneo prob Th=0.75 | 0.8717 ± 0.0027 | 0.8717–0.8737 | 32.72 | 498 | 7.30e-03 |
| 6 | torneo prob Th=0.95 | 0.8695 ± 0.0041 | 0.8673–0.8710 | 33.27 | 530 | 7.13e-03 |
| 7 | torneo prob Th=0.55 | 0.8684 ± 0.0022 | 0.8660–0.8703 | 33.57 | 518 | 8.26e-03 |

**No hay un ganador claro.** torneo M=10 tiene la media más alta, pero su rango intercuartil se solapa con el de torneo M=30: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **torneo prob Th=0.55**, que queda separado (0.8684).

- **Rango del eje**: 0.0120 de fitness entre el mejor y el peor (3.1 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 466; el más lento, en la 530.
- **Diversidad**: 1 de 7 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **torneo M=30** en la generación 696.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo M=30** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo M=30 | 0.9558 ± 0.0031 | 0.9541–0.9579 | 11.28 | 496 | 2.92e-03 |
| 2 | torneo M=10 | 0.9509 ± 0.0073 | 0.9507–0.9535 | 12.53 | 547 | 3.34e-03 |
| 3 | torneo M=4 | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 4 | torneo M=2 | 0.9393 ± 0.0038 | 0.9368–0.9400 | 15.49 | 569 | 6.81e-03 |
| 5 | torneo prob Th=0.95 | 0.9376 ± 0.0050 | 0.9326–0.9400 | 15.91 | 621 | 8.03e-03 |
| 6 | torneo prob Th=0.55 | 0.9358 ± 0.0061 | 0.9331–0.9388 | 16.38 | 605 | 9.74e-03 |
| 7 | torneo prob Th=0.75 | 0.9323 ± 0.0030 | 0.9319–0.9345 | 17.25 | 619 | 9.08e-03 |

**torneo M=30** gana con el rango intercuartil **separado** del segundo (torneo M=10): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0234 de fitness entre el mejor y el peor (6.0 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 496; el más lento, en la 621.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo M=30** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- Se espera una curva en U: poca presión no explota lo bueno que encuentra, demasiada colapsa la diversidad antes de haber explorado. Mirá dónde está el óptimo y **cuán plano es** — si es plano, la presión no es un parámetro crítico en este problema.
- En `diversidad`, el efecto tiene que ser **monótono** aunque el fitness no lo sea: más presión, colapso más temprano. Si eso se ve, es la evidencia directa del mecanismo.
- En `mejor_vs_promedio`, con presión alta el promedio alcanza al mejor mucho antes: la población entera se vuelve el mismo individuo.
- Comparar M=30 (sobre N=60) con `elite` del eje 01: son casi lo mismo conceptualmente y deberían comportarse parecido.

## Trampas y advertencias

- M está acotado por el tamaño de la población: `tournament_det` hace `min(M, N)`.
- Th < 0.5 invertiría el método (ganaría el peor). La cátedra lo acota a [0.5, 1] y por eso no se prueban valores menores.

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
| compleja | torneo M=10 | 0.9181 | `imagenes/compleja__torneo-m10.png` |
| compleja | torneo M=2 | 0.9137 | `imagenes/compleja__torneo-m2.png` |
| compleja | torneo M=30 | 0.9139 | `imagenes/compleja__torneo-m30.png` |
| compleja | torneo M=4 | 0.9119 | `imagenes/compleja__torneo-m4.png` |
| compleja | torneo prob Th=0.55 | 0.9125 | `imagenes/compleja__torneo-prob-th0.55.png` |
| compleja | torneo prob Th=0.75 | 0.9135 | `imagenes/compleja__torneo-prob-th0.75.png` |
| compleja | torneo prob Th=0.95 | 0.9090 | `imagenes/compleja__torneo-prob-th0.95.png` |
| detallada | torneo M=10 | 0.8787 | `imagenes/detallada__torneo-m10.png` |
| detallada | torneo M=2 | 0.8715 | `imagenes/detallada__torneo-m2.png` |
| detallada | torneo M=30 | 0.8852 | `imagenes/detallada__torneo-m30.png` |
| detallada | torneo M=4 | 0.8721 | `imagenes/detallada__torneo-m4.png` |
| detallada | torneo prob Th=0.55 | 0.8684 | `imagenes/detallada__torneo-prob-th0.55.png` |
| detallada | torneo prob Th=0.75 | 0.8737 | `imagenes/detallada__torneo-prob-th0.75.png` |
| detallada | torneo prob Th=0.95 | 0.8642 | `imagenes/detallada__torneo-prob-th0.95.png` |
| plana | torneo M=10 | 0.9507 | `imagenes/plana__torneo-m10.png` |
| plana | torneo M=2 | 0.9456 | `imagenes/plana__torneo-m2.png` |
| plana | torneo M=30 | 0.9506 | `imagenes/plana__torneo-m30.png` |
| plana | torneo M=4 | 0.9435 | `imagenes/plana__torneo-m4.png` |
| plana | torneo prob Th=0.55 | 0.9433 | `imagenes/plana__torneo-prob-th0.55.png` |
| plana | torneo prob Th=0.75 | 0.9345 | `imagenes/plana__torneo-prob-th0.75.png` |
| plana | torneo prob Th=0.95 | 0.9396 | `imagenes/plana__torneo-prob-th0.95.png` |

## Datos

- `datos/presion.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
