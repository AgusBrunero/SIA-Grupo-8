# Cantidad de triángulos

> Eje `triangulos` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Barre el segundo **parámetro del problema** (el primero es la imagen) **cruzado con el régimen de la tasa de mutación**, porque los dos no se pueden separar.

Más triángulos dan más capacidad de representación pero un cromosoma más largo: de 100 a 2000 genes. Y como `pm` es probabilidad **por gen**, dejar `pm` fijo mientras el cromosoma crece multiplica la carga de mutación por el mismo factor. Por eso cada cantidad se corre dos veces: con **pm fijo** (lo que sale de copiar el valor calibrado) y con **carga fija** (escalando pm como 1/L para mantener constante la cantidad esperada de genes mutados).

Sin ese cruce, cualquier conclusión sobre «cuántos triángulos conviene» estaría confundida con un efecto de mutación.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | 100 tri - carga fija | 0.9151 ± 0.0020 | 0.9141–0.9158 | 21.66 | 472 | 2.98e-03 |
| 2 | 100 tri - pm fijo | 0.9137 ± 0.0022 | 0.9117–0.9150 | 21.99 | 422 | 6.19e-03 |
| 3 | 200 tri - carga fija | 0.9136 ± 0.0036 | 0.9119–0.9164 | 22.02 | 534 | 2.47e-03 |
| 4 | 50 tri - pm fijo | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 5 | 50 tri - carga fija | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 6 | 200 tri - pm fijo | 0.9114 ± 0.0027 | 0.9099–0.9129 | 22.58 | 408 | 7.29e-03 |
| 7 | 25 tri - pm fijo | 0.9055 ± 0.0027 | 0.9037–0.9078 | 24.09 | 398 | 1.54e-03 |
| 8 | 25 tri - carga fija | 0.9043 ± 0.0037 | 0.9059–0.9061 | 24.39 | 403 | 4.26e-03 |
| 9 | 10 tri - pm fijo | 0.8937 ± 0.0039 | 0.8901–0.8973 | 27.10 | 250 | 2.06e-04 |
| 10 | 10 tri - carga fija | 0.8933 ± 0.0019 | 0.8928–0.8946 | 27.22 | 257 | 3.87e-03 |

**No hay un ganador claro.** 100 tri - carga fija tiene la media más alta, pero su rango intercuartil se solapa con el de 100 tri - pm fijo: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **10 tri - carga fija**, que queda separado (0.8933).

- **Rango del eje**: 0.0218 de fitness entre el mejor y el peor (5.6 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 250; el más lento, en la 534.
- **Diversidad**: 3 de 10 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **200 tri - carga fija** en la generación 77.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **10 tri - pm fijo** (0.0000); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 44,807. Comparadas todas a las **44,807 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | 100 tri - carga fija | 0.9141 |
| 2 | 100 tri - pm fijo | 0.9130 |
| 3 | 50 tri - pm fijo | 0.9125 |
| 4 | 50 tri - carga fija | 0.9125 |
| 5 | 200 tri - carga fija | 0.9121 |
| 6 | 200 tri - pm fijo | 0.9105 |
| 7 | 25 tri - pm fijo | 0.9050 |
| 8 | 25 tri - carga fija | 0.9039 |
| 9 | 10 tri - pm fijo | 0.8937 |
| 10 | 10 tri - carga fija | 0.8931 |

El orden **se mantiene**: *100 tri - carga fija* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | 50 tri - pm fijo | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | 50 tri - carga fija | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 3 | 100 tri - carga fija | 0.8753 ± 0.0050 | 0.8751–0.8787 | 31.79 | 576 | 3.44e-03 |
| 4 | 100 tri - pm fijo | 0.8733 ± 0.0040 | 0.8691–0.8771 | 32.30 | 528 | 5.98e-03 |
| 5 | 200 tri - pm fijo | 0.8686 ± 0.0040 | 0.8654–0.8706 | 33.51 | 556 | 7.98e-03 |
| 6 | 200 tri - carga fija | 0.8672 ± 0.0056 | 0.8634–0.8701 | 33.87 | 641 | 2.28e-03 |
| 7 | 25 tri - carga fija | 0.8670 ± 0.0042 | 0.8630–0.8704 | 33.92 | 420 | 5.57e-03 |
| 8 | 25 tri - pm fijo | 0.8646 ± 0.0020 | 0.8628–0.8662 | 34.53 | 392 | 2.48e-03 |
| 9 | 10 tri - carga fija | 0.8502 ± 0.0026 | 0.8488–0.8528 | 38.19 | 258 | 6.86e-03 |
| 10 | 10 tri - pm fijo | 0.8469 ± 0.0055 | 0.8437–0.8491 | 39.05 | 269 | 2.06e-04 |

**No hay un ganador claro.** 50 tri - pm fijo tiene la media más alta, pero su rango intercuartil se solapa con el de 50 tri - carga fija: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **10 tri - pm fijo**, que queda separado (0.8469).

- **Rango del eje**: 0.0313 de fitness entre el mejor y el peor (8.0 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 258; el más lento, en la 641.
- **Diversidad**: 3 de 10 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **200 tri - carga fija** en la generación 106.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **10 tri - pm fijo** (0.0000); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 44,807. Comparadas todas a las **44,807 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | 50 tri - pm fijo | 0.8772 |
| 2 | 50 tri - carga fija | 0.8772 |
| 3 | 100 tri - carga fija | 0.8739 |
| 4 | 100 tri - pm fijo | 0.8720 |
| 5 | 200 tri - pm fijo | 0.8669 |
| 6 | 25 tri - carga fija | 0.8663 |
| 7 | 200 tri - carga fija | 0.8647 |
| 8 | 25 tri - pm fijo | 0.8641 |
| 9 | 10 tri - carga fija | 0.8498 |
| 10 | 10 tri - pm fijo | 0.8469 |

El orden **se mantiene**: *50 tri - pm fijo* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | 25 tri - pm fijo | 0.9407 ± 0.0056 | 0.9404–0.9423 | 15.12 | 487 | 3.20e-03 |
| 2 | 50 tri - pm fijo | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 3 | 50 tri - carga fija | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 4 | 100 tri - pm fijo | 0.9345 ± 0.0069 | 0.9335–0.9383 | 16.70 | 662 | 8.42e-03 |
| 5 | 25 tri - carga fija | 0.9338 ± 0.0043 | 0.9320–0.9332 | 16.89 | 521 | 7.44e-03 |
| 6 | 100 tri - carga fija | 0.9316 ± 0.0083 | 0.9257–0.9400 | 17.45 | 620 | 3.58e-03 |
| 7 | 200 tri - pm fijo | 0.9212 ± 0.0136 | 0.9086–0.9327 | 20.10 | 644 | 9.13e-03 |
| 8 | 10 tri - carga fija | 0.9162 ± 0.0079 | 0.9119–0.9179 | 21.37 | 425 | 6.68e-03 |
| 9 | 10 tri - pm fijo | 0.9159 ± 0.0058 | 0.9159–0.9203 | 21.43 | 319 | 3.86e-04 |
| 10 | 200 tri - carga fija | 0.9130 ± 0.0158 | 0.9053–0.9104 | 22.19 | 695 | 3.16e-03 |

**No hay un ganador claro.** 25 tri - pm fijo tiene la media más alta, pero su rango intercuartil se solapa con el de 50 tri - pm fijo: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **200 tri - carga fija**, que queda separado (0.9130).

- **Rango del eje**: 0.0277 de fitness entre el mejor y el peor (7.1 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 319; el más lento, en la 695.
- **Diversidad**: 3 de 10 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **200 tri - carga fija** en la generación 114.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **10 tri - pm fijo** (0.0000); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 48,060 evaluaciones de fitness y la más barata 44,807. Comparadas todas a las **44,807 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | 25 tri - pm fijo | 0.9397 |
| 2 | 50 tri - pm fijo | 0.9384 |
| 3 | 50 tri - carga fija | 0.9384 |
| 4 | 25 tri - carga fija | 0.9325 |
| 5 | 100 tri - pm fijo | 0.9309 |
| 6 | 100 tri - carga fija | 0.9294 |
| 7 | 200 tri - pm fijo | 0.9183 |
| 8 | 10 tri - pm fijo | 0.9159 |
| 9 | 10 tri - carga fija | 0.9150 |
| 10 | 200 tri - carga fija | 0.9091 |

El orden **se mantiene**: *25 tri - pm fijo* gana también a presupuesto igualado, así que su ventaja no se explica por haber gastado más cómputo.

## Qué mirar

- **Compará las dos series por separado antes de sacar conclusiones.** Si con pm fijo más triángulos empeora y con carga fija mejora, entonces «más triángulos es peor» era un artefacto de no escalar la mutación, no una propiedad del problema.
- El punto donde las dos series se cruzan es 50 triángulos: ahí los dos regímenes son el mismo valor por construcción. Sirve de control.
- Mirá el **arranque** de las curvas además del final: con inicialización informada, más triángulos arranca mejor (grilla más fina) aunque termine peor. Son dos efectos opuestos que se pueden separar mirando la generación 1 contra la última.
- En las imágenes resultado, la comparación visual es más elocuente que el fitness: con la misma cantidad de generaciones, más triángulos puede verse **peor** aunque tenga más capacidad.

## Trampas y advertencias

- El fitness entre cantidades distintas mezcla capacidad y dificultad de búsqueda. Igualar el presupuesto de evaluaciones no lo arregla: el costo extra está **adentro** de cada evaluación (renderizar más triángulos), no en la cantidad de evaluaciones. Hay que mirar además el tiempo, que está en el resumen.
- La inicialización de la base es `random`, así que acá no se ve la ventaja de arranque que da la grilla informada. Ese cruce está en [`15-caso-final`](../15-caso-final/informe.md).

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
| compleja | 10 tri - carga fija | 0.8946 | `imagenes/compleja__10-tri--carga-fija.png` |
| compleja | 10 tri - pm fijo | 0.8901 | `imagenes/compleja__10-tri--pm-fijo.png` |
| compleja | 100 tri - carga fija | 0.9179 | `imagenes/compleja__100-tri--carga-fija.png` |
| compleja | 100 tri - pm fijo | 0.9174 | `imagenes/compleja__100-tri--pm-fijo.png` |
| compleja | 200 tri - carga fija | 0.9142 | `imagenes/compleja__200-tri--carga-fija.png` |
| compleja | 200 tri - pm fijo | 0.9085 | `imagenes/compleja__200-tri--pm-fijo.png` |
| compleja | 25 tri - carga fija | 0.9061 | `imagenes/compleja__25-tri--carga-fija.png` |
| compleja | 25 tri - pm fijo | 0.9037 | `imagenes/compleja__25-tri--pm-fijo.png` |
| compleja | 50 tri - carga fija | 0.9119 | `imagenes/compleja__50-tri--carga-fija.png` |
| compleja | 50 tri - pm fijo | 0.9119 | `imagenes/compleja__50-tri--pm-fijo.png` |
| detallada | 10 tri - carga fija | 0.8488 | `imagenes/detallada__10-tri--carga-fija.png` |
| detallada | 10 tri - pm fijo | 0.8481 | `imagenes/detallada__10-tri--pm-fijo.png` |
| detallada | 100 tri - carga fija | 0.8664 | `imagenes/detallada__100-tri--carga-fija.png` |
| detallada | 100 tri - pm fijo | 0.8691 | `imagenes/detallada__100-tri--pm-fijo.png` |
| detallada | 200 tri - carga fija | 0.8643 | `imagenes/detallada__200-tri--carga-fija.png` |
| detallada | 200 tri - pm fijo | 0.8654 | `imagenes/detallada__200-tri--pm-fijo.png` |
| detallada | 25 tri - carga fija | 0.8704 | `imagenes/detallada__25-tri--carga-fija.png` |
| detallada | 25 tri - pm fijo | 0.8628 | `imagenes/detallada__25-tri--pm-fijo.png` |
| detallada | 50 tri - carga fija | 0.8721 | `imagenes/detallada__50-tri--carga-fija.png` |
| detallada | 50 tri - pm fijo | 0.8721 | `imagenes/detallada__50-tri--pm-fijo.png` |
| plana | 10 tri - carga fija | 0.9119 | `imagenes/plana__10-tri--carga-fija.png` |
| plana | 10 tri - pm fijo | 0.9203 | `imagenes/plana__10-tri--pm-fijo.png` |
| plana | 100 tri - carga fija | 0.9189 | `imagenes/plana__100-tri--carga-fija.png` |
| plana | 100 tri - pm fijo | 0.9383 | `imagenes/plana__100-tri--pm-fijo.png` |
| plana | 200 tri - carga fija | 0.8973 | `imagenes/plana__200-tri--carga-fija.png` |
| plana | 200 tri - pm fijo | 0.9086 | `imagenes/plana__200-tri--pm-fijo.png` |
| plana | 25 tri - carga fija | 0.9330 | `imagenes/plana__25-tri--carga-fija.png` |
| plana | 25 tri - pm fijo | 0.9407 | `imagenes/plana__25-tri--pm-fijo.png` |
| plana | 50 tri - carga fija | 0.9435 | `imagenes/plana__50-tri--carga-fija.png` |
| plana | 50 tri - pm fijo | 0.9435 | `imagenes/plana__50-tri--pm-fijo.png` |

## Datos

- `datos/triangulos.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
