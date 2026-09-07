# Tamaño de población

> Eje `poblacion` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Barre el tamaño de población N, con K = N (brecha generacional 1). Es la **causa 3 de convergencia prematura** que enumera la cátedra: población demasiado escasa.

Una población chica pierde diversidad rápido por deriva genética; una grande la mantiene pero **gasta más evaluaciones por generación**, así que a presupuesto de cómputo fijo hace menos generaciones.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | N=K=120 | 0.9137 ± 0.0071 | 0.9159–0.9179 | 22.01 | 339 | 5.57e-03 |
| 2 | N=K=60 | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 3 | N=K=30 | 0.9104 ± 0.0030 | 0.9086–0.9132 | 22.84 | 440 | 3.26e-03 |
| 4 | N=K=15 | 0.9064 ± 0.0027 | 0.9044–0.9073 | 23.87 | 521 | 1.81e-03 |

**N=K=120** gana con el rango intercuartil **separado** del segundo (N=K=60): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0073 de fitness entre el mejor y el peor (1.9 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 339; el más lento, en la 521.
- **Diversidad**: 2 de 4 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **N=K=15** en la generación 271.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **N=K=30** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 95,859 evaluaciones de fitness y la más barata 11,981. Comparadas todas a las **11,981 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | N=K=15 | 0.9064 |
| 2 | N=K=30 | 0.8995 |
| 3 | N=K=60 | 0.8874 |
| 4 | N=K=120 | 0.8758 |

**El orden se da vuelta.** Por generación gana *N=K=120*; por evaluación gasta gana *N=K=15*. La ventaja de *N=K=120* no venía del método sino de **haber consumido más cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar a la presentación, no el de convergencia.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | N=K=120 | 0.8833 ± 0.0048 | 0.8800–0.8847 | 29.75 | 503 | 6.14e-03 |
| 2 | N=K=60 | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 3 | N=K=30 | 0.8730 ± 0.0034 | 0.8707–0.8738 | 32.37 | 565 | 3.41e-03 |
| 4 | N=K=15 | 0.8642 ± 0.0042 | 0.8605–0.8650 | 34.62 | 612 | 1.52e-03 |

**No hay un ganador claro.** N=K=120 tiene la media más alta, pero su rango intercuartil se solapa con el de N=K=60: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **N=K=15**, que queda separado (0.8642).

- **Rango del eje**: 0.0191 de fitness entre el mejor y el peor (4.9 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 466; el más lento, en la 612.
- **Diversidad**: 2 de 4 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **N=K=15** en la generación 146.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **N=K=60** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 95,859 evaluaciones de fitness y la más barata 11,981. Comparadas todas a las **11,981 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | N=K=15 | 0.8642 |
| 2 | N=K=30 | 0.8546 |
| 3 | N=K=60 | 0.8436 |
| 4 | N=K=120 | 0.8275 |

**El orden se da vuelta.** Por generación gana *N=K=120*; por evaluación gasta gana *N=K=15*. La ventaja de *N=K=120* no venía del método sino de **haber consumido más cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar a la presentación, no el de convergencia.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | N=K=120 | 0.9490 ± 0.0025 | 0.9472–0.9489 | 13.01 | 526 | 7.71e-03 |
| 2 | N=K=60 | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 3 | N=K=30 | 0.9365 ± 0.0114 | 0.9299–0.9430 | 16.18 | 614 | 3.42e-03 |
| 4 | N=K=15 | 0.9323 ± 0.0088 | 0.9288–0.9316 | 17.25 | 628 | 2.04e-03 |

**N=K=120** gana con el rango intercuartil **separado** del segundo (N=K=60): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0166 de fitness entre el mejor y el peor (4.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 526; el más lento, en la 628.
- **Diversidad**: 1 de 4 variantes colapsan por debajo del 1% de su diversidad inicial; la primera es **N=K=15** en la generación 168.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **N=K=60** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

#### A presupuesto de cómputo igualado

Las variantes de este eje **no gastan lo mismo**: al final del barrido, la más cara hizo 95,859 evaluaciones de fitness y la más barata 11,981. Comparadas todas a las **11,981 evaluaciones** que alcanza la más barata:

| # | Variante | Fitness a presupuesto igualado |
|---|---|---|
| 1 | N=K=15 | 0.9323 |
| 2 | N=K=30 | 0.9074 |
| 3 | N=K=60 | 0.8840 |
| 4 | N=K=120 | 0.8653 |

**El orden se da vuelta.** Por generación gana *N=K=120*; por evaluación gasta gana *N=K=15*. La ventaja de *N=K=120* no venía del método sino de **haber consumido más cómputo**. Es el gráfico `fitness_vs_evaluaciones` el que hay que llevar a la presentación, no el de convergencia.

## Qué mirar

- **`fitness_vs_evaluaciones` es el gráfico decisivo de este eje.** Comparar poblaciones por generación es directamente injusto: N=120 gasta 8 veces más evaluaciones por generación que N=15. Si al graficar contra evaluaciones el orden se da vuelta, la conclusión por generación era un artefacto del presupuesto.
- En `diversidad`, el efecto del tamaño tiene que ser claro y monótono: poblaciones chicas colapsan antes. Es la evidencia directa de la causa 3.
- En `mejor_vs_promedio`, con N chico el promedio alcanza al mejor mucho más rápido.
- Buscá el punto donde agrandar la población deja de pagar: es la respuesta práctica a «qué N usar».

## Trampas y advertencias

- K se mueve junto con N para mantener la brecha en 1. El efecto de la brecha se mide por separado en [`03-supervivencia`](../03-supervivencia/informe.md).
- El costo por generación es proporcional a K, no a N: lo que se paga es evaluar a los hijos.

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
| compleja | N=K=120 | 0.9171 | `imagenes/compleja__nk120.png` |
| compleja | N=K=15 | 0.9073 | `imagenes/compleja__nk15.png` |
| compleja | N=K=30 | 0.9144 | `imagenes/compleja__nk30.png` |
| compleja | N=K=60 | 0.9119 | `imagenes/compleja__nk60.png` |
| detallada | N=K=120 | 0.8800 | `imagenes/detallada__nk120.png` |
| detallada | N=K=15 | 0.8635 | `imagenes/detallada__nk15.png` |
| detallada | N=K=30 | 0.8707 | `imagenes/detallada__nk30.png` |
| detallada | N=K=60 | 0.8721 | `imagenes/detallada__nk60.png` |
| plana | N=K=120 | 0.9465 | `imagenes/plana__nk120.png` |
| plana | N=K=15 | 0.9316 | `imagenes/plana__nk15.png` |
| plana | N=K=30 | 0.9386 | `imagenes/plana__nk30.png` |
| plana | N=K=60 | 0.9435 | `imagenes/plana__nk60.png` |

## Datos

- `datos/poblacion.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
