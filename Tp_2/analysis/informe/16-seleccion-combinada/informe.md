# Selección combinada A%/B%

> Eje `seleccion_combinada` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

La cátedra pide poder seleccionar **A% de los padres con un método y (1−A)% con otro**. Está implementado (`selection.build`) y este eje lo mide, con tres métodos puros como referencia para que la comparación tenga sentido.

La intuición detrás de combinar es repartir el trabajo: un método de presión alta (elite) asegura que lo mejor se propague, y uno de presión baja (ruleta) mantiene diversidad. La pregunta es si esa mezcla rinde más que cualquiera de los dos solo.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo det (puro) | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 2 | 25% elite + 75% torneo | 0.9131 ± 0.0019 | 0.9121–0.9138 | 22.17 | 380 | 4.30e-03 |
| 3 | 50% elite + 50% torneo | 0.9125 ± 0.0013 | 0.9122–0.9126 | 22.32 | 383 | 3.89e-03 |
| 4 | 50% torneo + 50% ruleta | 0.9117 ± 0.0017 | 0.9101–0.9136 | 22.51 | 385 | 5.05e-03 |
| 5 | 50% elite + 50% ruleta | 0.9107 ± 0.0020 | 0.9098–0.9119 | 22.78 | 443 | 5.78e-03 |
| 6 | ruleta (pura) | 0.9086 ± 0.0021 | 0.9090–0.9096 | 23.30 | 450 | 9.15e-03 |
| 7 | elite (puro) | 0.9086 ± 0.0024 | 0.9063–0.9103 | 23.31 | 424 | 7.64e-03 |

**No hay un ganador claro.** torneo det (puro) tiene la media más alta, pero su rango intercuartil se solapa con el de 25% elite + 75% torneo: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **elite (puro)**, que queda separado (0.9086).

- **Rango del eje**: 0.0047 de fitness entre el mejor y el peor (1.2 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 380; el más lento, en la 450.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **50% torneo + 50% ruleta** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | torneo det (puro) | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | 25% elite + 75% torneo | 0.8763 ± 0.0025 | 0.8744–0.8773 | 31.54 | 511 | 4.91e-03 |
| 3 | 50% elite + 50% torneo | 0.8750 ± 0.0036 | 0.8728–0.8767 | 31.87 | 511 | 4.17e-03 |
| 4 | 50% torneo + 50% ruleta | 0.8746 ± 0.0035 | 0.8751–0.8761 | 31.98 | 488 | 5.28e-03 |
| 5 | 50% elite + 50% ruleta | 0.8719 ± 0.0035 | 0.8721–0.8737 | 32.66 | 523 | 6.76e-03 |
| 6 | ruleta (pura) | 0.8684 ± 0.0018 | 0.8677–0.8704 | 33.55 | 547 | 9.74e-03 |
| 7 | elite (puro) | 0.8684 ± 0.0027 | 0.8668–0.8675 | 33.56 | 515 | 8.08e-03 |

**No hay un ganador claro.** torneo det (puro) tiene la media más alta, pero su rango intercuartil se solapa con el de 25% elite + 75% torneo: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **elite (puro)**, que queda separado (0.8684).

- **Rango del eje**: 0.0098 de fitness entre el mejor y el peor (2.5 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 466; el más lento, en la 547.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo det (puro)** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | 25% elite + 75% torneo | 0.9502 ± 0.0034 | 0.9469–0.9511 | 12.70 | 580 | 5.60e-03 |
| 2 | 50% elite + 50% torneo | 0.9433 ± 0.0059 | 0.9418–0.9476 | 14.45 | 568 | 6.30e-03 |
| 3 | 50% torneo + 50% ruleta | 0.9421 ± 0.0024 | 0.9396–0.9426 | 14.76 | 594 | 6.86e-03 |
| 4 | 50% elite + 50% ruleta | 0.9403 ± 0.0008 | 0.9398–0.9410 | 15.22 | 596 | 7.34e-03 |
| 5 | torneo det (puro) | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |
| 6 | elite (puro) | 0.9335 ± 0.0032 | 0.9306–0.9337 | 16.95 | 606 | 8.90e-03 |
| 7 | ruleta (pura) | 0.9333 ± 0.0031 | 0.9322–0.9359 | 17.01 | 634 | 9.72e-03 |

**No hay un ganador claro.** 25% elite + 75% torneo tiene la media más alta, pero su rango intercuartil se solapa con el de 50% elite + 50% torneo: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **ruleta (pura)**, que queda separado (0.9333).

- **Rango del eje**: 0.0169 de fitness entre el mejor y el peor (4.3 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 551; el más lento, en la 634.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **torneo det (puro)** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- **Compará cada mezcla contra sus dos componentes puros**, que están en el mismo gráfico. Si una mezcla no le gana a los dos, combinar no aportó.
- `25% elite + 75% torneo` es la mezcla con más presión de las tres; si el orden sigue la presión, el resultado se explica por [`02-presion-seleccion`](../02-presion-seleccion/informe.md) y no por la combinación en sí.
- En `diversidad`: la promesa de combinar es sostener más dispersión que el método de presión alta solo. Si eso no se ve, el mecanismo propuesto no está operando.

## Trampas y advertencias

- `elite` con k=N devuelve la población entera (presión cero), así que `50% elite + 50% X` en realidad significa *la mitad de los padres sin ninguna presión*. No es 'medio elite': es medio azar.
- Este eje comparte variantes con [`01-seleccion`](../01-seleccion/informe.md) (los puros); los números tienen que coincidir, y sirven de control cruzado.

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
| compleja | 25% elite + 75% torneo | 0.9104 | `imagenes/compleja__25-elite-mas-75-torneo.png` |
| compleja | 50% elite + 50% ruleta | 0.9119 | `imagenes/compleja__50-elite-mas-50-ruleta.png` |
| compleja | 50% elite + 50% torneo | 0.9126 | `imagenes/compleja__50-elite-mas-50-torneo.png` |
| compleja | 50% torneo + 50% ruleta | 0.9109 | `imagenes/compleja__50-torneo-mas-50-ruleta.png` |
| compleja | elite (puro) | 0.9123 | `imagenes/compleja__elite-puro.png` |
| compleja | ruleta (pura) | 0.9104 | `imagenes/compleja__ruleta-pura.png` |
| compleja | torneo det (puro) | 0.9119 | `imagenes/compleja__torneo-det-puro.png` |
| detallada | 25% elite + 75% torneo | 0.8762 | `imagenes/detallada__25-elite-mas-75-torneo.png` |
| detallada | 50% elite + 50% ruleta | 0.8737 | `imagenes/detallada__50-elite-mas-50-ruleta.png` |
| detallada | 50% elite + 50% torneo | 0.8728 | `imagenes/detallada__50-elite-mas-50-torneo.png` |
| detallada | 50% torneo + 50% ruleta | 0.8751 | `imagenes/detallada__50-torneo-mas-50-ruleta.png` |
| detallada | elite (puro) | 0.8668 | `imagenes/detallada__elite-puro.png` |
| detallada | ruleta (pura) | 0.8704 | `imagenes/detallada__ruleta-pura.png` |
| detallada | torneo det (puro) | 0.8721 | `imagenes/detallada__torneo-det-puro.png` |
| plana | 25% elite + 75% torneo | 0.9465 | `imagenes/plana__25-elite-mas-75-torneo.png` |
| plana | 50% elite + 50% ruleta | 0.9398 | `imagenes/plana__50-elite-mas-50-ruleta.png` |
| plana | 50% elite + 50% torneo | 0.9502 | `imagenes/plana__50-elite-mas-50-torneo.png` |
| plana | 50% torneo + 50% ruleta | 0.9396 | `imagenes/plana__50-torneo-mas-50-ruleta.png` |
| plana | elite (puro) | 0.9333 | `imagenes/plana__elite-puro.png` |
| plana | ruleta (pura) | 0.9322 | `imagenes/plana__ruleta-pura.png` |
| plana | torneo det (puro) | 0.9435 | `imagenes/plana__torneo-det-puro.png` |

## Datos

- `datos/seleccion_combinada.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
