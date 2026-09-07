# Método de cruza

> Eje `cruza` · batch `20260906T233756Z` · 5 semillas · 800 generaciones

## Qué mide este eje

Compara los cinco métodos de cruza. La pregunta de fondo es cuánto **preserva bloques constructivos**: conjuntos de genes que juntos valen más que por separado. Acá un bloque constructivo es un triángulo bien ubicado, o un grupo de triángulos que se superponen bien.

Se puede **predecir antes de medir**: la cruza uniforme es la única que no mantiene correlación posicional entre loci, así que debería ser la más disruptiva. Que gane o pierda dice cuánto le importa a este problema conservar el orden.

La configuración exacta está en [`config.md`](config.md).

## Resultados — imagen `compleja`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | uniforme | 0.9143 ± 0.0014 | 0.9141–0.9154 | 21.84 | 351 | 4.03e-03 |
| 2 | un punto | 0.9133 ± 0.0023 | 0.9119–0.9144 | 22.11 | 405 | 3.78e-03 |
| 3 | dos puntos | 0.9130 ± 0.0023 | 0.9113–0.9144 | 22.18 | 384 | 4.54e-03 |
| 4 | espacial | 0.9117 ± 0.0034 | 0.9106–0.9124 | 22.53 | 394 | 4.04e-03 |
| 5 | anular | 0.9115 ± 0.0035 | 0.9083–0.9128 | 22.57 | 403 | 3.81e-03 |

**No hay un ganador claro.** uniforme tiene la media más alta, pero su rango intercuartil se solapa con el de un punto: con 5 semillas la diferencia está dentro del ruido. Lo que **sí** se puede afirmar es que le gana a **anular**, que queda separado (0.9115).

- **Rango del eje**: 0.0029 de fitness entre el mejor y el peor (0.7 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 351; el más lento, en la 405.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **anular** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `detallada`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | un punto | 0.8781 ± 0.0050 | 0.8731–0.8826 | 31.07 | 466 | 4.42e-03 |
| 2 | uniforme | 0.8774 ± 0.0039 | 0.8760–0.8814 | 31.28 | 444 | 5.20e-03 |
| 3 | anular | 0.8767 ± 0.0033 | 0.8743–0.8790 | 31.45 | 494 | 4.30e-03 |
| 4 | espacial | 0.8752 ± 0.0023 | 0.8737–0.8760 | 31.82 | 512 | 4.87e-03 |
| 5 | dos puntos | 0.8737 ± 0.0048 | 0.8705–0.8748 | 32.20 | 475 | 4.85e-03 |

**Las 5 variantes son indistinguibles a esta escala**: ni siquiera el primero se separa del último (0.8781 contra 0.8737, con rangos que se solapan). El resultado es un empate, y como tal hay que presentarlo.

- **Rango del eje**: 0.0044 de fitness entre el mejor y el peor (1.1 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 444; el más lento, en la 512.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **un punto** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Resultados — imagen `plana`

| # | Variante | Fitness final | IQR | RMSE | Gen. al 99% | Diversidad final |
|---|---|---|---|---|---|---|
| 1 | uniforme | 0.9539 ± 0.0061 | 0.9514–0.9565 | 11.75 | 524 | 5.47e-03 |
| 2 | anular | 0.9463 ± 0.0042 | 0.9426–0.9497 | 13.69 | 566 | 5.80e-03 |
| 3 | dos puntos | 0.9459 ± 0.0045 | 0.9433–0.9497 | 13.79 | 562 | 4.91e-03 |
| 4 | espacial | 0.9426 ± 0.0067 | 0.9352–0.9477 | 14.64 | 631 | 5.35e-03 |
| 5 | un punto | 0.9396 ± 0.0049 | 0.9393–0.9433 | 15.40 | 551 | 4.99e-03 |

**uniforme** gana con el rango intercuartil **separado** del segundo (anular): la diferencia no se explica por el ruido entre semillas.

- **Rango del eje**: 0.0143 de fitness entre el mejor y el peor (3.6 puntos de RMSE).
- **Velocidad**: el más rápido llega al 99% de su fitness final en la generación 524; el más lento, en la 631.
- **Convergencia de la población**: la brecha entre el mejor y el promedio es mínima en **un punto** (0.0001); cuanto más chica, más se parecen entre sí todos los individuos.

## Qué mirar

- Un punto y anular preservan bloques contiguos; uniforme los rompe. Si uniforme igual gana, es señal de que el problema **tolera o necesita** mucha mezcla — probablemente porque los triángulos interactúan poco entre sí salvo por superposición.
- La cruza **espacial** es propia: parte por dónde cae el triángulo en el canvas, no por su índice. Miralo junto a [`11-triangulos`](../11-triangulos/informe.md): con pocos triángulos una partición espacial casi no tiene qué repartir.
- En `diversidad`, si alguna cruza mantiene la población más dispersa. Una cruza disruptiva actúa parcialmente como mutación.
- El eje se lee junto con [`06-tasa-cruza`](../06-tasa-cruza/informe.md): si con pc=0 el algoritmo rinde casi igual, la elección de método de cruza importa poco por definición.

## Trampas y advertencias

- La granularidad está fija en este eje; se estudia aparte en [`05-granularidad`](../05-granularidad/informe.md).
- La cruza espacial **ignora** la granularidad configurada: siempre trabaja por triángulo entero, porque partir un triángulo al medio no tiene sentido geométrico.

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
| compleja | anular | 0.9083 | `imagenes/compleja__anular.png` |
| compleja | dos puntos | 0.9160 | `imagenes/compleja__dos-puntos.png` |
| compleja | espacial | 0.9171 | `imagenes/compleja__espacial.png` |
| compleja | un punto | 0.9119 | `imagenes/compleja__un-punto.png` |
| compleja | uniforme | 0.9116 | `imagenes/compleja__uniforme.png` |
| detallada | anular | 0.8730 | `imagenes/detallada__anular.png` |
| detallada | dos puntos | 0.8676 | `imagenes/detallada__dos-puntos.png` |
| detallada | espacial | 0.8793 | `imagenes/detallada__espacial.png` |
| detallada | un punto | 0.8721 | `imagenes/detallada__un-punto.png` |
| detallada | uniforme | 0.8761 | `imagenes/detallada__uniforme.png` |
| plana | anular | 0.9525 | `imagenes/plana__anular.png` |
| plana | dos puntos | 0.9497 | `imagenes/plana__dos-puntos.png` |
| plana | espacial | 0.9448 | `imagenes/plana__espacial.png` |
| plana | un punto | 0.9435 | `imagenes/plana__un-punto.png` |
| plana | uniforme | 0.9514 | `imagenes/plana__uniforme.png` |

## Datos

- `datos/cruza.csv` — una fila por generación, variante y semilla
- `datos/resumen.csv` — una fila por variante y target, con mediana e IQR
